"""Stage 1: catch a thrown ball (no bounces) with a cup on a KUKA iiwa 14.

The ball flies with aerodynamics the robot doesn't know: quadratic drag with a random
coefficient and mass per throw, and optionally random spin (Magnus force).

Pipeline, run at CTRL_HZ:
  ball position -> predictor (BallisticPredictor: gravity only, or DragPredictor: online drag estimate)
  -> CatchPlanner (earliest reachable point on the predicted path)
  -> DiffIK (move the cup mouth there, facing the incoming ball)
  -> joint position commands to the arm's position actuators.

Usage:
  .venv/bin/python stage1.py --trials 30 --spin 150 --blind 2.5
  .venv/bin/python stage1.py --trials 1 --video catch.mp4
  .venv/bin/mjpython stage1.py --trials 5 --view
"""
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from scene import BALL_RADIUS, CUP_DEPTH, CUP_RADIUS, QD_LIMITS, SHOULDER, load_model

G = np.array([0.0, 0.0, -9.81])
CTRL_HZ = 100
Q_HOME = np.array([0.0, 0.3, 0.0, -1.6, 0.0, -0.4, 0.0])
REACH = (0.35, 0.95)     # catch-point distance from the shoulder, m
MIN_CATCH_Z = 0.25
EE_SPEED = 1.8           # assumed achievable cup speed for feasibility checks, m/s
QD_MAX = QD_LIMITS        # per-joint speed limits from the iiwa14 URDF, rad/s
CUP_TURN_RATE = 4.0      # assumed achievable cup reorientation rate for feasibility checks, rad/s
GIVE_DIST = 0.0         # how far the cup retreats along the ball path during the catch, m
GIVE_TIME = 0.15
RHO_AIR = 1.2
BALL_AREA = np.pi * BALL_RADIUS**2


class Aero:
    """Per-throw aerodynamics: a = -k|v|v + c (w x v). Hidden from the predictors."""

    def __init__(self, k=0.0, c=0.0, spin=np.zeros(3)):
        self.k, self.c, self.spin = k, c, np.asarray(spin, float)

    @classmethod
    def sample(cls, rng, max_spin):
        mass = rng.uniform(0.045, 0.07)
        cd = rng.uniform(0.45, 0.75)
        k = 0.5 * RHO_AIR * cd * BALL_AREA / mass
        c = 0.5 * RHO_AIR * BALL_AREA * BALL_RADIUS / mass    # lift coefficient ~ r|w|/|v|
        axis = rng.normal(size=3)
        spin = rng.uniform(0, max_spin) * axis / np.linalg.norm(axis)
        return cls(k, c, spin)

    def accel(self, v):
        return G - self.k * np.linalg.norm(v) * v + self.c * np.cross(self.spin, v)


def rollout(p, v, aero, dt, n):
    """RK4 point-mass rollout; returns (n+1, 3) positions and velocities."""
    P, V = [p], [v]
    for _ in range(n):
        a1 = aero.accel(v)
        a2 = aero.accel(v + 0.5 * dt * a1)
        a3 = aero.accel(v + 0.5 * dt * a2)
        a4 = aero.accel(v + dt * a3)
        p = p + dt * v + dt**2 / 6 * (a1 + a2 + a3)
        v = v + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
        P.append(p)
        V.append(v)
    return np.array(P), np.array(V)


class BallisticPredictor:
    """Fits p(t) = p0 + v*(t - t0) + g*(t - t0)^2 / 2 to observations by linear least squares.

    Exact in a vacuum; biased under drag. The interface (observe / predict / ready) is what
    later stages swap a learned model into.
    """

    def __init__(self, min_obs=8):
        self.min_obs = min_obs
        self.reset()

    def reset(self):
        self.ts, self.ps = [], []
        self.p0 = self.v = None

    def observe(self, t, p):
        self.ts.append(t)
        self.ps.append(p)
        if len(self.ts) < self.min_obs:
            return
        tau = np.array(self.ts) - self.ts[0]
        A = np.stack([np.ones_like(tau), tau], axis=1)
        Y = np.array(self.ps) - 0.5 * np.outer(tau**2, G)
        sol, *_ = np.linalg.lstsq(A, Y, rcond=None)
        self.p0, self.v = sol[0], sol[1]

    @property
    def ready(self):
        return self.p0 is not None

    def predict(self, t):
        """Position and velocity at absolute time(s) t."""
        tau = np.atleast_1d(t) - self.ts[0]
        pos = self.p0 + np.outer(tau, self.v) + 0.5 * np.outer(tau**2, G)
        vel = self.v + np.outer(tau, G)
        return pos, vel


class DragPredictor:
    """Estimates the current state and a drag coefficient online, then rolls out g + drag.

    A quadratic fit over a short window gives position, velocity and acceleration now. The part of
    the acceleration opposing velocity, beyond gravity, gives k = -(a - g).v / |v|^3, averaged
    over time. Spin isn't modeled, so Magnus curvature is left as prediction error.
    """

    def __init__(self, window=15, dt=0.02, horizon=2.0):
        self.window, self.dt, self.horizon = window, dt, horizon
        self.reset()

    def reset(self):
        self.ts, self.ps = [], []
        self.ks = []
        self.state = None

    def observe(self, t, p):
        self.ts.append(t)
        self.ps.append(p)
        if len(self.ts) < self.window:
            return
        tau = np.array(self.ts[-self.window :]) - t
        A = np.stack([np.ones_like(tau), tau, 0.5 * tau**2], axis=1)
        (p_now, v_now, a_now), *_ = np.linalg.lstsq(A, np.array(self.ps[-self.window :]), rcond=None)
        speed = np.linalg.norm(v_now)
        self.ks.append(max(0.0, -(a_now - G) @ v_now / speed**3))
        self.state = (t, p_now, v_now)
        P, V = rollout(p_now, v_now, Aero(k=self.k), self.dt, round(self.horizon / self.dt))
        self.grid, self.P, self.V = t + self.dt * np.arange(len(P)), P, V

    @property
    def ready(self):
        return self.state is not None

    @property
    def k(self):
        return float(np.mean(self.ks))

    def predict(self, t):
        t = np.atleast_1d(t)
        pos = np.stack([np.interp(t, self.grid, self.P[:, i]) for i in range(3)], axis=1)
        vel = np.stack([np.interp(t, self.grid, self.V[:, i]) for i in range(3)], axis=1)
        return pos, vel


PREDICTORS = {"ballistic": BallisticPredictor, "drag": DragPredictor}

class CatchPlanner:
    """Picks the earliest predicted point the cup can reach in time, and sticks with that time."""

    def __init__(self, horizon=2.0, dt=0.01):
        self.horizon, self.dt = horizon, dt
        self.t_catch = None

    def reset(self):
        self.t_catch = None

    def _feasible(self, t_now, ts, pos, vel, cup_pos, cup_z):
        r = pos - SHOULDER
        d = np.linalg.norm(r, axis=1)
        travel = np.linalg.norm(pos - cup_pos, axis=1)
        z_des = -vel / np.linalg.norm(vel, axis=1, keepdims=True)
        turn = np.arccos(np.clip(z_des @ cup_z, -1, 1))
        return (
            (d > REACH[0]) & (d < REACH[1]) & (pos[:, 2] > MIN_CATCH_Z)
            & (vel[:, 2] < 0)                             # catch on the way down
            & (np.sum(z_des * r, axis=1) > 0)             # ball comes toward the robot: cup can face it
            & (travel <= EE_SPEED * (ts - t_now) + 0.05)
            & (turn <= CUP_TURN_RATE * (ts - t_now) + 0.2)
        )

    def plan(self, t_now, predictor, cup_pos, cup_z):
        ts = t_now + np.arange(0.05, self.horizon, self.dt)
        pos, vel = predictor.predict(ts)
        ok = self._feasible(t_now, ts, pos, vel, cup_pos, cup_z)
        if self.t_catch is not None and self.t_catch > t_now:
            # Keep the committed time unless the new prediction made it unreachable.
            k = np.argmin(np.abs(ts - self.t_catch))
            if ok[k] or self.t_catch - t_now < 0.1:
                p, v = predictor.predict(self.t_catch)
                return p[0], v[0]
        if not ok.any():
            return None
        k = np.argmax(ok)
        self.t_catch = ts[k]
        return pos[k], vel[k]


def box_qp(H, g, lo, hi, max_iter=50):
    """Solve min 0.5 x'Hx + g'x  s.t. lo <= x <= hi  (H positive definite) by a primal active set.

    Tiny and dependency-free; for 7 variables it converges in a handful of iterations.
    """
    x = np.clip(np.linalg.solve(H, -g), lo, hi)
    at_lo, at_hi = x <= lo, x >= hi
    for _ in range(max_iter):
        free = ~(at_lo | at_hi)
        x_new = x.copy()
        if free.any():
            fixed = ~free
            rhs = -g[free] - H[np.ix_(free, fixed)] @ x[fixed]
            x_new[free] = np.linalg.solve(H[np.ix_(free, free)], rhs)
        # Step toward x_new, stopping at the first bound hit by a free variable.
        d = x_new - x
        alpha, hit = 1.0, None
        for i in np.flatnonzero(free):
            if d[i] < 0 and x[i] + d[i] < lo[i]:
                a = (lo[i] - x[i]) / d[i]
            elif d[i] > 0 and x[i] + d[i] > hi[i]:
                a = (hi[i] - x[i]) / d[i]
            else:
                continue
            if a < alpha:
                alpha, hit = a, i
        x = x + alpha * d
        if hit is not None:
            x[hit] = lo[hit] if d[hit] < 0 else hi[hit]
            at_lo[hit], at_hi[hit] = d[hit] < 0, d[hit] > 0
            continue
        # Unconstrained optimum on the free set reached: release a bound with the wrong-sign multiplier.
        grad = H @ x + g
        viol = np.where(at_lo, -grad, 0.0) + np.where(at_hi, grad, 0.0)  # >0 means it wants to leave
        if viol.max(initial=0.0) <= 1e-12:
            return x
        i = int(np.argmax(viol))
        at_lo[i] = at_hi[i] = False
    return x


class DiffIK:
    """Damped least-squares differential IK on a kinematic copy of the arm.

    Tracks the cup mouth position and the cup axis direction (5 DoF) by solving a small box-constrained
    QP each step: joint speed and position limits are hard constraints, and a weak secondary term
    pulls the spare DoF toward Q_HOME.
    """

    w_rot = 0.5   # weight of cup-axis error relative to position error
    mu = 1e-2     # weight of the pull toward Q_HOME (soft null-space task)
    lam = 1e-4    # damping

    def __init__(self, model):
        self.m = model
        self.d = mujoco.MjData(model)
        self.site = model.site("cup_mouth").id
        self.nq = 7
        self.q = Q_HOME.copy()
        self.jp = np.zeros((3, model.nv))
        self.jr = np.zeros((3, model.nv))

    def reset(self, q):
        self.q = q.copy()

    def fk(self, q=None):
        self.d.qpos[: self.nq] = self.q if q is None else q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        R = self.d.site_xmat[self.site].reshape(3, 3)
        return self.d.site_xpos[self.site].copy(), R[:, 2].copy()

    def step(self, p_des, z_des, dt, kp=12.0, kr=10.0, v_max=3.0):
        p, z = self.fk()
        mujoco.mj_jacSite(self.m, self.d, self.jp, self.jr, self.site)
        Jp, Jr = self.jp[:, : self.nq], self.jr[:, : self.nq]
        # Rotation about the cup axis doesn't matter: keep only the two perpendicular rows.
        e1 = np.cross(z, [0, 0, 1.0] if abs(z[2]) < 0.9 else [1.0, 0, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(z, e1)
        J = np.vstack([Jp, e1 @ Jr, e2 @ Jr])
        v = kp * (p_des - p)
        n = np.linalg.norm(v)
        if n > v_max:
            v *= v_max / n
        w = kr * np.cross(z, z_des)
        err = np.concatenate([v, [e1 @ w, e2 @ w]])
        # QP: min |W(J qd - err)|^2 + mu |qd - qd_home|^2 + lam |qd|^2
        #     s.t. joint speed limits and "don't cross a joint limit this step".
        W2 = np.diag([1.0, 1.0, 1.0, self.w_rot**2, self.w_rot**2])
        qd_home = 2.0 * (Q_HOME - self.q)
        H = J.T @ W2 @ J + (self.mu + self.lam) * np.eye(self.nq)
        g = -(J.T @ W2 @ err + self.mu * qd_home)
        lo, hi = self.m.jnt_range[: self.nq].T
        qd_lo = np.maximum(-QD_MAX, (lo - self.q) / dt)
        qd_hi = np.minimum(QD_MAX, (hi - self.q) / dt)
        qd = box_qp(H, g, qd_lo, qd_hi)
        self.q = np.clip(self.q + qd * dt, lo, hi)  # guards only against round-off
        return self.q


def sample_throw(rng, aero):
    """Ball start state whose path (under this throw's true aerodynamics) passes near the arm."""
    p0 = np.array([rng.uniform(4.0, 6.0), rng.uniform(-1.5, 1.5), rng.uniform(0.8, 1.6)])
    r = rng.uniform(0.45, 0.9)
    az = rng.uniform(-1.0, 1.0)
    target = SHOULDER + np.array([r * np.cos(az), r * np.sin(az), rng.uniform(-0.15, 0.45)])
    T = rng.uniform(0.9, 1.3)
    v0 = (target - p0 - 0.5 * G * T**2) / T
    dt = 0.005
    n = round(T / dt)
    for _ in range(20):  # shooting: correct v0 by the miss
        P, _ = rollout(p0, v0, aero, dt, n)
        v0 = v0 + (target - P[-1]) / T
    return p0, v0


class Episode:
    def __init__(self, model, predictor, rng, noise_std=0.0, blind_radius=0.0):
        self.blind_radius = blind_radius
        self.m, self.d = model, mujoco.MjData(model)
        self.rng = rng
        self.noise_std = noise_std
        self.predictor = predictor
        self.ball = model.body("ball").id
        self.planner = CatchPlanner()
        self.ik = DiffIK(model)
        self.ball_q = model.joint("ball").qposadr[0]
        self.ball_v = model.joint("ball").dofadr[0]
        self.cup = model.body("cup").id
        self.mouth = model.site("cup_mouth").id
        self.bottom = model.site("cup_bottom").id

    def reset(self, p0, v0, aero):
        self.aero = aero
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:7] = Q_HOME
        self.d.ctrl[:] = Q_HOME
        self.d.qpos[self.ball_q : self.ball_q + 3] = p0
        self.d.qvel[self.ball_v : self.ball_v + 3] = v0
        mujoco.mj_forward(self.m, self.d)
        self.p0, self.v0 = p0, v0
        self.predictor.reset()
        self.planner.reset()
        self.ik.reset(Q_HOME)
        self.log = []   # (t, t_catch, predicted catch point) for error analysis
        self.caught = None
        self.track = []  # (t, true simulated ball position) at control ticks

    def true_ball(self, t):
        # Simulated position, not the exact parabola: the integrator drifts from it by ~g*dt*t/2.
        ts, ps = zip(*self.track)
        return np.array([np.interp(t, ts, np.array(ps)[:, i]) for i in range(3)])

    def ball_in_cup(self):
        R = self.d.xmat[self.cup].reshape(3, 3)
        rel = R.T @ (self.d.qpos[self.ball_q : self.ball_q + 3] - self.d.site_xpos[self.bottom])
        return np.hypot(rel[0], rel[1]) < CUP_RADIUS and -0.01 < rel[2] < CUP_DEPTH + BALL_RADIUS

    def control(self):
        t = self.d.time
        self.track.append((t, self.d.qpos[self.ball_q : self.ball_q + 3].copy()))
        p_obs = self.track[-1][1] + self.rng.normal(0, self.noise_std, 3)
        visible = np.linalg.norm(self.track[-1][1] - SHOULDER) > self.blind_radius
        if visible and (self.planner.t_catch is None or t < self.planner.t_catch + 0.05):
            self.predictor.observe(t, p_obs)  # stop listening once the ball is (supposedly) in the cup
        p_des, z_des = None, None
        t_c = self.planner.t_catch
        if t_c is not None and t >= t_c:
            # Catch phase: "give" by retreating along the ball's path, then tilt upright.
            if self.caught is None:
                self.caught = (self.last_target[0], self.last_target[1] / np.linalg.norm(self.last_target[1]))
            p_c, v_hat = self.caught
            s = min((t - t_c) / GIVE_TIME, 1.0)
            p_des = p_c + v_hat * GIVE_DIST * (1 - (1 - s) ** 2)
            z_des = -v_hat if t < t_c + 2 * GIVE_TIME else np.array([0.0, 0.0, 1.0])
        elif self.predictor.ready:
            cup_pos, cup_z = self.ik.fk()
            target = self.planner.plan(t, self.predictor, cup_pos, cup_z)
            if target is not None:
                p_c, v_c = target
                self.last_target = target
                p_des = p_c
                z_des = -v_c / np.linalg.norm(v_c)
                self.log.append((t, self.planner.t_catch, p_c))
        if p_des is None:
            p_des, _ = self.ik.fk(Q_HOME)
            z_des = self.ik.fk(Q_HOME)[1]
        self.d.ctrl[:] = self.ik.step(p_des, z_des, 1.0 / CTRL_HZ)

    def aero_accel(self):
        """Non-gravity aerodynamic acceleration on the ball right now."""
        return self.aero.accel(self.d.qvel[self.ball_v : self.ball_v + 3]) - G

    def draw_overlay(self, scn):
        """Ball trail (yellow), current predicted path (blue), planned catch point (green)."""
        def dot(p, r, rgba):
            if scn.ngeom >= scn.maxgeom:
                return
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                [r, 0, 0], np.asarray(p, float), np.eye(3).flatten(), np.asarray(rgba, np.float32))
            scn.ngeom += 1
        for _, p in self.track[::3]:
            dot(p, 0.018, [1, 0.85, 0.1, 0.9])
        t = self.d.time
        if self.predictor.ready and (self.planner.t_catch is None or t < self.planner.t_catch):
            end = self.planner.t_catch if self.planner.t_catch else t + 1.5
            pos, _ = self.predictor.predict(np.arange(t, end, 0.02))
            for p in pos:
                dot(p, 0.022, [0.2, 0.5, 1, 0.9])
        if self.planner.t_catch is not None and self.log:
            dot(self.log[-1][2], CUP_RADIUS, [0.1, 0.9, 0.2, 0.35])

    def post_step(self):
        """Hook for extra physics after each step (stage 2 uses it for wall bounces)."""

    def run(self, duration=2.5, viewer=None, frames=None, renderer=None):
        sub = round(1.0 / (CTRL_HZ * self.m.opt.timestep))
        steps = int(duration / self.m.opt.timestep)
        for i in range(steps):
            if i % sub == 0:
                self.control()
            self.d.xfrc_applied[self.ball, :3] = self.m.body_mass[self.ball] * self.aero_accel()
            mujoco.mj_step(self.m, self.d)
            self.post_step()
            if viewer is not None and i % 8 == 0:
                with viewer.lock():
                    viewer.user_scn.ngeom = 0
                    self.draw_overlay(viewer.user_scn)
                viewer.sync()
                time.sleep(8 * self.m.opt.timestep)
            if renderer is not None and i % 17 == 0:  # ~30 fps
                renderer.update_scene(self.d, camera=frames["cam"])
                self.draw_overlay(renderer.scene)
                frames["img"].append(renderer.render())
        return self.ball_in_cup()


def run_trials(model, name, args, viewer=None, frames=None, renderer=None):
    rng = np.random.default_rng(args.seed)
    ep = Episode(model, PREDICTORS[name](), np.random.default_rng(args.seed + 1), args.noise, args.blind)
    results, errors = [], []
    for k in range(args.trials):
        aero = Aero.sample(rng, args.spin) if not args.vacuum else Aero()
        p0, v0 = sample_throw(rng, aero)
        ep.reset(p0, v0, aero)
        ok = ep.run(viewer=viewer, frames=frames, renderer=renderer)
        results.append(ok)
        t_c = ep.planner.t_catch
        if t_c is not None:
            truth = ep.true_ball(t_c)
            for t, tc, p in ep.log:
                if abs(tc - t_c) < 1e-9:
                    errors.append((t_c - t, np.linalg.norm(p - truth)))
        if args.verbose:
            print(f"  [{name}] trial {k:3d}: {'CATCH' if ok else 'drop '}  k={aero.k:.4f}  "
                  f"spin={np.linalg.norm(aero.spin):5.1f}  t_catch={t_c if t_c is None else round(t_c, 2)}")
    return np.array(results), np.array(errors)


def plot_errors(all_err, title, path):
    """all_err: {predictor name: (catch rate, array of (time before catch, error))}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name, (rate, e) in all_err.items():
        if not len(e):
            continue
        bins = np.linspace(0, e[:, 0].max(), 15)
        idx = np.digitize(e[:, 0], bins)
        keep = [i for i in range(1, len(bins)) if (idx == i).any()]
        mid = [(bins[i - 1] + bins[i]) / 2 for i in keep]
        med = [np.median(e[idx == i, 1]) * 1000 for i in keep]
        ax.plot(mid, med, "o-", label=f"{name} (catch rate {rate:.0%})")
    ax.axhline(CUP_RADIUS * 1000, color="gray", lw=0.8, label="cup radius")
    ax.set_xlabel("time before catch (s)")
    ax.set_ylabel("median catch-point error (mm)")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--predictors", type=str, default="ballistic,drag")
    ap.add_argument("--spin", type=float, default=150.0, help="max random spin, rad/s (Magnus force)")
    ap.add_argument("--blind", type=float, default=2.5,
                    help="ball is unobservable within this distance of the shoulder (arm occludes it), m")
    ap.add_argument("--vacuum", action="store_true", help="turn off aerodynamics")
    ap.add_argument("--noise", type=float, default=0.0, help="optional observation noise std, m")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--view", action="store_true", help="interactive viewer (run with mjpython on macOS)")
    ap.add_argument("--video", type=str, default=None, help="write an mp4 of the episodes")
    ap.add_argument("--plot", type=str, default="stage1_results.png")
    args = ap.parse_args()

    model = load_model()
    viewer = renderer = frames = None
    if args.view:
        viewer = mujoco.viewer.launch_passive(model, mujoco.MjData(model))
    if args.video:
        renderer = mujoco.Renderer(model, 720, 1280)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [2.2, 0, 0.8]
        cam.distance, cam.azimuth, cam.elevation = 5.5, -125, -15
        frames = {"cam": cam, "img": []}

    cond = "vacuum" if args.vacuum else f"drag" + (f" + spin<={args.spin:.0f} rad/s" if args.spin else "")
    all_err = {}
    for name in args.predictors.split(","):
        if frames is not None:
            frames["img"] = []
        results, errors = run_trials(model, name, args, viewer, frames, renderer)
        if args.video:
            import imageio.v3 as iio
            path = args.video.replace(".mp4", f"_{name}.mp4")
            iio.imwrite(path, np.stack(frames["img"]), fps=30)
            print("wrote", path)
        all_err[name] = (results.mean(), errors)
        final = errors[errors[:, 0] < 0.05, 1] if len(errors) else np.array([np.nan])
        print(f"{name:>9}: catch rate {results.sum()}/{len(results)} = {results.mean():.0%}   "
              f"median catch-point error at catch {np.median(final)*1000:.0f} mm   [{cond}]")

    if args.plot:
        plot_errors(all_err, f"Stage 1 catch-point prediction, {cond}", args.plot)

if __name__ == "__main__":
    main()
