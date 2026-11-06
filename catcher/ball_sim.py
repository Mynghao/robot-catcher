"""Event-driven point-ball simulation in NumPy.

Between bounces the ball follows exact ballistic flight
    p(t) = p0 + v0 t + 1/2 g t^2,
so the time of each panel crossing is the root of a quadratic and bounces are
resolved exactly (no tunnelling, no time-step dependence). At a bounce:
    v_n' = -e v_n,      v_t' = (1 - friction) v_t.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .room import Room

GRAVITY = np.array([0.0, 0.0, -9.81])
REST_SPEED = 0.05  # normal speed (m/s) below which a bounce ends the trajectory
_EPS_T = 1e-9


@dataclass
class Bounce:
    t: float
    position: np.ndarray
    panel_index: int
    v_in: np.ndarray
    v_out: np.ndarray


@dataclass
class Segment:
    """Ballistic flight starting at time t0 from (p0, v0)."""

    t0: float
    p0: np.ndarray
    v0: np.ndarray

    def state(self, t: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        tau = np.asarray(t, dtype=float)[..., None] - self.t0
        return self.p0 + self.v0 * tau + 0.5 * GRAVITY * tau**2, self.v0 + GRAVITY * tau


@dataclass
class BallTrajectory:
    segments: list[Segment]
    bounces: list[Bounce]
    t_end: float
    at_rest: bool  # ball stopped bouncing (on the floor) before t_end

    def sample(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Positions and velocities at the given times, each (len(times), 3).

        After the ball comes to rest it stays put with zero velocity.
        """
        times = np.asarray(times, dtype=float)
        starts = np.array([s.t0 for s in self.segments])
        seg_idx = np.clip(np.searchsorted(starts, times, side="right") - 1, 0, None)
        pos = np.empty((len(times), 3))
        vel = np.empty((len(times), 3))
        for i, seg in enumerate(self.segments):
            m = seg_idx == i
            if m.any():
                pos[m], vel[m] = seg.state(np.minimum(times[m], self.t_end))
        if self.at_rest:
            m = times >= self.t_end
            vel[m] = 0.0
        return pos, vel

    def num_bounces_before(self, t: float) -> int:
        return sum(b.t <= t for b in self.bounces)


def _first_hit(room: Room, p0: np.ndarray, v0: np.ndarray, t_max: float,
               skip_panel: int | None) -> tuple[float, int] | None:
    """Earliest (dt, panel_index) at which the flight from (p0, v0) crosses a panel."""
    best = None
    for k, panel in enumerate(room.panels):
        n = panel.normal
        d0 = n.dot(p0 - panel.center)
        dv = n.dot(v0)
        a = 0.5 * n.dot(GRAVITY)
        # Side of the (two-sided) panel we start on. Right after a bounce d0 ~ 0,
        # so use the direction of motion instead.
        side = np.sign(d0) if abs(d0) > 1e-9 else np.sign(dv)
        if side == 0:
            continue
        if abs(a) < 1e-12:
            roots = [-d0 / dv] if abs(dv) > 1e-12 else []
        else:
            disc = dv * dv - 4 * a * d0
            if disc < 0:
                continue
            sq = np.sqrt(disc)
            roots = sorted([(-dv - sq) / (2 * a), (-dv + sq) / (2 * a)])
        for t in roots:
            if t <= _EPS_T or t > t_max or (best is not None and t >= best[0]):
                continue
            # Must be approaching the panel from our side at the crossing.
            if side * (dv + 2 * a * t) >= 0:
                continue
            if k == skip_panel and t < 1e-6:
                continue
            p = p0 + v0 * t + 0.5 * GRAVITY * t * t
            if panel.contains(p):
                best = (t, k)
                break
    return best


def simulate(room: Room, p0, v0, t_end: float = 3.0, max_bounces: int = 50) -> BallTrajectory:
    """Simulate a point ball from (p0, v0) at t=0 until t_end."""
    p = np.asarray(p0, dtype=float)
    v = np.asarray(v0, dtype=float)
    t = 0.0
    segments = [Segment(t, p.copy(), v.copy())]
    bounces: list[Bounce] = []
    last_panel = None
    while len(bounces) < max_bounces:
        hit = _first_hit(room, p, v, t_end - t, last_panel)
        if hit is None:
            return BallTrajectory(segments, bounces, t_end, at_rest=False)
        dt, k = hit
        p, v_in = segments[-1].state(t + dt)
        t += dt
        panel = room.panels[k]
        n = panel.normal
        vn = n.dot(v_in) * n
        vt = v_in - vn
        v = -panel.material.restitution * vn + (1.0 - panel.material.friction) * vt
        # Snap exactly onto the plane to avoid drift through it.
        p = p - n.dot(p - panel.center) * n
        bounces.append(Bounce(t, p.copy(), k, v_in, v.copy()))
        if abs(n.dot(v)) < REST_SPEED and n[2] > 0.7:
            return BallTrajectory(segments, bounces, t, at_rest=True)
        segments.append(Segment(t, p.copy(), v.copy()))
        last_panel = k
    return BallTrajectory(segments, bounces, t, at_rest=True)
