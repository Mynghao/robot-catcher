# `stage1.py` Explained: Catching a Thrown Ball with a Cup on a 7-DoF Arm

## Overview

`stage1.py` simulates a 7-joint robot arm in MuJoCo. A cup is mounted on the end of the arm, and the arm tries to catch a ball thrown toward it. There are no bounces in this stage.

The catch is hard because **the ball's aerodynamics are hidden from the robot**. On every throw the simulator picks random values for:
- the ball's **mass** and **drag coefficient**, which give quadratic air drag
- optionally, a random **spin**, which produces a sideways Magnus force

The robot only sees the ball's position over time. From that it has to predict where the ball will be, pick a point to catch it, and move the cup there in time.

### Control pipeline (runs at `CTRL_HZ` = 100 Hz)

```
ball position (observed)
   │
   ▼
Predictor ── BallisticPredictor (gravity only)
   │       └ DragPredictor     (estimates drag online)
   ▼
CatchPlanner ── earliest reachable point on the predicted path
   │
   ▼
DiffIK ── joint motion that puts the cup mouth there, facing the ball
   │
   ▼
joint position commands → arm actuators
```

---

## 1. Constants

| Constant | Value | Meaning |
|---|---|---|
| `G` | `[0,0,-9.81]` | Gravity |
| `CTRL_HZ` | 100 | Controller rate (Hz) |
| `Q_HOME` | 7 joint angles | Rest pose of the arm |
| `REACH` | (0.35, 0.95) m | Allowed distance between the catch point and the shoulder |
| `MIN_CATCH_Z` | 0.25 m | Lowest allowed catch height |
| `EE_SPEED` | 1.8 m/s | Cup speed the planner assumes the arm can reach |
| `QD_MAX` | 2.5 rad/s | Joint speed limit |
| `CUP_TURN_RATE` | 4.0 rad/s | Cup turning speed the planner assumes |
| `GIVE_DIST`, `GIVE_TIME` | 0.0 m, 0.15 s | "Give" motion during the catch (moving the cup back along the ball's path). It is turned off because `GIVE_DIST` is 0. |
| `RHO_AIR`, `BALL_AREA` | — | Air density and ball cross-section, used to compute drag |

`BALL_RADIUS`, `CUP_DEPTH`, `CUP_RADIUS`, `SHOULDER` and `load_model` are imported from `scene.py`.

---

## 2. True physics: `Aero` and `rollout`

### `Aero`
This class holds the ball's real aerodynamics for one throw. It computes the acceleration as:

$$a = g \;-\; k\,|v|\,v \;+\; c\,(\omega \times v)$$

- `k` is the drag constant: `0.5 · ρ · Cd · A / m`
- `c` is the Magnus (lift) constant, and `ω` is the spin vector
- `Aero.sample(rng, max_spin)` draws a random mass (45–70 g), a random `Cd` (0.45–0.75), and a spin vector with a random direction and a speed of up to `max_spin`

The predictors never see this object. Only the simulator uses it.

### `rollout(p, v, aero, dt, n)`
This function integrates a point mass forward in time with **4th-order Runge-Kutta (RK4)** and returns arrays of positions and velocities. Two parts of the code use it:
- `sample_throw`, to aim throws
- `DragPredictor`, to predict the ball's future path

---

## 3. Predictors

Both predictors have the same interface: `reset()`, `observe(t, p)`, `ready`, and `predict(t) → (pos, vel)`. Because the interface is shared, later stages can plug in other models, such as learned ones.

### `BallisticPredictor`
- Assumes **gravity only**: `p(t) = p0 + v·τ + ½·g·τ²`
- Once it has `min_obs` (8) observations, it finds `p0` and `v` with **linear least squares** over all observations. Gravity is known, so it is subtracted from the data before the fit.
- This is exact in a vacuum. With drag, the prediction is **biased**: the real ball falls short.

### `DragPredictor`
- Fits a **local quadratic** (position, velocity, acceleration) to the last `window` (15) observations to estimate the ball's current state.
- Estimates the drag constant from whatever acceleration is left after removing gravity, in the direction opposite the velocity:
  $$k = \frac{-(a - g)\cdot v}{|v|^3}$$
  It averages these estimates over time.
- Rolls out a 2 s future path with `rollout` and that `k`, and stores the path on a time grid. `predict` then interpolates along that grid.
- Spin is **not modeled**, so any curve caused by the Magnus force shows up as prediction error.

---

## 4. `CatchPlanner`

The planner chooses **when and where** to catch the ball.

1. It samples the predicted path every 10 ms, from 0.05 s to 2 s ahead.
2. `_feasible` keeps only the points that pass all of these checks:
   - distance from the shoulder is within `REACH`, and height is above `MIN_CATCH_Z`
   - the ball is **falling** (`vz < 0`)
   - the ball is moving **toward the robot**, so the cup can face it
   - the cup can **travel** there in time: distance ≤ `EE_SPEED · Δt + 5 cm`
   - the cup can **turn** to face the ball in time: angle ≤ `CUP_TURN_RATE · Δt + 0.2 rad`
3. It picks the **earliest** point that passes and stores `t_catch`.
4. **Commitment:** on later ticks it keeps the same `t_catch` as long as that time is still feasible, or if the catch is less than 0.1 s away. It only updates the predicted position for that time. This keeps the target from jumping around.

`plan` returns the predicted `(position, velocity)` at the catch time, or `None` if no point is feasible.

---

## 5. `DiffIK`: differential inverse kinematics

`DiffIK` converts "put the cup here, pointing this way" into joint motion. It works on its own kinematic copy of the arm (a separate `MjData`).

- `fk(q)` returns the cup mouth position and the cup's axis direction (`z`).
- `step(p_des, z_des, dt)` does the following:
  - Builds a **5-row Jacobian**: 3 rows for position, plus 2 rows for rotation about the axes perpendicular to the cup axis. Spinning the cup about its own axis doesn't matter, so that direction is left out.
  - Computes the desired motion as a P-controller: linear velocity `kp·(p_des − p)`, capped at `v_max`, and angular velocity `kr·(z × z_des)`.
  - Solves for joint velocities with a **damped least-squares pseudo-inverse**: `Jᵀ(JJᵀ + λI)⁻¹`.
  - Uses the 2 leftover degrees of freedom to pull the arm toward `Q_HOME` (null-space projection).
  - Clips joint velocities to `QD_MAX` and joint angles to their limits, integrates, and returns the new joint targets.

---

## 6. `sample_throw`

This function creates a random throw that **passes near the arm under that throw's real aerodynamics**:
1. Picks a random start point 4–6 m away and a random target near the shoulder, plus a flight time `T` of 0.9–1.3 s.
2. Computes an initial velocity using the vacuum (gravity-only) formula.
3. Corrects that velocity with **20 iterations of shooting**: it simulates the throw with `rollout`, measures how far it misses the target, and adjusts `v0` by `miss / T`.

---

## 7. `Episode`: one simulated throw

`Episode` connects all the pieces inside the MuJoCo simulation.

- **`reset`** places the arm at `Q_HOME`, sets the ball's start position and velocity, and clears the predictor, planner, IK and logs.
- **`control`**, called at 100 Hz:
  1. Records the true ball position and adds optional noise to make the observation.
  2. **Occlusion:** if the ball is within `blind_radius` of the shoulder, the robot can't see it, because the arm blocks the view. The default of 2.5 m means the robot only sees the early part of the flight. Observations also stop shortly after `t_catch`.
  3. **Catch phase** (`t ≥ t_catch`): the cup holds the last target, optionally moves back along the ball's path ("give"), then tilts upright to hold the ball.
  4. **Tracking phase:** asks the planner for a target and points the cup's opening against the ball's incoming velocity (`z_des = −v̂`).
  5. If there is no target, the arm goes back to its home pose.
  6. Sends the result of `DiffIK.step` to the actuators as `ctrl`.
- **`run`** steps the physics. On every step it applies the hidden aerodynamic force to the ball through `xfrc_applied`. It calls `control` once per control tick, can show the interactive viewer or record video frames (about 30 fps), and returns `ball_in_cup()`.
- **`ball_in_cup`** converts the ball position into the cup's frame and checks that the ball is inside the cup's cylinder.
- **`draw_overlay`** draws the ball's trail (yellow), the current predicted path (blue) and the planned catch point (green).
- **`post_step`** does nothing here. It is a hook that Stage 2 uses for wall bounces.

---

## 8. Experiments and output

### `run_trials`
Runs `args.trials` throws with one predictor. For each throw it:
- samples an `Aero` (or uses a vacuum) and a throw
- runs the episode and records whether the ball was caught
- compares every logged prediction of the final catch point with the ball's **true** position at `t_catch`, giving pairs of `(time before catch, error)`

### `plot_errors`
Plots the **median catch-point error (mm, log scale)** against the **time before the catch**, with one line per predictor and the cup radius as a reference line. The output goes to `stage1_results.png`.

### `main` and command-line options

| Flag | Default | Effect |
|---|---|---|
| `--trials` | 30 | Number of throws per predictor |
| `--predictors` | `ballistic,drag` | Predictors to compare |
| `--spin` | 150 | Maximum spin (rad/s) |
| `--blind` | 2.5 | Distance from the shoulder within which the ball can't be seen (m) |
| `--vacuum` | off | Turns off aerodynamics |
| `--noise` | 0 | Standard deviation of observation noise (m) |
| `--seed` | 0 | Random seed |
| `--verbose` | off | Prints the result of each trial |
| `--view` | off | Interactive viewer (use `mjpython` on macOS) |
| `--video` | — | Saves an mp4 for each predictor, e.g. `stage1_demo_drag.mp4` |
| `--plot` | `stage1_results.png` | Path for the error plot |

The script prints the catch rate and the median catch-point error at the moment of the catch for each predictor.

Examples:
```bash
.venv/bin/python stage1.py --trials 30 --spin 150 --blind 2.5
.venv/bin/python stage1.py --trials 1 --video catch.mp4
.venv/bin/mjpython stage1.py --trials 5 --view
```

---

## Takeaways

- **Ballistic vs. drag predictor:** the gravity-only model's error grows with drag. The online drag estimate removes most of that bias.
- **Spin isn't modeled** by either predictor, so the Magnus force puts a floor under the prediction error.
- **Occlusion (`--blind`)** makes the robot commit to a prediction early, so prediction quality decides whether the catch succeeds.
- Because the **predictor interface is modular**, Stage 2 (`stage2.py`) and later stages can swap in new models and add effects like wall bounces through `post_step`.
