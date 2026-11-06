# Trajectory Prediction Model — Architecture Spec & Stage 1 (No-Bounce) Plan

**Project:** Ball Catching in General Room Geometries via Learning Ball Trajectory
**Scope of this document:** (1) the model architecture, (2) generating a first, simple training set of *no-bounce* parabolic throws inside rooms, and (3) a staged sanity-check plan whose goal is to confirm the network can learn (and overfit) a parabola with the full input pipeline in place.

The simulator already exists. This plan only says *what* to generate and *how* to package it, not how to simulate.

---

## 0. Goal of this stage

Train the full architecture (wall encoder, trajectory decoder, Gaussian output head) on trajectories that never touch a wall. The room geometry is still fed to the model, even though it is irrelevant to the motion here.

Success means:
1. The model can **overfit** a tiny dataset (confirms the code is wired correctly).
2. It **generalizes** to held-out throws and held-out rooms for pure parabolic flight.
3. It learns to **ignore** wall geometry when no bounce happens.
4. Its predicted **uncertainty is calibrated** once observation noise is added.

Bounces come later (Stage 2). Nothing in this stage should be bounce-specific, so the same code carries over.

---

## 1. Conventions (use everywhere)

| Symbol | Meaning | Default |
|---|---|---|
| Frame | Robot base frame, $z$ up. Walls and ball are expressed in the **same** frame. | — |
| Units | SI (meters, seconds) in stored data | — |
| $s$ | Global length scale for normalization. **One constant for all rooms**, never per room, because gravity must keep the same value in normalized units. | 3.0 m |
| $\Delta t$ | Observation / simulation output timestep | 0.01 s (100 Hz) |
| $C$ | Chunk length (time steps per token) | 5 (= 50 ms) |
| $K$ | Max history chunks given to the model | 10 (= 0.5 s) |
| $N$ | Future chunks to predict | 10 (= 0.5 s) |
| $\mathbf{g}$ | Gravity | $(0, 0, -9.81)$ m/s² |
| $r$ | Ball radius | match simulator |
| $\sigma_{\text{obs}}$ | Std. dev. of Gaussian observation noise added to ball positions | 0 at first, then 0.005 m |

All of these should live in a single config file.

---

## 2. Data generation (Stage 1: free flight, no bounces)

### 2.1 Rooms

- **Shape:** axis-aligned box rooms (floor, ceiling, 4 walls) for now.
  - Width and depth: uniform in [4, 10] m.
  - Height: uniform in [3, 6] m.
  - Optional: 0–2 box obstacles at random positions, so the geometry input is non-trivial. Throws must still not touch them.
- **Pose:** random rotation about $z$ only, and a random translation, such that the robot base lies inside the room with at least 1 m clearance from every wall.
- **Point cloud:** sample $P = 4096$ points on all room surfaces (and obstacles). Sample proportionally to surface area, so large walls get more points.
  - Store the points ($P \times 3$), the per-point surface normals ($P \times 3$, optional input feature), and the room parameters.
  - Use a fixed $P$ for every room so batching is simple.

### 2.2 Throws

For each room, sample throws by rejection sampling:

1. **Launch position** $\mathbf{p}_0$: uniform inside the room, at least 0.5 m from every surface, height in [0.5, 2.5] m.
2. **Launch velocity** $\mathbf{v}_0$, built from three sampled quantities:
   - speed uniform in [3, 10] m/s,
   - azimuth uniform in [0, 2π),
   - elevation uniform in [−10°, 60°].
3. **Simulate** with the existing simulator, with **air drag disabled**, at output rate $\Delta t$, for the duration $T_{\text{total}} = (K + N)\,C\,\Delta t$ (1.0 s by default).
4. **Reject** the throw if, at any time in $[0, T_{\text{total}}]$, the ball comes within $r + 0.05$ m of any surface (walls, floor, ceiling, obstacles). This guarantees no bounces.
5. **Log** the acceptance rate. If it is below ~20%, shrink the speed range or bias launch directions toward the room's long axis.

### 2.3 Ground-truth check (run once, before training)

For every accepted trajectory, fit $\mathbf{p}(t) = \mathbf{p}_0 + \mathbf{v}_0 t + \tfrac12 \mathbf{g} t^2$ and confirm the maximum residual is below 1 mm. Here $\mathbf{p}(t)$ is the ball position at time $t$.

This verifies that the simulator output really is a drag-free parabola. If the check fails, the overfit test below is meaningless.

### 2.4 Dataset sizes and splits

| Split | Rooms | Throws per room | Purpose |
|---|---|---|---|
| `overfit` | 1 | 32 | Wiring check (E1) |
| `train` | 50 | 200 | Main training (10k trajectories) |
| `val_throws` | same 50 | 20 | Held-out throws, seen rooms |
| `test_rooms` | 10 new | 50 | Held-out rooms |

### 2.5 Storage

- **One file per room:** points, normals, room parameters, room ID.
- **One record per trajectory:**
  - clean positions ($T \times 3$, in meters),
  - timestamps,
  - room ID,
  - launch $\mathbf{p}_0$ and $\mathbf{v}_0$,
  - split label.

Store **clean** positions only. Observation noise is added on the fly during training, so $\sigma_{\text{obs}}$ can be changed without regenerating data. Use any convenient array format (e.g. `.npz` or HDF5).

---

## 3. Building training samples (windowing)

Each training example is built from one trajectory by choosing a "current time" index $t_{\text{now}}$:

1. **History:** take the observed positions up to and including $t_{\text{now}}$. Chunk them **backward from $t_{\text{now}}$** into at most $K$ chunks of $C$ steps, so the newest chunk always ends exactly at $t_{\text{now}}$.
   - Allow *variable* history length: anywhere from 2 to $K$ chunks.
   - Left-pad to $K$ chunks and pass a padding mask.
   - This mimics early-in-flight predictions, when little history exists.
2. **Future:** the next $N \cdot C$ positions after $t_{\text{now}}$, chunked forward into $N$ chunks.
3. **Noise:** add i.i.d. $\mathcal{N}(0, \sigma_{\text{obs}}^2)$ noise to the history positions only, never to the targets. Here i.i.d. means independent and identically distributed, and $\mathcal{N}(0, \sigma^2)$ is a zero-mean Gaussian with variance $\sigma^2$.
4. **Normalize:** divide all positions (history, targets, wall points) by $s$. No per-room centering, because everything is already in the robot base frame.
5. **Targets:** express every future position relative to the **last observed history position** $\mathbf{p}_{\text{now}}$:
$$\mathbf{y}_{i,j} = \frac{\mathbf{p}_{i,j} - \mathbf{p}_{\text{now}}}{s}$$
   where $i$ is the future chunk index and $j$ is the step within the chunk.

Sample $t_{\text{now}}$ uniformly from all valid indices on each epoch. Each trajectory then yields many different windows.

---

## 4. Architecture

Total size is ~1.5M parameters at width $d = 128$.

### 4.1 Wall encoder (run once per room; cacheable)

1. **Patch centers:** farthest-point sampling picks $M = 64$ centers from the $P$ points.
2. **Grouping:** for each center, take its $m = 32$ nearest points. Express them relative to the center (and append normals if used).
3. **Patch embedding:** a shared per-point MLP (3 or 6 → 64 → 128 → $d$), followed by a max-pool over the 32 points, gives one vector per patch. This is a mini PointNet.
4. **Position embedding:** add an MLP embedding of the patch center's (normalized) coordinates.
5. **Context:** 2 transformer encoder layers (self-attention among wall tokens, pre-LayerNorm).

**Output:** wall tokens $\mathbf{w}_1, \dots, \mathbf{w}_M \in \mathbb{R}^d$.

### 4.2 Chunk tokens

For each history chunk (and later, each predicted chunk), concatenate:
- the chunk's $C$ positions relative to the chunk's first position, flattened ($3C$ values),
- the chunk's first position in normalized coordinates (3 values),
- the chunk's time offset relative to $t_{\text{now}}$ (1 value).

An MLP maps this vector to $d$. Then add a learned embedding of the chunk index.

### 4.3 Trajectory decoder

$L = 4$ pre-LayerNorm blocks with width $d = 128$ and 4 heads. Each block applies, in order and with residual connections, to every chunk token $\mathbf{h}$:
$$
\begin{aligned}
\mathbf{h} &\leftarrow \mathbf{h} + \text{CausalSelfAttn}(\mathrm{LN}(\mathbf{h})) \\
\mathbf{h} &\leftarrow \mathbf{h} + \text{CrossAttn}(\mathrm{LN}(\mathbf{h}),\ \{\mathbf{w}_m\}) \\
\mathbf{h} &\leftarrow \mathbf{h} + \text{MLP}(\mathrm{LN}(\mathbf{h}))
\end{aligned}
$$
- $\mathrm{LN}$ is LayerNorm.
- **CausalSelfAttn:** a chunk token attends only to itself and earlier chunks. Padding is masked out.
- **CrossAttn:** queries come from chunk tokens; keys and values come from all $M$ wall tokens. Wall tokens never attend to chunk tokens.
- **MLP:** hidden width $4d$, GELU activation.

### 4.4 Output head

A linear layer maps each final chunk token to $6C$ numbers for the **next** chunk:
- a mean $\boldsymbol{\mu}_{j} \in \mathbb{R}^3$ for each step $j = 1, \dots, C$, as a normalized offset from $\mathbf{p}_{\text{now}}$ (same convention as the targets $\mathbf{y}$), and
- a log-variance $\log \boldsymbol{\sigma}^2_{j} \in \mathbb{R}^3$, clamped to [−14, 2] for stability.

During training, every token predicts its own next chunk (teacher forcing), so one forward pass yields up to $K$ predictions. During inference, only the last token's prediction is used.

> **Offset convention for chunks after the first.** When a predicted chunk is appended and the model predicts the chunk after it, keep $\mathbf{p}_{\text{now}}$ fixed to the last *observed* position for the whole rollout. Every predicted offset is then in the same reference frame.

### 4.5 Physics residual: OFF for this stage

The architecture should support an optional flag that adds a ballistic extrapolation to the predicted mean. **Keep it disabled in Stage 1.** With it on, the network's job becomes trivially "output zero" for parabolas, which would defeat the purpose of this test. It may be turned on in later stages.

---

## 5. Training

- **Loss:** Gaussian negative log-likelihood, summed over steps $j$, axes $d \in \{x,y,z\}$, and all predicted chunks, then averaged over the batch:
$$
\mathcal{L} = \sum_{i,j,d}\left[\frac{(y_{i,j,d} - \mu_{i,j,d})^2}{2\sigma_{i,j,d}^2} + \tfrac12\log\sigma_{i,j,d}^2\right]
$$
  Here $y_{i,j,d}$ is the target coordinate, and $\mu_{i,j,d}$, $\sigma^2_{i,j,d}$ are the predicted mean and variance for future chunk $i$, step $j$, axis $d$.
- **Optional warm-up:** for the first ~1k steps, use plain MSE on $\boldsymbol{\mu}$ only. This avoids an early collapse where the model inflates variance instead of fitting the mean.
- **Training rollout:**
  - Teacher forcing, as above.
  - For the future chunks beyond the first, also feed the **true** future chunks as decoder tokens (standard teacher forcing over the whole window).
  - Multi-step unrolling and input-noise injection are **off** for this stage.
- **Optimizer:** AdamW, learning rate 3e-4, weight decay 0.01, 1k warmup steps then cosine decay, batch size 256, gradient clipping at 1.0.
- **Logging:**
  - NLL,
  - mean-only RMSE in meters (un-normalized),
  - average predicted $\sigma$ in meters,
  - all three per split.

---

## 6. Inference (needed for evaluation)

Implement two rollout modes, both autoregressive over $N$ chunks:

1. **Mean rollout:** append the predicted mean chunk and repeat. This gives one trajectory.
2. **Sampled rollout:** run $S$ copies as a batch (default $S = 64$). At each step, sample the next chunk from the predicted Gaussian, append it, and repeat. This gives $S$ trajectories.

Cache the wall tokens per room. Caching attention keys/values for the decoder is optional at this stage.

---

## 7. Experiments, in order

| ID | Setup | What to check | Pass criterion |
|---|---|---|---|
| **E0** | Data only | Parabola-fit residual; no surface contact; plot 20 random throws inside their rooms | Residual < 1 mm; zero contacts |
| **E1** | `overfit` split, $\sigma_{\text{obs}} = 0$ | Train until loss plateaus | Teacher-forced mean RMSE < 2 mm; training NLL keeps decreasing (σ shrinks) |
| **E2** | `train` → `val_throws`, $\sigma_{\text{obs}} = 0$ | Generalization to new throws in seen rooms | 1-chunk RMSE < 1 cm; 0.5 s mean-rollout RMSE < 3 cm |
| **E3** | `train` → `test_rooms`, $\sigma_{\text{obs}} = 0$ | Generalization to unseen rooms | Within 10% of E2 errors |
| **E3b** | Wall-shuffle ablation on `test_rooms` | Swap each sample's point cloud with another room's | Error essentially unchanged, i.e. the model correctly ignores walls when there is no bounce |
| **E4** | Retrain with $\sigma_{\text{obs}} = 0.005$ m | Calibration of predicted uncertainty | True position inside the predicted 90% region (see below) for 85–95% of steps |
| **E5** | Compare to analytic baseline | Least-squares parabola fit to the same noisy history, extrapolated | Model mean-rollout RMSE within 1.5× of the baseline at every horizon |

**The 90% region in E4.** For a diagonal 3D Gaussian, the 90% region is
$$\left\{\mathbf{x} : \sum_{d} \frac{(x_d - \mu_d)^2}{\sigma_d^2} \le 6.25\right\}$$
where $\mathbf{x}$ is a candidate position, $d$ ranges over the three axes, and 6.25 is the 0.9 quantile of the chi-squared distribution with 3 degrees of freedom.

**The baseline in E5.** With noise, a parabola fit is the statistically optimal estimator, so the network should approach it but is not expected to beat it.

**Required plots:**
1. 3D plots of history, true future, predicted mean, and ~10 sampled rollouts, for a handful of throws.
2. RMSE vs. prediction horizon, for both the model and the E5 baseline.
3. Predicted σ vs. horizon (it should grow with horizon).
4. Calibration coverage vs. horizon (E4).

---

## 8. Code organization (suggested)

- `config` — all constants from Section 1 and all hyperparameters.
- `generate_data` — rooms, throws, rejection, the ground-truth check, and saving (calls the existing simulator).
- `dataset` — windowing, noise, normalization, padding masks (Section 3).
- `model` — wall encoder, chunk embedding, decoder, head (Section 4).
- `train` — loss, optimizer, logging, checkpointing.
- `evaluate` — rollouts, metrics, baseline, plots (Sections 6–7).

---

## 9. Out of scope here (next stages)

- Wall bounces with randomized restitution (Stage 2): requires removing the rejection rule in §2.2 step 4.
- Multi-step unrolled training and input-noise injection.
- The physics-residual flag ON.
- Ensembles or mixture outputs.
- Integration with the catching planner.
