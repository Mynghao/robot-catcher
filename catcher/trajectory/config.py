"""All trajectory-model constants and hyperparameters (trajectory_model_plan.md sections 1, 4, 5)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrajectoryConfig:
    # Conventions (section 1).
    length_scale: float = 3.0  # s: one constant for all rooms, so gravity keeps one normalized value
    dt: float = 0.01  # observation timestep (s)
    chunk_len: int = 5  # C: time steps per token
    max_history: int = 10  # K: history chunks
    min_history: int = 2
    future_chunks: int = 10  # N: chunks to predict
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    ball_radius: float = 0.0  # catcher.ball_sim treats the ball as a point
    obs_noise_std: float = 0.0  # sigma_obs (m); added to history only, on the fly

    # Wall encoder (section 4.1).
    n_points: int = 4096  # P
    n_patches: int = 64  # M
    patch_size: int = 32  # m
    use_normals: bool = False
    wall_layers: int = 2

    # Decoder (sections 4.2-4.5).
    d_model: int = 128
    n_heads: int = 4
    decoder_layers: int = 4  # L
    mlp_ratio: int = 4
    logvar_min: float = -14.0
    logvar_max: float = 2.0
    physics_residual: bool = False  # off for Stage 1 (section 4.5)

    # Training (section 5).
    lr: float = 3e-4
    weight_decay: float = 0.01
    lr_warmup_steps: int = 1000
    mse_warmup_steps: int = 1000
    batch_size: int = 256
    grad_clip: float = 1.0
    # Departures from the plan's section 5, which keeps future tokens clean and samples t_now first.
    # Fraction of each batch whose future tokens are the model's own rollout rather than the truth.
    # 0 is the plan's pure teacher forcing. Ramps up linearly between the two steps below.
    rollout_fraction: float = 1.0
    rollout_ramp_start: int = 1000
    rollout_ramp_end: int = 5000
    # "balanced": pick the history length first, so every length is equally common.
    # "t_now_first": the plan's section 3, where full histories are about 0.3% of windows.
    history_sampling: str = "balanced"

    @property
    def num_tokens(self) -> int:
        """Token slots: K history slots, then N - 1 future chunks (the last one is only a target)."""
        return self.max_history + self.future_chunks - 1

    @property
    def token_input_dim(self) -> int:
        """C relative positions, the chunk's first position, and its time offset."""
        return 3 * self.chunk_len + 3 + 1

    @property
    def window_steps(self) -> int:
        """Steps in the longest window, (K + N) C: also the plan's trajectory length."""
        return (self.max_history + self.future_chunks) * self.chunk_len
