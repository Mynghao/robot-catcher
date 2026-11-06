"""Turn clean trajectories into training windows (trajectory_model_plan.md section 3).

Token slots are fixed relative to t_now. Slots 0..K-1 hold history, left-padded so
slot K-1 always ends exactly at t_now. Slots K..K+N-2 hold future chunks for teacher
forcing. Slot K-1+i predicts future chunk i, so all N targets have a token.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import TrajectoryConfig


@dataclass
class WindowBatch:
    chunks: torch.Tensor  # (B, S, C, 3) normalized positions; padded slots are zero
    time_offsets: torch.Tensor  # (B, S) time of each chunk's first position minus t_now (s)
    mask: torch.Tensor  # (B, S) True where the slot holds a real chunk
    targets: torch.Tensor  # (B, N, C, 3) normalized offsets from p_now
    p_now: torch.Tensor  # (B, 3) last observed (possibly noisy) position, normalized
    room_index: torch.Tensor | None = None  # (B,) which room's wall tokens to use


def sample_window_indices(num_steps: int, batch_size: int, cfg: TrajectoryConfig,
                          generator: torch.Generator | None = None, device: torch.device | str = "cpu",
                          mode: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw (t_now, history length) pairs. mode defaults to cfg.history_sampling.

    "balanced" picks a history length uniformly, then a t_now late enough to fit it.
    "t_now_first" picks t_now uniformly, then a history length from what fits before it.
    """
    mode = cfg.history_sampling if mode is None else mode
    C = cfg.chunk_len
    low = C * cfg.min_history - 1
    high = num_steps - 1 - cfg.future_chunks * C
    if high < low:
        raise ValueError(f"trajectories of {num_steps} steps are too short for this window")
    if mode == "balanced":
        longest = min(cfg.max_history, (high + 1) // C)
        n_history = torch.randint(cfg.min_history, longest + 1, (batch_size,), generator=generator)
        earliest = C * n_history - 1
        t_now = earliest + (torch.rand(batch_size, generator=generator) * (high - earliest + 1)).long()
        return t_now.to(device), n_history.to(device)
    if mode != "t_now_first":
        raise ValueError(f"unknown history sampling mode {mode!r}")
    t_now = torch.randint(low, high + 1, (batch_size,), generator=generator)
    max_h = torch.minimum(((t_now + 1) // C), torch.tensor(cfg.max_history))
    u = torch.rand(batch_size, generator=generator)
    n_history = cfg.min_history + (u * (max_h - cfg.min_history + 1)).long()
    return t_now.to(device), n_history.to(device)


def build_windows(positions: torch.Tensor, t_now: torch.Tensor, n_history: torch.Tensor,
                  cfg: TrajectoryConfig, noise_std: float | None = None,
                  generator: torch.Generator | None = None) -> WindowBatch:
    """Slice one window per trajectory from clean positions (B, T, 3) in metres."""
    noise_std = cfg.obs_noise_std if noise_std is None else noise_std
    device = positions.device
    B = positions.shape[0]
    C, K, N = cfg.chunk_len, cfg.max_history, cfg.future_chunks
    batch = torch.arange(B, device=device)[:, None, None]
    steps = torch.arange(C, device=device)

    slots = torch.arange(K, device=device)
    chunk_ends = t_now[:, None] - (K - 1 - slots) * C  # (B, K)
    history_index = chunk_ends[:, :, None] - (C - 1) + steps  # (B, K, C)
    history_mask = slots[None, :] >= (K - n_history[:, None])
    history = positions[batch, history_index.clamp(min=0)]
    if noise_std > 0:
        # Generated on the CPU so a seeded generator behaves the same on every device.
        history = history + (noise_std * torch.randn(history.shape, generator=generator)).to(device)
    history = history * history_mask[:, :, None, None]

    future_index = t_now[:, None, None] + 1 + torch.arange(N, device=device)[None, :, None] * C + steps
    future = positions[batch, future_index]

    p_now = history[:, K - 1, C - 1]
    tokens = torch.cat([history, future[:, :N - 1]], dim=1) / cfg.length_scale
    first_index = torch.cat([history_index[:, :, 0], future_index[:, :N - 1, 0]], dim=1)
    time_offsets = (first_index - t_now[:, None]).to(positions.dtype) * cfg.dt
    mask = torch.cat([history_mask, torch.ones(B, N - 1, dtype=torch.bool, device=device)], dim=1)
    return WindowBatch(tokens, time_offsets * mask, mask, (future - p_now[:, None, None]) / cfg.length_scale,
                       p_now / cfg.length_scale)
