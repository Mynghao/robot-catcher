"""Autoregressive rollouts, the analytic baseline, and metrics (trajectory_model_plan.md sections 6-7)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import TrajectoryConfig
from .dataset import Split
from .model import TrajectoryModel
from .windows import WindowBatch, build_windows, sample_window_indices

CHI2_3D_90 = 6.25  # 0.9 quantile of the chi-squared distribution with 3 degrees of freedom


@dataclass
class Rollout:
    mean: torch.Tensor  # (B, N, C, 3) normalized offsets from p_now
    logvar: torch.Tensor  # (B, N, C, 3)


@torch.no_grad()
def rollout(model: TrajectoryModel, batch: WindowBatch, walls: torch.Tensor, sample: bool = False,
            generator: torch.Generator | None = None) -> Rollout:
    """Predict N chunks from history alone. p_now stays the last observed position throughout.

    Each step appends the predicted mean, or a draw from the predicted Gaussian if sample=True.
    """
    cfg = model.cfg
    C, K, N = cfg.chunk_len, cfg.max_history, cfg.future_chunks
    chunks = batch.chunks.clone()
    offsets = batch.time_offsets.clone()
    mask = batch.mask.clone()
    chunks[:, K:] = 0.0
    offsets[:, K:] = 0.0
    mask[:, K:] = False
    means, logvars = [], []
    for i in range(N):
        slot = K - 1 + i
        # Only slots up to `slot` are in view; causality makes the rest irrelevant.
        pred = model(chunks[:, :slot + 1], offsets[:, :slot + 1], mask[:, :slot + 1], walls, batch.p_now)
        mean, logvar = pred.mean[:, slot], pred.logvar[:, slot]
        means.append(mean)
        logvars.append(logvar)
        if i == N - 1:
            break
        offset = mean
        if sample:
            noise = torch.randn(mean.shape, generator=generator).to(mean.device)
            offset = mean + noise * torch.exp(0.5 * logvar)
        chunks[:, slot + 1] = batch.p_now[:, None] + offset
        offsets[:, slot + 1] = (i * C + 1) * cfg.dt
        mask[:, slot + 1] = True
    return Rollout(torch.stack(means, dim=1), torch.stack(logvars, dim=1))


def sampled_rollouts(model: TrajectoryModel, batch: WindowBatch, walls: torch.Tensor, num_samples: int = 64,
                     generator: torch.Generator | None = None) -> Rollout:
    """S sampled futures per window. Fields gain a sample dimension: (B, S, N, C, 3)."""
    B = batch.chunks.shape[0]
    expanded = WindowBatch(batch.chunks.repeat_interleave(num_samples, 0),
                           batch.time_offsets.repeat_interleave(num_samples, 0),
                           batch.mask.repeat_interleave(num_samples, 0),
                           batch.targets.repeat_interleave(num_samples, 0),
                           batch.p_now.repeat_interleave(num_samples, 0))
    result = rollout(model, expanded, walls.repeat_interleave(num_samples, 0), sample=True, generator=generator)
    shape = (B, num_samples, *result.mean.shape[1:])
    return Rollout(result.mean.view(shape), result.logvar.view(shape))


def parabola_baseline(batch: WindowBatch, cfg: TrajectoryConfig) -> torch.Tensor:
    """Section 7's E5 baseline: least-squares fit of p0 and v0 to the observed history with
    gravity known, extrapolated over the future. Returns (B, N, C, 3) offsets from p_now."""
    C, K, N = cfg.chunk_len, cfg.max_history, cfg.future_chunks
    dtype, device = batch.chunks.dtype, batch.chunks.device
    gravity = torch.tensor(cfg.gravity, dtype=dtype, device=device) / cfg.length_scale
    # Time of every history and future step relative to t_now. The history ends at t = 0.
    t_hist = cfg.dt * (torch.arange(K * C, dtype=dtype, device=device) - (K * C - 1))
    t_future = cfg.dt * torch.arange(1, N * C + 1, dtype=dtype, device=device)
    valid = batch.mask[:, :K, None].expand(-1, -1, C).flatten(1).to(dtype)  # (B, K C)
    observed = batch.chunks[:, :K].flatten(1, 2)  # (B, K C, 3)
    y = observed - 0.5 * gravity * t_hist[:, None] ** 2
    # Normal equations for y = p + v t, weighted by the padding mask.
    s0 = valid.sum(1)
    s1 = (valid * t_hist).sum(1)
    s2 = (valid * t_hist**2).sum(1)
    b0 = (valid[:, :, None] * y).sum(1)
    b1 = (valid[:, :, None] * t_hist[:, None] * y).sum(1)
    det = s0 * s2 - s1**2
    v = (s0[:, None] * b1 - s1[:, None] * b0) / det[:, None]
    p = (b0 - s1[:, None] * v) / s0[:, None]
    future = p[:, None] + v[:, None] * t_future[None, :, None] + 0.5 * gravity * t_future[None, :, None] ** 2
    return (future - batch.p_now[:, None]).view(-1, N, C, 3)


def position_rmse(mean: torch.Tensor, targets: torch.Tensor, cfg: TrajectoryConfig) -> torch.Tensor:
    """3D position RMSE in metres, per chunk and step: (N, C). Average squared errors yourself to combine."""
    return (mean - targets).square().sum(-1).mean(0).sqrt() * cfg.length_scale


def combined_rmse(rmse: torch.Tensor) -> float:
    """One RMSE from per-step RMSEs: the root of the mean squared error."""
    return float(rmse.square().mean().sqrt())


def coverage_90(mean: torch.Tensor, logvar: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Fraction of true positions inside the predicted 90% ellipsoid, per chunk and step (section 7, E4)."""
    distance = ((targets - mean).square() * torch.exp(-logvar)).sum(-1)
    return (distance <= CHI2_3D_90).to(mean.dtype).mean(0)


@torch.no_grad()
def evaluate_split(model: TrajectoryModel, split: Split, noise_std: float, num_windows: int = 2048, seed: int = 1234,
                   full_history: bool = False, shuffle_walls: bool = False) -> dict:
    """Metrics in metres on fixed windows, so runs and models are compared on identical inputs.

    Windows always use the "t_now_first" mix, whatever the model trained on, so scores stay
    comparable across runs. full_history uses K history chunks everywhere instead.
    shuffle_walls gives each window another room's walls (section 7, E3b).
    """
    cfg = model.cfg
    model.eval()
    device = split.positions.device
    generator = torch.Generator().manual_seed(seed)
    throw = torch.randint(0, split.positions.shape[0], (num_windows,), generator=generator).to(device)
    if full_history:
        t_now = torch.full((num_windows,), cfg.max_history * cfg.chunk_len - 1, device=device)
        n_history = torch.full((num_windows,), cfg.max_history, device=device)
    else:
        t_now, n_history = sample_window_indices(split.positions.shape[1], num_windows, cfg, generator, device,
                                                 mode="t_now_first")
    batch = build_windows(split.positions[throw], t_now, n_history, cfg, noise_std=noise_std, generator=generator)
    tokens = model.encode_walls(split.patches, split.centers)
    room = split.room_index[throw]
    if shuffle_walls:
        room = (room + 1) % split.patches.shape[0]

    tf_mean, ro_mean, ro_logvar = [], [], []
    for i in range(0, num_windows, cfg.batch_size):
        j = i + cfg.batch_size
        part = WindowBatch(batch.chunks[i:j], batch.time_offsets[i:j], batch.mask[i:j], batch.targets[i:j], batch.p_now[i:j])
        walls = tokens[room[i:j]]
        pred = model(part.chunks, part.time_offsets, part.mask, walls, part.p_now)
        tf_mean.append(pred.mean[:, cfg.max_history - 1:])
        result = rollout(model, part, walls)
        ro_mean.append(result.mean)
        ro_logvar.append(result.logvar)
    tf_mean, ro_mean, ro_logvar = torch.cat(tf_mean), torch.cat(ro_mean), torch.cat(ro_logvar)

    tf_rmse = position_rmse(tf_mean, batch.targets, cfg)
    ro_rmse = position_rmse(ro_mean, batch.targets, cfg)
    base_rmse = position_rmse(parabola_baseline(batch, cfg), batch.targets, cfg)
    coverage = coverage_90(ro_mean, ro_logvar, batch.targets)
    sigma = (torch.exp(0.5 * ro_logvar) * cfg.length_scale).mean(dim=(0, 3))
    return {
        "teacher_forced_rmse": combined_rmse(tf_rmse),
        "one_chunk_rmse": combined_rmse(ro_rmse[0]),
        "rollout_rmse": combined_rmse(ro_rmse),
        "rollout_rmse_final_step": float(ro_rmse[-1, -1]),
        "rollout_rmse_by_chunk": [combined_rmse(ro_rmse[i]) for i in range(cfg.future_chunks)],
        "baseline_rmse": combined_rmse(base_rmse),
        "baseline_rmse_by_chunk": [combined_rmse(base_rmse[i]) for i in range(cfg.future_chunks)],
        "coverage_90": float(coverage.mean()),
        "coverage_90_by_chunk": coverage.mean(1).tolist(),
        "sigma_by_chunk": sigma.mean(1).tolist(),
    }
