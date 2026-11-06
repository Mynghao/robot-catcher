"""Loss, optimizer and training loop (trajectory_model_plan.md section 5). Teacher forcing over the whole window."""

from __future__ import annotations

import math
from typing import Callable

import torch

from .config import TrajectoryConfig
from .dataset import Split
from .evaluate import rollout
from .model import Prediction, TrajectoryModel
from .windows import WindowBatch, build_windows, sample_window_indices


def future_predictions(pred: Prediction, cfg: TrajectoryConfig) -> Prediction:
    """The N predictions that have targets: slot K-1 predicts future chunk 0, and so on."""
    return Prediction(pred.mean[:, cfg.max_history - 1:], pred.logvar[:, cfg.max_history - 1:])


def gaussian_nll(pred: Prediction, targets: torch.Tensor) -> torch.Tensor:
    """Summed over chunks, steps and axes, averaged over the batch. Constant term dropped."""
    per_term = 0.5 * ((targets - pred.mean).square() * torch.exp(-pred.logvar) + pred.logvar)
    return per_term.sum(dim=(1, 2, 3)).mean()


def mean_squared_error(pred: Prediction, targets: torch.Tensor) -> torch.Tensor:
    """Same reduction as gaussian_nll, on the mean only."""
    return 0.5 * (targets - pred.mean).square().sum(dim=(1, 2, 3)).mean()


def teacher_forced_loss(model: TrajectoryModel, batch: WindowBatch, walls: torch.Tensor,
                        use_mse: bool = False) -> tuple[torch.Tensor, Prediction]:
    """walls are (B, M, d), already indexed per sample."""
    pred = future_predictions(model(batch.chunks, batch.time_offsets, batch.mask, walls, batch.p_now), model.cfg)
    loss = mean_squared_error(pred, batch.targets) if use_mse else gaussian_nll(pred, batch.targets)
    return loss, pred


def make_optimizer(model: TrajectoryModel, total_steps: int) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """AdamW with a linear warmup, then cosine decay to zero over total_steps."""
    cfg = model.cfg
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warmup = min(cfg.lr_warmup_steps, total_steps)

    def scale(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def rollout_fraction_at(step: int, cfg: TrajectoryConfig) -> float:
    """Share of the batch using rollout tokens: 0, then a linear ramp, then cfg.rollout_fraction."""
    span = max(1, cfg.rollout_ramp_end - cfg.rollout_ramp_start)
    return cfg.rollout_fraction * min(max((step - cfg.rollout_ramp_start) / span, 0.0), 1.0)


def with_rollout_tokens(model: TrajectoryModel, batch: WindowBatch, walls: torch.Tensor, fraction: float,
                        generator: torch.Generator | None = None) -> WindowBatch:
    """Swap the true future tokens for the model's own mean rollout, in a random `fraction` of the batch.

    The rollout runs without gradients. Thanks to causality, a forward pass on the result reproduces
    the rollout's predictions, so the loss scores exactly what inference produces, including how
    errors build up with horizon, while each step only backpropagates through one pass.
    """
    if fraction <= 0.0:
        return batch
    K, N = model.cfg.max_history, model.cfg.future_chunks
    chosen = torch.rand(batch.chunks.shape[0], generator=generator).to(batch.chunks.device) < fraction
    if not chosen.any():
        return batch
    subset = WindowBatch(batch.chunks[chosen], batch.time_offsets[chosen], batch.mask[chosen], batch.targets[chosen],
                         batch.p_now[chosen])
    predicted = rollout(model, subset, walls[chosen]).mean
    chunks = batch.chunks.clone()
    chunks[chosen, K:] = subset.p_now[:, None, None] + predicted[:, :N - 1]
    return WindowBatch(chunks, batch.time_offsets, batch.mask, batch.targets, batch.p_now, batch.room_index)


def fit(model: TrajectoryModel, split: Split, steps: int, noise_std: float, seed: int = 0,
        on_eval: Callable[[int, float], None] | None = None, eval_every: int = 1000,
        on_log: Callable[[int, dict], None] | None = None, log_every: int = 50) -> None:
    """Train in place. Windows are drawn fresh each step: random throw, t_now and history length.

    on_log gets training metrics every log_every steps; on_eval is for slower evaluation.
    """
    cfg = model.cfg
    device = split.positions.device
    optimizer, scheduler = make_optimizer(model, steps)
    generator = torch.Generator().manual_seed(seed)
    for step in range(steps):
        model.train()
        throw = torch.randint(0, split.positions.shape[0], (cfg.batch_size,), generator=generator).to(device)
        t_now, n_history = sample_window_indices(split.positions.shape[1], cfg.batch_size, cfg, generator, device)
        batch = build_windows(split.positions[throw], t_now, n_history, cfg, noise_std=noise_std, generator=generator)
        # Encode each room in the batch once, keeping gradients flowing into the wall encoder.
        present, inverse = split.room_index[throw].unique(return_inverse=True)
        walls = model.encode_walls(split.patches[present], split.centers[present])[inverse]
        use_mse = step < cfg.mse_warmup_steps
        fraction = rollout_fraction_at(step, cfg)
        batch = with_rollout_tokens(model, batch, walls, fraction, generator)
        loss, pred = teacher_forced_loss(model, batch, walls, use_mse=use_mse)
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        if on_log is not None and (step + 1) % log_every == 0:
            on_log(step + 1, {"train/mse_loss" if use_mse else "train/nll": loss.item(),
                              "train/grad_norm": grad_norm.item(),
                              "train/lr": scheduler.get_last_lr()[0],
                              "train/rollout_fraction": fraction,
                              "train/sigma_m": (torch.exp(0.5 * pred.logvar).mean() * cfg.length_scale).item(),
                              "train/rmse_m": ((pred.mean - batch.targets).square().sum(-1).mean().sqrt()
                                               * cfg.length_scale).item()})
        optimizer.step()
        scheduler.step()
        if on_eval is not None and ((step + 1) % eval_every == 0 or step + 1 == steps):
            on_eval(step + 1, loss.item())
