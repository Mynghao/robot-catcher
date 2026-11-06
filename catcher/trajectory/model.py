"""Wall encoder + causal trajectory decoder with a Gaussian head (trajectory_model_plan.md section 4)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .config import TrajectoryConfig


def farthest_point_sample(points: torch.Tensor, k: int) -> torch.Tensor:
    """Indices of k far-apart points. Starts from the point farthest from the
    centroid, so the chosen points don't depend on the order of the input."""
    chosen = [(points - points.mean(0)).square().sum(-1).argmax()]
    distance = (points - points[chosen[0]]).square().sum(-1)
    for _ in range(k - 1):
        chosen.append(distance.argmax())
        distance = torch.minimum(distance, (points - points[chosen[-1]]).square().sum(-1))
    return torch.stack(chosen)


def build_wall_patches(points: torch.Tensor, cfg: TrajectoryConfig,
                       normals: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Group a room's point cloud (P, 3) in metres into patches.

    Returns patches (M, m, 3 or 6), relative to their centers and normalized, and
    centers (M, 3), normalized. Depends only on the room, so compute it once per room.
    """
    points = points / cfg.length_scale
    centers = points[farthest_point_sample(points, cfg.n_patches)]
    neighbors = torch.cdist(centers, points).topk(cfg.patch_size, largest=False).indices
    patches = points[neighbors] - centers[:, None]
    if cfg.use_normals:
        if normals is None:
            raise ValueError("cfg.use_normals is set but no normals were given")
        patches = torch.cat([patches, normals[neighbors]], dim=-1)
    return patches, centers


class Attention(nn.Module):
    """Multi-head attention. mask is (B, 1, Q, K) and True where the query may attend."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], x.shape[1], self.n_heads, -1).transpose(1, 2)

    def forward(self, queries: torch.Tensor, context: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attended = F.scaled_dot_product_attention(
            self._split(self.query(queries)), self._split(self.key(context)), self._split(self.value(context)),
            attn_mask=mask)
        return self.out(attended.transpose(1, 2).flatten(2))


def _mlp(d_model: int, ratio: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_model, ratio * d_model), nn.GELU(), nn.Linear(ratio * d_model, d_model))


class WallLayer(nn.Module):
    """Pre-LayerNorm self-attention among wall tokens."""

    def __init__(self, cfg: TrajectoryConfig):
        super().__init__()
        self.norm_attn = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg.d_model, cfg.n_heads)
        self.norm_mlp = nn.LayerNorm(cfg.d_model)
        self.mlp = _mlp(cfg.d_model, cfg.mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm_attn(x)
        x = x + self.attn(h, h)
        return x + self.mlp(self.norm_mlp(x))


class WallEncoder(nn.Module):
    """Mini PointNet on each patch, a center embedding, then self-attention (section 4.1)."""

    def __init__(self, cfg: TrajectoryConfig):
        super().__init__()
        point_dim = 6 if cfg.use_normals else 3
        self.point_mlp = nn.Sequential(nn.Linear(point_dim, 64), nn.GELU(), nn.Linear(64, 128), nn.GELU(),
                                       nn.Linear(128, cfg.d_model))
        self.center_mlp = nn.Sequential(nn.Linear(3, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, cfg.d_model))
        self.layers = nn.ModuleList(WallLayer(cfg) for _ in range(cfg.wall_layers))
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, patches: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        """patches (R, M, m, point_dim), centers (R, M, 3) -> wall tokens (R, M, d)."""
        x = self.point_mlp(patches).amax(dim=2) + self.center_mlp(centers)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class DecoderBlock(nn.Module):
    """Causal self-attention, cross-attention to the walls, then an MLP (section 4.3)."""

    def __init__(self, cfg: TrajectoryConfig):
        super().__init__()
        self.norm_self = nn.LayerNorm(cfg.d_model)
        self.self_attn = Attention(cfg.d_model, cfg.n_heads)
        self.norm_cross = nn.LayerNorm(cfg.d_model)
        self.cross_attn = Attention(cfg.d_model, cfg.n_heads)
        self.norm_mlp = nn.LayerNorm(cfg.d_model)
        self.mlp = _mlp(cfg.d_model, cfg.mlp_ratio)

    def forward(self, x: torch.Tensor, walls: torch.Tensor, self_mask: torch.Tensor) -> torch.Tensor:
        h = self.norm_self(x)
        x = x + self.self_attn(h, h, self_mask)
        x = x + self.cross_attn(self.norm_cross(x), walls)
        return x + self.mlp(self.norm_mlp(x))


@dataclass
class Prediction:
    mean: torch.Tensor  # (B, S, C, 3) next-chunk mean, normalized offset from p_now
    logvar: torch.Tensor  # (B, S, C, 3)


class TrajectoryModel(nn.Module):
    def __init__(self, cfg: TrajectoryConfig):
        super().__init__()
        self.cfg = cfg
        self.wall_encoder = WallEncoder(cfg)
        self.chunk_mlp = nn.Sequential(nn.Linear(cfg.token_input_dim, cfg.d_model), nn.GELU(),
                                       nn.Linear(cfg.d_model, cfg.d_model))
        self.slot_embedding = nn.Embedding(cfg.num_tokens, cfg.d_model)
        self.blocks = nn.ModuleList(DecoderBlock(cfg) for _ in range(cfg.decoder_layers))
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, 6 * cfg.chunk_len)
        self.register_buffer("causal", torch.tril(torch.ones(cfg.num_tokens, cfg.num_tokens, dtype=torch.bool)),
                             persistent=False)
        # Next-chunk steps, in seconds after a chunk's last position, for the physics residual.
        self.register_buffer("step_times", cfg.dt * torch.arange(1, cfg.chunk_len + 1, dtype=torch.float32),
                             persistent=False)
        self.register_buffer("gravity", torch.tensor(cfg.gravity, dtype=torch.float32) / cfg.length_scale,
                             persistent=False)

    def encode_walls(self, patches: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        """Wall tokens (R, M, d). Cache these per room at inference time."""
        return self.wall_encoder(patches, centers)

    def _embed_chunks(self, chunks: torch.Tensor, time_offsets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        first = chunks[:, :, :1]
        features = torch.cat([(chunks - first).flatten(2), first[:, :, 0], time_offsets[:, :, None]], dim=-1)
        slots = torch.arange(chunks.shape[1], device=chunks.device)
        tokens = self.chunk_mlp(features) + self.slot_embedding(slots)
        return tokens * mask[:, :, None]

    def _ballistic_offset(self, chunks: torch.Tensor, p_now: torch.Tensor) -> torch.Tensor:
        """Constant-gravity extrapolation of each chunk into the next one, as an offset from p_now."""
        C, dt = self.cfg.chunk_len, self.cfg.dt
        duration = (C - 1) * dt
        # Mean velocity over a chunk is its midpoint velocity; advance it to the chunk's end.
        v_end = (chunks[:, :, -1] - chunks[:, :, 0]) / duration + self.gravity * duration / 2
        t = self.step_times[None, None, :, None]
        future = chunks[:, :, -1:] + v_end[:, :, None] * t + 0.5 * self.gravity * t**2
        return future - p_now[:, None, None]

    def forward(self, chunks: torch.Tensor, time_offsets: torch.Tensor, mask: torch.Tensor, walls: torch.Tensor,
                p_now: torch.Tensor | None = None) -> Prediction:
        """Predict each slot's next chunk.

        chunks (B, S, C, 3), time_offsets (B, S), mask (B, S), walls (B, M, d).
        p_now (B, 3) is only needed when cfg.physics_residual is on.
        """
        S = chunks.shape[1]
        # Valid tokens attend to themselves and earlier valid tokens. Every token keeps
        # itself in view so padded rows aren't empty, which would give NaNs.
        self_mask = self.causal[:S, :S][None, None] & mask[:, None, None, :]
        self_mask = self_mask | torch.eye(S, dtype=torch.bool, device=mask.device)[None, None]
        x = self._embed_chunks(chunks, time_offsets, mask)
        for block in self.blocks:
            x = block(x, walls, self_mask)
        out = self.head(self.norm(x)).view(chunks.shape[0], S, self.cfg.chunk_len, 6)
        mean, logvar = out[..., :3], out[..., 3:]
        if self.cfg.physics_residual:
            mean = mean + self._ballistic_offset(chunks, p_now)
        return Prediction(mean, logvar.clamp(self.cfg.logvar_min, self.cfg.logvar_max))
