"""Load generated trajectories and rooms for training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import TrajectoryConfig
from .model import build_wall_patches


@dataclass
class Split:
    positions: torch.Tensor  # (N, T, 3) clean positions in metres
    room_index: torch.Tensor  # (N,) index into patches and centers
    patches: torch.Tensor  # (R, M, m, 3 or 6)
    centers: torch.Tensor  # (R, M, 3)

    def first_per_room(self, per_room: int) -> "Split":
        """The first `per_room` throws of each room, so smaller training sets nest inside larger ones."""
        keep = []
        for room in range(self.patches.shape[0]):
            keep.append(torch.nonzero(self.room_index == room).flatten()[:per_room])
        keep = torch.cat(keep)
        return Split(self.positions[keep], self.room_index[keep], self.patches, self.centers)


def load_split(root: Path | str, split: str, cfg: TrajectoryConfig, device: str = "cpu") -> Split:
    root = Path(root)
    data = np.load(root / "trajectories.npz")
    names = data["split_names"].tolist()
    if split not in names:
        raise ValueError(f"unknown split {split!r}; expected one of {names}")
    keep = data["split"] == names.index(split)
    room_ids = data["room_id"][keep]
    unique, room_index = np.unique(room_ids, return_inverse=True)
    patches, centers = [], []
    for room_id in unique:
        room = np.load(root / "rooms" / f"room_{room_id:04d}.npz")
        normals = torch.from_numpy(room["normals"]) if cfg.use_normals else None
        p, c = build_wall_patches(torch.from_numpy(room["points"]), cfg, normals)
        patches.append(p)
        centers.append(c)
    return Split(torch.from_numpy(data["positions"][keep]).to(device), torch.from_numpy(room_index).to(device),
                 torch.stack(patches).to(device), torch.stack(centers).to(device))
