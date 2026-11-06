"""Stage 1 data: no-bounce parabolic throws in random rooms (trajectory_model_plan.md section 2).

Every throw goes through the event-driven simulator, which finds surface crossings exactly.
That catches a ball clipping a corner between two 10 ms samples, which checking the
samples alone would miss.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from catcher.ball_sim import simulate
from catcher.room import MATERIALS, Panel, Room, box_room

from .config import TrajectoryConfig

CLEARANCE_MARGIN = 0.05  # section 2.2: reject throws within r + this of any surface
MAX_PARABOLA_RESIDUAL = 1e-3  # section 2.3


@dataclass
class StageRoom:
    room: Room  # panels in the robot base frame, ready for the simulator
    width: float
    depth: float
    height: float
    yaw: float
    base: np.ndarray  # robot base position in room coordinates (x, y)
    obstacles: list[tuple[np.ndarray, np.ndarray]]  # (center, half size) in room coordinates

    def to_world(self, points: np.ndarray) -> np.ndarray:
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        xy = points[..., :2] - self.base
        return np.stack([c * xy[..., 0] - s * xy[..., 1], s * xy[..., 0] + c * xy[..., 1], points[..., 2]], axis=-1)

    def rotate(self, vector: np.ndarray) -> np.ndarray:
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return np.array([c * vector[0] - s * vector[1], s * vector[0] + c * vector[1], vector[2]])


def _obstacle_panels(center: np.ndarray, half: np.ndarray, index: int) -> list[Panel]:
    """The six faces of a box, in room coordinates."""
    ex, ey, ez = np.eye(3)
    material = MATERIALS["plywood"]
    panels = []
    for sign in (1.0, -1.0):
        panels += [
            Panel(f"obstacle_{index}_x{sign:+.0f}", center + sign * half[0] * ex, sign * ex, ey, (half[1], half[2]), material),
            Panel(f"obstacle_{index}_y{sign:+.0f}", center + sign * half[1] * ey, sign * ey, ex, (half[0], half[2]), material),
            Panel(f"obstacle_{index}_z{sign:+.0f}", center + sign * half[2] * ez, sign * ez, ex, (half[0], half[1]), material),
        ]
    return panels


def sample_room(rng: np.random.Generator) -> StageRoom:
    """Section 2.1: a box room rotated about z, floor at z = 0, with the robot at least 1 m
    from the side walls, and 0-2 box obstacles standing on the floor away from the robot."""
    width, depth = rng.uniform(4.0, 10.0, size=2)
    height = rng.uniform(3.0, 6.0)
    limit = np.array([width / 2 - 1.0, depth / 2 - 1.0])
    base = rng.uniform(-limit, limit)

    obstacles = []
    for _ in range(rng.integers(0, 3)):
        half = rng.uniform([0.2, 0.2, 0.2], [0.6, 0.6, 0.8])
        while True:
            center_xy = rng.uniform([-width / 2 + half[0], -depth / 2 + half[1]],
                                    [width / 2 - half[0], depth / 2 - half[1]])
            if np.linalg.norm(center_xy - base) > 1.5:
                break
        obstacles.append((np.array([center_xy[0], center_xy[1], half[2]]), half))

    local = box_room([-width / 2, -depth / 2, 0.0], [width / 2, depth / 2, height])
    panels = list(local.panels)
    for i, (center, half) in enumerate(obstacles):
        panels += _obstacle_panels(center, half, i)

    stage_room = StageRoom(local, width, depth, height, rng.uniform(-np.pi, np.pi), base, obstacles)
    world = [Panel(p.name, stage_room.to_world(p.center), stage_room.rotate(p.normal), stage_room.rotate(p.u),
                   p.half_extents, p.material) for p in panels]
    corners = stage_room.to_world(np.array([[x, y, 0.0] for x in (-width / 2, width / 2) for y in (-depth / 2, depth / 2)]))
    stage_room.room = Room(world, np.array([*corners[:, :2].min(0), 0.0]), np.array([*corners[:, :2].max(0), height]))
    return stage_room


def distance_to_surfaces(room: Room, points: np.ndarray) -> np.ndarray:
    """Distance from each point (..., 3) to the nearest panel. Exact for bounded rectangles."""
    nearest = np.full(points.shape[:-1], np.inf)
    for panel in room.panels:
        d = points - panel.center
        along_n = np.abs(d @ panel.normal)
        along_u = np.maximum(np.abs(d @ panel.u) - panel.half_extents[0], 0.0)
        along_v = np.maximum(np.abs(d @ panel.v) - panel.half_extents[1], 0.0)
        nearest = np.minimum(nearest, np.sqrt(along_n**2 + along_u**2 + along_v**2))
    return nearest


def _inside_obstacle(room: StageRoom, p: np.ndarray) -> bool:
    return any(np.all(np.abs(p - center) <= half) for center, half in room.obstacles)


def sample_throws(stage_room: StageRoom, n: int, rng: np.random.Generator, cfg: TrajectoryConfig) -> tuple[dict, float]:
    """Section 2.2, by rejection. Returns n throws in the robot base frame and the acceptance rate."""
    times = cfg.dt * np.arange(cfg.window_steps)
    t_end = cfg.window_steps * cfg.dt
    w, d, h = stage_room.width, stage_room.depth, stage_room.height
    clearance = cfg.ball_radius + CLEARANCE_MARGIN
    low = [-w / 2 + 0.5, -d / 2 + 0.5, 0.5]
    high = [w / 2 - 0.5, d / 2 - 0.5, min(2.5, h - 0.5)]
    positions, p0s, v0s = [], [], []
    tried = 0
    while len(positions) < n:
        tried += 1
        p_local = rng.uniform(low, high)
        if _inside_obstacle(stage_room, p_local):
            continue
        speed = rng.uniform(3.0, 10.0)
        azimuth = rng.uniform(0.0, 2 * np.pi)
        elevation = np.radians(rng.uniform(-10.0, 60.0))
        v_local = speed * np.array([np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])
        p0, v0 = stage_room.to_world(p_local), stage_room.rotate(v_local)
        trajectory = simulate(stage_room.room, p0, v0, t_end=t_end)
        if trajectory.bounces:
            continue
        pos, _ = trajectory.sample(times)
        if distance_to_surfaces(stage_room.room, pos).min() < clearance:
            continue
        positions.append(pos)
        p0s.append(p0)
        v0s.append(v0)
    return {"positions": np.stack(positions), "p0": np.stack(p0s), "v0": np.stack(v0s)}, n / tried


def sample_cloud(room: Room, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Section 2.1: n points (n, 3) and their surface normals, area-weighted across all panels."""
    areas = np.array([p.area for p in room.panels])
    index = rng.choice(len(room.panels), size=n, p=areas / areas.sum())
    ab = rng.uniform(-1.0, 1.0, size=(n, 2))
    centers = np.stack([p.center for p in room.panels])
    us = np.stack([p.u * p.half_extents[0] for p in room.panels])
    vs = np.stack([p.v * p.half_extents[1] for p in room.panels])
    normals = np.stack([p.normal for p in room.panels])
    points = centers[index] + ab[:, :1] * us[index] + ab[:, 1:] * vs[index]
    return points, normals[index]


def parabola_residuals(positions: np.ndarray, cfg: TrajectoryConfig) -> np.ndarray:
    """Section 2.3: per-throw maximum distance from the best fit of p0 + v0 t + g t^2 / 2. Shape (N,)."""
    t = cfg.dt * np.arange(positions.shape[1])
    y = positions - 0.5 * np.array(cfg.gravity) * t[None, :, None] ** 2
    A = np.stack([np.ones_like(t), t], axis=1)
    fit = A @ (np.linalg.pinv(A) @ y)
    return np.linalg.norm(y - fit, axis=-1).max(axis=1)


# Section 2.4. Split name -> (number of rooms, throws per room, shares rooms with).
SPLITS = {"train": (50, 200), "val_throws": (50, 20), "test_rooms": (10, 50), "overfit": (1, 32)}


def generate_stage1(out_dir: Path, cfg: TrajectoryConfig, seed: int = 0, log=print) -> dict:
    """Write data/stage1: one file per room, plus trajectories.npz with a split label per throw."""
    rng = np.random.default_rng(seed)
    (out_dir / "rooms").mkdir(parents=True, exist_ok=True)

    train_rooms = [sample_room(rng) for _ in range(SPLITS["train"][0])]
    test_rooms = [sample_room(rng) for _ in range(SPLITS["test_rooms"][0])]
    overfit_rooms = [sample_room(rng) for _ in range(SPLITS["overfit"][0])]
    assignments = [("train", train_rooms, 0), ("val_throws", train_rooms, 0),
                   ("test_rooms", test_rooms, len(train_rooms)),
                   ("overfit", overfit_rooms, len(train_rooms) + len(test_rooms))]

    for room_id, stage_room in enumerate(train_rooms + test_rooms + overfit_rooms):
        points, normals = sample_cloud(stage_room.room, cfg.n_points, rng)
        np.savez_compressed(out_dir / "rooms" / f"room_{room_id:04d}.npz", room_id=room_id,
                            points=points.astype(np.float32), normals=normals.astype(np.float32),
                            width=stage_room.width, depth=stage_room.depth, height=stage_room.height,
                            yaw=stage_room.yaw, base=stage_room.base,
                            obstacle_centers=np.array([c for c, _ in stage_room.obstacles]).reshape(-1, 3),
                            obstacle_half_sizes=np.array([h for _, h in stage_room.obstacles]).reshape(-1, 3))

    columns: dict[str, list] = {"positions": [], "p0": [], "v0": [], "room_id": [], "split": []}
    acceptance = []
    split_names = list(SPLITS)
    for split, rooms, first_id in assignments:
        per_room = SPLITS[split][1]
        for i, stage_room in enumerate(rooms):
            throws, rate = sample_throws(stage_room, per_room, rng, cfg)
            acceptance.append(rate)
            for key in ("positions", "p0", "v0"):
                columns[key].append(throws[key])
            columns["room_id"].append(np.full(per_room, first_id + i))
            columns["split"].append(np.full(per_room, split_names.index(split)))
        log(f"{split}: {len(rooms)} rooms x {per_room} throws, acceptance so far {np.mean(acceptance):.1%}")

    positions = np.concatenate(columns["positions"])
    residuals = parabola_residuals(positions, cfg)
    if residuals.max() >= MAX_PARABOLA_RESIDUAL:
        raise RuntimeError(f"parabola check failed: max residual {residuals.max()} m")
    data = {key: np.concatenate(columns[key]) for key in ("p0", "v0", "room_id", "split")}
    np.savez_compressed(out_dir / "trajectories.npz", positions=positions.astype(np.float32),
                        times=cfg.dt * np.arange(cfg.window_steps), split_names=np.array(split_names), **data)
    manifest = {
        "seed": seed,
        "throws": {name: int((data["split"] == i).sum()) for i, name in enumerate(split_names)},
        "acceptance_mean": float(np.mean(acceptance)),
        "acceptance_lowest_room": float(np.min(acceptance)),
        "max_parabola_residual_m": float(residuals.max()),
        "steps": cfg.window_steps,
        "dt": cfg.dt,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest
