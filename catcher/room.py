"""Room geometry: flat rectangular panels with per-surface bounce materials.

Everything in the scene (floor, walls, ceiling, free-standing boards) is a
`Panel`: a bounded rectangle in 3D. The same `Room` object drives the NumPy
ball simulator, the MuJoCo scene, and point-cloud sampling for the model, so
there is one source of truth for geometry.

World frame: robot base at the origin, z up, the thrower is on the +x side.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Material:
    name: str
    restitution: float  # e: normal speed kept after a bounce, v_n' = -e * v_n
    friction: float  # fraction of tangential speed lost in a bounce, v_t' = (1 - friction) * v_t
    rgba: tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0)


# Fixed elasticity values per object type. Stage 3+ randomizes on top of these.
MATERIALS = {
    "wood_floor": Material("wood_floor", restitution=0.75, friction=0.10, rgba=(0.55, 0.40, 0.25, 1.0)),
    "drywall": Material("drywall", restitution=0.60, friction=0.15, rgba=(0.85, 0.85, 0.80, 0.15)),
    "ceiling": Material("ceiling", restitution=0.50, friction=0.15, rgba=(0.90, 0.90, 0.90, 0.05)),
    "plywood": Material("plywood", restitution=0.70, friction=0.10, rgba=(0.80, 0.65, 0.40, 0.9)),
    "rubber": Material("rubber", restitution=0.90, friction=0.30, rgba=(0.20, 0.20, 0.20, 0.9)),
    "foam": Material("foam", restitution=0.30, friction=0.40, rgba=(0.30, 0.60, 0.90, 0.9)),
}


@dataclass
class Panel:
    """A two-sided rectangle centered at `center` spanning +/- half_extents along (u, v)."""

    name: str
    center: np.ndarray
    normal: np.ndarray
    u: np.ndarray
    half_extents: tuple[float, float]
    material: Material

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=float)
        n = np.asarray(self.normal, dtype=float)
        self.normal = n / np.linalg.norm(n)
        u = np.asarray(self.u, dtype=float)
        u = u - u.dot(self.normal) * self.normal  # make u exactly in-plane
        self.u = u / np.linalg.norm(u)

    @property
    def v(self) -> np.ndarray:
        return np.cross(self.normal, self.u)

    @property
    def area(self) -> float:
        return 4.0 * self.half_extents[0] * self.half_extents[1]

    def contains(self, p: np.ndarray, tol: float = 1e-9) -> bool:
        """Whether the in-plane projection of p lies within the rectangle."""
        d = p - self.center
        return (abs(d.dot(self.u)) <= self.half_extents[0] + tol
                and abs(d.dot(self.v)) <= self.half_extents[1] + tol)

    def rotation(self) -> np.ndarray:
        """R_WP with columns (u, v, n): panel frame expressed in world."""
        return np.column_stack([self.u, self.v, self.normal])

    def to_vector(self) -> np.ndarray:
        """Flat 13-d encoding: center(3), normal(3), u(3), half_extents(2), restitution, friction."""
        return np.concatenate([self.center, self.normal, self.u, self.half_extents,
                               [self.material.restitution, self.material.friction]])


PANEL_VECTOR_DIM = 13


@dataclass
class Room:
    panels: list[Panel] = field(default_factory=list)
    # Axis-aligned bounds of the enclosing box, used for sanity checks and sampling.
    lo: np.ndarray = field(default_factory=lambda: np.zeros(3))
    hi: np.ndarray = field(default_factory=lambda: np.ones(3))

    def sample_point_cloud(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Uniformly sample n points over all panel surfaces (area-weighted). Returns (n, 3)."""
        areas = np.array([p.area for p in self.panels])
        idx = rng.choice(len(self.panels), size=n, p=areas / areas.sum())
        ab = rng.uniform(-1.0, 1.0, size=(n, 2))
        pts = np.empty((n, 3))
        for i, panel in enumerate(self.panels):
            m = idx == i
            pts[m] = (panel.center
                      + (ab[m, :1] * panel.half_extents[0]) * panel.u
                      + (ab[m, 1:] * panel.half_extents[1]) * panel.v)
        return pts

    def panel_array(self, max_panels: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(P, 13) panel encodings, zero-padded to max_panels, plus a validity mask."""
        P = max_panels or len(self.panels)
        assert len(self.panels) <= P
        arr = np.zeros((P, PANEL_VECTOR_DIM))
        mask = np.zeros(P, dtype=bool)
        for i, p in enumerate(self.panels):
            arr[i] = p.to_vector()
            mask[i] = True
        return arr, mask


def box_room(lo=(-1.5, -3.0, 0.0), hi=(4.5, 3.0, 3.0)) -> Room:
    """Closed box room: floor, ceiling and four walls with inward-facing normals."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c = (lo + hi) / 2
    h = (hi - lo) / 2
    M = MATERIALS
    ex, ey, ez = np.eye(3)
    panels = [
        Panel("floor", [c[0], c[1], lo[2]], ez, ex, (h[0], h[1]), M["wood_floor"]),
        Panel("ceiling", [c[0], c[1], hi[2]], -ez, ex, (h[0], h[1]), M["ceiling"]),
        Panel("wall_x-", [lo[0], c[1], c[2]], ex, ey, (h[1], h[2]), M["drywall"]),
        Panel("wall_x+", [hi[0], c[1], c[2]], -ex, ey, (h[1], h[2]), M["drywall"]),
        Panel("wall_y-", [c[0], lo[1], c[2]], ey, ex, (h[0], h[2]), M["drywall"]),
        Panel("wall_y+", [c[0], hi[1], c[2]], -ey, ex, (h[0], h[2]), M["drywall"]),
    ]
    return Room(panels=panels, lo=lo, hi=hi)


def random_board(rng: np.random.Generator, room: Room, name: str,
                 material: Material | None = None, random_restitution: bool = False) -> Panel:
    """A free-standing flat board at a random position/orientation inside the room."""
    margin = 0.6
    center = rng.uniform(room.lo + margin, room.hi - margin)
    center[0] = rng.uniform(0.8, room.hi[0] - 1.5)  # keep it between robot and thrower
    yaw = rng.uniform(-np.pi, np.pi)
    tilt = rng.uniform(-np.pi / 4, np.pi / 4)  # vertical-ish, tilted up to 45 deg
    normal = np.array([np.cos(yaw) * np.cos(tilt), np.sin(yaw) * np.cos(tilt), np.sin(tilt)])
    up = np.array([0.0, 0.0, 1.0])
    u = np.cross(up, normal)
    half = (rng.uniform(0.4, 0.9), rng.uniform(0.3, 0.7))
    if material is None:
        material = MATERIALS[rng.choice(["plywood", "rubber", "foam"])]
    if random_restitution:
        material = Material(material.name, restitution=float(rng.uniform(0.3, 0.95)),
                            friction=float(rng.uniform(0.05, 0.4)), rgba=material.rgba)
    return Panel(name, center, normal, u, half, material)


def make_room(stage: int, rng: np.random.Generator) -> Room:
    """Procedural room for each project stage.

    1: plain box room (throws go straight to the robot).
    2: box room + one flat board with a random pose and material.
    3: box room + 1-3 boards with fully randomized restitution/friction.
    4: randomized room dimensions + 2-4 random boards.
    """
    if stage == 1:
        return box_room()
    if stage == 2:
        room = box_room()
        room.panels.append(random_board(rng, room, "board_0"))
        return room
    if stage == 3:
        room = box_room()
        for i in range(rng.integers(1, 4)):
            room.panels.append(random_board(rng, room, f"board_{i}", random_restitution=True))
        return room
    if stage == 4:
        lo = np.array([-rng.uniform(1.0, 2.0), -rng.uniform(2.0, 4.0), 0.0])
        hi = np.array([rng.uniform(4.0, 6.0), rng.uniform(2.0, 4.0), rng.uniform(2.5, 4.0)])
        room = box_room(lo, hi)
        for i in range(rng.integers(2, 5)):
            room.panels.append(random_board(rng, room, f"board_{i}", random_restitution=True))
        return room
    raise ValueError(f"unknown stage {stage}")
