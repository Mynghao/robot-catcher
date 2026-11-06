import numpy as np
import pytest

from catcher.ball_sim import simulate
from catcher.trajectory.config import TrajectoryConfig
from catcher.trajectory.generate_data import (
    CLEARANCE_MARGIN,
    distance_to_surfaces,
    parabola_residuals,
    sample_cloud,
    sample_room,
    sample_throws,
)

CFG = TrajectoryConfig()


@pytest.fixture(scope="module")
def rooms():
    rng = np.random.default_rng(0)
    return [sample_room(rng) for _ in range(8)]


def test_rooms_match_the_plan(rooms):
    for r in rooms:
        assert 4 <= r.width <= 10 and 4 <= r.depth <= 10 and 3 <= r.height <= 6
        # The robot is at least 1 m from all four side walls, and the floor is at z = 0.
        assert np.all(np.abs(r.base) <= np.array([r.width, r.depth]) / 2 - 1.0 + 1e-9)
        walls = [p for p in r.room.panels if p.name.startswith("wall")]
        assert len(walls) == 4 and min(abs(p.center @ p.normal) for p in walls) >= 1.0 - 1e-9
        floor = next(p for p in r.room.panels if p.name == "floor")
        assert floor.center[2] == 0 and floor.normal[2] == pytest.approx(1)
        assert len(r.room.panels) == 6 + 6 * len(r.obstacles)


def test_throws_stay_clear_and_are_parabolas(rooms):
    rng = np.random.default_rng(1)
    t_end = CFG.window_steps * CFG.dt
    for r in rooms:
        throws, rate = sample_throws(r, 10, rng, CFG)
        assert 0 < rate <= 1
        assert distance_to_surfaces(r.room, throws["positions"]).min() >= CFG.ball_radius + CLEARANCE_MARGIN
        assert parabola_residuals(throws["positions"], CFG).max() < 1e-9
        for p0, v0, pos in zip(throws["p0"], throws["v0"], throws["positions"]):
            result = simulate(r.room, p0, v0, t_end=t_end)
            assert not result.bounces
            np.testing.assert_allclose(pos[0], p0)
        speed = np.linalg.norm(throws["v0"], axis=1)
        assert np.all((speed >= 3) & (speed <= 10))


def test_clipping_a_corner_between_samples_is_rejected(rooms):
    """A fast ball can cross an obstacle's corner between 10 ms samples; the simulator must catch it."""
    room = next(r for r in rooms if r.obstacles)
    (center, half) = room.obstacles[0]
    corner = room.to_world(center + half * np.array([1, 1, 1]))
    direction = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
    start = corner - 0.05 * room.rotate(direction) - np.array([0, 0, 0.01])
    assert simulate(room.room, start, 10 * room.rotate(direction), t_end=0.02).bounces


def test_cloud_is_on_surfaces_and_area_weighted(rooms):
    r = rooms[0]
    points, normals = sample_cloud(r.room, 4096, np.random.default_rng(2))
    assert distance_to_surfaces(r.room, points).max() < 1e-9
    assert points.shape == normals.shape == (4096, 3)
    floor_share = np.mean(np.abs(points[:, 2]) < 1e-9)
    total = sum(p.area for p in r.room.panels)
    floor_area = sum(p.area for p in r.room.panels if abs(p.center[2]) < 1e-9 and abs(p.normal[2]) == 1)
    assert floor_share == pytest.approx(floor_area / total, abs=0.03)


def test_residual_check_catches_a_non_parabola():
    t = CFG.dt * np.arange(CFG.window_steps)
    drag = np.stack([np.exp(-t), 0 * t, 0.5 * np.array(CFG.gravity)[2] * t**2], axis=1)[None]
    assert parabola_residuals(drag, CFG).max() > 1e-3
