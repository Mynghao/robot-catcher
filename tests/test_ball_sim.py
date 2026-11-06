import numpy as np
import pytest

from catcher.ball_sim import GRAVITY, simulate
from catcher.room import MATERIALS, Material, Panel, Room, box_room
from catcher.throws import STAGE_BOUNCES, generate_dataset, in_catch_zone, sample_stage_throw


def test_free_flight_matches_analytic():
    room = box_room()
    p0, v0 = np.array([2.0, 0.0, 1.5]), np.array([-1.0, 0.5, 2.0])
    traj = simulate(room, p0, v0, t_end=0.5)
    assert not traj.bounces
    t = np.linspace(0, 0.5, 20)
    pos, vel = traj.sample(t)
    np.testing.assert_allclose(pos, p0 + v0 * t[:, None] + 0.5 * GRAVITY * t[:, None] ** 2)
    np.testing.assert_allclose(vel, v0 + GRAVITY * t[:, None])


def test_floor_bounce_restitution_and_friction():
    floor = MATERIALS["wood_floor"]
    room = box_room()
    traj = simulate(room, [1.0, 0.0, 1.0], [1.0, 0.0, 0.0], t_end=2.0)
    b = traj.bounces[0]
    assert room.panels[b.panel_index].name == "floor"
    assert b.t == pytest.approx(np.sqrt(2 * 1.0 / 9.81))
    assert b.position[2] == pytest.approx(0.0, abs=1e-12)
    assert b.v_out[2] == pytest.approx(-floor.restitution * b.v_in[2])
    assert b.v_out[0] == pytest.approx((1 - floor.friction) * b.v_in[0])


def test_two_sided_board_and_bounds():
    board = Panel("board", [1.0, 0.0, 1.0], [1, 0, 0], [0, 1, 0], (0.5, 0.5), Material("m", 1.0, 0.0))
    room = Room([board], lo=np.full(3, -10), hi=np.full(3, 10))
    # Hit from the -x side: elastic, velocity flips in x.
    traj = simulate(room, [0.0, 0.0, 1.0], [5.0, 0.0, 0.0], t_end=0.3)
    assert len(traj.bounces) == 1 and traj.bounces[0].v_out[0] == pytest.approx(-5.0)
    # Miss the board (passes beside it).
    traj = simulate(room, [0.0, 2.0, 1.0], [5.0, 0.0, 0.0], t_end=0.3)
    assert not traj.bounces


def test_ball_never_leaves_closed_room():
    rng = np.random.default_rng(0)
    room = box_room()
    t = np.linspace(0, 5, 400)
    for _ in range(500):
        traj = simulate(room, rng.uniform(room.lo + 0.1, room.hi - 0.1), rng.normal(0, 8, 3), t_end=5.0)
        pos, _ = traj.sample(t)
        assert np.all(pos >= room.lo - 1e-6) and np.all(pos <= room.hi + 1e-6)


@pytest.mark.parametrize("stage", [1, 2, 3, 4])
def test_stage_throws_reach_catch_zone(stage):
    rng = np.random.default_rng(stage)
    for _ in range(5):
        th = sample_stage_throw(stage, rng)
        lo, hi = STAGE_BOUNCES[stage]
        assert lo <= th.bounces_before_catch <= hi
        assert in_catch_zone(th.pos[th.catch_index])


def test_dataset_shapes():
    d = generate_dataset(stage=2, n=4, seed=0, n_cloud_points=64)
    N, T = d["pos"].shape[:2]
    assert (N, T) == (4, len(d["times"]))
    assert d["obs"].shape == d["pos"].shape == d["vel"].shape
    assert d["panels"].shape == (4, 10, 13) and d["panel_mask"].sum(1).tolist() == [7] * 4
    assert d["cloud"].shape == (4, 64, 3)
