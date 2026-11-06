import copy
import json

import numpy as np
import pytest

import sim_api
from catcher.ball_sim import simulate
from catcher.room import make_room


def request(**overrides):
    base = {"source": "custom", "config": sim_api.default_config(), "seed": 0, "t_end": 3.0, "noise_std": 0.005,
            "throw": {"mode": "sample", "n": 4, "min_bounces": 1, "max_bounces": 2}}
    return {**base, **overrides}


def test_response_is_json_serializable_and_matches_simulator():
    result = sim_api.simulate_runs(request())
    json.dumps(result, allow_nan=False)
    assert len(result["runs"]) == 4
    room = sim_api.room_from_config(sim_api.default_config())
    for run in result["runs"]:
        assert 1 <= run["bounces_before_catch"] <= 2
        # The JSON trajectory is what the simulator produces from the same room and throw.
        traj = simulate(room, run["p0"], run["v0"], t_end=3.0)
        pos, _ = traj.sample(np.array(result["times"]))
        assert np.max(np.abs(pos - np.array(run["pos"]))) < 0.05


def test_preview_matches_panel_math():
    config = sim_api.default_config()
    config["boards"] = [{"center": [1.0, 0.5, 1.0], "yaw_deg": 90.0, "tilt_deg": 0.0, "width": 2.0, "height": 1.0,
                         "material": "rubber", "restitution": 0.9, "friction": 0.3}]
    board = sim_api.preview_room({"config": config})["panels"][-1]
    assert board["name"] == "board_0"
    np.testing.assert_allclose(board["normal"], [0, 1, 0], atol=1e-12)
    assert board["half_extents"] == [1.0, 0.5]
    assert board["restitution"] == 0.9


def test_flat_board_bounces_ball_straight_up():
    config = sim_api.default_config()
    config["boards"] = [{"center": [2.0, 0.0, 1.0], "yaw_deg": 0.0, "tilt_deg": 90.0, "width": 1.0, "height": 1.0,
                         "material": "rubber", "restitution": 0.8, "friction": 0.0}]
    throw = {"mode": "manual", "p0": [2.0, 0.0, 2.0], "v0": [0.0, 0.0, 0.0]}
    run = sim_api.simulate_runs(request(config=config, throw=throw))["runs"][0]
    first = run["bounces"][0]
    assert first["surface"] == "board_0"
    assert first["position"][2] == pytest.approx(1.0)
    assert first["speed_out"] == pytest.approx(0.8 * first["speed_in"], abs=1e-2)


@pytest.mark.parametrize("stage", [1, 2, 3, 4])
def test_procedural_room_round_trips_through_config(stage):
    room = make_room(stage, np.random.default_rng(stage))
    rebuilt = sim_api.room_from_config(sim_api.config_from_room(room))
    assert len(rebuilt.panels) == len(room.panels)
    for a, b in zip(room.panels, rebuilt.panels):
        np.testing.assert_allclose(a.center, b.center, atol=1e-3)
        np.testing.assert_allclose(a.normal, b.normal, atol=1e-3)
        np.testing.assert_allclose(a.u, b.u, atol=1e-3)
        np.testing.assert_allclose(a.half_extents, b.half_extents, atol=1e-3)
        assert a.material.restitution == pytest.approx(b.material.restitution, abs=1e-3)


@pytest.mark.parametrize("stage", [1, 2, 3, 4])
def test_stage_source_runs(stage):
    result = sim_api.simulate_runs(request(source="stage", stage=stage, config=None, throw={"n": 2}))
    json.dumps(result, allow_nan=False)
    assert len(result["runs"]) == 2 and "config" in result["runs"][0]


def test_saved_dataset_is_exactly_what_was_displayed(tmp_path, monkeypatch):
    monkeypatch.setattr(sim_api, "DATA_DIR", tmp_path / "data" / "web")
    req = request()
    shown = sim_api.simulate_runs(req)
    first = sim_api.save_dataset({"run": req, "name": "../../evil name"})
    second = sim_api.save_dataset({"run": req, "name": "../../evil name"})
    assert first["path"] != second["path"]  # never overwrites
    files = sorted((tmp_path / "data" / "web").iterdir())
    assert len(files) == 2 and all(f.parent == tmp_path / "data" / "web" for f in files)
    data = np.load(files[0])  # plain arrays: must load without pickle
    assert data["pos"].shape == (4, len(shown["times"]), 3)
    for i, run in enumerate(shown["runs"]):
        np.testing.assert_allclose(data["pos"][i], run["pos"], atol=1e-4)
        np.testing.assert_allclose(data["obs"][i], run["obs"], atol=1e-4)
    assert json.loads(str(data["config_json"]))["seed"] == 0


def test_manual_throw_that_misses_can_be_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(sim_api, "DATA_DIR", tmp_path)
    req = request(throw={"mode": "manual", "p0": [4.0, 2.5, 2.5], "v0": [0.0, 0.0, 0.0]}, t_end=0.2)
    run = sim_api.simulate_runs(req)["runs"][0]
    assert run["catch_index"] == -1 and run["catch_time"] is None
    assert sim_api.save_dataset({"run": req, "name": "m"})["throws"] == 1


def _break(path, value):
    def apply(config):
        node = config
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
    return apply


@pytest.mark.parametrize("mutate", [
    _break(["room", "x"], [1.0, 2.0]),  # robot outside
    _break(["room", "height"], None),  # cleared field in the UI
    _break(["room", "surfaces", "floor", "restitution"], 1.5),  # adds energy
    _break(["room", "surfaces", "floor", "restitution"], 0.0),
    _break(["boards", 0, "center"], [99.0, 0.0, 1.0]),
    _break(["boards", 0, "width"], -1.0),
    _break(["throw_region", "hi"], [99.0, 1.0, 1.0]),
])
def test_bad_config_gives_a_readable_error(mutate):
    config = copy.deepcopy(sim_api.default_config())
    mutate(config)
    with pytest.raises(sim_api.ApiError):
        sim_api.preview_room({"config": config})


def test_impossible_constraints_explain_themselves():
    with pytest.raises(sim_api.ApiError, match="No throw met"):
        sim_api.simulate_runs(request(throw={"mode": "sample", "n": 1, "min_bounces": 20, "max_bounces": 20}))
