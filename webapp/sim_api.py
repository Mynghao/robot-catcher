"""Translate between the web UI's JSON and the catcher NumPy simulator.

All geometry math lives here and in catcher/, never in the browser: the UI sends
a config, gets back fully resolved panels, and draws exactly what the simulator
collides against. Every function takes and returns plain JSON-able data.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

from catcher.ball_sim import simulate
from catcher.room import MATERIALS, Material, Panel, Room, box_room
from catcher.throws import (
    CATCH_Z_MIN,
    OBS_RATE_HZ,
    REACH_MAX,
    REACH_MIN,
    SHOULDER,
    STAGE_BOUNCES,
    THROW_HI,
    THROW_LO,
    Throw,
    first_catchable_index,
    sample_stage_throw,
    sample_throw,
    throws_to_arrays,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "web"
SURFACES = ("floor", "ceiling", "wall_x-", "wall_x+", "wall_y-", "wall_y+")
DEFAULT_SURFACE_MATERIALS = {"floor": "wood_floor", "ceiling": "ceiling", "wall_x-": "drywall",
                             "wall_x+": "drywall", "wall_y-": "drywall", "wall_y+": "drywall"}
MAX_THROWS = 500
MAX_BOARDS = 20
MIN_RESTITUTION = 0.01  # e = 0 leaves the ball exactly on the plane, where its side is undefined


class ApiError(ValueError):
    """A bad request; the message is shown to the user."""


def _number(value, name: str, lo: float = -math.inf, hi: float = math.inf) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ApiError(f"{name} must be a number")
    if not lo <= value <= hi:
        raise ApiError(f"{name} must be between {lo:g} and {hi:g}")
    return float(value)


def _integer(value, name: str, lo: int, hi: int) -> int:
    number = _number(value, name, lo, hi)
    if number != int(number):
        raise ApiError(f"{name} must be a whole number")
    return int(number)


def _vector(value, name: str, size: int = 3) -> np.ndarray:
    if not isinstance(value, list) or len(value) != size:
        raise ApiError(f"{name} must have {size} values")
    return np.array([_number(v, f"{name}[{i}]") for i, v in enumerate(value)])


def _section(parent: dict, key: str) -> dict:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ApiError(f"missing {key}")
    return value


def _material(cfg: dict, label: str) -> Material:
    name = cfg.get("material")
    base = MATERIALS.get(name)
    return Material(name if base else "custom",
                    restitution=_number(cfg.get("restitution"), f"{label} restitution", MIN_RESTITUTION, 1.0),
                    friction=_number(cfg.get("friction"), f"{label} friction", 0.0, 1.0),
                    rgba=base.rgba if base else (0.6, 0.5, 0.8, 0.9))


def _board_panel(cfg: dict, index: int, room: Room) -> Panel:
    """A board from its center, yaw and tilt. Same convention as room.random_board."""
    label = f"board {index + 1}"
    center = _vector(cfg.get("center"), f"{label} center")
    if np.any(center <= room.lo) or np.any(center >= room.hi):
        raise ApiError(f"{label} center must be inside the room")
    yaw = math.radians(_number(cfg.get("yaw_deg"), f"{label} yaw", -360, 360))
    tilt = math.radians(_number(cfg.get("tilt_deg"), f"{label} tilt", -90, 90))
    half = (_number(cfg.get("width"), f"{label} width", 0.01, 50) / 2,
            _number(cfg.get("height"), f"{label} height", 0.01, 50) / 2)
    normal = np.array([math.cos(yaw) * math.cos(tilt), math.sin(yaw) * math.cos(tilt), math.sin(tilt)])
    u = np.cross([0.0, 0.0, 1.0], normal)
    if np.linalg.norm(u) < 1e-6:  # flat board (tilt = +/-90 deg): yaw sets its in-plane direction
        u = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    return Panel(f"board_{index}", center, normal, u, half, _material(cfg, label))


def room_from_config(config: dict) -> Room:
    """Build a Room of NumPy panels from the UI's config, validating every value."""
    room_cfg = _section(config, "room")
    x = _vector(room_cfg.get("x"), "room x range", 2)
    y = _vector(room_cfg.get("y"), "room y range", 2)
    height = _number(room_cfg.get("height"), "room height", 0.5, 50)
    if not (x[0] < 0 < x[1] and y[0] < 0 < y[1]):
        raise ApiError("the room must contain the robot at the origin: "
                       "x and y ranges need a negative minimum and a positive maximum")
    room = box_room([x[0], y[0], 0.0], [x[1], y[1], height])
    surfaces = _section(room_cfg, "surfaces")
    for panel in room.panels:
        panel.material = _material(_section(surfaces, panel.name), panel.name)
    boards = config.get("boards", [])
    if not isinstance(boards, list) or len(boards) > MAX_BOARDS:
        raise ApiError(f"at most {MAX_BOARDS} boards are supported")
    room.panels.extend(_board_panel(b, i, room) for i, b in enumerate(boards))
    return room


def _throw_region(config: dict, room: Room) -> tuple[np.ndarray, np.ndarray]:
    region = _section(config, "throw_region")
    lo = _vector(region.get("lo"), "throw region min")
    hi = _vector(region.get("hi"), "throw region max")
    if np.any(lo > hi):
        raise ApiError("throw region min must not exceed its max")
    if np.any(lo <= room.lo) or np.any(hi >= room.hi):
        raise ApiError("the throw region must be inside the room")
    return lo, hi


def config_from_room(room: Room) -> dict:
    """Inverse of room_from_config, so a procedural room can be opened in the editor."""
    def surface(panel: Panel) -> dict:
        m = panel.material
        return {"material": m.name, "restitution": round(m.restitution, 4), "friction": round(m.friction, 4)}

    boards = []
    for panel in room.panels:
        if panel.name in SURFACES:
            continue
        n = panel.normal
        boards.append({"center": np.round(panel.center, 4).tolist(),
                       "yaw_deg": round(math.degrees(math.atan2(n[1], n[0])), 3),
                       "tilt_deg": round(math.degrees(math.asin(np.clip(n[2], -1, 1))), 3),
                       "width": round(2 * panel.half_extents[0], 4),
                       "height": round(2 * panel.half_extents[1], 4),
                       **surface(panel)})
    return {
        "room": {"x": [round(float(room.lo[0]), 4), round(float(room.hi[0]), 4)],
                 "y": [round(float(room.lo[1]), 4), round(float(room.hi[1]), 4)],
                 "height": round(float(room.hi[2]), 4),
                 "surfaces": {p.name: surface(p) for p in room.panels if p.name in SURFACES}},
        "boards": boards,
        "throw_region": {"lo": THROW_LO.tolist(), "hi": THROW_HI.tolist()},
    }


def _panels_json(room: Room) -> list[dict]:
    return [{"name": p.name, "center": p.center.tolist(), "normal": p.normal.tolist(), "u": p.u.tolist(),
             "v": p.v.tolist(), "half_extents": list(p.half_extents), "restitution": p.material.restitution,
             "friction": p.material.friction, "rgba": list(p.material.rgba)} for p in room.panels]


def _bounds_json(room: Room) -> dict:
    return {"lo": room.lo.tolist(), "hi": room.hi.tolist()}


def default_config() -> dict:
    room = box_room()
    config = config_from_room(room)
    plywood = MATERIALS["plywood"]
    config["boards"] = [{"center": [1.2, 1.2, 1.2], "yaw_deg": -60.0, "tilt_deg": 10.0, "width": 1.8,
                         "height": 1.4, "material": plywood.name, "restitution": plywood.restitution,
                         "friction": plywood.friction}]
    return config


def init() -> dict:
    """Everything the UI needs on page load."""
    return {
        "config": default_config(),
        "materials": {name: {"restitution": m.restitution, "friction": m.friction, "rgba": list(m.rgba)}
                      for name, m in MATERIALS.items()},
        "stages": {str(s): list(b) for s, b in STAGE_BOUNCES.items()},
        "catch_zone": {"shoulder": SHOULDER.tolist(), "reach_min": REACH_MIN, "reach_max": REACH_MAX,
                       "z_min": CATCH_Z_MIN},
        "obs_rate_hz": OBS_RATE_HZ,
        "max_throws": MAX_THROWS,
    }


def preview_room(request: dict) -> dict:
    """Validate a config and return the panels the simulator would use."""
    config = _section(request, "config")
    room = room_from_config(config)
    lo, hi = _throw_region(config, room)
    return {"panels": _panels_json(room), "bounds": _bounds_json(room),
            "throw_region": {"lo": lo.tolist(), "hi": hi.tolist()}}


def _manual_throw(room: Room, p0: np.ndarray, v0: np.ndarray, t_end: float) -> Throw:
    if np.any(p0 <= room.lo) or np.any(p0 >= room.hi):
        raise ApiError("the start position must be inside the room")
    times = np.arange(0.0, t_end, 1.0 / OBS_RATE_HZ)
    traj = simulate(room, p0, v0, t_end=t_end)
    pos, vel = traj.sample(times)
    catch = first_catchable_index(pos)
    if catch is None:  # -1 means the ball never enters the catch zone
        return Throw(room, p0, v0, traj, times, pos, vel, -1, -1)
    return Throw(room, p0, v0, traj, times, pos, vel, catch, traj.num_bounces_before(times[catch]))


def _run(request: dict) -> tuple[list[Throw], dict[str, np.ndarray]]:
    """Sample the requested throws. Deterministic for a given request and seed."""
    seed = _integer(request.get("seed"), "seed", 0, 2**32 - 1)
    t_end = _number(request.get("t_end"), "duration", 0.1, 10.0)
    noise_std = _number(request.get("noise_std"), "observation noise", 0.0, 0.5)
    throw = _section(request, "throw")
    rng = np.random.default_rng(seed)
    if request.get("source") == "stage":
        stage = request.get("stage")
        if stage not in STAGE_BOUNCES:
            raise ApiError("unknown stage")
        n = _integer(throw.get("n"), "number of throws", 1, MAX_THROWS)
        throws = [sample_stage_throw(stage, rng, t_end) for _ in range(n)]
    else:
        config = _section(request, "config")
        room = room_from_config(config)
        if throw.get("mode") == "manual":
            throws = [_manual_throw(room, _vector(throw.get("p0"), "start position"),
                                    _vector(throw.get("v0"), "start velocity"), t_end)]
        else:
            lo, hi = _throw_region(config, room)
            n = _integer(throw.get("n"), "number of throws", 1, MAX_THROWS)
            min_b = _integer(throw.get("min_bounces"), "min bounces", 0, 20)
            max_b = _integer(throw.get("max_bounces"), "max bounces", 0, 20)
            if min_b > max_b:
                raise ApiError("min bounces must not exceed max bounces")
            throws = [sample_throw(room, rng, min_b, max_b, t_end=t_end, throw_lo=lo, throw_hi=hi)
                      for _ in range(n)]
    max_panels = max(10, max(len(t.room.panels) for t in throws))
    return throws, throws_to_arrays(throws, rng, noise_std, max_panels=max_panels)


def _run_or_explain(request: dict) -> tuple[list[Throw], dict[str, np.ndarray]]:
    try:
        return _run(request)
    except RuntimeError as e:
        raise ApiError("No throw met the constraints for this geometry. Try allowing a wider bounce "
                       "range, a longer duration, or moving boards that block the path to the robot.") from e


def _rounded(array: np.ndarray) -> list:
    return np.round(array, 4).tolist()


def simulate_runs(request: dict) -> dict:
    """Run the simulation and return every throw for playback."""
    throws, arrays = _run_or_explain(request)
    is_stage = request.get("source") == "stage"
    runs = []
    for i, th in enumerate(throws):
        run = {
            "index": i,
            "p0": _rounded(th.p0),
            "v0": _rounded(th.v0),
            "pos": _rounded(th.pos),
            "obs": _rounded(arrays["obs"][i]),
            "speed": _rounded(np.linalg.norm(th.vel, axis=1)),
            "bounces": [{"t": round(b.t, 4), "position": _rounded(b.position),
                         "surface": th.room.panels[b.panel_index].name,
                         "restitution": th.room.panels[b.panel_index].material.restitution,
                         "friction": th.room.panels[b.panel_index].material.friction,
                         "speed_in": round(float(np.linalg.norm(b.v_in)), 3),
                         "speed_out": round(float(np.linalg.norm(b.v_out)), 3)} for b in th.traj.bounces],
            "catch_index": th.catch_index,
            "catch_time": float(th.times[th.catch_index]) if th.catch_index >= 0 else None,
            "bounces_before_catch": int(th.bounces_before_catch),
            "at_rest": th.traj.at_rest,
            "stop_time": round(th.traj.t_end, 4),
            "panels": _panels_json(th.room),
            "bounds": _bounds_json(th.room),
        }
        if is_stage:
            run["config"] = config_from_room(th.room)
        runs.append(run)
    return {"times": _rounded(throws[0].times), "runs": runs}


def save_dataset(request: dict) -> dict:
    """Write the throws of `run` to data/web/<name>.npz. Re-running the same request and
    seed reproduces exactly what the UI displayed, so nothing is cached server-side."""
    run_request = _section(request, "run")
    _, arrays = _run_or_explain(run_request)
    name = re.sub(r"[^\w.-]+", "_", str(request.get("name") or "")).strip("._") or "dataset"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{name}.npz"
    suffix = 1
    while path.exists():  # never overwrite an earlier dataset
        path = DATA_DIR / f"{name}-{suffix}.npz"
        suffix += 1
    stage = run_request.get("stage") if run_request.get("source") == "stage" else 0
    np.savez_compressed(path, stage=np.array(stage), config_json=np.array(json.dumps(run_request)), **arrays)
    return {"path": str(path.relative_to(DATA_DIR.parent.parent)), "bytes": path.stat().st_size,
            "throws": len(arrays["pos"]),
            "arrays": {k: list(v.shape) for k, v in arrays.items()}}
