import numpy as np

from catcher.mujoco_scene import Q_READY, Kinematics, build_model, run_catch, solve_catch_ik
from catcher.throws import sample_stage_throw


def test_ik_reaches_target_inside_cone():
    kin = Kinematics(build_model())
    p, v = np.array([0.5, 0.1, 0.7]), np.array([-3.0, 0.0, -4.0])
    q, ok = solve_catch_ik(kin, p, v, Q_READY, cone_half_angle_deg=25)
    assert ok
    x, axis, _ = kin(q)
    assert np.linalg.norm(x - p) < 0.005
    assert np.degrees(np.arccos(axis.dot(-v / np.linalg.norm(v)))) <= 25.1


def test_stage1_oracle_catches():
    rng = np.random.default_rng(0)
    results = [run_catch(sample_stage_throw(1, rng)) for _ in range(5)]
    assert all(r.success for r in results)
