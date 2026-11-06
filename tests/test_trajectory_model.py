import numpy as np
import pytest
import torch

from catcher.trajectory.config import TrajectoryConfig
from catcher.trajectory.evaluate import coverage_90, parabola_baseline, position_rmse, rollout, sampled_rollouts
from catcher.trajectory.model import TrajectoryModel, build_wall_patches, farthest_point_sample
from catcher.trajectory.train import (
    future_predictions,
    gaussian_nll,
    make_optimizer,
    rollout_fraction_at,
    teacher_forced_loss,
    with_rollout_tokens,
)
from catcher.trajectory.windows import build_windows, sample_window_indices

CFG = TrajectoryConfig()
GRAVITY = torch.tensor(CFG.gravity)


def parabolas(n: int, seed: int = 0, steps: int = CFG.window_steps) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    p0 = torch.rand(n, 3, generator=g) * 2 - 1
    v0 = torch.rand(n, 3, generator=g) * 8 - 4
    t = CFG.dt * torch.arange(steps, dtype=torch.float32)[None, :, None]
    return p0[:, None] + v0[:, None] * t + 0.5 * GRAVITY * t**2


def room_cloud(n: int = 4096, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    points = torch.rand(n, 3, generator=g) * torch.tensor([6.0, 6.0, 3.0]) - torch.tensor([3.0, 3.0, 0.0])
    face = torch.randint(0, 3, (n,), generator=g)
    points[range(n), face] = torch.tensor([3.0, 3.0, 3.0])[face]  # snap each point to a surface
    return points


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return TrajectoryModel(CFG).eval()


@pytest.fixture(scope="module")
def walls(model):
    patches, centers = build_wall_patches(room_cloud(), CFG)
    return model.encode_walls(patches[None], centers[None])


def windows(n=8, h=None, t_now=None, positions=None, **kw):
    positions = parabolas(n) if positions is None else positions
    if h is None:
        t_now, h = sample_window_indices(positions.shape[1], n, CFG, torch.Generator().manual_seed(1))
    else:
        t_now, h = torch.full((n,), t_now), torch.full((n,), h)
    return positions, build_windows(positions, t_now, h, CFG, **kw)


def test_parameter_count_matches_the_plan():
    count = sum(p.numel() for p in TrajectoryModel(CFG).parameters())
    assert 1.3e6 < count < 1.7e6  # section 4 says about 1.5M


def test_window_contents():
    positions, b = windows(n=2, h=3, t_now=30)
    C, K = CFG.chunk_len, CFG.max_history
    # Newest chunk ends exactly at t_now; earlier chunks run backward from it, and the rest is padding.
    assert b.mask[0, :K].tolist() == [False] * (K - 3) + [True] * 3
    np.testing.assert_allclose(b.chunks[0, K - 1], positions[0, 26:31] / 3.0)
    np.testing.assert_allclose(b.chunks[0, K - 2], positions[0, 21:26] / 3.0)
    assert b.chunks[0, :K - 3].abs().sum() == 0
    # Targets are the next N C steps relative to p_now, and the same steps are the teacher-forced tokens.
    np.testing.assert_allclose(b.targets[0, 0], (positions[0, 31:36] - positions[0, 30]) / 3.0, atol=1e-6)
    np.testing.assert_allclose(b.chunks[0, K], positions[0, 31:36] / 3.0)
    np.testing.assert_allclose(b.p_now[0], positions[0, 30] / 3.0)
    # Time offsets are for each chunk's first position, relative to t_now.
    assert b.time_offsets[0, K - 1] == pytest.approx(-0.04)
    assert b.time_offsets[0, K] == pytest.approx(0.01)
    assert b.time_offsets[0, K + 1] == pytest.approx(0.06)
    # Slot K-1 predicts future chunk 0, and all N chunks have a token.
    assert CFG.num_tokens - (K - 1) == CFG.future_chunks


def test_balanced_sampling_makes_every_history_length_equally_common():
    t_now, h = sample_window_indices(CFG.window_steps, 90000, CFG, torch.Generator().manual_seed(0), mode="balanced")
    shares = torch.bincount(h, minlength=11)[2:] / 90000
    assert shares.min() > 0.105 and shares.max() < 0.117  # 1 / 9 = 0.111 each
    assert (h * CFG.chunk_len <= t_now + 1).all() and t_now.max() == 49
    assert (t_now[h == 2].min(), t_now[h == 10].min(), t_now[h == 10].max()) == (9, 49, 49)
    assert len(t_now[h == 5].unique()) == 49 - 24 + 1  # every t_now that fits is reachable


def test_t_now_first_sampling_keeps_the_plans_mix():
    _, h = sample_window_indices(CFG.window_steps, 90000, CFG, torch.Generator().manual_seed(0), mode="t_now_first")
    shares = torch.bincount(h, minlength=11).float() / 90000
    assert shares[2] == pytest.approx(0.334, abs=0.01) and shares[10] == pytest.approx(0.0027, abs=0.001)


def test_index_sampling_stays_in_bounds():
    t_now, h = sample_window_indices(CFG.window_steps, 20000, CFG, torch.Generator().manual_seed(0), mode="t_now_first")
    assert t_now.min() == 9 and t_now.max() == 49
    assert h.min() == 2 and h.max() == 10
    assert (h * CFG.chunk_len <= t_now + 1).all()


def test_noise_only_touches_history():
    clean = build_windows(parabolas(4), torch.full((4,), 40), torch.full((4,), 5), CFG)
    noisy = build_windows(parabolas(4), torch.full((4,), 40), torch.full((4,), 5), CFG, noise_std=0.01,
                          generator=torch.Generator().manual_seed(0))
    K = CFG.max_history
    assert (noisy.chunks[:, :K][noisy.mask[:, :K]] != clean.chunks[:, :K][clean.mask[:, :K]]).all()
    assert noisy.chunks[:, :K - 5].abs().sum() == 0  # padding stays zero
    np.testing.assert_array_equal(noisy.chunks[:, K:], clean.chunks[:, K:])
    # Targets stay clean positions, relative to the last *observed* position.
    p_true = parabolas(4)[:, 40] / 3.0
    np.testing.assert_allclose(noisy.targets + noisy.p_now[:, None, None] - p_true[:, None, None],
                               clean.targets, atol=1e-5)


def test_output_shapes_and_clamp(model, walls):
    _, b = windows()
    pred = model(b.chunks, b.time_offsets, b.mask, walls.expand(8, -1, -1), b.p_now)
    assert pred.mean.shape == (8, CFG.num_tokens, 5, 3) and pred.logvar.shape == pred.mean.shape
    assert torch.isfinite(pred.mean).all() and torch.isfinite(pred.logvar).all()
    big = model.head.bias.data.clone()
    model.head.bias.data[:] = 1e3
    assert model(b.chunks, b.time_offsets, b.mask, walls.expand(8, -1, -1), b.p_now).logvar.max() == CFG.logvar_max
    model.head.bias.data[:] = -1e3
    assert model(b.chunks, b.time_offsets, b.mask, walls.expand(8, -1, -1), b.p_now).logvar.min() == CFG.logvar_min
    model.head.bias.data.copy_(big)


@torch.no_grad()
def test_decoder_is_causal(model, walls):
    _, b = windows(n=2, h=5, t_now=40)
    w = walls.expand(2, -1, -1)
    base = model(b.chunks, b.time_offsets, b.mask, w).mean
    edited = b.chunks.clone()
    slot = CFG.max_history + 2
    edited[:, slot] += 1.0
    changed = model(edited, b.time_offsets, b.mask, w).mean
    np.testing.assert_allclose(base[:, :slot], changed[:, :slot], atol=1e-5)
    assert (base[:, slot:] - changed[:, slot:]).abs().amax() > 1e-3


@torch.no_grad()
def test_padding_is_ignored(model, walls):
    _, b = windows(n=2, h=4, t_now=40)
    w = walls.expand(2, -1, -1)
    base = model(b.chunks, b.time_offsets, b.mask, w).mean
    junk, offsets = b.chunks.clone(), b.time_offsets.clone()
    junk[:, :CFG.max_history - 4] = 123.0
    offsets[:, :CFG.max_history - 4] = -9.0
    out = model(junk, offsets, b.mask, w).mean
    np.testing.assert_allclose(base[:, CFG.max_history - 4:], out[:, CFG.max_history - 4:], atol=1e-5)


def test_wall_encoder_ignores_point_order(model):
    cloud = room_cloud()
    shuffled = cloud[torch.randperm(len(cloud), generator=torch.Generator().manual_seed(3))]
    a = model.encode_walls(*[x[None] for x in build_wall_patches(cloud, CFG)])
    b = model.encode_walls(*[x[None] for x in build_wall_patches(shuffled, CFG)])
    np.testing.assert_allclose(a.detach(), b.detach(), atol=1e-4)


def test_farthest_point_sampling_spreads_out():
    cloud = room_cloud() / CFG.length_scale
    chosen = cloud[farthest_point_sample(cloud, CFG.n_patches)]
    assert len(set(farthest_point_sample(cloud, CFG.n_patches).tolist())) == CFG.n_patches
    nearest = (torch.cdist(chosen, chosen) + 10 * torch.eye(CFG.n_patches)).amin(1)
    rng = torch.randperm(len(cloud), generator=torch.Generator().manual_seed(0))[:CFG.n_patches]
    random_nearest = (torch.cdist(cloud[rng], cloud[rng]) + 10 * torch.eye(CFG.n_patches)).amin(1)
    assert nearest.min() > random_nearest.min()


def test_walls_change_the_output(model, walls):
    _, b = windows(n=2, h=5, t_now=40)
    other = model.encode_walls(*[x[None] for x in build_wall_patches(room_cloud(seed=9) * 0.5, CFG)])
    a = model(b.chunks, b.time_offsets, b.mask, walls.expand(2, -1, -1)).mean
    c = model(b.chunks, b.time_offsets, b.mask, other.expand(2, -1, -1)).mean
    assert (a - c).abs().amax() > 1e-4


def test_rollout_matches_teacher_forcing_when_fed_its_own_means(model, walls):
    """Feeding a mean rollout's chunks through forward() must reproduce every prediction."""
    _, b = windows(n=3, h=5, t_now=40)
    w = walls.expand(3, -1, -1)
    r = rollout(model, b, w)
    K, N = CFG.max_history, CFG.future_chunks
    chunks, offsets = b.chunks.clone(), b.time_offsets.clone()
    chunks[:, K:] = b.p_now[:, None, None] + r.mean[:, :N - 1]
    tf = future_predictions(model(chunks, offsets, b.mask, w), CFG)
    np.testing.assert_allclose(tf.mean.detach(), r.mean, atol=1e-5)
    np.testing.assert_allclose(tf.logvar.detach(), r.logvar, atol=1e-5)


def test_rollout_does_not_peek_at_the_future(model, walls):
    positions, b = windows(n=2, h=5, t_now=40)
    w = walls.expand(2, -1, -1)
    base = rollout(model, b, w)
    b.chunks[:, CFG.max_history:] = 99.0
    b.targets[:] = 99.0
    other = rollout(model, b, w)
    np.testing.assert_array_equal(base.mean, other.mean)


def test_sampled_rollout(model, walls):
    _, b = windows(n=2, h=5, t_now=40)
    r = sampled_rollouts(model, b, walls.expand(2, -1, -1), num_samples=6, generator=torch.Generator().manual_seed(0))
    assert r.mean.shape == (2, 6, 10, 5, 3)
    assert torch.equal(r.mean[:, 0, 0], r.mean[:, 1, 0])  # first chunk is deterministic per sample
    assert (r.mean[:, 0, 1] - r.mean[:, 1, 1]).abs().amax() > 0  # samples diverge after that


def test_physics_residual_is_exact_for_parabolas():
    """With the network's output zeroed, the flag alone must extrapolate a parabola."""
    cfg = TrajectoryConfig(physics_residual=True)
    m = TrajectoryModel(cfg).eval()
    torch.nn.init.zeros_(m.head.weight)
    torch.nn.init.zeros_(m.head.bias)
    positions = parabolas(3)
    b = build_windows(positions, torch.full((3,), 40), torch.full((3,), 5), cfg)
    w = torch.zeros(3, cfg.n_patches, cfg.d_model)
    pred = future_predictions(m(b.chunks, b.time_offsets, b.mask, w, b.p_now), cfg)
    assert position_rmse(pred.mean.detach(), b.targets, cfg).max() < 1e-4
    off = TrajectoryModel(TrajectoryConfig()).eval()
    torch.nn.init.zeros_(off.head.weight)
    torch.nn.init.zeros_(off.head.bias)
    assert off(b.chunks, b.time_offsets, b.mask, w).mean.abs().max() == 0  # flag off by default


def test_rollout_tokens_make_training_score_exactly_what_inference_produces(model, walls):
    _, b = windows(n=4, h=5, t_now=40)
    w = walls.expand(4, -1, -1)
    swapped = with_rollout_tokens(model, b, w, fraction=1.0)
    trained = future_predictions(model(swapped.chunks, swapped.time_offsets, swapped.mask, w), CFG)
    r = rollout(model, b, w)
    np.testing.assert_allclose(trained.mean.detach(), r.mean, atol=1e-5)
    np.testing.assert_allclose(trained.logvar.detach(), r.logvar, atol=1e-5)
    # History, targets and masks are untouched, and the original batch isn't modified.
    K = CFG.max_history
    assert torch.equal(swapped.chunks[:, :K], b.chunks[:, :K]) and torch.equal(swapped.targets, b.targets)
    assert not torch.equal(swapped.chunks[:, K:], b.chunks[:, K:])


def test_rollout_tokens_respect_the_fraction(model, walls):
    positions, b = windows(n=200, h=5, t_now=40, positions=parabolas(200))
    w = walls.expand(200, -1, -1)
    assert with_rollout_tokens(model, b, w, fraction=0.0) is b
    swapped = with_rollout_tokens(model, b, w, fraction=0.5, generator=torch.Generator().manual_seed(0))
    changed = (swapped.chunks != b.chunks).flatten(1).any(1).float().mean().item()
    assert changed == pytest.approx(0.5, abs=0.12)


def test_rollout_fraction_ramp():
    assert rollout_fraction_at(0, CFG) == 0 and rollout_fraction_at(1000, CFG) == 0
    assert rollout_fraction_at(3000, CFG) == pytest.approx(0.5)
    assert rollout_fraction_at(5000, CFG) == 1 and rollout_fraction_at(20000, CFG) == 1
    off = TrajectoryConfig(rollout_fraction=0.0)
    assert rollout_fraction_at(20000, off) == 0


def test_nll_formula():
    from catcher.trajectory.model import Prediction
    mean = torch.zeros(1, 1, 1, 3)
    logvar = torch.full((1, 1, 1, 3), np.log(4.0), dtype=torch.float32)
    targets = torch.full((1, 1, 1, 3), 2.0)
    expected = 3 * (4.0 / (2 * 4.0) + 0.5 * np.log(4.0))
    assert gaussian_nll(Prediction(mean, logvar), targets).item() == pytest.approx(expected, rel=1e-5)


def test_baseline_is_exact_without_noise_and_handles_padding():
    positions = parabolas(4)
    for h in (2, 5, 10):
        b = build_windows(positions, torch.full((4,), 49), torch.full((4,), h), CFG)
        assert position_rmse(parabola_baseline(b, CFG), b.targets, CFG).max() < 2e-4


def test_baseline_error_matches_least_squares_theory():
    """With noise, the error at a horizon is known in closed form."""
    positions = parabolas(4000, steps=CFG.window_steps)
    sigma = 0.005
    b = build_windows(positions, torch.full((4000,), 49), torch.full((4000,), 10), CFG, noise_std=sigma,
                      generator=torch.Generator().manual_seed(0))
    rmse = position_rmse(parabola_baseline(b, CFG), b.targets, CFG)
    t_hist = CFG.dt * (np.arange(50) - 49)
    A = np.stack([np.ones(50), t_hist], axis=1)
    cov = np.linalg.inv(A.T @ A) * sigma**2
    t = 0.5
    # Prediction and target are both relative to the observed p_now, so its noise cancels.
    x = np.array([1.0, t])
    var = x @ cov @ x
    assert rmse[-1, -1].item() == pytest.approx(np.sqrt(3 * var), rel=0.05)


def test_coverage_formula():
    targets = torch.zeros(20000, 1, 1, 3)
    g = torch.Generator().manual_seed(0)
    mean = torch.randn(20000, 1, 1, 3, generator=g) * 0.5
    logvar = torch.full_like(mean, np.log(0.25))
    assert coverage_90(mean, logvar, targets).item() == pytest.approx(0.90, abs=0.01)


def test_training_step_updates_everything_and_lowers_loss():
    torch.manual_seed(0)
    m = TrajectoryModel(CFG)
    patches, centers = build_wall_patches(room_cloud(), CFG)
    opt, sched = make_optimizer(m, total_steps=40)
    positions = parabolas(32)
    t_now, h = sample_window_indices(positions.shape[1], 32, CFG, torch.Generator().manual_seed(0))
    b = build_windows(positions, t_now, h, CFG)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    losses = []
    for _ in range(40):
        loss, _ = teacher_forced_loss(m, b, m.encode_walls(patches[None], centers[None]).expand(32, -1, -1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), CFG.grad_clip)
        opt.step()
        sched.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
    frozen = [k for k, v in m.state_dict().items() if torch.equal(v, before[k])]
    assert not frozen, frozen
