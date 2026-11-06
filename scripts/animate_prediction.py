"""Run a trained trajectory model on individual throws and animate its predictions over time.

Each frame is one moment in the throw: the model sees only the observations so far and
predicts the next 0.5 s. Runs on the CPU by default so it doesn't disturb a training run.

    python scripts/animate_prediction.py --run e2_clean --split val_throws --n 3
"""

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Ellipse

from catcher.trajectory.config import TrajectoryConfig
from catcher.trajectory.evaluate import rollout
from catcher.trajectory.model import TrajectoryModel, build_wall_patches
from catcher.trajectory.windows import build_windows

INK, MUTED, WALL = "#222222", "#6d6b63", "#b5b3ab"
TRUE_PATH, PREDICTED, BALL, OBSERVED = "#8a887f", "#2a78d6", "#eb6834", "#222222"


def load_model(run_dir: Path, device: str) -> tuple[TrajectoryModel, int | None, float]:
    """The final weights if the run has finished, otherwise the latest ones."""
    path = run_dir / "model.pt" if (run_dir / "model.pt").exists() else run_dir / "latest.pt"
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    noise = checkpoint["config"]["noise"]
    model = TrajectoryModel(TrajectoryConfig(obs_noise_std=noise)).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint.get("step"), noise


@torch.no_grad()
def predict_over_time(model: TrajectoryModel, positions: torch.Tensor, room: dict, noise: float, seed: int) -> dict:
    """One rollout per valid t_now, using all the history available at that moment (up to K chunks)."""
    cfg = model.cfg
    C = cfg.chunk_len
    t_now = torch.arange(C * cfg.min_history - 1, positions.shape[0] - cfg.future_chunks * C)
    n_history = torch.minimum((t_now + 1) // C, torch.tensor(cfg.max_history))
    batch = build_windows(positions[None].expand(len(t_now), -1, -1), t_now, n_history, cfg, noise_std=noise,
                          generator=torch.Generator().manual_seed(seed))
    patches, centers = build_wall_patches(torch.from_numpy(room["points"]), cfg)
    walls = model.encode_walls(patches[None], centers[None]).expand(len(t_now), -1, -1)
    result = rollout(model, batch, walls)
    s = cfg.length_scale
    mean = (result.mean * s + batch.p_now[:, None, None] * s).flatten(1, 2).numpy()  # metres, (F, N C, 3)
    sigma = (torch.exp(0.5 * result.logvar) * s).flatten(1, 2).numpy()
    truth = (batch.targets * s + batch.p_now[:, None, None] * s).flatten(1, 2).numpy()
    return {"t_now": t_now.numpy(), "n_history": n_history.numpy(), "mean": mean, "sigma": sigma, "truth": truth,
            "observed": (batch.chunks[:, :cfg.max_history] * s).flatten(1, 2).numpy(),
            "observed_mask": batch.mask[:, :cfg.max_history, None].expand(-1, -1, C).flatten(1).numpy(),
            "error": np.linalg.norm(mean - truth, axis=-1)}


def room_outline(room) -> np.ndarray:
    w, d = float(room["width"]) / 2, float(room["depth"]) / 2
    c, s = np.cos(room["yaw"]), np.sin(room["yaw"])
    xy = np.array([[-w, -d], [w, -d], [w, d], [-w, d], [-w, -d]]) - room["base"]
    return np.stack([c * xy[:, 0] - s * xy[:, 1], s * xy[:, 0] + c * xy[:, 1]], axis=1)


def animate(positions: np.ndarray, room, pred: dict, cfg: TrajectoryConfig, title: str, out: Path, fps: int = 8):
    """Side view along the throw's direction of travel, plus a top view, one frame per t_now."""
    origin = positions[0, :2]
    direction = (positions[-1, :2] - origin) / np.linalg.norm(positions[-1, :2] - origin)
    along = lambda p: (p[..., :2] - origin) @ direction

    fig, (side, top) = plt.subplots(1, 2, figsize=(11.5, 5.2), gridspec_kw={"width_ratios": [1.25, 1]})
    margin = 0.6
    d = along(positions)
    side.set(xlim=(d.min() - margin, d.max() + margin), ylim=(-0.15, float(room["height"]) + 0.15),
             xlabel="distance along the throw (m)", ylabel="height (m)", title="Side view")
    side.axhline(0, color=WALL, lw=1.5)
    side.axhline(float(room["height"]), color=WALL, lw=1.5)
    top.set(xlim=(positions[:, 0].min() - margin, positions[:, 0].max() + margin),
            ylim=(positions[:, 1].min() - margin, positions[:, 1].max() + margin),
            xlabel="x (m)", ylabel="y (m)", title="Top view (grey lines are the walls)", aspect="equal")
    top.plot(*room_outline(room).T, color=WALL, lw=1.5)
    for ax in (side, top):
        ax.title.set(fontsize=10, color=INK)
        ax.tick_params(labelsize=8, colors=MUTED)
        ax.xaxis.label.set(fontsize=9, color=MUTED)
        ax.yaxis.label.set(fontsize=9, color=MUTED)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.plot([], [], color=TRUE_PATH, lw=2, label="True future")
        ax.plot([], [], color=PREDICTED, lw=2, label="Predicted mean")
        ax.plot([], [], "o", color=OBSERVED, ms=3, label="Observed")
        ax.plot([], [], "o", color=BALL, ms=8, label="Ball now")
    side.legend(fontsize=8, frameon=False, loc="upper right", ncol=2)
    fig.suptitle(title, fontsize=11, color=INK)
    readout = fig.text(0.5, 0.02, "", ha="center", fontsize=10, color=INK, family="monospace")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))

    drawn: list = []

    def draw(frame: int):
        for artist in drawn:
            artist.remove()
        drawn.clear()
        truth, mean, sigma = pred["truth"][frame], pred["mean"][frame], pred["sigma"][frame]
        observed = pred["observed"][frame][pred["observed_mask"][frame]]
        now = positions[pred["t_now"][frame]]
        # 2-sigma ellipses at the end of each predicted chunk.
        for i in range(cfg.chunk_len - 1, len(mean), cfg.chunk_len):
            sigma_along = np.sqrt((direction[0] * sigma[i, 0]) ** 2 + (direction[1] * sigma[i, 1]) ** 2)
            for ax, xy, w, h in ((side, (along(mean[i]), mean[i, 2]), sigma_along, sigma[i, 2]),
                                 (top, (mean[i, 0], mean[i, 1]), sigma[i, 0], sigma[i, 1])):
                drawn.append(ax.add_patch(Ellipse(xy, 4 * w, 4 * h, color=PREDICTED, alpha=0.18, lw=0)))
        for ax, x, y, nx, ny, ox, oy, tx, ty in (
                (side, along(mean), mean[:, 2], along(now), now[2], along(observed), observed[:, 2], along(truth), truth[:, 2]),
                (top, mean[:, 0], mean[:, 1], now[0], now[1], observed[:, 0], observed[:, 1], truth[:, 0], truth[:, 1])):
            drawn.extend(ax.plot(tx, ty, color=TRUE_PATH, lw=2))
            drawn.extend(ax.plot(x, y, color=PREDICTED, lw=2))
            drawn.extend(ax.plot(ox, oy, "o", color=OBSERVED, ms=2.5))
            drawn.extend(ax.plot(nx, ny, "o", color=BALL, ms=8, zorder=5))
        error = pred["error"][frame]
        readout.set_text(f"t = {pred['t_now'][frame] * cfg.dt:.2f} s   history {pred['n_history'][frame] * cfg.chunk_len * cfg.dt:.2f} s"
                         f"   error over next 0.5 s: {100 * np.sqrt(np.mean(error**2)):5.1f} cm"
                         f"   at 0.5 s: {100 * error[-1]:5.1f} cm")
        return drawn

    frames = list(range(len(pred["t_now"]))) + [len(pred["t_now"]) - 1] * fps  # hold the last frame
    out.parent.mkdir(parents=True, exist_ok=True)
    FuncAnimation(fig, draw, frames=frames, blit=False).save(out, writer=PillowWriter(fps=fps), dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run name under outputs/trajectory")
    ap.add_argument("--data", default="data/stage1")
    ap.add_argument("--split", default="val_throws", choices=["train", "val_throws", "test_rooms", "overfit"])
    ap.add_argument("--n", type=int, default=3, help="throws to animate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    run_dir = Path("outputs/trajectory") / args.run
    model, step, noise = load_model(run_dir, args.device)
    cfg = model.cfg
    root = Path(args.data)
    data = np.load(root / "trajectories.npz")
    pool = np.flatnonzero(data["split"] == data["split_names"].tolist().index(args.split))
    picks = np.random.default_rng(args.seed).choice(pool, size=args.n, replace=False)
    print(f"{args.run}" + (f" at step {step}" if step else "") + f", split {args.split}, noise {noise} m")

    for i in picks:
        positions = torch.from_numpy(data["positions"][i]).to(args.device)
        room_id = int(data["room_id"][i])
        room = np.load(root / "rooms" / f"room_{room_id:04d}.npz")
        pred = predict_over_time(model, positions, room, noise, args.seed)
        error = pred["error"]
        rmse = np.sqrt((error**2).mean(1))
        speed = np.linalg.norm(data["v0"][i])
        print(f"throw {i} (room {room_id}, {speed:.1f} m/s): error over next 0.5 s  "
              f"first {100 * rmse[0]:.1f} cm  middle {100 * rmse[len(rmse) // 2]:.1f} cm  last {100 * rmse[-1]:.1f} cm  "
              f"mean {100 * np.sqrt((error**2).mean()):.1f} cm")
        title = f"{args.run}: throw {i}, room {room_id}, {args.split} ({speed:.1f} m/s launch)"
        out = run_dir / "inference" / f"{args.split}_throw_{i}.gif"
        animate(positions.cpu().numpy(), room, pred, cfg, title, out)
        print(f"  saved {out}")

    # Latency of one prediction with cached wall tokens, the way the catching planner would call it.
    positions = torch.from_numpy(data["positions"][picks[0]]).to(args.device)
    room = np.load(root / "rooms" / f"room_{int(data['room_id'][picks[0]]):04d}.npz")
    patches, centers = build_wall_patches(torch.from_numpy(room["points"]), cfg)
    with torch.no_grad():
        start = time.perf_counter()
        walls = model.encode_walls(patches[None].to(args.device), centers[None].to(args.device))
        wall_ms = 1e3 * (time.perf_counter() - start)
        batch = build_windows(positions[None], torch.tensor([49]), torch.tensor([cfg.max_history]), cfg, noise_std=noise)
        timings = []
        for _ in range(20):
            start = time.perf_counter()
            rollout(model, batch, walls)
            timings.append(1e3 * (time.perf_counter() - start))
    print(f"latency on {args.device}: wall encoding {wall_ms:.1f} ms once per room, "
          f"full 0.5 s rollout {np.median(timings):.1f} ms median (observation interval is {1e3 * cfg.dt:.0f} ms)")


if __name__ == "__main__":
    main()
