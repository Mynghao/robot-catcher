"""E0 plot: random throws from the generated dataset, drawn inside their rooms.

    python scripts/plot_trajectories.py [--data data/stage1] [--n 20] [--out outputs/trajectory/e0_throws.png]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

INK, MUTED, WALL = "#222222", "#6d6b63", "#b5b3ab"
PATH, START = "#2a78d6", "#eb6834"


def room_outline(room) -> np.ndarray:
    """The floor rectangle in the robot base frame, closed, (5, 2)."""
    w, d = float(room["width"]) / 2, float(room["depth"]) / 2
    c, s = np.cos(room["yaw"]), np.sin(room["yaw"])
    xy = np.array([[-w, -d], [w, -d], [w, d], [-w, d], [-w, -d]]) - room["base"]
    return np.stack([c * xy[:, 0] - s * xy[:, 1], s * xy[:, 0] + c * xy[:, 1]], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/stage1")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/trajectory/e0_throws.png")
    args = ap.parse_args()

    root = Path(args.data)
    data = np.load(root / "trajectories.npz")
    train = np.flatnonzero(data["split"] == data["split_names"].tolist().index("train"))
    picks = np.random.default_rng(args.seed).choice(train, size=args.n, replace=False)
    cols = 5
    rows = int(np.ceil(args.n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 2.3 * rows), squeeze=False)
    for ax, i in zip(axes.flat, picks):
        pos = data["positions"][i]
        room = np.load(root / "rooms" / f"room_{data['room_id'][i]:04d}.npz")
        # Side view along the throw's own horizontal direction, so each arc shows its true shape.
        travel = pos[-1, :2] - pos[0, :2]
        direction = travel / np.linalg.norm(travel)
        distance = (pos[:, :2] - pos[0, :2]) @ direction
        ax.plot(distance, pos[:, 2], color=PATH, lw=2)
        ax.plot(distance[0], pos[0, 2], "o", color=START, ms=5)
        ax.axhline(0, color=WALL, lw=1.5)
        ax.axhline(float(room["height"]), color=WALL, lw=1.5)
        ax.set_ylim(-0.3, float(room["height"]) + 0.3)
        ax.set_xlim(-0.5, max(distance.max(), 1.0) + 0.5)
        ax.tick_params(labelsize=7, colors=MUTED)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        speed = np.linalg.norm(data["v0"][i])
        ax.set_title(f"room {data['room_id'][i]}, {speed:.1f} m/s", fontsize=8, color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel("height (m)", fontsize=8, color=MUTED)
    for ax in axes[-1]:
        ax.set_xlabel("distance travelled (m)", fontsize=8, color=MUTED)
    fig.suptitle("Generated throws in side view: orange is the launch, grey lines are the floor and ceiling", fontsize=10, color=INK)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
