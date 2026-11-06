"""Simulate a few throws with the NumPy point-ball sim and plot them.

    python scripts/simulate_ball.py --stage 2 --n 4 --seed 0 [--out outputs/stage2.png]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from catcher.throws import REACH_MAX, SHOULDER, STAGE_BOUNCES, sample_stage_throw, sample_throw

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed categorical order
INK, MUTED, PANEL = "#222222", "#888888", "#bbbbbb"


def draw_panel_outline(ax3d, ax2d, panel):
    a, b = panel.half_extents
    corners = np.array([panel.center + s * a * panel.u + t * b * panel.v
                        for s, t in [(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)]])
    is_board = panel.name.startswith("board")
    style = dict(color=MUTED if is_board else PANEL, lw=1.5 if is_board else 0.8)
    ax3d.plot(*corners.T, **style)
    ax2d.plot(corners[:, 0], corners[:, 2], **style)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=2)
    ap.add_argument("--n", type=int, default=3, help="number of throws (max 4)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    # All throws share the first throw's room so they can be drawn together.
    first = sample_stage_throw(args.stage, rng)
    throws = [first]
    lo, hi = STAGE_BOUNCES[args.stage]
    throws += [sample_throw(first.room, rng, lo, hi) for _ in range(min(args.n, 4) - 1)]

    fig = plt.figure(figsize=(13, 5.5))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax2d = fig.add_subplot(1, 2, 2)
    for panel in first.room.panels:
        draw_panel_outline(ax3d, ax2d, panel)
    circle = plt.Circle((SHOULDER[0], SHOULDER[2]), REACH_MAX, fill=False, ls="--", color=MUTED, lw=1)
    ax2d.add_patch(circle)
    ax2d.annotate("reach", (SHOULDER[0], SHOULDER[2] + REACH_MAX), color=MUTED, fontsize=8,
                  ha="center", va="bottom")

    for i, th in enumerate(throws):
        c = SERIES[i]
        end = th.catch_index + 1
        label = f"throw {i}: {th.bounces_before_catch} bounce(s) before catch"
        for ax, (xi, yi, zi) in [(ax3d, (0, 1, 2)), (ax2d, (0, 2, None))]:
            pts = [th.pos[:end, xi], th.pos[:end, yi]] + ([th.pos[:end, zi]] if zi is not None else [])
            rest = [th.pos[end - 1:, xi], th.pos[end - 1:, yi]] + ([th.pos[end - 1:, zi]] if zi is not None else [])
            ax.plot(*pts, color=c, lw=2, label=label if ax is ax2d else None)
            ax.plot(*rest, color=c, lw=1, alpha=0.3)
        for b in th.traj.bounces:
            if b.t <= th.times[th.catch_index]:
                ax3d.scatter(*b.position, color=c, s=30, marker="o", edgecolors="white")
                ax2d.scatter(b.position[0], b.position[2], color=c, s=40, marker="o", edgecolors="white", zorder=3)
        p = th.pos[th.catch_index]
        ax3d.scatter(*p, color=c, s=70, marker="*")
        ax2d.scatter(p[0], p[2], color=c, s=110, marker="*", zorder=3)

    ax3d.scatter(0, 0, 0, color=INK, s=40, marker="^")
    ax3d.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)")
    ax3d.set_box_aspect(first.room.hi - first.room.lo)
    ax2d.set(xlabel="x (m)", ylabel="z (m)", aspect="equal")
    ax2d.set_title("side view  (● bounce, ★ first catchable point, faded = after)", fontsize=9, color=INK)
    ax2d.legend(fontsize=8, frameon=False, loc="upper right")
    for s in ["top", "right"]:
        ax2d.spines[s].set_visible(False)
    room = first.room
    mats = ", ".join(f"{p.name}: e={p.material.restitution:.2f}" for p in room.panels if p.name.startswith("board"))
    fig.suptitle(f"Stage {args.stage} point-ball simulation" + (f"   [{mats}]" if mats else ""), color=INK)
    fig.tight_layout()

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=130)
        print(f"saved {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
