"""Generate a trajectory dataset for one stage and save it as .npz.

    python scripts/generate_dataset.py --stage 2 --n 1000 --seed 0 --out data/stage2_train.npz

See catcher.throws.generate_dataset for the array layout.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from catcher.throws import generate_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--t-end", type=float, default=3.0)
    ap.add_argument("--noise-std", type=float, default=0.005, help="observation noise (m)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    out = Path(args.out or f"data/stage{args.stage}_n{args.n}_seed{args.seed}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = generate_dataset(args.stage, args.n, seed=args.seed, t_end=args.t_end, noise_std=args.noise_std)
    np.savez_compressed(out, **data)
    print(f"stage {args.stage}: {args.n} throws in {time.time() - t0:.1f}s -> {out}")
    for k, v in data.items():
        print(f"  {k:22s} {str(v.shape):18s} {v.dtype}")
    counts = np.bincount(data["bounces_before_catch"])
    print("  bounces before catch:", {i: int(c) for i, c in enumerate(counts) if c})


if __name__ == "__main__":
    main()
