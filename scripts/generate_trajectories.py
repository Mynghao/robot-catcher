"""Generate the Stage 1 no-bounce dataset and run the plan's E0 checks.

    python scripts/generate_trajectories.py [--out data/stage1] [--seed 0]
"""

import argparse
import json
import time
from pathlib import Path

from catcher.trajectory.config import TrajectoryConfig
from catcher.trajectory.generate_data import generate_stage1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/stage1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    start = time.time()
    manifest = generate_stage1(Path(args.out), TrajectoryConfig(), seed=args.seed)
    print(f"done in {time.time() - start:.0f} s")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
