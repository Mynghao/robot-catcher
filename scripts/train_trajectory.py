"""Train the trajectory model on generated data and score it against the plan's section 7 criteria.

    python scripts/train_trajectory.py --split train --steps 20000 --name e2
    python scripts/train_trajectory.py --split overfit --steps 5000 --name e1
    python scripts/train_trajectory.py --split train --steps 20000 --noise 0.005 --name e4

By default this trains on the model's own rollouts with balanced history lengths. For the plan's
original section 3 and 5 behaviour, pass --rollout-fraction 0 --history-sampling t_now_first.

Logs to Weights & Biases: online if you've run `wandb login`, otherwise saved locally under
outputs/trajectory/<name>/wandb for a later `wandb sync`. Use --wandb disabled to turn it off.
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
import wandb

from catcher.trajectory.config import TrajectoryConfig
from catcher.trajectory.dataset import load_split
from catcher.trajectory.evaluate import evaluate_split
from catcher.trajectory.model import TrajectoryModel
from catcher.trajectory.train import fit


def start_wandb(args: argparse.Namespace, cfg: TrajectoryConfig, out: Path):
    """A wandb run, or None if disabled. Never blocks on a login prompt."""
    if args.wandb == "disabled":
        return None
    mode = args.wandb
    if mode == "auto":
        mode = "online" if wandb.login(prompt=False) else "offline"
    print(f"wandb mode: {mode}", flush=True)
    return wandb.init(project=args.wandb_project, name=args.name, group=args.wandb_group, mode=mode, dir=out,
                      config={**asdict(cfg), **vars(args)})


def eval_log(prefix: str, scores: dict) -> dict:
    """Scalar metrics for wandb, in metres. Per-horizon lists stay in metrics.json."""
    logged = {f"{prefix}/{k}": v for k, v in scores.items() if not isinstance(v, list)}
    logged[f"{prefix}/sigma_mean_m"] = sum(scores["sigma_by_chunk"]) / len(scores["sigma_by_chunk"])
    return logged


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/stage1")
    ap.add_argument("--split", default="train", choices=["train", "overfit"])
    ap.add_argument("--per-room", type=int, default=None, help="use only this many throws per room")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--noise", type=float, default=0.0, help="observation noise std (m)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rollout-fraction", type=float, default=TrajectoryConfig.rollout_fraction,
                    help="share of each batch trained on its own rollout; 0 is pure teacher forcing")
    ap.add_argument("--history-sampling", default=TrajectoryConfig.history_sampling, choices=["balanced", "t_now_first"])
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--name", required=True)
    ap.add_argument("--wandb", default="auto", choices=["auto", "online", "offline", "disabled"])
    ap.add_argument("--wandb-project", default="robot-catcher")
    ap.add_argument("--wandb-group", default=None)
    args = ap.parse_args()

    out = Path("outputs/trajectory") / args.name
    out.mkdir(parents=True, exist_ok=True)
    cfg = TrajectoryConfig(obs_noise_std=args.noise, rollout_fraction=args.rollout_fraction,
                           history_sampling=args.history_sampling)
    train = load_split(args.data, args.split, cfg, args.device)
    if args.per_room is not None:
        train = train.first_per_room(args.per_room)
    val = load_split(args.data, "val_throws", cfg, args.device)
    test = load_split(args.data, "test_rooms", cfg, args.device)
    print(f"{train.positions.shape[0]} throws in {train.patches.shape[0]} rooms", flush=True)

    run = start_wandb(args, cfg, out)
    torch.manual_seed(args.seed)
    model = TrajectoryModel(cfg).to(args.device)
    history = []
    start = time.time()

    def on_eval(step: int, loss: float):
        # The overfit split has its own room, so it's scored on itself while it trains.
        split_name = "train" if args.split == "overfit" else "val_throws"
        scored = evaluate_split(model, train if args.split == "overfit" else val, args.noise, num_windows=1024)
        if run is not None:
            run.log(eval_log(f"eval_{split_name}", scored), step=step)
        # Latest weights, so inference can run while training continues.
        torch.save({"model": model.state_dict(), "config": vars(args), "step": step}, out / "latest.pt")
        history.append({"step": step, "loss": loss, **{k: scored[k] for k in
                        ("teacher_forced_rmse", "one_chunk_rmse", "rollout_rmse", "coverage_90")}})
        print(f"step {step:6d}  loss {loss:9.2f}  teacher-forced {100 * scored['teacher_forced_rmse']:6.2f} cm  "
              f"1-chunk {100 * scored['one_chunk_rmse']:6.2f} cm  rollout {100 * scored['rollout_rmse']:6.2f} cm  "
              f"coverage {scored['coverage_90']:.2f}  ({time.time() - start:.0f} s)", flush=True)

    def on_log(step: int, metrics: dict):
        if run is not None:
            run.log(metrics, step=step)

    fit(model, train, args.steps, args.noise, args.seed, on_eval, args.eval_every, on_log)

    scores = {"train": evaluate_split(model, train, args.noise)}
    if args.split == "train":
        scores["val_throws"] = evaluate_split(model, val, args.noise)
        scores["val_throws_full_history"] = evaluate_split(model, val, args.noise, full_history=True)
        scores["test_rooms"] = evaluate_split(model, test, args.noise)
        scores["test_rooms_full_history"] = evaluate_split(model, test, args.noise, full_history=True)
        scores["test_rooms_shuffled_walls"] = evaluate_split(model, test, args.noise, shuffle_walls=True)
    torch.save({"model": model.state_dict(), "config": vars(args)}, out / "model.pt")
    (out / "metrics.json").write_text(json.dumps({"args": vars(args), "history": history, "scores": scores,
                                                  "seconds": time.time() - start}, indent=1))
    print("FINAL")
    for name, s in scores.items():
        print(f"  {name:28s} teacher-forced {100 * s['teacher_forced_rmse']:6.2f}  1-chunk {100 * s['one_chunk_rmse']:6.2f}  "
              f"rollout {100 * s['rollout_rmse']:6.2f}  final step {100 * s['rollout_rmse_final_step']:6.2f}  "
              f"baseline {100 * s['baseline_rmse']:6.2f} cm  coverage {s['coverage_90']:.2f}", flush=True)
    print(f"saved {out}  ({time.time() - start:.0f} s)", flush=True)
    if run is not None:
        for name, s in scores.items():
            run.summary.update(eval_log(f"final_{name}", s))
        run.finish()


if __name__ == "__main__":
    main()
