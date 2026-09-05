"""CLI entry point: run alternating self-play and save both team policies.

    uv run pe-train                          # full defaults (8 generations)
    uv run pe-train --generations 4 --steps-per-gen 20000 --n-envs 8
    uv run pe-train --quick                  # tiny run to prove the pipeline

Outputs (under --out, default ./models):
    pursuer.zip, evader.zip   trained SB3 policies
    history.json              per-generation eval capture rates
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .selfplay import TrainConfig, train_selfplay


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Self-play PPO trainer for 2v2 pursuit-evasion")
    p.add_argument("--generations", type=int, default=8)
    p.add_argument("--steps-per-gen", type=int, default=40_000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--pool-cap", type=int, default=6)
    p.add_argument("--eval-episodes", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="models")
    p.add_argument("--quick", action="store_true",
                   help="tiny smoke run: 2 gens x 4k steps, 2 envs")
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    if args.quick:
        cfg = TrainConfig(generations=2, steps_per_gen=4_000, n_envs=2,
                          eval_episodes=10, seed=args.seed)
    else:
        cfg = TrainConfig(generations=args.generations, steps_per_gen=args.steps_per_gen,
                          n_envs=args.n_envs, pool_cap=args.pool_cap,
                          eval_episodes=args.eval_episodes, lr=args.lr, seed=args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[train] generations={cfg.generations} steps/gen={cfg.steps_per_gen} "
          f"n_envs={cfg.n_envs} -> {out}")
    t0 = time.time()

    def on_gen(rec):
        dt = time.time() - t0
        print(f"[gen {rec['gen']}] "
              f"RL-pursuer vs scripted-evader capture={rec['rl_pursuer_vs_scripted_evader']:.2f} | "
              f"scripted-pursuer vs RL-evader capture={rec['scripted_pursuer_vs_rl_evader']:.2f} "
              f"| {dt:.0f}s elapsed")

    pursuer_model, evader_model, history = train_selfplay(cfg, on_gen=on_gen)

    pursuer_model.save(out / "pursuer")
    evader_model.save(out / "evader")
    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"[train] done in {time.time()-t0:.0f}s. saved -> {out}/pursuer.zip, {out}/evader.zip")
    return history


if __name__ == "__main__":
    main()
