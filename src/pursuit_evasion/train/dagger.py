"""DAgger: fix behaviour cloning's covariate-shift failure.

Plain BC of the scripted expert reaches MSE ~0.003 yet captures ~0%: the cloned
policy drifts into states the expert never visited (and blends between the two
evaders instead of committing), and small errors compound. DAgger closes that gap
by training on the *learner's own* state distribution with the *expert's* labels:

    for each iteration:
      roll out the current policy (β-mixed with the expert), collecting the states
      it actually visits; label every state with the scripted expert's action;
      aggregate into the dataset; retrain the policy on everything so far.

Because the expert always commits to a target assignment, DAgger also teaches the
policy to commit instead of averaging — the specific failure we diagnosed.

The same recipe is game-agnostic: ``--game assault`` clones the *interceptor*
expert instead, which is a materially harder imitation target because its
scripted teacher switches law (lead-intercept vs. hold-the-gate) discontinuously.

    uv run python -m pursuit_evasion.train.dagger --iterations 8 --episodes 100 --out models
    uv run python -m pursuit_evasion.train.dagger --game assault --iterations 6
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from ..env.core import TEAM_PURSUERS, PursuitEvasionCore
from ..env.games import GAME_KEYS, make_game
from ..env.gym_env import SelfPlayTeamEnv
from ..env.observations import build_team_obs
from ..scripted import default_controllers
from ..scripted.base import RLController
from .imitation import bc_pretrain


def collect_dagger(learner, expert, evader, n_episodes, arena, episode, beta, seed0,
                   game=None):
    """Roll out a β-mix of expert/learner; label every visited state with the expert."""
    core = PursuitEvasionCore(arena, episode, seed=seed0, game=game)
    rng = np.random.default_rng(seed0)
    X, Y = [], []
    for e in range(n_episodes):
        pv, ev = core.reset(seed=seed0 + e)
        learner.reset()
        expert.reset()
        evader.reset()
        while True:
            expert_a = np.asarray(expert.act(pv), dtype=np.float32).reshape(-1)
            X.append(build_team_obs(pv))
            Y.append(expert_a)                                   # label = expert
            drive = expert_a if rng.random() < beta else np.asarray(learner.act(pv)).reshape(-1)
            r = core.step(drive, evader.act(ev))
            pv, ev = r.pursuer_view, r.evader_view
            if r.terminated or r.truncated:
                break
    return X, Y


def quick_win_rate(model, arena, episode, opponent, game=None, n=40, seed0=9000):
    """Held-out win rate for the pursuer-slot team (tag: captures; assault: no breach)."""
    from ..eval.scenarios import run_batch
    ctrl = RLController(model, name="dagger")
    return run_batch(ctrl, opponent, n, seed0, arena, episode,
                     game=game).aggregate()["win_rate"]


def main(argv=None):
    p = argparse.ArgumentParser(description="DAgger against the scripted pursuer-slot expert")
    p.add_argument("--game", default="tag", choices=list(GAME_KEYS))
    p.add_argument("--iterations", type=int, default=8)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--bc-epochs", type=int, default=12)
    p.add_argument("--out", default="models")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    game = make_game(args.game)
    arena, episode = game.arena, game.episode
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = "pursuer_dagger" if args.game == "tag" else f"{args.game}_dagger"

    expert, opponent = default_controllers(args.game)
    boot = DummyVecEnv([lambda: Monitor(
        SelfPlayTeamEnv(TEAM_PURSUERS, default_controllers(args.game)[1],
                        arena=arena, episode=episode, game=game))])
    model = PPO("MlpPolicy", boot, device="cpu", seed=args.seed,
                policy_kwargs=dict(net_arch=[128, 128]), n_steps=1024, batch_size=256)
    learner = RLController(model, name="dagger")

    X, Y = [], []
    best = -1.0
    for it in range(args.iterations):
        beta = 1.0 if it == 0 else 0.0            # iter 0 = pure BC; then learner drives
        Xi, Yi = collect_dagger(learner, expert, opponent, args.episodes, arena, episode,
                                beta, seed0=args.seed + it * 1000, game=game)
        X.extend(Xi)
        Y.extend(Yi)
        mse = bc_pretrain(model, np.asarray(X, np.float32), np.asarray(Y, np.float32),
                          epochs=args.bc_epochs)
        rate = quick_win_rate(model, arena, episode, opponent, game=game)
        print(f"[dagger] {args.game} iter {it}: dataset={len(X)} mse={mse:.4f} win={rate:.2f}")
        if rate >= best:
            best = rate
            model.save(out / stem)
    print(f"[dagger] best win={best:.2f}; saved -> {out}/{stem}.zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
