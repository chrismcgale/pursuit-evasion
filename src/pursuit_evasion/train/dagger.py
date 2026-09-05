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

from ..env.core import TEAM_EVADERS, TEAM_PURSUERS, PursuitEvasionCore
from ..env.games import GAME_KEYS, make_game
from ..env.gym_env import SelfPlayTeamEnv
from ..env.observations import build_team_obs
from ..scripted import default_controllers
from ..scripted.base import RLController
from .imitation import bc_pretrain


def collect_dagger(learner, expert, opponent, n_episodes, arena, episode, beta, seed0,
                   game=None, team=TEAM_PURSUERS):
    """Roll out a β-mix of expert/learner; label every visited state with the expert.

    ``team`` selects which slot is being learned. The pursuer slot is the default
    (tag pursuers / air-defence interceptors); ``TEAM_EVADERS`` learns the evader
    slot instead, which is how a *raid* policy is cloned from the missile expert.
    """
    core = PursuitEvasionCore(arena, episode, seed=seed0, game=game)
    rng = np.random.default_rng(seed0)
    learning_pursuers = team == TEAM_PURSUERS
    X, Y = [], []
    for e in range(n_episodes):
        pv, ev = core.reset(seed=seed0 + e)
        learner.reset()
        expert.reset()
        opponent.reset()
        while True:
            own, other = (pv, ev) if learning_pursuers else (ev, pv)
            expert_a = np.asarray(expert.act(own), dtype=np.float32).reshape(-1)
            X.append(build_team_obs(own))
            Y.append(expert_a)                                   # label = expert
            drive = expert_a if rng.random() < beta else np.asarray(learner.act(own)).reshape(-1)
            opp_a = opponent.act(other)
            r = core.step(drive if learning_pursuers else opp_a,
                          opp_a if learning_pursuers else drive)
            pv, ev = r.pursuer_view, r.evader_view
            if r.terminated or r.truncated:
                break
    return X, Y


def quick_win_rate(model, arena, episode, opponent, game=None, n=40, seed0=9000,
                   team=TEAM_PURSUERS):
    """Held-out success rate for the team being learned.

    The harness always reports the *pursuer-slot* objective (see CLAUDE.md — it
    keeps the two runtimes comparable), so an evader-slot learner is scored by the
    complementary outcome: in the defence games an attacker only succeeds by
    breaching, and a timeout is a defender win.
    """
    from ..eval.scenarios import run_batch
    ctrl = RLController(model, name="dagger")
    if team == TEAM_PURSUERS:
        return run_batch(ctrl, opponent, n, seed0, arena, episode,
                         game=game).aggregate()["win_rate"]
    agg = run_batch(opponent, ctrl, n, seed0, arena, episode, game=game).aggregate()
    return 1.0 - agg["win_rate"]        # == breach_rate in the defence games


def main(argv=None):
    p = argparse.ArgumentParser(description="DAgger against the scripted pursuer-slot expert")
    p.add_argument("--game", default="tag", choices=list(GAME_KEYS))
    p.add_argument("--iterations", type=int, default=8)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--bc-epochs", type=int, default=12)
    p.add_argument("--out", default="models")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--side", default="pursuers", choices=["pursuers", "evaders"],
                   help="which slot to learn; 'evaders' clones the raid/attacker expert")
    p.add_argument("--select-episodes", type=int, default=40,
                   help="episodes in the checkpoint-selection battery. Best-of-N on a "
                        "small battery is biased high (winner's curse): the tag run "
                        "used 40 and its pick scored 0.30 there but 0.20 on a held-out "
                        "200. Always re-measure the chosen checkpoint independently.")
    args = p.parse_args(argv)

    game = make_game(args.game)
    arena, episode = game.arena, game.episode
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    team = TEAM_PURSUERS if args.side == "pursuers" else TEAM_EVADERS
    if args.side == "pursuers":
        stem = "pursuer_dagger" if args.game == "tag" else f"{args.game}_dagger"
    else:
        stem = f"{args.game}_attacker_dagger"

    pursuer_ctrl, evader_ctrl = default_controllers(args.game)
    # the expert teaches the learned slot; the other slot is the scripted opponent
    expert, opponent = ((pursuer_ctrl, evader_ctrl) if team == TEAM_PURSUERS
                        else (evader_ctrl, pursuer_ctrl))
    boot = DummyVecEnv([lambda: Monitor(
        SelfPlayTeamEnv(team, opponent,
                        arena=arena, episode=episode, game=game))])
    model = PPO("MlpPolicy", boot, device="cpu", seed=args.seed,
                policy_kwargs=dict(net_arch=[128, 128]), n_steps=1024, batch_size=256)
    learner = RLController(model, name="dagger")

    X, Y = [], []
    best = -1.0
    for it in range(args.iterations):
        beta = 1.0 if it == 0 else 0.0            # iter 0 = pure BC; then learner drives
        Xi, Yi = collect_dagger(learner, expert, opponent, args.episodes, arena, episode,
                                beta, seed0=args.seed + it * 1000, game=game, team=team)
        X.extend(Xi)
        Y.extend(Yi)
        mse = bc_pretrain(model, np.asarray(X, np.float32), np.asarray(Y, np.float32),
                          epochs=args.bc_epochs)
        rate = quick_win_rate(model, arena, episode, opponent, game=game, team=team,
                              n=args.select_episodes)
        print(f"[dagger] {args.game}/{args.side} iter {it}: dataset={len(X)} "
              f"mse={mse:.4f} win={rate:.2f}")
        if rate >= best:
            best = rate
            model.save(out / stem)
    print(f"[dagger] best win={best:.2f}; saved -> {out}/{stem}.zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
