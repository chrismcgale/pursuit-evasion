"""Single-predicate gate ablation: measure what each branch of the tree buys.

The project's sharpest result came from this procedure, so it lives in the repo
rather than in a scratch buffer. A gate profile is a *list* of predicates tried
in order; the ablation walks that list cumulatively — scripted-only, then the
first branch, then the first two, ... , then policy-only — and re-measures on
identical seeds. Because only one predicate changes between adjacent rows, a
score difference is attributable to that branch instead of to the tree as a whole.

That is what showed the defender profile must stop at two branches: adding tag's
``close_quarters -> RL`` instinct to air defence costs 42 points (0.70 -> 0.12).

Run it on more than one seed block. The per-cell noise at 150 episodes is a few
points, so a difference of that size is not a finding:

    uv run pe-ablate --game assault --side pursuers --episodes 150 --blocks 2
    uv run pe-ablate --game assault --side evaders --model models/assault_attacker_dagger.zip
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO

from ..bt.gating import _PREDICATES, GatedController, default_profile
from ..env.core import TEAM_EVADERS, TEAM_PURSUERS
from ..env.games import GAME_KEYS, make_game
from ..scripted import default_controllers
from ..scripted.base import RLController
from .scenarios import run_batch


def _truncated_profile(profile: str, keep: int):
    """A profile function exposing only the first ``keep`` predicates."""
    full = _PREDICATES[profile]

    def factory():
        return full()[:keep]

    return factory


def ablate(game_key: str, side: str, model_path: Path, episodes: int, blocks: int,
           seed0: int) -> dict:
    game = make_game(game_key)
    team = TEAM_PURSUERS if side == "pursuers" else TEAM_EVADERS
    profile = default_profile(team, game_key)
    names = [n for n, _, _ in _PREDICATES[profile]()]

    pursuer_ctrl, evader_ctrl = default_controllers(game_key)
    scripted = pursuer_ctrl if team == TEAM_PURSUERS else evader_ctrl
    opponent = evader_ctrl if team == TEAM_PURSUERS else pursuer_ctrl
    policy = RLController(PPO.load(str(model_path), device="cpu"), name="rl")

    def score(ctrl, seed):
        """Success rate for the side under test, on identical starts."""
        pair = (ctrl, opponent) if team == TEAM_PURSUERS else (opponent, ctrl)
        agg = run_batch(*pair, episodes, seed, game=game, label="ablate").aggregate()
        return agg["win_rate"] if team == TEAM_PURSUERS else 1.0 - agg["win_rate"]

    def gated(keep):
        """A gate carrying only the first ``keep`` predicates of the profile.

        GatedController builds its tree in __init__, so the profile table is
        swapped for the duration of construction and restored immediately — the
        controller keeps the truncated tree, the module-level table is untouched.
        """
        saved, _PREDICATES[profile] = _PREDICATES[profile], _truncated_profile(profile, keep)
        try:
            return GatedController(team, scripted, policy, game=game_key)
        finally:
            _PREDICATES[profile] = saved

    rows = []
    for keep in range(len(names) + 1):
        label = "scripted only" if keep == 0 else "+ " + names[keep - 1]
        ctrl_fn = (lambda: scripted) if keep == 0 else (lambda k=keep: gated(k))
        per_block = [score(ctrl_fn(), seed0 + b * 100_000) for b in range(blocks)]
        rows.append({"profile": label, "cumulative": names[:keep], "blocks": per_block})
        print(f"  {label:<34} " + "  ".join(f"{v:.2f}" for v in per_block))

    per_block = [score(policy, seed0 + b * 100_000) for b in range(blocks)]
    rows.append({"profile": "policy only", "cumulative": None, "blocks": per_block})
    print(f"  {'policy only':<34} " + "  ".join(f"{v:.2f}" for v in per_block))
    return {"game": game_key, "side": side, "profile": profile,
            "episodes": episodes, "rows": rows}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--game", default="assault", choices=list(GAME_KEYS))
    p.add_argument("--side", default="pursuers", choices=["pursuers", "evaders"])
    p.add_argument("--model", type=Path, default=None)
    p.add_argument("--episodes", type=int, default=150)
    p.add_argument("--blocks", type=int, default=2, help="independent seed blocks")
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--out", type=Path, default=Path("results"))
    args = p.parse_args(argv)

    model = args.model
    if model is None:
        stem = (f"{args.game}_dagger" if args.side == "pursuers"
                else f"{args.game}_attacker_dagger")
        model = Path("models") / f"{stem}.zip"
        if args.game == "tag" and args.side == "pursuers":
            model = Path("models/pursuer_dagger.zip")
    print(f"\n=== gate ablation — {args.game}/{args.side} ({args.episodes} eps x "
          f"{args.blocks} blocks, policy={model.name}) ===")
    res = ablate(args.game, args.side, model, args.episodes, args.blocks, args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / f"ablation_{args.game}_{args.side}.json"
    dest.write_text(json.dumps(res, indent=2))
    print(f"\n[ablate] wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
