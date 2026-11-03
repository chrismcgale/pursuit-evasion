"""Does training under the link close the robustness gap? (ROADMAP #11)

Same paired protocol as pe-robust (seeds 20000+, LinkedController seed 4242,
so every arm sees the same spawns and the same corruption), on the latency
axis and the preset ladder, comparing a policy trained on a perfect link with
the same policy fine-tuned under vicon_lab (dagger.py --link), each flying
alone, gated and gated + shield, against scripted and scripted + shield.

    .venv/bin/python scripts/analysis/link_trained_sweep.py --game assault \
        --base models/sphere_law/assault_dagger.zip \
        --linked models/sphere_law_link/assault_dagger_link.zip
"""
import argparse
import json
from pathlib import Path

import pursuit_evasion  # noqa: F401  (OpenBLAS pin before numpy)
import numpy as np
from stable_baselines3 import PPO

from pursuit_evasion.bt.gating import GatedController
from pursuit_evasion.env.core import TEAM_PURSUERS
from pursuit_evasion.env.games import make_game
from pursuit_evasion.env.link import PRESETS, LinkedController
from pursuit_evasion.eval.robustness import axis_profiles, paired_delta, wilson
from pursuit_evasion.eval.scenarios import run_batch
from pursuit_evasion.safety import ShieldedController
from pursuit_evasion.scripted import default_controllers
from pursuit_evasion.scripted.base import RLController


def arms(game, base, linked):
    def law():
        return default_controllers(game)[0]

    def pol(path, name):
        return RLController(PPO.load(path, device="cpu"), name=name)

    out = {"scripted": law(), "scripted_safe": ShieldedController(law())}
    for tag, path in (("base", base), ("link", linked)):
        out[f"policy_{tag}"] = pol(path, tag)
        out[f"gated_{tag}"] = GatedController(TEAM_PURSUERS, law(), pol(path, tag), game=game)
        out[f"gated_safe_{tag}"] = ShieldedController(
            GatedController(TEAM_PURSUERS, law(), pol(path, tag), game=game))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="assault")
    p.add_argument("--base", required=True)
    p.add_argument("--linked", required=True)
    p.add_argument("--episodes", type=int, default=150)
    p.add_argument("--seed", type=int, default=20_000)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    g = make_game(a.game)
    dt = g.arena.timestep * g.episode.control_repeat
    levels = [("latency", lv, prof) for lv, prof in axis_profiles("latency") if lv <= 0.08]
    levels += [("presets", i, prof) for i, prof in enumerate(PRESETS.values())]
    res = {"game": a.game, "episodes": a.episodes, "seed0": a.seed, "base": a.base,
           "linked": a.linked, "rows": []}
    for axis, lv, prof in levels:
        row = {"axis": axis, "level": lv, "profile": prof.name, "arms": {}}
        for name, ctrl in arms(a.game, a.base, a.linked).items():
            _, opp = default_controllers(a.game)
            L = LinkedController(ctrl, prof, seed=4242, dt=dt)
            w = [bool(r.win) for r in run_batch(L, opp, a.episodes, a.seed, game=a.game).records]
            row["arms"][name] = {"win": float(np.mean(w)), "ci": wilson(sum(w), len(w)), "wins": w}
        A = row["arms"]
        for x, y in [("policy_link", "policy_base"), ("gated_link", "gated_base"),
                     ("gated_safe_link", "gated_safe_base"), ("gated_base", "scripted"),
                     ("gated_link", "scripted"), ("gated_safe_link", "scripted_safe"),
                     ("gated_safe_base", "scripted_safe"), ("scripted_safe", "scripted")]:
            row.setdefault("deltas", {})[f"{x} - {y}"] = paired_delta(A[x]["wins"], A[y]["wins"])
        res["rows"].append(row)
        d = row["deltas"]
        print(f"{prof.name:<16} " + " ".join(f"{k}={v['win']:.3f}" for k, v in A.items())
              + f" | policy link-base {d['policy_link - policy_base']['delta']:+.3f}"
              f" gated link-base {d['gated_link - gated_base']['delta']:+.3f}", flush=True)
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
