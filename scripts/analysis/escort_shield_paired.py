"""Paired check: what does the shield cost escort, and which half costs it?

REVIEW 2026-09-30 recorded escort bt_gated 0.53 -> bt_gated_safe 0.47 from
unpaired pe-games cells and suspected the speed cap. Every arm here runs the
same seeds, so the per-episode differences are paired.

    .venv/bin/python scripts/analysis/escort_shield_paired.py --seed 10000 --episodes 200
"""
import argparse
import json
from pathlib import Path

import pursuit_evasion  # noqa: F401  (OpenBLAS pin before numpy)
from pursuit_evasion.eval.games import build_controllers
from pursuit_evasion.eval.robustness import paired_delta, wilson
from pursuit_evasion.eval.scenarios import run_batch
from pursuit_evasion.safety import SafetyConfig, ShieldedController
from pursuit_evasion.scripted import default_controllers


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="escort")
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--out", default="results/escort_shield_paired.json")
    a = p.parse_args()
    built = build_controllers(Path("models"), a.game)
    gate = built["bt_gated"]
    _, opp = default_controllers(a.game)
    arms = {
        "bt_gated": gate,
        "shield (shipped)": ShieldedController(gate),
        "fence only": ShieldedController(gate, SafetyConfig(speed_limit_frac=1e9)),
        "speed cap only": ShieldedController(
            gate, SafetyConfig(geofence_margin=-1e3, z_margin=-1e3, brake_horizon_s=0.0)),
        "scripted": built["scripted"],
        "scripted + shield": ShieldedController(built["scripted"]),
    }
    res = {}
    for name, c in arms.items():
        rep = run_batch(c, opp, a.episodes, a.seed, game=a.game, label=name)
        wins = [bool(r.win) for r in rep.records]
        agg = rep.aggregate()
        res[name] = {"wins": wins, "win_rate": agg["win_rate"],
                     "ci": wilson(sum(wins), len(wins)),
                     "geofence_viol": agg["geofence_viol"], "speed_viol": agg["speed_viol"]}
        print(f"{name:<20} win={agg['win_rate']:.3f} geo/spd={agg['geofence_viol']}/{agg['speed_viol']}", flush=True)
    for ref, arm in [("bt_gated", "shield (shipped)"), ("bt_gated", "fence only"),
                     ("bt_gated", "speed cap only"), ("scripted", "scripted + shield")]:
        d = paired_delta(res[arm]["wins"], res[ref]["wins"])
        res[arm][f"vs_{ref}"] = d
        print(f"{arm} - {ref}: {d['delta']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}]")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"game": a.game, "seed0": a.seed,
                                       "episodes": a.episodes, "arms": res}, indent=1))


if __name__ == "__main__":
    main()
