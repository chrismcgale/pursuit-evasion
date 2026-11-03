"""Choose the defender law's fallback rule (what to do with no intercept).

Selection on seed blocks 10000+ and 110000+, confirmation on 20000+ only for
the chosen rule (the same discipline that picked intercept_radius = 0.7).
Scripted law alone, paired against the legacy 'park' rule.

    .venv/bin/python scripts/analysis/fallback_select.py --blocks 10000 110000
"""
import argparse
import json
from pathlib import Path

import pursuit_evasion  # noqa: F401  (OpenBLAS pin before numpy)
import numpy as np

from pursuit_evasion.eval.robustness import paired_delta, wilson
from pursuit_evasion.eval.scenarios import run_batch
from pursuit_evasion.scripted import default_controllers
from pursuit_evasion.scripted.defenders import GuardDefenders


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--blocks", type=int, nargs="+", default=[10_000, 110_000])
    p.add_argument("--rules", nargs="+", default=["park", "pursue", "lead_gate", "adaptive"])
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    res = {}
    for game in ("assault", "escort"):
        for block in a.blocks:
            wins = {}
            for rule in a.rules:
                _, opp = default_controllers(game)
                law = GuardDefenders(fallback=rule)
                rep = run_batch(law, opp, a.episodes, block, game=game)
                wins[rule] = [bool(r.win) for r in rep.records]
            row = {}
            for rule in a.rules:
                w = wins[rule]
                row[rule] = {"win": float(np.mean(w)), "ci": wilson(sum(w), len(w)),
                             "vs_park": paired_delta(w, wins["park"])}
                d = row[rule]["vs_park"]
                print(f"{game:<8} {block:>6} {rule:<10} {np.mean(w):.3f}  vs park "
                      f"{d['delta']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}]", flush=True)
            res[f"{game}/{block}"] = row
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
