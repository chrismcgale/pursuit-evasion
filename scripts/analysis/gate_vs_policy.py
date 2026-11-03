"""Gate value against policy quality (ROADMAP #4, REVIEW #19).

For every DAgger iteration of a lineage (``<stem>_lineage.json``), measure on
held-out seeds, paired against the scripted law:

  * policy-only   — the checkpoint flying everything
  * gated         — scripted law + the shipped gate, handing the law's
                    fallback regime to that checkpoint (no shield: this isolates
                    the gate)

The falsifiable prediction from ROADMAP #4: gated - scripted stays flat (the
gain is regime coverage, not policy quality) while policy-only rises. The
slope of (gated - scripted) on policy-only win rate across iterations is
reported with a bootstrap CI over episodes.

    .venv/bin/python scripts/analysis/gate_vs_policy.py --game assault \
        --lineage models/sphere_law/assault_dagger_lineage.json --episodes 200
"""
import argparse
import json
from pathlib import Path

import pursuit_evasion  # noqa: F401  (OpenBLAS pin before numpy)
import numpy as np
from stable_baselines3 import PPO

from pursuit_evasion.bt.gating import GatedController
from pursuit_evasion.env.core import TEAM_PURSUERS
from pursuit_evasion.eval.robustness import paired_delta, wilson
from pursuit_evasion.eval.scenarios import run_batch
from pursuit_evasion.scripted import default_controllers
from pursuit_evasion.scripted.base import RLController


def wins_of(ctrl, game, n, seed0):
    _, opp = default_controllers(game)
    rep = run_batch(ctrl, opp, n, seed0, game=game)
    return [bool(r.win) for r in rep.records]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="assault")
    p.add_argument("--lineage", required=True)
    p.add_argument("--extra", nargs="*", default=[],
                   help="further checkpoints to include, e.g. the shipped models/assault_dagger.zip")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=20_000)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    lin = json.loads(Path(a.lineage).read_text())
    root = Path(a.lineage).parent
    ckpts = [(f"it{e['iter']}", root / e["path"], e) for e in lin["iterations"]]
    ckpts += [(Path(x).stem, Path(x), {}) for x in a.extra]

    scripted, _ = default_controllers(a.game)
    base = wins_of(scripted, a.game, a.episodes, a.seed)
    print(f"scripted win={np.mean(base):.3f}", flush=True)
    rows = []
    for name, path, meta in ckpts:
        pol = RLController(PPO.load(str(path), device="cpu"), name=name)
        pol_w = wins_of(pol, a.game, a.episodes, a.seed)
        s, _ = default_controllers(a.game)
        gate = GatedController(TEAM_PURSUERS, s, pol, game=a.game)
        gate_w = wins_of(gate, a.game, a.episodes, a.seed)
        row = {"name": name, "path": str(path), "select_win": meta.get("select_win"),
               "mse": meta.get("mse"), "dataset": meta.get("dataset"),
               "policy": {"win": float(np.mean(pol_w)), "ci": wilson(sum(pol_w), len(pol_w)),
                          "wins": pol_w, "vs_scripted": paired_delta(pol_w, base)},
               "gated": {"win": float(np.mean(gate_w)), "ci": wilson(sum(gate_w), len(gate_w)),
                         "wins": gate_w, "vs_scripted": paired_delta(gate_w, base)}}
        rows.append(row)
        g = row["gated"]["vs_scripted"]
        print(f"{name:<22} select={meta.get('select_win')}  policy={row['policy']['win']:.3f}  "
              f"gated={row['gated']['win']:.3f}  gated-scripted={g['delta']:+.3f} "
              f"[{g['lo']:+.3f},{g['hi']:+.3f}]", flush=True)

    # slope of (gated - scripted) on policy-only win across the lineage, with a
    # paired episode bootstrap (resample episodes once, recompute every arm)
    lin_rows = [r for r in rows if r["name"].startswith("it")]
    B = np.array(base, float)
    P = np.array([r["policy"]["wins"] for r in lin_rows], float)
    G = np.array([r["gated"]["wins"] for r in lin_rows], float)
    rng = np.random.default_rng(0)
    n = len(B)

    def slope(idx):
        x = P[:, idx].mean(1)
        y = G[:, idx].mean(1) - B[idx].mean()
        if np.ptp(x) == 0:
            return 0.0
        return float(np.polyfit(x, y, 1)[0])

    full = slope(np.arange(n))
    boots = [slope(rng.integers(0, n, n)) for _ in range(2000)]
    res = {"game": a.game, "episodes": a.episodes, "seed0": a.seed,
           "scripted": {"win": float(np.mean(base)), "wins": base}, "rows": rows,
           "slope_gain_on_policy": {"slope": full, "lo": float(np.percentile(boots, 2.5)),
                                    "hi": float(np.percentile(boots, 97.5))}}
    print(f"slope of (gated - scripted) on policy-only win: {full:+.3f} "
          f"[{res['slope_gain_on_policy']['lo']:+.3f},{res['slope_gain_on_policy']['hi']:+.3f}]")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
