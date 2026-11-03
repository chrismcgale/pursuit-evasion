"""What should fill the defender law's fallback regime?

The gate hands the regime where GuardDefenders.plan() has no intercept (and
parks on a gate point) to whatever sits on the RL branch. This puts simple
scripted alternatives on that branch and compares them, paired, with the
policies: if a one-line pursuit rule matches the best policy, the gain is a
fix the scripted law could make itself (the lesson of writeup §7.2).

    .venv/bin/python scripts/analysis/fallback_probe.py --game escort --seed 20000
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


class PurePursuit:
    """Each defender flies straight at its nearest live attacker."""
    name = "pure_pursuit"

    def reset(self):
        pass

    def act(self, v):
        out = np.zeros((len(v.self_pos), 3))
        live = [j for j in range(len(v.opp_pos)) if v.opp_alive[j]]
        for i in range(len(v.self_pos)):
            if not live:
                continue
            d = [np.linalg.norm(v.opp_pos[j] - v.self_pos[i]) for j in live]
            r = v.opp_pos[live[int(np.argmin(d))]] - v.self_pos[i]
            n = np.linalg.norm(r)
            out[i] = r / n if n > 1e-9 else 0.0
        return out.reshape(-1)


class LeadGate:
    """The law's own fallback, but with the gate point placed on the asset's
    predicted position when the threat arrives (asset_pos + asset_vel * t),
    instead of its current one. Identical to the law for a static asset."""
    name = "lead_gate"

    def __init__(self, game):
        from pursuit_evasion.scripted.defenders import time_to_asset
        self.law, _ = default_controllers(game)
        self.tta = time_to_asset

    def reset(self):
        self.law.reset()

    def act(self, v):
        a, fb = self.law.plan(v)
        a = a.reshape(-1, 3).copy()
        assign = self.law._assign(v, v.asset_pos)
        for i in range(len(fb)):
            if not (fb[i] and v.self_alive[i]):
                continue
            j = assign[i]
            t = self.tta(v.opp_pos[j], v.opp_vel[j], v.asset_pos, v.opp_vmax)
            t = min(t, 10.0) if np.isfinite(t) else 0.0
            ahead = v.asset_pos + v.asset_vel * t
            gate = self.law._gate_point(v.opp_pos[j], ahead)
            to_gate = gate - v.self_pos[i]
            n = np.linalg.norm(to_gate)
            if n < 0.8:
                r = v.opp_pos[j] - v.self_pos[i]
                a[i] = r / max(np.linalg.norm(r), 1e-9)
            else:
                a[i] = to_gate / n
        return a.reshape(-1)


class Hold:
    """Zero command: drift to a stop instead of flying to the gate point."""
    name = "hold"

    def reset(self):
        pass

    def act(self, v):
        return np.zeros(3 * len(v.self_pos))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="escort")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=20_000)
    p.add_argument("--models", nargs="*", default=[])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    fills = {"lead_gate": LeadGate(a.game), "pure_pursuit": PurePursuit(), "hold": Hold()}
    for m in a.models:
        key = f"{Path(m).parent.name}/{Path(m).stem}"
        fills[key] = RLController(PPO.load(m, device="cpu"), name=key)
    law, opp = default_controllers(a.game)
    base = [bool(r.win) for r in run_batch(law, opp, a.episodes, a.seed, game=a.game).records]
    res = {"game": a.game, "seed0": a.seed, "episodes": a.episodes,
           "scripted": {"win": float(np.mean(base))}, "fills": {}}
    print(f"scripted {np.mean(base):.3f}", flush=True)
    for name, fill in fills.items():
        law, opp = default_controllers(a.game)
        gate = GatedController(TEAM_PURSUERS, law, fill, game=a.game)
        w = [bool(r.win) for r in run_batch(gate, opp, a.episodes, a.seed, game=a.game).records]
        d = paired_delta(w, base)
        res["fills"][name] = {"win": float(np.mean(w)), "ci": wilson(sum(w), len(w)), "vs_scripted": d}
        print(f"gate -> {name:<26} {np.mean(w):.3f}  vs scripted {d['delta']:+.3f} "
              f"[{d['lo']:+.3f},{d['hi']:+.3f}]", flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
