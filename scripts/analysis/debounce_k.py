"""Re-measure the k-tick handover debounce on the current defender law.

The 2026-09-06 k-sweep (built, measured, reverted) ran on the point-solve law
and the proxy gate. This repeats it on the shipped sphere-solve law and the
law-read gate without touching shipped code: ``DebouncedLaw`` wraps
``GuardDefenders.plan()`` and reports an agent's fallback only after it has
persisted ``k`` consecutive ticks. k=1 is immediate handover. Since
2026-11-03 the shipped defender gate hands nothing over, so the handover is
built here with ``FallbackFill``.

Paired: same spawn seeds and the same link corruption (LinkedController seed)
for every k at a given latency.

    .venv/bin/python scripts/analysis/debounce_k.py --episodes 150
"""
import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, __import__('os').path.dirname(__file__))
import pursuit_evasion  # noqa: F401  (OpenBLAS pin before numpy)
import numpy as np
from stable_baselines3 import PPO

from pursuit_evasion.env.games import make_game
from pursuit_evasion.env.link import LinkedController, LinkProfile
from fallback_fill import FallbackFill  # noqa: E402  (the pre-2026-11-03 defender handover)
from pursuit_evasion.eval.robustness import paired_delta, wilson
from pursuit_evasion.eval.scenarios import run_batch
from pursuit_evasion.scripted import default_controllers
from pursuit_evasion.scripted.base import RLController


class DebouncedLaw:
    def __init__(self, law, k):
        self.law, self.k, self.name = law, k, f"{law.name}_k{k}"
        self.streak = None

    def reset(self):
        self.law.reset()
        self.streak = None

    def act(self, view):
        return self.law.act(view)

    def plan(self, view):
        a, fb = self.law.plan(view)
        if self.streak is None:
            self.streak = np.zeros(len(fb), dtype=int)
        self.streak = np.where(np.asarray(fb, bool), self.streak + 1, 0)
        return a, self.streak >= self.k


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="assault")
    p.add_argument("--episodes", type=int, default=150)
    p.add_argument("--seed", type=int, default=20_000)
    p.add_argument("--latencies", type=float, nargs="+", default=[0.0, 0.02, 0.04, 0.06])
    p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--model", default="models/assault_dagger.zip")
    p.add_argument("--out", default="results/debounce_k_assault.json")
    a = p.parse_args()
    game = make_game(a.game)
    dt = game.arena.timestep * game.episode.control_repeat
    policy = RLController(PPO.load(a.model, device="cpu"), name=Path(a.model).stem)
    res = {"game": a.game, "episodes": a.episodes, "seed0": a.seed, "model": a.model,
           "rows": []}
    for lat in a.latencies:
        prof = LinkProfile(name=f"latency={lat:g}", obs_latency_s=lat)
        row = {"latency_s": lat, "arms": {}}
        for k in [0] + a.ks:     # 0 = scripted, no gate
            law, opp = default_controllers(a.game)
            ctrl = law if k == 0 else FallbackFill(DebouncedLaw(law, k), policy)
            linked = LinkedController(ctrl, prof, seed=4242, dt=dt)
            rep = run_batch(linked, opp, a.episodes, a.seed, game=a.game)
            wins = [bool(r.win) for r in rep.records]
            name = "scripted" if k == 0 else f"k={k}"
            row["arms"][name] = {"win": float(np.mean(wins)), "ci": wilson(sum(wins), len(wins)),
                                 "wins": wins}
        for k in a.ks:
            arm = row["arms"][f"k={k}"]
            arm["vs_scripted"] = paired_delta(arm["wins"], row["arms"]["scripted"]["wins"])
            if k != a.ks[0]:
                arm[f"vs_k{a.ks[0]}"] = paired_delta(arm["wins"], row["arms"][f"k={a.ks[0]}"]["wins"])
        res["rows"].append(row)
        print(f"latency {lat*1000:.0f} ms: " + "  ".join(
            f"{n}={v['win']:.3f}" for n, v in row["arms"].items()), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
