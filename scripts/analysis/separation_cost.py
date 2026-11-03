"""What does own-team separation cost each game, and what does it buy?

Paired: every arm runs the same spawn seeds and (under a link) the same
corruption stream (LinkedController seeds per episode). Teammate proximity is
measured on TRUTH (the probe sits outside the link wrapper), so the contact
count reflects the airframes, not the estimate.

    .venv/bin/python scripts/analysis/separation_cost.py --games tag --episodes 150
"""
import argparse
import json
from pathlib import Path

import pursuit_evasion  # noqa: F401  (first: pins OpenBLAS before numpy loads)
import numpy as np
from pursuit_evasion.env.games import make_game
from pursuit_evasion.env.link import PERFECT, VICON_LAB, LinkedController
from pursuit_evasion.eval.games import build_controllers
from pursuit_evasion.eval.robustness import paired_delta, wilson
from pursuit_evasion.eval.scenarios import run_batch
from pursuit_evasion.safety import SafetyConfig, ShieldedController
from pursuit_evasion.scripted import default_controllers

# separation = twice the contact distance (2 * agent_radius) of each arena
SEP = {"tag": 1.4, "assault": 2.0, "escort": 2.0}


class Probe:
    """Records per-episode min distance between live teammates (truth)."""

    def __init__(self, inner):
        self.inner, self.name = inner, getattr(inner, "name", "x")
        self.ep_min, self.ticks, self.close = [], 0, 0
        self.contact_r = None

    def reset(self):
        self.inner.reset()
        self.ep_min.append(np.inf)

    def act(self, view):
        alive = [i for i in range(len(view.self_pos)) if view.self_alive[i]]
        for x in range(len(alive)):
            for y in range(x + 1, len(alive)):
                d = float(np.linalg.norm(view.self_pos[alive[x]] - view.self_pos[alive[y]]))
                self.ep_min[-1] = min(self.ep_min[-1], d)
                self.ticks += 1
                self.close += d < self.contact_r
        return self.inner.act(view)

    def __getattr__(self, k):
        return getattr(self.inner, k)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", nargs="+", default=["tag", "assault", "escort"])
    p.add_argument("--episodes", type=int, default=150)
    p.add_argument("--seed", type=int, default=20_000)
    p.add_argument("--out", default="results/separation_cost.json")
    a = p.parse_args()
    out = {}
    for g in a.games:
        game = make_game(g)
        contact = 2 * game.arena.agent_radius
        dt = game.arena.timestep * game.episode.control_repeat
        _, opp = default_controllers(g)
        out[g] = {"separation_m": SEP[g], "contact_m": contact, "links": {}}
        for prof in (PERFECT, VICON_LAB):
            built = build_controllers(Path("models"), g)
            on = SafetyConfig(separation_m=SEP[g])
            arms = {
                "scripted+shield": ShieldedController(built["scripted"]),
                "scripted+shield+sep": ShieldedController(built["scripted"], on),
                "bt_gated_safe": ShieldedController(built["bt_gated"]),
                "bt_gated_safe+sep": ShieldedController(built["bt_gated"], on),
            }
            res = {}
            for name, c in arms.items():
                pr = Probe(LinkedController(c, prof, seed=4242, dt=dt, name=name))
                pr.contact_r = contact
                rep = run_batch(pr, opp, a.episodes, a.seed, game=g, label=name)
                wins = [bool(r.win) for r in rep.records]
                m = np.array(pr.ep_min)
                filt = c.filter
                res[name] = {
                    "win_rate": float(np.mean(wins)), "ci": wilson(sum(wins), len(wins)),
                    "wins": wins,
                    "contact_episodes": float(np.mean(m < contact)),
                    "contact_tick_frac": pr.close / max(pr.ticks, 1),
                    "min_mate_dist_p10": float(np.percentile(m, 10)),
                    "min_mate_dist_median": float(np.median(m)),
                }
                print(f"{g:<8} {prof.name:<10} {name:<20} win={res[name]['win_rate']:.3f} "
                      f"contact_eps={res[name]['contact_episodes']:.2f} "
                      f"ticks<2r={res[name]['contact_tick_frac']:.4f} "
                      f"p10={res[name]['min_mate_dist_p10']:.2f}", flush=True)
            for base in ("scripted+shield", "bt_gated_safe"):
                d = paired_delta(res[base + "+sep"]["wins"], res[base]["wins"])
                res[base + "+sep"]["vs_off"] = d
                print(f"   {base}: sep - off = {d['delta']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}]",
                      flush=True)
            out[g]["links"][prof.name] = res
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"episodes": a.episodes, "seed0": a.seed,
                                           "games": out}, indent=1))


if __name__ == "__main__":
    main()
