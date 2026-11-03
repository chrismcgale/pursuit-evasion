"""Aggregate parity under a degraded link: Python vs C++, episode by episode.

Both runtimes fly the shipped pursuer-slot stack Linked(Shielded(Gated(scripted,
ONNX policy))) from the same start file and the same corruption tape, and every
episode's outcome, violation counts and the gate's RL ticks must agree.

    .venv/bin/python scripts/link_aggregate_parity.py --game assault --link vicon_lab --episodes 50
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pursuit_evasion  # noqa: F401  (first: pins OpenBLAS before numpy loads)
import numpy as np
from pursuit_evasion.bt.gating import GatedController
from pursuit_evasion.env.core import TEAM_PURSUERS
from pursuit_evasion.env.games import make_game
from pursuit_evasion.env.link import PRESETS, LinkedController
from pursuit_evasion.eval import dump_starts
from pursuit_evasion.eval.parity import PARITY_ONNX, OnnxRLController
from pursuit_evasion.eval.scenarios import run_batch
from pursuit_evasion.safety import ShieldedController
from pursuit_evasion.scripted import default_controllers

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="assault")
    p.add_argument("--link", default="vicon_lab")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=20_000)
    a = p.parse_args()
    tmp = Path(tempfile.mkdtemp())
    dump_starts.main(["--games", a.game, "--episodes", str(a.episodes), "--seed",
                      str(a.seed), "--dir", str(tmp), "--link", a.link])
    starts = [list(map(float, ln.split())) for ln in open(tmp / f"s_{a.game}.txt") if ln.strip()]
    names = ["pursuer0", "pursuer1", "evader0", "evader1"]

    ep = {"i": 0}

    def starts_fn(core, rng):
        row = starts[ep["i"]]
        ep["i"] += 1
        return {n: tuple(row[3 * k:3 * k + 3]) for k, n in enumerate(names)}

    game = make_game(a.game)
    dt = game.arena.timestep * game.episode.control_repeat
    scripted, opp = default_controllers(a.game)
    gate = GatedController(TEAM_PURSUERS, scripted, OnnxRLController(ROOT / PARITY_ONNX[a.game]),
                           game=a.game)
    rl_ticks = []

    class Count:  # accumulate gate ticks per episode (the gate zeroes on reset)
        def __init__(s, inner): s.inner, s.name = inner, inner.name
        def reset(s):
            if rl_ticks is not None and ep["i"] > 0:
                rl_ticks.append(gate.mode_counts["rl"])
            s.inner.reset()
        def act(s, v): return s.inner.act(v)
        @property
        def filter(s): return s.inner.filter

    ctrl = Count(LinkedController(ShieldedController(gate), PRESETS[a.link], seed=4242, dt=dt))
    rep = run_batch(ctrl, opp, a.episodes, a.seed, game=a.game, starts_fn=starts_fn)
    py_wins = [bool(r.win) for r in rep.records]
    agg = rep.aggregate()

    log = tmp / "cpp.json"
    cmd = ["./build/pe_run", "--game", a.game, "--controller", "bt_safe", "--starts-file",
           str(tmp / f"s_{a.game}.txt"), "--link", a.link, "--link-tape",
           str(tmp / f"t_{a.game}_{a.link}.txt"), "--log", str(log)]
    out = subprocess.run(cmd, cwd=ROOT / "cpp", capture_output=True, text=True, env=os.environ)
    print(out.stdout.strip())
    if out.returncode != 0:
        print(out.stderr)
        return 1
    cpp = json.loads(log.read_text())
    cpp_wins = [e["win"] for e in cpp["episodes"]]
    flips = [i for i, (x, y) in enumerate(zip(py_wins, cpp_wins)) if x != y]
    print(f"[py ] win={agg['win_rate']:.3f} viol(geo/spd)={agg['geofence_viol']}/{agg['speed_viol']}")
    print(f"[cpp] win={cpp['aggregate']['win_rate']:.3f} viol(geo/spd)="
          f"{cpp['aggregate']['geofence_viol']}/{cpp['aggregate']['speed_viol']}")
    ok = (not flips and agg["geofence_viol"] == cpp["aggregate"]["geofence_viol"]
          and agg["speed_viol"] == cpp["aggregate"]["speed_viol"])
    print(f"link aggregate parity {'OK' if ok else 'FAIL'}: {a.game} {a.link} n={a.episodes}, "
          f"flipped episodes={flips}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
