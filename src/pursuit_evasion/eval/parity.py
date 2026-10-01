"""Cross-runtime parity check — the Python half of ``pe_run --parity``.

CLAUDE.md's central invariant is that the Python (research) and C++ (production)
runtimes are semantically identical: same ``arena.xml``, same ``libmujoco``, and
line-for-line mirrors of dynamics, observations, scripted controllers, features,
gate predicates and the safety filter. That is only an invariant if something
checks it, so this module runs the *same* fixed start configurations that
``cpp/src/runner.cpp:parity_starts()`` hard-codes and prints a per-episode
checksum for line-by-line diffing::

    ./build/pe_run --parity --game assault --controller bt_safe   # from cpp/
    uv run pe-parity --game assault --controller bt_safe          # from the repo root

Every game has its own fixed starts, and two controller modes are checked:

* ``scripted`` — scripted-vs-scripted. Covers dynamics (incl. the missile
  transform), substep capture, breach / burn-out, and both scripted laws.
* ``bt_safe`` — the shipped pursuer-slot stack ``Shielded(Gated(scripted,
  policy))`` against the scripted opponent. Covers feature extraction, the gate
  predicates (py_trees vs the BehaviorTree.CPP XML), the safety filter and the
  policy. Both sides run the **same ONNX file through the same ONNX Runtime
  version**, so the policy path is exact too, not merely close; the line also
  carries the RL tick count and the shield's violation counts, which must match
  exactly.

Two separate things are being tested, and they fail differently:

* **Exact-start parity** (this file). Same starts, deterministic controllers, so
  the trajectories should agree to floating-point noise. A checksum that differs
  in the third decimal is a real semantic bug, not numerical drift.
* **Aggregate parity** (``dump_starts_file`` + ``pe_run --starts-file``). Runs
  100 episodes per side and compares win rate / captures / min separation. This
  is the honest end-to-end check, but note that a *handful* of long chaotic tag
  chases diverge in step count even when every win/loss outcome agrees. Judge
  aggregate parity on outcomes, not on the long tail of ``steps2win``.

The checksum is ``sum(min_dist)`` over the episode: it integrates the whole
trajectory rather than sampling its end, so a divergence anywhere shows up.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..env.games import GAME_KEYS, make_game
from ..scripted import default_controllers
from ..scripted.base import BaseController

# EXACTLY the configurations in cpp/src/runner.cpp:parity_starts(). If you change
# one side you must change the other — that is the point of the file.
_TAG = [
    {"pursuer0": (-8, 1, 3), "pursuer1": (-8, -1, 3), "evader0": (8, 1, 3), "evader1": (8, -1, 4)},
    {"pursuer0": (-6, 3, 2), "pursuer1": (-7, -2, 5), "evader0": (6, -3, 3), "evader1": (7, 2, 4)},
    {"pursuer0": (-9, 0, 4), "pursuer1": (-5, 4, 2), "evader0": (5, 0, 5), "evader1": (8, -4, 3)},
    {"pursuer0": (-4, -4, 3), "pursuer1": (-8, 2, 6), "evader0": (7, 3, 2), "evader1": (4, -2, 5)},
    {"pursuer0": (-7, -3, 5), "pursuer1": (-6, 1, 3), "evader0": (6, 4, 4), "evader1": (9, -1, 2)},
]
# assault: asset at the origin, raid ~30 m out on split bearings, CAP within 9 m
_ASSAULT = [
    {"pursuer0": (6, 0, 4), "pursuer1": (-5, 3, 5), "evader0": (30, 5, 10), "evader1": (-28, -12, 8)},
    {"pursuer0": (0, 7, 3), "pursuer1": (4, -6, 6), "evader0": (5, 32, 12), "evader1": (-20, -25, 7)},
    {"pursuer0": (-6, -2, 5), "pursuer1": (7, 4, 2), "evader0": (31, -8, 6), "evader1": (-31, 6, 14)},
    {"pursuer0": (3, 5, 7), "pursuer1": (-4, -5, 3), "evader0": (-10, 30, 9), "evader1": (12, -30, 11)},
    {"pursuer0": (8, 1, 4), "pursuer1": (-2, 8, 6), "evader0": (25, 20, 13), "evader1": (-25, -20, 5)},
]
# escort: convoy starts at (-32, 0, 1.5) heading +x; the raid waits downrange
_ESCORT = [
    {"pursuer0": (-26, 3, 4), "pursuer1": (-30, -6, 5), "evader0": (28, 10, 10), "evader1": (25, -15, 8)},
    {"pursuer0": (-25, -4, 3), "pursuer1": (-36, 5, 6), "evader0": (30, 0, 12), "evader1": (20, 20, 7)},
    {"pursuer0": (-28, 6, 5), "pursuer1": (-34, -3, 2), "evader0": (15, 25, 6), "evader1": (29, -5, 14)},
    {"pursuer0": (-24, 0, 7), "pursuer1": (-38, 2, 3), "evader0": (22, -22, 9), "evader1": (31, 8, 11)},
    {"pursuer0": (-30, 8, 4), "pursuer1": (-27, -7, 6), "evader0": (10, 28, 13), "evader1": (26, -18, 5)},
]
PARITY_STARTS = {"tag": _TAG, "assault": _ASSAULT, "escort": _ESCORT}

# The policy each game's shipped gate routes to — mirrors runner.cpp:default_onnx
# and eval/games.py:_model_stems (best-last).
PARITY_ONNX = {"tag": "models/pursuer_dagger.onnx",
               "assault": "models/assault_dagger.onnx",
               "escort": "models/escort_dagger.onnx"}


class OnnxRLController(BaseController):
    """The policy through ONNX Runtime — the exact graph the C++ runtime loads.

    Using SB3 here would compare torch against ORT and drift at ~1e-6 per tick,
    which a discrete gate amplifies into flipped episodes. Same file, same ORT
    version, same float32 obs: the two halves then compute the same numbers.
    """

    def __init__(self, path: str | Path, name: str = "rl"):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.name = name

    def act(self, view):
        from ..env.observations import build_team_obs
        obs = build_team_obs(view).astype(np.float32)[None, :]
        a = self.sess.run(["action"], {"obs": obs})[0][0]
        return np.clip(a.astype(np.float64), -1.0, 1.0)


def _build(game_key: str, controller: str, onnx: str | None):
    pursuer, evader = default_controllers(game_key)
    if controller == "scripted":
        return pursuer, evader, None
    from ..bt.gating import GatedController
    from ..env.core import TEAM_PURSUERS
    from ..safety import ShieldedController
    rl = OnnxRLController(onnx or PARITY_ONNX[game_key])
    gate = GatedController(TEAM_PURSUERS, pursuer, rl, game=game_key)
    return ShieldedController(gate), evader, gate


def run_parity(game_key: str = "tag", controller: str = "scripted",
               onnx: str | None = None) -> list[dict]:
    """Fixed starts for one game; returns one record per config."""
    from ..env.core import PursuitEvasionCore

    game = make_game(game_key)
    core = PursuitEvasionCore(game=game)
    pursuer, evader, gate = _build(game_key, controller, onnx)

    out = []
    for k, starts in enumerate(PARITY_STARTS[game_key]):
        pv, ev = core.reset_with_starts(starts)
        pursuer.reset()
        evader.reset()
        checksum = 0.0
        rl_ticks = 0
        while True:
            r = core.step(pursuer.act(pv), evader.act(ev))
            if gate is not None:
                rl_ticks = gate.mode_counts["rl"]
            pv, ev = r.pursuer_view, r.evader_view
            checksum += r.info["min_dist"]
            if r.terminated or r.truncated:
                break
        f = getattr(pursuer, "filter", None)
        geo, spd = (f.n_geofence, f.n_speed) if f is not None else (0, 0)
        rec = {"cfg": k, "win": int(r.info["pursuer_win"]), "steps": int(r.info["steps"]),
               "checksum": checksum, "rl": rl_ticks, "geo": geo, "spd": spd}
        out.append(rec)
        print(f"[py-parity] game={game_key} ctrl={controller} cfg={k} win={rec['win']} "
              f"steps={rec['steps']} checksum={checksum:.6f} rl={rl_ticks} "
              f"viol={geo}/{spd}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--game", default="tag", choices=list(GAME_KEYS))
    p.add_argument("--controller", default="scripted", choices=["scripted", "bt_safe"])
    p.add_argument("--onnx", default=None, help="policy for bt_safe (default per game)")
    args = p.parse_args(argv)
    run_parity(args.game, args.controller, args.onnx)
    print(f"[py-parity] compare against: (cd cpp && ./build/pe_run --parity "
          f"--game {args.game} --controller {args.controller})")


if __name__ == "__main__":
    main()
