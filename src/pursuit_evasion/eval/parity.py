"""Cross-runtime parity check — the Python half of ``pe_run --parity``.

CLAUDE.md's central invariant is that the Python (research) and C++ (production)
runtimes are semantically identical: same ``arena.xml``, same ``libmujoco``, and
line-for-line mirrors of dynamics, observations, scripted controllers, features,
gate predicates and the safety filter. That is only an invariant if something
checks it, so this module runs the *same* five fixed start configurations that
``cpp/src/runner.cpp:parity_starts()`` hard-codes, scripted-vs-scripted, and
prints a per-episode checksum for line-by-line diffing::

    ./build/pe_run --parity        # from cpp/
    uv run pe-parity               # from the repo root

Two separate things are being tested, and they fail differently:

* **Exact-start parity** (this file). Same starts, deterministic controllers, so
  the trajectories should agree to floating-point noise. Checksums matching to
  ~1e-6 means the *logic* is mirrored. A checksum that differs in the third
  decimal is a real semantic bug, not numerical drift.
* **Aggregate parity** (``dump_starts_file`` + ``pe_run --starts-file``). Runs
  100 episodes per side and compares win rate / captures / min separation. This
  is the honest end-to-end check, but note that a *handful* of long chaotic
  chases diverge in step count even when every win/loss outcome agrees — two
  agents orbiting at ~1e-9 separation eventually break symmetry differently.
  Judge aggregate parity on outcomes, not on the long tail of ``steps2win``.

The checksum is ``sum(min_dist)`` over the episode: it integrates the whole
trajectory rather than sampling its end, so a divergence anywhere shows up.
"""
from __future__ import annotations

import argparse

from ..env.games import make_game
from ..scripted import default_controllers

# EXACTLY the configurations in cpp/src/runner.cpp:parity_starts(). If you change
# one side you must change the other — that is the point of the file.
PARITY_STARTS = [
    {"pursuer0": (-8, 1, 3), "pursuer1": (-8, -1, 3), "evader0": (8, 1, 3), "evader1": (8, -1, 4)},
    {"pursuer0": (-6, 3, 2), "pursuer1": (-7, -2, 5), "evader0": (6, -3, 3), "evader1": (7, 2, 4)},
    {"pursuer0": (-9, 0, 4), "pursuer1": (-5, 4, 2), "evader0": (5, 0, 5), "evader1": (8, -4, 3)},
    {"pursuer0": (-4, -4, 3), "pursuer1": (-8, 2, 6), "evader0": (7, 3, 2), "evader1": (4, -2, 5)},
    {"pursuer0": (-7, -3, 5), "pursuer1": (-6, 1, 3), "evader0": (6, 4, 4), "evader1": (9, -1, 2)},
]


def run_parity(game_key: str = "tag") -> list[dict]:
    """Scripted-vs-scripted from the fixed starts; returns one record per config."""
    from ..env.core import PursuitEvasionCore

    game = make_game(game_key)
    core = PursuitEvasionCore(game=game)
    pursuer, evader = default_controllers(game_key)

    out = []
    for k, starts in enumerate(PARITY_STARTS):
        pv, ev = core.reset_with_starts(starts)
        pursuer.reset()
        evader.reset()
        checksum = 0.0
        while True:
            r = core.step(pursuer.act(pv), evader.act(ev))
            pv, ev = r.pursuer_view, r.evader_view
            checksum += r.info["min_dist"]
            if r.terminated or r.truncated:
                break
        rec = {"cfg": k, "win": int(r.info["all_captured"]),
               "steps": int(r.info["steps"]), "checksum": checksum}
        out.append(rec)
        print(f"[py-parity] cfg={k} win={rec['win']} steps={rec['steps']} "
              f"checksum={checksum:.6f}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--game", default="tag",
                   help="tag only for the fixed starts; the objective games spawn "
                        "on a different geometry, use --starts-file for those")
    args = p.parse_args(argv)
    run_parity(args.game)
    print("[py-parity] compare against: (cd cpp && ./build/pe_run --parity)")


if __name__ == "__main__":
    main()
