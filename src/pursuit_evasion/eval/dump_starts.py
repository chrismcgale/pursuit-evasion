"""Write per-game start-state files so both runtimes evaluate identical episodes.

numpy's Generator and C++'s std::mt19937_64 are different streams, so seeding
both with the same integer does NOT produce the same spawns. Aggregate parity
therefore has to be driven from an explicit start set: Python dumps it here, the
C++ runner reads it with ``--starts-file``.

    uv run python -m pursuit_evasion.eval.dump_starts
    (cd cpp && ./build/pe_run --game assault --controller bt_safe \
        --starts-file /tmp/s_assault.txt)
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..env.games import GAME_KEYS
from .scenarios import dump_starts_file


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--games", nargs="+", default=list(GAME_KEYS))
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--dir", type=Path, default=Path("/tmp"),
                   help="output directory; files are named s_<game>.txt")
    args = p.parse_args(argv)

    for key in args.games:
        path = args.dir / f"s_{key}.txt"
        dump_starts_file(path, args.episodes, args.seed, game=key)
        print(f"[starts] {path} — {args.episodes} episodes from seed {args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
