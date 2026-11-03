"""Write per-game start-state files so both runtimes evaluate identical episodes.

numpy's Generator and C++'s std::mt19937_64 are different streams, so seeding
both with the same integer does NOT produce the same spawns. Aggregate parity
therefore has to be driven from an explicit start set: Python dumps it here, the
C++ runner reads it with ``--starts-file``.

    uv run python -m pursuit_evasion.eval.dump_starts
    (cd cpp && ./build/pe_run --game assault --controller bt_safe \
        --starts-file /tmp/s_assault.txt)

With ``--link <preset>`` it also writes the link's corruption tape for the same
episodes (``t_<game>_<preset>.txt``), so a degraded C++ run pairs with
Python's ``LinkedController(seed=--link-seed)`` episode for episode:

    uv run python -m pursuit_evasion.eval.dump_starts --link vicon_lab
    (cd cpp && ./build/pe_run --game assault --controller bt_safe \
        --starts-file /tmp/s_assault.txt --link vicon_lab \
        --link-tape /tmp/t_assault_vicon_lab.txt)
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..env.games import GAME_KEYS, make_game
from ..env.link import PRESETS, write_tape
from .scenarios import dump_starts_file


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--games", nargs="+", default=list(GAME_KEYS))
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--dir", type=Path, default=Path("/tmp"),
                   help="output directory; files are named s_<game>.txt")
    p.add_argument("--link", default=None, choices=list(PRESETS),
                   help="also write the link corruption tape for these episodes")
    p.add_argument("--link-seed", type=int, default=4242)
    args = p.parse_args(argv)

    for key in args.games:
        path = args.dir / f"s_{key}.txt"
        dump_starts_file(path, args.episodes, args.seed, game=key)
        print(f"[starts] {path} — {args.episodes} episodes from seed {args.seed}")
        if args.link and not PRESETS[args.link].is_perfect():
            g = make_game(key)
            tape = args.dir / f"t_{key}_{args.link}.txt"
            # LinkedController resets once in its constructor: episode e -> stream e+1
            write_tape(tape, PRESETS[args.link], args.link_seed,
                       range(1, args.episodes + 1), g.arena.n_pursuers,
                       g.arena.n_evaders, g.episode.max_steps)
            print(f"[tape]   {tape} — {args.link}, seed {args.link_seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
