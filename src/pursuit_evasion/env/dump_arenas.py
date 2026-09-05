"""Regenerate the C++ runtime's arena XMLs from the Python env.

`cpp/assets/*.xml` are GENERATED, never hand-edited: both runtimes load the same
file, so the MJCF is the single source of geometry truth (CLAUDE.md). Run this
after any change to arena dimensions, agent radii, or the asset.

    uv run python -m pursuit_evasion.env.dump_arenas          # write
    uv run python -m pursuit_evasion.env.dump_arenas --check  # CI: fail on drift

``arena.xml`` (tag) must stay byte-identical unless you intend to move the tuned
40-50% scripted baseline, so --check is the useful mode in a pre-commit hook.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .core import PursuitEvasionCore
from .games import GAME_KEYS, make_game

DEFAULT_OUT = Path(__file__).resolve().parents[3] / "cpp" / "assets"


def arena_filename(game_key: str) -> str:
    """Mirrors cpp/include/pe/games.hpp:arena_asset_name."""
    return "arena.xml" if game_key == "tag" else f"arena_{game_key}.xml"


def dump(out_dir: Path = DEFAULT_OUT, check: bool = False) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    drift = 0
    for key in GAME_KEYS:
        path = out_dir / arena_filename(key)
        new = PursuitEvasionCore(game=make_game(key)).static_xml()
        old = path.read_text() if path.exists() else None
        if old == new:
            print(f"[arena] {path.name}: unchanged ({len(new)} bytes)")
        elif check:
            print(f"[arena] {path.name}: DRIFT — regenerate "
                  f"(on disk {'missing' if old is None else str(len(old)) + ' bytes'}, "
                  f"generated {len(new)} bytes)")
            drift += 1
        else:
            path.write_text(new)
            print(f"[arena] {path.name}: wrote {len(new)} bytes "
                  f"({'new' if old is None else 'updated'})")
    return drift


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--check", action="store_true",
                   help="don't write; exit non-zero if any file is stale")
    args = p.parse_args(argv)
    return 1 if dump(args.out, args.check) else 0


if __name__ == "__main__":
    raise SystemExit(main())
