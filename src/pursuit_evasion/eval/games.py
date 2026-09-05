"""Cross-game tournament: does the "gate the policy" result survive a change of game?

The tag game asks one question — can a faster drone corner a slower one? The
objective games ask a different one: can a *slower* interceptor stop a *faster*
missile before it reaches something. That inversion matters, because it moves the
regime where the scripted controller is weak:

  * ``tag``    — scripted lead-intercept is near-optimal in open space and only
                 breaks down in close-quarters reversals.
  * ``assault`` — the defenders are slower than the attackers, so for most of the
                 episode there is *no* intercept solution at all and the scripted
                 law degenerates to "sit on the gate point". That is a much bigger
                 hole for the learned branch to fill.
  * ``escort``  — same, plus the keep-out volume is moving, so the gate point is
                 never stable and pure geometry ages badly.

Run every available controller across every game on identical seeds:

    uv run pe-games --episodes 200
    uv run pe-games --games assault escort --episodes 300
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO

from ..bt.gating import GatedController
from ..env.core import TEAM_PURSUERS
from ..env.games import GAME_KEYS, make_game
from ..safety import ShieldedController
from ..scripted import default_controllers
from ..scripted.base import RLController
from .scenarios import run_batch


def _load(path: Path):
    if not path.exists():
        return None
    return RLController(PPO.load(str(path), device="cpu"), name=path.stem)


def _model_stems(game_key: str) -> list[str]:
    """Candidate policy files for a game, best-last (the gate takes the last hit)."""
    if game_key == "tag":
        return ["pursuer", "pursuer_bc", "pursuer_bc_ft", "pursuer_dagger"]
    return [f"{game_key}", f"{game_key}_dagger"]


def build_controllers(models: Path, game_key: str) -> dict:
    """Every controller we can assemble for one game, keyed by display name."""
    scripted, _ = default_controllers(game_key)
    ctrls = {"scripted": scripted}

    best = None
    for stem in _model_stems(game_key):
        pol = _load(models / f"{stem}.zip")
        if pol is None:
            continue
        ctrls[stem] = pol
        best = pol

    if best is not None:
        gate = GatedController(TEAM_PURSUERS, default_controllers(game_key)[0], best,
                               game=game_key)
        gate.name = f"bt_gated({best.name})"
        ctrls["bt_gated"] = gate
        ctrls["bt_gated_safe"] = ShieldedController(gate)
    return ctrls


def _md_table(section: dict, game) -> str:
    if game.has_asset:
        head = (f"| controller | defended | breach rate | attackers down | burned out | "
                f"closest approach (m) | geo/spd viol |")
        sep = "|---|---|---|---|---|---|---|"
    else:
        head = "| controller | win rate | mean captures | steps→win | min sep (m) | geo/spd viol |"
        sep = "|---|---|---|---|---|---|"
    rows = [head, sep]
    for name, a in section.items():
        viol = f"{a['geofence_viol']}/{a['speed_viol']}"
        if game.has_asset:
            rows.append(f"| `{name}` | {a['win_rate']:.2f} | {a['breach_rate']:.2f} | "
                        f"{a['mean_captures']:.2f} | {a['spent_rate']:.2f} | "
                        f"{a['mean_min_asset_dist']:.2f} | {viol} |")
        else:
            s2w = a["mean_steps_to_win"]
            s2w = "—" if s2w != s2w else f"{s2w:.0f}"
            rows.append(f"| `{name}` | {a['win_rate']:.2f} | {a['mean_captures']:.2f} | "
                        f"{s2w} | {a['mean_min_separation']:.2f} | {viol} |")
    title = (f"**{game.key}** — {game.pursuer_role} (pursuer slot) vs "
             f"{game.evader_role}; win = "
             + ("raid stopped" if game.has_asset else "both tagged"))
    return title + "\n\n" + "\n".join(rows) + "\n"


def write_writeup_tables(results: dict, writeup: Path):
    """Replace the GAMES:AUTOFILL marker in the writeup with generated tables."""
    if not writeup.exists():
        return
    block = "\n".join(_md_table(sec, make_game(key)) + "\n"
                      for key, sec in results.items())
    text = writeup.read_text()
    marker = "<!-- GAMES:AUTOFILL -->"
    if marker not in text:
        return
    head = text[:text.index(marker)] + marker + "\n\n"
    rest = text[text.index(marker) + len(marker):]
    idx = rest.find("\n## ")
    tail = rest[idx + 1:] if idx != -1 else ""
    writeup.write_text(head + block + tail)


def run_games(models: Path, game_keys, episodes: int, seed0: int, out: Path):
    results = {}
    for key in game_keys:
        game = make_game(key)
        _, opponent = default_controllers(key)
        ctrls = build_controllers(models, key)
        objective = ("stop the raid" if game.has_asset else "tag both evaders")
        print(f"\n=== {key.upper()} ({episodes} eps) — {game.pursuer_role} must {objective} "
              f"vs scripted {game.evader_role} ===")
        section = {}
        for name, ctrl in ctrls.items():
            rep = run_batch(ctrl, opponent, episodes, seed0,
                            game=game, label=name)
            section[name] = rep.aggregate()
            print("  " + rep.row())
            if name == "bt_gated":
                out.mkdir(parents=True, exist_ok=True)
                rep.save_failures(out / f"failures_{key}_bt_gated.json")
        results[key] = section

    out.mkdir(parents=True, exist_ok=True)
    # Merge into any previous run: re-running a single game (e.g. after retraining
    # one model) must not silently delete the other games' tables from the writeup.
    merged = {}
    prev = out / "games.json"
    if prev.exists():
        merged.update(json.loads(prev.read_text()))
    merged.update(results)
    merged = {k: merged[k] for k in GAME_KEYS if k in merged}   # stable game order
    prev.write_text(json.dumps(merged, indent=2))
    write_writeup_tables(merged, Path("writeup/when-each-wins.md"))
    print(f"\n[games] wrote {out}/games.json and refreshed writeup tables "
          f"({', '.join(results)} rerun; {len(merged)} games in the doc)")
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description="Cross-game controller tournament")
    p.add_argument("--models", default="models")
    p.add_argument("--games", nargs="+", default=list(GAME_KEYS), choices=list(GAME_KEYS))
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--out", default="results")
    args = p.parse_args(argv)
    run_games(Path(args.models), args.games, args.episodes, args.seed, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
