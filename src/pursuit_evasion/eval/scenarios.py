"""Scenario battery + instrumentation.

Runs a controller matchup over a batch of seeded random start states and records
per-episode metrics, then aggregates them. This is the "stand up and instrument
scenarios" layer: everything downstream (the tournament table, the writeup plots,
the failure replays) reads these records.

Metrics per episode:
  * win               — the pursuer-slot team achieved its objective (tag: both
                        evaders tagged before timeout; assault/escort: the raid
                        was stopped, i.e. the asset was never breached)
  * n_captured        — opponents neutralised (0..2)
  * n_spent           — of those, missiles that burned out rather than being hit
  * steps_to_all      — steps to neutralise both (None if not achieved)
  * time_to_first     — seconds to first kill (None if none)
  * min_separation    — closest pursuer↔evader approach over the episode (m)
  * min_asset_dist    — closest an attacker got to the defended asset (objective
                        games only; the "how close did it get" margin)
  * breach            — an attacker reached the asset
  * geofence_viol     — ticks the safety filter corrected a geofence breach
  * speed_viol        — ticks the safety filter corrected a speed breach
  * starts            — spawn positions, so any episode is exactly replayable
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..env.core import (ArenaConfig, EpisodeConfig, PursuitEvasionCore,
                        TeamView)
from ..env.games import GameSpec, make_game


def _dt(episode: EpisodeConfig, arena: ArenaConfig) -> float:
    return arena.timestep * episode.control_repeat


def _violations(controller) -> tuple[int, int]:
    f = getattr(controller, "filter", None)
    if f is not None:
        return int(f.n_geofence), int(f.n_speed)
    return 0, 0


@dataclass
class EpisodeRecord:
    seed: int
    win: bool
    n_captured: int
    steps_to_all: int | None
    time_to_first: float | None
    min_separation: float
    geofence_viol: int
    speed_viol: int
    starts: dict
    # objective games (defaults keep `tag` records byte-compatible)
    n_spent: int = 0
    min_asset_dist: float = float("nan")
    breach: bool = False
    asset_arrived: bool = False


@dataclass
class BatchReport:
    label: str
    records: list[EpisodeRecord] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.records)

    def aggregate(self) -> dict:
        r = self.records
        n = max(len(r), 1)
        wins = [x for x in r if x.win]
        # a defence-game win can arrive by timeout or spent missiles, with no
        # "both neutralised" step to report — only average the wins that have one
        win_steps = [x.steps_to_all for x in wins if x.steps_to_all is not None]
        first_times = [x.time_to_first for x in r if x.time_to_first is not None]
        return {
            "label": self.label,
            "n": len(r),
            "win_rate": sum(x.win for x in r) / n,
            "mean_captures": sum(x.n_captured for x in r) / n,
            "mean_steps_to_win": float(np.mean(win_steps)) if win_steps else float("nan"),
            "mean_time_to_first": float(np.mean(first_times)) if first_times else float("nan"),
            "mean_min_separation": float(np.mean([x.min_separation for x in r])),
            "geofence_viol": sum(x.geofence_viol for x in r),
            "speed_viol": sum(x.speed_viol for x in r),
            "breach_rate": sum(x.breach for x in r) / n,
            "spent_rate": sum(x.n_spent for x in r) / n,
            "mean_min_asset_dist": float(np.nanmean([x.min_asset_dist for x in r]))
            if any(np.isfinite(x.min_asset_dist) for x in r) else float("nan"),
        }

    def failures(self, side: str = "pursuers") -> list[EpisodeRecord]:
        # a "failure" for pursuers is a non-win; for evaders it is being wiped out
        if side == "pursuers":
            return [x for x in self.records if not x.win]
        return [x for x in self.records if x.n_captured == 2]

    def save_failures(self, path: str | Path, side: str = "pursuers"):
        fails = self.failures(side)
        Path(path).write_text(json.dumps(
            {"label": self.label, "side": side, "n_failures": len(fails),
             "episodes": [asdict(x) for x in fails]}, indent=2))
        return len(fails)

    def row(self) -> str:
        a = self.aggregate()
        s2w = f"{a['mean_steps_to_win']:.0f}" if not np.isnan(a["mean_steps_to_win"]) else "  -"
        extra = ""
        if not np.isnan(a["mean_min_asset_dist"]):
            extra = (f" breach={a['breach_rate']:.2f} "
                     f"assetmargin={a['mean_min_asset_dist']:.1f} spent={a['spent_rate']:.2f}")
        return (f"{a['label']:<28} win={a['win_rate']:.2f} caps={a['mean_captures']:.2f} "
                f"steps2win={s2w:>4} minsep={a['mean_min_separation']:.2f} "
                f"viol(geo/spd)={a['geofence_viol']}/{a['speed_viol']}{extra}")


def run_batch(pursuer, evader, n_episodes: int = 100, seed0: int = 10_000,
              arena: ArenaConfig | None = None, episode: EpisodeConfig | None = None,
              label: str = "match", starts_fn=None,
              game: GameSpec | str | None = None) -> BatchReport:
    if isinstance(game, str):
        game = make_game(game)
    core = PursuitEvasionCore(arena, episode, seed=seed0, game=game)
    arena, episode = core.arena, core.ep
    dt = _dt(episode, arena)
    report = BatchReport(label=label)

    for ep in range(n_episodes):
        seed = seed0 + ep
        if starts_fn is not None:
            rng = np.random.default_rng(seed)
            pv, ev = core.reset_with_starts(starts_fn(core, rng))
        else:
            pv, ev = core.reset(seed=seed)
        pursuer.reset()
        evader.reset()
        starts = dict(core.last_starts)
        g0, s0 = _violations(pursuer)  # after reset the shielded filter is zeroed

        min_sep = float("inf")
        min_asset = float("inf")
        time_to_first = None
        prev_captured = 0
        while True:
            r = core.step(pursuer.act(pv), evader.act(ev))
            pv, ev = r.pursuer_view, r.evader_view
            min_sep = min(min_sep, r.info["min_dist"] if r.info["n_captured"] < 2 else min_sep)
            if np.isfinite(r.info["asset_dist"]):
                min_asset = min(min_asset, r.info["asset_dist"])
            if time_to_first is None and r.info["n_captured"] > prev_captured:
                time_to_first = r.info["steps"] * dt
            prev_captured = r.info["n_captured"]
            if r.terminated or r.truncated:
                break

        g1, s1 = _violations(pursuer)
        report.records.append(EpisodeRecord(
            seed=seed,
            win=bool(r.info["pursuer_win"]),
            n_captured=int(r.info["n_captured"]),
            steps_to_all=int(r.info["steps"]) if r.info["all_captured"] else None,
            time_to_first=time_to_first,
            min_separation=float(min_sep),
            geofence_viol=g1 - g0,
            speed_viol=s1 - s0,
            starts=starts,
            n_spent=int(r.info["n_spent"]),
            min_asset_dist=float(min_asset) if np.isfinite(min_asset) else float("nan"),
            breach=bool(r.info["breach"]),
            asset_arrived=bool(r.info["asset_arrived"]),
        ))
    return report


def dump_starts_file(path: str | Path, n: int = 100, seed0: int = 10_000,
                     arena: ArenaConfig | None = None, episode: EpisodeConfig | None = None,
                     game: GameSpec | str | None = None):
    """Write the exact start states for seeds [seed0, seed0+n) as 12 floats/line.

    Order per line: pursuer0 xyz, pursuer1 xyz, evader0 xyz, evader1 xyz. The C++
    runner reads this so both runtimes evaluate byte-identical episodes.
    """
    core = PursuitEvasionCore(arena, episode, seed=seed0, game=game)
    names = core.pursuers + core.evaders
    lines = []
    for ep in range(n):
        core.reset(seed=seed0 + ep)
        s = core.last_starts
        lines.append(" ".join(f"{c:.6f}" for nm in names for c in s[nm]))
    Path(path).write_text("\n".join(lines) + "\n")
    return len(lines)


def replay_starts(record: EpisodeRecord | dict) -> dict:
    """Return the spawn dict from a record so an episode can be re-run exactly."""
    d = record if isinstance(record, dict) else asdict(record)
    return {k: tuple(v) for k, v in d["starts"].items()}
