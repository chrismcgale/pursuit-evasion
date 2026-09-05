"""Run matches between any pursuer controller and any evader controller.

This is the single source of truth for "who won and how fast", shared by the
self-play trainer (to track progress) and the tournament (to produce the
writeup's numbers). Controllers are the uniform ``Controller`` interface, so a
scripted policy, a trained RL policy, and a BT-gated policy are all just
arguments here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..env.core import (ArenaConfig, EpisodeConfig, PursuitEvasionCore,
                        TeamView)


@dataclass
class MatchStats:
    n_episodes: int
    pursuer_wins: int                       # both evaders captured before timeout
    capture_rate: float                     # pursuer_wins / n_episodes
    mean_captures: float                    # avg evaders tagged per episode (0..2)
    mean_steps_to_win: float                # over pursuer wins only (nan if none)
    trajectories: list = field(default_factory=list)   # optional, for viz

    def summary(self) -> str:
        s = (f"{self.n_episodes} eps | capture_rate={self.capture_rate:.2f} "
             f"| mean_captures={self.mean_captures:.2f}")
        if not np.isnan(self.mean_steps_to_win):
            s += f" | mean_steps_to_win={self.mean_steps_to_win:.0f}"
        return s


def run_match(pursuer, evader, n_episodes: int = 30, seed: int = 0,
              arena: ArenaConfig | None = None, episode: EpisodeConfig | None = None,
              record: bool = False) -> MatchStats:
    core = PursuitEvasionCore(arena, episode, seed=seed)
    wins = 0
    total_caps = 0
    steps_to_win = []
    trajectories = []
    names = core.pursuers + core.evaders

    for ep in range(n_episodes):
        pv, ev = core.reset(seed=seed + ep)
        pursuer.reset()
        evader.reset()
        traj = {n: [] for n in names} if record else None
        cap_flags = None
        while True:
            if record:
                for n in core.pursuers + core.evaders:
                    traj[n].append(core._pos(n).copy())
            r = core.step(pursuer.act(pv), evader.act(ev))
            pv, ev = r.pursuer_view, r.evader_view
            cap_flags = r.info["captured"]
            if r.terminated or r.truncated:
                if r.info["all_captured"]:
                    wins += 1
                    steps_to_win.append(r.info["steps"])
                total_caps += int(r.info["n_captured"])
                break
        if record:
            trajectories.append({"traj": traj, "captured": cap_flags,
                                 "win": bool(r.info["all_captured"])})

    return MatchStats(
        n_episodes=n_episodes,
        pursuer_wins=wins,
        capture_rate=wins / n_episodes,
        mean_captures=total_caps / n_episodes,
        mean_steps_to_win=float(np.mean(steps_to_win)) if steps_to_win else float("nan"),
        trajectories=trajectories,
    )
