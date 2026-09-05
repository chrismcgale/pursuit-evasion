"""Scripted attacker team (the *missile* side of ``assault`` / ``escort``).

A missile only has lateral control authority, so the controller's job is purely
to nominate a steering direction; ``env.dynamics.apply_dynamics`` then discards
whatever component it cannot physically produce. That means this controller can
be written as if it were holonomic and still fly like a missile — the airframe
is what makes the trajectory look like a missile's, not the guidance law.

Three behaviours, in priority order:

1. **Lead guidance to the asset.** Aim at where the asset *will be* (matters for
   the moving convoy in ``escort``), not where it is.
2. **Bearing split.** While still far out, each missile biases to a different
   side of the target so the raid arrives on separated bearings — a single
   interceptor cannot sit on both. The bias fades with range so the terminal
   phase is a clean run-in.
3. **Terminal break.** If an interceptor is close and on a collision bearing,
   apply a lateral break. Because lateral authority is limited and thrust stays
   locked to the body axis, the break costs closure — evading is not free, which
   is exactly the trade the defence is trying to force.
"""
from __future__ import annotations

import numpy as np

from ..env.core import TeamView
from .base import BaseController, unit
from .pursuers import lead_intercept_dir

_UP = np.array([0.0, 0.0, 1.0])


def _perp(v: np.ndarray) -> np.ndarray:
    """Any unit vector perpendicular to ``v`` (prefers the horizontal plane)."""
    p = np.cross(v, _UP)
    if np.linalg.norm(p) < 1e-6:
        p = np.cross(v, np.array([1.0, 0.0, 0.0]))
    return unit(p)


class MissileAttackers(BaseController):
    name = "scripted_attackers"

    def __init__(self, split: float = 0.55, split_range: float = 25.0,
                 dodge_range: float = 11.0, dodge_gain: float = 1.5):
        self.split = split
        self.split_range = split_range
        self.dodge_range = dodge_range
        self.dodge_gain = dodge_gain

    def act(self, view: TeamView) -> np.ndarray:
        n_self = view.self_pos.shape[0]
        actions = np.zeros((n_self, 3))
        if not view.has_asset:
            return actions.reshape(-1)

        asset_p, asset_v = view.asset_pos, view.asset_vel
        for i in range(n_self):
            if not view.self_alive[i]:
                continue
            pos = view.self_pos[i]
            rel = asset_p - pos
            rng = float(np.linalg.norm(rel))
            d = lead_intercept_dir(rel, asset_v, view.vmax)

            # 2. bearing split, fading to zero as the range closes
            if rng > 1e-6:
                bias = self.split * float(np.clip((rng - 6.0) / self.split_range, 0.0, 1.0))
                side = 1.0 if i % 2 == 0 else -1.0
                d = unit(d + side * bias * _perp(d))

            # 3. terminal break against the nearest interceptor on a collision bearing
            best_j, best_d = -1, float("inf")
            for j in range(view.opp_pos.shape[0]):
                if not view.opp_alive[j]:
                    continue
                dj = float(np.linalg.norm(view.opp_pos[j] - pos))
                if dj < best_d:
                    best_j, best_d = j, dj
            if best_j >= 0 and best_d < self.dodge_range:
                los = unit(view.opp_pos[best_j] - pos)
                # only break if the threat is roughly ahead; one behind us is beaten
                if float(los @ d) > 0.1:
                    urgency = 1.0 - best_d / self.dodge_range
                    side = 1.0 if (i + best_j) % 2 == 0 else -1.0
                    d = unit(d + side * self.dodge_gain * urgency * _perp(los))
            actions[i] = d
        return np.clip(actions.reshape(-1), -1, 1)
