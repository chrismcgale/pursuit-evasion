"""Scripted defender team (the *interceptor drone* side of ``assault`` / ``escort``).

The interceptors are ~40% slower than the missiles they are defending against,
which changes the problem completely: **a tail chase is not a strategy**. There
is no constant-speed pursuit that closes on a faster target that is opening, so
``lead_intercept_time`` returns ``None`` and the classic pursuit controller
degenerates to a hopeless stern chase.

``GuardDefenders`` therefore uses a two-regime law:

* **Feasible intercept** — if the quadratic has a positive root *and* that
  intercept happens before the missile reaches the asset, fly the lead bearing.
  This is the good case and it is exactly the tag-game controller.
* **Infeasible intercept** — otherwise, fall back on *position*: fly to a gate
  point on the missile's inbound bearing, a fixed standoff from the asset. You
  cannot catch it, so you wait where it must come. Terminal defence.

Assignment is by **threat time**, not distance: the missile with the smallest
time-to-asset gets the defender that can meet it soonest. Distance-greedy
assignment (the tag heuristic) is actively wrong here — the nearest missile is
often the least urgent.
"""
from __future__ import annotations

import numpy as np

from ..env.core import TeamView
from .base import BaseController, unit
from .pursuers import lead_intercept_dir, lead_intercept_time

_BIG = 1e9


def time_to_asset(pos: np.ndarray, vel: np.ndarray, asset: np.ndarray,
                  vmax: float) -> float:
    """Optimistic seconds until this attacker reaches the asset."""
    d = float(np.linalg.norm(asset - pos))
    speed = float(np.linalg.norm(vel))
    closing = float(vel @ unit(asset - pos)) if speed > 1e-6 else 0.0
    # use the better of its current closure and its top speed: a missile that is
    # currently turning is still a threat on its full-speed clock
    rate = max(closing, 0.35 * vmax, 1e-6)
    return d / rate


class GuardDefenders(BaseController):
    name = "scripted_defenders"

    def __init__(self, standoff: float = 9.0, min_standoff: float = 3.5,
                 patrol_radius: float = 6.0):
        self.standoff = standoff        # gate distance from the asset (m)
        self.min_standoff = min_standoff
        self.patrol_radius = patrol_radius

    # ------------------------------------------------------------- assignment
    def _assign(self, view: TeamView, asset: np.ndarray) -> list[int]:
        n_self = view.self_pos.shape[0]
        live = [j for j in range(view.opp_pos.shape[0]) if view.opp_alive[j]]
        if not live:
            return [-1] * n_self
        # order threats by time-to-asset (most urgent first)
        threat = sorted(live, key=lambda j: time_to_asset(
            view.opp_pos[j], view.opp_vel[j], asset, view.opp_vmax))
        assign = [-1] * n_self
        free = set(range(n_self))
        for j in threat:
            if not free:
                break
            # give this threat whichever free defender can meet it soonest;
            # if none can intercept, whichever is closest to its gate point
            def cost(i: int) -> float:
                rel = view.opp_pos[j] - view.self_pos[i]
                t = lead_intercept_time(rel, view.opp_vel[j], view.vmax)
                if t is not None:
                    return t
                gate = self._gate_point(view.opp_pos[j], asset)
                return _BIG + float(np.linalg.norm(gate - view.self_pos[i]))
            i = min(free, key=cost)
            assign[i] = j
            free.discard(i)
        # spare defenders back up the most urgent threat
        for i in list(free):
            assign[i] = threat[0]
        return assign

    def _gate_point(self, att_pos: np.ndarray, asset: np.ndarray) -> np.ndarray:
        """A point on the attacker's inbound bearing, a standoff out from the asset."""
        to_att = att_pos - asset
        r = float(np.linalg.norm(to_att))
        if r < 1e-6:
            return asset.copy()
        reach = float(np.clip(r * 0.5, self.min_standoff, self.standoff))
        return asset + unit(to_att) * reach

    # ------------------------------------------------------------------- act
    def act(self, view: TeamView) -> np.ndarray:
        n_self = view.self_pos.shape[0]
        actions = np.zeros((n_self, 3))
        if not view.has_asset:
            return actions.reshape(-1)
        asset = view.asset_pos
        assign = self._assign(view, asset)

        for i in range(n_self):
            if not view.self_alive[i]:
                continue
            pos = view.self_pos[i]
            j = assign[i]
            if j < 0:
                # nothing left to kill: settle back onto the asset's patrol ring
                to_asset = asset - pos
                r = float(np.linalg.norm(to_asset))
                actions[i] = unit(to_asset) * (1.0 if r > self.patrol_radius else -0.3)
                continue

            rel = view.opp_pos[j] - pos
            t_int = lead_intercept_time(rel, view.opp_vel[j], view.vmax)
            t_asset = time_to_asset(view.opp_pos[j], view.opp_vel[j], asset, view.opp_vmax)
            if t_int is not None and t_int <= t_asset:
                actions[i] = lead_intercept_dir(rel, view.opp_vel[j], view.vmax)
            else:
                # can't catch it — be where it has to come through
                gate = self._gate_point(view.opp_pos[j], asset)
                to_gate = gate - pos
                if float(np.linalg.norm(to_gate)) < 0.8:
                    # on station: hold the gate and face the threat
                    actions[i] = unit(rel)
                else:
                    actions[i] = unit(to_gate)
        return np.clip(actions.reshape(-1), -1, 1)
