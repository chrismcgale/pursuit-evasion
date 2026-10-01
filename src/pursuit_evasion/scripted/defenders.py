"""Scripted defender team (the *interceptor drone* side of ``assault`` / ``escort``).

The interceptors are ~40% slower than the missiles they are defending against,
which changes the problem completely: **a tail chase is not a strategy**. There
is no constant-speed pursuit that closes on a faster target that is opening, so
``lead_intercept_time`` returns ``None`` and the classic pursuit controller
degenerates to a hopeless stern chase.

``GuardDefenders`` therefore uses a two-regime law:

* **Feasible intercept** — if the defender can reach the missile's
  ``intercept_radius``-sphere at full speed *and* that happens before the
  missile reaches the asset, fly the lead bearing to that meeting point.
  The sphere matters: a kill only needs the gap under the 1.4 m capture radius,
  and the old point solve (radius 0) called a whole band of reachable geometry
  "infeasible". Solving against half the capture radius (0.7 m, a margin for
  the missile's last-second break) lifted scripted assault 0.53 -> 0.80 and
  escort 0.44 -> 0.55 on held-out seeds — more than the old gate ever added.
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
from .pursuers import lead_intercept_time_sphere

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
                 patrol_radius: float = 6.0, intercept_radius: float = 0.7):
        self.standoff = standoff        # gate distance from the asset (m)
        self.min_standoff = min_standoff
        self.patrol_radius = patrol_radius
        # solve the intercept against this sphere, not the point (0 = legacy)
        self.intercept_radius = intercept_radius

    def _intercept_time(self, rel: np.ndarray, vel: np.ndarray, vp: float) -> float | None:
        return lead_intercept_time_sphere(rel, vel, vp, self.intercept_radius)

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
                t = self._intercept_time(rel, view.opp_vel[j], view.vmax)
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
    def plan(self, view: TeamView) -> tuple[np.ndarray, np.ndarray]:
        """(team action, per-agent fallback flags).

        ``fallback[i]`` is True when agent i has a threat but this law has no
        intercept for it and is parking on the gate point — the one regime the
        air-defence gate hands to the policy. The gate reads it from HERE rather
        than recomputing a proxy, so the handover is the controller's own
        regime switch by construction (the old nearest-threat, point-solve
        proxy disagreed with it on ~1/3 of fallback ticks).
        """
        n_self = view.self_pos.shape[0]
        actions = np.zeros((n_self, 3))
        fallback = np.zeros(n_self, dtype=bool)
        if not view.has_asset:
            return actions.reshape(-1), fallback
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
            t_int = self._intercept_time(rel, view.opp_vel[j], view.vmax)
            t_asset = time_to_asset(view.opp_pos[j], view.opp_vel[j], asset, view.opp_vmax)
            if t_int is not None and t_int <= t_asset:
                aim = rel + view.opp_vel[j] * t_int
                actions[i] = unit(aim) if t_int > 0.0 else unit(rel)
            else:
                # can't catch it — be where it has to come through
                fallback[i] = True
                gate = self._gate_point(view.opp_pos[j], asset)
                to_gate = gate - pos
                if float(np.linalg.norm(to_gate)) < 0.8:
                    # on station: hold the gate and face the threat
                    actions[i] = unit(rel)
                else:
                    actions[i] = unit(to_gate)
        return np.clip(actions.reshape(-1), -1, 1), fallback

    def act(self, view: TeamView) -> np.ndarray:
        return self.plan(view)[0]
