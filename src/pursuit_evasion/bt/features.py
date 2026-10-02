"""Per-agent situational features the behaviour tree branches on.

These are cheap geometric summaries computed from a ``TeamView`` for a single
agent. The behaviour tree reads them to decide, tick by tick, whether that agent
should be driven by the scripted controller or by the learned policy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..env.core import TEAM_PURSUERS, TeamView
from ..scripted.defenders import time_to_asset
from ..scripted.pursuers import lead_intercept_time


def _unit(v, eps=1e-8):
    n = np.linalg.norm(v)
    return v / n if n > eps else np.zeros_like(v)


@dataclass
class AgentFeatures:
    alive: bool
    dist_nearest: float          # distance to nearest relevant other-team agent
    dist_second: float           # distance to 2nd nearest (inf if none)
    n_live_others: int           # live opponents (pursuer view) / threats (evader view)
    closing_rate: float          # rate the nearest other-agent gap is shrinking (m/s)
    target_lateral: float        # nearest other's speed perpendicular to LOS (maneuvering)
    intercept_ahead: bool        # nearest target is roughly ahead & catchable (pursuer)
    contested: bool              # two others at comparable range (ambiguous assignment)
    # --- objective games; inert (inf / True / 1.0) when the game has no asset --
    asset_dist: float = float("inf")        # own distance to the defended asset
    threat_time: float = float("inf")       # seconds until the nearest threat reaches it
    intercept_feasible: bool = True         # a positive lead-intercept solution exists
    fuel: float = 1.0                       # own remaining burn fraction
    # The defender law's OWN regime (GuardDefenders.plan): no intercept for its
    # assigned threat, so it is parking on a gate point. Set by GatedController
    # from the scripted controller, never recomputed here — see _defender_predicates.
    scripted_fallback: bool = False


def _pairwise(view: TeamView, i: int, others_pos, others_vel, others_alive):
    pos = view.self_pos[i]
    vel = view.self_vel[i]
    live = [j for j in range(others_pos.shape[0]) if others_alive[j]]
    if not live:
        return AgentFeatures(bool(view.self_alive[i]), float("inf"), float("inf"),
                             0, 0.0, 0.0, False, False)
    d = np.array([np.linalg.norm(others_pos[j] - pos) for j in live])
    order = np.argsort(d)
    j0 = live[order[0]]
    dist0 = float(d[order[0]])
    dist1 = float(d[order[1]]) if len(order) > 1 else float("inf")

    los = _unit(others_pos[j0] - pos)
    rel_v = others_vel[j0] - vel
    closing = float(-rel_v @ los)                      # >0 means gap shrinking
    tv = others_vel[j0]
    lateral = float(np.linalg.norm(tv - (tv @ los) * los))
    # "ahead & catchable": target in front of our motion and not already on top of us
    ahead = bool((los @ _unit(vel) > 0.3) or np.linalg.norm(vel) < 1e-3) and dist0 > 1.2
    contested = np.isfinite(dist1) and (dist1 - dist0) < 0.35 * max(dist0, 1e-6)

    feat = AgentFeatures(bool(view.self_alive[i]), dist0, dist1, len(live),
                         closing, lateral, ahead, bool(contested))

    # --- objective-game extras ------------------------------------------------
    if view.has_asset:
        feat.asset_dist = float(np.linalg.norm(view.asset_pos - pos))
        feat.threat_time = time_to_asset(others_pos[j0], others_vel[j0],
                                         view.asset_pos, view.opp_vmax)
        feat.intercept_feasible = lead_intercept_time(
            others_pos[j0] - pos, others_vel[j0], view.vmax) is not None
    if view.self_fuel is not None:
        feat.fuel = float(view.self_fuel[i])
    return feat


def agent_features(view: TeamView, i: int) -> AgentFeatures:
    if view.team == TEAM_PURSUERS:
        return _pairwise(view, i, view.opp_pos, view.opp_vel, view.opp_alive)
    # evader view: the "others" are the pursuers (all always alive)
    return _pairwise(view, i, view.opp_pos, view.opp_vel, view.opp_alive)
