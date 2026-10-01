"""Scripted pursuer team controllers.

The strong baseline is ``InterceptPursuers``: it assigns each pursuer to a
distinct evader (greedy min-distance), then steers along a *lead-intercept*
bearing — solving for where the evader will be given its current velocity and
the pursuer's top speed — rather than chasing the evader's current position
(which lags and overshoots). When only one evader remains, both pursuers pincer
it from offset bearings so a simple juke can't shake both.
"""
from __future__ import annotations

import numpy as np

from ..env.core import TeamView
from .base import BaseController, unit


def lead_intercept_time(rel_pos: np.ndarray, evader_vel: np.ndarray, vp: float) -> float | None:
    """Smallest positive time-to-intercept, or None if the target is uncatchable.

    Solves |rel_pos + evader_vel * t| = vp * t. ``None`` means no positive root:
    the target is faster and opening, so no constant-speed pursuit closes it —
    the caller must do something smarter than chase (see ``defenders.py``).
    """
    a = float(evader_vel @ evader_vel - vp * vp)
    b = float(2.0 * (rel_pos @ evader_vel))
    c = float(rel_pos @ rel_pos)
    t = None
    if abs(a) < 1e-6:
        if abs(b) > 1e-6:
            cand = -c / b
            if cand > 0:
                t = cand
    else:
        disc = b * b - 4 * a * c
        if disc >= 0:
            sq = np.sqrt(disc)
            roots = [(-b - sq) / (2 * a), (-b + sq) / (2 * a)]
            pos = [r for r in roots if r > 1e-6]
            if pos:
                t = min(pos)
    return t


def lead_intercept_time_sphere(rel_pos: np.ndarray, evader_vel: np.ndarray, vp: float,
                               radius: float) -> float | None:
    """Smallest t >= 0 at which a constant-speed pursuit reaches the target's
    ``radius``-sphere: solves |rel_pos + evader_vel * t| = vp * t + radius.

    A kill only needs the gap below the capture radius, not zero, so the point
    solve above (``radius = 0``) declares "no intercept exists" in a band of
    geometry where one does. That band is most of what the air-defence gate
    used to hand to the policy. ``radius <= 0`` is exactly ``lead_intercept_time``.
    """
    if radius <= 0.0:
        return lead_intercept_time(rel_pos, evader_vel, vp)
    c = float(rel_pos @ rel_pos) - radius * radius
    if c <= 0.0:
        return 0.0                                   # already inside the sphere
    a = float(evader_vel @ evader_vel - vp * vp)
    b = float(2.0 * (rel_pos @ evader_vel)) - 2.0 * vp * radius
    if abs(a) < 1e-9:
        return (-c / b) if b < 0.0 else None
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    sq = np.sqrt(disc)
    roots = [r for r in ((-b - sq) / (2 * a), (-b + sq) / (2 * a)) if r > 1e-6]
    return min(roots) if roots else None


def lead_intercept_dir(rel_pos: np.ndarray, evader_vel: np.ndarray, vp: float) -> np.ndarray:
    """Unit steering direction to the predicted meeting point.

    Falls back to pure pursuit if there is no positive intercept solution
    (target faster / diverging).
    """
    t = lead_intercept_time(rel_pos, evader_vel, vp)
    if t is None:
        return unit(rel_pos)  # pure pursuit fallback
    aim = rel_pos + evader_vel * t
    return unit(aim)


def assign_targets(p_pos: np.ndarray, e_pos: np.ndarray, live: np.ndarray) -> list[int]:
    """Greedy distinct assignment of pursuers to live evaders (indices into e_pos).

    Returns a list of length n_pursuers; entry is the evader index to chase, or
    -1 if no live evaders.
    """
    live_idx = [j for j in range(len(live)) if live[j]]
    n_p = p_pos.shape[0]
    if not live_idx:
        return [-1] * n_p
    if len(live_idx) == 1:
        return [live_idx[0]] * n_p  # both chase the survivor (pincer handled later)

    # cost matrix pursuer x live-evader
    cost = np.linalg.norm(p_pos[:, None, :] - e_pos[None, live_idx, :], axis=2)
    assignment = [-1] * n_p
    used_e: set[int] = set()
    # greedily take globally smallest (pursuer, evader) pairs
    pairs = sorted(
        ((cost[pi, ei], pi, live_idx[ei]) for pi in range(n_p) for ei in range(len(live_idx))),
        key=lambda x: x[0],
    )
    used_p: set[int] = set()
    for _, pi, ej in pairs:
        if pi in used_p or ej in used_e:
            continue
        assignment[pi] = ej
        used_p.add(pi)
        used_e.add(ej)
        if len(used_e) == len(live_idx):
            break
    # any unassigned pursuer chases the nearest live evader
    for pi in range(n_p):
        if assignment[pi] == -1:
            dists = np.linalg.norm(e_pos[live_idx] - p_pos[pi], axis=1)
            assignment[pi] = live_idx[int(dists.argmin())]
    return assignment


class InterceptPursuers(BaseController):
    name = "scripted_pursuers"

    def act(self, view: TeamView) -> np.ndarray:
        vp = view.vmax
        p_pos, p_vel = view.self_pos, view.self_vel
        e_pos, e_vel = view.opp_pos, view.opp_vel
        live = view.opp_alive
        assign = assign_targets(p_pos, e_pos, live)

        n_live = int(live.sum())
        actions = np.zeros((p_pos.shape[0], 3))
        for pi in range(p_pos.shape[0]):
            ej = assign[pi]
            if ej < 0:
                continue
            rel = e_pos[ej] - p_pos[pi]
            d = lead_intercept_dir(rel, e_vel[ej], vp)
            if n_live == 1:
                # pincer: offset each pursuer to a different bearing around the target
                side = 1.0 if pi % 2 == 0 else -1.0
                perp = unit(np.cross(d, np.array([0.0, 0.0, 1.0])))
                if np.linalg.norm(perp) < 1e-6:
                    perp = np.array([1.0, 0.0, 0.0])
                close = np.linalg.norm(rel)
                # blend in a flanking offset that fades as we close the distance
                flank = np.clip(close / 4.0, 0.0, 1.0)
                d = unit(d + side * 0.6 * flank * perp)
            actions[pi] = d
        return np.clip(actions.reshape(-1), -1, 1)
