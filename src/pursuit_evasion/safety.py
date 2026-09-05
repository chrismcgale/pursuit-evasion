"""Runtime safety filter: a shield below the BT and the policy.

Whatever the behaviour tree or the learned policy commands, the final action for
every agent passes through this filter. It enforces two hard runtime constraints:

  * **Geofence** — an operational keep-in box strictly inside the physical arena.
    If an agent is at/near a face and the command pushes further out, the
    outward component is removed and replaced with an inward correction.
  * **Speed limit** — a soft cap below the agent's aerodynamic terminal speed. If
    the agent is already over the cap, any command component that would increase
    speed is projected out.

The logic is a pure function (``filter_action``) so the C++ runtime can mirror it
exactly and so every branch is unit-testable. ``violations`` reports which
constraints fired this tick — that feeds the "constraint violations" metric and
the V&V story.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env.core import TeamView


@dataclass(frozen=True)
class SafetyConfig:
    geofence_margin: float = 0.75     # keep-in inset from the xy walls (m)
    z_margin: float = 0.75            # keep-in inset from floor/ceiling (m)
    speed_limit_frac: float = 0.98    # cap as a fraction of the agent's v_max
    correction_gain: float = 2.0      # how hard to steer back inside the fence


@dataclass
class Violation:
    geofence: bool = False
    speed: bool = False

    def any(self) -> bool:
        return self.geofence or self.speed


def filter_action(pos: np.ndarray, vel: np.ndarray, action: np.ndarray,
                  vmax: float, half_extent: float, z_min: float, z_max: float,
                  cfg: SafetyConfig) -> tuple[np.ndarray, Violation]:
    """Clamp a single agent's action to the geofence and speed limit.

    Returns the corrected action (each component in [-1, 1]) and which
    constraints were actively enforced this tick.
    """
    a = np.clip(np.asarray(action, dtype=np.float64).copy(), -1.0, 1.0)
    v = Violation()

    lo_xy = -(half_extent - cfg.geofence_margin)
    hi_xy = half_extent - cfg.geofence_margin
    bounds = [(lo_xy, hi_xy), (lo_xy, hi_xy), (z_min + cfg.z_margin, z_max - cfg.z_margin)]

    for ax, (lo, hi) in enumerate(bounds):
        if pos[ax] >= hi:
            # past the far face: kill outward push, steer inward
            over = pos[ax] - hi
            a[ax] = min(a[ax], 0.0) - cfg.correction_gain * over
            v.geofence = True
        elif pos[ax] <= lo:
            under = lo - pos[ax]
            a[ax] = max(a[ax], 0.0) + cfg.correction_gain * under
            v.geofence = True

    # speed cap: if already too fast, remove the speed-increasing component
    speed = float(np.linalg.norm(vel))
    limit = cfg.speed_limit_frac * vmax
    if speed > limit and speed > 1e-6:
        vhat = vel / speed
        along = float(a @ vhat)
        if along > 0:
            a = a - along * vhat
            v.speed = True

    return np.clip(a, -1.0, 1.0), v


class TeamSafetyFilter:
    """Applies ``filter_action`` to every agent of a team's action vector."""

    def __init__(self, cfg: SafetyConfig | None = None):
        self.cfg = cfg or SafetyConfig()
        self.n_geofence = 0
        self.n_speed = 0

    def reset(self):
        self.n_geofence = 0
        self.n_speed = 0

    def apply(self, view: TeamView, team_action: np.ndarray) -> np.ndarray:
        a = np.asarray(team_action, dtype=np.float64).reshape(-1, 3)
        arena = view.arena
        out = np.zeros_like(a)
        for i in range(a.shape[0]):
            if not view.self_alive[i]:
                continue
            corr, viol = filter_action(
                view.self_pos[i], view.self_vel[i], a[i], view.vmax,
                arena.half_extent, arena.z_min, arena.z_max, self.cfg)
            out[i] = corr
            self.n_geofence += int(viol.geofence)
            self.n_speed += int(viol.speed)
        return out.reshape(-1)


class ShieldedController:
    """Wraps any controller so its output always passes through the safety filter."""

    def __init__(self, inner, cfg: SafetyConfig | None = None, name: str | None = None):
        self.inner = inner
        self.filter = TeamSafetyFilter(cfg)
        self.name = name or f"shielded_{getattr(inner, 'name', 'ctrl')}"

    def reset(self):
        self.inner.reset()
        self.filter.reset()

    def act(self, view: TeamView) -> np.ndarray:
        return self.filter.apply(view, self.inner.act(view))
