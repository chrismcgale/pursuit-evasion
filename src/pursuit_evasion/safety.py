"""Runtime safety filter: a shield below the BT and the policy.

Whatever the behaviour tree or the learned policy commands, the final action for
every agent passes through this filter. It enforces two hard runtime constraints:

  * **Geofence** — an operational keep-in box strictly inside the physical arena.
    It acts on the *predicted* position ``pos + brake_horizon_s * vel`` as well
    as the current one: an agent closing on a face fast enough to cross it
    within the horizon has its outward command removed and an inward correction
    applied *before* it arrives. A position-only fence fires a full control
    period late (0.45-0.5 m of travel at these speeds) and then still needs
    braking distance, which together exceed the 0.75 m margin (measured: 0.77 m
    overruns and 11/200 wall contacts in tag with the position-only fence). The fenced axes
    also keep their authority through the downstream unit-norm clip
    (``apply_dynamics``): if the corrected command is longer than 1, the
    *unfenced* axes are shrunk, never the inward correction.
  * **Speed limit** — a soft cap below the agent's aerodynamic terminal speed. If
    the agent is already over the cap, any command component that would increase
    speed is projected out.
  * **Own-team separation** (off by default, ``separation_m > 0``) — keep live
    teammates at least ``separation_m`` apart, centre to centre. Brake-aware
    like the fence: it acts on the predicted relative position
    ``(p_i + h v_i) - (p_j + h v_j)`` as well as the current one, removes the
    command component toward the teammate and pushes away in proportion to the
    shortfall. The push keeps its authority through the norm clip (the
    component across the push direction is shrunk instead). It runs before the
    fence, so near a wall the fence wins. Each agent applies it independently,
    so a converging pair both turn away. Opponents are not included: capture
    commands near-contact, and separating from opponents conflicts with the
    objective (see docs/REVIEW.md #12).

The logic is a pure function (``filter_action``) so the C++ runtime can mirror it
exactly and so every branch is unit-testable. ``violations`` reports which
constraints fired this tick — that feeds the "constraint violations" metric and
the V&V story.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .env.core import TeamView


@dataclass(frozen=True)
class SafetyConfig:
    geofence_margin: float = 0.75     # keep-in inset from the xy walls (m)
    z_margin: float = 0.75            # keep-in inset from floor/ceiling (m)
    speed_limit_frac: float = 0.98    # cap as a fraction of the agent's v_max
    correction_gain: float = 2.0      # how hard to steer back inside the fence
    # Look-ahead for the fence (s): one control period of travel plus braking
    # time v/2a (tag: 0.10 + 0.06; defence: 0.06 + 0.06). 0 = position-only.
    # Swept on tag bt_gated_safe (n=200, paired vs the old fence):
    #   h=0     win 0.460  max overrun 0.76 m  wall contacts 3 (authority only)
    #   h=0.10  win 0.460             0.57 m                0
    #   h=0.15  win 0.425 (n.s.)      0.23 m                0   <- shipped
    #   h=0.25  win 0.375 (sig.)      0.24 m                0
    # Defence games: identical win rate at every h, overrun <= 0.03 m.
    brake_horizon_s: float = 0.15
    # Shrink unfenced axes (not the inward correction) when |cmd| > 1, so the
    # downstream norm clip cannot dilute the fence. False = legacy behaviour.
    # Free: at h=0.10 it is worth +0.06 tag win over the same fence without it.
    keep_fence_authority: bool = True
    # Minimum centre-to-centre distance between live teammates (m). 0 = off.
    # Uses brake_horizon_s for its look-ahead and correction_gain for its push.
    separation_m: float = 0.0


@dataclass
class Violation:
    geofence: bool = False
    speed: bool = False
    separation: bool = False

    def any(self) -> bool:
        return self.geofence or self.speed or self.separation


def separate(pos, vel, action, mates, cfg: SafetyConfig, rank: int = 0):
    """Own-team separation; returns (action, fired). Pure, mirrored in C++.

    ``mates`` is a sequence of ``(pos, vel, index)`` for the OTHER live agents
    of the same team; ``rank`` is this agent's index (it only breaks the tie
    when two agents sit exactly on top of each other). Scalar arithmetic
    throughout, in the order the C++ mirror evaluates it.
    """
    a = [float(action[0]), float(action[1]), float(action[2])]
    h = cfg.brake_horizon_s
    sep = cfg.separation_m
    fired = False
    for q, w, j in mates:
        r = [(pos[k] + h * vel[k]) - (q[k] + h * w[k]) for k in range(3)]
        r0 = [pos[k] - q[k] for k in range(3)]
        d = math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
        d0 = math.sqrt(r0[0] * r0[0] + r0[1] * r0[1] + r0[2] * r0[2])
        dmin = min(d, d0)
        if dmin >= sep:
            continue
        if d > 1e-9:
            n = [r[0] / d, r[1] / d, r[2] / d]
        elif d0 > 1e-9:
            n = [r0[0] / d0, r0[1] / d0, r0[2] / d0]
        else:
            n = [0.0, 0.0, 1.0 if rank < j else -1.0]
        along = a[0] * n[0] + a[1] * n[1] + a[2] * n[2]
        if along < 0:
            a = [a[k] - along * n[k] for k in range(3)]
        c = cfg.correction_gain * (sep - dmin)
        a = [a[k] + c * n[k] for k in range(3)]
        # keep the push through the downstream norm clip: shrink the rest
        n2 = a[0] * a[0] + a[1] * a[1] + a[2] * a[2]
        if n2 > 1.0:
            p = a[0] * n[0] + a[1] * n[1] + a[2] * n[2]
            if p >= 1.0:
                a = [n[0], n[1], n[2]]
            else:
                perp = [a[k] - p * n[k] for k in range(3)]
                pn = math.sqrt(perp[0] * perp[0] + perp[1] * perp[1] + perp[2] * perp[2])
                if pn > 1e-12:
                    kk = math.sqrt(1.0 - p * p) / pn
                    a = [p * n[k] + kk * perp[k] for k in range(3)]
        fired = True
    return np.array(a, dtype=np.float64), fired


def filter_action(pos: np.ndarray, vel: np.ndarray, action: np.ndarray,
                  vmax: float, half_extent: float, z_min: float, z_max: float,
                  cfg: SafetyConfig, mates=(), rank: int = 0
                  ) -> tuple[np.ndarray, Violation]:
    """Clamp a single agent's action to the geofence and speed limit.

    ``mates``/``rank`` feed the own-team separation constraint (``separate``),
    which only runs when ``cfg.separation_m > 0``. Returns the corrected action
    (each component in [-1, 1]) and which constraints were actively enforced
    this tick.
    """
    a = np.clip(np.asarray(action, dtype=np.float64).copy(), -1.0, 1.0)
    v = Violation()
    if cfg.separation_m > 0 and len(mates) > 0:
        a, v.separation = separate(pos, vel, a, mates, cfg, rank)

    lo_xy = -(half_extent - cfg.geofence_margin)
    hi_xy = half_extent - cfg.geofence_margin
    bounds = [(lo_xy, hi_xy), (lo_xy, hi_xy), (z_min + cfg.z_margin, z_max - cfg.z_margin)]

    fenced = [False, False, False]
    for ax, (lo, hi) in enumerate(bounds):
        pred = pos[ax] + cfg.brake_horizon_s * vel[ax]
        if pos[ax] >= hi:
            # past the far face: kill outward push, steer inward
            over = pos[ax] - hi
            a[ax] = min(a[ax], 0.0) - cfg.correction_gain * over
            fenced[ax] = True
        elif pos[ax] <= lo:
            under = lo - pos[ax]
            a[ax] = max(a[ax], 0.0) + cfg.correction_gain * under
            fenced[ax] = True
        elif pred > hi:
            # inside, but will cross within the horizon: brake now
            a[ax] = min(a[ax], 0.0) - cfg.correction_gain * (pred - hi)
            fenced[ax] = True
        elif pred < lo:
            a[ax] = max(a[ax], 0.0) + cfg.correction_gain * (lo - pred)
            fenced[ax] = True
    v.geofence = fenced[0] or fenced[1] or fenced[2]

    # speed cap: if already too fast, remove the speed-increasing component
    speed = float(np.linalg.norm(vel))
    limit = cfg.speed_limit_frac * vmax
    if speed > limit and speed > 1e-6:
        vhat = vel / speed
        along = float(a @ vhat)
        if along > 0:
            a = a - along * vhat
            v.speed = True

    a = np.clip(a, -1.0, 1.0)
    if cfg.keep_fence_authority and v.geofence:
        # explicit sums (not a @ a) so the C++ mirror computes the same bits
        n2 = a[0] * a[0] + a[1] * a[1] + a[2] * a[2]
        if n2 > 1.0:
            f2 = 0.0
            for ax in range(3):
                if fenced[ax]:
                    f2 += a[ax] * a[ax]
            if f2 >= 1.0:
                # the correction alone saturates: drop everything else
                f = math.sqrt(f2)
                for ax in range(3):
                    a[ax] = a[ax] / f if fenced[ax] else 0.0
            else:
                k = math.sqrt((1.0 - f2) / (n2 - f2))
                for ax in range(3):
                    if not fenced[ax]:
                        a[ax] = a[ax] * k
    return np.clip(a, -1.0, 1.0), v


class TeamSafetyFilter:
    """Applies ``filter_action`` to every agent of a team's action vector."""

    def __init__(self, cfg: SafetyConfig | None = None):
        self.cfg = cfg or SafetyConfig()
        self.n_geofence = 0
        self.n_speed = 0
        self.n_separation = 0

    def reset(self):
        self.n_geofence = 0
        self.n_speed = 0
        self.n_separation = 0

    def apply(self, view: TeamView, team_action: np.ndarray) -> np.ndarray:
        a = np.asarray(team_action, dtype=np.float64).reshape(-1, 3)
        arena = view.arena
        out = np.zeros_like(a)
        for i in range(a.shape[0]):
            if not view.self_alive[i]:
                continue
            mates = ()
            if self.cfg.separation_m > 0:
                mates = [(view.self_pos[j], view.self_vel[j], j)
                         for j in range(a.shape[0]) if j != i and view.self_alive[j]]
            corr, viol = filter_action(
                view.self_pos[i], view.self_vel[i], a[i], view.vmax,
                arena.half_extent, arena.z_min, arena.z_max, self.cfg, mates, i)
            out[i] = corr
            self.n_geofence += int(viol.geofence)
            self.n_speed += int(viol.speed)
            self.n_separation += int(viol.separation)
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
