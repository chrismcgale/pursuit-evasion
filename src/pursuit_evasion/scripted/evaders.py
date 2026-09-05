"""Scripted evader team controllers.

``FieldEvaders`` steers each evader by a small potential field:
  * strong repulsion from each pursuer (grows as 1/dist), which dominates when
    a pursuer is close;
  * a tangential "juke" component perpendicular to the incoming pursuer, so the
    evader doesn't flee in a straight line (straight flight is exactly what a
    lead-intercept pursuer wants);
  * wall repulsion so it doesn't corner itself (the classic evader failure);
  * mild separation from its ally so the two don't clump and get double-tagged.
"""
from __future__ import annotations

import numpy as np

from ..env.core import TeamView
from .base import BaseController, unit


class FieldEvaders(BaseController):
    name = "scripted_evaders"

    def __init__(self, juke: float = 0.7):
        self.juke = juke

    def act(self, view: TeamView) -> np.ndarray:
        L = view.arena.half_extent
        z_lo, z_hi = view.arena.z_min, view.arena.z_max
        e_pos, e_vel = view.self_pos, view.self_vel
        p_pos = view.opp_pos
        alive = view.self_alive
        n_self = e_pos.shape[0]

        actions = np.zeros((n_self, 3))
        for i in range(n_self):
            if not alive[i]:
                continue
            pos = e_pos[i]
            force = np.zeros(3)

            # flee from every pursuer, weighted by proximity
            for pj in range(p_pos.shape[0]):
                to_me = pos - p_pos[pj]
                d = np.linalg.norm(to_me) + 1e-6
                w = 1.0 / (d * d)
                force += w * unit(to_me)
                # tangential juke around the nearest threat
                if d < 5.0:
                    perp = unit(np.cross(to_me, np.array([0.0, 0.0, 1.0])))
                    if np.linalg.norm(perp) < 1e-6:
                        perp = np.array([0.0, 0.0, 1.0])
                    sign = 1.0 if (i + pj) % 2 == 0 else -1.0
                    force += self.juke * w * sign * perp

            # wall repulsion (xy box + z ceiling/floor)
            margin = 2.5
            for ax in (0, 1):
                if pos[ax] > L - margin:
                    force[ax] -= (pos[ax] - (L - margin)) / margin
                elif pos[ax] < -(L - margin):
                    force[ax] += ((-(L - margin)) - pos[ax]) / margin
            if pos[2] > z_hi - margin:
                force[2] -= (pos[2] - (z_hi - margin)) / margin
            elif pos[2] < z_lo + margin:
                force[2] += ((z_lo + margin) - pos[2]) / margin

            # separation from ally
            for k in range(n_self):
                if k == i or not alive[k]:
                    continue
                to_me = pos - e_pos[k]
                d = np.linalg.norm(to_me) + 1e-6
                if d < 3.0:
                    force += 0.4 * unit(to_me) / d

            actions[i] = unit(force)
        return np.clip(actions.reshape(-1), -1, 1)
