"""Flat observation encoding for a team, shared by the RL policy and the env.

A team jointly controls its N agents, so the observation concatenates an
egocentric block per agent. Everything is normalised to roughly [-1, 1] so the
policy sees well-scaled inputs regardless of arena size or speed limits.

Games with a defended asset append an **objective block** (per agent: bearing to
the asset, its relative velocity, own remaining fuel; plus one global flag).
Games without one — ``tag`` — are byte-identical to the original 43-dim layout,
so previously trained policies still load.
"""
from __future__ import annotations

import numpy as np

from .core import TeamView


def obs_dim(n_self: int, n_opp: int, has_asset: bool = False) -> int:
    # time(1) + per-self [pos3, vel3, alive1] + per-(self,opp) [relpos3, relvel3, alive1]
    base = 1 + n_self * 7 + n_self * n_opp * 7
    # objective: per-self [rel_asset3, rel_asset_vel3, fuel1] + global [asset_live1]
    return base + (n_self * 7 + 1 if has_asset else 0)


def act_dim(n_self: int) -> int:
    return n_self * 3


def build_team_obs(view: TeamView) -> np.ndarray:
    L = view.arena.half_extent
    vmax = max(view.vmax, 1e-6)
    ovmax = max(view.opp_vmax, 1e-6)
    n_self = view.self_pos.shape[0]
    n_opp = view.opp_pos.shape[0]

    out = [np.array([view.time_frac * 2 - 1], dtype=np.float32)]
    for i in range(n_self):
        out.append((view.self_pos[i] / L).astype(np.float32))
        out.append((view.self_vel[i] / vmax).astype(np.float32))
        out.append(np.array([1.0 if view.self_alive[i] else -1.0], dtype=np.float32))
    for i in range(n_self):
        for j in range(n_opp):
            rel_p = (view.opp_pos[j] - view.self_pos[i]) / (2 * L)
            rel_v = (view.opp_vel[j] - view.self_vel[i]) / ovmax
            out.append(rel_p.astype(np.float32))
            out.append(rel_v.astype(np.float32))
            out.append(np.array([1.0 if view.opp_alive[j] else -1.0], dtype=np.float32))
    if view.has_asset:
        fuel = view.self_fuel if view.self_fuel is not None else np.ones(n_self)
        for i in range(n_self):
            rel_p = (view.asset_pos - view.self_pos[i]) / (2 * L)
            rel_v = (view.asset_vel - view.self_vel[i]) / vmax
            out.append(rel_p.astype(np.float32))
            out.append(rel_v.astype(np.float32))
            out.append(np.array([float(fuel[i]) * 2 - 1], dtype=np.float32))
        out.append(np.array([1.0], dtype=np.float32))   # asset intact this tick
    return np.clip(np.concatenate(out), -5.0, 5.0)
