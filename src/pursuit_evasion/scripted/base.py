"""Controller interface shared by scripted policies, the RL policy, and the BT.

A Controller maps a team's ``TeamView`` to a flat action vector of shape
(n_self * 3,) with each component in [-1, 1] (per-agent 3D control force).
This uniform interface is what lets the self-play trainer, the tournament, and
the behaviour-tree gate all treat "how a team acts" as a single swappable object.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from ..env.core import TeamView
from ..env.observations import build_team_obs


class Controller(Protocol):
    name: str

    def act(self, view: TeamView) -> np.ndarray: ...

    def reset(self) -> None: ...


class BaseController:
    name = "base"

    def reset(self) -> None:  # most controllers are stateless
        pass


def unit(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > eps else np.zeros_like(v)


class RLController(BaseController):
    """Wraps a trained SB3 policy (or any obj with ``predict``)."""

    def __init__(self, model, deterministic: bool = True, name: str = "rl"):
        self.model = model
        self.deterministic = deterministic
        self.name = name

    def act(self, view: TeamView) -> np.ndarray:
        obs = build_team_obs(view)
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return np.asarray(action, dtype=np.float64).reshape(-1)


class ZeroController(BaseController):
    name = "zero"

    def __init__(self, n_self: int):
        self.n_self = n_self

    def act(self, view: TeamView) -> np.ndarray:
        return np.zeros(self.n_self * 3)


class NoisyController(BaseController):
    """Random-walk controller, useful as a self-play cold-start opponent."""

    name = "noisy"

    def __init__(self, n_self: int, seed: int = 0, scale: float = 0.6):
        self.n_self = n_self
        self.rng = np.random.default_rng(seed)
        self.scale = scale

    def act(self, view: TeamView) -> np.ndarray:
        return np.clip(self.rng.normal(0, self.scale, self.n_self * 3), -1, 1)
