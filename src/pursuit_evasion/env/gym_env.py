"""Single-team gymnasium view of the 2v2 core, for training with SB3.

SB3 is single-agent, so we expose exactly one team ("pursuers" or "evaders") as
the learner and drive the opposing team with a fixed ``Controller`` (scripted,
a frozen policy snapshot, or a BT-gated controller). Swapping that opponent is
the whole basis of the self-play loop in ``train.selfplay``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .core import (TEAM_EVADERS, TEAM_PURSUERS, ArenaConfig, EpisodeConfig,
                   PursuitEvasionCore)

if TYPE_CHECKING:  # avoids a scripted.base <-> env circular import at runtime
    from ..scripted.base import Controller
from .observations import act_dim, build_team_obs, obs_dim


class SelfPlayTeamEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, team: str, opponent: "Controller",
                 arena: ArenaConfig | None = None, episode: EpisodeConfig | None = None,
                 seed: int | None = None, starts_fn=None, game=None):
        super().__init__()
        assert team in (TEAM_PURSUERS, TEAM_EVADERS)
        self.team = team
        self.opponent = opponent
        # optional curriculum hook: starts_fn(core, rng) -> starts dict, used to
        # spawn specific configurations (e.g. the terminal-intercept sub-task).
        self.starts_fn = starts_fn
        self.core = PursuitEvasionCore(arena, episode, seed=seed, game=game)
        n_self = len(self.core.pursuers) if team == TEAM_PURSUERS else len(self.core.evaders)
        n_opp = len(self.core.evaders) if team == TEAM_PURSUERS else len(self.core.pursuers)
        self.observation_space = spaces.Box(
            -5.0, 5.0, (obs_dim(n_self, n_opp, self.core.game.has_asset),), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (act_dim(n_self),), np.float32)
        self._pview = None
        self._eview = None

    def set_opponent(self, opponent: "Controller"):
        self.opponent = opponent

    def _learner_view(self):
        return self._pview if self.team == TEAM_PURSUERS else self._eview

    def _opp_view(self):
        return self._eview if self.team == TEAM_PURSUERS else self._pview

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.starts_fn is not None:
            if seed is not None:
                self.core.rng = np.random.default_rng(seed)
            self._pview, self._eview = self.core.reset_with_starts(
                self.starts_fn(self.core, self.core.rng))
        else:
            self._pview, self._eview = self.core.reset(seed=seed)
        self.opponent.reset()
        return build_team_obs(self._learner_view()), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        opp_action = self.opponent.act(self._opp_view())
        if self.team == TEAM_PURSUERS:
            res = self.core.step(action, opp_action)
            reward = res.pursuer_reward
        else:
            res = self.core.step(opp_action, action)
            reward = res.evader_reward
        self._pview, self._eview = res.pursuer_view, res.evader_view
        obs = build_team_obs(self._learner_view())
        return obs, float(reward), res.terminated, res.truncated, res.info
