"""Alternating self-play trainer for both teams, using SB3 PPO.

SB3 is single-agent, so we train two separate PPO policies — one per team — and
have each learn against a *pool* of frozen snapshots of the other team (plus the
scripted baseline as a permanent pool member for a stable learning signal). Each
generation:

    1. pursuer PPO trains for `steps_per_gen` against a pool of evader opponents;
    2. a frozen snapshot of the pursuer policy is added to the pursuer pool;
    3. evader PPO trains against the (now updated) pursuer pool;
    4. a frozen snapshot of the evader policy is added to the evader pool.

Each sub-environment samples a fresh opponent from the shared pool list on every
`reset()` (see ``PoolController``), which decorrelates episodes and prevents the
policy from overfitting a single opponent. Pools reference the live list, so new
snapshots become available to the other side without touching the vec-env.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from ..env.core import TEAM_EVADERS, TEAM_PURSUERS, ArenaConfig, EpisodeConfig
from ..env.gym_env import SelfPlayTeamEnv
from ..scripted.base import BaseController, Controller, RLController
from ..scripted.evaders import FieldEvaders
from ..scripted.pursuers import InterceptPursuers
from ..eval.rollout import run_match


class PoolController(BaseController):
    """Delegates to an opponent sampled from a (growing) pool on each reset."""

    def __init__(self, pool: list[Controller], seed: int = 0, latest_bias: float = 0.5):
        self.pool = pool
        self.rng = np.random.default_rng(seed)
        self.latest_bias = latest_bias
        self.current: Controller = pool[0]
        self.name = "pool"

    def reset(self):
        # with prob latest_bias pick the most recent snapshot, else uniform
        if len(self.pool) > 1 and self.rng.random() < self.latest_bias:
            self.current = self.pool[-1]
        else:
            self.current = self.pool[int(self.rng.integers(len(self.pool)))]
        self.current.reset()

    def act(self, view):
        return self.current.act(view)


def freeze_policy(model: PPO, name: str) -> RLController:
    """Snapshot an SB3 policy into a predict-only controller.

    We serialise through an in-memory buffer rather than ``deepcopy`` because
    PPO's policy holds non-leaf tensors (e.g. the state-dependent ``log_std``)
    that torch refuses to deepcopy. Save/load gives a clean, detached predictor.
    """
    buf = io.BytesIO()
    model.save(buf)
    buf.seek(0)
    frozen = PPO.load(buf, device="cpu")
    return RLController(frozen, deterministic=False, name=name)


@dataclass
class TrainConfig:
    generations: int = 8
    steps_per_gen: int = 40_000
    n_envs: int = 4
    pool_cap: int = 6                 # keep scripted + last (cap-1) snapshots
    eval_episodes: int = 30
    lr: float = 3e-4
    seed: int = 0


def _make_ppo(team: str, pool: list[Controller], cfg: TrainConfig,
              arena: ArenaConfig, episode: EpisodeConfig) -> PPO:
    def factory(k):
        def _f():
            env = SelfPlayTeamEnv(team, PoolController(pool, seed=cfg.seed * 100 + k),
                                  arena=arena, episode=episode, seed=cfg.seed * 100 + k)
            return Monitor(env)
        return _f

    venv = DummyVecEnv([factory(k) for k in range(cfg.n_envs)])
    return PPO(
        "MlpPolicy", venv, verbose=0, device="cpu", seed=cfg.seed,
        n_steps=1024, batch_size=256, n_epochs=10, gamma=0.99, gae_lambda=0.95,
        learning_rate=cfg.lr, ent_coef=0.0, clip_range=0.2,
        policy_kwargs=dict(net_arch=[128, 128]),
    )


def _cap_pool(pool: list[Controller], cap: int):
    # keep index 0 (the scripted baseline) plus the most recent snapshots
    if len(pool) > cap:
        del pool[1:len(pool) - (cap - 1)]


def train_selfplay(cfg: TrainConfig | None = None, arena: ArenaConfig | None = None,
                   episode: EpisodeConfig | None = None, on_gen=None):
    cfg = cfg or TrainConfig()
    arena = arena or ArenaConfig()
    episode = episode or EpisodeConfig()

    pool_p: list[Controller] = [InterceptPursuers()]
    pool_e: list[Controller] = [FieldEvaders()]

    pursuer_model = _make_ppo(TEAM_PURSUERS, pool_e, cfg, arena, episode)
    evader_model = _make_ppo(TEAM_EVADERS, pool_p, cfg, arena, episode)

    history = []
    for gen in range(cfg.generations):
        pursuer_model.learn(cfg.steps_per_gen, reset_num_timesteps=False, progress_bar=False)
        pool_p.append(freeze_policy(pursuer_model, f"rl_pursuer_g{gen}"))
        _cap_pool(pool_p, cfg.pool_cap)

        evader_model.learn(cfg.steps_per_gen, reset_num_timesteps=False, progress_bar=False)
        pool_e.append(freeze_policy(evader_model, f"rl_evader_g{gen}"))
        _cap_pool(pool_e, cfg.pool_cap)

        # snapshot current head policies for evaluation
        cur_p = freeze_policy(pursuer_model, "rl_pursuer")
        cur_e = freeze_policy(evader_model, "rl_evader")
        vs_scripted_e = run_match(cur_p, FieldEvaders(), cfg.eval_episodes, seed=9000, arena=arena, episode=episode)
        scripted_p_vs = run_match(InterceptPursuers(), cur_e, cfg.eval_episodes, seed=9000, arena=arena, episode=episode)
        rec = {
            "gen": gen,
            "rl_pursuer_vs_scripted_evader": vs_scripted_e.capture_rate,
            "scripted_pursuer_vs_rl_evader": scripted_p_vs.capture_rate,
        }
        history.append(rec)
        if on_gen:
            on_gen(rec)

    return pursuer_model, evader_model, history
