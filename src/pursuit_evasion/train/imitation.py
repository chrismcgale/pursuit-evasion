"""Imitation learning + RL handoff.

Two-stage recipe the writeup analyses:

  1. **Behaviour cloning** — roll out the BT-gated controller (scripted tactics
     gated with the self-play RL policy) as the expert, collect (obs, action)
     pairs, and regress a fresh MLP policy onto them. This distils the whole
     hand-designed gate into one cheap network.
  2. **RL fine-tune on one hard sub-task** — take the BC weights and PPO-fine-tune
     them on *terminal intercept* only (spawns already inside the endgame, short
     horizon). This is where a learned policy can beat the hand-tuned tactics: the
     last-second closing geometry is exactly what scripted rules handle poorly.

    uv run python -m pursuit_evasion.train.imitation --expert bt \
        --model models/pursuer.zip --out models --bc-epochs 15 --finetune-steps 60000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from ..bt.gating import GatedController
from ..env.core import (TEAM_PURSUERS, ArenaConfig, EpisodeConfig,
                        PursuitEvasionCore)
from ..env.gym_env import SelfPlayTeamEnv
from ..env.observations import build_team_obs
from ..scripted.base import RLController
from ..scripted.evaders import FieldEvaders
from ..scripted.pursuers import InterceptPursuers


def terminal_intercept_starts(core: PursuitEvasionCore, rng: np.random.Generator) -> dict:
    """Spawn each pursuer 3–5 m from an evader: the endgame, not the whole chase."""
    L = core.arena.half_extent
    z_lo, z_hi = core.arena.z_min + 1.0, core.arena.z_max - 1.0
    starts, ev_pos = {}, []
    for n in core.evaders:
        ang = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(3.0, 8.0)
        p = np.array([r * np.cos(ang), r * np.sin(ang), rng.uniform(2, 6)])
        starts[n] = tuple(p)
        ev_pos.append(p)
    for i, n in enumerate(core.pursuers):
        target = ev_pos[i % len(ev_pos)]
        off = rng.normal(0, 1, 3)
        off = off / (np.linalg.norm(off) + 1e-9) * rng.uniform(3.0, 5.0)
        p = target + off
        p[0] = float(np.clip(p[0], -L + 1, L - 1))
        p[1] = float(np.clip(p[1], -L + 1, L - 1))
        p[2] = float(np.clip(p[2], z_lo, z_hi))
        starts[n] = tuple(p)
    return starts


def collect_dataset(expert, n_episodes: int, arena: ArenaConfig, episode: EpisodeConfig,
                    seed0: int = 0):
    """Roll out the expert (pursuers) vs scripted evaders, logging (obs, action)."""
    core = PursuitEvasionCore(arena, episode, seed=seed0)
    opp = FieldEvaders()
    X, Y = [], []
    for e in range(n_episodes):
        pv, ev = core.reset(seed=seed0 + e)
        expert.reset()
        opp.reset()
        while True:
            a = expert.act(pv)
            X.append(build_team_obs(pv))
            Y.append(np.asarray(a, dtype=np.float32).reshape(-1))
            r = core.step(a, opp.act(ev))
            pv, ev = r.pursuer_view, r.evader_view
            if r.terminated or r.truncated:
                break
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


def bc_pretrain(model: PPO, X, Y, epochs: int = 15, lr: float = 1e-3, batch: int = 256):
    """Supervised warm-start of the PPO policy's action head to imitate the expert."""
    policy = model.policy
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    Xt = torch.as_tensor(X)
    Yt = torch.as_tensor(Y)
    n = Xt.shape[0]
    last = float("nan")
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            obs, tgt = Xt[idx], Yt[idx]
            feats = policy.extract_features(obs)
            latent = policy.mlp_extractor.forward_actor(feats)
            pred = policy.action_net(latent)
            loss = torch.nn.functional.mse_loss(pred, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        last = tot / n
        print(f"[bc] epoch {ep+1}/{epochs}  mse={last:.4f}")
    return last


def _make_terminal_env(arena, seed=0):
    ep = EpisodeConfig(max_steps=120)   # short endgame horizon

    def factory():
        env = SelfPlayTeamEnv(TEAM_PURSUERS, FieldEvaders(), arena=arena, episode=ep,
                              seed=seed, starts_fn=terminal_intercept_starts)
        return Monitor(env)

    return DummyVecEnv([factory])


def main(argv=None):
    p = argparse.ArgumentParser(description="BC from BT + PPO fine-tune on terminal intercept")
    p.add_argument("--expert", choices=["bt", "scripted"], default="bt")
    p.add_argument("--model", default="models/pursuer.zip", help="RL policy for the BT's RL branch")
    p.add_argument("--out", default="models")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--bc-epochs", type=int, default=15)
    p.add_argument("--finetune-steps", type=int, default=60_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    arena = ArenaConfig()
    episode = EpisodeConfig()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- build the expert to clone -------------------------------------------
    if args.expert == "bt":
        rl = RLController(PPO.load(args.model, device="cpu"), name="rl_pursuer")
        expert = GatedController(TEAM_PURSUERS, InterceptPursuers(), rl)
        print(f"[il] expert = BT-gated(scripted + {args.model})")
    else:
        expert = InterceptPursuers()
        print("[il] expert = scripted intercept")

    # --- fresh PPO policy, BC warm-start --------------------------------------
    boot_env = DummyVecEnv([lambda: Monitor(
        SelfPlayTeamEnv(TEAM_PURSUERS, FieldEvaders(), arena=arena, episode=episode))])
    model = PPO("MlpPolicy", boot_env, device="cpu", seed=args.seed,
                policy_kwargs=dict(net_arch=[128, 128]), n_steps=1024, batch_size=256)

    print(f"[il] collecting {args.episodes} expert episodes...")
    X, Y = collect_dataset(expert, args.episodes, arena, episode, seed0=args.seed)
    print(f"[il] dataset: {X.shape[0]} transitions")
    bc_pretrain(model, X, Y, epochs=args.bc_epochs)
    model.save(out / "pursuer_bc")
    print(f"[il] saved BC policy -> {out}/pursuer_bc.zip")

    # --- PPO fine-tune on the terminal-intercept sub-task ---------------------
    print(f"[il] fine-tuning {args.finetune_steps} steps on terminal intercept...")
    model.set_env(_make_terminal_env(arena, seed=args.seed))
    model.learn(args.finetune_steps, reset_num_timesteps=False, progress_bar=False)
    model.save(out / "pursuer_bc_ft")
    print(f"[il] saved fine-tuned policy -> {out}/pursuer_bc_ft.zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
