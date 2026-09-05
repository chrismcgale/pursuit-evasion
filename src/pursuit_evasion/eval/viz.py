"""Render one episode to an mp4 for eyeballing behaviour.

    uv run pe-viz --controller bt --model models/pursuer.zip --out rollout.mp4
    uv run pe-viz --game assault --controller scripted --out assault.mp4

Pursuer-slot agents are red, evader-slot blue; a neutralised agent stops moving.
In the objective games the yellow sphere is the defended asset — attackers are
missiles trying to reach it, defenders are drones trying to stop them.

Offscreen rendering uses MuJoCo's renderer (EGL by default on this machine); set
MUJOCO_GL=egl|glfw|osmesa if your setup needs a different backend.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402

from ..bt.gating import GatedController  # noqa: E402
from ..env.core import TEAM_PURSUERS, PursuitEvasionCore  # noqa: E402
from ..env.games import GAME_KEYS, make_game  # noqa: E402
from ..safety import ShieldedController  # noqa: E402
from ..scripted import default_controllers  # noqa: E402


def _pursuer_controller(kind: str, model_path: str | None, game_key: str):
    scripted, opponent = default_controllers(game_key)
    if kind == "scripted":
        return scripted, opponent
    from stable_baselines3 import PPO

    from ..scripted.base import RLController
    rl = RLController(PPO.load(model_path, device="cpu"), name="rl")
    if kind == "rl":
        return rl, opponent
    gate = GatedController(TEAM_PURSUERS, scripted, rl, game=game_key)
    return (ShieldedController(gate) if kind == "bt_safe" else gate), opponent


def _camera(arena, game):
    cam = mujoco.MjvCamera()
    lookat = game.asset.start if game.has_asset else (0.0, 0.0, 3.0)
    cam.lookat[:] = [lookat[0], lookat[1], max(lookat[2], 3.0)]
    cam.distance = 2.6 * arena.half_extent
    cam.azimuth = 90
    cam.elevation = -35
    return cam


def _outcome(info, game):
    if not game.has_asset:
        return "CAPTURED both" if info["all_captured"] else f"{info['n_captured']}/2 tagged"
    if info["breach"]:
        return "BREACHED — asset lost"
    who = "convoy arrived" if info["asset_arrived"] else "raid stopped"
    return (f"DEFENDED ({who}); {info['n_captured']}/2 attackers down "
            f"({info['n_spent']} burned out)")


def render_episode(kind="bt", model_path=None, out="rollout.mp4", seed=0,
                   width=1280, height=720, fps=30, game="tag"):
    import imageio

    game = make_game(game) if isinstance(game, str) else game
    core = PursuitEvasionCore(seed=seed, game=game)
    pursuer, evader = _pursuer_controller(kind, model_path, game.key)
    pv, ev = core.reset(seed=seed)
    pursuer.reset()
    evader.reset()

    renderer = mujoco.Renderer(core.model, height=height, width=width)
    cam = _camera(core.arena, game)
    frames = []
    while True:
        pa, ea = pursuer.act(pv), evader.act(ev)
        r = core.step(pa, ea)
        renderer.update_scene(core.data, cam)
        frames.append(renderer.render())
        pv, ev = r.pursuer_view, r.evader_view
        if r.terminated or r.truncated:
            break
    renderer.close()
    imageio.mimsave(out, frames, fps=fps, macro_block_size=None)
    print(f"[viz] {game.key}/{kind}: {len(frames)} frames, {_outcome(r.info, game)} "
          f"in {r.info['steps']} steps -> {out}")
    if isinstance(pursuer, (GatedController,)) or hasattr(pursuer, "inner"):
        gate = pursuer.inner if hasattr(pursuer, "inner") else pursuer
        if hasattr(gate, "mode_counts"):
            print(f"[viz] gate mode usage ({gate.profile}): {gate.mode_counts}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Render a pursuit-evasion rollout to mp4")
    p.add_argument("--game", default="tag", choices=list(GAME_KEYS))
    p.add_argument("--controller", choices=["scripted", "rl", "bt", "bt_safe"], default="bt")
    p.add_argument("--model", default="models/pursuer.zip")
    p.add_argument("--out", default="rollout.mp4")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    render_episode(args.controller, args.model, args.out, args.seed, game=args.game)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
