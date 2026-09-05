"""Record one episode in enough detail to explain *why* the gate did what it did.

The mp4 in ``viz.py`` shows what happened; this shows the decision. Per control
step it captures the geometry, every agent's gate branch and mode, both
controllers' candidate actions, and the safety filter's correction — so the
viewer can answer "the tree handed this tick to the policy: which predicate
fired, what did the two controllers disagree about, and did the shield veto it?"

Output is a plain JSON dict consumed by ``eval.explorer``.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path

import numpy as np

from ..bt.gating import BRANCH_READS, _PREDICATES, GatedController
from ..env.core import TEAM_PURSUERS, PursuitEvasionCore
from ..env.games import GAME_KEYS, make_game
from ..safety import ShieldedController
from ..scripted import default_controllers
from ..scripted.base import RLController


def _num(x):
    """JSON-safe: inf/nan are meaningful here (no intercept solution) but illegal."""
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    v = float(x)
    return None if (math.isinf(v) or math.isnan(v)) else round(v, 4)


def _vec(v):
    return [round(float(c), 4) for c in np.asarray(v).reshape(-1)]


def _controller(kind: str, model_path: str | None, game_key: str):
    scripted, opponent = default_controllers(game_key)
    if kind == "scripted":
        return scripted, opponent
    rl = RLController(__import__("stable_baselines3").PPO.load(model_path, device="cpu"),
                      name="rl")
    if kind == "rl":
        return rl, opponent
    gate = GatedController(TEAM_PURSUERS, scripted, rl, game=game_key)
    return (ShieldedController(gate) if kind == "bt_safe" else gate), opponent


def _gate_of(ctrl):
    inner = getattr(ctrl, "inner", ctrl)
    return inner if isinstance(inner, GatedController) else None


def record(game_key="assault", kind="bt_safe", model_path=None, seed=0) -> dict:
    game = make_game(game_key)
    core = PursuitEvasionCore(seed=seed, game=game)
    ctrl, opponent = _controller(kind, model_path, game_key)
    gate = _gate_of(ctrl)
    if gate is not None:
        gate.trace = True

    pv, ev = core.reset(seed=seed)
    ctrl.reset()
    opponent.reset()
    if gate is not None:
        gate.trace = True                      # reset() must not clear the flag

    shield = getattr(ctrl, "filter", None)
    steps = []
    while True:
        before = (shield.n_geofence, shield.n_speed) if shield else (0, 0)
        applied = np.asarray(ctrl.act(pv), dtype=np.float64).reshape(-1, 3)
        # The filter only keeps cumulative counters, so diff them to attribute
        # corrections to this tick. The viewer pairs that with applied-vs-chosen
        # to show the shield overruling whichever controller the gate picked.
        fired = ((shield.n_geofence - before[0], shield.n_speed - before[1])
                 if shield else (0, 0))
        decisions = []
        for d in (gate.last_decisions if gate is not None else []):
            f = d["features"]
            decisions.append({
                "agent": int(d["agent"]),
                "mode": d["mode"],
                "branch": d["branch"],
                "scripted": _vec(d["scripted"]),
                "rl": _vec(d["rl"]),
                "evals": {b["name"]: b["passed"] for b in d["evals"]},
                "features": {k: _num(v) for k, v in dataclasses.asdict(f).items()},
            })
        r = core.step(applied.reshape(-1), opponent.act(ev))
        info = r.info
        steps.append({
            "t": int(info["steps"]),
            "self": [_vec(p) for p in pv.self_pos],
            "opp": [_vec(p) for p in pv.opp_pos],
            "self_vel": [_vec(v) for v in pv.self_vel],
            "opp_vel": [_vec(v) for v in pv.opp_vel],
            "self_alive": [bool(a) for a in pv.self_alive],
            "opp_alive": [bool(a) for a in pv.opp_alive],
            "applied": [_vec(a) for a in applied],
            "shield": {"geofence": fired[0], "speed": fired[1]},
            "decisions": decisions,
            "min_dist": _num(info["min_dist"]),
            "asset_dist": _num(info["asset_dist"]),
            "n_captured": int(info["n_captured"]),
        })
        pv, ev = r.pursuer_view, r.evader_view
        if r.terminated or r.truncated:
            break

    thr = gate.thresholds if gate is not None else None
    profile = gate.profile if gate is not None else None
    return {
        "game": game_key,
        "controller": kind,
        "seed": seed,
        "profile": profile,
        # Branches in Selector order, each with the terms it reads, so the viewer
        # can render the live comparison without knowing any predicate itself.
        "branches": ([{"name": n, "mode": m, "reads": BRANCH_READS[profile][n]}
                      for n, _, m in _PREDICATES[profile]()]
                     + [{"name": "default", "mode": "scripted", "reads": []}]) if profile else [],
        "thresholds": dataclasses.asdict(thr) if thr is not None else {},
        "arena": {"half_extent": core.arena.half_extent,
                  "z_min": core.arena.z_min, "z_max": core.arena.z_max},
        "capture_radius": core.ep.capture_radius,
        "asset": {"pos": _vec(core.asset_position()), "radius": game.asset.radius}
                 if game.has_asset else None,
        "dt": core.dt,                          # seconds per *control* step
        "outcome": {"pursuer_win": bool(r.info["pursuer_win"]),
                    "breach": bool(r.info["breach"]),
                    "n_captured": int(r.info["n_captured"]),
                    "steps": int(r.info["steps"])},
        "mode_counts": dict(gate.mode_counts) if gate is not None else {},
        "steps": steps,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--game", default="assault", choices=list(GAME_KEYS))
    p.add_argument("--controller", default="bt_safe",
                   choices=["scripted", "rl", "bt", "bt_safe"])
    p.add_argument("--model", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    model = args.model
    if model is None:
        model = ("models/pursuer_dagger.zip" if args.game == "tag"
                 else f"models/{args.game}_dagger.zip")
    data = record(args.game, args.controller, model, args.seed)
    out = args.out or Path(f"results/trace_{args.game}_{args.controller}_{args.seed}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data))
    print(f"[trace] {args.game}/{args.controller} seed={args.seed}: "
          f"{len(data['steps'])} steps, modes={data['mode_counts']} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
