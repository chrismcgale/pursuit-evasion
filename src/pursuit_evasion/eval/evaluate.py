"""Tournament: the "when each wins" experiment.

Two batteries, all controllers on identical seeds:

  1. **Full chase** (random open-arena starts): scripted / RL / BC / BT-gated /
     BT-gated+safety pursuers vs a fixed scripted evader.
  2. **Terminal intercept** (endgame starts): the sub-task the RL fine-tune targets,
     where scripted / RL / BC / BC→fine-tune are compared head-to-head.

Writes results.json + failure logs and prints a table. Missing model files are
skipped gracefully so this runs after `pe-train` alone or after `imitation` too.

    uv run pe-eval --models models --episodes 200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO

from ..bt.gating import GatedController
from ..env.core import TEAM_PURSUERS, ArenaConfig, EpisodeConfig
from ..safety import ShieldedController
from ..scripted.base import RLController
from ..scripted.evaders import FieldEvaders
from ..scripted.pursuers import InterceptPursuers
from ..train.imitation import terminal_intercept_starts
from .scenarios import BatchReport, run_batch


def _fmt(x, nan="—"):
    import math
    return nan if (x is None or (isinstance(x, float) and math.isnan(x))) else x


def _md_table(section: dict, title: str) -> str:
    rows = ["| controller | win rate | mean captures | steps→win | min sep (m) | geo/spd viol |",
            "|---|---|---|---|---|---|"]
    for name, a in section.items():
        s2w = a["mean_steps_to_win"]
        s2w = "—" if (isinstance(s2w, float) and s2w != s2w) else f"{s2w:.0f}"
        rows.append(f"| `{name}` | {a['win_rate']:.2f} | {a['mean_captures']:.2f} | "
                    f"{s2w} | {a['mean_min_separation']:.2f} | "
                    f"{a['geofence_viol']}/{a['speed_viol']} |")
    return f"**{title}**\n\n" + "\n".join(rows) + "\n"


def write_writeup_tables(results: dict, writeup: Path):
    """Replace the AUTOFILL marker in the writeup with generated results tables."""
    if not writeup.exists():
        return
    block = (_md_table(results["full_chase"], "Full chase (random open-arena starts)")
             + "\n" + _md_table(results["terminal_intercept"], "Terminal intercept (endgame starts)"))
    text = writeup.read_text()
    marker = "<!-- RESULTS:AUTOFILL -->"
    if marker in text:
        head = text[:text.index(marker)] + marker + "\n\n"
        rest = text[text.index(marker) + len(marker):]
        # replace everything up to the next section heading (idempotent re-runs)
        idx = rest.find("\n## ")
        tail = rest[idx + 1:] if idx != -1 else ""
        writeup.write_text(head + block + "\n" + tail)


def _load(path: Path):
    return RLController(PPO.load(str(path), device="cpu"), name=path.stem) if path.exists() else None


def build_pursuer_controllers(models: Path):
    """Assemble every available pursuer controller keyed by name.

    The BT gates the *best available* learned policy as its RL branch
    (fine-tuned > BC > self-play), since gating a broken policy would only ever
    hurt. Raw self-play is kept as a separate baseline so the writeup can show it.
    """
    ctrls = {"scripted": InterceptPursuers()}
    rl = _load(models / "pursuer.zip")
    if rl is not None:
        ctrls["rl_selfplay"] = rl
    bc = _load(models / "pursuer_bc.zip")
    if bc is not None:
        ctrls["bc_distill"] = bc          # imitation of the (scripted) expert
    bcft = _load(models / "pursuer_bc_ft.zip")
    if bcft is not None:
        ctrls["bc_finetuned"] = bcft      # + PPO fine-tune on terminal intercept
    dagger = _load(models / "pursuer_dagger.zip")
    if dagger is not None:
        ctrls["dagger"] = dagger          # DAgger: BC + expert relabeling of learner states

    # best policy for the gate's RL branch (DAgger fixes BC's covariate shift)
    best = dagger or bcft or bc or rl
    if best is not None:
        gate = GatedController(TEAM_PURSUERS, InterceptPursuers(), best)
        gate.name = f"bt_gated({best.name})"
        ctrls["bt_gated"] = gate
        ctrls["bt_gated_safe"] = ShieldedController(gate)
    return ctrls


def run_tournament(models: Path, episodes: int, seed0: int, out: Path):
    arena, episode = ArenaConfig(), EpisodeConfig()
    ctrls = build_pursuer_controllers(models)
    results = {"full_chase": {}, "terminal_intercept": {}}

    print(f"\n=== FULL CHASE (random starts, {episodes} eps) — pursuer vs scripted evader ===")
    full_reports = {}
    for name, ctrl in ctrls.items():
        rep = run_batch(ctrl, FieldEvaders(), episodes, seed0, arena, episode, label=name)
        full_reports[name] = rep
        results["full_chase"][name] = rep.aggregate()
        print("  " + rep.row())

    print(f"\n=== TERMINAL INTERCEPT (endgame starts, {episodes} eps) ===")
    # only policy-like controllers + scripted are interesting here
    tkeys = [k for k in ("scripted", "rl_selfplay", "bc_distill", "dagger", "bt_gated") if k in ctrls]
    for name in tkeys:
        rep = run_batch(ctrls[name], FieldEvaders(), episodes, seed0, arena, episode,
                        label=name, starts_fn=terminal_intercept_starts)
        results["terminal_intercept"][name] = rep.aggregate()
        print("  " + rep.row())

    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    # failure logs for the headline BT-gated controller (replayable)
    if "bt_gated" in full_reports:
        nf = full_reports["bt_gated"].save_failures(out / "failures_bt_gated.json")
        print(f"\n[eval] saved {nf} replayable BT-gated failures -> {out}/failures_bt_gated.json")
    write_writeup_tables(results, Path("writeup/when-each-wins.md"))
    print(f"[eval] wrote {out}/results.json and refreshed writeup tables")
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description="Pursuit-evasion tournament")
    p.add_argument("--models", default="models")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--out", default="results")
    args = p.parse_args(argv)
    run_tournament(Path(args.models), args.episodes, args.seed, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
