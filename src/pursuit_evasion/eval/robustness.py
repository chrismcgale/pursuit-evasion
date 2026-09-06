"""Robustness sweep: which controller survives the link to real hardware?

Every headline number in this repo was measured through a perfect link. This
sweep re-measures them through a degraded one, over the failure modes that
actually occur flying 2-4 tinywhoops off a Vicon volume (``env/link.py``), and
reports where the ordering between scripted, learned and gated **changes**.

The question is not "does everything get worse" — it does. The question is
whether the three controllers degrade at *different rates*, because that is what
decides which one you put on the aircraft. Three concrete hypotheses this is
built to falsify:

  H1  Latency hurts the scripted lead-intercept law more than the policy. The
      law solves for a future intercept point from a velocity estimate; stale
      velocity biases that solve systematically, while a reactive policy is
      merely late. If true, the gate's advantage should GROW with latency.

  H2  Identity swaps hurt the policy more than the tree. A swap keeps every
      position accurate and only exchanges labels, so it is invisible to any
      residual check. The BT's ``contested`` branch reasons about *counts* near
      the agent rather than about which specific body is which, so it may be
      structurally less sensitive.

  H3  The safety filter's advantage in tag is an artefact of perfect state. It
      is fed the same delayed estimate as everything else, and a geofence
      enforced on a stale position can command an excursion rather than prevent
      one. If the filter's +0.27 in tag collapses under latency, that is a
      finding worth more than the original number.

Statistics, because the point estimates in the earlier tables were not enough:
win rates carry Wilson intervals, and every delta-vs-baseline is a **paired**
bootstrap over episodes, since every arm runs the same spawn seeds AND the same
corruption realisations (see ``LinkedController``'s seeding note).

    uv run pe-robust --game assault --axis latency --episodes 150
    uv run pe-robust --game tag --axis all --episodes 200 --plot
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..env.games import make_game
from ..env.link import PERFECT, PRESETS, LinkedController, LinkProfile
from .games import build_controllers
from .scenarios import run_batch

# --------------------------------------------------------------------- stats


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — correct near 0 and 1, where win rates live."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def paired_delta(a: list[bool], b: list[bool], n_boot: int = 10_000,
                 seed: int = 0) -> dict:
    """Bootstrap CI for mean(a) - mean(b) over PAIRED episodes.

    Pairing is the whole reason this is worth doing: the arms share spawns and
    corruption, so the per-episode difference has far less variance than the two
    marginals, and a 5-point delta can be significant at n=150 where the
    unpaired intervals would overlap freely.
    """
    x = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(x)
    if n == 0:
        return {"delta": 0.0, "lo": 0.0, "hi": 0.0, "p_better": 0.5, "n": 0}
    rng = np.random.default_rng(seed)
    boots = x[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return {"delta": float(x.mean()),
            "lo": float(np.percentile(boots, 2.5)),
            "hi": float(np.percentile(boots, 97.5)),
            "p_better": float((boots > 0).mean()),
            "n": n}


# ---------------------------------------------------------------------- axes
# Each axis sweeps ONE link parameter from nominal upward, holding the rest
# perfect, so a curve attributes degradation to a single physical cause. The
# preset ladder below is the opposite: everything at once, which is what the rig
# actually delivers.

AXES: dict[str, dict] = {
    "latency": {
        "field": "obs_latency_s",
        # Sampled densely below 80 ms because that is where the whole transition
        # happens: see `latency_budget` — assault's budget is 76 ms and the
        # smoke test floored every controller at exactly 80.
        "levels": [0.0, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.12],
        "label": "state latency (s)",
        "why": "Vicon capture+solve+network. 8-15 ms is realistic; the tail is "
               "what a congested network or an onboard estimator looks like.",
    },
    "cmd_latency": {
        "field": "cmd_latency_s",
        "levels": [0.0, 0.01, 0.02, 0.04, 0.08],
        "label": "uplink latency (s)",
        "why": "ELRS transport. Separated from state latency because they are "
               "fixed by different purchases. NOTE the quantisation: a radio "
               "delivers whole frames, so this rounds to whole control ticks — "
               "at dt=60 ms, 10/20 ms are 0 ticks and 40/80 ms are both 1. The "
               "measured step (assault 0.71 -> 0.07 at one tick) is therefore "
               "exact, not an artefact: ONE stale command frame at this control "
               "rate forfeits the game. Sub-tick uplink resolution is another "
               "argument for a faster control loop, not for interpolating here.",
    },
    "motor_lag": {
        "field": "motor_tau_s",
        "levels": [0.0, 0.01, 0.02, 0.04, 0.08],
        "label": "motor time constant (s)",
        "why": "0802-0805 whoop motors with a ducted prop sit near 20 ms. This "
               "is a lag, not a delay: it also LOWERS achievable authority.",
    },
    "vel_noise": {
        "field": "vel_noise_mps",
        "levels": [0.0, 0.02, 0.05, 0.10, 0.25],
        "label": "velocity noise (m/s, 1σ)",
        "why": "Finite-differenced mocap velocity. Every gate predicate is "
               "velocity-derived, so this hits the tree where it thinks.",
    },
    "pos_noise": {
        "field": "pos_noise_m",
        "levels": [0.0, 0.01, 0.05, 0.15, 0.40],
        "label": "position noise (m, 1σ)",
        "why": "Vicon is sub-mm, so the interesting levels here are the "
               "UWB/onboard-estimator end of the axis.",
    },
    "swap": {
        "field": "swap_p",
        "levels": [0.0, 0.001, 0.004, 0.01, 0.03],
        "label": "identity-swap onset (per tick)",
        "why": "Two identical 65 mm airframes crossing inside a metre. "
               "Positions stay exact; only the labels are wrong, so nothing "
               "downstream can detect it.",
        "extra": {"swap_burst": 20.0},
    },
    "dropout": {
        "field": "dropout_p",
        "levels": [0.0, 0.01, 0.03, 0.08, 0.20],
        "label": "marker occlusion (per tick)",
        "why": "Body leaves the capture volume or is shadowed; the bridge holds "
               "the last pose.",
        "extra": {"dropout_burst": 5.0},
    },
    "packet_loss": {
        "field": "packet_loss_p",
        "levels": [0.0, 0.01, 0.05, 0.15, 0.30],
        "label": "uplink packet loss",
        "why": "The FC flies its last received command. Cheap to test, and it "
               "is the failure mode a range check is supposed to catch.",
    },
    "sag": {
        "field": "sag_frac_per_min",
        "levels": [0.0, 0.05, 0.15, 0.30, 0.60],
        "label": "thrust lost per minute",
        "why": "A whoop pack sags hard. Authority falls THROUGH the episode, so "
               "this asymmetrically punishes controllers that win late.",
    },
}


def latency_budget(game_key: str) -> float:
    """Seconds of lag that cost you one capture radius of closing travel.

    This is the prediction the latency sweep exists to test, and it needs no
    simulation: two aircraft close at ``v_pursuer + v_evader``, and an intercept
    has to be resolved to within ``capture_radius``, so

        budget = capture_radius / closing_speed

    is the lag at which your estimate of where the opponent *is* has drifted by
    the whole tolerance you had. Beyond it, no controller of any kind can be
    expected to intercept, because the geometry it is solving no longer exists.
    For assault that is 1.4 / 18.5 = 76 ms — which is inside a realistic
    hardware budget, and is the single most important number for anyone
    planning to fly this.
    """
    g = make_game(game_key)
    closing = g.arena.pursuer_dyn.vmax + g.arena.evader_dyn.vmax
    return g.episode.capture_radius / closing


def control_rate_configs(game_key: str, repeats=(1, 2, 3, 5, 8)) -> list[tuple]:
    """Vary the control period itself, holding everything else fixed.

    This is not a link parameter and not noise — it is the discretisation the
    whole project has been measured at, and comparing it to ``latency_budget``
    says it is too coarse to be defensible:

        tag      budget 70 ms   vs   control period 100 ms   (0.7 ticks)
        assault  budget 76 ms   vs   control period  60 ms   (1.3 ticks)

    Tag decides its engagements at a control period *longer than its own
    tolerance*, so the discretisation alone costs more than an entire realistic
    latency budget. That means part of the tuned "scripted wins 40-50%" baseline
    is an artefact of a slow control loop, and a real ground station running at
    50-200 Hz would not reproduce it. Every controller should improve as the
    period shrinks; what matters is whether they improve at the same rate.

    Two controls make the comparison honest:

      * ``max_steps`` scales inversely, so every arm gets the same wall-clock
        episode rather than more chances at a finer step;
      * capture checking is forced to substep rate for ALL levels, because tag
        normally tests capture once per control step — without this, a faster
        control rate would raise the capture rate purely by sampling the
        proximity test more often, which has nothing to do with control.
    """
    g = make_game(game_key)
    base_repeat = g.episode.control_repeat
    base_steps = g.episode.max_steps
    out = []
    for r in repeats:
        ep = replace(g.episode, control_repeat=r,
                     max_steps=int(round(base_steps * base_repeat / r)))
        spec = replace(g, episode=ep, capture_substeps=True)
        out.append((g.arena.timestep * r, ep, spec))
    return out


def axis_profiles(axis: str) -> list[tuple[float, LinkProfile]]:
    spec = AXES[axis]
    out = []
    for lv in spec["levels"]:
        kw = {spec["field"]: lv, "name": f"{axis}={lv:g}"}
        if lv > 0:
            kw.update(spec.get("extra", {}))
        out.append((lv, LinkProfile(**kw)))
    return out


def preset_ladder() -> list[tuple[float, LinkProfile]]:
    return [(i, p) for i, p in enumerate(PRESETS.values())]


# ----------------------------------------------------------------------- run


def sweep(game_key: str, axis: str, episodes: int, seed0: int, models: Path,
          link_seed: int = 4242) -> dict:
    game = make_game(game_key)
    from ..scripted import default_controllers
    _, opponent = default_controllers(game_key)
    dt = game.arena.timestep * game.episode.control_repeat

    if axis == "control_rate":
        levels = [(period, LinkProfile(name=f"dt={period*1000:.0f}ms"), spec)
                  for period, _ep, spec in control_rate_configs(game_key)]
    elif axis == "presets":
        levels = [(lv, p, game) for lv, p in preset_ladder()]
    else:
        levels = [(lv, p, game) for lv, p in axis_profiles(axis)]

    rows = []
    for lv, profile, spec in levels:
        built = build_controllers(models, game_key)
        entry = {"level": lv, "profile": profile.name, "arms": {}}
        for name, ctrl in built.items():
            linked = LinkedController(ctrl, profile, seed=link_seed,
                                      dt=spec.arena.timestep * spec.episode.control_repeat,
                                      name=f"{name}@{profile.name}")
            rep = run_batch(linked, opponent, episodes, seed0, game=spec,
                            label=linked.name)
            wins = [bool(r.win) for r in rep.records]
            agg = rep.aggregate()
            lo, hi = wilson(sum(wins), len(wins))
            entry["arms"][name] = {
                "win_rate": agg["win_rate"], "ci": [lo, hi], "wins": wins,
                "breach_rate": agg["breach_rate"],
                "mean_captures": agg["mean_captures"],
                "geofence_viol": agg["geofence_viol"],
                "speed_viol": agg["speed_viol"],
                "link": linked.stats(),
            }
            print(f"  {profile.name:<22} {name:<16} win={agg['win_rate']:.3f} "
                  f"[{lo:.3f},{hi:.3f}]  swaps={linked.stats()['swaps']} "
                  f"drops={linked.stats()['dropouts']}")
        rows.append(entry)

    # paired deltas against scripted, level by level
    for entry in rows:
        base = entry["arms"].get("scripted")
        if base is None:
            continue
        for name, arm in entry["arms"].items():
            if name == "scripted":
                continue
            arm["vs_scripted"] = paired_delta(arm["wins"], base["wins"])

    return {"game": game_key, "axis": axis, "episodes": episodes,
            "seed0": seed0, "link_seed": link_seed,
            "label": AXES.get(axis, {}).get("label", axis),
            "why": AXES.get(axis, {}).get("why", ""), "rows": rows}


# -------------------------------------------------------------------- output


def markdown(result: dict) -> str:
    names = list(result["rows"][0]["arms"].keys())
    head = f"| {result['label']} | " + " | ".join(names) + " |"
    sep = "|---" * (len(names) + 1) + "|"
    lines = [head, sep]
    for e in result["rows"]:
        cells = []
        for n in names:
            a = e["arms"][n]
            cells.append(f"{a['win_rate']:.2f} <sub>[{a['ci'][0]:.2f},{a['ci'][1]:.2f}]</sub>")
        lines.append(f"| {e['profile']} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Paired deltas vs `scripted` (95% bootstrap CI; ✓ = interval excludes 0):")
    lines.append("")
    lines.append("| level | " + " | ".join(n for n in names if n != "scripted") + " |")
    lines.append("|---" * len([n for n in names if n != "scripted"]) + "|---|")
    for e in result["rows"]:
        cells = []
        for n in names:
            if n == "scripted":
                continue
            d = e["arms"][n].get("vs_scripted")
            if not d:
                cells.append("—")
                continue
            sig = "✓" if (d["lo"] > 0 or d["hi"] < 0) else " "
            cells.append(f"{d['delta']:+.2f} [{d['lo']:+.2f},{d['hi']:+.2f}]{sig}")
        lines.append(f"| {e['profile']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot(result: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(result["rows"][0]["arms"].keys())
    xs = [e["level"] for e in result["rows"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for n in names:
        ys = [e["arms"][n]["win_rate"] for e in result["rows"]]
        lo = [e["arms"][n]["ci"][0] for e in result["rows"]]
        hi = [e["arms"][n]["ci"][1] for e in result["rows"]]
        line, = ax.plot(xs, ys, marker="o", label=n)
        ax.fill_between(xs, lo, hi, alpha=0.15, color=line.get_color())
    if result["axis"] in ("latency", "cmd_latency"):
        b = latency_budget(result["game"])
        if b <= max(xs):
            ax.axvline(b, color="crimson", ls="--", lw=1)
            ax.annotate(f"capture radius / closing speed = {b*1000:.0f} ms",
                        (b, 0.94), xycoords=("data", "axes fraction"),
                        rotation=90, va="top", ha="right", fontsize=7.5,
                        color="crimson")
    ax.set_xlabel(result["label"])
    ax.set_ylabel("win rate (defender / pursuer slot)")
    ax.set_title(f"{result['game']} — degradation under {result['axis']} "
                 f"(n={result['episodes']}/point, 95% Wilson)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description="Sim-to-real robustness sweep")
    p.add_argument("--game", default="assault")
    p.add_argument("--axis", default="latency",
                   help="one of " + ", ".join(list(AXES) + ["presets", "control_rate", "all"]))
    p.add_argument("--episodes", type=int, default=150)
    p.add_argument("--seed", type=int, default=20_000)
    p.add_argument("--models", default="models")
    p.add_argument("--out", default="results")
    p.add_argument("--plot", action="store_true")
    args = p.parse_args(argv)

    axes = (list(AXES) + ["presets", "control_rate"]) if args.axis == "all" else [args.axis]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for axis in axes:
        print(f"\n=== {args.game} / {axis} ({args.episodes} eps per point) ===")
        res = sweep(args.game, axis, args.episodes, args.seed, Path(args.models))
        (out / f"robust_{args.game}_{axis}.json").write_text(json.dumps(res, indent=1))
        print("\n" + markdown(res))
        if args.plot:
            plot(res, out / f"robust_{args.game}_{axis}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
