"""Figures for the writeup's robustness section (committed under
writeup/figures/, since results/ is not). Reads the pe-robust JSONs (plus the
backfilled scripted_safe arm and the fixed control-rate run where present).

    .venv/bin/python scripts/analysis/robust_figures.py
"""
import json
from pathlib import Path

import pursuit_evasion  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pursuit_evasion.eval.robustness import latency_budget  # noqa: E402

R = Path("results")
OUT = Path("writeup/figures")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# fixed categorical order (validated: scripts/validate_palette.js, light surface)
ARMS = [("scripted", "scripted", "#2a78d6", "o"),
        ("scripted_safe", "scripted + shield", "#eb6834", "s"),
        ("bt_gated", "gated", "#1baf7a", "^"),
        ("bt_gated_safe", "gated + shield", "#eda100", "D"),
        ("policy", "policy alone", "#e87ba4", "v")]
POLICY = {"tag": "pursuer_dagger", "assault": "assault_dagger"}


def load(game, axis):
    for d in (R / "fixed", R):
        p = d / f"robust_{game}_{axis}.json"
        if p.exists():
            res = json.loads(p.read_text())
            break
    else:
        return None
    bf = R / "backfill" / f"robust_{game}_{axis}.json"
    if bf.exists() and "scripted_safe" not in res["rows"][0]["arms"]:
        extra = json.loads(bf.read_text())
        for r, e in zip(res["rows"], extra["rows"]):
            assert e["arms"]["scripted"]["wins"] == r["arms"]["scripted"]["wins"]
            r["arms"]["scripted_safe"] = e["arms"]["scripted_safe"]
    for r in res["rows"]:
        if POLICY[game] in r["arms"]:
            r["arms"]["policy"] = r["arms"][POLICY[game]]
    return res


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_ylim(-0.02, 1.02)


def lines(ax, res, xs, xlabel, label_ends=True):
    style(ax)
    # draw back to front so the scripted arms sit on top where gated coincides
    # with them exactly; gated arms are dashed so both stay visible
    for key, label, color, marker in reversed(ARMS):
        if key not in res["rows"][0]["arms"]:
            continue
        ys = [r["arms"][key]["win_rate"] for r in res["rows"]]
        lo = [r["arms"][key]["ci"][0] for r in res["rows"]]
        hi = [r["arms"][key]["ci"][1] for r in res["rows"]]
        ax.fill_between(xs, lo, hi, color=color, alpha=0.12, lw=0)
        ax.plot(xs, ys, color=color, lw=2, marker=marker, ms=5, label=label,
                ls="--" if key.startswith("bt_gated") else "-",
                markeredgecolor=SURFACE, markeredgewidth=1)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel("win rate (pursuer slot)", color=INK2, fontsize=9)


def fig_latency():
    games = [g for g in ("assault", "tag") if load(g, "latency")]
    fig, axes = plt.subplots(1, len(games), figsize=(5.2 * len(games), 3.8), facecolor=SURFACE)
    axes = [axes] if len(games) == 1 else axes
    for ax, g in zip(axes, games):
        res = load(g, "latency")
        xs = [r["level"] * 1000 for r in res["rows"]]
        lines(ax, res, xs, "state latency (ms)")
        b = latency_budget(g) * 1000
        ax.axvline(b, color=INK2, ls="--", lw=1)
        ax.text(b + 1.5, 0.97, f"capture radius / closing speed\n= {b:.0f} ms", color=INK2,
                fontsize=7, va="top")
        ax.set_title(f"{g}: state latency, n={res['episodes']} per point, 95% Wilson",
                     color=INK, fontsize=10, loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], fontsize=8, frameon=False, ncol=5,
               loc="lower center")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "robust_latency.png", dpi=150, facecolor=SURFACE)


def fig_presets():
    games = [g for g in ("assault", "tag") if load(g, "presets")]
    fig, axes = plt.subplots(1, len(games), figsize=(5.2 * len(games), 3.8), facecolor=SURFACE)
    axes = [axes] if len(games) == 1 else axes
    for ax, g in zip(axes, games):
        res = load(g, "presets")
        xs = list(range(len(res["rows"])))
        lines(ax, res, xs, "link preset")
        ax.set_xticks(xs, [r["profile"] for r in res["rows"]])
        ax.set_title(f"{g}: preset ladder, n={res['episodes']} per point",
                     color=INK, fontsize=10, loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], fontsize=8, frameon=False, ncol=5,
               loc="lower center")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "robust_presets.png", dpi=150, facecolor=SURFACE)


def fig_control_rate():
    games = [g for g in ("assault", "tag") if load(g, "control_rate")]
    if not games:
        return
    fig, axes = plt.subplots(1, len(games), figsize=(5.2 * len(games), 3.8), facecolor=SURFACE)
    axes = [axes] if len(games) == 1 else axes
    for ax, g in zip(axes, games):
        res = load(g, "control_rate")
        xs = [r["level"] * 1000 for r in res["rows"]]
        lines(ax, res, xs, "control period (ms)")
        b = latency_budget(g) * 1000
        ax.axvline(b, color=INK2, ls="--", lw=1)
        ax.text(b + 2, 0.06, f"latency budget {b:.0f} ms", color=INK2, fontsize=7)
        ax.set_title(f"{g}: control period, n={res['episodes']} per point",
                     color=INK, fontsize=10, loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], fontsize=8, frameon=False, ncol=5,
               loc="lower center")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "robust_control_rate.png", dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_latency()
    fig_presets()
    fig_control_rate()
    print("wrote", sorted(p.name for p in OUT.glob("robust_*.png")))
