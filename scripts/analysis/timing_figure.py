"""Per-stage tick-time histogram from pe_bench's CSV (log2 buckets).

    (cd cpp && ./build/pe_bench --game tag --aircraft 4 --episodes 50 \
        --onnx-evader /tmp/evader.onnx --csv /tmp/bench_tag4.csv)
    .venv/bin/python scripts/analysis/timing_figure.py /tmp/bench_tag4.csv \
        writeup/figures/timing_tag4.png
"""
import csv
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
STAGES = [("onnx_policy/team", "ONNX inference (per team)"),
          ("bt_tick/aircraft", "features + BT tick (per aircraft)"),
          ("scripted_law/team", "scripted law (per team)"),
          ("safety_filter/team", "safety filter (per team)"),
          ("ground_station_tick/all_aircraft", "whole tick, all aircraft")]


def main(src, dst, title=""):
    h = defaultdict(dict)
    for row in csv.DictReader(open(src)):
        h[row["stage"]][float(row["bucket_lo_us"])] = int(row["count"])
    fig, ax = plt.subplots(figsize=(7.2, 3.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for (suffix, label), color in zip(STAGES, COLORS):
        merged = defaultdict(int)
        for stage, buckets in h.items():
            if stage.endswith(suffix):
                for b, c in buckets.items():
                    merged[b] += c
        if not merged:
            continue
        xs = sorted(merged)
        tot = sum(merged.values())
        ax.plot([max(x, 0.5) for x in xs], [merged[x] / tot for x in xs], marker="o", ms=4,
                lw=2, color=color, label=label)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1.5)
    ax.set_xlabel("time per call (us, log2 bucket lower edge)", color=INK2, fontsize=9)
    ax.set_ylabel("fraction of calls (log)", color=INK2, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(dst, dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
