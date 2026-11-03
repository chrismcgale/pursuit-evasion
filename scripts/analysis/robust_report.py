"""Markdown tables for the writeup's robustness section, from pe-robust JSONs.

Reads results/robust_<game>_<axis>.json (and, where an arm was backfilled in a
separate run with the same seeds and link seed, results/backfill/...), keeps
the arms the writeup discusses, and prints win rates with Wilson intervals plus
the paired deltas the text cites. Every delta is recomputed here from the
stored per-episode wins, so the arms being differenced must share seed0 and
link_seed (asserted).

    .venv/bin/python scripts/analysis/robust_report.py --game assault --axis latency
"""
import argparse
import json
from pathlib import Path

import pursuit_evasion  # noqa: F401
from pursuit_evasion.eval.robustness import paired_delta

ARMS = {"tag": ["scripted", "scripted_safe", "pursuer_dagger", "bt_gated", "bt_gated_safe"],
        "assault": ["scripted", "scripted_safe", "assault_dagger", "bt_gated", "bt_gated_safe"],
        "escort": ["scripted", "scripted_safe", "escort_dagger", "bt_gated", "bt_gated_safe"]}
DELTAS = [("bt_gated", "scripted", "gate"), ("bt_gated_safe", "bt_gated", "shield on gate"),
          ("scripted_safe", "scripted", "shield on scripted"),
          ("bt_gated_safe", "scripted_safe", "gate under shield")]


def load(results: Path, game: str, axis: str) -> dict:
    main = json.loads((results / f"robust_{game}_{axis}.json").read_text())
    bf = results / "backfill" / f"robust_{game}_{axis}.json"
    if bf.exists():
        extra = json.loads(bf.read_text())
        assert (extra["seed0"], extra["link_seed"], extra["episodes"]) == \
            (main["seed0"], main["link_seed"], main["episodes"])
        for r, e in zip(main["rows"], extra["rows"]):
            assert r["profile"] == e["profile"]
            assert e["arms"]["scripted"]["wins"] == r["arms"]["scripted"]["wins"], \
                "backfill is not paired with the main run"
            for k, v in e["arms"].items():
                r["arms"].setdefault(k, v)
    return main


def fmt(d):
    sig = "*" if (d["lo"] > 0 or d["hi"] < 0) else ""
    return f"{d['delta']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]{sig}"


def tables(res: dict, game: str) -> str:
    names = [n for n in ARMS[game] if n in res["rows"][0]["arms"]]
    out = [f"| {res['label']} | " + " | ".join(f"`{n}`" for n in names) + " |",
           "|---" * (len(names) + 1) + "|"]
    for e in res["rows"]:
        cells = [f"{e['arms'][n]['win_rate']:.3f} [{e['arms'][n]['ci'][0]:.2f}, "
                 f"{e['arms'][n]['ci'][1]:.2f}]" for n in names]
        out.append(f"| {e['profile']} | " + " | ".join(cells) + " |")
    ds = [(a, b, lab) for a, b, lab in DELTAS if a in names and b in names]
    out += ["", "| " + res["label"] + " | " + " | ".join(f"{lab} ({a} - {b})" for a, b, lab in ds)
            + " |", "|---" * (len(ds) + 1) + "|"]
    for e in res["rows"]:
        A = e["arms"]
        out.append(f"| {e['profile']} | " + " | ".join(
            fmt(paired_delta(A[a]["wins"], A[b]["wins"])) for a, b, _ in ds) + " |")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="assault")
    p.add_argument("--axis", nargs="+", default=["latency"])
    p.add_argument("--results", default="results")
    a = p.parse_args()
    for ax in a.axis:
        res = load(Path(a.results), a.game, ax)
        print(f"\n### {a.game} / {ax} (n={res['episodes']}, seeds {res['seed0']}+)\n")
        print(tables(res, a.game))


if __name__ == "__main__":
    main()
