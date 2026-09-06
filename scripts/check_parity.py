"""Diff the outputs of ``pe-parity`` and ``pe_run --parity`` for CI.

The two commands print one line per fixed start configuration::

    [py-parity]  cfg=0 win=1 steps=312 checksum=1234.567890
    [cpp-parity] cfg=0 win=1 steps=312 checksum=1234.567891

Exact-start parity (CLAUDE.md invariant): win and steps must match exactly;
the checksum integrates min_dist over the whole trajectory, so it accumulates
floating-point noise — allow a small absolute tolerance, far below the "third
decimal" that would indicate a real semantic divergence.

Usage: check_parity.py <py_output> <cpp_output>
"""
from __future__ import annotations

import re
import sys

LINE = re.compile(r"cfg=(\d+) win=(\d+) steps=(\d+) checksum=([-\d.]+)")
CHECKSUM_TOL = 1e-4


def parse(path: str) -> dict[int, tuple[int, int, float]]:
    out = {}
    with open(path) as fh:
        for m in (LINE.search(line) for line in fh):
            if m:
                out[int(m.group(1))] = (int(m.group(2)), int(m.group(3)), float(m.group(4)))
    return out


def main() -> int:
    py, cpp = parse(sys.argv[1]), parse(sys.argv[2])
    if not py or set(py) != set(cpp):
        print(f"parity FAIL: configs differ (py={sorted(py)}, cpp={sorted(cpp)})")
        return 1
    ok = True
    for cfg in sorted(py):
        (pw, ps, pc), (cw, cs, cc) = py[cfg], cpp[cfg]
        if pw != cw or ps != cs or abs(pc - cc) > CHECKSUM_TOL:
            print(f"parity FAIL cfg={cfg}: py win={pw} steps={ps} checksum={pc:.6f} "
                  f"vs cpp win={cw} steps={cs} checksum={cc:.6f}")
            ok = False
    if ok:
        print(f"parity OK: {len(py)} configs, max |Δchecksum| = "
              f"{max(abs(py[c][2] - cpp[c][2]) for c in py):.2e}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
