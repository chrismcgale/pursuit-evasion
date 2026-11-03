"""Diff the outputs of ``pe-parity`` and ``pe_run --parity`` for CI.

The two commands print one line per fixed start configuration::

    [py-parity]  game=assault ctrl=bt_safe cfg=0 win=1 steps=97 checksum=1234.567890 rl=40 viol=0/3
    [cpp-parity] game=assault ctrl=bt_safe cfg=0 win=1 steps=97 checksum=1234.567891 rl=40 viol=0/3

Exact-start parity (CLAUDE.md invariant): win, steps, the gate's RL tick count
and the shield's violation counts must match exactly; the checksum integrates
min_dist over the whole trajectory, so it accumulates floating-point noise —
allow a small absolute tolerance, far below the "third decimal" that would
indicate a real semantic divergence. (Scripted lines carry rl=0 viol=0/0.)

Usage: check_parity.py <py_output> <cpp_output>
"""
from __future__ import annotations

import re
import sys

LINE = re.compile(r"cfg=(\d+) win=(\d+) steps=(\d+) checksum=([-\d.]+)"
                  r"(?: rl=(\d+) viol=(\d+)/(\d+)(?:/(\d+))?)?")
CHECKSUM_TOL = 1e-4


def parse(path: str) -> dict[int, tuple]:
    out = {}
    with open(path) as fh:
        for m in (LINE.search(line) for line in fh):
            if m:
                g = m.groups()
                exact = tuple(int(x or 0) for x in (g[1], g[2], g[4], g[5], g[6], g[7]))
                out[int(g[0])] = (exact, float(g[3]))
    return out


def _diag_diff() -> None:
    """DIAGNOSTIC ONLY: first differing row of the per-tick dumps, if present."""
    import os
    a, b = "/tmp/pe_diag_py.txt", "/tmp/pe_diag_cpp.txt"
    if not (os.path.exists(a) and os.path.exists(b)):
        return
    pa, pb = open(a).read().splitlines(), open(b).read().splitlines()
    print(f"[diag] rows py={len(pa)} cpp={len(pb)}")
    for i, (x, y) in enumerate(zip(pa, pb)):
        if x != y:
            lo = max(0, i - 8)
            print("[diag] context (both equal):")
            for r in pa[lo:i]:
                print("   ", r)
            print("[diag] FIRST DIFF at row", i)
            print("  py :", x)
            print("  cpp:", y)
            xs, ys = x.split()[1:], y.split()[1:]
            for j, (u, v) in enumerate(zip(xs, ys)):
                if u != v:
                    print(f"  [{j}] py={u} cpp={v} diff={float(u) - float(v):.3e}")
            print("[diag] next rows:")
            for r1, r2 in list(zip(pa, pb))[i + 1:i + 6]:
                print("  py :", r1)
                print("  cpp:", r2)
            break
    else:
        print("[diag] dumps identical over the common prefix")
    os.remove(a)
    os.remove(b)


def main() -> int:
    py, cpp = parse(sys.argv[1]), parse(sys.argv[2])
    if not py or set(py) != set(cpp):
        print(f"parity FAIL: configs differ (py={sorted(py)}, cpp={sorted(cpp)})")
        return 1
    ok = True
    for cfg in sorted(py):
        (pe, pc), (ce, cc) = py[cfg], cpp[cfg]
        if pe != ce or abs(pc - cc) > CHECKSUM_TOL:
            print(f"parity FAIL cfg={cfg}: py (win,steps,rl,geo,spd,sep)={pe} checksum={pc:.6f} "
                  f"vs cpp {ce} checksum={cc:.6f}")
            ok = False
    _diag_diff()
    if ok:
        print(f"parity OK: {len(py)} configs, max |Δchecksum| = "
              f"{max(abs(py[c][1] - cpp[c][1]) for c in py):.2e}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
