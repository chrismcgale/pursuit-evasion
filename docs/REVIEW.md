# REVIEW — state of the project and known issues

*As of 2026-09-06 (commit `c2bc99b` + robustness data landed). This document and
`ROADMAP.md` are the session-carrying context: read these two, `CLAUDE.md`
(invariants), and the runbook skill before touching anything.*

## What exists

Full stack: env (MuJoCo, generated MJCF) → 3 games (`tag`/`assault`/`escort`,
heterogeneous missile-vs-drone dynamics as a pure command transform) → scripted
tactics → self-play PPO + BC + DAgger → BT gating (py_trees reference **and**
BehaviorTree.CPP/Groot2 XML) → runtime safety filter → C++ runtime (MuJoCo C
API + ONNX Runtime) with **exact** cross-runtime parity → instrumentation
(`pe-ablate`, `pe-trace`, `pe-explore` HTML replay with truth-vs-belief
rendering, `pe-robust` sweeps with Wilson CIs + paired bootstrap) → sim-to-real
link layer (`env/link.py`: Vicon/ELRS/whoop failure modes). 71 Python tests,
32 C++ test cases, all green.

## Headline results (n=200, perfect link, `results/games.json`)

| | scripted | best learned | BT-gated | + shield |
|---|---|---|---|---|
| tag (capture) | 0.455 | 0.200 | 0.190 | 0.460 |
| assault (asset held) | 0.540 | 0.040 | **0.725** | 0.720 |
| escort (convoy) | 0.395 | 0.250 | **0.460** | 0.440 |

Core claims, each measured not assumed:
1. **Gate where the scripted law is structurally undefined** (no lead-intercept
   root), not where things look messy. Defender gate: one branch, +0.17.
2. **The attacker gate hands over nothing** — ablation monotonically downhill
   (0.46/0.57 → 0.33/0.35). Both RL branches deleted, pinned by tests in both
   runtimes.
3. Robustness (new, `results/robust_*.json`, n=150/point, paired seeds+corruption):
   - Gate advantage in assault: +0.23 perfect → **+0.16 vicon_lab (survives a
     well-run volume)** → +0.08 vicon_busy → 0 field_degraded. On the
     single-axis latency sweep it loses significance at 40 ms.
   - **The shield inverts from redundant to load-bearing under degradation**
     (assault: beats bare gate by +0.21 at 60 ms latency; tag: only learned-side
     arm that stays within reach of scripted out to 40 ms).
   - Gate edge is robust to **label swaps** (+0.18 at 10× realistic rate) and
     moderate occlusion (+0.13 at 8%/tick). Latency is the killer axis, swaps
     are not — H2 resolved in the tree's favour, H1 backwards.
   - `latency_budget = capture_radius/closing_speed` (assault 76 ms) predicts
     the floor on every latency curve without simulation. Whoop-scale ≈ 50 ms.
   - **Escalation anti-pattern (mechanism, traced):** handover rises 16%→22%
     as the link degrades — `intercept_feasible` computed from a bad estimate
     reads as "no solution exists", so the tree hands over to the policy
     exactly when the policy is least trustworthy. Unfixed; see ROADMAP #3.
   - **One stale command frame forfeits assault** (0.71→0.07) at dt=60 ms.
   - **Tag's baseline is partly a clock artifact**: scripted 0.60→0.88 as
     control period drops 100→20 ms (substep capture held fixed) while the
     policy stays flat ~0.28. Tag's control period (100 ms) exceeds its own
     latency budget (70 ms).

## Known issues

**Data / statistics**
1. **Defender ablation is n=40, one seed block** (`ablation_assault_pursuers.json`)
   — the least-replicated number the writeup rests on. Attacker side got 150×2.
   Re-run to match before leaning on it.
2. **Robustness sweeps use seed block 20000+**, headline tables use 10000+.
   Cross-block variance is visible (tag scripted 0.56 vs 0.455; tag bt_gated
   0.17 vs 0.19). Robustness tables are internally consistent (paired) but not
   directly comparable to `games.json` — do not mix them in one table.
3. **Checkpoint selection bias**: best-of-N policy selection (winner's curse) —
   tag DAgger scored 0.30 on its selection battery, 0.20 held out. Writeup §4.1
   states it; any new policy comparison must use held-out seeds.
4. `results/` is gitignored: every artifact is regenerate-on-demand. Logs of
   the robustness runs are in `results/robust_{assault,tag}.log`.
5. **Assault `control_rate` axis was never run** (axis added after the sweep
   launched; only tag has it). One command: `pe-robust --game assault --axis
   control_rate --episodes 150 --plot`.

**Modelling honesty**
6. `cmd_latency` is whole-frame quantised at dt=60 ms (10/20 ms→0 ticks,
   40/80 ms→1) — rows are identical by construction. Documented in the axis
   metadata; the 0.71→0.07 step at one tick is exact, not an artifact.
7. **The opponent always flies a perfect link** in sweeps and traces. Defensible
   (opponent = environment) but unstated in the writeup; symmetric degradation
   is untested.
8. Only *opponent* bodies can identity-swap in `link.py`; own-team bodies can
   swap in a real volume too (defender chasing its wingman's estimate).
9. Battery sag scales the command post-hoc; a real sag also slows the *response*
   (tau grows). Second-order, but state it if quoted.
10. Point-mass dynamics: no attitude loop, no tilt-to-translate. Every gate
    threshold is calibrated on isotropic-instant thrust. This is the largest
    remaining sim-to-real gap (ROADMAP stage 2) and the declared reason the
    robustness layer came first.

**Architecture**
11. **The link layer is Python-only.** The C++ runtime cannot run degraded
    evals. Parity is unaffected (PERFECT bypasses bit-identically) but the
    "production runtime" story now lags the research one. Mirroring `link.py`
    + seeded-corruption parity is ROADMAP #10.
12. **The safety filter has no inter-agent separation** — geofence + speed cap
    only. For 2-4 real airframes in one room this is the *first* missing
    constraint, and "capture" currently commands near-contact.
13. The gate reads raw features with no estimate-quality input (see escalation
    anti-pattern above). The fix is designed, not built: require infeasibility
    to persist k ticks, or suppress handover under high innovation.
14. `LinkedController` history buffer caps at 64 views; fine for current
    latencies, silently wrong if anyone sweeps >3.8 s.
15. Explorer truth-vs-belief renders for the pursuer-slot side only (matches
    what's traced).

**Presentation / process**
16. Writeup has **no robustness section yet** — the biggest gap between what's
    measured and what's written. All data is in `results/robust_*.json` + this
    file's summary; plots exist for tag only (`--plot` wasn't set on the
    detached assault run; regenerate is cheap).
17. No CI, no git remote, no figures in the writeup (tables only), no GIF.
18. Two stale explorer artifacts predate the est-rendering (`explorer_assault_0`,
    `explorer_tag_1`); regenerate if shown.

## Key file map

- `env/link.py` — link profiles + `LinkedController` (wrap order:
  `Linked(Shielded(Gated))` = deployment topology).
- `eval/robustness.py` — axes, `latency_budget()`, `control_rate_configs()`,
  Wilson/paired-bootstrap, sweep CLI.
- `bt/gating.py` — `_PREDICATES`, `BRANCH_READS` (display metadata, pinned to
  predicates by test), `evaluate_branches()`.
- `eval/trace.py` / `eval/explorer.py` — per-tick recorder (`--link`) and the
  self-contained HTML viewer.
- `cpp/` — mirrored runtime; **any semantic change must land in both and pass
  `pe-parity` vs `pe_run --parity`** (CLAUDE.md invariant).
- Tests that pin findings: `test_attacker_gate_hands_over_nothing` (×2
  runtimes), `test_perfect_link_is_a_no_op`,
  `test_tag_control_period_exceeds_its_own_latency_budget`,
  `test_branch_reads_match_predicates`.
