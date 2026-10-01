# REVIEW — state of the project and known issues

*As of 2026-09-30 (deep review + follow-ups on branch `review-followups`). This
document and `ROADMAP.md` are the session-carrying context: read these two,
`CLAUDE.md` (invariants), and the runbook skill before touching anything.*

## 2026-09-30 deep review — what changed and why

A full read of both runtimes plus targeted experiments (scratch scripts, all
paired, n=150-200). Four findings changed shipped semantics; every one landed in
both runtimes with all six parity combinations exact.

1. **The defender gate's "undefined regime" was mostly a scripted-law bug.**
   `lead_intercept_time` solved for a *point* intercept; a kill needs only the
   1.4 m capture sphere. `GuardDefenders` now solves against a 0.7 m sphere.
   And the gate's `intercept_infeasible` was a proxy (nearest threat, root
   existence) that disagreed with the law's own switch (assigned threat,
   `t_int <= t_asset`) on 14% of handovers and 37% of fallback ticks; the gate
   now reads `GuardDefenders.plan()`'s own flag (`scripted_fallback`).
   Held-out seeds 20000+, n=200, defender win rate:

   | | assault | escort |
   |---|---|---|
   | old scripted | 0.525 | 0.440 |
   | old scripted + old gate (was shipped) | 0.730 | 0.495 |
   | sphere scripted | 0.795 | 0.545 |
   | sphere scripted + old proxy gate | loses −0.13 (blocks A+B) | loses −0.08 |
   | **sphere scripted + aligned gate (shipped)** | **0.845** | **0.620** |

   The thesis survives in a sharper, smaller form: the gate earns +0.05 /
   +0.075 over its own (fixed) law, CIs exclude 0 — but most of the old +0.17
   was a fix the scripted law could make itself, and gating on an
   *approximation* of the law's regime over the fixed law is actively harmful.
   R=0.7 was chosen on blocks 10000/110000 and confirmed on 20000.
   **The RL branch is still the DAgger clone of the OLD law** (see #19).
2. **Tag's cross-runtime "chaotic flips" were a bug, not chaos.** C++ computed
   `v * (1.0/n)` where numpy computes `v / n` (unit, shield `vhat`, missile
   `fwd`, obs, evader separation) — one ulp at tick 12 diverged a gated chase.
   Now exact everywhere, including 200-episode tag bt_safe aggregate parity
   (win 0.45, viol 6974/4397 identical).
3. **CI now checks parity for every game × {scripted, bt_safe}.** Previously
   tag scripted only — missile dynamics, substep capture, breach, both defence
   laws, the BT.CPP gate, the shield and ONNX were never compared in CI. The
   Python side runs the policy through the same ONNX file + ORT version.
   `models/` is committed (3.3 MB) so this — and every headline — reproduces
   from a clone. `pe_run`'s default tag ONNX was the 0%-capture self-play
   `pursuer.onnx`, not the `pursuer_dagger` Python gates to; fixed.
4. **Brake-aware geofence.** The fence was position-only (fires a control
   period late, then needs braking distance > the 0.75 m margin) and the
   downstream norm clip diluted its correction. Now acts on
   `pos + 0.15 s * vel` and shrinks unfenced axes instead. Tag overrun
   0.77 → 0.23 m, wall contacts 11 → 0/200, win 0.460 → 0.425 (n.s.;
   swept, see `SafetyConfig`); defence overrun → ≤0.04 m, win unchanged.
   **Under `field_degraded` it cannot help** (stale estimate; ~0.8 m overruns
   persist) — the FC-level fence in ROADMAP #19 is still required.

Checked and **not** a problem (recorded so nobody re-runs them):
- Latency cliff is real: constant-velocity forward prediction of the delayed
  estimate recovers only +0.01 (20 ms) to +0.12 (40 ms); scripted still 0.12
  at 60 ms. But `latency_budget`'s "no controller of any kind" is too strong.
- `dump_starts_file`'s `%.6f` rounding flips 0/200 tag outcomes.

Post-review headline (`pe-games`, seeds 10000+, n=200): tag 0.46 / gated 0.19 /
+shield 0.42; assault 0.77 / 0.80 / 0.81; escort 0.49 / 0.53 / **0.47** — note
the shield now *costs* escort 0.06 (the fence change itself was measured
neutral there, 0.460 → 0.465, so suspect the speed cap; unpaired, check it).
Pre-review result JSONs are preserved in `results/pre-review-2026-09-30/`.

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

> **Superseded 2026-09-30** — the table below is pre-review (old defender law,
> old gate, position-only fence). Current numbers: `writeup/when-each-wins.md`
> §6 (autofilled by `pe-games`) and the review section above.

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
   - **Escalation "anti-pattern" — resolved as a symptom, not a cost.**
     Handover rises 16%→22% as the link degrades (`intercept_feasible` from a
     bad estimate reads as "no solution exists"). The designed fix — require
     infeasibility to persist k ticks — was built, measured and **reverted**:
     clean dose-response loss (bt_gated 0.71/0.59/0.49 at perfect link for
     k=1/2/3, paired Δ(k3−k1) −0.22✓, recovering nothing at any latency).
     Traces: genuine infeasible regimes are 20–30-tick streaks that don't
     flicker, so a debounce is pure delay against an 11 m/s missile. Immediate
     handover is load-bearing; the latency collapse belongs to the *policy*,
     and the shield is the mitigation. Pinned by
     `test_defender_handover_is_immediate` (both runtimes); artifacts
     `results/robust_assault_latency_{debounce_k3,k2}.json`.
   - **One stale command frame forfeits assault** (0.71→0.07) at dt=60 ms.
   - **Tag's baseline is partly a clock artifact**: scripted 0.60→0.88 as
     control period drops 100→20 ms (substep capture held fixed) while the
     policy stays flat ~0.28. Tag's control period (100 ms) exceeds its own
     latency budget (70 ms).

## Known issues

**Data / statistics**
1. ~~Defender ablation is n=40, one seed block~~ **Resolved 2026-09-06**:
   re-run at 150×2 blocks — scripted 0.54/0.43, +`intercept_infeasible`
   0.70/0.68, policy-only 0.03/0.09. The headline +0.17 replicates.
   (`clean_intercept` bought exactly 0.00 in both blocks, as expected — it
   routes to scripted, same as the fall-through; it exists for Groot2
   legibility only.)
2. **Robustness sweeps use seed block 20000+**, headline tables use 10000+.
   Cross-block variance is visible (tag scripted 0.56 vs 0.455; tag bt_gated
   0.17 vs 0.19). Robustness tables are internally consistent (paired) but not
   directly comparable to `games.json` — do not mix them in one table.
3. **Checkpoint selection bias**: best-of-N policy selection (winner's curse) —
   tag DAgger scored 0.30 on its selection battery, 0.20 held out. Writeup §4.1
   states it; any new policy comparison must use held-out seeds.
4. `results/` is gitignored: every artifact is regenerate-on-demand. Logs of
   the robustness runs are in `results/robust_{assault,tag}.log`.
5. ~~Assault `control_rate` axis was never run~~ **Resolved 2026-09-06**
   (`robust_assault_control_rate.json`): gate edge significant at every
   period — +0.20/+0.23/+0.16/+0.08 for dt=40/60/100/160 ms — and at
   dt=20 ms assault saturates to 1.00 for *every* arm (the defence game gets
   easy for everyone at fast control, unlike tag where only scripted rises).
   Running it exposed and fixed a real crash: `aggregate()` died on a
   defence-game win with no capture step (timeout/spent-out win), first hit
   at dt=100 ms. Pinned in `test_robustness`.

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

**Found by the 2026-09-30 review, still open**
19. **The defender policy imitates the old law.** `assault_dagger` /
    `escort_dagger` are pure DAgger clones of the point-solve `GuardDefenders`;
    in the fallback regime they reproduce a smoothed version of the very law
    the gate bypasses. Retrain against the sphere law (or under the link,
    ROADMAP #11) before reading anything into the residual +0.05.
20. **Each gated agent gets an action from a team-joint policy** trained to
    drive both agents; mixed scripted/RL ticks are out of its training
    distribution.
21. **Captured agents coast, they do not freeze** (ctrl zeroed, velocity decays
    with tau ~0.25 s; one captured mid-step keeps thrusting until the step
    ends). Breach is checked once per control step but capture every substep,
    so a capture can pre-empt an earlier in-step breach (slightly pro-defender).
22. Robustness axes other than assault latency/presets were measured on the
    pre-review semantics — rerun before quoting (`pe-robust --axis all`).
    ~135 per-point CIs, no multiple-comparison correction; the control-rate
    axis runs policies at periods they were not trained at.
23. Writeup/attacker docstrings called the missile law "proportional
    navigation" — it is lead guidance + bearing split + terminal break (fixed
    in code; check any new prose).

**Architecture**
11. **The link layer is Python-only.** The C++ runtime cannot run degraded
    evals. Parity is unaffected (PERFECT bypasses bit-identically) but the
    "production runtime" story now lags the research one. Mirroring `link.py`
    + seeded-corruption parity is ROADMAP #10.
12. **The safety filter has no inter-agent separation** — geofence (now
    brake-aware, see review #4) + speed cap only. For 2-4 real airframes in one room this is the *first* missing
    constraint, and "capture" currently commands near-contact.
13. The gate reads raw features with no estimate-quality input. The
    k-tick-persistence fix was built, measured and **reverted** (see the
    escalation bullet above — it loses everywhere). The remaining designed-not-
    built idea is innovation-based suppression (hold handover when the track is
    jumpy), which unlike the debounce would leave perfect-link behaviour
    untouched — but the burden of proof is now the k-sweep table, and the
    measured story ("the shield, not the gate, carries degradation") may simply
    be the answer.
14. `LinkedController` history buffer caps at 64 views; fine for current
    latencies, silently wrong if anyone sweeps >3.8 s.
15. Explorer truth-vs-belief renders for the pursuer-slot side only (matches
    what's traced).

**Presentation / process**
16. Writeup has **no robustness section yet** — the biggest gap between what's
    measured and what's written. All data is in `results/robust_*.json` + this
    file's summary; plots exist for tag only (`--plot` wasn't set on the
    detached assault run; regenerate is cheap).
17. ~~No CI, no git remote~~ **Resolved 2026-09-06**: private repo at
    `github.com/chrismcgale/pursuit-evasion`, CI green on ubuntu-latest
    (pytest, arena check, C++ build+tests against fetched ORT 1.29.0,
    cross-runtime parity via `scripts/check_parity.py` — parity also holds
    under GCC 13 on foreign hardware). Note: local `origin` stays SSH
    (YubiKey); pushes without the key use
    `git -c credential.helper='!gh auth git-credential' push https://...`.
    Still missing: figures in the writeup (tables only), GIF.
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
  `pe-parity` vs `pe_run --parity` for every `--game` × `--controller
  scripted|bt_safe`** (CLAUDE.md invariant; CI runs all six).
- Tests that pin findings: `test_attacker_gate_hands_over_nothing` (×2
  runtimes), `test_perfect_link_is_a_no_op`,
  `test_tag_control_period_exceeds_its_own_latency_budget`,
  `test_branch_reads_match_predicates`.
