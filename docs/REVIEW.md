# REVIEW — state of the project and known issues

*As of 2026-11-03 (phases 1-3 of the post-review plan, PRs #2-#8, stacked).
This document and `ROADMAP.md` are the session-carrying context: read these
two, `CLAUDE.md` (invariants), and the runbook skill before touching
anything.*

## 2026-11-03 session — what changed

Semantics changed in nothing shipped: every new constraint is off by default,
every new model sits beside the shipped ones, and all six default parity
combinations stay exact.

- **Parity is CPU-dependent through OpenBLAS (#24).** Pinned at import on
  AVX-512 hosts, with a warning if numpy was imported first.
- **Own-team separation in the shield (#12)**, both runtimes, default off,
  recommended on. Tag pursuers overlapped in half of all episodes; separating
  them raises capture +0.08. Opponent separation is Chris's decision (options
  under #12).
- **The link layer runs in C++ (#11, #14)** with exact parity through a
  Python-written corruption tape.
- **Defender retrain + gate value vs policy quality (#19, writeup §7.3).** The
  gate's gain does not follow the policy's own quality; simple scripted
  fallbacks explain much of it (assault pure pursuit +0.06; escort
  predicted-asset gate point +0.08-0.10); escort's BC iteration 0 adds +0.25
  on two blocks. Which models ship is Chris's decision.
- **Robustness re-run and written up (writeup §8, #16, #22).** Recipe:
  scripted + shield. The old "+0.16 on vicon_lab" is retracted. The
  control-rate axis had a missile-burn artefact (#5). Training under the link
  changes nothing.
- **Shield's escort cost is the speed cap** (paired, two blocks; headline
  section below).
- **CI is red on `main` since 2026-11-03 13:14 (run 6), before any of this
  session's changes:** the same commit `a734f87` passed on 2026-10-02. Tag
  bt_safe parity fails on the runner (scripted parity is exact). The
  runner now has AVX-512: `main`'s Python gave 1340.882892 on start 2 (the
  OpenBLAS FMA value), and with PR #2's pin it gives 1340.887337, the value
  both runtimes give on the cloud Xeon used here. CI's C++ binary gives a
  third value, 1340.884633, in both runs, so a second, C++-side numerical
  difference exists on that runner and could not be reproduced here. Suspects:
  ONNX Runtime's MLAS kernel choice for that CPU (release tarball vs wheel)
  or glibc libm's CPU-dispatched variants. Next step: one CI run with a
  per-tick `%.17g` action dump on both sides (the 2026-09-30 method) to find
  the first differing number. Until then, CI cannot confirm tag bt_safe
  parity; every check here passes on the Xeon.
- **CI:** this session could not edit `.github/workflows` (token without
  `workflow` scope). New parity checks live in `scripts/ci_parity.sh`; the
  "Cross-runtime parity" step should call it (one-line edit, Chris).
- **Timing (ROADMAP #6):** `pe_bench` per-stage histograms; see ROADMAP #6.
- **Stage 2 proposal:** `docs/STAGE2.md`.

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
**Checked 2026-11-03, paired** (`scripts/analysis/escort_shield_paired.py`,
n=200 on seeds 10000+ and 20000+, pooled n=400): shield on the gated stack
-0.052 [-0.085, -0.020]; speed cap alone -0.040 [-0.072, -0.007]; fence alone
-0.010 [-0.022, 0.000]; on the scripted law the shield costs -0.040
[-0.075, -0.005]. The speed cap carries about four fifths of the cost. It
fires on ~350 ticks per 200 episodes in escort and removes the accelerating
component whenever a defender is above 0.98 v_max, which a crossing
intercept against an 11 m/s missile needs. Not changed here: the cap is a
safety limit, and whether escort should run with a higher cap fraction (or
none, with the fence alone) is a safety decision, not a tuning one.
**Robustness, re-run on the new semantics (assault, n=150, seeds 20000+,
paired):** the gate's edge is +0.027 at perfect link, +0.013 (n.s.) on
`vicon_lab`, and exactly zero from 20 ms of state latency on. The old headline
"gate advantage survives a well-run Vicon volume (+0.16)" **no longer holds** —
it was the old law's weakness surviving, not the gate. What changed instead:
the fixed scripted law degrades far more gracefully (40 ms: 0.49, was 0.19;
60 ms: 0.27, was 0.05), and the shield is the load-bearing mitigation
(+0.13 at 60 ms, +0.21 at 80 ms, CIs exclude 0). Hardware recipe is now simply
**scripted + shield**; the gate is a perfect-link refinement.
**2026-11-03: every axis re-run on these semantics and written up (writeup
§8, figures in `writeup/figures/`).** Additions: a `scripted + shield` arm
(the recipe was never measured directly before); under the shield the
defender gate adds a significant +0.04..+0.08 on every non-latency axis and
never a negative delta, and nothing from ~30 ms of latency; from 40 ms the
gated and scripted outcomes are identical episode for episode even though
handover rises 11.5% -> 26%; the k-debounce re-measured on the current law
still loses (k3-k1 = -0.027 at 0 ms, zero from 20 ms); one stale command
frame 0.787 -> 0.273 (equal to the 60 ms state-latency row, as it should be);
training the policy under `vicon_lab` changes nothing (§8.8); and the
assault control-rate "saturation at 1.00" was a missile-burn artefact
(#5).
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
   period — +0.20/+0.23/+0.16/+0.08 for dt=40/60/100/160 ms (old law) — and at
   dt=20 ms assault saturates to 1.00 for *every* arm. **Corrected
   2026-11-03:** the saturation was an artefact. `burn_steps` counts control
   steps and the axis rescaled `max_steps` but not the burn, so at dt=20 ms a
   missile burned 2.4 s instead of 7.2 s and never reached the asset (breach
   0.00, policy alone "1.00"). Fixed in `control_rate_configs` and pinned by
   `test_control_rate_holds_missile_burn_time_constant`; on the current law
   scripted goes 0.733 -> 0.833 from 160 to 20 ms and the gate's edge holds at
   +0.03..+0.04 (writeup §8.7). Units smell worth fixing in Stage 2: burn
   should be in seconds.
   Running it exposed and fixed a real crash: `aggregate()` died on a
   defence-game win with no capture step (timeout/spent-out win), first hit
   at dt=100 ms. Pinned in `test_robustness`.

**Modelling honesty**
6. `cmd_latency` is whole-frame quantised at dt=60 ms (10/20 ms→0 ticks,
   40/80 ms→1) — rows are identical by construction. Documented in the axis
   metadata; the 0.71→0.07 step at one tick is exact, not an artifact. On the
   current law the step is 0.787 -> 0.273, identical to the 60 ms
   state-latency row (writeup §8.6).
7. **The opponent always flies a perfect link** in sweeps and traces. Defensible
   (opponent = environment); now stated in writeup §8. Symmetric degradation
   is still untested.
8. Only *opponent* bodies can identity-swap in `link.py`; own-team bodies can
   swap in a real volume too (defender chasing its wingman's estimate).
9. Battery sag scales the command post-hoc; a real sag also slows the *response*
   (tau grows). Second-order, but state it if quoted.
10. Point-mass dynamics: no attitude loop, no tilt-to-translate. Every gate
    threshold is calibrated on isotropic-instant thrust. This is the largest
    remaining sim-to-real gap (ROADMAP stage 2) and the declared reason the
    robustness layer came first.

**Found by the 2026-09-30 review, still open**
19. **Retrained 2026-11-03; the shipped models are unchanged (Chris's
    decision).** Defender policies retrained against the sphere law with every
    DAgger iteration kept (`models/sphere_law/`, lineage JSONs alongside).
    Gate value vs policy quality on two held-out blocks (writeup §7.3): the
    gain does not rise with the policy's own win rate (assault: policy alone
    0.005 -> 0.175, gain +0.025..+0.075). Escort's iteration-0 behaviour clone
    is an outlier: gated 0.790 / 0.815 vs scripted 0.545 on both blocks
    (+0.245 / +0.270), against +0.075 / +0.035 for the shipped policy. Simple
    scripted fills of the same regime explain part of it: pure pursuit
    recovers all of assault's gate gain (+0.060), and leading the escort gate
    point to the convoy's predicted position gains +0.095 / +0.075. Options
    for shipping, none taken:
    (a) keep the shipped old-law clones (CI parity inputs unchanged);
    (b) ship the best-on-selection sphere-law checkpoints (assault it8, escort
        it7): no better than shipped on either block, so little reason to;
    (c) ship escort it0 (replicated +0.27 over scripted; picked post hoc
        from 10 checkpoints, but on an independent second block);
    (d) **recommended:** first fix the two scripted fallbacks (pure pursuit
        when no intercept in assault; predicted-asset gate point in escort),
        both runtimes + parity, then retrain and re-measure. Shipping a policy
        to cover a gap a ten-line rule closes repeats the §7.2 pattern.
20. **Each gated agent gets an action from a team-joint policy** trained to
    drive both agents; mixed scripted/RL ticks are out of its training
    distribution.
21. **Captured agents coast, they do not freeze** (ctrl zeroed, velocity decays
    with tau ~0.25 s; one captured mid-step keeps thrusting until the step
    ends). Breach is checked once per control step but capture every substep,
    so a capture can pre-empt an earlier in-step breach (slightly pro-defender).
22. ~~Robustness axes other than assault latency/presets were measured on the
    pre-review semantics~~ **Re-run 2026-11-03** (all assault axes and all tag
    axes, n=150, seeds 20000+, with a `scripted + shield` arm). ~300 paired deltas, still no multiple-comparison
    correction (stated in writeup §8, which leans on patterns across levels
    and games rather than single cells); the control-rate axis runs policies
    at periods they were not trained at.
23. Writeup/attacker docstrings called the missile law "proportional
    navigation" — it is lead guidance + bearing split + terminal break (fixed
    in code; check any new prose).

24. **Parity silently depends on the host CPU — half fixed 2026-11-03** (C++
    side open on the CI runner; see the 2026-11-03 summary): numpy's
    OpenBLAS uses an FMA `ddot` kernel on AVX-512 CPUs, so 3-vector dots differ
    from the C++ runtime in the last bit (tag bt_safe parity failed 3/5 on a
    Xeon). The package pins `OPENBLAS_CORETYPE=Haswell` at import on those
    hosts; `test_small_dot_is_sequential` pins it. Runs on an AVX-512 host
    before this fix are not on the laptop/CI semantics.

**Architecture**
11. ~~The link layer is Python-only~~ **Resolved 2026-11-03.** `pe/link.hpp`
    mirrors `env/link.py`; `pe_run --link <preset>` runs degraded evals on the
    production stack. The random half comes from a corruption *tape* that
    Python writes (`link.py:write_tape`, `dump_starts --link`, `pe-parity
    --tape`) and C++ replays (`--link-tape`), rather than a C++ port of
    numpy's PCG64 + ziggurat normal + `Generator.choice`: every draw depends
    only on (seed, stream, tick), never on state or commands, so a tape is
    exact by construction and Python stays the one source of the stochastic
    model (the `--starts-file` precedent). A native `mt19937_64` source runs
    standalone C++ evals that are statistically equivalent but not paired.
    Exact: fixed-start parity for 3 presets x 3 games x {scripted, bt_safe}
    (18/18); episode-level aggregate (`scripts/link_aggregate_parity.py`,
    n=40, 0 flips, identical violation counts) for assault and escort under
    `vicon_lab` and tag under perfect / `vicon_lab` / `field_degraded`. The
    `link.py` refactor (draws moved into `CorruptionStream`) was verified
    bit-identical on 60 golden episodes before anything else changed.
    `scripts/ci_parity.sh` runs `vicon_lab` for every game x stack.
12. **Own-team separation: built 2026-11-03, default off; separation from
    opponents is open (Chris's decision).** `SafetyConfig.separation_m` keeps
    live teammates apart, brake-aware like the fence (acts on
    `(p_i + h v_i) - (p_j + h v_j)`), in both runtimes, parity exact on and off
    (CI: tag 1.5 m, escort 5 m), 7 unit tests per side. Measured
    (`scripts/analysis/separation_cost.py`, seeds 20000+, n=150, paired,
    separation = 2x contact distance: tag 1.4 m, defence 2.0 m). "Contact" =
    teammates closer than two agent radii at any tick (agents have no physical
    collision in the sim, so this was invisible until now):

    | game, link | arm | win off | win on | paired delta | contact episodes off -> on |
    |---|---|---|---|---|---|
    | tag, perfect | scripted+shield | 0.527 | 0.607 | +0.080 [+0.033, +0.127] | 0.50 -> 0.01 |
    | tag, perfect | bt_gated_safe | 0.353 | 0.360 | +0.007 [-0.080, +0.087] | 0.73 -> 0.01 |
    | tag, vicon_lab | scripted+shield | 0.513 | 0.547 | +0.033 [-0.020, +0.093] | 0.50 -> 0.08 |
    | tag, vicon_lab | bt_gated_safe | 0.347 | 0.287 | -0.060 [-0.133, +0.013] | 0.70 -> 0.04 |
    | assault, perfect | scripted+shield | 0.727 | 0.727 | 0.000 | 0.01 -> 0.01 |
    | assault, perfect | bt_gated_safe | 0.767 | 0.760 | -0.007 [-0.020, 0.000] | 0.01 -> 0.01 |
    | assault, vicon_lab | scripted+shield | 0.680 | 0.687 | +0.007 [0.000, +0.020] | 0.01 -> 0.01 |
    | assault, vicon_lab | bt_gated_safe | 0.727 | 0.727 | 0.000 | 0.01 -> 0.01 |
    | escort, perfect | scripted+shield | 0.513 | 0.507 | -0.007 [-0.027, +0.013] | 0.10 -> 0.00 |
    | escort, perfect | bt_gated_safe | 0.600 | 0.593 | -0.007 [-0.033, +0.013] | 0.11 -> 0.00 |
    | escort, vicon_lab | scripted+shield | 0.460 | 0.447 | -0.013 [-0.033, 0.000] | 0.07 -> 0.01 |
    | escort, vicon_lab | bt_gated_safe | 0.587 | 0.580 | -0.007 [-0.033, +0.013] | 0.09 -> 0.00 |

    Tag's scripted pursuers overlap each other in half of all episodes (both
    converge on the last live evader; p10 of the closest approach is 0.03 m),
    and keeping them apart *raises* capture by 8 points: two pursuers on two
    sides of an evader beat two on top of each other. Defenders split across
    threats and almost never meet (the assault 1% is a spawn overlap that
    nothing can undo in one tick). Under `vicon_lab`, estimate noise lets 4-8%
    of tag episodes still close inside 2r. No cell shows a significant loss.
    **Recommendation: make it the default** (separation = 2x contact distance)
    with the next headline re-run, since it moves the tag numbers; it is the
    first constraint any 2-4 airframe flight needs. Not switched here so this
    change moves no shipped number.

    **Open — separation from opponents (Chris's decision).** Capture currently
    *is* contact: tag's capture radius (0.7 m) equals two agent radii, and the
    defence games' 1.4 m is 1.4x contact. Stopping distance from top speed
    (one control period of reaction + full reverse thrust against drag): tag
    0.53 + 0.40 = 0.93 m; defence 0.45 + 0.58 = 1.03 m. At whoop scale (5 m/s,
    ~12 m/s^2 lateral at 50 deg tilt, ~0.2 s of control + link + attitude lag)
    it is ~1.0 + 1.1 = 2.1 m, far above any sensible 0.3-0.5 m capture radius.
    Options, none picked:
    (a) *No opponent separation*; capture ends the engagement and both break
        off. Contact risk on every capture, carried by prop guards, foam and
        the kill switch. Keeps current semantics.
    (b) *Brake-aware floor below the capture radius.* No contact by
        construction if floor + stopping distance <= capture radius; at whoop
        scale that forbids fast terminal approaches, so capture of a fleeing
        target becomes rare. Measure before choosing.
    (c) *Stand-off capture.* Raise the capture radius to stopping distance plus
        a floor (~2 m at whoop scale) so "capture" is a virtual tag at range;
        safe, changes what the game means, re-tune the 40-50% balance.
    (d) *Altitude layering.* Teams hold separated altitude bands and capture
        is judged in the horizontal plane; prevents contact by construction,
        costs the vertical dimension in a ~2.5 m room.
    (e) *Post-capture separation only*, combined with (a) or (c): the shield
        separates from neutralised opponents so the pair breaks off; does not
        protect the capture instant itself.
13. The gate reads raw features with no estimate-quality input. The
    k-tick-persistence fix was built, measured and **reverted** (see the
    escalation bullet above — it loses everywhere). The remaining designed-not-
    built idea is innovation-based suppression (hold handover when the track is
    jumpy), which unlike the debounce would leave perfect-link behaviour
    untouched — but the burden of proof is now the k-sweep table, and the
    measured story ("the shield, not the gate, carries degradation") may simply
    be the answer.
14. ~~`LinkedController` history buffer caps at 64 views~~ **Fixed
    2026-11-03** in both runtimes: the view history is sized
    `ceil(latency / dt) + 1`, the command history `round(cmd latency / dt) + 1`
    (tested at 80 ticks of latency on both sides).
15. Explorer truth-vs-belief renders for the pursuer-slot side only (matches
    what's traced).

**Presentation / process**
16. ~~Writeup has no robustness section yet~~ **Written 2026-11-03**: writeup
    §8, figures committed in `writeup/figures/` (regenerate with
    `scripts/analysis/robust_figures.py`; tables with `robust_report.py`).
17. ~~No CI, no git remote~~ **Resolved 2026-09-06**: repo at
    `github.com/chrismcgale/pursuit-evasion` (public and MIT-licensed since
    2026-11; it was private when this item was written), CI green on ubuntu-latest
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
