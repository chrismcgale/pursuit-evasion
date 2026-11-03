# ROADMAP — toward 2-4 real tinywhoops

*Vision (Chris, 2026-09-06): a real-world sim with 2-4 tinywhoops, flown off a
motion-capture volume (U of T Vicon access to be confirmed). Strategy chosen:
**staged** — robustness layer on the point-mass model first (done), then 6-DOF
underactuated dynamics, then hardware. Read `REVIEW.md` first for what exists
and what is broken.*

## Stage 0 — close out the analysis (short tasks, high value)

1. ~~Writeup §robustness~~ **Done 2026-11-03** (writeup §8): every axis
   re-run on the current semantics with a `scripted + shield` arm, figures
   committed, the old "+0.16 on vicon_lab" headline retracted.
2. ~~Re-run the defender ablation at 150×2 seed blocks~~ **Done 2026-09-06**:
   +0.17 replicates (0.70/0.68 vs 0.54/0.43). REVIEW #1.
3. ~~Fix the gate escalation anti-pattern~~ **Done 2026-09-06 — hypothesis
   falsified and the fix reverted.** k∈{1,2,3} persistence swept on the
   latency axis: dose-response loss (0.71/0.59/0.49 at perfect link),
   nothing recovered at 20-60 ms. Genuine infeasibility doesn't flicker
   (20-30-tick streaks, traced), so persistence is pure delay. The
   escalation was a symptom; immediate handover is load-bearing and now
   pinned by `test_defender_handover_is_immediate` in both runtimes (since
   2026-11-03 moot: the defender gate hands nothing over). The
   negative result IS the design contribution — writeup §robustness should
   carry the k-table.
4. ~~Gate-value vs policy-quality curve~~ **Done 2026-11-03** (writeup §7.3,
   REVIEW #19). `dagger.py` keeps every iteration; retrained on the sphere
   law. Prediction half held: the gain never rises with policy-only quality
   (assault flat across 0.005 -> 0.175). Escort's BC iteration 0 is an
   outlier (+0.25 on two blocks). Simple scripted fills explain much of the
   gain (assault pure pursuit +0.06, escort predicted-asset gate +0.08-0.10).
   ~~Next: fix both scripted fallbacks~~ **Done 2026-11-03** (writeup §7.4):
   `adaptive` fallback, scripted +0.04 / +0.18 on a fresh block; afterwards
   no defender gate adds anything except escort's BC iteration 0 (+0.185),
   and the shipped escort gate loses 0.105. **Defender gate RL branch
   removed (Chris, 2026-11-03).** Next: what the escort BC policy does in the
   fallback regime that the law still does not.
5. ~~Assault `control_rate` axis~~ **Done 2026-09-06** (REVIEW #5): gate edge
   holds at every period; dt=20 ms saturates every arm to 1.00 — **a
   missile-burn artefact, fixed 2026-11-03** (REVIEW #5). Still open from this
   item: whether tag's published baseline should be restated at a defensible
   control rate (scripted 0.41 -> 0.89 from 160 to 20 ms, writeup §8.7).
6. ~~C++ runtime timing evidence~~ **Done 2026-11-03** (`pe_bench`, writeup
   §5.1): whole tick for 4 aircraft 21 us p50 / 47 us p99 on an idle cloud
   machine; ONNX and BT tick ~45% each, shield <1%; linear at ~10 us p99 per
   aircraft up to 16. Re-measure on the ground station itself.
7. Groot2: screenshot of the shipped trees + live-monitoring hookup.
8. Repo hygiene: ~~CI~~ ~~remote~~ (done 2026-09-06:
   `chrismcgale/pursuit-evasion`, public and MIT-licensed since 2026-11, CI
   green including cross-runtime parity).
   ~~Parity for all games + gate~~, ~~models committed~~ (done 2026-09-30).
   Remaining: robustness figures in the writeup, explorer GIF in the README.

## Stage 1 — robustness layer hardening (mostly done, gaps remain)

9. Symmetric degradation option (opponent flies the same link) + own-team
   swaps (REVIEW #7, #8).
10. ~~Mirror `link.py` in the C++ runtime~~ **Done 2026-11-03** via a
    Python-written corruption tape; exact for every preset x game x stack
    (REVIEW #11). Degraded evals now run on the production stack.
11. ~~Train under the link~~ **Done 2026-11-03, answer no** (writeup §8.8):
    the sphere-law defenders fine-tuned under `vicon_lab` (`dagger.py
    --link`, truth-labelled expert) are within +/-0.03 of their parents at
    every latency and preset, alone and gated. A DAgger student of a teacher
    that fails under latency cannot learn past it; a latency-aware teacher or
    an objective-trained policy would be the next test.

## Stage 2 — 6-DOF underactuated dynamics (the big rebuild)

*Design proposal (2026-11-03, not implemented): `docs/STAGE2.md` — model
choice (free joint with kinematic attitude, recommended), command-to-FC
mapping, whoop preset, what each change invalidates, and a 10-step migration
order that keeps parity checkable. The parameters and the response model are
Chris's decisions (STAGE2 §6).*

12. Replace slide joints with free joint + thrust-along-body-z + Betaflight-
    style rate/angle response model (attitude loop as first-order lag on tilt,
    ~100-200 ms to settle). Keep `apply_dynamics` as the single transform site.
13. Whoop-scale preset: ~5 m room, 0.3-0.5 m capture radius, 3-8 m/s, 65 mm
    ducted airframe params. Re-derive: timestep, control rate (the current
    60-100 ms periods are indefensible at whoop scale — budget ≈50 ms),
    spawn geometry, the 40-50 % scripted baseline, every gate threshold
    (`pe-ablate` from scratch — profiles are per-game AND per-airframe).
14. Re-run the **same** robustness sweep on the same axes — the point of the
    staged plan is the direct "what did realism cost" comparison.
15. Both runtimes + parity throughout, per CLAUDE.md. Budget: this invalidates
    all trained policies, both gate calibrations, and the arena XMLs at once.

## Stage 3 — hardware track (plan/analyse here; Chris executes physical steps)

16. **Confirm Vicon access** (U of T — Chris). Fallbacks: OptiTrack Duo/Trio,
    UWB (already modelled as `field_degraded`; results say don't bother for
    this task), lighthouse-style tracking.
17. **Platform decision** (flagged, not final): Betaflight whoop + ELRS means
    building the outer position loop ourselves on the ground station;
    Crazyflie + Crazyswarm2 gets that for free but flies 1-2 m/s (tame for
    pursuit). Current lean: whoops, speed is the problem statement.
18. Ground station architecture: the C++ runtime is the intended brain —
    Vicon bridge in, per-aircraft BT+policy+shield tick, CRSF/ELRS out.
    The Stage-0 timing histogram says whether one process holds 4 aircraft
    at 100+ Hz: **yes by compute** (~0.5% of a 10 ms period at p99 for 4
    aircraft, writeup §5.1); the open risk is scheduling jitter (worst ticks
    0.2-1.7 ms on the cloud machine), so plan a real-time thread and
    deadline-miss accounting, and re-measure with the Stage 2 attitude model
    in the loop.
19. **Safety before first flight** (also a sim task now): ~~brake-aware
    geofence~~ (done 2026-09-30; re-derive `brake_horizon_s` = control period
    + v/2a for the whoop airframe), ~~own-team separation in the shield~~
    (built 2026-11-03, default off, recommended on; REVIEW #12), separation
    from *opponents* (open, Chris's decision: options in REVIEW #12),
    hardware kill switch,
    geofence enforced below the ground station (FC-level), foam-walled volume,
    battery-sag cutoff. House rule applies: nothing physical happens without
    Chris's explicit go.
20. Sim-to-real checkpoints: fly ONE whoop under ground-station control
    replaying sim trajectories; compare tracked-vs-sim state error to the
    `vicon_lab` profile; only then 2v2.

## Standing questions

- **Revised 2026-11-03 (writeup §8, §7.4):** the hardware recipe is
  **scripted + shield**, with no defender gate: on the current law it adds
  nothing, and its RL branch is removed. The question below is answered for
  the OLD law only.
- ~~Does the escalation fix (task 3) restore the gate at realistic latency?~~
  **Answered 2026-09-06: no — it destroys the gate at low latency and restores
  nothing.** Refined conclusion: the gate earns its +0.23 below ~30 ms of
  state latency and the shield carries everything past that, so the hardware
  recipe at whoop scale (budget ≈50 ms, well-run Vicon ≈10-20 ms) is
  **gate + shield if the volume is clean, scripted + shield if it is not** —
  and the sim can tell those apart per-volume with one latency sweep.
- Is tag worth keeping at whoop scale, or do the defence games (where the gate
  earned its keep before the law was fixed) become the only story?
- 250 Hz ELRS vs 500 Hz: does the uplink one-stale-frame result (REVIEW #6)
  justify the higher rate at whoop control periods?
