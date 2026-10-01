# ROADMAP — toward 2-4 real tinywhoops

*Vision (Chris, 2026-09-06): a real-world sim with 2-4 tinywhoops, flown off a
motion-capture volume (U of T Vicon access to be confirmed). Strategy chosen:
**staged** — robustness layer on the point-mass model first (done), then 6-DOF
underactuated dynamics, then hardware. Read `REVIEW.md` first for what exists
and what is broken.*

## Stage 0 — close out the analysis (short tasks, high value)

1. **Writeup §robustness** — the biggest measured-but-unwritten result.
   Fold in: preset ladder, latency cliff + `latency_budget`, shield inversion,
   escalation anti-pattern, swap immunity, one-stale-frame, control-rate
   artifact. Regenerate assault plots (`pe-robust --game assault --axis all
   --episodes 150 --plot` re-reads nothing; or plot from existing JSONs).
2. ~~Re-run the defender ablation at 150×2 seed blocks~~ **Done 2026-09-06**:
   +0.17 replicates (0.70/0.68 vs 0.54/0.43). REVIEW #1.
3. ~~Fix the gate escalation anti-pattern~~ **Done 2026-09-06 — hypothesis
   falsified and the fix reverted.** k∈{1,2,3} persistence swept on the
   latency axis: dose-response loss (0.71/0.59/0.49 at perfect link),
   nothing recovered at 20-60 ms. Genuine infeasibility doesn't flicker
   (20-30-tick streaks, traced), so persistence is pure delay. The
   escalation was a symptom; immediate handover is load-bearing and now
   pinned by `test_defender_handover_is_immediate` in both runtimes. The
   negative result IS the design contribution — writeup §robustness should
   carry the k-table.
4. **Gate-value vs policy-quality curve** — NB (review 2026-09-30): there is
   no DAgger checkpoint lineage — `models/history.json` is the self-play PPO
   history and `dagger.py` overwrites one best checkpoint. Save per-iteration
   checkpoints first, and retrain against the sphere-solving law (REVIEW #19).
   Then sweep gate advantage as a function of policy strength. Falsifiable prediction: gated-minus-scripted stays flat
   (the gain is regime coverage, not policy quality) while policy-only rises.
5. ~~Assault `control_rate` axis~~ **Done 2026-09-06** (REVIEW #5): gate edge
   holds at every period; dt=20 ms saturates every arm to 1.00. Still open
   from this item: whether tag's published baseline should be restated at a
   defensible control rate.
6. **C++ runtime timing evidence**: per-tick latency histogram for BT tick /
   ONNX inference / safety filter in `pe_run` (~1 h; the strongest artifact for
   a robotics reader, and it feeds the ground-station budget in Stage 3).
7. Groot2: screenshot of the shipped trees + live-monitoring hookup.
8. Repo hygiene: ~~CI~~ ~~remote~~ (done 2026-09-06: private
   `chrismcgale/pursuit-evasion`, CI green including cross-runtime parity).
   ~~Parity for all games + gate~~, ~~models committed~~ (done 2026-09-30).
   Remaining: robustness figures in the writeup, explorer GIF in the README.

## Stage 1 — robustness layer hardening (mostly done, gaps remain)

9. Symmetric degradation option (opponent flies the same link) + own-team
   swaps (REVIEW #7, #8).
10. **Mirror `link.py` in the C++ runtime** with seeded-corruption parity, so
    degraded evals run on the production stack too (REVIEW #11).
11. **Train under the link**: retrain/fine-tune the DAgger policies with
    `vicon_lab` corruption active (domain randomisation). Open question with
    real stakes: does the learned side close the robustness gap when it is
    allowed to *experience* the link in training? If yes, the shield story
    changes again.

## Stage 2 — 6-DOF underactuated dynamics (the big rebuild)

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
    at 100+ Hz.
19. **Safety before first flight** (also a sim task now): ~~brake-aware
    geofence~~ (done 2026-09-30; re-derive `brake_horizon_s` = control period
    + v/2a for the whoop airframe), inter-agent separation constraint in the
    shield (REVIEW #12), hardware kill switch,
    geofence enforced below the ground station (FC-level), foam-walled volume,
    battery-sag cutoff. House rule applies: nothing physical happens without
    Chris's explicit go.
20. Sim-to-real checkpoints: fly ONE whoop under ground-station control
    replaying sim trajectories; compare tracked-vs-sim state error to the
    `vicon_lab` profile; only then 2v2.

## Standing questions

- ~~Does the escalation fix (task 3) restore the gate at realistic latency?~~
  **Answered 2026-09-06: no — it destroys the gate at low latency and restores
  nothing.** Refined conclusion: the gate earns its +0.23 below ~30 ms of
  state latency and the shield carries everything past that, so the hardware
  recipe at whoop scale (budget ≈50 ms, well-run Vicon ≈10-20 ms) is
  **gate + shield if the volume is clean, scripted + shield if it is not** —
  and the sim can tell those apart per-volume with one latency sweep.
- Is tag worth keeping at whoop scale, or do the defence games (where the gate
  earns its keep) become the only story?
- 250 Hz ELRS vs 500 Hz: does the uplink one-stale-frame result (REVIEW #6)
  justify the higher rate at whoop control periods?
