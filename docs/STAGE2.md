# STAGE2 — 6-DOF underactuated dynamics (design proposal)

*Status: proposal, 2026-11-03. Nothing here is implemented. The airframe
parameters and the attitude response model are Chris's decisions; the numbers
below are starting points to be measured on a real whoop, not choices. Covers
ROADMAP #12-#15.*

## 1. What changes and why

Every result so far is on a drag-limited point mass whose thrust vector is the
command (`apply_dynamics`, holonomic kind). A real quadrotor cannot do that. It
produces thrust only along its body z axis, so to accelerate sideways it must
first tilt, and the flight controller (FC) takes 100-200 ms to settle a new
attitude. Gravity also makes the vertical axis asymmetric: full thrust up gives
`T_max/m - g`, while down is limited to `g` plus drag. REVIEW #10 lists this as
the largest remaining sim-to-real gap, and every gate threshold, the 40-50 %
scripted baseline and both shield horizons are calibrated on the point mass.

Stage 2 replaces the point mass with a rigid body that has an attitude,
thrust along body z, gravity, and an attitude response that looks like a
Betaflight FC in angle mode. It keeps three things fixed:

1. `apply_dynamics` stays the single place where a command becomes actuator
   input (CLAUDE.md). It gains state; it does not gain a second physics path.
2. The controller interface stays "desired acceleration direction in the unit
   ball". Scripted laws, observations and the gate keep their meaning; only the
   response to a command changes.
3. Both runtimes and the parity check stay exact at every step of the migration
   (section 5).

## 2. Dynamics model

### 2.1 Rigid body

Two ways to give the body an attitude in MuJoCo:

| | A. free joint, kinematic attitude | B. slide joints + attitude as a side state |
|---|---|---|
| translation | MuJoCo integrates the free joint | MuJoCo integrates 3 slide joints (as today) |
| thrust direction | site actuator with gear `0 0 1 0 0 0`: MuJoCo applies thrust along the body z axis | `apply_dynamics` rotates the thrust vector by the side-state attitude |
| attitude | the response model sets the free joint's angular velocity each substep; MuJoCo integrates the quaternion | integrated by the response model outside MuJoCo |
| rendering, traces | orientation is in `qpos`; the explorer and videos show tilt | orientation must be drawn separately |
| state layout | `qpos` 7 / `qvel` 6 per agent: every index in `sim.cpp`, `core.py`, `reset_with_starts`, the starts file and parity code changes | unchanged |
| collisions (Stage 3 prop-guard contact, nets) | possible later | not possible |

Recommendation: **A**, with the attitude driven kinematically (the response
model writes body angular velocity; no simulated rate PID, no rotor torques).
It is what ROADMAP #12 specified, MuJoCo does the thrust projection, and the
orientation lives in the model state where replay and rendering already look.
The cost is a state-layout change that touches every index on both sides, which
is why section 5 isolates it in its own step. B is the fallback if the layout
change proves too disruptive: it gives identical translational physics.

Simulating rotor torques and a rate PID was considered and rejected for Stage
2: it adds inertia tensors, motor mixing and PID gains that would all need
identifying, and the outer loop only ever sees the closed-loop attitude
response, which a lag model captures.

### 2.2 Attitude response (the FC, as the ground station sees it)

Betaflight angle mode runs an angle P loop around a rate PID. Seen from the
ground station it is a lagged, rate-limited, tilt-limited follower of the
commanded attitude. Proposed model, per control substep:

    tilt_sp   = commanded tilt vector (roll, pitch), |tilt_sp| <= tilt_max
    tilt_rate = clip((tilt_sp - tilt) / tau_att, -rate_max, +rate_max)
    tilt     += tilt_rate * dt_phys
    yaw       = held at 0 (targets are points; heading does not matter yet)
    thrust   += (thrust_sp - thrust) * (1 - exp(-dt_phys / tau_motor))

`tau_att` sets the 100-200 ms settle time (about four time constants, so
`tau_att` 25-50 ms). `rate_max` (deg/s) caps large re-tilts, which is what
makes a reversal slower than a small correction. A second-order model (angle
loop natural frequency and damping) fits a step response with overshoot better;
it can replace the first-order one later without changing anything else,
because both live behind the same function.

**Motor lag moves from the link into the airframe.** `env/link.py` already
models a 20 ms first-order thrust lag (`motor_tau_s`). Once the airframe has
`tau_motor`, the link profiles must set `motor_tau_s = 0` for the whoop game,
or the lag is counted twice.

### 2.3 From command to FC setpoint

The controller emits `c` (unit ball), read as a desired acceleration
direction. `apply_dynamics` turns it into an FC setpoint the way the ground
station will on hardware (the outer position loop of ROADMAP #17):

    a_des     = c * a_max            (a_max: chosen authority, m/s^2)
    F         = m * (a_des + g * z_hat)
    thrust_sp = min(|F|, T_max)
    tilt_sp   = angle between F and z_hat, clipped to tilt_max, in F's azimuth

On hardware this function becomes the CRSF/ELRS command encoder, so the code
that is parity-checked is the code that flies. Gravity is compensated by
feed-forward; the residual (when `|F| > T_max` or the tilt clip binds) is the
real loss of authority a quadrotor has and the point mass does not.

### 2.4 Parameters to decide (Chris)

Starting points for a 65 mm 1S whoop, to be replaced by measured values
(ROADMAP #20: fly one whoop, compare to the model):

| parameter | starting point | how to measure |
|---|---|---|
| mass `m` | 25-30 g with battery | scale |
| `T_max / (m g)` | 2.5-3.5 | static thrust stand, or hover throttle |
| `tilt_max` | 45-55 deg (angle-mode limit) | Betaflight `angle_limit` |
| `tau_att` | 25-50 ms | step in roll setpoint, mocap |
| `rate_max` | 600-1000 deg/s | Betaflight rates |
| `tau_motor` | 15-25 ms | throttle step, mocap z |
| linear drag | so that top speed is 6-8 m/s at full tilt | flat-out run, mocap |

## 3. Whoop-scale preset

A new game spec (`tag_whoop` first), alongside the existing ones, never
replacing them:

| | current tag | proposed whoop tag |
|---|---|---|
| room | 24 x 24 x 11.5 m | ~5 x 5 x 2.5 m |
| capture radius | 0.7 m | 0.3-0.5 m |
| speeds | 5.25 / 4.75 m/s | 3-8 m/s, pursuer edge kept |
| physics step | 20 ms | 2-5 ms (must resolve `tau_att` 25-50 ms) |
| control period | 100 ms | 10-20 ms (100 Hz class) |
| latency budget (`capture_radius / closing speed`) | 70 ms | 0.4 / 10 = 40 ms |

The control period follows from the budget: at 100 ms the current tag already
decides engagements slower than its own tolerance (REVIEW, control-rate
result). At whoop scale a 40 ms budget leaves room for one or two control ticks
plus the realistic Vicon + ELRS chain (~15-20 ms). Whether one process can
tick 4 aircraft at that rate is ROADMAP #6/#18 (`pe_bench`); measure it on the
ground station itself, and with the attitude model in the loop.

The shield needs re-deriving at this scale: `brake_horizon_s` = control period
+ `v / 2a` (ROADMAP #19), the 0.75 m fence margin is too large for a 5 m room,
and separation becomes mandatory with real airframes (it is a `SafetyConfig`
option since 2026-11-03).

Whether the defence games carry over is open (ROADMAP standing question): a
missile airframe at whoop scale is a whoop flying a constrained mode, which is
a modelling choice in itself.

## 4. What each change invalidates

| change | invalidates | does not invalidate |
|---|---|---|
| airframe state in `apply_dynamics` (refactor) | nothing if exact | — |
| new `quad` airframe kind + `tag_whoop` game | nothing existing: new game, new arena file | all current games, `arena*.xml`, parity lines |
| attitude lag, gravity, tilt limit | every policy trained for the new game; scripted balance of the new game | existing games |
| whoop room, capture radius, speeds | gate thresholds in metres and seconds (`close_quarters` 3.0, `juke_*`, `defender_close` 6.0, `threat_imminent`, `committed_range`), 40-50 % baseline, fence margin, brake horizon, `latency_budget` values, the tuned-band test in `test_env.py`, parity starts | the gate *rule* (gate where the law is undefined), the link layer, the instrumentation |
| control period 10-20 ms | link presets expressed in ticks (they are in seconds, so they carry over), `cmd_latency` quantisation result (one tick is now 10-20 ms, not 60), max_steps | link code |
| motor lag in the airframe | `motor_tau_s` in link presets for the new game (set 0) | other link fields |

Gate thresholds are per game **and** per airframe (ROADMAP #13): re-derive with
`pe-ablate --game tag_whoop` on two seed blocks before writing any profile.
Expect `scripted_fallback`-style regimes to move: lead intercept with a 150 ms
attitude lag is a different law from instant thrust.

## 5. Migration order (parity checkable throughout)

Each step lands in both runtimes and adds or keeps exact parity lines before
the next starts. Existing games stay bit-identical until a step says otherwise.

1. **Airframe state refactor.** Replace `burn_left` in `apply_dynamics` with an
   airframe-state struct (holds burn today, attitude and thrust later). No
   behaviour change; all six current parity combinations exact.
2. **`quad` kind with gravity, no lag.** Gravity on for that kind only
   (per-body `gravcomp` or an explicit force), feed-forward compensation in
   `apply_dynamics`, tilt and thrust limits, instant attitude. New game
   `tag_whoop` on the *current* room so only the airframe changes; add its
   scripted parity line.
3. **Free joint.** Switch the `quad` kind's bodies to a free joint with the site
   thrust actuator; attitude still instant. Both `sim` implementations index by
   joint address, not by position in `qpos`. Parity for `tag_whoop` scripted;
   current games unchanged (they keep slide joints).
4. **Attitude response.** Add `tau_att`, `rate_max`, `tau_motor`. Unit tests for
   the step response in both runtimes (settle time, rate cap), then parity.
5. **Whoop room and speeds.** New arena file via `dump_arenas` (the existing
   files are untouched, `--check` stays green), new physics step and control
   period, new parity starts.
6. **Balance.** Tune the pursuer edge until scripted catches 40-50 % on two
   seed blocks; pin it with a tuned-band test like `test_env.py`'s.
7. **Shield.** Re-derive fence margin, brake horizon and separation; unit tests
   in both runtimes; measure overrun and contacts as in the 2026-09-30 review.
8. **Policies.** DAgger lineage against the new scripted law
   (`dagger.py` now keeps every iteration), ONNX export, `bt_safe` parity.
9. **Gate.** `pe-ablate` from scratch; ship a profile only if it wins on two
   blocks.
10. **Robustness.** The same `pe-robust` axes with `motor_tau_s = 0`, so the
    "what did realism cost" comparison with Stage 1 is direct (ROADMAP #14).

Steps 1-4 can be done on the current room so that each one changes one thing.
Step 3 is the riskiest for parity (state layout) and is isolated for that
reason.

## 6. Decisions needed from Chris

1. Free joint with kinematic attitude (A, recommended) or slide joints with an
   attitude side state (B).
2. First- or second-order attitude response, and the parameter values in 2.4
   (best measured on one whoop before step 4).
3. Whoop room size, capture radius and speed band (section 3).
4. Whether the defence games are carried to whoop scale or tag becomes the
   only hardware game.
