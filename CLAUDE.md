# CLAUDE.md — pursuit-evasion

Always-on source of truth for architecture and invariants. Operational runbook
lives in `.claude/skills/pursuit-evasion/SKILL.md`; the build plan and rubric
mapping in `docs/PLAN.md`.

## What this is

A 2v2 3D pursuit-evasion testbed built to compare **hand-written behaviour trees**
against **learned policies**, and to find **where the boundary between them
actually is**. It spans a small but complete autonomy stack: env → scripted
tactics → self-play RL → imitation learning + RL fine-tune → BT gating → safety
filter → instrumentation → a C++ production runtime.

## Games are a layer, not a fork

The simulator is generic — two teams, one of which can neutralise the other. A
`GameSpec` (`env/games.py` ↔ `cpp/include/pe/games.hpp`) layers on the objective,
airframes and spawn geometry:

| game | pursuer slot | evader slot | win | timeout favours |
|---|---|---|---|---|
| `tag` | drones 5.25 m/s | drones 4.75 m/s | tag both evaders | evaders |
| `assault` | interceptors 7.5 m/s | missiles 11 m/s | asset never breached | defenders |
| `escort` | interceptors 7.5 m/s | missiles, long burn | convoy reaches goal | defenders |

Defenders reuse the **pursuer slot**, attackers the **evader slot**, so every
controller, observation, metric, gate and the safety filter carry over unchanged.

Heterogeneous airframes are a **pure command transform** (`apply_dynamics`), never
a second physics path: a holonomic drone gets its command verbatim; a missile
keeps thrust on its velocity axis, spends at most `lat_authority` of the command
laterally, carries a finite burn, and is out once it coasts below stall. Turn
radius is therefore `v²/a_lat` — fast means committed.

**The gate profile is per-game and empirically derived, not ported.** Tag's
"close quarters is messy → hand it to the policy" is *actively harmful* in air
defence (0.70 → 0.12 measured). The defender gate hands over exactly one regime:
where the scripted law is **structurally undefined** (no positive lead-intercept
root, so it parks on a static gate point). Keep it at two branches; see
`writeup/when-each-wins.md` §7 and the ablation in `_defender_predicates`.

## Two runtimes, one set of semantics

- **Python (train + research):** MuJoCo (python), SB3 PPO self-play, py_trees gate
  (reference), imitation learning, the scenario/instrumentation harness.
- **C++ (`cpp/`, production):** MuJoCo **C API**, **BehaviorTree.CPP** gate
  (Groot2-editable XML), policy via **ONNX Runtime**, safety filter, headless runner.

The two are kept semantically identical and cross-checked:
- both load the **same** `cpp/assets/arena*.xml` (dumped from Python) and link the
  **same** `libmujoco.so.3.12.0` (from the venv);
- the observation encoding, scripted controllers, feature extraction, gate
  predicates, and safety filter are line-for-line mirrors;
- `pe_run --parity` (C++) vs `pe-parity` (Python) must print matching checksums
  for the same 5 fixed starts;
- the C++ runner can consume Python's exact start states (`--starts-file`, dumped
  by `eval.dump_starts`) so aggregate metrics match, not just single rollouts.
  numpy's Generator and `std::mt19937_64` are different streams — the same seed
  does *not* give the same spawns, which is why the starts file exists.
  On the defence games the two runtimes match **exactly**, including per-counter
  safety violations. On tag expect a few flipped binary outcomes in long chaotic
  chases; judge on outcomes, not on the tail of `steps2win`.

**Invariant:** any change to dynamics, the observation layout, the scripted
controllers, the gate predicates, or the safety filter MUST be applied to *both*
runtimes and re-validated with the parity check. If they diverge, the C++ result
is not trustworthy.

## Physics / balance invariants

- Point-mass agents, 3 slide joints each, drag-limited (`v_max = gear/damping`).
  That bound is only true if thrust is **isotropic**, so `apply_dynamics` clips
  the command by **norm**, never per component. The three actuators each accept
  [-1, 1], so a per-component clip would let a diagonal command pull `sqrt(3)*gear`
  — measured 7.42 m/s against a documented 5.25. Scripted controllers emit
  unit-norm directions and never expose this; learned policies fill the action
  cube and will exploit it silently. Pinned by `test_drone_thrust_is_isotropic`
  in both runtimes.
  In **tag** pursuers are faster than evaders (gears 21 vs 19) — without a speed
  edge a 2v2 capture is impossible. In the **defence** games the edge deliberately
  inverts (interceptor 7.5 vs missile 11 m/s): a tail chase is impossible, so the
  defender must solve a *crossing* intercept. That inversion is the point, not a
  balance bug.
- Balance is tuned so **scripted pursuers catch ~40–50%** of the time: enough
  headroom that a learned or gated policy can visibly beat or miss the baseline.
  Don't "fix" the sub-100% capture rate — it's the whole point.
- Capture = a pursuer within `capture_radius` of a live evader; that evader
  freezes. In tag, pursuers win iff both evaders are tagged before `max_steps`;
  in the defence games a timeout is a **defender** win (the missiles are the ones
  on a clock) and a breach ends it immediately.
- The defence games check captures **every physics substep**. Not because
  missiles are fast (per control step tag closes 1.43 capture radii, assault only
  0.79) but because a slower defender can never tail-chase, so every engagement is
  a high-offset *crossing pass* whose chord through the capture sphere is far
  shorter than its diameter. Tag stays once-per-step: its baseline is calibrated
  on that.
- A missile that burns out and decays below stall speed is **spent** — removed
  from the fight like a capture, but recorded separately (`n_spent`).

## Safety filter is authoritative

The safety filter (`safety.py` / `cpp/include/pe/safety.hpp`) is a runtime shield
*below* both the BT and the policy. It can override any command (geofence +
speed cap). It is pure and unit-tested on both sides; every BT node is unit-tested
in `cpp/tests`. Treat these tests as the V&V gate — keep them green.

**Known limit, state it honestly:** the shield can only *request* a correction
from a missile. The airframe transform runs after it, and 0.45 lateral authority
may be too little to comply. Missiles stay bounded by their MuJoCo joint ranges —
enforced by the *model*, not by the shield. On a real vehicle that distinction
matters.

## Reproducibility

- `cpp/assets/arena*.xml` are generated; regenerate/verify with
  `python -m pursuit_evasion.env.dump_arenas [--check]`. `arena.xml` must stay
  byte-identical unless you intend to move the tuned tag baseline.
- Failure logs store spawn states → any episode replays exactly
  (`reset_with_starts`). Determinism depends on this; don't reintroduce
  per-reset model rebuilds or nondeterministic spawns.
