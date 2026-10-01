# When does the tree win, when does the policy win, and when does gating win?

**TL;DR.** Across three games — symmetric `tag` and two air-defence problems
(`assault`, `escort`) — a hand-written behaviour tree built on lead-intercept
geometry + explicit target assignment is a **strong, cheap, legible baseline**,
and whether gating a learned policy beats it depends entirely on whether you can
name a regime the baseline does not serve.

- **In `tag` the baseline wins** (~46% capture of random starts) and no learned
  method here beats it; gating tracks and slightly trails it.
- **In air defence the first gate looked like a big win** — `assault` 0.54 →
  0.72 — **but most of that was a bug in the scripted law** (§7.2): it solved
  for a point intercept when a kill needs only the capture sphere. Fixed, the
  scripted law alone reaches 0.77 (`assault`) / 0.49 (`escort`), and a gate
  that hands over on the law's *own* fallback flag adds a smaller, real
  +0.03–0.075 on top (0.80 / 0.53; held-out: 0.845 / 0.62).
- **The gate profile does not transfer, and the failure is expensive.** Porting
  tag's "close quarters is messy, hand it to the policy" instinct to air defence
  costs **42 points** (0.70 → 0.12) — worse than not gating at all. The boundary
  that pays is not *"where is the world messy"* but *"where does the scripted law
  stop being defined"* — read off the law itself, not approximated: gating on
  a geometric proxy of that regime, over the fixed law, *loses* 13 points (§7.2).
- **Run on the *attacker* side of the same game, the identical procedure says
  "don't gate at all"** — every handoff loses breach rate (0.46 → 0.37), so both
  RL branches were deleted from the shipped missile profile. Two opposite
  verdicts from one method is the actual result: the defender's scripted law has
  an undefined regime and the missile's guidance law does not (§7.1).

The learned-side results are a compact tour of when each technique works:

- **RL from scratch (self-play PPO) fails** (0%) — the capture reward is too
  sparse for a random policy to bootstrap in the budget used.
- **Behaviour cloning fails despite near-perfect supervised loss** (MSE 0.003,
  yet ~0% capture) — the textbook covariate-shift failure: it drifts into states
  the expert never showed it and blends between the two evaders instead of
  committing.
- **DAgger fixes exactly that** — relabelling the learner's *own* visited states
  with the expert takes the same network from **0% → 20%** capture standalone,
  and to **42%** — near the scripted 46% — once the gate and safety filter are
  layered on. That is the headline: the *method*, not the budget, was
  the problem.
- **Gating only helps where the policy actually dominates.** In tag the gate
  hands close-quarters/juking/contested ticks to the (now competent) DAgger
  policy, but DAgger still isn't *better* than scripted at the precision endgame,
  so the gate tracks — and slightly trails — pure scripted. Gating a
  merely-competent policy into a strong baseline's home turf doesn't buy wins.
  The defence games are the converse of this, and the reason it's a statement
  about *regimes* rather than about gating.
- **The safety filter is a surprise net-positive.** Enabling the runtime shield on
  the gated controller more than **doubles** its win rate (0.19 → 0.42; 0.46
  with the pre-review position-only fence, which let it overrun the keep-in box
  by up to 0.77 m) by keeping the learned policy inside the geofence/speed
  envelope instead of wandering to the walls — safety here *improves* task
  performance rather than costing it. It is the largest single lever on the
  learned side, which was not the expected result for a V&V layer.
- **A physical invariant caught a bug that cross-runtime parity could not.** Both
  runtimes agreed exactly on an action-clipping rule that let a diagonal command
  buy 1.73× thrust; it inflated every learned result and was invisible to the
  scripted baseline. Two implementations agreeing proves they agree, not that
  either is right (§4.1).

## 1. Setup

- **Env.** Point-mass agents in a bounded 3D arena (MuJoCo). Drag-limited speed
  (`v_max = gear/damping`); pursuers faster than evaders (5.25 vs 4.75 m/s) —
  without a speed edge, 2v2 capture is impossible. Capture = pursuer within 0.7 m
  of a live evader (it then freezes). Pursuers win iff both evaders are tagged
  within 500 steps (~50 s).
- **Balance.** Tuned so the scripted baseline catches ~40–50% of random starts —
  enough headroom to see a controller beat or miss it, nothing saturated.
- **Controllers compared.**
  - `scripted` — lead-intercept + greedy target assignment (pursuers) / potential-field flee with juking (evaders).
  - `rl_selfplay` — SB3 PPO, alternating self-play against a growing opponent pool.
  - `bc_distill` — an MLP behaviour-cloned from the scripted expert (imitation).
  - `bc_finetuned` — `bc_distill` after PPO fine-tuning on the terminal-intercept sub-task.
  - `dagger` — `bc_distill` repaired with DAgger (expert relabelling of learner states).
  - `bt_gated` — the py_trees / BehaviorTree.CPP gate: policy in close/juking/contested regimes, scripted in clean geometry (routes to the best policy, `dagger`).
  - `bt_gated_safe` — `bt_gated` with the runtime safety filter enabled.

## 2. Methodology (instrumentation)

Every number below comes from the scenario harness (`eval/scenarios.py`): batched
runs over **seeded** random starts, identical seeds across controllers. Two
batteries: **full chase** (open-arena random starts) and **terminal intercept**
(endgame starts, pursuers spawned 3–5 m from an evader, short horizon). Metrics
per episode: **win/capture rate**, **time-to-first-intercept**, **min separation**
(closest approach), **constraint violations** (geofence/speed corrections by the
safety filter), and a **replayable failure log** (spawn state per lost episode →
re-run exactly with `reset_with_starts`). 200 episodes per cell below.

<!-- RESULTS:AUTOFILL -->

**Full chase (random open-arena starts)**

| controller | win rate | mean captures | steps→win | min sep (m) | geo/spd viol |
|---|---|---|---|---|---|
| `scripted` | 0.46 | 1.36 | 75 | 0.73 | 0/0 |
| `rl_selfplay` | 0.00 | 0.00 | — | 5.94 | 0/0 |
| `bc_distill` | 0.01 | 0.02 | 261 | 2.21 | 0/0 |
| `bc_finetuned` | 0.00 | 0.00 | — | 2.81 | 0/0 |
| `dagger` | 0.20 | 1.06 | 154 | 0.74 | 0/0 |
| `bt_gated` | 0.19 | 1.10 | 138 | 0.74 | 0/0 |
| `bt_gated_safe` | 0.46 | 1.36 | 152 | 0.74 | 7005/4296 |

**Terminal intercept (endgame starts)**

| controller | win rate | mean captures | steps→win | min sep (m) | geo/spd viol |
|---|---|---|---|---|---|
| `scripted` | 0.45 | 1.27 | 114 | 0.75 | 0/0 |
| `rl_selfplay` | 0.00 | 0.00 | — | 3.24 | 0/0 |
| `bc_distill` | 0.00 | 0.00 | — | 2.35 | 0/0 |
| `dagger` | 0.20 | 1.11 | 252 | 0.74 | 0/0 |
| `bt_gated` | 0.20 | 1.14 | 206 | 0.74 | 0/0 |

### 2.1 Per-decision instrumentation (the gate explorer)

Aggregate win rates say *whether* a gate helped, never *why*. The batteries above
are therefore paired with a per-tick recorder (`eval/trace.py`) and a
self-contained HTML viewer (`eval/explorer.py`, no server or CDN — the trace is
inlined):

    uv run pe-explore --game assault --seed 3     # -> results/explorer_assault_3.html

For every control step it captures each agent's fired branch, **all** branch
predicates with their live values (not just the winner, so you can see what
nearly fired), what the scripted controller and the policy each wanted at that
instant, the applied command, and whether the safety filter corrected it. It
records the same seed under several controllers so the dropdown is a true A/B on
identical starts, and the URL carries `#trace/step/agent` so a specific moment is
linkable.

Two things fell out of it that the aggregates hid:

- **`pe-explore --game assault --seed 3`** — the defender gate converts a breach
  into a double capture while handing the policy just **3 of 62 agent-ticks**,
  all in the terminal seconds. The gate's value is concentrated in a handful of
  decisions rather than spread across the episode, which is what §7's
  "structurally undefined regime" argument predicts and what a win-rate column
  cannot show.
- **`pe-explore --game tag --seed 13`** — an episode the safety filter *loses*
  (see §4), despite being strongly net-positive in aggregate.

(Both are regenerated on demand; `results/` is not committed.)

The branch metadata the viewer renders is generated from the predicate table
itself rather than re-implemented in JavaScript, and a test
(`test_branch_reads_match_predicates`) fails if a predicate is added without it,
so the explanation cannot drift from the tree that actually ran.

## 3. When each wins

- **Open chase & precision endgame → scripted.** Lead-intercept with explicit
  assignment is near-optimal when the bearing is clean, and it *commits* to a
  target, so it closes the final metre. It is the best single controller in both
  batteries (0.46 / 0.45). This is the regime where a hand-written tree is simply
  the right tool.
- **RL-from-scratch → nowhere (here).** With a capture-sparse reward and a modest
  self-play budget, PPO never sees enough reward to bootstrap and lands at 0%. Not
  a bug — a sample-efficiency reality. (Note the *same* PPO machinery works once
  warm-started, below, which rules out an implementation fault.)
- **Behaviour cloning → deceptively broken.** `bc_distill` reaches MSE 0.003 —
  it has "learned" the expert in the supervised sense — but captures ~0% and
  stalls ~2 m out. Two compounding effects: (a) covariate shift (it visits states
  the open-loop expert data never covered) and (b) it regresses toward the mean of
  a *bimodal* target (two evaders) and steers to the midpoint instead of
  committing. Low loss ≠ good closed-loop control.
- **DAgger → the learned-side win.** Feeding the learner its own visited states
  with expert labels (and the expert always commits to an assignment) drives it
  from **0% → 20%** standalone (1.06 mean captures; 0.21 on the endgame battery).
  This is *when imitation works*: on-policy relabelling, not more offline data.
  It is still well short of scripted's 0.46.
- **Gating → only as good as the policy you gate.** `bt_gated` routes the messy
  regimes to `dagger`, but `dagger` is competent, not superior to scripted in
  those regimes, so the gate lands at 0.19 — indistinguishable from the policy it
  gates (0.20), and far below pure scripted. The honest lesson: a gate is worth it
  precisely when the learned policy *dominates* some slice; identifying (or
  training) such a slice is the prerequisite, and a strong scripted baseline
  raises that bar. §7 is what happens when you find such a slice; §7.1 is what
  happens when you go looking and it is not there.
- **Safety filter → the biggest single lever on the learned side.**
  `bt_gated_safe` scores **0.46** against `bt_gated`'s 0.19: it more than doubles
  the gated controller and pulls level with the scripted baseline (0.455, and the
  same 1.365 mean captures). The DAgger policy habitually drives toward the arena
  walls; the shield fires constantly (7005 geofence + 4296 speed corrections over
  200 episodes) and redirects it into productive space. A layer added for V&V
  turns out to be the most effective *performance* component in the learned
  stack — see §4.

## 4. Safety / V&V

The safety filter is a runtime shield *below* the BT and the policy: an operational
geofence (keep-in box inside the arena) and a speed cap, either of which can
override the commanded action. It is a pure function, mirrored in Python and C++,
and unit-tested on both sides; every BT node has a unit test (`cpp/tests`,
`tests/test_env.py`). Unlike the usual "safety costs a little performance" story,
here it *helped*: on the gated DAgger controller it fired constantly (11970
geofence + 4924 speed corrections over 200 episodes with the brake-aware fence)
and took the win rate from 0.19 to 0.42 by keeping the policy in-bounds. (The
original position-only fence scored 0.46 but let the pursuers overrun the
keep-in box by up to 0.77 m and touch the arena wall in 11/200 episodes; the
current fence acts on predicted position and holds the overrun to 0.23 m with
no wall contacts — a non-significant −0.035 for a real safety margin.) That is a strong argument for a
*separable* safety layer — it both guarantees the constraints and, as a bonus,
clips a learned policy's out-of-distribution excursions, without touching the
reward.

Two caveats keep this honest. The shield's contribution is **game-specific**: in
air defence it does little (`assault` 0.80 → 0.81) and in `escort` it now
costs 0.06 (0.53 → 0.47, unpaired — the fence change itself measured neutral
there, so the speed cap is the suspect), because the defender gate's scripted
branch rarely goes near the geofence in the first place. And it is not uniformly positive even in
tag — `pe-explore --game tag --seed 13` records an episode where the gated
controller captures both evaders in 48 steps *without* the shield and times out
*with* it, the geofence having deflected a converging intercept. It wins on
average and loses individual episodes; a per-episode trace is the only way to see
that, which is what §2.1 is for.

### 4.1 The bug the V&V suite could not see

Adding the two defence games meant factoring the airframe map out of the core, and
that refactor changed one line for `tag` as well: commands are now clipped by
**norm** rather than per component. The three actuators each take `[-1, 1]`, so
under the old rule a drone commanded `(1, 1, 1)` pulled `√3 · gear` and reached a
measured **7.42 m/s** against a documented `v_max = gear/damping = 5.25`. Thrust
was anisotropic; the speed limit was only true along the axes.

What makes this worth writing down is *how well it hid*. The scripted controllers
emit unit-norm direction vectors, so they never touched the corner of the action
cube: the tag baseline, `arena.xml`, the exact-start checksums and the
Python↔C++ parity results were **bit-identical under both rules**. Every check in
§5 passed, before and after — because both runtimes had the same clip. Parity
demonstrates that two implementations agree, not that either is right.

A learned policy, of course, fills the action cube, and PPO will happily find 73%
of free thrust. So the entire cost of the bug landed on exactly the numbers this
writeup is about. The then-shipped DAgger checkpoint — selected on a 40-episode
battery during training — scores **0.28 on that battery under the old rule and
0.125 under the fixed one**. Roughly half of an earlier draft's headline number
was the exploit, not the method.

The policy was then retrained from scratch under the corrected dynamics, and the
refit recovers most of the loss honestly: **0.20** on the independent 200-episode
battery, against the old checkpoint's 0.14 there. That refit is what every table
in this document now uses. It also produced a clean demonstration of the
selection bias described in §8 — it scored 0.30 on its own 40-episode selection
battery and 0.20 on the held-out 200.

The fix is kept: isotropic thrust is the correct physics and it makes the speed
bound the rest of the document quotes actually true. All tables above were
regenerated under it. What caught it was not a cross-check but a **physical
invariant** — measure the achievable top speed and compare it to the bound you
claim — and that is now a unit test in both runtimes
(`test_drone_thrust_is_isotropic`). The general lesson for a V&V suite: mirror
tests catch divergence, property tests catch shared mistakes, and you need both.

## 5. Two runtimes, cross-validated

The research stack is Python (MuJoCo + SB3 + py_trees); the **production runtime is
C++** (MuJoCo C API + **BehaviorTree.CPP** with a Groot2-editable tree + policy via
**ONNX Runtime** + the safety filter). Both load the same `arena.xml` and link the
same `libmujoco`. Parity is enforced two ways:

- **Exact-start checksums** (`pe_run --parity` vs the Python rollout) match to
  ~1e-6 — identical dynamics and scripted logic. On a shared 100-episode start
  set, C++ scripted and Python scripted score identically (0.48 / 0.48).
- **The neural policy and the gate match exactly too.** Both runtimes run the
  same ONNX file through the same ONNX Runtime version, and on a shared
  200-episode tag start set the full gated + shielded stack scores identically
  in C++ and Python (win 0.45, geofence/speed interventions 6974/4397 — every
  counter equal). Exact-start parity holds for every game × {scripted,
  bt_safe} and runs in CI.

  This section used to say the opposite: that the gated tag controller drifted
  (0.13 C++ vs 0.18 Python) because "discrete gate decisions amplify ~1e-6
  policy differences over 500 chaotic steps" — and that this was the honest
  limit of cross-runtime validation. It was a bug. The C++ side computed
  `v * (1.0 / n)` wherever numpy computes `v / n`; the two differ by one ulp in
  a fraction of cases, and a per-tick action dump located the first divergence
  at tick 12 of a gated chase. Scripted-only parity never exposed it because
  the scripted law happened not to hit a differing case on the fixed starts.
  Lesson, in the same vein as §4.1: a plausible physical story ("chaos") for a
  residual is not evidence; bisect it to the first differing number.

## 6. Different games, different winners

Tag has one weakness as evidence: it is a single problem, so "the gate trails
scripted" could just be a fact about *this* chase rather than about gating. So
the env grew two more games that reuse the same slots, controllers, observation
encoding, gate machinery, safety filter and metrics — only the **objective**,
the **airframes** and the **spawn geometry** change:

| game | pursuer slot | evader slot | objective | timeout favours |
|---|---|---|---|---|
| `tag` | drone pursuers (5.25 m/s) | drone evaders (4.75 m/s) | tag both evaders | evaders |
| `assault` | interceptor drones (7.5 m/s) | **missiles** (11 m/s) | stop the raid on a static asset | **defenders** |
| `escort` | interceptor drones (7.5 m/s) | **missiles**, long burn | protect a convoy in transit | **defenders** |

Two things flip at once, and both matter:

- **The speed edge inverts.** In tag the pursuer is *faster*, so a tail chase is
  a valid plan. In the defence games the attacker is ~47% faster, so a defender
  can never convert to a stern chase — it must solve a *crossing* intercept or
  lose. That single change makes the terminal problem geometric rather than
  reactive.
- **Attackers are missiles, not drones.** Heterogeneous dynamics are a pure
  command transform (`dynamics.py:apply_dynamics`), not a new physics path: a
  holonomic drone gets its command verbatim, while a missile keeps thrust on its
  velocity axis, can only spend `lat_authority` (0.45) of its command laterally,
  carries a finite burn, and is out of the fight once it coasts below stall.
  Turn radius therefore grows as `v²/a_lat` — a fast missile is committed, which
  is exactly the property the gate exploits.

<!-- GAMES:AUTOFILL -->

**tag** — pursuers (pursuer slot) vs evaders; win = both tagged

| controller | win rate | mean captures | steps→win | min sep (m) | geo/spd viol |
|---|---|---|---|---|---|
| `scripted` | 0.46 | 1.36 | 75 | 0.73 | 0/0 |
| `pursuer` | 0.00 | 0.00 | — | 5.94 | 0/0 |
| `pursuer_bc` | 0.01 | 0.02 | 261 | 2.21 | 0/0 |
| `pursuer_bc_ft` | 0.00 | 0.00 | — | 2.81 | 0/0 |
| `pursuer_dagger` | 0.20 | 1.06 | 154 | 0.74 | 0/0 |
| `bt_gated` | 0.19 | 1.10 | 138 | 0.74 | 0/0 |
| `bt_gated_safe` | 0.42 | 1.38 | 202 | 0.73 | 11970/4924 |


**assault** — defenders (pursuer slot) vs attackers; win = raid stopped

| controller | defended | breach rate | attackers down | burned out | closest approach (m) | geo/spd viol |
|---|---|---|---|---|---|---|
| `scripted` | 0.77 | 0.23 | 1.72 | 0.01 | 11.28 | 0/0 |
| `assault_dagger` | 0.04 | 0.96 | 0.26 | 0.01 | 2.44 | 0/0 |
| `bt_gated` | 0.80 | 0.20 | 1.74 | 0.00 | 11.53 | 0/0 |
| `bt_gated_safe` | 0.81 | 0.18 | 1.74 | 0.00 | 11.50 | 8/546 |


**escort** — escorts (pursuer slot) vs attackers; win = raid stopped

| controller | defended | breach rate | attackers down | burned out | closest approach (m) | geo/spd viol |
|---|---|---|---|---|---|---|
| `scripted` | 0.49 | 0.51 | 1.24 | 0.00 | 3.85 | 0/0 |
| `escort_dagger` | 0.25 | 0.75 | 0.70 | 0.00 | 2.26 | 0/0 |
| `bt_gated` | 0.53 | 0.47 | 1.31 | 0.00 | 4.20 | 0/0 |
| `bt_gated_safe` | 0.47 | 0.54 | 1.26 | 0.00 | 3.38 | 555/345 |

## 7. The air-defence gate is a *different* tree — and that is the result

The tag gate says: *close quarters is messy, hand it to the policy.* Porting that
instinct to air defence and measuring it is the most useful thing in this repo,
because **the instinct is wrong, and expensively so**. Gating on one predicate
at a time (`assault`, 150 episodes × 2 independent seed blocks, DAgger policy on
the RL branch, defender win rate):

| gate profile | seed block A | seed block B |
|---|---|---|
| scripted only | 0.54 | 0.56 |
| **`intercept_infeasible` → RL** | **0.70** | **0.71** |
| + `threat_imminent` → RL | 0.45 | 0.45 |
| + `close_quarters` → RL | 0.12 | 0.12 |
| RL only | 0.03 | 0.03 |

Handing close quarters to the policy costs **42 points** (0.70 → 0.12) — worse
than not gating at all, and nearly as bad as deleting the scripted controller.
The reason is that the terminal endgame against a missile is *not* a scrappy
dogfight. It is a precise geometry problem, and the lead-intercept law solves it
near-optimally while the policy cannot.

The regime actually worth handing over is the one where the scripted law is not
merely imprecise but **structurally undefined**: when the target is faster and
opening, `lead_intercept_time` has no positive root, so no constant-speed pursuit
closes it. The scripted controller's fallback is to park on a static gate point —
it has nothing better to offer, and *anything* is an improvement. So the shipped
profile (`bt/gating.py:_defender_predicates`, `cpp/trees/gate_defenders.xml`) is
two branches, not four.

This is the sharpest version of the project's thesis. The right gate boundary is
**not** "where is the world messy" but **"where does the scripted controller stop
being defined"** — and that is a property you can read off the controller's own
math rather than tune by intuition. Two consequences worth noting:

- **The same tree adapts across games without retuning.** On `assault` it routes
  15% of ticks to the policy; on `escort` it routes 59% — a moving asset makes
  intercept solutions infeasible far more often. The predicate is about the
  scripted law's validity, not a threshold fitted to one game.
- **The gate now wins outright**, which it never did in tag. On the 200-episode
  battery in §6 (pre-review): `assault` scripted 0.54 → gated **0.72**, with the safety filter
  costing nothing (0.72 → 0.72); `escort` scripted 0.40 → gated **0.46** (0.44
  shielded). The ablation above ran 150 episodes on two independent seed blocks
  and put the same profile at 0.70/0.71 — consistent, and a useful reminder of
  how wide the ±1σ band is at this sample size. The tag conclusion
  ("gating a merely-competent policy into a strong baseline's home turf doesn't
  buy wins") was right, and the defence games show its converse: find a regime
  the baseline genuinely cannot serve, and gating pays immediately.

### 7.1 The attacker gate: the same experiment, the opposite answer

The defender result above could be a story about *air defence*. To find out, the
identical procedure was run on the other side of the same game: train an attacker
policy by DAgger against the scripted raid doctrine
(`assault_attacker_dagger`, breach 0.33/0.35), then ablate the missile gate one
predicate at a time. Scored as **breach rate**, so higher is better for the
attacker:

| gate profile | seed block A | seed block B |
|---|---|---|
| **scripted only** | **0.46** | **0.57** |
| + `committed` → SCRIPTED | 0.46 | 0.57 |
| + `threat_close` → RL | 0.41 | 0.49 |
| + `threat_closing_fast` → RL | 0.37 | 0.38 |
| policy only | 0.33 | 0.35 |

Monotonically downhill. **Every handoff costs breach rate**; the best attacker
gate is no gate at all. So both RL branches were deleted from
`_attacker_predicates` and `cpp/trees/gate_attackers.xml`, which now route
everything to the scripted law — the surviving `committed` branch is a
no-op kept only because it states the run-in regime legibly in Groot2.

That is a more useful outcome than a second win would have been, because it
falsifies the cheap reading of §7. The defender gains 16 points from exactly one
handoff; the attacker gains nothing from any. Same game, same machinery, same
sample sizes, opposite verdicts — so the earlier result is not "gating works in
air defence". The difference is structural:

> The defender's scripted law has a regime where it is **undefined** — no positive
> lead-intercept root, so it parks on a static gate point and has nothing to
> offer. The missile's guidance law has no such regime. A committed terminal
> run-in is precisely the case a lead-guidance law is *for*, and the policy
> is strictly worse everywhere.

**Hand a regime to the policy where the scripted controller is undefined, not
merely where it is imprecise, and not where the situation merely feels messy.**
That predicate is readable off the controller's own math, which is why it
transfers between `assault` and `escort` without retuning and why the two
intuition-driven profiles (tag's four branches, the missile's three) both lost.
It is also a cheap test to apply before writing a gate at all: if the scripted
law is well-defined everywhere in the state space, a gate is unlikely to pay,
and the honest recommendation is to skip it — as it is here.

Both the removal and the reasoning are pinned by tests in both runtimes
(`test_attacker_gate_hands_over_nothing`, and its doctest mirror ticking the
shipped XML), so the profile cannot silently regrow the branches that lost.

### 7.2 Revisited: the regime was partly a bug in the scripted law

A later review asked two questions of §7 that its ablation could not, and both
answers changed the shipped code.

**Was the predicate the law's regime?** No — an approximation of it.
`intercept_infeasible` asked whether a *point* intercept of the *nearest*
threat exists. The scripted law actually switches on its *assigned* threat
(ordered by time-to-asset) and also falls back when the intercept would land
after the missile reaches the asset. Over 13,148 agent-ticks of the shipped
gate, 14% of handovers happened while the law still had an intercept, and the
law was parked on its gate point on 957 ticks (37% of its fallback) that the
gate left with it.

**Was the regime genuinely undefined?** Mostly not. A kill only needs the gap
under the 1.4 m capture radius, but the law solved for the point. Solving
|rel + v·t| = v_p·t + r instead — about fifteen lines — shrinks the "no
solution" band dramatically. A 3×3 matrix (law: r ∈ {0, 0.7, 1.4} m × gate:
none / old proxy / the law's own fallback flag) on seed blocks 10000 and
110000 picked r = 0.7 m; then on held-out seeds 20000+ (n = 200, paired):

| defender stack | `assault` | `escort` |
|---|---|---|
| point-solve law (§7's baseline) | 0.525 | 0.440 |
| point-solve law + `intercept_infeasible` gate (§7, was shipped) | 0.730 | 0.495 |
| sphere-solve law, no gate | 0.795 | 0.545 |
| sphere-solve law + gate on the law's own fallback flag (**shipped**) | **0.845** | **0.620** |

Three consequences:

- **Most of §7's +0.17 was a fix the scripted law could make itself.** The
  corrected law alone beats the old gated stack in `assault`. What the policy
  added was largely covering for a solver that called reachable geometry
  "impossible" — and that policy is itself a DAgger clone of the old law, so in
  the fallback regime it was a smoothed copy of the very behaviour the gate was
  bypassing.
- **The thesis survives, smaller and sharper.** Over the corrected law, gating
  on the law's *own* fallback flag still earns +0.05 (`assault`) and +0.075
  (`escort`), both intervals excluding zero. Gating on the old *proxy* over the
  same corrected law loses 13 points (`assault`) and 8 (`escort`): it hands the
  policy geometry the law now handles. "Hand over where the scripted law is
  undefined" holds — but it must be read off the law (`GuardDefenders.plan()`
  returns the flag the gate consumes), not off an approximation of it.
- **The first question to ask of an "undefined" regime is whether the scripted
  law is solving the right problem.** A gate can paper over a solver bug and
  measure as a win. The method in §7 — ablate one predicate at a time — cannot
  tell those apart; changing the scripted law can.

### Cross-runtime validation of the new games

Both defence games were re-validated under the CLAUDE.md invariant. On a shared
100-episode start set (`--starts-file`), Python and C++ agree **exactly** on
every metric — including the ONNX policy path, the BehaviorTree.CPP gate, and the
safety filter's individual violation counts:

| assault, shared starts | win | captures | breach | asset margin | geo/spd viol |
|---|---|---|---|---|---|
| `scripted` (py = cpp) | 0.54 | 1.40 | 0.46 | 8.3 | 0/0 |
| `rl` (py = cpp) | 0.05 | 0.25 | 0.95 | 2.4 | 0/0 |
| `bt_gated` (py = cpp) | 0.68 | 1.56 | 0.32 | 9.7 | 0/0 |
| `bt_gated_safe` (py = cpp) | 0.70 | 1.62 | 0.30 | 9.7 | 19/238 |

(Pre-review numbers; the defender law and gate changed in §7.2. Parity for the
current stack is exact and checked in CI — and tag now matches exactly too, §5.)

**One honest V&V caveat.** The geofence can only *request* a correction from a
missile. The safety filter emits a desired command, but the airframe transform
runs after it, and a missile with 0.45 lateral authority cannot comply with an
arbitrary correction. Missiles stay hard-bounded by their MuJoCo joint ranges —
so the invariant is enforced, but by the *model*, not by the shield. On a real
vehicle that distinction matters: a shield you cannot actuate is not a shield.

## 8. Limitations & next

- **Beating the baseline in tag.** The best learned configuration (DAgger + gate
  + shield) reaches 0.42 (0.46 with the old, leakier fence), near scripted's
  0.46 but not beating it —
  and it gets there mostly via the safety filter rather than via the gate, since
  DAgger alone is 0.20 and the unshielded gate 0.19. Calling that "parity with the
  baseline" would be generous: it is a learned policy rescued by a hand-written
  constraint layer. More DAgger iterations or a stronger network would help. A
  *residual*
  policy (learning a correction on top of the scripted action) is the obvious
  textbook suggestion here; §7.2 is a reason to try the cheaper thing first —
  check whether the scripted law is solving the right problem. In air defence
  that one fix was worth more than the whole learned branch.
- **Attacker teams are always scripted *in the defender numbers*.** §7.1 now
  trains and ablates an attacker policy, but every *defender* win rate in §6/§7
  is still measured against the scripted raid doctrine. The attacker policy came
  out weaker than that doctrine (breach 0.33 vs 0.46), so it would not have made
  a harder opponent — but the defence win rates should still be read as "against
  this raid doctrine", not as absolute. Genuine co-training, where both sides
  adapt to each other's gate, is the open experiment.
- **No error bars, and a known selection bias.** Every table is a point estimate.
  The ±1σ band at 150–200 episodes is a few points, so differences of that size
  (e.g. `assault` 0.80 vs 0.81 shielded) are not findings; only the large gaps
  are. Worse, checkpoints are chosen best-of-N on a small battery, which is
  biased high: the tag DAgger refit scored 0.30 on its 40-episode selection
  battery and 0.20 on a held-out 200. Selection and reporting batteries are
  disjoint here, but the honest fix is repeated seeds with intervals.
- **Reward shaping.** The self-play failure is partly a shaping artefact (a
  per-step time penalty that outweighs the distance-closing bonus); fixing that is
  the honest way to give from-scratch RL a fair shot.
- **Dynamics.** Point-mass, no attitude/actuation limits. A drone model would make
  the terminal-intercept sub-task richer and blunt the scripted lead's optimality.
- **Learned gating.** The gate thresholds are hand-set; comparing hand-set vs
  learned gating boundaries — and gating on *predicted* policy advantage — is the
  natural follow-up.
