# When does the tree win, when does the policy win, and when does gating win?

**TL;DR.** Across three games — symmetric `tag` and two air-defence problems
(`assault`, `escort`) — a hand-written behaviour tree built on lead-intercept
geometry + explicit target assignment is a **strong, cheap, legible baseline**,
and whether gating a learned policy beats it depends entirely on whether you can
name a regime the baseline does not serve.

- **In `tag` the baseline wins** (~46% capture of random starts) and no learned
  method here beats it; gating tracks and slightly trails it.
- **In air defence the gate wins outright** — `assault` 0.54 → **0.72**,
  `escort` 0.40 → **0.46** — using the *same* machinery and no retuning.
- **The gate profile does not transfer, and the failure is expensive.** Porting
  tag's "close quarters is messy, hand it to the policy" instinct to air defence
  costs **42 points** (0.70 → 0.12) — worse than not gating at all. The boundary
  that pays is not *"where is the world messy"* but *"where does the scripted law
  stop being defined"*, which you can read off the controller's own math (§7).

The learned-side results are a compact tour of when each technique works:

- **RL from scratch (self-play PPO) fails** (0%) — the capture reward is too
  sparse for a random policy to bootstrap in the budget used.
- **Behaviour cloning fails despite near-perfect supervised loss** (MSE 0.003,
  yet ~0% capture) — the textbook covariate-shift failure: it drifts into states
  the expert never showed it and blends between the two evaders instead of
  committing.
- **DAgger fixes exactly that** — relabelling the learner's *own* visited states
  with the expert takes the same network from **0% → ~28%** capture. That is the
  headline: the *method*, not the budget, was the problem.
- **Gating only helps where the policy actually dominates.** In tag the gate
  hands close-quarters/juking/contested ticks to the (now competent) DAgger
  policy, but DAgger still isn't *better* than scripted at the precision endgame,
  so the gate tracks — and slightly trails — pure scripted. Gating a
  merely-competent policy into a strong baseline's home turf doesn't buy wins.
  The defence games are the converse of this, and the reason it's a statement
  about *regimes* rather than about gating.
- **The safety filter is a surprise net-positive.** Enabling the runtime shield on
  the gated controller more than **doubles** its win rate (0.11 → 0.28) by keeping
  the learned policy inside the geofence/speed envelope instead of wandering to the
  walls — safety here *improves* task performance rather than costing it.
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
| `dagger` | 0.14 | 1.09 | 206 | 0.73 | 0/0 |
| `bt_gated` | 0.11 | 1.06 | 137 | 0.74 | 0/0 |
| `bt_gated_safe` | 0.28 | 1.23 | 175 | 0.74 | 6193/5833 |

**Terminal intercept (endgame starts)**

| controller | win rate | mean captures | steps→win | min sep (m) | geo/spd viol |
|---|---|---|---|---|---|
| `scripted` | 0.45 | 1.27 | 114 | 0.75 | 0/0 |
| `rl_selfplay` | 0.00 | 0.00 | — | 3.24 | 0/0 |
| `bc_distill` | 0.00 | 0.00 | — | 2.35 | 0/0 |
| `dagger` | 0.17 | 1.11 | 238 | 0.74 | 0/0 |
| `bt_gated` | 0.12 | 1.07 | 186 | 0.74 | 0/0 |

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
  from **0% → 14%** standalone (1.09 mean captures; 0.17 on the endgame battery).
  This is *when imitation works*: on-policy relabelling, not more offline data.
  It is still well short of scripted — see §4.1 for why an earlier draft of this
  writeup claimed 28%.
- **Gating → only as good as the policy you gate.** `bt_gated` routes the messy
  regimes to `dagger`, but `dagger` is competent, not superior to scripted in
  those regimes, so the gate lands at 0.11 — below pure scripted. The honest
  lesson: a gate is worth it precisely when the learned policy *dominates* some
  slice; identifying (or training) such a slice is the prerequisite, and a strong
  scripted baseline raises that bar. §7 is what happens when you find one.
- **Safety filter → unexpectedly wins some.** `bt_gated_safe` (0.28) more than
  doubles `bt_gated` (0.11): the DAgger policy periodically drives toward the
  arena walls, and the geofence/speed shield redirects it back into productive
  space. A layer we added for V&V turned out to also regularise a learned policy's
  worst habits — see §4.

## 4. Safety / V&V

The safety filter is a runtime shield *below* the BT and the policy: an operational
geofence (keep-in box inside the arena) and a speed cap, either of which can
override the commanded action. It is a pure function, mirrored in Python and C++,
and unit-tested on both sides; every BT node has a unit test (`cpp/tests`,
`tests/test_env.py`). Unlike the usual "safety costs a little performance" story,
here it *helped*: on the gated DAgger controller it fired often (thousands of
geofence/speed corrections over 200 episodes) and nearly doubled the win rate by
keeping the policy in-bounds. That is a strong argument for a *separable* safety
layer — it both guarantees the constraints and, as a bonus, clips a learned
policy's out-of-distribution excursions, without touching the reward.

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
writeup is about. The shipped DAgger checkpoint — selected on a 40-episode
battery during training — scores **0.28 on that battery under the old rule and
0.125 under the fixed one**. Roughly half of the "DAgger reaches 28%" headline in
an earlier draft was the exploit, not the method.

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
- **The neural policy matches too.** ONNX matches SB3 to ~1e-6 per step
  (`export_onnx` parity), and on the shared start set the pure DAgger policy scores
  0.16 (C++) vs 0.15 (Python) — within one episode. The gated controller sits at
  0.13 (C++) vs 0.18 (Python): discrete per-tick gate decisions amplify the ~1e-6
  policy differences over 500 chaotic steps into a handful of flipped *binary*
  outcomes, while continuous metrics (min-sep 0.74) stay identical. That is the
  honest limit of cross-runtime validation for a chaotic closed loop — per-step
  equivalence, aggregate agreement, not bit-identical episode outcomes once a
  learned policy is in the loop.

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
| `pursuer_dagger` | 0.14 | 1.09 | 206 | 0.73 | 0/0 |
| `bt_gated` | 0.11 | 1.06 | 137 | 0.74 | 0/0 |
| `bt_gated_safe` | 0.28 | 1.23 | 175 | 0.74 | 6193/5833 |


**assault** — defenders (pursuer slot) vs attackers; win = raid stopped

| controller | defended | breach rate | attackers down | burned out | closest approach (m) | geo/spd viol |
|---|---|---|---|---|---|---|
| `scripted` | 0.54 | 0.46 | 1.39 | 0.01 | 8.15 | 0/0 |
| `assault_dagger` | 0.04 | 0.96 | 0.26 | 0.01 | 2.44 | 0/0 |
| `bt_gated` | 0.72 | 0.28 | 1.61 | 0.01 | 10.13 | 0/0 |
| `bt_gated_safe` | 0.72 | 0.28 | 1.61 | 0.01 | 9.93 | 72/484 |


**escort** — escorts (pursuer slot) vs attackers; win = raid stopped

| controller | defended | breach rate | attackers down | burned out | closest approach (m) | geo/spd viol |
|---|---|---|---|---|---|---|
| `scripted` | 0.40 | 0.60 | 1.05 | 0.00 | 2.93 | 0/0 |
| `escort_dagger` | 0.25 | 0.75 | 0.70 | 0.00 | 2.26 | 0/0 |
| `bt_gated` | 0.46 | 0.54 | 1.13 | 0.00 | 3.14 | 0/0 |
| `bt_gated_safe` | 0.44 | 0.56 | 1.11 | 0.00 | 2.89 | 385/245 |

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
  battery in §6: `assault` scripted 0.54 → gated **0.72**, with the safety filter
  costing nothing (0.72 → 0.72); `escort` scripted 0.40 → gated **0.46** (0.44
  shielded). The ablation above ran 150 episodes on two independent seed blocks
  and put the same profile at 0.70/0.71 — consistent, and a useful reminder of
  how wide the ±1σ band is at this sample size. The tag conclusion
  ("gating a merely-competent policy into a strong baseline's home turf doesn't
  buy wins") was right, and the defence games show its converse: find a regime
  the baseline genuinely cannot serve, and gating pays immediately.

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

That is a stronger parity result than tag's, where a learned policy in a 500-step
chaotic chase flipped a handful of binary outcomes (§5). The defence episodes are
shorter (200 steps) and more geometrically determined, so ~1e-6 per-step
differences don't get the chance to compound.

**One honest V&V caveat.** The geofence can only *request* a correction from a
missile. The safety filter emits a desired command, but the airframe transform
runs after it, and a missile with 0.45 lateral authority cannot comply with an
arbitrary correction. Missiles stay hard-bounded by their MuJoCo joint ranges —
so the invariant is enforced, but by the *model*, not by the shield. On a real
vehicle that distinction matters: a shield you cannot actuate is not a shield.

## 8. Limitations & next

- **Beating the baseline in tag.** DAgger reached ~28%, not scripted's ~46%.
  More DAgger iterations or a stronger network would narrow that. A *residual*
  policy (learning a correction on top of the scripted action) is the obvious
  textbook suggestion here, but §7 is a reason to be sceptical of it as the first
  move: residual learning assumes the scripted action is a good basis everywhere
  and only needs nudging, whereas the measurable win came from identifying the
  regime where the scripted action is not defined at all and replacing it
  wholesale. A residual would smear a correction across both regimes, which is
  closer to the 4-branch gate that lost 42 points than to the 2-branch one that
  won. Worth trying, but *after* the regime analysis, not instead of it.
- **The attacker gate profile is unvalidated.** `gate_attackers.xml` and
  `_attacker_predicates` are design intent, not a measured result: no raid policy
  has been trained, so there is nothing to gate yet. Unlike the defender profile,
  those thresholds have not earned their place and are flagged as such in both
  runtimes. Training an attacker policy and running the same single-predicate
  ablation is the direct next experiment.
- **Attacker teams are always scripted.** Every number here evaluates the
  pursuer/defender slot against a *scripted* opponent. A co-trained raid would
  likely find the gate's seams, and the defence win rates should be read as
  "against this raid doctrine", not as absolute.
- **Reward shaping.** The self-play failure is partly a shaping artefact (a
  per-step time penalty that outweighs the distance-closing bonus); fixing that is
  the honest way to give from-scratch RL a fair shot.
- **Dynamics.** Point-mass, no attitude/actuation limits. A drone model would make
  the terminal-intercept sub-task richer and blunt the scripted lead's optimality.
- **Learned gating.** The gate thresholds are hand-set; comparing hand-set vs
  learned gating boundaries — and gating on *predicted* policy advantage — is the
  natural follow-up.
