---
name: pursuit-evasion
description: Runbook for the 2v2 pursuit-evasion project — train, evaluate, build the C++ BT.CPP runtime, and cross-validate. Use when running, debugging, or extending this repo.
---

# pursuit-evasion runbook

Architecture + invariants: see `CLAUDE.md` (don't duplicate here). Build plan +
rubric mapping: `docs/PLAN.md`.

## Run

Python (train / research), from repo root:
```bash
uv sync                                   # Python 3.12 venv + deps (CPU torch)
uv run pytest -q                          # Python tests
uv run pe-train --generations 5 --steps-per-gen 30000 --n-envs 8   # self-play
uv run python -m pursuit_evasion.train.imitation --expert bt \
    --model models/pursuer.zip --finetune-steps 60000              # BC + RL fine-tune
uv run pe-eval --models models --episodes 200                      # tag tournament table
uv run pe-viz --controller bt --model models/pursuer.zip --out rollout.mp4
uv run python -m pursuit_evasion.train.export_onnx models/pursuer.zip models/pursuer.onnx
```

Games (`tag` | `assault` | `escort`). Most entry points take `--game`; the defence
games use the `<game>_dagger` model stem, not `pursuer*`:
```bash
uv run python -m pursuit_evasion.train.dagger --game assault --iters 6
uv run python -m pursuit_evasion.train.export_onnx models/assault_dagger.zip models/assault_dagger.onnx
uv run pe-games --games tag assault escort --episodes 200   # cross-game tournament, autofills the writeup
uv run pe-viz --game escort --controller bt --out escort.mp4
uv run python -m pursuit_evasion.env.dump_arenas --check    # cpp/assets/*.xml are GENERATED
```

C++ (production runtime), from `cpp/`:
```bash
cmake -S . -B build -G Ninja -DCMAKE_POLICY_VERSION_MINIMUM=3.5   # first configure fetches BT.CPP + doctest
ninja -C build
./build/pe_tests                                                  # BT-node + dynamics + safety unit tests
./build/pe_run --controller bt --side pursuers --onnx ../models/pursuer.onnx
./build/pe_run --game assault --controller bt_safe                # arena/tree/onnx all default per game
./build/pe_run --controller bt --groot                            # live tree view (Groot2 :1667)
```

Cross-runtime validation — the CLAUDE.md invariant. Run after touching dynamics,
observations, scripted controllers, gate predicates or the safety filter:
```bash
# 1. exact same-start checksums; both halves print the same 5 lines (tag only)
uv run pe-parity                                                  # from repo root
./build/pe_run --parity                                           # from cpp/

# 2. aggregate parity per game — catches gate / ONNX / shield drift
uv run python -m pursuit_evasion.eval.dump_starts                 # writes s_<game>.txt
./build/pe_run --game assault --controller bt_safe --starts-file /tmp/s_assault.txt
```
Compare against the same cell from `pe-games`. On the **defence** games these
match *exactly*, including violation counts. On **tag**, expect a few flipped
binary outcomes in long chaotic chases — judge on outcomes, not on the tail of
`steps2win`.

## Debug / common errors

- **ONNX export fails with `onnxscript`/dynamo trace error** — use the legacy
  exporter (`dynamo=False`) and compute the action head directly (already done in
  `export_onnx.py`). Don't route through the distribution object.
- **C++ segfault in `OnnxPolicy` ctor** — a `TypeInfo` temporary was let die
  before `GetShape()`. Keep it in a named local (see `onnx_policy.cpp`).
- **CMake configure error: "Compatibility with CMake < 3.5 has been removed"** —
  doctest's CMakeLists; pass `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`.
- **Build error in BT.CPP's bundled lexy (`-Wtemplate-body`)** — GCC 16 + BT.CPP
  `-Werror`; we disable `-Werror` on the `behaviortree_cpp` target in CMakeLists.
- **Empty training log** — Python block-buffers stdout to files; run with
  `PYTHONUNBUFFERED=1`. Also: don't run heavy `ninja` builds during training, they
  starve the DummyVecEnv rollouts (single-process).
- **Capture rate looks "too low"** — intended (~40–50% scripted). See CLAUDE.md.
- **MuJoCo "mass and inertia of moving bodies must be larger than mjMINVAL"** —
  every jointed body needs inertia; the x/y wrapper bodies carry a tiny inertial.

## Architecture (brief)

`env/` MuJoCo core + gym self-play wrapper · `scripted/` intercept + field
controllers · `bt/` py_trees gate (reference) · `train/` self-play, imitation,
onnx export · `eval/` scenario harness, tournament, viz · `safety.py` shield.
`cpp/` mirrors env+scripted+features+safety+gate in C++ with BehaviorTree.CPP +
ONNX Runtime + MuJoCo C API. Same `arena.xml`, same `libmujoco`.

**Games are a layer, not a fork.** One simulator ("team A can neutralise team
B"); a `GameSpec` (`env/games.py`, `cpp/include/pe/games.hpp`) adds the objective,
airframes and spawn geometry. Defenders reuse the *pursuer slot* and attackers the
*evader slot*, so every controller, observation, metric, gate and the safety
filter carry over unchanged. Heterogeneous dynamics are a **pure command
transform** (`apply_dynamics`), not a second physics path: holonomic drones pass
through, a missile projects its command onto its velocity axis with capped
lateral authority. Adding a game = one `GameSpec` per runtime + a generated arena
XML + a gate tree.

## Gotchas

- SB3 is single-agent → self-play trains two PPO policies against growing pools of
  frozen snapshots; each sub-env samples an opponent per reset (`PoolController`).
  Snapshots are serialised through a buffer (deepcopy fails on non-leaf tensors).
- The BT gate ticks **per agent per step**; it reads `AgentFeatures` from the
  blackboard. Any new predicate must be added to BOTH `bt/gating.py` and
  `cpp/src/bt_nodes.cpp` + the tree XML, and unit-tested.
- `cpp/assets/*.xml` are committed but generated — run `dump_arenas --check` after
  any geometry change. `arena.xml` must stay byte-identical unless you mean to
  move the tuned tag baseline.
- **Don't port the tag gate profile to the defence games.** Measured: adding
  tag's `close_quarters -> RL` branch to the defender gate drops it 0.70 to 0.12.
  The shipped defender gate is deliberately two branches; the ablation is in
  `_defender_predicates`' docstring and `trees/gate_defenders.xml`'s header, and
  `cpp/tests/test_games.cpp` fails if someone re-adds it.
- **Model stems differ per game**: tag uses `pursuer*`, the defence games use
  `<game>_dagger`. Encoded in `eval/games.py:_model_stems` and
  `cpp/src/runner.cpp:default_onnx` — change both together.
- **Command clipping is by norm, never per component** (`apply_dynamics`). A
  per-component clip silently gives diagonal commands 1.73x thrust — invisible to
  scripted controllers (unit-norm directions) and to the cross-runtime parity
  check, but learned policies find it. If a policy's win rate moves after a
  dynamics change, check the top speed against `gear/damping` first.
- The safety filter can only *request* a correction from a missile: the airframe
  transform runs after the shield, and 0.45 lateral authority may not be enough to
  comply. Missiles stay bounded by their MuJoCo joint ranges — by the model, not
  by the shield. State that plainly in any V&V discussion.

## Changelog / decisions (newest first)

- **2026-09-05** **Fixed anisotropic drone thrust — it moved every learned tag
  number.** The command is now clipped by norm; the pre-multi-game code clipped
  per component, so a diagonal command pulled 1.73x thrust and topped out at
  7.42 m/s against a documented v_max of 5.25.
  - Scripted controllers emit unit-norm directions, so the tag baseline, the
    arena XML and every parity check were bit-identical either way — the bug was
    **invisible to the entire V&V suite**. Learned policies fill the action cube
    and had been exploiting it: the shipped DAgger checkpoint scores 0.28 under
    the old rule and **0.125 under the fixed one** on its own selection battery.
  - Kept the fix (correct physics; it makes the documented speed limits true),
    regenerated `results.json`, and corrected the writeup's headline DAgger claim
    from "0% to ~28%" to the honest number. `test_drone_thrust_is_isotropic` pins
    it in both runtimes.
  - **Lesson: parity between two runtimes proves they agree, not that either is
    right.** Both had the same clip. The check that caught it was a physical
    invariant (top speed vs the documented bound), not a cross-check.
- **2026-09-05** **Multi-game + heterogeneous dynamics.** Added `assault` (point
  defence, static asset) and `escort` (convoy in transit) alongside `tag`, reusing
  the pursuer/evader *slots* so every controller, metric, gate and the safety
  filter carry over — only objective/airframe/spawn geometry differ. Attackers are
  **missiles** (thrust on the velocity axis, `lat_authority` 0.45, finite burn,
  stall-out), defenders are faster-turning but ~40% slower **drones**; implemented
  as a pure command transform (`env/dynamics.py`, `cpp/include/pe/dynamics.hpp`),
  not a second physics path. Asset is a MuJoCo **mocap** body.
  - **Key finding — the defender gate is NOT the tag gate.** Single-predicate
    ablation on `assault` (150 eps × 2 seed blocks): scripted 0.54/0.56;
    `intercept_infeasible`→RL **0.70/0.71**; +`threat_imminent` 0.45; +
    `close_quarters` **0.12**; RL-only 0.03. Porting tag's "close quarters is
    messy → hand it to the policy" costs **42 points**. Shipped profile is 2
    branches: hand over only where the scripted law is *structurally undefined*
    (no positive lead-intercept root → it parks on a static gate point).
    **Don't re-add the close-quarters branch** — the ablation is in the docstring
    of `_defender_predicates` and the header of `trees/gate_defenders.xml`, and
    `cpp/tests/test_games.cpp` pins it.
  - Gate now **wins outright**, unlike tag: assault scripted 0.54 → gated 0.70;
    escort 0.41 → 0.50. Same tree, no retuning: rl_share 0.15 (assault) vs 0.59
    (escort), because a moving asset makes intercepts infeasible more often.
  - **Cross-runtime parity is exact** on both defence games (shared `--starts-file`),
    including the ONNX path, BT.CPP gate and per-counter safety violations
    (19/238). Better than tag's, whose 500-step chaotic chases flip a few binary
    outcomes.
  - Added `pe-parity` (`eval/parity.py`) — the Python half of `pe_run --parity`,
    which CLAUDE.md claimed but was never checked in. Both sides now print
    matching checksums for the same 5 fixed starts.
  - Corrected an overstated comment: substep capture checking is **not** "missiles
    are fast so they tunnel" (per control step tag closes 1.43 capture radii,
    assault only 0.79). It is that a slower defender can never tail-chase, so every
    engagement is a high-offset **crossing pass** whose chord through the capture
    sphere is far shorter than its diameter. Tag stays once-per-step because its
    40–50% baseline is calibrated on that.
  - Attacker profile (`gate_attackers.xml`) is **unvalidated design intent** — no
    raid policy trained yet. Flagged in both runtimes; don't cite its thresholds.

- **2026-09-05** Learned-policy results in. Self-play PPO from scratch = 0%
  (capture-sparse reward + budget). BC of the scripted expert = 0% *despite MSE
  0.003* (covariate shift + two-target blending). PPO fine-tune drifted it worse.
  **DAgger fixed it: 0% → ~28%** (`train/dagger.py`) — expert relabelling of the
  learner's own states. Gate routes to the best policy (`dagger`) but doesn't beat
  scripted (0.46), because DAgger is competent, not superior, in the gated regimes.
  Surprise: the **safety filter nearly doubled** the gated policy (0.16→0.32) by
  keeping it in-bounds. Cross-runtime: scripted exact (0.48/0.48), pure policy
  0.16(cpp)/0.15(py), gated 0.13/0.18 (chaotic amplification of 1e-6 ONNX diffs
  through discrete gate ticks). Full analysis in `writeup/when-each-wins.md`.
  Takeaway for extending: to make the gate *win*, DAgger must surpass scripted in
  some slice (more iters / residual policy on top of the scripted action).

- **2026-09-04** Initial build. Chose MuJoCo over Genesis (mature, CPU-friendly,
  clean C API for the C++ port). CPU torch pinned (Blackwell sm_120 needs cu128;
  MLP PPO is CPU-bound anyway). Full C++ runtime with BehaviorTree.CPP + Groot2 +
  ONNX Runtime per the "industry-standard tooling" requirement. Added IL (BC from
  BT) + PPO fine-tune on terminal intercept, the scenario/instrumentation harness,
  and the safety filter with per-node unit tests. Cross-runtime parity validated
  to ~1e-6 via exact-start checksums.
