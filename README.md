# pursuit-evasion

A **2v2 pursuit–evasion** testbed in 3D (MuJoCo) for studying **when a hand-written
behaviour tree beats a learned policy, when the policy beats the tree, and when
gating one with the other beats both** — see
[`writeup/when-each-wins.md`](writeup/when-each-wins.md).

It carries a small but complete autonomy stack: scripted tactics → self-play RL
(SB3 PPO) → imitation learning + RL fine-tune → **BT gating** → a runtime **safety
filter** → an instrumented scenario harness → and a **C++ production runtime**
(MuJoCo C API + **BehaviorTree.CPP** with Groot2 + policy via **ONNX Runtime**).

Architecture & invariants: [`CLAUDE.md`](CLAUDE.md). Build plan & rubric mapping:
[`docs/PLAN.md`](docs/PLAN.md). Runbook: [`.claude/skills/pursuit-evasion/SKILL.md`](.claude/skills/pursuit-evasion/SKILL.md).

## Quickstart (Python)

```bash
uv sync                                              # Python 3.12 venv + deps (CPU torch)
uv run pytest -q                                     # V&V: env, safety, BT gate
uv run pe-train --generations 5 --steps-per-gen 30000 --n-envs 8   # self-play both teams
uv run python -m pursuit_evasion.train.imitation --expert bt \
    --model models/pursuer.zip --finetune-steps 60000             # BC from BT + RL fine-tune
uv run pe-eval --models models --episodes 200        # tournament: the "when each wins" table
uv run pe-viz --controller bt --model models/pursuer.zip --out rollout.mp4
```

## Games, gate analysis, and the explorer

Three games share one simulator, one observation layout and one set of
controllers — `tag` (symmetric drones) plus two air-defence problems, `assault`
and `escort`, where attackers fly missile dynamics and defenders fly drones.

```bash
uv run pe-games --games tag assault escort --episodes 200   # cross-game tournament
uv run pe-ablate --game assault --side pursuers             # what each gate branch is worth
uv run pe-explore --game assault --seed 3                   # -> self-contained HTML viewer
```

`pe-ablate` walks a gate profile one predicate at a time on identical seeds; it is
how every shipped profile was derived, and both defence profiles came out
*smaller* than the intuition that motivated them. `pe-explore` writes a
dependency-free HTML page that replays an episode step by step, showing which
branch fired, every predicate's live value, what the scripted controller and the
policy each wanted, and where the safety filter intervened.

Headline results (200 episodes; full detail and caveats in the writeup):

| | scripted | best learned | BT-gated | + safety filter |
|---|---|---|---|---|
| `tag` (capture rate) | **0.46** | 0.20 | 0.19 | 0.46 |
| `assault` (asset held) | 0.54 | 0.04 | **0.72** | 0.72 |
| `escort` (convoy through) | 0.40 | 0.25 | **0.46** | 0.44 |

The short version: gating pays exactly where the scripted controller is
*structurally undefined*, not where the situation merely looks messy. In `tag`,
where lead-intercept is well-defined everywhere, gating buys nothing and the
safety filter does the work. In air defence there is a regime with no valid
intercept solution, and handing precisely that regime to the policy is worth 16+
points. Run the same procedure on the attacker side and it says *don't gate at
all* — which is why the shipped missile profile has no learned branch.

## C++ production runtime

```bash
cd cpp
cmake -S . -B build -G Ninja -DCMAKE_POLICY_VERSION_MINIMUM=3.5    # first configure fetches BT.CPP + doctest
ninja -C build
./build/pe_tests                                                   # BT-node + safety unit tests
uv run python -m pursuit_evasion.train.export_onnx ../models/pursuer.zip ../models/pursuer.onnx
./build/pe_run --controller bt --onnx ../models/pursuer.onnx       # MuJoCo + BT.CPP gate + ONNX policy + safety
./build/pe_run --parity                                            # cross-runtime numerical parity vs Python
```
