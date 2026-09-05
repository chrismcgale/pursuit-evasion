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
