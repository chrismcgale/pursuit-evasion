# Build plan & rubric mapping

The project deliberately hits six capabilities. "BT gates a policy" alone is ~4/6;
the imitation-learning handoff, the instrumentation harness, and the safety/V&V
layer take it to 6/6.

| # | Capability                         | Where it lives                                              | Status |
|---|------------------------------------|-------------------------------------------------------------|--------|
| 1 | 3D pursuit-evasion env             | `src/pursuit_evasion/env` (MuJoCo), `cpp/src/sim` (C API)    | ✅ py + cpp (1e-6 parity) |
| 2 | Behaviour tree gating a policy     | `src/…/bt` (py_trees ref) + `cpp/src/bt` (**BehaviorTree.CPP + Groot2**) | ✅ ref + prod |
| 3 | Learned policy (RL, self-play)     | `src/…/train/selfplay.py` (SB3 PPO)                         | ✅ trained (0% — see writeup) |
| 4 | Imitation learning (+ DAgger)      | `src/…/train/imitation.py`, `train/dagger.py`               | ✅ BC 0% → DAgger 28% |
| 5 | Instrumentation / scenarios        | `src/…/eval/scenarios.py` (batched seeds, metrics, replay logs) | ✅ |
| 6 | Safety filter + V&V                 | `src/…/safety.py` + `cpp/include/pe/safety.hpp` + unit tests per BT node | ✅ (Py 36/36, C++ 31/31) |
| + | Multiple games, heterogeneous dynamics | `src/…/env/games.py` + `env/dynamics.py`, mirrored in `cpp/include/pe/` | ✅ tag / assault / escort |

## Runtime architecture (full C++ path)

```
        TRAIN (Python)                         RUN (C++, production)
  ┌───────────────────────────┐        ┌──────────────────────────────────┐
  │ MuJoCo (py)  SB3 PPO       │        │ MuJoCo C API   BehaviorTree.CPP   │
  │ self-play + IL             │        │ scripted C++   ONNX Runtime       │
  │        │                   │        │        │            │            │
  │  export policy → ONNX ─────┼───────▶│   BT gate ── picks scripted / ONNX│
  │  dump arena.xml ───────────┼───────▶│        │                          │
  └───────────────────────────┘        │   SAFETY FILTER (geofence+speed)  │
                                        │        │ overrides both           │
                                        │   MuJoCo step ── log metrics      │
                                        │   Groot2 live viz (zmq) / XML     │
                                        └──────────────────────────────────┘
```

Identical physics both sides: the C++ runtime loads the same `arena.xml` dumped
from Python and links the same `libmujoco.so.3.12.0`. The BT logic, scripted
controllers, and observation encoding are mirrored and cross-checked by a parity
test (aggregate capture rate must match within noise; explicit-start replays
must match closely).

## The "when each wins" experiment

Controllers compared on the same seeded scenario battery (side = pursuers, held
against a fixed scripted evader, and vice-versa):

* **Scripted** — lead-intercept + potential-field. Strong in clean geometry.
* **RL (self-play PPO)** — strong in contested/close-quarters micro-play.
* **BC (imitates BT)** — cheap policy that clones the gated controller.
* **BC→PPO finetune (terminal intercept)** — surpasses BT on the hard sub-task.
* **BT-gated** — routes each tick to scripted or policy by situation.
* **BT-gated + safety filter** — same, with the runtime shield enabled.

Metrics: win/capture rate, time-to-intercept, **min separation**, constraint
violations (geofence/speed), plus replayable failure logs.

## Three games, so the answer isn't about one chase

`tag` alone can't distinguish "gating loses here" from "gating loses". Two
objective games reuse the same slots and machinery — only the objective,
airframes and spawn geometry change:

| game | pursuer slot | evader slot | win condition | timeout favours |
|---|---|---|---|---|
| `tag` | drones, 5.25 m/s | drones, 4.75 m/s | tag both evaders | evaders |
| `assault` | interceptors, 7.5 m/s | **missiles**, 11 m/s | asset never breached | defenders |
| `escort` | interceptors, 7.5 m/s | **missiles**, long burn | convoy reaches its goal | defenders |

The speed edge **inverts**: a defender is slower than the missile it must stop,
so a tail chase is impossible and the terminal problem becomes geometric rather
than reactive. That is what makes the gate boundary move — and measuring where it
moves to is the project's sharpest result (`writeup/when-each-wins.md` §7):
porting tag's "close quarters → policy" instinct to air defence costs 42 points,
while gating on *"the scripted law has no solution here"* takes 0.54 → 0.70.
