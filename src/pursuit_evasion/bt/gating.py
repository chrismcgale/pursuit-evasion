"""A py_trees behaviour tree that gates a learned policy with scripted tactics.

Thesis encoded by the tree: a hand-written controller is excellent in *clean*
situations (an open-space lead-intercept is near-optimal; a straight sprint away
from a distant threat is fine) but brittle in *messy* ones (close-quarters
reversals, a hard-juking target, an ambiguous two-target assignment). A learned
policy is the opposite — strong at reactive, contested micro-play, wasteful when
the geometry is trivial. The tree routes each agent, every tick, to whichever
controller owns the current regime.

Per-agent decision (pursuer example), as a py_trees Selector (first success wins):

    Selector "gate"
      ├─ Sequence  close-quarters?  → MODE = RL      (reactive dogfight)
      ├─ Sequence  target juking?   → MODE = RL      (scripted lead overshoots)
      ├─ Sequence  contested?       → MODE = RL      (ambiguous assignment)
      ├─ Sequence  clean intercept? → MODE = SCRIPTED (open-space geometry)
      └─ MODE = SCRIPTED                              (default: approach/search)

The evader tree mirrors this: RL when a pursuer is close or closing fast,
scripted potential-field flee when threats are far.

The objective games (``assault`` / ``escort``) add two more profiles, because
the regime where the scripted controller is weak is a *different* regime:

    Selector "gate[defender]"
      ├─ Sequence  intercept infeasible? → MODE = RL       (scripted falls back
      │                                                     to a static gate)
      ├─ Sequence  threat imminent?      → MODE = RL       (terminal endgame)
      ├─ Sequence  close quarters?       → MODE = RL
      ├─ Sequence  clean intercept?      → MODE = SCRIPTED (lead geometry works)
      └─ MODE = SCRIPTED

The attacker (missile) profile hands over **nothing** — measured, not assumed.
Every RL branch tried there lost breach rate, because a committed run-in is
exactly what a guidance law is for. See ``_attacker_predicates``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import py_trees
from py_trees.common import Status

from ..env.core import TEAM_PURSUERS, TeamView
from ..scripted.base import BaseController, Controller
from .features import AgentFeatures, agent_features

MODE_SCRIPTED = "scripted"
MODE_RL = "rl"


MODE_SCRIPTED_ONLY = ("pursuer", "evader", "defender", "attacker")


@dataclass
class GateThresholds:
    close_quarters: float = 3.0      # dist below which we hand off to RL
    juke_lateral: float = 2.2        # target lateral speed that counts as "juking"
    juke_min_dist: float = 6.0       # only treat juking as messy inside this range
    closing_fast: float = 3.0        # evader: pursuer closing rate that triggers RL
    evader_danger: float = 4.0       # evader: threat distance below which RL takes over
    # objective games
    defender_close: float = 6.0      # defender: engagement range (scale is bigger here)
    threat_imminent: float = 1.6     # defender: seconds-to-asset that means "now"
    committed_range: float = 8.0     # attacker: inside this it is a ballistic run-in


# ---- leaf behaviours -------------------------------------------------------

class _Condition(py_trees.behaviour.Behaviour):
    """Returns SUCCESS iff ``predicate(features, thresholds)`` is true."""

    def __init__(self, name, predicate, bb):
        super().__init__(name)
        self.predicate = predicate
        self.bb = bb

    def update(self) -> Status:
        feat: AgentFeatures = self.bb.get("features")
        thr: GateThresholds = self.bb.get("thresholds")
        return Status.SUCCESS if self.predicate(feat, thr) else Status.FAILURE


class _SetMode(py_trees.behaviour.Behaviour):
    def __init__(self, name, mode, bb):
        super().__init__(name)
        self.mode = mode
        self.bb = bb

    def update(self) -> Status:
        self.bb.set("mode", self.mode)
        return Status.SUCCESS


def _branch(cond_name, predicate, mode, bb):
    seq = py_trees.composites.Sequence(name=f"{cond_name}->{mode}", memory=False)
    seq.add_children([_Condition(cond_name, predicate, bb), _SetMode(f"set:{mode}", mode, bb)])
    return seq


# ---- predicates ------------------------------------------------------------

def _pursuer_predicates():
    return [
        ("close_quarters", lambda f, t: f.dist_nearest < t.close_quarters, MODE_RL),
        ("target_juking",
         lambda f, t: (f.target_lateral > t.juke_lateral and f.dist_nearest < t.juke_min_dist),
         MODE_RL),
        ("contested", lambda f, t: f.contested and f.n_live_others >= 2, MODE_RL),
        ("clean_intercept", lambda f, t: f.intercept_ahead, MODE_SCRIPTED),
    ]


def _evader_predicates():
    return [
        ("threat_close", lambda f, t: f.dist_nearest < t.evader_danger, MODE_RL),
        ("threat_closing_fast", lambda f, t: f.closing_rate > t.closing_fast, MODE_RL),
    ]


def _defender_predicates():
    """Air-defence handoff — and it is NOT the tag heuristic.

    Measured on ``assault`` (150 eps x 2 seed blocks, DAgger policy as the RL
    branch), gating on one predicate at a time:

        scripted only ............................ 0.54 / 0.56
        intercept_infeasible -> RL ............... 0.70 / 0.71   <-- ship this
        + threat_imminent -> RL .................. 0.45 / 0.45
        + close_quarters -> RL ................... 0.12 / 0.12
        RL only .................................. 0.03

    So the tag profile's instinct — "close quarters is messy, hand it to the
    policy" — is *actively harmful* here, costing 42 points. The terminal
    endgame against a missile is not a scrappy dogfight; it is a precise
    geometry problem the lead-intercept law solves near-optimally and the
    policy cannot. The one regime worth handing over is the one where the
    scripted law is not merely imprecise but **structurally undefined**: no
    positive intercept solution exists, so it falls back to parking on a static
    gate point and has nothing better to offer.

    The handover is deliberately **immediate — a debounced variant was measured
    and lost**. Under a degraded link, handover share rises 16%→22% (feasibility
    computed from a bad estimate reads as "no solution exists"), which looked
    like an anti-pattern: the tree escalating to the policy exactly when the
    policy is least trustworthy. Requiring the infeasibility to persist k
    consecutive ticks before handing over was a clean dose-response *loss*
    (assault latency axis, n=150 paired seeds+corruption, bt_gated win rate):

        latency:      0 ms   20 ms   40 ms   60 ms
        k=1 (ship)    0.71    0.51    0.21    0.07
        k=2           0.59    0.37    0.19    0.06
        k=3           0.49    0.35    0.19    0.06

    Each extra tick of delay costs ~0.10-0.12 at the clean end and recovers
    nothing at the degraded end. Traces show why: at perfect link infeasibility
    does not flicker — genuine handover regimes are 20-30-tick streaks and the
    1-3-tick blips are real geometry, so a debounce only delays the handover by
    (k-1) ticks, and at 11 m/s missile speed those ~1.3 m decide the episode.
    The extra handovers under degradation were never the thing costing wins;
    the latency collapse belongs to the *policy*, and the shield — not the gate
    — is the load-bearing mitigation there. Pinned by
    ``test_defender_handover_is_immediate``.
    """
    return [
        # the scripted law has no answer here — it parks on a static gate point
        ("intercept_infeasible", lambda f, t: not f.intercept_feasible, MODE_RL),
        # kept for legibility in Groot2: this is the regime scripted owns outright
        ("clean_intercept", lambda f, t: f.intercept_feasible and f.intercept_ahead,
         MODE_SCRIPTED),
    ]


def _attacker_predicates():
    """Missile-side profile — and the measured answer is **do not gate at all**.

    This profile used to carry the evader tree's two RL branches as design intent,
    flagged unvalidated. An attacker DAgger policy (``assault_attacker_dagger``,
    breach 0.33/0.35) now exists, so the same cumulative ablation could be run
    against it (``pe-ablate --game assault --side evaders``, 150 eps x 2 blocks,
    scored as **breach rate**, higher = better for the attacker):

        scripted only ............................ 0.46 / 0.57
        + committed -> SCRIPTED .................. 0.46 / 0.57   (no-op by design)
        + threat_close -> RL ..................... 0.41 / 0.49
        + threat_closing_fast -> RL .............. 0.37 / 0.38
        policy only .............................. 0.33 / 0.35

    Monotonically downhill: every handoff costs breach rate, and the scripted
    guidance law dominates end to end. So the RL branches are gone, and what
    remains is one branch that routes to SCRIPTED — i.e. the attacker gate is
    deliberately a no-op, kept only because it documents the run-in regime
    legibly in Groot2.

    Read this against ``_defender_predicates``: same game, same machinery,
    opposite verdict. The defender gains 16 points from exactly one handoff; the
    attacker gains nothing from any. The difference is not "which side is
    harder" — it is that the defender's scripted law has a regime where it is
    **structurally undefined** (no positive intercept root, so it parks on a
    static gate point) and the missile's guidance law has no such regime. A
    committed run-in is precisely the case a proportional-navigation law is for.

    Generalised, that is the project's main transferable claim: hand a regime to
    the policy where the scripted controller is *undefined*, not merely where it
    is imprecise or where the situation merely feels messy.
    """
    return [
        # committed run-in: no lateral authority left to spend, fly the guidance law
        ("committed", lambda f, t: f.asset_dist < t.committed_range, MODE_SCRIPTED),
    ]


_PREDICATES = {
    "pursuer": _pursuer_predicates,
    "evader": _evader_predicates,
    "defender": _defender_predicates,
    "attacker": _attacker_predicates,
}


# Display-only decomposition of each predicate into its terms, for the trace
# viewer: (feature, operator, threshold-name-or-literal). The viewer renders the
# live numbers so you can see *how close* a branch came to firing, not just that
# it did. This duplicates the lambdas above, so ``test_branch_reads_match_predicates``
# asserts the key sets agree exactly and that every name resolves on
# ``AgentFeatures`` / ``GateThresholds`` — add a predicate without an entry here
# and the suite fails.
BRANCH_READS = {
    "pursuer": {
        "close_quarters": [("dist_nearest", "<", "close_quarters")],
        "target_juking": [("target_lateral", ">", "juke_lateral"),
                          ("dist_nearest", "<", "juke_min_dist")],
        "contested": [("contested", "is", True), ("n_live_others", ">=", 2)],
        "clean_intercept": [("intercept_ahead", "is", True)],
    },
    "evader": {
        "threat_close": [("dist_nearest", "<", "evader_danger")],
        "threat_closing_fast": [("closing_rate", ">", "closing_fast")],
    },
    "defender": {
        "intercept_infeasible": [("intercept_feasible", "is", False)],
        "clean_intercept": [("intercept_feasible", "is", True),
                            ("intercept_ahead", "is", True)],
    },
    "attacker": {
        "committed": [("asset_dist", "<", "committed_range")],
    },
}


def default_profile(team: str, game: str = "tag") -> str:
    """Which gate profile a team runs, given the game being played."""
    if game == "tag":
        return "pursuer" if team == TEAM_PURSUERS else "evader"
    return "defender" if team == TEAM_PURSUERS else "attacker"


def build_gate_tree(team: str, bb, profile: str | None = None) -> py_trees.trees.BehaviourTree:
    profile = profile or ("pursuer" if team == TEAM_PURSUERS else "evader")
    root = py_trees.composites.Selector(name=f"gate[{profile}]", memory=False)
    for cond_name, pred, mode in _PREDICATES[profile]():
        root.add_child(_branch(cond_name, pred, mode, bb))
    root.add_child(_SetMode("default:scripted", MODE_SCRIPTED, bb))
    return py_trees.trees.BehaviourTree(root)


# ---- the gated controller --------------------------------------------------

class GatedController(BaseController):
    """Composes a scripted controller and an RL controller behind a BT gate.

    Each tick it computes both controllers' team actions, then for every agent
    ticks the behaviour tree to pick which controller's sub-action to use. It
    records per-mode usage in ``self.mode_counts`` for the writeup's analysis.
    """

    def __init__(self, team: str, scripted: Controller, rl: Controller,
                 thresholds: GateThresholds | None = None, name: str = "bt_gated",
                 profile: str | None = None, game: str = "tag"):
        self.team = team
        self.scripted = scripted
        self.rl = rl
        self.thresholds = thresholds or GateThresholds()
        self.name = name
        self.profile = profile or default_profile(team, game)
        # unique blackboard namespace per instance so parallel envs don't collide
        self._ns = f"/gate/{team}/{id(self)}"
        self.bb = py_trees.blackboard.Client(name=f"{self.name}:{id(self)}")
        for key in ("features", "thresholds", "mode"):
            self.bb.register_key(f"{self._ns}/{key}", access=py_trees.common.Access.READ)
            self.bb.register_key(f"{self._ns}/{key}", access=py_trees.common.Access.WRITE)
        self.tree = build_gate_tree(team, _NS(self.bb, self._ns), self.profile)
        self.mode_counts = {MODE_SCRIPTED: 0, MODE_RL: 0}
        # Opt-in per-tick record for the trace viewer: which branch fired, the
        # features it saw, and what each controller wanted. Off by default — act()
        # runs millions of times during DAgger and must not allocate for nothing.
        self.trace = False
        self.last_decisions: list[dict] = []

    def reset(self):
        self.scripted.reset()
        self.rl.reset()
        self.mode_counts = {MODE_SCRIPTED: 0, MODE_RL: 0}
        self.last_decisions = []

    def evaluate_branches(self, features) -> list[dict]:
        """Every branch's truth value this tick, in Selector order.

        Re-evaluated rather than scraped out of py_trees: a Selector returns on
        the first child that succeeds, so the nodes after the winner are never
        ticked and have no status to read. Evaluating all of them here keeps the
        tracing read-only (no instrumented node subclasses) and lets the viewer
        show the branches that *nearly* fired, which is where the interesting
        gate behaviour lives.
        """
        return [{"name": name, "mode": mode, "passed": bool(pred(features, self.thresholds))}
                for name, pred, mode in _PREDICATES[self.profile]()]

    def _fired_branch(self, features):
        """Which predicate the Selector would have taken — the first that passes."""
        for b in self.evaluate_branches(features):
            if b["passed"]:
                return b["name"], b["mode"]
        return "default", MODE_SCRIPTED

    def act(self, view: TeamView) -> np.ndarray:
        scripted_a = np.asarray(self.scripted.act(view), dtype=np.float64).reshape(-1, 3)
        rl_a = np.asarray(self.rl.act(view), dtype=np.float64).reshape(-1, 3)
        n_self = view.self_pos.shape[0]
        out = np.zeros((n_self, 3))
        nsbb = _NS(self.bb, self._ns)
        if self.trace:
            self.last_decisions = []
        for i in range(n_self):
            if not view.self_alive[i]:
                continue
            features = agent_features(view, i)
            nsbb.set("features", features)
            nsbb.set("thresholds", self.thresholds)
            nsbb.set("mode", MODE_SCRIPTED)
            self.tree.tick()
            mode = nsbb.get("mode")
            self.mode_counts[mode] += 1
            out[i] = rl_a[i] if mode == MODE_RL else scripted_a[i]
            if self.trace:
                evals = self.evaluate_branches(features)
                branch = next((b["name"] for b in evals if b["passed"]), "default")
                self.last_decisions.append(
                    {"agent": i, "mode": mode, "branch": branch, "features": features,
                     "evals": evals, "scripted": scripted_a[i].copy(), "rl": rl_a[i].copy()})
        return np.clip(out.reshape(-1), -1, 1)


class _NS:
    """Tiny namespaced view over a py_trees blackboard client."""

    def __init__(self, client, ns):
        self.client = client
        self.ns = ns

    def set(self, key, value):
        self.client.set(f"{self.ns}/{key}", value)

    def get(self, key):
        return self.client.get(f"{self.ns}/{key}")
