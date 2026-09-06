"""V&V for the objective games: heterogeneous dynamics, assets, win conditions.

The tag game's tests live in ``test_env.py``; these cover the additions. The most
load-bearing test here is ``test_apply_dynamics_is_pure`` — ``apply_dynamics`` is
mirrored line-for-line in ``cpp/include/pe/dynamics.hpp``, so it must stay a pure
function of (profile, velocity, command, fuel) with no hidden state.
"""
import numpy as np
import pytest

from pursuit_evasion.bt import GatedController, agent_features
from pursuit_evasion.bt.features import AgentFeatures
from pursuit_evasion.bt.gating import MODE_RL, MODE_SCRIPTED, default_profile
from pursuit_evasion.env import (GAME_KEYS, PursuitEvasionCore, apply_dynamics,
                                 build_team_obs, is_spent, make_game, obs_dim,
                                 sample_starts)
from pursuit_evasion.env.dynamics import (DRONE_PURSUER, KIND_MISSILE,
                                          MISSILE_ATTACKER)
from pursuit_evasion.scripted import default_controllers


# ---- dynamics --------------------------------------------------------------

def test_apply_dynamics_is_pure_and_bounded():
    rng = np.random.default_rng(0)
    for _ in range(200):
        vel = rng.normal(0, 6, 3)
        cmd = rng.normal(0, 1.5, 3)
        fuel = int(rng.integers(-1, 50))
        a = apply_dynamics(MISSILE_ATTACKER, vel, cmd, fuel)
        b = apply_dynamics(MISSILE_ATTACKER, vel, cmd, fuel)
        assert np.array_equal(a, b)                       # deterministic, no state
        assert np.linalg.norm(a) <= 1.0 + 1e-9            # normalised command


def test_holonomic_passes_command_through():
    cmd = np.array([0.3, -0.4, 0.5])
    out = apply_dynamics(DRONE_PURSUER, np.array([1.0, 0, 0]), cmd, 1)
    assert np.allclose(out, cmd)


def test_drone_thrust_is_isotropic():
    """No command direction may buy more than unit thrust.

    Regression, and the reason every tag *policy* number in the writeup moved:
    the pre-multi-game code clipped commands per component, so a drone commanded
    (1,1,1) pulled sqrt(3)*gear and topped out at 7.42 m/s against a documented
    v_max of 5.25. The scripted controllers emit unit-norm directions and never
    touched it; learned policies fill the action cube and exploited it. Mirrored
    in ``cpp/tests/test_games.cpp``.
    """
    vel = np.array([5.0, 0.0, 0.0])
    for cmd in ([1, 1, 1], [1, 1, 0], [-1, 1, -1], [1, 0, 0]):
        out = apply_dynamics(DRONE_PURSUER, vel, np.array(cmd, dtype=float), 1)
        assert np.linalg.norm(out) <= 1.0 + 1e-9, cmd
    # scaled down, but the commanded direction is preserved
    out = apply_dynamics(DRONE_PURSUER, vel, np.ones(3), 1)
    assert np.allclose(out, out[0])


def test_missile_cannot_turn_around():
    """A missile commanded straight backwards still ends up going forwards."""
    vel = np.array([10.0, 0.0, 0.0])
    out = apply_dynamics(MISSILE_ATTACKER, vel, np.array([-1.0, 0.0, 0.0]), 50)
    assert out[0] > 0.0, "thrust stays on the velocity axis while powered"


def test_missile_lateral_authority_is_capped():
    """Hard-over lateral command buys exactly ``lat_authority`` g per g of thrust.

    The command is renormalised after ``fwd + lat``, so the invariant is the
    lateral:forward *ratio*, not the absolute lateral magnitude.
    """
    vel = np.array([10.0, 0.0, 0.0])
    out = apply_dynamics(MISSILE_ATTACKER, vel, np.array([0.0, 1.0, 0.0]), 50)
    ratio = np.linalg.norm(out[1:]) / out[0]
    assert ratio == pytest.approx(MISSILE_ATTACKER.lat_authority, rel=1e-6)


def test_missile_lateral_authority_binds_for_any_command():
    """No commanded direction can exceed the airframe's lateral:forward ratio."""
    rng = np.random.default_rng(1)
    vel = np.array([10.0, 0.0, 0.0])
    for _ in range(100):
        out = apply_dynamics(MISSILE_ATTACKER, vel, rng.normal(0, 1, 3), 50)
        assert np.linalg.norm(out[1:]) <= MISSILE_ATTACKER.lat_authority * out[0] + 1e-9


def test_missile_coasts_when_burnt_out():
    vel = np.array([10.0, 0.0, 0.0])
    out = apply_dynamics(MISSILE_ATTACKER, vel, np.array([1.0, 0.0, 0.0]), 0)
    assert out[0] == pytest.approx(0.0), "no forward thrust after burnout"


def test_missile_below_align_speed_has_no_body_axis():
    """With no meaningful velocity there is no forward direction to thrust along."""
    out = apply_dynamics(MISSILE_ATTACKER, np.zeros(3), np.array([0.0, 0.0, 1.0]), 50)
    assert np.allclose(out, [0.0, 0.0, 1.0])


def test_turn_radius_scales_with_speed_squared():
    r_slow = MISSILE_ATTACKER.turn_radius(5.0)
    r_fast = MISSILE_ATTACKER.turn_radius(10.0)
    assert r_fast == pytest.approx(4.0 * r_slow, rel=1e-6)


def test_is_spent_only_applies_to_burnt_out_slow_missiles():
    assert not is_spent(DRONE_PURSUER, np.zeros(3), 0)
    assert not is_spent(MISSILE_ATTACKER, np.array([9.0, 0, 0]), 0)   # fast: still lethal
    assert not is_spent(MISSILE_ATTACKER, np.zeros(3), 5)             # still burning
    assert is_spent(MISSILE_ATTACKER, np.array([0.5, 0, 0]), 0)


# ---- game specs ------------------------------------------------------------

def test_tag_is_unchanged_by_the_game_layer():
    """Regression guard: adding games must not perturb the original experiment."""
    core = PursuitEvasionCore(seed=0)
    assert core.game.key == "tag"
    assert not core.game.has_asset
    pv, _ = core.reset(seed=0)
    assert build_team_obs(pv).shape[0] == obs_dim(2, 2) == 43


@pytest.mark.parametrize("key", GAME_KEYS)
def test_static_xml_matches_the_committed_arena(key):
    """cpp/assets/arena*.xml are generated; both runtimes must load the same file.

    If this fails, the C++ results are running on different geometry than the
    Python ones and nothing downstream is comparable. Regenerate with
    ``python -m pursuit_evasion.env.dump_arenas``.
    """
    from pathlib import Path

    from pursuit_evasion.env.dump_arenas import arena_filename

    committed = (Path(__file__).resolve().parents[1] / "cpp" / "assets"
                 / arena_filename(key))
    if not committed.exists():
        pytest.skip("C++ assets not present")
    assert PursuitEvasionCore(game=make_game(key)).static_xml() == committed.read_text()


@pytest.mark.parametrize("key", GAME_KEYS)
def test_every_game_runs_to_a_decision(key):
    game = make_game(key)
    core = PursuitEvasionCore(seed=1, game=game)
    P, E = default_controllers(key)
    pv, ev = core.reset(seed=1)
    P.reset()
    E.reset()
    for _ in range(game.episode.max_steps + 1):
        r = core.step(P.act(pv), E.act(ev))
        pv, ev = r.pursuer_view, r.evader_view
        if r.terminated or r.truncated:
            break
    assert r.terminated or r.truncated
    assert isinstance(r.info["pursuer_win"], bool)


@pytest.mark.parametrize("key", ["assault", "escort"])
def test_objective_obs_is_wider_and_self_consistent(key):
    core = PursuitEvasionCore(seed=0, game=key)
    pv, ev = core.reset(seed=0)
    expected = obs_dim(2, 2, has_asset=True)
    assert expected > obs_dim(2, 2)
    assert build_team_obs(pv).shape[0] == expected
    assert build_team_obs(ev).shape[0] == expected
    assert np.all(np.isfinite(build_team_obs(pv)))


def test_breach_ends_the_episode_as_a_defender_loss():
    """Drive an attacker straight at the asset with no defenders acting."""
    game = make_game("assault")
    core = PursuitEvasionCore(seed=3, game=game)
    pv, ev = core.reset(seed=3)
    for _ in range(game.episode.max_steps):
        # attackers steer at the asset; defenders sit still
        cmd = []
        for i in range(ev.self_pos.shape[0]):
            d = ev.asset_pos - ev.self_pos[i]
            cmd.append(d / max(np.linalg.norm(d), 1e-6))
        r = core.step(np.zeros(6), np.concatenate(cmd))
        ev = r.evader_view
        if r.terminated or r.truncated:
            break
    assert r.info["breach"] is True
    assert r.terminated and not r.info["pursuer_win"]


def test_timeout_is_a_defender_win_in_objective_games():
    """Nobody moves: the raid never lands, so the defenders hold."""
    game = make_game("assault")
    core = PursuitEvasionCore(seed=5, game=game)
    core.reset(seed=5)
    for _ in range(game.episode.max_steps):
        r = core.step(np.zeros(6), np.zeros(6))
        if r.terminated or r.truncated:
            break
    assert not r.info["breach"]
    assert r.info["pursuer_win"] is True


def test_escort_asset_moves_and_arriving_wins():
    game = make_game("escort")
    core = PursuitEvasionCore(seed=7, game=game)
    core.reset(seed=7)
    start = np.asarray(game.asset.start)
    r = core.step(np.zeros(6), np.zeros(6))
    p0 = np.asarray(r.pursuer_view.asset_pos)
    for _ in range(game.episode.max_steps):
        r = core.step(np.zeros(6), np.zeros(6))
        if r.terminated or r.truncated:
            break
    p1 = np.asarray(r.pursuer_view.asset_pos)
    assert np.linalg.norm(p1 - start) > np.linalg.norm(p0 - start)  # it transits
    assert r.info["asset_arrived"] and r.info["pursuer_win"]


def test_sample_starts_respect_the_arena_bounds():
    rng = np.random.default_rng(0)
    for key in GAME_KEYS:
        game = make_game(key)
        a = game.arena
        for _ in range(20):
            starts = sample_starts(game, rng, ("pursuer0", "pursuer1"),
                                   ("evader0", "evader1"))
            for name, (x, y, z) in starts.items():
                lim = a.half_extent - a.agent_radius
                assert -lim <= x <= lim and -lim <= y <= lim, name
                assert a.z_min <= z <= a.z_max, name


def test_attackers_never_spawn_on_top_of_the_asset():
    """Regression: the escort ambush ring used to be centred on the origin."""
    game = make_game("escort")
    rng = np.random.default_rng(0)
    for _ in range(50):
        s = sample_starts(game, rng, ("pursuer0", "pursuer1"), ("evader0", "evader1"))
        for name in ("evader0", "evader1"):
            d = np.linalg.norm(np.asarray(s[name]) - np.asarray(game.asset.start))
            assert d > game.asset.radius * 2, f"{name} spawned inside the keep-out"


# ---- BT gate profiles ------------------------------------------------------

def test_default_profile_follows_the_game():
    assert default_profile("pursuers", "tag") == "pursuer"
    assert default_profile("evaders", "tag") == "evader"
    assert default_profile("pursuers", "assault") == "defender"
    assert default_profile("evaders", "escort") == "attacker"


def test_objective_features_are_populated():
    core = PursuitEvasionCore(seed=0, game="assault")
    pv, ev = core.reset(seed=0)
    core.step(np.zeros(6), np.zeros(6))
    f = agent_features(pv, 0)
    assert np.isfinite(f.asset_dist) and f.asset_dist > 0
    assert f.threat_time > 0
    g = agent_features(ev, 0)
    assert 0.0 <= g.fuel <= 1.0


def test_tag_features_leave_objective_fields_inert():
    core = PursuitEvasionCore(seed=0)
    pv, _ = core.reset(seed=0)
    f = agent_features(pv, 0)
    assert f.asset_dist == float("inf")
    assert f.threat_time == float("inf")
    assert f.intercept_feasible is True and f.fuel == 1.0


def test_attacker_gate_commits_inside_the_terminal_run_in():
    from pursuit_evasion.bt.gating import GateThresholds, _PREDICATES
    thr = GateThresholds()
    preds = _PREDICATES["attacker"]()
    f = agent_features(PursuitEvasionCore(seed=0, game="assault").reset(seed=0)[1], 0)
    f.asset_dist = thr.committed_range - 1.0
    f.dist_nearest = 0.5                       # a threat right on top of us...
    name, pred, mode = preds[0]
    assert name == "committed" and pred(f, thr) and mode == MODE_SCRIPTED


def test_attacker_gate_hands_over_nothing():
    """Pins the attacker ablation — see _attacker_predicates' docstring.

    The missile profile originally carried the evader tree's two RL branches as
    design intent. Measured against a real attacker policy they were monotonically
    harmful (breach 0.46/0.57 scripted -> 0.37/0.38 fully gated -> 0.33/0.35
    policy only), so they are gone and the attacker gate is a deliberate no-op.

    This is the counterweight to test_defender_gate_does_not_hand_close_quarters
    _to_the_policy: same game, same machinery, opposite verdict. A handoff is
    worth it where the scripted law is structurally undefined, not merely where
    the situation is messy.

    The C++ half ticks the shipped trees/gate_attackers.xml in
    cpp/tests/test_games.cpp.
    """
    from pursuit_evasion.bt.gating import GateThresholds, _PREDICATES

    thr = GateThresholds()
    f = agent_features(PursuitEvasionCore(seed=0, game="assault").reset(seed=0)[1], 0)
    f.asset_dist = 30.0      # not committed...
    f.dist_nearest = 0.5     # ...interceptor on top of us...
    f.closing_rate = 9.0     # ...and closing hard
    fired = [(n, m) for n, pred, m in _PREDICATES["attacker"]() if pred(f, thr)]
    assert all(m == MODE_SCRIPTED for _, m in fired), fired
    assert all(m == MODE_SCRIPTED for _, _, m in _PREDICATES["attacker"]()), (
        "the attacker gate must not route any regime to the policy")
    assert [n for n, _, _ in _PREDICATES["attacker"]()] == ["committed"]


def test_defender_gate_hands_off_when_intercept_is_infeasible():
    from pursuit_evasion.bt.gating import GateThresholds, _PREDICATES
    thr = GateThresholds()
    name, pred, mode = _PREDICATES["defender"]()[0]
    f = agent_features(PursuitEvasionCore(seed=0, game="assault").reset(seed=0)[0], 0)
    f.intercept_feasible = False
    assert name == "intercept_infeasible" and pred(f, thr) and mode == MODE_RL


class _ConstController:
    """Stub controller returning a recognisable constant command."""

    def __init__(self, value):
        self.value = value

    def reset(self):
        pass

    def act(self, view):
        return np.full(view.self_pos.shape[0] * 3, self.value)


def _defender_view(missile_vel_x):
    """One defender at the origin watching one missile 20 m out on +x.

    The missile flying away at 11 m/s outruns the 7.5 m/s defender — no
    positive lead-intercept root, genuinely infeasible. Flip its velocity and
    the intercept is head-on trivial.
    """
    from pursuit_evasion.env.core import (ArenaConfig, TEAM_PURSUERS,
                                          TeamView)
    return TeamView(
        team=TEAM_PURSUERS,
        self_pos=np.array([[0.0, 0.0, 3.0]]), self_vel=np.zeros((1, 3)),
        self_alive=np.array([True]),
        opp_pos=np.array([[20.0, 0.0, 3.0]]),
        opp_vel=np.array([[missile_vel_x, 0.0, 0.0]]),
        opp_alive=np.array([True]),
        time_frac=0.0, arena=ArenaConfig(), vmax=7.5, opp_vmax=11.0,
        game="assault", asset_pos=np.array([-5.0, 0.0, 3.0]))


def test_defender_handover_is_immediate():
    """Pins the negative result of the debounce experiment (ROADMAP #3).

    Requiring intercept infeasibility to persist k consecutive ticks before
    handing over — the designed fix for the "escalates under degradation"
    observation — was measured and LOST, as a clean dose-response on the
    assault latency axis (n=150, paired seeds+corruption, bt_gated win rate):

        latency:      0 ms   20 ms   40 ms   60 ms
        k=1 (ship)    0.71    0.51    0.21    0.07
        k=2           0.59    0.37    0.19    0.06
        k=3           0.49    0.35    0.19    0.06

    Traces show genuine infeasible regimes are 20-30-tick streaks that do not
    flicker at perfect link, so a debounce is pure delay — and (k-1) ticks of
    delay against an 11 m/s missile decide episodes. The first infeasible tick
    must therefore hand over immediately. Do not re-add a persistence
    threshold without beating this table. C++ mirror in
    cpp/tests/test_games.cpp.
    """
    from pursuit_evasion.env.core import TEAM_PURSUERS

    gc = GatedController(TEAM_PURSUERS, _ConstController(0.25),
                         _ConstController(-0.75), game="assault")
    infeasible, feasible = _defender_view(11.0), _defender_view(-11.0)
    assert not agent_features(infeasible, 0).intercept_feasible
    assert agent_features(feasible, 0).intercept_feasible

    assert gc.act(feasible)[0] == 0.25                    # feasible: scripted
    assert gc.act(infeasible)[0] == -0.75                 # first infeasible tick: RL
    assert gc.mode_counts[MODE_RL] == 1


def test_defender_gate_does_not_hand_close_quarters_to_the_policy():
    """Pins the ablation result — see _defender_predicates' docstring.

    Porting tag's "close quarters is messy, hand it to the policy" instinct into
    air defence measured 0.70 -> 0.12 defender win rate: worse than not gating at
    all. The terminal endgame against a missile is a precise geometry problem the
    lead-intercept law owns, not a scrappy dogfight. A close-in defender WITH a
    feasible intercept must therefore stay scripted.

    The C++ half of this lives in cpp/tests/test_games.cpp, ticking the shipped
    trees/gate_defenders.xml.
    """
    from pursuit_evasion.bt.gating import GateThresholds, _PREDICATES, MODE_SCRIPTED

    thr = GateThresholds()
    f = agent_features(PursuitEvasionCore(seed=0, game="assault").reset(seed=0)[0], 0)
    f.intercept_feasible = True
    f.intercept_ahead = True
    f.dist_nearest = 1.0     # point blank
    f.threat_time = 0.2      # and the asset is about to be hit

    fired = [(name, mode) for name, pred, mode in _PREDICATES["defender"]()
             if pred(f, thr)]
    assert all(mode == MODE_SCRIPTED for _, mode in fired), (
        f"defender gate routed a close-in feasible intercept to RL: {fired}")
    # and the profile stays at the two measured branches
    assert [n for n, _, _ in _PREDICATES["defender"]()] == [
        "intercept_infeasible", "clean_intercept"]


@pytest.mark.parametrize("key", ["assault", "escort"])
def test_gated_controller_runs_an_objective_episode(key):
    from pursuit_evasion.scripted.base import ZeroController
    game = make_game(key)
    scripted, attackers = default_controllers(key)
    gate = GatedController("pursuers", scripted, ZeroController(2), game=key)
    core = PursuitEvasionCore(seed=2, game=game)
    pv, ev = core.reset(seed=2)
    gate.reset()
    attackers.reset()
    for _ in range(30):
        r = core.step(gate.act(pv), attackers.act(ev))
        pv, ev = r.pursuer_view, r.evader_view
        if r.terminated or r.truncated:
            break
    assert gate.profile == "defender"
    assert sum(gate.mode_counts.values()) > 0


def test_branch_reads_match_predicates():
    """BRANCH_READS is display metadata for the trace viewer, duplicated from the
    predicate lambdas. This is the thing that stops it silently going stale: every
    profile's key set must match exactly, and every name it mentions must resolve
    on AgentFeatures / GateThresholds. Add a branch without annotating it and the
    viewer would render an empty explanation — fail here instead.
    """
    from dataclasses import fields

    from pursuit_evasion.bt.gating import BRANCH_READS, GateThresholds, _PREDICATES

    feat_names = {f.name for f in fields(AgentFeatures)}
    thr_names = {f.name for f in fields(GateThresholds)}
    assert BRANCH_READS.keys() == _PREDICATES.keys()
    for profile, preds in _PREDICATES.items():
        assert BRANCH_READS[profile].keys() == {n for n, _, _ in preds()}, profile
        for branch, terms in BRANCH_READS[profile].items():
            for feature, op, thr in terms:
                assert feature in feat_names, (profile, branch, feature)
                assert isinstance(thr, (bool, int, float)) or thr in thr_names, \
                    (profile, branch, thr)
                assert op in ("<", ">", "<=", ">=", "is"), (profile, branch, op)


def test_evaluate_branches_agrees_with_the_selector():
    """The viewer shows all branches, but the *fired* one must still be whatever
    py_trees actually picked — tracing is read-only and must not become a second,
    subtly-different gate implementation."""
    from pursuit_evasion.scripted.base import ZeroController

    scripted, attackers = default_controllers("assault")
    gate = GatedController("pursuers", scripted, ZeroController(2), game="assault")
    gate.trace = True
    core = PursuitEvasionCore(seed=3, game=make_game("assault"))
    pv, ev = core.reset(seed=3)
    gate.reset()
    gate.trace = True
    attackers.reset()
    seen = set()
    for _ in range(60):
        r = core.step(gate.act(pv), attackers.act(ev))
        for d in gate.last_decisions:
            first = next((b for b in d["evals"] if b["passed"]), None)
            assert d["branch"] == (first["name"] if first else "default")
            # the mode the tree wrote to the blackboard must match the branch's
            assert d["mode"] == (first["mode"] if first else MODE_SCRIPTED)
            seen.add(d["branch"])
        pv, ev = r.pursuer_view, r.evader_view
        if r.terminated or r.truncated:
            break
    assert seen, "no gate decisions were traced"


def test_trace_and_explorer_round_trip(tmp_path):
    """The explorer inlines the trace as JSON into a standalone HTML file, so the
    trace must stay JSON-serialisable (inf/nan are nulled) and the page must carry
    no external references — it gets opened straight off disk, often on a machine
    without the repo.
    """
    import json

    from pursuit_evasion.eval.explorer import build
    from pursuit_evasion.eval.trace import record

    tr = record("assault", "scripted", None, seed=1)
    assert tr["steps"] and tr["branches"] == []          # no gate on a scripted run
    json.dumps(tr)                                        # would raise on inf/nan

    dest = build([tr], tmp_path / "e.html")
    html = dest.read_text()
    assert "__DATA__" not in html                         # placeholder was substituted
    assert "src=\"http" not in html and "@import" not in html
    # the inlined payload must parse back out
    blob = html.split("const TRACES = ", 1)[1].split(";\nconst C", 1)[0]
    assert json.loads(blob)[0]["game"] == "assault"
