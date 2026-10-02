"""Python-side V&V: env mechanics, observation encoding, safety filter, BT gate."""
import numpy as np
import pytest

from pursuit_evasion.env import (ArenaConfig, EpisodeConfig, PursuitEvasionCore,
                                 TEAM_PURSUERS, build_team_obs, obs_dim)
from pursuit_evasion.scripted import FieldEvaders, InterceptPursuers, NoisyController
from pursuit_evasion.bt import GatedController, agent_features
from pursuit_evasion.safety import SafetyConfig, filter_action, TeamSafetyFilter


def test_obs_dim_matches_encoding():
    core = PursuitEvasionCore(seed=0)
    pv, _ = core.reset(seed=0)
    assert build_team_obs(pv).shape[0] == obs_dim(2, 2) == 43


def test_reset_with_starts_is_exact_and_replayable():
    core = PursuitEvasionCore(seed=0)
    starts = {"pursuer0": (-8, 1, 3), "pursuer1": (-8, -1, 3),
              "evader0": (8, 1, 3), "evader1": (8, -1, 4)}
    core.reset_with_starts(starts)
    assert np.allclose(core._pos("pursuer0"), [-8, 1, 3])
    P, E = InterceptPursuers(), FieldEvaders()
    sums = []
    for _ in range(2):
        pv, ev = core.reset_with_starts(starts)
        s = 0.0
        for _ in range(60):
            r = core.step(P.act(pv), E.act(ev))
            pv, ev = r.pursuer_view, r.evader_view
            s += r.info["min_dist"]
        sums.append(s)
    assert abs(sums[0] - sums[1]) < 1e-9


def test_capture_when_pursuer_on_evader():
    core = PursuitEvasionCore(seed=0)
    core.reset_with_starts({"pursuer0": (0, 0, 3), "pursuer1": (0, 0.3, 3),
                            "evader0": (0, 0, 3), "evader1": (9, 9, 3)})
    r = core.step(np.zeros(6), np.zeros(6))
    assert r.info["captured"][0]        # coincident evader tagged
    assert not r.info["captured"][1]


def test_scripted_capture_rate_in_designed_band():
    core = PursuitEvasionCore(seed=0)
    P, E = InterceptPursuers(), FieldEvaders()
    wins = 0
    for e in range(40):
        pv, ev = core.reset(seed=1000 + e)
        while True:
            r = core.step(P.act(pv), E.act(ev))
            pv, ev = r.pursuer_view, r.evader_view
            if r.terminated or r.truncated:
                wins += r.info["all_captured"]
                break
    assert 0.25 <= wins / 40 <= 0.75   # tuned band; see CLAUDE.md


def test_safety_geofence_blocks_outward_push():
    cfg = SafetyConfig()
    a, v = filter_action(np.array([12.0, 0, 3]), np.zeros(3), np.array([1.0, 0, 0]),
                         5.25, 12.0, 0.5, 12.0, cfg)
    assert v.geofence and a[0] < 0


def test_safety_speed_cap_removes_speed_increase():
    cfg = SafetyConfig()
    a, v = filter_action(np.array([0, 0, 3.0]), np.array([6.0, 0, 0]),
                         np.array([1.0, 0, 0]), 5.25, 12.0, 0.5, 12.0, cfg)
    assert v.speed and abs(a[0]) < 1e-6


def test_gate_routes_between_controllers():
    core = PursuitEvasionCore(seed=0)
    gate = GatedController(TEAM_PURSUERS, InterceptPursuers(), NoisyController(2, seed=1))
    E = FieldEvaders()
    pv, ev = core.reset(seed=3)
    gate.reset()
    for _ in range(200):
        r = core.step(gate.act(pv), E.act(ev))
        pv, ev = r.pursuer_view, r.evader_view
        if r.terminated or r.truncated:
            break
    total = gate.mode_counts["scripted"] + gate.mode_counts["rl"]
    assert total > 0


def test_agent_features_sane():
    core = PursuitEvasionCore(seed=0)
    pv, _ = core.reset_with_starts({"pursuer0": (0, 0, 3), "pursuer1": (-2, 0, 3),
                                    "evader0": (2, 0, 3), "evader1": (9, 0, 3)})
    f = agent_features(pv, 0)
    assert f.dist_nearest == pytest.approx(2.0, abs=1e-6)
    assert f.n_live_others == 2


def test_safety_fence_brakes_before_the_face():
    pos, vel = np.array([10.9, 0, 3.0]), np.array([5.0, 0, 0])
    a, v = filter_action(pos, vel, np.array([1.0, 0, 0]), 5.25, 12.0, 0.5, 12.0, SafetyConfig())
    assert v.geofence and a[0] < 0
    legacy = SafetyConfig(brake_horizon_s=0.0, keep_fence_authority=False)
    _, v = filter_action(pos, vel, np.array([1.0, 0, 0]), 5.25, 12.0, 0.5, 12.0, legacy)
    assert not v.geofence


def test_safety_fence_keeps_authority_through_the_norm_clip():
    a, _ = filter_action(np.array([12.0, 0, 3]), np.zeros(3), np.array([1.0, 1, 1]),
                         5.25, 12.0, 0.5, 12.0, SafetyConfig())
    assert np.linalg.norm(a) <= 1 + 1e-12 and abs(a[0] + 1.0) < 1e-12
