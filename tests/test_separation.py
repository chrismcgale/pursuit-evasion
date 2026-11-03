"""Own-team separation in the safety filter (REVIEW #12). Mirrored in
cpp/tests/test_safety.cpp — keep the two in step."""
import numpy as np

import pursuit_evasion  # noqa: F401
from pursuit_evasion.safety import SafetyConfig, filter_action, separate

ARGS = (5.25, 12.0, 0.5, 12.0)          # vmax, half_extent, z_min, z_max
ON = SafetyConfig(separation_m=1.5)
Z = np.zeros(3)


def _f(pos, vel, act, mates, cfg=ON, rank=0):
    return filter_action(np.array(pos, float), np.array(vel, float), np.array(act, float),
                         *ARGS, cfg, mates, rank)


def test_separation_is_off_by_default():
    assert SafetyConfig().separation_m == 0.0
    mate = [(np.array([0.5, 0, 3.0]), Z, 1)]
    a, v = _f([0, 0, 3], Z, [1, 0, 0], mate, SafetyConfig())
    b, _ = _f([0, 0, 3], Z, [1, 0, 0], [], SafetyConfig())
    assert not v.separation and np.array_equal(a, b)


def test_separation_turns_a_converging_command_away():
    # mate 1 m ahead on +x, agent commands straight at it
    a, v = _f([0, 0, 3], Z, [1, 0, 0], [(np.array([1.0, 0, 3]), Z, 1)])
    assert v.separation and a[0] < 0          # pushed back, not toward
    assert np.linalg.norm(a) <= 1 + 1e-12


def test_separation_is_brake_aware():
    # 2 m apart (> 1.5) but closing at 8 m/s: within 0.15 s they are 0.8 m apart
    a, v = _f([0, 0, 3], [4, 0, 0], [1, 0, 0], [(np.array([2.0, 0, 3]), np.array([-4.0, 0, 0]), 1)])
    assert v.separation and a[0] < 0
    # same geometry, at rest: no action
    _, v = _f([0, 0, 3], Z, [0, 1, 0], [(np.array([2.0, 0, 3]), Z, 1)])
    assert not v.separation


def test_separation_keeps_its_push_through_the_norm_clip():
    # full sideways command; mate 0.5 m behind on -x -> push along +x of 2*(1.5-0.5)=2
    a, v = _f([0, 0, 3], Z, [0, 1, 0], [(np.array([-0.5, 0, 3]), Z, 1)])
    assert v.separation
    assert abs(np.linalg.norm(a) - 1.0) < 1e-12 and abs(a[0] - 1.0) < 1e-12


def test_separation_breaks_a_coincident_tie_in_opposite_directions():
    p = np.array([0, 0, 3.0])
    a0, _ = separate(p, Z, np.zeros(3), [(p.copy(), Z, 1)], ON, rank=0)
    a1, _ = separate(p, Z, np.zeros(3), [(p.copy(), Z, 0)], ON, rank=1)
    assert a0[2] > 0 and a1[2] < 0


def test_fence_wins_over_separation_at_the_wall():
    # at the +x wall with the mate on the inside: separation pushes out, fence pulls in
    a, v = _f([12.0, 0, 3], Z, [0, 0, 0], [(np.array([11.0, 0, 3]), Z, 1)])
    assert v.separation and v.geofence and a[0] < 0


def test_team_filter_ignores_dead_teammates():
    from pursuit_evasion.env.core import PursuitEvasionCore
    from pursuit_evasion.safety import TeamSafetyFilter
    core = PursuitEvasionCore(seed=0)
    pv, _ = core.reset_with_starts({"pursuer0": (0, 0, 3), "pursuer1": (0.5, 0, 3),
                                    "evader0": (8, 1, 3), "evader1": (8, -1, 4)})
    f = TeamSafetyFilter(ON)
    f.apply(pv, np.zeros(6))
    assert f.n_separation == 2
    pv.self_alive[1] = False
    f.reset()
    f.apply(pv, np.zeros(6))
    assert f.n_separation == 0
