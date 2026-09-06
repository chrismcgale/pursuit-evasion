"""V&V for the sim-to-real link (``env/link.py``).

The load-bearing test here is ``test_perfect_link_is_a_no_op``: every number
already in the writeup was measured without this layer, so a PERFECT link must
be bit-identical, not merely close. If it ever isn't, the robustness curves stop
being comparable to the headline results and the whole study is anchored to
nothing.
"""
import numpy as np
import pytest

from pursuit_evasion.env import PursuitEvasionCore, make_game
from pursuit_evasion.env.link import (FIELD_DEGRADED, PERFECT, PRESETS,
                                      VICON_BUSY, VICON_LAB, LinkedController,
                                      LinkProfile, _Burst)
from pursuit_evasion.env.core import TEAM_PURSUERS
from pursuit_evasion.scripted import default_controllers


def _episode(ctrl, opponent, game="tag", seed=7, steps=60):
    """Run a fixed episode and return the applied pursuer commands, tick by tick."""
    core = PursuitEvasionCore(game=make_game(game), seed=seed)
    pv, ev = core.reset(seed=seed)
    ctrl.reset()
    opponent.reset()
    out = []
    for _ in range(steps):
        a = np.asarray(ctrl.act(pv), dtype=np.float64)
        out.append(a.copy())
        res = core.step(a, opponent.act(ev))
        pv, ev = res.pursuer_view, res.evader_view
        if res.terminated or res.truncated:
            break
    return np.array(out)


# ---- the invariant ---------------------------------------------------------

def test_perfect_link_is_a_no_op():
    """A PERFECT link must reproduce the unwrapped controller exactly."""
    for game in ("tag", "assault"):
        scripted, opp = default_controllers(game)
        bare = _episode(scripted, opp, game=game)
        scripted2, opp2 = default_controllers(game)
        linked = _episode(LinkedController(scripted2, PERFECT, dt=0.06), opp2, game=game)
        assert bare.shape == linked.shape
        np.testing.assert_array_equal(bare, linked)


def test_perfect_link_draws_no_randomness():
    """The bypass must not consume RNG, or seeded sweeps would shift under it."""
    rng = np.random.default_rng(0)
    scripted, opp = default_controllers("tag")
    _episode(LinkedController(scripted, PERFECT, rng=rng), opp)
    assert rng.random() == np.random.default_rng(0).random()


def test_is_perfect_only_for_the_identity_profile():
    assert PERFECT.is_perfect()
    for name, p in PRESETS.items():
        if name != "perfect":
            assert not p.is_perfect(), name
    # every individual knob must break perfection on its own
    for field, bad in [("obs_latency_s", 0.01), ("pos_noise_m", 0.01),
                       ("vel_noise_mps", 0.01), ("dropout_p", 0.01),
                       ("swap_p", 0.01), ("cmd_latency_s", 0.01),
                       ("motor_tau_s", 0.01), ("packet_loss_p", 0.01),
                       ("thrust_scale", 0.9), ("sag_frac_per_min", 0.1)]:
        assert not LinkProfile(**{field: bad}).is_perfect(), field


# ---- perception ------------------------------------------------------------

def test_latency_interpolates_below_one_tick():
    """At dt=60 ms a realistic 8-40 ms budget must not round to zero."""
    scripted, opp = default_controllers("tag")
    base = _episode(scripted, opp)
    outs = []
    for lat in (0.0, 0.01, 0.03, 0.05):
        s, o = default_controllers("tag")
        outs.append(_episode(LinkedController(s, LinkProfile(obs_latency_s=lat)), o))
    # each successive latency must move the commands further from nominal
    devs = [float(np.abs(o[:20] - base[:20]).mean()) for o in outs]
    assert devs[0] == 0.0
    assert devs[1] < devs[2] < devs[3], devs


def test_dropout_holds_the_last_pose_rather_than_zeroing():
    p = LinkProfile(dropout_p=1.0, dropout_burst=100.0)   # permanently occluded
    scripted, opp = default_controllers("tag")
    link = LinkedController(scripted, p, rng=np.random.default_rng(1))
    core = PursuitEvasionCore(game=make_game("tag"), seed=3)
    pv, ev = core.reset(seed=3)
    link.reset()
    link.act(pv)                     # first tick establishes the hold
    first = link._held[2].copy()
    for _ in range(5):
        res = core.step(np.zeros(6), np.zeros(6))
        pv = res.pursuer_view
        link.act(pv)
    # frozen at the first observation, and emphatically not at the origin
    np.testing.assert_allclose(link._held[2], first)
    assert np.linalg.norm(first) > 1.0


def test_label_swap_preserves_positions_but_exchanges_identities():
    """The danger of a swap is that it is undetectable from residuals."""
    p = LinkProfile(swap_p=1.0, swap_burst=100.0)
    scripted, opp = default_controllers("tag")
    link = LinkedController(scripted, p, rng=np.random.default_rng(0))
    core = PursuitEvasionCore(game=make_game("tag"), seed=5)
    pv, _ = core.reset(seed=5)
    link.reset()
    link.act(pv)
    seen = link._held[2]
    truth = np.asarray(pv.opp_pos)
    # the SET of reported positions is unchanged; the assignment is not
    assert sorted(np.linalg.norm(seen, axis=1).round(6).tolist()) == \
           sorted(np.linalg.norm(truth, axis=1).round(6).tolist())
    assert not np.allclose(seen, truth)
    assert link.n_swap == 1


def test_swap_is_sustained_not_per_tick_flicker():
    """A 1-tick swap is noise; a sustained one is the failure worth studying."""
    b = _Burst(p_on=1.0, mean_len=20.0, rng=np.random.default_rng(0))
    run = [b.tick() for _ in range(40)]
    assert run[0] and sum(run) > 5


# ---- command path ----------------------------------------------------------

def test_motor_lag_ramps_instead_of_stepping():
    p = LinkProfile(motor_tau_s=0.05)
    link = LinkedController(_Const(1.0), p, dt=0.06)
    link.reset()
    outs = [link._uplink(np.ones((2, 3)))[0, 0] for _ in range(5)]
    assert 0.0 < outs[0] < 1.0
    assert all(b > a for a, b in zip(outs, outs[1:]))     # monotone approach
    assert outs[-1] > 0.9


def test_packet_loss_holds_the_previous_command():
    p = LinkProfile(packet_loss_p=1.0)
    link = LinkedController(_Const(1.0), p, dt=0.06, rng=np.random.default_rng(0))
    link.reset()
    first = link._uplink(np.full((2, 3), 0.5))
    second = link._uplink(np.full((2, 3), -1.0))          # never arrives
    np.testing.assert_allclose(first, second)
    assert link.n_lost_packet == 1


def test_battery_sag_reduces_authority_over_the_flight():
    p = LinkProfile(sag_frac_per_min=0.6)
    link = LinkedController(_Const(1.0), p, dt=0.06)
    link.reset()
    early = link._uplink(np.ones((2, 3)))[0, 0]
    link._t = 50                                          # 3 s in at dt=60 ms
    late = link._uplink(np.ones((2, 3)))[0, 0]
    assert late < early


def test_commands_stay_in_the_actuator_box():
    """Whatever the link does, the core still receives a legal command."""
    scripted, opp = default_controllers("tag")
    for prof in (VICON_LAB, VICON_BUSY, FIELD_DEGRADED):
        s, o = default_controllers("tag")
        a = _episode(LinkedController(s, prof, rng=np.random.default_rng(2)), o)
        assert np.all(np.abs(a) <= 1.0 + 1e-12), prof.name


def test_degradation_is_ordered_across_the_presets():
    """The presets must form a ladder, or the sweep axis is meaningless."""
    order = [PERFECT, VICON_LAB, VICON_BUSY, FIELD_DEGRADED]
    sev = [(p.obs_latency_s + p.cmd_latency_s + p.motor_tau_s
            + p.pos_noise_m + p.vel_noise_mps + p.dropout_p + p.swap_p) for p in order]
    assert sev == sorted(sev), sev


class _Const:
    """Minimal controller stub: always commands the same value."""
    name = "const"

    def __init__(self, v):
        self.v = v

    def reset(self):
        pass

    def act(self, view):
        return np.full((2, 3), self.v)
