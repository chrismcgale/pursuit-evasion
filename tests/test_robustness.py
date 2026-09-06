"""V&V for the robustness sweep: the statistics and the fairness controls.

The sweep exists to change conclusions, so its machinery has to be right. Two
things are easy to get wrong and both are pinned here: interval coverage near
0 and 1 (where win rates actually live), and the control-rate axis, which would
silently confound "faster control" with "more capture checks" if the substep
control were dropped.
"""
import numpy as np
import pytest

from pursuit_evasion.env.games import make_game
from pursuit_evasion.eval.robustness import (AXES, axis_profiles,
                                             control_rate_configs,
                                             latency_budget, paired_delta,
                                             wilson)


# ---- interval statistics ---------------------------------------------------

def test_wilson_stays_inside_the_unit_interval_at_the_extremes():
    """Normal-approximation intervals go negative at k=0; Wilson must not."""
    for n in (5, 40, 200):
        lo, hi = wilson(0, n)
        assert lo == 0.0 and 0.0 < hi < 1.0
        lo, hi = wilson(n, n)
        assert hi == 1.0 and 0.0 < lo < 1.0


def test_wilson_narrows_with_n():
    widths = [wilson(int(0.5 * n), n)[1] - wilson(int(0.5 * n), n)[0]
              for n in (20, 100, 500)]
    assert widths[0] > widths[1] > widths[2]


def test_wilson_brackets_the_point_estimate():
    for k, n in [(1, 10), (7, 20), (150, 200), (3, 300)]:
        lo, hi = wilson(k, n)
        assert lo <= k / n <= hi


def test_paired_delta_detects_a_real_shift():
    a = [True] * 30 + [False] * 70
    b = [False] * 100
    d = paired_delta(a, b, n_boot=2000, seed=1)
    assert d["delta"] == pytest.approx(0.30)
    assert d["lo"] > 0 and d["p_better"] > 0.99


def test_paired_delta_reports_no_effect_for_identical_arms():
    a = [True, False] * 50
    d = paired_delta(a, list(a), n_boot=2000, seed=1)
    assert d["delta"] == 0.0 and d["lo"] == 0.0 and d["hi"] == 0.0


def test_pairing_is_tighter_than_treating_arms_as_independent():
    """The whole reason the sweep pairs seeds: correlated arms shrink the CI."""
    rng = np.random.default_rng(0)
    base = rng.random(200) < 0.5
    # b differs from base in only 10 episodes -> a small but consistent delta
    a = base.copy()
    a[:10] = True
    b = base.copy()
    b[:10] = False
    paired = paired_delta(list(a), list(b), n_boot=4000, seed=2)
    # unpaired 95% half-width for two independent proportions at n=200
    p1, p2 = a.mean(), b.mean()
    unpaired_half = 1.96 * np.sqrt(p1 * (1 - p1) / 200 + p2 * (1 - p2) / 200)
    paired_half = (paired["hi"] - paired["lo"]) / 2
    assert paired_half < unpaired_half


# ---- the derived prediction ------------------------------------------------

def test_latency_budget_matches_the_hand_calculation():
    g = make_game("assault")
    closing = g.arena.pursuer_dyn.vmax + g.arena.evader_dyn.vmax
    assert latency_budget("assault") == pytest.approx(
        g.episode.capture_radius / closing)
    # and it is the number quoted in the docstring / writeup
    assert 0.070 < latency_budget("assault") < 0.080


def test_latency_axis_brackets_the_budget():
    """A sweep that stops short of the predicted cliff cannot test it."""
    for game in ("tag", "assault"):
        levels = AXES["latency"]["levels"]
        assert min(levels) == 0.0
        assert max(levels) > latency_budget(game)


# ---- fairness of the control-rate axis -------------------------------------

def test_control_rate_holds_wall_clock_episode_length_constant():
    """Same seconds of flight at every control rate, not the same step count.

    Exact equality is unavailable — max_steps is an integer, so 500*5/3 rounds
    to 833 and the episode lands 0.08 s short of 50. The requirement is that no
    level gets a materially longer episode than another, i.e. agreement to
    within one control period.
    """
    for game in ("tag", "assault"):
        secs = [period * ep.max_steps for period, ep, _ in control_rate_configs(game)]
        periods = [period for period, _, _ in control_rate_configs(game)]
        assert max(secs) - min(secs) <= max(periods), (game, secs)


def test_control_rate_forces_substep_capture_at_every_level():
    """Otherwise a faster control loop wins by sampling proximity more often."""
    for game in ("tag", "assault"):
        assert all(spec.capture_substeps for _, _, spec in control_rate_configs(game))


def test_control_rate_changes_only_the_intended_fields():
    g = make_game("tag")
    for _, ep, spec in control_rate_configs("tag"):
        assert spec.arena == g.arena              # same physics, same XML
        assert spec.episode.capture_radius == g.episode.capture_radius
        assert spec.timeout_winner == g.timeout_winner


def test_tag_control_period_exceeds_its_own_latency_budget():
    """Pins the finding: tag decides engagements slower than its tolerance."""
    g = make_game("tag")
    period = g.arena.timestep * g.episode.control_repeat
    assert period > latency_budget("tag")


# ---- axis hygiene ----------------------------------------------------------

def test_every_axis_starts_at_nominal():
    """Level 0 of every axis must be the perfect link, or curves lack an anchor."""
    for axis in AXES:
        lv, prof = axis_profiles(axis)[0]
        assert prof.is_perfect(), axis


def test_every_axis_is_monotone_in_its_field():
    for axis, spec in AXES.items():
        levels = [lv for lv, _ in axis_profiles(axis)]
        assert levels == sorted(levels), axis


def test_axis_metadata_is_complete():
    """Each axis must explain what physical thing it models."""
    for axis, spec in AXES.items():
        assert spec["why"].strip() and spec["label"].strip(), axis
        assert spec["field"] in type(axis_profiles(axis)[0][1]).__dataclass_fields__
