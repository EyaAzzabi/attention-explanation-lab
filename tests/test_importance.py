"""Tests for the measures the whole reproduction rests on.

Every value here is derived by hand or from a closed form, never from running the
code and copying its output. A test that asserts what the code already does proves
nothing, and these particular numbers are the ones a reader is being asked to trust.
"""
import numpy as np
import pytest

from src.importance import (
    JSD_UPPER_BOUND,
    jsd,
    kendall_tau_per_instance,
    summarise,
    tvd,
)


# --------------------------------------------------------------------------- #
# Total variation distance
# --------------------------------------------------------------------------- #
def test_tvd_is_zero_for_identical_distributions():
    p = np.array([0.2, 0.3, 0.5])
    assert tvd(p, p) == pytest.approx(0.0)


def test_tvd_is_one_for_disjoint_point_masses():
    assert tvd(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(1.0)


def test_tvd_matches_hand_computation():
    # 0.5 * (|0.7-0.4| + |0.3-0.6|) = 0.5 * 0.6 = 0.3
    assert tvd(np.array([0.7, 0.3]), np.array([0.4, 0.6])) == pytest.approx(0.3)


def test_tvd_is_symmetric():
    p, q = np.array([0.1, 0.9]), np.array([0.6, 0.4])
    assert tvd(p, q) == pytest.approx(tvd(q, p))


def test_tvd_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        tvd(np.array([0.5, 0.5]), np.array([0.3, 0.3, 0.4]))


# --------------------------------------------------------------------------- #
# Jensen-Shannon divergence
# --------------------------------------------------------------------------- #
def test_jsd_is_zero_for_identical_distributions():
    p = np.array([0.25, 0.25, 0.5])
    assert jsd(p, p) == pytest.approx(0.0, abs=1e-9)


def test_jsd_hits_ln2_for_disjoint_support():
    """The paper's 0.69 upper bound, which is ln(2) in nats.

    Figure 4 is read against this bound, so if the unit changed to bits every
    histogram in the reproduction would be wrong by a factor of ln(2).
    """
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    assert jsd(p, q) == pytest.approx(JSD_UPPER_BOUND, abs=1e-9)
    assert JSD_UPPER_BOUND == pytest.approx(0.6931, abs=1e-4)


def test_jsd_never_exceeds_the_bound():
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = rng.dirichlet(np.ones(6))
        q = rng.dirichlet(np.ones(6))
        assert jsd(p, q) <= JSD_UPPER_BOUND + 1e-9


def test_jsd_is_symmetric():
    p, q = np.array([0.1, 0.2, 0.7]), np.array([0.5, 0.4, 0.1])
    assert jsd(p, q) == pytest.approx(jsd(q, p))


# --------------------------------------------------------------------------- #
# Kendall tau over unpadded positions
# --------------------------------------------------------------------------- #
def test_perfect_agreement_gives_tau_one():
    attention = np.array([[0.1, 0.2, 0.3, 0.4]])
    importance = np.array([[1.0, 2.0, 3.0, 4.0]])
    mask = np.ones((1, 4))
    taus, _ = kendall_tau_per_instance(attention, importance, mask)
    assert taus[0] == pytest.approx(1.0)


def test_perfect_disagreement_gives_tau_minus_one():
    attention = np.array([[0.1, 0.2, 0.3, 0.4]])
    importance = np.array([[4.0, 3.0, 2.0, 1.0]])
    mask = np.ones((1, 4))
    taus, _ = kendall_tau_per_instance(attention, importance, mask)
    assert taus[0] == pytest.approx(-1.0)


def test_padding_is_excluded_from_the_correlation():
    """Padding positions must not enter tau, or every long batch reads as correlated.

    Here the real tokens disagree perfectly while the padding agrees perfectly. If
    the mask were ignored the tau would come out positive instead of -1.
    """
    attention = np.array([[0.4, 0.3, 0.0, 0.0]])
    importance = np.array([[1.0, 2.0, 9.0, 9.9]])
    mask = np.array([[1.0, 1.0, 0.0, 0.0]])
    taus, _ = kendall_tau_per_instance(attention, importance, mask)
    assert taus[0] == pytest.approx(-1.0)


def test_instances_too_short_to_correlate_are_nan_not_zero():
    """A one-token instance has no correlation, and calling it 0.0 would drag the
    reported mean toward zero, which is the direction of the paper's own claim."""
    attention = np.array([[0.9, 0.0]])
    importance = np.array([[1.0, 0.0]])
    mask = np.array([[1.0, 0.0]])
    taus, _ = kendall_tau_per_instance(attention, importance, mask)
    assert np.isnan(taus[0])


def test_constant_side_is_skipped():
    attention = np.array([[0.25, 0.25, 0.25, 0.25]])
    importance = np.array([[1.0, 2.0, 3.0, 4.0]])
    mask = np.ones((1, 4))
    taus, _ = kendall_tau_per_instance(attention, importance, mask)
    assert np.isnan(taus[0])


# --------------------------------------------------------------------------- #
# Summary in the shape of the paper's Table 2
# --------------------------------------------------------------------------- #
def test_summarise_reports_significant_fraction():
    taus = np.array([0.5, 0.4, np.nan, 0.3])
    pvals = np.array([0.01, 0.20, np.nan, 0.04])
    out = summarise(taus, pvals)
    assert out["n"] == 3
    assert out["mean"] == pytest.approx(0.4)
    assert out["sig_frac"] == pytest.approx(2 / 3)


def test_summarise_survives_an_all_nan_input():
    out = summarise(np.array([np.nan, np.nan]), np.array([np.nan, np.nan]))
    assert out["n"] == 0
    assert np.isnan(out["mean"])
