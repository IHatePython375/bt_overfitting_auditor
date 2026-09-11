"""
Validation for sharpe.py.

Every test here checks against a number derived independently -- either
analytically or from the reference paper -- not against whatever the code
happened to print the first time. That distinction is the whole point of a
reference implementation: it is the thing the fast C++ version gets
differential-tested against later.
"""

import numpy as np
import pytest
from scipy.stats import norm

from auditor.sharpe import (
    annualize,
    deannualize,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    moments,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
    sharpe_standard_error,
)


# --------------------------------------------------------------------------
# Standard error
# --------------------------------------------------------------------------

def test_se_reduces_to_lo_2002_under_normality():
    """With g3=0, g4=3 the formula must collapse to sqrt((1+SR^2/2)/(T-1))."""
    sr, T = 0.1, 1000
    expected = np.sqrt((1 + 0.5 * sr ** 2) / (T - 1))
    assert sharpe_standard_error(sr, T, 0.0, 3.0) == pytest.approx(expected)


def test_se_matches_hand_computation():
    """5y daily, true SR 0, normal -> 0.0282 per period, 0.4475 annualized."""
    se = sharpe_standard_error(0.0, 1260, 0.0, 3.0)
    assert se == pytest.approx(np.sqrt(1 / 1259), rel=1e-12)
    assert annualize(se) == pytest.approx(0.447391, abs=1e-5)


def test_negative_skew_inflates_standard_error():
    """The -g3*SR term means crash-prone strategies have WIDER error bars."""
    base = sharpe_standard_error(0.1, 1000, 0.0, 3.0)
    negskew = sharpe_standard_error(0.1, 1000, -2.0, 3.0)
    assert negskew > base


def test_fat_tails_inflate_standard_error():
    base = sharpe_standard_error(0.1, 1000, 0.0, 3.0)
    fat = sharpe_standard_error(0.1, 1000, 0.0, 9.0)
    assert fat > base


def test_se_rejects_invalid_variance_term():
    with pytest.raises(ValueError, match="non-positive"):
        sharpe_standard_error(5.0, 1000, 10.0, 3.0)


# --------------------------------------------------------------------------
# Moments
# --------------------------------------------------------------------------

def test_moments_use_raw_kurtosis_convention():
    """Normal data must give kurtosis near 3.0, not near 0.0."""
    rng = np.random.default_rng(0)
    g3, g4 = moments(rng.standard_normal(200_000))
    assert g3 == pytest.approx(0.0, abs=0.02)
    assert g4 == pytest.approx(3.0, abs=0.05)


# --------------------------------------------------------------------------
# Expected max -- the luck bar
# --------------------------------------------------------------------------

def test_expected_max_matches_hand_computation():
    """sigma_sr = 0.45: N=100 -> ~1.139, N=1000 -> ~1.466 (annualized units)."""
    assert expected_max_sharpe(100, 0.45) == pytest.approx(1.139, abs=0.005)
    assert expected_max_sharpe(1000, 0.45) == pytest.approx(1.466, abs=0.005)


def test_expected_max_is_monotone_in_trials():
    vals = [expected_max_sharpe(n, 0.45) for n in (2, 10, 100, 1000, 10000)]
    assert all(a < b for a, b in zip(vals, vals[1:]))


def test_expected_max_scales_linearly_in_sigma():
    assert expected_max_sharpe(100, 0.90) == pytest.approx(
        2 * expected_max_sharpe(100, 0.45)
    )


def test_expected_max_grows_sublinearly_in_log_n():
    """
    Damage is front-loaded: going 100 -> 1000 buys more than 1000 -> 10000,
    even though both are a 10x increase in trials.
    """
    a = expected_max_sharpe(1000, 0.45) - expected_max_sharpe(100, 0.45)
    b = expected_max_sharpe(10000, 0.45) - expected_max_sharpe(1000, 0.45)
    assert a > b


def test_expected_max_against_monte_carlo():
    """
    The Gumbel approximation should track an actual simulated maximum.

    This is the test that would catch a wrong Euler-Mascheroni constant or a
    flipped quantile -- the formula is opaque enough that the algebra alone
    is not convincing.
    """
    rng = np.random.default_rng(42)
    n_trials, sigma = 100, 0.45
    sims = rng.standard_normal((20_000, n_trials)) * sigma
    empirical = sims.max(axis=1).mean()
    assert expected_max_sharpe(n_trials, sigma) == pytest.approx(
        empirical, abs=0.02
    )


# --------------------------------------------------------------------------
# PSR / DSR
# --------------------------------------------------------------------------

def test_psr_is_half_when_sharpe_equals_benchmark():
    """Observed exactly at the bar -> genuine coin flip."""
    assert probabilistic_sharpe_ratio(0.1, 1000, 0.1) == pytest.approx(0.5)


def test_psr_matches_explicit_normal_cdf():
    sr, T, bench = 0.08, 1260, 0.02
    se = sharpe_standard_error(sr, T, 0.0, 3.0)
    assert probabilistic_sharpe_ratio(sr, T, bench) == pytest.approx(
        norm.cdf((sr - bench) / se)
    )


def test_dsr_equals_psr_against_the_luck_bar():
    """DSR is not a new formula -- it is PSR with a shifted null."""
    rng = np.random.default_rng(7)
    r = rng.standard_normal(1260) * 0.01 + 0.0005
    out = deflated_sharpe_ratio(r, n_trials=100, sigma_sr=deannualize(0.45))
    manual = probabilistic_sharpe_ratio(
        out["sharpe"], out["n_obs"], out["luck_bar"], out["skew"], out["kurtosis"]
    )
    assert out["dsr"] == pytest.approx(manual)


def test_dsr_is_always_below_psr():
    """Raising the bar can only lower the probability."""
    rng = np.random.default_rng(11)
    r = rng.standard_normal(1260) * 0.01 + 0.0008
    out = deflated_sharpe_ratio(r, n_trials=100, sigma_sr=deannualize(0.45))
    assert out["dsr"] < out["psr"]


def test_dsr_falls_as_trial_count_rises():
    """Same returns, more things tried, less credible. The core behaviour."""
    rng = np.random.default_rng(3)
    r = rng.standard_normal(1260) * 0.01 + 0.0009
    sig = deannualize(0.45)
    vals = [
        deflated_sharpe_ratio(r, n, sig)["dsr"] for n in (10, 100, 1000, 10000)
    ]
    assert all(a > b for a, b in zip(vals, vals[1:]))


def test_dsr_kills_a_lucky_winner_from_pure_noise():
    """
    End-to-end. Generate 200 strategies with ZERO true edge, select the best,
    and confirm PSR is fooled while DSR is not.

    This is the demo in miniature.
    """
    rng = np.random.default_rng(2024)
    n_trials, T = 200, 1260
    panel = rng.standard_normal((T, n_trials)) * 0.01  # zero mean: no edge

    sharpes = np.array([sharpe_ratio(panel[:, i]) for i in range(n_trials)])
    winner = panel[:, int(sharpes.argmax())]

    out = deflated_sharpe_ratio(winner, n_trials, sharpes.std(ddof=1))

    assert out["sharpe_annual"] > 0.8      # looks like a real strategy
    assert out["psr"] > 0.95               # naive test is fooled
    assert out["dsr"] < 0.60               # deflated test is not