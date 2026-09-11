"""
Validation for bootstrap.py.

The two substantive tests here are the ones with closed-form targets:

  - long-run variance recovery. For AR(1) both the marginal variance and the
    long-run variance are known exactly, and they differ. A block bootstrap
    that works recovers the long-run one; an iid bootstrap recovers the
    marginal one. The gap between them is the entire reason this module
    exists, so it is worth asserting both ends.

  - the Politis-White selector, against the exact AR(1) optimum derived
    below rather than against a monotonicity check.

Everything else pins a structural property of the scheme.
"""

import numpy as np
import pytest
from scipy.stats import chisquare, geom

from auditor.bootstrap import (
    politis_white_block_length,
    rule_of_thumb_block_length,
    stationary_bootstrap_indices,
)


def ar1(n_obs, phi, sigma_e=1.0, rng=None, burn=500):
    """
    AR(1): x_t = phi*x_{t-1} + sigma_e*e_t, started from its stationary
    distribution so no burn-in bias leaks into the variance targets.
    """
    rng = rng or np.random.default_rng(0)
    x = np.empty(n_obs + burn)
    x[0] = rng.normal(0.0, sigma_e / np.sqrt(1.0 - phi ** 2))
    shocks = rng.normal(0.0, sigma_e, size=n_obs + burn)
    for t in range(1, n_obs + burn):
        x[t] = phi * x[t - 1] + shocks[t]
    return x[burn:]


def ar1_marginal_variance(phi, sigma_e=1.0):
    """Var(x_t) = sigma_e^2 / (1 - phi^2)."""
    return sigma_e ** 2 / (1.0 - phi ** 2)


def ar1_long_run_variance(phi, sigma_e=1.0):
    """
    lim T*Var(xbar) = sum_k Cov(x_0, x_k) = sigma_e^2 / (1 - phi)^2.

    Strictly larger than the marginal variance for phi > 0, which is exactly
    why an iid bootstrap understates uncertainty on dependent data.
    """
    return sigma_e ** 2 / (1.0 - phi) ** 2


def ar1_optimal_block_length(phi, n_obs):
    """
    Politis & White's optimum for the stationary bootstrap, evaluated
    analytically for AR(1).

    With R(k) = sigma^2 * phi^|k|:
        G    = sum_k |k| R(k)  = 2*sigma^2*phi / (1-phi)^2
        g(0) = sum_k R(k)      = sigma^2*(1+phi) / (1-phi)
    and since D_SB = 2*g(0)^2 the leading 2 cancels:
        b = (2*G^2 / D_SB)^(1/3) * T^(1/3) = (G/g(0))^(2/3) * T^(1/3)
    with G/g(0) = 2*phi/(1-phi^2). Sigma cancels entirely.
    """
    return (2.0 * phi / (1.0 - phi ** 2)) ** (2.0 / 3.0) * n_obs ** (1.0 / 3.0)


# --------------------------------------------------------------------------
# Structure of the resampled indices
# --------------------------------------------------------------------------

def test_indices_are_valid_and_shaped():
    idx = stationary_bootstrap_indices(200, 10.0, 50, np.random.default_rng(0))
    assert idx.shape == (50, 200)
    assert idx.min() >= 0 and idx.max() < 200


def test_every_observation_is_equally_likely():
    """
    The stationarity property, and the one thing the circular wrap buys.

    Without wrapping, observations near the end of the series can only ever
    appear at the end of a block, so they are under-sampled and the marginal
    distribution is not uniform. Chi-square against uniform catches that.
    """
    n_obs = 50
    idx = stationary_bootstrap_indices(n_obs, 8.0, 4000, np.random.default_rng(1))
    counts = np.bincount(idx.ravel(), minlength=n_obs)
    expected = np.full(n_obs, counts.sum() / n_obs)
    assert chisquare(counts, expected).pvalue > 0.01


def test_indices_wrap_around_the_end():
    """A block that runs off the end continues at observation 0."""
    n_obs = 40
    idx = stationary_bootstrap_indices(n_obs, 15.0, 500, np.random.default_rng(2))
    wraps = (idx[:, :-1] == n_obs - 1) & (idx[:, 1:] == 0)
    assert wraps.any()


@pytest.mark.parametrize("block_length", [4.0, 10.0, 25.0])
def test_block_lengths_are_geometric(block_length):
    """
    Run lengths of consecutive indices must be Geometric(1/L), mean L.

    Measured runs are very slightly long because a restart can land by chance
    on the next index (probability 1/T per restart), which merges two runs.
    At T = 400 that is a sub-percent effect, well inside the tolerance.
    """
    n_obs = 400
    idx = stationary_bootstrap_indices(n_obs, block_length, 2000, np.random.default_rng(3))

    continues = idx[:, 1:] == (idx[:, :-1] + 1) % n_obs
    # P(continue) = 1 - p is the direct handle on the geometric parameter.
    assert continues.mean() == pytest.approx(1.0 - 1.0 / block_length, abs=0.01)

    run_lengths = []
    for row in idx[:200]:
        breaks = np.flatnonzero(np.diff(row) != 1)
        run_lengths.extend(np.diff(np.concatenate([[-1], breaks, [len(row) - 1]])))
    run_lengths = np.array(run_lengths)
    assert run_lengths.mean() == pytest.approx(block_length, rel=0.15)

    # Compare the head of the distribution against the geometric pmf.
    top = 5
    observed = np.array([(run_lengths == k).sum() for k in range(1, top + 1)])
    pmf = geom.pmf(np.arange(1, top + 1), 1.0 / block_length)
    expected = pmf / pmf.sum() * observed.sum()
    assert chisquare(observed, expected).pvalue > 0.01


def test_unit_block_length_is_the_iid_bootstrap():
    """
    L = 1 gives restart probability 1, so indices are iid uniform and runs of
    consecutive observations occur only by coincidence, at rate 1/T.
    """
    n_obs = 200
    idx = stationary_bootstrap_indices(n_obs, 1.0, 500, np.random.default_rng(4))
    continues = (idx[:, 1:] == (idx[:, :-1] + 1) % n_obs).mean()
    assert continues == pytest.approx(1.0 / n_obs, abs=0.005)


def test_same_seed_reproduces():
    a = stationary_bootstrap_indices(100, 6.0, 20, np.random.default_rng(7))
    b = stationary_bootstrap_indices(100, 6.0, 20, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_bad_input_rejected():
    with pytest.raises(ValueError, match="block_length"):
        stationary_bootstrap_indices(100, 0.5, 10)
    with pytest.raises(ValueError, match="observation"):
        stationary_bootstrap_indices(0, 5.0, 10)
    with pytest.raises(ValueError, match="replicate"):
        stationary_bootstrap_indices(100, 5.0, 0)


# --------------------------------------------------------------------------
# The variance it is supposed to reproduce
# --------------------------------------------------------------------------

@pytest.mark.parametrize("phi", [0.0, 0.5, 0.8])
def test_recovers_long_run_variance(phi):
    """
    The test that says the block bootstrap actually works.

    T*Var(xbar*) across replicates estimates the long-run variance of the
    process. Both targets are closed form for AR(1) and differ by a factor of
    (1+phi)/(1-phi) -- 9x at phi = 0.8 -- so the two hypotheses are far apart
    and the test is not a tolerance exercise.
    """
    n_obs, n_boot = 2000, 4000
    rng = np.random.default_rng(11)
    x = ar1(n_obs, phi, rng=rng)

    block_length = max(1.0, ar1_optimal_block_length(phi, n_obs)) if phi > 0 else 1.0
    idx = stationary_bootstrap_indices(n_obs, block_length, n_boot, rng)
    boot_means = x[idx].mean(axis=1)
    estimated = n_obs * boot_means.var(ddof=1)

    assert estimated == pytest.approx(ar1_long_run_variance(phi), rel=0.35)


@pytest.mark.parametrize("phi", [0.5, 0.8])
def test_iid_bootstrap_recovers_only_the_marginal_variance(phi):
    """
    The other half, and the reason block length is not cosmetic.

    At L = 1 the resampled observations are independent, so T*Var(xbar*)
    estimates the MARGINAL variance and misses all the serial dependence.
    Under-blocking therefore understates uncertainty, which is precisely how
    a misspecified block length makes a test over-reject.
    """
    n_obs, n_boot = 2000, 4000
    rng = np.random.default_rng(12)
    x = ar1(n_obs, phi, rng=rng)

    idx = stationary_bootstrap_indices(n_obs, 1.0, n_boot, rng)
    estimated = n_obs * x[idx].mean(axis=1).var(ddof=1)

    assert estimated == pytest.approx(ar1_marginal_variance(phi), rel=0.2)
    assert estimated < 0.7 * ar1_long_run_variance(phi)


# --------------------------------------------------------------------------
# Block-length selection
# --------------------------------------------------------------------------

def test_rule_of_thumb_has_the_right_rate():
    """Theta(T^(1/3)) is the known optimal rate; only the constant is data
    dependent. A 1000x increase in T must multiply the block length by 10."""
    assert rule_of_thumb_block_length(1000) == pytest.approx(10.0)
    assert rule_of_thumb_block_length(1_000_000) == pytest.approx(100.0)


@pytest.mark.slow
@pytest.mark.parametrize("phi", [0.3, 0.5, 0.7])
def test_politis_white_matches_the_ar1_optimum(phi):
    """
    Against the closed form derived in ar1_optimal_block_length, not against
    a printed value.

    Averaged over independent series: the selector is a ratio of two noisy
    spectral estimates and is genuinely high-variance on a single path, so a
    single-sample assertion would be testing luck. The mean over 40 paths is
    what has the sampling behaviour worth pinning.
    """
    n_obs = 1000
    rng = np.random.default_rng(21)
    estimates = [
        politis_white_block_length(ar1(n_obs, phi, rng=rng)) for _ in range(40)
    ]
    assert np.mean(estimates) == pytest.approx(
        ar1_optimal_block_length(phi, n_obs), rel=0.4
    )


@pytest.mark.slow
def test_politis_white_increases_with_dependence():
    """Monotone in phi -- more persistence needs longer blocks."""
    n_obs = 1000
    rng = np.random.default_rng(22)
    means = [
        np.mean([politis_white_block_length(ar1(n_obs, phi, rng=rng)) for _ in range(20)])
        for phi in (0.0, 0.4, 0.8)
    ]
    assert means[0] < means[1] < means[2]


def test_politis_white_on_iid_data_is_near_one():
    """No dependence to preserve, so blocks should collapse."""
    rng = np.random.default_rng(23)
    estimates = [
        politis_white_block_length(rng.standard_normal(1000)) for _ in range(20)
    ]
    assert np.mean(estimates) < 3.0


def test_matrix_aggregation_orders_as_expected():
    """
    One block length must serve all N columns, because resampling is joint.
    max >= 90th quantile >= median by construction; the point of the default
    is that it leans high without inheriting a maximum's N-dependent bias.
    """
    rng = np.random.default_rng(24)
    m = np.column_stack([ar1(800, phi, rng=rng) for phi in (0.0, 0.2, 0.5, 0.8)])

    med = politis_white_block_length(m, aggregate="median")
    q90 = politis_white_block_length(m, aggregate="quantile")
    mx = politis_white_block_length(m, aggregate="max")

    assert med <= q90 <= mx
    with pytest.raises(ValueError, match="unknown aggregate"):
        politis_white_block_length(m, aggregate="mean")
