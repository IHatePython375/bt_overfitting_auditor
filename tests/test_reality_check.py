"""
Validation for reality_check.py.

The null-calibration test is the load-bearing one. A bootstrap test is
characterised by its size, and if the centering term is omitted the p-value
collapses toward 0.5 whatever the data -- which looks healthy in a spot check
and is caught immediately by a uniformity test.

test_bootstrap_distribution_is_location_invariant isolates the same term
algebraically rather than statistically, so between them the two tests pin
the centering from both directions.
"""

import numpy as np
import pytest
from scipy.signal import lfilter
from scipy.stats import kstest

from auditor.reality_check import reality_check


def ar1_panel(n_obs, n_strat, phi, rng, sigma=0.01):
    """
    n_strat independent AR(1) columns, started from the stationary
    distribution. lfilter rather than a Python loop -- the calibration tests
    build a few hundred of these.
    """
    shocks = rng.normal(0.0, sigma, size=(n_obs, n_strat))
    shocks[0] /= np.sqrt(1.0 - phi ** 2)
    return lfilter([1.0], [1.0, -phi], shocks, axis=0)


# --------------------------------------------------------------------------
# Size: the p-value must be uniform under the null
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_null_pvalues_are_uniform():
    """
    Under the least favourable configuration -- every strategy with exactly
    zero expected excess return -- the p-value is uniform on (0, 1).

    This is the strongest available check on a bootstrap procedure, because
    it tests the whole pipeline at once: resampling, centering, the max, and
    the (B+1) denominator. An omitted centering term centres V* on V itself
    and drives every p-value toward 0.5, which fails here with no ambiguity.

    Uniformity is asymptotic, and with B replicates the p-value lives on a
    grid of B+1 points, so the KS test is mildly conservative -- in the safe
    direction. A lenient threshold guards against flagging a correct
    implementation.
    """
    n_exp, n_obs, n_strat, n_boot = 300, 250, 8, 199
    rng = np.random.default_rng(4242)
    p_values = np.array(
        [
            reality_check(
                rng.standard_normal((n_obs, n_strat)) * 0.01,
                n_boot=n_boot,
                rng=rng,
            )["p_value"]
            for _ in range(n_exp)
        ]
    )

    assert kstest(p_values, "uniform").pvalue > 0.01
    # A collapse toward 0.5 is the specific failure mode; check the mean too,
    # since KS is weakest against a symmetric concentration.
    assert p_values.mean() == pytest.approx(0.5, abs=0.06)
    assert p_values.std() == pytest.approx(1 / np.sqrt(12), abs=0.05)


@pytest.mark.slow
def test_rejection_rate_matches_nominal_size():
    """
    A direct read of the same property at the threshold people actually use:
    a 5% test should reject 5% of the time under the null.
    """
    n_exp = 400
    rng = np.random.default_rng(909)
    rejects = sum(
        reality_check(
            rng.standard_normal((250, 8)) * 0.01, n_boot=199, rng=rng
        )["p_value"]
        <= 0.05
        for _ in range(n_exp)
    )
    # Binomial(400, 0.05) has sd 4.36, so +-3 sd is about +-13 rejections.
    assert abs(rejects - 0.05 * n_exp) < 13


# --------------------------------------------------------------------------
# Power
# --------------------------------------------------------------------------

def test_planted_alpha_is_detected():
    """
    One strategy with a real edge, sized so its t-statistic is around 6 --
    far enough above the multiplicity-corrected critical value that a correct
    test must reject, without being so extreme that the test is vacuous.
    """
    n_obs, n_strat, sigma = 600, 10, 0.01
    rng = np.random.default_rng(5)
    m = rng.standard_normal((n_obs, n_strat)) * sigma
    m[:, 3] += 6.0 * sigma / np.sqrt(n_obs)     # t-stat ~ 6

    result = reality_check(m, n_boot=999, rng=rng)
    assert result["best_strategy"] == 3
    assert result["p_value"] < 0.01


def test_multiplicity_correction_is_visible():
    """
    The corrected p-value must exceed the uncorrected one for the same
    strategy: searching over N candidates can only make the best one's margin
    less surprising, never more.

    The gap between them is the data-snooping adjustment, which is the whole
    point of returning both.
    """
    rng = np.random.default_rng(17)
    m = rng.standard_normal((400, 40)) * 0.01
    result = reality_check(m, n_boot=999, rng=rng)
    assert result["naive_p_value"] <= result["p_value"]


def test_single_strategy_needs_no_correction():
    """With N = 1 there is nothing to search over, so the two agree exactly."""
    rng = np.random.default_rng(18)
    m = rng.standard_normal((300, 1)) * 0.01
    result = reality_check(m, n_boot=499, rng=rng)
    assert result["p_value"] == result["naive_p_value"]


# --------------------------------------------------------------------------
# The known pathology, demonstrated rather than hidden
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_power_falls_as_useless_strategies_are_added():
    """
    White's RC loses power when obviously-bad strategies are included. This
    is the motivation for Hansen's SPA, and it belongs in the calibration
    study as a measured effect rather than a footnote.

    Mechanism, and why the direction is derived rather than tuned: the least
    favourable configuration recentres EVERY strategy to mean zero, including
    ones with a large negative mean that nobody would ever select. Those
    strategies contribute nothing to V -- the good strategy still wins -- but
    each one adds another term to max_k in the bootstrap null, so V* is
    stochastically larger and p = P(V* >= V) rises.

    The inequality is DETERMINISTIC, not statistical. Both runs are handed
    the same seed, and the bootstrap indices depend only on (T, L, B), which
    the padding does not change -- so both see identical resampled rows, and
    the padded run's first six columns are the unpadded run's columns. Hence
    V*_many = max(V*_few, the 95 extra) >= V*_few in EVERY replicate, so
    p_many >= p_few for every seed with no appeal to sampling. Ties occur
    where both p-values sit on the 1/(B+1) floor.
    """
    n_obs, sigma = 500, 0.01
    edge = 4.0 * sigma / np.sqrt(n_obs)

    p_few, p_many, stat_few, stat_many = [], [], [], []
    for seed in range(20):
        rng = np.random.default_rng(1000 + seed)
        good = rng.standard_normal((n_obs, 1)) * sigma + edge
        bad = rng.standard_normal((n_obs, 100)) * sigma - 5.0 * sigma / np.sqrt(n_obs)

        few = reality_check(np.hstack([good, bad[:, :5]]), n_boot=299,
                            rng=np.random.default_rng(seed))
        many = reality_check(np.hstack([good, bad]), n_boot=299,
                             rng=np.random.default_rng(seed))

        p_few.append(few["p_value"])
        p_many.append(many["p_value"])
        stat_few.append(few["statistic"])
        stat_many.append(many["statistic"])
        assert few["best_strategy"] == 0 and many["best_strategy"] == 0

    p_few, p_many = np.array(p_few), np.array(p_many)

    # The observed statistic is untouched: same winner, same margin.
    np.testing.assert_allclose(stat_few, stat_many, rtol=1e-12)

    # Only the null distribution moved. Weak inequality everywhere, by the
    # pointwise argument above.
    assert np.all(p_many >= p_few)

    # And the loss of power is substantial, not marginal: padding with 95
    # strategies that are all visibly worse than the benchmark roughly
    # quadruples the median p-value.
    assert np.median(p_many) > 3.0 * np.median(p_few)
    strictly_worse = p_few > 1.0 / 300.0          # room above the floor
    assert np.all(p_many[strictly_worse] > p_few[strictly_worse])


# --------------------------------------------------------------------------
# Block length
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_misspecified_block_length_over_rejects_under_autocorrelation():
    """
    With positively autocorrelated returns, the long-run variance exceeds the
    marginal variance -- at phi = 0.5 by a factor (1+phi)/(1-phi) = 3. An iid
    bootstrap (L = 1) only ever reproduces the marginal variance, so it
    understates the spread of fbar, shrinks the critical value, and rejects a
    true null far too often.

    A correctly blocked bootstrap should sit near nominal size. The direction
    is derived from the variance identity, not calibrated -- and it is a small
    rehearsal of the week-4 argument that ignoring serial correlation biases
    a detector.
    """
    n_exp, n_obs, n_strat, phi = 250, 300, 6, 0.5
    blocked = max(2.0, n_obs ** (1 / 3))

    rejects_iid = rejects_blocked = 0
    for seed in range(n_exp):
        rng = np.random.default_rng(7000 + seed)
        m = ar1_panel(n_obs, n_strat, phi, rng)
        rejects_iid += (
            reality_check(m, block_length=1.0, n_boot=199,
                          rng=np.random.default_rng(seed))["p_value"] <= 0.05
        )
        rejects_blocked += (
            reality_check(m, block_length=blocked, n_boot=199,
                          rng=np.random.default_rng(seed))["p_value"] <= 0.05
        )

    rate_iid = rejects_iid / n_exp
    rate_blocked = rejects_blocked / n_exp
    assert rate_iid > 0.10, f"iid bootstrap should over-reject, got {rate_iid:.3f}"
    assert rate_blocked < rate_iid
    assert rate_blocked < 0.15


def test_block_length_moves_the_pvalue_on_dependent_data():
    """Same data, different L, different answer -- so L is not cosmetic and
    is rightly an explicit parameter rather than a hidden default."""
    rng = np.random.default_rng(31)
    m = ar1_panel(400, 6, 0.6, rng)
    short = reality_check(m, block_length=1.0, n_boot=999, rng=np.random.default_rng(3))
    long = reality_check(m, block_length=20.0, n_boot=999, rng=np.random.default_rng(3))
    assert short["p_value"] != long["p_value"]
    assert short["statistic"] == long["statistic"]


# --------------------------------------------------------------------------
# The centering term, isolated algebraically
# --------------------------------------------------------------------------

def test_bootstrap_distribution_is_location_invariant():
    """
    The sharpest possible check on the centering term, and it needs no
    Monte Carlo.

    Add a constant c to every strategy's returns. Then fbar_k -> fbar_k + c
    and fbar*_k -> fbar*_k + c, so the centred difference is UNCHANGED: the
    bootstrap distribution must be bitwise identical. The observed statistic,
    which is deliberately uncentred, shifts by sqrt(T)*c.

    If the centering term were omitted, V* would shift by sqrt(T)*c as well,
    the two shifts would cancel, and the p-value would not move at all. So
    this single test separates correct centering from no centering by a
    property rather than by a tolerance.

    The invariance is exact in real arithmetic but not bitwise: summing 400
    shifted numbers and then averaging does not round identically to
    averaging and then shifting, so the two distributions differ by ~4e-17.
    The tolerance below is therefore 1e-10 while the effect it is guarding
    against -- an uncentred V* shifting with the data -- would be
    sqrt(400)*0.002 = 0.04. Seven orders of magnitude of margin.
    """
    rng = np.random.default_rng(77)
    m = rng.standard_normal((400, 6)) * 0.01
    shift = 0.002

    base = reality_check(m, n_boot=499, rng=np.random.default_rng(1))
    moved = reality_check(m + shift, n_boot=499, rng=np.random.default_rng(1))

    np.testing.assert_allclose(
        base["bootstrap_distribution"],
        moved["bootstrap_distribution"],
        rtol=1e-10,
        atol=1e-15,
    )
    assert moved["statistic"] == pytest.approx(
        base["statistic"] + np.sqrt(400) * shift
    )
    assert moved["p_value"] < base["p_value"]


def test_benchmark_subtraction_is_equivalent_to_shifting_returns():
    """Passing a benchmark series must equal subtracting it beforehand."""
    rng = np.random.default_rng(88)
    m = rng.standard_normal((300, 5)) * 0.01
    bench = rng.standard_normal(300) * 0.005

    with_bench = reality_check(m, benchmark=bench, n_boot=299,
                               rng=np.random.default_rng(2))
    pre_subtracted = reality_check(m - bench[:, None], n_boot=299,
                                   rng=np.random.default_rng(2))
    assert with_bench["p_value"] == pre_subtracted["p_value"]
    np.testing.assert_allclose(
        with_bench["mean_relative"], pre_subtracted["mean_relative"]
    )


# --------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------

def test_result_dict_is_inspectable():
    rng = np.random.default_rng(41)
    m = rng.standard_normal((300, 7)) * 0.01
    result = reality_check(m, n_boot=499, rng=rng)

    for key in (
        "p_value", "naive_p_value", "statistic", "bootstrap_distribution",
        "mean_relative", "best_strategy", "block_length", "n_boot",
        "n_obs", "n_strategies",
    ):
        assert key in result

    assert result["bootstrap_distribution"].shape == (499,)
    assert result["mean_relative"].shape == (7,)
    assert result["n_obs"] == 300 and result["n_strategies"] == 7
    assert result["block_length"] == pytest.approx(300 ** (1 / 3))

    # p can never be 0: the observed statistic is in its own reference set.
    assert result["p_value"] >= 1.0 / (result["n_boot"] + 1.0)
    assert result["p_value"] <= 1.0
    assert result["best_strategy"] == int(np.argmax(result["mean_relative"]))


def test_pvalue_uses_the_b_plus_one_denominator():
    """
    An overwhelming edge should drive the p-value to its floor, and that
    floor must be 1/(B+1) rather than 0 -- finitely many replicates cannot
    support infinite evidence.
    """
    rng = np.random.default_rng(42)
    m = rng.standard_normal((400, 5)) * 0.01
    m[:, 1] += 0.02                                   # enormous edge
    for n_boot in (99, 499):
        result = reality_check(m, n_boot=n_boot, rng=np.random.default_rng(0))
        assert result["p_value"] == pytest.approx(1.0 / (n_boot + 1.0))


def test_bad_input_rejected():
    rng = np.random.default_rng(43)
    with pytest.raises(ValueError, match="matrix"):
        reality_check(rng.standard_normal(100))
    with pytest.raises(ValueError, match="non-finite"):
        reality_check(np.full((100, 3), np.nan))
    with pytest.raises(ValueError, match="benchmark"):
        reality_check(rng.standard_normal((100, 3)), benchmark=np.zeros(50))
    with pytest.raises(ValueError, match="replicate"):
        reality_check(rng.standard_normal((100, 3)), n_boot=0)
