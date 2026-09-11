"""
Validation for pbo.py.

Same rule as test_sharpe.py: every assertion is against a number derived
independently of the implementation -- an exact combinatorial identity, an
exchangeability argument, a deliberately constructed adversarial matrix, or
Monte Carlo. Nothing here is pinned to whatever the code printed first.

The differential tests between the naive and sufficient-statistics paths
carry extra weight: the sufstats path is what the week-2 C++ kernel mirrors,
so a bad block decomposition caught here is a bug that never reaches C++.
"""

import numpy as np
import pytest
from math import comb
from scipy.stats import chisquare

from auditor.pbo import (
    DEGENERATE_SCORE,
    block_stats,
    make_blocks,
    n_partitions,
    partition_lambda,
    pbo,
    sharpe_direct,
    sharpe_from_stats,
)


# --------------------------------------------------------------------------
# Block construction
# --------------------------------------------------------------------------

def test_blocks_partition_the_whole_axis():
    """Blocks must tile [0, T) exactly: contiguous, no gaps, no overlap."""
    for n_obs in (100, 101, 250, 1000):
        for n_blocks in (2, 4, 10, 16):
            blocks = make_blocks(n_obs, n_blocks)
            assert len(blocks) == n_blocks
            assert blocks[0][0] == 0
            assert blocks[-1][1] == n_obs
            for (_, prev_stop), (start, _) in zip(blocks, blocks[1:]):
                assert prev_stop == start
            assert sum(stop - start for start, stop in blocks) == n_obs


def test_uneven_split_loses_no_observations():
    """
    997 into 16 blocks: 5 blocks of 63 and 11 of 62. Truncation would drop
    the 13-observation remainder; distributing keeps all of it.
    """
    blocks = make_blocks(997, 16)
    sizes = sorted(stop - start for start, stop in blocks)
    assert sizes == [62] * 11 + [63] * 5
    assert sum(sizes) == 997


def test_odd_block_count_rejected():
    """S must be even, or there is no balanced split."""
    with pytest.raises(ValueError, match="even"):
        make_blocks(100, 15)
    with pytest.raises(ValueError, match="even"):
        n_partitions(7)


def test_partition_count_is_the_binomial_coefficient():
    """C(S, S/2) exactly -- both directions of each complementary pair."""
    assert n_partitions(16) == 12_870
    assert n_partitions(24) == 2_704_156
    assert n_partitions(32) == 601_080_390

    rng = np.random.default_rng(0)
    m = rng.standard_normal((200, 4))
    for n_blocks in (4, 6, 8):
        assert pbo(m, n_blocks)["n_partitions"] == comb(n_blocks, n_blocks // 2)


# --------------------------------------------------------------------------
# The two scoring paths agree
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "n_blocks", [12, pytest.param(14, marks=pytest.mark.slow)]
)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_naive_and_sufstats_agree(n_blocks, seed):
    """
    The differential test the C++ kernel will reuse verbatim.

    Asserts on the lambda vector, not just the headline PBO. PBO is a mean of
    indicator functions, so it can agree by luck while individual partitions
    disagree; lambda is the finest-grained per-partition output there is.
    """
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((700, 8)) * 0.01 + 0.0003

    slow = pbo(m, n_blocks, method="naive")
    fast = pbo(m, n_blocks, method="sufstats")

    assert slow["n_partitions"] == fast["n_partitions"]
    assert np.array_equal(slow["n_star"], fast["n_star"])
    assert np.array_equal(slow["oos_ranks"], fast["oos_ranks"])
    np.testing.assert_allclose(slow["lambdas"], fast["lambdas"], rtol=1e-9, atol=1e-12)
    assert slow["pbo"] == fast["pbo"]


def test_sufstats_scores_match_direct_scores():
    """
    The decomposition itself, one level below the full detector: pooling
    per-block moments must reproduce the Sharpe of the concatenated rows.

    Averaging per-block Sharpes would also "work" on symmetric data and is a
    different number. This pins the right one.
    """
    rng = np.random.default_rng(7)
    m = rng.standard_normal((480, 6)) * 0.02
    blocks = make_blocks(480, 8)
    counts, sums, sumsq = block_stats(m, blocks)

    for chosen in [(0, 1, 2, 3), (0, 2, 4, 6), (5, 6, 7, 1), (3,)]:
        idx = np.array(chosen, dtype=np.int64)
        rows = np.concatenate([np.arange(*blocks[b]) for b in chosen])
        np.testing.assert_allclose(
            sharpe_from_stats(counts, sums, sumsq, idx),
            sharpe_direct(m, rows),
            rtol=1e-10,
            atol=1e-13,
        )


def _twins(separation: float, seed: int = 11) -> np.ndarray:
    """Six strategies, the last two separated ADDITIVELY by `separation`."""
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((600, 6)) * 0.01
    m[:, 5] = m[:, 4] + separation * rng.standard_normal(600) * 0.01
    return m


@pytest.mark.parametrize("separation", [1e-12, 1e-8, 1e-6])
def test_separated_strategies_give_identical_paths(separation):
    """
    Above the accumulation error the two paths must agree EXACTLY -- same
    winners, same ranks, same lambdas.

    The sufficient-statistics identity is exact in real arithmetic; the paths
    differ only in the order they accumulate, which perturbs each Sharpe by
    order 1e-15 relative. Any two strategies separated by more than that are
    ordered identically by both. Measured: a 1e-12 additive separation
    already produces in-sample Sharpe gaps of ~1e-15, and divergence stops.
    """
    m = _twins(separation)
    slow = pbo(m, 10, method="naive")
    fast = pbo(m, 10, method="sufstats")

    assert np.array_equal(slow["n_star"], fast["n_star"])
    assert np.array_equal(slow["oos_ranks"], fast["oos_ranks"])
    np.testing.assert_allclose(slow["lambdas"], fast["lambdas"], rtol=1e-12, atol=0)
    assert slow["pbo"] == fast["pbo"]


def test_exactly_identical_columns_are_safe():
    """
    Perfect duplicates do NOT diverge, which is the counter-intuitive half.

    Identical inputs run through identical arithmetic give bitwise identical
    scores in each path separately, and argmax then breaks the tie by index
    deterministically in both. The hazard is near-ties, not ties.
    """
    m = _twins(0.0)
    assert np.array_equal(m[:, 4], m[:, 5])
    slow = pbo(m, 10, method="naive")
    fast = pbo(m, 10, method="sufstats")
    assert np.array_equal(slow["n_star"], fast["n_star"])
    assert np.array_equal(slow["oos_ranks"], fast["oos_ranks"])
    assert slow["pbo"] == fast["pbo"]


def test_near_tie_divergence_is_confined_to_the_tied_pair():
    """
    The failure mode worth knowing before writing any C++.

    In the narrow band where two strategies are separated by LESS than the
    accumulation error (~1e-15 in Sharpe), the two paths can order them
    differently. Both the in-sample argmax and the out-of-sample ranking are
    discontinuous at a tie, so lambda jumps by a visible amount rather than
    by 1e-15 -- here 17 of 252 partitions.

    The invariant is not "the paths always agree". It is that divergence is
    confined to the tied pair:
      - every divergent partition is won by one of the twins in both paths;
      - the winner's out-of-sample rank moves by at most 1, i.e. the twins
        swap adjacent positions and nothing else reorders;
      - PBO therefore moves by at most (divergent partitions) / (partitions),
        and in practice by nothing at all.

    Anything outside that would be a wrong decomposition, not float noise.
    This is the check that survives being re-ported to C++, where the
    accumulation order changes again.
    """
    m = _twins(1e-16)
    slow = pbo(m, 10, method="naive")
    fast = pbo(m, 10, method="sufstats")

    divergent = np.flatnonzero(slow["lambdas"] != fast["lambdas"])
    assert divergent.size > 0, "expected the near-tie band to bite; it did not"
    assert divergent.size < 0.10 * slow["n_partitions"]

    assert np.isin(slow["n_star"][divergent], [4, 5]).all()
    assert np.isin(fast["n_star"][divergent], [4, 5]).all()

    rank_shift = np.abs(slow["oos_ranks"][divergent] - fast["oos_ranks"][divergent])
    assert rank_shift.max() <= 1.0

    # Each divergent partition can flip at most one indicator, so this bound
    # is exact rather than a tuned tolerance.
    assert abs(slow["pbo"] - fast["pbo"]) <= divergent.size / slow["n_partitions"]


def test_leverage_variants_tie_exactly():
    """
    Sharpe is scale invariant, so a strategy and a leveraged copy of it have
    exactly the same score -- for any leverage factor, not approximately.

    This is not a corner case. A parameter sweep over position size or
    leverage produces columns that are exact scalar multiples of each other,
    so a sweep of N variants can contain far fewer than N distinct Sharpes.
    That makes the average-rank tie policy load-bearing on real inputs, and
    it is also why N in a sweep is not automatically the right n_trials for
    the Deflated Sharpe Ratio.
    """
    rng = np.random.default_rng(21)
    m = rng.standard_normal((400, 4)) * 0.01
    m[:, 3] = m[:, 2] * 2.5          # same strategy, 2.5x the size

    rows = np.arange(400)
    scores = sharpe_direct(m, rows)
    assert scores[3] == pytest.approx(scores[2], rel=1e-12)

    slow = pbo(m, 8, method="naive")
    fast = pbo(m, 8, method="sufstats")
    assert slow["pbo"] == fast["pbo"]


# --------------------------------------------------------------------------
# Relative rank and tie handling
# --------------------------------------------------------------------------

def test_lambda_is_finite_at_both_extremes():
    """
    rank/(N+1), not rank/N.

    A winner that also finishes first out-of-sample has rank N; with rank/N
    that is w = 1 and lambda = +inf. PBO would still read correctly, which is
    exactly what makes the bug survive casual testing -- so assert on lambda
    directly, at both ends.
    """
    n_strat = 5
    best = np.arange(n_strat, dtype=float)          # strategy 4 is best
    worst = np.arange(n_strat, dtype=float)[::-1]   # strategy 4 is worst

    _, rank_hi, w_hi, lam_hi = partition_lambda(best, best)
    assert rank_hi == n_strat
    assert w_hi == pytest.approx(5 / 6)
    assert np.isfinite(lam_hi) and lam_hi > 0

    _, rank_lo, w_lo, lam_lo = partition_lambda(best, worst)
    assert rank_lo == 1
    assert w_lo == pytest.approx(1 / 6)
    assert np.isfinite(lam_lo) and lam_lo < 0

    # Symmetric by construction: logit(w) = -logit(1-w).
    assert lam_hi == pytest.approx(-lam_lo)


def test_ties_get_average_ranks_not_positional_ones():
    """
    All strategies identical out-of-sample: every rank is the average,
    (N+1)/2, so w = 1/2 and lambda = 0 exactly -- no strategy is favoured.

    With argsort's index-order tie-breaking the winner's rank would depend on
    its column position, which biases toward whichever variant sits earlier
    in the matrix. Parameter sweeps are usually generated in sorted order, so
    that is a bias toward smaller parameter values, not a random one.
    """
    n_strat = 6
    flat = np.zeros(n_strat)
    for winner in range(n_strat):
        is_scores = np.full(n_strat, -1.0)
        is_scores[winner] = 1.0
        _, rank, w, lam = partition_lambda(is_scores, flat)
        assert rank == pytest.approx((n_strat + 1) / 2)
        assert w == pytest.approx(0.5)
        assert lam == pytest.approx(0.0)


def test_duplicate_columns_do_not_move_pbo():
    """Appending an exact copy of a column changes N but not the verdict."""
    rng = np.random.default_rng(3)
    m = rng.standard_normal((400, 5)) * 0.01
    doubled = np.hstack([m, m[:, :1]])
    assert pbo(m, 8)["pbo"] == pytest.approx(pbo(doubled, 8)["pbo"], abs=0.1)


# --------------------------------------------------------------------------
# Analytic limits
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_null_winner_rank_is_uniform():
    """
    The exchangeability argument, checked by Monte Carlo.

    Under the null -- N independent zero-edge strategies -- selection uses
    only in-sample data, and the test blocks are disjoint from the training
    blocks. So conditional on which strategy won in-sample, the out-of-sample
    scores of all N strategies are still exchangeable, and the winner's
    out-of-sample rank is EXACTLY uniform on {1..N}. Not asymptotically.

    One partition per matrix, so the samples are independent: ranks from
    different partitions of the same matrix share blocks and are not.
    Chi-square, because the target distribution is discrete and uniform.
    """
    n_strat, n_reps = 8, 600
    rng = np.random.default_rng(2024)
    counts = np.zeros(n_strat, dtype=int)
    for _ in range(n_reps):
        m = rng.standard_normal((240, n_strat))
        first = pbo(m, 4)["oos_ranks"][0]
        counts[int(first) - 1] += 1

    expected = np.full(n_strat, n_reps / n_strat)
    assert chisquare(counts, expected).pvalue > 0.01


@pytest.mark.slow
def test_null_pbo_is_one_half():
    """
    Follows from the uniform rank: PBO = P(rank <= N/2) = 1/2 exactly, for
    EVEN N.

    Even N matters. With odd N the rank (N+1)/2 is an atom of probability 1/N
    sitting exactly at lambda = 0, and since PBO counts lambda < 0 strictly,
    it converges to (N-1)/(2N) instead. Using odd N here and asserting 0.5
    would produce a test that fails for a correct implementation.
    """
    rng = np.random.default_rng(99)
    values = [pbo(rng.standard_normal((500, 10)), 8)["pbo"] for _ in range(40)]
    assert np.mean(values) == pytest.approx(0.5, abs=0.05)


def test_real_alpha_drives_pbo_to_zero():
    """
    One strategy with an edge large enough to dominate sampling noise wins
    in-sample nearly always and stays on top out-of-sample, so lambda > 0
    almost everywhere. This is the "winners keep winning" end of the scale.
    """
    rng = np.random.default_rng(5)
    m = rng.standard_normal((800, 8)) * 0.01
    m[:, 3] += 0.02          # per-period Sharpe ~2.0, unmistakable
    assert pbo(m, 10)["pbo"] < 0.05


def test_adversarial_matrix_gives_pbo_exactly_one():
    """
    A construction where the in-sample winner is ALWAYS the out-of-sample
    loser, so PBO = 1.0 exactly -- the anti-predictive extreme.

    How it is exact, since "PBO near 1" would be a weak assertion:

    Each block carries a large alternating +/- d pattern, which contributes
    zero to the mean of any whole block but sets the dispersion. On top of
    that sits a small per-block, per-strategy offset eps*c[j,b], where each
    strategy's c row sums to zero across blocks.

    Train and test are exact complements of equal size, so a zero row sum
    means the out-of-sample mean is the exact negation of the in-sample mean.
    Mean ordering is therefore exactly reversed between the two halves. The
    standard deviation is d for every strategy up to O(eps^2/d^2), so with
    eps/d = 1e-3 the Sharpe ordering follows the mean ordering strictly --
    the mean gaps are O(eps) while the std differences are O(eps^2/d).

    Hence the in-sample argmax is the out-of-sample argmin in EVERY
    partition: rank 1, w = 1/(N+1) < 1/2, lambda < 0.

    The offsets are drawn continuously rather than permuted from a discrete
    set. Permutations of a centred integer sequence collide -- different
    strategies land on the same subset sum, tie, and the O(eps^2) standard
    deviation term then breaks the tie arbitrarily, so the winner is only
    somewhere in the bottom group rather than exactly last. Continuous
    offsets tie with probability zero, which is asserted below.
    """
    n_blocks, n_strat, block_len = 8, 6, 40
    d, eps = 1.0, 1e-3

    # Zero-sum rows: the out-of-sample mean is then the exact negation of the
    # in-sample mean, because train and test are complements of equal size.
    g = np.random.default_rng(1).standard_normal((n_strat, n_blocks))
    c = g - g.mean(axis=1, keepdims=True)
    np.testing.assert_allclose(c.sum(axis=1), 0.0, atol=1e-15)

    alternating = np.where(np.arange(block_len) % 2 == 0, d, -d)
    m = np.empty((n_blocks * block_len, n_strat))
    for b in range(n_blocks):
        rows = slice(b * block_len, (b + 1) * block_len)
        m[rows] = alternating[:, None] + eps * c[:, b][None, :]

    result = pbo(m, n_blocks, store_scores=True)

    # Strategies must be strictly separated in-sample by a margin far above
    # the O(eps^2/d) standard-deviation correction (~1e-9), or "rank exactly
    # 1" would not be a well-posed claim.
    gaps = np.diff(np.sort(result["is_scores"], axis=1), axis=1)
    assert gaps.min() > 1e-7

    assert result["pbo"] == 1.0
    assert np.all(result["oos_ranks"] == 1)


# --------------------------------------------------------------------------
# Invariances and degenerate input
# --------------------------------------------------------------------------

def test_pbo_is_invariant_to_positive_rescaling():
    """
    PBO sees the metric only through ranks, so any strictly monotone
    transform applied uniformly across strategies cancels. Scaling every
    return by 252 -- the shape an accidental annualization would take --
    must not move it at all.
    """
    rng = np.random.default_rng(4)
    m = rng.standard_normal((400, 6)) * 0.01
    assert pbo(m, 8)["pbo"] == pbo(m * 252.0, 8)["pbo"]
    np.testing.assert_allclose(pbo(m, 8)["lambdas"], pbo(m * 252.0, 8)["lambdas"])


def test_constant_column_is_never_selected():
    """
    A zero-variance strategy has an undefined Sharpe. sharpe.py raises; here
    that would kill an otherwise valid run over N strategies, so it scores
    -inf instead: never the argmax, always last out-of-sample, and counted in
    the result dict so it stays visible rather than silent.
    """
    rng = np.random.default_rng(6)
    m = rng.standard_normal((300, 4)) * 0.01
    m[:, 2] = 0.0

    result = pbo(m, 6)
    assert result["n_degenerate"] > 0
    assert not np.any(result["n_star"] == 2)

    rows = np.arange(300)
    assert sharpe_direct(m, rows)[2] == DEGENERATE_SCORE


def test_result_dict_is_inspectable():
    """Detectors return dicts, not floats; the intermediates are the point."""
    rng = np.random.default_rng(8)
    m = rng.standard_normal((300, 5)) * 0.01
    result = pbo(m, 6)

    for key in (
        "pbo", "lambdas", "oos_ranks", "relative_ranks", "n_star", "blocks",
        "n_partitions", "n_blocks", "n_obs", "n_strategies", "n_degenerate",
        "method",
    ):
        assert key in result

    n_part = result["n_partitions"]
    assert result["lambdas"].shape == (n_part,)
    assert result["n_star"].shape == (n_part,)
    assert result["n_obs"] == 300 and result["n_strategies"] == 5
    assert "is_scores" not in result           # off by default: memory ceiling

    with_scores = pbo(m, 6, store_scores=True)
    assert with_scores["is_scores"].shape == (n_part, 5)

    # PBO is exactly the fraction of partitions with negative lambda.
    assert result["pbo"] == pytest.approx(np.mean(result["lambdas"] < 0))


def test_bad_input_rejected():
    rng = np.random.default_rng(10)
    with pytest.raises(ValueError, match="matrix"):
        pbo(rng.standard_normal(100), 4)
    with pytest.raises(ValueError, match="2 strategies"):
        pbo(rng.standard_normal((100, 1)), 4)
    with pytest.raises(ValueError, match="non-finite"):
        pbo(np.full((100, 3), np.nan), 4)
    with pytest.raises(ValueError, match="unknown method"):
        pbo(rng.standard_normal((100, 3)), 4, method="fast")
