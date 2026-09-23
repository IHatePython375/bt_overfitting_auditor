"""
Validation for the C++20 CSCV kernel.

Same rule as the rest of the suite: assertions are against something derived
independently of the thing under test -- math.comb, itertools.combinations, an
analytic PBO target, or the Python paths in pbo.py.

The differential tests here are stronger than the usual "agrees to within a
tolerance". The kernel is written to reproduce the numpy path BIT FOR BIT
(ascending block accumulation, -ffp-contract=off, true division rather than
reciprocal multiplication), so the ranks, relative ranks and winners are
asserted EQUAL, not close. That matters because a tolerance has to be loose
enough to pass and is therefore loose enough to hide a genuine reordering: a
rank that moves by one is a completely different result that a 1e-9 rtol on
lambda would happily accept.

Lambdas are the one exception, held to a tight tolerance rather than equality,
because std::log and np.log may differ by an ulp across libm implementations.
On the development machine they are in fact bitwise identical.

Everything here skips rather than fails when the kernel is not built. An
optional accelerator that breaks `pytest` for anyone who has not run
scripts/build_kernel.py is not optional.
"""

from itertools import combinations
from math import comb

import numpy as np
import pytest

from auditor.pbo import kernel_available, kernel_info, n_partitions, pbo

pytestmark = pytest.mark.skipif(
    not kernel_available(),
    reason="C++ kernel not built; run scripts/build_kernel.py",
)


# --------------------------------------------------------------------------
# The build itself
# --------------------------------------------------------------------------

def test_build_flags_preserve_the_bitwise_guarantee():
    """
    Every exact assertion below rests on two compiler flags, so they are
    asserted rather than assumed.

    With FP contraction on, `s2 - s1*s1/n` becomes an FMA and the low bit of
    every Sharpe changes; with -ffast-math the compiler may also assume no
    infinities, and -inf is how a zero-variance strategy is represented. Both
    would leave the suite passing under a loosened tolerance, which is exactly
    the kind of silent downgrade worth a test.
    """
    info = kernel_info()
    assert info is not None
    assert info["fp_contract_off"], "kernel built without -ffp-contract=off"
    assert not info["fast_math"], "kernel built with -ffast-math"


# --------------------------------------------------------------------------
# Combinatorics: the thread-offset machinery
# --------------------------------------------------------------------------

def test_binomial_matches_math_comb():
    """
    Checked over the whole supported range, not a few spot values.

    This is not ceremony. The first implementation used the multiplicative
    form, which is exactly divisible at every step but whose INTERMEDIATE
    product overflows uint64 long before the result does -- C(63,31) fits
    comfortably and was still returned wrong. Only a sweep finds that, because
    every small case is correct.
    """
    from auditor import _cscv

    for n in range(0, 65):
        for k in range(0, n + 1):
            assert _cscv.binomial(n, k) == comb(n, k), (n, k)

    # Out-of-range k is 0, not an error: unrank_combination relies on it at
    # the top of its search.
    assert _cscv.binomial(5, 6) == 0
    assert _cscv.binomial(5, -1) == 0


def test_unrank_matches_itertools_exhaustively():
    """
    Lexicographic unranking against itertools.combinations, every rank.

    This is the correctness condition for threading. Each chunk of partitions
    jumps straight to its first combination instead of walking there, so an
    off-by-one in the unranking does not crash -- it silently evaluates the
    wrong partition at that index, and PBO shifts by a hair in a way that only
    shows up at particular chunk boundaries and thread counts.
    """
    from auditor import _cscv

    for n, k in [(6, 3), (8, 4), (10, 5), (12, 6), (16, 8)]:
        expected = list(combinations(range(n), k))
        assert len(expected) == comb(n, k)
        for r, want in enumerate(expected):
            assert tuple(_cscv.unrank(r, n, k)) == want, (n, k, r)


def test_unrank_matches_itertools_at_realistic_size():
    """
    C(20,10) = 184,756: large enough that chunks actually start mid-stream,
    still small enough for itertools to serve as the oracle.
    """
    from auditor import _cscv

    n, k = 20, 10
    expected = list(combinations(range(n), k))
    rng = np.random.default_rng(0)
    sample = list(rng.integers(0, len(expected), size=400))
    sample += [0, 1, len(expected) - 2, len(expected) - 1]
    for r in sample:
        assert tuple(_cscv.unrank(int(r), n, k)) == expected[int(r)], r


def test_unrank_rejects_out_of_range():
    from auditor import _cscv

    with pytest.raises(ValueError):
        _cscv.unrank(comb(8, 4), 8, 4)      # one past the last
    with pytest.raises(ValueError):
        _cscv.unrank(0, 4, 9)               # k > n


def test_partition_count_agrees_with_python():
    from auditor import _cscv

    for s in (4, 8, 16, 24, 32, 64):
        assert _cscv.partition_count(s) == n_partitions(s) == comb(s, s // 2)


# --------------------------------------------------------------------------
# Differential: C++ against both Python paths
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "n_blocks", [10, 12, pytest.param(16, marks=pytest.mark.slow)]
)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cpp_matches_sufstats(n_blocks, seed):
    """
    The port against the thing it was ported from, asserted exactly.

    Asserts on the per-partition vectors rather than only on PBO. PBO is a
    mean of indicators, so it can agree by luck while individual partitions
    disagree in compensating directions.
    """
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((700, 8)) * 0.01 + 0.0003

    py = pbo(m, n_blocks, method="sufstats")
    cpp = pbo(m, n_blocks, method="cpp")

    assert cpp["method"] == "cpp"
    assert py["n_partitions"] == cpp["n_partitions"]
    assert np.array_equal(py["n_star"], cpp["n_star"])
    assert np.array_equal(py["oos_ranks"], cpp["oos_ranks"])
    assert np.array_equal(py["relative_ranks"], cpp["relative_ranks"])
    assert py["n_degenerate"] == cpp["n_degenerate"]
    assert py["pbo"] == cpp["pbo"]
    np.testing.assert_allclose(py["lambdas"], cpp["lambdas"], rtol=1e-15, atol=0)


@pytest.mark.parametrize("n_blocks", [8, 10])
def test_cpp_matches_the_naive_oracle(n_blocks):
    """
    Against the path that does NOT share the block decomposition.

    The sufstats comparison above cannot catch a shared design error: the
    kernel mirrors that path deliberately, so if the block algebra itself were
    wrong the two would agree on the same wrong answer. The naive path slices
    raw rows and knows nothing about blocks, so it is the oracle that does not
    share the assumption.
    """
    rng = np.random.default_rng(4)
    m = rng.standard_normal((500, 6)) * 0.01 + 0.0002

    slow = pbo(m, n_blocks, method="naive")
    cpp = pbo(m, n_blocks, method="cpp")

    assert np.array_equal(slow["n_star"], cpp["n_star"])
    assert np.array_equal(slow["oos_ranks"], cpp["oos_ranks"])
    assert slow["pbo"] == cpp["pbo"]
    np.testing.assert_allclose(slow["lambdas"], cpp["lambdas"], rtol=1e-9, atol=1e-12)


# --------------------------------------------------------------------------
# Threading
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n_threads", [1, 2, 3, 5, 8, 16, 0])
def test_thread_count_is_not_an_answer_knob(n_threads):
    """
    Identical output for any thread count, including the awkward ones.

    Work is claimed dynamically from a shared counter, so WHICH thread runs a
    given chunk varies between runs; what must not vary is what lands at
    partition index p. Odd thread counts and counts above the chunk count are
    included because they are where a scheme that divides the range by hand
    tends to drop or double-count the tail.

    C(12,6) = 924 with a 64-partition chunk leaves a final chunk of 28, so the
    ragged-tail case is exercised here rather than assumed.
    """
    rng = np.random.default_rng(5)
    m = rng.standard_normal((600, 7)) * 0.01 + 0.0001

    ref = pbo(m, 12, method="cpp", n_threads=1)
    got = pbo(m, 12, method="cpp", n_threads=n_threads)

    assert ref["n_partitions"] == 924
    assert np.array_equal(ref["lambdas"], got["lambdas"])
    assert np.array_equal(ref["n_star"], got["n_star"])
    assert np.array_equal(ref["oos_ranks"], got["oos_ranks"])
    assert ref["pbo"] == got["pbo"]
    assert ref["n_degenerate"] == got["n_degenerate"]


def test_every_partition_slot_is_written():
    """
    No chunk is skipped and none is claimed twice.

    An unclaimed chunk would leave its slice of the freshly allocated output
    holding uninitialised memory, which is not reliably NaN and can easily
    look like a plausible lambda. Comparing element-wise against the Python
    path over a size with a ragged final chunk is what actually pins this;
    the finiteness check just names the failure mode.
    """
    rng = np.random.default_rng(6)
    m = rng.standard_normal((400, 5)) * 0.01

    cpp = pbo(m, 12, method="cpp", n_threads=8)
    py = pbo(m, 12, method="sufstats")

    assert np.isfinite(cpp["lambdas"]).all()
    assert np.array_equal(cpp["lambdas"], py["lambdas"])
    assert set(np.unique(cpp["n_star"])) <= set(range(5))


# --------------------------------------------------------------------------
# Summary-only mode
# --------------------------------------------------------------------------

def test_summary_only_agrees_with_the_full_run():
    """
    store_lambdas=False must change what is stored, never what is computed.

    It exists because the detail arrays cost 28 bytes per partition, which is
    17 GB at C(32,16) -- the regime the kernel was written for. PBO comes from
    the kernel's streaming counter in both modes, so the two cannot drift.
    """
    rng = np.random.default_rng(7)
    m = rng.standard_normal((500, 6)) * 0.01 + 0.0002

    full = pbo(m, 10, method="cpp", store_lambdas=True)
    lean = pbo(m, 10, method="cpp", store_lambdas=False)

    assert lean["pbo"] == full["pbo"]
    assert lean["n_partitions"] == full["n_partitions"]
    assert lean["n_degenerate"] == full["n_degenerate"]

    # Absent, not empty: a caller can test for the key rather than discover an
    # array of zeros.
    for key in ("lambdas", "oos_ranks", "relative_ranks", "n_star"):
        assert key in full
        assert key not in lean

    # And the summary keys every caller needs are still there.
    for key in ("pbo", "n_partitions", "n_blocks", "n_obs", "n_strategies",
                "n_degenerate", "method", "blocks"):
        assert key in lean


# --------------------------------------------------------------------------
# The awkward inputs, against the Python paths
# --------------------------------------------------------------------------

def test_degenerate_columns_match_python():
    """
    A zero-variance column scores -inf in both paths: never the in-sample
    argmax, always last out-of-sample, and counted.

    -inf is load-bearing rather than incidental, which is why -ffast-math is
    asserted off above. The counts must agree exactly, since the kernel
    increments its own counter inside the scoring loop rather than scanning
    for -inf afterwards.
    """
    rng = np.random.default_rng(8)
    m = rng.standard_normal((300, 5)) * 0.01
    m[:, 2] = 0.0

    py = pbo(m, 8, method="sufstats")
    cpp = pbo(m, 8, method="cpp")

    assert cpp["n_degenerate"] > 0
    assert cpp["n_degenerate"] == py["n_degenerate"]
    assert not np.any(cpp["n_star"] == 2)
    assert np.array_equal(py["oos_ranks"], cpp["oos_ranks"])
    assert py["pbo"] == cpp["pbo"]


def test_tied_columns_match_python():
    """
    Exact ties are common on real sweeps and the two paths must break them
    identically.

    Two sources here, and they are different mechanisms: a duplicated column
    (identical arithmetic, identical bits) and a leveraged copy (Sharpe is
    scale invariant, so a 2.5x position size gives the SAME Sharpe from
    different bits). The kernel counts ties with ==, which matches scipy's
    rankdata only if both paths produce identical bits for tied strategies --
    that is the assumption under test.
    """
    rng = np.random.default_rng(9)
    m = rng.standard_normal((400, 6)) * 0.01
    m[:, 4] = m[:, 3]             # exact duplicate
    m[:, 5] = m[:, 0] * 2.5       # same Sharpe, different bits

    py = pbo(m, 10, method="sufstats")
    cpp = pbo(m, 10, method="cpp")

    # The ties are real, or this test proves nothing.
    assert np.any(cpp["oos_ranks"] % 1 != 0), "expected fractional average ranks"
    assert np.array_equal(py["n_star"], cpp["n_star"])
    assert np.array_equal(py["oos_ranks"], cpp["oos_ranks"])
    assert py["pbo"] == cpp["pbo"]


def test_two_strategies_is_enough():
    """N = 2 is the minimum the ranking is defined for; the edge should run."""
    rng = np.random.default_rng(12)
    m = rng.standard_normal((200, 2)) * 0.01
    cpp = pbo(m, 6, method="cpp")
    py = pbo(m, 6, method="sufstats")
    assert cpp["pbo"] == py["pbo"]
    assert np.array_equal(cpp["oos_ranks"], py["oos_ranks"])


# --------------------------------------------------------------------------
# Analytic targets, straight through the kernel
# --------------------------------------------------------------------------

def test_adversarial_matrix_gives_pbo_exactly_one():
    """
    The anti-predictive extreme, hitting the kernel rather than the Python.

    Construction and its derivation are in test_pbo.py; in short, each
    strategy gets per-block offsets whose row sums are zero, so the
    out-of-sample mean is the exact negation of the in-sample mean and the
    in-sample winner is the out-of-sample loser in EVERY partition. The
    target is 1.0 exactly and rank 1 everywhere -- an analytic value, not a
    number copied from a previous run, so it constrains the kernel
    independently of the Python path.
    """
    n_blocks, n_strat, block_len = 8, 6, 40
    d, eps = 1.0, 1e-3

    g = np.random.default_rng(1).standard_normal((n_strat, n_blocks))
    c = g - g.mean(axis=1, keepdims=True)

    alternating = np.where(np.arange(block_len) % 2 == 0, d, -d)
    m = np.empty((n_blocks * block_len, n_strat))
    for b in range(n_blocks):
        rows = slice(b * block_len, (b + 1) * block_len)
        m[rows] = alternating[:, None] + eps * c[:, b][None, :]

    result = pbo(m, n_blocks, method="cpp")
    assert result["pbo"] == 1.0
    assert np.all(result["oos_ranks"] == 1)


def test_real_alpha_drives_pbo_to_zero():
    """The other end: an edge too large to be luck keeps winning."""
    rng = np.random.default_rng(5)
    m = rng.standard_normal((800, 8)) * 0.01
    m[:, 3] += 0.02
    assert pbo(m, 10, method="cpp")["pbo"] < 0.05


@pytest.mark.slow
def test_null_pbo_is_one_half():
    """
    Pure noise: being best in-sample predicts nothing, so the winner lands at
    a uniformly random out-of-sample rank and PBO converges to 0.5.

    Averaged over independent matrices because a single one has real sampling
    spread -- the target is a property of the null, not of any one draw.
    """
    rng = np.random.default_rng(100)
    values = [
        pbo(rng.standard_normal((600, 10)) * 0.01, 10, method="cpp")["pbo"]
        for _ in range(40)
    ]
    assert np.mean(values) == pytest.approx(0.5, abs=0.05)


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------

def test_auto_prefers_the_kernel():
    rng = np.random.default_rng(13)
    m = rng.standard_normal((300, 4)) * 0.01
    assert pbo(m, 8)["method"] == "cpp"


def test_auto_falls_back_when_scores_are_requested():
    """
    store_scores cannot be served by the kernel -- refusing to materialise
    n_partitions x N score matrices is the point of it -- so "auto" routes to
    the Python path instead. That is dispatch, not a silent downgrade: the
    result says which path ran.
    """
    rng = np.random.default_rng(14)
    m = rng.standard_normal((300, 4)) * 0.01
    result = pbo(m, 8, store_scores=True)
    assert result["method"] == "sufstats"
    assert result["is_scores"].shape == (result["n_partitions"], 4)


def test_explicit_cpp_with_store_scores_is_an_error():
    """Asking the kernel directly for something it cannot do must say so."""
    rng = np.random.default_rng(15)
    m = rng.standard_normal((300, 4)) * 0.01
    with pytest.raises(ValueError, match="store_scores"):
        pbo(m, 8, method="cpp", store_scores=True)


def test_kernel_info_reports_the_build():
    info = kernel_info()
    assert set(info) == {"version", "compiler", "fp_contract_off", "fast_math"}
    assert isinstance(info["compiler"], str) and info["compiler"]
