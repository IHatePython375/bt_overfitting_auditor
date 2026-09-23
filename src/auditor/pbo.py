"""
Probability of Backtest Overfitting via combinatorially symmetric
cross-validation (CSCV).

Reference: Bailey, Borwein, Lopez de Prado & Zhu (2017), "The Probability
of Backtest Overfitting."

The question CSCV answers is not "is this strategy good" but "does being
best in-sample predict anything out-of-sample." It splits the observation
axis into S contiguous blocks, enumerates every balanced train/test
partition, and asks where the in-sample winner lands in the out-of-sample
ranking. If selection is pure noise the winner lands at a uniformly random
rank, and PBO converges to 0.5.

UNITS
-----
Every Sharpe here is PER-PERIOD, as in sharpe.py. Nothing in this module
annualizes, and it would not matter if it did: PBO depends on the metric
only through ranks, so any strictly monotone transform applied uniformly
across strategies -- annualization included -- cancels out. The units
footgun that is lethal in DSR is inert here. Do not "fix" it.

THREE IMPLEMENTATIONS
---------------------
_pbo_naive      slices the actual return rows and computes Sharpe from
                scratch. Obviously correct by inspection; unusable past
                S = 16.
_pbo_sufstats   precomputes per-block (n, sum, sumsq) per strategy, so each
                partition is a reduction over the chosen blocks. This is the
                shape the C++ kernel mirrors.
_pbo_cpp        the same reduction in C++20 (src/kernel/cscv.hpp), threaded
                over partitions. Optional: absent unless
                scripts/build_kernel.py has been run, and every path here
                falls back rather than failing.

They exist side by side on purpose. Differential-testing Python against C++
catches implementation divergence, but it cannot catch a shared design
error: if both are structured around the same block decomposition and that
reasoning is wrong, both return the same wrong answer and every test passes.
The naive path is the oracle that does not share the assumption.

The kernel receives (counts, sums, sumsq) and nothing else -- make_blocks and
block_stats stay here. So the C++ cannot disagree with the Python about where
the block boundaries are, and a differential failure localises to the
combinatorial reduction, which is the only thing that was ported.
"""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
from scipy.stats import rankdata

try:
    from . import _cscv  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - depends on whether the build was run
    _cscv = None

# Score for strategies with no variance over the selected rows. -inf rather
# than nan so that argmax skips them instead of propagating, and so they
# rank last out-of-sample instead of poisoning the whole rank vector.
DEGENERATE_SCORE = -np.inf


def kernel_available() -> bool:
    """Whether the compiled C++ kernel is importable."""
    return _cscv is not None


def kernel_info() -> dict | None:
    """
    Build facts for the compiled kernel, or None if it is not built.

    ``fp_contract_off`` is not trivia. The kernel agrees with the Python path
    bit for bit only while FMA contraction is disabled; with it on, the
    differential test has to fall back to a tolerance, and a tolerance loose
    enough to pass is loose enough to hide a real bug. Reporting it makes a
    mis-built kernel visible instead of quietly weakening the guarantee.
    """
    if _cscv is None:
        return None
    return {
        "version": _cscv.__version__,
        "compiler": _cscv.compiler,
        "fp_contract_off": _cscv.fp_contract_off,
        "fast_math": _cscv.fast_math,
    }


# --------------------------------------------------------------------------
# Block construction
# --------------------------------------------------------------------------

def make_blocks(n_obs: int, n_blocks: int) -> list[tuple[int, int]]:
    """
    Split [0, n_obs) into `n_blocks` CONTIGUOUS half-open ranges.

    Contiguous, not shuffled: shuffling would destroy the serial correlation
    inside each block, which is exactly the structure week 4 needs in order
    to measure how badly it biases PBO.

    When n_obs is not divisible by n_blocks the remainder goes to the
    earliest blocks (numpy.array_split semantics) rather than being
    truncated. Truncating would silently discard up to n_blocks-1
    observations -- at S=32 on daily data that is a month and a half. Uneven
    blocks cost the reduction nothing, because the sufficient statistics
    carry their own counts.
    """
    if n_blocks < 2:
        raise ValueError("need at least 2 blocks")
    if n_blocks % 2 != 0:
        raise ValueError(f"n_blocks must be even, got {n_blocks}")
    if n_obs < n_blocks:
        raise ValueError(
            f"need at least one observation per block ({n_obs} < {n_blocks})"
        )

    base, extra = divmod(n_obs, n_blocks)
    bounds = []
    start = 0
    for b in range(n_blocks):
        stop = start + base + (1 if b < extra else 0)
        bounds.append((start, stop))
        start = stop
    return bounds


def _validate(returns: np.ndarray) -> np.ndarray:
    m = np.asarray(returns, dtype=float)
    if m.ndim != 2:
        raise ValueError(f"expected a (T, N) matrix, got shape {m.shape}")
    if m.shape[1] < 2:
        raise ValueError("need at least 2 strategies to rank")
    if not np.isfinite(m).all():
        raise ValueError("returns contain non-finite values")
    return m


# --------------------------------------------------------------------------
# The two scoring paths
# --------------------------------------------------------------------------

def sharpe_direct(returns: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """
    Per-period Sharpe of every strategy over `rows`, from the raw returns.
    Matches sharpe.py's sharpe_ratio(): mean / std(ddof=1).

    This is the oracle. It does not know that blocks exist.
    """
    x = returns[rows]
    sd = x.std(axis=0, ddof=1)
    out = np.full(returns.shape[1], DEGENERATE_SCORE)
    ok = sd > 0
    out[ok] = x.mean(axis=0)[ok] / sd[ok]
    return out


def block_stats(
    returns: np.ndarray, blocks: list[tuple[int, int]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-block sufficient statistics: counts (S,), sums (S, N), sumsq (S, N).

    Sufficient in the precise statistical sense: any subset of blocks can be
    scored from these alone, without revisiting the raw returns. That is what
    turns each partition from an O(T*N) recomputation into an O(S*N)
    reduction, and the reduction is a sum over a subset of rows -- SIMD
    friendly, and embarrassingly parallel across partitions.
    """
    n_blocks = len(blocks)
    n_strat = returns.shape[1]
    counts = np.empty(n_blocks, dtype=np.int64)
    sums = np.empty((n_blocks, n_strat))
    sumsq = np.empty((n_blocks, n_strat))
    for b, (start, stop) in enumerate(blocks):
        chunk = returns[start:stop]
        counts[b] = stop - start
        sums[b] = chunk.sum(axis=0)
        sumsq[b] = (chunk ** 2).sum(axis=0)
    return counts, sums, sumsq


def sharpe_from_stats(
    counts: np.ndarray,
    sums: np.ndarray,
    sumsq: np.ndarray,
    block_idx: np.ndarray,
) -> np.ndarray:
    """
    Per-period Sharpe of every strategy over the union of `block_idx`,
    assembled from sufficient statistics.

    Pools the moments and THEN forms the Sharpe. It does not average the
    per-block Sharpes -- those are different numbers, and the second one is
    wrong. That is the easiest way to get the kernel subtly wrong.

    Variance uses the textbook shortcut (sumsq - sum^2/n) / (n-1), which is
    what makes the reduction cheap and is also its weak point: catastrophic
    cancellation when mean^2 >> variance. For returns, whose mean is ~0, the
    condition number is benign. Feed it price levels and it is garbage.
    Welford per block plus Chan's parallel merge is the fix if that ever
    bites; the block decomposition already supports it.
    """
    n = counts[block_idx].sum()
    s1 = sums[block_idx].sum(axis=0)
    s2 = sumsq[block_idx].sum(axis=0)

    ss = s2 - s1 * s1 / n           # centred sum of squares
    var = ss / (n - 1)              # ddof=1, matching sharpe.py
    out = np.full(sums.shape[1], DEGENERATE_SCORE)
    ok = var > 0                    # cancellation can push this slightly negative
    out[ok] = (s1[ok] / n) / np.sqrt(var[ok])
    return out


# --------------------------------------------------------------------------
# Ranking: shared by both paths, deliberately
# --------------------------------------------------------------------------

def partition_lambda(
    is_scores: np.ndarray, oos_scores: np.ndarray
) -> tuple[int, float, float, float]:
    """
    Return (n_star, oos_rank, w, lambda) for one partition.

    n_star is the in-sample winner. Its out-of-sample rank is taken among all
    N strategies, ascending in performance and 1-indexed, so rank N means the
    winner also finished best out-of-sample.

    Relative rank is rank/(N+1), NOT rank/N. With rank/N a winner that also
    finishes first out-of-sample gives w = 1 and lambda = +inf. PBO itself
    survives that -- +inf is still "not below zero" -- so the headline number
    looks perfectly healthy while every distributional statistic downstream
    is poisoned. The (N+1) denominator is the Weibull plotting position; its
    job is keeping w strictly inside (0, 1) at BOTH ends.

    Ties get average ranks. np.argsort would break them by column index,
    biasing toward whichever variant sits earlier in the matrix -- and since
    parameter sweeps are usually generated in sorted order, "earlier"
    correlates with a systematically smaller parameter value, so that is a
    real bias rather than a hypothetical one. rankdata's 'min' or 'max' would
    instead push PBO in a fixed direction.

    Shared by both paths on purpose. The differential test exists to compare
    the two block decompositions, so holding the ranking fixed isolates what
    is actually under test; the ranking itself is pinned separately against
    analytic targets (null -> 0.5, adversarial -> 1.0).
    """
    n_strat = is_scores.size
    n_star = int(np.argmax(is_scores))
    rank = float(rankdata(oos_scores, method="average")[n_star])
    w = rank / (n_strat + 1.0)
    return n_star, rank, w, float(np.log(w / (1.0 - w)))


def _assemble(
    pbo_value: float,
    n_part: int,
    lambdas: np.ndarray | None,
    ranks: np.ndarray | None,
    rel_ranks: np.ndarray | None,
    winners: np.ndarray | None,
    blocks: list[tuple[int, int]],
    shape: tuple[int, int],
    n_degenerate: int,
    method: str,
    is_scores: np.ndarray | None = None,
    oos_scores: np.ndarray | None = None,
) -> dict:
    """
    Common result dict. Detectors return dicts, not floats.

    `pbo_value` and `n_part` are passed rather than derived from `lambdas`
    because the C++ path can run in summary-only mode, where the per-partition
    arrays are never materialised -- at C(32,16) they would be about 17 GB.
    The summary keys are therefore always present and the detail keys are not,
    which is the honest shape: a caller can test for "lambdas" rather than
    finding an empty array.
    """
    result = {
        "pbo": pbo_value,
        "blocks": blocks,
        "n_partitions": n_part,
        "n_blocks": len(blocks),
        "n_obs": shape[0],
        "n_strategies": shape[1],
        "n_degenerate": n_degenerate,
        "method": method,
    }
    if lambdas is not None:
        result["lambdas"] = lambdas
        result["oos_ranks"] = ranks
        result["relative_ranks"] = rel_ranks
        result["n_star"] = winners
    if is_scores is not None:
        result["is_scores"] = is_scores
        result["oos_scores"] = oos_scores
    return result


# --------------------------------------------------------------------------
# Path 1: the oracle
# --------------------------------------------------------------------------

def _pbo_naive(returns: np.ndarray, n_blocks: int, store_scores: bool) -> dict:
    """
    CSCV computed straight from the return rows, with no block algebra.

    Correct by inspection and far too slow to use in anger -- it re-reads
    O(T*N) floats per partition, so C(16,8) = 12,870 partitions already means
    walking the whole matrix 12,870 times. Its job is to be the thing the
    fast path gets checked against.
    """
    blocks = make_blocks(returns.shape[0], n_blocks)
    block_rows = [np.arange(start, stop) for start, stop in blocks]
    all_blocks = set(range(n_blocks))

    lam, rks, wvs, wins = [], [], [], []
    is_all, oos_all = [], []
    n_degenerate = 0

    for train in combinations(range(n_blocks), n_blocks // 2):
        test = sorted(all_blocks - set(train))
        is_rows = np.concatenate([block_rows[b] for b in train])
        oos_rows = np.concatenate([block_rows[b] for b in test])

        is_scores = sharpe_direct(returns, is_rows)
        oos_scores = sharpe_direct(returns, oos_rows)
        n_degenerate += int(
            np.isneginf(is_scores).sum() + np.isneginf(oos_scores).sum()
        )

        n_star, rank, w, lmbda = partition_lambda(is_scores, oos_scores)
        lam.append(lmbda)
        rks.append(rank)
        wvs.append(w)
        wins.append(n_star)
        if store_scores:
            is_all.append(is_scores)
            oos_all.append(oos_scores)

    lambdas = np.asarray(lam)
    return _assemble(
        float(np.mean(lambdas < 0.0)),
        int(lambdas.size),
        lambdas,
        np.asarray(rks),
        np.asarray(wvs),
        np.asarray(wins, dtype=int),
        blocks,
        returns.shape,
        n_degenerate,
        "naive",
        np.asarray(is_all) if store_scores else None,
        np.asarray(oos_all) if store_scores else None,
    )


# --------------------------------------------------------------------------
# Path 2: the week-2 porting target
# --------------------------------------------------------------------------

def _pbo_sufstats(returns: np.ndarray, n_blocks: int, store_scores: bool) -> dict:
    """
    CSCV via per-block sufficient statistics.

    Structurally what the C++ kernel will do: build the statistics once, then
    loop partitions doing a reduction over the chosen block rows. Vectorised
    across strategies, looping over partitions -- the same nesting the kernel
    uses, so the differential test compares like with like.
    """
    blocks = make_blocks(returns.shape[0], n_blocks)
    counts, sums, sumsq = block_stats(returns, blocks)
    all_blocks = set(range(n_blocks))
    half = n_blocks // 2

    lam, rks, wvs, wins = [], [], [], []
    is_all, oos_all = [], []
    n_degenerate = 0

    for train in combinations(range(n_blocks), half):
        train_idx = np.fromiter(train, dtype=np.int64, count=half)
        test_idx = np.fromiter(
            sorted(all_blocks - set(train)), dtype=np.int64, count=half
        )

        is_scores = sharpe_from_stats(counts, sums, sumsq, train_idx)
        oos_scores = sharpe_from_stats(counts, sums, sumsq, test_idx)
        n_degenerate += int(
            np.isneginf(is_scores).sum() + np.isneginf(oos_scores).sum()
        )

        n_star, rank, w, lmbda = partition_lambda(is_scores, oos_scores)
        lam.append(lmbda)
        rks.append(rank)
        wvs.append(w)
        wins.append(n_star)
        if store_scores:
            is_all.append(is_scores)
            oos_all.append(oos_scores)

    lambdas = np.asarray(lam)
    return _assemble(
        float(np.mean(lambdas < 0.0)),
        int(lambdas.size),
        lambdas,
        np.asarray(rks),
        np.asarray(wvs),
        np.asarray(wins, dtype=int),
        blocks,
        returns.shape,
        n_degenerate,
        "sufstats",
        np.asarray(is_all) if store_scores else None,
        np.asarray(oos_all) if store_scores else None,
    )


# --------------------------------------------------------------------------
# Path 3: the C++ kernel
# --------------------------------------------------------------------------

def _pbo_cpp(
    returns: np.ndarray,
    n_blocks: int,
    store_lambdas: bool,
    n_threads: int,
) -> dict:
    """
    CSCV with the partition loop in C++20, threaded over partitions.

    Everything before the loop stays in numpy: the blocks and the sufficient
    statistics are built here and handed over. What crosses the boundary is
    three small arrays -- (S,) and two (S, N) -- regardless of T, so the
    marshalling cost does not scale with the data.

    The kernel splits partitions into contiguous index ranges, one per thread,
    and each thread writes only into its own slice. The result is therefore
    identical for any thread count, which is what makes `n_threads` a
    performance knob and not a source of answers that drift with the machine.
    """
    if _cscv is None:  # pragma: no cover - guarded by the caller
        raise RuntimeError(
            "the C++ kernel is not built; run scripts/build_kernel.py "
            "or use method='sufstats'"
        )

    blocks = make_blocks(returns.shape[0], n_blocks)
    counts, sums, sumsq = block_stats(returns, blocks)

    raw = _cscv.run(counts, sums, sumsq, store_lambdas, n_threads)

    # PBO comes from the kernel's own streaming count, not from re-reducing
    # the lambda array here. In summary-only mode there is no array to reduce,
    # and having the headline number arrive by one route in both modes means
    # the two modes cannot disagree.
    return _assemble(
        float(raw["n_negative"]) / float(raw["n_partitions"]),
        int(raw["n_partitions"]),
        raw["lambdas"] if store_lambdas else None,
        raw["oos_ranks"] if store_lambdas else None,
        raw["relative_ranks"] if store_lambdas else None,
        raw["n_star"].astype(int) if store_lambdas else None,
        blocks,
        returns.shape,
        int(raw["n_degenerate"]),
        "cpp",
    )


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def pbo(
    returns: np.ndarray,
    n_blocks: int = 16,
    method: str = "auto",
    store_scores: bool = False,
    store_lambdas: bool = True,
    n_threads: int = 0,
) -> dict:
    """
    Probability of Backtest Overfitting for a (T, N) matrix of per-period
    strategy returns.

    Parameters
    ----------
    returns : (T observations, N strategy variants)
    n_blocks : S, must be even. C(S, S/2) partitions get enumerated, so
        C(16,8) = 12,870, C(24,12) = 2,704,156, C(32,16) = 601,080,390.
        The third row is why the C++ kernel exists.
    method : "auto", "cpp", "sufstats", or "naive".

        "auto" picks "cpp" when the kernel is built and "sufstats"
        otherwise, except that `store_scores=True` always selects
        "sufstats" -- the kernel structurally cannot produce per-partition
        score matrices, since refusing to materialise them is the point.
        That is dispatch rather than a silent downgrade: the result's
        "method" key always says which path actually ran.
    store_scores : keep the full per-partition score matrices. Off by default
        because they are n_partitions x N: at C(24,12) with N=500 that is
        10 GB. Not a tuning knob, a hard ceiling -- and the second
        independent reason the kernel must stream its reduction rather than
        materialise anything. Python paths only.
    store_lambdas : keep the per-partition lambda, rank and winner vectors.
        C++ path only; the Python paths accumulate them either way. Turning
        it off is what makes very large S feasible at all: the four detail
        arrays cost 28 bytes per partition, which is 17 GB at C(32,16), while
        the summary keys ("pbo", "n_degenerate") cost nothing and stream.
    n_threads : C++ path only. 0 means every core. The partition range is
        split contiguously and each thread writes only its own slice, so the
        result does not depend on this value -- it is a speed knob, never an
        answer knob.

    Reading the result: PBO near 0 means winners keep winning. Near 0.5 means
    selection is a coin flip. Above 0.5 means selection is actively
    anti-predictive -- picking the in-sample best is worse than picking at
    random.

    Note that all C(S, S/2) subsets are enumerated, so each complementary
    pair appears twice, once in each train/test direction. That is correct
    per the paper, and it is what makes the lambda distribution symmetric
    under the null. C(S, S/2) / 2 looks like the obvious de-duplication and
    is not.
    """
    m = _validate(returns)

    if method == "auto":
        method = "cpp" if (_cscv is not None and not store_scores) else "sufstats"

    if method == "cpp":
        if _cscv is None:
            raise RuntimeError(
                "method='cpp' requested but the kernel is not built; run "
                "scripts/build_kernel.py, or use method='sufstats'"
            )
        if store_scores:
            raise ValueError(
                "store_scores is not available on the C++ path: the kernel "
                "streams its reduction and never materialises the "
                "n_partitions x N score matrices. Use method='sufstats'."
            )
        return _pbo_cpp(m, n_blocks, store_lambdas, n_threads)
    if method == "sufstats":
        return _pbo_sufstats(m, n_blocks, store_scores)
    if method == "naive":
        return _pbo_naive(m, n_blocks, store_scores)
    raise ValueError(
        f"unknown method {method!r}; expected 'auto', 'cpp', 'sufstats' "
        "or 'naive'"
    )


def n_partitions(n_blocks: int) -> int:
    """C(S, S/2) -- the number of partitions CSCV will enumerate."""
    if n_blocks % 2 != 0:
        raise ValueError(f"n_blocks must be even, got {n_blocks}")
    return comb(n_blocks, n_blocks // 2)
