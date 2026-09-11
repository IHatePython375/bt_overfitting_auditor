"""
White's Reality Check for data snooping (White 2000).

The question is different from the other two detectors. DSR asks whether one
strategy's Sharpe survives the trial count. PBO asks whether in-sample
ranking predicts anything out-of-sample. Reality Check asks: given N
strategies and a benchmark, is the BEST one's outperformance significant once
you account for having searched over all N?

WHAT f_k IS, AND WHY IT IS NOT A SHARPE RATIO
---------------------------------------------
f_k,t is strategy k's excess return over the benchmark in period t, and the
statistic is built from its MEAN. That is forced: the bootstrap resamples
periods and averages them, so the performance measure has to be the mean of a
per-period quantity. A Sharpe ratio is a ratio of moments, not a mean, and
does not decompose that way.

The consequence is worth stating plainly rather than hiding: this detector
ranks strategies by mean excess return, while DSR and PBO rank them by
Sharpe. Those are different functionals -- one ignores volatility entirely.
So ROC curves from the week-3 calibration study are NOT strictly
commensurable across the three detectors, and any comparison has to say so.
Fidelity to the published method is worth more than internal tidiness here,
and "the three detectors do not measure the same thing" is itself a result.

The alternatives were: studentize f_k (Hansen's SPA direction, deliberately
out of scope), or bootstrap the Sharpe directly by recomputing it per
replicate (defensible, but then it is not White's Reality Check and should
not be called one).

THE CENTERING TERM
------------------
    V  = max_k sqrt(T) * fbar_k                 (observed, UNcentered)
    V* = max_k sqrt(T) * (fbar*_k - fbar_k)     (bootstrap, centered on the
                                                 SAMPLE mean)

The bootstrap resamples from the empirical distribution, whose mean is
fbar_k, not zero. Subtracting fbar_k strips out the location while leaving
the sampling variability intact -- serial dependence via the blocks,
cross-sectional dependence via the shared indices. It imposes the LEAST
FAVOURABLE configuration of the composite null max_k E[f_k] <= 0, namely
E[f_k] = 0 for every k.

The three ways to get it wrong fail in different directions, which is why
only some of them are caught by casual testing:

  - centering on each replicate's own mean (fbar*_k - fbar*_k) collapses V*
    to 0, so p = 0 for any positive statistic: ALWAYS REJECTS.
  - omitting it leaves V* centred on V itself, so p ~ 0.5 whatever the data:
    no power, and under the null it looks accidentally fine. This is the one
    the null-uniformity test exists to catch.
  - flipping the sign doubles the drift and p -> 1: NEVER rejects.

Two details: the max is taken AFTER centering, per replicate; and V uses the
uncentered mean. Centering both sides gives V = 0 and p ~ 1.

Reference: White, H. (2000), "A Reality Check for Data Snooping,"
Econometrica 68(5), 1097-1126.
"""

from __future__ import annotations

import numpy as np

from .bootstrap import rule_of_thumb_block_length, stationary_bootstrap_indices

__all__ = ["reality_check"]


def _bootstrap_means(f: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    Mean of every strategy over every bootstrap replicate, as (n_boot, N).

    Counts each row's multiplicity and contracts with a matmul rather than
    gathering f[indices]. The gather would materialise an (n_boot, T, N)
    array -- 40 GB at n_boot = 1000, T = 1000, N = 50 -- so this is not a
    micro-optimisation, it is what makes the null-calibration test runnable.
    Cost drops to one (n_boot, T) x (T, N) BLAS call.
    """
    n_boot, n_obs = indices.shape
    offsets = (np.arange(n_boot, dtype=np.int64) * n_obs)[:, None]
    flat = indices.astype(np.int64) + offsets
    counts = np.bincount(flat.ravel(), minlength=n_boot * n_obs)
    counts = counts.reshape(n_boot, n_obs).astype(float)
    return counts @ f / n_obs


def reality_check(
    returns: np.ndarray,
    benchmark: float | np.ndarray = 0.0,
    block_length: float | None = None,
    n_boot: int = 1000,
    rng: np.random.Generator | int | None = None,
) -> dict:
    """
    White's Reality Check p-value for the best of N strategies.

    Parameters
    ----------
    returns : (T observations, N strategy variants), per-period returns.
    benchmark : scalar or (T,) per-period benchmark return. f_k is
        returns[:, k] - benchmark.
    block_length : expected stationary-bootstrap block length. Defaults to
        T^(1/3), which has the right rate with an unjustified constant. Pass
        politis_white_block_length(returns) for a data-driven value -- it is
        opt-in so that a p-value never silently depends on an estimator that
        has not been validated for this data.
    n_boot : B. Monte Carlo error on the p-value is sqrt(p(1-p)/B), about
        0.007 near p = 0.05 at B = 1000. Raise it before trusting a small p.
    rng : Generator or seed.

    Returns a dict. The p-value alone hides everything that matters:

    p_value                 Reality Check p-value, multiplicity corrected
    naive_p_value           same test for the best strategy ALONE, with no
                            correction. The gap between the two IS the
                            data-snooping adjustment, in readable units.
    statistic               V = max_k sqrt(T) * fbar_k
    bootstrap_distribution  the B draws of V*
    mean_relative           fbar_k per strategy
    best_strategy           argmax_k fbar_k

    The null is "no strategy beats the benchmark", so a small p-value says
    the best strategy's margin is too large to be explained by having
    searched N of them.
    """
    r = np.asarray(returns, dtype=float)
    if r.ndim != 2:
        raise ValueError(f"expected a (T, N) matrix, got shape {r.shape}")
    if r.shape[1] < 1:
        raise ValueError("need at least one strategy")
    if not np.isfinite(r).all():
        raise ValueError("returns contain non-finite values")
    if n_boot < 1:
        raise ValueError("need at least one bootstrap replicate")

    b = np.asarray(benchmark, dtype=float)
    if b.ndim == 0:
        f = r - float(b)
    elif b.shape == (r.shape[0],):
        f = r - b[:, None]
    else:
        raise ValueError(
            f"benchmark must be scalar or shape ({r.shape[0]},), got {b.shape}"
        )

    n_obs, n_strat = f.shape
    if block_length is None:
        block_length = rule_of_thumb_block_length(n_obs)
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    root_t = np.sqrt(n_obs)
    mean_relative = f.mean(axis=0)
    best = int(np.argmax(mean_relative))
    statistic = float(root_t * mean_relative[best])

    indices = stationary_bootstrap_indices(n_obs, block_length, n_boot, rng)
    boot_means = _bootstrap_means(f, indices)

    # Centre on the SAMPLE mean, then take the max. Both halves matter.
    centred = root_t * (boot_means - mean_relative[None, :])
    v_star = centred.max(axis=1)

    # (B + 1) denominator: under the null the observed statistic is itself a
    # draw from this distribution, so it belongs in the reference set. That
    # makes the test exact at finite B and stops p from ever being exactly 0,
    # which would claim infinite evidence from finitely many replicates.
    # >= rather than > so ties count against rejection.
    p_value = (1.0 + np.sum(v_star >= statistic)) / (n_boot + 1.0)
    naive_p = (1.0 + np.sum(centred[:, best] >= statistic)) / (n_boot + 1.0)

    return {
        "p_value": float(p_value),
        "naive_p_value": float(naive_p),
        "statistic": statistic,
        "bootstrap_distribution": v_star,
        "mean_relative": mean_relative,
        "best_strategy": best,
        "block_length": float(block_length),
        "n_boot": n_boot,
        "n_obs": n_obs,
        "n_strategies": n_strat,
    }
