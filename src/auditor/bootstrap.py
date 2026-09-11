"""
Stationary bootstrap (Politis & Romano 1994) and automatic block-length
selection (Politis & White 2004, corrected by Patton, Politis & White 2009).

Its own module because two things need it: White's Reality Check, and the
autocorrelation-robust PBO variant in week 4.

WHY STATIONARY RATHER THAN FIXED-BLOCK
--------------------------------------
A fixed block bootstrap cuts the series into blocks of constant length. The
resampled series is then not stationary -- observations near a block boundary
behave differently from observations in the middle, and the dependence on
where the grid happens to fall does not vanish as T grows. Drawing block
lengths from a Geometric distribution instead makes the resampled series
strictly stationary, which is what the asymptotics of every downstream test
assume.

Two details do the real work:

  - Geometric block lengths with mean L. Restart with probability p = 1/L at
    each step, otherwise advance one observation.
  - A CIRCULAR wrap. Without it, observations near the end of the series can
    only appear at the end of a block and are systematically under-sampled,
    which breaks the stationarity the scheme exists to provide. Wrapping the
    series into a circle makes every observation equally likely -- a property
    that is directly asserted in the tests.

WHY THIS RETURNS INDICES, NOT RESAMPLED DATA
--------------------------------------------
Both callers resample a (T, N) matrix and must apply the SAME time indices to
every strategy. Resampling strategies independently would destroy the
cross-sectional correlation between them, and for a max-statistic that
correlation is the whole story: N highly correlated variants behave like far
fewer than N independent trials, and independent resampling would silently
assume the full N. Returning indices makes joint resampling the only thing
the interface can express.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "stationary_bootstrap_indices",
    "politis_white_block_length",
    "rule_of_thumb_block_length",
]


# --------------------------------------------------------------------------
# Resampling
# --------------------------------------------------------------------------

def stationary_bootstrap_indices(
    n_obs: int,
    block_length: float,
    n_boot: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Row indices for `n_boot` stationary-bootstrap replicates of a length
    `n_obs` series.

    Returns an (n_boot, n_obs) int array. Apply one row to every column of a
    (T, N) matrix to resample all strategies jointly.

    `block_length` is the EXPECTED block length L; restart probability is
    p = 1/L, so block lengths are Geometric(p) with mean L. L = 1 gives
    p = 1, i.e. the ordinary iid bootstrap -- useful as the deliberately
    misspecified comparison when returns are serially correlated.

    Memory is (n_boot, n_obs) int32 plus a few temporaries of the same shape:
    about 4 MB per array at n_boot = 1000, n_obs = 1000.
    """
    if n_obs < 1:
        raise ValueError("need at least one observation")
    if n_boot < 1:
        raise ValueError("need at least one bootstrap replicate")
    if not np.isfinite(block_length) or block_length < 1:
        raise ValueError(f"block_length must be >= 1, got {block_length}")
    if rng is None:
        rng = np.random.default_rng()

    p_restart = 1.0 / block_length

    # Restart flags. Position 0 always restarts, so every replicate begins at
    # a uniformly drawn point rather than at observation 0.
    restart = rng.random((n_boot, n_obs)) < p_restart
    restart[:, 0] = True

    # A candidate fresh start for every position; only those at a restart are
    # ever used.
    starts = rng.integers(0, n_obs, size=(n_boot, n_obs), dtype=np.int64)

    # Index of the most recent restart, propagated forward.
    positions = np.where(restart, np.arange(n_obs, dtype=np.int64), 0)
    np.maximum.accumulate(positions, axis=1, out=positions)

    # The value drawn at that restart, and how far we have walked since.
    base = np.take_along_axis(starts, positions, axis=1)
    offset = np.arange(n_obs, dtype=np.int64)[None, :] - positions

    # The modulo is the circular wrap.
    return ((base + offset) % n_obs).astype(np.int32)


# --------------------------------------------------------------------------
# Block-length selection
# --------------------------------------------------------------------------

def rule_of_thumb_block_length(n_obs: int) -> float:
    """
    T^(1/3), the default.

    The Politis-White optimal block length is Theta(T^(1/3)); only the
    constant depends on the data. So this has the right rate and an
    unjustified constant of 1, which is a defensible default rather than a
    magic number -- and unlike a data-driven default it keeps the block
    length an explicit, testable input rather than hidden machinery.
    """
    return float(n_obs) ** (1.0 / 3.0)


def _flat_top(s: np.ndarray) -> np.ndarray:
    """Politis-White trapezoidal flat-top lag window."""
    a = np.abs(s)
    return np.where(a <= 0.5, 1.0, np.where(a <= 1.0, 2.0 * (1.0 - a), 0.0))


def _autocovariance(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Biased (divide by n) sample autocovariances, lags 0..max_lag."""
    n = x.size
    e = x - x.mean()
    return np.array([float(e[: n - k] @ e[k:]) / n for k in range(max_lag + 1)])


def _block_length_1d(x: np.ndarray) -> float:
    n = x.size
    if n < 8:
        raise ValueError("need at least 8 observations to estimate a block length")

    k_n = max(5, int(np.ceil(np.log10(n))))
    m_max = int(np.ceil(np.sqrt(n))) + k_n
    b_max = float(np.ceil(min(3.0 * np.sqrt(n), n / 3.0)))

    acov = _autocovariance(x, m_max)
    if acov[0] <= 0:
        return 1.0
    rho = acov / acov[0]

    # Smallest lag beyond which the next k_n autocorrelations are all
    # insignificant. The 2*sqrt(log10(n)/n) band is Politis & White's.
    threshold = 2.0 * np.sqrt(np.log10(n) / n)
    m_hat = 0
    for m in range(0, m_max - k_n + 1):
        window = rho[m + 1 : m + k_n + 1]
        if window.size and np.all(np.abs(window) < threshold):
            m_hat = m
            break
    else:
        m_hat = m_max - k_n

    lag_window = max(1, min(2 * m_hat, m_max))

    k = np.arange(-lag_window, lag_window + 1)
    lam = _flat_top(k / lag_window)
    r = acov[np.abs(k)]

    g_hat = float(np.sum(lam * np.abs(k) * r))
    g_zero = float(np.sum(lam * r))
    if g_zero == 0.0:
        return 1.0

    # D_SB = 2 * g(0)^2 for the stationary bootstrap, so the factor of 2
    # cancels and b = (G / g(0))^(2/3) * T^(1/3).
    d_sb = 2.0 * g_zero ** 2
    b = (2.0 * g_hat ** 2 / d_sb) ** (1.0 / 3.0) * n ** (1.0 / 3.0)

    return float(np.clip(b, 1.0, b_max))


def politis_white_block_length(
    x: np.ndarray,
    aggregate: str = "quantile",
    quantile: float = 0.9,
) -> float:
    """
    Data-driven expected block length for the stationary bootstrap.

    Opt-in, not the default. A p-value that depends on an estimated block
    length depends on this estimator too, and it deserves to be validated on
    its own rather than inside a detector. For AR(1) the target it estimates
    has a closed form -- b = (2*phi/(1-phi^2))^(2/3) * T^(1/3) -- which is
    what the tests check it against.

    For a (T, N) matrix the block length must be a single number, because
    resampling is joint across strategies. `aggregate` says how to collapse
    the N per-column estimates:

      "quantile" (default, 0.9)  high but bounded bias
      "max"                      most conservative, but the bias of a maximum
                                 of N noisy estimators grows with N, so at
                                 N = 100 it systematically over-blocks
      "median"                   lowest variance, least conservative

    Under-blocking is the dangerous direction -- it understates the long-run
    variance and the test over-rejects -- so the default leans high without
    taking an unbounded maximum.
    """
    a = np.asarray(x, dtype=float)
    if a.ndim == 1:
        return _block_length_1d(a)
    if a.ndim != 2:
        raise ValueError(f"expected a 1-D or 2-D array, got shape {a.shape}")

    per_column = np.array([_block_length_1d(a[:, j]) for j in range(a.shape[1])])
    if aggregate == "quantile":
        return float(np.quantile(per_column, quantile))
    if aggregate == "max":
        return float(per_column.max())
    if aggregate == "median":
        return float(np.median(per_column))
    raise ValueError(
        f"unknown aggregate {aggregate!r}; expected 'quantile', 'max' or 'median'"
    )
