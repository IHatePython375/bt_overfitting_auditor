"""
Probabilistic and Deflated Sharpe Ratio.

Reference: Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting and Non-Normality."

UNITS WARNING
-------------
Every Sharpe in this module is PER-PERIOD unless the name says otherwise.
The T-1 scaling in the standard error assumes per-period units, so mixing
annualized and per-period Sharpes silently produces wrong probabilities.
Use annualize() / deannualize() at the boundaries only.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

EULER_MASCHERONI = 0.5772156649015329

# Periods per year, for the usual conventions.
PERIODS = {"daily": 252, "weekly": 52, "monthly": 12}


# --------------------------------------------------------------------------
# Basic estimators
# --------------------------------------------------------------------------

def sharpe_ratio(returns: np.ndarray, rf: float = 0.0) -> float:
    """Per-period Sharpe ratio. `rf` is the per-period risk-free rate."""
    r = np.asarray(returns, dtype=float)
    excess = r - rf
    sd = excess.std(ddof=1)
    if sd == 0:
        raise ValueError("zero return variance; Sharpe undefined")
    return float(excess.mean() / sd)


def annualize(sr_per_period: float, freq: str = "daily") -> float:
    return sr_per_period * np.sqrt(PERIODS[freq])


def deannualize(sr_annual: float, freq: str = "daily") -> float:
    return sr_annual / np.sqrt(PERIODS[freq])


def moments(returns: np.ndarray) -> tuple[float, float]:
    """
    Return (skewness, RAW kurtosis).

    Raw, not excess: a normal distribution gives 3.0, not 0.0. The DSR
    formula below expects this convention. Getting it wrong shifts the
    variance term by (3-1)/4 = 0.5 per unit of SR^2, which is large.

    Population moments (biased, ddof=0) to match the paper.
    """
    r = np.asarray(returns, dtype=float)
    d = r - r.mean()
    m2 = (d ** 2).mean()
    m3 = (d ** 3).mean()
    m4 = (d ** 4).mean()
    if m2 == 0:
        raise ValueError("zero return variance; moments undefined")
    return float(m3 / m2 ** 1.5), float(m4 / m2 ** 2)


# --------------------------------------------------------------------------
# Piece 2: how blurry is the estimate
# --------------------------------------------------------------------------

def sharpe_standard_error(
    sr: float, n_obs: int, skewness: float = 0.0, kurt: float = 3.0
) -> float:
    """
    Standard error of the per-period Sharpe estimate (Lo 2002 / Mertens).

        se = sqrt( (1 - g3*SR + (g4-1)/4 * SR^2) / (T-1) )

    Note the two ways this exceeds the naive 1/sqrt(T):
      - negative skew (g3 < 0) INFLATES it, because the -g3*SR term flips sign
      - fat tails (g4 > 3) inflate it via the SR^2 term

    So a strategy with a smooth equity curve and rare crashes has wider error
    bars than its chart suggests.
    """
    if n_obs < 2:
        raise ValueError("need at least 2 observations")
    variance_term = 1.0 - skewness * sr + (kurt - 1.0) / 4.0 * sr ** 2
    if variance_term <= 0:
        # Possible with extreme skew/kurtosis combinations; the asymptotic
        # approximation has broken down rather than the data being invalid.
        raise ValueError(
            f"variance term non-positive ({variance_term:.4f}); "
            "asymptotic approximation invalid for these moments"
        )
    return float(np.sqrt(variance_term / (n_obs - 1)))


# --------------------------------------------------------------------------
# Piece 3: the luck bar
# --------------------------------------------------------------------------

def expected_max_sharpe(n_trials: int, sigma_sr: float) -> float:
    """
    Expected maximum Sharpe across `n_trials` strategies with TRUE Sharpe 0.

        E[max] ~= sigma_sr * [ (1-g)*Z^-1(1 - 1/N) + g*Z^-1(1 - 1/(N*e)) ]

    `sigma_sr` is the CROSS-SECTIONAL standard deviation of the estimated
    Sharpes across your trials -- how much the variants differ from each
    other. It is NOT the standard error from sharpe_standard_error().
    This is the input people most often get wrong.

    Both `sigma_sr` and the return value are in the same units (pass
    per-period, get per-period).
    """
    if n_trials < 2:
        raise ValueError("expected-max approximation requires n_trials >= 2")
    if sigma_sr <= 0:
        raise ValueError("sigma_sr must be positive")

    g = EULER_MASCHERONI
    q1 = norm.ppf(1.0 - 1.0 / n_trials)
    q2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(sigma_sr * ((1.0 - g) * q1 + g * q2))


# --------------------------------------------------------------------------
# Pieces 4: PSR, then DSR as a special case
# --------------------------------------------------------------------------

def probabilistic_sharpe_ratio(
    sr: float,
    n_obs: int,
    sr_benchmark: float = 0.0,
    skewness: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """
    P(true Sharpe > sr_benchmark), given the observed per-period Sharpe.

    A one-sided test with the null shifted to `sr_benchmark`. With the
    default benchmark of 0 this is the standard PSR.
    """
    se = sharpe_standard_error(sr, n_obs, skewness, kurt)
    return float(norm.cdf((sr - sr_benchmark) / se))


def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int,
    sigma_sr: float,
) -> dict:
    """
    DSR = PSR evaluated against the luck bar instead of against zero.

    That is the whole idea: the null hypothesis moves from "true Sharpe is 0"
    to "true Sharpe is whatever the best of N worthless strategies would show."

    Parameters
    ----------
    returns : per-period return series of the SELECTED strategy
    n_trials : how many variants were tried before picking this one
    sigma_sr : cross-sectional std dev of per-period Sharpes across trials

    Returns a dict so the intermediate quantities stay inspectable -- when a
    DSR looks wrong it is almost always one of these, not the final formula.
    """
    r = np.asarray(returns, dtype=float)
    n_obs = r.size
    sr = sharpe_ratio(r)
    skewness, kurt = moments(r)
    sr0 = expected_max_sharpe(n_trials, sigma_sr)

    return {
        "sharpe": sr,
        "sharpe_annual": annualize(sr),
        "luck_bar": sr0,
        "luck_bar_annual": annualize(sr0),
        "std_error": sharpe_standard_error(sr, n_obs, skewness, kurt),
        "skew": skewness,
        "kurtosis": kurt,
        "n_obs": n_obs,
        "n_trials": n_trials,
        "psr": probabilistic_sharpe_ratio(sr, n_obs, 0.0, skewness, kurt),
        "dsr": probabilistic_sharpe_ratio(sr, n_obs, sr0, skewness, kurt),
    }