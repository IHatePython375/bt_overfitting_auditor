# backtest-auditor

Detectors for backtest overfitting.

## The problem

Try enough strategy variants and the best one looks great by luck alone. The best
of 100 worthless strategies shows an annualized Sharpe around 1.14; the best of
1,000 shows around 1.47. No standard backtest metric catches this, because the
number of trials never enters the calculation.

This is not a backtester. A backtester answers *how did this strategy perform*.
This answers *is that number real*.

## Status

- [x] Deflated Sharpe Ratio (`src/auditor/sharpe.py`)
- [x] PBO via combinatorially symmetric cross-validation (CSCV)
- [x] White's Reality Check (+ stationary bootstrap, `src/auditor/bootstrap.py`)
- [ ] C++20 CSCV kernel + pybind11
- [ ] Calibration study: ROC curves against known ground-truth alpha
- [ ] Autocorrelation-robust PBO (block bootstrap)

## Quickstart

```
py -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pytest              # fast suite
pytest -m slow      # Monte Carlo calibration + large-S differential tests
```

## Conventions

Two that are load-bearing, because violating either produces wrong numbers rather
than errors:

- **Sharpe ratios are per-period** unless the identifier ends in `_annual`. The
  `T-1` scaling in the standard error assumes per-period. Annualize only at
  presentation boundaries.
- **Kurtosis is raw, not excess** — a normal distribution gives 3.0, not 0.0.
  `scipy.stats.kurtosis` defaults to excess, so `moments()` computes it by hand.

Detectors return dicts rather than floats: when a result looks wrong, the culprit
is almost always an intermediate quantity, so they stay inspectable.

## References

- Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting, and Non-Normality.
  *Journal of Portfolio Management*, 40(5), 94–107.
- Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2017). The
  Probability of Backtest Overfitting. *Journal of Computational Finance*,
  20(4), 39–69.
