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
- [x] C++20 CSCV kernel + pybind11 (`src/kernel/`)
- [ ] Calibration study: ROC curves against known ground-truth alpha
- [ ] Autocorrelation-robust PBO (block bootstrap)

## Quickstart

```
py -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pytest              # fast suite
pytest -m slow      # Monte Carlo calibration + large-S differential tests

py scripts/build_kernel.py    # optional: the C++ kernel
```

## The C++ kernel

CSCV enumerates `C(S, S/2)` partitions, which is 12,870 at S=16 and
601,080,390 at S=32. The numpy path costs about 71 us per partition, so S=24
is roughly three minutes and S=32 is about twelve hours. `src/kernel/cscv.hpp`
is the same reduction in C++20, threaded over partitions.

Measured at T=2000, N=50 on an i7-13650HX (6 P-cores + 8 E-cores), each size
timed in a fresh process — `py scripts/bench_kernel.py` reproduces it:

| S  | partitions | numpy  | cpp 1 thread | cpp all cores |
|----|-----------:|-------:|-------------:|--------------:|
| 16 |     12,870 |  0.92s |        0.01s |         0.01s |
| 20 |    184,756 |  ~13s  |        0.15s |         0.03s |
| 24 |  2,704,156 |  ~193s |        2.45s |         0.71s |

About 80x on one thread; the rest is threading, which on this laptop tops out
near 3.5x against a single core running at full boost.

It is optional in the real sense. `pbo()` falls back to numpy when the kernel
is not built, `method` in the result says which path actually ran, and the
kernel's tests skip rather than fail. Two properties are worth knowing:

- **It agrees with numpy bitwise.** Ranks, relative ranks and winners are
  asserted *equal* in `tests/test_kernel.py`, not close, which is only
  possible because the kernel accumulates blocks in the same order, is built
  with `-ffp-contract=off`, and keeps true division instead of hoisting a
  reciprocal. A tolerance loose enough to pass would be loose enough to hide
  a genuine reordering.
- **Thread count is not an answer knob.** Work is claimed dynamically from a
  shared counter, so which thread runs a chunk varies; what lands at
  partition `p` does not.

`store_lambdas=False` keeps only the streaming summary. The per-partition
detail arrays cost 28 bytes each, which is 17 GB at S=32 — that ceiling, not
speed, is what makes the largest cases infeasible without it.

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
