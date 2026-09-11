"""Backtest overfitting auditor.

Not a backtester. A backtester answers "how did this strategy perform";
this answers "is that number real, or is it the best of many tries."

Unit convention: every Sharpe ratio is PER-PERIOD unless the identifier
ends in ``_annual``. Annualize only at presentation boundaries.
"""

__version__ = "0.1.0"
