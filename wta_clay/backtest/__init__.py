"""Backtest layer — production-grade ROI evaluation pipeline.

Modules
-------
odds_utils  Vig removal, implied prob, decimal odds conversion.
staking     Kelly + fractional Kelly + per-bet cap.
engine      Sequential bankroll-aware bet placement on test predictions.
metrics     ROI, Sharpe, max drawdown, hit rate, etc.
reports     Grouped breakdowns + equity-curve plot.

The mini-backtest in :mod:`wta_clay.models.lgbm` is **flat 1u stake** for
quick alpha signal at training time; this module is **bankroll-aware
fractional Kelly** for production-realistic ROI / Sharpe / drawdown.
"""
