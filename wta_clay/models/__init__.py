"""Model layer.

Modules:
    splits          time-based train/valid/test splits + feature matrix construction
    lgbm            LightGBM training + inline mini-backtest
    calibration     isotonic post-calibration on valid predictions
    leakage_checks  feature × target temporal leakage assertions (TBD)
"""
