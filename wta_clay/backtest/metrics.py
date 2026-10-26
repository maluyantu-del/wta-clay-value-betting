"""Backtest performance metrics.

Inputs are typically:
    trades   : DataFrame with columns ``stake, pnl, won, odds_taken, ev`` etc.
    equity   : Series of bankroll values indexed by date / bet number.

ROI conventions
---------------
**ROI per bet** = sum(pnl) / sum(stake)  → returns per unit at risk.
This is the most authoritative "edge" metric — independent of bankroll
sizing, comparable across staking strategies.

**Total return** = (final_bankroll − initial_bankroll) / initial_bankroll
Reflects the actual bankroll growth including compounding.

Both numbers matter; ``roi_per_bet`` measures the model's edge,
``total_return`` measures the realised compounding given Kelly sizing.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

#: Trading days in a year — used for annualisation. WTA clay season is
#: April-October but bets land throughout the year on test set; using
#: 365 calendar days is the simplest convention.
TRADING_DAYS_PER_YEAR: int = 365


# ===========================================================================
# Per-bet metrics (independent of bankroll sizing)
# ===========================================================================

def roi_per_bet(trades: pd.DataFrame) -> float:
    """ROI per unit staked: ``sum(pnl) / sum(stake)``.

    The single most authoritative edge metric.
    """
    if trades.empty:
        return float("nan")
    s = float(trades["stake"].sum())
    if s <= 0:
        return float("nan")
    return float(trades["pnl"].sum()) / s


def hit_rate(trades: pd.DataFrame) -> float:
    """Fraction of bets won."""
    if trades.empty:
        return float("nan")
    return float(trades["won"].mean())


def avg_odds_taken(trades: pd.DataFrame) -> float:
    """Mean decimal odd of bets placed."""
    if trades.empty:
        return float("nan")
    return float(trades["odds_taken"].mean())


def avg_ev(trades: pd.DataFrame) -> float:
    """Mean EV across bets placed."""
    if trades.empty:
        return float("nan")
    return float(trades["ev"].mean())


def avg_stake(trades: pd.DataFrame) -> float:
    """Mean absolute stake size."""
    if trades.empty:
        return float("nan")
    return float(trades["stake"].mean())


def avg_pnl_per_bet(trades: pd.DataFrame) -> float:
    """Mean PnL per bet (currency units)."""
    if trades.empty:
        return float("nan")
    return float(trades["pnl"].mean())


def n_bets(trades: pd.DataFrame) -> int:
    """Number of bets placed."""
    return int(len(trades))


def n_wins(trades: pd.DataFrame) -> int:
    return int(trades["won"].sum()) if not trades.empty else 0


def kelly_concentration(trades: pd.DataFrame) -> dict:
    """Distribution of bet sizes as fraction of bankroll.

    Returns mean / median / max / std of ``kelly_frac`` column.
    """
    if trades.empty or "kelly_frac" not in trades.columns:
        return {"mean": float("nan"), "median": float("nan"),
                "max": float("nan"), "std": float("nan")}
    f = trades["kelly_frac"]
    return {
        "mean":   round(float(f.mean()),   6),
        "median": round(float(f.median()), 6),
        "max":    round(float(f.max()),    6),
        "std":    round(float(f.std()),    6),
    }


# ===========================================================================
# Bankroll-aware metrics (compound returns)
# ===========================================================================

def total_return_pct(equity: pd.Series) -> float:
    """``(final − initial) / initial × 100``."""
    if equity.empty or equity.iloc[0] <= 0:
        return float("nan")
    return float((equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0] * 100)


def annualized_return_pct(equity: pd.Series, n_years: float) -> float:
    """Compound annual growth rate (CAGR), as percent.

    ``CAGR = (final / initial)^(1/n_years) − 1``
    """
    if equity.empty or equity.iloc[0] <= 0 or n_years <= 0:
        return float("nan")
    if equity.iloc[-1] <= 0:
        return -100.0
    growth = equity.iloc[-1] / equity.iloc[0]
    return float((growth ** (1.0 / n_years) - 1.0) * 100)


def max_drawdown_pct(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown, as percent (positive number).

    Returns 0 if equity monotonically rises.
    """
    if equity.empty:
        return float("nan")
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max
    return float(-dd.min() * 100)   # positive number


def sharpe(daily_pnl: pd.Series, *, rf: float = 0.0) -> float:
    """Annualised Sharpe ratio of a daily-PnL series.

    ``Sharpe = (mean(daily_returns) − rf) / std(daily_returns) × sqrt(252)``

    For betting, ``daily_returns`` is interpreted as fractional daily
    bankroll change. Uses 365-day year convention for sports (continuous
    activity; not a 252-day stock-trading year).
    """
    if daily_pnl.empty:
        return float("nan")
    s = daily_pnl.std()
    if s == 0 or pd.isna(s):
        return float("nan")
    return float((daily_pnl.mean() - rf) / s * math.sqrt(TRADING_DAYS_PER_YEAR))


# ===========================================================================
# Helper: build daily-PnL series from trades
# ===========================================================================

def daily_returns(trades: pd.DataFrame, equity: pd.Series) -> pd.Series:
    """Build a per-calendar-day fractional return series from equity.

    ``ret_t = (equity_t − equity_{t-1}) / equity_{t-1}``
    """
    if equity.empty:
        return pd.Series(dtype=float)
    daily = equity.copy()
    if not isinstance(daily.index, pd.DatetimeIndex):
        return pd.Series(dtype=float)
    daily = daily.resample("D").last().ffill()
    rets = daily.pct_change().dropna()
    return rets
