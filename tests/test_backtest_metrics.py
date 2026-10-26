"""Unit tests for ``wta_clay.backtest.metrics``."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from wta_clay.backtest.metrics import (
    annualized_return_pct, avg_ev, avg_odds_taken, avg_pnl_per_bet,
    avg_stake, daily_returns, hit_rate, kelly_concentration,
    max_drawdown_pct, n_bets, n_wins, roi_per_bet, sharpe,
    total_return_pct,
)


def _make_trades(n_win=6, n_loss=4, stake=100.0, odds=2.0):
    """Build a trivial trade log with n_win + n_loss bets."""
    rows = []
    for _ in range(n_win):
        rows.append({"stake": stake, "pnl": stake * (odds - 1.0),
                     "won": True, "odds_taken": odds, "ev": 0.10,
                     "kelly_frac": 0.01})
    for _ in range(n_loss):
        rows.append({"stake": stake, "pnl": -stake,
                     "won": False, "odds_taken": odds, "ev": 0.10,
                     "kelly_frac": 0.01})
    return pd.DataFrame(rows)


class TestPerBetMetrics:
    def test_roi_basic(self):
        # 6 wins × +100 = +600, 4 losses × -100 = -400 → +200 PnL
        # Total stake = 1000 → ROI = 200/1000 = 20%
        t = _make_trades(6, 4, stake=100, odds=2.0)
        assert math.isclose(roi_per_bet(t), 0.20, rel_tol=1e-9)

    def test_hit_rate(self):
        t = _make_trades(6, 4)
        assert math.isclose(hit_rate(t), 0.60, rel_tol=1e-9)

    def test_avg_odds(self):
        t = _make_trades(odds=2.5)
        assert math.isclose(avg_odds_taken(t), 2.5)

    def test_avg_ev(self):
        t = _make_trades()
        assert math.isclose(avg_ev(t), 0.10, rel_tol=1e-9)

    def test_avg_stake_and_pnl(self):
        t = _make_trades(6, 4, stake=100, odds=2.0)
        assert math.isclose(avg_stake(t), 100.0)
        # Mean PnL: (6×100 + 4×(-100))/10 = 20
        assert math.isclose(avg_pnl_per_bet(t), 20.0, rel_tol=1e-9)

    def test_n_bets_n_wins(self):
        t = _make_trades(6, 4)
        assert n_bets(t) == 10
        assert n_wins(t) == 6

    def test_kelly_concentration(self):
        t = _make_trades()
        kc = kelly_concentration(t)
        assert math.isclose(kc["mean"], 0.01, rel_tol=1e-9)
        assert math.isclose(kc["max"],  0.01, rel_tol=1e-9)

    def test_empty_trades(self):
        t = pd.DataFrame(columns=["stake","pnl","won","odds_taken","ev","kelly_frac"])
        assert math.isnan(roi_per_bet(t))
        assert math.isnan(hit_rate(t))
        assert n_bets(t) == 0


class TestBankrollMetrics:
    def test_total_return_simple(self):
        eq = pd.Series([10000, 11000, 12000, 11500, 13000])
        assert math.isclose(total_return_pct(eq), 30.0, rel_tol=1e-9)

    def test_total_return_negative(self):
        eq = pd.Series([10000, 9500, 9000])
        assert math.isclose(total_return_pct(eq), -10.0, rel_tol=1e-9)

    def test_annualized_return_doubling_in_one_year(self):
        eq = pd.Series([10000, 20000])
        # 100% return in 1 year → CAGR 100%
        assert math.isclose(annualized_return_pct(eq, 1.0), 100.0, rel_tol=1e-9)

    def test_annualized_return_doubling_in_two_years(self):
        eq = pd.Series([10000, 20000])
        # 100% return in 2 years → CAGR ~41.42%
        cagr = annualized_return_pct(eq, 2.0)
        assert math.isclose(cagr, 41.4213562, rel_tol=1e-5)

    def test_max_drawdown_simple(self):
        # 10000 → 12000 (peak) → 9000 (trough) → 11000
        # MDD = (9000 - 12000)/12000 = -25% → 25%
        eq = pd.Series([10000, 12000, 9000, 11000])
        assert math.isclose(max_drawdown_pct(eq), 25.0, rel_tol=1e-9)

    def test_max_drawdown_zero_for_monotonic(self):
        eq = pd.Series([10000, 11000, 12000, 13000])
        assert math.isclose(max_drawdown_pct(eq), 0.0, abs_tol=1e-9)


class TestSharpe:
    def test_constant_pnl_zero_std_returns_nan(self):
        s = pd.Series([0.001, 0.001, 0.001])
        assert math.isnan(sharpe(s))

    def test_positive_drift(self):
        # Daily returns mean 0.005, std 0.005 → Sharpe = 1 × √365 ≈ 19.1
        rng = np.random.default_rng(0)
        rets = pd.Series(rng.normal(0.005, 0.005, 1000))
        sh = sharpe(rets)
        assert sh > 0

    def test_empty(self):
        assert math.isnan(sharpe(pd.Series(dtype=float)))


class TestDailyReturns:
    def test_basic(self):
        idx = pd.date_range("2024-01-01", periods=4, freq="D")
        eq = pd.Series([10000, 11000, 12100, 13310], index=idx)
        rets = daily_returns(pd.DataFrame(), eq)
        # 10% per day
        assert math.isclose(rets.iloc[0], 0.10, rel_tol=1e-9)
        assert math.isclose(rets.iloc[1], 0.10, rel_tol=1e-9)

    def test_empty(self):
        assert daily_returns(pd.DataFrame(), pd.Series(dtype=float)).empty

    def test_non_datetime_index_returns_empty(self):
        # No DatetimeIndex → can't compute calendar daily returns
        eq = pd.Series([10000, 11000])
        assert daily_returns(pd.DataFrame(), eq).empty
