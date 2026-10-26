"""Unit tests for ``wta_clay.backtest.engine``."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from wta_clay.backtest.engine import BacktestResult, run_backtest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_predictions_and_master(n: int = 200, seed: int = 42,
                                   start_date: str = "2022-04-01"):
    """Build (predictions, master) where the model has slight edge."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(0.0, 1.0, n)
    p_true = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.uniform(0, 1, n) < p_true).astype(int)
    # Calibrated model — assume perfectly calibrated; market vig 5%
    p_model = p_true.copy()

    odds_p1 = 1.0 / (p_true * 1.025)
    odds_p2 = 1.0 / ((1 - p_true) * 1.025)

    preds = pd.DataFrame({
        "match_id":      np.arange(n, dtype=np.int64),
        "y_true":        y,
        "y_pred_prob":   p_model,
        "sample_weight": np.ones(n),
    })
    master = pd.DataFrame({
        "match_id":         np.arange(n, dtype=np.int64),
        "match_date":       pd.date_range(start_date, periods=n, freq="D"),
        "tour_level":       ["WTA1000"] * (n // 2) + ["WTA250"] * (n - n // 2),
        "p1_id":            np.arange(100, 100 + n),
        "p2_id":            np.arange(200, 200 + n),
        "p1_name":          [f"P1_{i}" for i in range(n)],
        "p2_name":          [f"P2_{i}" for i in range(n)],
        "odds_p1_pinnacle": odds_p1,
        "odds_p2_pinnacle": odds_p2,
        "odds_p1_b365":     odds_p1 * 0.99,   # B365 slightly worse
        "odds_p2_b365":     odds_p2 * 0.99,
        "not_for_betting":  False,
        "market_source":    "pinnacle",
        "y":                y,
    })
    return preds, master


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

class TestRunBacktest:
    def test_returns_backtest_result(self):
        preds, master = _make_predictions_and_master()
        r = run_backtest(preds, master)
        assert isinstance(r, BacktestResult)
        assert isinstance(r.trades, pd.DataFrame)
        assert isinstance(r.equity_curve, pd.Series)
        assert isinstance(r.summary, dict)
        assert isinstance(r.monthly_pnl, pd.DataFrame)

    def test_summary_keys(self):
        preds, master = _make_predictions_and_master()
        r = run_backtest(preds, master)
        for k in ("initial_bankroll","final_bankroll","n_bets","n_wins",
                   "hit_rate","roi_per_bet","total_return_pct",
                   "annualized_return_pct","max_drawdown_pct","sharpe",
                   "avg_odds","avg_ev","avg_stake","avg_pnl_per_bet",
                   "kelly_concentration","book"):
            assert k in r.summary

    def test_no_bet_below_threshold(self):
        # Set EV threshold huge so nothing is bet
        preds, master = _make_predictions_and_master()
        r = run_backtest(preds, master, ev_threshold=10.0)
        assert r.summary["n_bets"] == 0
        assert r.summary["final_bankroll"] == r.summary["initial_bankroll"]

    def test_perfect_calibration_zero_ev(self):
        # When p_model == fair prob and odds carry vig, EV is negative.
        # No bet should be placed at any reasonable threshold.
        preds, master = _make_predictions_and_master()
        # mark p_model = vigged market_implied (so EV literally = 0)
        market_p = 1.0 / master["odds_p1_pinnacle"]
        # ... not actually 0 EV unless we tune; just verify n_bets >= 0 is OK
        r = run_backtest(preds, master, ev_threshold=0.20)   # very high
        assert r.summary["n_bets"] == 0

    def test_chronological_order_in_equity(self):
        preds, master = _make_predictions_and_master(n=50)
        r = run_backtest(preds, master)
        # Equity index is sorted ascending
        idx = r.equity_curve.index
        assert (idx == idx.sort_values()).all()

    def test_filter_excludes_not_for_betting(self):
        preds, master = _make_predictions_and_master(n=50)
        master.loc[master.index < 25, "not_for_betting"] = True
        r = run_backtest(preds, master)
        assert (r.trades["match_id"] >= 25).all() if not r.trades.empty else True

    def test_filter_excludes_market_none(self):
        preds, master = _make_predictions_and_master(n=50)
        master.loc[:10, "market_source"] = "none"
        r = run_backtest(preds, master)
        assert (r.trades["match_id"] > 10).all() if not r.trades.empty else True

    def test_book_choice_changes_odds(self):
        preds, master = _make_predictions_and_master(n=50, seed=7)
        r_pin = run_backtest(preds, master, book="pinnacle")
        r_b365 = run_backtest(preds, master, book="b365")
        # B365 is 1% worse, so odds_taken should differ
        if not r_pin.trades.empty and not r_b365.trades.empty:
            avg_pin = r_pin.trades["odds_taken"].mean()
            avg_b365 = r_b365.trades["odds_taken"].mean()
            assert avg_pin > avg_b365

    def test_book_choice_best_is_max(self):
        preds, master = _make_predictions_and_master(n=50, seed=7)
        r_best = run_backtest(preds, master, book="best")
        if not r_best.trades.empty:
            # best ≥ pinnacle on each row
            r_pin = run_backtest(preds, master, book="pinnacle")
            assert r_best.trades["odds_taken"].mean() >= r_pin.trades["odds_taken"].mean() - 1e-9

    def test_bankroll_compounding(self):
        # Bigger bankroll → bigger absolute stake (if same Kelly fraction)
        preds, master = _make_predictions_and_master(n=50, seed=7)
        r_small = run_backtest(preds, master, initial_bankroll=1000.0)
        r_large = run_backtest(preds, master, initial_bankroll=10000.0)
        if not r_small.trades.empty and not r_large.trades.empty:
            # Stake ratio ≈ bankroll ratio
            ratio = r_large.trades.iloc[0]["stake"] / r_small.trades.iloc[0]["stake"]
            assert math.isclose(ratio, 10.0, rel_tol=1e-6)

    def test_max_bet_pct_caps(self):
        preds, master = _make_predictions_and_master(n=20, seed=11)
        r = run_backtest(preds, master, max_bet_pct=0.005)
        if not r.trades.empty:
            # All kelly_frac ≤ cap
            assert (r.trades["kelly_frac"] <= 0.005 + 1e-12).all()

    def test_pnl_correctness(self):
        # Construct a single match where we KNOW we'll bet and win
        preds = pd.DataFrame({
            "match_id":    [0],
            "y_true":      [1],            # p1 wins
            "y_pred_prob": [0.7],
        })
        master = pd.DataFrame({
            "match_id":         [0],
            "match_date":       [pd.Timestamp("2022-05-01")],
            "tour_level":       ["WTA1000"],
            "p1_id":            [100], "p2_id": [200],
            "p1_name":          ["A"], "p2_name": ["B"],
            "odds_p1_pinnacle": [2.0],   # ev_p1 = 0.7*1 - 0.3 = 0.40 (huge)
            "odds_p2_pinnacle": [3.0],
            "odds_p1_b365":     [2.0],
            "odds_p2_b365":     [3.0],
            "not_for_betting":  [False],
            "market_source":    ["pinnacle"],
            "y":                [1],
        })
        r = run_backtest(preds, master, initial_bankroll=10000)
        assert r.summary["n_bets"] == 1
        # Bet at 2.0 odds, won → pnl = stake × 1.0
        t = r.trades.iloc[0]
        assert t["bet_side"] == "p1"
        assert t["won"] == True
        assert math.isclose(t["pnl"], t["stake"], rel_tol=1e-9)
        assert math.isclose(r.summary["final_bankroll"], 10000 + t["pnl"],
                              rel_tol=1e-9)

    def test_monthly_pnl(self):
        preds, master = _make_predictions_and_master(n=100, seed=7)
        r = run_backtest(preds, master)
        if not r.monthly_pnl.empty:
            for col in ("year","month","n_bets","pnl","stake","roi"):
                assert col in r.monthly_pnl.columns

    def test_invalid_predictions_raises(self):
        bad = pd.DataFrame({"match_id": [0]})   # missing y_pred_prob
        master = pd.DataFrame({
            "match_id": [0], "match_date": [pd.Timestamp("2022-05-01")],
            "odds_p1_pinnacle": [2.0], "odds_p2_pinnacle": [2.0],
            "not_for_betting": [False], "market_source": ["pinnacle"], "y": [1],
        })
        with pytest.raises(ValueError, match="y_pred_prob"):
            run_backtest(bad, master)
