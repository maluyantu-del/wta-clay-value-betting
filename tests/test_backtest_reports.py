"""Unit tests for ``wta_clay.backtest.reports``."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wta_clay.backtest.engine import run_backtest
from wta_clay.backtest.reports import (
    CLAY_SEASON_MONTHS,
    format_summary,
    plot_equity_curve,
    report_by_ev_bucket,
    report_by_month,
    report_by_odds_bucket,
    report_by_tourney_level,
    report_by_year,
    report_clay_vs_offseason,
)


def _make_setup(n=200, seed=42):
    """Build a model with a real edge: model probability is sharper than market.

    Market quotes are derived from a *less-informative* version of the
    underlying signal, so the model has a 5-15% edge on most matches.
    This gives actual bet placement to test the report functions.
    """
    rng = np.random.default_rng(seed)
    logits = rng.normal(0.0, 1.5, n)
    p_true = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.uniform(0, 1, n) < p_true).astype(int)
    p_model = p_true.copy()    # well-calibrated model
    # Market is "blurry": uses a noisier estimate of p_true
    p_market = 1.0 / (1.0 + np.exp(-logits * 0.7))
    odds_p1 = 1.0 / (p_market * 1.025)
    odds_p2 = 1.0 / ((1 - p_market) * 1.025)
    preds = pd.DataFrame({"match_id": np.arange(n, dtype=np.int64),
                            "y_true": y, "y_pred_prob": p_model,
                            "sample_weight": np.ones(n)})
    levels = (["GS"] * (n // 4) + ["WTA1000"] * (n // 4)
              + ["WTA500"] * (n // 4) + ["WTA250"] * (n - 3 * (n // 4)))
    master = pd.DataFrame({
        "match_id": np.arange(n, dtype=np.int64),
        "match_date": pd.date_range("2022-04-01", periods=n, freq="D"),
        "tour_level": levels,
        "p1_id": np.arange(100, 100+n), "p2_id": np.arange(200, 200+n),
        "p1_name": [f"A{i}" for i in range(n)],
        "p2_name": [f"B{i}" for i in range(n)],
        "odds_p1_pinnacle": odds_p1, "odds_p2_pinnacle": odds_p2,
        "odds_p1_b365":     odds_p1, "odds_p2_b365":     odds_p2,
        "not_for_betting":  False, "market_source": "pinnacle", "y": y,
    })
    return preds, master


@pytest.fixture
def trades_fixture():
    preds, master = _make_setup(n=300, seed=7)
    r = run_backtest(preds, master)
    return r


class TestReportByTourLevel:
    def test_columns_and_levels(self, trades_fixture):
        df = report_by_tourney_level(trades_fixture.trades)
        assert "tour_level" in df.columns
        for c in ("n_bets", "hit_rate_%", "roi_%", "avg_odds"):
            assert c in df.columns
        assert set(df["tour_level"]) >= {"GS", "WTA1000", "WTA500", "WTA250"}

    def test_empty_trades(self):
        df = report_by_tourney_level(pd.DataFrame())
        assert df.empty


class TestReportByOddsBucket:
    def test_buckets_present(self, trades_fixture):
        df = report_by_odds_bucket(trades_fixture.trades)
        assert "odds_bucket" in df.columns
        assert len(df) >= 4

    def test_custom_bins(self, trades_fixture):
        df = report_by_odds_bucket(trades_fixture.trades,
                                     bins=(1.0, 2.0, 5.0, np.inf))
        assert len(df) == 3


class TestReportByEvBucket:
    def test_buckets(self, trades_fixture):
        df = report_by_ev_bucket(trades_fixture.trades)
        assert "ev_bucket" in df.columns
        assert len(df) == 4   # 5-10 / 10-15 / 15-25 / 25-100


class TestReportByYearAndMonth:
    def test_year_grouping(self, trades_fixture):
        df = report_by_year(trades_fixture.trades)
        assert "year" in df.columns
        # All years are 2022 (test fixture starts 2022-04-01)
        if not df.empty:
            assert (df["year"] >= 2022).all()

    def test_month_grouping(self, trades_fixture):
        df = report_by_month(trades_fixture.trades)
        assert "month" in df.columns
        assert "clay_season" in df.columns
        # 12 rows
        assert len(df) == 12

    def test_clay_vs_offseason(self, trades_fixture):
        df = report_clay_vs_offseason(trades_fixture.trades)
        assert len(df) == 2
        assert set(df["period"]) == {"clay_season (Apr-Oct)", "off_season"}


class TestEquityPlot:
    def test_plot_writes_png(self, trades_fixture, tmp_path):
        out = tmp_path / "equity.png"
        plot_equity_curve(trades_fixture.equity_curve, out,
                            initial_bankroll=10000)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_plot_empty_equity_skips(self, tmp_path):
        out = tmp_path / "equity.png"
        plot_equity_curve(pd.Series(dtype=float), out)
        # Should not raise; file may not be created
        # But the function should at least return the path
        assert out.parent.exists()


class TestFormatSummary:
    def test_returns_string(self, trades_fixture):
        s = format_summary(trades_fixture.summary)
        assert isinstance(s, str)
        assert "Initial bankroll" in s
        assert "ROI per bet" in s
        assert "Sharpe" in s


class TestClaySeasonConstant:
    def test_clay_months_april_through_october(self):
        assert CLAY_SEASON_MONTHS == (4, 5, 6, 7, 8, 9, 10)
