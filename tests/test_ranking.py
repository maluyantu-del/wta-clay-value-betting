"""Unit tests for wta_clay.features.ranking."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wta_clay.features import ranking
from wta_clay.features.ranking import (
    EXTREME_TREND_WARN, FEATURE_COLUMNS, MIN_POINTS_FOR_TREND,
    TREND_WINDOW_WEEKS, RankingResult, _build_rankings_index,
    _trend_slope, compute_ranking_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _master_row(match_id, *, p1=100, p2=200,
                p1_rank=10, p2_rank=50,
                date="2020-05-01", tourney_id="2020-001"):
    return {
        "match_id": match_id,
        "match_date": pd.Timestamp(date),
        "tourney_id": tourney_id,
        "p1_id": p1, "p2_id": p2,
        "p1_rank": p1_rank, "p2_rank": p2_rank,
        "y": 1,
        "stable_match_key": f"{tourney_id}_{p1}_{p2}",
    }


def _ranking_row(player_id, ranking_date, rank):
    return {
        "ranking_date": pd.Timestamp(ranking_date),
        "player_id": player_id, "rank": rank,
    }


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_window_size(self):
        assert TREND_WINDOW_WEEKS == 8
        assert MIN_POINTS_FOR_TREND == 4
        assert EXTREME_TREND_WARN == 30.0

    def test_feature_columns(self):
        for c in ("match_id",
                  "rank_p1", "rank_p2", "rank_diff",
                  "rank_trend_p1", "rank_trend_p2", "rank_trend_diff"):
            assert c in FEATURE_COLUMNS
        assert len(FEATURE_COLUMNS) == 7


# ---------------------------------------------------------------------------
# Schema check
# ---------------------------------------------------------------------------

class TestSchema:
    def test_master_missing_p1_rank_raises(self):
        master = pd.DataFrame([{
            "match_id": 0, "match_date": pd.Timestamp("2020-05-01"),
            "p1_id": 100, "p2_id": 200,
            "stable_match_key": "k",
            # Missing p1_rank, p2_rank
        }])
        rk = pd.DataFrame([_ranking_row(100, "2020-04-01", 10)])
        with pytest.raises(ValueError, match="p1_rank|p2_rank"):
            compute_ranking_features(master, rk, run_sanity_checks=False)

    def test_rankings_missing_player_id_raises(self):
        master = pd.DataFrame([_master_row(0)])
        rk = pd.DataFrame([{
            "ranking_date": pd.Timestamp("2020-04-01"), "rank": 10,
        }])
        with pytest.raises(ValueError, match="player_id"):
            compute_ranking_features(master, rk, run_sanity_checks=False)


# ---------------------------------------------------------------------------
# Direction lock
# ---------------------------------------------------------------------------

class TestDirectionLock:
    def test_rank_diff_positive_when_p1_stronger(self):
        # p1 ranked 5 (stronger), p2 ranked 50 → rank_diff = 50 - 5 = +45
        master = pd.DataFrame([_master_row(0, p1_rank=5, p2_rank=50)])
        rk = pd.DataFrame([])
        result = compute_ranking_features(master, rk, run_sanity_checks=False)
        assert int(result.features.iloc[0]["rank_diff"]) == 45

    def test_rank_diff_negative_when_p2_stronger(self):
        master = pd.DataFrame([_master_row(0, p1_rank=50, p2_rank=5)])
        rk = pd.DataFrame([])
        result = compute_ranking_features(master, rk, run_sanity_checks=False)
        assert int(result.features.iloc[0]["rank_diff"]) == -45


# ---------------------------------------------------------------------------
# Trend slope
# ---------------------------------------------------------------------------

class TestTrendSlope:
    def _build_history(self, ranks_per_week, start="2020-03-01"):
        """Build weekly rankings from a list of rank values."""
        rows = []
        for i, r in enumerate(ranks_per_week):
            rows.append(_ranking_row(
                100, pd.Timestamp(start) + pd.Timedelta(weeks=i), r,
            ))
        return pd.DataFrame(rows)

    def test_no_history_returns_nan(self):
        slope = _trend_slope(None, pd.Timestamp("2020-05-01"), 8, 4)
        assert np.isnan(slope)

    def test_below_min_points_returns_nan(self):
        # Only 3 weekly points in window — below min_points=4
        rk = self._build_history([10, 9, 8], start="2020-04-01")
        rk_g = rk.sort_values("ranking_date").reset_index(drop=True)
        slope = _trend_slope(rk_g, pd.Timestamp("2020-05-01"), 8, 4)
        assert np.isnan(slope)

    def test_steady_decline_negative_slope(self):
        # Ranks dropping by 1 per week (player rising up) → slope = -1
        ranks = [50, 49, 48, 47, 46, 45, 44, 43]
        rk = self._build_history(ranks, start="2020-03-06")
        rk_g = rk.sort_values("ranking_date").reset_index(drop=True)
        slope = _trend_slope(rk_g, pd.Timestamp("2020-05-04"), 8, 4)
        # Expected slope: -1.0/week (descending rank value)
        assert abs(slope - (-1.0)) < 1e-9

    def test_steady_rise_positive_slope(self):
        # Ranks growing by 2 per week (player falling) → slope = +2
        ranks = [10, 12, 14, 16, 18, 20, 22, 24]
        rk = self._build_history(ranks, start="2020-03-06")
        rk_g = rk.sort_values("ranking_date").reset_index(drop=True)
        slope = _trend_slope(rk_g, pd.Timestamp("2020-05-04"), 8, 4)
        assert abs(slope - 2.0) < 1e-9

    def test_strict_upper_bound(self):
        # Ranking exactly on target_date should be EXCLUDED (strict <)
        ranks = [10, 9, 8, 7, 6, 5]
        rk = self._build_history(ranks, start="2020-03-30")
        rk_g = rk.sort_values("ranking_date").reset_index(drop=True)
        # target_date = last entry's date — should exclude it (strict <)
        target = pd.Timestamp("2020-03-30") + pd.Timedelta(weeks=5)
        slope = _trend_slope(rk_g, target, 8, 4)
        # Only first 5 entries: 10, 9, 8, 7, 6 → slope = -1
        assert abs(slope - (-1.0)) < 1e-9


# ---------------------------------------------------------------------------
# End-to-end synthetic
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_basic_match_with_rising_p1(self):
        # p1 has 8 weeks of declining rank (rising up); p2 has 8 weeks flat
        master = pd.DataFrame([_master_row(
            0, p1=100, p2=200, p1_rank=20, p2_rank=50,
            date="2020-05-04",
        )])
        # p1 history: rank 30 → 20 over 8 weeks (slope = -1.25/week)
        p1_rows = [
            _ranking_row(100, pd.Timestamp("2020-03-09") + pd.Timedelta(weeks=i),
                         30 - i * 1.25)
            for i in range(8)
        ]
        # p2 history: rank 50 flat
        p2_rows = [
            _ranking_row(200, pd.Timestamp("2020-03-09") + pd.Timedelta(weeks=i), 50)
            for i in range(8)
        ]
        rk = pd.DataFrame(p1_rows + p2_rows)
        result = compute_ranking_features(master, rk, run_sanity_checks=False)
        f = result.features.iloc[0]
        assert int(f["rank_p1"]) == 20
        assert int(f["rank_p2"]) == 50
        assert int(f["rank_diff"]) == 30      # p2_rank - p1_rank
        # p1 trend = -1.25/week (rising)
        assert abs(f["rank_trend_p1"] - (-1.25)) < 1e-6
        # p2 trend = 0 (flat)
        assert abs(f["rank_trend_p2"] - 0.0) < 1e-9
        # rank_trend_diff = p2 - p1 = 0 - (-1.25) = +1.25 (positive = p1 rising faster)
        assert abs(f["rank_trend_diff"] - 1.25) < 1e-6

    def test_master_rank_nan_propagates_to_diff(self):
        master = pd.DataFrame([_master_row(0, p1_rank=pd.NA, p2_rank=50)])
        rk = pd.DataFrame([])
        result = compute_ranking_features(master, rk, run_sanity_checks=False)
        assert pd.isna(result.features.iloc[0]["rank_diff"])

    def test_player_with_no_ranking_history_trend_nan(self):
        master = pd.DataFrame([_master_row(0, p1=999, p2=200,
                                            p1_rank=10, p2_rank=50)])
        # No rankings for player 999
        p2_rows = [_ranking_row(200, pd.Timestamp("2020-03-09") + pd.Timedelta(weeks=i), 50)
                   for i in range(8)]
        rk = pd.DataFrame(p2_rows)
        result = compute_ranking_features(master, rk, run_sanity_checks=False)
        f = result.features.iloc[0]
        assert pd.isna(f["rank_trend_p1"])
        assert pd.isna(f["rank_trend_diff"])


# ---------------------------------------------------------------------------
# NaN consistency self-check
# ---------------------------------------------------------------------------

class TestNanConsistency:
    def test_diff_nan_iff_either_side_nan(self):
        # Build a few rows: some with both ranks, some missing.
        master = pd.DataFrame([
            _master_row(0, p1_rank=10, p2_rank=50),
            _master_row(1, p1_rank=pd.NA, p2_rank=20, p1=300, p2=400,
                        tourney_id="2020-002"),
            _master_row(2, p1_rank=15, p2_rank=pd.NA, p1=500, p2=600,
                        tourney_id="2020-003"),
        ])
        rk = pd.DataFrame([])
        result = compute_ranking_features(master, rk, run_sanity_checks=False)
        f = result.features
        # Row 0: both present → diff present
        assert pd.notna(f.loc[f["match_id"] == 0, "rank_diff"].iloc[0])
        # Row 1: p1_rank NaN → diff NaN
        assert pd.isna(f.loc[f["match_id"] == 1, "rank_diff"].iloc[0])
        # Row 2: p2_rank NaN → diff NaN
        assert pd.isna(f.loc[f["match_id"] == 2, "rank_diff"].iloc[0])


# ---------------------------------------------------------------------------
# Empty / output schema
# ---------------------------------------------------------------------------

class TestEmpty:
    def test_empty_master(self):
        result = compute_ranking_features(pd.DataFrame(), pd.DataFrame(),
                                            run_sanity_checks=False)
        assert result.features.empty


class TestOutputSchema:
    def test_columns_match_canonical_order(self):
        master = pd.DataFrame([_master_row(0)])
        rk = pd.DataFrame([])
        result = compute_ranking_features(master, rk, run_sanity_checks=False)
        assert list(result.features.columns) == list(FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

class TestRankingsIndex:
    def test_index_groups_by_player(self):
        rk = pd.DataFrame([
            _ranking_row(100, "2020-04-01", 10),
            _ranking_row(100, "2020-04-08", 9),
            _ranking_row(200, "2020-04-01", 50),
        ])
        idx = _build_rankings_index(rk)
        assert 100 in idx and 200 in idx
        assert len(idx[100]) == 2
        assert len(idx[200]) == 1

    def test_each_player_sorted_ascending(self):
        rk = pd.DataFrame([
            _ranking_row(100, "2020-04-15", 8),
            _ranking_row(100, "2020-04-01", 10),
            _ranking_row(100, "2020-04-08", 9),
        ])
        idx = _build_rankings_index(rk)
        assert (idx[100]["ranking_date"].diff().dropna() >= pd.Timedelta(0)).all()
