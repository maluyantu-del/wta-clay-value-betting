"""Unit tests for wta_clay.features.fatigue."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wta_clay.features import fatigue
from wta_clay.features.fatigue import (
    CLAY_SEASON_END_MONTH, CLAY_SEASON_START_MONTH, FEATURE_COLUMNS,
    FatigueResult, _query_clay_season_cum, _query_matches_this_week,
    _query_rest_days, compute_fatigue_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _master_row(match_id, *, p1=100, p2=200, y=1, date="2020-05-15",
                tourney_id="2020-001"):
    return {
        "match_id": match_id,
        "match_date": pd.Timestamp(date),
        "tourney_id": tourney_id,
        "p1_id": p1, "p2_id": p2, "y": y,
        "stable_match_key": f"{tourney_id}_{p1}_{p2}",
    }


def _sackmann_row(*, winner_id, loser_id, date="2020-05-15",
                  tourney_id="2020-001", round_="R32", surface="Clay"):
    return {
        "tourney_id": tourney_id, "round": round_,
        "match_date": pd.Timestamp(date),
        "winner_id": winner_id, "loser_id": loser_id,
        "surface": surface,
    }


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_season_window(self):
        assert CLAY_SEASON_START_MONTH == 4
        assert CLAY_SEASON_END_MONTH == 10

    def test_feature_columns(self):
        for c in ("match_id",
                  "rest_days_p1", "rest_days_p2", "rest_days_diff",
                  "matches_this_week_p1", "matches_this_week_p2",
                  "matches_this_week_diff",
                  "clay_season_cum_p1", "clay_season_cum_p2",
                  "clay_season_cum_diff"):
            assert c in FEATURE_COLUMNS
        assert len(FEATURE_COLUMNS) == 10


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

class TestRestDays:
    def test_no_history_returns_nan(self):
        v = _query_rest_days([], pd.Timestamp("2020-05-15"), "k_self")
        assert np.isnan(v)

    def test_single_prior_match(self):
        history = [(pd.Timestamp("2020-05-08"), True, "k1")]
        v = _query_rest_days(history, pd.Timestamp("2020-05-15"), "k_self")
        assert v == 7.0

    def test_takes_most_recent_prior(self):
        history = [
            (pd.Timestamp("2020-04-01"), True, "k1"),
            (pd.Timestamp("2020-04-15"), True, "k2"),
            (pd.Timestamp("2020-05-10"), False, "k3"),
        ]
        v = _query_rest_days(history, pd.Timestamp("2020-05-15"), "k_self")
        # Most recent prior is May 10 → 5 days
        assert v == 5.0

    def test_self_excluded_via_key(self):
        history = [(pd.Timestamp("2020-05-15"), True, "k_self")]
        v = _query_rest_days(history, pd.Timestamp("2020-05-15"), "k_self")
        assert np.isnan(v)


class TestMatchesThisWeek:
    def test_no_prior_in_week(self):
        history = [(pd.Timestamp("2020-05-04"), True, "k1")]   # prev week
        v = _query_matches_this_week(history, pd.Timestamp("2020-05-15"),
                                       "k_self")
        assert v == 0

    def test_two_prior_in_same_week(self):
        # ISO week of 2020-05-15 (Friday) is week 20 (May 11-17).
        # NB: history is chronologically sorted — that's the precondition
        # the early-termination loop relies on.
        history = [
            (pd.Timestamp("2020-05-04"), True, "k0"),  # prev week
            (pd.Timestamp("2020-05-11"), True, "k1"),  # Mon same week
            (pd.Timestamp("2020-05-13"), True, "k2"),  # Wed same week
        ]
        v = _query_matches_this_week(history, pd.Timestamp("2020-05-15"),
                                       "k_self")
        assert v == 2

    def test_same_day_excluded_strict(self):
        history = [
            (pd.Timestamp("2020-05-15"), True, "k_other"),  # same day
        ]
        v = _query_matches_this_week(history, pd.Timestamp("2020-05-15"),
                                       "k_self")
        # date == current_date → strict < excludes it
        assert v == 0


class TestClaySeasonCum:
    def test_in_season_with_prior_clay(self):
        history = [
            (pd.Timestamp("2020-04-05"), True,  "k1"),
            (pd.Timestamp("2020-04-15"), True,  "k2"),
            (pd.Timestamp("2020-04-20"), False, "k3"),  # hard — excluded
            (pd.Timestamp("2020-05-01"), True,  "k4"),
        ]
        v = _query_clay_season_cum(history, pd.Timestamp("2020-05-15"),
                                     "k_self", 4, 10)
        # Only clay matches in [Apr 1, May 15) → 3 (k1, k2, k4)
        assert v == 3

    def test_out_of_season_returns_zero(self):
        history = [(pd.Timestamp("2020-08-15"), True, "k1")]
        # Match in February → out of season
        v = _query_clay_season_cum(history, pd.Timestamp("2020-02-15"),
                                     "k_self", 4, 10)
        assert v == 0

    def test_pre_season_clay_not_counted(self):
        # March clay matches (Bogotá) shouldn't count toward main season
        history = [
            (pd.Timestamp("2020-02-20"), True, "k1"),  # Bogotá clay
            (pd.Timestamp("2020-03-10"), True, "k2"),  # Bogotá clay
            (pd.Timestamp("2020-04-15"), True, "k3"),  # Charleston (in season)
        ]
        v = _query_clay_season_cum(history, pd.Timestamp("2020-05-15"),
                                     "k_self", 4, 10)
        # Only k3 is in [Apr 1, May 15) → 1
        assert v == 1

    def test_year_isolation(self):
        # Prior year's clay shouldn't count
        history = [
            (pd.Timestamp("2019-08-15"), True, "k1"),  # prev year
            (pd.Timestamp("2020-04-10"), True, "k2"),  # this year
        ]
        v = _query_clay_season_cum(history, pd.Timestamp("2020-05-15"),
                                     "k_self", 4, 10)
        assert v == 1


# ---------------------------------------------------------------------------
# End-to-end on synthetic data
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_basic_match_with_prior_history(self):
        # Player 100: 1 hard match April 30; player 200: no prior history
        sk_rows = [
            _sackmann_row(winner_id=100, loser_id=999,
                          date="2020-04-30", tourney_id="2020-EARLY",
                          surface="Hard"),
            _sackmann_row(winner_id=100, loser_id=200,
                          date="2020-05-15", tourney_id="2020-MAIN",
                          surface="Clay"),
        ]
        master = pd.DataFrame([_master_row(
            1, p1=100, p2=200, date="2020-05-15", tourney_id="2020-MAIN",
        )])
        sk = pd.DataFrame(sk_rows)
        result = compute_fatigue_features(master, sk, run_sanity_checks=False)
        f = result.features.iloc[0]
        # p1 last played April 30 → 15 days rest
        assert f["rest_days_p1"] == 15.0
        # p2 has no prior → NaN
        assert pd.isna(f["rest_days_p2"])
        assert pd.isna(f["rest_days_diff"])
        # No matches this week for either
        assert f["matches_this_week_p1"] == 0
        assert f["matches_this_week_p2"] == 0
        # p1 played Hard April 30 → not counted in clay_season (hard)
        assert f["clay_season_cum_p1"] == 0
        assert f["clay_season_cum_p2"] == 0

    def test_busy_week_increments_count(self):
        # Player 100 plays Tue + Wed + Fri in same ISO week
        sk_rows = [
            _sackmann_row(winner_id=100, loser_id=300, date="2020-05-12",
                          tourney_id="2020-T1"),
            _sackmann_row(winner_id=100, loser_id=400, date="2020-05-13",
                          tourney_id="2020-T2"),
            _sackmann_row(winner_id=100, loser_id=200, date="2020-05-15",
                          tourney_id="2020-MAIN"),
        ]
        master = pd.DataFrame([_master_row(
            2, p1=100, p2=200, date="2020-05-15", tourney_id="2020-MAIN",
        )])
        sk = pd.DataFrame(sk_rows)
        result = compute_fatigue_features(master, sk, run_sanity_checks=False)
        f = result.features.iloc[0]
        # Player 100 has 2 prior matches in the same ISO week (Tue + Wed)
        assert f["matches_this_week_p1"] == 2
        # Player 200 has no prior matches
        assert f["matches_this_week_p2"] == 0
        # matches_this_week_diff = p2 - p1 = 0 - 2 = -2 (p1 more loaded)
        assert f["matches_this_week_diff"] == -2

    def test_clay_season_cum_accumulates(self):
        # Player 100 plays 3 prior clay matches in the season
        sk_rows = []
        for i, day in enumerate(["2020-04-05", "2020-04-15", "2020-04-25"]):
            sk_rows.append(_sackmann_row(
                winner_id=100, loser_id=300 + i, date=day,
                tourney_id=f"2020-T{i:03d}", surface="Clay",
            ))
        sk_rows.append(_sackmann_row(
            winner_id=100, loser_id=200, date="2020-05-15",
            tourney_id="2020-MAIN", surface="Clay",
        ))
        master = pd.DataFrame([_master_row(
            3, p1=100, p2=200, date="2020-05-15", tourney_id="2020-MAIN",
        )])
        sk = pd.DataFrame(sk_rows)
        result = compute_fatigue_features(master, sk, run_sanity_checks=False)
        f = result.features.iloc[0]
        assert f["clay_season_cum_p1"] == 3
        assert f["clay_season_cum_p2"] == 0
        # diff = p2 - p1 = -3 (p1 has more clay-season wear)
        assert f["clay_season_cum_diff"] == -3

    def test_diff_signs_match_convention(self):
        # rest_days_diff = p1 - p2 (positive = p1 more rested)
        sk_rows = [
            _sackmann_row(winner_id=100, loser_id=999, date="2020-04-01",
                          tourney_id="2020-T1"),    # p1 rested 44 days
            _sackmann_row(winner_id=200, loser_id=998, date="2020-05-10",
                          tourney_id="2020-T2"),    # p2 rested 5 days
            _sackmann_row(winner_id=100, loser_id=200, date="2020-05-15",
                          tourney_id="2020-MAIN"),
        ]
        master = pd.DataFrame([_master_row(
            4, p1=100, p2=200, date="2020-05-15", tourney_id="2020-MAIN",
        )])
        sk = pd.DataFrame(sk_rows)
        result = compute_fatigue_features(master, sk, run_sanity_checks=False)
        f = result.features.iloc[0]
        # p1 rest = 44, p2 rest = 5 → diff = +39 (p1 advantage)
        assert f["rest_days_p1"] == 44.0
        assert f["rest_days_p2"] == 5.0
        assert f["rest_days_diff"] == 39.0


# ---------------------------------------------------------------------------
# Empty / output schema
# ---------------------------------------------------------------------------

class TestEmpty:
    def test_empty_master(self):
        sk = pd.DataFrame([_sackmann_row(winner_id=100, loser_id=200)])
        result = compute_fatigue_features(pd.DataFrame(), sk,
                                            run_sanity_checks=False)
        assert result.features.empty


class TestOutputSchema:
    def test_columns_match_canonical_order(self):
        master = pd.DataFrame([_master_row(0)])
        sk = pd.DataFrame([_sackmann_row(winner_id=100, loser_id=200)])
        result = compute_fatigue_features(master, sk, run_sanity_checks=False)
        assert list(result.features.columns) == list(FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# Schema check
# ---------------------------------------------------------------------------

class TestSchema:
    def test_master_missing_stable_key_raises(self):
        master = pd.DataFrame([{
            "match_id": 0, "match_date": pd.Timestamp("2020-05-15"),
            "tourney_id": "T", "p1_id": 100, "p2_id": 200, "y": 1,
        }])
        sk = pd.DataFrame([_sackmann_row(winner_id=100, loser_id=200)])
        with pytest.raises(ValueError, match="stable_match_key"):
            compute_fatigue_features(master, sk, run_sanity_checks=False)
