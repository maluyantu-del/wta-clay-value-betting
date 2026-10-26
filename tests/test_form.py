"""Unit tests for wta_clay.features.form."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wta_clay.features import form
from wta_clay.features.form import (
    DOMINANCE_CAP, FEATURE_COLUMNS, MIN_MATCHES_FOR_VALUE, WINDOW_SIZE,
    FormResult, _build_unified_iteration_frame, _compute_dominance_arrays,
    _query_history, compute_form_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _master_row(match_id, *, p1=100, p2=200, y=1, date="2020-05-01",
                tourney_id="2020-001"):
    return {
        "match_id": match_id,
        "match_date": pd.Timestamp(date),
        "tourney_id": tourney_id,
        "p1_id": p1, "p2_id": p2, "y": y,
        "stable_match_key": f"{tourney_id}_{p1}_{p2}",
    }


def _sackmann_row(*, winner_id, loser_id, date="2020-05-01",
                  tourney_id="2020-001", round_="R32",
                  w_svpt=70, w_1stWon=40, w_2ndWon=15,
                  l_svpt=70, l_1stWon=30, l_2ndWon=10):
    return {
        "tourney_id": tourney_id, "round": round_,
        "match_date": pd.Timestamp(date),
        "winner_id": winner_id, "loser_id": loser_id,
        "w_svpt": w_svpt, "w_1stWon": w_1stWon, "w_2ndWon": w_2ndWon,
        "l_svpt": l_svpt, "l_1stWon": l_1stWon, "l_2ndWon": l_2ndWon,
    }


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_window_size_constant(self):
        assert WINDOW_SIZE == 15
        assert MIN_MATCHES_FOR_VALUE == 5
        assert DOMINANCE_CAP == 5.0

    def test_feature_columns(self):
        assert "match_id" in FEATURE_COLUMNS
        assert "form_winrate_p1" in FEATURE_COLUMNS
        assert "form_n_matches_p1" in FEATURE_COLUMNS
        assert "form_dominance_p1" in FEATURE_COLUMNS
        assert "form_winrate_diff" in FEATURE_COLUMNS
        assert "form_dominance_diff" in FEATURE_COLUMNS
        # No master_row_count — sidecar JSON
        assert "master_row_count" not in FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Dominance computation
# ---------------------------------------------------------------------------

class TestDominanceArrays:
    def test_basic_computation(self):
        sk = pd.DataFrame([{
            "w_svpt": 100, "w_1stWon": 50, "w_2ndWon": 20,
            "l_svpt": 100, "l_1stWon": 40, "l_2ndWon": 15,
        }])
        dr_w, dr_l = _compute_dominance_arrays(sk, cap=5.0)
        # winner: serve_won = 70/100 = 0.7; opp_serve_won = 55/100 = 0.55
        #         dr_w = (1-0.55)/(1-0.7) = 0.45/0.30 = 1.5
        assert abs(dr_w[0] - 1.5) < 1e-9
        # loser: serve_won = 0.55; opp_serve_won = 0.7
        #        dr_l = (1-0.7)/(1-0.55) = 0.30/0.45 = 0.6666...
        assert abs(dr_l[0] - 0.30/0.45) < 1e-9

    def test_cap_applied(self):
        # Winner held nearly all serve, lost almost none → DR explodes
        sk = pd.DataFrame([{
            "w_svpt": 100, "w_1stWon": 60, "w_2ndWon": 39,   # serve_won = 0.99
            "l_svpt": 100, "l_1stWon": 30, "l_2ndWon": 10,   # serve_won = 0.40
        }])
        dr_w, _ = _compute_dominance_arrays(sk, cap=5.0)
        # raw DR = (1-0.4) / (1-0.99) = 0.6/0.01 = 60 → cap 5.0
        assert dr_w[0] == 5.0

    def test_serve_won_one_returns_nan(self):
        # serve_won = 1.0 (perfect serve) → denom = 0 → NaN
        sk = pd.DataFrame([{
            "w_svpt": 100, "w_1stWon": 60, "w_2ndWon": 40,   # serve_won = 1.0
            "l_svpt": 100, "l_1stWon": 30, "l_2ndWon": 10,
        }])
        dr_w, _ = _compute_dominance_arrays(sk, cap=5.0)
        assert np.isnan(dr_w[0])

    def test_missing_stats_returns_nan(self):
        sk = pd.DataFrame([{
            "w_svpt": np.nan, "w_1stWon": 50, "w_2ndWon": 20,
            "l_svpt": 100, "l_1stWon": 40, "l_2ndWon": 15,
        }])
        dr_w, dr_l = _compute_dominance_arrays(sk, cap=5.0)
        assert np.isnan(dr_w[0])
        assert np.isnan(dr_l[0])

    def test_zero_svpt_returns_nan(self):
        sk = pd.DataFrame([{
            "w_svpt": 0, "w_1stWon": 0, "w_2ndWon": 0,
            "l_svpt": 100, "l_1stWon": 40, "l_2ndWon": 15,
        }])
        dr_w, dr_l = _compute_dominance_arrays(sk, cap=5.0)
        assert np.isnan(dr_w[0])
        assert np.isnan(dr_l[0])


# ---------------------------------------------------------------------------
# History query (cold start, strict date filter, window cap)
# ---------------------------------------------------------------------------

class TestQueryHistory:
    def test_cold_start_returns_nan(self):
        wr, n, dr = _query_history([], pd.Timestamp("2020-05-01"), 15, 5)
        assert np.isnan(wr)
        assert n == 0
        assert np.isnan(dr)

    def test_below_min_matches_returns_nan_winrate(self):
        history = [
            (pd.Timestamp("2020-04-01"), 1, 1.5),
            (pd.Timestamp("2020-04-15"), 0, 1.2),
            (pd.Timestamp("2020-04-20"), 1, np.nan),
        ]
        wr, n, dr = _query_history(history, pd.Timestamp("2020-05-01"), 15, 5)
        assert np.isnan(wr)
        assert n == 3
        assert np.isnan(dr)

    def test_strict_date_filter_excludes_same_day(self):
        # Same-day matches MUST NOT contribute to current match's history.
        history = [
            (pd.Timestamp("2020-04-30"), 1, 1.0),  # day before — included
            (pd.Timestamp("2020-05-01"), 0, 1.0),  # same day — EXCLUDED
        ] * 3   # repeat to satisfy min_matches
        wr, n, dr = _query_history(history, pd.Timestamp("2020-05-01"), 15, 5)
        # Only 3 same-day-before entries kept; need 5 → NaN
        assert n == 3
        assert np.isnan(wr)

        # Larger history with enough strictly-before entries
        history2 = [
            (pd.Timestamp(f"2020-04-{d:02d}"), 1, 1.0) for d in range(1, 11)
        ]
        wr, n, dr = _query_history(history2, pd.Timestamp("2020-05-01"), 15, 5)
        assert n == 10
        assert wr == 1.0   # all wins

    def test_window_caps_at_15(self):
        history = [
            (pd.Timestamp(f"2020-04-{d:02d}"), 1, 1.0) for d in range(1, 21)
        ]   # 20 wins
        wr, n, dr = _query_history(history, pd.Timestamp("2020-05-01"), 15, 5)
        assert n == 15
        assert wr == 1.0

    def test_winrate_with_mixed_outcomes(self):
        history = [
            (pd.Timestamp(f"2020-04-{d:02d}"), d % 2, 1.0)
            for d in range(1, 11)   # 10 matches alternating 1/0/1/0...
        ]
        wr, n, dr = _query_history(history, pd.Timestamp("2020-05-01"), 15, 5)
        assert n == 10
        # d % 2 for d in 1..10 = 1,0,1,0,1,0,1,0,1,0 → 5 wins
        assert wr == 0.5

    def test_dominance_drops_nan_entries(self):
        # 5 matches with 3 valid DR + 2 NaN → 3 valid samples ≥ 5? No, 3 < 5
        history = [
            (pd.Timestamp("2020-04-01"), 1, 1.0),
            (pd.Timestamp("2020-04-02"), 1, 2.0),
            (pd.Timestamp("2020-04-03"), 0, 1.5),
            (pd.Timestamp("2020-04-04"), 0, np.nan),
            (pd.Timestamp("2020-04-05"), 1, np.nan),
        ]
        wr, n, dr = _query_history(history, pd.Timestamp("2020-05-01"), 15, 5)
        assert n == 5
        assert wr == 0.6   # 3 wins of 5 = 0.6
        assert np.isnan(dr)   # only 3 valid DRs < 5

    def test_winrate_independent_of_dominance(self):
        # 5 matches all with NaN DR — winrate still computable
        history = [
            (pd.Timestamp(f"2020-04-{d:02d}"), 1, np.nan)
            for d in range(1, 6)
        ]
        wr, n, dr = _query_history(history, pd.Timestamp("2020-05-01"), 15, 5)
        assert n == 5
        assert wr == 1.0   # all wins
        assert np.isnan(dr)   # no valid DR samples


# ---------------------------------------------------------------------------
# End-to-end on synthetic data
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def _build(self, n_matches=10, dates_offset_days=1):
        """Build a master + sackmann pair where a player has many matches."""
        # Player 100 plays player 200, 300, 400, ... in sequence
        sackmann_rows = []
        master_rows = []
        for i in range(n_matches):
            opp = 200 + i
            date = pd.Timestamp("2020-04-01") + pd.Timedelta(
                days=dates_offset_days * i
            )
            sackmann_rows.append(_sackmann_row(
                winner_id=100, loser_id=opp, date=date,
                tourney_id=f"2020-T{i:03d}", round_="R32",
            ))
            # Last match goes into master
            if i == n_matches - 1:
                master_rows.append({
                    "match_id": i,
                    "match_date": date,
                    "tourney_id": f"2020-T{i:03d}",
                    "p1_id": 100, "p2_id": opp, "y": 1,
                    "stable_match_key": f"2020-T{i:03d}_100_{opp}",
                })
        return pd.DataFrame(master_rows), pd.DataFrame(sackmann_rows)

    def test_player_with_5_prior_wins(self):
        master, sackmann = self._build(n_matches=6)   # 5 prior wins + 1 master
        result = compute_form_features(master, sackmann,
                                        run_sanity_checks=False)
        assert len(result.features) == 1
        f = result.features.iloc[0]
        assert f["form_n_matches_p1"] == 5
        assert f["form_winrate_p1"] == 1.0   # 5 prior wins
        # p2 (opponent) has 0 matches before — cold start
        assert f["form_n_matches_p2"] == 0
        assert pd.isna(f["form_winrate_p2"])

    def test_n_matches_caps_at_window_size(self):
        master, sackmann = self._build(n_matches=20)   # 19 prior + 1 master
        result = compute_form_features(master, sackmann,
                                        run_sanity_checks=False)
        f = result.features.iloc[0]
        assert f["form_n_matches_p1"] == 15

    def test_below_min_matches_winrate_is_nan(self):
        master, sackmann = self._build(n_matches=5)   # 4 prior wins (< 5)
        result = compute_form_features(master, sackmann,
                                        run_sanity_checks=False)
        f = result.features.iloc[0]
        assert f["form_n_matches_p1"] == 4
        assert pd.isna(f["form_winrate_p1"])

    def test_strict_same_day_exclusion(self):
        # 6 matches all on the same day — last match's history filter
        # should EXCLUDE all preceding same-day entries.
        sackmann_rows = []
        master_rows = []
        for i in range(6):
            opp = 200 + i
            sackmann_rows.append(_sackmann_row(
                winner_id=100, loser_id=opp, date="2020-05-01",
                tourney_id=f"2020-T{i:03d}",
            ))
            if i == 5:
                master_rows.append({
                    "match_id": i,
                    "match_date": pd.Timestamp("2020-05-01"),
                    "tourney_id": f"2020-T{i:03d}",
                    "p1_id": 100, "p2_id": opp, "y": 1,
                    "stable_match_key": f"2020-T{i:03d}_100_{opp}",
                })
        master = pd.DataFrame(master_rows)
        sackmann = pd.DataFrame(sackmann_rows)
        result = compute_form_features(master, sackmann,
                                        run_sanity_checks=False)
        f = result.features.iloc[0]
        # All 5 prior matches were on the SAME date → strict < excludes them
        assert f["form_n_matches_p1"] == 0


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

class TestSanityChecks:
    def _build_clean(self, n=30):
        """Build a clean dataset with rotating outcomes and many players."""
        sk_rows = []
        for i in range(n):
            sk_rows.append(_sackmann_row(
                winner_id=100, loser_id=200 + i,
                date=pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
                tourney_id=f"2020-T{i:03d}",
            ))
        master_rows = []
        # Master = the last 10 matches (so history exists)
        for i in range(20, 30):
            master_rows.append({
                "match_id": i, "match_date": sk_rows[i]["match_date"],
                "tourney_id": f"2020-T{i:03d}",
                "p1_id": 100, "p2_id": 200 + i, "y": 1,
                "stable_match_key": f"2020-T{i:03d}_100_{200+i}",
            })
        return pd.DataFrame(master_rows), pd.DataFrame(sk_rows)

    def test_sanity_log_populated(self):
        master, sackmann = self._build_clean(50)
        result = compute_form_features(master, sackmann,
                                        run_sanity_checks=True,
                                        leakage_sample_size=5)
        # Required ✓ entries should appear
        joined = "\n".join(result.sanity_log)
        assert "form_n_matches in [0, 15]" in joined
        assert "Cold-start NaN" in joined
        assert "form_winrate in [0, 1]" in joined
        assert "Leakage cross-check" in joined
        assert "Winrate independent of dominance" in joined

    def test_leakage_check_passes(self):
        master, sackmann = self._build_clean(50)
        # Should NOT raise
        compute_form_features(master, sackmann, run_sanity_checks=True,
                              leakage_sample_size=5)


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------

class TestEmpty:
    def test_empty_master(self):
        result = compute_form_features(
            pd.DataFrame(), pd.DataFrame(), run_sanity_checks=False,
        )
        assert result.features.empty


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class TestOutputSchema:
    def test_columns_match_canonical_order(self):
        sk = pd.DataFrame([
            _sackmann_row(winner_id=100, loser_id=200,
                          date=pd.Timestamp("2020-04-01") + pd.Timedelta(days=i),
                          tourney_id=f"T{i}")
            for i in range(6)
        ])
        master = pd.DataFrame([{
            "match_id": 5,
            "match_date": pd.Timestamp("2020-04-06"),
            "tourney_id": "T5",
            "p1_id": 100, "p2_id": 200, "y": 1,
            "stable_match_key": "T5_100_200",
        }])
        result = compute_form_features(master, sk, run_sanity_checks=False)
        assert list(result.features.columns) == list(FEATURE_COLUMNS)
