"""Unit tests for wta_clay.features.specialist."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wta_clay.features import specialist
from wta_clay.features.specialist import (
    FEATURE_COLUMNS, MIN_CLAY_MATCHES, MIN_TOTAL_MATCHES,
    SpecialistResult, _query_player_history, _specialist_index,
    compute_specialist_features,
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
    def test_thresholds(self):
        assert MIN_TOTAL_MATCHES == 30
        assert MIN_CLAY_MATCHES == 5

    def test_feature_columns(self):
        for c in ("match_id", "specialist_p1", "specialist_p2",
                  "specialist_diff",
                  "specialist_n_clay_p1", "specialist_n_total_p1",
                  "specialist_n_clay_p2", "specialist_n_total_p2"):
            assert c in FEATURE_COLUMNS
        assert len(FEATURE_COLUMNS) == 8


# ---------------------------------------------------------------------------
# Schema check
# ---------------------------------------------------------------------------

class TestSchema:
    def test_missing_sackmann_surface_raises(self):
        master = pd.DataFrame([_master_row(0)])
        sk = pd.DataFrame([{
            "tourney_id": "T", "match_date": pd.Timestamp("2020-05-01"),
            "winner_id": 100, "loser_id": 200, "round": "R32",
        }])
        with pytest.raises(ValueError, match="surface"):
            compute_specialist_features(master, sk, run_sanity_checks=False)

    def test_missing_master_stable_key_raises(self):
        master = pd.DataFrame([{
            "match_id": 0, "match_date": pd.Timestamp("2020-05-01"),
            "tourney_id": "T", "p1_id": 100, "p2_id": 200, "y": 1,
        }])
        sk = pd.DataFrame([_sackmann_row(winner_id=100, loser_id=200)])
        with pytest.raises(ValueError, match="stable_match_key"):
            compute_specialist_features(master, sk, run_sanity_checks=False)


# ---------------------------------------------------------------------------
# Specialist formula — semantic lock
# ---------------------------------------------------------------------------

class TestSpecialistFormula:
    def test_locked_example_8_clay_6_hard(self):
        # Player A: 8 wins / 10 clay matches, 6 wins / 10 hard matches.
        # overall_winrate = 14/20 = 0.70; clay_winrate = 8/10 = 0.80
        # specialist = 0.80 - 0.70 = +0.10
        result = _specialist_index(
            n_clay=10, wins_clay=8, n_total=20, wins_total=14,
            min_total=15, min_clay=5,
        )
        assert abs(result - 0.10) < 1e-9

    def test_overall_includes_all_surfaces(self):
        # Player B: 5 wins / 10 clay, 7 wins / 10 hard, 4 wins / 10 grass
        # overall = 16/30; clay = 5/10 = 0.5
        # specialist = 0.5 - 16/30 = 0.5 - 0.5333 = -0.0333
        result = _specialist_index(
            n_clay=10, wins_clay=5, n_total=30, wins_total=16,
            min_total=15, min_clay=5,
        )
        assert abs(result - (0.5 - 16/30)) < 1e-9
        assert result < 0   # slightly below average on clay

    def test_below_min_total_returns_nan(self):
        result = _specialist_index(
            n_clay=8, wins_clay=8, n_total=20, wins_total=14,
            min_total=30, min_clay=5,
        )
        assert np.isnan(result)

    def test_below_min_clay_returns_nan(self):
        result = _specialist_index(
            n_clay=3, wins_clay=3, n_total=50, wins_total=30,
            min_total=30, min_clay=5,
        )
        assert np.isnan(result)


# ---------------------------------------------------------------------------
# Strict-< filter
# ---------------------------------------------------------------------------

class TestStrictDateFilter:
    def test_excludes_same_day(self):
        history = [
            (pd.Timestamp("2020-05-01"), 1, True,  "k1"),  # same day — excluded
            (pd.Timestamp("2020-04-30"), 1, True,  "k2"),
            (pd.Timestamp("2020-04-29"), 0, False, "k3"),
        ]
        n_c, w_c, n_t, w_t = _query_player_history(
            history, pd.Timestamp("2020-05-01"), "k_cur",
        )
        assert n_t == 2     # k2 + k3
        assert n_c == 1     # only k2
        assert w_c == 1     # k2 won
        assert w_t == 1     # k2 won, k3 lost

    def test_excludes_self_via_key(self):
        history = [
            (pd.Timestamp("2020-05-01"), 1, True, "k_self"),
            (pd.Timestamp("2020-04-15"), 1, True, "k_other"),
        ]
        n_c, _, n_t, _ = _query_player_history(
            history, pd.Timestamp("2020-05-01"), "k_self",
        )
        # Self excluded by date-< AND key check; only k_other remains
        assert n_t == 1


# ---------------------------------------------------------------------------
# Rolling vs cumulative
# ---------------------------------------------------------------------------

class TestRollingWindow:
    def test_cumulative_uses_all(self):
        history = [
            (pd.Timestamp(f"2020-04-{d:02d}"), 1, True, f"k{d}")
            for d in range(1, 21)
        ]
        n_c, w_c, n_t, w_t = _query_player_history(
            history, pd.Timestamp("2020-05-01"), "k_cur",
            rolling_window=None,
        )
        assert n_t == 20
        assert n_c == 20

    def test_rolling_caps_at_window(self):
        history = [
            (pd.Timestamp(f"2020-04-{d:02d}"), 1, True, f"k{d}")
            for d in range(1, 21)
        ]
        n_c, w_c, n_t, w_t = _query_player_history(
            history, pd.Timestamp("2020-05-01"), "k_cur",
            rolling_window=10,
        )
        assert n_t == 10


# ---------------------------------------------------------------------------
# End-to-end synthetic
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_below_threshold_returns_nan(self):
        # Only 1 prior match — well below threshold.
        sk = pd.DataFrame([
            _sackmann_row(winner_id=100, loser_id=200, date="2020-04-01"),
            _sackmann_row(winner_id=100, loser_id=200, date="2020-05-01",
                          tourney_id="2020-MAIN"),
        ])
        master = pd.DataFrame([_master_row(
            1, p1=100, p2=200, date="2020-05-01", tourney_id="2020-MAIN",
        )])
        result = compute_specialist_features(master, sk, run_sanity_checks=False)
        f = result.features.iloc[0]
        assert pd.isna(f["specialist_p1"])
        assert pd.isna(f["specialist_p2"])
        assert f["specialist_n_clay_p1"] == 1
        assert f["specialist_n_total_p1"] == 1

    def test_above_thresholds_emits_value(self):
        # Build 35 prior all-surface matches for player 100 across mixed surfaces:
        # 10 clay, 25 hard. Player 100 wins 8/10 clay, 15/25 hard (=23/35 overall).
        # specialist = 0.80 - 23/35 = 0.80 - 0.6571 = +0.1429
        sk_rows = []
        # 10 clay: player 100 wins 8 (win opp 200..207), loses 2 (to 208, 209)
        for i in range(10):
            day = i + 1
            if i < 8:
                sk_rows.append(_sackmann_row(
                    winner_id=100, loser_id=200 + i,
                    date=f"2019-04-{day:02d}",
                    tourney_id=f"2019-CLAY-{i}", surface="Clay",
                ))
            else:
                sk_rows.append(_sackmann_row(
                    winner_id=200 + i, loser_id=100,
                    date=f"2019-04-{day:02d}",
                    tourney_id=f"2019-CLAY-{i}", surface="Clay",
                ))
        # 25 hard: wins 15 (opp 300..314), loses 10 (315..324)
        for i in range(25):
            day = (i % 28) + 1
            month = 5 + i // 28
            if i < 15:
                sk_rows.append(_sackmann_row(
                    winner_id=100, loser_id=300 + i,
                    date=f"2019-{month:02d}-{day:02d}",
                    tourney_id=f"2019-HARD-{i}", surface="Hard",
                ))
            else:
                sk_rows.append(_sackmann_row(
                    winner_id=300 + i, loser_id=100,
                    date=f"2019-{month:02d}-{day:02d}",
                    tourney_id=f"2019-HARD-{i}", surface="Hard",
                ))
        # Master match in 2020-05-01 between player 100 and a fresh opponent 999
        # (opponent will have NaN specialist due to insufficient history).
        sk_rows.append(_sackmann_row(
            winner_id=100, loser_id=999, date="2020-05-01",
            tourney_id="2020-MAIN", surface="Clay",
        ))
        master = pd.DataFrame([_master_row(
            len(sk_rows) - 1, p1=100, p2=999, date="2020-05-01",
            tourney_id="2020-MAIN",
        )])
        sk = pd.DataFrame(sk_rows)
        result = compute_specialist_features(master, sk, run_sanity_checks=False)
        f = result.features.iloc[0]
        assert f["specialist_n_clay_p1"] == 10
        assert f["specialist_n_total_p1"] == 35
        # specialist = 0.80 - 23/35
        assert abs(f["specialist_p1"] - (0.80 - 23/35)) < 1e-9
        # p2 (id 999) has 0 prior matches → NaN
        assert pd.isna(f["specialist_p2"])
        assert pd.isna(f["specialist_diff"])


# ---------------------------------------------------------------------------
# Empty / output schema
# ---------------------------------------------------------------------------

class TestEmpty:
    def test_empty_master(self):
        sk = pd.DataFrame([_sackmann_row(winner_id=100, loser_id=200)])
        result = compute_specialist_features(pd.DataFrame(), sk,
                                              run_sanity_checks=False)
        assert result.features.empty


class TestOutputSchema:
    def test_columns_match_canonical_order(self):
        master = pd.DataFrame([_master_row(0)])
        sk = pd.DataFrame([_sackmann_row(winner_id=100, loser_id=200)])
        result = compute_specialist_features(master, sk,
                                              run_sanity_checks=False)
        assert list(result.features.columns) == list(FEATURE_COLUMNS)
