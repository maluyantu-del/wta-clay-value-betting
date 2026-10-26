"""Unit tests for wta_clay.features.h2h."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wta_clay.features import h2h
from wta_clay.features.h2h import (
    ALPHA, BETA, ELO_CORRELATION_WARN_THRESHOLD, FEATURE_COLUMNS,
    H2HResult, _interpret_correlation, _query_pair_history,
    compute_h2h_features,
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
    def test_alpha_beta(self):
        assert ALPHA == 5.0
        assert BETA == 5.0

    def test_warn_threshold(self):
        assert ELO_CORRELATION_WARN_THRESHOLD == 0.70

    def test_feature_columns(self):
        assert "match_id" in FEATURE_COLUMNS
        assert "h2h_n_clay" in FEATURE_COLUMNS
        assert "h2h_p1_wins" in FEATURE_COLUMNS
        assert "h2h_smoothed_ratio_p1" in FEATURE_COLUMNS
        assert "h2h_n_overall" in FEATURE_COLUMNS
        assert "h2h_p1_wins_overall" in FEATURE_COLUMNS
        assert "h2h_smoothed_ratio_overall" not in FEATURE_COLUMNS  # skipped


# ---------------------------------------------------------------------------
# Schema check
# ---------------------------------------------------------------------------

class TestSchemaValidation:
    def test_missing_sackmann_column_raises(self):
        master = pd.DataFrame([_master_row(0)])
        # Missing 'surface'
        sk = pd.DataFrame([{
            "tourney_id": "T", "match_date": pd.Timestamp("2020-05-01"),
            "winner_id": 100, "loser_id": 200, "round": "R32",
        }])
        with pytest.raises(ValueError, match="surface"):
            compute_h2h_features(master, sk, run_sanity_checks=False)

    def test_missing_master_column_raises(self):
        master = pd.DataFrame([{
            "match_id": 0, "match_date": pd.Timestamp("2020-05-01"),
            "tourney_id": "T", "p1_id": 100, "p2_id": 200, "y": 1,
            # Missing stable_match_key
        }])
        sk = pd.DataFrame([_sackmann_row(winner_id=100, loser_id=200)])
        with pytest.raises(ValueError, match="stable_match_key"):
            compute_h2h_features(master, sk, run_sanity_checks=False)


# ---------------------------------------------------------------------------
# Smoothing math
# ---------------------------------------------------------------------------

class TestSmoothingMath:
    def test_cold_start_returns_alpha_over_alpha_beta(self):
        # n=0 → smoothed = α/(α+β) = 5/10 = 0.5
        n_clay, p1_wins, n_ov, p1_wins_ov = _query_pair_history(
            [], pd.Timestamp("2020-05-01"), "key1",
        )
        smoothed = (p1_wins + ALPHA) / (n_clay + ALPHA + BETA)
        assert smoothed == 0.5

    def test_n10_with_7_wins(self):
        history = [
            (pd.Timestamp(f"2020-04-{d:02d}"),
             1 if d <= 7 else 0,
             "clay", f"key{d}")
            for d in range(1, 11)
        ]
        n_clay, p1_wins, n_ov, p1_wins_ov = _query_pair_history(
            history, pd.Timestamp("2020-05-01"), "key_self",
        )
        assert n_clay == 10
        assert p1_wins == 7
        smoothed = (7 + 5) / (10 + 5 + 5)
        assert smoothed == 0.6


# ---------------------------------------------------------------------------
# Strict-< filter
# ---------------------------------------------------------------------------

class TestStrictDateFilter:
    def test_excludes_same_day(self):
        # Same-day prior matches must be excluded
        history = [
            (pd.Timestamp("2020-05-01"), 1, "clay", "k1"),
            (pd.Timestamp("2020-05-01"), 1, "clay", "k2"),
            (pd.Timestamp("2020-04-30"), 1, "clay", "k3"),
        ]
        n_clay, _, _, _ = _query_pair_history(
            history, pd.Timestamp("2020-05-01"), "k_self",
        )
        # Only k3 (2020-04-30) included
        assert n_clay == 1

    def test_excludes_self_via_key(self):
        # Defensive key check — same date AND same key would be self
        history = [
            (pd.Timestamp("2020-05-01"), 1, "clay", "k_self"),
            (pd.Timestamp("2020-04-30"), 1, "clay", "k_other"),
        ]
        n_clay, _, _, _ = _query_pair_history(
            history, pd.Timestamp("2020-05-01"), "k_self",
        )
        # Self excluded by date<; k_other (earlier) included
        assert n_clay == 1


# ---------------------------------------------------------------------------
# Surface aggregation
# ---------------------------------------------------------------------------

class TestSurfaceAggregation:
    def test_clay_subset_of_overall(self):
        history = [
            (pd.Timestamp("2020-04-01"), 1, "hard",  "k1"),
            (pd.Timestamp("2020-04-05"), 0, "clay",  "k2"),
            (pd.Timestamp("2020-04-10"), 1, "clay",  "k3"),
            (pd.Timestamp("2020-04-15"), 1, "grass", "k4"),
        ]
        n_clay, p1_wins, n_ov, p1_wins_ov = _query_pair_history(
            history, pd.Timestamp("2020-05-01"), "k_self",
        )
        # Overall: 4 matches, 3 wins for p1
        assert n_ov == 4
        assert p1_wins_ov == 3
        # Clay only: 2 matches (k2, k3), 1 win
        assert n_clay == 2
        assert p1_wins == 1


# ---------------------------------------------------------------------------
# End-to-end on synthetic data
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_first_match_returns_cold_start(self):
        master = pd.DataFrame([_master_row(0, p1=100, p2=200)])
        sk = pd.DataFrame([_sackmann_row(winner_id=100, loser_id=200)])
        result = compute_h2h_features(master, sk, run_sanity_checks=False)
        f = result.features.iloc[0]
        assert f["h2h_n_clay"] == 0
        assert f["h2h_p1_wins"] == 0
        assert f["h2h_smoothed_ratio_p1"] == 0.5
        assert f["h2h_n_overall"] == 0
        assert f["h2h_p1_wins_overall"] == 0

    def test_master_with_3_prior_clay_wins(self):
        # 3 prior clay matches: p1 wins all → n_clay=3, p1_wins=3, smoothed=(3+5)/(3+10)=8/13
        sk_rows = [
            _sackmann_row(winner_id=100, loser_id=200,
                          date=f"2020-04-{d:02d}",
                          tourney_id=f"2020-T{d:03d}")
            for d in (1, 8, 15)
        ]
        master_row = _master_row(99, p1=100, p2=200, y=1, date="2020-05-01",
                                  tourney_id="2020-MAIN")
        sk_rows.append(_sackmann_row(
            winner_id=100, loser_id=200, date="2020-05-01",
            tourney_id="2020-MAIN",
        ))
        master = pd.DataFrame([master_row])
        sk = pd.DataFrame(sk_rows)
        result = compute_h2h_features(master, sk, run_sanity_checks=False)
        f = result.features.iloc[0]
        assert f["h2h_n_clay"] == 3
        assert f["h2h_p1_wins"] == 3
        assert abs(f["h2h_smoothed_ratio_p1"] - 8/13) < 1e-9

    def test_overall_includes_non_clay(self):
        # 1 hard match + 2 clay matches as prior history
        sk_rows = [
            _sackmann_row(winner_id=100, loser_id=200, date="2020-04-01",
                          tourney_id="2020-T1", surface="Hard"),
            _sackmann_row(winner_id=100, loser_id=200, date="2020-04-15",
                          tourney_id="2020-T2", surface="Clay"),
            _sackmann_row(winner_id=200, loser_id=100, date="2020-04-22",
                          tourney_id="2020-T3", surface="Clay"),
        ]
        master_row = _master_row(99, p1=100, p2=200, y=1, date="2020-05-01",
                                  tourney_id="2020-MAIN")
        sk_rows.append(_sackmann_row(winner_id=100, loser_id=200,
                                      date="2020-05-01",
                                      tourney_id="2020-MAIN"))
        master = pd.DataFrame([master_row])
        sk = pd.DataFrame(sk_rows)
        result = compute_h2h_features(master, sk, run_sanity_checks=False)
        f = result.features.iloc[0]
        assert f["h2h_n_overall"] == 3
        assert f["h2h_p1_wins_overall"] == 2   # p1 won T1 and T2; T3 went to p2
        assert f["h2h_n_clay"] == 2
        assert f["h2h_p1_wins"] == 1


# ---------------------------------------------------------------------------
# Sanity checks (raise behaviour)
# ---------------------------------------------------------------------------

class TestSanityRaises:
    def _build_clean(self, n=20):
        sk_rows = []
        for i in range(n):
            sk_rows.append(_sackmann_row(
                winner_id=100, loser_id=200,
                date=pd.Timestamp("2020-01-01") + pd.Timedelta(days=i*5),
                tourney_id=f"2020-T{i:03d}",
            ))
        # Master = last 10 of those
        master_rows = []
        for i in range(10, 20):
            master_rows.append({
                "match_id": i,
                "match_date": sk_rows[i]["match_date"],
                "tourney_id": f"2020-T{i:03d}",
                "p1_id": 100, "p2_id": 200, "y": 1,
                "stable_match_key": f"2020-T{i:03d}_100_200",
            })
        return pd.DataFrame(master_rows), pd.DataFrame(sk_rows)

    def test_baseline_passes_all_checks(self):
        master, sk = self._build_clean()
        result = compute_h2h_features(master, sk, run_sanity_checks=True,
                                       leakage_sample_size=5)
        joined = "\n".join(result.sanity_log)
        assert "h2h_p1_wins ∈ [0, h2h_n_clay]" in joined
        assert "h2h_n_overall ≥ h2h_n_clay" in joined
        assert "smoothed_ratio ∈ [0, 1]" in joined
        assert "Cold-start (n_clay=0)" in joined
        assert "h2h_n_clay distribution" in joined
        assert "Leakage cross-check" in joined


# ---------------------------------------------------------------------------
# Correlation interpretation
# ---------------------------------------------------------------------------

class TestCorrelationInterp:
    def test_low_corr_recommends_add(self):
        msg = _interpret_correlation(0.30)
        assert "new signal" in msg.lower() or "add directly" in msg.lower()

    def test_mid_corr_recommends_ablation(self):
        msg = _interpret_correlation(0.60)
        assert "ablation" in msg.lower() or "auc" in msg.lower()

    def test_high_corr_warns(self):
        msg = _interpret_correlation(0.85)
        assert "skip" in msg.lower() or "high redundancy" in msg.lower()

    def test_negative_corr_treated_by_abs(self):
        # Negative r with same |r| should be interpreted similarly
        msg_neg = _interpret_correlation(-0.85)
        msg_pos = _interpret_correlation(0.85)
        assert msg_neg == msg_pos


# ---------------------------------------------------------------------------
# Empty / output schema
# ---------------------------------------------------------------------------

class TestEmpty:
    def test_empty_master(self):
        sk = pd.DataFrame([_sackmann_row(winner_id=100, loser_id=200)])
        result = compute_h2h_features(pd.DataFrame(), sk,
                                       run_sanity_checks=False)
        assert result.features.empty


class TestOutputSchema:
    def test_columns_match_canonical_order(self):
        master = pd.DataFrame([_master_row(0)])
        sk = pd.DataFrame([_sackmann_row(winner_id=100, loser_id=200)])
        result = compute_h2h_features(master, sk, run_sanity_checks=False)
        assert list(result.features.columns) == list(FEATURE_COLUMNS)
