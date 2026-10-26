"""Unit tests for wta_clay.data.merge."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wta_clay.data import merge as merge_mod
from wta_clay.data.merge import (
    HAND_CATEGORIES, MARKET_SOURCE_CATEGORIES, MASTER_COLUMNS,
    MergeResult, ROUND_ORDER, TOUR_LEVEL_ORDER,
    _RETIRED_RE, _resolve_market_source,
    build_master_dataset, load_master,
)


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------

def _make_sackmann_row(idx, *,
                      tourney_id="2020-001", tourney_name="Test Open",
                      match_date="2020-05-01", match_num=1,
                      winner_id, winner_name="WName",
                      loser_id, loser_name="LName",
                      tour_level="WTA250", sample_weight=1.0,
                      score="6-4 6-2", best_of=3, minutes=90,
                      winner_age=25.0, winner_hand="R", winner_ht=170,
                      winner_ioc="USA", winner_rank=10, winner_rank_points=2000,
                      loser_age=27.0, loser_hand="L", loser_ht=175,
                      loser_ioc="ESP", loser_rank=50, loser_rank_points=900,
                      not_for_betting=False):
    return {
        "tourney_id": tourney_id, "tourney_name": tourney_name,
        "tour_level": tour_level, "match_date": pd.Timestamp(match_date),
        "score": score, "best_of": best_of, "minutes": minutes,
        "winner_id": winner_id, "winner_name": winner_name,
        "winner_age": winner_age, "winner_hand": winner_hand,
        "winner_ht": winner_ht, "winner_ioc": winner_ioc,
        "winner_rank": winner_rank, "winner_rank_points": winner_rank_points,
        "loser_id": loser_id, "loser_name": loser_name,
        "loser_age": loser_age, "loser_hand": loser_hand,
        "loser_ht": loser_ht, "loser_ioc": loser_ioc,
        "loser_rank": loser_rank, "loser_rank_points": loser_rank_points,
        "sample_weight": sample_weight, "not_for_betting": not_for_betting,
    }


def _make_matched_row(sackmann_idx, *,
                      tournament="Test Open", match_date="2020-05-01",
                      winner_name_raw="W.", loser_name_raw="L.",
                      psw=1.5, psl=2.5,
                      b365w=1.55, b365l=2.45,
                      avgw=1.52, avgl=2.48,
                      maxw=1.58, maxl=2.55,
                      surface="Clay", round_norm="R32",
                      source="2020.csv",
                      match_path="primary", match_score=99.0,
                      winner_mismatch=False):
    return {
        "tournament": tournament, "surface": surface,
        "round_norm": round_norm,
        "match_date": pd.Timestamp(match_date),
        "winner_name_raw": winner_name_raw,
        "loser_name_raw":  loser_name_raw,
        "psw": psw, "psl": psl,
        "b365w": b365w, "b365l": b365l,
        "avgw": avgw, "avgl": avgl,
        "maxw": maxw, "maxl": maxl,
        "_source_file": source,
        "sackmann_match_id": sackmann_idx,
        "match_path": match_path, "match_score": match_score,
        "winner_mismatch": winner_mismatch,
    }


# ---------------------------------------------------------------------------
# Constants and regex
# ---------------------------------------------------------------------------

class TestConstants:
    def test_master_columns_count(self):
        # Schema lock — 13 metadata (incl. stable_match_key) + 9 p1 + 9 p2 +
        # y + 9 market + 2 weights + 4 audit = 47 columns.
        assert len(MASTER_COLUMNS) == 47

    def test_stable_match_key_in_columns(self):
        assert "stable_match_key" in MASTER_COLUMNS
        # Required keys present
        for c in ("match_id", "match_date", "tourney_date", "p1_id", "p2_id",
                  "y", "market_source", "retired", "sample_weight",
                  "not_for_betting", "score"):
            assert c in MASTER_COLUMNS

    def test_retired_regex(self):
        assert _RETIRED_RE.search("6-3 RET")
        assert _RETIRED_RE.search("W/O")
        assert _RETIRED_RE.search("6-3 0-0 ret")     # case-insensitive
        assert _RETIRED_RE.search("Walkover")
        assert not _RETIRED_RE.search("6-4 6-2")
        assert not _RETIRED_RE.search("")


# ---------------------------------------------------------------------------
# Market source resolution
# ---------------------------------------------------------------------------

class TestResolveMarketSource:
    def test_pinnacle_priority(self):
        df = pd.DataFrame({
            "psw": [1.5, np.nan, np.nan],
            "psl": [2.5, np.nan, np.nan],
            "b365w": [1.55, 1.6, np.nan],
            "b365l": [2.45, 2.4, np.nan],
        })
        out = _resolve_market_source(df).tolist()
        assert out == ["pinnacle", "bet365", "none"]

    def test_partial_pinnacle_falls_through_to_b365(self):
        # Pinnacle requires BOTH sides; if only one is present, fall through.
        df = pd.DataFrame({
            "psw": [1.5], "psl": [np.nan],
            "b365w": [1.55], "b365l": [2.45],
        })
        assert _resolve_market_source(df).tolist() == ["bet365"]


# ---------------------------------------------------------------------------
# End-to-end: p1/p2 ordering + odds reorientation
# ---------------------------------------------------------------------------

class TestStableMatchKey:
    def test_format_is_tourney_p1_p2(self):
        sackmann = pd.DataFrame([
            _make_sackmann_row(0, winner_id=100, loser_id=200,
                              tourney_id="2020-001"),
        ])
        matched = pd.DataFrame([_make_matched_row(0)])
        result = build_master_dataset(matched, sackmann)
        # p1=100 (smaller), p2=200, tourney_id="2020-001"
        assert result.master.iloc[0]["stable_match_key"] == "2020-001_100_200"

    def test_independent_of_winner_loser_orientation(self):
        # Same physical match (Charleston 2020 between players 100 and 200)
        # whether 100 won or 200 won — stable key should be identical.
        sk_100_won = pd.DataFrame([
            _make_sackmann_row(0, winner_id=100, loser_id=200,
                              tourney_id="2020-001"),
        ])
        sk_200_won = pd.DataFrame([
            _make_sackmann_row(0, winner_id=200, loser_id=100,
                              tourney_id="2020-001"),
        ])
        m = pd.DataFrame([_make_matched_row(0)])
        r1 = build_master_dataset(m, sk_100_won)
        r2 = build_master_dataset(m, sk_200_won)
        assert (r1.master.iloc[0]["stable_match_key"] ==
                r2.master.iloc[0]["stable_match_key"])

    def test_uniqueness_self_check(self):
        # Two Sackmann rows with same (tourney, p1, p2) should be impossible
        # for the matcher to produce, but the self-check guards against it.
        # We construct a scenario where the matcher dedup MIGHT have failed
        # and verify the self-check catches it.
        sackmann = pd.DataFrame([
            _make_sackmann_row(i, winner_id=100, loser_id=200,
                              tourney_id="2020-001",
                              match_num=i+1)
            for i in range(2)   # two matches with same players in same tournament
        ])
        matched = pd.DataFrame([_make_matched_row(i) for i in range(2)])
        # Self-check should raise on stable_match_key duplicates
        with pytest.raises(AssertionError, match="stable_match_key has 1 duplicates"):
            build_master_dataset(matched, sackmann)


class TestPlayerOrderingAndOdds:
    def test_p1_is_smaller_id_winner(self):
        # winner_id=100 < loser_id=200 → p1 = winner side; y = 1
        sackmann = pd.DataFrame([
            _make_sackmann_row(0, winner_id=100, loser_id=200,
                              winner_name="Alice A", loser_name="Bob B"),
        ])
        matched = pd.DataFrame([_make_matched_row(0)])
        result = build_master_dataset(matched, sackmann, run_self_checks=True)
        m = result.master
        assert m.iloc[0]["p1_id"] == 100
        assert m.iloc[0]["p2_id"] == 200
        assert m.iloc[0]["y"] == 1
        assert m.iloc[0]["odds_p1_pinnacle"] == 1.5    # winner-side
        assert m.iloc[0]["odds_p2_pinnacle"] == 2.5

    def test_p1_is_smaller_id_loser(self):
        # winner_id=200 > loser_id=100 → p1 = loser side; y = 0
        sackmann = pd.DataFrame([
            _make_sackmann_row(0, winner_id=200, loser_id=100,
                              winner_name="Alice A", loser_name="Bob B"),
        ])
        matched = pd.DataFrame([_make_matched_row(0)])
        result = build_master_dataset(matched, sackmann)
        m = result.master
        assert m.iloc[0]["p1_id"] == 100        # smaller id
        assert m.iloc[0]["p2_id"] == 200
        assert m.iloc[0]["y"] == 0              # p1 (loser side) lost
        # Odds reoriented: p1 was loser → p1 odds = original loser odds (psl)
        assert m.iloc[0]["odds_p1_pinnacle"] == 2.5
        assert m.iloc[0]["odds_p2_pinnacle"] == 1.5

    def test_strict_p1_lt_p2(self):
        sackmann = pd.DataFrame([
            _make_sackmann_row(i, winner_id=100 + i, loser_id=200 + i)
            for i in range(5)
        ])
        matched = pd.DataFrame([_make_matched_row(i) for i in range(5)])
        result = build_master_dataset(matched, sackmann)
        assert (result.master["p1_id"] < result.master["p2_id"]).all()


# ---------------------------------------------------------------------------
# Self-check assertions
# ---------------------------------------------------------------------------

class TestSelfChecks:
    def _baseline(self, n=3):
        sackmann = pd.DataFrame([
            _make_sackmann_row(i, winner_id=100 + i, loser_id=200 + i)
            for i in range(n)
        ])
        matched = pd.DataFrame([_make_matched_row(i) for i in range(n)])
        return matched, sackmann

    def test_baseline_passes_all_checks(self):
        matched, sackmann = self._baseline()
        result = build_master_dataset(matched, sackmann)
        # All checks passed → log has only ✓ entries (and possibly tourney
        # quali ⚠ if present; we have none here)
        assert all("[✓]" in line for line in result.self_check_log
                   if "[✓]" in line or "[⚠]" in line)

    def test_market_source_none_with_odds_raises(self):
        # Force a row with market='none' but partial b365 — actually this
        # would route to bet365. Instead test directly: build a master
        # frame where market_source is forcibly 'none' but pinnacle odds
        # are present. We bypass the builder to inject the bug.
        matched, sackmann = self._baseline(1)
        # Set b365 to NaN but keep pinnacle so pinnacle odds will populate
        # the master, but we'll force market_source='none' post-build.
        result = build_master_dataset(matched, sackmann)
        m = result.master.copy()
        m.loc[0, "market_source"] = "none"      # corrupt
        with pytest.raises(AssertionError, match="market_source='none' but odds present"):
            merge_mod._validate_master(m, 0.05, [])

    def test_match_id_duplicate_raises(self):
        matched, sackmann = self._baseline(2)
        result = build_master_dataset(matched, sackmann)
        m = result.master.copy()
        m.loc[1, "match_id"] = m.loc[0, "match_id"]   # duplicate
        with pytest.raises(AssertionError, match="match_id has 1 duplicates"):
            merge_mod._validate_master(m, 0.05, [])

    def test_y_nan_raises(self):
        matched, sackmann = self._baseline(1)
        result = build_master_dataset(matched, sackmann)
        m = result.master.copy()
        m.loc[0, "y"] = pd.NA
        with pytest.raises(AssertionError, match="y has 1 NaN"):
            merge_mod._validate_master(m, 0.05, [])

    def test_p1_id_not_lt_p2_id_raises(self):
        matched, sackmann = self._baseline(1)
        result = build_master_dataset(matched, sackmann)
        m = result.master.copy()
        # Swap so p1_id == p2_id
        m.loc[0, "p1_id"] = m.loc[0, "p2_id"]
        with pytest.raises(AssertionError, match="p1_id < p2_id"):
            merge_mod._validate_master(m, 0.05, [])

    def test_odds_single_sided_raises(self):
        matched, sackmann = self._baseline(1)
        result = build_master_dataset(matched, sackmann)
        m = result.master.copy()
        m.loc[0, "odds_p2_pinnacle"] = np.nan   # only p1 side
        with pytest.raises(AssertionError, match="Odds pairing"):
            merge_mod._validate_master(m, 0.05, [])

    def test_invalid_sample_weight_raises(self):
        matched, sackmann = self._baseline(1)
        result = build_master_dataset(matched, sackmann)
        m = result.master.copy()
        m.loc[0, "sample_weight"] = 0.75    # not in {0.25, 0.5, 1.0}
        with pytest.raises(AssertionError, match="sample_weight outside"):
            merge_mod._validate_master(m, 0.05, [])

    def test_index_not_range_raises(self):
        matched, sackmann = self._baseline(2)
        result = build_master_dataset(matched, sackmann)
        m = result.master.set_index(pd.Index([5, 7]))
        with pytest.raises(AssertionError, match="index is"):
            merge_mod._validate_master(m, 0.05, [])

    def test_retired_inconsistent_raises(self):
        matched, sackmann = self._baseline(1)
        result = build_master_dataset(matched, sackmann)
        m = result.master.copy()
        m.loc[0, "retired"] = True            # but score is "6-4 6-2"
        with pytest.raises(AssertionError, match="retired column inconsistent"):
            merge_mod._validate_master(m, 0.05, [])

    def test_rank_missing_above_threshold_warns_only(self):
        # Build with most ranks NaN
        sackmann = pd.DataFrame([
            _make_sackmann_row(0, winner_id=100, loser_id=200,
                              winner_rank=None, loser_rank=None),
            _make_sackmann_row(1, winner_id=101, loser_id=201,
                              winner_rank=None, loser_rank=None),
        ])
        matched = pd.DataFrame([_make_matched_row(0), _make_matched_row(1)])
        # Should not raise — rank missing 100% > 5% triggers WARN only
        result = build_master_dataset(matched, sackmann,
                                       rank_missing_warn_threshold=0.05)
        assert any("[⚠] p1_rank missing rate" in line
                   for line in result.self_check_log)


# ---------------------------------------------------------------------------
# Retired column derivation
# ---------------------------------------------------------------------------

class TestRetiredColumn:
    def test_retired_score_marked(self):
        sackmann = pd.DataFrame([
            _make_sackmann_row(0, winner_id=100, loser_id=200,
                              score="6-3 0-0 RET"),
        ])
        matched = pd.DataFrame([_make_matched_row(0)])
        result = build_master_dataset(matched, sackmann)
        assert bool(result.master.iloc[0]["retired"]) is True

    def test_walkover_marked(self):
        sackmann = pd.DataFrame([
            _make_sackmann_row(0, winner_id=100, loser_id=200, score="W/O"),
        ])
        matched = pd.DataFrame([_make_matched_row(0)])
        result = build_master_dataset(matched, sackmann)
        assert bool(result.master.iloc[0]["retired"]) is True

    def test_normal_score_not_marked(self):
        sackmann = pd.DataFrame([
            _make_sackmann_row(0, winner_id=100, loser_id=200,
                              score="6-4 7-6(3)"),
        ])
        matched = pd.DataFrame([_make_matched_row(0)])
        result = build_master_dataset(matched, sackmann)
        assert bool(result.master.iloc[0]["retired"]) is False


# ---------------------------------------------------------------------------
# tourney_date carry-through
# ---------------------------------------------------------------------------

class TestTourneyDate:
    def test_tourney_date_from_sackmann(self):
        sackmann = pd.DataFrame([
            _make_sackmann_row(0, winner_id=100, loser_id=200,
                              match_date="2020-05-04"),
        ])
        # TD says match was on 2020-05-08 (mid-tournament)
        matched = pd.DataFrame([_make_matched_row(0, match_date="2020-05-08")])
        result = build_master_dataset(matched, sackmann)
        assert result.master.iloc[0]["tourney_date"] == pd.Timestamp("2020-05-04")
        assert result.master.iloc[0]["match_date"] == pd.Timestamp("2020-05-08")


# ---------------------------------------------------------------------------
# Players DOB join
# ---------------------------------------------------------------------------

class TestPlayersDobJoin:
    def test_dob_pulled_from_players(self):
        sackmann = pd.DataFrame([
            _make_sackmann_row(0, winner_id=100, loser_id=200),
        ])
        matched = pd.DataFrame([_make_matched_row(0)])
        players = pd.DataFrame({
            "player_id": [100, 200],
            "dob": [pd.Timestamp("1995-01-01"), pd.Timestamp("1993-06-15")],
        })
        result = build_master_dataset(matched, sackmann, players=players)
        assert result.master.iloc[0]["p1_dob"] == pd.Timestamp("1995-01-01")
        assert result.master.iloc[0]["p2_dob"] == pd.Timestamp("1993-06-15")

    def test_no_players_dataframe_dob_is_nat(self):
        sackmann = pd.DataFrame([
            _make_sackmann_row(0, winner_id=100, loser_id=200),
        ])
        matched = pd.DataFrame([_make_matched_row(0)])
        result = build_master_dataset(matched, sackmann, players=None)
        assert pd.isna(result.master.iloc[0]["p1_dob"])
        assert pd.isna(result.master.iloc[0]["p2_dob"])


# ---------------------------------------------------------------------------
# Parquet round-trip
# ---------------------------------------------------------------------------

class TestParquetRoundtrip:
    def test_writes_and_reads_back_equal(self, tmp_path):
        sackmann = pd.DataFrame([
            _make_sackmann_row(i, winner_id=100 + i, loser_id=200 + i,
                              match_date=f"2020-05-{(i%4)+1:02d}")
            for i in range(5)
        ])
        matched = pd.DataFrame([_make_matched_row(i) for i in range(5)])
        path = tmp_path / "master.parquet"
        result = build_master_dataset(matched, sackmann, out_path=path)
        assert path.exists()
        loaded = load_master(path)
        # match_id order preserved
        assert list(loaded["match_id"]) == list(result.master["match_id"])

    def test_empty_input_returns_empty_result(self):
        result = build_master_dataset(pd.DataFrame(), pd.DataFrame())
        assert result.master.empty


# ---------------------------------------------------------------------------
# Sort stability
# ---------------------------------------------------------------------------

class TestSortStability:
    def test_sort_by_date_then_tourney(self):
        # Build out-of-order matches; sort should put earliest first.
        sackmann = pd.DataFrame([
            _make_sackmann_row(0, winner_id=100, loser_id=200,
                              match_date="2020-08-01", tourney_id="B"),
            _make_sackmann_row(1, winner_id=300, loser_id=400,
                              match_date="2020-04-01", tourney_id="A"),
            _make_sackmann_row(2, winner_id=500, loser_id=600,
                              match_date="2020-06-01", tourney_id="C"),
        ])
        matched = pd.DataFrame([
            _make_matched_row(0, match_date="2020-08-01"),
            _make_matched_row(1, match_date="2020-04-01"),
            _make_matched_row(2, match_date="2020-06-01"),
        ])
        result = build_master_dataset(matched, sackmann)
        dates = list(result.master["match_date"])
        assert dates == sorted(dates)
        # Index is RangeIndex(0..N-1)
        assert list(result.master.index) == [0, 1, 2]


# ---------------------------------------------------------------------------
# Schema report
# ---------------------------------------------------------------------------

class TestNotForBetting:
    def test_wta125_flagged_as_not_for_betting(self):
        # WTA125 rows must always have not_for_betting=True regardless of
        # Sackmann's per-row flag, per the locked Phase-1 decision.
        sackmann = pd.DataFrame([
            _make_sackmann_row(0, winner_id=100, loser_id=200,
                              tour_level="WTA125", sample_weight=0.5,
                              not_for_betting=False),  # Sackmann says betable
        ])
        matched = pd.DataFrame([_make_matched_row(0)])
        result = build_master_dataset(matched, sackmann)
        assert bool(result.master.iloc[0]["not_for_betting"]) is True

    def test_olympics_flagged_via_sackmann(self):
        # Sackmann sets not_for_betting=True for Olympics; merge respects it.
        sackmann = pd.DataFrame([
            _make_sackmann_row(0, winner_id=100, loser_id=200,
                              tour_level="WTA125", sample_weight=0.5,
                              not_for_betting=True),
        ])
        matched = pd.DataFrame([_make_matched_row(0)])
        result = build_master_dataset(matched, sackmann)
        assert bool(result.master.iloc[0]["not_for_betting"]) is True

    def test_wta250_main_draw_is_betable(self):
        sackmann = pd.DataFrame([
            _make_sackmann_row(0, winner_id=100, loser_id=200,
                              tour_level="WTA250", sample_weight=1.0),
        ])
        matched = pd.DataFrame([_make_matched_row(0)])
        result = build_master_dataset(matched, sackmann)
        assert bool(result.master.iloc[0]["not_for_betting"]) is False

    def test_low_sample_weight_implies_not_for_betting(self):
        # Defensive: if sample_weight is 0.25 (ITF), enforce not_for_betting
        # even if tour_level got corrupted to WTA250.
        sackmann = pd.DataFrame([
            _make_sackmann_row(0, winner_id=100, loser_id=200,
                              tour_level="WTA250", sample_weight=0.25),
        ])
        matched = pd.DataFrame([_make_matched_row(0)])
        # The sample_weight self-check would normally raise on 0.25 if
        # tour_level isn't in the allowed combination, but the rule check
        # is on the value alone — 0.25 is allowed.
        result = build_master_dataset(matched, sackmann)
        assert bool(result.master.iloc[0]["not_for_betting"]) is True


class TestDeduplication:
    def test_duplicate_sackmann_id_resolved(self):
        # Two TD rows both pointing to sackmann_match_id=0; higher score wins.
        sackmann = pd.DataFrame([
            _make_sackmann_row(0, winner_id=100, loser_id=200,
                              winner_name="Real Winner",
                              loser_name="Real Loser"),
        ])
        # First TD row: false-positive 85% match; second: perfect 100%
        matched = pd.DataFrame([
            _make_matched_row(0, winner_name_raw="Lookalike A.",
                              loser_name_raw="Lookalike B.", match_score=85.0),
            _make_matched_row(0, winner_name_raw="Real W.",
                              loser_name_raw="Real L.", match_score=100.0),
        ])
        result = build_master_dataset(matched, sackmann)
        # Only the higher-scored row survives in master
        assert len(result.master) == 1
        assert result.master.iloc[0]["match_score"] == 100.0
        # The lower-scored row is recorded in duplicate_rejected
        assert len(result.duplicate_rejected) == 1
        assert result.duplicate_rejected.iloc[0]["match_score"] == 85.0


class TestSchemaReport:
    def test_columns_in_report(self):
        sackmann = pd.DataFrame([
            _make_sackmann_row(0, winner_id=100, loser_id=200),
        ])
        matched = pd.DataFrame([_make_matched_row(0)])
        result = build_master_dataset(matched, sackmann)
        rep = result.schema_report
        assert {"column", "dtype", "missing_%"}.issubset(rep.columns)
        assert "match_id" in set(rep["column"])
        assert "y" in set(rep["column"])
