"""Unit tests for wta_clay.data.name_matcher.

All fixtures are synthetic; no real data downloads required.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wta_clay.data import name_matcher as nm
from wta_clay.data.name_matcher import (
    AliasHit, MatchResult, PairMatchResult, TierMismatchError,
    _check_alias_substring_conflicts, _lookup_alias, _normalize_name,
    _pair_score, _score_histogram, _validate_tier,
    build_sackmann_tourney_index, compute_unmatched_audit,
    find_match_candidates_fallback, find_tourney_candidates_primary,
    load_alias, match_player_pair, match_td_to_sackmann,
    write_manual_review,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _alias_df(rows):
    return pd.DataFrame(rows, columns=[
        "td_pattern", "sackmann_name", "year_from", "year_to",
        "tier_expected", "notes",
    ]).astype({
        "year_from": "Int64", "year_to": "Int64",
        "td_pattern": "string", "sackmann_name": "string",
        "tier_expected": "string", "notes": "string",
    })


def _sackmann_match(idx, tourney_id, tourney_name, match_date,
                    winner_name, loser_name, tour_level="WTA250",
                    winner_id=100, loser_id=200,
                    winner_rank=10, loser_rank=50):
    return {
        "tourney_id":   tourney_id,
        "tourney_name": tourney_name,
        "match_date":   pd.Timestamp(match_date),
        "winner_name":  winner_name,
        "loser_name":   loser_name,
        "winner_id":    winner_id,
        "loser_id":     loser_id,
        "winner_rank":  winner_rank,
        "loser_rank":   loser_rank,
        "tour_level":   tour_level,
    }


def _td_row(match_date, tournament, winner, loser,
            psw=1.6, psl=2.4, source="2020.csv"):
    return {
        "match_date":      pd.Timestamp(match_date),
        "tournament":      tournament,
        "surface":         "Clay",
        "round_raw":       "1st Round",
        "winner_name_raw": winner,
        "loser_name_raw":  loser,
        "winner_rank":     pd.NA,
        "loser_rank":      pd.NA,
        "psw":             psw,
        "psl":             psl,
        "_source_file":    source,
    }


# ---------------------------------------------------------------------------
# Section C — name normalization and pair fuzzy matching
# ---------------------------------------------------------------------------

class TestNormalizeName:
    def test_td_style_simple(self):
        assert _normalize_name("Williams S.") == "williams s"

    def test_sackmann_western_simple(self):
        assert _normalize_name("Serena Williams") == "williams s"

    def test_td_compound_surname(self):
        assert _normalize_name("Sorribes Tormo S.") == "sorribes tormo s"

    def test_sackmann_compound_surname(self):
        assert _normalize_name("Sara Sorribes Tormo") == "sorribes tormo s"

    def test_td_multi_initial_takes_first(self):
        assert _normalize_name("Hsieh S.W.") == "hsieh s"

    def test_diacritic_stripping(self):
        # "Świątek I." → "swiatek i"
        assert _normalize_name("Świątek I.") == "swiatek i"
        # "Iga Świątek" → "swiatek i"
        assert _normalize_name("Iga Świątek") == "swiatek i"

    def test_hyphenated_first_name(self):
        # Hyphen is replaced with space → "su wei hsieh" → no period →
        # Western-style → first token 'su' = initial 's', remainder is
        # the surname. Result is "wei hsieh s". Matching against TD form
        # "Hsieh S.W." → "hsieh s" via token_set_ratio still scores ≥85.
        assert _normalize_name("Su-Wei Hsieh") == "wei hsieh s"

    def test_hyphenated_surname_td_split(self):
        # TD's "Lucic-Baroni M." should split hyphen so token_set_ratio
        # can match Sackmann's "Mirjana Lucic" via subset.
        assert _normalize_name("Lucic-Baroni M.") == "lucic baroni m"

    def test_empty_and_none(self):
        assert _normalize_name("") == ""
        assert _normalize_name(None) == ""
        assert _normalize_name("   ") == ""

    def test_case_insensitive(self):
        assert _normalize_name("WILLIAMS S.") == "williams s"
        assert _normalize_name("serena williams") == "williams s"


class TestPairScore:
    def test_perfect_aligned_match(self):
        score, mismatch = _pair_score("williams s", "hampton j",
                                       "williams s", "hampton j")
        assert score == 100.0
        assert mismatch is False

    def test_perfect_swapped_match(self):
        # TD says winner=Williams; Sackmann says winner=Hampton.
        # Pair is the same, just direction swapped.
        score, mismatch = _pair_score("williams s", "hampton j",
                                       "hampton j", "williams s")
        assert score == 100.0
        assert mismatch is True

    def test_no_match_low_score(self):
        score, _ = _pair_score("foo a", "bar b",
                                "baz c", "qux d")
        # rapidfuzz.fuzz.ratio on these single-letter+space strings
        # gives ~20-40 (lots of incidental shared chars). Just make sure
        # it's well below the primary threshold of 85.
        assert score < 60


class TestMatchPlayerPair:
    def _pool(self, rows):
        return pd.DataFrame(rows)

    def test_single_perfect_match(self):
        pool = self._pool([
            _sackmann_match(0, "T-1", "Test", "2020-05-01",
                            "Serena Williams", "Jamie Hampton"),
        ])
        out = match_player_pair("Williams S.", "Hampton J.", pool)
        assert out.score == 100.0
        assert out.sackmann_match_idx == 0
        assert out.winner_mismatch is False
        assert out.ambiguous is False

    def test_winner_mismatch_flag(self):
        pool = self._pool([
            _sackmann_match(0, "T-1", "Test", "2020-05-01",
                            "Jamie Hampton", "Serena Williams"),
        ])
        out = match_player_pair("Williams S.", "Hampton J.", pool)
        assert out.score == 100.0
        assert out.winner_mismatch is True

    def test_picks_best_among_pool(self):
        pool = self._pool([
            _sackmann_match(0, "T-1", "Test", "2020-05-01",
                            "Aryna Sabalenka", "Elise Mertens"),
            _sackmann_match(1, "T-1", "Test", "2020-05-01",
                            "Serena Williams", "Jamie Hampton"),
            _sackmann_match(2, "T-1", "Test", "2020-05-01",
                            "Iga Swiatek", "Jelena Ostapenko"),
        ])
        out = match_player_pair("Williams S.", "Hampton J.", pool)
        assert out.sackmann_match_idx == 1
        assert out.score >= 95

    def test_ambiguous_marked_when_no_date_info(self):
        # Two Sackmann rows with identical (similar) pairs and no date diff
        pool = self._pool([
            _sackmann_match(0, "T-1", "Test", "2020-05-01",
                            "Serena Williams", "Jamie Hampton"),
            _sackmann_match(1, "T-1", "Test", "2020-05-01",
                            "Serena Williams", "Jamie Hampton"),
        ])
        out = match_player_pair("Williams S.", "Hampton J.", pool,
                                td_match_date=None)
        assert out.ambiguous is True

    def test_ambiguous_resolved_by_match_date(self):
        # Two equal-score candidates; tiebreak on date.
        pool = self._pool([
            _sackmann_match(0, "T-A", "Test", "2020-04-01",
                            "Serena Williams", "Jamie Hampton"),
            _sackmann_match(1, "T-B", "Test", "2020-05-15",
                            "Serena Williams", "Jamie Hampton"),
        ])
        out = match_player_pair("Williams S.", "Hampton J.", pool,
                                td_match_date=pd.Timestamp("2020-05-14"))
        # Should pick the second row (closer date)
        assert out.sackmann_match_idx == 1
        assert out.ambiguous is False

    def test_ambiguous_when_date_also_tied(self):
        # Two equal-score candidates with EQUAL date distance — must flag.
        pool = self._pool([
            _sackmann_match(0, "T-A", "Test", "2020-04-30",
                            "Serena Williams", "Jamie Hampton"),
            _sackmann_match(1, "T-B", "Test", "2020-05-02",
                            "Serena Williams", "Jamie Hampton"),
        ])
        out = match_player_pair("Williams S.", "Hampton J.", pool,
                                td_match_date=pd.Timestamp("2020-05-01"))
        # Both 1 day from TD date → ambiguous
        assert out.ambiguous is True

    def test_empty_pool(self):
        out = match_player_pair("Williams S.", "Hampton J.", pd.DataFrame())
        assert out.sackmann_match_idx is None
        assert out.score == 0.0


# ---------------------------------------------------------------------------
# Section A — alias loading and validation
# ---------------------------------------------------------------------------

class TestAliasLoading:
    def test_load_valid_alias(self, tmp_path):
        p = tmp_path / "alias.csv"
        p.write_text(
            "td_pattern,sackmann_name,year_from,year_to,tier_expected,notes\n"
            "# comment line ignored\n"
            "French Open,Roland Garros,,,GS,\n"
        )
        df = load_alias(p)
        assert len(df) == 1
        assert df.iloc[0]["td_pattern"] == "French Open"

    def test_load_invalid_tier_raises(self, tmp_path):
        p = tmp_path / "alias.csv"
        p.write_text(
            "td_pattern,sackmann_name,year_from,year_to,tier_expected,notes\n"
            "Foo,Bar,,,BOGUS,\n"
        )
        with pytest.raises(ValueError, match="invalid tier_expected"):
            load_alias(p)

    def test_substring_conflict_different_canonical_raises(self):
        # 'A' is substring of 'A B' — different sackmann_name + overlap → conflict
        df = _alias_df([
            ("A",   "FooEvent", 2020, 2025, "WTA250", ""),
            ("A B", "BarEvent", 2020, 2025, "WTA250", ""),
        ])
        with pytest.raises(ValueError, match="substring × year-overlap"):
            _check_alias_substring_conflicts(df)

    def test_substring_no_year_overlap_ok(self):
        df = _alias_df([
            ("A",   "FooEvent", 2010, 2015, "WTA250", ""),
            ("A B", "BarEvent", 2020, 2025, "WTA250", ""),
        ])
        _check_alias_substring_conflicts(df)  # should not raise

    def test_substring_same_canonical_and_tier_ok(self):
        # Same TD pattern with different year ranges (FCC tier split) — OK.
        df = _alias_df([
            ("Family Circle Cup", "Charleston", 2007, 2008, "WTA1000", ""),
            ("Family Circle Cup", "Charleston", 2009, 2019, "WTA500",  ""),
        ])
        _check_alias_substring_conflicts(df)  # OK; same pattern not flagged


class TestLookupAlias:
    def test_hit(self):
        df = _alias_df([
            ("Foo", "Bar", 2010, 2015, "WTA250", ""),
        ])
        hit = _lookup_alias("Foo", 2012, df)
        assert hit is not None
        assert hit.sackmann_name == "Bar"
        assert hit.tier_expected == "WTA250"

    def test_miss_year_out_of_range(self):
        df = _alias_df([("Foo", "Bar", 2010, 2015, "WTA250", "")])
        assert _lookup_alias("Foo", 2020, df) is None

    def test_miss_unknown_pattern(self):
        df = _alias_df([("Foo", "Bar", 2010, 2015, "WTA250", "")])
        assert _lookup_alias("Other", 2012, df) is None

    def test_year_split_picks_correct_row(self):
        # Family-Circle-Cup-style year-tier split.
        df = _alias_df([
            ("FCC", "Charleston", 2007, 2008, "WTA1000", ""),
            ("FCC", "Charleston", 2009, 2019, "WTA500", ""),
        ])
        a = _lookup_alias("FCC", 2008, df)
        assert a.tier_expected == "WTA1000"
        b = _lookup_alias("FCC", 2015, df)
        assert b.tier_expected == "WTA500"


# ---------------------------------------------------------------------------
# Section B — Sackmann index + candidate pools
# ---------------------------------------------------------------------------

class TestSackmannIndex:
    def test_basic_grouping(self):
        sm = pd.DataFrame([
            _sackmann_match(0, "T-1", "Charleston", "2020-04-05",
                            "Sloane Stephens", "Madison Keys"),
            _sackmann_match(1, "T-1", "Charleston", "2020-04-06",
                            "Coco Gauff", "Jessica Pegula"),
            _sackmann_match(2, "T-2", "Roland Garros", "2020-05-31",
                            "Iga Swiatek", "Sofia Kenin"),
        ])
        idx = build_sackmann_tourney_index(sm)
        assert ("charleston", 2020) in idx
        assert ("roland garros", 2020) in idx
        assert len(idx[("charleston", 2020)]) == 2


class TestCandidatesPrimary:
    def _index(self):
        sm = pd.DataFrame([
            _sackmann_match(0, "T-1", "Charleston", "2020-04-05",
                            "A", "B"),
            _sackmann_match(1, "T-1", "Charleston", "2020-04-05",
                            "C", "D"),
        ])
        return build_sackmann_tourney_index(sm)

    def test_in_window(self):
        idx = self._index()
        out, status = find_tourney_candidates_primary(
            "Charleston", 2020, pd.Timestamp("2020-04-04"), idx,
            window_days=7,
        )
        assert status == "in_window"
        assert len(out) == 2

    def test_sackmann_missing(self):
        idx = self._index()
        out, status = find_tourney_candidates_primary(
            "Acapulco", 2020, pd.Timestamp("2020-04-04"), idx,
        )
        assert status == "sackmann_missing"
        assert out.empty

    def test_date_window_miss(self):
        idx = self._index()
        out, status = find_tourney_candidates_primary(
            "Charleston", 2020, pd.Timestamp("2020-06-01"), idx,
            window_days=7,
        )
        assert status == "date_window_miss"
        assert len(out) == 2

    def test_case_insensitive_sackmann_name(self):
        idx = self._index()
        out, status = find_tourney_candidates_primary(
            "CHARLESTON", 2020, pd.Timestamp("2020-04-04"), idx,
        )
        assert status == "in_window"
        assert len(out) == 2


class TestCandidatesFallback:
    def test_filters_by_date_and_rank(self):
        sm = pd.DataFrame([
            _sackmann_match(0, "T", "X", "2020-05-01", "A", "B",
                            winner_rank=20, loser_rank=300),
            _sackmann_match(1, "T", "X", "2020-05-01", "C", "D",
                            winner_rank=900, loser_rank=950),  # both > 500
            _sackmann_match(2, "T", "X", "2020-05-15", "E", "F",  # date too far
                            winner_rank=5, loser_rank=10),
        ])
        td = pd.Series({"match_date": pd.Timestamp("2020-05-02")})
        out = find_match_candidates_fallback(td, sm, days=3, rank_top=500)
        # Row 0 kept (date ok, winner rank 20 ≤ 500); row 1 dropped; row 2 dropped (date)
        assert len(out) == 1
        assert out.iloc[0]["winner_name"] == "A"


# ---------------------------------------------------------------------------
# Section D — tier validation
# ---------------------------------------------------------------------------

class TestValidateTier:
    def test_match_returns_true(self):
        assert _validate_tier("WTA1000", "WTA1000",
                              td_event="Foo", td_year=2020,
                              allow_mismatch=False) is True

    def test_mismatch_raises_by_default(self):
        with pytest.raises(TierMismatchError):
            _validate_tier("WTA500", "WTA1000",
                           td_event="Foo", td_year=2020,
                           allow_mismatch=False)

    def test_allow_flag_returns_false(self):
        result = _validate_tier(
            "WTA500", "WTA1000",
            td_event="Foo", td_year=2020, allow_mismatch=True,
        )
        assert result is False


# ---------------------------------------------------------------------------
# Section E — main entry (end-to-end on synthetic data)
# ---------------------------------------------------------------------------

class TestMatchTdToSackmann:
    def _build(self):
        alias = _alias_df([
            ("French Open", "Roland Garros", 2020, 2020, "GS", ""),
            ("Mutua Madrid Open", "Madrid", 2020, 2020, "WTA1000", ""),
            ("Iasi Open", "Iasi", 2025, 2025, "WTA250",
             "expect sackmann_missing"),
        ])
        sm = pd.DataFrame([
            # Roland Garros 2020 — one match
            _sackmann_match(0, "2020-520", "Roland Garros", "2020-05-31",
                            "Iga Swiatek", "Sofia Kenin", tour_level="GS"),
            # Madrid 2020 — but TD will say WTA1000 expected; we'll set
            # Sackmann tier to WTA500 to test the tier-mismatch raise.
            _sackmann_match(1, "2020-1038", "Madrid", "2020-09-12",
                            "Sara Errani", "Karolina Pliskova",
                            tour_level="WTA500"),  # intentionally wrong
        ])
        td = pd.DataFrame([
            _td_row("2020-05-31", "French Open", "Swiatek I.", "Kenin S."),
            _td_row("2025-07-14", "Iasi Open", "Tomova V.", "Linette M."),
        ])
        return alias, sm, td

    def test_basic_match(self):
        alias, sm, td = self._build()
        # Drop the buggy Madrid row to focus on RG match
        td = td.iloc[[0]]
        result = match_td_to_sackmann(td, sm, alias)
        assert isinstance(result, MatchResult)
        assert len(result.matched) == 1
        assert result.matched.iloc[0]["match_path"] == "primary"
        assert result.matched.iloc[0]["match_score"] == 100.0
        # winner_mismatch may be a numpy bool — compare with bool(.)
        assert bool(result.matched.iloc[0]["winner_mismatch"]) is False
        assert result.matched.iloc[0]["sackmann_tourney_id"] == "2020-520"

    def test_sackmann_missing_routed(self):
        alias, sm, td = self._build()
        # Iasi Open 2025 — alias maps to Sackmann name 'Iasi' but no Sackmann
        # data → sackmann_missing
        td = td.iloc[[1]]
        result = match_td_to_sackmann(td, sm, alias)
        assert len(result.matched) == 0
        assert len(result.sackmann_missing) == 1
        assert result.sackmann_missing.iloc[0]["failure_mode"] == "sackmann_missing"

    def test_unaliased_routed_to_unmatched(self):
        alias, sm, td = self._build()
        td = pd.DataFrame([
            _td_row("2020-05-01", "Mystery Open", "X.", "Y."),
        ])
        result = match_td_to_sackmann(td, sm, alias)
        assert len(result.unmatched) == 1
        assert result.unmatched.iloc[0]["failure_mode"] == "unaliased"

    def test_tier_mismatch_raises_default(self):
        alias, sm, td = self._build()
        td = td.iloc[[0]].copy()
        # Mutate alias to declare WTA1000 for a Sackmann row tagged GS
        bad_alias = _alias_df([
            ("French Open", "Roland Garros", 2020, 2020, "WTA1000",
             "intentional bad tier"),
        ])
        with pytest.raises(TierMismatchError):
            match_td_to_sackmann(td, sm, bad_alias)

    def test_tier_mismatch_allow_flag_continues(self):
        alias, sm, td = self._build()
        td = td.iloc[[0]].copy()
        bad_alias = _alias_df([
            ("French Open", "Roland Garros", 2020, 2020, "WTA1000",
             "intentional bad tier"),
        ])
        result = match_td_to_sackmann(td, sm, bad_alias,
                                       allow_tier_mismatch=True)
        # Match still happens (tier doesn't block), but tier_mismatches log
        # has the entry and manual_review captures it.
        assert len(result.matched) == 1
        assert len(result.tier_mismatches) == 1
        assert "tier_mismatch" in set(result.manual_review["match_status"])

    def test_low_score_routes_to_manual_review(self):
        alias = _alias_df([
            ("Foo Open", "Foo", 2020, 2020, "WTA250", ""),
        ])
        sm = pd.DataFrame([
            _sackmann_match(0, "T", "Foo", "2020-05-01",
                            "Alice Aaa", "Bob Bbb"),
        ])
        td = pd.DataFrame([
            _td_row("2020-05-01", "Foo Open", "ZZZ X.", "QQQ Y."),
        ])
        result = match_td_to_sackmann(td, sm, alias)
        assert len(result.matched) == 0
        assert len(result.unmatched) == 1
        assert result.unmatched.iloc[0]["failure_mode"] == "low_score"
        assert len(result.manual_review) == 1
        assert "low_score" in result.manual_review.iloc[0]["match_status"]

    def test_date_window_miss_routes_to_separate_bucket(self):
        alias = _alias_df([
            ("Foo Open", "Foo", 2020, 2020, "WTA250", ""),
        ])
        sm = pd.DataFrame([
            _sackmann_match(0, "T", "Foo", "2020-05-01",
                            "Alice Aaa", "Bob Bbb"),
        ])
        # TD date is way off (Sept) — will be date_window_miss
        td = pd.DataFrame([
            _td_row("2020-09-15", "Foo Open", "Aaa A.", "Bbb B."),
        ])
        result = match_td_to_sackmann(td, sm, alias,
                                       tourney_date_window_days=7)
        assert len(result.date_window_miss) == 1
        assert result.date_window_miss.iloc[0]["failure_mode"] == "date_window_miss"
        # And recorded in manual_review for investigation
        assert "date_window_miss" in set(result.manual_review["match_status"])


# ---------------------------------------------------------------------------
# Section F — audit
# ---------------------------------------------------------------------------

class TestAudit:
    def test_score_histogram_buckets(self):
        out = _score_histogram([45, 65, 75, 82, 90, np.nan])
        assert out["0-50"] == 1
        assert out["50-70"] == 1
        assert out["70-80"] == 1
        assert out["80-85"] == 1
        assert out["85+"] == 1
        assert out["no_score"] == 1

    def test_audit_per_period_gates(self):
        # Build minimal matched/unmatched frames for two periods.
        td = pd.DataFrame([
            {"match_date": pd.Timestamp("2008-05-01"), "psw": 1.5},  # period 1
            {"match_date": pd.Timestamp("2008-05-02"), "psw": 1.5},
            {"match_date": pd.Timestamp("2015-05-01"), "psw": 1.5},  # period 2
            {"match_date": pd.Timestamp("2015-05-02"), "psw": 1.5},
            {"match_date": pd.Timestamp("2015-05-03"), "psw": 1.5},
        ])
        matched = pd.DataFrame([
            {"match_date": pd.Timestamp("2008-05-01"), "psw": 1.5,
             "winner_mismatch": False, "match_path": "primary"},
            {"match_date": pd.Timestamp("2008-05-02"), "psw": 1.5,
             "winner_mismatch": False, "match_path": "primary"},
            {"match_date": pd.Timestamp("2015-05-01"), "psw": 1.5,
             "winner_mismatch": False, "match_path": "primary"},
            {"match_date": pd.Timestamp("2015-05-02"), "psw": 1.5,
             "winner_mismatch": False, "match_path": "primary"},
        ])
        unmatched = pd.DataFrame([
            {"match_date": pd.Timestamp("2015-05-03"), "psw": 1.5,
             "failure_mode": "low_score", "best_score": 78.0},
        ])
        sk_missing = pd.DataFrame()
        dwm = pd.DataFrame()
        audit = compute_unmatched_audit(matched, unmatched, sk_missing, dwm, td)
        # Period 2010-2025: 2 matched + 1 unmatched → 33.33% > 2% gate
        p2 = audit[audit["period"] == "2010-2025"].iloc[0]
        assert p2["n_matched"] == 2
        assert p2["n_unmatched"] == 1
        assert abs(p2["unmatched_rate_%"] - 33.333) < 0.01
        assert bool(p2["gate_passed"]) is False
        # Period 2007-2008: 2 matched, 0 unmatched → 0% < 5% gate
        p1 = audit[audit["period"] == "2007-2008"].iloc[0]
        assert p1["n_matched"] == 2
        assert p1["unmatched_rate_%"] == 0
        assert bool(p1["gate_passed"]) is True

    def test_winner_mismatch_in_audit(self):
        td = pd.DataFrame([{"match_date": pd.Timestamp("2015-05-01"), "psw": 1.5}])
        matched = pd.DataFrame([{
            "match_date": pd.Timestamp("2015-05-01"), "psw": 1.5,
            "winner_mismatch": True, "match_path": "primary",
        }])
        audit = compute_unmatched_audit(
            matched, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), td,
        )
        p2 = audit[audit["period"] == "2010-2025"].iloc[0]
        assert p2["n_winner_mismatch"] == 1
        assert abs(p2["winner_mismatch_rate_%"] - 100.0) < 1e-6


# ---------------------------------------------------------------------------
# Section G — manual_review writer
# ---------------------------------------------------------------------------

class TestManualReviewWriter:
    def test_overwrite_mode(self, tmp_path):
        path = tmp_path / "review.csv"
        df1 = pd.DataFrame([{
            "td_source_file": "2020.csv",
            "td_date": pd.Timestamp("2020-05-01"),
            "td_tournament_raw": "Foo",
            "td_winner_raw": "A.",
            "td_loser_raw":  "B.",
            "match_status":  "low_score_primary",
            "reason":        "test",
            "best_sackmann_candidate": "X|Y",
            "best_score":    78.0,
            "candidate_top3": "",
        }])
        write_manual_review(df1, path)
        assert path.exists()
        # Re-write with empty rows → header-only
        write_manual_review(pd.DataFrame(), path)
        out = pd.read_csv(path)
        assert len(out) == 0
        assert "td_source_file" in out.columns

    def test_column_order_stable(self, tmp_path):
        path = tmp_path / "review.csv"
        df = pd.DataFrame([{
            # Provide cols in scrambled order
            "candidate_top3": "",
            "td_source_file": "2020.csv",
            "td_date": pd.Timestamp("2020-05-01"),
            "td_tournament_raw": "Foo",
            "td_winner_raw": "A.",
            "td_loser_raw":  "B.",
            "match_status":  "tier_mismatch",
            "reason":        "test",
            "best_sackmann_candidate": "X|Y",
            "best_score":    np.nan,
        }])
        write_manual_review(df, path)
        out = pd.read_csv(path)
        # Stable order per _MANUAL_REVIEW_COLS
        assert list(out.columns) == [
            "td_source_file", "td_date", "td_tournament_raw",
            "td_winner_raw", "td_loser_raw",
            "match_status", "reason",
            "best_sackmann_candidate", "best_score", "candidate_top3",
        ]
