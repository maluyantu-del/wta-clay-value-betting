"""Unit tests for wta_clay.data.oddsfeed.

Fixtures mimic real Tennis-Data.co.uk WTA files: per-year CSV with the
canonical column set and realistic schema drift across years.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from wta_clay.data import oddsfeed
from wta_clay.data.oddsfeed import (
    CANONICAL_COLUMNS,
    HEADER_ALIASES,
    ROUND_NORM_ORDER,
    _canonicalize_headers,
    audit_bookmaker_coverage,
    audit_unknown_rounds,
    drop_walkovers,
    filter_clay_odds,
    load_tennisdata_odds,
    normalize_round,
    parse_dates,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Schema for a modern year (2019+): Pinnacle + Bet365 + Max/Avg present.
MODERN_COLS = [
    "Date", "Tournament", "Series", "Court", "Surface", "Round", "Best of",
    "Winner", "Loser", "WRank", "LRank",
    "PSW", "PSL", "B365W", "B365L", "MaxW", "MaxL", "AvgW", "AvgL",
    "Comment",
]

# Schema for an older year (pre-Pinnacle, pre-aggregates).
LEGACY_COLS = [
    "Date", "Tournament", "Tier", "Court", "Surface", "Round", "Best of",
    "Winner", "Loser", "WRank", "LRank", "B365W", "B365L", "Comment",
]


def _make_modern_row(date, winner, loser, round_="Quarterfinals",
                    surface="Clay", psw=1.80, psl=2.05, **kw):
    base = dict.fromkeys(MODERN_COLS)
    base.update({
        "Date": date, "Tournament": "Test Open", "Series": "WTA1000",
        "Court": "Outdoor", "Surface": surface, "Round": round_,
        "Best of": 3, "Winner": winner, "Loser": loser,
        "WRank": 10, "LRank": 50, "PSW": psw, "PSL": psl,
        "B365W": 1.75, "B365L": 2.10, "MaxW": 1.85, "MaxL": 2.15,
        "AvgW": 1.78, "AvgL": 2.08, "Comment": "Completed",
    })
    base.update(kw)
    return base


def _make_legacy_row(date, winner, loser, **kw):
    base = dict.fromkeys(LEGACY_COLS)
    base.update({
        "Date": date, "Tournament": "Old Open", "Tier": "Tier I",
        "Court": "Outdoor", "Surface": "Clay", "Round": "Semifinals",
        "Best of": 3, "Winner": winner, "Loser": loser,
        "WRank": 5, "LRank": 20, "B365W": 1.60, "B365L": 2.40,
        "Comment": "Completed",
    })
    base.update(kw)
    return base


def _write_tennisdata_dir(tmp_path: Path) -> Path:
    d = tmp_path / "tennis_data"
    d.mkdir()
    # 2019: modern schema, mix of clay + hard + walkover
    pd.DataFrame([
        _make_modern_row("20/05/2019", "Swiatek I.", "Halep S.",
                        round_="Quarterfinals", psw=1.75, psl=2.10),
        _make_modern_row("21/05/2019", "Osaka N.", "Williams S.",
                        surface="Hard"),   # should be filtered out
        _make_modern_row("22/05/2019", "Kerber A.", "Vondrousova M.",
                        round_="The Final", **{"Comment": "Walkover"}),
    ], columns=MODERN_COLS).to_csv(d / "2019.csv", index=False)

    # 2006: legacy schema (no Pinnacle, uses 'Tier' not 'Series', no Max/Avg)
    pd.DataFrame([
        _make_legacy_row("15/05/2006", "Henin J.", "Kuznetsova S."),
    ], columns=LEGACY_COLS).to_csv(d / "2006.csv", index=False)

    return d


# ---------------------------------------------------------------------------
# Header canonicalization
# ---------------------------------------------------------------------------

class TestHeaderCanonicalization:
    def test_modern_headers_all_map(self):
        df = pd.DataFrame(columns=MODERN_COLS + ["_source_file"])
        out = _canonicalize_headers(df)
        # Every modern column has a canonical mapping
        assert "match_date" in out.columns
        assert "series_raw" in out.columns
        assert "psw" in out.columns and "psl" in out.columns
        assert "avgw" in out.columns

    def test_legacy_tier_column_aliased_to_series_raw(self):
        df = pd.DataFrame(columns=LEGACY_COLS + ["_source_file"])
        out = _canonicalize_headers(df)
        assert "series_raw" in out.columns
        # Legacy doesn't have Pinnacle columns — those should be absent here
        assert "psw" not in out.columns

    def test_case_insensitive_match(self):
        df = pd.DataFrame(columns=["DATE", "Winner", "loser", "PsW"])
        out = _canonicalize_headers(df)
        assert set(out.columns) == {"match_date", "winner_name_raw",
                                    "loser_name_raw", "psw"}

    def test_unknown_columns_dropped(self):
        df = pd.DataFrame(columns=["Date", "Winner", "Loser", "MysteryCol"])
        out = _canonicalize_headers(df)
        assert "MysteryCol" not in out.columns


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

class TestParseDates:
    def test_dayfirst_dmy(self):
        s = pd.Series(["20/05/2019", "01/06/2019"])
        out = parse_dates(s)
        assert out.iloc[0] == pd.Timestamp("2019-05-20")
        assert out.iloc[1] == pd.Timestamp("2019-06-01")

    def test_already_timestamp_passthrough(self):
        s = pd.to_datetime(pd.Series(["2019-05-20", "2019-06-01"]))
        out = parse_dates(s)
        assert out.iloc[0] == pd.Timestamp("2019-05-20")

    def test_iso_strings_fall_back_to_non_dayfirst(self):
        s = pd.Series(["2019-05-20", "2019-06-01"])
        out = parse_dates(s)
        assert out.iloc[0] == pd.Timestamp("2019-05-20")

    def test_invalid_becomes_nat(self):
        s = pd.Series(["20/05/2019", "garbage"])
        out = parse_dates(s)
        assert pd.isna(out.iloc[1])


# ---------------------------------------------------------------------------
# Round normalization
# ---------------------------------------------------------------------------

class TestNormalizeRound:
    def test_tennisdata_phrases(self):
        s = pd.Series(["Quarterfinals", "Semifinals", "The Final",
                      "2nd Round", "Round Robin"])
        out = normalize_round(s)
        assert list(out) == ["QF", "SF", "F", "R64", "RR"]

    def test_sackmann_short_codes(self):
        s = pd.Series(["QF", "SF", "F", "R32"])
        out = normalize_round(s)
        assert list(out) == ["QF", "SF", "F", "R32"]

    def test_case_insensitive(self):
        s = pd.Series(["quarterfinals", "SEMIFINALS"])
        out = normalize_round(s)
        assert list(out) == ["QF", "SF"]

    def test_unknown_becomes_nan(self):
        s = pd.Series(["Mystery Round"])
        out = normalize_round(s)
        assert out.isna().all()

    def test_ordered_category(self):
        s = pd.Series(["Quarterfinals"])
        out = normalize_round(s)
        assert list(out.cat.categories) == list(ROUND_NORM_ORDER)
        assert out.cat.ordered


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

class TestFilterClayOdds:
    def test_case_insensitive(self):
        df = pd.DataFrame({"surface": ["Clay", "clay", "Hard", "Grass"]})
        assert len(filter_clay_odds(df)) == 2


class TestDropWalkovers:
    def test_drops_walkover_only(self):
        df = pd.DataFrame({"comment": ["Completed", "Walkover", "Retired",
                                       "walkover"]})
        out = drop_walkovers(df)
        assert len(out) == 2
        assert set(out["comment"]) == {"Completed", "Retired"}

    def test_missing_comment_column_noop(self):
        df = pd.DataFrame({"other": [1, 2, 3]})
        out = drop_walkovers(df)
        assert len(out) == 3


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

class TestLoadTennisdataOdds:
    def test_end_to_end(self, tmp_path):
        d = _write_tennisdata_dir(tmp_path)
        df = load_tennisdata_odds(d, clay_only=True, drop_walkover=True)

        # 2019: 3 rows written, 1 Hard dropped, 1 Walkover dropped → 1 kept
        # 2006: 1 row kept
        assert len(df) == 2

        # Canonical columns all present, in the documented order
        assert list(df.columns) == list(CANONICAL_COLUMNS)

        # Pinnacle present in 2019 row, missing (NA) in 2006 row
        year_2019 = df[df["match_date"].dt.year == 2019].iloc[0]
        year_2006 = df[df["match_date"].dt.year == 2006].iloc[0]
        assert pd.notna(year_2019["psw"])
        assert pd.isna(year_2006["psw"])

        # Date parsed correctly
        assert year_2019["match_date"] == pd.Timestamp("2019-05-20")

        # Round normalized
        assert year_2019["round_norm"] == "QF"
        assert year_2006["round_norm"] == "SF"

        # Dtypes
        assert df["psw"].dtype == "float64"
        assert str(df["winner_rank"].dtype) == "Int64"

    def test_no_clay_filter(self, tmp_path):
        d = _write_tennisdata_dir(tmp_path)
        df = load_tennisdata_odds(d, clay_only=False, drop_walkover=False)
        # 3 rows in 2019 + 1 row in 2006 = 4
        assert len(df) == 4

    def test_sorted_by_date(self, tmp_path):
        d = _write_tennisdata_dir(tmp_path)
        df = load_tennisdata_odds(d)
        assert df["match_date"].is_monotonic_increasing

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_tennisdata_odds(tmp_path / "nope")

    def test_mixed_schema_tier_and_series_coalesce(self, tmp_path, caplog):
        """Regression: legacy 'Tier' and modern 'Series' should collapse into
        one ``series_raw`` column without a 'duplicate canonical column' warning.
        """
        d = _write_tennisdata_dir(tmp_path)
        with caplog.at_level("WARNING"):
            df = load_tennisdata_odds(d)
        assert not any("Duplicate canonical" in r.message for r in caplog.records)

        # Both years present, each with a non-null series_raw value
        y2006 = df[df["match_date"].dt.year == 2006].iloc[0]
        y2019 = df[df["match_date"].dt.year == 2019].iloc[0]
        assert y2006["series_raw"] == "Tier I"
        assert y2019["series_raw"] == "WTA1000"


# ---------------------------------------------------------------------------
# Audit helpers
# ---------------------------------------------------------------------------

class TestAuditHelpers:
    def test_bookmaker_coverage(self, tmp_path):
        d = _write_tennisdata_dir(tmp_path)
        df = load_tennisdata_odds(d)
        cov = audit_bookmaker_coverage(df)
        assert set(cov.columns) == {"n", "pinnacle_%", "b365_%",
                                    "avg_%", "max_%"}
        # 2006 row has no Pinnacle → 0%
        assert cov.loc[2006, "pinnacle_%"] == 0.0
        # 2019 row has Pinnacle → 100%
        assert cov.loc[2019, "pinnacle_%"] == 100.0
        # Both have Bet365
        assert cov.loc[2006, "b365_%"] == 100.0

    def test_unknown_rounds_counts(self):
        df = pd.DataFrame({
            "round_raw": pd.array(["Quarterfinals", "Mystery", "Mystery"],
                                  dtype="string"),
            "round_norm": normalize_round(pd.Series(
                ["Quarterfinals", "Mystery", "Mystery"])),
        })
        out = audit_unknown_rounds(df)
        assert out["Mystery"] == 2


# ---------------------------------------------------------------------------
# Surface blacklist + consistency audit
# ---------------------------------------------------------------------------

class TestSurfaceBlacklist:
    def test_rouen_is_in_default_blacklist(self):
        # Known Rouen/Clay TD tagging bug — all years.
        assert ("Open de Rouen", "Clay", None, None) in oddsfeed.SURFACE_BLACKLIST

    def test_istanbul_2017_plus_is_in_default_blacklist(self):
        # Istanbul Cup 2017+ Clay rows are TD mis-tags (event was Hard 2015+).
        assert ("Istanbul Cup", "Clay", 2017, None) in oddsfeed.SURFACE_BLACKLIST
        assert ("TEB BNP Paribas Istanbul Cup", "Clay", 2017, None) \
            in oddsfeed.SURFACE_BLACKLIST

    def test_apply_drops_blacklisted_rows_all_years(self):
        df = pd.DataFrame({
            "tournament": ["Open de Rouen", "Open de Rouen", "Roland Garros"],
            "surface":    ["Clay",          "Hard",          "Clay"],
            "match_date": pd.to_datetime(["2024-04-15", "2024-04-15",
                                          "2024-05-28"]),
        })
        out = oddsfeed._apply_surface_blacklist(df)
        # Rouen/Clay dropped; Rouen/Hard + RG/Clay kept.
        assert len(out) == 2
        assert (out["tournament"] == "Open de Rouen").sum() == 1
        assert (out.loc[out["tournament"] == "Open de Rouen", "surface"]
                .iloc[0]) == "Hard"

    def test_apply_case_and_whitespace_insensitive(self):
        df = pd.DataFrame({
            "tournament": ["  OPEN DE ROUEN  "],
            "surface":    ["clay"],
            "match_date": pd.to_datetime(["2024-04-15"]),
        })
        out = oddsfeed._apply_surface_blacklist(df)
        assert out.empty

    def test_year_lower_bound_drops_only_in_range(self):
        # Istanbul Cup: 2016 Clay legit (kept), 2017+ Clay mis-tag (dropped).
        df = pd.DataFrame({
            "tournament": ["Istanbul Cup"] * 4,
            "surface":    ["Clay"] * 4,
            "match_date": pd.to_datetime(
                ["2014-04-20", "2016-04-20", "2017-04-24", "2022-04-15"]),
        })
        out = oddsfeed._apply_surface_blacklist(df)
        # 2014 and 2016 kept (year < 2017); 2017 and 2022 dropped.
        kept_yrs = sorted(out["match_date"].dt.year.tolist())
        assert kept_yrs == [2014, 2016]

    def test_year_upper_bound_drops_only_in_range(self):
        bl = frozenset({("Test Event", "Clay", None, 2010)})
        df = pd.DataFrame({
            "tournament": ["Test Event"] * 3,
            "surface":    ["Clay"] * 3,
            "match_date": pd.to_datetime(
                ["2008-01-01", "2010-12-31", "2011-01-01"]),
        })
        out = oddsfeed._apply_surface_blacklist(df, blacklist=bl)
        # 2008 + 2010 dropped (≤ 2010); 2011 kept.
        assert sorted(out["match_date"].dt.year.tolist()) == [2011]

    def test_year_both_bounds(self):
        bl = frozenset({("Test Event", "Clay", 2018, 2020)})
        df = pd.DataFrame({
            "tournament": ["Test Event"] * 5,
            "surface":    ["Clay"] * 5,
            "match_date": pd.to_datetime(
                ["2017-05-01", "2018-05-01", "2019-05-01",
                 "2020-05-01", "2021-05-01"]),
        })
        out = oddsfeed._apply_surface_blacklist(df, blacklist=bl)
        assert sorted(out["match_date"].dt.year.tolist()) == [2017, 2021]

    def test_year_boundary_inclusive(self):
        bl = frozenset({("Test Event", "Clay", 2017, None)})
        df = pd.DataFrame({
            "tournament": ["Test Event", "Test Event"],
            "surface":    ["Clay", "Clay"],
            "match_date": pd.to_datetime(["2016-12-31", "2017-01-01"]),
        })
        out = oddsfeed._apply_surface_blacklist(df, blacklist=bl)
        # 2017-01-01 is the inclusive lower bound → dropped.
        assert (out["match_date"].dt.year == 2016).all()

    def test_nat_match_date_never_blacklisted(self):
        # Defensive: rows with missing date should not match year-scoped rules.
        bl = frozenset({("Test Event", "Clay", 2017, None)})
        df = pd.DataFrame({
            "tournament": ["Test Event"],
            "surface":    ["Clay"],
            "match_date": pd.to_datetime([None]),
        })
        out = oddsfeed._apply_surface_blacklist(df, blacklist=bl)
        assert len(out) == 1

    def test_apply_custom_blacklist_overrides_default(self):
        df = pd.DataFrame({
            "tournament": ["Foo Open", "Bar Cup", "Open de Rouen"],
            "surface":    ["Clay",     "Clay",    "Clay"],
            "match_date": pd.to_datetime(["2024-04-15"] * 3),
        })
        bl = frozenset({("Foo Open", "Clay", None, None)})
        out = oddsfeed._apply_surface_blacklist(df, blacklist=bl)
        # Default Rouen entry is bypassed when a custom set is passed.
        assert sorted(out["tournament"]) == ["Bar Cup", "Open de Rouen"]

    def test_apply_empty_blacklist_is_noop(self):
        df = pd.DataFrame({
            "tournament": ["Anything"],
            "surface":    ["Clay"],
            "match_date": pd.to_datetime(["2024-04-15"]),
        })
        out = oddsfeed._apply_surface_blacklist(df, blacklist=frozenset())
        assert len(out) == 1

    def test_blacklist_applied_in_load_pipeline(self, tmp_path):
        # End-to-end: split rows across year-named files (which is what
        # discover_tennisdata_files looks for). Verify Rouen/Clay all-years
        # entry + Istanbul/Clay 2017+ year-range entry both fire.
        d = tmp_path / "tennis_data"
        d.mkdir()
        common = dict(Round="1st Round", PSW=1.5, PSL=2.6,
                      Comment="Completed", Surface="Clay")
        # 2016: legit Istanbul Clay (kept)
        pd.DataFrame([{
            "Date": "2016-04-20", "Tournament": "Istanbul Cup",
            "Winner": "A.", "Loser": "B.", **common,
        }]).to_csv(d / "2016.csv", index=False)
        # 2018: Istanbul Clay (dropped — year ≥ 2017)
        pd.DataFrame([{
            "Date": "2018-04-25", "Tournament": "Istanbul Cup",
            "Winner": "C.", "Loser": "D.", **common,
        }]).to_csv(d / "2018.csv", index=False)
        # 2024: Rouen Clay (dropped — all years) + French Open Clay (kept)
        pd.DataFrame([
            {"Date": "2024-04-15", "Tournament": "Open de Rouen",
             "Winner": "E.", "Loser": "F.", **common},
            {"Date": "2024-05-28", "Tournament": "French Open",
             "Winner": "G.", "Loser": "H.", **common},
        ]).to_csv(d / "2024.csv", index=False)

        df = load_tennisdata_odds(d, clay_only=True)
        assert (df["tournament"] == "Open de Rouen").sum() == 0
        assert ((df["tournament"] == "Istanbul Cup") &
                (df["match_date"].dt.year == 2018)).sum() == 0
        assert ((df["tournament"] == "Istanbul Cup") &
                (df["match_date"].dt.year == 2016)).sum() == 1
        assert (df["tournament"] == "French Open").sum() == 1


class TestAuditSurfaceConsistency:
    def test_single_surface_tournaments_not_flagged(self):
        df = pd.DataFrame({
            "tournament": ["A"] * 3 + ["B"] * 2,
            "surface":    ["Clay"] * 3 + ["Hard"] * 2,
            "match_date": pd.to_datetime(
                ["2020-05-01", "2021-05-01", "2022-05-01",
                 "2020-07-01", "2021-07-01"]),
        })
        out = oddsfeed.audit_surface_consistency(df)
        assert out.empty

    def test_multi_surface_tournament_flagged(self):
        # Stuttgart-style: 2 hard years + 3 clay years → flagged with
        # Clay as majority and Hard 2007/2008 as the minority years.
        df = pd.DataFrame({
            "tournament": ["Stuttgart"] * 5,
            "surface":    ["Hard", "Hard", "Clay", "Clay", "Clay"],
            "match_date": pd.to_datetime(
                ["2007-05-01", "2008-05-01", "2009-05-01",
                 "2010-05-01", "2011-05-01"]),
        })
        out = oddsfeed.audit_surface_consistency(df)
        assert "Stuttgart" in out.index
        assert out.loc["Stuttgart", "majority_surface"] == "Clay"
        assert out.loc["Stuttgart", "minority_n"] == 2
        assert out.loc["Stuttgart", "minority_years"] == [2007, 2008]

    def test_sorted_by_minority_n_descending(self):
        df = pd.DataFrame({
            "tournament": ["Big"] * 6 + ["Small"] * 3,
            "surface":    (["Hard"] * 4 + ["Clay"] * 2 +
                           ["Hard"] * 2 + ["Clay"]),
            "match_date": pd.to_datetime(["2020-01-01"] * 9),
        })
        out = oddsfeed.audit_surface_consistency(df)
        assert list(out.index) == ["Big", "Small"]

    def test_empty_input_returns_empty(self):
        assert oddsfeed.audit_surface_consistency(pd.DataFrame()).empty


class TestVigSanity:
    """Tests for ``oddsfeed._apply_vig_sanity`` — drop rows where the
    implied vig is below ``VIG_SANITY_MIN`` (default 0.99)."""

    def test_constant_vig_threshold(self):
        assert oddsfeed.VIG_SANITY_MIN == 0.99

    def test_drops_pinnacle_arbitrage_row(self):
        df = pd.DataFrame({
            "match_date": pd.to_datetime(["2020-09-21", "2020-05-15"]),
            "tournament": ["M009", "Madrid"],
            "winner_name_raw": ["A.", "B."],
            "loser_name_raw":  ["C.", "D."],
            # Row 0: known real-data error — odds 52.0/2.77, implied 0.380
            # Row 1: typical Pinnacle pricing — odds 1.85/2.10, implied 1.017
            "psw":   [52.00, 1.85],
            "psl":   [2.77, 2.10],
            "b365w": [pd.NA, 1.85],
            "b365l": [pd.NA, 2.10],
        })
        out = oddsfeed._apply_vig_sanity(df)
        assert len(out) == 1
        assert out.iloc[0]["tournament"] == "Madrid"

    def test_drops_b365_arbitrage_row_when_pinnacle_missing(self):
        df = pd.DataFrame({
            "match_date": pd.to_datetime(["2010-05-01"]),
            "tournament": ["TestEvent"],
            "winner_name_raw": ["X."],
            "loser_name_raw":  ["Y."],
            "psw":   [pd.NA],
            "psl":   [pd.NA],
            "b365w": [3.5],
            "b365l": [3.5],   # implied = 0.286 + 0.286 = 0.571 → drop
        })
        out = oddsfeed._apply_vig_sanity(df)
        assert len(out) == 0

    def test_keeps_normal_pinnacle_vig(self):
        df = pd.DataFrame({
            "match_date": pd.to_datetime(["2020-05-01"] * 3),
            "tournament": ["A", "B", "C"],
            "winner_name_raw": ["x"] * 3, "loser_name_raw": ["y"] * 3,
            # All implied vigs > 0.99 (typical Pinnacle):
            # 1.85/2.10 = 1.017,  1.85/2.20 = 0.995 (just above),
            # 1.80/2.10 = 1.032
            "psw":   [1.85, 1.85, 1.80],
            "psl":   [2.10, 2.20, 2.10],
            "b365w": [pd.NA, pd.NA, pd.NA],
            "b365l": [pd.NA, pd.NA, pd.NA],
        })
        out = oddsfeed._apply_vig_sanity(df)
        assert len(out) == 3

    def test_keeps_rows_with_no_odds_present(self):
        # No-odds rows can't fail vig sanity → keep them.
        df = pd.DataFrame({
            "match_date": pd.to_datetime(["2007-01-01"]),
            "tournament": ["Old"], "winner_name_raw": ["x"],
            "loser_name_raw": ["y"],
            "psw":   [pd.NA], "psl":   [pd.NA],
            "b365w": [pd.NA], "b365l": [pd.NA],
        })
        out = oddsfeed._apply_vig_sanity(df)
        assert len(out) == 1

    def test_borderline_below_threshold(self):
        # Implied = 0.989 just below 0.99 → drop
        df = pd.DataFrame({
            "match_date": pd.to_datetime(["2024-05-01", "2024-05-02"]),
            "tournament": ["X", "Y"],
            "winner_name_raw": ["a", "b"], "loser_name_raw": ["c", "d"],
            "psw": [2.00, 2.00],
            "psl": [2.10, 2.05],   # 0.5 + 0.476 = 0.976 (drop), 0.5 + 0.488 = 0.988 (drop)
            "b365w": [pd.NA, pd.NA], "b365l": [pd.NA, pd.NA],
        })
        out = oddsfeed._apply_vig_sanity(df)
        assert len(out) == 0

    def test_custom_threshold(self):
        # Stricter threshold drops cleaner rows
        df = pd.DataFrame({
            "match_date": pd.to_datetime(["2024-05-01"]),
            "tournament": ["X"],
            "winner_name_raw": ["a"], "loser_name_raw": ["b"],
            "psw": [1.95], "psl": [2.10],   # implied = 0.989
            "b365w": [pd.NA], "b365l": [pd.NA],
        })
        # default 0.99: keep at the boundary? 0.989 < 0.99 → drop
        out = oddsfeed._apply_vig_sanity(df, threshold=0.99)
        assert len(out) == 0
        # threshold = 0.95: keep
        out = oddsfeed._apply_vig_sanity(df, threshold=0.95)
        assert len(out) == 1

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({
            "match_date": pd.to_datetime(["2020-09-21"]),
            "tournament": ["M009"],
            "winner_name_raw": ["A"], "loser_name_raw": ["B"],
            "psw": [52.00], "psl": [2.77],
            "b365w": [pd.NA], "b365l": [pd.NA],
        })
        before = df.copy()
        _ = oddsfeed._apply_vig_sanity(df)
        pd.testing.assert_frame_equal(df, before)

    def test_zero_or_negative_odds_are_ignored(self):
        # Defensive: psw=0 should not propagate inf into 1/odds; we treat
        # it as missing for the vig check.
        df = pd.DataFrame({
            "match_date": pd.to_datetime(["2020-05-01"]),
            "tournament": ["X"],
            "winner_name_raw": ["a"], "loser_name_raw": ["b"],
            "psw": [0.0], "psl": [2.0],
            "b365w": [pd.NA], "b365l": [pd.NA],
        })
        out = oddsfeed._apply_vig_sanity(df)
        # Treated as no Pinnacle pair present → kept (no signal to drop on).
        assert len(out) == 1

    def test_vig_sanity_applied_in_load_pipeline(self, tmp_path):
        d = tmp_path / "tennis_data"
        d.mkdir()
        common = dict(Round="1st Round", Surface="Clay",
                      Comment="Completed", Court="Outdoor")
        # One legit row (implied 1.017) + one arbitrage row (implied 0.380)
        pd.DataFrame([
            {"Date": "2020-05-01", "Tournament": "OK Cup",
             "Winner": "A.", "Loser": "B.",
             "PSW": 1.85, "PSL": 2.10, **common},
            {"Date": "2020-09-21", "Tournament": "Bad Cup",
             "Winner": "C.", "Loser": "D.",
             "PSW": 52.0, "PSL": 2.77, **common},
        ]).to_csv(d / "2020.csv", index=False)

        df = load_tennisdata_odds(d, clay_only=True)
        assert len(df) == 1
        assert df.iloc[0]["tournament"] == "OK Cup"
