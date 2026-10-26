"""Unit tests for wta_clay.data.sackmann."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wta_clay.data import sackmann
from wta_clay.data.sackmann import (
    TOUR_LEVELS,
    SAMPLE_WEIGHTS,
    classify_tour_level,
    describe_tour_levels,
    filter_clay,
    load_players,
    load_rankings,
    load_sackmann_matches,
    parse_tourney_date,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row(tourney_level, tourney_name, surface="Clay", year=2015,
        tourney_id=None, match_num=1):
    """Build a single match-row dict with required columns."""
    return {
        "tourney_id": tourney_id or f"{year}-X",
        "tourney_name": tourney_name,
        "surface": surface,
        "tourney_level": tourney_level,
        "tourney_date": year * 10000 + 520,
        "match_num": match_num,
        "winner_id": 100, "winner_name": "A P",
        "loser_id": 200, "loser_name": "B Q",
        "score": "6-0 6-0",
    }


# ---------------------------------------------------------------------------
# parse_tourney_date
# ---------------------------------------------------------------------------

class TestParseDate:
    def test_int_input(self):
        s = pd.Series([20190520, 20230101])
        out = parse_tourney_date(s)
        assert out.iloc[0] == pd.Timestamp("2019-05-20")
        assert out.iloc[1] == pd.Timestamp("2023-01-01")

    def test_invalid_becomes_nat(self):
        s = pd.Series([20190520, 99999999])
        out = parse_tourney_date(s)
        assert out.iloc[0] == pd.Timestamp("2019-05-20")
        assert pd.isna(out.iloc[1])


# ---------------------------------------------------------------------------
# classify_tour_level
# ---------------------------------------------------------------------------

class TestClassify:
    def test_grand_slam(self):
        df = pd.DataFrame([_row("G", "Roland Garros")])
        out = classify_tour_level(df)
        assert out.loc[0, "tour_level"] == "GS"
        assert out.loc[0, "sample_weight"] == 1.0
        assert out.loc[0, "not_for_betting"] is np.False_ or \
               out.loc[0, "not_for_betting"] == False

    def test_premier5_upgrade_rome(self):
        # 'P' is default WTA500, but Rome should upgrade to WTA1000
        df = pd.DataFrame([_row("P", "Internazionali BNL d'Italia - Rome")])
        out = classify_tour_level(df)
        assert out.loc[0, "tour_level"] == "WTA1000"

    def test_premier5_whitelist_hits_multiple_names(self):
        df = pd.DataFrame([
            _row("P", "Doha Open"),
            _row("P", "Dubai Duty Free"),
            _row("P", "Cincinnati Open"),
            _row("P", "Rogers Cup Toronto"),
        ])
        out = classify_tour_level(df)
        assert (out["tour_level"] == "WTA1000").all()

    def test_premier_plain_stays_wta500(self):
        df = pd.DataFrame([_row("P", "Charleston Open")])
        out = classify_tour_level(df)
        assert out.loc[0, "tour_level"] == "WTA500"

    def test_premier5_whitelist_does_not_promote_t_codes(self):
        # T2 with name "Rome" should stay WTA500, not be promoted.
        # (Defensive: whitelist only upgrades P/PM, not Tier codes.)
        df = pd.DataFrame([_row("T2", "Rome 2007")])
        out = classify_tour_level(df)
        assert out.loc[0, "tour_level"] == "WTA500"

    def test_premier_mandatory(self):
        df = pd.DataFrame([_row("PM", "Mutua Madrid Open")])
        out = classify_tour_level(df)
        assert out.loc[0, "tour_level"] == "WTA1000"

    def test_international(self):
        df = pd.DataFrame([_row("I", "Strasbourg")])
        out = classify_tour_level(df)
        assert out.loc[0, "tour_level"] == "WTA250"

    def test_tier_mapping(self):
        df = pd.DataFrame([
            _row("T1", "Berlin German Open", year=2006),
            _row("T2", "Warsaw", year=2006),
            _row("T3", "Palermo", year=2006),
            _row("T4", "Estoril", year=2006),
        ])
        out = classify_tour_level(df)
        assert list(out["tour_level"]) == ["WTA1000", "WTA500", "WTA250", "WTA250"]

    def test_fed_cup_excluded(self):
        df = pd.DataFrame([
            _row("D", "Fed Cup"),
            _row("G", "Roland Garros"),
        ])
        out = classify_tour_level(df)
        assert len(out) == 1
        assert out.loc[0, "tour_level"] == "GS"

    def test_olympics_weighted_and_flagged(self):
        df = pd.DataFrame([_row("O", "Paris Olympics 2024", year=2024)])
        out = classify_tour_level(df)
        assert out.loc[0, "tour_level"] == "WTA125"
        assert out.loc[0, "sample_weight"] == 0.5
        assert bool(out.loc[0, "not_for_betting"]) is True

    def test_wta125_code(self):
        df = pd.DataFrame([_row("W", "Cagnes-sur-Mer")])
        out = classify_tour_level(df)
        assert out.loc[0, "tour_level"] == "WTA125"
        assert out.loc[0, "sample_weight"] == 0.5
        assert bool(out.loc[0, "not_for_betting"]) is False

    def test_challenger_c_code_upgraded_to_wta125(self):
        # C (WTA 125 Challenger) — same tier as W.
        df = pd.DataFrame([_row("C", "Bucharest 125")])
        out = classify_tour_level(df)
        assert out.loc[0, "tour_level"] == "WTA125"
        assert out.loc[0, "sample_weight"] == 0.5
        assert bool(out.loc[0, "not_for_betting"]) is False

    # -- TOURNEY_NAME_OVERRIDES: Madrid PM mislabel correction --
    def test_override_madrid_2016_promoted_to_wta1000(self):
        # Sackmann coded Madrid 2016 as P; override forces WTA1000.
        df = pd.DataFrame([_row("P", "Mutua Madrid Open", year=2016)])
        out = classify_tour_level(df)
        assert out.loc[0, "tour_level"] == "WTA1000"
        assert out.loc[0, "sample_weight"] == 1.0

    def test_override_madrid_2021_promoted_to_wta1000(self):
        df = pd.DataFrame([_row("P", "Mutua Madrid Open", year=2021)])
        out = classify_tour_level(df)
        assert out.loc[0, "tour_level"] == "WTA1000"

    def test_override_madrid_2022_promoted_to_wta1000(self):
        df = pd.DataFrame([_row("P", "Mutua Madrid Open", year=2022)])
        out = classify_tour_level(df)
        assert out.loc[0, "tour_level"] == "WTA1000"

    def test_override_madrid_outside_year_range_keeps_default(self):
        # 2020 is not in the override range — P stays WTA500 (defensive).
        df = pd.DataFrame([_row("P", "Madrid", year=2020)])
        out = classify_tour_level(df)
        assert out.loc[0, "tour_level"] == "WTA500"

    def test_override_does_not_touch_itf_madrid_25k(self):
        # ITF Madrid 25K row should NEVER be promoted to WTA1000 even
        # in 2016 — raw level is a numeric ITF code, not promotable.
        df = pd.DataFrame([_row("25", "Madrid 25K", year=2016)])
        out = classify_tour_level(df)
        assert out.loc[0, "tour_level"] == "ITF"
        assert out.loc[0, "sample_weight"] == 0.25

    def test_override_does_not_touch_olympics_named_madrid(self):
        # Defensive: an Olympics row with 'Madrid' in name in override
        # year range should stay WTA125 (not_for_betting), not WTA1000.
        df = pd.DataFrame([_row("O", "Olympics Madrid 2016", year=2016)])
        out = classify_tour_level(df)
        assert out.loc[0, "tour_level"] == "WTA125"
        assert bool(out.loc[0, "not_for_betting"]) is True

    def test_override_strasbourg_2021_w_to_wta250(self):
        # Sackmann bug fix: raw='W' (WTA125) for Strasbourg 2021 must be
        # promoted to WTA250. Verifies that 'W' is in _PROMOTABLE_RAW_LEVELS.
        df = pd.DataFrame([_row("W", "Internationaux de Strasbourg",
                                 year=2021)])
        out = classify_tour_level(df)
        assert out.loc[0, "tour_level"] == "WTA250"
        assert out.loc[0, "sample_weight"] == 1.0

    def test_override_year_inclusive_at_boundaries(self):
        # Override key ('Madrid', (2021, 2022)) — both endpoints must hit.
        for yr in (2021, 2022):
            df = pd.DataFrame([_row("P", "Mutua Madrid Open", year=yr)])
            out = classify_tour_level(df)
            assert out.loc[0, "tour_level"] == "WTA1000", f"failed for {yr}"
        # 2023 is outside range — should fall back to default (P → WTA500),
        # though in real data 2023 Madrid is correctly coded PM elsewhere.
        df = pd.DataFrame([_row("P", "Mutua Madrid Open", year=2023)])
        out = classify_tour_level(df)
        assert out.loc[0, "tour_level"] == "WTA500"

    def test_override_invalid_tier_raises(self):
        df = pd.DataFrame([_row("P", "Madrid", year=2016)])
        with pytest.raises(ValueError, match="not a valid tier"):
            classify_tour_level(
                df, name_overrides={("Madrid", (2016, 2016)): "BOGUS"}
            )

    def test_unknown_code_falls_back_to_itf(self):
        df = pd.DataFrame([_row("XYZ", "Mystery Open")])
        out = classify_tour_level(df)
        assert out.loc[0, "tour_level"] == "ITF"
        assert out.loc[0, "sample_weight"] == 0.25

    def test_sample_weight_matches_table(self):
        df = pd.DataFrame([
            _row("G", "RG"),
            _row("PM", "Madrid"),
            _row("P", "Charleston"),
            _row("I", "Strasbourg"),
            _row("W", "125K"),
        ])
        out = classify_tour_level(df)
        expected = [SAMPLE_WEIGHTS[lvl] for lvl in
                    ["GS", "WTA1000", "WTA500", "WTA250", "WTA125"]]
        assert list(out["sample_weight"]) == expected

    def test_categorical_order(self):
        df = pd.DataFrame([_row("G", "RG")])
        out = classify_tour_level(df)
        assert list(out["tour_level"].cat.categories) == list(TOUR_LEVELS)


class TestNameOverrides:
    """Year-scoped TOURNEY_NAME_OVERRIDES corrections (Madrid etc.)."""

    def test_madrid_2016_upgraded_from_p_to_wta1000(self):
        df = pd.DataFrame([_row("P", "Mutua Madrid Open", year=2016)])
        out = classify_tour_level(df)
        assert out.loc[0, "tour_level"] == "WTA1000"
        assert out.loc[0, "sample_weight"] == 1.0

    def test_madrid_2020_not_overridden_stays_wta500(self):
        # 2020 falls outside BOTH override windows ((2016,2016) and (2021,2022))
        # so the raw P code should win → WTA500. Defensive test on bounds.
        df = pd.DataFrame([_row("P", "Madrid", year=2020)])
        out = classify_tour_level(df)
        assert out.loc[0, "tour_level"] == "WTA500"

    def test_madrid_2022_upgraded_via_range(self):
        df = pd.DataFrame([_row("P", "Madrid Open", year=2022)])
        out = classify_tour_level(df)
        assert out.loc[0, "tour_level"] == "WTA1000"

    def test_madrid_pm_years_untouched(self):
        # 2018 Madrid was correctly coded PM → should stay WTA1000 without overrides
        df = pd.DataFrame([_row("PM", "Madrid", year=2018)])
        out = classify_tour_level(df)
        assert out.loc[0, "tour_level"] == "WTA1000"

    def test_invalid_override_target_raises(self):
        df = pd.DataFrame([_row("P", "Madrid", year=2016)])
        with pytest.raises(ValueError):
            classify_tour_level(df, name_overrides={
                ("Madrid", (2016, 2016)): "NOT_A_TIER",
            })

    def test_custom_overrides_apply(self):
        # Caller-provided override dict replaces the default.
        df = pd.DataFrame([_row("I", "Strasbourg", year=2019)])
        out = classify_tour_level(df, name_overrides={
            ("Strasbourg", (2019, 2019)): "WTA500",
        })
        assert out.loc[0, "tour_level"] == "WTA500"

    def test_override_does_not_promote_itf_city_collision(self):
        # Regression: ITF 10K / W80 events hosted IN Madrid shouldn't be
        # promoted to WTA1000 just because their name contains "Madrid"
        # during the override year window.
        df = pd.DataFrame([
            _row("10", "Madrid 10K",  year=2016),   # ITF fallback
            _row("80", "W80 Madrid",  year=2022),   # ITF fallback
            _row("P",  "Madrid",      year=2016),   # real WTA1000 event
        ])
        out = classify_tour_level(df)
        assert out.loc[0, "tour_level"] == "ITF"
        assert out.loc[1, "tour_level"] == "ITF"
        assert out.loc[2, "tour_level"] == "WTA1000"


class TestQualItfUnifiedClassification:
    """W / C / numeric codes in qual_itf files go through classify_tour_level."""

    def test_w_code_in_qual_file_becomes_wta125(self, tmp_path):
        # Build a minimal sackmann_dir with only qual_itf containing a 'W' row
        d = tmp_path / "sackmann"
        d.mkdir()
        # Main draw needed (loader requires at least one main-draw file)
        pd.DataFrame([_row("G", "RG", year=2021, tourney_id="2021-RG")]).to_csv(
            d / "wta_matches_2021.csv", index=False
        )
        pd.DataFrame([
            _row("W", "Cagnes-sur-Mer 125", year=2021,
                 tourney_id="2021-W-CAG", match_num=1),
        ]).to_csv(d / "wta_matches_qual_itf_2021.csv", index=False)
        pd.DataFrame({"player_id": [100], "name_first": ["A"],
                     "name_last": ["X"]}).to_csv(d / "wta_players.csv", index=False)

        df = load_sackmann_matches(d, include_qual_itf=True)
        w_rows = df[df["raw_level"] == "W"]
        assert len(w_rows) == 1
        assert w_rows.iloc[0]["tour_level"] == "WTA125"
        assert w_rows.iloc[0]["sample_weight"] == 0.5

    def test_numeric_itf_code_in_qual_file_becomes_itf(self, tmp_path):
        d = tmp_path / "sackmann"
        d.mkdir()
        pd.DataFrame([_row("G", "RG", year=2020, tourney_id="2020-RG")]).to_csv(
            d / "wta_matches_2020.csv", index=False
        )
        pd.DataFrame([
            _row("25", "ITF 25K Event", year=2020,
                 tourney_id="2020-ITF-25", match_num=1),
        ]).to_csv(d / "wta_matches_qual_itf_2020.csv", index=False)
        pd.DataFrame({"player_id": [100], "name_first": ["A"],
                     "name_last": ["X"]}).to_csv(d / "wta_players.csv", index=False)

        df = load_sackmann_matches(d, include_qual_itf=True)
        itf_rows = df[df["raw_level"] == "25"]
        assert len(itf_rows) == 1
        assert itf_rows.iloc[0]["tour_level"] == "ITF"
        assert itf_rows.iloc[0]["sample_weight"] == 0.25


# ---------------------------------------------------------------------------
# filter_clay
# ---------------------------------------------------------------------------

class TestFilterClay:
    def test_case_insensitive(self):
        df = pd.DataFrame({"surface": ["Clay", "clay", "Hard", "Grass"]})
        out = filter_clay(df)
        assert len(out) == 2

    def test_index_reset(self):
        df = pd.DataFrame({"surface": ["Hard", "Clay", "Hard", "Clay"]})
        out = filter_clay(df)
        assert list(out.index) == [0, 1]


# ---------------------------------------------------------------------------
# load_sackmann_matches (end-to-end on synthetic files)
# ---------------------------------------------------------------------------

def _write_sackmann_dir(tmp_path: Path) -> Path:
    d = tmp_path / "sackmann"
    d.mkdir()

    # Year 2015: G (RG) + I (Strasbourg) + D (Fed Cup, should drop) + P Rome (upgrade)
    pd.DataFrame([
        _row("G",  "Roland Garros", year=2015, tourney_id="2015-RG",  match_num=1),
        _row("I",  "Strasbourg",    year=2015, tourney_id="2015-STR", match_num=2),
        _row("D",  "Fed Cup",       year=2015, tourney_id="2015-FC",  match_num=3),
        _row("P",  "Internazionali BNL d'Italia Rome",
             year=2015, tourney_id="2015-ROM", match_num=4),
    ]).to_csv(d / "wta_matches_2015.csv", index=False)

    # Year 2006: T1/T2/T3/T4
    pd.DataFrame([
        _row("T1", "Berlin German Open", year=2006, tourney_id="2006-BER",
             match_num=1),
        _row("T2", "Warsaw",   year=2006, tourney_id="2006-WAR", match_num=2),
        _row("T3", "Palermo",  year=2006, tourney_id="2006-PAL", match_num=3),
        _row("T4", "Estoril",  year=2006, tourney_id="2006-EST", match_num=4),
    ]).to_csv(d / "wta_matches_2006.csv", index=False)

    # Qual/ITF file for 2015 — numeric code (ITF 25K prize money tier).
    # Under the unified classifier, numeric raw codes fall back to ITF.
    pd.DataFrame([
        _row("25", "ITF 25K Something", year=2015,
             tourney_id="2015-ITF-25", match_num=1),
    ]).to_csv(d / "wta_matches_qual_itf_2015.csv", index=False)

    # Players + rankings (minimal, so load_players / load_rankings are testable)
    pd.DataFrame({
        "player_id": [100, 200],
        "name_first": ["Alice", "Bob"],
        "name_last": ["Test", "Case"],
        "hand": ["R", "L"],
        "dob": [19950101, 19930615],
        "ioc": ["USA", "ESP"],
    }).to_csv(d / "wta_players.csv", index=False)
    pd.DataFrame({
        "ranking_date": [20150601, 20150608],
        "rank": [10, 12], "player": [100, 100], "points": [2000, 1900],
    }).to_csv(d / "wta_rankings_10s.csv", index=False)

    return d


class TestLoadSackmannMatches:
    def test_end_to_end(self, tmp_path):
        d = _write_sackmann_dir(tmp_path)
        df = load_sackmann_matches(d, include_qual_itf=True)

        # Fed Cup row dropped → 4 + 4 + 1 = 9, minus 1 Fed Cup = 8
        assert len(df) == 8

        # Columns present
        for col in ("tour_level", "sample_weight", "not_for_betting",
                    "raw_level", "match_date"):
            assert col in df.columns

        # Rome promoted to WTA1000
        rome = df[df["tourney_name"].str.contains("Rome")].iloc[0]
        assert rome["tour_level"] == "WTA1000"

        # ITF block tagged correctly
        itf = df[df["tour_level"] == "ITF"]
        assert len(itf) == 1
        assert itf["sample_weight"].iloc[0] == 0.25

        # Sort order: 2006 rows come before 2015 rows
        assert df["match_date"].is_monotonic_increasing

    def test_no_qual_itf(self, tmp_path):
        d = _write_sackmann_dir(tmp_path)
        df = load_sackmann_matches(d, include_qual_itf=False)
        assert "ITF" not in df["tour_level"].astype(str).unique()

    def test_year_range(self, tmp_path):
        d = _write_sackmann_dir(tmp_path)
        df = load_sackmann_matches(d, year_from=2015, include_qual_itf=False)
        assert (df["match_date"].dt.year == 2015).all()

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_sackmann_matches(tmp_path / "nope")


class TestPlayersRankings:
    def test_load_players(self, tmp_path):
        d = _write_sackmann_dir(tmp_path)
        players = load_players(d)
        assert "full_name" in players.columns
        assert set(players["full_name"]) == {"Alice Test", "Bob Case"}
        assert players["dob"].dtype.kind == "M"

    def test_load_rankings(self, tmp_path):
        d = _write_sackmann_dir(tmp_path)
        rk = load_rankings(d)
        assert "ranking_date" in rk.columns
        assert rk["ranking_date"].dtype.kind == "M"
        assert list(rk.columns).count("player_id") == 1


# ---------------------------------------------------------------------------
# describe_tour_levels
# ---------------------------------------------------------------------------

class TestDescribe:
    def test_by_year_shape(self, tmp_path):
        d = _write_sackmann_dir(tmp_path)
        df = load_sackmann_matches(d)
        tab = describe_tour_levels(df, by_year=True)
        assert tab.shape[1] >= 2  # at least 2006 and 2015
        # Fed Cup row was dropped, so "D" should not appear
        assert "D" not in tab.index

    def test_flat(self, tmp_path):
        d = _write_sackmann_dir(tmp_path)
        df = load_sackmann_matches(d)
        tab = describe_tour_levels(df, by_year=False)
        assert {"raw_level", "tour_level", "n"}.issubset(tab.columns)
