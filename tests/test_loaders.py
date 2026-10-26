"""Unit tests for wta_clay.data.loaders.

These tests DO NOT require the actual Sackmann or Tennis-Data downloads.
They build tiny synthetic files in ``tmp_path`` that mimic the real layout,
so the test suite is hermetic and fast.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from wta_clay.data import loaders


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SACKMANN_MAIN_COLS = [
    "tourney_id", "tourney_name", "surface", "draw_size", "tourney_level",
    "tourney_date", "match_num",
    "winner_id", "winner_name", "winner_hand", "winner_ioc", "winner_age",
    "loser_id", "loser_name", "loser_hand", "loser_ioc", "loser_age",
    "score", "best_of", "round", "minutes",
    "winner_rank", "loser_rank",
]


def _make_sackmann_dir(tmp_path: Path, years: list[int],
                      with_qual: bool = False) -> Path:
    d = tmp_path / "sackmann"
    d.mkdir()
    for y in years:
        pd.DataFrame({
            "tourney_id": [f"{y}-001"],
            "tourney_name": ["Test Open"],
            "surface": ["Clay"],
            "draw_size": [32],
            "tourney_level": ["P"],
            "tourney_date": [y * 10000 + 415],  # YYYYMMDD = Apr 15
            "match_num": [1],
            "winner_id": [100],
            "winner_name": ["Alice Test"],
            "winner_hand": ["R"],
            "winner_ioc": ["USA"],
            "winner_age": [25.0],
            "loser_id": [200],
            "loser_name": ["Bob Test"],
            "loser_hand": ["R"],
            "loser_ioc": ["ESP"],
            "loser_age": [27.0],
            "score": ["6-4 6-2"],
            "best_of": [3],
            "round": ["R32"],
            "minutes": [90],
            "winner_rank": [10],
            "loser_rank": [50],
        }).to_csv(d / f"wta_matches_{y}.csv", index=False)
        if with_qual:
            (d / f"wta_matches_qual_itf_{y}.csv").write_text(
                "tourney_id,surface\nQ{0}-001,Clay\n".format(y)
            )
    # players
    pd.DataFrame({
        "player_id": [100, 200],
        "name_first": ["Alice", "Bob"],
        "name_last": ["Test", "Test"],
        "hand": ["R", "R"],
        "dob": [19950101, 19930101],
        "ioc": ["USA", "ESP"],
    }).to_csv(d / "wta_players.csv", index=False)
    # rankings (one decade file)
    pd.DataFrame({
        "ranking_date": [20200106, 20200106],
        "rank": [10, 50],
        "player": [100, 200],
        "points": [2000, 800],
    }).to_csv(d / "wta_rankings_20s.csv", index=False)
    return d


def _make_tennisdata_dir(tmp_path: Path, years: list[int],
                         ext: str = "csv") -> Path:
    d = tmp_path / "tennis_data"
    d.mkdir()
    for y in years:
        df = pd.DataFrame({
            "Date": [f"{y}-05-01"],
            "Tournament": ["Test Open"],
            "Surface": ["Clay"],
            "Round": ["1st Round"],
            "Winner": ["Test A."],
            "Loser": ["Test B."],
            "PSW": [1.80],
            "PSL": [2.05],
            "B365W": [1.75],
            "B365L": [2.10],
        })
        path = d / f"{y}.{ext}"
        if ext == "csv":
            df.to_csv(path, index=False)
        elif ext == "xlsx":
            df.to_excel(path, index=False, engine="openpyxl")
        else:
            raise ValueError(ext)
    return d


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

class TestSackmannDiscovery:
    def test_main_files_full_range(self, tmp_path):
        d = _make_sackmann_dir(tmp_path, [2005, 2010, 2020])
        files = loaders.discover_sackmann_match_files(d)
        assert [f.name for f in files] == [
            "wta_matches_2005.csv",
            "wta_matches_2010.csv",
            "wta_matches_2020.csv",
        ]

    def test_main_files_year_range_filter(self, tmp_path):
        d = _make_sackmann_dir(tmp_path, [2005, 2010, 2020, 2024])
        files = loaders.discover_sackmann_match_files(d, year_from=2010, year_to=2020)
        assert [f.name for f in files] == [
            "wta_matches_2010.csv",
            "wta_matches_2020.csv",
        ]

    def test_main_files_excludes_qual_and_futures(self, tmp_path):
        d = _make_sackmann_dir(tmp_path, [2020], with_qual=True)
        # Add a futures file too
        (d / "wta_matches_futures_2020.csv").write_text("x\n1\n")
        files = loaders.discover_sackmann_match_files(d)
        assert [f.name for f in files] == ["wta_matches_2020.csv"]

    def test_qual_itf_discovery(self, tmp_path):
        d = _make_sackmann_dir(tmp_path, [2020, 2021], with_qual=True)
        files = loaders.discover_sackmann_qual_itf_files(d, year_from=2021)
        assert [f.name for f in files] == ["wta_matches_qual_itf_2021.csv"]

    def test_rankings_discovery(self, tmp_path):
        d = _make_sackmann_dir(tmp_path, [2020])
        files = loaders.discover_sackmann_ranking_files(d)
        assert any("rankings" in f.name for f in files)

    def test_players_path(self, tmp_path):
        d = _make_sackmann_dir(tmp_path, [2020])
        p = loaders.sackmann_players_path(d)
        assert p.name == "wta_players.csv"

    def test_players_missing_raises(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        with pytest.raises(FileNotFoundError):
            loaders.sackmann_players_path(d)

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            loaders.discover_sackmann_match_files(tmp_path / "nope")


class TestTennisDataDiscovery:
    def test_csv_discovery(self, tmp_path):
        d = _make_tennisdata_dir(tmp_path, [2005, 2019])
        files = loaders.discover_tennisdata_files(d)
        assert [f.name for f in files] == ["2005.csv", "2019.csv"]

    def test_year_filter(self, tmp_path):
        d = _make_tennisdata_dir(tmp_path, [2005, 2019, 2024])
        files = loaders.discover_tennisdata_files(d, year_from=2019)
        assert [f.name for f in files] == ["2019.csv", "2024.csv"]

    def test_prefers_csv_over_xlsx_for_same_year(self, tmp_path):
        d = tmp_path / "tennis_data"
        d.mkdir()
        pd.DataFrame({"a": [1]}).to_csv(d / "2020.csv", index=False)
        pd.DataFrame({"a": [1]}).to_excel(d / "2020.xlsx", index=False, engine="openpyxl")
        files = loaders.discover_tennisdata_files(d)
        assert [f.name for f in files] == ["2020.csv"]


# ---------------------------------------------------------------------------
# CSV reading
# ---------------------------------------------------------------------------

class TestCsvReaders:
    def test_read_csv_robust_utf8(self, tmp_path):
        p = tmp_path / "x.csv"
        p.write_text("a,b\n1,héllo\n", encoding="utf-8")
        df = loaders.read_csv_robust(p)
        assert df["b"].iloc[0] == "héllo"

    def test_read_csv_robust_latin1_fallback(self, tmp_path):
        p = tmp_path / "x.csv"
        # Write a byte that's invalid utf-8 but valid latin-1
        p.write_bytes(b"a,b\n1,caf\xe9\n")
        df = loaders.read_csv_robust(p)
        assert df["b"].iloc[0] == "café"

    def test_read_many_csv_concatenates_and_tags_source(self, tmp_path):
        d = _make_sackmann_dir(tmp_path, [2005, 2006])
        files = loaders.discover_sackmann_match_files(d)
        df = loaders.read_many_csv(files)
        assert len(df) == 2
        assert set(df["_source_file"]) == {
            "wta_matches_2005.csv", "wta_matches_2006.csv"
        }

    def test_read_many_csv_empty_input(self):
        df = loaders.read_many_csv([])
        assert df.empty


# ---------------------------------------------------------------------------
# Tennis-Data reading
# ---------------------------------------------------------------------------

class TestTennisDataReaders:
    def test_read_single_csv(self, tmp_path):
        d = _make_tennisdata_dir(tmp_path, [2019], ext="csv")
        path = next(d.iterdir())
        df = loaders.read_tennisdata_file(path)
        assert {"Winner", "Loser", "PSW", "PSL"}.issubset(df.columns)
        assert df["_source_file"].iloc[0] == "2019.csv"

    def test_read_single_xlsx(self, tmp_path):
        d = _make_tennisdata_dir(tmp_path, [2019], ext="xlsx")
        path = d / "2019.xlsx"
        df = loaders.read_tennisdata_file(path)
        assert df.loc[0, "Winner"] == "Test A."

    def test_read_many_concat_mixed_years(self, tmp_path):
        d = _make_tennisdata_dir(tmp_path, [2018, 2019], ext="csv")
        files = loaders.discover_tennisdata_files(d)
        df = loaders.read_many_tennisdata(files)
        assert len(df) == 2

    def test_unsupported_extension_raises(self, tmp_path):
        p = tmp_path / "2020.parquet"
        p.write_bytes(b"")
        with pytest.raises(ValueError):
            loaders.read_tennisdata_file(p)
