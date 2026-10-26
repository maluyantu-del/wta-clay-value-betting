"""Minimal runnable demo for wta_clay.data.loaders.

Usage (from project root):
    python examples/demo_loaders.py

The demo creates a small synthetic Sackmann + Tennis-Data layout in a
temporary directory and exercises every public loader function. No real
data downloads required.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from wta_clay.data import loaders


def build_synthetic_sackmann(root: Path) -> Path:
    d = root / "sackmann"
    d.mkdir(parents=True)
    for year in (2005, 2010, 2020, 2024):
        pd.DataFrame({
            "tourney_id": [f"{year}-RG"],
            "tourney_name": ["Roland Garros"],
            "surface": ["Clay"],
            "tourney_level": ["G"],
            "tourney_date": [year * 10000 + 528],
            "winner_id": [100], "winner_name": ["Alice Example"],
            "loser_id": [200], "loser_name": ["Bob Example"],
            "score": ["6-3 6-4"],
        }).to_csv(d / f"wta_matches_{year}.csv", index=False)
        pd.DataFrame({"tourney_id": [f"Q-{year}"]}).to_csv(
            d / f"wta_matches_qual_itf_{year}.csv", index=False
        )
    pd.DataFrame({
        "player_id": [100, 200],
        "name_first": ["Alice", "Bob"],
        "name_last": ["Example", "Example"],
    }).to_csv(d / "wta_players.csv", index=False)
    pd.DataFrame({
        "ranking_date": [20200106],
        "rank": [10], "player": [100], "points": [2000],
    }).to_csv(d / "wta_rankings_20s.csv", index=False)
    return d


def build_synthetic_tennisdata(root: Path) -> Path:
    d = root / "tennis_data"
    d.mkdir(parents=True)
    for year in (2005, 2019, 2024):
        pd.DataFrame({
            "Date": [f"{year}-05-28"],
            "Tournament": ["French Open"],
            "Surface": ["Clay"],
            "Winner": ["Example A."],
            "Loser": ["Example B."],
            "PSW": [1.80], "PSL": [2.05],
            "B365W": [1.75], "B365L": [2.10],
        }).to_csv(d / f"{year}.csv", index=False)
    return d


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sack_dir = build_synthetic_sackmann(root)
        odds_dir = build_synthetic_tennisdata(root)

        print("=" * 60)
        print("Sackmann main-draw discovery (2010–2020)")
        print("=" * 60)
        main_files = loaders.discover_sackmann_match_files(
            sack_dir, year_from=2010, year_to=2020
        )
        for f in main_files:
            print("  ", f.name)

        print("\nSackmann qual+ITF discovery (all years):")
        for f in loaders.discover_sackmann_qual_itf_files(sack_dir):
            print("  ", f.name)

        print("\nSackmann rankings discovery:")
        for f in loaders.discover_sackmann_ranking_files(sack_dir):
            print("  ", f.name)

        print("\nPlayers file:", loaders.sackmann_players_path(sack_dir).name)

        print("\n" + "=" * 60)
        print("Read & concat main-draw matches")
        print("=" * 60)
        matches = loaders.read_many_csv(main_files)
        print(matches[["tourney_date", "winner_name", "loser_name",
                       "_source_file"]].to_string(index=False))

        print("\n" + "=" * 60)
        print("Tennis-Data discovery + read")
        print("=" * 60)
        td_files = loaders.discover_tennisdata_files(odds_dir, year_from=2019)
        print("Selected files:", [f.name for f in td_files])
        odds = loaders.read_many_tennisdata(td_files)
        print(odds.to_string(index=False))

        print("\nAll loaders exercised successfully.")


if __name__ == "__main__":
    main()
