"""Minimal runnable demo for wta_clay.data.oddsfeed.

If real Tennis-Data.co.uk files are present in data/raw/tennis_data/, runs
end-to-end on them and prints bookmaker coverage by year. Otherwise
falls back to a synthetic fixture so the demo is always executable.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from wta_clay.data import oddsfeed

TD_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "tennis_data"


# Column set that matches the real tennis-data.co.uk WTA file schema.
SCHEMA_MODERN = [
    "Date", "Tournament", "Series", "Court", "Surface", "Round", "Best of",
    "Winner", "Loser", "WRank", "LRank",
    "PSW", "PSL", "B365W", "B365L", "MaxW", "MaxL", "AvgW", "AvgL",
    "Comment",
]
SCHEMA_LEGACY = [  # pre-Pinnacle
    "Date", "Tournament", "Tier", "Court", "Surface", "Round", "Best of",
    "Winner", "Loser", "WRank", "LRank", "B365W", "B365L", "Comment",
]


def _build_synthetic(tmp: Path) -> Path:
    d = tmp / "tennis_data"
    d.mkdir()
    # 2006 — legacy, no Pinnacle
    pd.DataFrame([
        {"Date": "15/05/2006", "Tournament": "Rome", "Tier": "Tier I",
         "Court": "Outdoor", "Surface": "Clay", "Round": "Quarterfinals",
         "Best of": 3, "Winner": "Henin J.", "Loser": "Kuznetsova S.",
         "WRank": 1, "LRank": 5, "B365W": 1.40, "B365L": 2.80,
         "Comment": "Completed"},
    ], columns=SCHEMA_LEGACY).to_csv(d / "2006.csv", index=False)
    # 2019 — modern, mix of clay + hard + walkover
    pd.DataFrame([
        {"Date": "20/05/2019", "Tournament": "Rome", "Series": "Premier 5",
         "Court": "Outdoor", "Surface": "Clay", "Round": "Semifinals",
         "Best of": 3, "Winner": "Pliskova K.", "Loser": "Konta J.",
         "WRank": 7, "LRank": 42, "PSW": 1.85, "PSL": 1.95,
         "B365W": 1.80, "B365L": 2.00, "MaxW": 1.90, "MaxL": 2.05,
         "AvgW": 1.82, "AvgL": 1.98, "Comment": "Completed"},
        {"Date": "21/05/2019", "Tournament": "Nuremberg", "Series": "International",
         "Court": "Outdoor", "Surface": "Clay", "Round": "The Final",
         "Best of": 3, "Winner": "Kontaveit A.", "Loser": "Putintseva Y.",
         "WRank": 23, "LRank": 40, "PSW": 1.65, "PSL": 2.25,
         "B365W": 1.62, "B365L": 2.30, "MaxW": 1.68, "MaxL": 2.35,
         "AvgW": 1.64, "AvgL": 2.28, "Comment": "Completed"},
        {"Date": "25/05/2019", "Tournament": "Rome", "Series": "Premier 5",
         "Court": "Outdoor", "Surface": "Hard", "Round": "Quarterfinals",
         "Best of": 3, "Winner": "X Y.", "Loser": "W Z.",
         "WRank": 1, "LRank": 2, "PSW": 1.9, "PSL": 1.9,
         "B365W": 1.9, "B365L": 1.9, "MaxW": 2.0, "MaxL": 2.0,
         "AvgW": 1.9, "AvgL": 1.9, "Comment": "Completed"},
        {"Date": "26/05/2019", "Tournament": "Walkover Cup",
         "Series": "International", "Court": "Outdoor", "Surface": "Clay",
         "Round": "2nd Round", "Best of": 3,
         "Winner": "Pavlyuchenkova A.", "Loser": "Svitolina E.",
         "WRank": 15, "LRank": 10, "PSW": 2.5, "PSL": 1.55,
         "B365W": 2.4, "B365L": 1.60, "MaxW": 2.6, "MaxL": 1.65,
         "AvgW": 2.45, "AvgL": 1.58, "Comment": "Walkover"},
    ], columns=SCHEMA_MODERN).to_csv(d / "2019.csv", index=False)
    return d


def main() -> None:
    if TD_DIR.exists() and any(TD_DIR.iterdir()):
        print(f"Using real Tennis-Data files at {TD_DIR}")
        src_dir = TD_DIR
    else:
        print(f"(No real files at {TD_DIR} — running on synthetic fixture.)")
        tmp = Path(tempfile.mkdtemp())
        src_dir = _build_synthetic(tmp)

    df = oddsfeed.load_tennisdata_odds(src_dir, clay_only=True,
                                       drop_walkover=True)

    print(f"\nLoaded {len(df):,} clay rows, "
          f"{df['match_date'].min()} → {df['match_date'].max()}")

    print("\n=== Sample rows (first 5) ===")
    print(df.head(5).to_string())

    print("\n=== Bookmaker coverage by year ===")
    cov = oddsfeed.audit_bookmaker_coverage(df)
    print(cov.to_string())

    print("\n=== Round distribution ===")
    print(df["round_norm"].value_counts(dropna=False).to_string())

    print("\n=== Unknown round strings (needs alias if non-empty) ===")
    unk = oddsfeed.audit_unknown_rounds(df)
    if len(unk):
        print(unk.to_string())
    else:
        print("(none)")


if __name__ == "__main__":
    main()
