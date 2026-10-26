"""Minimal runnable demo for wta_clay.data.sackmann.

Runs on the real cloned Sackmann repo at data/raw/sackmann/ and prints:
  1. Summary of the tier classification on the full clay dataset (2005+)
  2. Audit crosstab (raw_level × year) — confirms pre-2009 Tier mapping
  3. A sanity look at Premier 5 upgrades (should include Rome)
  4. A sample of Fed Cup / BJK Cup rows that were dropped

If the real data is not present, falls back to a tiny synthetic demo.
"""
from __future__ import annotations

from pathlib import Path

from wta_clay.data import sackmann


SACK_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "sackmann"


def main() -> None:
    if not SACK_DIR.exists() or not any(SACK_DIR.glob("wta_matches_*.csv")):
        print(f"(No data at {SACK_DIR} — please `git clone` tennis_wta first.)")
        return

    print("Loading Sackmann matches 2005+ (main draw + qual/ITF)…")
    df = sackmann.load_sackmann_matches(
        SACK_DIR, year_from=2005, include_qual_itf=True
    )
    clay = sackmann.filter_clay(df)

    print(f"\nTotal rows:         {len(df):>8,}")
    print(f"Clay rows:          {len(clay):>8,}")
    print(f"Not-for-betting:    {clay['not_for_betting'].sum():>8,}"
          "  (Olympics — train-only)")

    print("\n=== Clay tour_level distribution ===")
    print(clay["tour_level"].value_counts().to_string())

    print("\n=== Clay sample_weight distribution ===")
    print(clay["sample_weight"].value_counts().to_string())

    print("\n=== Clay raw_level × year (audit table) ===")
    audit = sackmann.describe_tour_levels(clay, by_year=True)
    print(audit.to_string())

    print("\n=== Premier 5 upgrade sanity-check (P-coded → WTA1000 by name) ===")
    upgraded = clay[(clay["raw_level"] == "P") & (clay["tour_level"] == "WTA1000")]
    print(f"Upgraded rows: {len(upgraded):,}")
    if len(upgraded):
        print("Sample tourney_names (unique):")
        for name in upgraded["tourney_name"].drop_duplicates().head(10):
            print(f"  - {name}")

    print("\n=== Premier kept as WTA500 (sanity) ===")
    p_500 = clay[(clay["raw_level"] == "P") & (clay["tour_level"] == "WTA500")]
    print(f"Rows: {len(p_500):,}")
    for name in p_500["tourney_name"].drop_duplicates().head(10):
        print(f"  - {name}")

    print("\n=== Fed Cup rows are gone (should print 0) ===")
    print((clay["raw_level"] == "D").sum())


if __name__ == "__main__":
    main()
