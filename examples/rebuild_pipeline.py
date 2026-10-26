"""Rebuild master + 6 features from raw data.

This script is run when upstream changes invalidate downstream artefacts:

  - oddsfeed.py vig-sanity filter added (drops bad-data rows like
    2020-09-21 odds=52.0/2.77)
  - splits/lgbm/calibration refactored for the 4-set design
    (train / calib / valid / test)

The order is fixed:
  1. Tennis-Data odds (oddsfeed)
  2. Sackmann main-draw matches + qual/ITF
  3. Sackmann all-surface matches (for h2h/specialist/fatigue)
  4. Sackmann rankings (for ranking module)
  5. Sackmann players (for DOB-based age + dominant-hand)
  6. Name-matcher TD ↔ Sackmann
  7. Master build via merge.build_master_dataset → master.parquet
  8. Features in dependency order: clay_elo → form / h2h / specialist / ranking / fatigue
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from wta_clay.data import loaders, merge, name_matcher, oddsfeed, sackmann
from wta_clay.features import (
    clay_elo as f_clay_elo,
    fatigue   as f_fatigue,
    form      as f_form,
    h2h       as f_h2h,
    ranking   as f_ranking,
    specialist as f_specialist,
)
from wta_clay.features.io import write_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("rebuild_pipeline")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw"
PROCESSED = ROOT / "data/processed"
FEATURES = PROCESSED / "features"


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    FEATURES.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1 — Tennis-Data odds (will apply vig sanity filter inline)
    # ------------------------------------------------------------------
    log.info("Step 1/8 — load Tennis-Data odds (with vig sanity filter)")
    odds = oddsfeed.load_tennisdata_odds(
        RAW / "tennis_data", year_from=2007, year_to=None, clay_only=True,
    )
    log.info("odds: %d rows × %d cols  (%s → %s)",
             len(odds), odds.shape[1],
             odds["match_date"].min().date(), odds["match_date"].max().date())

    # ------------------------------------------------------------------
    # Step 2 — Sackmann main+ITF (for matched build)
    # ------------------------------------------------------------------
    log.info("Step 2/8 — load Sackmann matches (main + qual/ITF)")
    sk_full = sackmann.load_sackmann_matches(
        RAW / "sackmann", year_from=2005, include_qual_itf=True,
    )
    sk_clay = sackmann.filter_clay(sk_full)
    sk_clay = sk_clay.reset_index(drop=True)
    log.info("sackmann clay (main+ITF): %d rows", len(sk_clay))

    # ------------------------------------------------------------------
    # Step 3 — Sackmann all-surface (main only, used by h2h/specialist/fatigue)
    # ------------------------------------------------------------------
    log.info("Step 3/8 — load Sackmann all-surface (main only)")
    sk_all = sackmann.load_sackmann_matches(
        RAW / "sackmann", year_from=2005, include_qual_itf=False,
    )
    log.info("sackmann all-surface (main): %d rows", len(sk_all))

    # ------------------------------------------------------------------
    # Step 4 — Sackmann rankings
    # ------------------------------------------------------------------
    log.info("Step 4/8 — load Sackmann rankings")
    rankings = sackmann.load_rankings(RAW / "sackmann")
    log.info("rankings: %d rows", len(rankings))

    # ------------------------------------------------------------------
    # Step 5 — Sackmann players (for DOB)
    # ------------------------------------------------------------------
    log.info("Step 5/8 — load Sackmann players")
    players = sackmann.load_players(RAW / "sackmann")
    log.info("players: %d rows", len(players))

    # ------------------------------------------------------------------
    # Step 6 — Name match TD ↔ Sackmann
    # ------------------------------------------------------------------
    log.info("Step 6/8 — name match TD ↔ Sackmann")
    alias_path = ROOT / "config/tournament_alias.csv"
    nm = name_matcher.match_td_to_sackmann(
        odds, sk_clay, alias_path, rankings=rankings,
    )
    matched = nm.matched
    log.info("name match: %d matched / %d total TD rows  (%.1f%%)",
             len(matched), len(odds), 100 * len(matched) / max(len(odds), 1))

    # ------------------------------------------------------------------
    # Step 7 — Build master + write
    # ------------------------------------------------------------------
    log.info("Step 7/8 — build master")
    master_path = PROCESSED / "clay_matches_master.parquet"
    mr = merge.build_master_dataset(
        matched=matched, sackmann=sk_clay, players=players,
        out_path=master_path, run_self_checks=True,
    )
    master = mr.master
    log.info("master: %d rows × %d cols", len(master), master.shape[1])
    log.info("Year coverage: %s",
             master["match_date"].dt.year.value_counts().sort_index().to_dict())
    log.info("y=1 rate by year (post-build):")
    for yr, g in master.groupby(master["match_date"].dt.year):
        log.info("  %d: n=%d  y=%.4f", int(yr), len(g), float(g["y"].mean()))

    # ------------------------------------------------------------------
    # Step 8 — Features (in dependency order)
    # ------------------------------------------------------------------
    log.info("Step 8/8 — features")

    # 8a. clay_elo
    log.info("  8a. clay_elo")
    elo = f_clay_elo.compute_clay_elo(
        master, sackmann_seed=sk_full, players=players,
    )
    write_features(elo.features, master,
                   path=FEATURES / "clay_elo.parquet",
                   feature_module="clay_elo")

    # 8b. form
    log.info("  8b. form")
    fm = f_form.compute_form_features(master, sk_clay, players=players)
    write_features(fm.features, master,
                   path=FEATURES / "form.parquet",
                   feature_module="form")

    # 8c. h2h
    log.info("  8c. h2h")
    h2h = f_h2h.compute_h2h_features(master, sk_all,
                                       clay_elo_features=elo.features,
                                       players=players)
    write_features(h2h.features, master,
                   path=FEATURES / "h2h.parquet",
                   feature_module="h2h")

    # 8d. specialist
    log.info("  8d. specialist")
    sp = f_specialist.compute_specialist_features(
        master, sk_all, clay_elo_features=elo.features, players=players,
    )
    write_features(sp.features, master,
                   path=FEATURES / "specialist.parquet",
                   feature_module="specialist")

    # 8e. ranking
    log.info("  8e. ranking")
    rk = f_ranking.compute_ranking_features(
        master, rankings, clay_elo_features=elo.features, players=players,
    )
    write_features(rk.features, master,
                   path=FEATURES / "ranking.parquet",
                   feature_module="ranking")

    # 8f. fatigue
    log.info("  8f. fatigue")
    fa = f_fatigue.compute_fatigue_features(
        master, sk_all, clay_elo_features=elo.features, players=players,
    )
    write_features(fa.features, master,
                   path=FEATURES / "fatigue.parquet",
                   feature_module="fatigue")

    log.info("=" * 60)
    log.info("DONE — master + 6 features written to %s", PROCESSED)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
