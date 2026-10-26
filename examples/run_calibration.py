"""Run vig-check + 4 baselines + dual-strategy calibration on real data.

Deliverables (per locked spec):
  1. Empirical Pinnacle vig: overall + distribution + by-year
  2. 4-baseline ROI under valid_self (in-sample upper bound)
  3. 4-baseline ROI under train_tail (holdout — production proxy)
  4. valid_self vs train_tail comparison table (in-sample optimism)
  5. EV-bucket comparison table (per baseline × strategy)
  6. 10-bucket calibration plot data (train_tail strategy)
  7. Critical number: Baseline C train_tail post-calibration ROI
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from wta_clay.models.calibration import (
    DEFAULT_HOLDOUT_YEARS,
    calibrate_baselines,
    format_calibration_comparison,
    format_calibration_report,
    format_ev_bucket_comparison,
    format_strategy_comparison,
)
from wta_clay.models.lgbm import run_baselines

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("calibration_run")

PD_OPTS = (
    "display.width", 220,
    "display.max_columns", None,
    "display.float_format", lambda x: f"{x:.4f}",
)


# ---------------------------------------------------------------------------
# 1. Vig check
# ---------------------------------------------------------------------------

def vig_check(master: pd.DataFrame) -> None:
    """Compute and print Pinnacle implied-vig statistics."""
    print()
    print("=" * 78)
    print(" 1. PINNACLE VIG CHECK (post-mini-backtest filters)")
    print("=" * 78)

    m = master.copy()
    m = m[m["market_source"].astype(str) != "none"]
    m = m[m["match_date"].dt.year != 2009]
    m = m[m["odds_p1_pinnacle"].notna() & m["odds_p2_pinnacle"].notna()]
    implied = 1.0 / m["odds_p1_pinnacle"] + 1.0 / m["odds_p2_pinnacle"]

    print(f"  Overall (n={len(m):,}):")
    print(f"    mean    = {implied.mean():.4f}    (Pinnacle expected ~1.02-1.05)")
    print(f"    std     = {implied.std():.4f}")
    print(f"    min     = {implied.min():.4f}")
    print(f"    p10     = {implied.quantile(0.10):.4f}")
    print(f"    median  = {implied.median():.4f}")
    print(f"    p90     = {implied.quantile(0.90):.4f}")
    print(f"    max     = {implied.max():.4f}")

    n_neg = int((implied < 1.0).sum())
    n_high = int((implied > 1.10).sum())
    print(f"\n    rows implied < 1.00:  {n_neg}  ({n_neg/len(m)*100:.2f}%)  "
          f"<-- DATA-ERROR CANDIDATES")
    print(f"    rows implied > 1.10:  {n_high}  ({n_high/len(m)*100:.2f}%)")

    if n_neg > 0:
        bad = m.assign(implied=implied)[m.assign(implied=implied)["implied"] < 1.0]
        bad = bad.sort_values("implied").head(5)
        print(f"\n    Top {len(bad)} bad-vig rows:")
        cols = ["match_date", "tourney_id", "p1_id", "p2_id", "y",
                "odds_p1_pinnacle", "odds_p2_pinnacle", "implied"]
        for _, r in bad.iterrows():
            print(f"      {r['match_date'].date()}  {r['tourney_id']:<28s}  "
                  f"odds={r['odds_p1_pinnacle']:.2f}/{r['odds_p2_pinnacle']:.2f}  "
                  f"y={int(r['y'])}  implied={r['implied']:.4f}")

    # By year
    print("\n  By year:")
    print("    year    n    mean    p10    p50    p90")
    m_yr = m.assign(yr=m["match_date"].dt.year)
    for yr, g in m_yr.groupby("yr"):
        impl = 1.0 / g["odds_p1_pinnacle"] + 1.0 / g["odds_p2_pinnacle"]
        print(f"    {yr}  {len(g):>4}  {impl.mean():.4f}  "
              f"{impl.quantile(0.10):.4f}  {impl.median():.4f}  "
              f"{impl.quantile(0.90):.4f}")

    # Valid + train-tail subsets
    print("\n  Valid window (2019-2021) — what calibration acts on:")
    g_v = m_yr[m_yr["yr"].between(2019, 2021)]
    impl_v = 1.0 / g_v["odds_p1_pinnacle"] + 1.0 / g_v["odds_p2_pinnacle"]
    print(f"    n = {len(g_v)}, mean = {impl_v.mean():.4f}, "
          f"p10/p50/p90 = {impl_v.quantile(0.10):.4f}/"
          f"{impl_v.median():.4f}/{impl_v.quantile(0.90):.4f}")

    print("\n  Train-tail window (2017-2018) — calibration FIT set:")
    g_t = m_yr[m_yr["yr"].between(2017, 2018)]
    impl_t = 1.0 / g_t["odds_p1_pinnacle"] + 1.0 / g_t["odds_p2_pinnacle"]
    print(f"    n = {len(g_t)}, mean = {impl_t.mean():.4f}, "
          f"p10/p50/p90 = {impl_t.quantile(0.10):.4f}/"
          f"{impl_t.median():.4f}/{impl_t.quantile(0.90):.4f}")


# ---------------------------------------------------------------------------
# 2. Headline pre/post comparison
# ---------------------------------------------------------------------------

def print_comparison(label: str, results: dict) -> None:
    print()
    print("=" * 78)
    print(f" {label}")
    print("=" * 78)
    df = format_calibration_comparison(results)
    with pd.option_context(*PD_OPTS):
        print(df.to_string(index=False))


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    master_path  = root / "data/processed/clay_matches_master.parquet"
    feat_dir     = root / "data/processed/features"
    feature_parquets = {
        "clay_elo":   feat_dir / "clay_elo.parquet",
        "form":       feat_dir / "form.parquet",
        "h2h":        feat_dir / "h2h.parquet",
        "specialist": feat_dir / "specialist.parquet",
        "ranking":    feat_dir / "ranking.parquet",
        "fatigue":    feat_dir / "fatigue.parquet",
    }

    log.info("Loading master from %s", master_path)
    master = pd.read_parquet(master_path)
    log.info("master: %d rows × %d cols", len(master), master.shape[1])

    # =========================================================================
    # 1. Vig check
    # =========================================================================
    vig_check(master)

    # =========================================================================
    # 2. Train all 4 baselines
    # =========================================================================
    log.info("\n========== Running 4 baselines ==========")
    baselines = run_baselines(master, feature_parquets, verbose=0)

    # =========================================================================
    # 3. Calibrate via valid_self (in-sample upper bound)
    # =========================================================================
    log.info("\n========== Calibrating: valid_self (in-sample) ==========")
    cal_self = calibrate_baselines(
        baselines, master, holdout_strategy="valid_self",
    )

    # =========================================================================
    # 4. Calibrate via train_tail (production proxy)
    # =========================================================================
    log.info("\n========== Calibrating: train_tail (holdout %s-%s) ==========",
             *DEFAULT_HOLDOUT_YEARS)
    cal_tail = calibrate_baselines(
        baselines, master, holdout_strategy="train_tail",
        holdout_years=DEFAULT_HOLDOUT_YEARS,
    )

    # =========================================================================
    # 5. Per-baseline reports — train_tail only (the trustable numbers)
    # =========================================================================
    print()
    print("=" * 78)
    print(" 2. PER-BASELINE REPORTS (strategy = train_tail)")
    print("=" * 78)
    for label, result in cal_tail.items():
        print()
        print(format_calibration_report(label, result))

    # =========================================================================
    # 6. Wide pre/post table per strategy
    # =========================================================================
    print_comparison(
        "3a. PRE/POST COMPARISON — strategy = valid_self (in-sample)",
        cal_self,
    )
    print_comparison(
        "3b. PRE/POST COMPARISON — strategy = train_tail (holdout, "
        f"fit on {DEFAULT_HOLDOUT_YEARS[0]}-{DEFAULT_HOLDOUT_YEARS[1]})",
        cal_tail,
    )

    # =========================================================================
    # 7. Strategy comparison: valid_self vs train_tail
    # =========================================================================
    print()
    print("=" * 78)
    print(" 4. STRATEGY COMPARISON — valid_self vs train_tail (per baseline)")
    print("=" * 78)
    cmp = format_strategy_comparison(cal_self, cal_tail)
    with pd.option_context(*PD_OPTS):
        print(cmp.to_string(index=False))
    print(f"\n  roi_optimism_pp = ROI(valid_self) − ROI(train_tail)")
    print(f"  Larger = more in-sample bias from valid_self.")

    # =========================================================================
    # 8. EV-bucket comparison
    # =========================================================================
    print()
    print("=" * 78)
    print(" 5. EV-BUCKET COMPARISON (per baseline × strategy)")
    print("=" * 78)
    bucket = format_ev_bucket_comparison(cal_self, cal_tail)
    with pd.option_context(*PD_OPTS):
        print(bucket.to_string(index=False))

    # =========================================================================
    # 9. 10-bucket calibration plot data — train_tail strategy
    # =========================================================================
    print()
    print("=" * 78)
    print(" 6. 10-BUCKET CALIBRATION PLOT DATA  (strategy = train_tail)")
    print("=" * 78)
    for label, result in cal_tail.items():
        print()
        print(f"--- {label} ---")
        with pd.option_context(*PD_OPTS):
            print(result.plot_data.to_string(index=False))

    # =========================================================================
    # 10. Critical numbers + decision
    # =========================================================================
    print()
    print("=" * 78)
    print(" 7. CRITICAL NUMBERS  (train_tail = trustable; valid_self = upper bound)")
    print("=" * 78)
    print(f"  {'baseline':<22s}  {'ROI valid_self':>14s}  "
          f"{'ROI train_tail':>14s}  {'optimism (pp)':>14s}  decision")
    for label in cal_tail.keys():
        roi_self = cal_self[label].post_mini_backtest.flat_roi_pct
        roi_tail = cal_tail[label].post_mini_backtest.flat_roi_pct
        opt = (roi_self - roi_tail
               if pd.notna(roi_self) and pd.notna(roi_tail)
               else float("nan"))

        decision = ""
        if label.startswith("C_") and pd.notna(roi_tail):
            if 0.0 <= roi_tail <= 4.0:
                decision = "  <-- ALPHA CONFIRMED, proceed to leakage_checks"
            elif roi_tail > 5.0:
                decision = "  <-- BUG, run fallback debug"
            elif roi_tail < -2.0:
                decision = "  <-- CALIBRATION FAILED, revisit features"
            else:
                decision = "  <-- borderline; user judgment call"

        print(f"  {label:<22s}  {roi_self:+13.2f}%  {roi_tail:+13.2f}%  "
              f"{opt:+13.2f}{decision}")

    # =========================================================================
    # 11. Sensitivity flag — wild odds row
    # =========================================================================
    print()
    print("=" * 78)
    print(" 8. KNOWN ISSUE — bad-vig row sensitivity")
    print("=" * 78)
    bad = master[(master["match_date"] == pd.Timestamp("2020-09-21"))
                  & (master["odds_p1_pinnacle"] > 40)]
    if len(bad) > 0:
        r = bad.iloc[0]
        print(f"  Row 2020-09-21 (tourney {r['tourney_id']}, match_id {r['match_id']}):")
        print(f"    odds = {r['odds_p1_pinnacle']:.2f} / {r['odds_p2_pinnacle']:.2f}, "
              f"y = {int(r['y'])}, implied vig = "
              f"{1/r['odds_p1_pinnacle'] + 1/r['odds_p2_pinnacle']:.4f}")
        # Estimate single-row contribution to ROI for each baseline (post-cal)
        for label, result in cal_tail.items():
            bets = result.post_mini_backtest.bets_df
            if len(bets) == 0:
                continue
            row = bets[bets["match_id"] == r["match_id"]]
            if len(row) == 0:
                print(f"    {label}: NOT in bet set")
                continue
            pnl = float(row.iloc[0]["pnl"])
            n_bets = result.post_mini_backtest.n_bets
            roi_share = pnl / n_bets * 100
            print(f"    {label}: bet_side={row.iloc[0]['bet_side']}  "
                  f"won={bool(row.iloc[0]['won'])}  pnl={pnl:+.2f}  "
                  f"contributes {roi_share:+.2f}pp to ROI ({n_bets} bets)")


if __name__ == "__main__":
    main()
