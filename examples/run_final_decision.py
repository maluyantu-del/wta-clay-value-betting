"""Final-decision run: train, calibrate (calib_holdout), evaluate on valid + test.

This is the project's "alpha real or not" deliverable. Splits:

    Train:  2009-2016  (model fitting only)
    Calib:  2017-2018  (calibration fit only — model never sees these)
    Valid:  2019-2021  (early-stopping target + calibration apply set)
    Test:   2022-2024  (FINAL OOS evaluation — break the seal once)

Deliverables:
  1. Vig sanity drop count (already filtered upstream)
  2. 4-set row counts + date ranges + y=1 rate
  3. 4 baselines:
       - train AUC (in-sample)
       - valid AUC (early-stopping)
       - calib AUC (OOS, what calibrator fits on)
       - train-vs-valid AUC gap (overfit indicator)
       - calib mean_pred vs y_calib (pre/post)
  4. Mini-backtest on VALID  (post-calibration)  — first credible OOS number
  5. Mini-backtest on TEST   (post-calibration)  — most authoritative number
  6. Critical: Baseline C TEST ROI → judgment matrix
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

from wta_clay.models.calibration import (
    DEFAULT_HOLDOUT_YEARS,
    apply_to_test_all,
    calibrate_baselines,
    format_calibration_comparison,
    format_calibration_report,
    format_test_eval,
    format_test_ev_buckets,
)
from wta_clay.models.lgbm import format_baseline_report, run_baselines

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("final_decision")

PD_OPTS = (
    "display.width", 220,
    "display.max_columns", None,
    "display.float_format", lambda x: f"{x:.4f}",
)


def _split_summary_table(master: pd.DataFrame) -> pd.DataFrame:
    """Show n / date range / y=1 rate / Pinnacle-vig for each of the 4 splits."""
    sets = [
        ("train", (2009, 2016)),
        ("calib", (2017, 2018)),
        ("valid", (2019, 2021)),
        ("test",  (2022, 2024)),  # 2025 not yet in data
    ]
    rows = []
    for name, (lo, hi) in sets:
        m = master[master["match_date"].dt.year.between(lo, hi)]
        bettable = m[
            (m["market_source"].astype(str) != "none")
            & m["odds_p1_pinnacle"].notna()
            & m["odds_p2_pinnacle"].notna()
        ]
        impl = (1.0 / bettable["odds_p1_pinnacle"]
                + 1.0 / bettable["odds_p2_pinnacle"])
        rows.append({
            "split":      name,
            "year_range": f"{lo}-{hi}",
            "n_total":    int(len(m)),
            "n_bettable": int(len(bettable)),
            "date_min":   str(m["match_date"].min().date()) if len(m) else "—",
            "date_max":   str(m["match_date"].max().date()) if len(m) else "—",
            "y_rate":     round(float(m["y"].mean()), 4) if len(m) else None,
            "vig_mean":   round(float(impl.mean()), 4) if len(impl) else None,
        })
    return pd.DataFrame(rows)


def _baseline_diagnostics(baselines: dict) -> pd.DataFrame:
    """Per-baseline AUCs on each split + train-valid overfit gap."""
    rows = []
    for label, tr in baselines.items():
        train_p = tr.train_predictions
        calib_p = tr.calib_predictions
        valid_p = tr.valid_predictions
        test_p  = tr.test_predictions

        def _auc(p):
            if p is None or p.empty:
                return None
            y, pp = p["y_true"].to_numpy(int), p["y_pred_prob"].to_numpy(float)
            if len(np.unique(y)) < 2:
                return None
            return float(roc_auc_score(y, pp))

        train_auc = _auc(train_p)
        calib_auc = _auc(calib_p)
        valid_auc = _auc(valid_p)
        test_auc  = _auc(test_p)
        rows.append({
            "baseline":         label,
            "n_train":          int(len(train_p)) if train_p is not None else 0,
            "n_calib":          int(len(calib_p)) if calib_p is not None else 0,
            "n_valid":          int(len(valid_p)) if valid_p is not None else 0,
            "n_test":           int(len(test_p))  if test_p  is not None else 0,
            "auc_train":        round(train_auc, 4) if train_auc is not None else None,
            "auc_calib":        round(calib_auc, 4) if calib_auc is not None else None,
            "auc_valid":        round(valid_auc, 4) if valid_auc is not None else None,
            "auc_test":         round(test_auc,  4) if test_auc  is not None else None,
            "auc_train_minus_valid":
                round(train_auc - valid_auc, 4)
                if (train_auc is not None and valid_auc is not None)
                else None,
            "auc_calib_minus_valid":
                round(calib_auc - valid_auc, 4)
                if (calib_auc is not None and valid_auc is not None)
                else None,
            "best_iter":        tr.config.get("best_iteration"),
        })
    return pd.DataFrame(rows)


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

    # ====================================================================
    # 1. Split summary
    # ====================================================================
    print()
    print("=" * 78)
    print(" 1. 4-SET SPLIT SUMMARY (years / n / y_rate / vig)")
    print("=" * 78)
    summary = _split_summary_table(master)
    with pd.option_context(*PD_OPTS):
        print(summary.to_string(index=False))
    print()
    print("Note: vig sanity drop @ oddsfeed dropped 6 rows upstream "
          "(see rebuild_pipeline log).")

    # ====================================================================
    # 2. Train all 4 baselines
    # ====================================================================
    log.info("\n========== Training 4 baselines ==========")
    baselines = run_baselines(master, feature_parquets, verbose=0)

    # ====================================================================
    # 3. Per-baseline diagnostics
    # ====================================================================
    print()
    print("=" * 78)
    print(" 2. BASELINE DIAGNOSTICS — AUC across train / calib / valid / test")
    print("=" * 78)
    diag = _baseline_diagnostics(baselines)
    with pd.option_context(*PD_OPTS):
        print(diag.to_string(index=False))
    print()
    print("  auc_train_minus_valid  → overfit indicator (healthy: 0.02-0.05)")
    print("  auc_calib_minus_valid  → calib vs valid distribution shift")

    # ====================================================================
    # 4. Calibrate via calib_holdout (the production strategy)
    # ====================================================================
    log.info("\n========== Calibrating: calib_holdout (years %s) ==========",
             DEFAULT_HOLDOUT_YEARS)
    cal = calibrate_baselines(
        baselines, master, holdout_strategy="calib_holdout",
    )

    # Per-baseline calibration reports
    print()
    print("=" * 78)
    print(" 3. CALIBRATION REPORTS (strategy = calib_holdout)")
    print("=" * 78)
    for label, result in cal.items():
        print()
        print(format_calibration_report(label, result))

    # Wide pre/post on valid
    print()
    print("=" * 78)
    print(" 4. VALID PRE/POST — first credible OOS ROI (calib_holdout)")
    print("=" * 78)
    valid_cmp = format_calibration_comparison(cal)
    with pd.option_context(*PD_OPTS):
        print(valid_cmp.to_string(index=False))

    # ====================================================================
    # 5. Apply to TEST set — break the seal ONCE
    # ====================================================================
    log.info("\n========== Apply calibrator to TEST (years 2022-2024) ==========")
    test_eval = apply_to_test_all(baselines, cal, master)

    print()
    print("=" * 78)
    print(" 5. TEST PRE/POST — final OOS ROI (the authoritative number)")
    print("=" * 78)
    test_df = format_test_eval(test_eval)
    with pd.option_context(*PD_OPTS):
        print(test_df.to_string(index=False))

    print()
    print("=" * 78)
    print(" 6. TEST EV-BUCKET DISTRIBUTION (post-calibration)")
    print("=" * 78)
    bucket_df = format_test_ev_buckets(test_eval)
    with pd.option_context(*PD_OPTS):
        print(bucket_df.to_string(index=False))

    # ====================================================================
    # 6. 10-bucket reliability data on VALID (calibrator's diagnostic)
    # ====================================================================
    print()
    print("=" * 78)
    print(" 7. 10-BUCKET CALIBRATION PLOT DATA (valid set)")
    print("=" * 78)
    for label, r in cal.items():
        print()
        print(f"--- {label} ---")
        with pd.option_context(*PD_OPTS):
            print(r.plot_data.to_string(index=False))

    # ====================================================================
    # 7. Critical numbers + decision matrix
    # ====================================================================
    print()
    print("=" * 78)
    print(" 8. CRITICAL NUMBERS — VALID vs TEST (project-decision row)")
    print("=" * 78)
    print(f"  {'baseline':<22s}  {'ROI valid':>10s}  "
          f"{'ROI test':>10s}  {'gap':>8s}  decision")
    for label in baselines:
        roi_v = cal[label].post_mini_backtest.flat_roi_pct
        roi_t = test_eval[label].post_mini_backtest.flat_roi_pct
        gap = (roi_v - roi_t) if pd.notna(roi_v) and pd.notna(roi_t) else float("nan")

        decision = ""
        if label.startswith("C_") and pd.notna(roi_t):
            if 0.0 <= roi_t <= 4.0:
                decision = "  <-- ALPHA CONFIRMED, proceed to leakage_checks + backtest"
            elif -2.0 <= roi_t < 0.0:
                decision = "  <-- borderline; ablation + tuning"
            elif roi_t < -2.0:
                decision = "  <-- features need rework, recheck independence"
            elif roi_t > 5.0:
                decision = "  <-- SUSPICIOUS; fallback debug"

        print(f"  {label:<22s}  {roi_v:+9.2f}%  {roi_t:+9.2f}%  "
              f"{gap:+7.2f}{decision}")

    # ====================================================================
    # 8. Sanity warns surfaced
    # ====================================================================
    print()
    print("=" * 78)
    print(" 9. CALIBRATION SANITY LOG SUMMARY")
    print("=" * 78)
    for label, r in cal.items():
        warns = [s for s in r.sanity_log if "[WARN]" in s]
        if warns:
            print(f"  {label}: {len(warns)} WARN(s)")
            for w in warns:
                print(f"    {w}")
        else:
            print(f"  {label}: all sanity checks OK")


if __name__ == "__main__":
    main()
