"""Final project deliverable: full backtest pipeline + leakage audit.

Pipeline
--------
1. Load master + 6 feature parquets
2. Train 4 baselines (A clay-Elo only, B 4-core, C all-38, D all-38+decay)
3. Calibrate via calib_holdout (2017-2018 dedicated calibration set)
4. Apply calibrators to test set (2022-2024) — first peek at test
5. Run bankroll-aware backtest (Baseline C primary, B for comparison)
6. Robustness sweep: book × ev_threshold × kelly_fraction
7. Run leakage audit on the production data + features
8. Print full report + save equity curve PNG + audit JSON
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from wta_clay.backtest.engine import run_backtest
from wta_clay.backtest.reports import (
    format_summary, plot_equity_curve,
    report_by_ev_bucket, report_by_month, report_by_odds_bucket,
    report_by_tourney_level, report_by_year, report_clay_vs_offseason,
)
from wta_clay.data import sackmann
from wta_clay.models.calibration import (
    apply_to_test_all, calibrate_baselines,
)
from wta_clay.models.leakage_checks import (
    format_audit_report, run_leakage_audit,
)
from wta_clay.models.lgbm import run_baselines

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("final_backtest")

PD_OPTS = ("display.width", 220, "display.max_columns", None,
           "display.float_format", lambda x: f"{x:.4f}")


# ===========================================================================
# Helpers
# ===========================================================================

def _build_test_predictions_df(
    train_result, cal_result, master,
) -> pd.DataFrame:
    """Build a backtest-ready DataFrame from a TestSetEvalResult.

    Columns: match_id, y_true, y_pred_prob (calibrated). The engine merges
    in match_date / odds / tier from master.
    """
    test_pred = train_result.test_predictions.copy()
    # Apply the calibrator from cal_result
    from wta_clay.models.calibration import apply_calibration
    return apply_calibration(test_pred, cal_result.calibrator)


def _print_section(title: str) -> None:
    print()
    print("=" * 78)
    print(f" {title}")
    print("=" * 78)


def _print_df(df: pd.DataFrame) -> None:
    if df.empty:
        print("  (empty)")
        return
    with pd.option_context(*PD_OPTS):
        print(df.to_string(index=False))


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    root = Path(__file__).resolve().parent.parent
    master_path = root / "data/processed/clay_matches_master.parquet"
    feat_dir    = root / "data/processed/features"
    audit_dir   = root / "data/processed/audit"
    out_dir     = root / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_parquets = {
        "clay_elo":   feat_dir / "clay_elo.parquet",
        "form":       feat_dir / "form.parquet",
        "h2h":        feat_dir / "h2h.parquet",
        "specialist": feat_dir / "specialist.parquet",
        "ranking":    feat_dir / "ranking.parquet",
        "fatigue":    feat_dir / "fatigue.parquet",
    }

    log.info("Loading master + features")
    master = pd.read_parquet(master_path)
    log.info("master: %d rows", len(master))

    # ====================================================================
    # 1. Train + calibrate
    # ====================================================================
    log.info("Training 4 baselines")
    baselines = run_baselines(master, feature_parquets, verbose=0)

    log.info("Calibrating via calib_holdout (2017-2018)")
    cal = calibrate_baselines(baselines, master,
                                holdout_strategy="calib_holdout")

    log.info("Applying calibrators to test predictions")
    test_eval = apply_to_test_all(baselines, cal, master)

    # ====================================================================
    # 2. Build test_predictions DataFrames for each baseline
    # ====================================================================
    cal_test_preds: dict[str, pd.DataFrame] = {}
    for label in baselines:
        cal_test_preds[label] = _build_test_predictions_df(
            baselines[label], cal[label], master,
        )

    # ====================================================================
    # 3. Run main backtest on Baseline C (38 features)
    # ====================================================================
    LABEL_PRIMARY = "C_all_38"
    LABEL_COMPARE = "B_4_core"

    _print_section(f"PRIMARY BACKTEST — {LABEL_PRIMARY} (Pinnacle, 1/4-Kelly, "
                    "EV>5%, max 2% per bet)")
    bt_main = run_backtest(
        cal_test_preds[LABEL_PRIMARY], master,
        initial_bankroll=10000.0,
        ev_threshold=0.05,
        kelly_fraction_arg=0.25,
        max_bet_pct=0.02,
        book="pinnacle",
    )
    print(format_summary(bt_main.summary))

    # ====================================================================
    # 4. Grouped reports — Baseline C primary
    # ====================================================================
    _print_section(f"{LABEL_PRIMARY} — REPORT BY TOURNEY LEVEL")
    _print_df(report_by_tourney_level(bt_main.trades))

    _print_section(f"{LABEL_PRIMARY} — REPORT BY ODDS BUCKET")
    _print_df(report_by_odds_bucket(bt_main.trades))

    _print_section(f"{LABEL_PRIMARY} — REPORT BY EV BUCKET")
    _print_df(report_by_ev_bucket(bt_main.trades))

    _print_section(f"{LABEL_PRIMARY} — REPORT BY YEAR")
    _print_df(report_by_year(bt_main.trades))

    _print_section(f"{LABEL_PRIMARY} — REPORT BY MONTH")
    _print_df(report_by_month(bt_main.trades))

    _print_section(f"{LABEL_PRIMARY} — CLAY-SEASON VS OFF-SEASON")
    _print_df(report_clay_vs_offseason(bt_main.trades))

    # ====================================================================
    # 5. Equity curve plot
    # ====================================================================
    eq_path = out_dir / f"equity_curve_{LABEL_PRIMARY}.png"
    plot_equity_curve(bt_main.equity_curve, eq_path,
                       title=f"{LABEL_PRIMARY} backtest equity curve "
                             f"(test 2022-2024)",
                       initial_bankroll=10000)
    log.info("Equity curve saved → %s", eq_path)

    # ====================================================================
    # 6. Comparison: Baseline B
    # ====================================================================
    _print_section(f"COMPARISON BACKTEST — {LABEL_COMPARE} (4-core features)")
    bt_cmp = run_backtest(
        cal_test_preds[LABEL_COMPARE], master,
        initial_bankroll=10000.0,
        ev_threshold=0.05,
        kelly_fraction_arg=0.25,
        max_bet_pct=0.02,
        book="pinnacle",
    )
    print(format_summary(bt_cmp.summary))
    eq_path_b = out_dir / f"equity_curve_{LABEL_COMPARE}.png"
    plot_equity_curve(bt_cmp.equity_curve, eq_path_b,
                       title=f"{LABEL_COMPARE} backtest equity curve",
                       initial_bankroll=10000)

    # ====================================================================
    # 7. Robustness — book / ev_threshold / kelly_fraction sensitivity
    # ====================================================================
    _print_section(f"ROBUSTNESS SWEEP — {LABEL_PRIMARY}")

    # 7a. Book sensitivity
    rows = []
    for book in ("pinnacle", "b365", "best"):
        bt = run_backtest(cal_test_preds[LABEL_PRIMARY], master, book=book)
        rows.append({
            "book": book,
            "n_bets":    bt.summary["n_bets"],
            "roi_per_bet_%":     round((bt.summary["roi_per_bet"] or 0) * 100, 2),
            "total_return_%":    round(bt.summary["total_return_pct"], 2),
            "max_drawdown_%":    round(bt.summary["max_drawdown_pct"], 2),
            "sharpe":            round(bt.summary["sharpe"], 3) if pd.notna(bt.summary["sharpe"]) else None,
            "final_bankroll":    round(bt.summary["final_bankroll"], 0),
        })
    print("\n  Book sensitivity:")
    _print_df(pd.DataFrame(rows))

    # 7b. EV threshold sensitivity
    rows = []
    for thr in (0.03, 0.05, 0.07, 0.10):
        bt = run_backtest(cal_test_preds[LABEL_PRIMARY], master,
                            ev_threshold=thr)
        rows.append({
            "ev_threshold":      f"{thr*100:.0f}%",
            "n_bets":            bt.summary["n_bets"],
            "roi_per_bet_%":     round((bt.summary["roi_per_bet"] or 0) * 100, 2),
            "total_return_%":    round(bt.summary["total_return_pct"], 2),
            "max_drawdown_%":    round(bt.summary["max_drawdown_pct"], 2),
            "sharpe":            round(bt.summary["sharpe"], 3) if pd.notna(bt.summary["sharpe"]) else None,
        })
    print("\n  EV-threshold sensitivity:")
    _print_df(pd.DataFrame(rows))

    # 7c. Kelly fraction sensitivity
    rows = []
    for kf in (0.125, 0.25, 0.50):
        bt = run_backtest(cal_test_preds[LABEL_PRIMARY], master,
                            kelly_fraction_arg=kf)
        rows.append({
            "kelly_fraction":    kf,
            "n_bets":            bt.summary["n_bets"],
            "roi_per_bet_%":     round((bt.summary["roi_per_bet"] or 0) * 100, 2),
            "total_return_%":    round(bt.summary["total_return_pct"], 2),
            "max_drawdown_%":    round(bt.summary["max_drawdown_pct"], 2),
            "sharpe":            round(bt.summary["sharpe"], 3) if pd.notna(bt.summary["sharpe"]) else None,
        })
    print("\n  Kelly-fraction sensitivity:")
    _print_df(pd.DataFrame(rows))

    # ====================================================================
    # 8. Leakage audit
    # ====================================================================
    _print_section("LEAKAGE AUDIT (production data + features)")
    log.info("Loading Sackmann data for the strict-< recompute")
    sk_full = sackmann.load_sackmann_matches(
        root / "data/raw/sackmann", year_from=2005, include_qual_itf=True,
    )
    sk_clay = sackmann.filter_clay(sk_full).reset_index(drop=True)
    sk_all  = sackmann.load_sackmann_matches(
        root / "data/raw/sackmann", year_from=2005, include_qual_itf=False,
    )
    rankings = sackmann.load_rankings(root / "data/raw/sackmann")

    # Build a test_predictions df with odds for the EV-bucket check
    tp = cal_test_preds[LABEL_PRIMARY].merge(
        master[["match_id", "odds_p1_pinnacle", "odds_p2_pinnacle"]]
        .rename(columns={"odds_p1_pinnacle": "odds_p1",
                          "odds_p2_pinnacle": "odds_p2"}),
        on="match_id", how="left",
    )

    # Rebuild splits (so the audit gets a fresh SplitResult)
    from wta_clay.models.splits import make_splits
    sp = make_splits(master, feature_parquets, run_sanity_checks=False)

    audit_path = audit_dir / "leakage_audit_latest.json"
    audit = run_leakage_audit(
        master, sp, feature_parquets, test_predictions=tp,
        sackmann_clay=sk_clay, sackmann_all=sk_all, rankings=rankings,
        n_random_per_module=30, n_consistency=50,
        out_path=audit_path,
    )
    print(format_audit_report(audit))
    log.info("Audit JSON → %s", audit_path)

    # ====================================================================
    # 9. Final headline
    # ====================================================================
    _print_section("FINAL HEADLINE — project verdict")
    s = bt_main.summary
    sb = bt_cmp.summary
    print(f"  Baseline {LABEL_PRIMARY}:")
    print(f"    ROI per bet:       {(s['roi_per_bet'] or 0)*100:+.2f}%")
    print(f"    Total return:      {s['total_return_pct']:+.2f}%")
    print(f"    CAGR (annualized): {s['annualized_return_pct']:+.2f}%")
    print(f"    Max drawdown:      {s['max_drawdown_pct']:.2f}%")
    print(f"    Sharpe:            {s['sharpe']:.3f}")
    print(f"    n_bets / hit_rate: {s['n_bets']} / {(s['hit_rate'] or 0)*100:.2f}%")
    print()
    print(f"  Baseline {LABEL_COMPARE}:")
    print(f"    ROI per bet:       {(sb['roi_per_bet'] or 0)*100:+.2f}%")
    print(f"    Total return:      {sb['total_return_pct']:+.2f}%")
    print(f"    CAGR (annualized): {sb['annualized_return_pct']:+.2f}%")
    print(f"    Max drawdown:      {sb['max_drawdown_pct']:.2f}%")
    print(f"    Sharpe:            {sb['sharpe']:.3f}")
    print(f"    n_bets / hit_rate: {sb['n_bets']} / {(sb['hit_rate'] or 0)*100:.2f}%")
    print()
    print(f"  Leakage audit:       {'PASS' if audit.passed else 'FAIL'}")
    print(f"  Equity plots:        {eq_path}")
    print(f"                       {eq_path_b}")
    print(f"  Audit JSON:          {audit_path}")


if __name__ == "__main__":
    main()
