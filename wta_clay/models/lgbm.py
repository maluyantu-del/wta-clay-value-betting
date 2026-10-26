"""LightGBM training + mini-backtest module.

Trains a LightGBM binary classifier on `SplitResult.X_train` to predict p1's
win probability, evaluates on `SplitResult.X_valid`, and runs an inline
flat-stake mini-backtest using Pinnacle odds with EV threshold.

The mini-backtest is **NOT** the production backtest (that's in
``wta_clay/backtest/``). It exists to give an immediate go/no-go ROI
signal at training time so we can see whether the feature pipeline +
modeling stack produce real alpha before investing in calibration,
hyperparameter tuning, and Kelly staking.

Mini-backtest filtering rules
-----------------------------
- ``not_for_betting=True`` rows excluded (WTA125 etc.)
- ``market_source='none'`` rows excluded (no Pinnacle line)
- 2009 rows excluded (Pinnacle absent that whole year)
- Rows with NaN ``odds_p1_pinnacle`` or ``odds_p2_pinnacle`` excluded

Bet logic
---------
For each row, compute EV for both directions::

    ev_p1 = p_model       * (odds_p1 - 1) − (1 − p_model)
    ev_p2 = (1 − p_model) * (odds_p2 - 1) − p_model

Bet on whichever side has ``ev > ev_threshold`` (default 5%). If both
sides exceed threshold (rare, requires negative bookie vig), pick the
higher EV. **Flat 1-unit stake** — no Kelly here.

Self-checks (5)
---------------
1. Valid AUC ≥ 0.55  (lower → model didn't learn)
2. ``clay_elo_diff`` in top-5 feature importance (when present)
3. ``|mean_pred − valid_y_rate| ≤ 0.05`` (calibration sanity)
4. ``|train_logloss − valid_logloss| ≤ 0.05`` (overfit guard)
5. Mini-backtest ``n_bets > 50`` (else EV threshold or model too conservative)

Diagnostic ranges (run-time interpretation)
-------------------------------------------
- AUC < 0.60 → leakage in wrong direction or sign-flip bug
- 0.60 ≤ AUC ≤ 0.75 → expected band for this problem
- AUC > 0.75 → leakage in right direction; STOP and investigate
- ROI > +5% → suspicious; run overfit checks before celebrating
- ROI in [-3%, +3%] → realistic project performance band
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "lgbm.py requires lightgbm; install with `pip install lightgbm`."
    ) from exc

from sklearn.metrics import (
    roc_auc_score, log_loss, brier_score_loss,
)

from wta_clay.models.splits import SplitResult, make_splits

log = logging.getLogger(__name__)


# ===========================================================================
# Constants
# ===========================================================================

#: Conservative LightGBM defaults — Optuna tuning lives at a later stage.
DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 50,        # over-fit defense
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 42,
    "deterministic": True,
}

#: 4-feature core baseline subset (per locked baseline B).
CORE_4_FEATURES: tuple[str, ...] = (
    "clay_elo_diff",
    "form_winrate_diff",
    "specialist_diff",
    "rank_diff",
)

#: Default EV threshold for mini-backtest (5%).
DEFAULT_EV_THRESHOLD: float = 0.05

#: EV buckets used for the mini-backtest segmentation report.
EV_BUCKETS: tuple[tuple[float, float, str], ...] = (
    (0.05, 0.10, "5-10%"),
    (0.10, 0.20, "10-20%"),
    (0.20, float("inf"), "20%+"),
)

#: Self-check thresholds.
SANITY_MIN_VALID_AUC: float = 0.55
SANITY_MEAN_PRED_DIFF_MAX: float = 0.05
SANITY_TRAIN_VALID_LOGLOSS_GAP_MAX: float = 0.05
SANITY_MIN_BETS: int = 50


# ===========================================================================
# Result dataclasses
# ===========================================================================

@dataclass
class MiniBacktestResult:
    """Output of :func:`mini_backtest`."""
    n_bets: int
    n_wins: int
    hit_rate: float
    flat_roi_pct: float
    by_ev_bucket: pd.DataFrame
    bets_df: pd.DataFrame                # row-level for forensic inspection
    n_eligible: int                       # post-filter row count
    n_pos_ev_p1: int                      # rows where ev_p1 > threshold
    n_pos_ev_p2: int                      # rows where ev_p2 > threshold


@dataclass
class TrainResult:
    """Output of :func:`train_lgbm`.

    Notes
    -----
    The ``*_predictions`` DataFrames all use the same column schema
    (match_id / y_true / y_pred_prob / sample_weight) and are produced
    at ``best_iteration`` so the calibration layer is consistent with
    the early-stopped model.

    The **calib** set (2017-2018 by default) is the production
    calibration holdout: the model never sees it during training, so
    ``calib_predictions`` is genuinely OOS. Fit the isotonic regressor
    on these and apply to ``valid_predictions`` (diagnostic) or
    ``test_predictions`` (final eval).
    """
    model: Any                            # lgb.Booster
    valid_auc: float
    valid_logloss: float
    valid_brier: float
    feature_importance: pd.DataFrame      # cols: feature, gain, split
    train_history: dict
    config: dict
    valid_predictions: pd.DataFrame       # match_id, y_true, y_pred_prob, sample_weight
    mini_backtest: MiniBacktestResult
    sanity_log: list[str] = field(default_factory=list)
    #: Predictions on the **train** set at ``best_iteration``. Same column
    #: schema as ``valid_predictions``. Diagnostic only (in-sample).
    train_predictions: pd.DataFrame | None = None
    #: Predictions on the **calib** set (2017-2018) at ``best_iteration``.
    #: True OOS holdout — model never trained on these rows. Use for
    #: fitting the calibration isotonic regression.
    calib_predictions: pd.DataFrame | None = None
    #: Predictions on the **test** set (2022-2025) at ``best_iteration``.
    #: Final OOS evaluation; do NOT inspect during model selection.
    test_predictions: pd.DataFrame | None = None


# ===========================================================================
# Main entry — train + evaluate + mini-backtest
# ===========================================================================

def train_lgbm(
    splits: SplitResult,
    master: pd.DataFrame,
    feature_subset: list[str] | tuple[str, ...] | None = None,
    *,
    params: dict | None = None,
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 50,
    verbose: int = 100,
    ev_threshold: float = DEFAULT_EV_THRESHOLD,
    run_sanity_checks: bool = True,
) -> TrainResult:
    """Train a LightGBM binary classifier and run inline mini-backtest.

    Parameters
    ----------
    splits : SplitResult
        Output of :func:`wta_clay.models.splits.make_splits`. Provides
        X_train/y_train/w_train + X_valid/y_valid + match_ids_valid.
    master : DataFrame
        Used by mini-backtest to fetch Pinnacle odds + filter columns.
    feature_subset : list | tuple | None
        Columns to use as features. ``None`` → all columns in
        ``splits.feature_names``.
    params : dict | None
        LightGBM training parameters. ``None`` → :data:`DEFAULT_PARAMS`.
    """
    if params is None:
        params = dict(DEFAULT_PARAMS)
    feats = list(feature_subset) if feature_subset else list(splits.feature_names)

    # Validate that requested features exist in splits.X_train
    missing = set(feats) - set(splits.X_train.columns)
    if missing:
        raise ValueError(f"feature_subset contains columns not in X_train: "
                          f"{sorted(missing)}")

    X_tr = splits.X_train[feats].copy()
    X_ca = splits.X_calib[feats].copy() if not splits.X_calib.empty else None
    X_va = splits.X_valid[feats].copy()
    X_te = splits.X_test[feats].copy()  if not splits.X_test.empty  else None
    y_tr = splits.y_train.astype(int).to_numpy()
    y_va = splits.y_valid.astype(int).to_numpy()
    w_tr = splits.w_train.astype(float).to_numpy()

    log.info("train_lgbm: train=%d  calib=%d  valid=%d  test=%d  features=%d",
             len(X_tr),
             len(X_ca) if X_ca is not None else 0,
             len(X_va),
             len(X_te) if X_te is not None else 0,
             len(feats))

    train_set = lgb.Dataset(X_tr, label=y_tr, weight=w_tr,
                             feature_name=feats)
    valid_set = lgb.Dataset(X_va, label=y_va, reference=train_set,
                             feature_name=feats)

    eval_history: dict[str, dict] = {}
    callbacks = [
        lgb.early_stopping(early_stopping_rounds, first_metric_only=True),
        lgb.log_evaluation(period=verbose),
        lgb.record_evaluation(eval_history),
    ]

    model = lgb.train(
        params, train_set,
        num_boost_round=num_boost_round,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )

    # Predict on all 4 sets at best_iteration (early-stopped model snapshot).
    # train_probs: in-sample (diagnostic)
    # calib_probs: TRUE OOS — used to fit the calibrator
    # valid_probs: early-stopping target — used for model selection
    # test_probs:  TRUE OOS — final evaluation, do not look unless reporting
    best_iter = model.best_iteration if model.best_iteration else model.current_iteration()
    train_probs = model.predict(X_tr, num_iteration=best_iter)
    valid_probs = model.predict(X_va, num_iteration=best_iter)
    calib_probs = (model.predict(X_ca, num_iteration=best_iter)
                   if X_ca is not None else None)
    test_probs  = (model.predict(X_te, num_iteration=best_iter)
                   if X_te is not None else None)

    # Standard metrics on valid (early-stopping target)
    valid_auc = float(roc_auc_score(y_va, valid_probs))
    valid_logloss = float(log_loss(y_va, valid_probs, labels=[0, 1]))
    valid_brier = float(brier_score_loss(y_va, valid_probs))

    # Feature importance (gain + split)
    fi_gain = model.feature_importance(importance_type="gain")
    fi_split = model.feature_importance(importance_type="split")
    fi = pd.DataFrame({
        "feature": feats, "gain": fi_gain, "split": fi_split,
    }).sort_values("gain", ascending=False).reset_index(drop=True)

    # Per-split predictions DataFrames — uniform schema.
    valid_predictions = pd.DataFrame({
        "match_id":      splits.match_ids_valid.astype("Int64"),
        "y_true":        y_va,
        "y_pred_prob":   valid_probs,
        "sample_weight": splits.w_valid.astype(float).to_numpy(),
    })
    train_predictions = pd.DataFrame({
        "match_id":      splits.match_ids_train.astype("Int64"),
        "y_true":        y_tr,
        "y_pred_prob":   train_probs,
        "sample_weight": splits.w_train.astype(float).to_numpy(),
    })
    calib_predictions: pd.DataFrame | None = None
    if calib_probs is not None and len(splits.match_ids_calib) > 0:
        calib_predictions = pd.DataFrame({
            "match_id":      splits.match_ids_calib.astype("Int64"),
            "y_true":        splits.y_calib.astype(int).to_numpy(),
            "y_pred_prob":   calib_probs,
            "sample_weight": splits.w_calib.astype(float).to_numpy(),
        })
    test_predictions: pd.DataFrame | None = None
    if test_probs is not None and len(splits.match_ids_test) > 0:
        test_predictions = pd.DataFrame({
            "match_id":      splits.match_ids_test.astype("Int64"),
            "y_true":        splits.y_test.astype(int).to_numpy(),
            "y_pred_prob":   test_probs,
            "sample_weight": splits.w_test.astype(float).to_numpy(),
        })

    # Mini-backtest on valid
    mb = mini_backtest(valid_predictions, master, ev_threshold)

    # Self-checks
    sanity_log: list[str] = []
    if run_sanity_checks:
        _run_sanity_checks(
            y_tr, y_va, train_probs, valid_probs,
            fi, mb, sanity_log,
        )

    config = {
        "n_features": len(feats),
        "feature_subset": feats,
        "params": params,
        "num_boost_round": num_boost_round,
        "early_stopping_rounds": early_stopping_rounds,
        "best_iteration": int(best_iter),
        "ev_threshold": ev_threshold,
    }

    return TrainResult(
        model=model,
        valid_auc=valid_auc,
        valid_logloss=valid_logloss,
        valid_brier=valid_brier,
        feature_importance=fi,
        train_history=dict(eval_history),
        config=config,
        valid_predictions=valid_predictions,
        mini_backtest=mb,
        sanity_log=sanity_log,
        train_predictions=train_predictions,
        calib_predictions=calib_predictions,
        test_predictions=test_predictions,
    )


# ===========================================================================
# Mini-backtest
# ===========================================================================

def mini_backtest(
    predictions: pd.DataFrame,
    master: pd.DataFrame,
    ev_threshold: float = DEFAULT_EV_THRESHOLD,
) -> MiniBacktestResult:
    """Flat-stake EV-filtered backtest on Pinnacle odds.

    See module docstring for filter rules and bet logic.
    """
    odds_cols = [
        "match_id", "match_date", "odds_p1_pinnacle", "odds_p2_pinnacle",
        "not_for_betting", "market_source",
    ]
    df = predictions.merge(master[odds_cols], on="match_id", how="left")

    # Filter
    keep = ~df["not_for_betting"].fillna(False).astype(bool)
    keep &= df["market_source"].astype(str) != "none"
    keep &= df["match_date"].dt.year != 2009
    keep &= df["odds_p1_pinnacle"].notna()
    keep &= df["odds_p2_pinnacle"].notna()
    df = df.loc[keep].reset_index(drop=True)
    n_eligible = len(df)

    if n_eligible == 0:
        empty_buckets = pd.DataFrame(columns=[
            "ev_bucket", "n_bets", "hit_rate_%", "flat_roi_%",
        ])
        return MiniBacktestResult(
            n_bets=0, n_wins=0, hit_rate=float("nan"),
            flat_roi_pct=float("nan"),
            by_ev_bucket=empty_buckets,
            bets_df=pd.DataFrame(),
            n_eligible=0, n_pos_ev_p1=0, n_pos_ev_p2=0,
        )

    # EV per side
    p = df["y_pred_prob"].astype(float)
    df["ev_p1"] = p * (df["odds_p1_pinnacle"] - 1) - (1 - p)
    df["ev_p2"] = (1 - p) * (df["odds_p2_pinnacle"] - 1) - p

    p1_hits = df["ev_p1"] > ev_threshold
    p2_hits = df["ev_p2"] > ev_threshold
    both = p1_hits & p2_hits

    # Choose bet side
    df["bet_side"] = pd.NA
    df.loc[p1_hits & ~both, "bet_side"] = "p1"
    df.loc[p2_hits & ~both, "bet_side"] = "p2"
    df.loc[both & (df["ev_p1"] >= df["ev_p2"]), "bet_side"] = "p1"
    df.loc[both & (df["ev_p2"] >  df["ev_p1"]), "bet_side"] = "p2"

    bets = df[df["bet_side"].notna()].copy()

    # PnL on flat 1-unit stake
    bet_p1 = bets["bet_side"] == "p1"
    bets["effective_ev"] = np.where(bet_p1, bets["ev_p1"], bets["ev_p2"])
    bets["effective_odds"] = np.where(
        bet_p1, bets["odds_p1_pinnacle"], bets["odds_p2_pinnacle"]
    )
    bets["won"] = ((bet_p1 & (bets["y_true"] == 1)) |
                   (~bet_p1 & (bets["y_true"] == 0)))
    bets["pnl"] = np.where(bets["won"],
                            bets["effective_odds"] - 1.0,
                            -1.0)

    n_bets = int(len(bets))
    n_wins = int(bets["won"].sum())
    hit_rate = float(n_wins / n_bets) if n_bets else float("nan")
    flat_roi_pct = (
        float(bets["pnl"].sum() / n_bets * 100) if n_bets else float("nan")
    )

    # Per-bucket breakdown
    bucket_rows = []
    for lo, hi, label in EV_BUCKETS:
        m = (bets["effective_ev"] >= lo) & (bets["effective_ev"] < hi)
        sub = bets[m]
        n = int(len(sub))
        if n:
            hr = float(sub["won"].mean())
            roi = float(sub["pnl"].sum() / n * 100)
        else:
            hr = float("nan"); roi = float("nan")
        bucket_rows.append({
            "ev_bucket": label,
            "n_bets": n,
            "hit_rate_%": round(hr * 100, 2) if pd.notna(hr) else None,
            "flat_roi_%": round(roi, 2) if pd.notna(roi) else None,
        })

    return MiniBacktestResult(
        n_bets=n_bets, n_wins=n_wins,
        hit_rate=hit_rate, flat_roi_pct=flat_roi_pct,
        by_ev_bucket=pd.DataFrame(bucket_rows),
        bets_df=bets,
        n_eligible=n_eligible,
        n_pos_ev_p1=int(p1_hits.sum()),
        n_pos_ev_p2=int(p2_hits.sum()),
    )


# ===========================================================================
# 4-baseline orchestration
# ===========================================================================

def run_baselines(
    master: pd.DataFrame,
    feature_parquets: dict[str, Path | str],
    *,
    ev_threshold: float = DEFAULT_EV_THRESHOLD,
    verbose: int = 0,
) -> dict[str, TrainResult]:
    """Run all 4 baselines back-to-back. Returns dict keyed by baseline label."""
    log.info("=== Running 4 baselines ===")
    results: dict[str, TrainResult] = {}

    # Baselines A, B, C share splits (lambda=0)
    splits_no_decay = make_splits(master, feature_parquets,
                                    time_decay_lambda=0.0,
                                    run_sanity_checks=False)

    log.info("Baseline A: clay_elo_diff only")
    results["A_clay_elo_only"] = train_lgbm(
        splits_no_decay, master,
        feature_subset=["clay_elo_diff"],
        verbose=verbose, ev_threshold=ev_threshold,
    )

    log.info("Baseline B: 4 core features")
    results["B_4_core"] = train_lgbm(
        splits_no_decay, master,
        feature_subset=list(CORE_4_FEATURES),
        verbose=verbose, ev_threshold=ev_threshold,
    )

    log.info("Baseline C: all 38 features (no decay)")
    results["C_all_38"] = train_lgbm(
        splits_no_decay, master,
        feature_subset=None,
        verbose=verbose, ev_threshold=ev_threshold,
    )

    log.info("Baseline D: all 38 features + time_decay_lambda=0.05")
    splits_decay = make_splits(master, feature_parquets,
                                 time_decay_lambda=0.05,
                                 time_decay_anchor_year=2016,
                                 run_sanity_checks=False)
    results["D_all_38_decay"] = train_lgbm(
        splits_decay, master,
        feature_subset=None,
        verbose=verbose, ev_threshold=ev_threshold,
    )

    return results


# ===========================================================================
# Self-checks (5)
# ===========================================================================

def _run_sanity_checks(
    y_tr, y_va, train_probs, valid_probs,
    feature_importance: pd.DataFrame, mb: MiniBacktestResult,
    log_list: list[str],
) -> None:
    """5 self-checks per locked spec.

    ``train_probs`` and ``valid_probs`` are the model's predictions at
    ``best_iteration`` on the train and valid splits respectively.
    """
    # 1. Valid AUC ≥ min
    auc = float(roc_auc_score(y_va, valid_probs))
    if auc < SANITY_MIN_VALID_AUC:
        raise AssertionError(
            f"Valid AUC = {auc:.4f} < {SANITY_MIN_VALID_AUC} threshold — "
            f"model failed to learn"
        )
    log_list.append(f"[✓] Valid AUC = {auc:.4f} ≥ {SANITY_MIN_VALID_AUC}")

    # 2. clay_elo_diff in top-5 by gain (when present in feature subset)
    feats_in_subset = set(feature_importance["feature"].tolist())
    if "clay_elo_diff" in feats_in_subset:
        top5 = feature_importance.head(5)["feature"].tolist()
        if "clay_elo_diff" in top5:
            log_list.append(f"[✓] clay_elo_diff in top-5 by gain")
        else:
            log_list.append(
                f"[⚠] clay_elo_diff NOT in top-5 by gain  "
                f"(top-5: {top5}) — possible bug or sign issue"
            )
    else:
        log_list.append(f"[ ] clay_elo_diff not in feature subset (skipped)")

    # 3. Mean predicted prob ≈ valid y=1 rate (calibration sanity)
    mean_pred = float(np.mean(valid_probs))
    actual_rate = float(np.mean(y_va))
    diff = abs(mean_pred - actual_rate)
    if diff <= SANITY_MEAN_PRED_DIFF_MAX:
        log_list.append(
            f"[✓] mean_pred = {mean_pred:.4f}, valid y=1 = {actual_rate:.4f}, "
            f"|diff| = {diff:.4f} ≤ {SANITY_MEAN_PRED_DIFF_MAX}"
        )
    else:
        log_list.append(
            f"[⚠] mean_pred = {mean_pred:.4f}, valid y=1 = {actual_rate:.4f}, "
            f"|diff| = {diff:.4f} > {SANITY_MEAN_PRED_DIFF_MAX} — "
            f"calibration off, isotonic step expected"
        )

    # 4. Train vs valid logloss gap (overfit guard)
    train_logloss = float(log_loss(y_tr, train_probs, labels=[0, 1]))
    valid_logloss = float(log_loss(y_va, valid_probs, labels=[0, 1]))
    gap = valid_logloss - train_logloss
    if abs(gap) <= SANITY_TRAIN_VALID_LOGLOSS_GAP_MAX:
        log_list.append(
            f"[✓] train logloss = {train_logloss:.4f}, valid = "
            f"{valid_logloss:.4f}, gap = {gap:+.4f} ≤ "
            f"{SANITY_TRAIN_VALID_LOGLOSS_GAP_MAX}"
        )
    else:
        log_list.append(
            f"[⚠] train logloss = {train_logloss:.4f}, valid = "
            f"{valid_logloss:.4f}, gap = {gap:+.4f} > "
            f"{SANITY_TRAIN_VALID_LOGLOSS_GAP_MAX} — possible overfit"
        )

    # 5. Mini-backtest n_bets > min
    if mb.n_bets > SANITY_MIN_BETS:
        log_list.append(
            f"[✓] mini-backtest n_bets = {mb.n_bets} > {SANITY_MIN_BETS}"
        )
    else:
        log_list.append(
            f"[⚠] mini-backtest n_bets = {mb.n_bets} ≤ {SANITY_MIN_BETS} "
            f"— EV threshold too high or model conservative"
        )


# ===========================================================================
# Pretty printer for baseline reports
# ===========================================================================

def format_baseline_report(label: str, result: TrainResult) -> str:
    """Format a single baseline's report as a multi-line string."""
    lines = []
    lines.append(f"=== Baseline {label} ===")
    lines.append(f"Features used:  {len(result.config['feature_subset'])}")
    lines.append(f"Best iteration: {result.config['best_iteration']}")
    lines.append(f"Valid AUC:      {result.valid_auc:.4f}")
    lines.append(f"Valid logloss:  {result.valid_logloss:.4f}")
    lines.append(f"Valid brier:    {result.valid_brier:.4f}")
    lines.append("")
    lines.append("Top 10 features by gain:")
    for i, row in result.feature_importance.head(10).iterrows():
        lines.append(f"  {i+1:>2}. {row['feature']:<28s} "
                     f"gain={row['gain']:>10.0f}  split={row['split']:>4d}")
    lines.append("")
    mb = result.mini_backtest
    lines.append(
        f"Mini-backtest on valid (Pinnacle, EV > "
        f"{result.config['ev_threshold']*100:.0f}%, flat 1u):"
    )
    lines.append(f"  Eligible rows: {mb.n_eligible}")
    lines.append(f"  n_bets:        {mb.n_bets}")
    lines.append(f"  hit_rate:      "
                 f"{mb.hit_rate*100:.2f}%" if pd.notna(mb.hit_rate) else "  hit_rate: N/A")
    lines.append(f"  flat_roi:      "
                 f"{mb.flat_roi_pct:+.2f}%" if pd.notna(mb.flat_roi_pct)
                 else "  flat_roi: N/A")
    lines.append("")
    lines.append("  By EV bucket:")
    for _, r in mb.by_ev_bucket.iterrows():
        if r["n_bets"]:
            lines.append(f"    {r['ev_bucket']:<6s}  n={r['n_bets']:>4}  "
                         f"hit={r['hit_rate_%']:>5}%  ROI={r['flat_roi_%']:+.2f}%")
        else:
            lines.append(f"    {r['ev_bucket']:<6s}  n=0  (no bets)")
    lines.append("")
    lines.append("Sanity checks:")
    for s in result.sanity_log:
        lines.append(f"  {s}")
    return "\n".join(lines)
