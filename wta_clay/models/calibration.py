"""Isotonic-regression post-calibration for trained model predictions.

Why this module exists
----------------------
LightGBM's binary objective produces uncalibrated sigmoid outputs. On
our 4 baselines, mean_pred over-predicts y by 6-8 percentage points,
collapsing flat-stake EV-filtered ROI into the −4% to −7% range despite
healthy AUC (0.64-0.69). Calibration removes this bias and, ideally,
flips ROI back to the realistic +1% to +3% project band.

Three strategies for fitting the isotonic regressor
---------------------------------------------------
``calib_holdout`` (DEFAULT, the trusted strategy)
    Fit on the **calib** split (2017-2018 by default), which the model
    NEVER trained on. Apply to ``valid_predictions`` (diagnostic) and
    ``test_predictions`` (final eval). This is the only strategy that
    eliminates the in-window contamination feedback loop. Requires
    ``train_result.calib_predictions`` populated.

``valid_self``
    Fit on valid predictions, apply on valid. *In-sample best-case*
    diagnostic — yields an OPTIMISTIC ROI estimate. Mean preservation
    and strict Brier decrease are mathematical guarantees on this path,
    but the ROI is upper-bounded by in-sample bias.

``train_tail`` (LEGACY, contaminated)
    Fit on the last N years of the train split (default 2017-2018),
    apply on valid. **Contaminated**: the model has seen 2017-2018
    during training (subject to early-stopping regularisation via
    valid), so the calibrator inherits the model's overfit on those
    years. We confirmed AUC inflation on the train tail of 0.07-0.16
    above valid AUC — that overfit propagates into PAV's empirical y
    rate estimation, producing aggressive over-correction. Kept only
    for diagnostic comparison; **never use for production decisions**.

The 2026-05-03 redesign:
    1. ``splits.py`` now produces 4 splits (train + calib + valid + test).
    2. ``train_lgbm`` trains only on the train split, early-stops on valid,
       predicts on all 4 (calib + test predictions are TRUE OOS).
    3. ``calibrate_baseline`` fits on calib, applies to valid + test.
    4. The bad-vig row 2020-09-21 (implied 0.380, +5pp ROI inflation
       single-handedly) is filtered upstream in oddsfeed.

Self-checks (4)
---------------
The thresholds and pass/fail semantics depend on strategy:

| Check                              | valid_self | train_tail | calib_holdout |
|------------------------------------|------------|------------|---------------|
| 1. |mean_pred_post − y_valid_rate| | hard ≤0.01 | warn ≤0.05 | warn ≤0.05    |
| 2. brier_post < brier_pre          | hard       | warn       | warn          |
| 3. |auc_post − auc_pre|            | warn ≤0.01 | warn ≤0.01 | warn ≤0.01    |
| 4. n_bets pre/post                 | info       | info       | info          |

For ``valid_self`` the math guarantees both 1 and 2; failure means a
calibrator-construction bug. For OOS strategies (``calib_holdout`` and
``train_tail``) the fit and apply sets differ, so the math properties
are warns instead of hard raises.

Exports
-------
``fit_calibrator``                  PAV fit on raw probs vs y_true
``apply_calibration``               Add y_pred_prob_orig + overwrite y_pred_prob
``calibration_plot_data``           10-bucket reliability table (decile binning)
``calibrate_baseline``              Single TrainResult → CalibrationResult
``calibrate_baselines``             dict of TrainResult → dict of CalibrationResult
``apply_to_test``                   Use calib-fit calibrator on test predictions
``format_calibration_report``       Per-baseline multi-line text report
``format_calibration_comparison``   Wide pre/post DataFrame across baselines
``format_ev_bucket_comparison``     Per-bucket cross-strategy wide table
``format_strategy_comparison``      Wide table comparing two strategies
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    brier_score_loss, log_loss, roc_auc_score,
)

from wta_clay.models.lgbm import (
    DEFAULT_EV_THRESHOLD, MiniBacktestResult, TrainResult, mini_backtest,
)

log = logging.getLogger(__name__)


# ===========================================================================
# Constants
# ===========================================================================

#: Number of buckets for the calibration reliability plot.
CALIBRATION_PLOT_BINS: int = 10

# Self-check thresholds (split by strategy — see module docstring).
SANITY_MEAN_PRED_DIFF_MAX_VALID_SELF: float = 0.01
SANITY_MEAN_PRED_DIFF_MAX_TRAIN_TAIL: float = 0.05
#: Backwards-compat alias (= valid_self threshold).
SANITY_MEAN_PRED_DIFF_MAX: float = SANITY_MEAN_PRED_DIFF_MAX_VALID_SELF
#: PAV-introduced ties + finite-sample noise can shift AUC on the
#: fitting set by a few thousandths in either direction; 0.01 is a
#: comfortable warn boundary for typical valid-set sizes (~1500 rows).
SANITY_AUC_DELTA_MAX: float = 0.01

#: Numerical clip applied to calibrated probabilities to avoid log_loss(0/1).
CAL_PROB_EPS: float = 1e-9

#: Default holdout window (last 2 years of the train split).
DEFAULT_HOLDOUT_YEARS: tuple[int, int] = (2017, 2018)

#: Recognised strategy values.
HoldoutStrategy = Literal["valid_self", "train_tail", "calib_holdout"]
_VALID_STRATEGIES: tuple[str, ...] = (
    "valid_self", "train_tail", "calib_holdout",
)


# ===========================================================================
# Result dataclass
# ===========================================================================

@dataclass
class CalibrationResult:
    """Output of :func:`calibrate_baseline`.

    The ``pre_*`` and ``post_*`` metric pairs are always evaluated on the
    **valid set** — that's the apply target. The ``holdout_*`` fields
    describe the calibration FIT set (only populated for ``train_tail``).

    Attributes
    ----------
    label
        Short identifier ("A_clay_elo_only" etc.) for reporting.
    calibrator
        Fitted ``sklearn.isotonic.IsotonicRegression`` instance.
    strategy
        ``"valid_self"`` or ``"train_tail"``.
    holdout_years
        ``(year_from, year_to)`` of the calibration fit set; ``None``
        for ``valid_self``.
    holdout_n
        Row count of the calibration fit set.
    holdout_y_rate
        y=1 rate in the calibration fit set.
    holdout_auc
        Model AUC on the calibration fit set (uncalibrated).
        ``None`` for ``valid_self`` (would equal ``pre_auc``).
    pre_*, post_*
        Standard metrics on the *valid* set, before and after calibration.
    pre_mini_backtest, post_mini_backtest
        Flat-stake EV-filtered Pinnacle backtest before/after.
    valid_predictions_calibrated
        DataFrame (match_id, y_true, y_pred_prob_orig, y_pred_prob,
        sample_weight) — the ``y_pred_prob`` column is overwritten with
        the calibrated probability so the row plugs into
        :func:`mini_backtest` directly.
    plot_data
        10-row reliability table (decile binning of raw prob).
    sanity_log
        Self-check verdicts.
    """
    label: str
    calibrator: Any
    strategy: str
    holdout_years: tuple[int, int] | None
    holdout_n: int | None
    holdout_y_rate: float | None
    holdout_auc: float | None

    # Pre-calibration metrics on valid set
    pre_auc: float
    pre_logloss: float
    pre_brier: float
    pre_mean_pred: float
    pre_mini_backtest: MiniBacktestResult

    # Post-calibration metrics on valid set
    post_auc: float
    post_logloss: float
    post_brier: float
    post_mean_pred: float
    post_mini_backtest: MiniBacktestResult

    # Calibrated valid predictions DataFrame
    valid_predictions_calibrated: pd.DataFrame

    # 10-bucket reliability data
    plot_data: pd.DataFrame

    # Self-check log
    sanity_log: list[str] = field(default_factory=list)


# ===========================================================================
# Core API — fit + apply
# ===========================================================================

def fit_calibrator(
    raw_probs: np.ndarray | pd.Series,
    y_true: np.ndarray | pd.Series,
) -> IsotonicRegression:
    """Fit isotonic regression: raw_prob → empirical y rate.

    Uses ``out_of_bounds='clip'`` so applying to predictions outside the
    fit-set range is safe. ``y_min/y_max`` enforce [0, 1] output range.
    """
    raw = np.asarray(raw_probs, dtype=float)
    y = np.asarray(y_true, dtype=float)
    if len(raw) != len(y):
        raise ValueError(
            f"length mismatch: raw_probs={len(raw)}, y_true={len(y)}"
        )
    if len(raw) == 0:
        raise ValueError("cannot fit calibrator on empty input")
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw, y)
    return iso


def apply_calibration(
    predictions_df: pd.DataFrame,
    calibrator: IsotonicRegression,
) -> pd.DataFrame:
    """Apply ``calibrator`` to the ``y_pred_prob`` column.

    Returns a new DataFrame with:
      - ``y_pred_prob_orig`` (the original raw probability)
      - ``y_pred_prob``      (the calibrated probability — overwrites orig)
      - all other columns preserved.

    The calibrated column is clipped to ``[CAL_PROB_EPS, 1 − CAL_PROB_EPS]``
    so downstream ``log_loss`` evaluation never hits ±inf.
    """
    if "y_pred_prob" not in predictions_df.columns:
        raise ValueError("predictions_df must contain 'y_pred_prob' column")
    out = predictions_df.copy()
    out["y_pred_prob_orig"] = out["y_pred_prob"].astype(float)
    cal = calibrator.predict(out["y_pred_prob_orig"].to_numpy())
    out["y_pred_prob"] = np.clip(cal, CAL_PROB_EPS, 1.0 - CAL_PROB_EPS)
    return out


# ===========================================================================
# Reliability plot data
# ===========================================================================

def calibration_plot_data(
    raw_probs: np.ndarray | pd.Series,
    calibrated_probs: np.ndarray | pd.Series,
    y_true: np.ndarray | pd.Series,
    n_bins: int = CALIBRATION_PLOT_BINS,
) -> pd.DataFrame:
    """Build a decile-binned reliability table (bins by ``raw_probs``)."""
    raw = np.asarray(raw_probs, dtype=float)
    cal = np.asarray(calibrated_probs, dtype=float)
    y = np.asarray(y_true, dtype=float)
    if not (len(raw) == len(cal) == len(y)):
        raise ValueError(
            f"length mismatch: raw={len(raw)}, cal={len(cal)}, y={len(y)}"
        )
    cols = [
        "bin", "lo", "hi", "n",
        "mean_pred_pre", "mean_pred_post", "actual_y_rate",
        "calibration_gap_pre", "calibration_gap_post",
    ]
    if len(raw) == 0:
        return pd.DataFrame(columns=cols)

    quants = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(raw, quants))
    if len(edges) < 2:
        edges = np.array([float(raw.min()), float(raw.min()) + 1e-9])
    edges = edges.astype(float)
    edges[0]  = min(edges[0],  float(raw.min())) - 1e-12
    edges[-1] = max(edges[-1], float(raw.max())) + 1e-12

    bin_idx = np.digitize(raw, edges[1:-1], right=False)
    rows = []
    for b in range(len(edges) - 1):
        m = bin_idx == b
        nb = int(m.sum())
        if nb == 0:
            continue
        mp_pre  = float(raw[m].mean())
        mp_post = float(cal[m].mean())
        yr      = float(y[m].mean())
        rows.append({
            "bin": b,
            "lo": round(float(edges[b]),     4),
            "hi": round(float(edges[b + 1]), 4),
            "n": nb,
            "mean_pred_pre":         round(mp_pre,         4),
            "mean_pred_post":        round(mp_post,        4),
            "actual_y_rate":         round(yr,             4),
            "calibration_gap_pre":   round(mp_pre  - yr,   4),
            "calibration_gap_post":  round(mp_post - yr,   4),
        })
    return pd.DataFrame(rows, columns=cols)


# ===========================================================================
# Holdout predictions extraction
# ===========================================================================

def _extract_train_tail_predictions(
    train_predictions: pd.DataFrame,
    master: pd.DataFrame,
    holdout_years: tuple[int, int],
) -> pd.DataFrame:
    """Filter ``train_predictions`` to rows whose ``match_date.year`` is in
    ``holdout_years`` (inclusive on both ends).

    Looks up dates from ``master`` via ``match_id``; rows in
    ``train_predictions`` whose match_id isn't in master are silently
    dropped (shouldn't happen given the build pipeline, but defensive).
    """
    required = {"match_id", "y_true", "y_pred_prob"}
    miss = required - set(train_predictions.columns)
    if miss:
        raise ValueError(
            f"train_predictions missing required columns: {sorted(miss)}"
        )
    if "match_date" not in master.columns:
        raise ValueError("master must have 'match_date' column")

    mid_to_date = master.set_index("match_id")["match_date"]
    dates = mid_to_date.reindex(train_predictions["match_id"].astype(int))
    years = dates.dt.year.to_numpy()
    lo, hi = holdout_years
    mask = (years >= lo) & (years <= hi)
    out = train_predictions.loc[mask].reset_index(drop=True)
    if len(out) == 0:
        raise ValueError(
            f"holdout window {holdout_years} has zero rows in "
            f"train_predictions; check train_years vs holdout_years"
        )
    return out


# ===========================================================================
# Per-baseline calibration
# ===========================================================================

def calibrate_baseline(
    label: str,
    train_result: TrainResult,
    master: pd.DataFrame,
    *,
    ev_threshold: float = DEFAULT_EV_THRESHOLD,
    holdout_strategy: HoldoutStrategy = "calib_holdout",
    holdout_years: tuple[int, int] = DEFAULT_HOLDOUT_YEARS,
    holdout_predictions: pd.DataFrame | None = None,
    run_sanity_checks: bool = True,
) -> CalibrationResult:
    """Calibrate one baseline + re-run mini-backtest on valid.

    Parameters
    ----------
    label
        Identifier for reporting.
    train_result
        Output of :func:`wta_clay.models.lgbm.train_lgbm`. Must have
        non-empty ``valid_predictions``. For ``calib_holdout`` strategy
        (default), also requires ``calib_predictions`` populated.
    master
        Master DataFrame; passed through to mini-backtest, and used to
        look up match dates for the year-window filter (``train_tail``).
    ev_threshold
        EV cutoff for placing bets (default 5%).
    holdout_strategy
        ``"calib_holdout"`` (default) — fit on the dedicated calibration
        split (2017-2018) which the model never saw during training.
        ``"valid_self"`` — fit + apply on valid (in-sample diagnostic).
        ``"train_tail"`` — legacy contaminated path; use only for
        diagnostic comparison.
    holdout_years
        ``(year_from, year_to)`` for ``train_tail``; ignored otherwise.
    holdout_predictions
        Optional explicit holdout DataFrame; if supplied it is used as
        the calibration fit set verbatim. Bypasses both the calib split
        lookup (``calib_holdout``) and the train-year filter (``train_tail``).
    run_sanity_checks
        If True, run the 4 self-checks and populate ``sanity_log``;
        ``valid_self`` raises on hard checks 1 and 2.
    """
    if holdout_strategy not in _VALID_STRATEGIES:
        raise ValueError(
            f"holdout_strategy must be one of {_VALID_STRATEGIES}, "
            f"got {holdout_strategy!r}"
        )
    vp = train_result.valid_predictions
    if vp.empty:
        raise ValueError(
            f"baseline {label!r}: valid_predictions is empty; "
            f"nothing to calibrate"
        )
    required = {"match_id", "y_true", "y_pred_prob"}
    missing = required - set(vp.columns)
    if missing:
        raise ValueError(
            f"valid_predictions missing required columns: {sorted(missing)}"
        )

    # ----- Build the calibration FIT set -----------------------------------
    if holdout_strategy == "valid_self":
        fit_preds = vp
        h_years: tuple[int, int] | None = None
    elif holdout_strategy == "calib_holdout":
        if holdout_predictions is not None:
            fit_preds = holdout_predictions
        else:
            if train_result.calib_predictions is None:
                raise ValueError(
                    f"baseline {label!r}: holdout_strategy='calib_holdout' "
                    f"requires train_result.calib_predictions. Re-run "
                    f"train_lgbm() with a SplitResult that has a non-empty "
                    f"calib split."
                )
            fit_preds = train_result.calib_predictions
        # For reporting: derive the year span from match_dates (if available)
        if "match_id" in fit_preds.columns and "match_date" in master.columns:
            mid_to_date = master.set_index("match_id")["match_date"]
            d = mid_to_date.reindex(fit_preds["match_id"].astype(int))
            yrs = d.dt.year.dropna()
            h_years = (
                (int(yrs.min()), int(yrs.max())) if not yrs.empty else None
            )
        else:
            h_years = None
    else:  # train_tail (legacy)
        if holdout_predictions is not None:
            fit_preds = holdout_predictions
        else:
            if train_result.train_predictions is None:
                raise ValueError(
                    f"baseline {label!r}: holdout_strategy='train_tail' "
                    f"requires train_result.train_predictions."
                )
            fit_preds = _extract_train_tail_predictions(
                train_result.train_predictions, master, holdout_years,
            )
        h_years = tuple(holdout_years)  # type: ignore[assignment]

    # ----- Pre metrics (always evaluated on valid) -------------------------
    raw_v  = vp["y_pred_prob"].to_numpy(dtype=float)
    y_v    = vp["y_true"].to_numpy(dtype=int)
    pre_auc     = float(roc_auc_score(y_v, raw_v))
    pre_logloss = float(log_loss(y_v, raw_v, labels=[0, 1]))
    pre_brier   = float(brier_score_loss(y_v, raw_v))
    pre_mean    = float(np.mean(raw_v))

    # ----- Holdout descriptive stats (for any OOS strategy) -----------
    if holdout_strategy in ("train_tail", "calib_holdout"):
        h_raw = fit_preds["y_pred_prob"].to_numpy(dtype=float)
        h_y   = fit_preds["y_true"].to_numpy(dtype=int)
        h_n            = int(len(fit_preds))
        h_y_rate       = float(np.mean(h_y))
        h_auc          = (float(roc_auc_score(h_y, h_raw))
                          if len(np.unique(h_y)) == 2 else float("nan"))
    else:
        h_n = h_y_rate = h_auc = None

    # ----- Fit + apply -----------------------------------------------------
    calibrator = fit_calibrator(
        fit_preds["y_pred_prob"], fit_preds["y_true"],
    )
    vp_cal = apply_calibration(vp, calibrator)
    cal_v = vp_cal["y_pred_prob"].to_numpy(dtype=float)

    post_auc     = float(roc_auc_score(y_v, cal_v))
    post_logloss = float(log_loss(y_v, cal_v, labels=[0, 1]))
    post_brier   = float(brier_score_loss(y_v, cal_v))
    post_mean    = float(np.mean(cal_v))

    # ----- Mini-backtests --------------------------------------------------
    pre_mb  = train_result.mini_backtest
    post_mb = mini_backtest(vp_cal, master, ev_threshold=ev_threshold)

    # ----- Plot data -------------------------------------------------------
    plot = calibration_plot_data(
        raw_v, cal_v, y_v, n_bins=CALIBRATION_PLOT_BINS,
    )

    # ----- Self-checks -----------------------------------------------------
    sanity_log: list[str] = []
    if run_sanity_checks:
        _run_calibration_sanity(
            holdout_strategy=holdout_strategy,
            y_v=y_v,
            pre_brier=pre_brier, post_brier=post_brier,
            post_mean=post_mean,
            pre_auc=pre_auc, post_auc=post_auc,
            pre_mb=pre_mb, post_mb=post_mb,
            log_list=sanity_log,
        )

    return CalibrationResult(
        label=label,
        calibrator=calibrator,
        strategy=holdout_strategy,
        holdout_years=h_years,
        holdout_n=h_n,
        holdout_y_rate=h_y_rate,
        holdout_auc=h_auc,
        pre_auc=pre_auc, pre_logloss=pre_logloss,
        pre_brier=pre_brier, pre_mean_pred=pre_mean,
        pre_mini_backtest=pre_mb,
        post_auc=post_auc, post_logloss=post_logloss,
        post_brier=post_brier, post_mean_pred=post_mean,
        post_mini_backtest=post_mb,
        valid_predictions_calibrated=vp_cal,
        plot_data=plot,
        sanity_log=sanity_log,
    )


def calibrate_baselines(
    baseline_results: dict[str, TrainResult],
    master: pd.DataFrame,
    *,
    ev_threshold: float = DEFAULT_EV_THRESHOLD,
    holdout_strategy: HoldoutStrategy = "calib_holdout",
    holdout_years: tuple[int, int] = DEFAULT_HOLDOUT_YEARS,
    run_sanity_checks: bool = True,
) -> dict[str, CalibrationResult]:
    """Calibrate every baseline in ``baseline_results``."""
    out: dict[str, CalibrationResult] = {}
    for label, tr in baseline_results.items():
        log.info("Calibrating baseline %s  (strategy=%s)",
                 label, holdout_strategy)
        out[label] = calibrate_baseline(
            label, tr, master,
            ev_threshold=ev_threshold,
            holdout_strategy=holdout_strategy,
            holdout_years=holdout_years,
            run_sanity_checks=run_sanity_checks,
        )
    return out


# ===========================================================================
# Test-set evaluation (the final OOS number)
# ===========================================================================

@dataclass
class TestSetEvalResult:
    """Output of :func:`apply_to_test`.

    Test-set metrics + mini-backtest both **before** and **after** applying
    the calibrator (the same one that was fit during ``calibrate_baseline``).
    """
    label: str
    n_test: int
    test_y_rate: float

    pre_auc: float
    pre_logloss: float
    pre_brier: float
    pre_mean_pred: float
    pre_mini_backtest: MiniBacktestResult

    post_auc: float
    post_logloss: float
    post_brier: float
    post_mean_pred: float
    post_mini_backtest: MiniBacktestResult

    test_predictions_calibrated: pd.DataFrame


def apply_to_test(
    label: str,
    train_result: TrainResult,
    cal_result: CalibrationResult,
    master: pd.DataFrame,
    *,
    ev_threshold: float = DEFAULT_EV_THRESHOLD,
) -> TestSetEvalResult:
    """Apply the (calib-fitted) calibrator to the test predictions.

    This is the single allowed peek at the test set — only run after
    you've fully committed to the calibration strategy. The function
    computes pre/post metrics and a mini-backtest on the test split.

    Parameters
    ----------
    label
        Baseline identifier for reporting.
    train_result
        Must have ``test_predictions`` populated (auto-populated since
        the 2026-05-03 lgbm.py refactor).
    cal_result
        Output of :func:`calibrate_baseline`. The ``calibrator`` field
        is what gets applied to test_predictions.
    master
        Master DataFrame; passed to mini_backtest.
    ev_threshold
        EV cutoff (default 5%).
    """
    if train_result.test_predictions is None:
        raise ValueError(
            f"baseline {label!r}: test_predictions is None; cannot "
            f"evaluate. Re-run train_lgbm() with a SplitResult that has "
            f"a non-empty test split."
        )
    tp = train_result.test_predictions
    if tp.empty:
        raise ValueError(f"baseline {label!r}: test_predictions is empty")

    raw = tp["y_pred_prob"].to_numpy(dtype=float)
    y   = tp["y_true"].to_numpy(dtype=int)

    pre_auc     = float(roc_auc_score(y, raw)) if len(np.unique(y)) == 2 else float("nan")
    pre_logloss = float(log_loss(y, raw, labels=[0, 1]))
    pre_brier   = float(brier_score_loss(y, raw))
    pre_mean    = float(np.mean(raw))

    pre_mb = mini_backtest(tp, master, ev_threshold=ev_threshold)

    tp_cal = apply_calibration(tp, cal_result.calibrator)
    cal = tp_cal["y_pred_prob"].to_numpy(dtype=float)

    post_auc     = float(roc_auc_score(y, cal)) if len(np.unique(y)) == 2 else float("nan")
    post_logloss = float(log_loss(y, cal, labels=[0, 1]))
    post_brier   = float(brier_score_loss(y, cal))
    post_mean    = float(np.mean(cal))

    post_mb = mini_backtest(tp_cal, master, ev_threshold=ev_threshold)

    return TestSetEvalResult(
        label=label,
        n_test=int(len(tp)),
        test_y_rate=float(np.mean(y)),
        pre_auc=pre_auc, pre_logloss=pre_logloss,
        pre_brier=pre_brier, pre_mean_pred=pre_mean,
        pre_mini_backtest=pre_mb,
        post_auc=post_auc, post_logloss=post_logloss,
        post_brier=post_brier, post_mean_pred=post_mean,
        post_mini_backtest=post_mb,
        test_predictions_calibrated=tp_cal,
    )


def apply_to_test_all(
    baseline_results: dict[str, TrainResult],
    cal_results: dict[str, CalibrationResult],
    master: pd.DataFrame,
    *,
    ev_threshold: float = DEFAULT_EV_THRESHOLD,
) -> dict[str, TestSetEvalResult]:
    """Apply calibrators to test predictions for every baseline."""
    if set(baseline_results.keys()) != set(cal_results.keys()):
        raise ValueError(
            f"key mismatch between baseline_results and cal_results: "
            f"{sorted(set(baseline_results) ^ set(cal_results))}"
        )
    out: dict[str, TestSetEvalResult] = {}
    for label in baseline_results:
        log.info("Applying calibrator to TEST: %s", label)
        out[label] = apply_to_test(
            label, baseline_results[label], cal_results[label], master,
            ev_threshold=ev_threshold,
        )
    return out


# ===========================================================================
# Self-checks (4) — strategy-aware
# ===========================================================================

def _run_calibration_sanity(
    *,
    holdout_strategy: str,
    y_v: np.ndarray,
    pre_brier: float, post_brier: float,
    post_mean: float,
    pre_auc: float, post_auc: float,
    pre_mb: MiniBacktestResult, post_mb: MiniBacktestResult,
    log_list: list[str],
) -> None:
    """Strategy-aware self-checks.

    valid_self: hard raise on (1) and (2). train_tail: warn-only on
    both — train-tail vs valid distribution drift can break the math
    properties without indicating a bug.
    """
    actual_rate = float(np.mean(y_v))
    diff = abs(post_mean - actual_rate)

    is_self = holdout_strategy == "valid_self"
    # OOS strategies (calib_holdout, train_tail) get the looser threshold
    threshold = (SANITY_MEAN_PRED_DIFF_MAX_VALID_SELF if is_self
                 else SANITY_MEAN_PRED_DIFF_MAX_TRAIN_TAIL)

    # ---- (1) mean_pred close to valid y rate ------------------------------
    if diff <= threshold:
        log_list.append(
            f"[OK] mean_pred_post = {post_mean:.4f}, "
            f"y=1 rate = {actual_rate:.4f}, "
            f"|diff| = {diff:.4f} <= {threshold} "
            f"(strategy={holdout_strategy})"
        )
    else:
        msg = (
            f"mean_pred convergence: post-calibration mean "
            f"{post_mean:.4f} vs y=1 rate {actual_rate:.4f}, "
            f"|diff| = {diff:.4f} > {threshold} "
            f"(strategy={holdout_strategy})"
        )
        if is_self:
            raise AssertionError(
                f"{msg}. valid_self must satisfy mean preservation; "
                f"calibrator was likely fit on a different distribution."
            )
        else:
            log_list.append(
                f"[WARN] {msg} — train-tail / valid distribution shift, "
                f"informative but not a bug"
            )

    # ---- (2) Brier strictly decreases ------------------------------------
    if post_brier < pre_brier:
        log_list.append(
            f"[OK] Brier  pre = {pre_brier:.6f} -> post = "
            f"{post_brier:.6f}  (delta = {post_brier - pre_brier:+.6f})"
        )
    else:
        msg = (
            f"Brier did NOT strictly decrease: pre = {pre_brier:.6f}, "
            f"post = {post_brier:.6f} (delta = "
            f"{post_brier - pre_brier:+.6f}, strategy={holdout_strategy})"
        )
        if is_self:
            raise AssertionError(
                f"{msg}. valid_self guarantees this; either calibrator "
                f"was applied to a different set, or raw probs were "
                f"already isotonically optimal."
            )
        else:
            log_list.append(
                f"[WARN] {msg} — calibrator fit on a distribution that "
                f"doesn't generalise to valid"
            )

    # ---- (3) AUC essentially unchanged (always soft warn) -----------------
    auc_delta = abs(post_auc - pre_auc)
    if auc_delta <= SANITY_AUC_DELTA_MAX:
        log_list.append(
            f"[OK] AUC pre = {pre_auc:.4f} -> post = {post_auc:.4f}  "
            f"(|delta| = {auc_delta:.4f} <= {SANITY_AUC_DELTA_MAX})"
        )
    else:
        log_list.append(
            f"[WARN] AUC pre = {pre_auc:.4f} -> post = {post_auc:.4f}  "
            f"(|delta| = {auc_delta:.4f} > {SANITY_AUC_DELTA_MAX}) — "
            f"tie-induced + finite-sample drift larger than expected"
        )

    # ---- (4) n_bets pre/post (info) --------------------------------------
    log_list.append(
        f"[..] n_bets pre = {pre_mb.n_bets}, post = {post_mb.n_bets}  "
        f"(delta = {post_mb.n_bets - pre_mb.n_bets:+d})"
    )


# ===========================================================================
# Pretty printers
# ===========================================================================

def format_calibration_report(label: str, result: CalibrationResult) -> str:
    """Multi-line text report for a single CalibrationResult."""
    pre_mb, post_mb = result.pre_mini_backtest, result.post_mini_backtest
    lines: list[str] = []
    lines.append(f"=== Calibration {label} (strategy={result.strategy}) ===")
    if result.strategy == "train_tail":
        lines.append(
            f"  Holdout: years {result.holdout_years[0]}-{result.holdout_years[1]}  "
            f"n = {result.holdout_n}  y=1 rate = {result.holdout_y_rate:.4f}  "
            f"AUC(pre-cal) = {result.holdout_auc:.4f}"
        )
    lines.append(
        f"  AUC      pre = {result.pre_auc:.4f}  -> post = "
        f"{result.post_auc:.4f}  "
        f"(delta = {result.post_auc - result.pre_auc:+.4f})"
    )
    lines.append(
        f"  Brier    pre = {result.pre_brier:.6f}  -> post = "
        f"{result.post_brier:.6f}  "
        f"(delta = {result.post_brier - result.pre_brier:+.6f})"
    )
    lines.append(
        f"  LogLoss  pre = {result.pre_logloss:.4f}  -> post = "
        f"{result.post_logloss:.4f}  "
        f"(delta = {result.post_logloss - result.pre_logloss:+.4f})"
    )
    lines.append(
        f"  meanP    pre = {result.pre_mean_pred:.4f}  -> post = "
        f"{result.post_mean_pred:.4f}"
    )
    lines.append("")
    lines.append("  Mini-backtest (valid):")
    lines.append(
        f"    n_bets       pre = {pre_mb.n_bets:>5}  -> post = "
        f"{post_mb.n_bets:>5}  "
        f"(delta = {post_mb.n_bets - pre_mb.n_bets:+d})"
    )
    lines.append(
        f"    hit_rate     pre = "
        f"{_fmt_pct(pre_mb.hit_rate * 100 if pd.notna(pre_mb.hit_rate) else float('nan'))}"
        f"  -> post = "
        f"{_fmt_pct(post_mb.hit_rate * 100 if pd.notna(post_mb.hit_rate) else float('nan'))}"
    )
    lines.append(
        f"    flat_roi_%   pre = {_fmt_signed(pre_mb.flat_roi_pct)}"
        f"  -> post = {_fmt_signed(post_mb.flat_roi_pct)}"
    )
    lines.append("")
    lines.append("  By EV bucket (post-calibration):")
    if not post_mb.by_ev_bucket.empty:
        for _, r in post_mb.by_ev_bucket.iterrows():
            if r["n_bets"]:
                lines.append(
                    f"    {r['ev_bucket']:<6s}  n={r['n_bets']:>4}  "
                    f"hit={r['hit_rate_%']:>5}%  ROI={r['flat_roi_%']:+.2f}%"
                )
            else:
                lines.append(f"    {r['ev_bucket']:<6s}  n=0  (no bets)")
    lines.append("")
    lines.append("  Sanity checks:")
    for s in result.sanity_log:
        lines.append(f"    {s}")
    return "\n".join(lines)


def format_calibration_comparison(
    results: dict[str, CalibrationResult],
) -> pd.DataFrame:
    """Wide pre/post DataFrame across baselines (one strategy)."""
    rows = []
    for label, r in results.items():
        roi_pre  = (round(r.pre_mini_backtest.flat_roi_pct,  2)
                    if pd.notna(r.pre_mini_backtest.flat_roi_pct)
                    else None)
        roi_post = (round(r.post_mini_backtest.flat_roi_pct, 2)
                    if pd.notna(r.post_mini_backtest.flat_roi_pct)
                    else None)
        roi_delta = (round(roi_post - roi_pre, 2)
                     if (roi_pre is not None and roi_post is not None)
                     else None)
        rows.append({
            "baseline":         label,
            "strategy":         r.strategy,
            "auc_pre":          round(r.pre_auc, 4),
            "auc_post":         round(r.post_auc, 4),
            "brier_pre":        round(r.pre_brier, 6),
            "brier_post":       round(r.post_brier, 6),
            "mean_pred_pre":    round(r.pre_mean_pred, 4),
            "mean_pred_post":   round(r.post_mean_pred, 4),
            "n_bets_pre":       r.pre_mini_backtest.n_bets,
            "n_bets_post":      r.post_mini_backtest.n_bets,
            "roi_pre":          roi_pre,
            "roi_post":         roi_post,
            "roi_delta":        roi_delta,
        })
    return pd.DataFrame(rows)


def format_strategy_comparison(
    self_results: dict[str, CalibrationResult],
    tail_results: dict[str, CalibrationResult],
) -> pd.DataFrame:
    """Side-by-side comparison: valid_self vs train_tail."""
    if set(self_results.keys()) != set(tail_results.keys()):
        raise ValueError(
            f"key mismatch: self={sorted(self_results)} "
            f"vs tail={sorted(tail_results)}"
        )
    rows = []
    for label in self_results:
        rs, rt = self_results[label], tail_results[label]
        rows.append({
            "baseline":            label,
            "auc_self":            round(rs.post_auc, 4),
            "auc_tail":            round(rt.post_auc, 4),
            "brier_self":          round(rs.post_brier, 6),
            "brier_tail":          round(rt.post_brier, 6),
            "meanP_self":          round(rs.post_mean_pred, 4),
            "meanP_tail":          round(rt.post_mean_pred, 4),
            "n_bets_self":         rs.post_mini_backtest.n_bets,
            "n_bets_tail":         rt.post_mini_backtest.n_bets,
            "roi_self":            (round(rs.post_mini_backtest.flat_roi_pct, 2)
                                    if pd.notna(rs.post_mini_backtest.flat_roi_pct)
                                    else None),
            "roi_tail":            (round(rt.post_mini_backtest.flat_roi_pct, 2)
                                    if pd.notna(rt.post_mini_backtest.flat_roi_pct)
                                    else None),
        })
    df = pd.DataFrame(rows)
    df["roi_optimism_pp"] = df["roi_self"] - df["roi_tail"]
    return df


def format_ev_bucket_comparison(
    self_results: dict[str, CalibrationResult],
    tail_results: dict[str, CalibrationResult],
) -> pd.DataFrame:
    """Long-form EV-bucket × baseline × strategy comparison.

    Columns: baseline, ev_bucket,
             n_self, hit_self_%, roi_self_%,
             n_tail, hit_tail_%, roi_tail_%.
    """
    if set(self_results.keys()) != set(tail_results.keys()):
        raise ValueError("key mismatch between self and tail dicts")
    rows = []
    for label in self_results:
        s_buckets = self_results[label].post_mini_backtest.by_ev_bucket
        t_buckets = tail_results[label].post_mini_backtest.by_ev_bucket
        if s_buckets.empty and t_buckets.empty:
            continue
        s_idx = s_buckets.set_index("ev_bucket") if not s_buckets.empty else pd.DataFrame()
        t_idx = t_buckets.set_index("ev_bucket") if not t_buckets.empty else pd.DataFrame()
        all_buckets = sorted(set(s_idx.index) | set(t_idx.index),
                              key=_ev_bucket_sort_key)
        for bk in all_buckets:
            s_row = s_idx.loc[bk] if bk in s_idx.index else None
            t_row = t_idx.loc[bk] if bk in t_idx.index else None
            rows.append({
                "baseline":   label,
                "ev_bucket":  bk,
                "n_self":     int(s_row["n_bets"]) if s_row is not None else 0,
                "hit_self_%": s_row["hit_rate_%"]   if s_row is not None else None,
                "roi_self_%": s_row["flat_roi_%"]   if s_row is not None else None,
                "n_tail":     int(t_row["n_bets"]) if t_row is not None else 0,
                "hit_tail_%": t_row["hit_rate_%"]   if t_row is not None else None,
                "roi_tail_%": t_row["flat_roi_%"]   if t_row is not None else None,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _fmt_pct(x: float) -> str:
    return f"{x:5.2f}%" if pd.notna(x) else "  N/A "


def _fmt_signed(x: float) -> str:
    return f"{x:+6.2f}%" if pd.notna(x) else "  N/A "


def format_test_eval(
    results: dict[str, TestSetEvalResult],
) -> pd.DataFrame:
    """Wide pre/post DataFrame across baselines for test-set evaluation."""
    rows = []
    for label, r in results.items():
        rows.append({
            "baseline":      label,
            "n_test":        r.n_test,
            "test_y_rate":   round(r.test_y_rate, 4),
            "auc_pre":       round(r.pre_auc, 4),
            "auc_post":      round(r.post_auc, 4),
            "brier_pre":     round(r.pre_brier, 6),
            "brier_post":    round(r.post_brier, 6),
            "meanP_pre":     round(r.pre_mean_pred, 4),
            "meanP_post":    round(r.post_mean_pred, 4),
            "n_bets_pre":    r.pre_mini_backtest.n_bets,
            "n_bets_post":   r.post_mini_backtest.n_bets,
            "roi_pre":       (round(r.pre_mini_backtest.flat_roi_pct, 2)
                              if pd.notna(r.pre_mini_backtest.flat_roi_pct)
                              else None),
            "roi_post":      (round(r.post_mini_backtest.flat_roi_pct, 2)
                              if pd.notna(r.post_mini_backtest.flat_roi_pct)
                              else None),
        })
    return pd.DataFrame(rows)


def format_test_ev_buckets(
    results: dict[str, TestSetEvalResult],
) -> pd.DataFrame:
    """Long-form (baseline × ev_bucket) test-set EV buckets, post-calibration."""
    rows = []
    for label, r in results.items():
        bk = r.post_mini_backtest.by_ev_bucket
        if bk.empty:
            continue
        for _, b in bk.iterrows():
            rows.append({
                "baseline":   label,
                "ev_bucket":  b["ev_bucket"],
                "n_bets":     int(b["n_bets"]),
                "hit_rate_%": b["hit_rate_%"],
                "flat_roi_%": b["flat_roi_%"],
            })
    return pd.DataFrame(rows)


def _ev_bucket_sort_key(label: str) -> float:
    """Sort EV buckets by their lower edge: '5-10%' → 5, '10-20%' → 10, '20%+' → 20."""
    digits = "".join(c for c in label.split("-")[0] if c.isdigit() or c == ".")
    return float(digits) if digits else float("inf")
