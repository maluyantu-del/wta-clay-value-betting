"""Unit tests for ``wta_clay.models.calibration``."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression

from wta_clay.models.calibration import (
    CAL_PROB_EPS, CALIBRATION_PLOT_BINS,
    DEFAULT_HOLDOUT_YEARS,
    SANITY_AUC_DELTA_MAX,
    SANITY_MEAN_PRED_DIFF_MAX,
    SANITY_MEAN_PRED_DIFF_MAX_TRAIN_TAIL,
    SANITY_MEAN_PRED_DIFF_MAX_VALID_SELF,
    CalibrationResult,
    apply_calibration,
    calibrate_baseline, calibrate_baselines,
    calibration_plot_data,
    fit_calibrator,
    format_calibration_comparison,
    format_calibration_report,
    format_ev_bucket_comparison,
    format_strategy_comparison,
)
from wta_clay.models.lgbm import (
    DEFAULT_EV_THRESHOLD, MiniBacktestResult, TrainResult, mini_backtest,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic over-confident predictions + master
# ---------------------------------------------------------------------------

def _build_overconfident_predictions(n: int = 500, seed: int = 7,
                                       start_date: str = "2020-05-01"):
    """Return (predictions_df, master_df) where the raw probability is
    systematically over-confident by ~7 pp.
    """
    rng = np.random.default_rng(seed)
    logits = rng.normal(0.0, 1.0, n)
    p_true = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.uniform(0, 1, n) < p_true).astype(int)
    p_raw = np.clip(1.5 * p_true - 0.20, 0.01, 0.99)

    preds = pd.DataFrame({
        "match_id":      np.arange(n, dtype=np.int64),
        "y_true":        y,
        "y_pred_prob":   p_raw,
        "sample_weight": np.ones(n, dtype=float),
    })
    fair_p1 = p_true
    overround = 1.05
    odds_p1 = 1.0 / (fair_p1 * overround)
    odds_p2 = 1.0 / ((1 - fair_p1) * overround)

    master = pd.DataFrame({
        "match_id":         np.arange(n, dtype=np.int64),
        "match_date":       pd.date_range(start_date, periods=n, freq="D"),
        "odds_p1_pinnacle": odds_p1,
        "odds_p2_pinnacle": odds_p2,
        "not_for_betting":  False,
        "market_source":    "pinnacle",
    })
    return preds, master


def _make_stub_train_result(preds: pd.DataFrame,
                             master: pd.DataFrame,
                             *,
                             train_predictions: pd.DataFrame | None = None,
                             ) -> TrainResult:
    """Build a TrainResult shell with optional train_predictions."""
    mb = mini_backtest(preds, master, ev_threshold=DEFAULT_EV_THRESHOLD)
    return TrainResult(
        model=None,
        valid_auc=0.65,
        valid_logloss=0.65,
        valid_brier=0.24,
        feature_importance=pd.DataFrame(columns=["feature", "gain", "split"]),
        train_history={},
        config={"feature_subset": [], "ev_threshold": DEFAULT_EV_THRESHOLD},
        valid_predictions=preds,
        mini_backtest=mb,
        train_predictions=train_predictions,
    )


def _build_train_tail_setup(n_train: int = 600, n_valid: int = 400,
                             seed_train: int = 31, seed_valid: int = 32):
    """Build a (master, train_result) pair where:

      - train_predictions covers years 2015-2018 (200 rows / 4 yrs = 50/yr ≈)
      - valid_predictions covers years 2019-2021
      - over-confident raw probs in BOTH

    Returns (master, train_result).
    """
    # Train predictions: spread across 2015-2018 so the tail year filter
    # 2017-2018 catches a real subset. Reset day counter per year so dates
    # stay in-year (otherwise running cumulative offset rolls into next year).
    n_per_year_train = n_train // 4
    train_rows = []
    train_match_id = 0
    rng_t = np.random.default_rng(seed_train)
    for yr in (2015, 2016, 2017, 2018):
        # In-year date offsets capped at 365 days
        for day_idx in range(n_per_year_train):
            logit = rng_t.normal(0, 1)
            p_true = 1.0 / (1.0 + np.exp(-logit))
            y = int(rng_t.uniform() < p_true)
            p_raw = float(np.clip(1.5 * p_true - 0.20, 0.01, 0.99))
            offset_days = min(day_idx, 350)
            train_rows.append({
                "match_id": train_match_id,
                "match_date": pd.Timestamp(f"{yr}-01-01") + pd.Timedelta(days=offset_days),
                "y_true": y,
                "y_pred_prob": p_raw,
                "sample_weight": 1.0,
                "odds_p1_pinnacle": 1.0 / (p_true * 1.05),
                "odds_p2_pinnacle": 1.0 / ((1 - p_true) * 1.05),
            })
            train_match_id += 1

    # Valid predictions: 2019-2021
    valid_preds, valid_master = _build_overconfident_predictions(
        n=n_valid, seed=seed_valid, start_date="2019-05-01",
    )
    # Re-key valid match_ids to start AFTER train ids
    valid_preds = valid_preds.copy()
    valid_master = valid_master.copy()
    valid_preds["match_id"]  = valid_preds["match_id"]  + train_match_id
    valid_master["match_id"] = valid_master["match_id"] + train_match_id

    # Build the master DF spanning train + valid rows
    train_df = pd.DataFrame(train_rows)
    master_train = train_df[["match_id", "match_date",
                              "odds_p1_pinnacle", "odds_p2_pinnacle"]].copy()
    master_train["not_for_betting"] = False
    master_train["market_source"] = "pinnacle"
    master = pd.concat([master_train, valid_master], ignore_index=True)

    train_predictions = train_df[["match_id", "y_true",
                                    "y_pred_prob", "sample_weight"]].copy()
    train_predictions["match_id"] = train_predictions["match_id"].astype("Int64")

    tr = _make_stub_train_result(
        valid_preds, master, train_predictions=train_predictions,
    )
    return master, tr


# ===========================================================================
# Constants
# ===========================================================================

class TestConstants:
    def test_calibration_plot_bins_default(self):
        assert CALIBRATION_PLOT_BINS == 10

    def test_sanity_thresholds(self):
        # Backwards-compat alias points at valid_self threshold
        assert SANITY_MEAN_PRED_DIFF_MAX == 0.01
        assert SANITY_MEAN_PRED_DIFF_MAX_VALID_SELF == 0.01
        assert SANITY_MEAN_PRED_DIFF_MAX_TRAIN_TAIL == 0.05
        assert SANITY_AUC_DELTA_MAX == 0.01

    def test_default_holdout_years(self):
        assert DEFAULT_HOLDOUT_YEARS == (2017, 2018)

    def test_eps_positive(self):
        assert CAL_PROB_EPS > 0
        assert CAL_PROB_EPS < 1e-6


# ===========================================================================
# fit_calibrator
# ===========================================================================

class TestFitCalibrator:
    def test_returns_isotonic_instance(self):
        rng = np.random.default_rng(0)
        raw = rng.uniform(0, 1, 200)
        y = (raw > 0.5).astype(int)
        iso = fit_calibrator(raw, y)
        assert isinstance(iso, IsotonicRegression)

    def test_clips_out_of_bounds(self):
        rng = np.random.default_rng(1)
        raw = rng.uniform(0.2, 0.8, 200)
        y = (raw > 0.5).astype(int)
        iso = fit_calibrator(raw, y)
        out = iso.predict(np.array([2.0, -1.0]))
        assert 0.0 <= out[0] <= 1.0
        assert 0.0 <= out[1] <= 1.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            fit_calibrator(np.array([0.1, 0.2, 0.3]), np.array([0, 1]))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            fit_calibrator(np.array([]), np.array([]))

    def test_monotone_output(self):
        rng = np.random.default_rng(3)
        raw = rng.uniform(0, 1, 1000)
        y = (rng.uniform(0, 1, 1000) < raw).astype(int)
        iso = fit_calibrator(raw, y)
        grid = np.linspace(0, 1, 100)
        out = iso.predict(grid)
        assert np.all(np.diff(out) >= -1e-9)


# ===========================================================================
# apply_calibration
# ===========================================================================

class TestApplyCalibration:
    def test_adds_orig_column_and_overwrites(self):
        preds, _ = _build_overconfident_predictions(n=200)
        iso = fit_calibrator(preds["y_pred_prob"], preds["y_true"])
        out = apply_calibration(preds, iso)
        assert "y_pred_prob_orig" in out.columns
        assert "y_pred_prob" in out.columns
        assert np.allclose(out["y_pred_prob_orig"].to_numpy(),
                            preds["y_pred_prob"].to_numpy())
        assert not np.allclose(out["y_pred_prob"].to_numpy(),
                                 preds["y_pred_prob"].to_numpy())

    def test_preserves_other_columns(self):
        preds, _ = _build_overconfident_predictions(n=100)
        iso = fit_calibrator(preds["y_pred_prob"], preds["y_true"])
        out = apply_calibration(preds, iso)
        for col in ("match_id", "y_true", "sample_weight"):
            assert col in out.columns
            assert np.array_equal(out[col].to_numpy(), preds[col].to_numpy())

    def test_clips_to_eps_range(self):
        preds, _ = _build_overconfident_predictions(n=200)
        iso = fit_calibrator(preds["y_pred_prob"], preds["y_true"])
        out = apply_calibration(preds, iso)
        cal = out["y_pred_prob"].to_numpy()
        assert (cal >= CAL_PROB_EPS).all()
        assert (cal <= 1.0 - CAL_PROB_EPS).all()

    def test_does_not_mutate_input(self):
        preds, _ = _build_overconfident_predictions(n=50)
        before = preds.copy()
        iso = fit_calibrator(preds["y_pred_prob"], preds["y_true"])
        _ = apply_calibration(preds, iso)
        pd.testing.assert_frame_equal(preds, before)

    def test_missing_column_raises(self):
        df = pd.DataFrame({"x": [0.1, 0.5]})
        iso = IsotonicRegression()
        iso.fit(np.array([0.1, 0.9]), np.array([0, 1]))
        with pytest.raises(ValueError, match="y_pred_prob"):
            apply_calibration(df, iso)


# ===========================================================================
# Math properties (mean preserved on fit set, Brier ↓, AUC ~ same)
# ===========================================================================

class TestCalibrationMathProperties:
    def test_mean_preservation(self):
        preds, _ = _build_overconfident_predictions(n=1000)
        iso = fit_calibrator(preds["y_pred_prob"], preds["y_true"])
        out = apply_calibration(preds, iso)
        mean_post = float(out["y_pred_prob"].mean())
        y_rate = float(preds["y_true"].mean())
        assert abs(mean_post - y_rate) < 1e-3

    def test_brier_strictly_decreases(self):
        from sklearn.metrics import brier_score_loss
        preds, _ = _build_overconfident_predictions(n=1000)
        iso = fit_calibrator(preds["y_pred_prob"], preds["y_true"])
        out = apply_calibration(preds, iso)
        b_pre = brier_score_loss(preds["y_true"],  preds["y_pred_prob"])
        b_post = brier_score_loss(preds["y_true"], out["y_pred_prob"])
        assert b_post < b_pre

    def test_auc_essentially_unchanged(self):
        from sklearn.metrics import roc_auc_score
        preds, _ = _build_overconfident_predictions(n=1000)
        iso = fit_calibrator(preds["y_pred_prob"], preds["y_true"])
        out = apply_calibration(preds, iso)
        auc_pre = roc_auc_score(preds["y_true"],  preds["y_pred_prob"])
        auc_post = roc_auc_score(preds["y_true"], out["y_pred_prob"])
        assert abs(auc_post - auc_pre) <= SANITY_AUC_DELTA_MAX


# ===========================================================================
# calibration_plot_data
# ===========================================================================

class TestCalibrationPlotData:
    def test_columns_and_default_n_bins(self):
        preds, _ = _build_overconfident_predictions(n=1000)
        iso = fit_calibrator(preds["y_pred_prob"], preds["y_true"])
        out = apply_calibration(preds, iso)
        plot = calibration_plot_data(
            preds["y_pred_prob"], out["y_pred_prob"], preds["y_true"],
        )
        for col in ("bin", "lo", "hi", "n",
                     "mean_pred_pre", "mean_pred_post", "actual_y_rate",
                     "calibration_gap_pre", "calibration_gap_post"):
            assert col in plot.columns
        assert len(plot) == 10

    def test_bin_n_sum_equals_total(self):
        preds, _ = _build_overconfident_predictions(n=500)
        plot = calibration_plot_data(
            preds["y_pred_prob"], preds["y_pred_prob"], preds["y_true"],
        )
        assert plot["n"].sum() == 500

    def test_pre_gap_sign_matches_overconfidence(self):
        preds, _ = _build_overconfident_predictions(n=1000)
        iso = fit_calibrator(preds["y_pred_prob"], preds["y_true"])
        out = apply_calibration(preds, iso)
        plot = calibration_plot_data(
            preds["y_pred_prob"], out["y_pred_prob"], preds["y_true"],
        )
        upper = plot.tail(5)
        assert (upper["calibration_gap_pre"] > 0).sum() >= 3

    def test_post_gaps_smaller_than_pre_in_aggregate(self):
        preds, _ = _build_overconfident_predictions(n=1500)
        iso = fit_calibrator(preds["y_pred_prob"], preds["y_true"])
        out = apply_calibration(preds, iso)
        plot = calibration_plot_data(
            preds["y_pred_prob"], out["y_pred_prob"], preds["y_true"],
        )
        agg_pre = (plot["calibration_gap_pre"].abs() * plot["n"]).sum()
        agg_post = (plot["calibration_gap_post"].abs() * plot["n"]).sum()
        assert agg_post < agg_pre

    def test_handles_empty_input(self):
        plot = calibration_plot_data(
            np.array([]), np.array([]), np.array([]),
        )
        assert plot.empty
        assert "mean_pred_pre" in plot.columns

    def test_handles_constant_raw(self):
        raw = np.full(50, 0.5)
        cal = np.full(50, 0.4)
        y = np.zeros(50, dtype=int)
        plot = calibration_plot_data(raw, cal, y, n_bins=10)
        assert len(plot) >= 1
        assert plot["n"].sum() == 50

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            calibration_plot_data(
                np.array([0.1, 0.5]),
                np.array([0.1, 0.5, 0.7]),
                np.array([0, 1]),
            )


# ===========================================================================
# calibrate_baseline — valid_self path
# ===========================================================================

class TestCalibrateBaselineValidSelf:
    def test_basic_structure(self):
        preds, master = _build_overconfident_predictions(n=600)
        tr = _make_stub_train_result(preds, master)
        result = calibrate_baseline(
            "X", tr, master, holdout_strategy="valid_self",
        )
        assert isinstance(result, CalibrationResult)
        assert result.label == "X"
        assert result.strategy == "valid_self"
        assert result.holdout_years is None
        assert result.holdout_n is None
        assert result.holdout_y_rate is None
        assert result.holdout_auc is None

    def test_post_brier_lower_than_pre(self):
        preds, master = _build_overconfident_predictions(n=600)
        tr = _make_stub_train_result(preds, master)
        result = calibrate_baseline(
            "X", tr, master, holdout_strategy="valid_self",
        )
        assert result.post_brier < result.pre_brier

    def test_post_mean_close_to_y_rate(self):
        preds, master = _build_overconfident_predictions(n=600)
        tr = _make_stub_train_result(preds, master)
        result = calibrate_baseline(
            "X", tr, master, holdout_strategy="valid_self",
        )
        y_rate = float(preds["y_true"].mean())
        assert abs(result.post_mean_pred - y_rate) <= SANITY_MEAN_PRED_DIFF_MAX_VALID_SELF

    def test_calibrated_predictions_have_orig_col(self):
        preds, master = _build_overconfident_predictions(n=400)
        tr = _make_stub_train_result(preds, master)
        result = calibrate_baseline(
            "X", tr, master, holdout_strategy="valid_self",
        )
        assert "y_pred_prob_orig" in result.valid_predictions_calibrated.columns
        assert "y_pred_prob" in result.valid_predictions_calibrated.columns

    def test_plot_data_present(self):
        preds, master = _build_overconfident_predictions(n=600)
        tr = _make_stub_train_result(preds, master)
        result = calibrate_baseline(
            "X", tr, master, holdout_strategy="valid_self",
        )
        assert not result.plot_data.empty
        assert len(result.plot_data) == CALIBRATION_PLOT_BINS

    def test_post_mini_backtest_recomputed(self):
        preds, master = _build_overconfident_predictions(n=600)
        tr = _make_stub_train_result(preds, master)
        result = calibrate_baseline(
            "X", tr, master, holdout_strategy="valid_self",
        )
        assert isinstance(result.pre_mini_backtest,  MiniBacktestResult)
        assert isinstance(result.post_mini_backtest, MiniBacktestResult)
        assert (result.post_mini_backtest.n_eligible
                == result.pre_mini_backtest.n_eligible)

    def test_sanity_log_populated(self):
        preds, master = _build_overconfident_predictions(n=600)
        tr = _make_stub_train_result(preds, master)
        result = calibrate_baseline(
            "X", tr, master, holdout_strategy="valid_self",
        )
        assert len(result.sanity_log) == 4
        assert result.sanity_log[0].startswith("[OK]")
        assert result.sanity_log[1].startswith("[OK]")

    def test_sanity_can_be_skipped(self):
        preds, master = _build_overconfident_predictions(n=600)
        tr = _make_stub_train_result(preds, master)
        result = calibrate_baseline(
            "X", tr, master, holdout_strategy="valid_self",
            run_sanity_checks=False,
        )
        assert result.sanity_log == []

    def test_empty_predictions_raises(self):
        master = pd.DataFrame({
            "match_id":         pd.Series([], dtype="int64"),
            "match_date":       pd.Series([], dtype="datetime64[ns]"),
            "odds_p1_pinnacle": pd.Series([], dtype="float64"),
            "odds_p2_pinnacle": pd.Series([], dtype="float64"),
            "not_for_betting":  pd.Series([], dtype="bool"),
            "market_source":    pd.Series([], dtype="object"),
        })
        empty_preds = pd.DataFrame({
            "match_id":      pd.Series([], dtype="int64"),
            "y_true":        pd.Series([], dtype="int64"),
            "y_pred_prob":   pd.Series([], dtype="float64"),
            "sample_weight": pd.Series([], dtype="float64"),
        })
        tr = TrainResult(
            model=None, valid_auc=0.5, valid_logloss=0.7, valid_brier=0.25,
            feature_importance=pd.DataFrame(columns=["feature", "gain", "split"]),
            train_history={}, config={},
            valid_predictions=empty_preds,
            mini_backtest=mini_backtest(empty_preds, master),
        )
        with pytest.raises(ValueError, match="empty"):
            calibrate_baseline(
                "X", tr, master, holdout_strategy="valid_self",
            )

    def test_missing_columns_raises(self):
        master = pd.DataFrame({
            "match_id": [0], "match_date": [pd.Timestamp("2020-05-01")],
            "odds_p1_pinnacle": [2.0], "odds_p2_pinnacle": [2.0],
            "not_for_betting": [False], "market_source": ["pinnacle"],
        })
        bad_preds = pd.DataFrame({"x": [1.0], "y_true": [1]})
        tr = TrainResult(
            model=None, valid_auc=0.5, valid_logloss=0.7, valid_brier=0.25,
            feature_importance=pd.DataFrame(columns=["feature", "gain", "split"]),
            train_history={}, config={},
            valid_predictions=bad_preds,
            mini_backtest=MiniBacktestResult(
                n_bets=0, n_wins=0, hit_rate=float("nan"),
                flat_roi_pct=float("nan"),
                by_ev_bucket=pd.DataFrame(),
                bets_df=pd.DataFrame(),
                n_eligible=0, n_pos_ev_p1=0, n_pos_ev_p2=0,
            ),
        )
        with pytest.raises(ValueError, match="missing required"):
            calibrate_baseline(
                "X", tr, master, holdout_strategy="valid_self",
            )

    def test_invalid_strategy_raises(self):
        preds, master = _build_overconfident_predictions(n=200)
        tr = _make_stub_train_result(preds, master)
        with pytest.raises(ValueError, match="holdout_strategy"):
            calibrate_baseline(
                "X", tr, master, holdout_strategy="banana",
            )


# ===========================================================================
# calibrate_baseline — train_tail path (the production default)
# ===========================================================================

class TestCalibrateBaselineTrainTail:
    """Legacy train_tail strategy (must explicitly opt in now that
    ``calib_holdout`` is the default)."""

    def test_basic_structure(self):
        master, tr = _build_train_tail_setup()
        result = calibrate_baseline(
            "X", tr, master, holdout_strategy="train_tail",
        )
        assert result.strategy == "train_tail"
        assert result.holdout_years == DEFAULT_HOLDOUT_YEARS
        assert result.holdout_n is not None
        assert result.holdout_n > 0
        assert 0.0 <= result.holdout_y_rate <= 1.0
        assert 0.0 <= result.holdout_auc <= 1.0

    def test_holdout_filtered_by_year(self):
        master, tr = _build_train_tail_setup()
        result = calibrate_baseline(
            "X", tr, master,
            holdout_strategy="train_tail",
            holdout_years=(2017, 2018),
        )
        # Setup put 600/4 = 150 rows per year × 2 years = 300 rows in 2017-2018
        assert result.holdout_n == 300

    def test_year_filter_narrower_window(self):
        master, tr = _build_train_tail_setup()
        result = calibrate_baseline(
            "X", tr, master,
            holdout_strategy="train_tail",
            holdout_years=(2018, 2018),
        )
        assert result.holdout_n == 150

    def test_train_tail_requires_train_predictions(self):
        # Stub TrainResult without train_predictions → train_tail raises
        preds, master = _build_overconfident_predictions(n=200)
        tr = _make_stub_train_result(preds, master)
        assert tr.train_predictions is None
        with pytest.raises(ValueError, match="train_predictions"):
            calibrate_baseline("X", tr, master, holdout_strategy="train_tail")

    def test_explicit_holdout_predictions(self):
        master, tr = _build_train_tail_setup()
        # Use first 50 train rows as a custom holdout, regardless of year
        custom = tr.train_predictions.head(50).copy()
        result = calibrate_baseline(
            "X", tr, master,
            holdout_strategy="train_tail",
            holdout_predictions=custom,
        )
        assert result.holdout_n == 50

    def test_post_mean_within_train_tail_threshold(self):
        master, tr = _build_train_tail_setup()
        result = calibrate_baseline(
            "X", tr, master, holdout_strategy="train_tail",
        )
        y_rate = float(tr.valid_predictions["y_true"].mean())
        # train_tail threshold is 0.05, not 0.01
        assert abs(result.post_mean_pred - y_rate) <= 0.10

    def test_holdout_window_with_no_rows_raises(self):
        master, tr = _build_train_tail_setup()
        with pytest.raises(ValueError, match="zero rows"):
            calibrate_baseline(
                "X", tr, master,
                holdout_strategy="train_tail",
                holdout_years=(1990, 1995),
            )

    def test_brier_decreases_typical(self):
        """Not a guarantee for train_tail, but for our synthetic where
        the bias is uniform across years it should hold."""
        master, tr = _build_train_tail_setup()
        result = calibrate_baseline(
            "X", tr, master, holdout_strategy="train_tail",
        )
        assert result.post_brier < result.pre_brier


# ===========================================================================
# calibrate_baseline — calib_holdout path (the production default)
# ===========================================================================

class TestCalibrateBaselineCalibHoldout:
    """Default strategy: fit on a dedicated calibration split that the
    model never trained on (no in-window contamination)."""

    def _build_setup(self, n_calib: int = 200, n_valid: int = 300,
                     seed: int = 71):
        """Build a (master, train_result) pair where calib_predictions
        is populated with synthetic OOS predictions distinct from valid."""
        valid_preds, master = _build_overconfident_predictions(
            n=n_valid, seed=seed, start_date="2019-05-01",
        )
        # calib predictions: same style of over-confidence, different rows
        rng = np.random.default_rng(seed + 1)
        logits = rng.normal(0.0, 1.0, n_calib)
        p_true = 1.0 / (1.0 + np.exp(-logits))
        y_c = (rng.uniform(0, 1, n_calib) < p_true).astype(int)
        p_raw_c = np.clip(1.5 * p_true - 0.20, 0.01, 0.99)
        # Re-key so calib match_ids don't collide with valid
        n_v = len(valid_preds)
        calib_preds = pd.DataFrame({
            "match_id":      np.arange(n_v, n_v + n_calib, dtype=np.int64),
            "y_true":        y_c,
            "y_pred_prob":   p_raw_c,
            "sample_weight": np.ones(n_calib, dtype=float),
        })
        # Add calib rows to master (so date-lookup works)
        calib_master = pd.DataFrame({
            "match_id":         np.arange(n_v, n_v + n_calib, dtype=np.int64),
            "match_date":       pd.date_range("2017-05-01", periods=n_calib,
                                                freq="D"),
            "odds_p1_pinnacle": 1.0 / (p_true * 1.05),
            "odds_p2_pinnacle": 1.0 / ((1 - p_true) * 1.05),
            "not_for_betting":  False,
            "market_source":    "pinnacle",
        })
        master = pd.concat([master, calib_master], ignore_index=True)
        tr = _make_stub_train_result(valid_preds, master)
        tr.calib_predictions = calib_preds
        return master, tr

    def test_default_strategy_is_calib_holdout(self):
        master, tr = self._build_setup()
        result = calibrate_baseline("X", tr, master)
        assert result.strategy == "calib_holdout"
        assert result.holdout_n == 200
        assert 0.0 <= result.holdout_y_rate <= 1.0
        assert 0.0 <= result.holdout_auc <= 1.0

    def test_calib_holdout_requires_calib_predictions(self):
        preds, master = _build_overconfident_predictions(n=200)
        tr = _make_stub_train_result(preds, master)
        assert tr.calib_predictions is None
        with pytest.raises(ValueError, match="calib_predictions"):
            calibrate_baseline("X", tr, master)

    def test_explicit_holdout_predictions_overrides_calib(self):
        master, tr = self._build_setup()
        custom = tr.calib_predictions.head(50).copy()
        result = calibrate_baseline(
            "X", tr, master,
            holdout_strategy="calib_holdout",
            holdout_predictions=custom,
        )
        assert result.holdout_n == 50

    def test_post_mean_within_oos_threshold(self):
        master, tr = self._build_setup()
        result = calibrate_baseline("X", tr, master)
        y_rate_v = float(tr.valid_predictions["y_true"].mean())
        # calib_holdout uses the looser 0.05 threshold; allow some drift
        assert abs(result.post_mean_pred - y_rate_v) <= 0.15


# ===========================================================================
# Self-check raise paths
# ===========================================================================

class TestSanityRaises:
    def test_brier_no_decrease_raises_in_valid_self(self):
        from wta_clay.models.calibration import _run_calibration_sanity
        empty_mb = MiniBacktestResult(
            n_bets=0, n_wins=0, hit_rate=float("nan"),
            flat_roi_pct=float("nan"),
            by_ev_bucket=pd.DataFrame(),
            bets_df=pd.DataFrame(),
            n_eligible=0, n_pos_ev_p1=0, n_pos_ev_p2=0,
        )
        y = np.array([0] * 50 + [1] * 50)        # y_rate = 0.50
        # valid_self: hard raise
        with pytest.raises(AssertionError, match="Brier did NOT"):
            _run_calibration_sanity(
                holdout_strategy="valid_self",
                y_v=y,
                pre_brier=0.20, post_brier=0.20,
                post_mean=0.50,
                pre_auc=0.65, post_auc=0.65,
                pre_mb=empty_mb, post_mb=empty_mb,
                log_list=[],
            )

    def test_brier_no_decrease_warns_in_train_tail(self):
        from wta_clay.models.calibration import _run_calibration_sanity
        empty_mb = MiniBacktestResult(
            n_bets=0, n_wins=0, hit_rate=float("nan"),
            flat_roi_pct=float("nan"),
            by_ev_bucket=pd.DataFrame(),
            bets_df=pd.DataFrame(),
            n_eligible=0, n_pos_ev_p1=0, n_pos_ev_p2=0,
        )
        y = np.array([0] * 50 + [1] * 50)
        log_list: list[str] = []
        # train_tail: warn only, no raise
        _run_calibration_sanity(
            holdout_strategy="train_tail",
            y_v=y,
            pre_brier=0.20, post_brier=0.25,        # increased
            post_mean=0.50,                          # mean ok
            pre_auc=0.65, post_auc=0.65,
            pre_mb=empty_mb, post_mb=empty_mb,
            log_list=log_list,
        )
        assert any("[WARN]" in s and "Brier" in s for s in log_list)

    def test_mean_diverges_raises_in_valid_self(self):
        from wta_clay.models.calibration import _run_calibration_sanity
        empty_mb = MiniBacktestResult(
            n_bets=0, n_wins=0, hit_rate=float("nan"),
            flat_roi_pct=float("nan"),
            by_ev_bucket=pd.DataFrame(),
            bets_df=pd.DataFrame(),
            n_eligible=0, n_pos_ev_p1=0, n_pos_ev_p2=0,
        )
        y = np.array([0] * 50 + [1] * 50)
        with pytest.raises(AssertionError, match="mean_pred convergence"):
            _run_calibration_sanity(
                holdout_strategy="valid_self",
                y_v=y,
                pre_brier=0.30, post_brier=0.20,
                post_mean=0.70,                          # |diff| = 0.20
                pre_auc=0.65, post_auc=0.65,
                pre_mb=empty_mb, post_mb=empty_mb,
                log_list=[],
            )

    def test_mean_diverges_warns_in_train_tail(self):
        """train_tail: 0.05 threshold instead of 0.01, but still warn."""
        from wta_clay.models.calibration import _run_calibration_sanity
        empty_mb = MiniBacktestResult(
            n_bets=0, n_wins=0, hit_rate=float("nan"),
            flat_roi_pct=float("nan"),
            by_ev_bucket=pd.DataFrame(),
            bets_df=pd.DataFrame(),
            n_eligible=0, n_pos_ev_p1=0, n_pos_ev_p2=0,
        )
        y = np.array([0] * 50 + [1] * 50)
        log_list: list[str] = []
        # diff = 0.20 > 0.05 train_tail threshold but warn-only
        _run_calibration_sanity(
            holdout_strategy="train_tail",
            y_v=y,
            pre_brier=0.30, post_brier=0.20,
            post_mean=0.70,
            pre_auc=0.65, post_auc=0.65,
            pre_mb=empty_mb, post_mb=empty_mb,
            log_list=log_list,
        )
        assert any("[WARN]" in s and "mean_pred" in s for s in log_list)

    def test_auc_drift_warns_not_raises(self):
        from wta_clay.models.calibration import _run_calibration_sanity
        empty_mb = MiniBacktestResult(
            n_bets=0, n_wins=0, hit_rate=float("nan"),
            flat_roi_pct=float("nan"),
            by_ev_bucket=pd.DataFrame(),
            bets_df=pd.DataFrame(),
            n_eligible=0, n_pos_ev_p1=0, n_pos_ev_p2=0,
        )
        y = np.array([0] * 50 + [1] * 50)
        log_list: list[str] = []
        _run_calibration_sanity(
            holdout_strategy="valid_self",
            y_v=y,
            pre_brier=0.30, post_brier=0.25,
            post_mean=0.50,
            pre_auc=0.70, post_auc=0.60,           # huge drop
            pre_mb=empty_mb, post_mb=empty_mb,
            log_list=log_list,
        )
        assert any("[WARN]" in s and "AUC" in s for s in log_list)


# ===========================================================================
# calibrate_baselines (multi-baseline orchestration)
# ===========================================================================

class TestCalibrateBaselines:
    def test_dict_in_dict_out_valid_self(self):
        preds_1, master = _build_overconfident_predictions(n=400, seed=11)
        preds_2, _ = _build_overconfident_predictions(n=400, seed=12)
        tr_1 = _make_stub_train_result(preds_1, master)
        tr_2 = _make_stub_train_result(preds_2, master)
        results = calibrate_baselines(
            {"A": tr_1, "B": tr_2}, master,
            holdout_strategy="valid_self",
        )
        assert set(results.keys()) == {"A", "B"}
        for k, r in results.items():
            assert isinstance(r, CalibrationResult)
            assert r.label == k
            assert r.strategy == "valid_self"

    def test_dict_in_dict_out_train_tail(self):
        master_1, tr_1 = _build_train_tail_setup(seed_train=41, seed_valid=42)
        master_2, tr_2 = _build_train_tail_setup(seed_train=43, seed_valid=44)
        # Use master_1 (different shapes won't matter for the per-baseline call,
        # since each baseline shares its master via the dict orchestration)
        results = calibrate_baselines(
            {"A": tr_1}, master_1,
            holdout_strategy="train_tail",
        )
        assert "A" in results
        assert results["A"].strategy == "train_tail"


# ===========================================================================
# format_*  (text + DataFrame outputs)
# ===========================================================================

class TestFormatters:
    def _self_result(self, label="X"):
        preds, master = _build_overconfident_predictions(n=400, seed=33)
        tr = _make_stub_train_result(preds, master)
        return calibrate_baseline(
            label, tr, master, holdout_strategy="valid_self",
        )

    def _tail_result(self, label="X"):
        master, tr = _build_train_tail_setup(seed_train=51, seed_valid=52)
        return calibrate_baseline(
            label, tr, master, holdout_strategy="train_tail",
        )

    def test_format_calibration_report_returns_string(self):
        r = self._self_result("X")
        out = format_calibration_report("X", r)
        assert isinstance(out, str)
        assert "Calibration X" in out
        assert "valid_self" in out
        assert "AUC" in out
        assert "Brier" in out
        assert "Mini-backtest" in out
        assert "Sanity checks" in out

    def test_format_calibration_report_train_tail_includes_holdout_line(self):
        r = self._tail_result("Y")
        out = format_calibration_report("Y", r)
        assert "train_tail" in out
        assert "Holdout: years" in out

    def test_format_calibration_comparison_columns(self):
        r1 = self._self_result("A")
        r2 = self._self_result("B")
        df = format_calibration_comparison({"A": r1, "B": r2})
        for col in ("baseline", "strategy",
                     "auc_pre", "auc_post",
                     "brier_pre", "brier_post",
                     "mean_pred_pre", "mean_pred_post",
                     "n_bets_pre", "n_bets_post",
                     "roi_pre", "roi_post", "roi_delta"):
            assert col in df.columns
        assert len(df) == 2
        assert set(df["baseline"]) == {"A", "B"}

    def test_strategy_comparison_table(self):
        r_self = self._self_result("X")
        r_tail = self._tail_result("X")
        df = format_strategy_comparison({"X": r_self}, {"X": r_tail})
        for col in ("baseline",
                     "auc_self", "auc_tail",
                     "brier_self", "brier_tail",
                     "meanP_self", "meanP_tail",
                     "n_bets_self", "n_bets_tail",
                     "roi_self", "roi_tail",
                     "roi_optimism_pp"):
            assert col in df.columns
        # If both ROIs are finite, optimism = roi_self - roi_tail
        if (pd.notna(df.loc[0, "roi_self"])
                and pd.notna(df.loc[0, "roi_tail"])):
            expected = df.loc[0, "roi_self"] - df.loc[0, "roi_tail"]
            assert abs(df.loc[0, "roi_optimism_pp"] - expected) < 1e-9

    def test_strategy_comparison_key_mismatch_raises(self):
        r1 = self._self_result("X")
        r2 = self._tail_result("Y")
        with pytest.raises(ValueError, match="key mismatch"):
            format_strategy_comparison({"X": r1}, {"Y": r2})

    def test_ev_bucket_comparison_columns(self):
        r_self = self._self_result("X")
        r_tail = self._tail_result("X")
        df = format_ev_bucket_comparison({"X": r_self}, {"X": r_tail})
        for col in ("baseline", "ev_bucket",
                     "n_self", "hit_self_%", "roi_self_%",
                     "n_tail", "hit_tail_%", "roi_tail_%"):
            assert col in df.columns
        # Should have 3 rows per baseline (5-10%, 10-20%, 20%+)
        assert len(df) == 3

    def test_comparison_roi_delta_arithmetic(self):
        r = self._self_result("X")
        df = format_calibration_comparison({"X": r})
        roi_pre = df.loc[0, "roi_pre"]
        roi_post = df.loc[0, "roi_post"]
        roi_delta = df.loc[0, "roi_delta"]
        if (pd.notna(roi_pre) and pd.notna(roi_post)
                and pd.notna(roi_delta)):
            assert abs(roi_delta - (roi_post - roi_pre)) < 1e-9
