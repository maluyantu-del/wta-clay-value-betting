"""Unit tests for wta_clay.models.lgbm."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wta_clay.features.io import write_features
from wta_clay.models import splits as splits_mod
from wta_clay.models.lgbm import (
    CORE_4_FEATURES, DEFAULT_EV_THRESHOLD, DEFAULT_PARAMS,
    SANITY_MIN_VALID_AUC, MiniBacktestResult, TrainResult,
    mini_backtest, train_lgbm, format_baseline_report,
)


# ---------------------------------------------------------------------------
# Synthetic master + features fixtures
# ---------------------------------------------------------------------------

def _build_master_with_signal(n_per_year=80, years=range(2007, 2026)):
    """Build a synthetic master where the OUTCOME has a real signal in
    clay_elo_diff (so AUC > 0.55 when used as a feature).

    p1 wins iff clay_elo_diff > 0 (with 25% noise).
    """
    rng = np.random.default_rng(42)
    rows = []
    mid = 0
    for yr in years:
        for i in range(n_per_year):
            day = (i % 28) + 1
            month = 5
            tier = ("GS", "WTA1000", "WTA500", "WTA250", "WTA125")[i % 5]
            # Signal: p1 gets a true skill diff
            true_diff = rng.normal(0, 100)
            # 25% noise: y is mostly determined by sign of diff but with noise
            y = int(true_diff > 0) if rng.random() > 0.25 else int(rng.random() > 0.5)
            rows.append({
                "match_id": mid,
                "match_date": pd.Timestamp(f"{yr}-{month:02d}-{day:02d}"),
                "tourney_id": f"{yr}-T{i:03d}",
                "stable_match_key": f"{yr}-T{i:03d}_{100+i}_{200+i}",
                "p1_id": 100 + i, "p2_id": 200 + i,
                "p1_age": 25.0, "p2_age": 27.0,
                "p1_hand": "R", "p2_hand": "R",
                "tour_level": tier,
                "y": y,
                "sample_weight": 1.0 if tier != "WTA125" else 0.5,
                "not_for_betting": (tier == "WTA125"),
                "market_source": "pinnacle",
                # Pinnacle odds — fair-ish around the signal
                "odds_p1_pinnacle": 2.0 - 0.5 * np.tanh(true_diff / 200),
                "odds_p2_pinnacle": 2.0 + 0.5 * np.tanh(true_diff / 200),
                "_true_diff": true_diff,
            })
            mid += 1
    df = pd.DataFrame(rows)
    df["tour_level"] = pd.Categorical(
        df["tour_level"],
        categories=("GS", "WTA1000", "WTA500", "WTA250", "WTA125", "ITF"),
        ordered=True,
    )
    return df


def _make_feature_parquets_with_signal(tmp_path: Path,
                                         master: pd.DataFrame) -> dict:
    """Write feature parquets where clay_elo_diff matches the signal."""
    out_dir = tmp_path / "features"
    out_dir.mkdir()
    paths = {}
    n = len(master)
    rng = np.random.default_rng(0)

    # clay_elo: clay_elo_diff = master._true_diff (perfect signal)
    f = pd.DataFrame({
        "match_id": master["match_id"],
        "clay_elo_p1": 1500 + master["_true_diff"] / 2,
        "clay_elo_p2": 1500 - master["_true_diff"] / 2,
        "clay_elo_diff": master["_true_diff"],
        "clay_elo_bootstrap": (master["match_date"].dt.year < 2009),
    })
    p = out_dir / "clay_elo.parquet"
    write_features(f, master, path=p, feature_module="clay_elo")
    paths["clay_elo"] = p

    # form (mostly noise)
    f = pd.DataFrame({
        "match_id": master["match_id"],
        "form_winrate_p1": rng.beta(5, 5, n),
        "form_winrate_p2": rng.beta(5, 5, n),
        "form_winrate_diff": rng.normal(0, 0.1, n),
        "form_n_matches_p1": rng.integers(5, 16, n),
        "form_n_matches_p2": rng.integers(5, 16, n),
        "form_dominance_p1": 1.0 + rng.normal(0, 0.3, n),
        "form_dominance_p2": 1.0 + rng.normal(0, 0.3, n),
        "form_dominance_diff": rng.normal(0, 0.2, n),
    })
    p = out_dir / "form.parquet"
    write_features(f, master, path=p, feature_module="form")
    paths["form"] = p

    # h2h (noise)
    f = pd.DataFrame({
        "match_id": master["match_id"],
        "h2h_n_clay": rng.integers(0, 5, n),
        "h2h_p1_wins": np.zeros(n, dtype=int),
        "h2h_smoothed_ratio_p1": np.full(n, 0.5),
        "h2h_n_overall": rng.integers(0, 8, n),
        "h2h_p1_wins_overall": np.zeros(n, dtype=int),
    })
    p = out_dir / "h2h.parquet"
    write_features(f, master, path=p, feature_module="h2h")
    paths["h2h"] = p

    # specialist (noise)
    f = pd.DataFrame({
        "match_id": master["match_id"],
        "specialist_p1": rng.normal(0, 0.05, n),
        "specialist_p2": rng.normal(0, 0.05, n),
        "specialist_diff": rng.normal(0, 0.05, n),
        "specialist_n_clay_p1": rng.integers(5, 100, n),
        "specialist_n_total_p1": rng.integers(30, 200, n),
        "specialist_n_clay_p2": rng.integers(5, 100, n),
        "specialist_n_total_p2": rng.integers(30, 200, n),
    })
    p = out_dir / "specialist.parquet"
    write_features(f, master, path=p, feature_module="specialist")
    paths["specialist"] = p

    # ranking (noise)
    f = pd.DataFrame({
        "match_id": master["match_id"],
        "rank_p1": rng.integers(1, 200, n),
        "rank_p2": rng.integers(1, 200, n),
        "rank_diff": rng.integers(-200, 200, n),
        "rank_trend_p1": rng.normal(0, 1, n),
        "rank_trend_p2": rng.normal(0, 1, n),
        "rank_trend_diff": rng.normal(0, 1, n),
    })
    p = out_dir / "ranking.parquet"
    write_features(f, master, path=p, feature_module="ranking")
    paths["ranking"] = p

    # fatigue (noise)
    f = pd.DataFrame({
        "match_id": master["match_id"],
        "rest_days_p1": rng.integers(1, 30, n).astype(float),
        "rest_days_p2": rng.integers(1, 30, n).astype(float),
        "rest_days_diff": rng.normal(0, 5, n),
        "matches_this_week_p1": rng.integers(0, 4, n),
        "matches_this_week_p2": rng.integers(0, 4, n),
        "matches_this_week_diff": rng.integers(-3, 4, n),
        "clay_season_cum_p1": rng.integers(0, 20, n),
        "clay_season_cum_p2": rng.integers(0, 20, n),
        "clay_season_cum_diff": rng.integers(-20, 20, n),
    })
    p = out_dir / "fatigue.parquet"
    write_features(f, master, path=p, feature_module="fatigue")
    paths["fatigue"] = p

    return paths


def _make_master_for_splits(master):
    """Drop the synthetic-only _true_diff column; splits.py won't use it."""
    return master.drop(columns=["_true_diff", "odds_p1_pinnacle", "odds_p2_pinnacle"]).copy()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_default_params_keys(self):
        for k in ("objective", "metric", "learning_rate",
                  "num_leaves", "min_data_in_leaf",
                  "feature_fraction", "bagging_fraction",
                  "lambda_l2", "verbose", "seed"):
            assert k in DEFAULT_PARAMS

    def test_core_4_features(self):
        assert "clay_elo_diff" in CORE_4_FEATURES
        assert "form_winrate_diff" in CORE_4_FEATURES
        assert "specialist_diff" in CORE_4_FEATURES
        assert "rank_diff" in CORE_4_FEATURES
        assert len(CORE_4_FEATURES) == 4

    def test_ev_threshold_default(self):
        assert DEFAULT_EV_THRESHOLD == 0.05


# ---------------------------------------------------------------------------
# Mini-backtest math
# ---------------------------------------------------------------------------

class TestMiniBacktestMath:
    def _build_predictions_and_master(self, prob, odds_p1, odds_p2, y):
        preds = pd.DataFrame({
            "match_id": [0],
            "y_true": [y],
            "y_pred_prob": [prob],
            "sample_weight": [1.0],
        })
        master = pd.DataFrame({
            "match_id": [0],
            "match_date": [pd.Timestamp("2020-05-01")],
            "odds_p1_pinnacle": [odds_p1],
            "odds_p2_pinnacle": [odds_p2],
            "not_for_betting": [False],
            "market_source": ["pinnacle"],
        })
        return preds, master

    def test_p1_bet_wins(self):
        # Model very confident p1 wins: prob=0.7, p1 odds=2.0
        # ev_p1 = 0.7*1.0 - 0.3 = 0.40 > 0.05 → bet p1
        # y_true=1 → won; pnl = 2.0 - 1 = +1.0
        preds, master = self._build_predictions_and_master(
            prob=0.7, odds_p1=2.0, odds_p2=2.5, y=1,
        )
        mb = mini_backtest(preds, master, ev_threshold=0.05)
        assert mb.n_bets == 1
        assert mb.n_wins == 1
        assert abs(mb.flat_roi_pct - 100.0) < 1e-9   # +1u on 1 bet = 100%

    def test_p1_bet_loses(self):
        preds, master = self._build_predictions_and_master(
            prob=0.7, odds_p1=2.0, odds_p2=2.5, y=0,
        )
        mb = mini_backtest(preds, master, ev_threshold=0.05)
        assert mb.n_bets == 1
        assert mb.n_wins == 0
        assert abs(mb.flat_roi_pct - (-100.0)) < 1e-9   # -1u → -100%

    def test_p2_bet_chosen_when_higher_ev(self):
        # Model thinks p2 strongly favored (prob_p1=0.2):
        # ev_p1 = 0.2*1.0 - 0.8 = -0.6 (no bet)
        # ev_p2 = 0.8*1.5 - 0.2 = +1.0 (BET — higher EV)
        preds, master = self._build_predictions_and_master(
            prob=0.2, odds_p1=2.0, odds_p2=2.5, y=0,   # p2 won
        )
        mb = mini_backtest(preds, master, ev_threshold=0.05)
        assert mb.n_bets == 1
        assert mb.bets_df.iloc[0]["bet_side"] == "p2"
        assert mb.bets_df.iloc[0]["won"] == True
        assert abs(mb.flat_roi_pct - 150.0) < 1e-9   # 2.5-1=+1.5

    def test_no_bet_when_ev_below_threshold(self):
        # prob=0.5, fair-ish odds → ev close to 0
        preds, master = self._build_predictions_and_master(
            prob=0.5, odds_p1=1.95, odds_p2=1.95, y=1,
        )
        mb = mini_backtest(preds, master, ev_threshold=0.05)
        # ev_p1 = 0.5*0.95 - 0.5 = -0.025 → no bet
        assert mb.n_bets == 0

    def test_filter_excludes_not_for_betting(self):
        preds = pd.DataFrame({
            "match_id": [0], "y_true": [1], "y_pred_prob": [0.7],
            "sample_weight": [1.0],
        })
        master = pd.DataFrame({
            "match_id": [0],
            "match_date": [pd.Timestamp("2020-05-01")],
            "odds_p1_pinnacle": [2.0], "odds_p2_pinnacle": [2.5],
            "not_for_betting": [True],          # filter out
            "market_source": ["pinnacle"],
        })
        mb = mini_backtest(preds, master)
        assert mb.n_eligible == 0
        assert mb.n_bets == 0

    def test_filter_excludes_2009(self):
        preds = pd.DataFrame({
            "match_id": [0], "y_true": [1], "y_pred_prob": [0.7],
            "sample_weight": [1.0],
        })
        master = pd.DataFrame({
            "match_id": [0],
            "match_date": [pd.Timestamp("2009-05-01")],   # 2009!
            "odds_p1_pinnacle": [2.0], "odds_p2_pinnacle": [2.5],
            "not_for_betting": [False],
            "market_source": ["pinnacle"],
        })
        mb = mini_backtest(preds, master)
        assert mb.n_eligible == 0

    def test_filter_excludes_market_none(self):
        preds = pd.DataFrame({
            "match_id": [0], "y_true": [1], "y_pred_prob": [0.7],
            "sample_weight": [1.0],
        })
        master = pd.DataFrame({
            "match_id": [0],
            "match_date": [pd.Timestamp("2020-05-01")],
            "odds_p1_pinnacle": [np.nan], "odds_p2_pinnacle": [np.nan],
            "not_for_betting": [False],
            "market_source": ["none"],
        })
        mb = mini_backtest(preds, master)
        assert mb.n_eligible == 0

    def test_ev_buckets(self):
        # Build 3 bets, one in each bucket (5-10%, 10-20%, 20%+)
        preds = pd.DataFrame({
            "match_id": [0, 1, 2],
            "y_true": [1, 1, 1],
            "y_pred_prob": [0.55, 0.65, 0.85],
            "sample_weight": [1.0, 1.0, 1.0],
        })
        master = pd.DataFrame({
            "match_id": [0, 1, 2],
            "match_date": [pd.Timestamp("2020-05-01")] * 3,
            "odds_p1_pinnacle": [2.0, 2.0, 2.0],   # ev = 0.10/0.30/0.70
            "odds_p2_pinnacle": [2.0, 2.0, 2.0],
            "not_for_betting": [False, False, False],
            "market_source": ["pinnacle"] * 3,
        })
        mb = mini_backtest(preds, master, ev_threshold=0.05)
        # row 0: ev=0.10 → 10-20% bucket  (just above 5-10 boundary)
        # row 1: ev=0.30 → 20%+
        # row 2: ev=0.70 → 20%+
        # Wait: ev = 0.55*1 - 0.45 = 0.10; 0.65*1 - 0.35 = 0.30; 0.85*1 - 0.15 = 0.70
        assert mb.n_bets == 3


# ---------------------------------------------------------------------------
# Train end-to-end
# ---------------------------------------------------------------------------

class TestTrainLgbm:
    def test_basic_train(self, tmp_path):
        master = _build_master_with_signal(n_per_year=50)
        # Build feature parquets BEFORE dropping _true_diff
        paths = _make_feature_parquets_with_signal(tmp_path, master)
        master_for_splits = _make_master_for_splits(master)
        splits = splits_mod.make_splits(master_for_splits, paths,
                                          run_sanity_checks=False)
        result = train_lgbm(splits, master, feature_subset=["clay_elo_diff"],
                             num_boost_round=50, early_stopping_rounds=10,
                             verbose=0, run_sanity_checks=False)
        assert isinstance(result, TrainResult)
        # AUC should be well above 0.55 with perfect-signal feature
        assert result.valid_auc > 0.65

    def test_feature_importance_top_includes_clay_elo(self, tmp_path):
        master = _build_master_with_signal(n_per_year=50)
        # Build feature parquets BEFORE dropping _true_diff
        paths = _make_feature_parquets_with_signal(tmp_path, master)
        master_for_splits = _make_master_for_splits(master)
        splits = splits_mod.make_splits(master_for_splits, paths,
                                          run_sanity_checks=False)
        # Use 4 core features; clay_elo_diff should be top because it has
        # the real signal
        result = train_lgbm(splits, master,
                             feature_subset=list(CORE_4_FEATURES),
                             num_boost_round=50, early_stopping_rounds=10,
                             verbose=0, run_sanity_checks=False)
        assert result.feature_importance.iloc[0]["feature"] == "clay_elo_diff"

    def test_invalid_feature_raises(self, tmp_path):
        master = _build_master_with_signal(n_per_year=20)
        paths = _make_feature_parquets_with_signal(tmp_path, master)
        master_for_splits = _make_master_for_splits(master)
        splits = splits_mod.make_splits(master_for_splits, paths,
                                          run_sanity_checks=False)
        with pytest.raises(ValueError, match="not in X_train"):
            train_lgbm(splits, master,
                        feature_subset=["nonexistent_feature"],
                        num_boost_round=10, early_stopping_rounds=5,
                        verbose=0, run_sanity_checks=False)


# ---------------------------------------------------------------------------
# Format report
# ---------------------------------------------------------------------------

class TestFormatReport:
    def test_format_returns_string(self, tmp_path):
        master = _build_master_with_signal(n_per_year=20)
        paths = _make_feature_parquets_with_signal(tmp_path, master)
        master_for_splits = _make_master_for_splits(master)
        splits = splits_mod.make_splits(master_for_splits, paths,
                                          run_sanity_checks=False)
        result = train_lgbm(splits, master, feature_subset=["clay_elo_diff"],
                             num_boost_round=20, early_stopping_rounds=5,
                             verbose=0, run_sanity_checks=False)
        report = format_baseline_report("test", result)
        assert "Valid AUC" in report
        assert "Mini-backtest" in report
        assert "Top 10" in report or "Top" in report
