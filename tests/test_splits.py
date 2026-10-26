"""Unit tests for wta_clay.models.splits."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wta_clay.features.io import write_features
from wta_clay.models import splits as splits_mod
from wta_clay.models.splits import (
    REQUIRED_FEATURES, REQUIRED_MASTER_COLS, TOUR_LEVELS,
    SplitResult, make_splits,
)


# ---------------------------------------------------------------------------
# Synthetic master + features fixtures
# ---------------------------------------------------------------------------

def _build_master(n_per_year=20, years=range(2007, 2026)):
    """Build a synthetic master table covering all required cols."""
    rows = []
    mid = 0
    for yr in years:
        for i in range(n_per_year):
            day = (i % 28) + 1
            month = 5  # all in May (clay season)
            tier = ("GS", "WTA1000", "WTA500", "WTA250", "WTA125")[i % 5]
            rows.append({
                "match_id": mid,
                "match_date": pd.Timestamp(f"{yr}-{month:02d}-{day:02d}"),
                "tourney_id": f"{yr}-T{i:03d}",
                "stable_match_key": f"{yr}-T{i:03d}_{100+i}_{200+i}",
                "p1_id": 100 + i, "p2_id": 200 + i,
                "p1_age": 25.0 + (i % 10),
                "p2_age": 27.0 + (i % 10),
                "p1_hand": "R" if i % 3 else "L",
                "p2_hand": "R",
                "tour_level": tier,
                "y": i % 2,
                "sample_weight": 1.0 if tier != "WTA125" else 0.5,
                "not_for_betting": (tier == "WTA125"),
                "market_source": "pinnacle" if i % 10 != 0 else "bet365",
            })
            mid += 1
    df = pd.DataFrame(rows)
    df["tour_level"] = pd.Categorical(df["tour_level"], categories=TOUR_LEVELS,
                                       ordered=True)
    return df


def _make_feature_parquets(tmp_path: Path, master: pd.DataFrame) -> dict:
    """Write 6 synthetic feature parquets that match the master."""
    out_dir = tmp_path / "features"
    out_dir.mkdir()
    paths = {}
    n = len(master)
    rng = np.random.default_rng(42)

    # clay_elo
    f = pd.DataFrame({
        "match_id": master["match_id"],
        "clay_elo_p1": 1500 + rng.normal(0, 100, n),
        "clay_elo_p2": 1500 + rng.normal(0, 100, n),
        "clay_elo_diff": rng.normal(0, 50, n),
        "clay_elo_bootstrap": (master["match_date"].dt.year < 2009),
    })
    p = out_dir / "clay_elo.parquet"
    write_features(f, master, path=p, feature_module="clay_elo")
    paths["clay_elo"] = p

    # form
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

    # h2h
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

    # specialist
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

    # ranking
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

    # fatigue
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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_required_features(self):
        for name in ("clay_elo", "form", "h2h", "specialist",
                      "ranking", "fatigue"):
            assert name in REQUIRED_FEATURES

    def test_required_master_cols(self):
        for c in ("match_id", "match_date", "stable_match_key",
                  "y", "sample_weight", "not_for_betting", "market_source"):
            assert c in REQUIRED_MASTER_COLS


# ---------------------------------------------------------------------------
# Schema check
# ---------------------------------------------------------------------------

class TestSchema:
    def test_master_missing_stable_key_raises(self, tmp_path):
        master = _build_master(n_per_year=2)
        master = master.drop(columns="stable_match_key")
        paths = {f: tmp_path / f"{f}.parquet" for f in REQUIRED_FEATURES}
        with pytest.raises(ValueError, match="stable_match_key"):
            make_splits(master, paths, run_sanity_checks=False)

    def test_missing_feature_module_raises(self, tmp_path):
        master = _build_master(n_per_year=2)
        paths = {"clay_elo": tmp_path / "clay_elo.parquet"}  # only 1 of 6
        with pytest.raises(ValueError, match="feature_parquets missing"):
            make_splits(master, paths, run_sanity_checks=False)


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_basic_split(self, tmp_path):
        master = _build_master(n_per_year=20)
        paths = _make_feature_parquets(tmp_path, master)
        result = make_splits(master, paths, run_sanity_checks=True)
        assert isinstance(result, SplitResult)
        # Train: 2007-2018 (12 years × 20 - 2007/2008 dropped) = ~200
        # Valid: 2019-2021 (3 × 20) = 60
        # Test: 2022-2025 (4 × 20) = 80
        assert len(result.X_train) > 0
        assert len(result.X_valid) > 0
        assert len(result.X_test) > 0

    def test_y_and_w_have_correct_lengths(self, tmp_path):
        master = _build_master(n_per_year=10)
        paths = _make_feature_parquets(tmp_path, master)
        result = make_splits(master, paths, run_sanity_checks=False)
        for X, y, w in [(result.X_train, result.y_train, result.w_train),
                        (result.X_valid, result.y_valid, result.w_valid),
                        (result.X_test, result.y_test, result.w_test)]:
            assert len(X) == len(y) == len(w)

    def test_feature_columns_consistent_across_splits(self, tmp_path):
        master = _build_master(n_per_year=10)
        paths = _make_feature_parquets(tmp_path, master)
        result = make_splits(master, paths, run_sanity_checks=False)
        assert list(result.X_train.columns) == result.feature_names
        assert list(result.X_valid.columns) == result.feature_names
        assert list(result.X_test.columns) == result.feature_names


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

class TestFilters:
    def test_drop_clay_bootstrap_removes_2007_2008(self, tmp_path):
        master = _build_master(n_per_year=10)
        paths = _make_feature_parquets(tmp_path, master)
        result = make_splits(master, paths, drop_clay_bootstrap=True,
                              run_sanity_checks=False)
        # Train rows should all be 2009+
        ids = result.match_ids_train
        train_dates = master.set_index("match_id").loc[ids, "match_date"]
        assert (train_dates.dt.year >= 2009).all()

    def test_keep_clay_bootstrap_includes_2007_2008(self, tmp_path):
        # With drop_clay_bootstrap=False AND train_years widened back to
        # (2007, 2016), 2007 + 2008 must reappear in train.
        master = _build_master(n_per_year=10)
        paths = _make_feature_parquets(tmp_path, master)
        result = make_splits(
            master, paths,
            drop_clay_bootstrap=False,
            train_years=(2007, 2016),
            run_sanity_checks=False,
        )
        ids = result.match_ids_train
        train_dates = master.set_index("match_id").loc[ids, "match_date"]
        assert (train_dates.dt.year == 2007).any()
        assert (train_dates.dt.year == 2008).any()


# ---------------------------------------------------------------------------
# Time decay
# ---------------------------------------------------------------------------

class TestTimeDecay:
    def test_lambda_zero_means_uniform_weight(self, tmp_path):
        master = _build_master(n_per_year=10)
        paths = _make_feature_parquets(tmp_path, master)
        result = make_splits(master, paths, time_decay_lambda=0.0,
                              run_sanity_checks=False)
        # All weights should equal sample_weight (1.0 or 0.5)
        # Master has not_for_betting=True for WTA125 (sample_weight=0.5),
        # but those rows are still kept in train. Both 1.0 and 0.5 exist.
        assert set(result.w_train.unique()) <= {1.0, 0.5}

    def test_lambda_positive_decays_old(self, tmp_path):
        master = _build_master(n_per_year=10)
        paths = _make_feature_parquets(tmp_path, master)
        result = make_splits(master, paths, time_decay_lambda=0.10,
                              time_decay_anchor_year=2018,
                              run_sanity_checks=False)
        # Older training rows should have smaller weights than recent ones
        ids = result.match_ids_train
        train_dates = master.set_index("match_id").loc[ids, "match_date"]
        old_idx = train_dates.dt.year == 2009
        recent_idx = train_dates.dt.year == 2017
        if old_idx.any() and recent_idx.any():
            old_mean = result.w_train[old_idx.values].mean()
            recent_mean = result.w_train[recent_idx.values].mean()
            assert old_mean < recent_mean


# ---------------------------------------------------------------------------
# Time + key non-overlap
# ---------------------------------------------------------------------------

class TestNonOverlap:
    def test_time_disjoint(self, tmp_path):
        master = _build_master(n_per_year=10)
        paths = _make_feature_parquets(tmp_path, master)
        result = make_splits(master, paths, run_sanity_checks=True)
        m = master.set_index("match_id")
        tr_max = m.loc[result.match_ids_train, "match_date"].max()
        va_min = m.loc[result.match_ids_valid, "match_date"].min()
        va_max = m.loc[result.match_ids_valid, "match_date"].max()
        te_min = m.loc[result.match_ids_test, "match_date"].min()
        assert tr_max < va_min
        assert va_max < te_min

    def test_match_id_disjoint(self, tmp_path):
        master = _build_master(n_per_year=10)
        paths = _make_feature_parquets(tmp_path, master)
        result = make_splits(master, paths, run_sanity_checks=False)
        s_tr = set(result.match_ids_train.astype(int).tolist())
        s_va = set(result.match_ids_valid.astype(int).tolist())
        s_te = set(result.match_ids_test.astype(int).tolist())
        assert not (s_tr & s_va)
        assert not (s_va & s_te)
        assert not (s_tr & s_te)


# ---------------------------------------------------------------------------
# Tour level reference dynamic selection
# ---------------------------------------------------------------------------

class TestTourLevelReference:
    def test_picks_most_populous_when_itf_zero(self, tmp_path):
        # Build master with NO ITF rows. Most populous would be one of GS/...
        master = _build_master(n_per_year=10)
        # Confirm no ITF
        assert (master["tour_level"] == "ITF").sum() == 0
        paths = _make_feature_parquets(tmp_path, master)
        result = make_splits(master, paths, run_sanity_checks=False)
        # Reference should be one of the populous tiers, NOT ITF
        assert result.tour_level_reference != "ITF"
        # Reference tier should NOT appear in feature columns
        assert f"tour_level_{result.tour_level_reference}" not in result.feature_names


# ---------------------------------------------------------------------------
# Empty
# ---------------------------------------------------------------------------

class TestEmpty:
    def test_empty_master_raises(self, tmp_path):
        master = pd.DataFrame()
        paths = {f: tmp_path / f"{f}.parquet" for f in REQUIRED_FEATURES}
        # Empty master fails schema check (no required cols)
        with pytest.raises(ValueError):
            make_splits(master, paths, run_sanity_checks=False)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

class TestDiagnostics:
    def test_drift_report_columns(self, tmp_path):
        master = _build_master(n_per_year=10)
        paths = _make_feature_parquets(tmp_path, master)
        result = make_splits(master, paths, run_sanity_checks=False)
        assert {"feature", "train_mean", "test_mean",
                "drift_sigma", "flag"}.issubset(set(result.drift_report.columns))

    def test_descriptive_stats_columns(self, tmp_path):
        master = _build_master(n_per_year=10)
        paths = _make_feature_parquets(tmp_path, master)
        result = make_splits(master, paths, run_sanity_checks=False)
        assert {"feature", "split", "n", "mean", "std",
                "min", "p10", "p90", "max"}.issubset(
            set(result.descriptive_stats.columns)
        )

    def test_drift_by_year_indexed_by_year(self, tmp_path):
        master = _build_master(n_per_year=10)
        paths = _make_feature_parquets(tmp_path, master)
        result = make_splits(master, paths, run_sanity_checks=False)
        assert "year" in result.drift_by_year.columns
