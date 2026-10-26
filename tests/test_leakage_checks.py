"""Unit tests for ``wta_clay.models.leakage_checks``."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wta_clay.features.io import write_features
from wta_clay.models import splits as splits_mod
from wta_clay.models.leakage_checks import (
    DEFAULT_N_RANDOM_PER_MODULE,
    LeakageCheckResult,
    REQUIRED_FEATURE_MODULES,
    _check_ev_bucket_monotonicity,
    _check_hash_consistency,
    _check_split_non_overlap,
    format_audit_report,
    run_leakage_audit,
)


# ---------------------------------------------------------------------------
# Synthetic master + features fixtures
# ---------------------------------------------------------------------------

def _build_master(n_per_year=10, years=range(2009, 2026)):
    rows = []
    mid = 0
    for yr in years:
        for i in range(n_per_year):
            day = (i % 28) + 1
            tier = ("GS","WTA1000","WTA500","WTA250","WTA125")[i % 5]
            rows.append({
                "match_id": mid,
                "match_date": pd.Timestamp(f"{yr}-05-{day:02d}"),
                "tourney_id": f"{yr}-T{i:03d}",
                "stable_match_key": f"{yr}-T{i:03d}_{100+i}_{200+i}",
                "p1_id": 100+i, "p2_id": 200+i,
                "p1_age": 25.0+(i%10), "p2_age": 27.0+(i%10),
                "p1_hand": "R", "p2_hand": "R",
                "tour_level": tier, "y": i % 2,
                "sample_weight": 1.0 if tier != "WTA125" else 0.5,
                "not_for_betting": (tier == "WTA125"),
                "market_source": "pinnacle",
            })
            mid += 1
    df = pd.DataFrame(rows)
    df["tour_level"] = pd.Categorical(df["tour_level"],
        categories=("GS","WTA1000","WTA500","WTA250","WTA125","ITF"),
        ordered=True)
    return df


def _make_feature_parquets(tmp_path: Path, master: pd.DataFrame) -> dict:
    out_dir = tmp_path / "features"
    out_dir.mkdir()
    paths = {}
    n = len(master)
    rng = np.random.default_rng(42)
    schemas = {
        "clay_elo": {"clay_elo_p1": 1500.0, "clay_elo_p2": 1500.0,
                     "clay_elo_diff": 0.0, "clay_elo_bootstrap": False},
        "form": {"form_winrate_p1": 0.5, "form_winrate_p2": 0.5,
                 "form_winrate_diff": 0.0,
                 "form_n_matches_p1": 15, "form_n_matches_p2": 15,
                 "form_dominance_p1": 1.0, "form_dominance_p2": 1.0,
                 "form_dominance_diff": 0.0},
        "h2h": {"h2h_n_clay": 0, "h2h_p1_wins": 0,
                "h2h_smoothed_ratio_p1": 0.5,
                "h2h_n_overall": 0, "h2h_p1_wins_overall": 0},
        "specialist": {"specialist_p1": 0.0, "specialist_p2": 0.0,
                        "specialist_diff": 0.0,
                        "specialist_n_clay_p1": 10, "specialist_n_total_p1": 50,
                        "specialist_n_clay_p2": 10, "specialist_n_total_p2": 50},
        "ranking": {"rank_p1": 50, "rank_p2": 75,
                    "rank_diff": 25, "rank_trend_p1": 0.0,
                    "rank_trend_p2": 0.0, "rank_trend_diff": 0.0},
        "fatigue": {"rest_days_p1": 7.0, "rest_days_p2": 7.0,
                    "rest_days_diff": 0.0,
                    "matches_this_week_p1": 1, "matches_this_week_p2": 1,
                    "matches_this_week_diff": 0,
                    "clay_season_cum_p1": 5, "clay_season_cum_p2": 5,
                    "clay_season_cum_diff": 0},
    }
    for mod, default in schemas.items():
        f = pd.DataFrame({"match_id": master["match_id"]})
        for col, val in default.items():
            f[col] = val
        p = out_dir / f"{mod}.parquet"
        write_features(f, master, path=p, feature_module=mod)
        paths[mod] = p
    return paths


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_module_list(self):
        assert "form" in REQUIRED_FEATURE_MODULES
        assert "specialist" in REQUIRED_FEATURE_MODULES
        assert "h2h" in REQUIRED_FEATURE_MODULES
        assert "clay_elo" in REQUIRED_FEATURE_MODULES
        assert "ranking" in REQUIRED_FEATURE_MODULES
        assert "fatigue" in REQUIRED_FEATURE_MODULES
        assert len(REQUIRED_FEATURE_MODULES) == 6

    def test_default_n_random(self):
        assert DEFAULT_N_RANDOM_PER_MODULE == 30


# ---------------------------------------------------------------------------
# Hash consistency
# ---------------------------------------------------------------------------

class TestHashConsistency:
    def test_passes_when_all_hashes_match(self, tmp_path):
        master = _build_master()
        paths = _make_feature_parquets(tmp_path, master)
        passed, log, per = _check_hash_consistency(paths)
        assert passed
        assert any("identical" in s for s in log)

    def test_fails_when_hashes_differ(self, tmp_path):
        master_a = _build_master(n_per_year=10)
        master_b = _build_master(n_per_year=20)   # different size → different hash
        d_a = tmp_path / "fa"
        d_a.mkdir()
        d_b = tmp_path / "fb"
        d_b.mkdir()
        f = pd.DataFrame({"match_id": master_a["match_id"],
                           "x": 1.0})
        write_features(f, master_a, path=d_a / "form.parquet",
                        feature_module="form")
        f2 = pd.DataFrame({"match_id": master_b["match_id"],
                            "x": 1.0})
        write_features(f2, master_b, path=d_b / "h2h.parquet",
                        feature_module="h2h")
        passed, log, per = _check_hash_consistency({
            "form": d_a / "form.parquet",
            "h2h":  d_b / "h2h.parquet",
        })
        assert not passed
        assert any("MISMATCH" in s for s in log)


# ---------------------------------------------------------------------------
# Split non-overlap
# ---------------------------------------------------------------------------

class TestSplitNonOverlap:
    def test_passes_with_disjoint_ids(self, tmp_path):
        master = _build_master()
        paths = _make_feature_parquets(tmp_path, master)
        sp = splits_mod.make_splits(master, paths, run_sanity_checks=False)
        passed, log = _check_split_non_overlap(sp)
        assert passed

    def test_fails_with_overlapping_ids(self, tmp_path):
        # Build a fake SplitResult with overlapping match_ids
        from wta_clay.models.splits import SplitResult
        base_ids = pd.Series([1, 2, 3], dtype="Int64")
        sp = SplitResult(
            X_train=pd.DataFrame(), y_train=pd.Series([], dtype="Int8"),
            w_train=pd.Series([], dtype="float64"),
            match_ids_train=base_ids,
            X_calib=pd.DataFrame(), y_calib=pd.Series([], dtype="Int8"),
            w_calib=pd.Series([], dtype="float64"),
            match_ids_calib=pd.Series([2, 4], dtype="Int64"),  # overlap with train
            X_valid=pd.DataFrame(), y_valid=pd.Series([], dtype="Int8"),
            w_valid=pd.Series([], dtype="float64"),
            match_ids_valid=pd.Series([5, 6], dtype="Int64"),
            X_test=pd.DataFrame(),  y_test=pd.Series([], dtype="Int8"),
            w_test=pd.Series([], dtype="float64"),
            match_ids_test=pd.Series([7, 8], dtype="Int64"),
            feature_names=[], config={},
        )
        passed, log = _check_split_non_overlap(sp)
        assert not passed


# ---------------------------------------------------------------------------
# EV-bucket monotonicity
# ---------------------------------------------------------------------------

class TestEvBucketMonotonicity:
    def test_warn_only_on_inversion(self):
        # Build synthetic test predictions where hit rate is INVERTED
        df = pd.DataFrame({
            "y_true": [1] * 100 + [0] * 100,
            "y_pred_prob": [0.9] * 50 + [0.7] * 50 + [0.5] * 50 + [0.3] * 50,
            "odds_p1": [1.05, 1.30, 1.50, 1.80] * 50,
            "odds_p2": [10.0, 5.0, 3.0, 1.5] * 50,
        })
        passed, log, table = _check_ev_bucket_monotonicity(df)
        # Should NOT raise either way (warn-only check)
        assert passed   # always returns True (it's a warn)

    def test_skips_when_no_required_cols(self):
        df = pd.DataFrame({"y_true": [0, 1], "y_pred_prob": [0.4, 0.6]})
        passed, log, table = _check_ev_bucket_monotonicity(df)
        assert passed
        assert any("skipped" in s.lower() for s in log)


# ---------------------------------------------------------------------------
# End-to-end orchestrator (synthetic)
# ---------------------------------------------------------------------------

class TestRunLeakageAudit:
    def test_runs_with_no_predictions(self, tmp_path):
        master = _build_master()
        paths = _make_feature_parquets(tmp_path, master)
        sp = splits_mod.make_splits(master, paths, run_sanity_checks=False)
        result = run_leakage_audit(
            master, sp, paths, test_predictions=None,
            n_random_per_module=5, n_consistency=10,
        )
        assert isinstance(result, LeakageCheckResult)
        # Hash consistency should pass
        assert result.hash_consistency_passed
        # Split non-overlap should pass
        assert result.split_non_overlap_passed

    def test_writes_audit_json(self, tmp_path):
        master = _build_master()
        paths = _make_feature_parquets(tmp_path, master)
        sp = splits_mod.make_splits(master, paths, run_sanity_checks=False)
        out = tmp_path / "audit.json"
        run_leakage_audit(
            master, sp, paths, test_predictions=None,
            n_random_per_module=5,
            out_path=out,
        )
        assert out.exists()
        import json
        d = json.loads(out.read_text())
        assert "passed" in d
        assert "hash_consistency" in d


class TestFormatReport:
    def test_returns_string(self, tmp_path):
        master = _build_master()
        paths = _make_feature_parquets(tmp_path, master)
        sp = splits_mod.make_splits(master, paths, run_sanity_checks=False)
        result = run_leakage_audit(master, sp, paths, n_random_per_module=5)
        out = format_audit_report(result)
        assert isinstance(out, str)
        assert "Leakage audit" in out
        assert "Check 1" in out
        assert "Check 5" in out
