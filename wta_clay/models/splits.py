"""Time-based train/calib/valid/test splits + feature matrix construction.

Produces FOUR disjoint, era-coherent slices of the master:

    Train: 2009-01-01 → 2016-12-31  (8 years)
    Calib: 2017-01-01 → 2018-12-31  (2 years)  ← NEW: calibration holdout
    Valid: 2019-01-01 → 2021-12-31  (3 years)
    Test:  2022-01-01 → 2025-12-31  (3.5+ years)

The **calib** set is brand new (2026-05-03 redesign). It is a true
out-of-sample holdout used **only** to fit the isotonic-regression
calibrator. The training pipeline never touches calib rows: train_lgbm
uses train for fitting, valid for early-stopping. The previous
"train_tail" calibration approach was contaminated because the model
HAD seen 2017-2018 during training (model AUC 0.83 on train tail vs
0.69 on valid → overfit feedback loop into calibrator). Removing
2017-2018 from training breaks that loop.

Strict time ordering: ``train.max < calib.min < valid.min < test.min``.
**No random shuffle**. The Test set is sacrosanct — only run
``model.predict(X_test)`` ONCE at final eval.

Special row handling
--------------------
- ``clay_elo_bootstrap=True`` (year < 2009) → DROPPED from train (Elo
  cold-start signal too noisy for modeling).
- ``market_source='none'`` (~56 rows) → DROPPED from all splits (no
  backtest possible later).
- ``not_for_betting=True`` (92 WTA125 rows) → KEPT in train/valid/test
  for AUC measurement; backtest layer filters them at Kelly time.
- 2009 (Pinnacle gap) → KEPT in train (training only cares about ``y``;
  market is a backtest concern). Backtest uses B365 fallback for 2009.
- ``retired=True`` (~360 rows) → KEPT (real outcomes).

Time decay weights
------------------
``time_decay_lambda`` defaults to 0.0 (= no decay, weights ≡ sample_weight).
At ablation stage you may try λ = 0.05-0.10 to weight recent matches
more (formula: ``exp(-λ × max(0, anchor_year - match_year))``). Default
disabled because clay surface and rules haven't shifted dramatically;
the decay introduces a hyperparameter without obvious upside until
the modeling stage proves otherwise.

Feature matrix X
----------------
38 columns total (locked schema, see :data:`FEATURE_NAMES_TEMPLATE`).
Reference category for the tour_level one-hot is determined dynamically
post-filter — typically WTA250 (most populous) since ITF is dropped at
master-build time.

Column families:
    Player meta:        p1_age, p2_age, p1_left_handed, p2_left_handed (4)
    Tour level one-hot: 4 cols (5 tiers minus 1 reference)
    clay-Elo:           p1, p2, diff (3)
    form:               winrate × 3, dominance × 3 (6)
    h2h:                smoothed_ratio_p1, n_clay (2)  — n_clay kept
                        because n=0 vs n=10 disambiguates smoothed=0.5
    specialist:         p1, p2, diff (3)
    ranking:            p1, p2, diff, trend × 3 (6)
    fatigue:            rest × 3, week × 3, season × 3 (9)
    Subtotal:           29 module + 4 player + 5 tier - 1 reference = 37
                        (or 38 if reference is dropped only when ITF=0)

Architecture
------------
Each feature is loaded via :func:`wta_clay.features.io.load_features`
which validates against the current master via ``master_key_hash``.
Any drift in master vs the saved feature parquets raises immediately
during X construction.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wta_clay.features.io import load_features

log = logging.getLogger(__name__)


# ===========================================================================
# Constants
# ===========================================================================

#: Tier categories considered for one-hot. ITF appears in the master
#: definition but typically has 0 rows post-filter (since ITF rows are
#: ``not_for_betting=True`` and backtest filters them — but training
#: keeps them). We dynamically pick the reference category at runtime.
TOUR_LEVELS: tuple[str, ...] = ("GS", "WTA1000", "WTA500",
                                "WTA250", "WTA125", "ITF")

#: Module-level feature columns (after dropping count metadata per locked rule,
#: with the exception of h2h_n_clay which provides sample-size disambiguation).
_MODULE_FEATURE_COLS: tuple[str, ...] = (
    # clay_elo
    "clay_elo_p1", "clay_elo_p2", "clay_elo_diff",
    # form
    "form_winrate_p1", "form_winrate_p2", "form_winrate_diff",
    "form_dominance_p1", "form_dominance_p2", "form_dominance_diff",
    # h2h (kept n_clay for sample-size disambiguation)
    "h2h_smoothed_ratio_p1", "h2h_n_clay",
    # specialist
    "specialist_p1", "specialist_p2", "specialist_diff",
    # ranking
    "rank_p1", "rank_p2", "rank_diff",
    "rank_trend_p1", "rank_trend_p2", "rank_trend_diff",
    # fatigue
    "rest_days_p1", "rest_days_p2", "rest_days_diff",
    "matches_this_week_p1", "matches_this_week_p2", "matches_this_week_diff",
    "clay_season_cum_p1", "clay_season_cum_p2", "clay_season_cum_diff",
)

#: Player-meta columns derived from master.
_PLAYER_META_COLS: tuple[str, ...] = (
    "p1_age", "p2_age",
    "p1_left_handed", "p2_left_handed",
)

#: Required master columns.
REQUIRED_MASTER_COLS: tuple[str, ...] = (
    "match_id", "match_date", "stable_match_key",
    "p1_age", "p2_age", "p1_hand", "p2_hand", "tour_level",
    "y", "sample_weight", "not_for_betting", "market_source",
)

#: Required feature module names (must be present in feature_parquets dict).
REQUIRED_FEATURES: tuple[str, ...] = (
    "clay_elo", "form", "h2h", "specialist", "ranking", "fatigue",
)


# ===========================================================================
# Result dataclass
# ===========================================================================

@dataclass
class SplitResult:
    """Output of :func:`make_splits`.

    4 ordered splits, all inclusive of their year boundary:
        train  →  calib  →  valid  →  test

    The ``calib`` arrays are populated when ``calib_years`` is set
    (default ``(2017, 2018)``); they are the calibration-only holdout
    that never enters training.
    """
    X_train: pd.DataFrame
    y_train: pd.Series
    w_train: pd.Series
    match_ids_train: pd.Series

    X_calib: pd.DataFrame
    y_calib: pd.Series
    w_calib: pd.Series
    match_ids_calib: pd.Series

    X_valid: pd.DataFrame
    y_valid: pd.Series
    w_valid: pd.Series
    match_ids_valid: pd.Series

    X_test: pd.DataFrame
    y_test: pd.Series
    w_test: pd.Series
    match_ids_test: pd.Series

    feature_names: list[str]
    config: dict[str, Any]
    sanity_log: list[str] = field(default_factory=list)

    drift_report: pd.DataFrame = field(default_factory=pd.DataFrame)
    drift_by_year: pd.DataFrame = field(default_factory=pd.DataFrame)
    descriptive_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    nan_rates: pd.DataFrame = field(default_factory=pd.DataFrame)
    tour_level_reference: str = ""
    tour_level_counts: pd.DataFrame = field(default_factory=pd.DataFrame)


# ===========================================================================
# Main entry
# ===========================================================================

def make_splits(
    master: pd.DataFrame,
    feature_parquets: dict[str, Path | str],
    *,
    train_years: tuple[int, int] = (2009, 2016),
    calib_years: tuple[int, int] = (2017, 2018),
    valid_years: tuple[int, int] = (2019, 2021),
    test_years:  tuple[int, int] = (2022, 2025),
    drop_clay_bootstrap: bool = True,
    drop_market_none: bool = True,
    time_decay_lambda: float = 0.0,
    time_decay_anchor_year: int = 2016,
    drift_warn_sigma: float = 1.0,
    run_sanity_checks: bool = True,
) -> SplitResult:
    """Build the 4-set modeling splits (train / calib / valid / test).

    Parameters
    ----------
    master : DataFrame
        From data/processed/clay_matches_master.parquet.
    feature_parquets : dict
        Map ``feature_module_name -> path``. Required keys:
        ``clay_elo, form, h2h, specialist, ranking, fatigue``.
        Each parquet is loaded with ``master_key_hash`` validation.
    train_years, calib_years, valid_years, test_years : tuple[int, int]
        Inclusive year ranges. Defaults are the 2026-05-03 lock:
        train (2009-2016) — 8 years, calib (2017-2018) — 2 years,
        valid (2019-2021) — 3 years, test (2022-2025) — 3-4 years.
        The four ranges must be **strictly non-overlapping** and in
        chronological order; the sanity checks enforce this.
    """
    config = {
        "train_years": list(train_years),
        "calib_years": list(calib_years),
        "valid_years": list(valid_years),
        "test_years":  list(test_years),
        "drop_clay_bootstrap": drop_clay_bootstrap,
        "drop_market_none": drop_market_none,
        "time_decay_lambda": time_decay_lambda,
        "time_decay_anchor_year": time_decay_anchor_year,
        "drift_warn_sigma": drift_warn_sigma,
        "n_master_rows": int(len(master)),
    }
    log.info("make_splits: master=%d  train=%s  calib=%s  valid=%s  test=%s",
             len(master), train_years, calib_years, valid_years, test_years)

    _validate_inputs(master, feature_parquets)
    _validate_year_ranges(train_years, calib_years, valid_years, test_years)

    # 1. Load all features via load_features (master_key_hash validates)
    feats = {}
    for name in REQUIRED_FEATURES:
        feats[name] = load_features(Path(feature_parquets[name]), master)
    log.info("Loaded %d feature parquets, all hashes match master", len(feats))

    # 2. Inner-join master + all features
    X = master.copy()
    for name, f in feats.items():
        X = X.merge(f, on="match_id", how="inner", validate="1:1")

    # 3. Apply row-level filters
    n_before = len(X)
    keep = pd.Series(True, index=X.index)
    if drop_clay_bootstrap and "clay_elo_bootstrap" in X.columns:
        keep &= ~X["clay_elo_bootstrap"]
    if drop_market_none:
        keep &= X["market_source"].astype(str) != "none"
    X = X.loc[keep].reset_index(drop=True)
    log.info("Filter: %d → %d (-%d rows)", n_before, len(X), n_before - len(X))

    # 4. Derive player-meta + tour-level columns
    X = _add_derived_features(X)

    # 5. Determine tour_level reference + final feature column list
    tour_level_counts, tour_level_reference, feature_names = \
        _resolve_tour_level_reference(X)

    # 6. Compute weights = sample_weight × time_decay
    X["_match_year"] = X["match_date"].dt.year
    decay = np.exp(-time_decay_lambda *
                    np.maximum(0, time_decay_anchor_year - X["_match_year"]))
    X["_weight"] = X["sample_weight"] * decay

    # 7. Time split — 4 sets
    yr = X["_match_year"]
    train_mask = yr.between(*train_years)
    calib_mask = yr.between(*calib_years)
    valid_mask = yr.between(*valid_years)
    test_mask  = yr.between(*test_years)

    def _slice(mask):
        sub = X.loc[mask].reset_index(drop=True)
        return (
            sub[feature_names].copy(),
            sub["y"].astype("Int8").reset_index(drop=True),
            sub["_weight"].astype("float64").reset_index(drop=True),
            sub["match_id"].astype("Int64").reset_index(drop=True),
        )

    X_tr, y_tr, w_tr, ids_tr = _slice(train_mask)
    X_ca, y_ca, w_ca, ids_ca = _slice(calib_mask)
    X_va, y_va, w_va, ids_va = _slice(valid_mask)
    X_te, y_te, w_te, ids_te = _slice(test_mask)

    log.info("Split sizes: train=%d  calib=%d  valid=%d  test=%d",
             len(X_tr), len(X_ca), len(X_va), len(X_te))

    # 8. Diagnostics — drift / desc-stats now span all 4 sets
    nan_rates = _compute_nan_rates_4(X_tr, X_ca, X_va, X_te)
    drift_report = _compute_drift_report(X_tr, X_va, X_te, drift_warn_sigma)
    descriptive_stats = _compute_descriptive_stats_4(X_tr, X_ca, X_va, X_te)
    drift_by_year = _compute_drift_by_year(X, feature_names)

    sanity_log: list[str] = []
    if run_sanity_checks:
        _run_sanity_checks(
            master, X,
            train_mask, calib_mask, valid_mask, test_mask,
            X_tr, X_ca, X_va, X_te,
            y_tr, y_ca, y_va, y_te,
            ids_tr, ids_ca, ids_va, ids_te,
            feature_names, drift_report, drift_warn_sigma,
            tour_level_reference, sanity_log,
        )

    config["feature_names"] = feature_names
    config["n_features"] = len(feature_names)
    config["tour_level_reference"] = tour_level_reference

    return SplitResult(
        X_train=X_tr, y_train=y_tr, w_train=w_tr, match_ids_train=ids_tr,
        X_calib=X_ca, y_calib=y_ca, w_calib=w_ca, match_ids_calib=ids_ca,
        X_valid=X_va, y_valid=y_va, w_valid=w_va, match_ids_valid=ids_va,
        X_test=X_te, y_test=y_te, w_test=w_te, match_ids_test=ids_te,
        feature_names=feature_names,
        config=config,
        sanity_log=sanity_log,
        drift_report=drift_report,
        drift_by_year=drift_by_year,
        descriptive_stats=descriptive_stats,
        nan_rates=nan_rates,
        tour_level_reference=tour_level_reference,
        tour_level_counts=tour_level_counts,
    )


def _validate_year_ranges(
    train_years: tuple[int, int],
    calib_years: tuple[int, int],
    valid_years: tuple[int, int],
    test_years:  tuple[int, int],
) -> None:
    """Enforce strict chronological ordering: train < calib < valid < test.

    Each range is inclusive on both ends; the upper end of one range must
    be strictly less than the lower end of the next.
    """
    ranges = [
        ("train", train_years), ("calib", calib_years),
        ("valid", valid_years), ("test",  test_years),
    ]
    for name, rng in ranges:
        if rng[0] > rng[1]:
            raise ValueError(
                f"{name}_years invalid: {rng[0]} > {rng[1]}"
            )
    for (n1, r1), (n2, r2) in zip(ranges, ranges[1:]):
        if r1[1] >= r2[0]:
            raise ValueError(
                f"{n1}_years and {n2}_years overlap: "
                f"{n1}.max={r1[1]} >= {n2}.min={r2[0]}"
            )


# ===========================================================================
# Schema validation
# ===========================================================================

def _validate_inputs(
    master: pd.DataFrame, feature_parquets: dict[str, Path | str],
) -> None:
    miss_m = set(REQUIRED_MASTER_COLS) - set(master.columns)
    if miss_m:
        raise ValueError(
            f"master missing required columns: {sorted(miss_m)}"
        )
    miss_f = set(REQUIRED_FEATURES) - set(feature_parquets.keys())
    if miss_f:
        raise ValueError(
            f"feature_parquets missing required modules: {sorted(miss_f)}. "
            f"Required keys: {list(REQUIRED_FEATURES)}"
        )


# ===========================================================================
# Derived features
# ===========================================================================

def _add_derived_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add player meta + tour_level one-hot binary cols.

    p1_left_handed = 1 if p1_hand == 'L' else 0
    p2_left_handed = 1 if p2_hand == 'L' else 0
    tour_level_<TIER> = 1 if tour_level == TIER else 0  (all 6 tiers)
    """
    X = X.copy()
    X["p1_left_handed"] = (X["p1_hand"].astype("string") == "L").astype("int8")
    X["p2_left_handed"] = (X["p2_hand"].astype("string") == "L").astype("int8")
    # p1_age, p2_age are already on master
    for tier in TOUR_LEVELS:
        X[f"tour_level_{tier}"] = (
            X["tour_level"].astype("string") == tier
        ).astype("int8")
    return X


def _resolve_tour_level_reference(
    X: pd.DataFrame,
) -> tuple[pd.DataFrame, str, list[str]]:
    """Pick the tour_level reference category to drop.

    Rule:
      - If ITF count > 0 in post-filter X: drop ITF (standard convention)
      - Else: drop the most-populous tier (avoids retaining a useless
        all-zero column as the reference) — typically WTA250.

    Returns (counts_df, reference_tier, feature_names_list).
    """
    counts = X["tour_level"].astype("string").value_counts(dropna=False)
    counts_df = counts.reset_index()
    counts_df.columns = ["tour_level", "n"]

    if counts.get("ITF", 0) > 0:
        ref = "ITF"
    else:
        ref = counts.idxmax()    # most populous

    log.info("tour_level reference = %r  (ITF count = %d)",
             ref, int(counts.get("ITF", 0)))

    feature_names: list[str] = list(_PLAYER_META_COLS)
    for tier in TOUR_LEVELS:
        if tier == ref:
            continue
        feature_names.append(f"tour_level_{tier}")
    feature_names.extend(_MODULE_FEATURE_COLS)
    return counts_df, ref, feature_names


# ===========================================================================
# Diagnostics
# ===========================================================================

def _compute_nan_rates(
    X_tr: pd.DataFrame, X_va: pd.DataFrame, X_te: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for col in X_tr.columns:
        rows.append({
            "column": col,
            "train_nan_%": round(X_tr[col].isna().mean() * 100, 2),
            "valid_nan_%": round(X_va[col].isna().mean() * 100, 2),
            "test_nan_%":  round(X_te[col].isna().mean() * 100, 2),
        })
    return pd.DataFrame(rows)


def _compute_nan_rates_4(
    X_tr: pd.DataFrame, X_ca: pd.DataFrame,
    X_va: pd.DataFrame, X_te: pd.DataFrame,
) -> pd.DataFrame:
    """4-set version including calibration holdout."""
    rows = []
    for col in X_tr.columns:
        rows.append({
            "column": col,
            "train_nan_%": round(X_tr[col].isna().mean() * 100, 2),
            "calib_nan_%": round(X_ca[col].isna().mean() * 100, 2),
            "valid_nan_%": round(X_va[col].isna().mean() * 100, 2),
            "test_nan_%":  round(X_te[col].isna().mean() * 100, 2),
        })
    return pd.DataFrame(rows)


def _compute_drift_report(
    X_tr: pd.DataFrame, X_va: pd.DataFrame, X_te: pd.DataFrame,
    sigma: float,
) -> pd.DataFrame:
    """Per-column train-mean / train-std / valid-mean / test-mean.

    drift_sigma = |train_mean - test_mean| / train_std
    flag = ⚠ if drift_sigma > sigma else ✓
    """
    rows = []
    for col in X_tr.columns:
        tr = pd.to_numeric(X_tr[col], errors="coerce")
        va = pd.to_numeric(X_va[col], errors="coerce")
        te = pd.to_numeric(X_te[col], errors="coerce")
        tr_mean = float(tr.mean())
        tr_std = float(tr.std())
        va_mean = float(va.mean())
        te_mean = float(te.mean())
        drift = (
            abs(tr_mean - te_mean) / tr_std
            if tr_std > 0 and not np.isnan(tr_mean) and not np.isnan(te_mean)
            else float("nan")
        )
        flag = "⚠" if (not np.isnan(drift) and drift > sigma) else "✓"
        rows.append({
            "feature": col,
            "train_mean": round(tr_mean, 4),
            "train_std":  round(tr_std, 4),
            "valid_mean": round(va_mean, 4),
            "test_mean":  round(te_mean, 4),
            "drift_sigma": round(drift, 3) if not np.isnan(drift) else None,
            "flag": flag,
        })
    return pd.DataFrame(rows)


def _compute_descriptive_stats(
    X_tr: pd.DataFrame, X_va: pd.DataFrame, X_te: pd.DataFrame,
) -> pd.DataFrame:
    """Per (feature, split): n, mean, std, min, p10, p90, max."""
    rows = []
    for split_name, X in (("train", X_tr), ("valid", X_va), ("test", X_te)):
        for col in X.columns:
            s = pd.to_numeric(X[col], errors="coerce").dropna()
            if s.empty:
                continue
            rows.append({
                "feature": col, "split": split_name,
                "n": int(s.notna().sum()),
                "mean": round(float(s.mean()), 4),
                "std":  round(float(s.std()), 4),
                "min":  round(float(s.min()), 4),
                "p10":  round(float(s.quantile(0.10)), 4),
                "p90":  round(float(s.quantile(0.90)), 4),
                "max":  round(float(s.max()), 4),
            })
    return pd.DataFrame(rows)


def _compute_descriptive_stats_4(
    X_tr: pd.DataFrame, X_ca: pd.DataFrame,
    X_va: pd.DataFrame, X_te: pd.DataFrame,
) -> pd.DataFrame:
    """4-set version: per (feature, split): n, mean, std, min, p10, p90, max."""
    rows = []
    for split_name, X in (("train", X_tr), ("calib", X_ca),
                           ("valid", X_va), ("test", X_te)):
        for col in X.columns:
            s = pd.to_numeric(X[col], errors="coerce").dropna()
            if s.empty:
                continue
            rows.append({
                "feature": col, "split": split_name,
                "n": int(s.notna().sum()),
                "mean": round(float(s.mean()), 4),
                "std":  round(float(s.std()), 4),
                "min":  round(float(s.min()), 4),
                "p10":  round(float(s.quantile(0.10)), 4),
                "p90":  round(float(s.quantile(0.90)), 4),
                "max":  round(float(s.max()), 4),
            })
    return pd.DataFrame(rows)


def _compute_drift_by_year(
    X: pd.DataFrame, feature_names: list[str],
) -> pd.DataFrame:
    """Per-year mean of each feature (rows = years, cols = features).

    Diagnostic for spotting year-specific anomalies (e.g. COVID 2020).
    """
    if X.empty or "_match_year" not in X.columns:
        return pd.DataFrame()
    rows = []
    for yr, g in X.groupby("_match_year"):
        row = {"year": int(yr), "n": len(g)}
        for col in feature_names:
            if col not in g.columns:
                continue
            v = pd.to_numeric(g[col], errors="coerce").mean()
            row[col] = round(float(v), 4) if pd.notna(v) else None
        rows.append(row)
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


# ===========================================================================
# Self-checks
# ===========================================================================

def _run_sanity_checks(
    master: pd.DataFrame, X_full: pd.DataFrame,
    train_mask: pd.Series, calib_mask: pd.Series,
    valid_mask: pd.Series, test_mask: pd.Series,
    X_tr, X_ca, X_va, X_te,
    y_tr, y_ca, y_va, y_te,
    ids_tr, ids_ca, ids_va, ids_te,
    feature_names: list[str],
    drift_report: pd.DataFrame, drift_warn_sigma: float,
    tour_level_reference: str,
    log_list: list[str],
) -> None:
    n_tr, n_ca, n_va, n_te = len(X_tr), len(X_ca), len(X_va), len(X_te)

    # 1. Time non-overlap across all 4 sets
    tr_max = X_full.loc[train_mask, "match_date"].max()
    ca_min = X_full.loc[calib_mask, "match_date"].min()
    ca_max = X_full.loc[calib_mask, "match_date"].max()
    va_min = X_full.loc[valid_mask, "match_date"].min()
    va_max = X_full.loc[valid_mask, "match_date"].max()
    te_min = X_full.loc[test_mask,  "match_date"].min()
    boundaries = [
        ("train", tr_max, "calib", ca_min),
        ("calib", ca_max, "valid", va_min),
        ("valid", va_max, "test",  te_min),
    ]
    for left, lmax, right, rmin in boundaries:
        if not (pd.notna(lmax) and pd.notna(rmin) and lmax < rmin):
            raise AssertionError(
                f"Time overlap {left}/{right}: "
                f"{left}.max={lmax} >= {right}.min={rmin}"
            )
    log_list.append(
        f"[✓] Time non-overlap  train.max={tr_max.date()} < "
        f"calib.min={ca_min.date()} < calib.max={ca_max.date()} < "
        f"valid.min={va_min.date()} < valid.max={va_max.date()} < "
        f"test.min={te_min.date()}"
    )

    # 2. match_id non-overlap across all 4 sets (defensive)
    sets = {
        "train": set(ids_tr.dropna().astype(int).tolist()),
        "calib": set(ids_ca.dropna().astype(int).tolist()),
        "valid": set(ids_va.dropna().astype(int).tolist()),
        "test":  set(ids_te.dropna().astype(int).tolist()),
    }
    pairs = [("train", "calib"), ("train", "valid"), ("train", "test"),
             ("calib", "valid"), ("calib", "test"),
             ("valid", "test")]
    for a, b in pairs:
        if sets[a] & sets[b]:
            raise AssertionError(
                f"match_id overlap {a}×{b}: {len(sets[a] & sets[b])} ids"
            )
    log_list.append(f"[✓] match_id non-overlap across all 4 splits "
                  f"(train×calib×valid×test)")

    # 3. stable_match_key non-overlap (double protection)
    sk = {
        "train": set(X_full.loc[train_mask, "stable_match_key"].astype(str).tolist()),
        "calib": set(X_full.loc[calib_mask, "stable_match_key"].astype(str).tolist()),
        "valid": set(X_full.loc[valid_mask, "stable_match_key"].astype(str).tolist()),
        "test":  set(X_full.loc[test_mask,  "stable_match_key"].astype(str).tolist()),
    }
    for a, b in pairs:
        if sk[a] & sk[b]:
            raise AssertionError(
                f"stable_match_key overlap {a}×{b}: {len(sk[a] & sk[b])} keys"
            )
    log_list.append(f"[✓] stable_match_key non-overlap across all 4 splits")

    # 4. stable_match_key COMPLETENESS (no rows lost in splitting)
    full_set = set(X_full["stable_match_key"].astype(str).tolist())
    union = sk["train"] | sk["calib"] | sk["valid"] | sk["test"]
    missing = full_set - union
    extra = union - full_set
    if missing or extra:
        raise AssertionError(
            f"stable_match_key completeness: post-filter X has "
            f"{len(full_set)} keys; train+calib+valid+test union has "
            f"{len(union)}. "
            f"Missing from splits: {len(missing)}, "
            f"Extra in splits: {len(extra)}. "
            f"Year-boundary or filter logic bug."
        )
    log_list.append(f"[✓] stable_match_key completeness: "
                  f"|full|={len(full_set)} = "
                  f"|train ∪ calib ∪ valid ∪ test|")

    # 5. All feature cols present in all 4 splits
    for split_name, X in (("train", X_tr), ("calib", X_ca),
                           ("valid", X_va), ("test", X_te)):
        miss = set(feature_names) - set(X.columns)
        if miss:
            raise AssertionError(
                f"{split_name} missing feature cols: {sorted(miss)}"
            )
    log_list.append(f"[✓] All {len(feature_names)} feature cols present "
                  f"in all 4 splits")

    # 6. y class balance per split (warn if outside [0.45, 0.55])
    for name, y in (("train", y_tr), ("calib", y_ca),
                     ("valid", y_va), ("test", y_te)):
        rate = float(y.mean()) if len(y) else float("nan")
        if 0.45 <= rate <= 0.55:
            log_list.append(f"[✓] {name} y=1 rate = {rate:.4f}  ⊆ [0.45, 0.55]")
        else:
            log_list.append(f"[⚠] {name} y=1 rate = {rate:.4f}  outside [0.45, 0.55]")

    # 7. Drift report — list ⚠-flagged columns
    if not drift_report.empty:
        warned = drift_report[drift_report["flag"] == "⚠"]
        if warned.empty:
            log_list.append(f"[✓] Drift train→test: 0 cols exceed "
                          f"{drift_warn_sigma:.1f}σ")
        else:
            log_list.append(f"[⚠] Drift train→test: {len(warned)} cols "
                          f"exceed {drift_warn_sigma:.1f}σ — see drift_report")
            for _, r in warned.iterrows():
                log_list.append(f"     {r['feature']}: "
                              f"train_mean={r['train_mean']} → "
                              f"test_mean={r['test_mean']} "
                              f"(drift={r['drift_sigma']}σ)")

    # 8. tour_level reference reporting
    log_list.append(f"[✓] tour_level reference category = "
                  f"{tour_level_reference!r}  "
                  f"(dropped from feature matrix)")
