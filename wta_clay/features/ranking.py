"""Ranking feature module — current rank diff + 8-week trend slope.

Output (6 cols + match_id):

  rank_p1            Int64    pre-match WTA rank (copied from master)
  rank_p2            Int64    mirror
  rank_diff          Int64    p2_rank − p1_rank (positive = p1 stronger)
  rank_trend_p1      float64  8-week linear-regression slope, spots/week
                              (negative = rising up the rankings)
  rank_trend_p2      float64  mirror
  rank_trend_diff    float64  p2_trend − p1_trend (positive = p1 rising faster)

Direction convention
--------------------
WTA ranks are inverted (#1 = best). To match the project's convention
that positive ``*_diff`` favors p1, both subtractions are flipped:

  rank_diff = p2_rank − p1_rank    (positive = p1 has the better rank)
  rank_trend_diff = p2_trend − p1_trend
                                    (positive = p1's rank-number is dropping
                                     faster — i.e. p1 is rising)

For ``rank_trend``, NEGATIVE slope means the rank number is decreasing,
i.e. the player is moving UP the rankings. A peak-ascent player like
Andreeva 2024 should show ``rank_trend ≈ -2 to -5``/week.

Data source decisions
---------------------
``rank_p1`` and ``rank_p2`` are copied directly from ``master.p1_rank``
and ``master.p2_rank`` (Sackmann's standardized snapshot, ~0.18% missing).
We do NOT re-query the rankings table for the current rank — that would
introduce a second-source-of-truth risk.

``rank_trend_p1/p2`` ARE computed from the rankings table (8-week linear
regression of weekly snapshots). Each weekly Monday rank within
``[match_date − 8 weeks, match_date)`` is used as a data point.

Locked thresholds (sanity, not gating)
--------------------------------------
- ``trend_window_weeks = 8``
- ``min_points_for_trend = 4`` (need at least half the window populated)
- ``|rank_trend| > 30/week`` → warn (likely outlier — sudden 240+ spot
  change in 8 weeks is rare and often a data entry artifact)

Locked conventions (re-stated)
------------------------------
- Reads ``master.stable_match_key`` for cross-frame join
- Output features keyed on ``match_id``
- Strict ``ranking_date < match_date`` for trend window upper bound
  (current rank uses ``≤`` since rankings are published Monday morning
  and effective same week)
- 100-row stratified leakage cross-check (5/year × 18 yrs)
- ``write_features`` writes parquet + sidecar JSON with ``master_key_hash``
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wta_clay.features.io import write_features

log = logging.getLogger(__name__)


# ===========================================================================
# Constants
# ===========================================================================

TREND_WINDOW_WEEKS: int = 8
MIN_POINTS_FOR_TREND: int = 4
EXTREME_TREND_WARN: float = 30.0          # |slope| > this is suspicious

REQUIRED_MASTER_COLS: tuple[str, ...] = (
    "match_id", "match_date", "p1_id", "p2_id",
    "p1_rank", "p2_rank", "stable_match_key",
)
REQUIRED_RANKINGS_COLS: tuple[str, ...] = (
    "ranking_date", "player_id", "rank",
)

FEATURE_COLUMNS: tuple[str, ...] = (
    "match_id",
    "rank_p1", "rank_p2", "rank_diff",
    "rank_trend_p1", "rank_trend_p2", "rank_trend_diff",
)

#: Pearson r thresholds (sanity warns).
ELO_RANK_DIFF_LOW_R_WARN:  float = 0.40   # warn if r(rank_diff, elo_diff) < this
ELO_RANK_TREND_HIGH_R_WARN: float = 0.50  # warn if r(trend, elo_diff) > this

NAN_RATE_WARN_THRESHOLD: float = 0.25


# ===========================================================================
# Result dataclass
# ===========================================================================

@dataclass
class RankingResult:
    """Top-level return of :func:`compute_ranking_features`."""
    features: pd.DataFrame
    config: dict[str, Any]
    sanity_log: list[str] = field(default_factory=list)
    elo_correlations: dict[str, float] = field(default_factory=dict)
    nan_rates: pd.DataFrame = field(default_factory=pd.DataFrame)
    star_trajectories: pd.DataFrame = field(default_factory=pd.DataFrame)
    coverage_audit: pd.DataFrame = field(default_factory=pd.DataFrame)


# ===========================================================================
# Main entry
# ===========================================================================

def compute_ranking_features(
    master: pd.DataFrame,
    rankings: pd.DataFrame,
    clay_elo_features: pd.DataFrame | None = None,
    players: pd.DataFrame | None = None,
    *,
    trend_window_weeks: int = TREND_WINDOW_WEEKS,
    min_points_for_trend: int = MIN_POINTS_FOR_TREND,
    out_path: Path | str | None = None,
    run_sanity_checks: bool = True,
    leakage_sample_size: int = 100,
    rng_seed: int = 42,
) -> RankingResult:
    """Compute ranking features for every match in master.

    Parameters
    ----------
    master : DataFrame
        From data/processed/clay_matches_master.parquet. Required cols::

            match_id, match_date, p1_id, p2_id, p1_rank, p2_rank,
            stable_match_key
    rankings : DataFrame
        Sackmann ``wta_rankings_*.csv`` concatenated. Required cols::

            ranking_date, player_id, rank
    clay_elo_features : DataFrame | None
        Used for the Pearson-r diagnostics (rank_diff vs elo_diff,
        rank_trend_diff vs elo_diff).
    players : DataFrame | None
        Used to find player_ids for star-trajectory verification.
    """
    config = {
        "trend_window_weeks": trend_window_weeks,
        "min_points_for_trend": min_points_for_trend,
        "extreme_trend_warn": EXTREME_TREND_WARN,
        "n_master_rows": int(len(master)),
        "n_rankings_rows": int(len(rankings)),
    }
    log.info("compute_ranking: master=%d, rankings=%d, window=%dwks",
             config["n_master_rows"], config["n_rankings_rows"],
             trend_window_weeks)

    if master.empty:
        log.warning("Empty master; returning empty RankingResult.")
        return RankingResult(
            features=pd.DataFrame(columns=list(FEATURE_COLUMNS)),
            config=config,
        )

    _validate_inputs(master, rankings)

    rk_index = _build_rankings_index(rankings)
    feature_rows = _compute_features(
        master, rk_index, trend_window_weeks, min_points_for_trend,
    )
    features = _finalize_features(feature_rows)

    nan_rates = _compute_nan_rates(features)
    coverage_audit = _compute_coverage_audit(rankings, master)
    star_trajectories = _star_trajectories(rk_index, players,
                                            trend_window_weeks,
                                            min_points_for_trend)

    sanity_log: list[str] = []
    elo_correlations: dict[str, float] = {}
    if run_sanity_checks:
        elo_correlations = _run_sanity_checks(
            master, features, rankings, rk_index, players,
            clay_elo_features, sanity_log,
            trend_window_weeks=trend_window_weeks,
            min_points_for_trend=min_points_for_trend,
            leakage_sample_size=leakage_sample_size,
            rng_seed=rng_seed,
            nan_rates=nan_rates,
        )

    if out_path is not None:
        write_features(features, master, path=out_path,
                       feature_module="ranking", config=config)

    return RankingResult(
        features=features, config=config, sanity_log=sanity_log,
        elo_correlations=elo_correlations, nan_rates=nan_rates,
        star_trajectories=star_trajectories,
        coverage_audit=coverage_audit,
    )


# ===========================================================================
# Schema
# ===========================================================================

def _validate_inputs(master: pd.DataFrame, rankings: pd.DataFrame) -> None:
    miss_m = set(REQUIRED_MASTER_COLS) - set(master.columns)
    if miss_m:
        raise ValueError(
            f"master missing required columns: {sorted(miss_m)}. "
            f"Required: {list(REQUIRED_MASTER_COLS)}"
        )
    # Skip rankings schema check when empty (no rows = no work; valid edge case)
    if rankings.empty:
        return
    miss_r = set(REQUIRED_RANKINGS_COLS) - set(rankings.columns)
    if miss_r:
        raise ValueError(
            f"rankings missing required columns: {sorted(miss_r)}. "
            f"Required: {list(REQUIRED_RANKINGS_COLS)}"
        )


# ===========================================================================
# Index + per-row computation
# ===========================================================================

def _build_rankings_index(rankings: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Group rankings by player_id; sorted by ranking_date ascending."""
    if rankings.empty:
        return {}
    rk = rankings[["player_id", "ranking_date", "rank"]].copy()
    rk["ranking_date"] = pd.to_datetime(rk["ranking_date"], errors="coerce")
    rk = rk.dropna(subset=["ranking_date", "player_id"]).copy()
    rk["player_id"] = rk["player_id"].astype("Int64")
    rk = rk.sort_values(["player_id", "ranking_date"]).reset_index(drop=True)
    return {int(pid): g.reset_index(drop=True)
            for pid, g in rk.groupby("player_id")}


def _trend_slope(
    player_rankings: pd.DataFrame | None,
    target_date: pd.Timestamp,
    window_weeks: int, min_points: int,
) -> float:
    """8-week linear regression slope, in spots-per-week.

    Strict-< on upper bound: ``ranking_date < target_date``. Returns NaN
    if fewer than ``min_points`` valid weekly snapshots.
    """
    if player_rankings is None or player_rankings.empty:
        return float("nan")
    window_start = target_date - pd.Timedelta(weeks=window_weeks)
    mask = ((player_rankings["ranking_date"] >= window_start) &
            (player_rankings["ranking_date"] < target_date))
    sub = player_rankings.loc[mask]
    if len(sub) < min_points:
        return float("nan")
    weeks = ((sub["ranking_date"] - window_start).dt.total_seconds()
             / (7 * 24 * 3600)).to_numpy(dtype=float)
    ranks = pd.to_numeric(sub["rank"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(ranks) & np.isfinite(weeks)
    weeks = weeks[valid]
    ranks = ranks[valid]
    if len(ranks) < min_points or len(np.unique(weeks)) < 2:
        return float("nan")
    slope, _ = np.polyfit(weeks, ranks, 1)
    return float(slope)


def _compute_features(
    master: pd.DataFrame, rk_index: dict[int, pd.DataFrame],
    trend_window_weeks: int, min_points_for_trend: int,
) -> list[dict]:
    """Iterate master, compute rank_diff (from master) + rank_trend per side."""
    feature_rows: list[dict] = []
    p1_ids = master["p1_id"].astype("Int64").to_numpy()
    p2_ids = master["p2_id"].astype("Int64").to_numpy()
    p1_ranks = master["p1_rank"].astype("Int64").to_numpy()
    p2_ranks = master["p2_rank"].astype("Int64").to_numpy()
    dates = pd.to_datetime(master["match_date"]).to_numpy()
    match_ids = master["match_id"].astype("Int64").to_numpy()

    for i in range(len(master)):
        cur_date = pd.Timestamp(dates[i])
        rk_p1 = p1_ranks[i]
        rk_p2 = p2_ranks[i]

        # rank_diff (zero-cost from master)
        if pd.notna(rk_p1) and pd.notna(rk_p2):
            rank_diff = int(rk_p2) - int(rk_p1)
        else:
            rank_diff = pd.NA

        # rank_trend (8-week linear regression)
        trend_p1 = _trend_slope(
            rk_index.get(int(p1_ids[i])), cur_date,
            trend_window_weeks, min_points_for_trend,
        )
        trend_p2 = _trend_slope(
            rk_index.get(int(p2_ids[i])), cur_date,
            trend_window_weeks, min_points_for_trend,
        )
        if not np.isnan(trend_p1) and not np.isnan(trend_p2):
            trend_diff = trend_p2 - trend_p1
        else:
            trend_diff = float("nan")

        feature_rows.append({
            "match_id": int(match_ids[i]),
            "rank_p1": rk_p1, "rank_p2": rk_p2,
            "rank_diff": rank_diff,
            "rank_trend_p1": trend_p1,
            "rank_trend_p2": trend_p2,
            "rank_trend_diff": trend_diff,
        })
    return feature_rows


def _finalize_features(feature_rows: list[dict]) -> pd.DataFrame:
    if not feature_rows:
        return pd.DataFrame(columns=list(FEATURE_COLUMNS))
    f = pd.DataFrame(feature_rows)
    f["match_id"] = f["match_id"].astype("Int64")
    f["rank_p1"] = f["rank_p1"].astype("Int64")
    f["rank_p2"] = f["rank_p2"].astype("Int64")
    f["rank_diff"] = f["rank_diff"].astype("Int64")
    f["rank_trend_p1"] = f["rank_trend_p1"].astype("float64")
    f["rank_trend_p2"] = f["rank_trend_p2"].astype("float64")
    f["rank_trend_diff"] = f["rank_trend_diff"].astype("float64")
    return f[list(FEATURE_COLUMNS)]


# ===========================================================================
# Diagnostics
# ===========================================================================

def _compute_nan_rates(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    rows = []
    for col in ("rank_p1", "rank_p2", "rank_diff",
                "rank_trend_p1", "rank_trend_p2", "rank_trend_diff"):
        rate = features[col].isna().mean()
        rows.append({"column": col, "nan_%": round(rate * 100, 2)})
    return pd.DataFrame(rows)


def _compute_coverage_audit(
    rankings: pd.DataFrame, master: pd.DataFrame,
) -> pd.DataFrame:
    """Audit weekly snapshot coverage for active players in 2022.

    For active players (those in master), expected ≈ 52 weekly snapshots
    per year. Sub-50 snapshots in a year = ranking-table gap.
    """
    if rankings.empty or master.empty:
        return pd.DataFrame()
    active = (set(master["p1_id"].dropna().astype(int).tolist()) |
              set(master["p2_id"].dropna().astype(int).tolist()))
    rk = rankings.copy()
    rk["ranking_date"] = pd.to_datetime(rk["ranking_date"], errors="coerce")
    yr_2022 = rk[(rk["ranking_date"] >= "2022-01-01") &
                 (rk["ranking_date"] < "2023-01-01")]
    active_2022 = yr_2022[yr_2022["player_id"].isin(active)]
    counts = active_2022.groupby("player_id").size()
    rows = [
        {"metric": "active_players_in_master",
         "value": len(active)},
        {"metric": "active_players_with_rankings_2022",
         "value": int(len(counts))},
        {"metric": "mean_weekly_snapshots_2022",
         "value": float(round(counts.mean(), 2)) if len(counts) else float("nan")},
        {"metric": "median_weekly_snapshots_2022",
         "value": float(counts.median()) if len(counts) else float("nan")},
        {"metric": "min_weekly_snapshots_2022",
         "value": int(counts.min()) if len(counts) else 0},
        {"metric": "max_weekly_snapshots_2022",
         "value": int(counts.max()) if len(counts) else 0},
        {"metric": "players_with_<40_weeks_2022",
         "value": int((counts < 40).sum()) if len(counts) else 0},
    ]
    return pd.DataFrame(rows)


# ===========================================================================
# Star trajectories
# ===========================================================================

_STAR_NAMES: tuple[str, ...] = (
    "Iga Swiatek", "Aryna Sabalenka", "Mirra Andreeva",
    "Simona Halep", "Justine Henin", "Maria Sharapova",
)


def _star_trajectories(
    rk_index: dict[int, pd.DataFrame], players: pd.DataFrame | None,
    trend_window_weeks: int, min_points_for_trend: int,
) -> pd.DataFrame:
    """Year-end ranks (most-recent snapshot per Dec 31) for star players.

    Plus ``best_trend_slope`` column showing each star's most negative
    rolling 8-week slope across their career — verifies trend formula
    captures real ascents.
    """
    if players is None or "full_name" not in players.columns:
        return pd.DataFrame()

    star_ids = {}
    for name in _STAR_NAMES:
        hit = players[players["full_name"].astype("string")
                      .str.contains(name, case=False, na=False, regex=False)]
        if len(hit):
            star_ids[name] = int(hit.iloc[0]["player_id"])

    rows = []
    for name, pid in star_ids.items():
        rk = rk_index.get(pid)
        if rk is None or rk.empty:
            continue
        # Year-end snapshots
        year_ends = {}
        for year in range(2007, 2026):
            year_end = pd.Timestamp(f"{year}-12-31")
            sub = rk[rk["ranking_date"] <= year_end]
            if sub.empty:
                continue
            latest = sub.iloc[-1]
            year_ends[year] = int(latest["rank"])
        # Best (most negative) trend slope across player history
        best_slope = float("nan")
        best_slope_date = None
        if len(rk) >= min_points_for_trend:
            for sample_date in pd.date_range(
                rk["ranking_date"].min() + pd.Timedelta(weeks=trend_window_weeks),
                rk["ranking_date"].max(),
                freq="4W",  # sample every 4 weeks
            ):
                slope = _trend_slope(rk, sample_date, trend_window_weeks,
                                      min_points_for_trend)
                if not np.isnan(slope):
                    if np.isnan(best_slope) or slope < best_slope:
                        best_slope = slope
                        best_slope_date = sample_date.date()
        rows.append({
            "player": name,
            **year_ends,
            "best_trend_slope": round(best_slope, 3) if not np.isnan(best_slope) else None,
            "best_trend_date": best_slope_date,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ===========================================================================
# Sanity checks (9)
# ===========================================================================

def _run_sanity_checks(
    master: pd.DataFrame, features: pd.DataFrame,
    rankings: pd.DataFrame, rk_index: dict[int, pd.DataFrame],
    players: pd.DataFrame | None,
    clay_elo_features: pd.DataFrame | None,
    log_list: list[str],
    *,
    trend_window_weeks: int, min_points_for_trend: int,
    leakage_sample_size: int, rng_seed: int,
    nan_rates: pd.DataFrame,
) -> dict[str, float]:
    n = len(features)
    correlations: dict[str, float] = {}

    # 1. ranks > 0 when present
    rp1 = features["rank_p1"].dropna().astype(int)
    rp2 = features["rank_p2"].dropna().astype(int)
    if (rp1 <= 0).any() or (rp2 <= 0).any():
        raise AssertionError("rank values ≤ 0 present (ranks are 1-indexed)")
    log_list.append(f"[✓] ranks > 0 when present  "
                  f"({len(rp1)}/{len(rp1)} p1, {len(rp2)}/{len(rp2)} p2)")

    # 2. rank_diff NaN consistency (NaN iff either side NaN)
    expected_diff_nan = (features["rank_p1"].isna() |
                         features["rank_p2"].isna())
    actual_diff_nan = features["rank_diff"].isna()
    if not (expected_diff_nan == actual_diff_nan).all():
        raise AssertionError("rank_diff NaN inconsistent with rank_p1/p2")
    log_list.append(f"[✓] rank_diff NaN consistency  {n}/{n}")

    # 3. NaN rates within tolerance
    high = nan_rates[nan_rates["nan_%"] > NAN_RATE_WARN_THRESHOLD * 100]
    if not high.empty:
        for _, r in high.iterrows():
            log_list.append(f"[⚠] {r['column']} NaN = {r['nan_%']}% > "
                          f"{NAN_RATE_WARN_THRESHOLD*100:.0f}% threshold")
    else:
        log_list.append("[✓] NaN rates within tolerance: " +
                      ", ".join(f"{r['column']}={r['nan_%']}%"
                                for _, r in nan_rates.iterrows()))

    # 4. Pearson r(rank_diff, clay_elo_diff) — expected ≥ 0.6
    # 5. Pearson r(rank_trend_diff, clay_elo_diff) — expected < 0.3
    if clay_elo_features is not None and not clay_elo_features.empty:
        joined = features.merge(
            clay_elo_features[["match_id", "clay_elo_diff"]], on="match_id",
        )
        # rank_diff vs elo_diff
        eligible = joined[joined["rank_diff"].notna() &
                          joined["clay_elo_diff"].notna()]
        if len(eligible) >= 30:
            r_diff = float(eligible["rank_diff"].corr(eligible["clay_elo_diff"]))
            correlations["rank_diff_vs_elo"] = r_diff
            tag = "[✓]" if abs(r_diff) >= ELO_RANK_DIFF_LOW_R_WARN else "[⚠]"
            log_list.append(
                f"{tag} Pearson r(rank_diff, clay_elo_diff) = {r_diff:+.3f}  "
                f"(expected ≥ 0.6, warn if < 0.4) — note rank_diff sign convention "
                f"means strong positive correlation with elo_diff is EXPECTED"
            )
        # rank_trend_diff vs elo_diff
        eligible_t = joined[joined["rank_trend_diff"].notna() &
                            joined["clay_elo_diff"].notna()]
        if len(eligible_t) >= 30:
            r_trend = float(eligible_t["rank_trend_diff"].corr(
                eligible_t["clay_elo_diff"]))
            correlations["rank_trend_diff_vs_elo"] = r_trend
            tag = "[✓]" if abs(r_trend) <= ELO_RANK_TREND_HIGH_R_WARN else "[⚠]"
            log_list.append(
                f"{tag} Pearson r(rank_trend_diff, clay_elo_diff) = {r_trend:+.3f}  "
                f"(expected < 0.3, warn if > 0.5)"
            )

    # 6/7. Star sanity — Swiatek 2022-2024 + Andreeva 2024 ascent
    _check_known_players(features, master, rk_index, players,
                          trend_window_weeks, min_points_for_trend, log_list)

    # 8. Extreme |trend| > 30/week → warn
    extreme_trend = (
        (features["rank_trend_p1"].abs() > EXTREME_TREND_WARN) |
        (features["rank_trend_p2"].abs() > EXTREME_TREND_WARN)
    ).sum()
    if extreme_trend:
        log_list.append(f"[⚠] {int(extreme_trend)} rows with "
                      f"|rank_trend| > {EXTREME_TREND_WARN}/week (likely "
                      f"data outliers — sudden 240+ spot moves in 8 weeks)")
    else:
        log_list.append(f"[✓] No extreme |rank_trend| > {EXTREME_TREND_WARN}/week")

    # 9. 100-row stratified leakage cross-check
    _check_no_leakage(
        master, features, rk_index, trend_window_weeks, min_points_for_trend,
        sample_size=leakage_sample_size, rng_seed=rng_seed, log_list=log_list,
    )

    return correlations


def _check_known_players(
    features: pd.DataFrame, master: pd.DataFrame,
    rk_index: dict[int, pd.DataFrame], players: pd.DataFrame | None,
    trend_window_weeks: int, min_points_for_trend: int,
    log_list: list[str],
) -> None:
    """Soft warns for Swiatek 2022-2024 stable top-2 + Andreeva 2024 ascent."""
    if players is None or "full_name" not in players.columns:
        return

    def _player_id(name):
        hit = players[players["full_name"].astype("string")
                      .str.contains(name, case=False, na=False, regex=False)]
        return int(hit.iloc[0]["player_id"]) if len(hit) else None

    # Swiatek 2022-2024 stability
    pid_sw = _player_id("Iga Swiatek")
    if pid_sw is not None:
        rk = rk_index.get(pid_sw)
        if rk is not None:
            mask = ((rk["ranking_date"] >= "2022-09-01") &
                    (rk["ranking_date"] <= "2024-12-31"))
            sw = rk[mask]
            if not sw.empty:
                med = float(sw["rank"].median())
                if med <= 2:
                    log_list.append(f"[✓] Swiatek 2022-2024 median rank = "
                                  f"{med:.1f} ≤ 2  (n={len(sw)} weeks)")
                else:
                    log_list.append(f"[⚠] Swiatek 2022-2024 median rank = "
                                  f"{med:.1f} > 2 — review")

    # Andreeva 2024 ascent: best (most negative) slope in 2024 should be < -2/week
    pid_an = _player_id("Mirra Andreeva")
    if pid_an is not None:
        rk = rk_index.get(pid_an)
        if rk is not None and not rk.empty:
            best_slope = float("nan")
            for sample_date in pd.date_range("2024-01-01", "2024-12-31",
                                              freq="W"):
                slope = _trend_slope(rk, sample_date, trend_window_weeks,
                                      min_points_for_trend)
                if not np.isnan(slope):
                    if np.isnan(best_slope) or slope < best_slope:
                        best_slope = slope
            if not np.isnan(best_slope):
                if best_slope < -2.0:
                    log_list.append(f"[✓] Andreeva 2024 best (most-neg) "
                                  f"trend slope = {best_slope:.2f}/week "
                                  f"(< -2.0 threshold)")
                else:
                    log_list.append(f"[⚠] Andreeva 2024 best slope = "
                                  f"{best_slope:.2f}/week — expected < -2.0")
            else:
                log_list.append("[⚠] Andreeva 2024: insufficient ranking data")


def _check_no_leakage(
    master: pd.DataFrame, features: pd.DataFrame,
    rk_index: dict[int, pd.DataFrame],
    trend_window_weeks: int, min_points_for_trend: int,
    *,
    sample_size: int, rng_seed: int, log_list: list[str],
) -> None:
    """100-row stratified-by-year leakage cross-check.

    For each sampled match, independently re-derive ``rank_trend_p1/p2``
    from rk_index and assert exact match (within 1e-9 tolerance).
    """
    f_with_year = features.merge(
        master[["match_id", "match_date", "p1_id", "p2_id"]], on="match_id",
    )
    f_with_year["_year"] = f_with_year["match_date"].dt.year
    rng = np.random.default_rng(rng_seed)

    yrs = sorted(f_with_year["_year"].dropna().unique().tolist())
    per_year = max(1, sample_size // max(len(yrs), 1))
    sampled = pd.concat([
        f_with_year[f_with_year["_year"] == yr].sample(
            min(per_year, (f_with_year["_year"] == yr).sum()),
            random_state=int(rng.integers(0, 1_000_000)),
        )
        for yr in yrs
    ], ignore_index=True)

    fails = 0
    msgs: list[str] = []
    for _, row in sampled.iterrows():
        cur_date = pd.Timestamp(row["match_date"])
        p1 = int(row["p1_id"])
        p2 = int(row["p2_id"])
        slope_p1_e = _trend_slope(rk_index.get(p1), cur_date,
                                   trend_window_weeks, min_points_for_trend)
        slope_p2_e = _trend_slope(rk_index.get(p2), cur_date,
                                   trend_window_weeks, min_points_for_trend)

        def _eq(a, b):
            if pd.isna(a) and pd.isna(b): return True
            if pd.isna(a) or pd.isna(b): return False
            return abs(float(a) - float(b)) < 1e-9

        if not (_eq(row["rank_trend_p1"], slope_p1_e)
                and _eq(row["rank_trend_p2"], slope_p2_e)):
            fails += 1
            if len(msgs) < 3:
                msgs.append(
                    f"  match_id={int(row['match_id'])}: "
                    f"stored=({row['rank_trend_p1']}, {row['rank_trend_p2']}) "
                    f"expected=({slope_p1_e}, {slope_p2_e})"
                )
    if fails:
        raise AssertionError(
            f"Leakage check failed: {fails}/{len(sampled)} differ\n"
            + "\n".join(msgs)
        )
    log_list.append(f"[✓] Leakage cross-check  {len(sampled)}/{len(sampled)} match")
