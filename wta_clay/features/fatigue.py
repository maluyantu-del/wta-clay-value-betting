"""Fatigue feature module — rest days + week load + clay-season cumulative.

Output (9 cols + match_id):

  rest_days_p1            float64    days since p1's last match (any surface)
  rest_days_p2            float64    mirror
  rest_days_diff          float64    p1 − p2 (positive = p1 more rested = advantage)
  matches_this_week_p1    Int64      p1's prior matches in same ISO week as
                                     the current match
  matches_this_week_p2    Int64      mirror
  matches_this_week_diff  Int64      p2 − p1 (positive = p2 more loaded = p1 advantage)
  clay_season_cum_p1      Int64      p1's clay matches since Apr 1 of current
                                     year, count = 0 outside Apr-Oct window
  clay_season_cum_p2      Int64      mirror
  clay_season_cum_diff    Int64      p2 − p1 (positive = p2 more loaded = p1 advantage)

Design rationale
----------------
**rest_days** uses ALL surfaces — physical fatigue doesn't distinguish
hard from clay. A player who played a 3-set hard match on Wednesday is
still tired Saturday on clay.

**matches_this_week** counts STRICTLY-PRIOR matches in the same ISO
calendar week as the current match. A player playing R64 on Tuesday and
R32 on Wednesday: at the Wed match, matches_this_week_p1 = 1.

**clay_season_cum** measures cumulative clay workload IN the current
year's clay window only. Window = [Apr 1, Oct 31] of the current year
(locked). Outside this window (Nov-Mar including South-American clay
swing in Feb-Mar) the value is 0 — those matches don't count toward
"main clay-season fatigue".

Cold-start
----------
- ``rest_days``: NaN if player has no prior match in history
- ``matches_this_week``: 0 (legitimately zero — no matches yet this week)
- ``clay_season_cum``: 0 (no matches yet this season, or out of season window)

Locked conventions
------------------
- Reads ``master.stable_match_key`` for cross-frame join
- Strict ``date < current_date AND stable_match_key != current_key``
- ``write_features`` writes parquet + sidecar JSON with master_key_hash
- 100-row stratified leakage cross-check
"""
from __future__ import annotations

import logging
from collections import defaultdict
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

CLAY_SEASON_START_MONTH: int = 4    # April 1
CLAY_SEASON_END_MONTH:   int = 10   # October 31 (inclusive)

ROUND_ORDER: tuple[str, ...] = ("R128", "R64", "R32", "R16",
                                "QF", "SF", "F", "RR")

REQUIRED_MASTER_COLS: tuple[str, ...] = (
    "match_id", "match_date", "tourney_id", "p1_id", "p2_id", "y",
    "stable_match_key",
)
REQUIRED_SACKMANN_COLS: tuple[str, ...] = (
    "tourney_id", "match_date", "winner_id", "loser_id",
    "surface", "round",
)

FEATURE_COLUMNS: tuple[str, ...] = (
    "match_id",
    "rest_days_p1", "rest_days_p2", "rest_days_diff",
    "matches_this_week_p1", "matches_this_week_p2", "matches_this_week_diff",
    "clay_season_cum_p1", "clay_season_cum_p2", "clay_season_cum_diff",
)

#: Pearson r threshold (warn if any diff feature correlates > this with elo_diff).
ELO_INDEPENDENCE_WARN_THRESHOLD: float = 0.20

#: Sanity bounds for clean data.
REST_DAYS_REASONABLE_MAX: int = 365 * 3       # 3 years between matches → outlier
MATCHES_PER_WEEK_REASONABLE_MAX: int = 7      # impossible to play >7 single matches/week
CLAY_SEASON_CUM_REASONABLE_MAX: int = 50      # a busy clay season is ~30 matches


# ===========================================================================
# Result dataclass
# ===========================================================================

@dataclass
class FatigueResult:
    """Top-level return of :func:`compute_fatigue_features`."""
    features: pd.DataFrame
    config: dict[str, Any]
    sanity_log: list[str] = field(default_factory=list)
    elo_correlations: dict[str, float] = field(default_factory=dict)
    nan_rates: pd.DataFrame = field(default_factory=pd.DataFrame)
    distribution: pd.DataFrame = field(default_factory=pd.DataFrame)
    star_seasons: pd.DataFrame = field(default_factory=pd.DataFrame)


# ===========================================================================
# Main entry
# ===========================================================================

def compute_fatigue_features(
    master: pd.DataFrame,
    sackmann_all: pd.DataFrame,
    clay_elo_features: pd.DataFrame | None = None,
    players: pd.DataFrame | None = None,
    *,
    season_start_month: int = CLAY_SEASON_START_MONTH,
    season_end_month: int = CLAY_SEASON_END_MONTH,
    out_path: Path | str | None = None,
    run_sanity_checks: bool = True,
    leakage_sample_size: int = 100,
    rng_seed: int = 42,
) -> FatigueResult:
    """Compute fatigue features for every master match.

    Parameters
    ----------
    master : DataFrame
        From data/processed/clay_matches_master.parquet. Must contain
        ``stable_match_key``.
    sackmann_all : DataFrame
        Sackmann main-draw matches across ALL surfaces (Hard/Clay/Grass/
        Carpet). Required cols: tourney_id, match_date, winner_id,
        loser_id, surface, round.
    """
    config = {
        "season_start_month": season_start_month,
        "season_end_month": season_end_month,
        "n_master_rows": int(len(master)),
        "n_sackmann_rows": int(len(sackmann_all)),
    }
    log.info("compute_fatigue: master=%d sackmann=%d clay_season=Mo%d-%d",
             config["n_master_rows"], config["n_sackmann_rows"],
             season_start_month, season_end_month)

    if master.empty:
        log.warning("Empty master; returning empty FatigueResult.")
        return FatigueResult(
            features=pd.DataFrame(columns=list(FEATURE_COLUMNS)),
            config=config,
        )

    _validate_inputs(master, sackmann_all)

    unified = _build_unified_iteration_frame(master, sackmann_all)
    feature_rows = _run_fatigue_iteration(
        unified, season_start_month, season_end_month,
    )
    features = _finalize_features(feature_rows)

    nan_rates = _compute_nan_rates(features)
    distribution = _compute_distribution(features)
    star_seasons = _star_season_audit(unified, players, season_start_month,
                                       season_end_month)

    sanity_log: list[str] = []
    elo_correlations: dict[str, float] = {}
    if run_sanity_checks:
        elo_correlations = _run_sanity_checks(
            master, features, unified, clay_elo_features,
            sanity_log,
            season_start_month=season_start_month,
            season_end_month=season_end_month,
            leakage_sample_size=leakage_sample_size,
            rng_seed=rng_seed,
            nan_rates=nan_rates,
        )

    if out_path is not None:
        write_features(features, master, path=out_path,
                       feature_module="fatigue", config=config)

    return FatigueResult(
        features=features, config=config, sanity_log=sanity_log,
        elo_correlations=elo_correlations, nan_rates=nan_rates,
        distribution=distribution, star_seasons=star_seasons,
    )


# ===========================================================================
# Schema
# ===========================================================================

def _validate_inputs(master: pd.DataFrame, sackmann_all: pd.DataFrame) -> None:
    miss_m = set(REQUIRED_MASTER_COLS) - set(master.columns)
    if miss_m:
        raise ValueError(
            f"master missing required columns: {sorted(miss_m)}"
        )
    if sackmann_all.empty:
        return
    miss_s = set(REQUIRED_SACKMANN_COLS) - set(sackmann_all.columns)
    if miss_s:
        raise ValueError(
            f"sackmann_all missing required columns: {sorted(miss_s)}"
        )


# ===========================================================================
# Iteration frame
# ===========================================================================

def _build_unified_iteration_frame(
    master: pd.DataFrame, sackmann_all: pd.DataFrame,
) -> pd.DataFrame:
    """Same pattern as h2h/specialist — chronological frame over ALL surfaces."""
    sk = sackmann_all.copy()
    w = pd.to_numeric(sk["winner_id"], errors="coerce").astype("Int64")
    l = pd.to_numeric(sk["loser_id"], errors="coerce").astype("Int64")
    p1_is_winner = w.lt(l).fillna(False)
    sk["p1_id"] = w.where(p1_is_winner, l).astype("Int64")
    sk["p2_id"] = l.where(p1_is_winner, w).astype("Int64")
    sk["y"] = p1_is_winner.astype("Int8")
    sk["_key"] = (sk["tourney_id"].astype(str) + "_" +
                  sk["p1_id"].astype("Int64").astype(str) + "_" +
                  sk["p2_id"].astype("Int64").astype(str))

    m_link = master[["stable_match_key", "match_id", "match_date"]].rename(
        columns={"stable_match_key": "_key", "match_date": "td_match_date"}
    )
    sk = sk.merge(m_link, on="_key", how="left")
    sk["is_master"] = sk["match_id"].notna()
    sk["match_id_or_na"] = sk["match_id"].astype("Int64")

    sk["sk_match_date"] = pd.to_datetime(sk["match_date"], errors="coerce")
    sk["td_match_date"] = pd.to_datetime(sk["td_match_date"], errors="coerce")
    sk["effective_date"] = sk["td_match_date"].fillna(sk["sk_match_date"])

    round_idx = {r: i for i, r in enumerate(ROUND_ORDER)}
    sk["_rd_idx"] = sk["round"].astype("string").map(round_idx).fillna(99).astype(int)
    sk["_idx_sort"] = sk.index

    sk = sk.sort_values(
        ["effective_date", "tourney_id", "_rd_idx", "_idx_sort"],
        kind="mergesort",
    ).reset_index(drop=True)

    sk["surface_lower"] = sk["surface"].astype("string").str.lower()
    sk = sk.drop(columns=["match_date"]).rename(
        columns={"effective_date": "match_date"}
    )
    return sk[[
        "match_id_or_na", "match_date", "tourney_id",
        "p1_id", "p2_id", "y", "surface_lower",
        "_key", "is_master",
    ]]


# ===========================================================================
# Per-player query helpers (early-termination for speed)
# ===========================================================================

def _iso_year_week(d: pd.Timestamp) -> tuple[int, int]:
    """Return (iso_year, iso_week) — handles year boundary correctly."""
    iso = pd.Timestamp(d).isocalendar()
    return (int(iso.year), int(iso.week))


def _query_rest_days(
    history: list, current_date: pd.Timestamp, current_key: str,
) -> float:
    """Days since most-recent prior match (any surface). NaN if none."""
    # history is chronologically sorted; iterate from end for fastest hit.
    for entry in reversed(history):
        d, _, k = entry
        if d < current_date and k != current_key:
            delta = pd.Timestamp(current_date) - pd.Timestamp(d)
            return float(delta.days)
    return float("nan")


def _query_matches_this_week(
    history: list, current_date: pd.Timestamp, current_key: str,
) -> int:
    """Count strictly-prior matches in the same ISO week as current_date."""
    cur_iso = _iso_year_week(current_date)
    count = 0
    for entry in reversed(history):
        d, _, k = entry
        if not (d < current_date and k != current_key):
            continue
        d_iso = _iso_year_week(d)
        if d_iso != cur_iso:
            break  # sorted: any earlier entry is also out-of-week
        count += 1
    return count


def _query_clay_season_cum(
    history: list, current_date: pd.Timestamp, current_key: str,
    season_start_month: int, season_end_month: int,
) -> int:
    """Count strictly-prior CLAY matches since Apr 1 of current year.

    Returns 0 if current_date is outside the [Apr-Oct] season window.
    """
    cur = pd.Timestamp(current_date)
    if cur.month < season_start_month or cur.month > season_end_month:
        return 0
    season_start = pd.Timestamp(f"{cur.year}-{season_start_month:02d}-01")
    count = 0
    for entry in reversed(history):
        d, is_clay, k = entry
        if not (d < current_date and k != current_key):
            continue
        if pd.Timestamp(d) < season_start:
            break  # sorted: any earlier entry is also pre-season
        if is_clay:
            count += 1
    return count


# ===========================================================================
# Iteration
# ===========================================================================

def _run_fatigue_iteration(
    unified: pd.DataFrame,
    season_start_month: int, season_end_month: int,
) -> list[dict]:
    """Iterate matches; emit features for master rows; update state."""
    state: dict[int, list] = defaultdict(list)
    feature_rows: list[dict] = []

    p1_ids = unified["p1_id"].astype("Int64").to_numpy()
    p2_ids = unified["p2_id"].astype("Int64").to_numpy()
    dates = pd.to_datetime(unified["match_date"]).to_numpy()
    surfaces = unified["surface_lower"].to_numpy()
    keys = unified["_key"].to_numpy()
    is_master = unified["is_master"].to_numpy(dtype=bool)
    match_ids = unified["match_id_or_na"].to_numpy()

    for i in range(len(unified)):
        p1, p2 = int(p1_ids[i]), int(p2_ids[i])
        cur_date = pd.Timestamp(dates[i])
        cur_key = keys[i]
        is_clay = str(surfaces[i]).lower() == "clay"

        if is_master[i]:
            h1 = state.get(p1, [])
            h2 = state.get(p2, [])
            rest_p1 = _query_rest_days(h1, cur_date, cur_key)
            rest_p2 = _query_rest_days(h2, cur_date, cur_key)
            wk_p1 = _query_matches_this_week(h1, cur_date, cur_key)
            wk_p2 = _query_matches_this_week(h2, cur_date, cur_key)
            cum_p1 = _query_clay_season_cum(
                h1, cur_date, cur_key, season_start_month, season_end_month,
            )
            cum_p2 = _query_clay_season_cum(
                h2, cur_date, cur_key, season_start_month, season_end_month,
            )

            # diffs (positive = p1 advantage)
            rest_diff = (rest_p1 - rest_p2
                          if not (np.isnan(rest_p1) or np.isnan(rest_p2))
                          else float("nan"))
            wk_diff = wk_p2 - wk_p1
            cum_diff = cum_p2 - cum_p1

            feature_rows.append({
                "match_id": int(match_ids[i]),
                "rest_days_p1": rest_p1,
                "rest_days_p2": rest_p2,
                "rest_days_diff": rest_diff,
                "matches_this_week_p1": wk_p1,
                "matches_this_week_p2": wk_p2,
                "matches_this_week_diff": wk_diff,
                "clay_season_cum_p1": cum_p1,
                "clay_season_cum_p2": cum_p2,
                "clay_season_cum_diff": cum_diff,
            })

        # Post-emission state update
        state[p1].append((cur_date, is_clay, cur_key))
        state[p2].append((cur_date, is_clay, cur_key))

    return feature_rows


def _finalize_features(feature_rows: list[dict]) -> pd.DataFrame:
    if not feature_rows:
        return pd.DataFrame(columns=list(FEATURE_COLUMNS))
    f = pd.DataFrame(feature_rows)
    f["match_id"] = f["match_id"].astype("Int64")
    for c in ("rest_days_p1", "rest_days_p2", "rest_days_diff"):
        f[c] = f[c].astype("float64")
    for c in ("matches_this_week_p1", "matches_this_week_p2",
              "matches_this_week_diff",
              "clay_season_cum_p1", "clay_season_cum_p2",
              "clay_season_cum_diff"):
        f[c] = f[c].astype("Int64")
    return f[list(FEATURE_COLUMNS)]


# ===========================================================================
# Diagnostics
# ===========================================================================

def _compute_nan_rates(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    rows = []
    for col in FEATURE_COLUMNS:
        if col == "match_id":
            continue
        rate = features[col].isna().mean()
        rows.append({"column": col, "nan_%": round(rate * 100, 2)})
    return pd.DataFrame(rows)


def _compute_distribution(features: pd.DataFrame) -> pd.DataFrame:
    """Distribution percentiles for the 3 *_p1 columns."""
    if features.empty:
        return pd.DataFrame()
    rows = []
    qs = (("min", 0.0), ("p10", 0.10), ("p25", 0.25),
          ("median", 0.50), ("p75", 0.75), ("p90", 0.90),
          ("p99", 0.99), ("max", 1.0))
    for col in ("rest_days_p1", "matches_this_week_p1", "clay_season_cum_p1"):
        s = features[col].dropna()
        if s.empty:
            continue
        row = {"feature": col}
        for label, q in qs:
            row[label] = round(float(s.quantile(q)), 2)
        rows.append(row)
    return pd.DataFrame(rows)


# ===========================================================================
# Star-season audit (real-data verification of clay_season_cum)
# ===========================================================================

_STAR_NAMES: tuple[str, ...] = (
    "Iga Swiatek", "Simona Halep", "Aryna Sabalenka",
    "Sara Errani", "Justine Henin",
)


def _star_season_audit(
    unified: pd.DataFrame, players: pd.DataFrame | None,
    season_start_month: int, season_end_month: int,
) -> pd.DataFrame:
    """For each star, output their peak clay-season cumulative match count.

    Re-iterates the unified frame (small) tracking only star players.
    Reports each year's end-of-clay-season total per star.
    """
    if players is None or "full_name" not in players.columns:
        return pd.DataFrame()

    star_ids: dict[str, int] = {}
    for name in _STAR_NAMES:
        hit = players[players["full_name"].astype("string")
                      .str.contains(name, case=False, na=False, regex=False)]
        if len(hit):
            star_ids[name] = int(hit.iloc[0]["player_id"])

    # For each (star, year), count clay matches in [Apr 1, Oct 31]
    target_ids = set(star_ids.values())
    name_lookup = {pid: pat for pat, pid in star_ids.items()}

    counts: dict[tuple[int, int], int] = defaultdict(int)
    p1_ids = unified["p1_id"].astype("Int64").to_numpy()
    p2_ids = unified["p2_id"].astype("Int64").to_numpy()
    dates = pd.to_datetime(unified["match_date"]).to_numpy()
    surfaces = unified["surface_lower"].to_numpy()

    for i in range(len(unified)):
        d = pd.Timestamp(dates[i])
        if not (season_start_month <= d.month <= season_end_month):
            continue
        if str(surfaces[i]).lower() != "clay":
            continue
        for pid in (int(p1_ids[i]), int(p2_ids[i])):
            if pid in target_ids:
                counts[(pid, d.year)] += 1

    rows = []
    for (pid, yr), c in counts.items():
        rows.append({
            "player": name_lookup[pid],
            "year": yr,
            "clay_season_total": c,
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="player", columns="year",
                            values="clay_season_total", aggfunc="last")
    return pivot.astype("Int64")


# ===========================================================================
# Sanity checks
# ===========================================================================

def _run_sanity_checks(
    master: pd.DataFrame, features: pd.DataFrame,
    unified: pd.DataFrame,
    clay_elo_features: pd.DataFrame | None,
    log_list: list[str],
    *,
    season_start_month: int, season_end_month: int,
    leakage_sample_size: int, rng_seed: int,
    nan_rates: pd.DataFrame,
) -> dict[str, float]:
    n = len(features)
    correlations: dict[str, float] = {}

    # 1. rest_days >= 0 when present
    rd1 = features["rest_days_p1"].dropna()
    rd2 = features["rest_days_p2"].dropna()
    if (rd1 < 0).any() or (rd2 < 0).any():
        raise AssertionError("rest_days < 0 (impossible)")
    log_list.append(f"[✓] rest_days ≥ 0 when present  ({len(rd1)} p1, {len(rd2)} p2)")

    # 2. matches_this_week and clay_season_cum >= 0
    if (features["matches_this_week_p1"] < 0).any() or \
       (features["matches_this_week_p2"] < 0).any():
        raise AssertionError("matches_this_week < 0")
    if (features["clay_season_cum_p1"] < 0).any() or \
       (features["clay_season_cum_p2"] < 0).any():
        raise AssertionError("clay_season_cum < 0")
    log_list.append(f"[✓] matches_this_week, clay_season_cum ≥ 0  {n}/{n}")

    # 3. matches_this_week within reasonable max
    max_wk = int(features["matches_this_week_p1"].max())
    if max_wk > MATCHES_PER_WEEK_REASONABLE_MAX:
        log_list.append(f"[⚠] matches_this_week_p1 max = {max_wk} > "
                      f"{MATCHES_PER_WEEK_REASONABLE_MAX} (suspicious)")
    else:
        log_list.append(f"[✓] matches_this_week max = {max_wk} "
                      f"≤ {MATCHES_PER_WEEK_REASONABLE_MAX}")

    # 4. clay_season_cum within reasonable max
    max_cum = int(features["clay_season_cum_p1"].max())
    if max_cum > CLAY_SEASON_CUM_REASONABLE_MAX:
        log_list.append(f"[⚠] clay_season_cum_p1 max = {max_cum} > "
                      f"{CLAY_SEASON_CUM_REASONABLE_MAX}")
    else:
        log_list.append(f"[✓] clay_season_cum max = {max_cum} "
                      f"≤ {CLAY_SEASON_CUM_REASONABLE_MAX}")

    # 5. NaN rates check
    high = nan_rates[nan_rates["nan_%"] > 5.0]   # >5% is high for fatigue
    if not high.empty:
        for _, r in high.iterrows():
            log_list.append(f"[⚠] {r['column']} NaN = {r['nan_%']}% > 5%")
    else:
        log_list.append("[✓] NaN rates: " + ", ".join(
            f"{r['column']}={r['nan_%']}%" for _, r in nan_rates.iterrows()
        ))

    # 6. Pearson r vs clay_elo_diff for all 3 diff features
    if clay_elo_features is not None and not clay_elo_features.empty:
        joined = features.merge(
            clay_elo_features[["match_id", "clay_elo_diff"]], on="match_id",
        )
        for diff_col in ("rest_days_diff", "matches_this_week_diff",
                          "clay_season_cum_diff"):
            sub = joined[joined[diff_col].notna() &
                          joined["clay_elo_diff"].notna()]
            if len(sub) >= 30:
                r = float(sub[diff_col].corr(sub["clay_elo_diff"]))
                correlations[diff_col] = r
                tag = "[✓]" if abs(r) <= ELO_INDEPENDENCE_WARN_THRESHOLD else "[⚠]"
                log_list.append(
                    f"{tag} r({diff_col}, clay_elo_diff) = {r:+.3f}  "
                    f"(expected < {ELO_INDEPENDENCE_WARN_THRESHOLD})"
                )

    # 7. 100-row stratified leakage cross-check
    _check_no_leakage(
        master, features, unified, season_start_month, season_end_month,
        sample_size=leakage_sample_size, rng_seed=rng_seed, log_list=log_list,
    )

    return correlations


def _check_no_leakage(
    master: pd.DataFrame, features: pd.DataFrame, unified: pd.DataFrame,
    season_start_month: int, season_end_month: int,
    *,
    sample_size: int, rng_seed: int, log_list: list[str],
) -> None:
    """Re-derive 3 fatigue features per sampled row; assert exact match."""
    f_with_year = features.merge(
        master[["match_id", "match_date", "p1_id", "p2_id",
                "stable_match_key"]],
        on="match_id", how="left",
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

    # Rebuild full per-player history
    state: dict[int, list] = defaultdict(list)
    p1_ids = unified["p1_id"].astype("Int64").to_numpy()
    p2_ids = unified["p2_id"].astype("Int64").to_numpy()
    dates = pd.to_datetime(unified["match_date"]).to_numpy()
    surfaces = unified["surface_lower"].to_numpy()
    keys = unified["_key"].to_numpy()
    for i in range(len(unified)):
        is_clay = str(surfaces[i]).lower() == "clay"
        state[int(p1_ids[i])].append(
            (pd.Timestamp(dates[i]), is_clay, keys[i])
        )
        state[int(p2_ids[i])].append(
            (pd.Timestamp(dates[i]), is_clay, keys[i])
        )

    fails = 0
    msgs: list[str] = []
    for _, row in sampled.iterrows():
        cur_date = pd.Timestamp(row["match_date"])
        p1, p2 = int(row["p1_id"]), int(row["p2_id"])
        cur_key = str(row["stable_match_key"])
        rest1 = _query_rest_days(state.get(p1, []), cur_date, cur_key)
        rest2 = _query_rest_days(state.get(p2, []), cur_date, cur_key)
        wk1 = _query_matches_this_week(state.get(p1, []), cur_date, cur_key)
        wk2 = _query_matches_this_week(state.get(p2, []), cur_date, cur_key)
        cum1 = _query_clay_season_cum(
            state.get(p1, []), cur_date, cur_key,
            season_start_month, season_end_month,
        )
        cum2 = _query_clay_season_cum(
            state.get(p2, []), cur_date, cur_key,
            season_start_month, season_end_month,
        )

        def _eq_f(a, b):
            if pd.isna(a) and pd.isna(b): return True
            if pd.isna(a) or pd.isna(b): return False
            return abs(float(a) - float(b)) < 1e-9

        ok = (
            _eq_f(row["rest_days_p1"], rest1)
            and _eq_f(row["rest_days_p2"], rest2)
            and int(row["matches_this_week_p1"]) == wk1
            and int(row["matches_this_week_p2"]) == wk2
            and int(row["clay_season_cum_p1"]) == cum1
            and int(row["clay_season_cum_p2"]) == cum2
        )
        if not ok:
            fails += 1
            if len(msgs) < 3:
                msgs.append(
                    f"  match_id={int(row['match_id'])}: "
                    f"stored rest=({row['rest_days_p1']},{row['rest_days_p2']}) "
                    f"wk=({row['matches_this_week_p1']},{row['matches_this_week_p2']}) "
                    f"cum=({row['clay_season_cum_p1']},{row['clay_season_cum_p2']}) "
                    f"expected rest=({rest1},{rest2}) wk=({wk1},{wk2}) "
                    f"cum=({cum1},{cum2})"
                )
    if fails:
        raise AssertionError(
            f"Leakage check failed: {fails}/{len(sampled)}\n"
            + "\n".join(msgs)
        )
    log_list.append(f"[✓] Leakage cross-check  {len(sampled)}/{len(sampled)} match")
