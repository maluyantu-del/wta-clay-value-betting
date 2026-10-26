"""Rolling-form feature module — last-N clay match winrate + dominance ratio.

Computes per-master-match recent-form features for both players, using a
fixed window of N most recent clay matches ending strictly BEFORE the
current match's date. Window is per-match (not per-day) because clay
season concentration makes any day-window effectively empty during the
9-month off-season.

Output (8 columns + match_id):

  form_winrate_p1, form_winrate_p2, form_winrate_diff
  form_n_matches_p1, form_n_matches_p2
  form_dominance_p1, form_dominance_p2, form_dominance_diff

Cold-start
----------
``form_n_matches < MIN_MATCHES_FOR_VALUE`` (default 5) → both winrate and
dominance are NaN. LightGBM handles NaN natively (Hessian-based split
direction); we never fall back to a fake 0.5 baseline that the model
might mistake for a real signal.

Dominance ratio
---------------
For player ``P`` at match ``m``::

    serve_won_P    = (P.1stWon + P.2ndWon) / P.svpt
    serve_won_opp  = (opp.1stWon + opp.2ndWon) / opp.svpt
    DR(P, m)       = (1 - serve_won_opp) / (1 - serve_won_P)

Capped at 5.0 (typical match DR is 0.5-2.5; 5.0 is a comfortable upper
bound). ``serve_won_P ≥ 1.0`` (player held every serve point — implies
no breaks possible) yields NaN. NaN matches are dropped from the rolling
average; if fewer than ``MIN_MATCHES_FOR_VALUE`` valid DRs in the
window, the feature is NaN.

Leakage prevention
------------------
History query uses STRICT ``date < current_date``. Same-day matches do
NOT contribute to each other's form (some Sackmann files lack
intraday round ordering; the conservative rule is to ignore same-day).

Independence of winrate vs dominance
------------------------------------
Winrate uses only ``y`` and is computable for every match (100% coverage
when ``n ≥ min``). Dominance requires Sackmann serve/return stats which
are sparse in older years (2007-2010 ~30-50% coverage). The two features
are decoupled: missing dominance never invalidates winrate.

Architecture
------------
Same I/O contract as :mod:`wta_clay.features.clay_elo` — writes a parquet
+ sidecar JSON via :func:`wta_clay.features.io.write_features`.
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
# Constants (configurable, NOT hardcoded in body)
# ===========================================================================

WINDOW_SIZE: int = 15                     # last-N clay matches in the window
MIN_MATCHES_FOR_VALUE: int = 5            # below this → NaN
DOMINANCE_CAP: float = 5.0                # upper bound for DR; > 5 → 5
ROUND_ORDER: tuple[str, ...] = ("R128", "R64", "R32", "R16",
                                "QF", "SF", "F", "RR")

#: Output feature columns (joined onto master via ``match_id``).
FEATURE_COLUMNS: tuple[str, ...] = (
    "match_id",
    "form_winrate_p1", "form_winrate_p2", "form_winrate_diff",
    "form_n_matches_p1", "form_n_matches_p2",
    "form_dominance_p1", "form_dominance_p2", "form_dominance_diff",
)

#: Sackmann stat columns required for dominance computation.
_STAT_COLS: tuple[str, ...] = (
    "w_svpt", "w_1stWon", "w_2ndWon",
    "l_svpt", "l_1stWon", "l_2ndWon",
)


# ===========================================================================
# Result dataclass
# ===========================================================================

@dataclass
class FormResult:
    """Top-level return of :func:`compute_form_features`."""
    features: pd.DataFrame
    config: dict[str, Any]
    sanity_log: list[str] = field(default_factory=list)
    coverage_by_year: pd.DataFrame = field(default_factory=pd.DataFrame)
    star_trajectories: pd.DataFrame = field(default_factory=pd.DataFrame)


# ===========================================================================
# Main entry
# ===========================================================================

def compute_form_features(
    master: pd.DataFrame,
    sackmann_clay: pd.DataFrame,
    players: pd.DataFrame | None = None,
    *,
    window_size: int = WINDOW_SIZE,
    min_matches_for_value: int = MIN_MATCHES_FOR_VALUE,
    dominance_cap: float = DOMINANCE_CAP,
    out_path: Path | str | None = None,
    run_sanity_checks: bool = True,
    leakage_sample_size: int = 100,
    rng_seed: int = 42,
) -> FormResult:
    """Compute rolling-form features for every match in master.

    Parameters
    ----------
    master : DataFrame
        From data/processed/clay_matches_master.parquet. Required columns:
        ``match_id, match_date, p1_id, p2_id, y``.
    sackmann_clay : DataFrame
        Sackmann main-draw clay matches (2005+). Required columns:
        ``match_date, tourney_id, winner_id, loser_id, round`` plus the
        stat columns ``w_svpt, w_1stWon, w_2ndWon, l_svpt, l_1stWon,
        l_2ndWon`` for dominance computation. Master rows must align by
        ``sackmann_match_id`` to ``sackmann_clay`` row index.
    players : DataFrame | None
        Sackmann players for star-trajectory diagnostics. Optional.
    """
    config = {
        "window_size": window_size,
        "min_matches_for_value": min_matches_for_value,
        "dominance_cap": dominance_cap,
        "leakage_sample_size": leakage_sample_size,
        "rng_seed": rng_seed,
        "n_master_rows": int(len(master)),
        "n_sackmann_rows": int(len(sackmann_clay)),
    }
    log.info("compute_form_features: master=%d, sackmann=%d",
             config["n_master_rows"], config["n_sackmann_rows"])

    if master.empty:
        log.warning("Empty master; returning empty FormResult.")
        return FormResult(features=pd.DataFrame(columns=list(FEATURE_COLUMNS)),
                          config=config)

    unified = _build_unified_iteration_frame(master, sackmann_clay,
                                              dominance_cap)
    feature_rows, history_state = _run_form_iteration(
        unified, window_size, min_matches_for_value,
    )
    features = _finalize_features(feature_rows)
    coverage_by_year = _compute_coverage_by_year(features, master)
    star_trajectories = _star_trajectories(unified, players,
                                            window_size,
                                            min_matches_for_value)

    sanity_log: list[str] = []
    if run_sanity_checks:
        _run_sanity_checks(
            master, features, unified, sanity_log,
            window_size=window_size,
            min_matches_for_value=min_matches_for_value,
            leakage_sample_size=leakage_sample_size,
            rng_seed=rng_seed,
            coverage_by_year=coverage_by_year,
        )

    if out_path is not None:
        write_features(features, master, path=out_path,
                       feature_module="form", config=config)

    return FormResult(features=features, config=config,
                      sanity_log=sanity_log,
                      coverage_by_year=coverage_by_year,
                      star_trajectories=star_trajectories)


# ===========================================================================
# Iteration frame
# ===========================================================================

def _build_unified_iteration_frame(
    master: pd.DataFrame,
    sackmann_clay: pd.DataFrame,
    dominance_cap: float,
) -> pd.DataFrame:
    """Build a chronological iteration frame over all clay matches.

    Master rows are flagged ``is_master=True``; only those produce feature
    output. Per-match dominance is precomputed for both winner and loser
    sides (capped at ``dominance_cap``).

    The ``effective_date`` column is the per-match canonical date used for
    BOTH sorting and the strict-< filter:
      - master rows: TD's individual ``match_date`` (per-match precision)
      - seed rows:   Sackmann's ``match_date`` (= tourney_date, week start)

    Using TD's individual date for master rows is essential — same-
    tournament rounds happen on different days, and the strict-< filter
    correctly includes earlier rounds in later rounds' history. Using
    Sackmann's tourney_date (which is the same for all rounds within a
    tournament) would over-exclude legitimate prior-round history.
    """
    sk = sackmann_clay.copy()
    # Compute id-sorted p1/p2 framing for Sackmann rows
    w = pd.to_numeric(sk["winner_id"], errors="coerce").astype("Int64")
    l = pd.to_numeric(sk["loser_id"], errors="coerce").astype("Int64")
    p1_is_winner = w.lt(l).fillna(False)
    sk["p1_id"] = w.where(p1_is_winner, l).astype("Int64")
    sk["p2_id"] = l.where(p1_is_winner, w).astype("Int64")
    sk["y"] = p1_is_winner.astype("Int8")

    # Compute dominance for each match (winner + loser perspectives), then
    # remap to p1/p2 perspective using the id-sort flag.
    dr_w_arr, dr_l_arr = _compute_dominance_arrays(sk, dominance_cap)
    sk["dr_p1"] = np.where(p1_is_winner, dr_w_arr, dr_l_arr)
    sk["dr_p2"] = np.where(p1_is_winner, dr_l_arr, dr_w_arr)

    # Mark master rows via the canonical ``stable_match_key`` (added to
    # master by :mod:`wta_clay.data.merge`). We do NOT use ``match_id``
    # for cross-frame linkage because that's a Sackmann row index tied to
    # the specific filter merge.py used; if form.py is called with a
    # different filter range, indices shift silently. ``stable_match_key``
    # is `f"{tourney_id}_{p1_id}_{p2_id}"` and stable across any filter.
    if "stable_match_key" not in master.columns:
        raise ValueError(
            "master must contain 'stable_match_key' column. Re-run "
            "merge.py to regenerate the master parquet."
        )
    sk["_key"] = (sk["tourney_id"].astype(str) + "_" +
                  sk["p1_id"].astype("Int64").astype(str) + "_" +
                  sk["p2_id"].astype("Int64").astype(str))
    m_link = master[["stable_match_key", "match_id", "match_date"]].rename(
        columns={"stable_match_key": "_key", "match_date": "td_match_date"}
    )
    sk = sk.merge(m_link, on="_key", how="left").drop(columns="_key")
    sk["is_master"] = sk["match_id"].notna()
    sk["match_id_or_na"] = sk["match_id"].astype("Int64")
    sk = sk.drop(columns="match_id")

    # Sackmann's per-row match_date is the tourney start (week-level granularity).
    sk["sk_match_date"] = pd.to_datetime(sk["match_date"], errors="coerce")
    sk["td_match_date"] = pd.to_datetime(sk["td_match_date"], errors="coerce")
    # effective_date: TD individual for master, Sackmann tourney_date for seed
    sk["effective_date"] = sk["td_match_date"].fillna(sk["sk_match_date"])

    # Sort: chronological by effective_date + tourney + round + index tiebreak
    round_idx = {r: i for i, r in enumerate(ROUND_ORDER)}
    sk["_rd_idx"] = sk["round"].astype("string").map(round_idx)
    sk["_rd_idx"] = sk["_rd_idx"].fillna(99).astype(int)
    sk["_idx_sort"] = sk.index

    sk = sk.sort_values(
        ["effective_date", "tourney_id", "_rd_idx", "_idx_sort"],
        kind="mergesort",
    ).reset_index(drop=True)

    # Drop the original Sackmann match_date column and rename
    # effective_date → match_date for downstream consumers.
    sk = sk.drop(columns="match_date")
    sk = sk.rename(columns={"effective_date": "match_date"})

    return sk[[
        "match_id_or_na", "match_date", "tourney_id",
        "p1_id", "p2_id", "y",
        "dr_p1", "dr_p2", "is_master",
    ]]


def _compute_dominance_arrays(
    sk: pd.DataFrame, cap: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row DR for (winner, loser). Returns float arrays with NaN where
    stats are missing or denominators are degenerate.
    """
    n = len(sk)
    dr_w = np.full(n, np.nan, dtype=float)
    dr_l = np.full(n, np.nan, dtype=float)

    # Pull as float arrays; missing → NaN
    arrays = {}
    for col in _STAT_COLS:
        if col in sk.columns:
            arrays[col] = pd.to_numeric(sk[col], errors="coerce").to_numpy(
                dtype=float, na_value=np.nan)
        else:
            arrays[col] = np.full(n, np.nan, dtype=float)

    w_svpt = arrays["w_svpt"]
    w_1st = arrays["w_1stWon"]
    w_2nd = arrays["w_2ndWon"]
    l_svpt = arrays["l_svpt"]
    l_1st = arrays["l_1stWon"]
    l_2nd = arrays["l_2ndWon"]

    with np.errstate(divide="ignore", invalid="ignore"):
        # Stats valid mask
        valid = (
            np.isfinite(w_svpt) & (w_svpt > 0) &
            np.isfinite(l_svpt) & (l_svpt > 0) &
            np.isfinite(w_1st) & np.isfinite(w_2nd) &
            np.isfinite(l_1st) & np.isfinite(l_2nd)
        )

        serve_won_w = np.where(valid, (w_1st + w_2nd) / w_svpt, np.nan)
        serve_won_l = np.where(valid, (l_1st + l_2nd) / l_svpt, np.nan)

        denom_w = 1.0 - serve_won_w
        denom_l = 1.0 - serve_won_l

        # Winner DR
        ok_w = valid & np.isfinite(denom_w) & (denom_w > 0)
        raw_w = (1.0 - serve_won_l) / denom_w
        dr_w = np.where(ok_w, np.minimum(raw_w, cap), np.nan)

        # Loser DR
        ok_l = valid & np.isfinite(denom_l) & (denom_l > 0)
        raw_l = (1.0 - serve_won_w) / denom_l
        dr_l = np.where(ok_l, np.minimum(raw_l, cap), np.nan)

    return dr_w, dr_l


# ===========================================================================
# Iteration loop
# ===========================================================================

def _run_form_iteration(
    unified: pd.DataFrame,
    window_size: int,
    min_matches: int,
) -> tuple[list[dict], dict[int, list]]:
    """Iterate matches chronologically, maintaining per-player history.

    For each master row, query the player's history (strictly before
    current date) for last-N matches and compute (winrate, n, dominance).
    Then update history with this match's outcome.

    Returns (feature_rows, final_state).
    """
    # Per-player history: list of (date, won_int, dr_or_nan)
    state: dict[int, list] = defaultdict(list)
    feature_rows: list[dict] = []

    p1_ids = unified["p1_id"].astype("Int64").to_numpy()
    p2_ids = unified["p2_id"].astype("Int64").to_numpy()
    ys = unified["y"].astype("Int8").to_numpy()
    dates = pd.to_datetime(unified["match_date"]).to_numpy()
    dr_p1s = unified["dr_p1"].astype(float).to_numpy()
    dr_p2s = unified["dr_p2"].astype(float).to_numpy()
    is_master = unified["is_master"].to_numpy(dtype=bool)
    match_ids = unified["match_id_or_na"].to_numpy()

    for i in range(len(unified)):
        p1, p2 = int(p1_ids[i]), int(p2_ids[i])
        y = int(ys[i])
        cur_date = dates[i]

        if is_master[i]:
            p1_w, p1_n, p1_d = _query_history(
                state.get(p1, []), cur_date, window_size, min_matches,
            )
            p2_w, p2_n, p2_d = _query_history(
                state.get(p2, []), cur_date, window_size, min_matches,
            )
            feature_rows.append({
                "match_id": int(match_ids[i]),
                "form_winrate_p1": p1_w,
                "form_winrate_p2": p2_w,
                "form_winrate_diff": (p1_w - p2_w
                                       if not (np.isnan(p1_w) or np.isnan(p2_w))
                                       else float("nan")),
                "form_n_matches_p1": p1_n,
                "form_n_matches_p2": p2_n,
                "form_dominance_p1": p1_d,
                "form_dominance_p2": p2_d,
                "form_dominance_diff": (p1_d - p2_d
                                         if not (np.isnan(p1_d) or np.isnan(p2_d))
                                         else float("nan")),
            })

        # Update state AFTER feature emission (post-match)
        state[p1].append((cur_date, 1 if y == 1 else 0, float(dr_p1s[i])))
        state[p2].append((cur_date, 1 if y == 0 else 0, float(dr_p2s[i])))

    return feature_rows, state


def _query_history(
    history: list, current_date, window_size: int, min_matches: int,
) -> tuple[float, int, float]:
    """Compute (winrate, n_matches, dominance_avg) for one player.

    History is a list of (date, won_int, dr_float_or_nan) tuples in
    insertion order (chronological). The strict ``date < current_date``
    filter is applied via reverse iteration since older entries appear
    first in the list.
    """
    # Filter strictly: only matches before current_date
    valid = [(d, w, dr) for d, w, dr in history if d < current_date]
    if not valid:
        return float("nan"), 0, float("nan")
    recent = valid[-window_size:]
    n = len(recent)

    if n < min_matches:
        return float("nan"), n, float("nan")

    winrate = sum(w for _, w, _ in recent) / n
    valid_dr = [dr for _, _, dr in recent if not np.isnan(dr)]
    if len(valid_dr) >= min_matches:
        dominance = sum(valid_dr) / len(valid_dr)
    else:
        dominance = float("nan")
    return winrate, n, dominance


def _finalize_features(feature_rows: list[dict]) -> pd.DataFrame:
    """Convert feature row dicts to a typed DataFrame in canonical column order."""
    if not feature_rows:
        return pd.DataFrame(columns=list(FEATURE_COLUMNS))
    f = pd.DataFrame(feature_rows)
    f["match_id"] = f["match_id"].astype("Int64")
    f["form_winrate_p1"] = f["form_winrate_p1"].astype("float64")
    f["form_winrate_p2"] = f["form_winrate_p2"].astype("float64")
    f["form_winrate_diff"] = f["form_winrate_diff"].astype("float64")
    f["form_n_matches_p1"] = f["form_n_matches_p1"].astype("Int64")
    f["form_n_matches_p2"] = f["form_n_matches_p2"].astype("Int64")
    f["form_dominance_p1"] = f["form_dominance_p1"].astype("float64")
    f["form_dominance_p2"] = f["form_dominance_p2"].astype("float64")
    f["form_dominance_diff"] = f["form_dominance_diff"].astype("float64")
    return f[list(FEATURE_COLUMNS)]


# ===========================================================================
# Coverage report (per-year)
# ===========================================================================

def _compute_coverage_by_year(features: pd.DataFrame,
                               master: pd.DataFrame) -> pd.DataFrame:
    """Per-year non-NaN coverage of winrate and dominance."""
    if features.empty:
        return pd.DataFrame()
    f = features.merge(master[["match_id", "match_date"]], on="match_id")
    f["yr"] = f["match_date"].dt.year
    rows = []
    for yr, g in f.groupby("yr"):
        n = len(g)
        winrate_cov = g["form_winrate_p1"].notna().sum() / n
        dominance_cov = g["form_dominance_p1"].notna().sum() / n
        rows.append({
            "year": int(yr), "n": n,
            "winrate_coverage_%": round(winrate_cov * 100, 2),
            "dominance_coverage_%": round(dominance_cov * 100, 2),
        })
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


# ===========================================================================
# Sanity checks
# ===========================================================================

def _run_sanity_checks(
    master: pd.DataFrame,
    features: pd.DataFrame,
    unified: pd.DataFrame,
    log_list: list[str],
    *,
    window_size: int,
    min_matches_for_value: int,
    leakage_sample_size: int,
    rng_seed: int,
    coverage_by_year: pd.DataFrame,
) -> None:
    """Run all 9 sanity checks; raise on hard failures."""
    n = len(features)

    # 1. n_matches in [0, window_size]
    nm1 = features["form_n_matches_p1"]
    nm2 = features["form_n_matches_p2"]
    if (nm1 < 0).any() or (nm1 > window_size).any() or \
       (nm2 < 0).any() or (nm2 > window_size).any():
        raise AssertionError(f"form_n_matches outside [0, {window_size}]")
    log_list.append(f"[✓] form_n_matches in [0, {window_size}]  {n}/{n}")

    # 2. Cold-start NaN consistency
    cold1 = features["form_n_matches_p1"] < min_matches_for_value
    bad1 = cold1 & features["form_winrate_p1"].notna()
    cold2 = features["form_n_matches_p2"] < min_matches_for_value
    bad2 = cold2 & features["form_winrate_p2"].notna()
    if bad1.any() or bad2.any():
        raise AssertionError(
            f"Cold-start NaN violation: {int(bad1.sum() + bad2.sum())} rows "
            f"have winrate when n < {min_matches_for_value}"
        )
    log_list.append(f"[✓] Cold-start NaN consistency  "
                  f"(n_matches < {min_matches_for_value} → winrate NaN)")

    # 3. Winrate in [0, 1] when present
    eligible_w1 = features["form_winrate_p1"].dropna()
    eligible_w2 = features["form_winrate_p2"].dropna()
    if (~eligible_w1.between(0.0, 1.0)).any() or \
       (~eligible_w2.between(0.0, 1.0)).any():
        raise AssertionError("form_winrate outside [0, 1]")
    log_list.append(f"[✓] form_winrate in [0, 1]  "
                  f"({len(eligible_w1) + len(eligible_w2)} present)")

    # 4. Mean winrate ≈ 0.50 across master rows where n_matches >= 10 (warn)
    deep_p1 = features[features["form_n_matches_p1"] >= 10]["form_winrate_p1"]
    deep_p2 = features[features["form_n_matches_p2"] >= 10]["form_winrate_p2"]
    if len(deep_p1) and len(deep_p2):
        combined = pd.concat([deep_p1, deep_p2])
        mean_wr = float(combined.mean())
        if 0.45 <= mean_wr <= 0.55:
            log_list.append(f"[✓] Mean form_winrate (n≥10) = {mean_wr:.3f} "
                          f"⊆ [0.45, 0.55]")
        else:
            log_list.append(f"[⚠] Mean form_winrate (n≥10) = {mean_wr:.3f} "
                          f"outside [0.45, 0.55]")

    # 5. Distribution shape: 90th pct ≤ 0.85, 10th pct ≥ 0.20 (warn) for n≥10
    if len(deep_p1):
        combined = pd.concat([deep_p1, deep_p2])
        p10 = float(combined.quantile(0.10))
        p90 = float(combined.quantile(0.90))
        if p10 >= 0.20 and p90 <= 0.85:
            log_list.append(f"[✓] Winrate p10={p10:.2f}, p90={p90:.2f} "
                          f"⊆ [0.20, 0.85]")
        else:
            log_list.append(f"[⚠] Winrate p10={p10:.2f}, p90={p90:.2f} "
                          f"outside [0.20, 0.85]")

    # 6. Extreme outliers concentrated in low n_matches (warn)
    extreme = features[
        ((features["form_winrate_p1"] < 0.20) |
         (features["form_winrate_p1"] > 0.85)) &
        features["form_winrate_p1"].notna()
    ]
    low_n = (extreme["form_n_matches_p1"] < 8).sum()
    if len(extreme):
        rate = low_n / len(extreme)
        if rate >= 0.5:
            log_list.append(f"[✓] Extreme winrates concentrated in n<8 "
                          f"({rate*100:.1f}%, {low_n}/{len(extreme)})")
        else:
            log_list.append(f"[⚠] Extreme winrates poorly concentrated in n<8 "
                          f"({rate*100:.1f}%) — review")
    else:
        log_list.append(f"[✓] No extreme winrates (<0.20 or >0.85)")

    # 7. Leakage cross-check (raise)
    _check_no_leakage(
        features, unified, master, window_size, min_matches_for_value,
        sample_size=leakage_sample_size, rng_seed=rng_seed, log_list=log_list,
    )

    # 8. Dominance coverage per-year (warn for years <50% post-2010)
    if not coverage_by_year.empty:
        late = coverage_by_year[coverage_by_year["year"] >= 2011]
        bad_late = late[late["dominance_coverage_%"] < 50.0]
        if bad_late.empty:
            log_list.append(f"[✓] Dominance coverage ≥50% for all years "
                          f"≥2011")
        else:
            log_list.append(f"[⚠] Dominance coverage <50% for years: "
                          f"{bad_late['year'].tolist()}")

    # 9. Winrate is decoupled from dominance (raise)
    # Where winrate is non-NaN, n_matches >= min. Where dominance is NaN,
    # winrate may still be non-NaN — that's the decoupling guarantee.
    has_w_no_d = (
        features["form_winrate_p1"].notna() &
        features["form_dominance_p1"].isna()
    )
    log_list.append(f"[✓] Winrate independent of dominance  "
                  f"({int(has_w_no_d.sum())} rows with winrate but no dominance)")


def _check_no_leakage(
    features: pd.DataFrame,
    unified: pd.DataFrame,
    master: pd.DataFrame,
    window_size: int,
    min_matches: int,
    *,
    sample_size: int,
    rng_seed: int,
    log_list: list[str],
) -> None:
    """Stratified-by-year sample; re-derive winrate via independent path.

    For each sampled match_id, independently filter sackmann (via unified)
    to the player's strictly-before-date matches, take last N, compute
    winrate, and compare with stored value. Winrates are rationals so we
    use exact equality (np.isclose with tiny tolerance for safety).
    """
    f_with_year = features.merge(
        master[["match_id", "match_date"]], on="match_id", how="left",
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

    # Build per-player history dict from unified (full pre-iteration history)
    by_player: dict[int, list] = defaultdict(list)
    for row in unified.itertuples():
        by_player[int(row.p1_id)].append(
            (row.match_date, 1 if row.y == 1 else 0, float(row.dr_p1))
        )
        by_player[int(row.p2_id)].append(
            (row.match_date, 1 if row.y == 0 else 0, float(row.dr_p2))
        )

    fails = 0
    failure_msgs = []
    master_lookup = master.set_index("match_id")[["match_date",
                                                    "p1_id", "p2_id"]]

    for _, row in sampled.iterrows():
        mid = int(row["match_id"])
        m_row = master_lookup.loc[mid]
        cur_date = pd.Timestamp(m_row["match_date"]).to_datetime64()
        p1_id = int(m_row["p1_id"])
        p2_id = int(m_row["p2_id"])

        wr_p1, n_p1, _ = _query_history(
            by_player.get(p1_id, []), cur_date, window_size, min_matches,
        )
        wr_p2, n_p2, _ = _query_history(
            by_player.get(p2_id, []), cur_date, window_size, min_matches,
        )

        stored_p1 = float(row["form_winrate_p1"]) if not pd.isna(
            row["form_winrate_p1"]) else np.nan
        stored_p2 = float(row["form_winrate_p2"]) if not pd.isna(
            row["form_winrate_p2"]) else np.nan

        # NaN-aware comparison
        ok_p1 = (np.isnan(wr_p1) and np.isnan(stored_p1)) or \
                (np.isfinite(wr_p1) and abs(wr_p1 - stored_p1) < 1e-9)
        ok_p2 = (np.isnan(wr_p2) and np.isnan(stored_p2)) or \
                (np.isfinite(wr_p2) and abs(wr_p2 - stored_p2) < 1e-9)

        if not (ok_p1 and ok_p2):
            fails += 1
            if len(failure_msgs) < 3:
                failure_msgs.append(
                    f"  match_id={mid}: p1 stored={stored_p1} expected={wr_p1}, "
                    f"p2 stored={stored_p2} expected={wr_p2}"
                )

    if fails:
        raise AssertionError(
            f"Leakage check failed: {fails} of {len(sampled)} sampled "
            f"matches show form_winrate discrepancy.\n"
            + "\n".join(failure_msgs)
        )
    log_list.append(f"[✓] Leakage cross-check  {len(sampled)}/{len(sampled)} "
                  f"match (winrate exact)")


# ===========================================================================
# Star trajectories (diagnostic — year-end winrate snapshots)
# ===========================================================================

_STAR_PATTERNS: tuple[str, ...] = (
    "Iga Swiatek", "Simona Halep", "Aryna Sabalenka",
    "Sara Errani", "Justine Henin",
)


def _star_trajectories(
    unified: pd.DataFrame,
    players: pd.DataFrame | None,
    window_size: int,
    min_matches: int,
) -> pd.DataFrame:
    """Re-run iteration capturing each star's year-end form_winrate.

    Each year boundary, we query the star's history (strictly before
    Jan 1 of the next year) for last-N matches and record winrate.
    """
    if players is None or "full_name" not in players.columns:
        return pd.DataFrame()

    star_ids = {}
    for pat in _STAR_PATTERNS:
        hit = players[players["full_name"].astype("string")
                      .str.contains(pat, case=False, na=False, regex=False)]
        if len(hit):
            star_ids[pat] = int(hit.iloc[0]["player_id"])
    if not star_ids:
        return pd.DataFrame()

    state: dict[int, list] = defaultdict(list)
    snapshots = []
    last_year = None

    p1_ids = unified["p1_id"].astype("Int64").to_numpy()
    p2_ids = unified["p2_id"].astype("Int64").to_numpy()
    ys = unified["y"].astype("Int8").to_numpy()
    dates = pd.to_datetime(unified["match_date"]).to_numpy()
    dr_p1s = unified["dr_p1"].astype(float).to_numpy()
    dr_p2s = unified["dr_p2"].astype(float).to_numpy()
    star_set = set(star_ids.values())

    for i in range(len(unified)):
        cur_year = pd.Timestamp(dates[i]).year
        if last_year is not None and cur_year != last_year:
            year_end = pd.Timestamp(f"{last_year}-12-31").to_datetime64()
            for pid in star_set:
                wr, n, _ = _query_history(state.get(pid, []), year_end,
                                           window_size, min_matches)
                if not np.isnan(wr):
                    snapshots.append({
                        "year": last_year, "player_id": pid, "winrate": wr,
                    })
        last_year = cur_year

        p1, p2 = int(p1_ids[i]), int(p2_ids[i])
        y = int(ys[i])
        state[p1].append((dates[i], 1 if y == 1 else 0, float(dr_p1s[i])))
        state[p2].append((dates[i], 1 if y == 0 else 0, float(dr_p2s[i])))

    if last_year is not None:
        year_end = pd.Timestamp(f"{last_year}-12-31").to_datetime64()
        for pid in star_set:
            wr, n, _ = _query_history(state.get(pid, []), year_end,
                                       window_size, min_matches)
            if not np.isnan(wr):
                snapshots.append({
                    "year": last_year, "player_id": pid, "winrate": wr,
                })

    snap = pd.DataFrame(snapshots)
    if snap.empty:
        return snap
    name_lookup = {pid: pat for pat, pid in star_ids.items()}
    snap["player_name"] = snap["player_id"].map(name_lookup)
    pivot = snap.pivot_table(index="player_name", columns="year",
                              values="winrate", aggfunc="last")
    return (pivot * 100).round(1)
