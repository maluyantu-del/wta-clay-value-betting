"""Head-to-head feature module — clay + all-surface H2H with Bayesian smoothing.

Output (6 cols + match_id):

  h2h_n_clay              Int64    prior clay matches between (p1, p2)
  h2h_p1_wins             Int64    p1 wins on clay
  h2h_smoothed_ratio_p1   float64  Bayesian-smoothed p1 win rate on clay [0, 1]
                                   (NEVER NaN — returns α/(α+β) at n_clay=0)
  h2h_n_overall           Int64    prior matches across ALL surfaces
  h2h_p1_wins_overall     Int64    p1 wins across all surfaces

Smoothing
---------
``smoothed = (wins_clay + α) / (n_clay + α + β)`` with default ``α = β = 5``.

  - At n_clay = 0:  smoothed = 5/(5+5) = 0.5      (no information)
  - At n_clay = 10, 7 wins: smoothed = (7+5)/(10+10) = 0.60
                            (data 7/10=0.70 weighted 50%; prior 0.50 weighted 50%)
  - At n_clay = 20, 14 wins: smoothed = (14+5)/(20+10) = 0.633
                            (data weighted 67%; prior weighted 33%)

α/β are locked at 5/5 for v1; future ablation may try 3/3 (weaker prior, faster
signal at moderate n).

Operational guidance — interpret the Pearson r of ``h2h_smoothed_ratio_p1``
vs ``clay_elo_diff`` (computed in :func:`compute_h2h_features` self-check):

  | r            | action                                                       |
  |--------------|--------------------------------------------------------------|
  | r < 0.5      | H2H carries new signal — add directly                        |
  | 0.5 ≤ r ≤ 0.7| partial overlap — gate via ablation; add only if valid AUC   |
  |              | lift > 0.005 vs clay-Elo-only                                |
  | r > 0.7      | H2H is a noisy proxy for Elo — skip                          |

The self-check WARNS at r > 0.7 but never raises — the call is deferred to
the modeling phase per the operational guidance above.

Leakage prevention
------------------
Strict ``history.date < current.date AND history.stable_match_key !=
current.stable_match_key``. Same-day matches between the same pair (rare,
e.g. a re-do) are excluded. Match-key check is defensive — strict-< on
date already excludes self, but key check guards against any future
edge case where two distinct matches share a date.

Sort tiebreaker
---------------
``(effective_date, tourney_id, round_norm_idx)`` ascending. Sackmann's
match_date is week-level granularity in older years; without round-order
tiebreaker, two same-week matches at the same tournament could be
processed out of round order.

Reads ``master.stable_match_key`` for cross-frame stable linkage. Writes
features parquet + sidecar via :func:`wta_clay.features.io.write_features`,
which embeds ``master_key_hash`` (SHA-256 of master.stable_match_key set)
to detect any subsequent master drift on load.
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

ALPHA: float = 5.0
BETA: float = 5.0

ROUND_ORDER: tuple[str, ...] = ("R128", "R64", "R32", "R16",
                                "QF", "SF", "F", "RR")

#: Required columns on the input ``sackmann_all`` DataFrame.
REQUIRED_SACKMANN_COLS: tuple[str, ...] = (
    "tourney_id", "match_date", "winner_id", "loser_id",
    "surface", "round",
)

#: Required columns on master.
REQUIRED_MASTER_COLS: tuple[str, ...] = (
    "match_id", "match_date", "tourney_id", "p1_id", "p2_id", "y",
    "stable_match_key",
)

FEATURE_COLUMNS: tuple[str, ...] = (
    "match_id",
    "h2h_n_clay",
    "h2h_p1_wins",
    "h2h_smoothed_ratio_p1",
    "h2h_n_overall",
    "h2h_p1_wins_overall",
)

#: Pearson r soft-warn threshold vs clay_elo_diff.
ELO_CORRELATION_WARN_THRESHOLD: float = 0.70

#: Distribution buckets for h2h_n_clay (locked breakdown).
_N_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("0",     0,  0),
    ("1-2",   1,  2),
    ("3-5",   3,  5),
    ("6-10",  6, 10),
    ("11+",  11, 10**9),
)


# ===========================================================================
# Result dataclass
# ===========================================================================

@dataclass
class H2HResult:
    """Top-level return of :func:`compute_h2h_features`."""
    features: pd.DataFrame
    config: dict[str, Any]
    sanity_log: list[str] = field(default_factory=list)
    elo_correlation: float | None = None
    n_distribution: pd.DataFrame = field(default_factory=pd.DataFrame)
    coverage_by_year: pd.DataFrame = field(default_factory=pd.DataFrame)
    star_matchups: pd.DataFrame = field(default_factory=pd.DataFrame)


# ===========================================================================
# Main entry
# ===========================================================================

def compute_h2h_features(
    master: pd.DataFrame,
    sackmann_all: pd.DataFrame,
    clay_elo_features: pd.DataFrame | None = None,
    players: pd.DataFrame | None = None,
    *,
    alpha: float = ALPHA,
    beta: float = BETA,
    out_path: Path | str | None = None,
    run_sanity_checks: bool = True,
    leakage_sample_size: int = 100,
    rng_seed: int = 42,
) -> H2HResult:
    """Compute Bayesian-smoothed H2H features for every match in master.

    Inputs
    ------
    master : DataFrame
        Output of :mod:`wta_clay.data.merge`. Must contain
        ``stable_match_key`` (for cross-frame linkage; row-index ``match_id``
        is filter-dependent and intentionally NOT used here).
    sackmann_all : DataFrame
        Sackmann main-draw matches (ALL surfaces — clay, hard, grass,
        carpet) for the historical window. Required columns::

            tourney_id, match_date, winner_id, loser_id, surface, round

        Loaded externally (caller responsibility) so that h2h.py doesn't
        re-load on every call. Schema is validated at function start.
    clay_elo_features : DataFrame | None
        Output of :func:`wta_clay.features.clay_elo.compute_clay_elo`.
        If provided, the self-check computes Pearson r of
        ``h2h_smoothed_ratio_p1`` vs ``clay_elo_diff`` (rows where
        ``h2h_n_clay > 0``) and warns above 0.7.
    players : DataFrame | None
        Sackmann players for star-matchup labels.
    """
    config = {
        "alpha": alpha, "beta": beta,
        "n_master_rows": int(len(master)),
        "n_sackmann_rows": int(len(sackmann_all)),
        "elo_correlation_warn_threshold": ELO_CORRELATION_WARN_THRESHOLD,
    }
    log.info("compute_h2h_features: master=%d, sackmann=%d, alpha=%g, beta=%g",
             config["n_master_rows"], config["n_sackmann_rows"], alpha, beta)

    if master.empty:
        log.warning("Empty master; returning empty H2HResult.")
        return H2HResult(
            features=pd.DataFrame(columns=list(FEATURE_COLUMNS)),
            config=config,
        )

    _validate_inputs(master, sackmann_all)

    unified = _build_unified_iteration_frame(master, sackmann_all)
    feature_rows = _run_h2h_iteration(unified, alpha, beta)
    features = _finalize_features(feature_rows)

    n_distribution = _compute_n_distribution(features)
    coverage_by_year = _compute_coverage_by_year(features, master)
    star_matchups = _star_matchups(unified, players)

    sanity_log: list[str] = []
    elo_correlation: float | None = None
    if run_sanity_checks:
        elo_correlation = _run_sanity_checks(
            master, features, unified, clay_elo_features,
            sanity_log,
            alpha=alpha, beta=beta,
            leakage_sample_size=leakage_sample_size, rng_seed=rng_seed,
        )

    if out_path is not None:
        write_features(features, master, path=out_path,
                       feature_module="h2h", config=config)

    return H2HResult(
        features=features,
        config=config,
        sanity_log=sanity_log,
        elo_correlation=elo_correlation,
        n_distribution=n_distribution,
        coverage_by_year=coverage_by_year,
        star_matchups=star_matchups,
    )


# ===========================================================================
# Schema validation
# ===========================================================================

def _validate_inputs(master: pd.DataFrame, sackmann_all: pd.DataFrame) -> None:
    """Schema check both inputs; raise ``ValueError`` on missing columns."""
    miss_sk = set(REQUIRED_SACKMANN_COLS) - set(sackmann_all.columns)
    if miss_sk:
        raise ValueError(
            f"sackmann_all missing required columns: {sorted(miss_sk)}. "
            f"Required: {list(REQUIRED_SACKMANN_COLS)}"
        )
    miss_m = set(REQUIRED_MASTER_COLS) - set(master.columns)
    if miss_m:
        raise ValueError(
            f"master missing required columns: {sorted(miss_m)}. "
            f"Required: {list(REQUIRED_MASTER_COLS)}. "
            f"Re-run merge.py if 'stable_match_key' is missing."
        )


# ===========================================================================
# Iteration frame
# ===========================================================================

def _build_unified_iteration_frame(
    master: pd.DataFrame, sackmann_all: pd.DataFrame,
) -> pd.DataFrame:
    """Build a chronological iteration frame over all clay+non-clay matches.

    Master rows are flagged via ``stable_match_key`` cross-link. The
    ``effective_date`` column is TD's individual ``match_date`` for
    master rows and Sackmann's ``match_date`` for non-master rows
    (Sackmann's is week-start tourney_date in the data layer).

    Sort: (effective_date, tourney_id, round_norm_idx) ascending.
    """
    sk = sackmann_all.copy()

    # id-sorted p1/p2
    w = pd.to_numeric(sk["winner_id"], errors="coerce").astype("Int64")
    l = pd.to_numeric(sk["loser_id"], errors="coerce").astype("Int64")
    p1_is_winner = w.lt(l).fillna(False)
    sk["p1_id"] = w.where(p1_is_winner, l).astype("Int64")
    sk["p2_id"] = l.where(p1_is_winner, w).astype("Int64")
    sk["y"] = p1_is_winner.astype("Int8")

    # Stable composite key
    sk["_key"] = (sk["tourney_id"].astype(str) + "_" +
                  sk["p1_id"].astype("Int64").astype(str) + "_" +
                  sk["p2_id"].astype("Int64").astype(str))

    # Master link via stable_match_key
    m_link = master[["stable_match_key", "match_id", "match_date"]].rename(
        columns={"stable_match_key": "_key", "match_date": "td_match_date"}
    )
    sk = sk.merge(m_link, on="_key", how="left")
    sk["is_master"] = sk["match_id"].notna()
    sk["match_id_or_na"] = sk["match_id"].astype("Int64")
    # Stable key for non-master rows = the auto-generated _key (no master entry).
    # This ensures the strict ``key != current`` check in queries works
    # uniformly — every history entry has a unique key.
    sk["sk_match_date"] = pd.to_datetime(sk["match_date"], errors="coerce")
    sk["td_match_date"] = pd.to_datetime(sk["td_match_date"], errors="coerce")
    sk["effective_date"] = sk["td_match_date"].fillna(sk["sk_match_date"])

    # Round ordering for tiebreaker
    round_idx = {r: i for i, r in enumerate(ROUND_ORDER)}
    sk["_rd_idx"] = sk["round"].astype("string").map(round_idx)
    sk["_rd_idx"] = sk["_rd_idx"].fillna(99).astype(int)
    sk["_idx_sort"] = sk.index

    sk = sk.sort_values(
        ["effective_date", "tourney_id", "_rd_idx", "_idx_sort"],
        kind="mergesort",
    ).reset_index(drop=True)

    # Lower-case surface for case-insensitive comparison
    sk["surface_lower"] = sk["surface"].astype("string").str.lower()

    # Drop original Sackmann match_date to avoid duplicate-name conflict
    sk = sk.drop(columns=["match_date"])
    sk = sk.rename(columns={"effective_date": "match_date"})

    return sk[[
        "match_id_or_na", "match_date", "tourney_id",
        "p1_id", "p2_id", "y", "surface_lower",
        "_key", "is_master",
    ]]


# ===========================================================================
# H2H iteration
# ===========================================================================

def _run_h2h_iteration(
    unified: pd.DataFrame, alpha: float, beta: float,
) -> list[dict]:
    """Iterate matches, maintain per-pair state, emit features for master rows.

    State key: ``(p1_id, p2_id)`` — already id-sorted in unified.
    State value: list of tuples ``(date, p1_won_int, surface_lower, key)``.
    """
    state: dict[tuple[int, int], list] = defaultdict(list)
    feature_rows: list[dict] = []

    p1_ids = unified["p1_id"].astype("Int64").to_numpy()
    p2_ids = unified["p2_id"].astype("Int64").to_numpy()
    ys = unified["y"].astype("Int8").to_numpy()
    dates = pd.to_datetime(unified["match_date"]).to_numpy()
    surfaces = unified["surface_lower"].to_numpy()
    keys = unified["_key"].to_numpy()
    is_master = unified["is_master"].to_numpy(dtype=bool)
    match_ids = unified["match_id_or_na"].to_numpy()

    for i in range(len(unified)):
        p1, p2 = int(p1_ids[i]), int(p2_ids[i])
        cur_date = dates[i]
        cur_key = keys[i]
        pair_key = (p1, p2)

        if is_master[i]:
            history = state.get(pair_key, [])
            n_clay, p1_wins_clay, n_overall, p1_wins_overall = \
                _query_pair_history(history, cur_date, cur_key)
            smoothed = (p1_wins_clay + alpha) / (n_clay + alpha + beta)
            feature_rows.append({
                "match_id": int(match_ids[i]),
                "h2h_n_clay": n_clay,
                "h2h_p1_wins": p1_wins_clay,
                "h2h_smoothed_ratio_p1": smoothed,
                "h2h_n_overall": n_overall,
                "h2h_p1_wins_overall": p1_wins_overall,
            })

        # Update state AFTER feature emission
        p1_won = 1 if int(ys[i]) == 1 else 0
        state[pair_key].append(
            (cur_date, p1_won, str(surfaces[i]) if surfaces[i] is not None else "",
             cur_key)
        )

    return feature_rows


def _query_pair_history(
    history: list, current_date, current_key,
) -> tuple[int, int, int, int]:
    """Return (n_clay, p1_wins_clay, n_overall, p1_wins_overall).

    Strict filter: ``date < current_date AND key != current_key``.
    """
    n_clay = 0
    p1_wins_clay = 0
    n_overall = 0
    p1_wins_overall = 0
    for d, p1_won, surf, key in history:
        if not (d < current_date and key != current_key):
            continue
        n_overall += 1
        p1_wins_overall += p1_won
        if surf == "clay":
            n_clay += 1
            p1_wins_clay += p1_won
    return n_clay, p1_wins_clay, n_overall, p1_wins_overall


def _finalize_features(feature_rows: list[dict]) -> pd.DataFrame:
    """Convert feature rows to typed DataFrame in canonical column order."""
    if not feature_rows:
        return pd.DataFrame(columns=list(FEATURE_COLUMNS))
    f = pd.DataFrame(feature_rows)
    f["match_id"] = f["match_id"].astype("Int64")
    f["h2h_n_clay"] = f["h2h_n_clay"].astype("Int64")
    f["h2h_p1_wins"] = f["h2h_p1_wins"].astype("Int64")
    f["h2h_smoothed_ratio_p1"] = f["h2h_smoothed_ratio_p1"].astype("float64")
    f["h2h_n_overall"] = f["h2h_n_overall"].astype("Int64")
    f["h2h_p1_wins_overall"] = f["h2h_p1_wins_overall"].astype("Int64")
    return f[list(FEATURE_COLUMNS)]


# ===========================================================================
# Distribution / coverage diagnostics
# ===========================================================================

def _compute_n_distribution(features: pd.DataFrame) -> pd.DataFrame:
    """Bucket counts of ``h2h_n_clay`` per the locked breakdown."""
    if features.empty:
        return pd.DataFrame(columns=["bucket", "n", "pct"])
    rows = []
    n_total = len(features)
    n = features["h2h_n_clay"].astype(int)
    for label, lo, hi in _N_BUCKETS:
        mask = (n >= lo) & (n <= hi)
        cnt = int(mask.sum())
        rows.append({
            "bucket": label,
            "n": cnt,
            "pct": round(cnt / n_total * 100, 2) if n_total else 0.0,
        })
    return pd.DataFrame(rows)


def _compute_coverage_by_year(
    features: pd.DataFrame, master: pd.DataFrame,
) -> pd.DataFrame:
    """Per-year fraction of master rows with ``h2h_n_clay > 0``."""
    if features.empty:
        return pd.DataFrame()
    merged = features.merge(master[["match_id", "match_date"]], on="match_id")
    merged["yr"] = merged["match_date"].dt.year
    rows = []
    for yr, g in merged.groupby("yr"):
        n = len(g)
        any_h2h = int((g["h2h_n_clay"] > 0).sum())
        rows.append({
            "year": int(yr),
            "n_master": n,
            "any_h2h_n": any_h2h,
            "any_h2h_%": round(any_h2h / n * 100, 2) if n else 0.0,
        })
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


# ===========================================================================
# Star matchups
# ===========================================================================

#: Pairs to verify against published H2H records.
_STAR_PAIRS: tuple[tuple[str, str], ...] = (
    ("Iga Swiatek", "Aryna Sabalenka"),
    ("Simona Halep", "Maria Sharapova"),
    ("Iga Swiatek", "Simona Halep"),
    ("Sara Errani", "Serena Williams"),
)


def _star_matchups(
    unified: pd.DataFrame, players: pd.DataFrame | None,
) -> pd.DataFrame:
    """Print every clay match between each star pair for verification.

    Returns a long-format DataFrame: pair_label, date, tourney_id, surface,
    p1_id, p2_id, p1_won_int.
    """
    if players is None or "full_name" not in players.columns:
        return pd.DataFrame()

    rows = []
    for name_a, name_b in _STAR_PAIRS:
        ha = players[players["full_name"].astype("string")
                     .str.contains(name_a, case=False, na=False, regex=False)]
        hb = players[players["full_name"].astype("string")
                     .str.contains(name_b, case=False, na=False, regex=False)]
        if not len(ha) or not len(hb):
            continue
        pid_a = int(ha.iloc[0]["player_id"])
        pid_b = int(hb.iloc[0]["player_id"])
        p1_id = min(pid_a, pid_b)
        p2_id = max(pid_a, pid_b)
        # Names by id-order
        if p1_id == pid_a:
            p1_name, p2_name = name_a, name_b
        else:
            p1_name, p2_name = name_b, name_a
        # Filter unified to this pair only, clay only
        mask = ((unified["p1_id"].astype("Int64") == p1_id) &
                (unified["p2_id"].astype("Int64") == p2_id) &
                (unified["surface_lower"] == "clay"))
        sub = unified[mask].sort_values("match_date")
        for _, r in sub.iterrows():
            rows.append({
                "pair": f"{p1_name} vs {p2_name}",
                "match_date": pd.Timestamp(r["match_date"]).date(),
                "tourney_id": str(r["tourney_id"]),
                "p1_won": int(r["y"]) == 1,
                "p1_name": p1_name,
                "p2_name": p2_name,
            })
    return pd.DataFrame(rows)


# ===========================================================================
# Self-checks (7)
# ===========================================================================

def _run_sanity_checks(
    master: pd.DataFrame,
    features: pd.DataFrame,
    unified: pd.DataFrame,
    clay_elo_features: pd.DataFrame | None,
    log_list: list[str],
    *,
    alpha: float, beta: float,
    leakage_sample_size: int, rng_seed: int,
) -> float | None:
    """Run all 7 sanity checks. Return the Elo correlation (or None)."""
    n = len(features)

    # 1. h2h_n_clay >= 0 AND h2h_p1_wins <= h2h_n_clay
    nc = features["h2h_n_clay"].astype(int)
    pw = features["h2h_p1_wins"].astype(int)
    if (nc < 0).any() or (pw < 0).any() or (pw > nc).any():
        raise AssertionError("h2h_p1_wins outside [0, h2h_n_clay] range")
    log_list.append(f"[✓] h2h_p1_wins ∈ [0, h2h_n_clay]  {n}/{n}")

    # 2. h2h_n_overall >= h2h_n_clay
    nov = features["h2h_n_overall"].astype(int)
    pwov = features["h2h_p1_wins_overall"].astype(int)
    if (nov < nc).any():
        raise AssertionError("h2h_n_overall < h2h_n_clay (clay must be a subset)")
    if (pwov > nov).any() or (pwov < 0).any():
        raise AssertionError("h2h_p1_wins_overall outside [0, h2h_n_overall]")
    log_list.append(f"[✓] h2h_n_overall ≥ h2h_n_clay  {n}/{n}")

    # 3. smoothed in [0, 1], no NaN
    sr = features["h2h_smoothed_ratio_p1"]
    if sr.isna().any():
        raise AssertionError(f"h2h_smoothed_ratio_p1 has {sr.isna().sum()} NaN")
    if (sr < 0).any() or (sr > 1).any():
        raise AssertionError("h2h_smoothed_ratio_p1 outside [0, 1]")
    log_list.append(f"[✓] smoothed_ratio ∈ [0, 1] no NaN  {n}/{n}")

    # 4. Cold-start invariant: where n_clay = 0, smoothed = α/(α+β)
    cold = features[nc == 0]
    expected_cold_value = alpha / (alpha + beta)
    if not np.allclose(cold["h2h_smoothed_ratio_p1"], expected_cold_value):
        bad = (~np.isclose(cold["h2h_smoothed_ratio_p1"],
                           expected_cold_value)).sum()
        raise AssertionError(
            f"Cold-start invariant violated: {int(bad)} rows with n_clay=0 "
            f"but smoothed != {expected_cold_value}"
        )
    log_list.append(f"[✓] Cold-start (n_clay=0) → smoothed = "
                  f"{expected_cold_value:.4f}  {len(cold)} rows")

    # 5. Distribution audit (printed to log)
    n_dist = _compute_n_distribution(features)
    log_list.append(f"[✓] h2h_n_clay distribution: " +
                  ", ".join(f"{r['bucket']}={r['pct']}%"
                            for _, r in n_dist.iterrows()))

    # 6. Pearson correlation vs clay_elo_diff (warn, not raise)
    elo_corr = None
    if clay_elo_features is not None and not clay_elo_features.empty:
        elo_corr = _compute_elo_correlation(features, clay_elo_features)
        if elo_corr is None:
            log_list.append(f"[⚠] Elo correlation skipped (no eligible rows)")
        else:
            tag = ("[✓]" if abs(elo_corr) < ELO_CORRELATION_WARN_THRESHOLD
                   else "[⚠]")
            interp = _interpret_correlation(elo_corr)
            log_list.append(
                f"{tag} Pearson r(h2h_smoothed_ratio_p1, clay_elo_diff) = "
                f"{elo_corr:+.3f}  → {interp}"
            )

    # 7. 100-row stratified leakage cross-check
    _check_no_leakage(
        master, features, unified, alpha, beta,
        sample_size=leakage_sample_size, rng_seed=rng_seed, log_list=log_list,
    )

    return elo_corr


def _interpret_correlation(r: float) -> str:
    """Human-readable interpretation per locked operational guidance."""
    a = abs(r)
    if a < 0.5:
        return "H2H carries new signal — add directly"
    if a <= 0.7:
        return "partial overlap — gate via ablation (AUC lift > 0.005)"
    return "high redundancy with Elo — likely skip (above 0.7 warn threshold)"


def _compute_elo_correlation(
    features: pd.DataFrame, clay_elo: pd.DataFrame,
) -> float | None:
    """Pearson r of ``h2h_smoothed_ratio_p1`` vs ``clay_elo_diff``,
    restricted to rows with ``h2h_n_clay > 0`` (where H2H carries signal).
    """
    joined = features.merge(
        clay_elo[["match_id", "clay_elo_diff"]], on="match_id", how="inner",
    )
    eligible = joined[(joined["h2h_n_clay"] > 0) &
                       joined["clay_elo_diff"].notna() &
                       joined["h2h_smoothed_ratio_p1"].notna()]
    if len(eligible) < 30:   # need a minimum sample for Pearson to be meaningful
        return None
    r = eligible["h2h_smoothed_ratio_p1"].corr(eligible["clay_elo_diff"])
    return float(r)


def _check_no_leakage(
    master: pd.DataFrame, features: pd.DataFrame, unified: pd.DataFrame,
    alpha: float, beta: float,
    *,
    sample_size: int, rng_seed: int, log_list: list[str],
) -> None:
    """Stratified 100-row leakage cross-check (5 per year × ~18 years).

    For each sampled master row, independently re-derive the 5 numeric
    H2H outputs and assert exact equality. Uses the same unified frame
    but rebuilds the per-pair history list and runs ``_query_pair_history``
    against it.
    """
    f_with_year = features.merge(
        master[["match_id", "match_date", "stable_match_key"]],
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

    # Build full per-pair history dict from unified
    state: dict[tuple[int, int], list] = defaultdict(list)
    p1_ids = unified["p1_id"].astype("Int64").to_numpy()
    p2_ids = unified["p2_id"].astype("Int64").to_numpy()
    ys = unified["y"].astype("Int8").to_numpy()
    dates = pd.to_datetime(unified["match_date"]).to_numpy()
    surfaces = unified["surface_lower"].to_numpy()
    keys = unified["_key"].to_numpy()

    for i in range(len(unified)):
        pair = (int(p1_ids[i]), int(p2_ids[i]))
        state[pair].append(
            (dates[i], 1 if int(ys[i]) == 1 else 0,
             str(surfaces[i]) if surfaces[i] is not None else "",
             keys[i])
        )

    # Build master lookup for sampled rows
    master_lookup = master.set_index("match_id")[
        ["p1_id", "p2_id", "match_date", "stable_match_key"]
    ]

    fails = 0
    failure_msgs: list[str] = []
    for _, row in sampled.iterrows():
        mid = int(row["match_id"])
        m_row = master_lookup.loc[mid]
        p1, p2 = int(m_row["p1_id"]), int(m_row["p2_id"])
        cur_date = pd.Timestamp(m_row["match_date"]).to_datetime64()
        cur_key = str(m_row["stable_match_key"])

        n_clay_e, wins_clay_e, n_ov_e, wins_ov_e = _query_pair_history(
            state.get((p1, p2), []), cur_date, cur_key,
        )
        smoothed_e = (wins_clay_e + alpha) / (n_clay_e + alpha + beta)

        ok = (
            int(row["h2h_n_clay"]) == n_clay_e
            and int(row["h2h_p1_wins"]) == wins_clay_e
            and int(row["h2h_n_overall"]) == n_ov_e
            and int(row["h2h_p1_wins_overall"]) == wins_ov_e
            and abs(float(row["h2h_smoothed_ratio_p1"]) - smoothed_e) < 1e-9
        )
        if not ok:
            fails += 1
            if len(failure_msgs) < 3:
                failure_msgs.append(
                    f"  match_id={mid}: stored=("
                    f"n_clay={int(row['h2h_n_clay'])},"
                    f"wins={int(row['h2h_p1_wins'])},"
                    f"smoothed={float(row['h2h_smoothed_ratio_p1']):.4f},"
                    f"n_ov={int(row['h2h_n_overall'])},"
                    f"wins_ov={int(row['h2h_p1_wins_overall'])}) "
                    f"expected=("
                    f"n_clay={n_clay_e},wins={wins_clay_e},"
                    f"smoothed={smoothed_e:.4f},"
                    f"n_ov={n_ov_e},wins_ov={wins_ov_e})"
                )

    if fails:
        raise AssertionError(
            f"Leakage check failed: {fails} of {len(sampled)} sampled rows "
            f"show H2H discrepancy.\n" + "\n".join(failure_msgs)
        )
    log_list.append(
        f"[✓] Leakage cross-check  {len(sampled)}/{len(sampled)} match (exact)"
    )
