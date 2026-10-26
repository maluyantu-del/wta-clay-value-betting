"""Clay-court Elo feature module.

Computes pre-match clay-Elo for both players in every master match.
Iterates chronologically over (Sackmann seed 2005-2006) ⊕ (master 2007+),
applying tier-graded K-factors. Output is keyed on ``match_id`` for
clean join back to master.

Algorithm
---------
For every clay match in time order::

    elo_p1, elo_p2 = state[p1_id], state[p2_id]      # PRE-match (recorded)
    K = K_FACTORS[tour_level]                         # 32/24/20/16/12/0
    expected_p1 = 1 / (1 + 10**((elo_p2 - elo_p1) / 400))
    actual_p1   = 1 if y == 1 else 0                  # y from id-sort target
    delta = K * (actual_p1 - expected_p1)
    state[p1_id] += delta                             # POST-match
    state[p2_id] -= delta

K-factor exposure
-----------------
``K = 0`` for ITF means **no Elo update** for that match (excluded from
rating signal because of volume — 180K+ ITF rows would swamp main-tour
information). It does NOT mean "skip reading Elo": the players' current
ratings are still observable on the row's other features. ITF rows just
don't perturb anyone's rating after they finish.

Bootstrap window
----------------
Master rows whose match year is < ``BOOTSTRAP_YEAR_CUTOFF`` (default 2009)
get ``clay_elo_bootstrap=True``. Training and backtest layers must filter
on this flag — bootstrap matches participate in Elo state evolution but
should NOT be used for model fitting (Elo state is too unstable in the
first ~2 years of observations). The Sackmann 2005-2006 seed extends
bootstrap further back without showing up in the master output at all.

Concerns separated
------------------
The 2009 Pinnacle-coverage gap is a market/backtest concern handled by
``backtest`` layer (filters ``market_source != 'none'``). This module
only manages the Elo bootstrap window 2007-2008. Don't conflate.

Architecture defense
--------------------
The output features DataFrame carries a ``master_row_count`` column with
the size of the master used. The downstream trainer asserts this matches
``len(current_master)`` to detect stale features (master updated, this
module not re-run). Saves debugging time on data-pipeline desyncs.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from wta_clay.features.io import write_features

log = logging.getLogger(__name__)


# ===========================================================================
# Constants
# ===========================================================================

#: K-factor by canonical tier. 0 = "no update" (ITF excluded by volume).
DEFAULT_K_FACTORS: dict[str, float] = {
    "GS":      32.0,
    "WTA1000": 24.0,
    "WTA500":  20.0,
    "WTA250":  16.0,
    "WTA125":  12.0,
    "ITF":      0.0,
}

INITIAL_ELO: float = 1500.0

#: Master rows with year < this are flagged ``clay_elo_bootstrap=True``.
BOOTSTRAP_YEAR_CUTOFF: int = 2009

#: Round-order convention (R128 first, F last). Matches data layer.
ROUND_ORDER: tuple[str, ...] = ("R128", "R64", "R32", "R16",
                                "QF", "SF", "F", "RR")

#: Output feature columns (joined onto master via ``match_id``).
#: Master-frame metadata (row count, date range) is written to the
#: sidecar JSON, NOT a per-row column — see :mod:`wta_clay.features.io`.
FEATURE_COLUMNS: tuple[str, ...] = (
    "match_id",
    "clay_elo_p1",
    "clay_elo_p2",
    "clay_elo_diff",
    "clay_elo_bootstrap",
)


# ===========================================================================
# Dataclass
# ===========================================================================

@dataclass
class ClayEloResult:
    """Top-level return of :func:`compute_clay_elo`.

    Attributes
    ----------
    features : pd.DataFrame
        Indexed by ``match_id``; columns per :data:`FEATURE_COLUMNS`.
    final_state : pd.DataFrame
        Player-level final ratings: ``player_id, name, n_clay_matches,
        final_elo``. Sorted by ``final_elo`` descending.
    config : dict
        Reproducibility metadata.
    sanity_log : list[str]
        Self-check results (✓/⚠/✗ entries).
    star_trajectories : pd.DataFrame
        Year-end Elo for a small set of star players, shape
        (player, year) → end-of-year Elo. Diagnostic only.
    """
    features: pd.DataFrame
    final_state: pd.DataFrame
    config: dict
    sanity_log: list[str] = field(default_factory=list)
    star_trajectories: pd.DataFrame = field(default_factory=pd.DataFrame)


# ===========================================================================
# Main entry
# ===========================================================================

def compute_clay_elo(
    master: pd.DataFrame,
    sackmann_seed: pd.DataFrame | None = None,
    players: pd.DataFrame | None = None,
    *,
    initial_elo: float = INITIAL_ELO,
    k_factors: dict[str, float] = DEFAULT_K_FACTORS,
    bootstrap_year_cutoff: int = BOOTSTRAP_YEAR_CUTOFF,
    run_sanity_checks: bool = True,
    leakage_sample_size: int = 100,
    rng_seed: int = 42,
    out_path: Path | str | None = None,
) -> ClayEloResult:
    """Compute pre-match clay Elo for every match in master.

    Parameters
    ----------
    master : DataFrame
        Output of :mod:`wta_clay.data.merge`. Required cols:
        ``match_id, match_date, tourney_date, tourney_id, tour_level,
        round_norm, p1_id, p2_id, y``.
    sackmann_seed : DataFrame | None
        Sackmann clay main-draw matches BEFORE the master's earliest year
        (typically 2005-2006). Required cols: ``tourney_id, tourney_name,
        tour_level, match_date (=tourney_date), winner_id, loser_id,
        round``. If None, Elo state starts cold at master's first match.
    players : DataFrame | None
        Sackmann ``wta_players.csv`` with ``player_id, full_name``. Used
        only to enrich ``final_state`` with names for sanity reporting.

    Returns
    -------
    :class:`ClayEloResult`. The ``features`` DataFrame is suitable for
    persistence (parquet) and joining to master by ``match_id``.
    """
    config = {
        "initial_elo": initial_elo,
        "k_factors": dict(k_factors),
        "bootstrap_year_cutoff": bootstrap_year_cutoff,
        "leakage_sample_size": leakage_sample_size,
        "rng_seed": rng_seed,
        "n_master_rows": len(master),
        "n_seed_rows": int(len(sackmann_seed)) if sackmann_seed is not None else 0,
    }
    log.info("compute_clay_elo: master=%d, seed=%d",
             config["n_master_rows"], config["n_seed_rows"])

    if master.empty:
        log.warning("Empty master; returning empty ClayEloResult.")
        return ClayEloResult(
            features=pd.DataFrame(columns=list(FEATURE_COLUMNS)),
            final_state=pd.DataFrame(),
            config=config,
        )

    unified = _build_unified_iteration_frame(master, sackmann_seed)
    state, n_matches, feature_rows = _run_elo_iteration(
        unified, initial_elo, k_factors,
    )
    features = _finalize_features(feature_rows, len(master),
                                   bootstrap_year_cutoff)
    # Set bootstrap flag based on master.match_date (eager — not deferred
    # to sanity-check phase, so it works with run_sanity_checks=False too).
    features = _set_bootstrap_flag(features, master, bootstrap_year_cutoff)
    final_state = _build_final_state(state, n_matches, players)

    star_trajectories = _star_trajectories(unified, initial_elo, k_factors,
                                            players=players)

    sanity_log: list[str] = []
    if run_sanity_checks:
        _run_sanity_checks(
            master, features, final_state, unified, sanity_log,
            bootstrap_year_cutoff=bootstrap_year_cutoff,
            initial_elo=initial_elo,
            k_factors=k_factors,
            leakage_sample_size=leakage_sample_size,
            rng_seed=rng_seed,
        )

    if out_path is not None:
        write_features(features, master, path=out_path,
                       feature_module="clay_elo", config=config)

    return ClayEloResult(
        features=features,
        final_state=final_state,
        config=config,
        sanity_log=sanity_log,
        star_trajectories=star_trajectories,
    )


# ===========================================================================
# Iteration frame construction
# ===========================================================================

def _build_unified_iteration_frame(
    master: pd.DataFrame,
    sackmann_seed: pd.DataFrame | None,
) -> pd.DataFrame:
    """Combine Sackmann seed + master into a unified chronological frame.

    Both sources are converted to a common schema with id-sorted
    ``p1_id < p2_id`` and ``y ∈ {0, 1}`` so the iteration loop is uniform.
    Seed rows are marked ``_is_master=False``; only master rows produce
    feature output.
    """
    frames: list[pd.DataFrame] = []
    if sackmann_seed is not None and not sackmann_seed.empty:
        seed = _seed_to_pair_form(sackmann_seed)
        seed["_is_master"] = False
        seed["match_id"] = pd.NA
        frames.append(seed)
    m = master[[
        "match_id", "match_date", "tourney_date", "tourney_id",
        "tour_level", "round_norm", "p1_id", "p2_id", "y",
    ]].copy()
    m["_is_master"] = True
    frames.append(m)
    unified = pd.concat(frames, ignore_index=True, sort=False)

    # Sort: tourney_date ascending → tourney_id → round (R128 first) →
    # match_id (deterministic tiebreak within a round).
    # Map round to integer for sortable order.
    round_idx = {r: i for i, r in enumerate(ROUND_ORDER)}
    unified["_rd_idx"] = unified["round_norm"].astype("string").map(round_idx)
    # Unknown rounds → push to end (high index).
    unified["_rd_idx"] = unified["_rd_idx"].fillna(99).astype(int)
    # match_id is NA for seed rows; fill with -1 so it sorts before any
    # master match_id (and pandas can sort the column without dtype hiccups).
    unified["_mid_sort"] = unified["match_id"].fillna(-1).astype("int64")
    unified = unified.sort_values(
        ["tourney_date", "tourney_id", "_rd_idx", "_mid_sort"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)
    return unified.drop(columns=["_rd_idx", "_mid_sort"])


def _seed_to_pair_form(seed: pd.DataFrame) -> pd.DataFrame:
    """Convert Sackmann winner/loser-form to id-sorted p1/p2 form."""
    s = seed.copy()
    w = pd.to_numeric(s["winner_id"], errors="coerce").astype("Int64")
    l = pd.to_numeric(s["loser_id"], errors="coerce").astype("Int64")
    s = s.dropna(subset=["winner_id", "loser_id"]).copy()
    p1_is_winner = (w < l).fillna(False).reindex(s.index)
    s["p1_id"] = w.where(p1_is_winner, l).reindex(s.index).astype("Int64")
    s["p2_id"] = l.where(p1_is_winner, w).reindex(s.index).astype("Int64")
    s["y"] = p1_is_winner.reindex(s.index).astype("Int8")
    # Use match_date (parsed Timestamp) as tourney_date — Sackmann's raw
    # `tourney_date` column is int YYYYMMDD which we don't want here. The
    # data layer's match_date is the parsed version of tourney_date.
    s["tourney_date"] = pd.to_datetime(s["match_date"], errors="coerce")
    if "round_norm" not in s.columns and "round" in s.columns:
        s["round_norm"] = s["round"].astype("string")
    return s[[
        "match_date", "tourney_date", "tourney_id", "tour_level",
        "round_norm", "p1_id", "p2_id", "y",
    ]]


# ===========================================================================
# Elo iteration
# ===========================================================================

def _run_elo_iteration(
    unified: pd.DataFrame,
    initial_elo: float,
    k_factors: dict[str, float],
) -> tuple[dict[int, float], dict[int, int], list[dict]]:
    """Iterate matches, update state, capture pre-match Elo for master rows.

    The pre-match capture happens BEFORE any state update — this is the
    leakage-prevention invariant (Layer 1 in the design doc).
    """
    state: dict[int, float] = {}
    n_matches: dict[int, int] = {}
    feature_rows: list[dict] = []

    # Tuple iteration is faster than itertuples here for hot loop
    p1_ids = unified["p1_id"].astype("Int64").to_numpy()
    p2_ids = unified["p2_id"].astype("Int64").to_numpy()
    ys = unified["y"].astype("Int8").to_numpy()
    levels = unified["tour_level"].astype("string").to_numpy()
    is_master = unified["_is_master"].to_numpy()
    match_ids = unified["match_id"].to_numpy()

    for i in range(len(unified)):
        p1, p2 = int(p1_ids[i]), int(p2_ids[i])
        y = int(ys[i])
        elo_p1 = state.get(p1, initial_elo)
        elo_p2 = state.get(p2, initial_elo)

        if is_master[i]:
            feature_rows.append({
                "match_id": int(match_ids[i]),
                "clay_elo_p1": elo_p1,
                "clay_elo_p2": elo_p2,
                "clay_elo_diff": elo_p1 - elo_p2,
            })

        k = float(k_factors.get(str(levels[i]), 16.0))
        if k > 0:
            expected_p1 = 1.0 / (1.0 + 10.0 ** ((elo_p2 - elo_p1) / 400.0))
            actual_p1 = 1.0 if y == 1 else 0.0
            delta = k * (actual_p1 - expected_p1)
            state[p1] = elo_p1 + delta
            state[p2] = elo_p2 - delta
        else:
            # K=0 still establishes initial state for new players.
            state.setdefault(p1, elo_p1)
            state.setdefault(p2, elo_p2)

        n_matches[p1] = n_matches.get(p1, 0) + 1
        n_matches[p2] = n_matches.get(p2, 0) + 1

    return state, n_matches, feature_rows


def _finalize_features(
    feature_rows: list[dict],
    master_row_count: int,
    bootstrap_year_cutoff: int,
) -> pd.DataFrame:
    """Convert feature row records to typed DataFrame; add bootstrap flag."""
    if not feature_rows:
        return pd.DataFrame(columns=list(FEATURE_COLUMNS))
    f = pd.DataFrame(feature_rows)
    f["match_id"] = f["match_id"].astype("Int64")
    f["clay_elo_p1"] = f["clay_elo_p1"].astype("float64")
    f["clay_elo_p2"] = f["clay_elo_p2"].astype("float64")
    f["clay_elo_diff"] = f["clay_elo_diff"].astype("float64")
    # Need master.match_date for bootstrap flag — caller passes year cutoff;
    # but match_id alone doesn't tell us year. We carry year through the
    # iteration: redo by joining? No — already separate. We'll attach
    # bootstrap by year of match_date later via an inner join in the caller.
    # For now, we don't have date info on f, so set bootstrap=False and
    # the caller will overwrite. Simpler: do bootstrap via a parallel pass.
    return _attach_bootstrap_flag(f, bootstrap_year_cutoff, master_row_count)


def _attach_bootstrap_flag(
    f: pd.DataFrame, cutoff: int, master_row_count: int,
) -> pd.DataFrame:
    """Attach bootstrap placeholder column; year filled by _set_bootstrap_flag."""
    f = f.copy()
    f["clay_elo_bootstrap"] = False  # placeholder; overwritten by _set_bootstrap_flag
    return f[list(FEATURE_COLUMNS)]


def _set_bootstrap_flag(
    features: pd.DataFrame,
    master: pd.DataFrame,
    bootstrap_year_cutoff: int,
) -> pd.DataFrame:
    """Compute ``clay_elo_bootstrap`` from master.match_date year."""
    if features.empty:
        return features
    f = features.merge(
        master[["match_id", "match_date"]], on="match_id", how="left",
    )
    bootstrap = (f["match_date"].dt.year < bootstrap_year_cutoff).fillna(False)
    f = f.drop(columns=["match_date"])
    f["clay_elo_bootstrap"] = bootstrap.astype(bool)
    return f[list(FEATURE_COLUMNS)]


# ===========================================================================
# Final-state DataFrame
# ===========================================================================

def _build_final_state(
    state: dict[int, float],
    n_matches: dict[int, int],
    players: pd.DataFrame | None,
) -> pd.DataFrame:
    """Build sorted player-level final-Elo DataFrame."""
    rows = [
        {"player_id": pid, "final_elo": elo,
         "n_clay_matches": n_matches[pid]}
        for pid, elo in state.items()
    ]
    df = pd.DataFrame(rows)
    if players is not None and len(df):
        plr = players[["player_id", "full_name"]].copy() \
            if "full_name" in players.columns else \
            players.assign(full_name=lambda d: (
                d.get("name_first", "").astype("string").fillna("") + " " +
                d.get("name_last", "").astype("string").fillna("")
            ).str.strip())[["player_id", "full_name"]]
        df = df.merge(plr, on="player_id", how="left")
    return df.sort_values("final_elo", ascending=False).reset_index(drop=True)


# ===========================================================================
# Star trajectories (diagnostic)
# ===========================================================================

#: Names tested for end-of-year Elo plausibility.
_STAR_PATTERNS: tuple[str, ...] = (
    "Iga Swiatek", "Simona Halep", "Aryna Sabalenka",
    "Sara Errani", "Justine Henin",
)


def _star_trajectories(
    unified: pd.DataFrame,
    initial_elo: float,
    k_factors: dict[str, float],
    players: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Re-run iteration capturing year-end Elo for star players.

    Returns a DataFrame with rows = star player names, cols = years.
    """
    if players is None or "full_name" not in players.columns:
        return pd.DataFrame()
    star_ids = {}
    for pat in _STAR_PATTERNS:
        hit = players[players["full_name"].astype("string")
                      .str.contains(pat, case=False, na=False, regex=False)]
        if len(hit):
            pid = int(hit.iloc[0]["player_id"])
            star_ids[pat] = pid

    if not star_ids:
        return pd.DataFrame()

    # Re-run minimal Elo capturing year-end snapshots
    state: dict[int, float] = {}
    snapshots: list[dict] = []  # year, player_id, elo

    p1_ids = unified["p1_id"].astype("Int64").to_numpy()
    p2_ids = unified["p2_id"].astype("Int64").to_numpy()
    ys = unified["y"].astype("Int8").to_numpy()
    levels = unified["tour_level"].astype("string").to_numpy()
    dates = pd.to_datetime(unified["match_date"]).to_numpy()
    star_set = set(star_ids.values())

    last_year = None
    for i in range(len(unified)):
        cur_year = pd.Timestamp(dates[i]).year
        if last_year is not None and cur_year != last_year:
            # year boundary — snapshot star Elo for last_year
            for pid in star_set:
                if pid in state:
                    snapshots.append({
                        "year": last_year, "player_id": pid,
                        "elo": state[pid],
                    })
        last_year = cur_year

        p1, p2 = int(p1_ids[i]), int(p2_ids[i])
        y = int(ys[i])
        elo_p1 = state.get(p1, initial_elo)
        elo_p2 = state.get(p2, initial_elo)
        k = float(k_factors.get(str(levels[i]), 16.0))
        if k > 0:
            expected = 1.0 / (1.0 + 10.0 ** ((elo_p2 - elo_p1) / 400.0))
            actual = 1.0 if y == 1 else 0.0
            d = k * (actual - expected)
            state[p1] = elo_p1 + d
            state[p2] = elo_p2 - d
        else:
            state.setdefault(p1, elo_p1)
            state.setdefault(p2, elo_p2)
    # Final-year snapshot
    if last_year is not None:
        for pid in star_set:
            if pid in state:
                snapshots.append({"year": last_year, "player_id": pid,
                                  "elo": state[pid]})

    snap_df = pd.DataFrame(snapshots)
    if snap_df.empty:
        return snap_df

    name_lookup = {pid: pat for pat, pid in star_ids.items()}
    snap_df["player_name"] = snap_df["player_id"].map(name_lookup)
    pivot = snap_df.pivot_table(
        index="player_name", columns="year", values="elo", aggfunc="last",
    ).round(0).astype("Int64")
    return pivot


# ===========================================================================
# Self-checks
# ===========================================================================

def _run_sanity_checks(
    master: pd.DataFrame,
    features: pd.DataFrame,
    final_state: pd.DataFrame,
    unified: pd.DataFrame,
    log_list: list[str],
    *,
    bootstrap_year_cutoff: int,
    initial_elo: float,
    k_factors: dict[str, float],
    leakage_sample_size: int,
    rng_seed: int,
) -> None:
    """Run all 8 sanity checks; raise on hard failures, warn on soft."""
    n = len(features)

    # Check 1: All Elo values finite (raise)
    elo_cols = ["clay_elo_p1", "clay_elo_p2", "clay_elo_diff"]
    for col in elo_cols:
        if features[col].isna().any() or (~np.isfinite(features[col])).any():
            n_bad = features[col].isna().sum() + (~np.isfinite(features[col])).sum()
            raise AssertionError(f"{col} has {n_bad} NaN/inf values")
    log_list.append(f"[✓] All Elo values finite  {n}/{n}")

    # Check 2: Top-50 (by final_elo, n>=20) in [1700, 2200] (warn)
    active = final_state[final_state["n_clay_matches"] >= 20]
    if len(active) >= 50:
        top50 = active.head(50)
        lo, hi = top50["final_elo"].min(), top50["final_elo"].max()
        # Soft check — "in range" means: max ≥ 1700 (top has signal),
        # max ≤ 2200 (no overflow), min ≥ 1500 (top-50 is above mean).
        if 1700 <= hi <= 2200 and lo >= 1500:
            log_list.append(f"[✓] Top-50 final Elo range [{lo:.0f}, {hi:.0f}] "
                          f"⊆ [1500, 2200]")
        else:
            log_list.append(f"[⚠] Top-50 final Elo range [{lo:.0f}, {hi:.0f}] "
                          f"OUTSIDE soft band [1500, 2200]")
    else:
        log_list.append(f"[⚠] Only {len(active)} active players with n≥20")

    # Check 3: Bottom-quartile in [1200, 1500] (warn)
    if len(active) >= 100:
        q25 = active["final_elo"].quantile(0.25)
        if 1200 <= q25 <= 1500:
            log_list.append(f"[✓] Active-player Q25 = {q25:.0f} ⊆ [1200, 1500]")
        else:
            log_list.append(f"[⚠] Active-player Q25 = {q25:.0f} outside [1200, 1500]")

    # Check 4: No overflow (raise)
    if len(active):
        max_elo = active["final_elo"].max()
        min_elo = active["final_elo"].min()
        if max_elo > 2500:
            raise AssertionError(f"Elo overflow: max={max_elo:.1f} > 2500")
        if min_elo < 800:
            raise AssertionError(f"Elo underflow: min={min_elo:.1f} < 800")
        log_list.append(f"[✓] No overflow  active range [{min_elo:.0f}, {max_elo:.0f}]")

    # Check 5: Mean Elo close to initial (warn) — closed system zero-sum invariant
    mean_elo = final_state["final_elo"].mean()
    drift = abs(mean_elo - initial_elo)
    if drift < 20:
        log_list.append(f"[✓] Mean final Elo = {mean_elo:.1f} "
                      f"(drift {drift:.1f} < 20)")
    else:
        log_list.append(f"[⚠] Mean final Elo = {mean_elo:.1f} "
                      f"(drift {drift:.1f} ≥ 20)")

    # Check 6: Re-derivation cross-check on N random matches (raise)
    _check_no_leakage_via_resampling(
        features, unified, master, initial_elo, k_factors,
        sample_size=leakage_sample_size, rng_seed=rng_seed, log_list=log_list,
    )

    # Check 7: Bootstrap row count = master rows with year < cutoff (raise)
    expected_bootstrap = (master["match_date"].dt.year < bootstrap_year_cutoff).sum()
    actual_bootstrap = int(features["clay_elo_bootstrap"].sum())
    if actual_bootstrap != expected_bootstrap:
        raise AssertionError(
            f"Bootstrap count mismatch: features={actual_bootstrap}, "
            f"master year<{bootstrap_year_cutoff}={expected_bootstrap}"
        )
    log_list.append(f"[✓] Bootstrap rows = master year<{bootstrap_year_cutoff}  "
                  f"{actual_bootstrap}/{actual_bootstrap}")

    # Check 8: Total updates = 2 × n_matches in unified (raise)
    expected_total = 2 * len(unified)
    actual_total = int(final_state["n_clay_matches"].sum())
    if actual_total != expected_total:
        raise AssertionError(
            f"Update count mismatch: total={actual_total}, "
            f"expected 2 × {len(unified)} = {expected_total}"
        )
    log_list.append(f"[✓] Total updates = 2 × n_matches  {actual_total}={expected_total}")

    # Additional: features.match_id.min() ≥ master.match_id.min() (raise)
    if not features.empty and not master.empty:
        f_min = int(features["match_id"].min())
        m_min = int(master["match_id"].min())
        if f_min < m_min:
            raise AssertionError(
                f"features.match_id.min()={f_min} < master.match_id.min()={m_min}; "
                f"seed leaked into features"
            )
        log_list.append(f"[✓] features.match_id.min()={f_min} ≥ "
                      f"master.match_id.min()={m_min}  (seed isolated)")


def _check_no_leakage_via_resampling(
    features: pd.DataFrame,
    unified: pd.DataFrame,
    master: pd.DataFrame,
    initial_elo: float,
    k_factors: dict[str, float],
    *,
    sample_size: int,
    rng_seed: int,
    log_list: list[str],
) -> None:
    """Sample matches uniformly across years; re-derive pre-match Elo and
    assert ``abs(diff) < 1e-9`` against the recorded value.

    Sample is stratified by year so 2024 doesn't dominate (it has 4-5×
    the rows of 2007). Within each year we sample proportionally.
    """
    f_with_year = features.merge(
        master[["match_id", "match_date"]], on="match_id", how="left",
    )
    f_with_year["_year"] = f_with_year["match_date"].dt.year
    rng = np.random.default_rng(rng_seed)

    # Stratified by year, sampling uniformly
    yrs = sorted(f_with_year["_year"].dropna().unique().tolist())
    per_year = max(1, sample_size // len(yrs))
    sampled = pd.concat([
        f_with_year[f_with_year["_year"] == yr].sample(
            min(per_year, (f_with_year["_year"] == yr).sum()),
            random_state=int(rng.integers(0, 1_000_000)),
        )
        for yr in yrs
    ], ignore_index=True)

    # Re-derive: build a position lookup in unified
    unified_sorted = unified.reset_index(drop=True)
    sample_match_ids = set(sampled["match_id"].astype(int).tolist())

    state: dict[int, float] = {}
    recorded_at_match: dict[int, tuple[float, float]] = {}

    p1_ids = unified_sorted["p1_id"].astype("Int64").to_numpy()
    p2_ids = unified_sorted["p2_id"].astype("Int64").to_numpy()
    ys = unified_sorted["y"].astype("Int8").to_numpy()
    levels = unified_sorted["tour_level"].astype("string").to_numpy()
    is_master = unified_sorted["_is_master"].to_numpy()
    match_ids = unified_sorted["match_id"].to_numpy()

    for i in range(len(unified_sorted)):
        p1, p2 = int(p1_ids[i]), int(p2_ids[i])
        y = int(ys[i])
        elo_p1 = state.get(p1, initial_elo)
        elo_p2 = state.get(p2, initial_elo)

        if is_master[i] and int(match_ids[i]) in sample_match_ids:
            recorded_at_match[int(match_ids[i])] = (elo_p1, elo_p2)

        k = float(k_factors.get(str(levels[i]), 16.0))
        if k > 0:
            expected = 1.0 / (1.0 + 10.0 ** ((elo_p2 - elo_p1) / 400.0))
            actual = 1.0 if y == 1 else 0.0
            d = k * (actual - expected)
            state[p1] = elo_p1 + d
            state[p2] = elo_p2 - d
        else:
            state.setdefault(p1, elo_p1)
            state.setdefault(p2, elo_p2)

    # Compare
    fails = 0
    for _, row in sampled.iterrows():
        mid = int(row["match_id"])
        recorded_p1, recorded_p2 = recorded_at_match.get(mid, (np.nan, np.nan))
        actual_p1 = float(row["clay_elo_p1"])
        actual_p2 = float(row["clay_elo_p2"])
        if abs(recorded_p1 - actual_p1) > 1e-9 or \
           abs(recorded_p2 - actual_p2) > 1e-9:
            fails += 1
    if fails:
        raise AssertionError(
            f"Leakage check: {fails} of {len(sampled)} sampled matches show "
            f"Elo discrepancy ≥ 1e-9 between recorded and re-derivation"
        )
    log_list.append(f"[✓] Leakage cross-check  {len(sampled)}/{len(sampled)} "
                  f"match (abs diff < 1e-9)")
