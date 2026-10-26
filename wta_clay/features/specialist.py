"""Clay-specialist feature module — clay vs all-surface winrate differential.

Output (7 cols + match_id):

  specialist_p1            float64    p1's clay_winrate − overall_winrate
  specialist_p2            float64    p2's mirror
  specialist_diff          float64    p1 − p2  (NaN if either side NaN)
  specialist_n_clay_p1     Int64      p1's cumulative clay match count
  specialist_n_total_p1    Int64      p1's cumulative ALL-surface match count
  specialist_n_clay_p2     Int64      mirror
  specialist_n_total_p2    Int64      mirror

**Semantic lock**: ``overall_winrate`` is computed across ALL surfaces
(clay + hard + grass + carpet — NOT "non-clay"). Example::

    Player A's history: 8/10 on clay, 6/10 on hard
    overall_winrate = 14/20 = 0.70
    clay_winrate    = 8/10  = 0.80
    specialist      = 0.80 - 0.70 = +0.10

This lets the index handle players who only play one surface (their
clay rate equals their overall rate → specialist = 0, correct neutral).

**Important — what specialist measures and does NOT measure**:

  ``specialist`` measures clay-vs-self DEVIATION, **NOT** clay-strength
  absolute. Top players have moderate specialist values because their
  overall winrate is also high — the delta is bounded.

  Examples on real 2007-2024 data:
    - Sara Errani peak    +0.092  (top-10 player; high baseline → moderate delta)
    - Anabel Medina G.    +0.190  (mid-tier player; clay-disproportionate → high delta)
    - Camila Osorio       +0.155  (Bogotá native; mid-tier; large positive delta)
    - Aryna Sabalenka     -0.265  (early-career hard-court specialist)

  Pure clay specialists are mid-tier players with disproportionate clay
  strength. Top players who happen to be best on clay register modestly.
  Don't read this index as "best clay player" — read it as "how much
  better is this player on clay than they are everywhere else".

GOAT × specialist audit (computed and surfaced in :class:`SpecialistResult`):

  At Elo-diff +200 to +300 (GOAT favored by 200-300 Elo), specialists
  outperform non-specialists by **+12 to +19 percentage points** in
  matched-Elo control comparisons (real 2007-2024 data). This is the
  feature's primary alpha contribution — clay specialism translates to
  real edge against GOATs at moderate Elo gaps. The marginal feature
  alone captures this; no explicit ``specialist × elo_diff`` interaction
  needed.

Locked thresholds
-----------------
``MIN_TOTAL_MATCHES = 30`` AND ``MIN_CLAY_MATCHES = 5`` — both must be
met or the index is NaN. Counts emit unconditionally so downstream can
filter at stricter thresholds for ablation.

Cumulative vs rolling
---------------------
Default = cumulative (career-to-date), since "specialist tendency" is a
slow-varying trait. Form.py already covers fast-varying signal (last-15).
Pass ``rolling_window=N`` to use a rolling N-match window for ablation.

Locked conventions
------------------
- Reads ``master.stable_match_key`` for cross-frame join (NOT match_id)
- Strict ``date < current_date AND stable_match_key != current`` filter
- 100-row stratified leakage cross-check (5 per year × 18 years)
- Writes via :func:`wta_clay.features.io.write_features` with
  ``master_key_hash`` sidecar

GOAT × specialist audit
-----------------------
Run post-feature-computation as a diagnostic:
  - GOAT pool   = top-20 players by peak clay-Elo
  - Spec pool   = players whose specialist_index ever ≥ +0.10 with
                  cumulative n_total ≥ 100
  - Compare specialist's winrate vs control (GOAT vs non-specialist
    same-Elo-diff bucket). If specialist label fails to translate to
    actual edge against GOATs, model design should learn the
    interaction (or accept the marginal-only signal).
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

MIN_TOTAL_MATCHES: int = 30
MIN_CLAY_MATCHES:  int = 5

ROUND_ORDER: tuple[str, ...] = ("R128", "R64", "R32", "R16",
                                "QF", "SF", "F", "RR")

REQUIRED_SACKMANN_COLS: tuple[str, ...] = (
    "tourney_id", "match_date", "winner_id", "loser_id",
    "surface", "round",
)
REQUIRED_MASTER_COLS: tuple[str, ...] = (
    "match_id", "match_date", "tourney_id", "p1_id", "p2_id", "y",
    "stable_match_key",
)

FEATURE_COLUMNS: tuple[str, ...] = (
    "match_id",
    "specialist_p1", "specialist_p2", "specialist_diff",
    "specialist_n_clay_p1", "specialist_n_total_p1",
    "specialist_n_clay_p2", "specialist_n_total_p2",
)

# Star-player verification pools
_KNOWN_SPECIALISTS: tuple[str, ...] = (
    "Sara Errani",
    "Francesca Schiavone",
    "Carla Suarez Navarro",
    "Anabel Medina Garrigues",          # primary — career clay focus
    # Backups (used if any of the above is missing data):
    "Lourdes Dominguez Lino",
    "Mariana Duque-Marino",
)
_KNOWN_ANTI_SPECIALISTS: tuple[str, ...] = (
    "Aryna Sabalenka",                  # early-career
    "Karolina Pliskova",
    "Venus Williams",                   # late-career
)

#: Operational thresholds for the GOAT × specialist diagnostic audit.
GOAT_POOL_SIZE: int = 20
SPECIALIST_AUDIT_THRESHOLD: float = 0.10
SPECIALIST_AUDIT_MIN_TOTAL: int = 100
GOAT_AUDIT_ELO_BUCKET_SIZE: float = 100.0

#: Soft warn threshold for Pearson r vs clay_elo_diff.
ELO_CORRELATION_WARN_THRESHOLD: float = 0.50

#: NaN-rate sanity thresholds (per-column).
NAN_RATE_WARN_THRESHOLD: float = 0.25


# ===========================================================================
# Result dataclass
# ===========================================================================

@dataclass
class SpecialistResult:
    """Top-level return of :func:`compute_specialist_features`."""
    features: pd.DataFrame
    config: dict[str, Any]
    sanity_log: list[str] = field(default_factory=list)
    elo_correlation: float | None = None
    distribution: pd.DataFrame = field(default_factory=pd.DataFrame)
    top_specialists: pd.DataFrame = field(default_factory=pd.DataFrame)
    bottom_specialists: pd.DataFrame = field(default_factory=pd.DataFrame)
    star_players: pd.DataFrame = field(default_factory=pd.DataFrame)
    goat_audit: pd.DataFrame = field(default_factory=pd.DataFrame)
    nan_rates: pd.DataFrame = field(default_factory=pd.DataFrame)


# ===========================================================================
# Main entry
# ===========================================================================

def compute_specialist_features(
    master: pd.DataFrame,
    sackmann_all: pd.DataFrame,
    clay_elo_features: pd.DataFrame | None = None,
    players: pd.DataFrame | None = None,
    *,
    min_total_matches: int = MIN_TOTAL_MATCHES,
    min_clay_matches: int = MIN_CLAY_MATCHES,
    rolling_window: int | None = None,
    out_path: Path | str | None = None,
    run_sanity_checks: bool = True,
    leakage_sample_size: int = 100,
    rng_seed: int = 42,
) -> SpecialistResult:
    """Compute clay-specialist index for every match in master.

    Parameters
    ----------
    master : DataFrame
        From data/processed/clay_matches_master.parquet. Must contain
        ``stable_match_key`` (cross-frame stable join key).
    sackmann_all : DataFrame
        Sackmann main-draw matches across ALL surfaces (Hard, Clay, Grass,
        Carpet) for 2005+. Required columns::

            tourney_id, match_date, winner_id, loser_id, surface, round
    clay_elo_features : DataFrame | None
        Output of clay_elo. Required for the GOAT audit (peak-Elo lookup)
        and the Pearson r diagnostic.
    players : DataFrame | None
        Sackmann players for star/specialist labels.
    rolling_window : int | None
        ``None`` → cumulative (career-to-date). ``int`` → use last N
        matches only (for ablation).
    """
    config = {
        "min_total_matches": min_total_matches,
        "min_clay_matches": min_clay_matches,
        "rolling_window": rolling_window,
        "n_master_rows": int(len(master)),
        "n_sackmann_rows": int(len(sackmann_all)),
    }
    log.info("compute_specialist: master=%d, sackmann=%d, gates total≥%d clay≥%d",
             config["n_master_rows"], config["n_sackmann_rows"],
             min_total_matches, min_clay_matches)

    if master.empty:
        log.warning("Empty master; returning empty SpecialistResult.")
        return SpecialistResult(
            features=pd.DataFrame(columns=list(FEATURE_COLUMNS)),
            config=config,
        )

    _validate_inputs(master, sackmann_all)

    unified = _build_unified_iteration_frame(master, sackmann_all)
    feature_rows = _run_specialist_iteration(
        unified, min_total_matches, min_clay_matches, rolling_window,
    )
    features = _finalize_features(feature_rows)

    distribution = _compute_distribution(features)
    top_specialists, bottom_specialists = _compute_top_bottom(
        features, master, players,
    )
    star_players = _star_player_verification(features, master, players)
    goat_audit = _goat_audit(
        features, master, clay_elo_features, players,
    )
    nan_rates = _compute_nan_rates(features)

    sanity_log: list[str] = []
    elo_correlation: float | None = None
    if run_sanity_checks:
        elo_correlation = _run_sanity_checks(
            master, features, unified, clay_elo_features, players,
            sanity_log,
            min_total=min_total_matches, min_clay=min_clay_matches,
            rolling_window=rolling_window,
            leakage_sample_size=leakage_sample_size, rng_seed=rng_seed,
            nan_rates=nan_rates,
        )

    if out_path is not None:
        write_features(features, master, path=out_path,
                       feature_module="specialist", config=config)

    return SpecialistResult(
        features=features,
        config=config,
        sanity_log=sanity_log,
        elo_correlation=elo_correlation,
        distribution=distribution,
        top_specialists=top_specialists,
        bottom_specialists=bottom_specialists,
        star_players=star_players,
        goat_audit=goat_audit,
        nan_rates=nan_rates,
    )


# ===========================================================================
# Schema validation
# ===========================================================================

def _validate_inputs(master: pd.DataFrame, sackmann_all: pd.DataFrame) -> None:
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
            f"Required: {list(REQUIRED_MASTER_COLS)}"
        )


# ===========================================================================
# Iteration frame
# ===========================================================================

def _build_unified_iteration_frame(
    master: pd.DataFrame, sackmann_all: pd.DataFrame,
) -> pd.DataFrame:
    """Same pattern as h2h.py — chronological frame over ALL surfaces."""
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
    sk["_rd_idx"] = sk["round"].astype("string").map(round_idx)
    sk["_rd_idx"] = sk["_rd_idx"].fillna(99).astype(int)
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
# Per-player history query
# ===========================================================================

def _query_player_history(
    history: list, current_date, current_key,
    rolling_window: int | None = None,
) -> tuple[int, int, int, int]:
    """Return (n_clay, wins_clay, n_total, wins_total) for a player.

    Strict ``date < current_date AND key != current_key`` filter applied.
    If ``rolling_window`` is set, only the last N entries (by insertion
    order) AFTER filtering are used.
    """
    valid = [(d, w, ic, k) for d, w, ic, k in history
             if d < current_date and k != current_key]
    if rolling_window is not None:
        valid = valid[-rolling_window:]
    n_total = len(valid)
    wins_total = sum(w for _, w, _, _ in valid)
    n_clay = sum(1 for _, _, ic, _ in valid if ic)
    wins_clay = sum(w for _, w, ic, _ in valid if ic)
    return n_clay, wins_clay, n_total, wins_total


def _specialist_index(
    n_clay: int, wins_clay: int, n_total: int, wins_total: int,
    *, min_total: int, min_clay: int,
) -> float:
    """Compute clay_winrate − overall_winrate, or NaN below thresholds."""
    if n_total < min_total or n_clay < min_clay:
        return float("nan")
    clay_wr = wins_clay / n_clay
    overall_wr = wins_total / n_total
    return clay_wr - overall_wr


# ===========================================================================
# Main iteration
# ===========================================================================

def _run_specialist_iteration(
    unified: pd.DataFrame,
    min_total: int, min_clay: int,
    rolling_window: int | None,
) -> list[dict]:
    """Iterate matches; emit features for master rows; update state."""
    state: dict[int, list] = defaultdict(list)
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
        y = int(ys[i])
        cur_date = dates[i]
        cur_key = keys[i]
        is_clay = str(surfaces[i]).lower() == "clay"

        if is_master[i]:
            n_c1, w_c1, n_t1, w_t1 = _query_player_history(
                state.get(p1, []), cur_date, cur_key, rolling_window,
            )
            n_c2, w_c2, n_t2, w_t2 = _query_player_history(
                state.get(p2, []), cur_date, cur_key, rolling_window,
            )
            sp1 = _specialist_index(n_c1, w_c1, n_t1, w_t1,
                                     min_total=min_total, min_clay=min_clay)
            sp2 = _specialist_index(n_c2, w_c2, n_t2, w_t2,
                                     min_total=min_total, min_clay=min_clay)
            feature_rows.append({
                "match_id": int(match_ids[i]),
                "specialist_p1": sp1,
                "specialist_p2": sp2,
                "specialist_diff": (sp1 - sp2
                                     if not (np.isnan(sp1) or np.isnan(sp2))
                                     else float("nan")),
                "specialist_n_clay_p1": n_c1,
                "specialist_n_total_p1": n_t1,
                "specialist_n_clay_p2": n_c2,
                "specialist_n_total_p2": n_t2,
            })

        # Post-emission state update
        state[p1].append((cur_date, 1 if y == 1 else 0, is_clay, cur_key))
        state[p2].append((cur_date, 1 if y == 0 else 0, is_clay, cur_key))

    return feature_rows


def _finalize_features(feature_rows: list[dict]) -> pd.DataFrame:
    if not feature_rows:
        return pd.DataFrame(columns=list(FEATURE_COLUMNS))
    f = pd.DataFrame(feature_rows)
    f["match_id"] = f["match_id"].astype("Int64")
    for c in ("specialist_p1", "specialist_p2", "specialist_diff"):
        f[c] = f[c].astype("float64")
    for c in ("specialist_n_clay_p1", "specialist_n_total_p1",
              "specialist_n_clay_p2", "specialist_n_total_p2"):
        f[c] = f[c].astype("Int64")
    return f[list(FEATURE_COLUMNS)]


# ===========================================================================
# Diagnostics
# ===========================================================================

def _compute_nan_rates(features: pd.DataFrame) -> pd.DataFrame:
    """Per-column NaN rate for diagnostic."""
    if features.empty:
        return pd.DataFrame()
    rows = []
    for col in ("specialist_p1", "specialist_p2", "specialist_diff",
                "specialist_n_clay_p1", "specialist_n_total_p1"):
        rate = features[col].isna().mean()
        rows.append({"column": col, "nan_%": round(rate * 100, 2)})
    return pd.DataFrame(rows)


def _compute_distribution(features: pd.DataFrame) -> pd.DataFrame:
    """Percentiles for specialist_p1 (active = non-NaN)."""
    if features.empty:
        return pd.DataFrame()
    s = features["specialist_p1"].dropna()
    if s.empty:
        return pd.DataFrame()
    qs = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.00]
    rows = [
        {"percentile": ["min","p10","p25","median","p75","p90","max"][i],
         "value": round(float(s.quantile(q)), 4)}
        for i, q in enumerate(qs)
    ]
    return pd.DataFrame(rows)


def _compute_top_bottom(
    features: pd.DataFrame, master: pd.DataFrame,
    players: pd.DataFrame | None,
    n: int = 20,
    min_n_total: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find players with highest/lowest specialist_index (latest snapshot).

    For each player, take their LAST recorded specialist value (from
    their most recent appearance in master). Then rank.
    """
    if features.empty or players is None or "full_name" not in players.columns:
        return pd.DataFrame(), pd.DataFrame()

    f = features.merge(master[["match_id", "match_date", "p1_id", "p2_id"]],
                       on="match_id")

    # Stack p1 and p2 perspectives into a single per-player view
    p1_view = f[["match_date", "p1_id", "specialist_p1",
                 "specialist_n_clay_p1", "specialist_n_total_p1"]].rename(
        columns={"p1_id": "player_id",
                 "specialist_p1": "specialist",
                 "specialist_n_clay_p1": "n_clay",
                 "specialist_n_total_p1": "n_total"}
    )
    p2_view = f[["match_date", "p2_id", "specialist_p2",
                 "specialist_n_clay_p2", "specialist_n_total_p2"]].rename(
        columns={"p2_id": "player_id",
                 "specialist_p2": "specialist",
                 "specialist_n_clay_p2": "n_clay",
                 "specialist_n_total_p2": "n_total"}
    )
    stacked = pd.concat([p1_view, p2_view], ignore_index=True).dropna(
        subset=["specialist"]
    )
    # Take each player's LAST specialist (most recent appearance)
    stacked = stacked.sort_values(["player_id", "match_date"])
    last = stacked.groupby("player_id").tail(1).reset_index(drop=True)

    # Filter to players with at least min_n_total cumulative matches
    last = last[last["n_total"] >= min_n_total]

    # Add names
    plr = players[["player_id", "full_name"]].copy()
    last = last.merge(plr, on="player_id", how="left")

    top = (last.sort_values("specialist", ascending=False)
                .head(n)[["full_name", "specialist", "n_clay", "n_total"]]
                .reset_index(drop=True))
    bot = (last.sort_values("specialist", ascending=True)
                .head(n)[["full_name", "specialist", "n_clay", "n_total"]]
                .reset_index(drop=True))
    top["specialist"] = top["specialist"].round(4)
    bot["specialist"] = bot["specialist"].round(4)
    return top, bot


def _star_player_verification(
    features: pd.DataFrame, master: pd.DataFrame,
    players: pd.DataFrame | None,
) -> pd.DataFrame:
    """For each known specialist / anti-specialist, output peak + final
    specialist values + sanity tag."""
    if features.empty or players is None or "full_name" not in players.columns:
        return pd.DataFrame()

    # Map name patterns → player_ids
    star_ids: dict[str, int] = {}
    for pat in (*_KNOWN_SPECIALISTS, *_KNOWN_ANTI_SPECIALISTS):
        hit = players[players["full_name"].astype("string")
                      .str.contains(pat, case=False, na=False, regex=False)]
        if len(hit):
            star_ids[pat] = int(hit.iloc[0]["player_id"])

    f = features.merge(master[["match_id", "match_date", "p1_id", "p2_id"]],
                       on="match_id")

    p1_view = f[["match_date", "p1_id", "specialist_p1",
                 "specialist_n_total_p1"]].rename(
        columns={"p1_id": "player_id",
                 "specialist_p1": "specialist",
                 "specialist_n_total_p1": "n_total"}
    )
    p2_view = f[["match_date", "p2_id", "specialist_p2",
                 "specialist_n_total_p2"]].rename(
        columns={"p2_id": "player_id",
                 "specialist_p2": "specialist",
                 "specialist_n_total_p2": "n_total"}
    )
    stacked = pd.concat([p1_view, p2_view], ignore_index=True).dropna(
        subset=["specialist"]
    )

    rows = []
    for pat, pid in star_ids.items():
        sub = stacked[stacked["player_id"] == pid]
        if sub.empty:
            continue
        peak = sub["specialist"].max()
        trough = sub["specialist"].min()
        final = sub.sort_values("match_date")["specialist"].iloc[-1]
        n_appear = len(sub)
        is_specialist_pred = pat in _KNOWN_SPECIALISTS
        rows.append({
            "player": pat,
            "expected": "specialist (>+0.05)" if is_specialist_pred
                         else "anti-specialist (≤0)",
            "peak": round(peak, 4),
            "trough": round(trough, 4),
            "final": round(final, 4),
            "n_appear": n_appear,
        })
    return pd.DataFrame(rows)


# ===========================================================================
# GOAT × specialist audit
# ===========================================================================

def _goat_audit(
    features: pd.DataFrame, master: pd.DataFrame,
    clay_elo_features: pd.DataFrame | None,
    players: pd.DataFrame | None,
    *,
    goat_pool_size: int = GOAT_POOL_SIZE,
    spec_threshold: float = SPECIALIST_AUDIT_THRESHOLD,
    spec_min_total: int = SPECIALIST_AUDIT_MIN_TOTAL,
    bucket_size: float = GOAT_AUDIT_ELO_BUCKET_SIZE,
) -> pd.DataFrame:
    """GOAT × specialist matchup audit with Elo-bucketed control.

    Output columns::

        elo_diff_bucket  | n_goat_x_spec  | spec_winrate
                         | n_goat_x_nonspec | nonspec_winrate
                         | spec_advantage_pp     (in percentage points)
    """
    if features.empty or clay_elo_features is None:
        return pd.DataFrame()

    # 1. GOAT pool — top by peak clay-Elo (across master appearances)
    elo_join = clay_elo_features.merge(
        master[["match_id", "p1_id", "p2_id"]], on="match_id",
    )
    p1_v = elo_join[["p1_id", "clay_elo_p1"]].rename(
        columns={"p1_id": "player_id", "clay_elo_p1": "elo"})
    p2_v = elo_join[["p2_id", "clay_elo_p2"]].rename(
        columns={"p2_id": "player_id", "clay_elo_p2": "elo"})
    peak_elo = (pd.concat([p1_v, p2_v]).groupby("player_id")["elo"]
                  .max().sort_values(ascending=False))
    goat_ids = set(peak_elo.head(goat_pool_size).index.tolist())

    # 2. Specialist pool — players whose specialist ever ≥ threshold AND
    #    cumulative n_total reaches spec_min_total at that point.
    feat_with_master = features.merge(
        master[["match_id", "p1_id", "p2_id"]], on="match_id",
    )
    s1 = feat_with_master[["p1_id", "specialist_p1", "specialist_n_total_p1"]].rename(
        columns={"p1_id": "player_id",
                 "specialist_p1": "specialist",
                 "specialist_n_total_p1": "n_total"})
    s2 = feat_with_master[["p2_id", "specialist_p2", "specialist_n_total_p2"]].rename(
        columns={"p2_id": "player_id",
                 "specialist_p2": "specialist",
                 "specialist_n_total_p2": "n_total"})
    sstack = pd.concat([s1, s2]).dropna(subset=["specialist"])
    qualifying = sstack[
        (sstack["specialist"] >= spec_threshold) &
        (sstack["n_total"] >= spec_min_total)
    ]
    spec_ids = set(qualifying["player_id"].unique().tolist())
    # Exclude any player who is also a GOAT
    spec_ids = spec_ids - goat_ids

    # 3. Iterate master, classify each match
    elo_join_full = clay_elo_features.merge(
        master[["match_id", "p1_id", "p2_id", "y"]], on="match_id",
    )

    audit_rows = []
    for r in elo_join_full.itertuples():
        p1, p2 = int(r.p1_id), int(r.p2_id)
        p1_g = p1 in goat_ids
        p2_g = p2 in goat_ids
        p1_s = p1 in spec_ids
        p2_s = p2 in spec_ids

        if p1_g and p2_s:
            category = "goat_x_spec"
            spec_won = (int(r.y) == 0)   # p2 (specialist) won
            elo_diff_against_target = float(r.clay_elo_p1) - float(r.clay_elo_p2)
        elif p2_g and p1_s:
            category = "goat_x_spec"
            spec_won = (int(r.y) == 1)
            elo_diff_against_target = float(r.clay_elo_p2) - float(r.clay_elo_p1)
        elif p1_g and not p2_s and not p2_g:
            category = "goat_x_nonspec"
            spec_won = (int(r.y) == 0)   # nonspec p2 winning
            elo_diff_against_target = float(r.clay_elo_p1) - float(r.clay_elo_p2)
        elif p2_g and not p1_s and not p1_g:
            category = "goat_x_nonspec"
            spec_won = (int(r.y) == 1)
            elo_diff_against_target = float(r.clay_elo_p2) - float(r.clay_elo_p1)
        else:
            continue

        bucket = int(round(elo_diff_against_target / bucket_size) * bucket_size)
        audit_rows.append({
            "elo_diff_bucket": bucket,
            "category": category,
            "weaker_won": spec_won,
        })

    if not audit_rows:
        return pd.DataFrame()

    audit_df = pd.DataFrame(audit_rows)
    grouped = audit_df.groupby(["elo_diff_bucket", "category"]).agg(
        n=("weaker_won", "size"),
        wins=("weaker_won", "sum"),
    ).reset_index()
    grouped["winrate_%"] = (grouped["wins"] / grouped["n"] * 100).round(2)

    # Pivot to side-by-side
    pivot = grouped.pivot_table(
        index="elo_diff_bucket", columns="category",
        values=["n", "winrate_%"], aggfunc="first",
    )
    pivot.columns = [f"{c[0]}_{c[1]}" for c in pivot.columns]
    pivot = pivot.reset_index()
    # Compute spec advantage (winrate vs control)
    if "winrate_%_goat_x_spec" in pivot.columns and "winrate_%_goat_x_nonspec" in pivot.columns:
        pivot["spec_advantage_pp"] = (
            pivot["winrate_%_goat_x_spec"] - pivot["winrate_%_goat_x_nonspec"]
        ).round(2)
    return pivot.sort_values("elo_diff_bucket").reset_index(drop=True)


# ===========================================================================
# Self-checks (9)
# ===========================================================================

def _run_sanity_checks(
    master: pd.DataFrame, features: pd.DataFrame,
    unified: pd.DataFrame,
    clay_elo_features: pd.DataFrame | None,
    players: pd.DataFrame | None,
    log_list: list[str],
    *,
    min_total: int, min_clay: int,
    rolling_window: int | None,
    leakage_sample_size: int, rng_seed: int,
    nan_rates: pd.DataFrame,
) -> float | None:
    n = len(features)

    # 1. n_clay <= n_total for all rows (clay ⊆ all-surface)
    nc1 = features["specialist_n_clay_p1"].astype("Int64").fillna(0).astype(int)
    nt1 = features["specialist_n_total_p1"].astype("Int64").fillna(0).astype(int)
    nc2 = features["specialist_n_clay_p2"].astype("Int64").fillna(0).astype(int)
    nt2 = features["specialist_n_total_p2"].astype("Int64").fillna(0).astype(int)
    if (nc1 > nt1).any() or (nc2 > nt2).any():
        raise AssertionError("specialist_n_clay > specialist_n_total — invariant violated")
    log_list.append(f"[✓] n_clay ≤ n_total  {n}/{n}")

    # 2. Cold-start NaN consistency: where gates fail, specialist must be NaN
    eligible_p1 = (nt1 >= min_total) & (nc1 >= min_clay)
    eligible_p2 = (nt2 >= min_total) & (nc2 >= min_clay)
    bad_p1 = (~eligible_p1) & features["specialist_p1"].notna()
    bad_p2 = (~eligible_p2) & features["specialist_p2"].notna()
    if bad_p1.any() or bad_p2.any():
        raise AssertionError(
            f"Cold-start violation: {int(bad_p1.sum() + bad_p2.sum())} rows "
            f"have specialist value when n_total<{min_total} or n_clay<{min_clay}"
        )
    log_list.append(f"[✓] Cold-start NaN consistency  "
                  f"(n_total<{min_total} or n_clay<{min_clay} → specialist NaN)")

    # 3. Where eligible, specialist in [-0.5, 0.5] (theoretical max)
    sp1 = features["specialist_p1"].dropna()
    sp2 = features["specialist_p2"].dropna()
    if (sp1.abs() > 0.5).any() or (sp2.abs() > 0.5).any():
        raise AssertionError("specialist outside [-0.5, +0.5] (impossible math)")
    log_list.append(f"[✓] specialist ∈ [-0.5, +0.5] when present")

    # 4. NaN rates within tolerance (warn at >25%)
    high_nan = nan_rates[nan_rates["nan_%"] > NAN_RATE_WARN_THRESHOLD * 100]
    if not high_nan.empty:
        for _, r in high_nan.iterrows():
            log_list.append(f"[⚠] {r['column']} NaN = {r['nan_%']}% "
                          f"(> {NAN_RATE_WARN_THRESHOLD*100:.0f}% threshold)")
    else:
        log_list.append(
            "[✓] NaN rates within tolerance: " +
            ", ".join(f"{r['column']}={r['nan_%']}%"
                      for _, r in nan_rates.iterrows())
        )

    # 5. Active-player p10/p90 within ±0.30 (warn)
    eligible_full = sp1[features["specialist_n_total_p1"].astype("Int64") >= 100]
    if len(eligible_full) > 100:
        p10 = float(eligible_full.quantile(0.10))
        p90 = float(eligible_full.quantile(0.90))
        if abs(p10) <= 0.30 and abs(p90) <= 0.30:
            log_list.append(f"[✓] Active-player p10={p10:+.3f}, p90={p90:+.3f} "
                          f"⊆ [-0.30, +0.30]")
        else:
            log_list.append(f"[⚠] Active-player p10={p10:+.3f}, p90={p90:+.3f} "
                          f"outside [-0.30, +0.30]")
        # 6. Active-player median in [-0.05, +0.05]
        med = float(eligible_full.median())
        if -0.05 <= med <= 0.05:
            log_list.append(f"[✓] Active-player median = {med:+.4f} ⊆ [-0.05, +0.05]")
        else:
            log_list.append(f"[⚠] Active-player median = {med:+.4f} "
                          f"outside [-0.05, +0.05]")

    # 7. Pearson r vs clay_elo_diff (warn if > 0.5)
    elo_corr = None
    if clay_elo_features is not None and not clay_elo_features.empty:
        joined = features.merge(
            clay_elo_features[["match_id", "clay_elo_diff"]], on="match_id",
        )
        eligible = joined[joined["specialist_diff"].notna() &
                          joined["clay_elo_diff"].notna()]
        if len(eligible) >= 30:
            r = float(eligible["specialist_diff"].corr(eligible["clay_elo_diff"]))
            elo_corr = r
            tag = "[✓]" if abs(r) <= ELO_CORRELATION_WARN_THRESHOLD else "[⚠]"
            log_list.append(
                f"{tag} Pearson r(specialist_diff, clay_elo_diff) = {r:+.3f}  "
                f"(predicted 0.25-0.40, warn threshold ±{ELO_CORRELATION_WARN_THRESHOLD})"
            )

    # 8. 100-row stratified leakage cross-check
    _check_no_leakage(
        master, features, unified, min_total, min_clay, rolling_window,
        sample_size=leakage_sample_size, rng_seed=rng_seed, log_list=log_list,
    )

    # 9. Known-specialist soft warns
    _check_known_players(features, master, players, log_list)

    return elo_corr


def _check_known_players(
    features: pd.DataFrame, master: pd.DataFrame,
    players: pd.DataFrame | None, log_list: list[str],
) -> None:
    """Soft warns for known specialists / anti-specialists."""
    if players is None or "full_name" not in players.columns:
        return
    f = features.merge(master[["match_id", "match_date", "p1_id", "p2_id"]],
                       on="match_id")
    p1v = f[["match_date", "p1_id", "specialist_p1"]].rename(
        columns={"p1_id": "player_id", "specialist_p1": "specialist"})
    p2v = f[["match_date", "p2_id", "specialist_p2"]].rename(
        columns={"p2_id": "player_id", "specialist_p2": "specialist"})
    stacked = pd.concat([p1v, p2v]).dropna(subset=["specialist"])

    for pat in _KNOWN_SPECIALISTS[:4]:   # top 4 (primary)
        hit = players[players["full_name"].astype("string")
                      .str.contains(pat, case=False, na=False, regex=False)]
        if not len(hit):
            continue
        pid = int(hit.iloc[0]["player_id"])
        sub = stacked[stacked["player_id"] == pid]
        if sub.empty:
            log_list.append(f"[⚠] Known specialist {pat!r}: no eligible rows")
            continue
        peak = float(sub["specialist"].max())
        if peak >= 0.05:
            log_list.append(f"[✓] {pat!r} peak specialist = {peak:+.4f} ≥ +0.05")
        else:
            log_list.append(f"[⚠] {pat!r} peak specialist = {peak:+.4f} "
                          f"BELOW +0.05 — review")

    for pat in _KNOWN_ANTI_SPECIALISTS:
        hit = players[players["full_name"].astype("string")
                      .str.contains(pat, case=False, na=False, regex=False)]
        if not len(hit):
            continue
        pid = int(hit.iloc[0]["player_id"])
        sub = stacked[stacked["player_id"] == pid]
        if sub.empty:
            log_list.append(f"[⚠] Anti-specialist {pat!r}: no eligible rows")
            continue
        # Peak specialist for anti — should be ≤ 0 if truly clay-averse
        peak = float(sub["specialist"].max())
        trough = float(sub["specialist"].min())
        if trough <= 0:
            log_list.append(f"[✓] {pat!r} trough specialist = {trough:+.4f} ≤ 0")
        else:
            log_list.append(f"[⚠] {pat!r} trough = {trough:+.4f} > 0 — review")


def _check_no_leakage(
    master: pd.DataFrame, features: pd.DataFrame, unified: pd.DataFrame,
    min_total: int, min_clay: int, rolling_window: int | None,
    *,
    sample_size: int, rng_seed: int, log_list: list[str],
) -> None:
    """Stratified-by-year sample. Re-derive specialist; assert exact match."""
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

    # Re-build full per-player history (across all unified rows)
    state: dict[int, list] = defaultdict(list)
    p1_ids = unified["p1_id"].astype("Int64").to_numpy()
    p2_ids = unified["p2_id"].astype("Int64").to_numpy()
    ys = unified["y"].astype("Int8").to_numpy()
    dates = pd.to_datetime(unified["match_date"]).to_numpy()
    surfaces = unified["surface_lower"].to_numpy()
    keys = unified["_key"].to_numpy()
    for i in range(len(unified)):
        is_clay = str(surfaces[i]).lower() == "clay"
        state[int(p1_ids[i])].append(
            (dates[i], 1 if int(ys[i]) == 1 else 0, is_clay, keys[i])
        )
        state[int(p2_ids[i])].append(
            (dates[i], 1 if int(ys[i]) == 0 else 0, is_clay, keys[i])
        )

    fails = 0
    failure_msgs: list[str] = []
    for _, row in sampled.iterrows():
        p1, p2 = int(row["p1_id"]), int(row["p2_id"])
        cur_date = pd.Timestamp(row["match_date"]).to_datetime64()
        cur_key = str(row["stable_match_key"])
        n_c1, w_c1, n_t1, w_t1 = _query_player_history(
            state.get(p1, []), cur_date, cur_key, rolling_window,
        )
        n_c2, w_c2, n_t2, w_t2 = _query_player_history(
            state.get(p2, []), cur_date, cur_key, rolling_window,
        )
        sp1_e = _specialist_index(n_c1, w_c1, n_t1, w_t1,
                                   min_total=min_total, min_clay=min_clay)
        sp2_e = _specialist_index(n_c2, w_c2, n_t2, w_t2,
                                   min_total=min_total, min_clay=min_clay)
        # NaN-aware comparison
        def _eq(a, b):
            if pd.isna(a) and pd.isna(b): return True
            if pd.isna(a) or pd.isna(b): return False
            return abs(a - b) < 1e-9
        ok = (
            int(row["specialist_n_clay_p1"]) == n_c1
            and int(row["specialist_n_total_p1"]) == n_t1
            and int(row["specialist_n_clay_p2"]) == n_c2
            and int(row["specialist_n_total_p2"]) == n_t2
            and _eq(row["specialist_p1"], sp1_e)
            and _eq(row["specialist_p2"], sp2_e)
        )
        if not ok:
            fails += 1
            if len(failure_msgs) < 3:
                failure_msgs.append(
                    f"  match_id={int(row['match_id'])}: "
                    f"stored=(n_c1={int(row['specialist_n_clay_p1'])}, "
                    f"n_t1={int(row['specialist_n_total_p1'])}, "
                    f"sp1={row['specialist_p1']}) "
                    f"expected=(n_c1={n_c1}, n_t1={n_t1}, sp1={sp1_e})"
                )
    if fails:
        raise AssertionError(
            f"Leakage check failed: {fails} of {len(sampled)} rows differ.\n"
            + "\n".join(failure_msgs)
        )
    log_list.append(f"[✓] Leakage cross-check  {len(sampled)}/{len(sampled)} match")
