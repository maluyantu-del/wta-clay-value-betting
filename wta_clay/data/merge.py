"""Master table builder.

Joins the matcher's output (TD ⨝ Sackmann) with Sackmann player meta and
optional players.csv DOB data, then re-orients to a fixed
(p1_id < p2_id) order to remove label leakage. The downstream feature
layer must read this file via :func:`load_master` and use the ``y`` target
exclusively — never raw ``winner_*`` / ``loser_*`` columns.

Output columns (32, see :data:`MASTER_COLUMNS` for the canonical order):

  Identifiers     : match_id, match_date, tourney_id, tourney_name,
                    tourney_date (Sackmann tournament-start), tour_level,
                    round_norm, surface, best_of, score, retired, minutes
  Player p1       : p1_id, p1_name, p1_dob, p1_age, p1_hand, p1_ht,
                    p1_ioc, p1_rank, p1_rank_points
  Player p2       : (mirror of p1)
  Target          : y  (1 if p1 won)
  Market          : market_source ∈ {pinnacle, bet365, none},
                    odds_p1_pinnacle / odds_p2_pinnacle, ... (b365/avg/max)
  Weights         : sample_weight, not_for_betting
  Provenance      : td_source_file, match_path, match_score, winner_mismatch

Hard self-check assertions raise on violation (see :func:`_validate_master`):
  1. p1_id < p2_id strictly
  2. y has no NaN and is in {0, 1}
  3. match_id is unique
  4. odds_p1_X ⇔ odds_p2_X paired (no single-sided rows)
  5. market_source='none' → ALL odds columns NaN
  6. sample_weight ∈ {0.25, 0.5, 1.0}
  7. Index is RangeIndex(0, N) after sort
  8. parquet round-trip preserves row order
  9. retired bool is consistent with score regex
  10. tourney_date is on or before match_date
  11. p1_rank / p2_rank missing rate ≤ threshold (WARN, not raise)

Known data anomaly (NOT a bug — informational):

  ``tourney_date > match_date`` fires on ~6% of rows (678/11098 in 2007-
  2024). Investigated 2026-04-21: delta is 1-2 days max (NEVER > 7), and
  80% of affected tourney_ids have all rows sharing one match_date — TD
  records some early-round matches at "week-start" (Sunday) while
  Sackmann's ``tourney_date`` is the official Monday start.
  Implication for downstream features:

    **Time-window features (Elo decay, recent-form rolling stats, fatigue
    days-since-last-match) MUST use Sackmann's ``tourney_date`` rather
    than TD's ``match_date`` for these 678 rows** (or universally — the
    tourney_date is more semantically stable for week-level aggregation).
    A 1-2 day error in the wrong direction can silently leak future
    information into recent-form windows of width 1-2 weeks.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ===========================================================================
# Constants
# ===========================================================================

ROUND_ORDER: tuple[str, ...] = ("R128", "R64", "R32", "R16",
                                "QF", "SF", "F", "RR")
TOUR_LEVEL_ORDER: tuple[str, ...] = ("GS", "WTA1000", "WTA500",
                                     "WTA250", "WTA125", "ITF")
HAND_CATEGORIES: tuple[str, ...] = ("L", "R", "U")
MARKET_SOURCE_CATEGORIES: tuple[str, ...] = ("pinnacle", "bet365", "none")
MATCH_PATH_CATEGORIES: tuple[str, ...] = ("primary", "fallback")

#: Regex matching anomalous score strings (retired / walkover / default).
#: Matches whole-word RET/WO/DEF (case-insensitive). Used to populate the
#: ``retired`` bool column on the master so feature/backtest layers don't
#: re-derive it.
_RETIRED_RE = re.compile(r"\b(?:RET|W/O|WO|DEF|Def|Walkover|ABN)\b",
                          re.IGNORECASE)

#: Canonical column order for the master parquet. Downstream code should
#: rely on this list rather than introspecting columns at runtime.
#:
#: ``stable_match_key`` is the cross-frame stable identifier
#: (``f"{tourney_id}_{p1_id}_{p2_id}"``) — feature modules MUST use this
#: instead of ``match_id`` when joining against a re-loaded Sackmann
#: frame, because ``match_id`` is a row index in a SPECIFIC filtering of
#: Sackmann (year_from / include_qual_itf flags) and silently shifts
#: when a different filter is applied. ``match_id`` remains the compact
#: primary key for joining features-parquet to master.
MASTER_COLUMNS: tuple[str, ...] = (
    # Identifiers / metadata
    "match_id", "stable_match_key",
    "match_date", "tourney_id", "tourney_name",
    "tourney_date", "tour_level", "round_norm", "surface",
    "best_of", "score", "retired", "minutes",
    # Player p1 (smaller id)
    "p1_id", "p1_name", "p1_dob", "p1_age", "p1_hand",
    "p1_ht", "p1_ioc", "p1_rank", "p1_rank_points",
    # Player p2 (larger id)
    "p2_id", "p2_name", "p2_dob", "p2_age", "p2_hand",
    "p2_ht", "p2_ioc", "p2_rank", "p2_rank_points",
    # Target
    "y",
    # Market
    "market_source",
    "odds_p1_pinnacle", "odds_p2_pinnacle",
    "odds_p1_b365", "odds_p2_b365",
    "odds_p1_avg", "odds_p2_avg",
    "odds_p1_max", "odds_p2_max",
    # Weights
    "sample_weight", "not_for_betting",
    # Provenance / audit
    "td_source_file", "match_path", "match_score", "winner_mismatch",
)


# ===========================================================================
# Result dataclass
# ===========================================================================

@dataclass
class MergeResult:
    """Top-level return of :func:`build_master_dataset`."""
    master: pd.DataFrame
    schema_report: pd.DataFrame
    self_check_log: list[str] = field(default_factory=list)
    duplicate_rejected: pd.DataFrame = field(default_factory=pd.DataFrame)


# ===========================================================================
# Main entry
# ===========================================================================

def build_master_dataset(
    matched: pd.DataFrame,
    sackmann: pd.DataFrame,
    players: pd.DataFrame | None = None,
    *,
    out_path: Path | str | None = None,
    rank_missing_warn_threshold: float = 0.05,
    run_self_checks: bool = True,
) -> MergeResult:
    """Build the master dataset and optionally write to parquet.

    Inputs
    ------
    matched : DataFrame
        Output of :func:`name_matcher.match_td_to_sackmann.matched`.
        Must contain ``sackmann_match_id`` (Sackmann row index in the
        clay-filtered frame) and TD odds + player columns.
    sackmann : DataFrame
        Output of ``filter_clay(load_sackmann_matches(...))``. Index is
        expected to be a contiguous RangeIndex (filter_clay resets index).
    players : DataFrame | None
        Optional. Used only to source ``p1_dob`` / ``p2_dob``. If absent,
        DOB columns are NaT. All other player meta (age, hand, ht, ioc)
        comes from inline Sackmann match-file columns.

    Notes
    -----
    The downstream feature engineering layer MUST use the ``y`` column to
    determine match outcome. Raw ``winner_*`` / ``loser_*`` should not
    leak into features — the ``winner_mismatch`` flag indicates that 0.02%
    of rows have TD/Sackmann winner-direction disagreement; ``y`` is
    Sackmann-authoritative and aligned with the p1/p2 id-sorted order.
    """
    if matched.empty:
        log.warning("Empty matched DataFrame; returning empty master.")
        return MergeResult(
            master=pd.DataFrame(columns=list(MASTER_COLUMNS)),
            schema_report=pd.DataFrame(columns=["column", "dtype", "missing_%"]),
            self_check_log=[],
        )

    matched, duplicate_rejected = _dedup_by_sackmann_id(matched)
    df = _join_sackmann_meta(matched, sackmann)
    df = _join_dob_from_players(df, players)
    p1_is_winner = _compute_p1_orientation(df)

    out = _build_player_columns(df, p1_is_winner)
    out["y"] = p1_is_winner.astype("Int8")
    _attach_odds_columns(out, df, p1_is_winner)
    out["market_source"] = _resolve_market_source(df)
    _enforce_market_source_consistency(out)
    _attach_metadata_columns(out, df)
    _attach_retired_flag(out)
    _enforce_categoricals(out)

    out = out.reindex(columns=list(MASTER_COLUMNS))

    # Stable sort (mergesort); reset index → required by self-check #7.
    out = (out.sort_values(["match_date", "tourney_id", "match_id"],
                          kind="mergesort")
              .reset_index(drop=True))

    self_check_log: list[str] = []
    if run_self_checks:
        _validate_master(out, rank_missing_warn_threshold, self_check_log)

    schema_report = _schema_report(out)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_path, index=False, compression="snappy")
        log.info("Wrote master to %s (%d rows × %d cols)",
                 out_path, len(out), len(out.columns))
        if run_self_checks:
            _check_parquet_roundtrip(out, out_path, self_check_log)

    return MergeResult(master=out, schema_report=schema_report,
                       self_check_log=self_check_log,
                       duplicate_rejected=duplicate_rejected)


# ===========================================================================
# Pre-merge dedup
# ===========================================================================

def _dedup_by_sackmann_id(matched: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop TD rows that map to the same Sackmann match as a better-scoring row.

    The matcher does per-row best-scoring selection independently, so two
    different TD rows can both fuzzy-match (above threshold) to the same
    Sackmann row when names are similar (e.g. Kozlova/Kucova/Kuzmova).
    Without dedup, the master would have duplicate ``match_id`` and the
    self-check would fail.

    Strategy: keep the row with highest ``match_score`` per
    ``sackmann_match_id``. Ties broken by primary > fallback, then by date.
    The losers are returned in a separate DataFrame for audit — they're
    almost always real WTA matches that Sackmann lacks under the older
    name (player surname change). Counts should be very small (single
    digits over the whole 2007-2025 dataset).
    """
    if matched.empty or "sackmann_match_id" not in matched.columns:
        return matched, pd.DataFrame()

    df = matched.copy()
    df["_path_pri"] = (df["match_path"] == "primary").astype(int)
    df = df.sort_values(
        ["sackmann_match_id", "match_score", "_path_pri", "match_date"],
        ascending=[True, False, False, True],
        kind="mergesort",
    )
    keep_mask = ~df["sackmann_match_id"].duplicated(keep="first")
    kept = df[keep_mask].drop(columns="_path_pri").sort_index()
    rejected = df[~keep_mask].drop(columns="_path_pri").sort_index()
    if not rejected.empty:
        log.warning(
            "_dedup_by_sackmann_id: %d TD rows dropped as duplicate "
            "Sackmann claims (likely surname-change false-positives)",
            len(rejected),
        )
    return kept, rejected


# ===========================================================================
# Joins
# ===========================================================================

#: Sackmann inline columns we pull into the master.
_SACKMANN_META_COLS: tuple[str, ...] = (
    "tourney_id", "tourney_name", "tour_level", "match_date",
    "score", "best_of", "minutes",
    "winner_id", "winner_name", "winner_hand", "winner_ht",
    "winner_ioc", "winner_age", "winner_rank", "winner_rank_points",
    "loser_id", "loser_name", "loser_hand", "loser_ht",
    "loser_ioc", "loser_age", "loser_rank", "loser_rank_points",
    "sample_weight", "not_for_betting",
)


def _join_sackmann_meta(matched: pd.DataFrame,
                        sackmann: pd.DataFrame) -> pd.DataFrame:
    """Lookup-join Sackmann meta via ``sackmann_match_id`` (= row index)."""
    avail = [c for c in _SACKMANN_META_COLS if c in sackmann.columns]
    sk = sackmann[avail].add_suffix("_sk")
    return matched.merge(
        sk, left_on="sackmann_match_id", right_index=True,
        how="left", validate="m:1",
    )


def _join_dob_from_players(df: pd.DataFrame,
                           players: pd.DataFrame | None) -> pd.DataFrame:
    """Add ``winner_dob`` / ``loser_dob`` columns from players.csv if given."""
    if players is None or "player_id" not in players.columns \
            or "dob" not in players.columns:
        df = df.copy()
        df["winner_dob"] = pd.NaT
        df["loser_dob"] = pd.NaT
        return df
    plr = players[["player_id", "dob"]].copy()
    df = df.merge(plr.rename(columns={"player_id": "winner_id_sk",
                                       "dob": "winner_dob"}),
                  on="winner_id_sk", how="left")
    df = df.merge(plr.rename(columns={"player_id": "loser_id_sk",
                                       "dob": "loser_dob"}),
                  on="loser_id_sk", how="left")
    return df


# ===========================================================================
# p1/p2 orientation + column construction
# ===========================================================================

def _compute_p1_orientation(df: pd.DataFrame) -> pd.Series:
    """Return Series[bool]: True iff Sackmann winner has the smaller id.

    Smaller id → p1; this fixes the label-leakage problem (model can't
    learn that ``winner_*`` is ``p1_*`` because they're not aligned).
    """
    w = df["winner_id_sk"].astype("Int64")
    l = df["loser_id_sk"].astype("Int64")
    # Compare with .lt to avoid pandas NA-propagation pitfalls; NA defaults
    # to False — an unmatched id pair shouldn't ship to master anyway.
    return w.lt(l).fillna(False)


# (player_field, sackmann_winner_col, sackmann_loser_col, target_dtype)
_PLAYER_FIELD_MAP: tuple[tuple[str, str, str, str], ...] = (
    ("id",          "winner_id_sk",          "loser_id_sk",          "Int64"),
    ("name",        "winner_name_sk",        "loser_name_sk",        "string"),
    ("dob",         "winner_dob",            "loser_dob",            "datetime64[ns]"),
    ("age",         "winner_age_sk",         "loser_age_sk",         "float64"),
    ("hand",        "winner_hand_sk",        "loser_hand_sk",        "string"),
    ("ht",          "winner_ht_sk",          "loser_ht_sk",          "Int64"),
    ("ioc",         "winner_ioc_sk",         "loser_ioc_sk",         "string"),
    ("rank",        "winner_rank_sk",        "loser_rank_sk",        "Int64"),
    ("rank_points", "winner_rank_points_sk", "loser_rank_points_sk", "Int64"),
)


def _build_player_columns(df: pd.DataFrame,
                          p1_is_winner: pd.Series) -> pd.DataFrame:
    """Compose p1_* / p2_* by selecting from winner/loser side."""
    out = pd.DataFrame(index=df.index)
    for field, w_col, l_col, dtype in _PLAYER_FIELD_MAP:
        if w_col not in df.columns or l_col not in df.columns:
            out[f"p1_{field}"] = pd.NA
            out[f"p2_{field}"] = pd.NA
            continue
        p1 = df[w_col].where(p1_is_winner, df[l_col])
        p2 = df[l_col].where(p1_is_winner, df[w_col])
        if dtype == "datetime64[ns]":
            out[f"p1_{field}"] = pd.to_datetime(p1, errors="coerce")
            out[f"p2_{field}"] = pd.to_datetime(p2, errors="coerce")
        elif dtype == "Int64":
            out[f"p1_{field}"] = pd.to_numeric(p1, errors="coerce").astype("Int64")
            out[f"p2_{field}"] = pd.to_numeric(p2, errors="coerce").astype("Int64")
        elif dtype == "float64":
            out[f"p1_{field}"] = pd.to_numeric(p1, errors="coerce").astype("float64")
            out[f"p2_{field}"] = pd.to_numeric(p2, errors="coerce").astype("float64")
        else:
            out[f"p1_{field}"] = p1.astype(dtype)
            out[f"p2_{field}"] = p2.astype(dtype)
    return out


# ===========================================================================
# Odds + market source
# ===========================================================================

# (canonical_name, td_winner_col, td_loser_col)
_ODDS_MAP: tuple[tuple[str, str, str], ...] = (
    ("pinnacle", "psw",   "psl"),
    ("b365",     "b365w", "b365l"),
    ("avg",      "avgw",  "avgl"),
    ("max",      "maxw",  "maxl"),
)


def _attach_odds_columns(out: pd.DataFrame, df: pd.DataFrame,
                         p1_is_winner: pd.Series) -> None:
    """Reorient TD odds columns to p1/p2 perspective.

    Single-sided rows (one side present, other NaN) are coerced to NaN-NaN
    on both sides — partial odds can't be used for betting and would
    violate the master's odds-pairing self-check. TD has ~4 such rows in
    2007-2025 across all bookmakers (data-entry issues at the source).
    """
    for source, td_w, td_l in _ODDS_MAP:
        if td_w not in df.columns or td_l not in df.columns:
            out[f"odds_p1_{source}"] = np.nan
            out[f"odds_p2_{source}"] = np.nan
            continue
        w = df[td_w].astype("float64")
        l = df[td_l].astype("float64")
        # Force pairing: if only one side present, drop both.
        single_sided = w.notna() ^ l.notna()
        if single_sided.any():
            log.info("attach_odds: %s has %d single-sided rows → coerced to NaN",
                     source, int(single_sided.sum()))
            w = w.where(~single_sided)
            l = l.where(~single_sided)
        out[f"odds_p1_{source}"] = w.where(p1_is_winner, l)
        out[f"odds_p2_{source}"] = l.where(p1_is_winner, w)


def _resolve_market_source(df: pd.DataFrame) -> pd.Series:
    """Determine market source per row: pinnacle > bet365 > none."""
    has_ps = (df.get("psw", pd.Series(dtype=float)).notna() &
              df.get("psl", pd.Series(dtype=float)).notna())
    has_b3 = (df.get("b365w", pd.Series(dtype=float)).notna() &
              df.get("b365l", pd.Series(dtype=float)).notna())
    src = pd.Series("none", index=df.index, dtype="object")
    src[has_b3 & ~has_ps] = "bet365"
    src[has_ps] = "pinnacle"
    return src


def _enforce_market_source_consistency(out: pd.DataFrame) -> None:
    """When ``market_source='none'``, force ALL odds columns to NaN.

    This is the user's locked rule: rows that can't be bet on (no Pinnacle
    AND no Bet365) must not carry stale avg/max prices that look bettable.
    The avg/max are TD's cross-book aggregates and may be present even
    when Pinnacle/Bet365 are absent — we drop them here for consistency
    with the gate-self-check #5.
    """
    none_mask = out["market_source"] == "none"
    if not none_mask.any():
        return
    odds_cols = [f"odds_p1_{s}" for s in ("pinnacle", "b365", "avg", "max")] + \
                [f"odds_p2_{s}" for s in ("pinnacle", "b365", "avg", "max")]
    n_nulled = int(out.loc[none_mask, odds_cols].notna().any(axis=1).sum())
    if n_nulled:
        log.info("market_source='none': nulling avg/max odds in %d rows "
                 "for consistency", n_nulled)
    for col in odds_cols:
        out.loc[none_mask, col] = np.nan


# ===========================================================================
# Identifiers / metadata
# ===========================================================================

def _attach_metadata_columns(out: pd.DataFrame, df: pd.DataFrame) -> None:
    """Carry through identifier + audit columns from the joined frame."""
    out["match_id"] = pd.to_numeric(df["sackmann_match_id"],
                                    errors="coerce").astype("Int64")
    # stable_match_key — composite cross-frame identifier. Use AFTER p1/p2
    # ordering is fixed (id-sorted), so the same physical match always
    # produces the same key regardless of who happened to be tagged
    # winner/loser in the source DataFrame.
    out["stable_match_key"] = (
        df.get("tourney_id_sk", df.get("tourney_id")).astype("string")
        + "_" + out["p1_id"].astype("Int64").astype("string")
        + "_" + out["p2_id"].astype("Int64").astype("string")
    )
    out["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    # NB: match_date_sk in the joined frame is Sackmann's parsed
    # tourney_date. Distinct from TD's per-match match_date.
    out["tourney_date"] = pd.to_datetime(df.get("match_date_sk"),
                                          errors="coerce")
    out["tourney_id"] = df.get("tourney_id_sk", df.get("tourney_id")).astype("string")
    out["tourney_name"] = df.get("tourney_name_sk",
                                  df.get("tourney_name")).astype("string")
    out["tour_level"] = df.get("tour_level_sk", df.get("tour_level"))
    out["round_norm"] = df.get("round_norm")
    out["surface"] = df.get("surface", "Clay").astype("string")
    out["best_of"] = pd.to_numeric(df.get("best_of_sk"),
                                    errors="coerce").astype("Int64")
    out["score"] = df.get("score_sk", "").astype("string")
    if "minutes_sk" in df.columns:
        out["minutes"] = pd.to_numeric(df["minutes_sk"],
                                        errors="coerce").astype("Int64")
    else:
        out["minutes"] = pd.NA
    out["sample_weight"] = pd.to_numeric(df.get("sample_weight_sk"),
                                          errors="coerce").astype("float64")
    # not_for_betting locked rule (combines three sources):
    #   1. Sackmann's per-row flag (currently fires only for Olympics raw='O')
    #   2. sample_weight < 1.0 — covers WTA125 (0.5) and ITF (0.25)
    #   3. tour_level explicitly in {WTA125, ITF} — defensive in case
    #      sample_weight gets corrupted upstream
    # The locked decision: WTA 125 events go INTO training (with downweight)
    # but are NEVER bet on at backtest time. Same for ITF.
    sk_not_for_betting = (df.get("not_for_betting_sk", False)
                            .fillna(False).astype(bool))
    sw_below_one = out["sample_weight"] < 1.0
    tier_unbettable = out["tour_level"].isin(["WTA125", "ITF"])
    # round_norm-based qualifying flag (defensive — qual rounds shouldn't
    # be in master since we filter to main draw, but check anyway)
    qual_rounds = out["round_norm"].astype("string").isin(
        ["Q1", "Q2", "Q3", "Q4", "QR"]
    )
    out["not_for_betting"] = (sk_not_for_betting | sw_below_one
                              | tier_unbettable | qual_rounds)
    out["td_source_file"] = df.get("_source_file", "").astype("string")
    out["match_path"] = df.get("match_path")
    out["match_score"] = pd.to_numeric(df.get("match_score"),
                                        errors="coerce").astype("float64")
    out["winner_mismatch"] = df.get("winner_mismatch", False).astype(bool)


def _attach_retired_flag(out: pd.DataFrame) -> None:
    """Derive ``retired`` bool from raw ``score`` string."""
    score = out["score"].astype("string").fillna("")
    out["retired"] = score.apply(lambda s: bool(_RETIRED_RE.search(s)))


def _enforce_categoricals(out: pd.DataFrame) -> None:
    """Coerce ordered/unordered categorical columns in place."""
    out["tour_level"] = pd.Categorical(out["tour_level"].astype("string"),
                                        categories=TOUR_LEVEL_ORDER, ordered=True)
    out["round_norm"] = pd.Categorical(
        out["round_norm"].astype("string"),
        categories=ROUND_ORDER, ordered=True,
    )
    out["match_path"] = pd.Categorical(out["match_path"].astype("string"),
                                        categories=MATCH_PATH_CATEGORIES)
    out["market_source"] = pd.Categorical(out["market_source"].astype("string"),
                                          categories=MARKET_SOURCE_CATEGORIES)
    out["p1_hand"] = pd.Categorical(
        out["p1_hand"].astype("string").fillna("U"),
        categories=HAND_CATEGORIES,
    )
    out["p2_hand"] = pd.Categorical(
        out["p2_hand"].astype("string").fillna("U"),
        categories=HAND_CATEGORIES,
    )


# ===========================================================================
# Self-checks
# ===========================================================================

def _validate_master(out: pd.DataFrame, rank_warn: float,
                     log_list: list[str]) -> None:
    """Run all hard assertions; raise on violation, warn on rank threshold."""
    n = len(out)

    # 1. p1_id < p2_id strictly
    bad = (out["p1_id"] >= out["p2_id"]).sum()
    if bad:
        raise AssertionError(f"p1_id < p2_id violation in {bad} rows")
    log_list.append(f"[✓] p1_id < p2_id strictly  {n}/{n}")

    # 2. y has no NaN, in {0, 1}
    if out["y"].isna().any():
        raise AssertionError(f"y has {out['y'].isna().sum()} NaN values")
    bad_y = (~out["y"].astype(int).isin([0, 1])).sum()
    if bad_y:
        raise AssertionError(f"y has {bad_y} values outside {{0, 1}}")
    log_list.append(f"[✓] y ∈ {{0, 1}} no NaN  {n}/{n}")

    # 3. match_id unique
    dup = out["match_id"].duplicated().sum()
    if dup:
        raise AssertionError(f"match_id has {dup} duplicates")
    log_list.append(f"[✓] match_id unique  {n}/{n}")

    # 3b. stable_match_key unique (cross-frame stability invariant)
    smk_dup = out["stable_match_key"].duplicated().sum()
    if smk_dup:
        raise AssertionError(
            f"stable_match_key has {smk_dup} duplicates — id-sort produced "
            f"non-unique (tourney_id, p1_id, p2_id) tuples"
        )
    smk_na = out["stable_match_key"].isna().sum()
    if smk_na:
        raise AssertionError(f"stable_match_key has {smk_na} NaN values")
    log_list.append(f"[✓] stable_match_key unique + non-null  {n}/{n}")

    # 4. Odds pairing — each odds_p1_X iff odds_p2_X
    for src in ("pinnacle", "b365", "avg", "max"):
        c1, c2 = f"odds_p1_{src}", f"odds_p2_{src}"
        single = (out[c1].notna() ^ out[c2].notna()).sum()
        if single:
            raise AssertionError(
                f"Odds pairing violation: {single} rows have "
                f"{c1} XOR {c2} (must be paired)"
            )
        log_list.append(f"[✓] {c1} ⇔ {c2}  no single-sided rows")

    # 5. market_source='none' rows must have ALL odds NaN
    none_rows = out[out["market_source"] == "none"]
    odds_cols = [f"odds_p1_{s}" for s in ("pinnacle", "b365", "avg", "max")] + \
                [f"odds_p2_{s}" for s in ("pinnacle", "b365", "avg", "max")]
    bad_none = none_rows[odds_cols].notna().any(axis=1).sum()
    if bad_none:
        raise AssertionError(
            f"market_source='none' but odds present in {bad_none} rows"
        )
    log_list.append(
        f"[✓] market_source='none' → all odds NaN  ({len(none_rows)} rows)"
    )

    # 6. sample_weight ∈ {0.25, 0.5, 1.0}
    bad_sw = (~out["sample_weight"].isin([0.25, 0.5, 1.0])).sum()
    if bad_sw:
        raise AssertionError(
            f"sample_weight outside {{0.25, 0.5, 1.0}} in {bad_sw} rows"
        )
    log_list.append(f"[✓] sample_weight ∈ {{0.25, 0.5, 1.0}}  {n}/{n}")

    # 7. Index is RangeIndex(0, N) after sort
    if not isinstance(out.index, pd.RangeIndex):
        raise AssertionError(
            f"index is {type(out.index).__name__}, expected RangeIndex"
        )
    if list(out.index) != list(range(n)):
        raise AssertionError("index is not contiguous 0..N-1 after sort")
    log_list.append(f"[✓] index is RangeIndex(0, {n})")

    # 9. retired bool must align with score regex (defensive)
    score = out["score"].astype("string").fillna("")
    expected_retired = score.apply(lambda s: bool(_RETIRED_RE.search(s)))
    if not (out["retired"] == expected_retired).all():
        n_bad = (out["retired"] != expected_retired).sum()
        raise AssertionError(
            f"retired column inconsistent with score regex in {n_bad} rows"
        )
    log_list.append(f"[✓] retired bool matches score regex  {n}/{n}")

    # 10. tourney_date ≤ match_date (Sackmann tournament starts on or before
    #     individual match date; can be equal for day-1 R128).
    if out["tourney_date"].notna().any() and out["match_date"].notna().any():
        date_violation = (out["tourney_date"] > out["match_date"]).sum()
        if date_violation:
            # Some legitimate cases exist: TD's match_date may have been
            # mis-recorded as before the official tournament start by 1-2
            # days due to qualifying overlap. Warn rather than raise.
            log.warning("tourney_date > match_date in %d rows (likely qualifying)",
                        int(date_violation))
            log_list.append(
                f"[⚠] tourney_date > match_date in {int(date_violation)} rows "
                f"(quali overlap; not fatal)"
            )
        else:
            log_list.append(f"[✓] tourney_date ≤ match_date  {n}/{n}")

    # 11. rank missing rate (warn only)
    for col in ("p1_rank", "p2_rank"):
        rate = out[col].isna().mean()
        if rate > rank_warn:
            log.warning("%s missing rate %.2f%% > %.2f%% threshold",
                        col, rate * 100, rank_warn * 100)
            log_list.append(
                f"[⚠] {col} missing rate {rate*100:.2f}% > "
                f"{rank_warn*100:.0f}% threshold"
            )
        else:
            log_list.append(
                f"[✓] {col} missing rate {rate*100:.2f}% "
                f"(< {rank_warn*100:.0f}% threshold)"
            )


def _check_parquet_roundtrip(orig: pd.DataFrame, path: Path,
                              log_list: list[str]) -> None:
    """Self-check #8: re-read parquet and verify match_id order preserved."""
    rt = pd.read_parquet(path)
    if len(orig) != len(rt):
        raise AssertionError(
            f"parquet round-trip row mismatch: {len(orig)} → {len(rt)}"
        )
    if not (orig["match_id"].reset_index(drop=True) ==
            rt["match_id"].reset_index(drop=True)).all():
        raise AssertionError("parquet round-trip changed row order")
    log_list.append(f"[✓] parquet round-trip preserves order  {len(orig)}/{len(orig)}")


def _schema_report(out: pd.DataFrame) -> pd.DataFrame:
    """Per-column dtype + missing rate as a DataFrame."""
    rows = []
    for col in out.columns:
        rows.append({
            "column": col,
            "dtype": str(out[col].dtype),
            "missing_%": round(float(out[col].isna().mean()) * 100, 2),
        })
    return pd.DataFrame(rows)


# ===========================================================================
# Convenience reader
# ===========================================================================

def load_master(path: Path | str) -> pd.DataFrame:
    """Convenience parquet reader. Restores categorical ordering."""
    df = pd.read_parquet(path)
    if "tour_level" in df.columns:
        df["tour_level"] = pd.Categorical(df["tour_level"],
                                           categories=TOUR_LEVEL_ORDER, ordered=True)
    if "round_norm" in df.columns:
        df["round_norm"] = pd.Categorical(df["round_norm"],
                                           categories=ROUND_ORDER, ordered=True)
    return df
