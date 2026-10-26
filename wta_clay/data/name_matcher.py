"""TD ↔ Sackmann fuzzy player-pair matcher with audit gating.

This module joins Tennis-Data.co.uk WTA odds rows to Sackmann match rows
via a tournament-aligned, year-scoped, fuzzy player-pair lookup.

Design (locked 2026-04-21 with 6 modifications + sackmann_missing 4-class
classification + 8 secondary corrections):

  1. Alias.csv maps TD ``tournament`` strings → (sackmann_name, year_range,
     tier_expected). Exact case-sensitive match on TD side; exact
     case-insensitive match on Sackmann side. NOT substring on either side
     (the alias is curated to exact values).
  2. Substring × year-overlap conflicts in alias are detected at load time.
  3. For each TD (tournament, year) group, primary candidate pool =
     Sackmann rows with ``tourney_name.lower() == sackmann_name.lower()``
     AND year matches the alias row, AND
     ``|td_group_earliest_date − sackmann_tourney_date| ≤ 7 days``.
  4. If primary pool is empty, retry without the date window (sackmann_name
     + year only). Outcomes:
        - still empty → ``sackmann_missing`` (Sackmann has no data)
        - non-empty → ``date_window_miss`` (data exists but date >7d off;
                     suspicious, not auto-matched)
  5. Tier check: ``actual_tour_level`` (Sackmann mode within pool) vs
     ``tier_expected`` (alias). Default = raise. With
     ``allow_tier_mismatch=True`` = warn + manual_review entry.
  6. Player pair match via rapidfuzz on a "lastname + first-initial"
     canonical form. Pair score = avg of two-orientation scores; winner
     direction is captured in ``winner_mismatch`` flag (does NOT block).
  7. Score thresholds: primary ≥85, fallback ≥92. Below → manual_review
     with status ∈ {low_score_primary, low_score_fallback,
     tier_mismatch, ambiguous_multi_hit, date_window_miss}.
  8. Ambiguous tiebreak: if top two candidates have score diff < 2,
     pick the one with closest match_date. If match_date also tied,
     write to manual_review (no random pick).
  9. Audit per period (2007-2008 / 2009 / 2010-2025) with hard/soft
     gates and a score histogram of unmatched rows for tuning.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:
    from rapidfuzz import fuzz
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "name_matcher requires rapidfuzz; install with `pip install rapidfuzz`."
    ) from exc

log = logging.getLogger(__name__)


# ===========================================================================
# Custom exceptions and dataclasses
# ===========================================================================

class TierMismatchError(ValueError):
    """Raised when alias.tier_expected disagrees with Sackmann.tour_level."""
    def __init__(self, td_event: str, td_year: int,
                 expected: str, actual: str):
        super().__init__(
            f"Tier mismatch for {td_event!r} ({td_year}): "
            f"alias expects {expected}, Sackmann classifies as {actual}. "
            f"Either fix alias.tier_expected, fix Sackmann classification "
            f"(TOURNEY_NAME_OVERRIDES / RAW_LEVEL_MAP), or pass "
            f"allow_tier_mismatch=True."
        )


@dataclass(frozen=True)
class AliasHit:
    """Single alias.csv row resolved for a (TD pattern, year)."""
    td_pattern: str
    sackmann_name: str
    year_from: int
    year_to: int
    tier_expected: str
    notes: str


@dataclass(frozen=True)
class PairMatchResult:
    """Outcome of one player-pair fuzzy match."""
    sackmann_match_idx: int | None       # index into the candidate pool DF
    score: float                          # 0-100
    winner_mismatch: bool                 # True if TD winner ↔ Sackmann loser
    top3: list[tuple[int, float]] = field(default_factory=list)
    ambiguous: bool = False               # True if tied within 2 points and dates also tied


@dataclass
class MatchResult:
    """Top-level return of :func:`match_td_to_sackmann`."""
    matched: pd.DataFrame
    unmatched: pd.DataFrame
    sackmann_missing: pd.DataFrame
    date_window_miss: pd.DataFrame
    manual_review: pd.DataFrame
    audit_report: pd.DataFrame
    tier_mismatches: pd.DataFrame
    config: dict


# ===========================================================================
# Section A — Alias loading and validation
# ===========================================================================

_ALIAS_REQUIRED_COLS = (
    "td_pattern", "sackmann_name", "year_from", "year_to",
    "tier_expected", "notes",
)
_VALID_TIERS = frozenset({"GS", "WTA1000", "WTA500", "WTA250", "WTA125"})


def load_alias(path: Path | str) -> pd.DataFrame:
    """Load tournament_alias.csv with comment-line skipping and validation.

    Performs at load time:
      - schema check (required columns)
      - tier_expected ∈ valid set
      - substring × year-overlap conflict scan (raises ``ValueError``)
    """
    path = Path(path)
    df = pd.read_csv(
        path,
        comment="#",
        dtype={
            "td_pattern": "string",
            "sackmann_name": "string",
            "year_from": "Int64",
            "year_to": "Int64",
            "tier_expected": "string",
            "notes": "string",
        },
    )
    missing = set(_ALIAS_REQUIRED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"alias.csv missing required columns: {sorted(missing)}")

    bad_tiers = (~df["tier_expected"].isin(_VALID_TIERS)).fillna(True)
    if bad_tiers.any():
        rows = df.loc[bad_tiers, ["td_pattern", "tier_expected"]].to_dict("records")
        raise ValueError(
            f"alias.csv has rows with invalid tier_expected: {rows}. "
            f"Allowed: {sorted(_VALID_TIERS)}"
        )

    _check_alias_substring_conflicts(df)
    log.info("Loaded alias from %s: %d rows, %d unique td_patterns",
             path, len(df), df["td_pattern"].nunique())
    return df


def _check_alias_substring_conflicts(alias: pd.DataFrame) -> None:
    """O(n²) scan: raise on (substring × year-overlap) conflict pairs.

    Two rows conflict when row A's td_pattern is a case-insensitive
    substring of row B's (or vice versa) AND their year ranges overlap
    AND they map to different (sackmann_name, tier_expected) combos.
    Same-canonical aliases are OK (e.g., 'Family Circle Cup' year-tier split).
    """
    rows = alias.reset_index(drop=True)
    pats = rows["td_pattern"].astype(str).tolist()
    yfs = rows["year_from"].fillna(2000).astype(int).tolist()
    yts = rows["year_to"].fillna(2099).astype(int).tolist()
    sks = rows["sackmann_name"].astype(str).tolist()
    tiers = rows["tier_expected"].astype(str).tolist()

    conflicts: list[str] = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            pi, pj = pats[i].lower(), pats[j].lower()
            if pi == pj:
                # Same TD pattern is OK — multi-row entries (year/tier split)
                continue
            if pi in pj or pj in pi:
                if yfs[i] <= yts[j] and yfs[j] <= yts[i]:
                    if (sks[i].lower() != sks[j].lower()
                        or tiers[i] != tiers[j]):
                        conflicts.append(
                            f"  - {pats[i]!r} ({yfs[i]}-{yts[i]}, "
                            f"sackmann={sks[i]}, tier={tiers[i]}) "
                            f"<> {pats[j]!r} ({yfs[j]}-{yts[j]}, "
                            f"sackmann={sks[j]}, tier={tiers[j]})"
                        )
    if conflicts:
        raise ValueError(
            "alias.csv has substring × year-overlap conflicts:\n"
            + "\n".join(conflicts)
        )


def _lookup_alias(td_pattern: str, year: int,
                  alias: pd.DataFrame) -> AliasHit | None:
    """Find the alias row for a (TD pattern, year). Returns None on miss."""
    sub = alias[alias["td_pattern"] == td_pattern]
    if sub.empty:
        return None
    yf = sub["year_from"].fillna(2000)
    yt = sub["year_to"].fillna(2099)
    hit_mask = (yf <= year) & (year <= yt)
    hits = sub[hit_mask]
    if hits.empty:
        return None
    if len(hits) > 1:
        log.warning(
            "Multiple alias rows match (%r, %d); using first: %s",
            td_pattern, year, hits["sackmann_name"].tolist()
        )
    r = hits.iloc[0]
    return AliasHit(
        td_pattern=str(r["td_pattern"]),
        sackmann_name=str(r["sackmann_name"]),
        year_from=int(r["year_from"]) if pd.notna(r["year_from"]) else 2000,
        year_to=int(r["year_to"]) if pd.notna(r["year_to"]) else 2099,
        tier_expected=str(r["tier_expected"]),
        notes=str(r["notes"]) if pd.notna(r["notes"]) else "",
    )


# ===========================================================================
# Section B — Sackmann indexing and candidate pools
# ===========================================================================

def build_sackmann_tourney_index(
    sackmann: pd.DataFrame,
) -> dict[tuple[str, int], pd.DataFrame]:
    """Pre-group Sackmann by (lowercase tourney_name, year) for fast lookup.

    Returned DataFrames preserve the original Sackmann index so candidate
    rows can be joined back to the source. Empty-key combos are absent.
    """
    if sackmann.empty:
        return {}
    df = sackmann.assign(
        _tn_lower=sackmann["tourney_name"].astype("string").str.lower(),
        _yr=sackmann["match_date"].dt.year.astype("Int64"),
    )
    return {
        (str(name), int(yr)): grp.drop(columns=["_tn_lower", "_yr"]).copy()
        for (name, yr), grp in df.groupby(["_tn_lower", "_yr"], dropna=True)
    }


def find_tourney_candidates_primary(
    sackmann_name: str,
    td_year: int,
    td_group_earliest_date: pd.Timestamp,
    sk_index: dict[tuple[str, int], pd.DataFrame],
    *,
    window_days: int = 7,
) -> tuple[pd.DataFrame, str]:
    """Find Sackmann candidates for a TD (tournament, year) group.

    Match steps (in order):
      1. EXACT case-insensitive ``sackmann_name`` match + year match.
      2. ``|td_group_earliest_date − sackmann.match_date| ≤ window_days``.

    Returns ``(candidates, status)`` where status is one of:
      - ``"in_window"``: primary success
      - ``"date_window_miss"``: name+year hit, but all candidates outside
        the date window (suspicious; goes to manual_review)
      - ``"sackmann_missing"``: no Sackmann data for (name, year) at all

    NOTE: ``td_group_earliest_date`` MUST be the EARLIEST match_date within
    the TD (tournament, year) group, NOT a single TD row's match_date.
    Sackmann ``match_date`` for all rows in one tourney = tourney_date
    (start-of-week), so the comparison is "TD group day-1 vs Sackmann
    tourney_date", which stays within the window even for 14-day GS draws.
    """
    key = (sackmann_name.lower(), td_year)
    if key not in sk_index:
        return pd.DataFrame(), "sackmann_missing"
    base = sk_index[key]
    delta = (base["match_date"] - td_group_earliest_date).abs()
    in_window = base[delta <= pd.Timedelta(days=window_days)]
    if len(in_window) > 0:
        return in_window, "in_window"
    return base, "date_window_miss"


def find_match_candidates_fallback(
    td_row: pd.Series,
    sackmann: pd.DataFrame,
    rankings: pd.DataFrame | None = None,
    *,
    days: int = 3,
    rank_top: int = 500,
) -> pd.DataFrame:
    """Fallback search when the primary alias-aligned pool fails.

    Filter:
      - Sackmann ``match_date`` within ±days of TD ``match_date``
      - winner_rank ≤ rank_top OR loser_rank ≤ rank_top

    The ``rankings`` DataFrame is currently unused (Sackmann's per-row
    winner_rank / loser_rank columns are sufficient); the parameter is
    accepted for API symmetry with future ranking-history queries.
    """
    td_date = td_row["match_date"]
    delta = (sackmann["match_date"] - td_date).abs()
    in_window = sackmann[delta <= pd.Timedelta(days=days)]
    if in_window.empty:
        return in_window
    rank_w = pd.to_numeric(in_window["winner_rank"], errors="coerce")
    rank_l = pd.to_numeric(in_window["loser_rank"], errors="coerce")
    mask = (rank_w <= rank_top) | (rank_l <= rank_top)
    return in_window[mask.fillna(False)]


# ===========================================================================
# Section C — Name normalization and pair fuzzy matching
# ===========================================================================

_PUNCT_RE = re.compile(r"[\.,'\"]")
_PAREN_RE = re.compile(r"\([^)]*\)")
_WS_RE = re.compile(r"\s+")


def _strip_diacritics(s: str) -> str:
    """Decompose unicode and drop combining marks (NFKD)."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )


def _normalize_name(s: str | None) -> str:
    """Convert any tennis name to canonical 'lastname f' (lowercase).

    Heuristic uses **period presence** as the canonical TD-vs-Sackmann signal:
        - Period anywhere in input → TD style (last token = initial)
        - No period → Sackmann Western style (first token = given name)

    This correctly handles short surnames (Tu/Li/Yu/Su) that would otherwise
    be misread as initials by a length-only heuristic.

    Examples
    --------
    >>> _normalize_name("Williams S.")            # TD style
    'williams s'
    >>> _normalize_name("Serena Williams")        # Sackmann Western
    'williams s'
    >>> _normalize_name("Sorribes Tormo S.")      # TD compound surname
    'sorribes tormo s'
    >>> _normalize_name("Sara Sorribes Tormo")    # Sackmann compound
    'sorribes tormo s'
    >>> _normalize_name("Hsieh S.W.")             # multi-initial → keep first
    'hsieh s'
    >>> _normalize_name("Tu M.")                  # short surname TD
    'tu m'
    >>> _normalize_name("Meilen Tu")              # short surname Sackmann
    'tu m'
    >>> _normalize_name("Na Li")                  # short surname Sackmann
    'li n'
    >>> _normalize_name("")
    ''
    """
    if not isinstance(s, str):
        return ""
    raw = s.strip()
    has_period = "." in raw
    s = _strip_diacritics(raw.lower())
    s = _PAREN_RE.sub(" ", s)
    s = _PUNCT_RE.sub("", s)
    # Replace hyphen with space — "Lucic-Baroni" becomes 2 tokens, which
    # token_set_ratio can match against Sackmann's older single-surname
    # form "Lucic" via subset detection. Trade-off: "Su-Wei Hsieh" splits
    # to 3 tokens; token_set_ratio still matches "Hsieh S.W." TD form
    # since "hsieh" + initial overlap dominates.
    s = s.replace("-", " ")
    s = _WS_RE.sub(" ", s).strip()
    if not s:
        return ""
    parts = s.split()
    if len(parts) == 1:
        return parts[0]
    if has_period:
        # TD style: last token is the initial(s); take first letter only.
        surname = " ".join(parts[:-1])
        initial = parts[-1][0]
        return f"{surname} {initial}"
    # Sackmann Western style: first token = given name, rest = surname.
    initial = parts[0][0]
    surname = " ".join(parts[1:])
    return f"{surname} {initial}"


def _name_score(a: str, b: str) -> float:
    """Per-name score: max of fuzz.ratio and fuzz.token_set_ratio.

    The ``max`` combination handles two distinct discrepancy patterns:
      - ``token_set_ratio`` rescues subset cases (Sackmann "Lucic" vs TD
        "Lucic-Baroni M." → tokens overlap but lengths differ).
      - ``ratio`` rescues spacing cases (Sackmann "Larcher De Brito" vs TD
        "Larcherde Brito M." → no token overlap but char-level near-identical).
    """
    return max(fuzz.ratio(a, b), fuzz.token_set_ratio(a, b))


def _pair_score(td_w: str, td_l: str, sk_w: str, sk_l: str) -> tuple[float, bool]:
    """Score an unordered (winner, loser) pair both ways.

    Uses :func:`_name_score` per side (max of fuzz.ratio + token_set_ratio).
    Returns ``(score, winner_mismatch)``. ``winner_mismatch`` is True iff
    the swapped orientation scored strictly higher than aligned.
    """
    aligned = (_name_score(td_w, sk_w) + _name_score(td_l, sk_l)) / 2.0
    swapped = (_name_score(td_w, sk_l) + _name_score(td_l, sk_w)) / 2.0
    if swapped > aligned:
        return swapped, True
    return aligned, False


# Tiebreak window: top two candidates within this many points are "close".
_AMBIGUOUS_SCORE_DIFF = 2.0


def match_player_pair(
    td_winner_raw: str,
    td_loser_raw: str,
    candidates: pd.DataFrame,
    td_match_date: pd.Timestamp | None = None,
) -> PairMatchResult:
    """Find the best Sackmann candidate row for an unordered TD player pair.

    If multiple candidates score within ``_AMBIGUOUS_SCORE_DIFF`` of the
    top, tiebreak by closest ``match_date`` to ``td_match_date``. If
    ``match_date`` is also tied, return ``ambiguous=True`` so the caller
    can route the row to manual_review (NO random pick).
    """
    if candidates.empty:
        return PairMatchResult(sackmann_match_idx=None, score=0.0,
                               winner_mismatch=False, top3=[])

    td_w = _normalize_name(td_winner_raw)
    td_l = _normalize_name(td_loser_raw)
    sk_w_norm = candidates["winner_name"].apply(_normalize_name).tolist()
    sk_l_norm = candidates["loser_name"].apply(_normalize_name).tolist()

    rows: list[tuple[int, float, bool]] = []
    for i in range(len(candidates)):
        score, mismatch = _pair_score(td_w, td_l, sk_w_norm[i], sk_l_norm[i])
        rows.append((i, score, mismatch))
    rows.sort(key=lambda t: t[1], reverse=True)
    best_i, best_score, best_mismatch = rows[0]
    top3 = [(i, round(s, 2)) for i, s, _ in rows[:3]]

    # Ambiguous tiebreak
    ambiguous = False
    close = [r for r in rows if best_score - r[1] < _AMBIGUOUS_SCORE_DIFF]
    if len(close) > 1 and td_match_date is not None and "match_date" in candidates.columns:
        # Pick closest match_date
        date_diffs = [
            (i, score, mm,
             abs((candidates.iloc[i]["match_date"] - td_match_date).total_seconds()))
            for i, score, mm in close
        ]
        date_diffs.sort(key=lambda t: t[3])
        # All same distance → ambiguous
        if len(date_diffs) > 1 and date_diffs[0][3] == date_diffs[1][3]:
            ambiguous = True
        best_i, best_score, best_mismatch, _ = date_diffs[0]
    elif len(close) > 1:
        # No date info → ambiguous
        ambiguous = True

    return PairMatchResult(
        sackmann_match_idx=best_i,
        score=best_score,
        winner_mismatch=best_mismatch,
        top3=top3,
        ambiguous=ambiguous,
    )


# ===========================================================================
# Section D — Tier validation
# ===========================================================================

def _validate_tier(
    actual_tier: str,
    expected_tier: str,
    *,
    td_event: str,
    td_year: int,
    allow_mismatch: bool,
) -> bool:
    """Return True if tier matches; False (and log) if mismatch and allowed.

    Raises ``TierMismatchError`` on mismatch when ``allow_mismatch=False``.
    """
    if actual_tier == expected_tier:
        return True
    if not allow_mismatch:
        raise TierMismatchError(td_event, td_year, expected_tier, actual_tier)
    log.warning("Tier mismatch %r (%d): expected=%s actual=%s",
                td_event, td_year, expected_tier, actual_tier)
    return False


# ===========================================================================
# Section E — Main entry
# ===========================================================================

# Default audit periods + gates (per locked spec)
DEFAULT_PERIODS: tuple[tuple[str, int, int], ...] = (
    ("2007-2008", 2007, 2008),
    ("2009",      2009, 2009),
    ("2010-2025", 2010, 2025),
)
DEFAULT_GATES: dict[str, float | None] = {
    "2007-2008": 0.05,   # soft 5%
    "2009":      None,   # not gated (Pinnacle absent)
    "2010-2025": 0.02,   # hard 2%
}
WINNER_MISMATCH_SOFT_GATE = 0.01  # 1% — warn above
WINNER_MISMATCH_HARD_GATE = 0.05  # 5% — pipeline bug


def match_td_to_sackmann(
    td: pd.DataFrame,
    sackmann: pd.DataFrame,
    alias: pd.DataFrame | Path | str,
    rankings: pd.DataFrame | None = None,
    *,
    primary_threshold: float = 85.0,
    fallback_threshold: float = 92.0,
    tourney_date_window_days: int = 7,
    fallback_date_window_days: int = 3,
    fallback_rank_top: int = 500,
    allow_tier_mismatch: bool = False,
    manual_review_path: Path | None = None,
    period_breakpoints: Sequence[tuple[str, int, int]] = DEFAULT_PERIODS,
    period_gates: dict[str, float | None] = DEFAULT_GATES,
) -> MatchResult:
    """Top-level matcher. See module docstring for the full algorithm.

    Inputs
    ------
    td : DataFrame
        Output of :func:`oddsfeed.load_tennisdata_odds` (clay-filtered).
    sackmann : DataFrame
        Output of :func:`sackmann.filter_clay(load_sackmann_matches(...))`.
    alias : DataFrame | Path | str
        Either a pre-loaded alias frame or a path to ``tournament_alias.csv``.

    Returns
    -------
    MatchResult dataclass with ``matched``, ``unmatched``,
    ``sackmann_missing``, ``date_window_miss``, ``manual_review``,
    ``audit_report``, ``tier_mismatches``, ``config``.
    """
    if isinstance(alias, (str, Path)):
        alias = load_alias(alias)
    else:
        _check_alias_substring_conflicts(alias)

    sk_index = build_sackmann_tourney_index(sackmann)

    matched_rows: list[dict] = []
    unmatched_rows: list[dict] = []
    sackmann_missing_rows: list[dict] = []
    date_window_miss_rows: list[dict] = []
    manual_review_rows: list[dict] = []
    tier_mismatch_log: list[dict] = []

    if td.empty:
        log.warning("Empty TD input; returning empty MatchResult.")
        return _empty_result(alias, primary_threshold, fallback_threshold)

    td = td.copy()
    td["_yr"] = td["match_date"].dt.year.astype("Int64")

    for (td_event, td_year), group in td.groupby(["tournament", "_yr"],
                                                  dropna=True, sort=False):
        td_year = int(td_year)
        hit = _lookup_alias(str(td_event), td_year, alias)
        if hit is None:
            for _, row in group.iterrows():
                unmatched_rows.append(_unmatched_row(row, "unaliased",
                                                    best_score=np.nan,
                                                    best_cand=""))
            continue

        td_earliest = group["match_date"].min()
        cands_pool, status = find_tourney_candidates_primary(
            hit.sackmann_name, td_year, td_earliest, sk_index,
            window_days=tourney_date_window_days,
        )

        if status == "sackmann_missing":
            for _, row in group.iterrows():
                sackmann_missing_rows.append(
                    _unmatched_row(row, "sackmann_missing",
                                   best_score=np.nan,
                                   best_cand=hit.sackmann_name)
                )
            continue

        if status == "date_window_miss":
            for _, row in group.iterrows():
                date_window_miss_rows.append(
                    _unmatched_row(row, "date_window_miss",
                                   best_score=np.nan,
                                   best_cand=hit.sackmann_name)
                )
                manual_review_rows.append(_review_row(
                    row, "date_window_miss",
                    f"Sackmann has data for ({hit.sackmann_name}, {td_year}) "
                    f"but earliest tourney_date is "
                    f"{cands_pool['match_date'].min().date()}, TD group "
                    f"earliest is {td_earliest.date()} (>{tourney_date_window_days}d off)",
                    best_score=np.nan,
                    best_cand=hit.sackmann_name,
                    top3=[],
                ))
            continue

        # Tier validation
        actual_tier = str(cands_pool["tour_level"].mode().iloc[0])
        tier_ok = _validate_tier(
            actual_tier, hit.tier_expected,
            td_event=str(td_event), td_year=td_year,
            allow_mismatch=allow_tier_mismatch,
        )
        if not tier_ok:
            tier_mismatch_log.append({
                "td_pattern": str(td_event), "td_year": td_year,
                "tier_expected": hit.tier_expected,
                "tier_actual": actual_tier,
                "n_rows": len(group),
            })
            for _, row in group.iterrows():
                manual_review_rows.append(_review_row(
                    row, "tier_mismatch",
                    f"Alias expected {hit.tier_expected} but Sackmann "
                    f"classifies {hit.sackmann_name} {td_year} as {actual_tier}",
                    best_score=np.nan, best_cand=hit.sackmann_name, top3=[],
                ))

        # Per-row fuzzy match
        for _, td_row in group.iterrows():
            primary = match_player_pair(
                td_row["winner_name_raw"], td_row["loser_name_raw"],
                cands_pool, td_match_date=td_row["match_date"],
            )

            if primary.score >= primary_threshold and not primary.ambiguous:
                sk_row = cands_pool.iloc[primary.sackmann_match_idx]
                matched_rows.append(_matched_row(
                    td_row, sk_row, "primary", primary.score,
                    primary.winner_mismatch, hit.tier_expected, actual_tier,
                ))
                continue

            # Primary failed — try fallback
            fb_pool = find_match_candidates_fallback(
                td_row, sackmann, rankings,
                days=fallback_date_window_days, rank_top=fallback_rank_top,
            )
            fb = match_player_pair(
                td_row["winner_name_raw"], td_row["loser_name_raw"],
                fb_pool, td_match_date=td_row["match_date"],
            )

            if fb.score >= fallback_threshold and not fb.ambiguous:
                sk_row = fb_pool.iloc[fb.sackmann_match_idx]
                fb_actual_tier = str(sk_row["tour_level"])
                matched_rows.append(_matched_row(
                    td_row, sk_row, "fallback", fb.score,
                    fb.winner_mismatch, hit.tier_expected, fb_actual_tier,
                ))
                continue

            # Both failed — classify and emit manual_review.
            # ambiguous_multi_hit only fires when the score WOULD have
            # qualified but ambiguity blocks the match. Below-threshold
            # cases are just low_score regardless of ambiguity.
            primary_blocked_by_ambig = (primary.ambiguous and
                                         primary.score >= primary_threshold)
            fallback_blocked_by_ambig = (fb.ambiguous and
                                          fb.score >= fallback_threshold)
            if primary_blocked_by_ambig:
                status_lbl = "ambiguous_multi_hit"
                reason = (f"Primary score {primary.score:.1f} ≥ "
                          f"{primary_threshold} but tied within "
                          f"{_AMBIGUOUS_SCORE_DIFF} pts AND match_date tied; "
                          f"cannot disambiguate")
                best = primary
                pool_for_topn = cands_pool
            elif fallback_blocked_by_ambig:
                status_lbl = "ambiguous_multi_hit"
                reason = (f"Fallback score {fb.score:.1f} ≥ "
                          f"{fallback_threshold} but ambiguous on date tiebreak")
                best = fb
                pool_for_topn = fb_pool
            elif primary.score > fb.score:
                status_lbl = "low_score_primary"
                reason = (f"Primary best {primary.score:.1f} < {primary_threshold} "
                          f"(pool n={len(cands_pool)})")
                best = primary
                pool_for_topn = cands_pool
            else:
                status_lbl = "low_score_fallback"
                reason = (f"Fallback best {fb.score:.1f} < {fallback_threshold} "
                          f"(pool n={len(fb_pool)})")
                best = fb
                pool_for_topn = fb_pool

            best_cand_str = ""
            if best.sackmann_match_idx is not None and not pool_for_topn.empty:
                bc = pool_for_topn.iloc[best.sackmann_match_idx]
                best_cand_str = f"{bc['winner_name']} | {bc['loser_name']}"

            top3_str = "; ".join(
                f"{pool_for_topn.iloc[i]['winner_name']}"
                f"|{pool_for_topn.iloc[i]['loser_name']}:{s}"
                for i, s in best.top3
            ) if best.top3 else ""

            unmatched_rows.append(_unmatched_row(
                td_row, "low_score",
                best_score=max(primary.score, fb.score),
                best_cand=best_cand_str,
            ))
            manual_review_rows.append(_review_row(
                td_row, status_lbl, reason,
                best_score=best.score, best_cand=best_cand_str, top3=top3_str,
            ))

    matched_df = pd.DataFrame(matched_rows)
    unmatched_df = pd.DataFrame(unmatched_rows)
    sackmann_missing_df = pd.DataFrame(sackmann_missing_rows)
    date_window_miss_df = pd.DataFrame(date_window_miss_rows)
    manual_review_df = pd.DataFrame(manual_review_rows)
    tier_mismatch_df = pd.DataFrame(tier_mismatch_log)

    audit = compute_unmatched_audit(
        matched_df, unmatched_df, sackmann_missing_df, date_window_miss_df,
        td, period_breakpoints, period_gates,
    )

    if manual_review_path is not None:
        write_manual_review(manual_review_df, manual_review_path)

    config = {
        "primary_threshold": primary_threshold,
        "fallback_threshold": fallback_threshold,
        "tourney_date_window_days": tourney_date_window_days,
        "fallback_date_window_days": fallback_date_window_days,
        "fallback_rank_top": fallback_rank_top,
        "allow_tier_mismatch": allow_tier_mismatch,
        "n_alias_rows": len(alias),
    }

    return MatchResult(
        matched=matched_df,
        unmatched=unmatched_df,
        sackmann_missing=sackmann_missing_df,
        date_window_miss=date_window_miss_df,
        manual_review=manual_review_df,
        audit_report=audit,
        tier_mismatches=tier_mismatch_df,
        config=config,
    )


def _empty_result(alias, p_thr, f_thr) -> MatchResult:
    return MatchResult(
        matched=pd.DataFrame(),
        unmatched=pd.DataFrame(),
        sackmann_missing=pd.DataFrame(),
        date_window_miss=pd.DataFrame(),
        manual_review=pd.DataFrame(),
        audit_report=pd.DataFrame(),
        tier_mismatches=pd.DataFrame(),
        config={"primary_threshold": p_thr, "fallback_threshold": f_thr,
                "n_alias_rows": len(alias)},
    )


# Row-builder helpers --------------------------------------------------------

def _matched_row(td_row, sk_row, path, score, winner_mismatch,
                 tier_expected, tier_actual):
    """Compose a row for the matched DataFrame."""
    out = td_row.to_dict()
    out.update({
        "sackmann_match_id":   int(sk_row.name),  # original Sackmann index
        "sackmann_tourney_id": str(sk_row.get("tourney_id", "")),
        "sackmann_winner_id":  int(sk_row.get("winner_id"))
            if pd.notna(sk_row.get("winner_id")) else pd.NA,
        "sackmann_loser_id":   int(sk_row.get("loser_id"))
            if pd.notna(sk_row.get("loser_id")) else pd.NA,
        "sackmann_winner_name": sk_row.get("winner_name", ""),
        "sackmann_loser_name":  sk_row.get("loser_name", ""),
        "match_path":          path,
        "match_score":         float(score),
        "winner_mismatch":     bool(winner_mismatch),
        "tier_expected":       tier_expected,
        "tier_actual":         tier_actual,
    })
    return out


def _unmatched_row(td_row, failure_mode, *, best_score, best_cand):
    out = td_row.to_dict()
    out.update({
        "failure_mode": failure_mode,
        "best_score":   float(best_score) if pd.notna(best_score) else np.nan,
        "best_sackmann_candidate": best_cand,
    })
    return out


def _review_row(td_row, match_status, reason, *,
                best_score, best_cand, top3):
    return {
        "td_source_file":        td_row.get("_source_file", ""),
        "td_date":               td_row.get("match_date"),
        "td_tournament_raw":     td_row.get("tournament", ""),
        "td_winner_raw":         td_row.get("winner_name_raw", ""),
        "td_loser_raw":          td_row.get("loser_name_raw", ""),
        "match_status":          match_status,
        "reason":                reason,
        "best_sackmann_candidate": best_cand,
        "best_score":            float(best_score) if pd.notna(best_score) else np.nan,
        "candidate_top3":        top3 if isinstance(top3, str) else "",
    }


# ===========================================================================
# Section F — Audit and reporting
# ===========================================================================

_HISTOGRAM_BUCKETS = [(-1, 50), (50, 70), (70, 80), (80, 85), (85, 200)]
_HISTOGRAM_LABELS = ["0-50", "50-70", "70-80", "80-85", "85+"]


def _score_histogram(scores: Iterable[float]) -> dict[str, int]:
    """Bucket counts for unmatched best_scores; missing → 'no_score'."""
    out = {label: 0 for label in _HISTOGRAM_LABELS}
    out["no_score"] = 0
    for s in scores:
        if pd.isna(s):
            out["no_score"] += 1
            continue
        for (lo, hi), label in zip(_HISTOGRAM_BUCKETS, _HISTOGRAM_LABELS):
            if lo < s <= hi:
                out[label] += 1
                break
    return out


def compute_unmatched_audit(
    matched: pd.DataFrame,
    unmatched: pd.DataFrame,
    sackmann_missing: pd.DataFrame,
    date_window_miss: pd.DataFrame,
    td: pd.DataFrame,
    periods: Sequence[tuple[str, int, int]] = DEFAULT_PERIODS,
    gates: dict[str, float | None] = DEFAULT_GATES,
) -> pd.DataFrame:
    """Compute per-period unmatched-rate gate report.

    Per the locked spec:
      - Denominator = ``n_with_pinnacle - n_sackmann_missing - n_date_window_miss``
      - ``unmatched`` rows count toward the rate (numerator)
      - 2009 has Pinnacle=0% so gate=None (unmatched_rate is informational)
      - winner_mismatch_rate computed within matched rows; soft gate 1%

    Adds a ``score_histogram_unmatched`` column showing unmatched best_score
    distribution as ``{0-50: n, 50-70: n, ...}`` for tuning the threshold.
    """
    rows = []
    td_yr = td["match_date"].dt.year if "match_date" in td.columns else pd.Series(dtype=int)
    has_pinnacle = td.get("psw", pd.Series(dtype=float)).notna() if not td.empty else pd.Series(dtype=bool)

    for label, lo, hi in periods:
        period_mask = td_yr.between(lo, hi)
        n_total = int(period_mask.sum())
        n_with_pinnacle = int((period_mask & has_pinnacle).sum())

        def _yr_in(df):
            if df.empty or "match_date" not in df.columns:
                return df.iloc[0:0]
            return df[df["match_date"].dt.year.between(lo, hi)]

        m = _yr_in(matched)
        u = _yr_in(unmatched)
        sk_m = _yr_in(sackmann_missing)
        dwm = _yr_in(date_window_miss)

        # Restrict denominator to with-Pinnacle rows so the gate matches the
        # eventual betting universe.
        def _has_psw(df):
            return df[df["psw"].notna()] if "psw" in df.columns and not df.empty else df

        m_p = _has_psw(m); u_p = _has_psw(u)
        sk_m_p = _has_psw(sk_m); dwm_p = _has_psw(dwm)

        gate_denom = (len(m_p) + len(u_p))   # exclude sackmann_missing + date_window_miss
        unmatched_rate = (len(u_p) / gate_denom) if gate_denom else float("nan")
        gate_thr = gates.get(label, None)
        gate_passed = (
            (unmatched_rate < gate_thr) if (gate_thr is not None and gate_denom)
            else None
        )

        # Winner-mismatch rate (within matched, with-pinnacle subset)
        if len(m_p) and "winner_mismatch" in m_p.columns:
            wm_rate = float(m_p["winner_mismatch"].sum()) / len(m_p)
        else:
            wm_rate = float("nan")

        # Score histogram on unmatched (with-pinnacle)
        scores = u_p["best_score"].tolist() if "best_score" in u_p.columns else []
        hist = _score_histogram(scores)

        n_unaliased = int((u_p["failure_mode"] == "unaliased").sum()) \
            if not u_p.empty and "failure_mode" in u_p.columns else 0
        n_low_score = int((u_p["failure_mode"] == "low_score").sum()) \
            if not u_p.empty and "failure_mode" in u_p.columns else 0
        n_fallback = int((m_p["match_path"] == "fallback").sum()) \
            if not m_p.empty and "match_path" in m_p.columns else 0

        rows.append({
            "period":                  label,
            "n_total":                 n_total,
            "n_with_pinnacle":         n_with_pinnacle,
            "n_matched":               len(m_p),
            "n_unmatched":             len(u_p),
            "n_low_score":             n_low_score,
            "n_unaliased":             n_unaliased,
            "n_sackmann_missing":      len(sk_m_p),
            "n_date_window_miss":      len(dwm_p),
            "n_fallback_used":         n_fallback,
            "n_winner_mismatch":       int(m_p["winner_mismatch"].sum())
                if len(m_p) and "winner_mismatch" in m_p.columns else 0,
            "unmatched_rate_%":        round(unmatched_rate * 100, 3)
                if not pd.isna(unmatched_rate) else float("nan"),
            "winner_mismatch_rate_%":  round(wm_rate * 100, 3)
                if not pd.isna(wm_rate) else float("nan"),
            "gate_threshold_%":        gate_thr * 100 if gate_thr is not None else None,
            "gate_passed":             gate_passed,
            "wm_soft_gate_passed":     (wm_rate < WINNER_MISMATCH_SOFT_GATE)
                if not pd.isna(wm_rate) else None,
            "score_histogram_unmatched": str(hist),
        })
    return pd.DataFrame(rows)


# ===========================================================================
# Section G — Manual review writer
# ===========================================================================

_MANUAL_REVIEW_COLS = (
    "td_source_file", "td_date", "td_tournament_raw",
    "td_winner_raw", "td_loser_raw",
    "match_status", "reason",
    "best_sackmann_candidate", "best_score", "candidate_top3",
)


def write_manual_review(rows: pd.DataFrame, path: Path) -> None:
    """Write manual_review.csv (overwrite mode; no append).

    Stable column order; sorted by (td_date, td_tournament_raw).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows.empty:
        # Write header-only file so downstream tools don't fail
        pd.DataFrame(columns=list(_MANUAL_REVIEW_COLS)).to_csv(path, index=False)
        return
    out = rows.reindex(columns=list(_MANUAL_REVIEW_COLS))
    out = out.sort_values(["td_date", "td_tournament_raw"], kind="mergesort")
    out.to_csv(path, index=False)
    log.info("write_manual_review: wrote %d rows to %s", len(out), path)
