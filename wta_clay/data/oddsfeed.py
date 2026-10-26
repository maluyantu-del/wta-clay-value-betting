"""Tennis-Data.co.uk WTA odds feed cleaning and schema unification.

Schema reality at tennis-data.co.uk (WTA files):
    * Per-year yearly file (``YYYY.xls`` pre-2013, ``YYYY.xlsx`` later, some
      years also republished as ``YYYY.csv``).
    * Column headers drift over time — bookmaker columns get added and
      removed as providers change. Early years (2005-2008) don't have
      Pinnacle at all; mid-era (2013+) add aggregate Max/Avg columns.
    * Player names are "Surname F." style (e.g. "Williams S.", "Vondrousova M.").
    * ``Comment`` = "Completed" / "Retired" / "Walkover" / etc.

This module:
    1. Loads all year files via :mod:`wta_clay.data.loaders`.
    2. Normalizes the column schema to a canonical set, case-insensitive.
    3. Parses dates robustly (Excel returns Timestamp, CSVs often DD/MM/YYYY).
    4. Normalizes round strings to an ordered short-code Categorical.
    5. Filters to clay.
    6. Drops rows where the match never happened (Walkover / missing score).
    7. Reports bookmaker coverage by year so you can see the Pinnacle gap
       (Pinnacle column is typically absent in the oldest files).

Canonical output columns (``load_tennisdata_odds`` return):
    match_date          datetime64[ns]
    tournament          string
    series_raw          string    (Tennis-Data tier name — NOT our tier)
    court               string    ("Outdoor" / "Indoor")
    surface             string
    round_raw           string    (as-shipped)
    round_norm          category  (R128 < R64 < R32 < R16 < QF < SF < F < RR)
    best_of             Int64
    winner_name_raw     string    ("Surname F.")
    loser_name_raw      string
    winner_rank         Int64
    loser_rank          Int64
    psw, psl            float64   (Pinnacle — primary)
    b365w, b365l        float64   (Bet365 — secondary)
    avgw, avgl          float64   (cross-book average — optional)
    maxw, maxl          float64   (cross-book max — optional)
    comment             string
    _source_file        string
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from wta_clay.data import loaders

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical schema
# ---------------------------------------------------------------------------

#: Canonical lower-snake column names for the final output.
CANONICAL_COLUMNS: tuple[str, ...] = (
    "match_date", "tournament", "series_raw", "court", "surface",
    "round_raw", "round_norm", "best_of",
    "winner_name_raw", "loser_name_raw",
    "winner_rank", "loser_rank",
    "psw", "psl", "b365w", "b365l",
    "avgw", "avgl", "maxw", "maxl",
    "comment", "_source_file",
)

#: Map of Tennis-Data header (case-insensitive, whitespace-stripped) → canonical.
#: Left side is already lowercased + ``.strip()``-ed in ``_canonicalize_headers``.
HEADER_ALIASES: dict[str, str] = {
    "date":        "match_date",
    "tournament":  "tournament",
    "series":      "series_raw",
    "tier":        "series_raw",            # pre-2009 files used 'Tier'
    "court":       "court",
    "surface":     "surface",
    "round":       "round_raw",
    "best of":     "best_of",
    "bestof":      "best_of",
    "winner":      "winner_name_raw",
    "loser":       "loser_name_raw",
    "wrank":       "winner_rank",
    "lrank":       "loser_rank",
    "comment":     "comment",
    "psw":         "psw",
    "psl":         "psl",
    "b365w":       "b365w",
    "b365l":       "b365l",
    "avgw":        "avgw",
    "avgl":        "avgl",
    "maxw":        "maxw",
    "maxl":        "maxl",
    "_source_file": "_source_file",
}

#: Normalized round-code ordering. Rounds not in this map map to <NA>.
_ROUND_NORM_MAP: dict[str, str] = {
    # Tennis-Data phrasing
    "1st round":       "R128",   # only in G-size draws
    "2nd round":       "R64",
    "3rd round":       "R32",
    "4th round":       "R16",
    "quarterfinals":   "QF",
    "semifinals":      "SF",
    "the final":       "F",
    "round robin":     "RR",
    # Sackmann-style short codes (defensive; users may pass Sackmann rounds)
    "r128": "R128", "r64": "R64", "r32": "R32", "r16": "R16",
    "qf": "QF", "sf": "SF", "f": "F", "rr": "RR",
}

ROUND_NORM_ORDER: tuple[str, ...] = ("R128", "R64", "R32", "R16",
                                    "QF", "SF", "F", "RR")


# ---------------------------------------------------------------------------
# Surface blacklist
# ---------------------------------------------------------------------------

#: 4-tuples ``(tournament, surface, year_from, year_to)`` that Tennis-Data.co.uk
#: has tagged incorrectly. Rows matching any entry are dropped BEFORE any
#: surface filtering in :func:`load_tennisdata_odds`.
#:
#:   - ``tournament``  exact (case-insensitive after stripping)
#:   - ``surface``     case-insensitive
#:   - ``year_from``   inclusive, or ``None`` for "no lower bound"
#:   - ``year_to``     inclusive, or ``None`` for "no upper bound"
#:
#: Each entry must cite the known-correct surface and a source note inline.
#: Add to this set rather than post-filtering downstream so all consumers
#: see the same cleaned view.
#: Lower bound on implied vig: rows where ``1/win_odds + 1/lose_odds`` is below
#: this threshold are physically impossible (would imply a guaranteed-arbitrage
#: bookie price). Such rows are scraping/typo errors and must be dropped.
#: The 0.99 cutoff allows for tiny negative vig from price tick noise (e.g.
#: 1.95/2.10 implied 0.989) while catching genuine errors (the worst observed
#: real bug had implied 0.380 — odds_p1=52.0 / odds_p2=2.77 on 2020-09-21).
VIG_SANITY_MIN: float = 0.99


SURFACE_BLACKLIST: frozenset[tuple[str, str, int | None, int | None]] = frozenset({
    # Open de Rouen is an indoor HARD court event. TD tags it Clay 2024-2025.
    # Source: WTA Rouen Open official page + Wikipedia.
    ("Open de Rouen", "Clay", None, None),

    # WTA Istanbul Cup was outdoor clay 2005-2014, switched to Hard 2015+.
    # 2016 outdoor clay edition is REAL (TEB BNP Paribas branding) and kept;
    # 2017+ rows tagged Clay are TD mis-tags (the event was Hard those years).
    # Source: WTA Istanbul Cup history; verified via 2017+ raw shows Outdoor
    # tag but field of players consistent with the post-2014 Hard era.
    ("Istanbul Cup", "Clay", 2017, None),
    # Defensive: same blacklist for the alternate sponsor name in case TD
    # ever uses it for 2017+ (currently only 2016 row exists for this string).
    ("TEB BNP Paribas Istanbul Cup", "Clay", 2017, None),

    # Acapulco "Abierto Mexicano" was clay 2005-2013; switched to Hard in
    # 2014, WTA withdrew 2015. TD rows 2014-2020 tagged Clay are mis-tags
    # (the men's tournament continued; TD likely scraped from a single
    # surface field that didn't update post-switch).
    ("Abierto Mexicano", "Clay", 2014, None),
})


def _apply_vig_sanity(
    df: pd.DataFrame,
    *,
    threshold: float = VIG_SANITY_MIN,
) -> pd.DataFrame:
    """Drop rows whose implied vig sits below ``threshold`` for any present
    bookmaker pair (Pinnacle ``psw/psl`` or Bet365 ``b365w/b365l``).

    Implied vig = ``1/win_odds + 1/lose_odds``.

    Theory: a fair market always has implied > 1.0 (the bookmaker's
    overround). Implied < 1.0 means the offered prices guarantee an
    arbitrage profit when stacked — physically impossible at any
    real-world book. Rows with implied < 0.99 are essentially always
    scraping errors or typos and must be filtered before they pollute
    backtesting (one such row with implied 0.380 was caught contributing
    +51u PnL on a single 1u flat-stake bet, distorting ROI by +4-5pp).

    Filter logic:
      - check Pinnacle pair: drop if both present AND implied < threshold
      - check Bet365 pair:   drop if both present AND implied < threshold
      - rows with neither pair fully present are kept (no signal)

    Returns a new DataFrame with index reset.
    """
    keep = pd.Series(True, index=df.index)
    drops_by_book: dict[str, int] = {}
    bad_rows: list[dict] = []

    for prefix, w_col, l_col in (
        ("pinnacle", "psw", "psl"),
        ("bet365",   "b365w", "b365l"),
    ):
        if w_col not in df.columns or l_col not in df.columns:
            continue
        w = pd.to_numeric(df[w_col], errors="coerce")
        l = pd.to_numeric(df[l_col], errors="coerce")
        present = w.notna() & l.notna() & (w > 0) & (l > 0)
        impl = 1.0 / w + 1.0 / l
        bad = present & (impl < threshold)
        n = int(bad.sum())
        if n:
            drops_by_book[prefix] = n
            for idx in df.index[bad]:
                bad_rows.append({
                    "match_date": df.at[idx, "match_date"]
                                  if "match_date" in df.columns else None,
                    "tournament": df.at[idx, "tournament"]
                                  if "tournament" in df.columns else None,
                    "winner_name_raw":
                        df.at[idx, "winner_name_raw"]
                        if "winner_name_raw" in df.columns else None,
                    "loser_name_raw":
                        df.at[idx, "loser_name_raw"]
                        if "loser_name_raw" in df.columns else None,
                    "book":      prefix,
                    "win_odds":  float(w.at[idx]),
                    "lose_odds": float(l.at[idx]),
                    "implied":   float(impl.at[idx]),
                })
        keep &= ~bad

    n_dropped = int((~keep).sum())
    if n_dropped:
        log.info("apply_vig_sanity: dropped %d rows (threshold=%.3f)",
                 n_dropped, threshold)
        for book, n in drops_by_book.items():
            log.info("  - %s vig < %.3f: %d rows", book, threshold, n)
        # Show up to 10 example rows so the user can spot-verify
        for r in bad_rows[:10]:
            d = r.get("match_date")
            log.info("  drop: %s | %s | %s vs %s | %s implied=%.4f "
                     "(odds=%.2f / %.2f)",
                     d.date() if hasattr(d, "date") else d,
                     r.get("tournament"),
                     r.get("winner_name_raw"), r.get("loser_name_raw"),
                     r["book"], r["implied"], r["win_odds"], r["lose_odds"])
    return df.loc[keep].reset_index(drop=True)


def _apply_surface_blacklist(
    df: pd.DataFrame,
    blacklist: frozenset[tuple[str, str, int | None, int | None]] = SURFACE_BLACKLIST,
    tournament_col: str = "tournament",
    surface_col: str = "surface",
    date_col: str = "match_date",
) -> pd.DataFrame:
    """Drop rows matching any ``(tournament, surface, year_from, year_to)``.

    Match details:
      - tournament: case-insensitive, whitespace-stripped exact match
      - surface:    case-insensitive
      - year:       inclusive; ``None`` on either side means "unbounded"
      - rows with NaT match_date are NEVER blacklisted (defensive)

    Returns a new DataFrame with index reset.
    """
    if not blacklist:
        return df.copy()

    t = df[tournament_col].astype("string").str.strip().str.lower().fillna("")
    s = df[surface_col].astype("string").str.strip().str.lower().fillna("")
    if date_col in df.columns:
        yr = df[date_col].dt.year
    else:
        yr = pd.Series([None] * len(df), index=df.index, dtype="Int64")

    keep = pd.Series(True, index=df.index)
    drops_by_key: dict[tuple, int] = {}
    for tn, sf, yf, yto in blacklist:
        tn_n = tn.strip().lower()
        sf_n = sf.strip().lower()
        match = (t == tn_n) & (s == sf_n)
        if yf is not None:
            match &= yr.notna() & (yr >= yf)
        if yto is not None:
            match &= yr.notna() & (yr <= yto)
        n = int(match.sum())
        if n:
            drops_by_key[(tn, sf, yf, yto)] = n
        keep &= ~match

    n_dropped = int((~keep).sum())
    if n_dropped:
        log.info("apply_surface_blacklist: dropped %d rows", n_dropped)
        for (tn, sf, yf, yto), n in drops_by_key.items():
            yr_desc = (
                f"{yf}-{yto}" if (yf is not None and yto is not None)
                else f"{yf}+" if yf is not None
                else f"<={yto}" if yto is not None
                else "all years"
            )
            log.info("  - %r on %s (%s): %d rows", tn, sf, yr_desc, n)
    return df.loc[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------------

def _canonicalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical snake_case, case-insensitive match.

    Unknown columns are dropped from the result silently but logged at DEBUG
    level. We log an INFO line summarizing which canonical columns were
    missing in this file — useful for bookmaker-coverage auditing.
    """
    norm_to_orig: dict[str, str] = {}
    for orig in df.columns:
        key = str(orig).strip().lower()
        if key in HEADER_ALIASES:
            canon = HEADER_ALIASES[key]
            # Tennis-Data has no true dup columns, but if it ever did the
            # last-one-wins; warn in that case.
            if canon in norm_to_orig:
                log.warning("Duplicate canonical column %s (from %r and %r); "
                            "keeping latter", canon, norm_to_orig[canon], orig)
            norm_to_orig[canon] = orig

    out = df[list(norm_to_orig.values())].copy()
    out.columns = list(norm_to_orig.keys())

    missing = [c for c in CANONICAL_COLUMNS if c not in out.columns
               and c not in ("round_norm",)]  # round_norm derived later
    if missing:
        log.debug("Canonical columns missing from this block: %s", missing)
    return out


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

def parse_dates(series: pd.Series) -> pd.Series:
    """Robustly parse Tennis-Data 'Date' column.

    Excel files (.xls/.xlsx read via pandas) return a Timestamp already.
    CSVs store the column as strings that could be DD/MM/YYYY, YYYY-MM-DD,
    or occasionally M/D/YY. We try in order:

        1. pass-through if already datetime-like
        2. dayfirst=True (European) parse
        3. pandas default fallback

    Unparseable values become NaT.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")

    # Try dayfirst first (matches the DD/MM/YYYY convention Tennis-Data uses)
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
    # If dayfirst produced too many NaT relative to the input, retry with
    # dayfirst=False — catches ISO strings like "2019-05-20".
    if parsed.isna().sum() > series.notna().sum() * 0.5:
        retry = pd.to_datetime(series, errors="coerce", dayfirst=False)
        if retry.notna().sum() > parsed.notna().sum():
            parsed = retry
    return parsed


# ---------------------------------------------------------------------------
# Round normalization
# ---------------------------------------------------------------------------

def normalize_round(series: pd.Series) -> pd.Series:
    """Map free-text round strings to ordered short codes.

    Unknown values become ``<NA>`` — callers can inspect via
    ``audit_unknown_rounds`` to decide whether to add more aliases.
    """
    lower = series.astype("string").str.strip().str.lower()
    mapped = lower.map(_ROUND_NORM_MAP)
    out = pd.Categorical(mapped, categories=ROUND_NORM_ORDER, ordered=True)
    return pd.Series(out, index=series.index, name=series.name)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def filter_clay_odds(df: pd.DataFrame, surface_col: str = "surface"
                     ) -> pd.DataFrame:
    """Keep only Clay rows (case-insensitive)."""
    mask = df[surface_col].astype("string").str.lower() == "clay"
    out = df.loc[mask].reset_index(drop=True)
    log.info("filter_clay_odds: %d / %d rows kept", len(out), len(df))
    return out


def drop_walkovers(df: pd.DataFrame, comment_col: str = "comment"
                   ) -> pd.DataFrame:
    """Drop rows where the match never actually happened.

    Tennis-Data's ``Comment`` column contains "Walkover" for forfeits and
    (rarely) empty / non-standard values for matches that were voided. We
    drop Walkover only — "Retired" matches did play at least one game and
    provide a valid result for our modeling target, so we keep them.
    """
    if comment_col not in df.columns:
        return df.copy()
    raw = df[comment_col].astype("string").str.strip().str.lower()
    mask = raw != "walkover"
    out = df.loc[mask].reset_index(drop=True)
    n_dropped = (~mask).sum()
    if n_dropped:
        log.info("drop_walkovers: dropped %d walkover rows", int(n_dropped))
    return out


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def load_tennisdata_odds(
    tennisdata_dir: Path,
    year_from: int = 2005,
    year_to: int | None = None,
    *,
    clay_only: bool = True,
    drop_walkover: bool = True,
) -> pd.DataFrame:
    """End-to-end load + normalization of Tennis-Data WTA odds files.

    Pipeline::

        discover files → read_many_tennisdata → canonicalize headers
        → parse_dates → normalize_round → ensure numeric odds
        → (optional) filter_clay_odds → (optional) drop_walkovers
        → stable sort by (match_date, tournament, round_norm)
    """
    files = loaders.discover_tennisdata_files(
        tennisdata_dir, year_from=year_from, year_to=year_to
    )
    if not files:
        raise FileNotFoundError(
            f"No Tennis-Data files found in {tennisdata_dir} "
            f"for {year_from}–{year_to}"
        )

    # Canonicalize headers PER-FILE before concat so that schema drift
    # across years (e.g. 'Tier' → 'Series') collapses cleanly into
    # a single ``series_raw`` column instead of two parallel columns.
    frames = []
    for p in files:
        raw = loaders.read_tennisdata_file(p)
        canon_frame = _canonicalize_headers(raw)
        # Fill any canonical columns this file happens to lack with NA so
        # concat produces a uniform schema.
        for col in CANONICAL_COLUMNS:
            if col not in canon_frame.columns and col != "round_norm":
                canon_frame[col] = pd.NA
        frames.append(canon_frame)
    canon = pd.concat(frames, ignore_index=True, sort=False)

    # Dtype coercion — odds + ranks are numeric; names / comments are string.
    numeric_cols = ["best_of", "winner_rank", "loser_rank",
                    "psw", "psl", "b365w", "b365l",
                    "avgw", "avgl", "maxw", "maxl"]
    for c in numeric_cols:
        canon[c] = pd.to_numeric(canon[c], errors="coerce")

    # Integer columns use nullable Int64 so missing bookie ranks stay NA.
    for c in ("best_of", "winner_rank", "loser_rank"):
        canon[c] = canon[c].astype("Int64")

    string_cols = ["tournament", "series_raw", "court", "surface",
                   "round_raw", "winner_name_raw", "loser_name_raw",
                   "comment", "_source_file"]
    for c in string_cols:
        canon[c] = canon[c].astype("string")

    canon["match_date"] = parse_dates(canon["match_date"])
    canon["round_norm"] = normalize_round(canon["round_raw"])

    # Surface-tagging corrections (e.g. Rouen tagged as Clay by TD).
    # Applied BEFORE filter_clay_odds so blacklisted rows never enter the
    # clay dataset even if ``clay_only=False`` downstream would reload them.
    canon = _apply_surface_blacklist(canon)

    if clay_only:
        canon = filter_clay_odds(canon)
    if drop_walkover:
        canon = drop_walkovers(canon)

    # Drop rows with physically-impossible implied vig (< 0.99). These are
    # scraping / typo errors. See ``_apply_vig_sanity`` for the rationale.
    canon = _apply_vig_sanity(canon)

    canon = canon[list(CANONICAL_COLUMNS)].copy()
    canon = canon.sort_values(
        ["match_date", "tournament", "round_norm"],
        kind="mergesort",
    ).reset_index(drop=True)

    log.info("load_tennisdata_odds: %d rows, %s → %s", len(canon),
             canon["match_date"].min(), canon["match_date"].max())
    return canon


# ---------------------------------------------------------------------------
# Audit helpers
# ---------------------------------------------------------------------------

def audit_bookmaker_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Coverage % of Pinnacle / Bet365 / Avg / Max odds by year.

    Returns a DataFrame indexed by year with columns
    ``n, pinnacle_%, b365_%, avg_%, max_%`` — useful to see the Pinnacle
    gap in early years at a glance.
    """
    if df.empty:
        return pd.DataFrame()
    yr = df["match_date"].dt.year
    grp = df.groupby(yr)

    def pct(col: str) -> pd.Series:
        return grp[col].apply(lambda s: 100.0 * s.notna().mean())

    out = pd.DataFrame({
        "n":          grp.size(),
        "pinnacle_%": pct("psw").round(1),
        "b365_%":     pct("b365w").round(1),
        "avg_%":      pct("avgw").round(1),
        "max_%":      pct("maxw").round(1),
    })
    out.index.name = "year"
    return out


def audit_unknown_rounds(df: pd.DataFrame) -> pd.Series:
    """Return value_counts of raw round strings that didn't normalize.

    Use after ``load_tennisdata_odds`` to decide whether to extend
    ``_ROUND_NORM_MAP`` with additional aliases.
    """
    mask = df["round_norm"].isna() & df["round_raw"].notna()
    return df.loc[mask, "round_raw"].value_counts()


def audit_surface_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Find tournaments that appear under >1 surface across years.

    Must be called on the FULL Tennis-Data frame (``clay_only=False``)
    for it to be meaningful — running on clay-filtered data is trivially
    single-surface.

    Returns a DataFrame indexed by tournament with:
      - one column per surface (counts)
      - ``majority_surface``  : str
      - ``minority_n``        : int
      - ``minority_years``    : list[int]

    Rows are sorted by ``minority_n`` descending — the ones most likely
    to be TD tagging errors appear first. A human should review each and
    either:
      (a) add the bad (tournament, surface) pair to SURFACE_BLACKLIST, or
      (b) confirm the split is legitimate (e.g. Stuttgart moved from hard
          to clay in 2009 — both tags are historically correct).
    """
    if df.empty or "tournament" not in df.columns or "surface" not in df.columns:
        return pd.DataFrame()

    counts = (df.groupby(["tournament", "surface"]).size()
                .unstack(fill_value=0))
    multi = counts[(counts > 0).sum(axis=1) > 1].copy()
    if multi.empty:
        return multi

    def _majority_and_minority(row: pd.Series) -> pd.Series:
        ordered = row.sort_values(ascending=False)
        maj = ordered.index[0]
        minority_n = int(ordered.iloc[1:].sum())
        return pd.Series({"majority_surface": maj, "minority_n": minority_n})

    extra = multi.apply(_majority_and_minority, axis=1)
    multi = multi.join(extra)

    # Collect the years each minority surface appears in, per tournament.
    yr = df["match_date"].dt.year
    minority_years_map: dict[str, list[int]] = {}
    for tourney in multi.index:
        sub = df[df["tournament"] == tourney]
        maj = multi.loc[tourney, "majority_surface"]
        minority_rows = sub[sub["surface"] != maj]
        minority_years_map[tourney] = sorted(
            int(y) for y in yr.loc[minority_rows.index].dropna().unique()
        )
    multi["minority_years"] = pd.Series(minority_years_map)

    return multi.sort_values("minority_n", ascending=False)
