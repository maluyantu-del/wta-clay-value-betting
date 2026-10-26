"""Sackmann tennis_wta cleaning and clay-surface filtering.

This module sits on top of :mod:`wta_clay.data.loaders` and produces the
canonical Sackmann-side DataFrames used by everything downstream.

Key transformations:

1. ``tourney_date`` integers (``YYYYMMDD``) → ``pd.Timestamp``.
2. Six-tier classification stored in ``tour_level`` column:
   ``{GS, WTA1000, WTA500, WTA250, WTA125, ITF}``.
3. Sample weights attached per your spec:
   GS / WTA1000 / WTA500 / WTA250 = 1.0, WTA125 = 0.5, ITF = 0.25.
4. Premier 5 whitelist by ``tourney_name`` forces ``WTA1000`` over the
   raw Sackmann ``P`` / ``PM`` code (Premier 5 events were coded as
   plain ``P`` pre-2020).
5. ``D`` (Fed Cup / BJK Cup) rows are filtered out entirely.
6. Olympics (``O``) rows are kept but flagged ``not_for_betting=True``
   and get ``WTA125`` weight (0.5).
7. Qual/ITF files are read separately and stacked with ``tour_level=ITF``
   and ``sample_weight=0.25``.

Downstream modules should never look at the raw ``tourney_level`` code.
Use the derived ``tour_level`` column.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from wta_clay.data import loaders

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Classification tables  (exported so callers can override)
# ---------------------------------------------------------------------------

#: Canonical six-tier enumeration. Order matters: used as pandas Categorical.
TOUR_LEVELS = ("GS", "WTA1000", "WTA500", "WTA250", "WTA125", "ITF")

#: Sample weight applied during training, keyed by ``tour_level``.
SAMPLE_WEIGHTS: dict[str, float] = {
    "GS": 1.0,
    "WTA1000": 1.0,
    "WTA500": 1.0,
    "WTA250": 1.0,
    "WTA125": 0.5,
    "ITF": 0.25,
}

#: Mapping from raw Sackmann ``tourney_level`` code → canonical level.
#: ``None`` means "consult tourney_name / Premier 5 whitelist".
#: ``"_EXCLUDE"`` means drop the row entirely (Fed Cup / BJK Cup).
RAW_LEVEL_MAP: dict[str, str] = {
    # Modern era (2009+)
    "G":  "GS",
    "PM": "WTA1000",
    "P":  "WTA500",      # default; Premier 5 names upgraded via whitelist below
    "I":  "WTA250",
    "W":  "WTA125",
    "C":  "WTA125",      # WTA 125K Challenger series (tourney_id prefix W-CH)
    "D":  "_EXCLUDE",    # Fed Cup / BJK Cup
    "O":  "WTA125",      # Olympics: weight like WTA125, but not_for_betting
    "F":  "WTA1000",     # Year-end Finals (defensive; not on clay)
    # Pre-2009 Tier system
    "T1": "WTA1000",
    "T2": "WTA500",
    "T3": "WTA250",
    "T4": "WTA250",
    "T5": "WTA125",
}

#: Tournaments that were "Premier 5" (WTA1000-equivalent) but coded ``P``
#: in Sackmann prior to the 2021 rebrand. Match against ``tourney_name``
#: via case-insensitive substring.
PREMIER_5_NAME_WHITELIST: tuple[str, ...] = (
    "Doha",
    "Dubai",
    "Rome",           # Italian Open / Internazionali BNL — CLAY
    "Cincinnati",
    "Montreal",
    "Toronto",
    "Wuhan",
    # Alternate canonical names Sackmann has used:
    "Canadian Open",
    "Rogers Cup",
)

#: Raw Sackmann level codes eligible for ``TOURNEY_NAME_OVERRIDES``.
#: We INCLUDE ``W`` and ``C`` (WTA 125 codes) because Sackmann sometimes
#: mis-tags real WTA 250 events as 'W' (e.g. Strasbourg 2021 — verified
#: as a Sackmann data bug). Numeric ITF codes (10/15/25/etc.) and special
#: codes (D = Fed Cup, O = Olympics, F = Finals) remain excluded so that
#: ITF "<city> 25K" rows are NEVER promoted via name+year alias.
#:
#: The Premier 5 name whitelist still uses the narrower ``{P, PM}`` check
#: and is unaffected by this set.
_PROMOTABLE_RAW_LEVELS: frozenset[str] = frozenset(
    {"G", "PM", "P", "I", "W", "C", "T1", "T2", "T3", "T4", "T5"}
)

#: Manual tier corrections keyed by ``(name_substring, (year_from, year_to))``.
#: Applied AFTER the Premier 5 whitelist as the final authority.
#: Name match is case-insensitive substring; year range is inclusive.
#:
#: Safety: only rows whose raw Sackmann code is in ``_PROMOTABLE_RAW_LEVELS``
#: can be promoted. ITF / WTA125 / Olympics / Fed Cup / Finals rows whose
#: tourney_name happens to contain the substring are never touched.
#:
#: Historical bugs we correct here:
#:   - Madrid 2016: Sackmann coded ``P``, but it was Premier Mandatory.
#:   - Madrid 2021 / 2022: same coding bug.
#: All three years were actual WTA1000-equivalent events.
TOURNEY_NAME_OVERRIDES: dict[tuple[str, tuple[int, int]], str] = {
    ("Madrid", (2016, 2016)): "WTA1000",
    ("Madrid", (2021, 2022)): "WTA1000",
    # Strasbourg 2021: Sackmann wrongly tagged as 'W' (WTA125). Real event
    # was WTA 250 (32-draw, late-May Roland-Garros lead-in). Champion was
    # Barbora Krejcikova (top-30); draw included Andreescu (#7), Pegula
    # (#29), Alexandrova (#33). Verified 2026-04-21 against Sackmann
    # 2021-406 raw rows.
    ("Strasbourg", (2021, 2021)): "WTA250",
}


def _matches_premier5_name(name: str) -> bool:
    if not isinstance(name, str):
        return False
    lower = name.lower()
    return any(kw.lower() in lower for kw in PREMIER_5_NAME_WHITELIST)


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

def parse_tourney_date(series: pd.Series) -> pd.Series:
    """Convert ``YYYYMMDD`` integers (or strings) to ``pd.Timestamp``.

    Invalid / missing values become ``NaT``. The Sackmann schema stores
    ``tourney_date`` as int; a minority of rows in very old files are
    strings, so we coerce via ``errors='coerce'``.
    """
    s = pd.to_datetime(series.astype("Int64").astype("string"),
                       format="%Y%m%d", errors="coerce")
    return s


# ---------------------------------------------------------------------------
# Tier classification
# ---------------------------------------------------------------------------

def classify_tour_level(
    df: pd.DataFrame,
    raw_level_col: str = "tourney_level",
    tourney_name_col: str = "tourney_name",
    tourney_date_col: str = "tourney_date",
    *,
    level_map: dict[str, str] | None = None,
    premier5_whitelist: tuple[str, ...] | None = None,
    name_overrides: dict[tuple[str, tuple[int, int]], str] | None = None,
) -> pd.DataFrame:
    """Derive a canonical ``tour_level`` column and drop excluded rows.

    Returns a new DataFrame. Adds columns:
        tour_level      (category, ordered per TOUR_LEVELS)
        sample_weight   (float)
        not_for_betting (bool)  # True for Olympics / anything outside WTA tour
        raw_level       (string, copy of original for audit)

    Resolution order (later steps override earlier):
        1. raw code via ``level_map`` (ITF fallback for unknowns)
        2. Premier 5 name whitelist (only upgrades raw ``P`` / ``PM`` → WTA1000)
        3. ``TOURNEY_NAME_OVERRIDES`` (year-scoped manual corrections)

    Rows with raw level marked ``_EXCLUDE`` (Fed Cup / BJK Cup) are dropped.
    """
    lm = dict(RAW_LEVEL_MAP if level_map is None else level_map)
    whitelist = PREMIER_5_NAME_WHITELIST if premier5_whitelist is None \
        else premier5_whitelist
    overrides = TOURNEY_NAME_OVERRIDES if name_overrides is None \
        else name_overrides

    out = df.copy()
    raw = out[raw_level_col].astype("string").str.strip().str.upper()
    out["raw_level"] = raw

    # Step 1: raw code → canonical, ITF fallback for unknowns
    mapped = raw.map(lm).fillna("ITF")

    unknown = raw[~raw.isin(list(lm.keys())) & raw.notna()]
    if len(unknown):
        log.warning(
            "classify_tour_level: %d rows with unknown raw level(s) %s "
            "— defaulted to ITF",
            len(unknown), sorted(unknown.unique().tolist()),
        )

    # Step 2: Premier 5 name whitelist — upgrade P/PM to WTA1000
    name_lower = out[tourney_name_col].astype("string").str.lower().fillna("")
    is_p5 = name_lower.apply(
        lambda s: any(kw.lower() in s for kw in whitelist)
    )
    premier_eligible = raw.isin({"P", "PM"})
    upgrade_mask = is_p5 & premier_eligible
    mapped = mapped.where(~upgrade_mask, "WTA1000")

    # Step 3: name-scoped year-range overrides (final authority for WTA tour).
    #
    # SAFETY: overrides ONLY fire on rows whose raw Sackmann code is in
    # ``_PROMOTABLE_RAW_LEVELS``. This guards against ITF/WTA125/Olympics/
    # Fed Cup rows that happen to contain a city name like "Madrid" being
    # wrongly promoted to WTA1000.
    year = (pd.to_numeric(out[tourney_date_col], errors="coerce") // 10000)
    promotable = raw.isin(_PROMOTABLE_RAW_LEVELS)
    for (name_sub, (yr_from, yr_to)), forced in overrides.items():
        if forced not in TOUR_LEVELS and forced != "_EXCLUDE":
            raise ValueError(
                f"TOURNEY_NAME_OVERRIDES value {forced!r} is not a valid tier"
            )
        name_hit = name_lower.str.contains(name_sub.lower(), na=False,
                                           regex=False)
        year_hit = year.between(yr_from, yr_to)
        hit = name_hit & year_hit & promotable
        if hit.any():
            mapped = mapped.where(~hit, forced)
            log.info(
                "TOURNEY_NAME_OVERRIDES: %r (%d-%d) applied to %d rows → %s",
                name_sub, yr_from, yr_to, int(hit.sum()), forced,
            )

    # Drop EXCLUDED rows
    keep = mapped != "_EXCLUDE"
    dropped_n = (~keep).sum()
    if dropped_n:
        log.info("classify_tour_level: dropping %d rows (Fed Cup / BJK Cup)",
                 dropped_n)
    out = out.loc[keep].copy()
    mapped = mapped.loc[keep]

    out["tour_level"] = pd.Categorical(mapped, categories=TOUR_LEVELS,
                                       ordered=True)
    out["sample_weight"] = out["tour_level"].astype(str).map(SAMPLE_WEIGHTS)
    out["not_for_betting"] = raw.loc[keep].eq("O").fillna(False)

    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def load_sackmann_matches(
    sackmann_dir: Path,
    year_from: int = 2005,
    year_to: int | None = None,
    include_qual_itf: bool = True,
) -> pd.DataFrame:
    """Load and normalize Sackmann matches (main + optional qual/ITF).

    Pipeline:
        1. Read main-draw CSVs for the year range.
        2. If ``include_qual_itf``, also read qual/ITF CSVs and force
           ``tour_level = ITF`` / ``sample_weight = 0.25`` on them.
        3. Classify main-draw rows via ``classify_tour_level``.
        4. Parse tourney_date into ``match_date`` timestamp column.
        5. Stable sort by (match_date, tourney_id, match_num).

    The returned DataFrame preserves all original Sackmann columns and adds:
        match_date (datetime64[ns])
        tour_level (Categorical over TOUR_LEVELS)
        sample_weight (float)
        not_for_betting (bool)
        raw_level (string)
    """
    main_files = loaders.discover_sackmann_match_files(
        sackmann_dir, year_from=year_from, year_to=year_to
    )
    if not main_files:
        raise FileNotFoundError(
            f"No wta_matches_YYYY.csv in {sackmann_dir} for {year_from}–{year_to}"
        )
    main_raw = loaders.read_many_csv(main_files)
    main_raw["_block"] = "main"

    if include_qual_itf:
        qual_files = loaders.discover_sackmann_qual_itf_files(
            sackmann_dir, year_from=year_from, year_to=year_to
        )
        if qual_files:
            qual_raw = loaders.read_many_csv(qual_files)
            qual_raw["_block"] = "qual_itf"
            combined = pd.concat([main_raw, qual_raw], ignore_index=True,
                                 sort=False)
        else:
            log.info("No qual/ITF files found; proceeding with main draw only.")
            combined = main_raw
    else:
        combined = main_raw

    # Route ALL rows through classify_tour_level so that qual/ITF files'
    # W/C/ITF codes get the same canonical tiering (W → WTA125, etc.).
    out = classify_tour_level(combined)

    out["match_date"] = parse_tourney_date(out["tourney_date"])
    out = out.sort_values(
        ["match_date", "tourney_id", "match_num"], kind="mergesort"
    ).reset_index(drop=True)

    log.info("Loaded %d Sackmann matches (%s → %s)", len(out),
             out["match_date"].min(), out["match_date"].max())
    return out


def filter_clay(df: pd.DataFrame, surface_col: str = "surface") -> pd.DataFrame:
    """Keep only Clay rows. Case-insensitive match."""
    mask = df[surface_col].astype("string").str.lower() == "clay"
    out = df.loc[mask].reset_index(drop=True)
    log.info("filter_clay: kept %d / %d rows", len(out), len(df))
    return out


# ---------------------------------------------------------------------------
# Players and rankings
# ---------------------------------------------------------------------------

def load_players(sackmann_dir: Path) -> pd.DataFrame:
    """Load wta_players.csv and return with parsed DOB.

    Adds ``full_name`` = ``name_first + ' ' + name_last`` (used by the
    name_matcher module).
    """
    path = loaders.sackmann_players_path(sackmann_dir)
    df = loaders.read_csv_robust(path, dtype={"player_id": "Int64",
                                              "hand": "string",
                                              "ioc": "string"})
    if "dob" in df.columns:
        df["dob"] = pd.to_datetime(
            df["dob"].astype("Int64").astype("string"),
            format="%Y%m%d", errors="coerce",
        )
    df["full_name"] = (
        df["name_first"].astype("string").fillna("") + " " +
        df["name_last"].astype("string").fillna("")
    ).str.strip()
    return df


def load_rankings(sackmann_dir: Path) -> pd.DataFrame:
    """Load all wta_rankings_*.csv files, concat, parse dates.

    Schema (Sackmann):
        ranking_date (YYYYMMDD int), rank (int), player (player_id int), points
    Returns dtypes standardized + ``ranking_date`` as Timestamp.
    """
    files = loaders.discover_sackmann_ranking_files(sackmann_dir)
    if not files:
        raise FileNotFoundError(f"No wta_rankings_*.csv in {sackmann_dir}")
    df = loaders.read_many_csv(
        files,
        dtype={"rank": "Int64", "player": "Int64", "points": "Int64"},
    )
    df["ranking_date"] = pd.to_datetime(
        df["ranking_date"].astype("Int64").astype("string"),
        format="%Y%m%d", errors="coerce",
    )
    df = df.rename(columns={"player": "player_id"})
    df = df.sort_values(["player_id", "ranking_date"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def describe_tour_levels(df: pd.DataFrame, by_year: bool = True) -> pd.DataFrame:
    """Human-readable audit of the tier classification.

    If ``by_year`` is True, returns a pivot of raw_level × year; else a
    flat counts table grouped by (raw_level, tour_level, not_for_betting).
    Useful to call after ``classify_tour_level`` to verify the mapping.
    """
    if by_year:
        if "match_date" in df.columns:
            yr = df["match_date"].dt.year
        else:
            yr = (df["tourney_date"] // 10000).astype("Int64")
        return pd.crosstab(df["raw_level"], yr)
    return (df.groupby(
                ["raw_level", "tour_level", "not_for_betting"], observed=True,
                dropna=False,
            )
            .size()
            .rename("n")
            .reset_index()
            .sort_values("n", ascending=False))
