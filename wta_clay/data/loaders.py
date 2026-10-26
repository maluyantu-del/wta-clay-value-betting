"""Low-level file readers for raw data sources.

This module is intentionally *source-agnostic* below the helper layer:
it discovers files and reads them into DataFrames, but does NOT apply
any cleaning, renaming, or filtering. Source-specific logic lives in
:mod:`wta_clay.data.sackmann` and :mod:`wta_clay.data.oddsfeed`.

Data sources assumed on disk:

1. Sackmann ``tennis_wta`` repo checked out under ``<raw>/sackmann/``.
   Relevant files::

       wta_matches_YYYY.csv             # main tour (main draw)
       wta_matches_qual_itf_YYYY.csv    # qualifying + ITF (for sample weighting)
       wta_players.csv
       wta_rankings_00s.csv / 10s.csv / 20s.csv / current.csv

2. Tennis-Data.co.uk WTA odds files under ``<raw>/tennis_data/``.
   Naming is just the year: ``2005.xls``, ``2019.xlsx``, ``2024.xlsx`` etc.
   Older years are ``.xls`` (needs ``xlrd==1.2.0`` or a converted ``.csv``).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path discovery
# ---------------------------------------------------------------------------

_SACKMANN_MAIN_RE = re.compile(r"wta_matches_(\d{4})\.csv$")
_SACKMANN_QUAL_ITF_RE = re.compile(r"wta_matches_qual_itf_(\d{4})\.csv$")
_SACKMANN_FUTURES_RE = re.compile(r"wta_matches_futures_(\d{4})\.csv$")
_TENNISDATA_RE = re.compile(r"(\d{4})\.(xls|xlsx|csv)$", re.IGNORECASE)


def _discover_by_regex(
    directory: Path,
    regex: re.Pattern,
    year_from: int,
    year_to: int | None,
) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    paths: list[tuple[int, Path]] = []
    for p in directory.iterdir():
        m = regex.search(p.name)
        if not m:
            continue
        year = int(m.group(1))
        if year < year_from:
            continue
        if year_to is not None and year > year_to:
            continue
        paths.append((year, p))
    paths.sort(key=lambda t: t[0])
    return [p for _, p in paths]


def discover_sackmann_match_files(
    sackmann_dir: Path,
    year_from: int = 2005,
    year_to: int | None = None,
) -> list[Path]:
    """Main-draw WTA match CSVs: ``wta_matches_YYYY.csv``."""
    return _discover_by_regex(sackmann_dir, _SACKMANN_MAIN_RE, year_from, year_to)


def discover_sackmann_qual_itf_files(
    sackmann_dir: Path,
    year_from: int = 2005,
    year_to: int | None = None,
) -> list[Path]:
    """Qualifying + ITF match CSVs. Used with reduced sample weight in training."""
    return _discover_by_regex(
        sackmann_dir, _SACKMANN_QUAL_ITF_RE, year_from, year_to
    )


def discover_sackmann_ranking_files(sackmann_dir: Path) -> list[Path]:
    """Return all ``wta_rankings_*.csv`` files found (decades + current)."""
    if not sackmann_dir.exists():
        raise FileNotFoundError(f"Directory not found: {sackmann_dir}")
    out = sorted(sackmann_dir.glob("wta_rankings_*.csv"))
    return out


def sackmann_players_path(sackmann_dir: Path) -> Path:
    """Return the path to ``wta_players.csv`` (raise if missing)."""
    p = sackmann_dir / "wta_players.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing players file: {p}")
    return p


def discover_tennisdata_files(
    tennisdata_dir: Path,
    year_from: int = 2005,
    year_to: int | None = None,
) -> list[Path]:
    """Tennis-Data.co.uk yearly odds files: ``YYYY.{xls,xlsx,csv}``.

    If multiple extensions exist for the same year (e.g. a manually-converted
    ``.csv`` alongside the original ``.xls``), the CSV is preferred — CSVs
    are faster and avoid the xlrd-version dance for legacy ``.xls``.
    """
    if not tennisdata_dir.exists():
        raise FileNotFoundError(f"Directory not found: {tennisdata_dir}")

    # Group by year, pick best extension
    by_year: dict[int, dict[str, Path]] = {}
    for p in tennisdata_dir.iterdir():
        m = _TENNISDATA_RE.search(p.name)
        if not m:
            continue
        year = int(m.group(1))
        ext = m.group(2).lower()
        if year < year_from or (year_to is not None and year > year_to):
            continue
        by_year.setdefault(year, {})[ext] = p

    preference = ("csv", "xlsx", "xls")
    chosen: list[tuple[int, Path]] = []
    for year, exts in by_year.items():
        for ext in preference:
            if ext in exts:
                chosen.append((year, exts[ext]))
                break
    chosen.sort(key=lambda t: t[0])
    return [p for _, p in chosen]


# ---------------------------------------------------------------------------
# CSV readers
# ---------------------------------------------------------------------------

def read_csv_robust(
    path: Path,
    dtype: dict | None = None,
    parse_dates: list[str] | None = None,
    encodings: Iterable[str] = ("utf-8", "latin-1", "cp1252"),
    **kwargs,
) -> pd.DataFrame:
    """Read a CSV trying multiple encodings.

    Sackmann files are utf-8, but some manually-edited Tennis-Data exports
    end up as latin-1 / cp1252. We try in order and raise only if all fail.
    """
    last_err: Exception | None = None
    for enc in encodings:
        try:
            df = pd.read_csv(
                path,
                dtype=dtype,
                parse_dates=parse_dates,
                encoding=enc,
                low_memory=False,
                **kwargs,
            )
            log.debug("read %s (%d rows) encoding=%s", path.name, len(df), enc)
            return df
        except UnicodeDecodeError as exc:
            last_err = exc
            continue
    raise UnicodeDecodeError(
        f"Could not decode {path} with any of {list(encodings)}: {last_err}"
    )


def read_many_csv(
    paths: Iterable[Path],
    dtype: dict | None = None,
    parse_dates: list[str] | None = None,
    add_source_col: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Read and concat CSVs.

    If ``add_source_col`` is True, a ``_source_file`` column is added so
    downstream auditing can trace any row back to its originating year.
    Empty input returns an empty DataFrame (no error).
    """
    frames: list[pd.DataFrame] = []
    for p in paths:
        df = read_csv_robust(p, dtype=dtype, parse_dates=parse_dates, **kwargs)
        if add_source_col:
            df["_source_file"] = p.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    return out


# ---------------------------------------------------------------------------
# Tennis-Data (mixed .xls / .xlsx / .csv)
# ---------------------------------------------------------------------------

def read_tennisdata_file(path: Path) -> pd.DataFrame:
    """Read a single Tennis-Data yearly file, dispatching on extension.

    No cleaning is done — column names and dtypes are returned as-is.
    Cleaning lives in :mod:`wta_clay.data.oddsfeed`.
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = read_csv_robust(path)
    elif suffix in (".xlsx", ".xlsm"):
        df = pd.read_excel(path, engine="openpyxl")
    elif suffix == ".xls":
        # Needs xlrd==1.2.0. If not installed, caller should pre-convert to csv.
        try:
            df = pd.read_excel(path, engine="xlrd")
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                f"Reading {path.name} requires xlrd==1.2.0. "
                "Install it, or convert the file to CSV/XLSX."
            ) from exc
    else:
        raise ValueError(f"Unsupported extension for Tennis-Data file: {path}")

    df["_source_file"] = path.name
    return df


def read_many_tennisdata(paths: Iterable[Path]) -> pd.DataFrame:
    """Concat Tennis-Data yearly files.

    Column sets differ across years (later years add more bookmakers and
    set-by-set fields). We concat with ``sort=False`` and let oddsfeed.py
    normalize the schema.
    """
    frames = [read_tennisdata_file(p) for p in paths]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)
