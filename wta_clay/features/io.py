"""I/O helpers for feature parquets with sidecar metadata.

Each feature module writes ``<name>.parquet`` for the feature columns AND
``<name>.parquet.json`` as a metadata sidecar. The sidecar records the
master frame snapshot used to compute the features so that downstream
loaders can detect stale features when the master is regenerated.

Convention:
  - Both files share the basename (``X.parquet`` ↔ ``X.parquet.json``).
  - Sidecar is canonical JSON with sorted keys (deterministic for VCS diff).
  - The validation in :func:`load_features` raises on master mismatch.

Why a sidecar instead of a per-row column? A column duplicates a constant
across every row (~88 KB for 11K rows × 1 Int64); a sidecar is ~400 bytes
once per parquet AND can carry richer metadata (date ranges, configs,
column lists, computed_at) without bloating the data table.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)


def _master_key_hash(master: pd.DataFrame) -> str:
    """SHA-256 of sorted ``stable_match_key`` set.

    This is the strongest link-integrity check available — any change to
    the master's set of matches (additions, deletions, or different
    matchings) produces a different hash. Catches subtle drift that
    row-count-only checks would miss.
    """
    if "stable_match_key" not in master.columns:
        return ""
    keys = sorted(master["stable_match_key"].dropna().astype(str).tolist())
    h = hashlib.sha256("\n".join(keys).encode("utf-8"))
    return h.hexdigest()


@dataclass
class FeatureMetadata:
    """Sidecar metadata. One per feature parquet."""
    feature_module: str                       # e.g. "clay_elo"
    master_row_count: int
    master_match_id_min: int
    master_match_id_max: int
    master_date_min: str                      # ISO date
    master_date_max: str                      # ISO date
    master_key_hash: str                      # sha256 of sorted stable_match_keys
    n_features: int                           # rows in features parquet
    feature_columns: list[str]
    computed_at: str                          # ISO 8601 UTC
    config: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to deterministic JSON (sorted keys, indented)."""
        return json.dumps(asdict(self), sort_keys=True, indent=2,
                          default=str)

    @classmethod
    def from_path(cls, path: Path | str) -> "FeatureMetadata":
        """Load from sidecar JSON path."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Sidecar JSON not found: {path}")
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


def sidecar_path(parquet_path: Path | str) -> Path:
    """Return the canonical sidecar JSON path for a parquet path."""
    p = Path(parquet_path)
    return p.with_suffix(p.suffix + ".json")


def write_features(
    features: pd.DataFrame,
    master: pd.DataFrame,
    *,
    path: Path | str,
    feature_module: str,
    config: dict[str, Any] | None = None,
) -> FeatureMetadata:
    """Write features parquet AND sidecar JSON metadata.

    The metadata snapshot is taken from ``master`` so that loaders can
    detect when the master has been regenerated since this feature was
    computed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if "match_id" not in features.columns:
        raise ValueError("features must have a 'match_id' column")
    if "match_id" not in master.columns or "match_date" not in master.columns:
        raise ValueError("master must have 'match_id' and 'match_date' columns")

    meta = FeatureMetadata(
        feature_module=feature_module,
        master_row_count=int(len(master)),
        master_match_id_min=int(master["match_id"].min()),
        master_match_id_max=int(master["match_id"].max()),
        master_date_min=str(master["match_date"].min().date()),
        master_date_max=str(master["match_date"].max().date()),
        master_key_hash=_master_key_hash(master),
        n_features=int(len(features)),
        feature_columns=[c for c in features.columns if c != "match_id"],
        computed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        config=config or {},
    )

    features.to_parquet(path, index=False, compression="snappy")
    sidecar_path(path).write_text(meta.to_json())
    log.info("Wrote %s (%d rows) + sidecar metadata", path, len(features))
    return meta


def load_features(
    path: Path | str,
    master: pd.DataFrame | None = None,
    *,
    allow_stale: bool = False,
) -> pd.DataFrame:
    """Load features parquet, validating the sidecar against current master.

    Validation checks (when ``master`` is provided):
      - master_row_count matches
      - master_match_id_min / max match
      - master_date_min / max match

    Raises ``ValueError`` on mismatch. Use ``allow_stale=True`` to bypass
    (e.g. during exploratory feature work — NEVER in production).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Features parquet not found: {path}")

    features = pd.read_parquet(path)

    if master is None:
        return features

    sc = sidecar_path(path)
    if not sc.exists():
        if allow_stale:
            log.warning("No sidecar at %s; skipping validation (allow_stale)", sc)
            return features
        raise FileNotFoundError(
            f"Sidecar JSON {sc} missing; cannot validate against master. "
            f"Use allow_stale=True to bypass."
        )

    meta = FeatureMetadata.from_path(sc)
    mismatches = _check_master_consistency(meta, master)
    if mismatches:
        msg = (
            f"Stale features at {path}:\n"
            + "\n".join(f"  - {m}" for m in mismatches)
            + f"\nRe-run feature module {meta.feature_module!r} on current master."
        )
        if allow_stale:
            log.warning(msg)
        else:
            raise ValueError(msg)

    return features


def _check_master_consistency(meta: FeatureMetadata,
                               master: pd.DataFrame) -> list[str]:
    """Return a list of human-readable mismatch descriptions."""
    issues = []
    if meta.master_row_count != len(master):
        issues.append(
            f"master_row_count: features-snapshot={meta.master_row_count} "
            f"vs current={len(master)}"
        )
    if meta.master_match_id_min != int(master["match_id"].min()):
        issues.append(
            f"master_match_id_min: features-snapshot={meta.master_match_id_min} "
            f"vs current={int(master['match_id'].min())}"
        )
    if meta.master_match_id_max != int(master["match_id"].max()):
        issues.append(
            f"master_match_id_max: features-snapshot={meta.master_match_id_max} "
            f"vs current={int(master['match_id'].max())}"
        )
    if meta.master_date_min != str(master["match_date"].min().date()):
        issues.append(
            f"master_date_min: features-snapshot={meta.master_date_min} "
            f"vs current={master['match_date'].min().date()}"
        )
    if meta.master_date_max != str(master["match_date"].max().date()):
        issues.append(
            f"master_date_max: features-snapshot={meta.master_date_max} "
            f"vs current={master['match_date'].max().date()}"
        )
    # Strongest check: stable_match_key set hash. Any change to the master's
    # match population (additions, deletions, mis-mappings) produces a
    # different hash. Catches subtle drift that row-count checks miss.
    if "stable_match_key" in master.columns and meta.master_key_hash:
        current_hash = _master_key_hash(master)
        if meta.master_key_hash != current_hash:
            issues.append(
                f"master_key_hash: features-snapshot={meta.master_key_hash[:12]}... "
                f"vs current={current_hash[:12]}... "
                f"(stable_match_key set differs — strongest evidence of drift)"
            )
    return issues
