"""Unit tests for wta_clay.features.io (sidecar metadata mechanism)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from wta_clay.features.io import (
    FeatureMetadata, load_features, sidecar_path, write_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_master(n=10):
    return pd.DataFrame({
        "match_id": list(range(n)),
        "match_date": pd.to_datetime(
            [f"2020-05-{(i%30)+1:02d}" for i in range(n)]
        ),
        "p1_id": [i + 100 for i in range(n)],
        "p2_id": [i + 200 for i in range(n)],
        "stable_match_key": [f"2020-T{i:03d}_{i+100}_{i+200}"
                              for i in range(n)],
    })


def _make_features(n=10):
    return pd.DataFrame({
        "match_id": list(range(n)),
        "test_feature_p1": [1.0 + i * 0.1 for i in range(n)],
        "test_feature_p2": [2.0 + i * 0.1 for i in range(n)],
    })


# ---------------------------------------------------------------------------
# write_features + sidecar
# ---------------------------------------------------------------------------

class TestWriteFeatures:
    def test_writes_parquet_and_sidecar(self, tmp_path):
        master = _make_master(10)
        features = _make_features(10)
        path = tmp_path / "test_feature.parquet"
        meta = write_features(features, master, path=path,
                              feature_module="test_feature")
        assert path.exists()
        assert sidecar_path(path).exists()
        assert sidecar_path(path).name == "test_feature.parquet.json"
        assert meta.feature_module == "test_feature"
        assert meta.master_row_count == 10
        assert meta.n_features == 10

    def test_sidecar_is_valid_json(self, tmp_path):
        master = _make_master(5)
        features = _make_features(5)
        path = tmp_path / "x.parquet"
        write_features(features, master, path=path, feature_module="x")
        # Reload as plain JSON
        data = json.loads(sidecar_path(path).read_text())
        assert data["feature_module"] == "x"
        assert data["master_row_count"] == 5
        assert data["master_match_id_min"] == 0
        assert data["master_match_id_max"] == 4
        assert "computed_at" in data

    def test_sidecar_includes_config(self, tmp_path):
        master = _make_master(3)
        features = _make_features(3)
        path = tmp_path / "x.parquet"
        config = {"k": 16, "method": "test"}
        write_features(features, master, path=path,
                       feature_module="x", config=config)
        data = json.loads(sidecar_path(path).read_text())
        assert data["config"] == config

    def test_missing_master_columns_raises(self, tmp_path):
        master_bad = pd.DataFrame({"match_id": [0, 1]})  # missing match_date
        features = _make_features(2)
        path = tmp_path / "x.parquet"
        with pytest.raises(ValueError, match="match_date"):
            write_features(features, master_bad, path=path,
                           feature_module="x")

    def test_missing_features_match_id_raises(self, tmp_path):
        master = _make_master(2)
        features = pd.DataFrame({"foo": [1, 2]})  # missing match_id
        path = tmp_path / "x.parquet"
        with pytest.raises(ValueError, match="match_id"):
            write_features(features, master, path=path, feature_module="x")


# ---------------------------------------------------------------------------
# load_features + validation
# ---------------------------------------------------------------------------

class TestLoadFeatures:
    def _setup(self, tmp_path):
        master = _make_master(10)
        features = _make_features(10)
        path = tmp_path / "x.parquet"
        write_features(features, master, path=path, feature_module="x")
        return path, master, features

    def test_load_and_validate_against_same_master(self, tmp_path):
        path, master, _ = self._setup(tmp_path)
        loaded = load_features(path, master)
        assert len(loaded) == 10

    def test_load_without_master_skips_validation(self, tmp_path):
        path, _, _ = self._setup(tmp_path)
        # No master argument → no validation
        loaded = load_features(path)
        assert len(loaded) == 10

    def test_stale_master_row_count_raises(self, tmp_path):
        path, _, _ = self._setup(tmp_path)
        # Master grew from 10 to 15 — stale features
        new_master = _make_master(15)
        with pytest.raises(ValueError, match="master_row_count"):
            load_features(path, new_master)

    def test_stale_master_date_range_raises(self, tmp_path):
        path, _, _ = self._setup(tmp_path)
        new_master = _make_master(10).copy()
        # Same row count but date shifted
        new_master["match_date"] = pd.to_datetime(
            [f"2024-05-{(i%30)+1:02d}" for i in range(10)]
        )
        with pytest.raises(ValueError, match="master_date"):
            load_features(path, new_master)

    def test_stale_match_id_range_raises(self, tmp_path):
        path, _, _ = self._setup(tmp_path)
        new_master = _make_master(10).copy()
        new_master["match_id"] = list(range(100, 110))  # shifted ids
        with pytest.raises(ValueError, match="master_match_id"):
            load_features(path, new_master)

    def test_allow_stale_warns_but_passes(self, tmp_path, caplog):
        path, _, _ = self._setup(tmp_path)
        new_master = _make_master(15)
        with caplog.at_level("WARNING"):
            loaded = load_features(path, new_master, allow_stale=True)
        assert any("Stale features" in r.message for r in caplog.records)
        assert len(loaded) == 10

    def test_missing_sidecar_raises_unless_stale(self, tmp_path):
        path, master, _ = self._setup(tmp_path)
        # Delete sidecar
        sidecar_path(path).unlink()
        with pytest.raises(FileNotFoundError, match="Sidecar"):
            load_features(path, master)
        # With allow_stale, falls through with a warning
        loaded = load_features(path, master, allow_stale=True)
        assert len(loaded) == 10

    def test_missing_parquet_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Features parquet"):
            load_features(tmp_path / "nope.parquet")


# ---------------------------------------------------------------------------
# FeatureMetadata
# ---------------------------------------------------------------------------

class TestFeatureMetadata:
    def test_to_json_sorted_keys(self, tmp_path):
        meta = FeatureMetadata(
            feature_module="z",
            master_row_count=5,
            master_match_id_min=0,
            master_match_id_max=4,
            master_date_min="2020-01-01",
            master_date_max="2020-12-31",
            master_key_hash="abc123",
            n_features=5,
            feature_columns=["a", "b"],
            computed_at="2026-04-21T00:00:00+00:00",
            config={"k": 1},
        )
        s = meta.to_json()
        idx_c = s.index('"computed_at"')
        idx_cfg = s.index('"config"')
        idx_fm = s.index('"feature_module"')
        assert idx_c < idx_cfg < idx_fm

    def test_round_trip_via_disk(self, tmp_path):
        meta = FeatureMetadata(
            feature_module="z",
            master_row_count=5,
            master_match_id_min=0,
            master_match_id_max=4,
            master_date_min="2020-01-01",
            master_date_max="2020-12-31",
            master_key_hash="hash_abc",
            n_features=5,
            feature_columns=["a"],
            computed_at="2026-04-21T00:00:00+00:00",
        )
        p = tmp_path / "m.json"
        p.write_text(meta.to_json())
        loaded = FeatureMetadata.from_path(p)
        assert loaded.master_row_count == 5
        assert loaded.master_key_hash == "hash_abc"
        assert loaded.feature_columns == ["a"]


class TestMasterKeyHashValidation:
    def _setup(self, tmp_path):
        master = _make_master(5)
        features = pd.DataFrame({
            "match_id": list(range(5)),
            "test_feature": [1.0 + i for i in range(5)],
        })
        path = tmp_path / "x.parquet"
        write_features(features, master, path=path, feature_module="x")
        return path, master

    def test_hash_recorded_in_sidecar(self, tmp_path):
        path, master = self._setup(tmp_path)
        sc = path.with_suffix(path.suffix + ".json")
        data = json.loads(sc.read_text())
        assert "master_key_hash" in data
        assert len(data["master_key_hash"]) == 64   # sha256 hex length

    def test_same_master_hash_matches(self, tmp_path):
        path, master = self._setup(tmp_path)
        # Same master → load passes
        loaded = load_features(path, master)
        assert len(loaded) == 5

    def test_different_player_set_caught_by_hash(self, tmp_path):
        # Build master, then create a "drift" master with same row count +
        # ID range + dates but DIFFERENT players (different stable keys).
        path, master = self._setup(tmp_path)
        drift = master.copy()
        drift["stable_match_key"] = [f"DRIFTED_{i}" for i in range(len(drift))]
        # row count, ID min/max, date range all match — only key set differs
        with pytest.raises(ValueError, match="master_key_hash"):
            load_features(path, drift)


# Bring needed imports into scope for new test
import json   # noqa: E402
