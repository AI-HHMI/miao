"""Tests for OME-NGFF metadata reading."""

from pathlib import Path

import numpy as np
import pytest

from miao.zarr_meta import detect_zarr_version, read_ome_metadata


class TestDetectZarrVersion:
    def test_zarr2(self, zarr2_volume: Path):
        version = detect_zarr_version(zarr2_volume / "raw" / "0")
        assert version == "zarr2"

    def test_missing_path(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Cannot detect"):
            detect_zarr_version(tmp_path / "nonexistent")


class TestReadOmeMetadata:
    def test_read_image_metadata(self, zarr2_volume: Path):
        meta = read_ome_metadata(zarr2_volume, "raw", "zarr2")
        assert meta.axis_names == ["z", "y", "x"]
        assert meta.zarr_version == "zarr2"
        assert len(meta.scales) == 3

        # Scale 0: full resolution
        s0 = meta.scales[0]
        assert s0.shape == [64, 64, 64]
        assert s0.scale_factors == [1.0, 1.0, 1.0]
        assert s0.dtype == np.dtype("float32")

        # Scale 1: 2x downsampled
        s1 = meta.scales[1]
        assert s1.shape == [32, 32, 32]
        assert s1.scale_factors == [2.0, 2.0, 2.0]

        # Scale 2: 4x downsampled
        s2 = meta.scales[2]
        assert s2.shape == [16, 16, 16]
        assert s2.scale_factors == [4.0, 4.0, 4.0]

    def test_read_label_metadata(self, zarr2_volume: Path):
        meta = read_ome_metadata(zarr2_volume, "labels/seg", "zarr2")
        assert meta.scales[0].dtype == np.dtype("uint32")

    def test_read_specific_scales(self, zarr2_volume: Path):
        meta = read_ome_metadata(zarr2_volume, "raw", "zarr2", requested_scales=[0, 2])
        assert set(meta.scales.keys()) == {0, 2}
        assert 1 not in meta.scales

    def test_invalid_scale_index(self, zarr2_volume: Path):
        with pytest.raises(IndexError, match="Requested scale level 5"):
            read_ome_metadata(zarr2_volume, "raw", "zarr2", requested_scales=[5])

    def test_missing_group(self, zarr2_volume: Path):
        with pytest.raises(FileNotFoundError):
            read_ome_metadata(zarr2_volume, "nonexistent", "zarr2")
