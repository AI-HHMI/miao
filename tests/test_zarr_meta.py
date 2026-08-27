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

    def test_multiscale_level_transform(self, tmp_path: Path):
        """A multiscale-level (outer) scale transform multiplies every level's factors."""
        from conftest import _create_ome_ngff_zarr2

        zarr_path = tmp_path / "outer_scale.zarr"
        _create_ome_ngff_zarr2(
            zarr_path,
            group_key="raw",
            base_shape=(64, 64, 64),
            num_scales=3,
            base_scale_factors=[4.0, 2.0, 2.0],
            outer_scale=[0.5, 0.5, 0.5],
        )

        meta = read_ome_metadata(zarr_path, "raw", "zarr2")
        assert meta.scales[0].scale_factors == [2.0, 1.0, 1.0]
        assert meta.scales[1].scale_factors == [4.0, 2.0, 2.0]
        assert meta.scales[2].scale_factors == [8.0, 4.0, 4.0]


# --- OME-NGFF `translation`: where a level's origin sits --------------------------------------
#
# Read because a label written as a crop of a larger image records its position this way, and
# without it there is nothing to say where in the image the crop belongs.


def test_translation_is_parsed_per_level(tmp_path):
    from conftest import _create_ome_ngff_zarr2

    _create_ome_ngff_zarr2(
        tmp_path, "raw", (32, 32, 32), num_scales=2,
        base_scale_factors=[2.0, 2.0, 2.0], base_translation=[100.0, 50.0, 25.0],
    )
    meta = read_ome_metadata(tmp_path, "raw", "zarr2")
    for level in (0, 1):
        assert meta.scales[level].translation == [100.0, 50.0, 25.0]


def test_translation_absent_reads_as_zeros(tmp_path):
    """The overwhelmingly common case, and the one that must not change behaviour."""
    from conftest import _create_ome_ngff_zarr2

    _create_ome_ngff_zarr2(
        tmp_path, "raw", (32, 32, 32), num_scales=2, base_scale_factors=[2.0, 2.0, 2.0]
    )
    meta = read_ome_metadata(tmp_path, "raw", "zarr2")
    assert meta.outer_translation == [0.0, 0.0, 0.0]
    assert meta.scales[0].translation_or_zeros() == [0.0, 0.0, 0.0]


def test_outer_translation_composes_with_the_per_level_one(tmp_path):
    """OME-NGFF applies the dataset transform, then the multiscale-level one:

        physical = outer_scale * (level_scale * index + level_translation) + outer_translation

    so the effective translation is outer_scale * level_translation + outer_translation.
    """
    from conftest import _create_ome_ngff_zarr2

    _create_ome_ngff_zarr2(
        tmp_path, "raw", (32, 32, 32), num_scales=1,
        base_scale_factors=[1.0, 1.0, 1.0],
        base_translation=[10.0, 20.0, 30.0],
        outer_scale=[2.0, 2.0, 2.0],
        outer_translation=[1.0, 1.0, 1.0],
    )
    meta = read_ome_metadata(tmp_path, "raw", "zarr2")
    assert meta.scales[0].translation == [21.0, 41.0, 61.0]
    assert meta.scales[0].scale_factors == [2.0, 2.0, 2.0]


def test_translation_survives_a_scale_only_outer_transform(tmp_path):
    """A pre-existing config carrying only an outer scale must still report zero translation."""
    from conftest import _create_ome_ngff_zarr2

    _create_ome_ngff_zarr2(
        tmp_path, "raw", (32, 32, 32), num_scales=1,
        base_scale_factors=[1.0, 1.0, 1.0], outer_scale=[3.0, 3.0, 3.0],
    )
    meta = read_ome_metadata(tmp_path, "raw", "zarr2")
    assert meta.scales[0].translation == [0.0, 0.0, 0.0]
    assert meta.scales[0].scale_factors == [3.0, 3.0, 3.0]
