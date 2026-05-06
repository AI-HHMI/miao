"""Tests for VolumeDataset."""

from pathlib import Path
from itertools import product as iproduct

import numpy as np
import pytest
import torch

from miao.config import MiaoConfig
from miao.dataset import VolumeDataset


class TestVolumeDataset:
    def test_basic_getitem(self, sample_config: dict):
        cfg = MiaoConfig(**sample_config)
        ds = VolumeDataset(cfg)

        assert len(ds) == 10

        sample = ds[0]
        assert "img" in sample
        assert "label" in sample
        assert "meta" in sample

        # output_axes="lzyx" → img shape: (1, L, Z, Y, X) = (1, 3, 8, 8, 8)
        assert sample["img"].shape == (3, 8, 8, 8)
        assert sample["img"].dtype == torch.float32

        # label shape: same as img
        assert sample["label"] is not None
        assert sample["label"].shape == (3, 8, 8, 8)

    def test_no_labels(self, zarr2_volume: Path):
        cfg = MiaoConfig(
            volumes=[
                {
                    "name": "test",
                    "path": str(zarr2_volume),
                    "image_key": "raw",
                    "scales": [0, 1],
                }
            ],
            n_scales=2,
            output_axes="lzyx",
            patch_size=[8, 8, 8],
            samples_per_epoch=5,
        )
        ds = VolumeDataset(cfg)
        sample = ds[0]

        assert sample["img"].shape == (2, 8, 8, 8)
        assert isinstance(sample["label"], torch.Tensor)
        assert sample["label"].dtype == torch.long
        assert sample["label"].numel() == 0

    def test_no_labels_with_dataloader(self, zarr2_volume: Path):
        """Default collate should batch unlabeled samples without error."""
        cfg = MiaoConfig(
            volumes=[
                {
                    "name": "test",
                    "path": str(zarr2_volume),
                    "image_key": "raw",
                    "scales": [0, 1],
                }
            ],
            n_scales=2,
            output_axes="lzyx",
            patch_size=[8, 8, 8],
            samples_per_epoch=5,
        )
        ds = VolumeDataset(cfg)
        dl = torch.utils.data.DataLoader(ds, batch_size=2, num_workers=0)
        batch = next(iter(dl))

        assert batch["img"].shape == (2, 2, 8, 8, 8)
        assert isinstance(batch["label"], torch.Tensor)
        assert batch["label"].shape == (2, 0)
        assert batch["label"].dtype == torch.long

    def test_axis_reorientation(self, zarr2_volume: Path):
        """Test that output axes are correctly reoriented."""
        cfg = MiaoConfig(
            volumes=[
                {
                    "name": "test",
                    "path": str(zarr2_volume),
                    "image_key": "raw",
                    "scales": [0],
                }
            ],
            n_scales=1,
            output_axes="lxyz",
            patch_size=[8, 8, 8],
            samples_per_epoch=5,
        )
        ds = VolumeDataset(cfg)
        sample = ds[0]

        # output_axes="lxyz" → (1, L, X, Y, Z)
        assert sample["img"].shape == (1, 8, 8, 8)

    def test_level_dim_placement(self, zarr2_volume: Path):
        """Test that 'l' can be placed at different positions."""
        cfg = MiaoConfig(
            volumes=[
                {
                    "name": "test",
                    "path": str(zarr2_volume),
                    "image_key": "raw",
                    "scales": [0, 1],
                }
            ],
            n_scales=2,
            output_axes="xyzl",
            patch_size=[8, 8, 8],
            samples_per_epoch=5,
        )
        ds = VolumeDataset(cfg)
        sample = ds[0]

        # output_axes="xyzl" → (1, X, Y, Z, L) = (1, 8, 8, 8, 2)
        assert sample["img"].shape == (8, 8, 8, 2)

    def test_meta_contents(self, sample_config: dict):
        cfg = MiaoConfig(**sample_config)
        ds = VolumeDataset(cfg)
        sample = ds[0]

        meta = sample["meta"]
        assert "volume" in meta
        assert "coordinate" in meta
        assert "scale_levels" in meta
        assert meta["volume"] == "test_raw"
        assert meta["scale_levels"] == [0, 1, 2]
        assert len(meta["coordinate"]) == 3

    def test_multiple_volumes_weighted(self, zarr2_volume: Path):
        """Test that volumes are sampled according to weights."""
        cfg = MiaoConfig(
            volumes=[
                {
                    "name": "vol_a",
                    "path": str(zarr2_volume),
                    "image_key": "raw",
                    "scales": [0],
                    "weight": 0.99,
                },
                {
                    "name": "vol_b",
                    "path": str(zarr2_volume),
                    "image_key": "raw",
                    "scales": [0],
                    "weight": 0.01,
                },
            ],
            n_scales=1,
            output_axes="lzyx",
            patch_size=[8, 8, 8],
            samples_per_epoch=100,
        )
        ds = VolumeDataset(cfg)

        vol_counts = {"vol_a": 0, "vol_b": 0}
        for i in range(100):
            sample = ds[i]
            vol_counts[sample["meta"]["volume"]] += 1

        # vol_a should be sampled much more often
        assert vol_counts["vol_a"] > vol_counts["vol_b"]

    def test_single_scale(self, zarr2_volume: Path):
        cfg = MiaoConfig(
            volumes=[
                {
                    "name": "test",
                    "path": str(zarr2_volume),
                    "image_key": "raw",
                    "scales": [0],
                }
            ],
            n_scales=1,
            output_axes="lzyx",
            patch_size=[8, 8, 8],
            samples_per_epoch=5,
        )
        ds = VolumeDataset(cfg)
        sample = ds[0]
        assert sample["img"].shape == (1, 8, 8, 8)

    def test_scales_length_mismatch(self, zarr2_volume: Path):
        """Test that mismatched n_scales raises an error."""
        with pytest.raises(ValueError, match="2 scales.*n_scales=3"):
            MiaoConfig(
                volumes=[
                    {
                        "name": "test",
                        "path": str(zarr2_volume),
                        "image_key": "raw",
                        "scales": [0, 1],
                    }
                ],
                n_scales=3,
                output_axes="lzyx",
                patch_size=[8, 8, 8],
            )

    def test_missing_l_in_output_axes(self, zarr2_volume: Path):
        """Test that output_axes without 'l' raises an error."""
        with pytest.raises(ValueError, match="must contain 'l'"):
            MiaoConfig(
                volumes=[
                    {
                        "name": "test",
                        "path": str(zarr2_volume),
                        "image_key": "raw",
                        "scales": [0],
                    }
                ],
                n_scales=1,
                output_axes="zyx",
                patch_size=[8, 8, 8],
            )

    @pytest.mark.parametrize(
        "zarr2_volume",
        [{"dtype": "uint8", "fill_value": 150, "num_scales": 1, "base_shape": (32, 32, 32)}],
        indirect=True,
    )
    def test_normalize_custom_min_max(self, zarr2_volume: Path):
        """Clip to [min, max] then linear map to [0, 1]."""
        cfg = MiaoConfig(
            volumes=[
                {
                    "name": "test",
                    "path": str(zarr2_volume),
                    "image_key": "raw",
                    "scales": [0],
                    "normalize": True,
                    "normalize_min": 100,
                    "normalize_max": 200,
                }
            ],
            n_scales=1,
            output_axes="lzyx",
            patch_size=[8, 8, 8],
            samples_per_epoch=1,
        )
        ds = VolumeDataset(cfg)
        sample = ds[0]
        expected = (150.0 - 100.0) / (200.0 - 100.0)
        assert torch.allclose(sample["img"], torch.full_like(sample["img"], expected))

    @pytest.mark.parametrize(
        "zarr2_volume",
        [{"dtype": "uint8", "fill_value": 255, "num_scales": 1, "base_shape": (32, 32, 32)}],
        indirect=True,
    )
    def test_normalize_clips_outside_range(self, zarr2_volume: Path):
        cfg = MiaoConfig(
            volumes=[
                {
                    "name": "test",
                    "path": str(zarr2_volume),
                    "image_key": "raw",
                    "scales": [0],
                    "normalize": True,
                    "normalize_min": 0,
                    "normalize_max": 128,
                }
            ],
            n_scales=1,
            output_axes="lzyx",
            patch_size=[8, 8, 8],
            samples_per_epoch=1,
        )
        ds = VolumeDataset(cfg)
        sample = ds[0]
        # 255 clipped to 128 → 1.0
        assert torch.allclose(sample["img"], torch.ones_like(sample["img"]))

    @pytest.mark.parametrize(
        "zarr2_volume",
        [{"dtype": "float32", "fill_value": 10.0, "num_scales": 1, "base_shape": (32, 32, 32)}],
        indirect=True,
    )
    def test_normalize_float_with_explicit_range(self, zarr2_volume: Path):
        """Explicit range must scale floats; legacy float path leaves raw values unchanged."""
        raw_value = 10.0
        lo, hi = 0.0, 100.0
        expected = (raw_value - lo) / (hi - lo)
        cfg = MiaoConfig(
            volumes=[
                {
                    "name": "test",
                    "path": str(zarr2_volume),
                    "image_key": "raw",
                    "scales": [0],
                    "normalize": True,
                    "normalize_min": lo,
                    "normalize_max": hi,
                }
            ],
            n_scales=1,
            output_axes="lzyx",
            patch_size=[8, 8, 8],
            samples_per_epoch=1,
        )
        ds = VolumeDataset(cfg)
        sample = ds[0]
        assert torch.allclose(sample["img"], torch.full_like(sample["img"], expected))

    # ── Random mode: grid_index is None ──────────────────────────────────────

    def test_random_mode_no_grid_index(self, zarr2_volume: Path):
        """Random mode: meta does not contain 'grid_index' (avoids DataLoader collation issues)."""
        cfg = MiaoConfig(
            volumes=[{"name": "test", "path": str(zarr2_volume), "image_key": "raw", "scales": [0]}],
            n_scales=1,
            output_axes="lzyx",
            patch_size=[8, 8, 8],
            samples_per_epoch=5,
        )
        ds = VolumeDataset(cfg)
        for i in range(5):
            assert "grid_index" not in ds[i]["meta"]

    # ── Sequential sampling ───────────────────────────────────────────────────

    def test_sequential_basic(self, zarr2_volume: Path):
        """Sequential: __len__ equals precomputed grid size; sample shape is correct."""
        cfg = MiaoConfig(
            volumes=[{"name": "test", "path": str(zarr2_volume), "image_key": "raw", "scales": [0]}],
            n_scales=1,
            output_axes="lzyx",
            patch_size=[8, 8, 8],
            sampling="sequential",
        )
        ds = VolumeDataset(cfg)
        # 64^3 volume, patch 8^3, overlap 0, stride 8
        # min_center=4, max_center=60 per axis → positions [4,12,...,60] = 8 per axis
        assert len(ds) == 8 ** 3
        sample = ds[0]
        assert sample["img"].shape == (1, 8, 8, 8)
        assert sample["img"].dtype == torch.float32

    def test_sequential_deterministic(self, zarr2_volume: Path):
        """Same idx always returns the same coordinate and grid_index."""
        cfg = MiaoConfig(
            volumes=[{"name": "test", "path": str(zarr2_volume), "image_key": "raw", "scales": [0]}],
            n_scales=1,
            output_axes="lzyx",
            patch_size=[8, 8, 8],
            sampling="sequential",
        )
        ds = VolumeDataset(cfg)
        for idx in [0, 7, 63, 511]:
            s1 = ds[idx]
            s2 = ds[idx]
            assert s1["meta"]["coordinate"] == s2["meta"]["coordinate"]
            assert s1["meta"]["grid_index"] == s2["meta"]["grid_index"]

    def test_sequential_grid_index_in_meta(self, zarr2_volume: Path):
        """Sequential mode: meta['grid_index'] is a tuple; first is (0,0,0)."""
        cfg = MiaoConfig(
            volumes=[{"name": "test", "path": str(zarr2_volume), "image_key": "raw", "scales": [0]}],
            n_scales=1,
            output_axes="lzyx",
            patch_size=[8, 8, 8],
            sampling="sequential",
        )
        ds = VolumeDataset(cfg)
        first = ds[0]
        last = ds[len(ds) - 1]
        assert isinstance(first["meta"]["grid_index"], tuple)
        assert first["meta"]["grid_index"] == (0, 0, 0)
        assert last["meta"]["grid_index"] == (7, 7, 7)

    def test_sequential_full_coverage(self, zarr2_volume: Path):
        """Every voxel in the volume is covered by at least one patch."""
        vol_shape = (64, 64, 64)
        patch_size = [8, 8, 8]
        cfg = MiaoConfig(
            volumes=[{"name": "test", "path": str(zarr2_volume), "image_key": "raw", "scales": [0]}],
            n_scales=1,
            output_axes="lzyx",
            patch_size=patch_size,
            sampling="sequential",
        )
        ds = VolumeDataset(cfg)
        covered = np.zeros(vol_shape, dtype=bool)
        half = [p // 2 for p in patch_size]  # [4, 4, 4]
        for i in range(len(ds)):
            z, y, x = ds[i]["meta"]["coordinate"]  # center in ZYX order
            covered[z - half[0]: z + half[0], y - half[1]: y + half[1], x - half[2]: x + half[2]] = True
        assert covered.all(), "Some voxels not covered by any patch"

    def test_sequential_zero_overlap_stride_equals_patch(self, zarr2_volume: Path):
        """overlap=0: consecutive patches are exactly patch_size apart (no overlap, no gap)."""
        cfg = MiaoConfig(
            volumes=[{"name": "test", "path": str(zarr2_volume), "image_key": "raw", "scales": [0]}],
            n_scales=1,
            output_axes="lzyx",
            patch_size=[8, 8, 8],
            sampling="sequential",
            overlap=0,
        )
        ds = VolumeDataset(cfg)
        # grid[0] = (z0, y0, x0), grid[1] = (z0, y0, x1) — last axis varies fastest
        c0 = ds._grid[0][1]  # center of first patch
        c1 = ds._grid[1][1]  # center of second patch (next x position)
        assert abs(int(c1[-1]) - int(c0[-1])) == 8  # stride = patch_size - overlap = 8

    def test_sequential_overlap(self, zarr2_volume: Path):
        """overlap=4: stride=4, consecutive patch centers are 4 apart; grid is larger."""
        cfg = MiaoConfig(
            volumes=[{"name": "test", "path": str(zarr2_volume), "image_key": "raw", "scales": [0]}],
            n_scales=1,
            output_axes="lzyx",
            patch_size=[8, 8, 8],
            sampling="sequential",
            overlap=4,
        )
        ds = VolumeDataset(cfg)
        # stride=4, positions per axis: range(4, 61, 4) → [4,8,...,60] = 15 positions
        assert len(ds) == 15 ** 3
        c0 = ds._grid[0][1]
        c1 = ds._grid[1][1]
        assert abs(int(c1[-1]) - int(c0[-1])) == 4

    def test_sequential_multi_volume(self, zarr2_volume: Path):
        """Multi-volume: all volumes are iterated; grid_index resets per volume."""
        cfg = MiaoConfig(
            volumes=[
                {"name": "vol_a", "path": str(zarr2_volume), "image_key": "raw", "scales": [0]},
                {"name": "vol_b", "path": str(zarr2_volume), "image_key": "raw", "scales": [0]},
            ],
            n_scales=1,
            output_axes="lzyx",
            patch_size=[8, 8, 8],
            sampling="sequential",
        )
        ds = VolumeDataset(cfg)
        per_vol = 8 ** 3  # 512 per volume
        assert len(ds) == 2 * per_vol
        # vol_a fills first half, vol_b fills second half
        assert ds[0]["meta"]["volume"] == "vol_a"
        assert ds[per_vol - 1]["meta"]["volume"] == "vol_a"
        assert ds[per_vol]["meta"]["volume"] == "vol_b"
        # grid_index resets at volume boundary
        assert ds[0]["meta"]["grid_index"] == (0, 0, 0)
        assert ds[per_vol]["meta"]["grid_index"] == (0, 0, 0)

    def test_sequential_overlap_too_large_raises(self, zarr2_volume: Path):
        """overlap >= patch_size raises ValueError at config creation time."""
        with pytest.raises(ValueError, match="overlap"):
            MiaoConfig(
                volumes=[{"name": "test", "path": str(zarr2_volume), "image_key": "raw", "scales": [0]}],
                n_scales=1,
                output_axes="lzyx",
                patch_size=[8, 8, 8],
                sampling="sequential",
                overlap=8,  # equal to patch_size → stride=0
            )

    def test_sequential_overlap_negative_raises(self, zarr2_volume: Path):
        """Negative overlap raises ValueError."""
        with pytest.raises(ValueError, match="overlap"):
            MiaoConfig(
                volumes=[{"name": "test", "path": str(zarr2_volume), "image_key": "raw", "scales": [0]}],
                n_scales=1,
                output_axes="lzyx",
                patch_size=[8, 8, 8],
                sampling="sequential",
                overlap=-1,
            )

    def test_sequential_per_axis_overlap(self, zarr2_volume: Path):
        """Per-axis overlap list: each axis uses its own overlap value."""
        cfg = MiaoConfig(
            volumes=[{"name": "test", "path": str(zarr2_volume), "image_key": "raw", "scales": [0]}],
            n_scales=1,
            output_axes="lzyx",
            patch_size=[8, 8, 8],
            sampling="sequential",
            overlap=[4, 0, 0],  # only Z has overlap; in output ZYX order
        )
        ds = VolumeDataset(cfg)
        # Z: stride=4 → 15 positions; Y,X: stride=8 → 8 positions each
        assert len(ds) == 15 * 8 * 8
