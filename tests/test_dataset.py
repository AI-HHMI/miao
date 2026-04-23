"""Tests for VolumeDataset."""

from pathlib import Path

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
