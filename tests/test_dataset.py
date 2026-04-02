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

        # img shape: (1, L, Z, Y, X) = (1, 3, 8, 8, 8)
        assert sample["img"].shape == (1, 3, 8, 8, 8)
        assert sample["img"].dtype == torch.float32

        # label shape: same as img
        assert sample["label"] is not None
        assert sample["label"].shape == (1, 3, 8, 8, 8)

    def test_no_labels(self, zarr2_volume: Path):
        cfg = MiaoConfig(
            volumes=[
                {
                    "name": "test",
                    "path": str(zarr2_volume),
                    "image_key": "raw",
                    "axes": "zyx",
                    "scales": [0, 1],
                }
            ],
            output_axes="zyx",
            patch_size=[8, 8, 8],
            samples_per_epoch=5,
        )
        ds = VolumeDataset(cfg)
        sample = ds[0]

        assert sample["img"].shape == (1, 2, 8, 8, 8)
        assert sample["label"] is None

    def test_axis_reorientation(self, zarr2_volume: Path):
        """Test that output axes are correctly reoriented."""
        cfg = MiaoConfig(
            volumes=[
                {
                    "name": "test",
                    "path": str(zarr2_volume),
                    "image_key": "raw",
                    "axes": "zyx",
                    "scales": [0],
                }
            ],
            output_axes="xyz",
            patch_size=[8, 8, 8],
            samples_per_epoch=5,
        )
        ds = VolumeDataset(cfg)
        sample = ds[0]

        # Shape should follow output_axes: (1, L, X, Y, Z)
        assert sample["img"].shape == (1, 1, 8, 8, 8)

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
                    "axes": "zyx",
                    "scales": [0],
                    "weight": 0.99,
                },
                {
                    "name": "vol_b",
                    "path": str(zarr2_volume),
                    "image_key": "raw",
                    "axes": "zyx",
                    "scales": [0],
                    "weight": 0.01,
                },
            ],
            output_axes="zyx",
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
                    "axes": "zyx",
                    "scales": [0],
                }
            ],
            output_axes="zyx",
            patch_size=[8, 8, 8],
            samples_per_epoch=5,
        )
        ds = VolumeDataset(cfg)
        sample = ds[0]
        assert sample["img"].shape == (1, 1, 8, 8, 8)
