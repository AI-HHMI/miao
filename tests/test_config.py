"""Tests for config loading and validation."""

import pytest
import yaml

from miao.config import MiaoConfig, VolumeConfig, load_config


class TestVolumeConfig:
    def test_valid(self):
        v = VolumeConfig(
            name="raw",
            path="/data/test.zarr",
            image_key="raw",
            scales=[0, 1],
        )
        assert v.weight == 1.0
        assert v.label_key is None

    def test_negative_weight(self):
        with pytest.raises(ValueError, match="positive"):
            VolumeConfig(
                name="raw",
                path="/data/test.zarr",
                image_key="raw",
                scales=[0],
                weight=-1.0,
            )


class TestMiaoConfig:
    def test_valid(self, sample_config: dict):
        cfg = MiaoConfig(**sample_config)
        assert len(cfg.volumes) == 1
        assert cfg.samples_per_epoch == 10

    def test_patch_size_dim_mismatch(self, sample_config: dict):
        sample_config["patch_size"] = [8, 8]
        with pytest.raises(ValueError, match="elements"):
            MiaoConfig(**sample_config)

    def test_duplicate_names(self, sample_config: dict):
        sample_config["volumes"].append(sample_config["volumes"][0].copy())
        with pytest.raises(ValueError, match="unique"):
            MiaoConfig(**sample_config)


class TestLoadConfig:
    def test_load_yaml(self, tmp_path, sample_config):
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(sample_config, f)
        cfg = load_config(config_path)
        assert cfg.volumes[0].name == "test_raw"
