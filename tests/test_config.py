"""Tests for config loading and validation."""

import pytest
import yaml

from miao.config import MiaoConfig, ResolutionSampling, VolumeConfig, load_config


def _vol(name="a", **kw):
    return {"name": name, "path": "/data/test.zarr", "image_key": "raw", **kw}


SAMPLING = {
    "strategy": "log_uniform",
    "n_scales": 3,
    "min": [8, 8, 8],
    "max": [64, 64, 64],
    "isotropic": True,
}


class TestVolumeConfig:
    def test_valid(self):
        v = VolumeConfig(
            name="raw",
            path="/data/test.zarr",
            image_key="raw",
        )
        assert v.weight == 1.0
        assert v.label_key is None

    def test_negative_weight(self):
        with pytest.raises(ValueError, match="positive"):
            VolumeConfig(
                name="raw",
                path="/data/test.zarr",
                image_key="raw",
                weight=-1.0,
            )

    def test_normalize_range_both_required(self):
        with pytest.raises(ValueError, match="both be set or both omitted"):
            VolumeConfig(
                name="raw",
                path="/data/test.zarr",
                image_key="raw",
                normalize_min=0.0,
            )
        with pytest.raises(ValueError, match="both be set or both omitted"):
            VolumeConfig(
                name="raw",
                path="/data/test.zarr",
                image_key="raw",
                normalize_max=1.0,
            )

    def test_normalize_max_must_exceed_min(self):
        with pytest.raises(ValueError, match="greater than normalize_min"):
            VolumeConfig(
                name="raw",
                path="/data/test.zarr",
                image_key="raw",
                normalize_min=1.0,
                normalize_max=1.0,
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


class TestResolutionSampling:
    def test_valid_global(self):
        cfg = MiaoConfig(
            volumes=[_vol()],
            resolution_sampling=SAMPLING,
            output_axes="lzyx",
            patch_size=[8, 8, 8],
        )
        assert cfg.n_scales == 3
        assert cfg.is_sampling(cfg.volumes[0])

    def test_neither_set(self):
        with pytest.raises(ValueError, match="exactly one of"):
            MiaoConfig(volumes=[_vol()], output_axes="lzyx", patch_size=[8, 8, 8])

    def test_both_set_global(self):
        with pytest.raises(ValueError, match="exactly one of"):
            MiaoConfig(
                volumes=[_vol()],
                resolutions=[[1, 1, 1]],
                resolution_sampling={**SAMPLING, "n_scales": 1},
                output_axes="lzyx",
                patch_size=[8, 8, 8],
            )

    def test_both_set_per_volume(self):
        with pytest.raises(ValueError, match="at most one"):
            MiaoConfig(
                volumes=[
                    _vol(resolutions=[[1, 1, 1]], resolution_sampling={**SAMPLING, "n_scales": 1})
                ],
                resolutions=[[1, 1, 1]],
                output_axes="lzyx",
                patch_size=[8, 8, 8],
            )

    def test_min_greater_than_max(self):
        with pytest.raises(ValueError, match="min\\[i\\] <= max\\[i\\]"):
            MiaoConfig(
                volumes=[_vol()],
                resolution_sampling={**SAMPLING, "min": [9, 8, 8], "max": [8, 8, 8]},
                output_axes="lzyx",
                patch_size=[8, 8, 8],
            )

    def test_wrong_axis_count(self):
        with pytest.raises(ValueError, match="must each have 3 elements"):
            MiaoConfig(
                volumes=[_vol()],
                resolution_sampling={**SAMPLING, "min": [8, 8], "max": [64, 64]},
                output_axes="lzyx",
                patch_size=[8, 8, 8],
            )

    def test_rejected_with_sequential(self):
        with pytest.raises(ValueError, match="incompatible with sampling='sequential'"):
            MiaoConfig(
                volumes=[_vol()],
                resolution_sampling=SAMPLING,
                output_axes="lzyx",
                patch_size=[8, 8, 8],
                sampling="sequential",
            )

    def test_sample_windows_requires_isotropic(self):
        with pytest.raises(ValueError, match="requires\\s+isotropic=True"):
            MiaoConfig(
                volumes=[_vol()],
                resolution_sampling={**SAMPLING, "isotropic": False},
                output_axes="lzyx",
                patch_size=[8, 8, 8],
                sample_windows=True,
            )

    def test_sample_windows_isotropic_ok(self):
        cfg = MiaoConfig(
            volumes=[_vol()],
            resolution_sampling={**SAMPLING, "isotropic": True},
            output_axes="lzyx",
            patch_size=[8, 8, 8],
            sample_windows=True,
        )
        assert cfg.n_scales == 3

    def test_per_volume_override_mixes_modes(self):
        cfg = MiaoConfig(
            volumes=[
                _vol("fixed"),
                _vol("sampled", resolution_sampling={**SAMPLING, "n_scales": 1}),
            ],
            resolutions=[[1, 1, 1]],
            output_axes="lzyx",
            patch_size=[8, 8, 8],
        )
        assert not cfg.is_sampling(cfg.volumes[0])
        assert cfg.is_sampling(cfg.volumes[1])

    def test_scale_count_mismatch(self):
        with pytest.raises(ValueError, match="same number of scales"):
            MiaoConfig(
                volumes=[_vol("a"), _vol("b", resolutions=[[1, 1, 1]])],
                resolution_sampling={**SAMPLING, "n_scales": 2},
                output_axes="lzyx",
                patch_size=[8, 8, 8],
            )


class TestLoadConfig:
    def test_load_yaml(self, tmp_path, sample_config):
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(sample_config, f)
        cfg = load_config(config_path)
        assert cfg.volumes[0].name == "test_raw"
