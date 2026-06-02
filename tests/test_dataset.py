"""Tests for VolumeDataset."""

from pathlib import Path
from itertools import product as iproduct

import numpy as np
import pytest
import torch

from miao.config import MiaoConfig
from miao.dataset import VolumeDataset

# Level voxel sizes of the standard 64^3 fixture are [1,1,1], [2,2,2], [4,4,4], so these
# resolutions map exactly to pyramid levels 0, 1, 2 (downsample ratio 1).
RES_1 = [[1, 1, 1]]
RES_2 = [[1, 1, 1], [2, 2, 2]]
RES_3 = [[1, 1, 1], [2, 2, 2], [4, 4, 4]]


class TestVolumeDataset:
    def test_basic_getitem(self, sample_config: dict):
        cfg = MiaoConfig(**sample_config)
        ds = VolumeDataset(cfg)

        assert len(ds) == 10

        sample = ds[0]
        assert "img" in sample
        assert "label" in sample
        assert "meta" in sample

        # output_axes="lzyx" → img shape: (L, Z, Y, X) = (3, 8, 8, 8)
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
                }
            ],
            resolutions=RES_2,
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
                }
            ],
            resolutions=RES_2,
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
                }
            ],
            resolutions=RES_1,
            output_axes="lxyz",
            patch_size=[8, 8, 8],
            samples_per_epoch=5,
        )
        ds = VolumeDataset(cfg)
        sample = ds[0]

        # output_axes="lxyz" → (L, X, Y, Z)
        assert sample["img"].shape == (1, 8, 8, 8)

    def test_level_dim_placement(self, zarr2_volume: Path):
        """Test that 'l' can be placed at different positions."""
        cfg = MiaoConfig(
            volumes=[
                {
                    "name": "test",
                    "path": str(zarr2_volume),
                    "image_key": "raw",
                }
            ],
            resolutions=RES_2,
            output_axes="xyzl",
            patch_size=[8, 8, 8],
            samples_per_epoch=5,
        )
        ds = VolumeDataset(cfg)
        sample = ds[0]

        # output_axes="xyzl" → (X, Y, Z, L) = (8, 8, 8, 2)
        assert sample["img"].shape == (8, 8, 8, 2)

    def test_meta_contents(self, sample_config: dict):
        cfg = MiaoConfig(**sample_config)
        ds = VolumeDataset(cfg)
        sample = ds[0]

        meta = sample["meta"]
        assert "volume" in meta
        assert "coordinate" in meta
        assert "resolutions" in meta
        assert "source_levels" in meta
        assert meta["volume"] == "test_raw"
        assert meta["resolutions"] == [[1, 1, 1], [2, 2, 2], [4, 4, 4]]
        # These resolutions map exactly to pyramid levels 0, 1, 2.
        assert meta["source_levels"] == [0, 1, 2]
        assert len(meta["coordinate"]) == 3

    def test_multiple_volumes_weighted(self, zarr2_volume: Path):
        """Test that volumes are sampled according to weights."""
        cfg = MiaoConfig(
            volumes=[
                {
                    "name": "vol_a",
                    "path": str(zarr2_volume),
                    "image_key": "raw",
                    "weight": 0.99,
                },
                {
                    "name": "vol_b",
                    "path": str(zarr2_volume),
                    "image_key": "raw",
                    "weight": 0.01,
                },
            ],
            resolutions=RES_1,
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
                }
            ],
            resolutions=RES_1,
            output_axes="lzyx",
            patch_size=[8, 8, 8],
            samples_per_epoch=5,
        )
        ds = VolumeDataset(cfg)
        sample = ds[0]
        assert sample["img"].shape == (1, 8, 8, 8)

    def test_downsample_from_pyramid(self, zarr2_volume: Path):
        """A resolution between stored levels reads the coarsest qualifying level and downsamples.

        Target [3,3,3]: level voxels are [1,1,1],[2,2,2],[4,4,4]. The coarsest level <= 3 on
        every axis is level 1 (voxel 2), so it reads level 1 and resamples 2->3.
        """
        cfg = MiaoConfig(
            volumes=[
                {
                    "name": "test",
                    "path": str(zarr2_volume),
                    "image_key": "raw",
                }
            ],
            resolutions=[[1, 1, 1], [3, 3, 3]],
            output_axes="lzyx",
            patch_size=[8, 8, 8],
            samples_per_epoch=3,
        )
        ds = VolumeDataset(cfg)
        sample = ds[0]
        assert sample["img"].shape == (2, 8, 8, 8)
        assert sample["meta"]["source_levels"] == [0, 1]
        # Physical extents: scale 0 covers 8*1=8 units; scale 1 covers 8*3=24 units per axis.
        bbox = sample["bbox"].numpy()  # (L, 2, n_spatial)
        ext0 = bbox[0, 1] - bbox[0, 0]
        ext1 = bbox[1, 1] - bbox[1, 0]
        assert np.allclose(ext0, 8.0)
        assert np.allclose(ext1, 24.0)

    def test_resolutions_length_mismatch(self, zarr2_volume: Path):
        """A per-volume resolutions override of a different length than the global raises."""
        with pytest.raises(ValueError, match="must define the same number of scales"):
            MiaoConfig(
                volumes=[
                    {
                        "name": "test",
                        "path": str(zarr2_volume),
                        "image_key": "raw",
                        "resolutions": [[1, 1, 1]],
                    }
                ],
                resolutions=RES_3,
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
                    }
                ],
                resolutions=RES_1,
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
                    "normalize": True,
                    "normalize_min": 100,
                    "normalize_max": 200,
                }
            ],
            resolutions=RES_1,
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
                    "normalize": True,
                    "normalize_min": 0,
                    "normalize_max": 128,
                }
            ],
            resolutions=RES_1,
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
                    "normalize": True,
                    "normalize_min": lo,
                    "normalize_max": hi,
                }
            ],
            resolutions=RES_1,
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
            volumes=[{"name": "test", "path": str(zarr2_volume), "image_key": "raw"}],
            resolutions=RES_1,
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
            volumes=[{"name": "test", "path": str(zarr2_volume), "image_key": "raw"}],
            resolutions=RES_1,
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
            volumes=[{"name": "test", "path": str(zarr2_volume), "image_key": "raw"}],
            resolutions=RES_1,
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
            volumes=[{"name": "test", "path": str(zarr2_volume), "image_key": "raw"}],
            resolutions=RES_1,
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
            volumes=[{"name": "test", "path": str(zarr2_volume), "image_key": "raw"}],
            resolutions=RES_1,
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
            volumes=[{"name": "test", "path": str(zarr2_volume), "image_key": "raw"}],
            resolutions=RES_1,
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
            volumes=[{"name": "test", "path": str(zarr2_volume), "image_key": "raw"}],
            resolutions=RES_1,
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
                {"name": "vol_a", "path": str(zarr2_volume), "image_key": "raw"},
                {"name": "vol_b", "path": str(zarr2_volume), "image_key": "raw"},
            ],
            resolutions=RES_1,
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
                volumes=[{"name": "test", "path": str(zarr2_volume), "image_key": "raw"}],
                resolutions=RES_1,
                output_axes="lzyx",
                patch_size=[8, 8, 8],
                sampling="sequential",
                overlap=8,  # equal to patch_size → stride=0
            )

    def test_sequential_overlap_negative_raises(self, zarr2_volume: Path):
        """Negative overlap raises ValueError."""
        with pytest.raises(ValueError, match="overlap"):
            MiaoConfig(
                volumes=[{"name": "test", "path": str(zarr2_volume), "image_key": "raw"}],
                resolutions=RES_1,
                output_axes="lzyx",
                patch_size=[8, 8, 8],
                sampling="sequential",
                overlap=-1,
            )

    def test_sequential_per_axis_overlap(self, zarr2_volume: Path):
        """Per-axis overlap list: each axis uses its own overlap value."""
        cfg = MiaoConfig(
            volumes=[{"name": "test", "path": str(zarr2_volume), "image_key": "raw"}],
            resolutions=RES_1,
            output_axes="lzyx",
            patch_size=[8, 8, 8],
            sampling="sequential",
            overlap=[4, 0, 0],  # only Z has overlap; in output ZYX order
        )
        ds = VolumeDataset(cfg)
        # Z: stride=4 → 15 positions; Y,X: stride=8 → 8 positions each
        assert len(ds) == 15 * 8 * 8

    # ── Anisotropic volume: resolution targeting (replaces the old isotropic flag) ──
    #
    # The anisotropic fixture is 20×100×100 (ZYX) with voxel [5,1,1] and a single pyramid
    # level. Requesting an isotropic 1-unit output resolution [1,1,1] cannot downsample on Z
    # (voxel 5 > 1), so it reads level 0 and upsamples Z: read shape ceil([10,10,10]*[1/5,1,1])
    # = [2,10,10], resampled up to the [10,10,10] output patch.

    def test_anisotropic_sequential_grid_size(self, zarr2_volume_anisotropic: Path):
        """Isotropic-target sequential grid tiles the volume at the output resolution."""
        cfg = MiaoConfig(
            volumes=[{"name": "test", "path": str(zarr2_volume_anisotropic), "image_key": "raw"}],
            resolutions=[[1, 1, 1]],
            output_axes="lzyx",
            patch_size=[10, 10, 10],
            sampling="sequential",
        )
        ds = VolumeDataset(cfg)
        # min_center=[1,5,5], max_center=[19,95,95]; ref stride per axis = patch * res/voxel
        # = [10,10,10]*[1/5,1,1] = [2,10,10].
        # Z: range(1,20,2) = 10 positions; Y,X: range(5,96,10) = 10 positions each.
        assert len(ds) == 10 ** 3

    def test_anisotropic_first_coordinate(self, zarr2_volume_anisotropic: Path):
        """First sequential position is at min_center in the level-0 reference frame."""
        cfg = MiaoConfig(
            volumes=[{"name": "test", "path": str(zarr2_volume_anisotropic), "image_key": "raw"}],
            resolutions=[[1, 1, 1]],
            output_axes="lzyx",
            patch_size=[10, 10, 10],
            sampling="sequential",
        )
        ds = VolumeDataset(cfg)
        sample = ds[0]
        assert "coordinate" in sample["meta"]
        assert sample["meta"]["coordinate"] == [1, 5, 5]
        assert sample["meta"]["resolutions"] == [[1, 1, 1]]
        # Single stored level → reads level 0 and upsamples.
        assert sample["meta"]["source_levels"] == [0]

    def test_anisotropic_output_shape(self, zarr2_volume_anisotropic: Path):
        """Output tensor matches patch_size after resampling the anisotropic read."""
        cfg = MiaoConfig(
            volumes=[{"name": "test", "path": str(zarr2_volume_anisotropic), "image_key": "raw"}],
            resolutions=[[1, 1, 1]],
            output_axes="lzyx",
            patch_size=[10, 10, 10],
            sampling="sequential",
        )
        ds = VolumeDataset(cfg)
        sample = ds[0]
        assert sample["img"].shape == (1, 10, 10, 10)

    def test_meta_has_resolutions_and_levels(self, zarr2_volume: Path):
        """meta carries the target resolutions and resolved source levels."""
        cfg = MiaoConfig(
            volumes=[{"name": "test", "path": str(zarr2_volume), "image_key": "raw"}],
            resolutions=RES_1,
            output_axes="lzyx",
            patch_size=[8, 8, 8],
            samples_per_epoch=5,
        )
        ds = VolumeDataset(cfg)
        meta = ds[0]["meta"]
        assert meta["resolutions"] == [[1, 1, 1]]
        assert meta["source_levels"] == [0]


class TestResolutionSampling:
    """Random per-sample resolution sampling (resolution_sampling)."""

    def _cfg(self, zarr2_volume, **spec_overrides):
        spec = {
            "strategy": "log_uniform",
            "n_scales": 2,
            "min": [1, 1, 1],
            "max": [4, 4, 4],
            "isotropic": True,
            **spec_overrides,
        }
        return MiaoConfig(
            volumes=[{"name": "test", "path": str(zarr2_volume), "image_key": "raw"}],
            resolution_sampling=spec,
            output_axes="lzyx",
            patch_size=[8, 8, 8],
            samples_per_epoch=30,
        )

    def test_output_shape_stable(self, zarr2_volume: Path):
        """Output is always patch_size regardless of the sampled resolution."""
        ds = VolumeDataset(self._cfg(zarr2_volume))
        np.random.seed(0)
        for i in range(30):
            assert ds[i]["img"].shape == (2, 8, 8, 8)

    def test_sampled_within_range_and_sorted(self, zarr2_volume: Path):
        ds = VolumeDataset(self._cfg(zarr2_volume))
        np.random.seed(1)
        for i in range(30):
            res = np.array(ds[i]["meta"]["resolutions"])
            assert res.min() >= 1.0 - 1e-6 and res.max() <= 4.0 + 1e-6
            # sorted fine -> coarse by per-scale geometric mean
            assert np.prod(res[0]) <= np.prod(res[1]) + 1e-9

    def test_isotropic_equal_per_axis(self, zarr2_volume: Path):
        ds = VolumeDataset(self._cfg(zarr2_volume, isotropic=True))
        np.random.seed(2)
        for i in range(10):
            for r in ds[i]["meta"]["resolutions"]:
                assert len(set(r)) == 1, r

    def test_per_axis_varies(self, zarr2_volume: Path):
        """Non-isotropic sampling can produce anisotropic voxels."""
        ds = VolumeDataset(self._cfg(zarr2_volume, isotropic=False, n_scales=1))
        np.random.seed(3)
        saw_aniso = False
        for i in range(30):
            r = ds[i]["meta"]["resolutions"][0]
            if len(set(round(v, 6) for v in r)) > 1:
                saw_aniso = True
                break
        assert saw_aniso

    def test_draws_vary_across_calls(self, zarr2_volume: Path):
        ds = VolumeDataset(self._cfg(zarr2_volume))
        np.random.seed(4)
        first = [tuple(ds[i]["meta"]["resolutions"][0]) for i in range(20)]
        assert len(set(first)) > 1

    def test_seeded_reproducible(self, zarr2_volume: Path):
        ds = VolumeDataset(self._cfg(zarr2_volume))
        np.random.seed(7)
        a = [ds[i]["meta"]["resolutions"] for i in range(10)]
        np.random.seed(7)
        b = [ds[i]["meta"]["resolutions"] for i in range(10)]
        assert a == b

    def test_source_levels_valid(self, zarr2_volume: Path):
        """Sampled resolutions in [1,4] resolve to existing pyramid levels 0/1/2."""
        ds = VolumeDataset(self._cfg(zarr2_volume))
        np.random.seed(5)
        for i in range(20):
            for lvl in ds[i]["meta"]["source_levels"]:
                assert lvl in (0, 1, 2)

    def test_with_labels(self, zarr2_volume: Path):
        """Sampling works alongside labels; label output matches patch_size."""
        cfg = MiaoConfig(
            volumes=[
                {
                    "name": "test",
                    "path": str(zarr2_volume),
                    "image_key": "raw",
                    "label_key": "labels/seg",
                }
            ],
            resolution_sampling={
                "strategy": "log_uniform",
                "n_scales": 2,
                "min": [1, 1, 1],
                "max": [4, 4, 4],
                "isotropic": True,
            },
            output_axes="lzyx",
            patch_size=[8, 8, 8],
            samples_per_epoch=10,
        )
        ds = VolumeDataset(cfg)
        np.random.seed(6)
        sample = ds[0]
        assert sample["img"].shape == (2, 8, 8, 8)
        assert sample["label"].shape == (2, 8, 8, 8)


class TestSampleWindows:
    """Multi-scale window sampling (sample_windows=True)."""

    def _cfg(self, zarr2_volume, resolutions):
        return MiaoConfig(
            volumes=[{"name": "test", "path": str(zarr2_volume), "image_key": "raw"}],
            resolutions=resolutions,
            output_axes="lzyx",
            patch_size=[8, 8, 8],
            sample_windows=True,
            samples_per_epoch=200,
        )

    def test_pyramid_aligned_resolutions(self, zarr2_volume: Path):
        """Resolutions matching pyramid levels never raise."""
        ds = VolumeDataset(self._cfg(zarr2_volume, [[1, 1, 1], [2, 2, 2]]))
        np.random.seed(0)
        for i in range(200):
            assert ds[i]["img"].shape == (2, 8, 8, 8)

    def test_misaligned_resolutions_do_not_raise(self, zarr2_volume: Path):
        """Regression: resolutions not aligned to a pyramid level produce odd (ceil'd) read
        shapes, which previously made the covering+ROI sampler infeasible at volume boundaries
        and raised ValueError. Without a bounding_box the coarse origin is now bounded only by
        covering + volume, so sampling always succeeds."""
        ds = VolumeDataset(self._cfg(zarr2_volume, [[1.3, 1.3, 1.3], [2, 2, 2]]))
        np.random.seed(0)
        for i in range(200):
            assert ds[i]["img"].shape == (2, 8, 8, 8)

    def test_coarse_covers_fine(self, zarr2_volume: Path):
        """The coarser scale's patch must contain the finer scale's patch (physical bbox)."""
        ds = VolumeDataset(self._cfg(zarr2_volume, [[1.3, 1.3, 1.3], [2, 2, 2]]))
        np.random.seed(1)
        for i in range(50):
            bb = ds[i]["bbox"].numpy()  # (L, 2, n_spatial) absolute physical coords
            assert np.all(bb[1, 0] <= bb[0, 0] + 1e-6)  # coarse min <= fine min
            assert np.all(bb[1, 1] >= bb[0, 1] - 1e-6)  # coarse max >= fine max


class TestBoundingBox:
    """bounding_box strictly contains every window's read extent (all scales)."""

    # bbox in finest/level-0 voxels, storage (zyx) order — same frame as the bbox tensor when
    # output_axes == "lzyx".
    BB = [[10, 50], [12, 48], [9, 55]]

    def _assert_inside(self, ds, n=300, seed=0):
        fv = ds._volumes[0].finest_voxel_size  # zyx
        bb = np.array(self.BB, dtype=float)
        np.random.seed(seed)
        for i in range(n):
            bbx = ds[i]["bbox"].numpy()  # (L, 2, 3) absolute physical, output spatial = zyx
            ref_lo = bbx[:, 0, :] / fv
            ref_hi = bbx[:, 1, :] / fv
            assert np.all(ref_lo >= bb[:, 0] - 1e-6), (ref_lo, bb[:, 0])
            assert np.all(ref_hi <= bb[:, 1] + 1e-6), (ref_hi, bb[:, 1])

    def _cfg(self, zarr2_volume, **kw):
        vol = {"name": "v", "path": str(zarr2_volume), "image_key": "raw", "bounding_box": self.BB}
        vol.update(kw.pop("vol", {}))
        return MiaoConfig(
            volumes=[vol], output_axes="lzyx", patch_size=[8, 8, 8], samples_per_epoch=300, **kw
        )

    def test_centered_multiscale(self, zarr2_volume: Path):
        self._assert_inside(VolumeDataset(self._cfg(zarr2_volume, resolutions=RES_3)))

    def test_centered_misaligned(self, zarr2_volume: Path):
        cfg = self._cfg(zarr2_volume, resolutions=[[1.3, 1.3, 1.3], [2.0, 2.0, 2.0]])
        self._assert_inside(VolumeDataset(cfg))

    def test_sample_windows_misaligned(self, zarr2_volume: Path):
        cfg = self._cfg(
            zarr2_volume, resolutions=[[1.3, 1.3, 1.3], [2.0, 2.0, 2.0]], sample_windows=True
        )
        self._assert_inside(VolumeDataset(cfg))

    def test_sample_windows_three_scales(self, zarr2_volume: Path):
        cfg = self._cfg(
            zarr2_volume,
            resolutions=[[1.1, 1.1, 1.1], [1.9, 1.9, 1.9], [3.3, 3.3, 3.3]],
            sample_windows=True,
        )
        self._assert_inside(VolumeDataset(cfg))

    def test_with_labels(self, zarr2_volume: Path):
        cfg = self._cfg(
            zarr2_volume,
            resolutions=[[1.3, 1.3, 1.3], [2.0, 2.0, 2.0]],
            sample_windows=True,
            vol={"label_key": "labels/seg"},
        )
        self._assert_inside(VolumeDataset(cfg))

    def test_resolution_sampling(self, zarr2_volume: Path):
        cfg = self._cfg(
            zarr2_volume,
            resolution_sampling={
                "strategy": "log_uniform",
                "n_scales": 2,
                "min": [1, 1, 1],
                "max": [4, 4, 4],
                "isotropic": True,
            },
        )
        self._assert_inside(VolumeDataset(cfg))

    def test_too_small_box_raises(self, zarr2_volume: Path):
        """A box smaller than the coarsest window raises a clear error at dataset build."""
        cfg = MiaoConfig(
            volumes=[
                {
                    "name": "v",
                    "path": str(zarr2_volume),
                    "image_key": "raw",
                    "bounding_box": [[10, 14], [10, 14], [10, 14]],
                }
            ],
            resolutions=[[1, 1, 1], [4, 4, 4]],
            output_axes="lzyx",
            patch_size=[8, 8, 8],
        )
        with pytest.raises(ValueError, match="too small"):
            VolumeDataset(cfg)
