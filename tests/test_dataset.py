"""Tests for VolumeDataset."""

from pathlib import Path

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

    def test_axis_reorientation_cyclic(self, zarr2_volume: Path):
        """Non-cubic patch with a cyclic storage↔output permutation.

        Storage is zyx and output spatial order is xzy — a 3-cycle, unlike the
        zyx↔xyz reversal which is its own inverse and so can't distinguish a
        permutation from its inverse. Guards the patch_size/resolution mapping
        in axes.map_patch_size_to_input.
        """
        import zarr

        cfg = MiaoConfig(
            volumes=[
                {
                    "name": "test",
                    "path": str(zarr2_volume),
                    "image_key": "raw",
                }
            ],
            resolutions=RES_1,
            output_axes="lxzy",
            patch_size=[6, 8, 12],  # X=6, Z=8, Y=12 — distinct per axis
            samples_per_epoch=5,
        )
        ds = VolumeDataset(cfg)
        sample = ds[0]

        # output_axes="lxzy" → (L, X, Z, Y)
        assert sample["img"].shape == (1, 6, 8, 12)

        # meta["coordinate"] is in output_axes spatial order ("xzy"); map it back to
        # storage (zyx) to reconstruct the on-disk crop, transposed zyx → xzy. At
        # resolution [1,1,1] the read is level 0 with no resampling, so the match is exact.
        coord = sample["meta"]["coordinate"]  # output order: x, z, y
        center = np.array([coord["xzy".index(c)] for c in "zyx"])  # storage z, y, x order
        storage_shape = np.array([8, 12, 6])  # patch_size in storage (z, y, x) order
        origin = np.clip(center - storage_shape // 2, 0, 64 - storage_shape)
        raw = zarr.open_group(str(zarr2_volume), mode="r")["raw"]["0"][
            origin[0] : origin[0] + 8,
            origin[1] : origin[1] + 12,
            origin[2] : origin[2] + 6,
        ]
        expected = np.transpose(raw, (2, 0, 1))  # zyx → xzy
        np.testing.assert_allclose(sample["img"][0].numpy(), expected)

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

    def test_pixel_size_output(self, sample_config: dict):
        """pixel_size is a (L, Nd_spatial) float tensor matching the requested resolutions."""
        cfg = MiaoConfig(**sample_config)
        ds = VolumeDataset(cfg)
        sample = ds[0]

        assert "pixel_size" in sample
        ps = sample["pixel_size"]
        assert isinstance(ps, torch.Tensor)
        assert ps.dtype == torch.float32
        # 3 levels, 3 spatial axes
        assert ps.shape == (3, 3)
        # output_axes="lzyx" → output spatial order == storage order, so pixel_size
        # equals the config resolutions exactly.
        assert torch.allclose(ps, torch.tensor([[1.0, 1, 1], [2, 2, 2], [4, 4, 4]]))

    def test_pixel_size_output_axis_order(self, zarr2_volume: Path):
        """pixel_size is returned in output spatial order (exercises non-identity perm)."""
        cfg = MiaoConfig(
            volumes=[{"name": "test", "path": str(zarr2_volume), "image_key": "raw"}],
            resolutions=RES_2,
            output_axes="lxyz",  # storage is zyx → spatial perm is non-trivial
            patch_size=[8, 8, 8],
            samples_per_epoch=5,
        )
        ds = VolumeDataset(cfg)
        ps = ds[0]["pixel_size"]
        # Recovers the config resolutions (given in output spatial order) per level.
        assert ps.shape == (2, 3)
        assert torch.allclose(ps, torch.tensor([[1.0, 1, 1], [2, 2, 2]]))

    def test_pixel_size_dataloader(self, sample_config: dict):
        """Default collate stacks pixel_size to (B, L, Nd_spatial)."""
        cfg = MiaoConfig(**sample_config)
        ds = VolumeDataset(cfg)
        dl = torch.utils.data.DataLoader(ds, batch_size=2, num_workers=0)
        batch = next(iter(dl))
        assert batch["pixel_size"].shape == (2, 3, 3)

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

    def test_patch_normalize_single_scale(self, zarr2_volume: Path):
        """Single scale: the returned sample has zero mean and unit standard deviation."""
        cfg = MiaoConfig(
            volumes=[
                {
                    "name": "test",
                    "path": str(zarr2_volume),
                    "image_key": "raw",
                    "patch_normalize": True,
                }
            ],
            resolutions=RES_1,
            output_axes="lzyx",
            patch_size=[8, 8, 8],
            samples_per_epoch=1,
        )
        ds = VolumeDataset(cfg)
        img = ds[0]["img"]
        assert img.mean().abs() < 1e-5
        assert abs(img.std().item() - 1.0) < 1e-5

    def test_patch_normalize_multi_scale_uses_coarsest(self, zarr2_volume: Path):
        """Multi-scale: every scale is normalized by the coarsest crop's mean/std."""

        def sample(patch_normalize: bool) -> torch.Tensor:
            cfg = MiaoConfig(
                volumes=[
                    {
                        "name": "test",
                        "path": str(zarr2_volume),
                        "image_key": "raw",
                        "patch_normalize": patch_normalize,
                    }
                ],
                resolutions=RES_3,
                output_axes="lzyx",
                patch_size=[8, 8, 8],
                sampling="sequential",
            )
            return VolumeDataset(cfg)[0]["img"]

        raw = sample(False)
        normed = sample(True)

        coarsest = raw[-1]
        expected = (raw - coarsest.mean()) / coarsest.std()
        assert torch.allclose(normed, expected, atol=1e-5)

        # Only the reference scale is exactly standardized; the finer ones share its statistics.
        assert normed[-1].mean().abs() < 1e-5
        assert abs(normed[-1].std().item() - 1.0) < 1e-5

    def test_patch_normalize_constant_patch(self, zarr2_volume: Path):
        """A zero-variance patch must not produce inf/nan."""
        cfg = MiaoConfig(
            volumes=[
                {
                    "name": "test",
                    "path": str(zarr2_volume),
                    "image_key": "raw",
                    "normalize": True,
                    "normalize_min": 0.0,
                    "normalize_max": 1e-12,  # clips everything to the upper bound
                    "patch_normalize": True,
                }
            ],
            resolutions=RES_1,
            output_axes="lzyx",
            patch_size=[8, 8, 8],
            samples_per_epoch=1,
        )
        ds = VolumeDataset(cfg)
        img = ds[0]["img"]
        assert torch.isfinite(img).all()

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

    # Default: one isotropic range 1..4 (shorthand), 2 scales.
    DEFAULT_SPEC = {"strategy": "log_uniform", "ranges": [[[1], [4]]], "n_scales": 2}

    def _cfg(self, zarr2_volume, spec=None, **vol):
        v = {"name": "test", "path": str(zarr2_volume), "image_key": "raw", **vol}
        return MiaoConfig(
            volumes=[v],
            resolution_sampling=spec or self.DEFAULT_SPEC,
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
        ds = VolumeDataset(self._cfg(zarr2_volume))  # isotropic ranges
        np.random.seed(2)
        for i in range(10):
            for r in ds[i]["meta"]["resolutions"]:
                assert len(set(r)) == 1, r

    def test_per_axis_varies(self, zarr2_volume: Path):
        """A per-axis range (full-length bounds) can produce anisotropic voxels."""
        spec = {"strategy": "log_uniform", "ranges": [[[1, 1, 1], [4, 4, 4]]], "n_scales": 1}
        ds = VolumeDataset(self._cfg(zarr2_volume, spec))
        np.random.seed(3)
        saw_aniso = False
        for i in range(30):
            r = ds[i]["meta"]["resolutions"][0]
            if len(set(round(v, 6) for v in r)) > 1:
                saw_aniso = True
                break
        assert saw_aniso

    def test_multiple_ranges_total_scales_and_bands(self, zarr2_volume: Path):
        """Two disjoint ranges with per-range counts -> total scales = sum; draws fall in bands."""
        spec = {
            "strategy": "log_uniform",
            "ranges": [[[1], [2]], [[3], [4]]],
            "n_scales": [1, 2],
        }
        ds = VolumeDataset(self._cfg(zarr2_volume, spec))
        np.random.seed(8)
        for i in range(30):
            res = np.array(ds[i]["meta"]["resolutions"])
            assert res.shape == (3, 3)  # total 3 scales, 3 axes
            vals = sorted(res[:, 0])  # isotropic -> per-scale scalar
            # one draw in [1,2], two draws in [3,4] (after sorting, first in band 1, rest band 2)
            assert 1.0 - 1e-6 <= vals[0] <= 2.0 + 1e-6
            assert all(3.0 - 1e-6 <= v <= 4.0 + 1e-6 for v in vals[1:])

    def test_draws_vary_across_calls(self, zarr2_volume: Path):
        ds = VolumeDataset(self._cfg(zarr2_volume))
        np.random.seed(4)
        first = [tuple(ds[i]["meta"]["resolutions"][0]) for i in range(20)]
        assert len(set(first)) > 1

    def test_pixel_size_reflects_sampled_resolutions(self, zarr2_volume: Path):
        """In sampling mode, pixel_size is recomputed per sample from the drawn resolutions."""
        ds = VolumeDataset(self._cfg(zarr2_volume))
        np.random.seed(4)
        seen = set()
        for i in range(20):
            sample = ds[i]
            ps = sample["pixel_size"]
            assert ps.shape == (2, 3)
            # output_axes="lzyx" → output spatial order == storage order, so pixel_size
            # matches meta["resolutions"] (which reports the actually-drawn resolutions).
            expected = torch.tensor(sample["meta"]["resolutions"], dtype=torch.float32)
            assert torch.allclose(ps, expected)
            seen.add(tuple(ps[0].tolist()))
        # Draws genuinely vary across calls (not a constant broadcast).
        assert len(seen) > 1

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
        ds = VolumeDataset(self._cfg(zarr2_volume, label_key="labels/seg"))
        np.random.seed(6)
        sample = ds[0]
        assert sample["img"].shape == (2, 8, 8, 8)
        assert sample["label"].shape == (2, 8, 8, 8)

    def test_level_spanning_range_no_overflow(self, zarr2_volume: Path):
        """A range spanning multiple pyramid levels (voxels 1/2/4) reads cleanly across many
        draws — different draws select different levels with different read shapes, and the
        per-scale origin must stay in-bounds regardless of the (coarsest-resolution) bounds."""
        spec = {"strategy": "log_uniform", "ranges": [[[1], [4]]], "n_scales": 1}
        ds = VolumeDataset(self._cfg(zarr2_volume, spec))
        np.random.seed(0)
        levels_seen = set()
        for i in range(400):
            s = ds[i]
            assert s["img"].shape == (1, 8, 8, 8)
            levels_seen.update(s["meta"]["source_levels"])
        assert len(levels_seen) > 1  # the range really does span levels

    def test_window_too_large_raises(self, zarr2_volume: Path):
        """If the coarsest sampled resolution makes the window exceed the volume, fail clearly."""
        # 64^3 fixture, level voxels 1/2/4. target 40 -> level 2 (voxel 4), read 8*40/4=80 > 64.
        cfg = MiaoConfig(
            volumes=[{"name": "test", "path": str(zarr2_volume), "image_key": "raw"}],
            resolution_sampling={"strategy": "log_uniform", "ranges": [[[1], [40]]], "n_scales": 1},
            output_axes="lzyx",
            patch_size=[8, 8, 8],
        )
        with pytest.raises(ValueError, match="does not fit the volume"):
            VolumeDataset(cfg)


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
                "ranges": [[[1], [4]]],
                "n_scales": 2,
            },
        )
        self._assert_inside(VolumeDataset(cfg))

    def test_output_axes_order(self, zarr2_volume: Path):
        """bounding_box is given in output_axes spatial order, not storage order.

        Storage is zyx; output spatial order is xyz. An asymmetric box exercises the
        permutation: it is reordered internally to storage (zyx) and every returned
        window (reported in output xyz order) stays inside the box as specified.
        """
        # output xyz order: x in [9, 55], y in [12, 48], z in [10, 50].
        bb_xyz = [[9, 55], [12, 48], [10, 50]]
        cfg = MiaoConfig(
            volumes=[{
                "name": "v", "path": str(zarr2_volume), "image_key": "raw",
                "bounding_box": bb_xyz,
            }],
            output_axes="lxyz", patch_size=[8, 8, 8], resolutions=RES_3,
            samples_per_epoch=300,
        )
        ds = VolumeDataset(cfg)

        # Stored bounding_box is reordered to storage (zyx): [z, y, x].
        stored = ds._volumes[0].bounding_box
        np.testing.assert_array_equal(stored, [[10, 50], [12, 48], [9, 55]])

        # Every window (bbox tensor is in output xyz order; isotropic volume so fv=1) fits.
        fv = ds._volumes[0].finest_voxel_size
        bb = np.array(bb_xyz, dtype=float)
        np.random.seed(0)
        for i in range(300):
            bbx = ds[i]["bbox"].numpy()  # (L, 2, 3), output spatial = xyz
            assert np.all(bbx[:, 0, :] / fv >= bb[:, 0] - 1e-6)
            assert np.all(bbx[:, 1, :] / fv <= bb[:, 1] + 1e-6)

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

            
class TestExpFactor:
    """exp_factor divides metadata voxel sizes, so targets are effective (pre-expansion) units.

    The 64^3 fixture stores level voxel sizes [1,1,1], [2,2,2], [4,4,4]; with exp_factor=4 the
    effective sizes are [0.25,...], [0.5,...], [1.0,...].
    """

    def _cfg(self, zarr2_volume, resolutions, exp_factor=1.0, **kw):
        vol = {"name": "v", "path": str(zarr2_volume), "image_key": "raw"}
        if exp_factor != 1.0:
            vol["exp_factor"] = exp_factor
        vol.update(kw.pop("vol", {}))
        return MiaoConfig(
            volumes=[vol],
            output_axes="lzyx",
            patch_size=[8, 8, 8],
            samples_per_epoch=10,
            resolutions=resolutions,
            **kw,
        )

    def test_scales_metadata_voxel_sizes(self, zarr2_volume: Path):
        ds = VolumeDataset(self._cfg(zarr2_volume, [[0.5, 0.5, 0.5]], exp_factor=4.0))
        vi = ds._volumes[0]
        assert np.allclose(vi.finest_voxel_size, [0.25, 0.25, 0.25])
        assert np.allclose(vi.img_level_voxels[2], [1.0, 1.0, 1.0])

    def test_selects_coarser_level_than_without(self, zarr2_volume: Path):
        # 0.5 is finer than every stored size, so without exp_factor it falls back to level 0;
        # with exp_factor=4 it matches the effective size of level 1 exactly.
        plain = VolumeDataset(self._cfg(zarr2_volume, [[0.5, 0.5, 0.5]]))
        assert plain._volumes[0].scales.chosen_levels == [0]

        ds = VolumeDataset(self._cfg(zarr2_volume, [[0.5, 0.5, 0.5]], exp_factor=4.0))
        scales = ds._volumes[0].scales
        assert scales.chosen_levels == [1]
        # target == effective level voxel size, so no resampling: read == patch_size
        assert scales.read_shapes[0].tolist() == [8, 8, 8]

    def test_equivalent_to_scaled_resolutions(self, zarr2_volume: Path):
        """exp_factor=F at target R/F must behave exactly like exp_factor=1 at target R."""
        res = [[1, 1, 1], [2, 2, 2], [4, 4, 4]]
        scaled = [[r / 4.0 for r in lvl] for lvl in res]
        plain = VolumeDataset(self._cfg(zarr2_volume, res))._volumes[0]
        exp = VolumeDataset(self._cfg(zarr2_volume, scaled, exp_factor=4.0))._volumes[0]

        assert exp.scales.chosen_levels == plain.scales.chosen_levels
        for a, b in zip(exp.scales.read_shapes, plain.scales.read_shapes):
            assert a.tolist() == b.tolist()
        for a, b in zip(exp.scales.relative_scale_factors, plain.scales.relative_scale_factors):
            assert np.allclose(a, b)
        # Center bounds live in level-0 voxels, so they are identical too.
        assert exp.min_center.tolist() == plain.min_center.tolist()
        assert exp.max_center.tolist() == plain.max_center.tolist()

    def test_labels_use_same_factor(self, zarr2_volume: Path):
        cfg = self._cfg(
            zarr2_volume,
            [[0.25, 0.25, 0.25], [0.5, 0.5, 0.5]],
            exp_factor=4.0,
            vol={"label_key": "labels/seg"},
        )
        scales = VolumeDataset(cfg)._volumes[0].scales
        assert scales.chosen_levels == [0, 1]
        assert scales.label_chosen_levels == scales.chosen_levels

    def test_reported_units_are_effective(self, zarr2_volume: Path):
        cfg = self._cfg(zarr2_volume, [[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]], exp_factor=4.0)
        ds = VolumeDataset(cfg)
        sample = ds[0]
        assert np.allclose(sample["pixel_size"].numpy(), [[0.5] * 3, [1.0] * 3])
        assert np.allclose(sample["meta"]["resolutions"], [[0.5] * 3, [1.0] * 3])
        # coordinate stays in level-0 voxel indices, so it is bounded by the volume shape.
        coord = np.array(sample["meta"]["coordinate"])
        assert np.all(coord >= 0) and np.all(coord < 64)
        # bbox is in effective physical units: extent == patch_size * pixel_size
        bbox = sample["bbox"].numpy()  # (L, 2, 3)
        assert np.allclose(bbox[:, 1, :] - bbox[:, 0, :], [[8 * 0.5] * 3, [8 * 1.0] * 3])

    def test_warns_when_outer_transform_also_present(self, tmp_path: Path):
        """exp_factor + a non-trivial outer OME-NGFF transform on the same volume risks
        applying the expansion factor twice."""
        from conftest import _create_ome_ngff_zarr2

        zarr_path = tmp_path / "outer_and_exp.zarr"
        _create_ome_ngff_zarr2(
            zarr_path,
            group_key="raw",
            base_shape=(64, 64, 64),
            num_scales=3,
            outer_scale=[0.25, 0.25, 0.25],
        )
        cfg = self._cfg(zarr_path, [[0.5, 0.5, 0.5]], exp_factor=4.0)
        with pytest.warns(UserWarning, match="exp_factor"):
            VolumeDataset(cfg)

    def test_no_warning_without_outer_transform(self, zarr2_volume: Path, recwarn):
        """The default fixture has no outer transform, so exp_factor alone should not warn."""
        cfg = self._cfg(zarr2_volume, [[0.5, 0.5, 0.5]], exp_factor=4.0)
        VolumeDataset(cfg)
        assert len(recwarn) == 0


class TestChunkAlignedSampling:
    """Tests for chunk_aligned=True random sampling."""

    def test_centers_within_chunk(self, zarr2_volume: Path):
        """Every sampled center produces a patch fitting within a single chunk."""
        # zarr2_volume: 64^3, chunks=32^3, patch=8^3 → chunk > patch
        cfg = MiaoConfig(
            volumes=[{
                "name": "test",
                "path": str(zarr2_volume),
                "image_key": "raw",
            }],
            resolutions=RES_1,
            output_axes="lzyx",
            patch_size=[8, 8, 8],
            samples_per_epoch=200,
            chunk_aligned=True,
        )
        ds = VolumeDataset(cfg)
        vol_info = ds._volumes[0]
        full_chunks = vol_info.image_meta.scales[0].chunks  # level 0 = reference frame
        spatial_chunks = [full_chunks[i] for i in vol_info.img_spatial_idx]

        for _ in range(200):
            center = ds._sample_chunk_aligned_center(vol_info)
            for ax in range(len(center)):
                half = vol_info.read_shape[ax] // 2
                patch_start = int(center[ax]) - half
                patch_end = patch_start + vol_info.read_shape[ax]
                chunk_sz = spatial_chunks[ax]
                assert patch_start // chunk_sz == (patch_end - 1) // chunk_sz, (
                    f"axis {ax}: patch [{patch_start}, {patch_end}) straddles "
                    f"chunk boundary (chunk_size={chunk_sz})"
                )

    def test_getitem_shapes(self, zarr2_volume: Path):
        """chunk_aligned=True produces correct output tensor shapes."""
        cfg = MiaoConfig(
            volumes=[{
                "name": "test",
                "path": str(zarr2_volume),
                "image_key": "raw",
                "label_key": "labels/seg",
            }],
            resolutions=RES_3,
            output_axes="lzyx",
            patch_size=[8, 8, 8],
            samples_per_epoch=10,
            chunk_aligned=True,
        )
        ds = VolumeDataset(cfg)
        sample = ds[0]
        assert sample["img"].shape == (3, 8, 8, 8)
        assert sample["label"].shape == (3, 8, 8, 8)

    def test_false_unchanged(self, zarr2_volume: Path):
        """chunk_aligned=False behaves identically to the default."""
        cfg = MiaoConfig(
            volumes=[{
                "name": "test",
                "path": str(zarr2_volume),
                "image_key": "raw",
            }],
            resolutions=RES_1,
            output_axes="lzyx",
            patch_size=[8, 8, 8],
            samples_per_epoch=5,
            chunk_aligned=False,
        )
        ds = VolumeDataset(cfg)
        sample = ds[0]
        assert sample["img"].shape == (1, 8, 8, 8)
        assert "grid_index" not in sample["meta"]

    def test_patch_equals_chunk_grid_locked(self, zarr2_volume: Path):
        """When patch_size == chunk_size, centers snap to chunk-boundary positions."""
        # zarr2_volume: 64^3 with 32^3 chunks, patch=32
        # half=16, valid center per chunk: c_lo=start+16, c_hi=start+32-32+16=start+16
        # → exactly 1 position per chunk. Chunks at [0,32]: centers are {16, 48}
        cfg = MiaoConfig(
            volumes=[{
                "name": "test",
                "path": str(zarr2_volume),
                "image_key": "raw",
            }],
            resolutions=RES_1,
            output_axes="lzyx",
            patch_size=[32, 32, 32],
            samples_per_epoch=50,
            chunk_aligned=True,
        )
        ds = VolumeDataset(cfg)
        vol_info = ds._volumes[0]
        valid_centers = {16, 48}
        for _ in range(50):
            center = ds._sample_chunk_aligned_center(vol_info)
            for ax in range(3):
                assert int(center[ax]) in valid_centers, (
                    f"axis {ax}: center {center[ax]} not in {valid_centers}"
                )

    def test_larger_chunk_has_diversity(self, zarr2_volume: Path):
        """When chunk > patch, centers have random freedom within each chunk."""
        # 64^3, chunks=32, patch=8 → 25 valid positions per chunk × 2 chunks
        cfg = MiaoConfig(
            volumes=[{
                "name": "test",
                "path": str(zarr2_volume),
                "image_key": "raw",
            }],
            resolutions=RES_1,
            output_axes="lzyx",
            patch_size=[8, 8, 8],
            samples_per_epoch=100,
            chunk_aligned=True,
        )
        ds = VolumeDataset(cfg)
        vol_info = ds._volumes[0]
        centers_ax0 = set()
        for _ in range(100):
            center = ds._sample_chunk_aligned_center(vol_info)
            centers_ax0.add(int(center[0]))
        assert len(centers_ax0) > 2, (
            f"Expected diverse centers when chunk > patch, got only {centers_ax0}"
        )


def _build_axes_volume(root: Path, img_axes: str, lbl_axes: str, n: int = 16) -> Path:
    """A minimal container whose image and label groups declare the given axis orders.

    Both hold the same physical data (values encode the physical voxel coordinate) written in
    each group's own on-disk order, with isotropic unit voxels.
    """
    import json

    import zarr
    from zarr.storage import LocalStore

    zz, yy, xx = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    val_zyx = (zz * 10000 + yy * 100 + xx).astype(np.float32)

    grp_root = zarr.open_group(LocalStore(str(root)), mode="a", zarr_format=2)
    for key, axes_str, dtype in [("raw", img_axes, "float32"), ("seg", lbl_axes, "uint32")]:
        spatial = "".join(c for c in axes_str if c in "xyz")
        data = np.transpose(val_zyx, ["zyx".index(c) for c in spatial]).astype(dtype)
        if "c" in axes_str:  # singleton channel at the position declared in axes_str
            data = np.expand_dims(data, axis=axes_str.index("c"))
        g = grp_root.create_group(key)
        a = g.create_array("0", shape=data.shape, chunks=data.shape, dtype=dtype, overwrite=True)
        a[:] = data
        axes = [
            {"name": c, "type": ("channel" if c == "c" else "space")}
            | ({} if c == "c" else {"unit": "micrometer"})
            for c in axes_str
        ]
        (root / key / ".zattrs").write_text(json.dumps({"multiscales": [{
            "version": "0.4", "axes": axes,
            "datasets": [{"path": "0", "coordinateTransformations": [
                {"type": "scale", "scale": [1.0] * len(axes_str)}]}],
        }]}))
    return root


class TestLabelAxisOrderValidation:
    """A label group whose SPATIAL axis order differs from the image's must be rejected.

    Reads are addressed in each array's own on-disk order while label read coordinates are
    derived from image-order quantities, so a mismatch silently returns transposed labels — with
    the correct shape and dtype, hence undetectable downstream. Real data exhibits this (an
    OME-NGFF container may store `raw` as xyz and a label group as zyx), so it is rejected up
    front instead.
    """

    def _cfg(self, path: Path) -> MiaoConfig:
        return MiaoConfig(
            volumes=[{"name": "v", "path": str(path), "image_key": "raw", "label_key": "seg"}],
            resolutions=RES_1, output_axes="lzyx", patch_size=[4, 4, 4], samples_per_epoch=2,
        )

    def test_matching_axes_accepted(self, tmp_path: Path):
        p = _build_axes_volume(tmp_path / "ok.zarr", "zyx", "zyx")
        ds = VolumeDataset(self._cfg(p))
        assert ds[0]["label"].shape == (1, 4, 4, 4)

    def test_reversed_label_axes_rejected(self, tmp_path: Path):
        p = _build_axes_volume(tmp_path / "rev.zarr", "zyx", "xyz")
        with pytest.raises(ValueError, match="do not match image spatial axes"):
            VolumeDataset(self._cfg(p))

    def test_cyclic_label_axes_rejected(self, tmp_path: Path):
        p = _build_axes_volume(tmp_path / "cyc.zarr", "zyx", "yzx")
        with pytest.raises(ValueError, match="do not match image spatial axes"):
            VolumeDataset(self._cfg(p))

    def test_error_names_both_keys_and_orders(self, tmp_path: Path):
        """The message has to be actionable — which volume, which keys, which orders."""
        p = _build_axes_volume(tmp_path / "msg.zarr", "zyx", "xyz")
        with pytest.raises(ValueError) as exc:
            VolumeDataset(self._cfg(p))
        msg = str(exc.value)
        for expected in ["'v'", "'zyx'", "'xyz'", "'raw'", "'seg'"]:
            assert expected in msg, f"{expected} missing from error:\n{msg}"

    def test_channel_only_difference_accepted(self, tmp_path: Path):
        """image 'cxyz' vs label 'xyz' is a channel-presence difference, not a spatial-order
        one — common in real data (all the nisb/* datasets) and handled correctly."""
        p = _build_axes_volume(tmp_path / "chan.zarr", "cxyz", "xyz")
        cfg = MiaoConfig(
            volumes=[{"name": "v", "path": str(p), "image_key": "raw", "label_key": "seg"}],
            resolutions=RES_1, output_axes="lzyx", patch_size=[4, 4, 4], samples_per_epoch=2,
        )
        ds = VolumeDataset(cfg)
        assert ds[0]["label"].shape == (1, 4, 4, 4)

    def test_channel_on_label_only_accepted(self, tmp_path: Path):
        p = _build_axes_volume(tmp_path / "chan2.zarr", "zyx", "zyxc")
        ds = VolumeDataset(self._cfg(p))
        assert ds[0]["label"].shape == (1, 4, 4, 4)

    def test_no_label_key_unaffected(self, tmp_path: Path):
        """Volumes without labels never hit the check."""
        p = _build_axes_volume(tmp_path / "nolbl.zarr", "zyx", "xyz")
        cfg = MiaoConfig(
            volumes=[{"name": "v", "path": str(p), "image_key": "raw"}],
            resolutions=RES_1, output_axes="lzyx", patch_size=[4, 4, 4], samples_per_epoch=2,
        )
        ds = VolumeDataset(cfg)
        assert ds[0]["label"].numel() == 0

    def test_rejected_regardless_of_output_axes(self, tmp_path: Path):
        """output_axes standardizes the returned tensor, not the read path — so it cannot
        rescue a storage-order mismatch, whichever order is requested."""
        p = _build_axes_volume(tmp_path / "out.zarr", "zyx", "xyz")
        for out_axes in ("lzyx", "lxyz", "xyzl"):
            cfg = MiaoConfig(
                volumes=[{"name": "v", "path": str(p), "image_key": "raw",
                          "label_key": "seg"}],
                resolutions=RES_1, output_axes=out_axes, patch_size=[4, 4, 4],
                samples_per_epoch=2,
            )
            with pytest.raises(ValueError, match="do not match image spatial axes"):
                VolumeDataset(cfg)
