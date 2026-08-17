"""Tests for the pure-function augmentation library."""

from pathlib import Path

import numpy as np
import pytest
import torch

from miao.augment import (
    _QUANTILE_MAX_ELEMENTS,
    _SPATIAL_PERMS,
    apply_rot90,
    draw_rot90,
    intensity_jitter,
    percentile_normalize,
    rot90isocube,
)
from miao.config import MiaoConfig
from miao.dataset import VolumeDataset


class TestApplyRot90:
    """Unit tests for the axis-aligned transform helper (no RNG)."""

    def test_identity(self):
        # "lzyx" → x=3, y=2, z=1
        t = torch.arange(2 * 4 * 4 * 4).reshape(2, 4, 4, 4)
        out = apply_rot90(t, (0, 1, 2), (False, False, False), spatial_dims=(3, 2, 1))
        assert torch.equal(out, t)

    def test_flip_x_only(self):
        t = torch.arange(1 * 4 * 4 * 4).reshape(1, 4, 4, 4)
        out = apply_rot90(t, (0, 1, 2), (True, False, False), spatial_dims=(3, 2, 1))
        assert torch.equal(out, torch.flip(t, dims=[3]))

    def test_swap_x_z(self):
        # perm (2,1,0) swaps the x and z slots → transpose of dims 1 (z) and 3 (x).
        t = torch.arange(1 * 4 * 4 * 4).reshape(1, 4, 4, 4)
        out = apply_rot90(t, (2, 1, 0), (False, False, False), spatial_dims=(3, 2, 1))
        assert torch.equal(out, t.transpose(1, 3))

    def test_leaves_nonspatial_dims(self):
        # "lzyxc": l=0, z=1, y=2, x=3, c=4 → spatial dims (x,y,z)=(3,2,1); l and c untouched.
        t = torch.arange(2 * 4 * 4 * 4 * 3).reshape(2, 4, 4, 4, 3)
        out = apply_rot90(t, (1, 2, 0), (True, False, True), spatial_dims=(3, 2, 1))
        assert out.shape == t.shape

    def test_negative_dims_default(self):
        # The (-3, -2, -1) default addresses the trailing spatial dims of any layout:
        # flipping spatial slot 2 (= dim -1) flips the last dim.
        t = torch.arange(2 * 4 * 4 * 4).reshape(2, 4, 4, 4)
        out = apply_rot90(t, (0, 1, 2), (False, False, True))
        assert torch.equal(out, torch.flip(t, dims=[3]))

    def test_group_has_48_distinct_elements(self):
        # Distinct labels per voxel so any two different transforms differ.
        t = torch.arange(1 * 3 * 3 * 3).reshape(1, 3, 3, 3)
        seen = set()
        for perm in _SPATIAL_PERMS:
            for fmask in range(8):
                flips = (bool(fmask & 1), bool(fmask & 2), bool(fmask & 4))
                out = apply_rot90(t, perm, flips, spatial_dims=(3, 2, 1))
                seen.add(out.numpy().tobytes())
        assert len(_SPATIAL_PERMS) == 6
        assert len(seen) == 48

    def test_low_rank_asserts(self):
        # The empty no-label sentinel and 2D tensors must be rejected, not wrapped into
        # duplicate dims by the negative-index modulo.
        for bad in (torch.empty(0), torch.zeros(4, 4)):
            with pytest.raises(AssertionError, match="at least 3 dims"):
                apply_rot90(bad, (0, 1, 2), (True, True, False))


class TestDrawApplyRot90:
    """One draw applied across separately-held tensors (per-level lists, mixed layouts)."""

    def test_same_draw_across_levels_matches_stacked(self):
        # Looping apply_rot90 over per-level tensors with one draw equals rotating the
        # stacked L Z Y X tensor in a single rot90isocube call (same rng seed, same draw).
        levels = [torch.randn(5, 5, 5) for _ in range(4)]
        perm, flips = draw_rot90(np.random.default_rng(3))
        looped = torch.stack([apply_rot90(lvl, perm, flips) for lvl in levels])
        (stacked,) = rot90isocube(np.random.default_rng(3), torch.stack(levels))
        assert torch.equal(looped, stacked)

    def test_mixed_layouts_stay_aligned(self):
        # An L Z Y X C image and an L Z Y X label share one draw via per-tensor spatial_dims.
        label = torch.arange(3 * 3 * 3).reshape(1, 3, 3, 3)
        img = label.unsqueeze(-1).float()  # L Z Y X C
        perm, flips = draw_rot90(np.random.default_rng(5))
        out_img = apply_rot90(img, perm, flips, spatial_dims=(-4, -3, -2))
        out_label = apply_rot90(label, perm, flips)
        assert torch.equal(out_img.squeeze(-1).long(), out_label)

    def test_recorded_draw_replays_identically(self):
        t = torch.arange(4 * 4 * 4).reshape(4, 4, 4)
        perm, flips = draw_rot90(np.random.default_rng(9))
        assert torch.equal(apply_rot90(t, perm, flips), apply_rot90(t, perm, flips))


class TestRot90IsoCube:
    """Random axis-aligned rotations/flips (48 total)."""

    def test_shape_preserved(self):
        rng = np.random.default_rng(0)
        img = torch.arange(2 * 8 * 8 * 8, dtype=torch.float32).reshape(2, 8, 8, 8)
        for _ in range(20):
            (out,) = rot90isocube(rng, img)
            assert out.shape == img.shape

    def test_preserves_value_multiset(self):
        """A signed permutation is a bijection over a cubic patch, so the sorted voxel values
        are invariant."""
        img = torch.arange(8 * 8 * 8, dtype=torch.float32).reshape(1, 8, 8, 8)
        for i in range(10):
            (out,) = rot90isocube(np.random.default_rng(i), img)
            assert out.shape == img.shape
            assert torch.allclose(
                torch.sort(out.flatten()).values, torch.sort(img.flatten()).values
            )

    def test_produces_variation(self):
        """Over many draws of a fixed patch, more than one distinct orientation appears."""
        rng = np.random.default_rng(0)
        img = torch.arange(8 * 8 * 8, dtype=torch.float32).reshape(1, 8, 8, 8)
        seen = {rot90isocube(rng, img)[0].numpy().tobytes() for _ in range(30)}
        assert len(seen) > 1

    def test_image_and_label_share_transform(self):
        """Image and label get the same transform. Distinct value at every voxel so the
        orientation is recoverable, and img/label stay aligned."""
        data = torch.arange(8 * 8 * 8).reshape(1, 8, 8, 8)
        for i in range(10):
            out_img, out_lbl = rot90isocube(
                np.random.default_rng(i), data.float(), data.clone()
            )
            assert torch.equal(out_img, out_lbl.float())

    def test_deterministic_given_rng(self):
        img = torch.arange(8 * 8 * 8, dtype=torch.float32).reshape(1, 8, 8, 8)
        (a,) = rot90isocube(np.random.default_rng(7), img)
        (b,) = rot90isocube(np.random.default_rng(7), img)
        assert torch.equal(a, b)

    def test_input_not_mutated(self):
        img = torch.arange(8 * 8 * 8, dtype=torch.float32).reshape(1, 8, 8, 8)
        original = img.clone()
        rot90isocube(np.random.default_rng(0), img)
        assert torch.equal(img, original)

    def test_isotropic_pixel_size_ok(self):
        rng = np.random.default_rng(0)
        img = torch.zeros(1, 8, 8, 8)
        for ps in ([1.0, 1.0, 1.0], [[1, 1, 1], [4, 4, 4]], torch.ones(2, 3) * 5.0):
            rot90isocube(rng, img, pixel_size=ps)

    def test_anisotropic_pixel_size_asserts(self):
        """Anisotropic output resolution → error at apply time."""
        rng = np.random.default_rng(0)
        img = torch.zeros(1, 8, 8, 8)
        for bad in ([1, 1, 2], [[1, 1, 1], [2, 4, 4]]):
            with pytest.raises(AssertionError, match="isotropic output resolution"):
                rot90isocube(rng, img, pixel_size=bad)

    def test_non_cubic_patch_asserts(self):
        rng = np.random.default_rng(0)
        with pytest.raises(AssertionError, match="cubic patch"):
            rot90isocube(rng, torch.zeros(1, 8, 8, 4))

    def test_bad_pixel_size_shape_asserts(self):
        # A scalar or transposed pixel_size would pass the isotropy check vacuously.
        rng = np.random.default_rng(0)
        img = torch.zeros(1, 8, 8, 8)
        for bad in (2.0, np.array([[1.0], [1.0], [2.0]])):
            with pytest.raises(AssertionError, match="3 spatial entries"):
                rot90isocube(rng, img, pixel_size=bad)


class TestIntensityJitter:
    def test_within_expected_scale_and_shift_bounds(self):
        rng = np.random.default_rng(0)
        img = torch.ones(4, 4, 4)
        out = intensity_jitter(rng, img)
        assert out.min() >= 1 * 0.9 - 0.1
        assert out.max() <= 1 * 1.1 + 0.1

    def test_is_affine_transform_of_input(self):
        rng = np.random.default_rng(0)
        img = torch.tensor([0.0, 1.0, 2.0])
        out = intensity_jitter(rng, img)
        scale = out[1] - out[0]
        assert torch.allclose(out[2] - out[1], scale, atol=1e-6)

    def test_params_respected(self):
        rng = np.random.default_rng(0)
        img = torch.ones(4, 4, 4)
        out = intensity_jitter(rng, img, scale=(2.0, 2.0), shift=(0.0, 0.0))
        assert torch.allclose(out, 2 * img)

    def test_input_not_mutated(self):
        img = torch.ones(4, 4, 4)
        intensity_jitter(np.random.default_rng(0), img)
        assert torch.equal(img, torch.ones(4, 4, 4))


class TestPercentileNormalize:
    """Deterministic percentile scaling to ~[0, 1] (no RNG)."""

    def test_full_range_maps_to_unit_interval(self):
        img = torch.linspace(-5.0, 37.0, 4 * 4 * 4).reshape(4, 4, 4)
        out = percentile_normalize(img, lower=0.0, upper=100.0)
        assert torch.allclose(out.min(), torch.tensor(0.0), atol=1e-6)
        assert torch.allclose(out.max(), torch.tensor(1.0), atol=1e-6)

    def test_matches_numpy_percentile_oracle(self):
        # torch.quantile's default linear interpolation matches np.percentile's.
        img = torch.from_numpy(np.random.default_rng(0).normal(100.0, 20.0, (6, 6, 6))).float()
        x = img.numpy()
        lo, hi = np.percentile(x, 1.0), np.percentile(x, 99.0)
        expected = (x - lo) / (hi - lo)
        out = percentile_normalize(img)
        assert np.allclose(out.numpy(), np.clip(expected, 0.0, 1.0), atol=1e-5)
        out_unclamped = percentile_normalize(img, clamp=False)
        assert np.allclose(out_unclamped.numpy(), expected, atol=1e-5)

    def test_clamp_default_bounds_output(self):
        # With 1/99 percentiles the tails fall outside the range and get clipped by default.
        img = torch.from_numpy(np.random.default_rng(1).normal(0.0, 1.0, (8, 8, 8))).float()
        out = percentile_normalize(img)
        assert out.min() == 0.0
        assert out.max() == 1.0

    def test_clamp_false_leaves_tails_outside_unit_interval(self):
        img = torch.from_numpy(np.random.default_rng(1).normal(0.0, 1.0, (8, 8, 8))).float()
        out = percentile_normalize(img, clamp=False)
        assert out.min() < 0.0
        assert out.max() > 1.0

    def test_integer_input_promoted_to_float(self):
        img = torch.arange(4 * 4 * 4, dtype=torch.uint8).reshape(4, 4, 4)
        out = percentile_normalize(img)
        assert out.is_floating_point()
        assert torch.isfinite(out).all()

    def test_constant_image_is_finite(self):
        out = percentile_normalize(torch.full((4, 4, 4), 7.0))
        assert torch.equal(out, torch.zeros(4, 4, 4))

    def test_input_not_mutated(self):
        img = torch.linspace(0.0, 9.0, 4 * 4 * 4).reshape(4, 4, 4)
        original = img.clone()
        percentile_normalize(img)
        assert torch.equal(img, original)

    def test_bad_percentile_order_asserts(self):
        img = torch.ones(4, 4, 4)
        for lower, upper in ((99.0, 1.0), (50.0, 50.0), (-1.0, 99.0), (1.0, 101.0)):
            with pytest.raises(AssertionError, match="lower < upper"):
                percentile_normalize(img, lower=lower, upper=upper)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_half_dtype_input_does_not_crash(self, dtype):
        # torch.quantile rejects half dtypes even though img.is_floating_point() is True for them.
        img = torch.arange(4 * 4 * 4, dtype=torch.float32).reshape(4, 4, 4).to(dtype)
        out = percentile_normalize(img)
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()

    def test_oversized_tensor_falls_back_to_subsample(self):
        # torch.quantile refuses inputs above _QUANTILE_MAX_ELEMENTS; a stacked multi-scale
        # sample can easily exceed it even though a single level fits.
        img = torch.rand(_QUANTILE_MAX_ELEMENTS + 1)
        out = percentile_normalize(img)
        assert torch.isfinite(out).all()

    def test_oversized_tensor_subsample_estimate_is_deterministic(self):
        img = torch.rand(_QUANTILE_MAX_ELEMENTS + 1)
        first = percentile_normalize(img)
        second = percentile_normalize(img)
        assert torch.equal(first, second)


class TestVolumeDatasetAugmentFn:
    """The augment_fn arg: called per sample as augment_fn(sample) -> sample; rng closed over."""

    def _cfg(self, zarr2_volume, resolutions=[[1, 1, 1]]):
        return MiaoConfig(
            volumes=[{
                "name": "v",
                "path": str(zarr2_volume),
                "image_key": "raw",
                "label_key": "labels/seg",
            }],
            resolutions=resolutions,
            output_axes="lzyx",
            patch_size=[8, 8, 8],
            sampling="sequential",  # sample 0 is deterministic
        )

    def test_identity_fn_passthrough(self, zarr2_volume: Path):
        plain = VolumeDataset(self._cfg(zarr2_volume))[0]
        ident = VolumeDataset(self._cfg(zarr2_volume), augment_fn=lambda s: s)[0]
        assert torch.equal(plain["img"], ident["img"])
        assert torch.equal(plain["label"], ident["label"])

    def test_non_callable_asserts_at_construction(self, zarr2_volume: Path):
        with pytest.raises(AssertionError, match="callable"):
            VolumeDataset(self._cfg(zarr2_volume), augment_fn=True)

    def test_rot90_and_jitter_closure(self, zarr2_volume: Path):
        """The intended usage: compose the pure fns in a closure with an rng closed over."""
        rng = np.random.default_rng(0)

        def augment(sample):
            img, label = rot90isocube(
                rng, sample["img"], sample["label"], pixel_size=sample["pixel_size"]
            )
            return {**sample, "img": intensity_jitter(rng, img), "label": label}

        plain = VolumeDataset(self._cfg(zarr2_volume))[0]
        aug = VolumeDataset(self._cfg(zarr2_volume), augment_fn=augment)[0]

        assert aug["img"].shape == plain["img"].shape
        assert aug["label"].shape == plain["label"].shape
        # Rotation is a bijection, so the label's voxel multiset is invariant.
        assert torch.equal(
            torch.sort(aug["label"].flatten()).values,
            torch.sort(plain["label"].flatten()).values,
        )
        # Jitter changed the image values.
        assert not torch.allclose(
            torch.sort(aug["img"].flatten()).values,
            torch.sort(plain["img"].flatten()).values,
        )

    def test_anisotropic_resolution_asserts_via_pixel_size(self, zarr2_volume: Path):
        """Gary's anisotropy check fires at read time when the closure forwards pixel_size."""
        rng = np.random.default_rng(0)

        def augment(sample):
            img, label = rot90isocube(
                rng, sample["img"], sample["label"], pixel_size=sample["pixel_size"]
            )
            return {**sample, "img": img, "label": label}

        ds = VolumeDataset(self._cfg(zarr2_volume, resolutions=[[1, 1, 2]]), augment_fn=augment)
        with pytest.raises(AssertionError, match="isotropic output resolution"):
            ds[0]
