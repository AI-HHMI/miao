"""Tests for the pure-function augmentation library."""

from pathlib import Path

import numpy as np
import pytest
import torch

from miao.augment import _SPATIAL_PERMS, apply_rot90, intensity_jitter, rot90isocube
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
