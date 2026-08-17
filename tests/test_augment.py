"""Tests for the pure-function augmentation library."""

from pathlib import Path

import numpy as np
import pytest
import torch

from miao.augment import (
    _SPATIAL_PERMS,
    additive_noise,
    apply_rot90,
    draw_rot90,
    drop_sections,
    intensity_jitter,
    rot90inplane,
    rot90isocube,
    shift_sections,
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


class TestRot90InPlane:
    """The 16-transform subgroup that never exchanges the anisotropic axis."""

    def test_fixed_axis_is_never_exchanged(self):
        """The property the whole function exists for.

        The volume varies along the fixed axis alone, so if that axis were ever permuted into a
        free slot the variation would move with it. A flip along it is allowed and leaves the
        variation where it is.
        """
        img = torch.arange(6, dtype=torch.float32).reshape(6, 1, 1).expand(6, 6, 6).contiguous()
        rng = np.random.default_rng(0)
        for _ in range(64):
            (out,) = rot90inplane(rng, img, fixed_axis=0)
            varies_within_section = out.reshape(6, -1).std(dim=1)
            assert torch.allclose(varies_within_section, torch.zeros(6), atol=1e-6), (
                "values varied inside a section, so the fixed axis was permuted away"
            )

    def test_offers_exactly_sixteen_transforms(self):
        """8 per-axis flips times the single swap of the two free axes."""
        img = torch.arange(4 * 4 * 4, dtype=torch.float32).reshape(4, 4, 4)
        rng = np.random.default_rng(0)
        seen = {rot90inplane(rng, img, fixed_axis=0)[0].numpy().tobytes() for _ in range(2000)}
        assert len(seen) == 16

    def test_is_a_subgroup_of_the_full_forty_eight(self):
        img = torch.arange(4 * 4 * 4, dtype=torch.float32).reshape(4, 4, 4)
        rng = np.random.default_rng(1)
        inplane = {rot90inplane(rng, img, fixed_axis=0)[0].numpy().tobytes() for _ in range(2000)}
        every = {
            apply_rot90(img, perm, flips).numpy().tobytes()
            for perm in _SPATIAL_PERMS
            for flips in [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
        }
        assert inplane <= every

    @pytest.mark.parametrize("fixed_axis", [0, 1, 2])
    def test_any_slot_can_be_the_fixed_one(self, fixed_axis):
        shape = [1, 1, 1]
        shape[fixed_axis] = 6
        img = torch.arange(6, dtype=torch.float32).reshape(shape).expand(6, 6, 6).contiguous()
        rng = np.random.default_rng(0)
        for _ in range(32):
            (out,) = rot90inplane(rng, img, fixed_axis=fixed_axis)
            spread = out.movedim(fixed_axis, 0).reshape(6, -1).std(dim=1)
            assert torch.allclose(spread, torch.zeros(6), atol=1e-6)

    def test_anisotropy_on_the_fixed_axis_is_allowed(self):
        """The weaker condition this subgroup needs, and the reason it exists."""
        img = torch.zeros(4, 4, 4)
        rot90inplane(np.random.default_rng(0), img, fixed_axis=2, pixel_size=[9.0, 9.0, 20.0])

    def test_anisotropy_between_the_free_axes_is_rejected(self):
        img = torch.zeros(4, 4, 4)
        with pytest.raises(AssertionError, match="share a voxel size"):
            rot90inplane(np.random.default_rng(0), img, fixed_axis=2, pixel_size=[9.0, 11.0, 20.0])

    def test_image_and_label_share_the_transform(self):
        data = torch.arange(4 * 4 * 4).reshape(4, 4, 4)
        for i in range(10):
            img, lbl = rot90inplane(np.random.default_rng(i), data.float(), data.clone())
            assert torch.equal(img, lbl.float())

    def test_deterministic_given_rng(self):
        img = torch.arange(4 * 4 * 4, dtype=torch.float32).reshape(4, 4, 4)
        (a,) = rot90inplane(np.random.default_rng(3), img)
        (b,) = rot90inplane(np.random.default_rng(3), img)
        assert torch.equal(a, b)


class TestDropSections:
    """Whole blanked sections, image only."""

    def test_dropped_sections_are_entirely_blank(self):
        img = torch.ones(8, 8, 8)
        out = drop_sections(np.random.default_rng(0), img, prob=0.5)
        per_section = [out.movedim(d, 0).reshape(8, -1) for d in range(3)]
        assert any(
            all(float(s[i].max()) in (0.0, 1.0) and float(s[i].min()) == float(s[i].max())
                for i in range(8))
            for s in per_section
        ), "a drop must blank a whole section, not part of one"

    def test_certain_probability_blanks_everything(self):
        out = drop_sections(np.random.default_rng(0), torch.ones(4, 4, 4), prob=1.0)
        assert torch.equal(out, torch.zeros(4, 4, 4))

    def test_zero_probability_is_the_identity(self):
        img = torch.rand(4, 4, 4)
        assert torch.equal(drop_sections(np.random.default_rng(0), img, prob=0.0), img)

    def test_does_not_mutate_its_input(self):
        img = torch.ones(8, 8, 8)
        before = img.clone()
        drop_sections(np.random.default_rng(0), img, prob=0.9)
        assert torch.equal(img, before)

    def test_deterministic_given_rng(self):
        img = torch.rand(6, 6, 6)
        a = drop_sections(np.random.default_rng(5), img, prob=0.4)
        b = drop_sections(np.random.default_rng(5), img, prob=0.4)
        assert torch.equal(a, b)


class TestShiftSections:
    """Misaligned sections, image and labels together."""

    def test_image_and_label_receive_the_same_shift(self):
        """The divergence from the reference recipes: alignment is preserved, not broken."""
        data = torch.arange(8 * 8 * 8).reshape(8, 8, 8)
        img, lbl = shift_sections(
            np.random.default_rng(0), data.float(), data.clone(), prob=1.0, magnitude=3
        )
        assert torch.equal(img, lbl.float())

    def test_shape_and_values_are_preserved(self):
        """A roll is a bijection, so no voxel is created or destroyed."""
        img = torch.arange(8 * 8 * 8, dtype=torch.float32).reshape(8, 8, 8)
        (out,) = shift_sections(np.random.default_rng(0), img, prob=1.0, magnitude=3)
        assert out.shape == img.shape
        assert torch.equal(torch.sort(out.flatten()).values, torch.sort(img.flatten()).values)

    def test_zero_magnitude_is_the_identity(self):
        img = torch.rand(4, 4, 4)
        (out,) = shift_sections(np.random.default_rng(0), img, prob=1.0, magnitude=0)
        assert torch.equal(out, img)

    def test_does_not_mutate_its_input(self):
        img = torch.arange(8 * 8 * 8, dtype=torch.float32).reshape(8, 8, 8)
        before = img.clone()
        shift_sections(np.random.default_rng(0), img, prob=1.0, magnitude=3)
        assert torch.equal(img, before)

    def test_deterministic_given_rng(self):
        img = torch.rand(6, 6, 6)
        (a,) = shift_sections(np.random.default_rng(2), img, prob=0.5, magnitude=2)
        (b,) = shift_sections(np.random.default_rng(2), img, prob=0.5, magnitude=2)
        assert torch.equal(a, b)


class TestAdditiveNoise:
    """Zero-mean Gaussian noise at a randomly drawn deviation."""

    def test_shape_and_dtype_survive(self):
        img = torch.rand(4, 4, 4)
        out = additive_noise(np.random.default_rng(0), img, scale=0.5)
        assert out.shape == img.shape and out.dtype == img.dtype

    def test_zero_scale_is_the_identity(self):
        img = torch.rand(4, 4, 4)
        assert torch.equal(additive_noise(np.random.default_rng(0), img, scale=0.0), img)

    def test_noise_is_zero_mean_and_bounded_by_the_scale(self):
        img = torch.zeros(64, 64, 64)
        out = additive_noise(np.random.default_rng(0), img, scale=0.5)
        assert abs(float(out.mean())) < 0.01
        assert float(out.std()) <= 0.5 + 1e-3, "the drawn deviation must not exceed `scale`"

    def test_does_not_mutate_its_input(self):
        img = torch.rand(8, 8, 8)
        before = img.clone()
        additive_noise(np.random.default_rng(0), img, scale=0.5)
        assert torch.equal(img, before)

    def test_deterministic_given_rng(self):
        img = torch.rand(6, 6, 6)
        a = additive_noise(np.random.default_rng(4), img, scale=0.3)
        b = additive_noise(np.random.default_rng(4), img, scale=0.3)
        assert torch.equal(a, b)


class TestSerialSectionRecipeIntegration:
    """The serial-section EM recipe, composed through the dataset the way a trainer would."""

    def _aniso_cfg(self, path):
        return MiaoConfig(
            volumes=[{"name": "v", "path": str(path), "image_key": "raw"}],
            resolutions=[[5, 1, 1]],
            output_axes="lzyx",
            patch_size=[8, 8, 8],
            sampling="sequential",
        )

    def _iso_cfg(self, path):
        return MiaoConfig(
            volumes=[{
                "name": "v", "path": str(path),
                "image_key": "raw", "label_key": "labels/seg",
            }],
            resolutions=[[1, 1, 1]],
            output_axes="lzyx",
            patch_size=[8, 8, 8],
            sampling="sequential",
        )

    def test_inplane_serves_the_data_isocube_refuses(self, zarr2_volume_anisotropic: Path):
        """Why this subgroup exists, stated as a contrast on one volume.

        At [5, 1, 1] the sectioning axis is 5x coarser than the in-plane ones. Exchanging it with
        y or x would relabel a 5-unit neighbour relationship as a 1-unit one, so `rot90isocube`
        refuses; `rot90inplane` keeps the 16 transforms that never touch it and proceeds.
        """
        def with_isocube(sample):
            (img,) = rot90isocube(
                np.random.default_rng(0), sample["img"], pixel_size=sample["pixel_size"]
            )
            return {**sample, "img": img}

        def with_inplane(sample):
            (img,) = rot90inplane(
                np.random.default_rng(0), sample["img"],
                fixed_axis=0, pixel_size=sample["pixel_size"],
            )
            return {**sample, "img": img}

        cfg = self._aniso_cfg(zarr2_volume_anisotropic)
        with pytest.raises(AssertionError, match="isotropic output resolution"):
            VolumeDataset(cfg, augment_fn=with_isocube)[0]

        out = VolumeDataset(cfg, augment_fn=with_inplane)[0]
        assert out["img"].shape == VolumeDataset(cfg)[0]["img"].shape

    def test_the_whole_recipe_composes_and_keeps_labels_aligned(self, zarr2_volume: Path):
        """All five operations in one closure, which is how a trainer would use them.

        Ordering is the part worth pinning: the geometric operations come first and receive both
        tensors, and the photometric ones come last and receive only the image. `drop_sections`
        sits with the photometric group on purpose -- an object still passes through a lost
        section, so its label must survive the blanking.
        """
        rng = np.random.default_rng(0)

        def augment(sample):
            img, label = rot90inplane(
                rng, sample["img"], sample["label"],
                fixed_axis=0, pixel_size=sample["pixel_size"],
            )
            img, label = shift_sections(rng, img, label, prob=0.3, magnitude=2)
            img = drop_sections(rng, img, prob=0.2)
            img = intensity_jitter(rng, img)
            img = additive_noise(rng, img, scale=0.1)
            return {**sample, "img": img, "label": label}

        cfg = self._iso_cfg(zarr2_volume)
        plain, aug = VolumeDataset(cfg)[0], VolumeDataset(cfg, augment_fn=augment)[0]

        assert aug["img"].shape == plain["img"].shape
        assert aug["label"].shape == plain["label"].shape
        # The label went through the geometric operations only, so it is a rearrangement of the
        # original: same voxel multiset. The image is not, having been jittered and noised.
        assert torch.equal(
            torch.sort(aug["label"].flatten()).values,
            torch.sort(plain["label"].flatten()).values,
        )
        assert not torch.allclose(
            torch.sort(aug["img"].flatten()).values,
            torch.sort(plain["img"].flatten()).values,
        )
