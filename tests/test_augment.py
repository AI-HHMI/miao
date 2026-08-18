"""Tests for the pure-function augmentation library."""

from pathlib import Path

import numpy as np
import pytest
import torch

from miao.augment import _SPATIAL_PERMS, apply_rot90, draw_rot90, intensity_jitter, rot90isocube
from miao.augment_std import em_default
from miao.config import MiaoConfig, load_config
from miao.dataset import VolumeDataset


def identity(sample):
    return sample


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


def _sequential_cfg(zarr2_volume, resolutions=None, augment_fn=None):
    return MiaoConfig(
        volumes=[{
            "name": "v",
            "path": str(zarr2_volume),
            "image_key": "raw",
            "label_key": "labels/seg",
        }],
        resolutions=resolutions or [[1, 1, 1]],
        output_axes="lzyx",
        patch_size=[8, 8, 8],
        sampling="sequential",  # sample 0 is deterministic
        augment_fn=augment_fn,
    )


class TestVolumeDatasetAugmentFn:
    """The augment_fn arg: called per sample as augment_fn(sample) -> sample; rng closed over."""

    def test_identity_fn_passthrough(self, zarr2_volume: Path):
        plain = VolumeDataset(_sequential_cfg(zarr2_volume))[0]
        ident = VolumeDataset(_sequential_cfg(zarr2_volume), augment_fn=identity)[0]
        assert torch.equal(plain["img"], ident["img"])
        assert torch.equal(plain["label"], ident["label"])

    def test_direct_augment_fn_runs_in_spawned_dataloader(self, zarr2_volume: Path):
        plain = VolumeDataset(_sequential_cfg(zarr2_volume))[0]
        ds = VolumeDataset(_sequential_cfg(zarr2_volume), augment_fn=HalveFactory((0.5, 0.5)))
        batch = next(iter(torch.utils.data.DataLoader(
            ds, batch_size=1, num_workers=1, multiprocessing_context="spawn"
        )))
        assert torch.allclose(batch["img"][0], 0.5 * plain["img"])

    def test_non_callable_asserts_at_construction(self, zarr2_volume: Path):
        with pytest.raises(AssertionError, match="callable"):
            VolumeDataset(_sequential_cfg(zarr2_volume), augment_fn=True)

    def test_rot90_and_jitter_closure(self, zarr2_volume: Path):
        """The intended usage: compose the pure fns in a closure with an rng closed over."""
        rng = np.random.default_rng(0)

        def augment(sample):
            img, label = rot90isocube(
                rng, sample["img"], sample["label"], pixel_size=sample["pixel_size"]
            )
            return {**sample, "img": intensity_jitter(rng, img), "label": label}

        plain = VolumeDataset(_sequential_cfg(zarr2_volume))[0]
        aug = VolumeDataset(_sequential_cfg(zarr2_volume), augment_fn=augment)[0]

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

        ds = VolumeDataset(_sequential_cfg(zarr2_volume, resolutions=[[1, 1, 2]]), augment_fn=augment)
        with pytest.raises(AssertionError, match="isotropic output resolution"):
            ds[0]


class HalveFactory:
    """Class-as-factory: the class itself is the factory, instances are picklable augment_fns."""

    def __init__(self, scale=(1.0, 1.0)):
        self.scale = scale

    def __call__(self, sample):
        img = intensity_jitter(np.random, sample["img"], scale=self.scale, shift=(0.0, 0.0))
        return {**sample, "img": img}


def _closure_factory():
    """A factory returning a nested closure — the documented anti-pattern."""

    def augment(sample):
        return sample

    return augment


class TestConfigAugmentFn:
    """config.augment_fn: {factory, kwargs} resolved at dataset construction (v1: class factory)."""


    def test_factory_resolved_and_kwargs_applied(self, zarr2_volume: Path):
        plain = VolumeDataset(_sequential_cfg(zarr2_volume))[0]
        spec = {"factory": "test_augment.HalveFactory", "kwargs": {"scale": [0.5, 0.5]}}
        aug = VolumeDataset(_sequential_cfg(zarr2_volume, augment_fn=spec))[0]
        assert torch.allclose(aug["img"], 0.5 * plain["img"])

    def test_load_yaml_factory_and_kwargs(self, tmp_path: Path, zarr2_volume: Path):
        example_path = Path(__file__).parents[1] / "examples" / "config_augment_fn.yaml"
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            example_path.read_text()
            .replace("/data/sample.zarr", str(zarr2_volume))
            .replace("patch_size: [64, 64, 64]", "patch_size: [8, 8, 8]")
        )
        cfg = load_config(config_path)
        assert cfg.augment_fn.factory == "miao.augment_std.em_default"
        assert cfg.augment_fn.kwargs == {"scale": [0.8, 1.2], "shift": [-0.1, 0.1]}
        assert VolumeDataset(cfg)[0]["img"].shape == (1, 8, 8, 8)

    def test_em_default_factory_and_kwargs(self, zarr2_volume: Path):
        plain = VolumeDataset(_sequential_cfg(zarr2_volume))[0]
        spec = {
            "factory": "miao.augment_std.em_default",
            "kwargs": {"scale": [0.5, 0.5], "shift": [0.0, 0.0], "rotate": False},
        }
        aug = VolumeDataset(_sequential_cfg(zarr2_volume, augment_fn=spec))[0]
        assert torch.allclose(aug["img"], 0.5 * plain["img"])
        assert torch.equal(aug["label"], plain["label"])

    def test_direct_callable_path(self, zarr2_volume: Path):
        cfg = _sequential_cfg(zarr2_volume, augment_fn="test_augment.identity")
        assert cfg.augment_fn == "test_augment.identity"
        plain = VolumeDataset(_sequential_cfg(zarr2_volume))[0]
        aug = VolumeDataset(cfg)[0]
        assert torch.equal(aug["img"], plain["img"])

    def test_load_yaml_direct_callable_path(self, tmp_path: Path, zarr2_volume: Path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            f"""
augment_fn: test_augment.identity
volumes:
  - name: v
    path: {zarr2_volume}
    image_key: raw
resolutions: [[1, 1, 1]]
output_axes: lzyx
patch_size: [8, 8, 8]
sampling: sequential
"""
        )
        cfg = load_config(config_path)
        assert cfg.augment_fn == "test_augment.identity"
        assert VolumeDataset(cfg)[0]["img"].shape == (1, 8, 8, 8)

    def test_dataset_pickles_and_still_augments(self, zarr2_volume: Path):
        import pickle

        spec = {"factory": "test_augment.HalveFactory", "kwargs": {"scale": [0.5, 0.5]}}
        ds = VolumeDataset(_sequential_cfg(zarr2_volume, augment_fn=spec))
        clone = pickle.loads(pickle.dumps(ds))  # what a spawned worker receives
        assert torch.allclose(clone[0]["img"], ds[0]["img"])

    def test_both_sources_assert(self, zarr2_volume: Path):
        cfg = _sequential_cfg(zarr2_volume, augment_fn="test_augment.identity")
        with pytest.raises(AssertionError, match="not both"):
            VolumeDataset(cfg, augment_fn=identity)

    def test_bad_kwarg_fails_at_construction(self, zarr2_volume: Path):
        spec = {"factory": "test_augment.HalveFactory", "kwargs": {"scal": [0.5, 0.5]}}
        with pytest.raises(TypeError):
            VolumeDataset(_sequential_cfg(zarr2_volume, augment_fn=spec))

    def test_bad_path_fails_at_construction(self, zarr2_volume: Path):
        for bad in ("test_augment.NoSuchFactory", "test_augment.", ".rel.path", "nodots"):
            with pytest.raises(AssertionError, match="augment_fn"):
                VolumeDataset(_sequential_cfg(zarr2_volume, augment_fn=bad))

    def test_bare_string_naming_a_factory_asserts_at_first_sample(self, zarr2_volume: Path):
        # em_default is a factory; as a bare string it resolves as the augment_fn itself and
        # returns a partial instead of a sample — the __getitem__ guard catches it.
        ds = VolumeDataset(
            _sequential_cfg(zarr2_volume, augment_fn="miao.augment_std.em_default")
        )
        with pytest.raises(AssertionError, match="factories need"):
            ds[0]

    def test_em_default_rotates_labels_with_image(self, tmp_path: Path):
        """Distinct value at every voxel so any misalignment between img and label is visible."""
        import json

        import zarr
        from zarr.storage import LocalStore

        zpath = tmp_path / "grad.zarr"
        root = zarr.open_group(LocalStore(str(zpath)), mode="a", zarr_format=2)
        axes = [{"name": n, "type": "space", "unit": "micrometer"} for n in "zyx"]
        data = np.arange(8 * 8 * 8, dtype=np.float32).reshape(8, 8, 8)
        for key, dtype, arr in [("raw", "float32", data), ("seg", "uint32", data.astype("uint32"))]:
            grp = root.create_group(key)
            a = grp.create_array("0", shape=(8, 8, 8), chunks=(8, 8, 8), dtype=dtype, overwrite=True)
            a[:] = arr
            (zpath / key / ".zattrs").write_text(json.dumps({"multiscales": [{
                "version": "0.4", "axes": axes,
                "datasets": [{"path": "0", "coordinateTransformations": [
                    {"type": "scale", "scale": [1.0, 1.0, 1.0]}]}],
            }]}))

        cfg = MiaoConfig(
            volumes=[{"name": "v", "path": str(zpath), "image_key": "raw", "label_key": "seg",
                      "normalize": False}],
            resolutions=[[1, 1, 1]],
            output_axes="lzyx",
            patch_size=[8, 8, 8],
            sampling="sequential",
            augment_fn={"factory": "miao.augment_std.em_default",
                        "kwargs": {"scale": [1.0, 1.0], "shift": [0.0, 0.0]}},
        )
        ds = VolumeDataset(cfg)
        plain_label = torch.from_numpy(data.astype("int64")).reshape(1, 8, 8, 8)
        rotated_any = False
        for seed in range(5):
            np.random.seed(seed)
            s = ds[0]
            # scale 1 / shift 0 makes jitter a no-op, so img must equal the rotated label exactly
            assert torch.equal(s["img"][0], s["label"][0].float()), f"misaligned at seed {seed}"
            rotated_any = rotated_any or not torch.equal(s["label"], plain_label)
        assert rotated_any, "no seed produced a non-identity rotation"

    def test_em_default_partial_pickles(self):
        import pickle

        pickle.loads(pickle.dumps(em_default(scale=(0.8, 1.2))))

    def test_closure_factory_return_does_not_pickle(self):
        import pickle

        with pytest.raises((AttributeError, pickle.PicklingError)):
            pickle.dumps(_closure_factory())
