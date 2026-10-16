"""Tests for `sampling: foreground` — oversampling patches that contain sparse foreground."""

from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from conftest import _create_ome_ngff_zarr2
from miao.config import MiaoConfig
from miao.dataset import VolumeDataset

RES_1 = [[1, 1, 1]]
RES_3 = [[1, 1, 1], [2, 2, 2], [4, 4, 4]]

# One 4^3 object in a 64^3 volume, aligned to the level-2 grid so it is a single coarse voxel
# along each axis there.
BLOB = (slice(40, 44), slice(24, 28), slice(36, 40))


def _blob_fn(fg=1, bg=0, blob=BLOB):
    """Label contents: `fg` inside the blob, `bg` elsewhere, max-pooled at coarser levels."""

    def fn(shape, level):
        out = np.full(shape, bg, dtype=np.float64)
        f = 2**level
        out[tuple(slice(s.start // f, -(-s.stop // f)) for s in blob)] = fg
        return out

    return fn


def _build(root: Path, *, label_fn=None, label_dtype="uint16", half_voxel_shift=True, axes=None,
           label_shape=(64, 64, 64)) -> Path:
    _create_ome_ngff_zarr2(root, "raw", (64, 64, 64), num_scales=3,
                           half_voxel_shift=half_voxel_shift)
    _create_ome_ngff_zarr2(
        root, "labels/seg", label_shape, num_scales=3, dtype=label_dtype, axes=axes,
        half_voxel_shift=half_voxel_shift, data_fn=label_fn or _blob_fn(),
    )
    return root


def _config(root: Path, resolutions=RES_3, foreground=None, volume=None, **kwargs) -> MiaoConfig:
    vol = {"name": "v", "path": str(root), "image_key": "raw", "label_key": "labels/seg"}
    vol.update(volume or {})
    kwargs.setdefault("samples_per_epoch", 40)
    kwargs.setdefault("patch_size", [8, 8, 8])
    if "resolution_sampling" not in kwargs:
        kwargs["resolutions"] = resolutions
    return MiaoConfig(
        volumes=[vol], output_axes="lzyx", sampling="foreground",
        foreground={"fraction": 1.0} if foreground is None else foreground, **kwargs,
    )


def _blob_in_crop(sample, scale: int) -> bool:
    return bool((sample["label"][scale] != 0).any())


class TestForegroundConfig:
    def test_defaults_filled_in(self, tmp_path: Path):
        cfg = _config(_build(tmp_path), foreground={})
        assert cfg.foreground.fraction == 0.5
        assert cfg.foreground.threshold is None
        assert cfg.foreground.level == -1

    def test_omitted_block_gets_defaults(self, tmp_path: Path):
        cfg = MiaoConfig(
            volumes=[{"name": "v", "path": str(tmp_path), "image_key": "raw",
                      "label_key": "labels/seg"}],
            resolutions=RES_1, output_axes="lzyx", patch_size=[8, 8, 8], sampling="foreground",
        )
        assert cfg.foreground is not None and cfg.foreground.fraction == 0.5

    def test_foreground_block_outside_foreground_mode_raises(self, tmp_path: Path):
        with pytest.raises(ValidationError, match="only applies to sampling='foreground'"):
            MiaoConfig(
                volumes=[{"name": "v", "path": str(tmp_path), "image_key": "raw"}],
                resolutions=RES_1, output_axes="lzyx", patch_size=[8, 8, 8],
                foreground={"fraction": 0.5},
            )

    def test_volume_without_labels_raises(self, tmp_path: Path):
        with pytest.raises(ValidationError, match="no label_key"):
            MiaoConfig(
                volumes=[{"name": "v", "path": str(tmp_path), "image_key": "raw"}],
                resolutions=RES_1, output_axes="lzyx", patch_size=[8, 8, 8],
                sampling="foreground",
            )

    @pytest.mark.parametrize("fraction", [-0.1, 1.5])
    def test_fraction_out_of_range_raises(self, tmp_path: Path, fraction):
        with pytest.raises(ValidationError, match="fraction"):
            _config(tmp_path, foreground={"fraction": fraction})

    def test_unknown_key_rejected(self, tmp_path: Path):
        with pytest.raises(ValidationError):
            _config(tmp_path, foreground={"fraction": 0.5, "thresh": 1})

    def test_yaml_round_trip(self, tmp_path: Path):
        cfg = _config(tmp_path, foreground={"fraction": 0.3, "threshold": 0.5, "level": 1})
        again = MiaoConfig(**__import__("yaml").safe_load(cfg.to_yaml()))
        assert again.foreground == cfg.foreground


class TestForegroundSampling:
    def test_every_scale_contains_the_object(self, tmp_path: Path):
        """fraction=1: the object is inside the crop at every scale, for every draw."""
        np.random.seed(0)
        ds = VolumeDataset(_config(_build(tmp_path)))
        for i in range(len(ds)):
            sample = ds[i]
            assert sample["meta"]["foreground"] is True
            for s in range(3):
                assert _blob_in_crop(sample, s), (i, s, sample["meta"]["coordinate"])

    def test_whole_coarse_voxel_inside_finest_crop(self, tmp_path: Path):
        """The 4^3 object is one level-2 voxel, and an 8^3 finest crop can hold all of it."""
        np.random.seed(1)
        ds = VolumeDataset(_config(_build(tmp_path)))
        for i in range(len(ds)):
            assert int((ds[i]["label"][0] != 0).sum()) == 4**3

    def test_object_position_is_jittered(self, tmp_path: Path):
        """The object lands at varying positions in the crop, not pinned to its centre."""
        np.random.seed(2)
        ds = VolumeDataset(_config(_build(tmp_path), samples_per_epoch=80))
        coords = {tuple(ds[i]["meta"]["coordinate"]) for i in range(len(ds))}
        # The 8^3 crop admits 5 positions of a 4^3 object per axis.
        assert len(coords) > 20
        arr = np.array(sorted(coords))
        assert np.all(arr.min(axis=0) < arr.max(axis=0))

    def test_fraction_zero_is_random(self, tmp_path: Path):
        np.random.seed(3)
        ds = VolumeDataset(_config(_build(tmp_path), foreground={"fraction": 0.0}))
        assert not any(ds[i]["meta"]["foreground"] for i in range(len(ds)))

    def test_fraction_mixes_draws(self, tmp_path: Path):
        np.random.seed(4)
        ds = VolumeDataset(_config(
            _build(tmp_path), resolutions=RES_1, foreground={"fraction": 0.5},
            samples_per_epoch=400,
        ))
        samples = [ds[i] for i in range(len(ds))]
        flags = np.array([x["meta"]["foreground"] for x in samples])
        hits = np.array([_blob_in_crop(x, 0) for x in samples])
        assert 0.4 < flags.mean() < 0.6
        # Foreground draws always hold the object; a random 8^3 crop of 64^3 rarely does.
        assert hits[flags].all()
        assert hits[~flags].mean() < 0.1

    def test_threshold_binarizes_float_labels(self, tmp_path: Path):
        np.random.seed(5)
        root = _build(tmp_path, label_fn=_blob_fn(fg=0.9, bg=0.2), label_dtype="float32")
        ds = VolumeDataset(_config(root, foreground={"fraction": 1.0, "threshold": 0.5}))
        assert len(ds._volumes[0].fg_voxels) > 0
        for i in range(len(ds)):
            sample = ds[i]
            # Float labels are returned as integers, so check the image region via coordinates.
            c = np.array(sample["meta"]["coordinate"])
            lo = np.array([s.start for s in BLOB])
            hi = np.array([s.stop for s in BLOB])
            # Crop [c - 4, c + 4) overlaps the object on every axis.
            assert np.all((c - 4 < hi) & (c + 4 > lo)), c

    def test_without_threshold_everything_nonzero_is_foreground(self, tmp_path: Path):
        root = _build(tmp_path, label_fn=_blob_fn(fg=0.9, bg=0.2), label_dtype="float32")
        ds = VolumeDataset(_config(root, foreground={"fraction": 1.0}))
        # Every coarse voxel is non-zero, but only those a finest crop can reach are kept: the
        # 32-voxel coarsest window keeps centres in [18, 49], so finest crops span [14, 52].
        fg = ds._volumes[0].fg_voxels
        assert len(fg) == 11**3
        assert fg.min() == 3 and fg.max() == 13

    def test_level_selects_scanned_level(self, tmp_path: Path):
        root = _build(tmp_path)
        coarse = VolumeDataset(_config(root))._volumes[0]
        fine = VolumeDataset(_config(root, foreground={"fraction": 1.0, "level": 0}))._volumes[0]
        assert coarse.fg_level == 2 and len(coarse.fg_voxels) == 1
        assert fine.fg_level == 0 and len(fine.fg_voxels) == 4**3
        np.testing.assert_allclose(fine.fg_rel, [1, 1, 1])
        np.testing.assert_allclose(coarse.fg_rel, [4, 4, 4])

    def test_level_out_of_range_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="out of range"):
            VolumeDataset(_config(_build(tmp_path), foreground={"fraction": 1.0, "level": 5}))

    def test_no_foreground_warns_and_falls_back(self, tmp_path: Path):
        np.random.seed(6)
        root = _build(tmp_path, label_fn=lambda shape, level: np.zeros(shape))
        with pytest.warns(UserWarning, match="no reachable foreground"):
            ds = VolumeDataset(_config(root))
        assert not any(ds[i]["meta"]["foreground"] for i in range(5))

    def test_foreground_outside_bounding_box_is_unreachable(self, tmp_path: Path):
        # bounding_box is in output_axes spatial order (zyx); the object sits at z 40..44.
        with pytest.warns(UserWarning, match="no reachable foreground"):
            VolumeDataset(_config(
                _build(tmp_path), resolutions=RES_1,
                volume={"bounding_box": [[0, 30], [0, 64], [0, 64]]},
            ))

    def test_coarse_voxel_wider_than_crop_warns(self, tmp_path: Path):
        np.random.seed(7)
        with pytest.warns(UserWarning, match="wider than the crop"):
            ds = VolumeDataset(_config(_build(tmp_path), resolutions=RES_1, patch_size=[2, 2, 2]))
        # The 2^3 crop, [c - 1, c], still overlaps the 4^3 coarse voxel.
        lo = np.array([s.start for s in BLOB])
        hi = np.array([s.stop for s in BLOB])
        for i in range(10):
            c = np.array(ds[i]["meta"]["coordinate"])
            assert np.all((c >= lo) & (c - 1 < hi)), c

    def test_pyramid_without_half_voxel_shift(self, tmp_path: Path):
        """Levels declared as sharing one origin are slightly mis-registered; the object is still
        inside every crop, if not always all of it."""
        np.random.seed(8)
        ds = VolumeDataset(_config(_build(tmp_path, half_voxel_shift=False)))
        for i in range(len(ds)):
            sample = ds[i]
            assert all(_blob_in_crop(sample, s) for s in range(3))

    def test_anisotropic(self, tmp_path: Path):
        """Voxels twice as thick in z, read at a resolution that resamples z."""
        np.random.seed(11)
        _create_ome_ngff_zarr2(tmp_path, "raw", (64, 64, 64), num_scales=3,
                               base_scale_factors=[2.0, 1.0, 1.0], half_voxel_shift=True)
        _create_ome_ngff_zarr2(tmp_path, "labels/seg", (64, 64, 64), num_scales=3,
                               dtype="uint16", base_scale_factors=[2.0, 1.0, 1.0],
                               half_voxel_shift=True, data_fn=_blob_fn())
        ds = VolumeDataset(_config(tmp_path, resolutions=[[2, 2, 2], [4, 4, 4]]))
        for i in range(len(ds)):
            sample = ds[i]
            assert all(_blob_in_crop(sample, s) for s in range(2)), sample["meta"]

    def test_sample_windows(self, tmp_path: Path):
        np.random.seed(9)
        ds = VolumeDataset(_config(_build(tmp_path), sample_windows=True))
        for i in range(len(ds)):
            sample = ds[i]
            assert all(_blob_in_crop(sample, s) for s in range(3))

    def test_resolution_sampling(self, tmp_path: Path):
        np.random.seed(10)
        ds = VolumeDataset(_config(
            _build(tmp_path),
            resolution_sampling={"ranges": [[[1], [3]]], "n_scales": 2},
        ))
        for i in range(len(ds)):
            sample = ds[i]
            assert all(_blob_in_crop(sample, s) for s in range(2)), sample["meta"]

    def test_label_channel_selection(self, tmp_path: Path):
        """With label_channels, only the selected channels count as foreground."""

        def two_channel(shape, level):
            out = np.zeros(shape)
            out[1] = _blob_fn()(shape[1:], level)
            return out

        axes = [{"name": "c", "type": "channel"}] + [
            {"name": n, "type": "space", "unit": "micrometer"} for n in "zyx"
        ]
        root = tmp_path
        _create_ome_ngff_zarr2(root, "raw", (64, 64, 64), num_scales=3, half_voxel_shift=True)
        # The channel axis must not be downsampled, so write each level by hand-shaped data.
        _create_ome_ngff_zarr2(
            root, "labels/seg", (2, 64, 64, 64), num_scales=1, axes=axes, dtype="uint8",
            data_fn=two_channel,
        )
        with_blob = VolumeDataset(_config(root, resolutions=RES_1, volume={"label_channels": [1]}))
        assert len(with_blob._volumes[0].fg_voxels) == 4**3
        with pytest.warns(UserWarning, match="no reachable foreground"):
            VolumeDataset(_config(root, resolutions=RES_1, volume={"label_channels": [0]}))

    def test_dataloader_collates_foreground_flag(self, tmp_path: Path):
        ds = VolumeDataset(_config(_build(tmp_path), foreground={"fraction": 0.5}))
        batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=4, num_workers=2)))
        assert batch["meta"]["foreground"].shape == (4,)

    def test_summary_reports_foreground(self, tmp_path: Path, capsys):
        VolumeDataset(_config(_build(tmp_path)))
        out = capsys.readouterr().out
        assert "foreground, 40 samples/epoch, fraction=1.0, label != 0" in out
        assert "foreground: 1 reachable voxels at label level 2" in out
