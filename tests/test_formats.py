"""N5 and Neuroglancer precomputed stores read like OME-NGFF zarr (miao.formats).

One pyramid -- an image and its labels -- is written as OME-NGFF zarr, as N5 in each of the three
metadata conventions found on the Janelia cluster, and as precomputed. Every copy must give miao
the same levels and the same patches as the zarr one.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tensorstore as ts
from pydantic import ValidationError

from conftest import _create_ome_ngff_zarr2

from miao.config import MiaoConfig, VolumeConfig
from miao.dataset import VolumeDataset
from miao.formats import read_metadata
from miao.store import open_store

SHAPE = (64, 64, 48)  # level 0, x y z
VOXEL = [8.0, 8.0, 8.0]  # nm
LEVELS = 3
XYZ_NM = [{"name": n, "type": "space", "unit": "nanometer"} for n in "xyz"]


def _pyramid(img0: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Image (uint8, block means) and label (uint64, strided picks) arrays per level, x y z."""
    blocks = [s // 4 for s in SHAPE]
    lbl0 = np.arange(np.prod(blocks), dtype=np.uint64).reshape(blocks) + np.uint64(2**40)
    lbl0 = lbl0.repeat(4, 0).repeat(4, 1).repeat(4, 2)
    imgs, lbls = [img0], [lbl0]
    for k in range(1, LEVELS):
        f = 2**k
        s = [n // f for n in SHAPE]
        imgs.append(img0.reshape(s[0], f, s[1], f, s[2], f).mean((1, 3, 5)).astype(np.uint8))
        lbls.append(lbl0[::f, ::f, ::f])
    return imgs, lbls


IMGS, LBLS = _pyramid(np.random.RandomState(0).randint(0, 256, SHAPE).astype(np.uint8))


def _zarr(root: Path, imgs=IMGS, base_translation=None) -> dict:
    for key, arrays, dtype in (("raw", imgs, "uint8"), ("labels/seg", LBLS, "uint64")):
        _create_ome_ngff_zarr2(
            root, key, SHAPE, num_scales=LEVELS, axes=XYZ_NM, dtype=dtype,
            base_scale_factors=VOXEL, half_voxel_shift=True, base_translation=base_translation,
            data_fn=lambda shape, level, a=arrays: a[level],
        )
    return {"path": str(root), "image_key": "raw", "label_key": "labels/seg"}


def _n5_array(path: Path, data: np.ndarray, attrs: dict) -> None:
    store = ts.open(
        {"driver": "n5", "kvstore": {"driver": "file", "path": str(path)},
         "metadata": {"dimensions": list(data.shape), "blockSize": [16, 16, 16],
                      "dataType": str(data.dtype), "compression": {"type": "gzip"}}},
        create=True, delete_existing=True,
    ).result()
    store[...] = data
    meta = json.loads((path / "attributes.json").read_text())
    (path / "attributes.json").write_text(json.dumps({**meta, **attrs}))


BASE = {"dimensions": VOXEL, "unit": "nm"}
# (attributes of level k, attributes of the multiscale group), as the cluster's N5s write them.
N5_CONVENTIONS = {
    # FAFB v14: each level's own `resolution`, `downsamplingFactors` from s1 on.
    "resolution": (
        lambda k: {"resolution": [v * 2**k for v in VOXEL],
                   **({"downsamplingFactors": [2**k] * 3} if k else {})},
        {},
    ),
    # render export of the male CNS: the base `pixelResolution` on the group and on every level.
    "pixelResolution": (
        lambda k: {"pixelResolution": BASE, "downsamplingFactors": [2**k] * 3},
        {"pixelResolution": BASE},
    ),
    # render export of D. yakuba: base and factors on the group only.
    "scales": (
        lambda k: {},
        {"pixelResolution": BASE, "scales": [[2**k] * 3 for k in range(LEVELS)]},
    ),
}


def _n5(root: Path, convention: str, first_level: int = 0) -> dict:
    level_attrs, group_attrs = N5_CONVENTIONS[convention]
    for key, arrays in (("raw", IMGS), ("seg", LBLS)):
        for k in range(first_level, LEVELS):
            _n5_array(root / key / f"s{k}", arrays[k], level_attrs(k))
        (root / key / "attributes.json").write_text(json.dumps({"n5": "2.0.0", **group_attrs}))
    return {"path": str(root), "image_key": "raw", "label_key": "seg", "format": "n5"}


def _precomputed(root: Path, offset=(0, 0, 0), channels: list[list[np.ndarray]] | None = None) -> dict:
    """`channels`: per channel, its image arrays per level (default: IMGS alone)."""
    channels = channels or [IMGS]
    for key, kind, encoding in (("raw", "image", "raw"), ("seg", "segmentation",
                                                          "compressed_segmentation")):
        for k in range(LEVELS):
            data = (np.stack([c[k] for c in channels], axis=-1) if key == "raw"
                    else LBLS[k][..., None])
            scale = {"resolution": [v * 2**k for v in VOXEL], "encoding": encoding,
                     "chunk_size": [16, 16, 16], "size": list(data.shape[:3]),
                     "voxel_offset": [o // 2**k for o in offset]}
            if encoding == "compressed_segmentation":
                scale["compressed_segmentation_block_size"] = [8, 8, 8]
            store = ts.open(
                {"driver": "neuroglancer_precomputed",
                 "kvstore": {"driver": "file", "path": str(root / key)},
                 "multiscale_metadata": {"type": kind, "data_type": str(data.dtype),
                                         "num_channels": data.shape[-1]},
                 "scale_metadata": scale},
                open=True, create=True,
            ).result()
            store[...] = data
    return {"path": str(root), "image_key": "raw", "label_key": "seg", "format": "precomputed"}


@pytest.fixture(scope="module")
def stores(tmp_path_factory) -> dict:
    base = tmp_path_factory.mktemp("formats")
    out = {"zarr": _zarr(base / "ref.zarr")}
    for convention in N5_CONVENTIONS:
        out[f"n5-{convention}"] = _n5(base / f"{convention}.n5", convention)
    out["precomputed"] = _precomputed(base / "precomputed")
    return out


COPIES = [*(f"n5-{c}" for c in N5_CONVENTIONS), "precomputed"]


def _meta(entry: dict, key: str):
    return read_metadata(entry["path"], entry[key], entry.get("format", "zarr2"))


def _dataset(entry: dict, **settings) -> VolumeDataset:
    cfg = {"volumes": [{"name": "v", **entry}], "output_axes": "lcxyz", "patch_size": [8, 8, 8]}
    return VolumeDataset(MiaoConfig(**{**cfg, **settings}))


def _draws(ds: VolumeDataset, n: int = 8) -> list[dict]:
    np.random.seed(0)
    return [ds[i] for i in range(n)]


def _assert_same_patches(a: VolumeDataset, b: VolumeDataset) -> None:
    for x, y in zip(_draws(a), _draws(b)):
        assert x["meta"]["resolutions"] == y["meta"]["resolutions"]
        np.testing.assert_array_equal(x["img"].numpy(), y["img"].numpy())
        assert ("label" in x) == ("label" in y)
        if "label" in x:
            np.testing.assert_array_equal(x["label"].numpy(), y["label"].numpy())


# --- every copy describes and delivers the zarr pyramid ---------------------------------------------


@pytest.mark.parametrize("copy", COPIES)
@pytest.mark.parametrize("key", ["image_key", "label_key"])
def test_levels_match_zarr(stores, copy, key):
    ref, got = _meta(stores["zarr"], key), _meta(stores[copy], key)
    assert got.axis_names == ref.axis_names == ["x", "y", "z"]
    assert sorted(got.scales) == sorted(ref.scales)
    for level, want in ref.scales.items():
        have = got.scales[level]
        assert have.shape == want.shape and have.dtype == want.dtype
        np.testing.assert_allclose(have.scale_factors, want.scale_factors)
        np.testing.assert_allclose(have.translation_or_zeros(), want.translation_or_zeros())


@pytest.mark.parametrize("copy", COPIES)
@pytest.mark.parametrize("settings", [
    {"resolutions": [[8, 8, 8]]},
    {"resolutions": [[32, 32, 32]]},  # level 2
    {"resolutions": [[8, 8, 8], [16, 16, 16]]},
    {"resolution_sampling": {"ranges": [[[8], [32]]], "n_scales": 1}},
], ids=["8nm", "32nm", "ladder", "sampled"])
def test_patches_match_zarr(stores, copy, settings):
    _assert_same_patches(_dataset(stores["zarr"], **settings), _dataset(stores[copy], **settings))


# --- N5 metadata --------------------------------------------------------------------------------


def test_n5_pyramid_without_s0(tmp_path):
    """The male CNS export starts at s1: level 0 is then s1, half a base voxel off the grid."""
    entry = _n5(tmp_path / "m.n5", "pixelResolution", first_level=1)
    meta = _meta(entry, "image_key")
    assert [s.path for s in meta.scales.values()] == ["s1", "s2"]
    np.testing.assert_allclose(meta.scales[0].scale_factors, [16, 16, 16])
    np.testing.assert_allclose(meta.scales[0].translation, [4, 4, 4])
    sample = _dataset(entry, resolutions=[[16, 16, 16]])[0]
    assert sample["img"].shape == (1, 1, 8, 8, 8)


def test_n5_micrometer_pixel_resolution(tmp_path):
    _n5_array(tmp_path / "um.n5" / "s0", IMGS[0], {})
    (tmp_path / "um.n5" / "attributes.json").write_text(
        json.dumps({"pixelResolution": {"dimensions": [0.008] * 3, "unit": "um"}}))
    meta = read_metadata(tmp_path, "um.n5", "n5")
    np.testing.assert_allclose(meta.scales[0].scale_factors, VOXEL)


def test_n5_single_dataset(tmp_path):
    """A group that is itself an array is a one-level pyramid."""
    _n5_array(tmp_path / "one", IMGS[0], {"resolution": VOXEL})
    meta = read_metadata(tmp_path, "one", "n5")
    assert list(meta.scales) == [0] and meta.scales[0].path == ""
    np.testing.assert_allclose(meta.scales[0].scale_factors, VOXEL)


@pytest.mark.parametrize("level_attrs, group_attrs, message", [
    ({}, {}, "no voxel size"),
    ({"resolution": [8, 8, 8], "downsamplingFactors": [2, 2, 2]}, {"pixelResolution": BASE},
     "disagrees"),
    ({"resolution": [8, 8, 8], "offset": [8, 0, 0]}, {}, "offset"),
    ({}, {"pixelResolution": {"dimensions": VOXEL, "unit": "furlong"}}, "unit"),
], ids=["no-voxel-size", "inconsistent", "offset", "unit"])
def test_n5_metadata_miao_cannot_trust_is_refused(tmp_path, level_attrs, group_attrs, message):
    _n5_array(tmp_path / "bad.n5" / "s1", IMGS[1], level_attrs)
    (tmp_path / "bad.n5" / "attributes.json").write_text(json.dumps(group_attrs))
    with pytest.raises(ValueError, match=message):
        read_metadata(tmp_path, "bad.n5", "n5")


# --- Neuroglancer precomputed -------------------------------------------------------------------


def test_precomputed_voxel_offset_is_a_translation(tmp_path):
    """A precomputed volume whose array starts at a voxel offset reads like a zarr store whose
    levels are translated by that offset."""
    entry = _precomputed(tmp_path / "off", offset=(8, 4, 0))
    ref = _zarr(tmp_path / "off.zarr", base_translation=[64.0, 32.0, 0.0])
    for level, want in _meta(ref, "image_key").scales.items():
        np.testing.assert_allclose(_meta(entry, "image_key").scales[level].translation,
                                   want.translation)
    _assert_same_patches(_dataset(ref, resolutions=[[16, 16, 16]]),
                         _dataset(entry, resolutions=[[16, 16, 16]]))


def test_precomputed_single_channel_opens_as_xyz_from_zero(tmp_path):
    entry = _precomputed(tmp_path / "p", offset=(8, 4, 0))
    key = _meta(entry, "image_key").scales[0].path
    store = open_store(Path(entry["path"]) / "raw" / key, "precomputed")
    assert store.rank == 3 and list(store.domain.inclusive_min) == [0, 0, 0]
    np.testing.assert_array_equal(store.read().result(), IMGS[0])


def test_precomputed_channels_are_pinned_with_fixed_axes(tmp_path):
    """A multi-channel volume keeps a c axis, which fixed_axes pins (and a pinned axis may not be in
    output_axes). Such volumes carry no labels, so the copies compared here are image-only."""
    inverted = [255 - a for a in IMGS]
    entry = _precomputed(tmp_path / "two", channels=[IMGS, inverted])
    assert _meta(entry, "image_key").axis_names == ["x", "y", "z", "c"]
    ref = _zarr(tmp_path / "inv.zarr", imgs=inverted)
    image_only = {k: v for k, v in entry.items() if k != "label_key"}
    settings = {"resolutions": [[8, 8, 8]], "output_axes": "lxyz"}
    _assert_same_patches(
        _dataset({k: v for k, v in ref.items() if k != "label_key"}, **settings),
        _dataset({**image_only, "fixed_axes": {"c": 1}}, **settings),
    )


# --- config ---------------------------------------------------------------------------------------


def test_format_is_optional_and_names_the_store_format():
    vol = {"name": "v", "path": "/x", "image_key": "raw"}
    assert VolumeConfig(**vol).store_format == "zarr2"
    assert VolumeConfig(**vol, zarr_version="zarr3").store_format == "zarr3"
    assert VolumeConfig(**vol, format="n5").store_format == "n5"
    assert VolumeConfig(**vol, format="precomputed").store_format == "precomputed"
    with pytest.raises(ValidationError, match="zarr_version applies to zarr stores only"):
        VolumeConfig(**vol, format="n5", zarr_version="zarr3")
    with pytest.raises(ValidationError):
        VolumeConfig(**vol, format="hdf5")


def test_a_volume_with_a_format_loads_from_its_own_dump():
    """A dump writes every field, zarr_version's default included. The config must load from that
    dump again."""
    cfg = MiaoConfig(volumes=[{"name": "v", "path": "/x", "image_key": "raw", "format": "n5"}],
                     resolutions=[[8, 8, 8]], output_axes="lcxyz", patch_size=[8, 8, 8])
    for dump in (cfg.model_dump(), cfg.model_dump(mode="json")):
        assert dump["volumes"][0]["zarr_version"] == "zarr2"
        assert MiaoConfig(**dump).volumes[0].store_format == "n5"


def test_a_zarr_volume_is_saved_without_a_format_key(stores):
    cfg = MiaoConfig(volumes=[{"name": "v", **stores["zarr"]}], resolutions=[[8, 8, 8]],
                     output_axes="lcxyz", patch_size=[8, 8, 8])
    assert "format" not in cfg.model_dump(exclude_none=True)["volumes"][0]
