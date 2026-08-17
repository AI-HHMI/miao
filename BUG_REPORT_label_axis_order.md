# Bug: labels are silently misaligned when the label group's axis order differs from the image's

**Severity:** High — silent data corruption. No exception, no warning; training just gets
labels that don't correspond to the image.

**Affected:** `VolumeDataset` on `master` (`24ea796`) and every earlier revision. Any volume
where `image_key` and `label_key` resolve to OME-NGFF groups that declare their spatial axes in
different orders.

**Fix:** branch `fix/label-axis-order`, commit `f653cb2`.

---

## Summary

OME-NGFF lets each multiscale group declare its own axis order. Within one zarr container, the
image group and a label group may disagree — e.g. `raw` written as `xyz` and a label written as
`zyx`. miao reads both groups' metadata correctly, but then combines *per-axis vectors from the
two different frames elementwise* without reordering. The result is that label crops are read
from transposed coordinates.

This is masked whenever the two orders happen to agree, which is the overwhelmingly common case
— so it never surfaced in the test suite or in day-to-day use.

## Impact

Two independent failure modes, which can fire together:

1. **Coordinate-order swap.** The sampled patch centre (image axis order) is used directly as a
   label coordinate (label axis order). Fires **even with perfectly isotropic voxels** — it is a
   pure index permutation, not a scaling error.
2. **Voxel-ratio mismatch.** Label voxel sizes are divided by image voxel sizes across mismatched
   axes. Invisible when voxels are isotropic; produces wrong pyramid-level scaling when
   anisotropic.

A third, narrower manifestation: label crops are resampled to the *image-order* patch size, so a
non-square `patch_size` also produces a wrongly-shaped interpolation target.

Nothing raises. The returned `label` tensor has the correct dtype and shape, so downstream code
has no way to notice.

## Root cause

Two frames are silently conflated. `VolumeInfo.finest_voxel_size` and the sampled `center` are
indexed by `img_spatial_idx` (**image** axis order); `lbl_level_voxels`, `label_read_shapes` and
`_build_lbl_slices`' `origin` argument are indexed by `lbl_spatial_idx` (**label** axis order).

Line numbers are against `master` @ `24ea796`:

| # | Location | Code | Problem |
|---|---|---|---|
| 1 | `dataset.py:595` (`_resolve_scales`) | `label_relative_scale_factors.append(lbl_voxel / finest)` | label-order ÷ image-order |
| 2 | `dataset.py:1065` (and `:1061` for `sample_windows`) | `lbl_center = np.floor(center / lbl_rel_factors)` | image-order ÷ label-order; result used as a label-order origin |
| 3 | `dataset.py:664` → `_apply` at `:629` | label-order `rel`/`eff`/`sp_shape` folded straight into image-order `min_center`/`max_center`; the image-order `bounding_box` is also applied to label-order extents | constraint lands on the wrong axes |
| 4 | `dataset.py:1144` | `F.interpolate(lbl_t, size=target_size, ...)` where `target_size` is the image-order read shape | wrong resample target for the label's frame |

Notably this is an **inconsistency, not a blanket assumption**: immediately above site 1, the
same function *does* handle ordering correctly —

```python
lbl_target = np.array(map_patch_size_to_input(list(target_out), lbl_spatial, output_spatial), ...)
lbl_patch  = np.array(map_patch_size_to_input(self.config.patch_size, lbl_spatial, output_spatial), ...)
```

Both explicitly translate into `lbl_spatial` order. The permutation is applied in two of four
places, which is why this reads as an oversight rather than a design decision.

## Reproduction

Build a container whose label holds the *same value* as the image at each physical voxel, but
written in a different axis order. A correct read gives `img == label` elementwise.

```python
import json
import numpy as np, zarr
from zarr.storage import LocalStore
from miao.config import MiaoConfig
from miao.dataset import VolumeDataset

N = 16
zz, yy, xx = np.meshgrid(*[np.arange(N)] * 3, indexing="ij")
val_zyx = (zz * 10000 + yy * 100 + xx).astype(np.float32)   # unique per physical voxel

root = Path("repro.zarr")
g = zarr.open_group(LocalStore(str(root)), mode="a", zarr_format=2)
for key, axes_str, scale, dtype in [
    ("raw", "zyx", [1.0, 1.0, 1.0], "float32"),
    ("seg", "xyz", [1.0, 1.0, 1.0], "uint32"),      # <-- same data, transposed on disk
]:
    data = np.transpose(val_zyx, ["zyx".index(c) for c in axes_str]).astype(dtype)
    grp = g.create_group(key)
    a = grp.create_array("0", shape=data.shape, chunks=data.shape, dtype=dtype, overwrite=True)
    a[:] = data
    (root / key / ".zattrs").write_text(json.dumps({"multiscales": [{
        "version": "0.4",
        "axes": [{"name": c, "type": "space", "unit": "micrometer"} for c in axes_str],
        "datasets": [{"path": "0", "coordinateTransformations": [
            {"type": "scale", "scale": scale}]}]}]}))

cfg = MiaoConfig(
    volumes=[{"name": "v", "path": str(root), "image_key": "raw",
              "label_key": "seg", "normalize": False}],
    resolutions=[[1, 1, 1]], output_axes="lzyx",
    patch_size=[4, 4, 4], samples_per_epoch=20,
)
ds = VolumeDataset(cfg)
np.random.seed(0)
bad = sum(
    not np.array_equal(ds[i]["img"][0].numpy(), ds[i]["label"][0].numpy().astype(np.float32))
    for i in range(20)
)
print(f"{bad}/20 samples misaligned")     # master: ~17/20   fixed: 0/20
```

Not all 20 fail, because the swap is a no-op whenever the permuted coordinates happen to
coincide — a signature of a coordinate permutation rather than a scaling error.

## Affected data in `/groups/miaai/miaai`

I scanned every OME-NGFF group under `/groups/miaai/miaai`: **2070 multiscale groups across 783
zarr containers**, enumerating all **1310** `(image_key, label_key)` combinations expressible in
a single `VolumeConfig` (miao resolves both keys as plain paths under one `path`, so labels
nested inside a sub-`.zarr` are pairable too).

**16 pairs mismatch**, reducing to **8 distinct `raw`-based pairs across 4 containers / 2 logical
datasets** (each dataset has a `.tmp` staging twin with identical structure). The count of 16
double-counts because each container root also declares `multiscales`, so `image_key="."` and
`image_key="raw"` both pair.

### 1. Drosophila FlyLICONN organelle — **confirmed real problem**

```
/groups/miaai/miaai/lmd-v0.0.1/liconn_data/Organelle/20260216_FlyID25_MOPS_2ndGel_B1_40XW003.zarr
/groups/miaai/miaai/lmd-v0.0.1/data/exm-drosophila-flyliconn-organelle.tmp/crop-001.zarr
```

| group | axes | level-0 voxel (nm) | level-0 shape | dtype | levels |
|---|---|---|---|---|---|
| `raw` | `xyz` | `[160, 160, 400]` | `[2304, 2304, 484]` | uint16 | 6 |
| `labels/mito_combined` | `xyz` | `[160, 160, 400]` | `[2304, 2304, 484]` | uint8 | 4 | 
| **`labels/mito001_combined`** | **`zyx`** | `[400, 160, 160]` | `[484, 2304, 2304]` | uint32 | 4 |

**Affected pair: `image_key="raw"` + `label_key="labels/mito001_combined"`.**

This is the worst case: the voxels are anisotropic (400 nm vs 160 nm), so *both* failure modes
fire. Verified against the real container — miao derives

```
label_relative_scale_factors = [2.5, 1.0, 0.4]      # should be [1, 1, 1]
```

The sibling `labels/mito_combined` is correctly `xyz`, so this looks like a one-off inconsistency
in how that single annotation was written, not a convention.

### 2. Mouse LICONN ExPID82-1 — **only reachable via cross-container pairing**

```
/groups/miaai/miaai/lmd-v0.0.1/liconn_data/ExPID82-1.zarr
/groups/miaai/miaai/lmd-v0.0.1/data/exm-mouse-liconn-expid82-1.tmp/crop-001_fullvol.zarr
```

| group | axes | spatial | dtype |
|---|---|---|---|
| `raw` | `zyx` | `zyx` | uint8 |
| `labels/seg_231030_agg_240123` | `zyx` | `zyx` — matches, fine | |
| `ExPID82-1-dense_internal_tmp.zarr/segmentation_dense` | `cxyz` | **`xyz`** | uint64 |
| `ExPID82-1_wk.zarr/color` | `cxyz` | `xyz` | uint8 |
| `ExPID82-1_wk.zarr/segmentation` | `cxyz` | `xyz` | uint64 |
| `ExPID82-1_wk.zarr/segmentation_dense` | `cxyz` | `xyz` | uint64 |

The intended label, `labels/seg_231030_agg_240123`, is `zyx` and matches `raw` — fine.

The remaining groups live in nested `.zarr` subdirectories, and they split into two very
different cases:

- **`ExPID82-1_wk.zarr` is self-contained and internally consistent.** It is a webKnossos export
  (note the `datasource-properties.json` and the wk convention of naming the image layer
  `color`), and it ships its own image: `color` (uint8, `cxyz`) alongside `segmentation` /
  `segmentation_dense` (uint64, `cxyz`). All three agree on spatial order, so the natural config
  — `path=.../ExPID82-1_wk.zarr, image_key=color, label_key=segmentation` — has **no mismatch**
  and loads fine (verified). A mismatch arises only if you point `path` at the *outer* container
  and reach into the wk export for labels while ignoring its own `color` layer. That is an
  artificial pairing; treat these as **not affected in practice**.

- **`ExPID82-1-dense_internal_tmp.zarr` contains only `segmentation_dense`, with no image
  layer.** So it cannot be used self-contained: pairing it with an image necessarily means
  reaching out to the outer `raw` (`zyx`) against its own `xyz`, which *is* a mismatch. The
  `_internal_tmp` naming suggests scratch that nobody trains on, but if it is used at all, it is
  affected.

Recount on this basis, of the 8 `raw`-based mismatched pairs:

| | pairs | assessment |
|---|---|---|
| Organelle `labels/mito001_combined` | 2 | **genuine** — a real annotation beside a correct sibling |
| ExPID82-1 `_internal_tmp/segmentation_dense` | 2 | **plausible** — labels-only, must cross-pair to be used |
| ExPID82-1 `_wk/segmentation{,_dense}` | 4 | **artificial** — self-contained with a matching `color` image |

### Everything else is clean

The remaining 779 containers agree between image and label. Worth noting for context: **both
conventions are in active use collection-wide** — 1531 groups `xyz`, 257 `zyx`, 132 `cxyz`, 128
`tczyx`, 22 `czyx` — so this is not a one-dataset quirk that can be safely ignored going forward.

One methodological note: an early version of this scan compared *full* axes and produced false
positives across all the `nisb/*` datasets (image `cxyz`, label `xyz`). Those differ only in
channel presence and their **spatial** order is `xyz` in both; miao handles channel presence
explicitly. Only spatial order matters here.

## Fix

Adds `axes.reorder_per_axis(values, from_axes, to_axes)` and applies it at each of the four
image/label frame boundaries, plus a note on `ScaleResolution` documenting that its `label_*`
fields are in label axis order while every other field is in image axis order.

```
 src/miao/axes.py      |  21 ++++++
 src/miao/dataset.py   |  79 ++++++++++++++++----
 tests/test_axes.py    |  41 +++++++++-
 tests/test_dataset.py | 162 ++++++++++++++++++++++++++++++++++++++++++
```

No behaviour change when the two orders agree — the helper short-circuits on `from_axes ==
to_axes`, and all 124 pre-existing tests pass untouched.

## Test coverage added

`TestReorderPerAxis` (8 unit tests) and `TestLabelAxisOrder` (10 integration tests) covering
reversed and cyclic label orders across isotropic, anisotropic, non-square-patch, multiscale,
`bounding_box` and `sequential` cases, plus a direct assertion on the derived
`label_relative_scale_factors`.

**9 of the 10 integration tests fail against pre-fix `dataset.py`** (verified by reverting the
source and re-running); the tenth is the matching-axes control, which was never broken.

Two traps worth knowing about if these tests are ever modified:

- The anisotropic fixtures deliberately use three *distinct* voxel sizes (`x=4, y=2, z=1`). An
  earlier draft used `[1, 4, 1]`, which is palindromic — the mixed-order ratio coincidentally
  came out `[1, 1, 1]` and the test passed against broken code.
- The multiscale fixture has real pyramid levels so the requested resolutions map exactly onto
  them. With a single level the image resamples trilinear (averaging) while the label resamples
  nearest, so `img == label` fails for reasons unrelated to this bug.

## Related

`feature/ImageDataset` (a 2D-slice dataset built on the same shared machinery) inherits this bug
and needs rebasing onto the fix.
