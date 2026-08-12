# miao
[![PyPI version](https://badge.fury.io/py/miao-io.svg)](https://pypi.org/project/miao-io/)

Scalable PyTorch data loaders for OME-NGFF zarr datasets, powered by TensorStore.

## Installation

```bash
pip install miao-io
```

## Quick start

**1. Write a config**

```yaml
# config.yaml
volumes:
  - name: "raw"
    path: "/data/sample_A.zarr"
    image_key: "raw"
    zarr_version: "zarr2"      # or "zarr3" (default: "zarr2")
    label_key: "labels/seg"    # optional
    weight: 0.5                # optional

  - name: "membrane"
    path: "/data/sample_B.zarr"
    image_key: "predictions"
    zarr_version: "zarr3"
    weight: 0.2

  - name: "expanded"
    path: "/data/expanded.zarr"
    image_key: "raw"
    exp_factor: 4.0            # optional, default: 1.0 — see Effective voxel sizes
    weight: 0.3

resolutions: [[8, 8, 8], [16, 16, 16], [32, 32, 32]]  # effective output voxel size per scale
output_axes: "lcxyz"            # layer, channels, X, Y, Z. Shuffle as you please!!!
patch_size: [64, 64, 64]
samples_per_epoch: 1000
cache_bytes: 1073741824         # 1 GB tensorstore cache
```

**2. Build a dataset**

```python
from torch.utils.data import DataLoader
from miao import VolumeDataset, load_config

dataset = VolumeDataset(load_config("config.yaml"))
loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8,
                    pin_memory=True, persistent_workers=True)

for batch in loader:
    img, label = batch["img"], batch["label"]
```

Runnable notebooks live in [`examples/`](examples/).

## Batch format

| Key | Shape | Description |
|---|---|---|
| `img` | `(B, L, X, Y, Z)`, or `(B, L, C, X, Y, Z)` with a channel axis | Image patch per scale level |
| `label` | `(B, L, X, Y, Z)`, empty without a `label_key` | Label patch per scale level |
| `bbox` | `(B, L, 2, Nd_spatial)` | Patch extent per level, world or crop-relative (see `bbox_mode`) |
| `pixel_size` | `(B, L, Nd_spatial)` | Effective physical output voxel size per level |
| `meta` | dict | `name`, `coordinate`, `resolutions` (mirroring `pixel_size`), source pyramid levels, plus `grid_index` in sequential mode |

Spatial axes follow `output_axes` throughout. `pixel_size` sits at the top level rather than inside `meta` so default collate stacks it cleanly,
ready to feed a scale-conditioned model. It reports the resolutions *actually used* for that
sample, which under random resolution sampling differ from sample to sample.

## How scales work

Scales are addressed by **desired output resolution** — physical voxel size per axis, in the zarr's
OME `coordinateTransformations` unit (e.g. nanometers) — never by pyramid level index. One
`resolutions` list therefore serves volumes with wildly different pyramid layouts, and any volume
can override it with its own.

### Effective voxel sizes

**Every voxel size and resolution in this README is an *effective* voxel size: the size of one
voxel in the specimen, not in the microscope.** For a volume with `exp_factor: F`, the effective
voxel size is the stored `coordinateTransformations` scale divided by `F` — a 4× expanded sample
imaged at 200 nm holds 50 nm of biology per voxel. `exp_factor` defaults to `1.0`, so for
unexpanded data the stored and effective sizes are identical.

Level selection, read extents, `pixel_size`, and `meta["resolutions"]` are all in effective units,
which is what lets one `resolutions` list mix expanded and unexpanded volumes correctly.
`bounding_box` and `meta["coordinate"]` are level-0 voxel *indices*, not physical sizes, so
`exp_factor` does not affect them.

Per sample, miao:

1. draws a volume by `weight`, then a random coordinate in its finest-scale (level-0) space;
2. for each requested resolution, picks the coarsest pyramid level whose effective voxel size is
   still ≤ the target on every axis, preferring downsampling — or the finest stored level,
   upsampled, if the target is finer than anything on disk;
3. reads `ceil(patch_size × target_resolution / effective_level_voxel_size)` voxels centered on
   that coordinate and resamples them to `patch_size` — trilinear for images, nearest for labels.

Every crop thus holds the same voxel count while covering a wider physical extent at coarser
scales.

### Random resolution sampling

To train resolution-agnostic models, let each sample draw its own scales. Set
`resolution_sampling` *instead of* `resolutions`, globally or per volume:

```yaml
resolution_sampling:
  strategy: log_uniform     # only option for now (pluggable; gaussian etc. can be added)
  ranges: [[[8], [64]]]     # one or more [min, max] ranges
  n_scales: 3               # scales per range — scalar (all ranges) or list (one per range)
  sort: true                # sort drawn scales fine -> coarse
```

The shape of each bound decides isotropy:

- `[[v], [w]]` is **isotropic** — one value per scale, broadcast to all axes. `[[[1], [4]]]` gives
  cubic voxels between 1 and 4.
- `[[a, b, c], [d, e, f]]` draws **each axis independently**, so `[[[1, 1, 1], [4, 4, 4]]]` may
  yield `[1.8, 3.2, 2.5]` — not the same distribution as `[[[1], [4]]]`.

Ranges stack: `[[[1], [2]], [[4], [8]]]` with `n_scales: [1, 1]` gives two scales, the `l`
dimension being the sum across ranges. Draws are log-uniform and redrawn every `__getitem__`.

Exactly one of `resolutions` / `resolution_sampling` may be set. `resolution_sampling` rules out
`sampling: "sequential"`, and `sample_windows` demands isotropic ranges.

## Sampling

Random sampling — the default, and the mode described above — draws `samples_per_epoch` patches per
epoch. Two settings change *where* patches land:

### Sequential (inference / evaluation)

Walk the whole volume on a deterministic grid instead.

```yaml
sampling: "sequential"
overlap: 16              # or per-axis list, e.g. [16, 16, 8]
```

```python
dataset = VolumeDataset(config)                            # len() = grid positions, all volumes
loader = DataLoader(dataset, batch_size=4, shuffle=False)   # shuffle=False required

for batch in loader:
    # e.g. (2, 0, 3) — grid position per axis; use it to stitch predictions into a full volume
    grid_index = batch["meta"]["grid_index"]
```

The grid tiles the volume at the **first scale's** target resolution, striding one output patch
minus `overlap` worth of physical extent. Coverage is gap-free even on anisotropic source data.
`samples_per_epoch` and `weight` are ignored here, and volumes are exhausted in order.

### Multi-scale windows (`sample_windows`)

By default all scales share one center, nesting finer patches inside coarser ones. With
`sample_windows: true` each coarser level instead drops its patch origin uniformly at random among
the positions that still cover the previous level's patch.

```yaml
sample_windows: true
```

Those sampled origins stay strictly inside any per-volume `bounding_box`. `resolutions` must run
finest to coarsest (non-decreasing voxel size per axis, e.g. `[[8,8,8], [16,16,16]]`); putting a
coarser resolution first raises.

## Augmentation (`aug_rot`)

Set `aug_rot: true` on a volume to reorient every sample at random:

```yaml
volumes:
  - name: "raw"
    path: "/data/sample_A.zarr"
    image_key: "raw"
    label_key: "labels/seg"
    aug_rot: true            # optional, default: false
```

Each `__getitem__` draws **one** transform uniformly from the 48 symmetries of the cube (3! axis
permutations × 2³ per-axis flips) — the largest augmentation set reachable without interpolation,
since every such transform maps voxels exactly onto voxels. The draw applies to image and labels
together, and to every scale, so alignment and nesting survive.

**Isotropic output is required**, since permuting axes only means something when voxels are cubes.
Two checks enforce it: `patch_size` must be equal across `x`, `y`, `z` (at config load), and the
resolution being read must be equal on all axes (per sample, because `resolution_sampling` can
draw a fresh one each call — use isotropic single-value bounds). Anisotropic data *on disk* is
fine — e.g. coarse `z` upsampled to match — as long as the requested output is isotropic.

> **Note:** `aug_rot` reorients `img` and `label` only. `bbox` and `meta["coordinate"]` still
> describe the original read location; `pixel_size` is isotropic by construction, so unaffected.

### Custom augmentations (`augment_fn`)

For anything beyond `aug_rot`, pass a callable to the dataset. It is called once per sample on
the finished sample dict, as `augment_fn(sample) -> sample`, and composes the pure functions in
`miao/augment.py` with an rng of your choosing:

```python
import numpy as np
from miao import VolumeDataset, load_config
from miao.augment import intensity_jitter, rot90isocube

def augment(sample):
    rng = np.random  # the global numpy RNG; PyTorch reseeds it per DataLoader worker
    img, lab = sample["img"], sample["label"]
    img, lab = rot90isocube(rng, img, lab, pixel_size=sample["pixel_size"])
    img = intensity_jitter(rng, img)
    return {**sample, "img": img, "label": lab}

dataset = VolumeDataset(load_config("config.yaml"), augment_fn=augment)
```

Using the `np.random` module keeps multi-worker loading correct for free; a
`np.random.default_rng()` Generator closed over from the parent process works too, but
fork-inherited copies draw identical streams across workers unless you reseed it in a
`worker_init_fn`. Forwarding `sample["pixel_size"]` to `rot90isocube` enables its isotropy check.
For volumes without labels, skip the empty label sentinel (`sample["label"].numel() == 0`)
instead of rotating it. Don't enable `aug_rot: true` and a rotating `augment_fn` together — the
sample would rotate twice.

## Configuration reference

### Per-volume fields

| Field | Description |
|---|---|
| `name` | Unique name for the volume |
| `path` | Path to the OME-NGFF zarr container |
| `image_key` | Group key within the zarr for image data |
| `zarr_version` | `"zarr2"` or `"zarr3"` (default: `"zarr2"`) |
| `exp_factor` | Divides the zarr's metadata voxel size to give the effective (pre-expansion) voxel size (default: `1.0`). For expansion microscopy the stored value is the microscope's; `stored / exp_factor` is the specimen's. Applies to image and labels, before level selection — see [Effective voxel sizes](#effective-voxel-sizes) |
| `label_key` | Optional group key for labels in the same zarr |
| `weight` | Sampling probability weight (default: equal across volumes) |
| `resolutions` | Optional per-volume override of the global `resolutions` (same format) |
| `normalize` | Scale images to [0, 1] by dtype max (default: `true`); `normalize_min` / `normalize_max` set the bounds explicitly |
| `patch_normalize` | Standardize each sample to zero mean / unit variance after `normalize` (default: `false`). Multi-scale: statistics come from the coarsest crop and apply to all scales |
| `bounding_box` | Optional `[[min, max], ...]` per spatial axis in level-0 voxels. Confines every read extent at every scale, `sample_windows` patches included — not merely the patch center. Must be at least as large as the coarsest window, or construction raises |
| `aug_rot` | Random axis-aligned rotation/flip per sample (default: `false`) — see [Augmentation](#augmentation-aug_rot) |

### Dataset fields

| Field | Description |
|---|---|
| `resolutions` | One output resolution tuple per scale; `len()` is the `l` dimension. Mutually exclusive with `resolution_sampling` |
| `resolution_sampling` | Draw resolutions per sample instead: `{strategy, ranges, n_scales, sort}` — see [above](#random-resolution-sampling). Mutually exclusive with `resolutions` |
| `output_axes` | Tensor dim order: `l` (levels), optional `c` (channel), spatial dims (e.g. `"lcxyz"`, `"lxyz"`) |
| `patch_size` | Voxel count per crop, in `output_axes` spatial order |
| `samples_per_epoch` | Number of samples per epoch |
| `cache_bytes` | TensorStore cache size in bytes (default: 1 GB) |
| `bbox_mode` | `"absolute"` world coords (default) or `"relative"` to the finest-level crop origin |
| `sampling` | `"random"` (default) or `"sequential"` — see [Sampling](#sampling) |
| `overlap` | Patch overlap in voxels, sequential mode only (default: `0`). Integer or per-axis list |
| `sample_windows` | Randomize each coarser scale's patch origin (default: `false`) — see [above](#multi-scale-windows-sample_windows) |

Input axes come from OME-NGFF metadata (`multiscales.axes`) and never need specifying; channel
dimensions are picked up automatically when present.

## Recipe: 2D image datasets

2D models are often strong baselines in 3D domains. Build one from three degenerate
`VolumeDataset`s — `patch_size` `(1,P,P)`, `(P,1,P)`, `(P,P,1)` — glued with `ConcatDataset`.

<details>
<summary><b>Show the recipe</b></summary>

```python
from torch.utils.data import ConcatDataset, DataLoader, default_collate
from miao import MiaoConfig, VolumeDataset, load_config

base = load_config("config.yaml")
P = 128
spatial_order = "".join(c for c in base.output_axes if c in "xyzt")   # "xyz"

def plane_config(cfg: MiaoConfig, thin_axis: str, size: int) -> MiaoConfig:
    """Config for a one-voxel-thick patch, i.e. a plane normal to `thin_axis`."""
    data = cfg.model_dump()
    data["patch_size"] = [1 if c == thin_axis else size for c in spatial_order]
    data["samples_per_epoch"] = cfg.samples_per_epoch // len(spatial_order)
    return MiaoConfig(**data)

dataset = ConcatDataset(
    [VolumeDataset(plane_config(base, ax, P)) for ax in spatial_order]
)
```

Each sub-dataset puts its singleton on a different axis, so a shuffled batch mixes shapes and
default collate chokes. Squeeze the thin axis away instead:

```python
def make_plane_collate(output_axes: str):
    """Collate 1-voxel-thick samples by squeezing away the singleton spatial axis."""
    img_dim = {c: i for i, c in enumerate(output_axes) if c in "xyzt"}
    label_axes = output_axes.replace("c", "")     # labels carry no channel axis

    def collate(samples):
        squeezed, planes = [], []
        for sample in samples:
            img = sample["img"]
            thin = [c for c, d in img_dim.items() if img.shape[d] == 1]
            if len(thin) != 1:
                raise ValueError(f"expected one singleton spatial axis, found {thin}")
            axis = thin[0]
            sample = dict(sample)
            sample["img"] = img.squeeze(img_dim[axis])
            if sample["label"].numel():           # empty when a volume has no labels
                sample["label"] = sample["label"].squeeze(label_axes.index(axis))
            squeezed.append(sample)
            planes.append("".join(c for c in img_dim if c != axis))
        batch = default_collate(squeezed)
        batch["plane"] = planes                   # e.g. "yz" — one entry per sample
        return batch

    return collate

loader = DataLoader(dataset, batch_size=8, shuffle=True,
                    collate_fn=make_plane_collate(base.output_axes))

for batch in loader:
    img = batch["img"]                # (B, L, C, P, P)
    label = batch["label"]            # (B, L, P, P)
    bbox = batch["bbox"]              # (B, L, 2, Nd_spatial) — still 3D
    plane = batch["plane"]            # "yz" / "xz" / "xy", one per sample
```

`bbox` and `pixel_size` keep their full spatial rank, so a plane's position, orientation and
physical thickness all survive collation. Index a single scale out with `batch["img"][:, 0]` for a
`(B, C, H, W)` tensor. Runnable: [`examples/example_2d.ipynb`](examples/example_2d.ipynb).

### Caveats

- `samples_per_epoch` counts per sub-dataset, so `len(ConcatDataset)` is the sum of three. Set
  them unequally to bias the orientation mix.
- Each `VolumeDataset` opens its own TensorStore cache per worker — divide `cache_bytes` by three
  to hold the footprint constant.
- `aug_rot` is out, since it wants a cubic `patch_size`. Flip and transpose the collated planes
  yourself.
- Mixing labelled and unlabelled volumes breaks collation: an empty `label` tensor will not stack
  against a real one.
- Under `sampling: "sequential"`, a non-zero scalar `overlap` fails validation — it must be
  smaller than *every* `patch_size` entry, so pass a per-axis list with `0` on the thin axis (the
  default `overlap: 0` is fine as a scalar). The grid strides one output plane's extent along the
  normal, which equals one stored plane only when the read count is 1; otherwise planes overlap.
- The three orientations are not equivalent on anisotropic data: at 4×4×33 nm, an `xy` plane is a
  stored section while `xz` and `yz` are mostly interpolation along `z`.
- Nearest-neighbor label resampling takes the *first* plane of a slab, not the middle one. So on a
  resampled thin axis the label plane comes from a different depth than the image plane, off by up
  to half a thickness. Read counts resolve from the *label* pyramid independently, so forcing the
  image count to 1 is not enough — check both. If the two pyramids differ enough, the label plane
  can fall outside the image slab entirely.

</details>

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- TensorStore >= 0.1.60
- Zarr datasets following the [OME-NGFF](https://ngff.openmicroscopy.org/latest/) spec (v2 or v3)
