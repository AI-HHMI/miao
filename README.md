# miao

[![PyPI version](https://img.shields.io/pypi/v/miao-io)](https://pypi.org/project/miao-io/)

Scalable PyTorch data loaders for OME-NGFF zarr datasets, powered by TensorStore.

- Multi-scale patches from arbitrary pyramid layouts, addressed by **physical resolution** rather than level index
- Random or deterministic (grid) sampling across many weighted volumes
- Input axis order auto-detected from OME-NGFF metadata; zarr v2 and v3

## Installation

```bash
pip install miao-io
```

## Quick start

**1. Write a config file**

```yaml
# config.yaml
volumes:
  - name: "raw"
    path: "/data/sample_A.zarr"
    image_key: "raw"
    zarr_version: "zarr2"      # "zarr2" or "zarr3" (default: "zarr2")
    label_key: "labels/seg"    # optional
    weight: 0.7                # optional, default: equal

  - name: "membrane"
    path: "/data/sample_B.zarr"
    image_key: "predictions"
    zarr_version: "zarr3"
    weight: 0.3

resolutions: [[8, 8, 8], [16, 16, 16], [32, 32, 32]]  # output voxel size per scale
output_axes: "lcxyz"            # layer, channels, X, Y, Z. Shuffle as you please!!!
patch_size: [64, 64, 64]
samples_per_epoch: 1000
cache_bytes: 1073741824         # 1 GB tensorstore cache
```

**2. Build a dataset**

```python
from torch.utils.data import DataLoader
from miao import VolumeDataset, load_config

config = load_config("config.yaml")
dataset = VolumeDataset(config)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)

for batch in loader:
    img = batch["img"]
    label = batch["label"]
    bbox = batch["bbox"]
    pixel_size = batch["pixel_size"]
    meta = batch["meta"]
```

Runnable notebooks live in [`examples/`](examples/).

## Batch format

| Key | Shape | Description |
|---|---|---|
| `img` | `(B, L, X, Y, Z)`, or `(B, L, C, X, Y, Z)` with a channel axis | Image patches, one per scale level |
| `label` | `(B, L, X, Y, Z)`, or empty when the volume has no `label_key` | Label patches |
| `bbox` | `(B, L, 2, Nd_spatial)` | Patch extent per level, `absolute` or `relative` (see `bbox_mode`) |
| `pixel_size` | `(B, L, Nd_spatial)` | Physical output voxel size per level |
| `meta` | dict | `name`, `coordinate`, `resolutions`, source pyramid levels, and `grid_index` in sequential mode |

Spatial axis order follows `output_axes` everywhere — `img`, `label`, `bbox`, `pixel_size`, and every spatial field in `meta`.

`pixel_size` is a first-class top-level tensor (not nested under `meta`) so the default PyTorch
collate stacks it cleanly, which is convenient for scale-conditioned models. Its unit is the
zarr's OME `coordinateTransformations` unit (e.g. nanometers). It reports the resolutions
*actually used* for each sample, so under [random resolution
sampling](#random-resolution-sampling) it reflects the freshly drawn values on every
`__getitem__`. The same values are mirrored in `meta["resolutions"]`.

## How it works

Each sample:

1. Picks one volume at random, according to the sampling weights
2. Picks a random coordinate in that volume's finest-scale (level-0) space
3. For each requested resolution, reads from the chosen pyramid level, centered at that
   coordinate, however many voxels are needed to yield `patch_size` voxels at the target
   resolution after resampling — i.e. `ceil(patch_size × target_resolution / level_voxel_size)`
   voxels — then resamples that read to `patch_size`

All crops therefore have the same voxel count (`patch_size`) but cover increasing physical
extents at coarser resolutions. Images are resampled trilinearly, labels with nearest neighbor.

## Resolutions

Scales are defined by **desired output resolution** — physical voxel size per axis, in the same
unit as the zarr's OME `coordinateTransformations` — not by pyramid level index. For each volume
and each requested resolution, miao reads from the coarsest pyramid level whose voxel size is
still ≤ the target on every axis (preferring downsampling), then resamples the patch to that
resolution. If a target is finer than the finest stored level, miao reads the finest level and
upsamples instead.

The same `resolutions` list applies to every volume regardless of how each volume's pyramid is
laid out. A volume may override it with its own `resolutions`.

There is no separate `isotropic` flag: request equal-valued resolution tuples (e.g. `[8, 8, 8]`)
and miao resamples each axis to that common voxel size automatically.

### Random resolution sampling

Instead of a fixed `resolutions` list, each sample can draw its own resolutions from a range —
useful for training resolution-agnostic models. Set `resolution_sampling` (globally or as a
per-volume override) *instead of* `resolutions`:

```yaml
resolution_sampling:
  strategy: log_uniform     # only option for now (pluggable; gaussian etc. can be added)
  ranges: [[[8], [64]]]     # one or more [min, max] ranges
  n_scales: 3               # scales per range — scalar (all ranges) or list (one per range)
  sort: true                # sort the drawn scales fine -> coarse
```

`ranges` is a list of `[min, max]` pairs. How each bound is written controls whether the drawn
resolution is isotropic:

- **Single-element bounds** `[[v], [w]]` are **isotropic**: one value is drawn per scale and
  broadcast to all axes (cubic voxel). E.g. `[[[1], [4]]]` — one range, cubic, 1 to 4.
- **Per-axis bounds** `[[a, b, c], [d, e, f]]` draw **each axis independently**, producing an
  anisotropic voxel. E.g. `[[[1, 1, 1], [4, 4, 4]]]` draws x, y, z separately in 1..4, so a
  sample might be `[1.8, 3.2, 2.5]` — this is *not* the same as the isotropic `[[[1], [4]]]`.

Multiple ranges are allowed, e.g. `[[[1], [2]], [[4], [8]]]`. `n_scales` is the number of scales
drawn from each range; the total number of scales (the `l` dimension) is the sum. Each
`__getitem__` draws log-uniformly within each range and sorts the result fine→coarse. The output
is always `patch_size` per scale.

Constraints: exactly one of `resolutions` / `resolution_sampling` may be set;
`resolution_sampling` is incompatible with `sampling: "sequential"`; and `sample_windows`
requires every range to be isotropic.

## Sampling

By default (`sampling: "random"`) each of the `samples_per_epoch` samples picks a volume by
weight and a random location within it, and every scale is centered on that location.

### Sequential sampling (inference / evaluation)

Set `sampling: "sequential"` to iterate over the entire volume in a deterministic grid instead
of sampling at random. Useful for dense inference and evaluation.

```yaml
sampling: "sequential"
overlap: 16              # or per-axis list, e.g. [16, 16, 8]
```

```python
dataset = VolumeDataset(config)
# len(dataset) = total grid positions across all volumes

loader = DataLoader(dataset, batch_size=4, shuffle=False)  # shuffle=False required

for batch in loader:
    # meta["grid_index"]: tuple e.g. (2, 0, 3) = position in the grid per axis.
    # Use it to stitch patch predictions back into a full-volume output.
    grid_index = batch["meta"]["grid_index"]
```

The grid tiles the volume at the **first scale's** target resolution: the stride is one output
patch (minus `overlap`) worth of physical extent. This gives full coverage of the output volume
with no gaps, even when the source data is anisotropic.

`samples_per_epoch` and per-volume `weight` are ignored in this mode. With multiple volumes, all
positions of volume 0 are yielded before volume 1.

### Multi-scale window sampling (`sample_windows`)

By default every scale level shares the same center location, so finer patches sit centered
inside coarser ones. With `sample_windows: true`, each coarser level instead picks its patch
origin uniformly at random among the positions that still cover the previous level's patch.

```yaml
sample_windows: true
```

Sampled coarse patch locations stay strictly within any per-volume `bounding_box` — the box
bounds every scale's read extent, not just the patch center.

`resolutions` must be listed finest to coarsest (non-decreasing voxel size per axis, e.g.
`[[8,8,8], [16,16,16]]`); ordering a coarser resolution before a finer one raises an error.

## Augmentation (`aug_rot`)

Set `aug_rot: true` on a volume to augment each returned sample with a random axis-aligned
rotation/flip:

```yaml
volumes:
  - name: "raw"
    path: "/data/sample_A.zarr"
    image_key: "raw"
    label_key: "labels/seg"
    aug_rot: true            # optional, default: false
```

Each `__getitem__` draws **one** transform uniformly from the 48 symmetries of the cube — the
full set of axis-aligned 3D rotations and reflections (3! axis permutations × 2³ per-axis flips).
This is the largest augmentation set achievable without interpolation, since every transform maps
voxels onto voxels exactly. The same transform is applied to the image and its labels together
(so they stay aligned) and to every requested scale (so the nested scales stay mutually
consistent).

**Isotropic output is required**, because an axis permutation mixes the spatial axes, which is
only meaningful when the output voxels are cubes. Two conditions are checked:

- `patch_size` must be equal across `x`, `y`, `z` (validated when the config is loaded).
- The output resolution being read must be equal on all spatial axes (validated per sample, since
  `resolution_sampling` can draw a different resolution each call — use isotropic single-value
  bounds).

It is fine for the **stored** data to be anisotropic — e.g. `z` coarser than `x`/`y` and
upsampled — as long as the requested **output** resolution is isotropic.

> **Note:** `aug_rot` reorients the returned `img` and `label` tensors only. `bbox` and
> `meta["coordinate"]` still describe the original, pre-augmentation read location in the source
> volume; `pixel_size` is unaffected because it is isotropic by construction.

## Configuration reference

### Per-volume fields

| Field | Description |
|---|---|
| `name` | Unique name for the volume |
| `path` | Path to the OME-NGFF zarr container |
| `image_key` | Group key within the zarr for image data |
| `zarr_version` | `"zarr2"` or `"zarr3"` (default: `"zarr2"`) |
| `label_key` | Optional group key for labels in the same zarr |
| `weight` | Sampling probability weight (default: equal across volumes) |
| `resolutions` | Optional per-volume override of the global `resolutions` (same format) |
| `normalize` | Auto-normalize images to [0, 1] by dtype max (default: `true`). See also `normalize_min` / `normalize_max` to set the bounds |
| `patch_normalize` | Standardize each sample to zero mean / unit standard deviation, applied after `normalize` (default: `false`). With multiple scales, statistics come from the coarsest-resolution crop and are applied to every scale |
| `bounding_box` | Optional `[[min, max], ...]` per spatial axis (finest-level voxels, `output_axes` spatial order). Every window's read extent — at every scale, including coarser `sample_windows` patches — is kept strictly inside the box. Must be at least as large as the coarsest window, or dataset construction raises |
| `aug_rot` | Apply a random axis-aligned rotation/flip to each sample (default: `false`). Requires isotropic output — see [Augmentation](#augmentation-aug_rot) |

### Dataset fields

| Field | Description |
|---|---|
| `resolutions` | List of desired output resolutions, one tuple per scale, each in `output_axes` spatial order. The number of scales (the `l` dimension) is `len(resolutions)`. Mutually exclusive with `resolution_sampling` |
| `resolution_sampling` | Draw resolutions randomly per sample instead: `{strategy, ranges, n_scales, sort}`. See [Random resolution sampling](#random-resolution-sampling). Mutually exclusive with `resolutions` |
| `output_axes` | Full tensor dim order including `l` (levels), optional `c` (channel), and spatial dims (e.g. `"lcxyz"`, `"lxyz"`) |
| `patch_size` | Voxel count per crop, in `output_axes` spatial order |
| `samples_per_epoch` | Number of samples per epoch |
| `cache_bytes` | TensorStore cache size in bytes (default: 1 GB) |
| `bbox_mode` | `"absolute"` (world coords, e.g. nm) or `"relative"` (relative to the finest-level crop origin). Default: `"absolute"` |
| `sampling` | `"random"` (default) or `"sequential"` — see [Sampling](#sampling) |
| `overlap` | Voxels of overlap between adjacent patches in sequential mode (default: `0`). Integer, or a list in `output_axes` spatial order, e.g. `[16, 16, 8]` |
| `sample_windows` | Randomize each coarser scale's patch origin (default: `false`). Requires more than one scale and fine-to-coarse `resolutions` — see [Multi-scale window sampling](#multi-scale-window-sampling-sample_windows) |

Input axes are auto-detected from OME-NGFF metadata (`multiscales.axes`) and never need to be
specified. Channel dimensions, if present in the image, are included automatically.

## Recipe: 2D image datasets

2D models are often strong baselines in 3D domains. You can build a 2D dataset out of
`VolumeDataset` by creating three degenerate datasets with `patch_size` `(1,P,P)`, `(P,1,P)`, and
`(P,P,1)`, then concatenating them with `torch.utils.data.ConcatDataset`:

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

The three sub-datasets put the singleton on different axes, so a shuffled batch mixes shapes and
the default collate function fails. A custom collate function fixes that by squeezing the thin
axis away:

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
physical thickness all survive collation. To retrieve a single scale, index it out —
`batch["img"][:, 0]` returns a `(B, C, H, W)` tensor. A runnable version of all of this is in
[`examples/example_2d.ipynb`](examples/example_2d.ipynb).

### Caveats

- `samples_per_epoch` applies per sub-dataset, so `len(ConcatDataset)` is the sum of the three.
  Set them unequally to bias the orientation mixture.
- Each `VolumeDataset` builds its own TensorStore cache per worker, so three of them triple the
  cache footprint — divide `cache_bytes` by three to keep it constant.
- `aug_rot` is unavailable, since it requires a cubic `patch_size`. Flip and transpose the
  collated planes yourself.
- Mixing labelled and unlabelled volumes in one config breaks collation: the unlabelled samples
  carry an empty `label` tensor, which will not stack against a real one.
- With `sampling: "sequential"`, any non-zero scalar `overlap` fails validation, since it must be
  smaller than every `patch_size` entry — use a per-axis list with `0` on the thin axis (the
  default `overlap: 0` is fine as a scalar). The grid steps one output plane's worth of extent
  along the normal, which is one stored plane only when the read count is 1; otherwise the stride
  can be shorter than the plane is thick, so planes overlap.
- On anisotropic data the three orientations are not equivalent: for a volume stored at 4×4×33 nm,
  an `xy` plane is a stored section while `xz` and `yz` planes are mostly interpolated along `z`.
- Labels use nearest-neighbor resampling, which for a one-voxel output takes the *first* plane of
  the slab rather than the middle one. So when the thin axis is resampled, the returned label
  plane comes from a different depth than the image plane — by up to about half the plane's
  thickness. The label read count is resolved from the *label* pyramid independently of the
  image, so forcing the image read count to 1 is not sufficient: check both. When the two
  pyramids differ enough, the label plane can even fall outside the image slab.

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- TensorStore >= 0.1.60
- Zarr datasets must follow the [OME-NGFF](https://ngff.openmicroscopy.org/latest/) specification
  (zarr v2 and v3 both supported)
