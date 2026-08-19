# miao

[![PyPI version](https://badge.fury.io/py/miao-io.svg)](https://pypi.org/project/miao-io/)

Scalable PyTorch data loaders for OME-NGFF zarr datasets, powered by TensorStore.

## Installation

```bash
pip install miao-io
```

## Quick start

```yaml
# config.yaml
volumes:
  - name: sample_a
    path: /data/sample_A.zarr
    image_key: raw
    label_key: labels/seg      # optional
    zarr_version: zarr2        # or zarr3
    weight: 0.5                # optional

  - name: sample_b
    path: /data/sample_B.zarr
    image_key: predictions
    zarr_version: zarr3
    weight: 0.5

resolutions: [[8, 8, 8], [16, 16, 16], [32, 32, 32]]   # output voxel size per scale
output_axes: lcxyz
patch_size: [64, 64, 64]
samples_per_epoch: 1000
```

```python
from torch.utils.data import DataLoader
from miao import VolumeDataset, load_config

dataset = VolumeDataset(load_config("config.yaml"))
loader = DataLoader(dataset, batch_size=4, shuffle=True)

for batch in loader:
    img, label = batch["img"], batch["label"]
```

Runnable notebooks live in [`examples/`](examples/).

## Batch format

Tensors follow `output_axes` with the batch dim prepended, so `output_axes: "lczyx"` yields `img`
as `(B, L, C, Z, Y, X)`. `L` is the number of scales, `Nd` the number of spatial axes.

| Key | Shape | Description |
|---|---|---|
| `img` | `output_axes` order | Image patch per scale |
| `label` | `output_axes` minus `c` | Label patch per scale; empty tensor without a `label_key` |
| `bbox` | `(B, L, 2, Nd)` | Patch extent in physical units, world or center-relative (see `bbox_mode`) |
| `pixel_size` | `(B, L, Nd)` | Effective output voxel size, as actually used for that sample |
| `meta` | dict | `volume`, `coordinate`, `resolutions`, `source_levels`, plus `grid_index` in sequential mode |

Spatial entries of `bbox`, `pixel_size` and every `meta` field are in `output_axes` spatial
order.

## Resolutions

Scales are addressed by **desired output resolution** — physical voxel size per axis, in the zarr's
OME `coordinateTransformations` unit (e.g. nanometers). One `resolutions` list therefore serves
volumes with wildly different pyramid layouts, and any volume can override it with its own.

Per sample, miao:

1. draws a volume with probability ∝ `weight * size ** size_weighting_exponent`, then a random
   coordinate in its finest-scale (level-0) space;
2. for each requested resolution, picks the coarsest pyramid level whose effective voxel size is
   still ≤ the target on every axis, preferring downsampling — or the finest stored level,
   upsampled, if the target is finer than anything on disk;
3. reads `ceil(patch_size × target_resolution / effective_level_voxel_size)` voxels centered on
   that coordinate and resamples them to `patch_size` — trilinear for images, nearest for labels.

Every crop thus holds the same voxel count while covering a wider physical extent at coarser scales.

### Effective voxel sizes

Every resolution here is an **effective** voxel size: the size of one voxel in the specimen, not in
the microscope. With `exp_factor: F` it is the stored `coordinateTransformations` scale divided by
`F` — a 4× expanded sample imaged at 200 nm holds 50 nm of biology per voxel. `exp_factor` defaults
to `1.0`, so stored and effective sizes are identical for unexpanded data.

Set `exp_factor` only for stores whose OME metadata lacks the correction. A multiscale-level (outer)
`coordinateTransformations` scale is applied automatically, and would otherwise count twice.

Level selection, read extents, `pixel_size` and `meta["resolutions"]` are all in effective units,
which is what lets one `resolutions` list mix expanded and unexpanded volumes. `bounding_box` and
`meta["coordinate"]` are level-0 voxel *indices*, so `exp_factor` does not affect them.

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

Ranges stack: `[[[1], [2]], [[4], [8]]]` with `n_scales: [1, 1]` gives two scales, `L` being the sum
across ranges. Draws are log-uniform and redrawn every `__getitem__`.

Constraints: exactly one of `resolutions` / `resolution_sampling` per config and per volume
override; `resolution_sampling` rules out `sampling: sequential`; `sample_windows` demands isotropic
ranges.

## Sampling

### Random (default)

Draws `samples_per_epoch` patches per epoch. Each sample picks a volume first — uniformly by
default, so a 512³ ground-truth crop is chosen as often as a whole brain, biasing training toward
small volumes. `size_weighting_exponent` tempers the draw by how much data each volume holds:

```yaml
size_weighting_exponent: 0.3   # p ∝ weight * size ** size_weighting_exponent
```

| Value | Effect |
|---|---|
| `0.0` (default) | size ignored; probabilities come from `weight` alone |
| `0.3` – `0.5` | tempered; the usual choice for a corpus spanning orders of magnitude |
| `1.0` | exactly proportional to reachable content; the largest volumes dominate |

`size` is the number of distinct stored voxels the sampler can reach, measured on the finest pyramid
level any configured scale reads. It is not the raw array size: it accounts for the patch window,
`bounding_box`, and the level actually used, so a crop admitting only one patch center scores one
patch window. It is reported per volume in the dataset summary alongside the resulting probability.
`weight` multiplies the size term, so per-volume overrides keep working.

### Sequential (inference / evaluation)

Walks the whole volume on a deterministic grid instead.

```yaml
sampling: sequential
overlap: 16              # or per-axis list, e.g. [16, 16, 8]
```

```python
dataset = VolumeDataset(config)                            # len() = grid positions, all volumes
loader = DataLoader(dataset, batch_size=4, shuffle=False)   # walk the grid in order

for batch in loader:
    # Use the grid position to stitch predictions into a full volume. Default collate gives one
    # (B,) tensor per spatial axis, in output_axes order; un-batched, dataset[i] gives a plain
    # tuple like (2, 0, 3).
    grid_index = batch["meta"]["grid_index"]
```

The grid tiles the volume at the **first scale's** target resolution, striding one output patch
minus `overlap` worth of physical extent. Coverage is gap-free even on anisotropic source data.
`samples_per_epoch`, `weight` and `size_weighting_exponent` are ignored here, and volumes are
exhausted in order — the grid already visits each in proportion to its sampleable extent.

### Multi-scale windows

By default all scales share one center, nesting finer patches inside coarser ones. With
`sample_windows: true` each coarser level instead drops its patch origin uniformly at random among
the positions that still cover the previous level's patch, staying strictly inside any
`bounding_box`. `resolutions` must then run fine to coarse (non-decreasing voxel size per axis,
e.g. `[[8,8,8], [16,16,16]]`); a coarser resolution first raises.

## Augmentation

Two composable modules, both used from one `augment_fn`:

- [`miao.augment`](src/miao/augment.py) — stochastic augmentations. Pure functions: deterministic
  given `rng`, tensors in, tensors out, inputs never mutated.
- [`miao.labels`](src/miao/labels.py) — deterministic label-target transforms.

### Setup

`augment_fn(sample) -> sample` is called once per sample. Pass it to the dataset:

```python
import numpy as np
from miao import VolumeDataset, load_config
from miao.augment import intensity_jitter, rot90isocube

def augment(sample):
    rng = np.random # PyTorch reseeds this per DataLoader worker
    img, lab = rot90isocube(rng, sample["img"], sample["label"], pixel_size=sample["pixel_size"])
    return {**sample, "img": intensity_jitter(rng, img), "label": lab}

dataset = VolumeDataset(load_config("config.yaml"), augment_fn=augment)
```

Or configure it. A bare string names a ready `augment_fn`; a `{factory, kwargs}` mapping names a
callable that *returns* one, keeping composition in code and the swept parameters diffable across
configs — see [`examples/config_augment_fn.yaml`](examples/config_augment_fn.yaml):

```yaml
augment_fn: myproject.augs.augment_sample     # a ready callable
```

```yaml
augment_fn:                                   # or a factory plus its parameters
  factory: miao.augment_std.em_default
  kwargs:
    scale: [0.8, 1.2]
```

The config field and the constructor argument are mutually exclusive. A directly passed callable is
pickled to workers, so it must be a module-level function, `functools.partial`, or class instance;
a factory instead runs once *inside* each worker and may return anything, closures included.
`miao.augment_std.em_default` (cube rotation plus intensity jitter) rotates by default — set
`rotate: false` for anisotropic data. For volumes without labels, skip the empty label sentinel
(`sample["label"].numel() == 0`) instead of transforming it.

### Functions

Geometric ops take `*tensors` and apply one draw to all of them, so images and labels stay aligned.
Photometric ops take the image alone. None is gated internally — apply your own probability in the
closure.

| Function | Applies to | Effect |
|---|---|---|
| `rot90isocube(rng, *tensors, spatial_dims, pixel_size)` | image + labels | One of the 48 cube symmetries. Requires a cubic patch and isotropic voxels |
| `rot90inplane(rng, *tensors, spatial_dims, fixed_axis, pixel_size)` | image + labels | The 16 symmetries that never exchange `fixed_axis`, for anisotropic voxels. Also requires a cubic patch |
| `shift_sections(rng, *tensors, prob=0.05, magnitude=10)` | image + labels | Offsets whole sections within their own plane, modeling imperfect section alignment |
| `drop_sections(rng, img, prob=0.05)` | image **only** | Blanks whole sections, modeling lost or unusable ones |
| `intensity_jitter(rng, img, scale=(0.9, 1.1), shift=(-0.1, 0.1))` | image only | `img * U(*scale) + U(*shift)` |
| `additive_noise(rng, img, scale=0.1)` | image only | Gaussian noise whose std is drawn per sample from `[0, scale]` |
| `percentile_normalize(img, lower=1.0, upper=99.0, clamp=True)` | image only | Rescales intensities to `[0, 1]` by the given percentiles. Deterministic, hence no `rng` |
| `draw_rot90(rng)` / `apply_rot90(t, perm, flips, spatial_dims)` | — | Split the draw from the application, to replay one transform across tensors |
| `spatial_dims_for(axes)` | — | Derive `spatial_dims` from a layout string |

Both rotations permute spatial axes, so both require a **cubic patch** — a z-thin patch like
`[8, 32, 32]` raises regardless of which one you use. They differ only in the voxel-size condition:
`rot90isocube` mixes all three axes and so needs isotropic voxels, while `rot90inplane` needs only
the two *free* axes to share one, which is what makes it the option for anisotropic data. Forward
`sample["pixel_size"]` to have the relevant condition asserted per sample, which matters under
`resolution_sampling`.
`drop_sections` is deliberately image-only: an object still passes through a lost section, so
blanking its label would teach the model that a gap is a boundary. Photometric defaults assume
normalized input (~[0, 1]).

Order geometric first, given both tensors, and photometric last, given only the image:

```python
img, lab = rot90inplane(rng, img, lab, fixed_axis=0, pixel_size=sample["pixel_size"])
img, lab = shift_sections(rng, img, lab, prob=0.05, magnitude=10)
img = drop_sections(rng, img, prob=0.05)
img = additive_noise(rng, intensity_jitter(rng, img), scale=0.1)
```

Rotation reorients `img` and `label` only: `bbox` and `meta["coordinate"]` still describe the
original read location. `pixel_size` stays valid either way, since both ops only exchange axes that
share a voxel size.

### Axis layouts (`spatial_dims`)

Every geometric op takes `spatial_dims`: the positions of the spatial axes counted from the end,
listed **in the order they appear in the layout** — `"lcxyz"` gives x, y, z and `"lczyx"` gives
z, y, x, both `(-3, -2, -1)`. The order matters because everything indexes these slots
positionally: `apply_rot90` permutes them, `fixed_axis` names one, and `pixel_size` follows the
same order.

The default `(-3, -2, -1)` is right when `output_axes` puts the spatial axes last (`"lczyx"`,
`"lcxyz"`) and wrong otherwise: `"lzyxc"` pushes the image's to `(-4, -3, -2)`, while the label,
carrying no channel axis, keeps `(-3, -2, -1)`. Pass one tuple per tensor when they differ; a
single tuple still means "the same axes in every tensor". Derive them with `spatial_dims_for`
rather than hard-coding:

```python
from miao.augment import rot90inplane, rot90isocube, spatial_dims_for

img, lab = rot90isocube(
    rng, img, lab,
    spatial_dims=[spatial_dims_for("lzyxc"), spatial_dims_for("lzyx")],   # image, label
)
```

`fixed_axis` indexes the same slot ordering, so it too depends on the layout — 0 for `"lczyx"`,
2 for `"lcxyz"`:

```python
# Serial-section EM: z is coarser, so it must not be permuted into y or x.
axes = "lczyx"                                   # the dataset's output_axes
spatial = [a for a in axes if a in "zyx"]        # -> ['z', 'y', 'x']
img, lab = rot90inplane(
    rng, img, lab,
    spatial_dims=[spatial_dims_for(axes), spatial_dims_for(axes.replace("c", ""))],
    fixed_axis=spatial.index("z"),               # 0 here, 2 for "lcxyz"
    pixel_size=sample["pixel_size"],
)
```

When one call cannot cover everything — per-level tensors that are not stacked yet, say — draw once
and apply the same transform per tensor. `apply_rot90` is deterministic given `(perm, flips)`, so a
recorded draw can be replayed later:

```python
from miao.augment import apply_rot90, draw_rot90

perm, flips = draw_rot90(rng)
levels = [apply_rot90(lvl, perm, flips) for lvl in per_level_tensors]  # same rotation across L
img = apply_rot90(img, perm, flips, spatial_dims=(-4, -3, -2))         # L Z Y X C layout
```

### Randomness

Draw from the `np.random` module and pass it as `rng`. This is correct with no further work:
PyTorch reseeds numpy's global RNG in every DataLoader worker (from the loader's base seed and the
worker id), so streams decorrelate across workers and epochs, and the whole pipeline stays
reproducible by seeding the loader —
`DataLoader(..., generator=torch.Generator().manual_seed(0))`.

For a private `np.random.Generator`, where you create it decides whether workers decorrelate:

| Where it is created | Result |
|---|---|
| In a configured factory, seeded from the global: `default_rng(np.random.randint(2**31))` | Independent per worker, reproducible via the loader seed. The factory runs after the per-worker reseeding |
| In a factory with a fixed seed: `default_rng(0)` | Every worker recreates the same stream — correlated augmentations |
| Captured in a directly passed callable | The Generator's *state* is cloned to every worker — correlated, and only fixable with a `worker_init_fn` |

### Label targets

| Function | Effect |
|---|---|
| `erode_labels(labels, steps=1, only_xy=False)` | Replaces instance borders with background: a `steps`-voxel gap on each side of a touch. `only_xy` compares in-plane neighbors only, for anisotropic data |
| `affinities(labels, neighborhood=NEAREST_NEIGHBORHOOD)` | One foreground affinity map per neighbor offset, stacked `C Z Y X`; 1.0 where a voxel and its offset neighbor share a non-zero label |

Both take a single `Z Y X` volume, so loop over the scale dim. Compose them after the stochastic
augmentations — affinity channels are direction-dependent — with `erode_labels` before `affinities`.

```python
import torch
from miao.labels import affinities, erode_labels

def augment(sample):
    # ... geometric / intensity augmentations first ...
    lab = torch.stack([erode_labels(l) for l in sample["label"]])   # per scale; output_axes lzyx
    return {**sample, "label": lab,
            "affinities": torch.stack([affinities(l) for l in lab])}
```

## Configuration reference

Input axes come from OME-NGFF metadata (`multiscales.axes`) and never need specifying; channel
dimensions are picked up automatically when present. Unknown keys are rejected, so typos fail loudly.

### Dataset fields

| Field | Default | Description |
|---|---|---|
| `volumes` | — | List of volumes; see [below](#per-volume-fields) |
| `resolutions` | — | Output voxel size per scale, in `output_axes` spatial order; `len()` is `L`. Mutually exclusive with `resolution_sampling` |
| `resolution_sampling` | — | Draw resolutions per sample instead: `{strategy, ranges, n_scales, sort}` — see [Random resolution sampling](#random-resolution-sampling) |
| `output_axes` | — | Tensor dim order: `l` (scales, required), optional `c` (channel), spatial `x`/`y`/`z`/`t` — e.g. `"lcxyz"` |
| `patch_size` | — | Voxel count per crop, in `output_axes` spatial order |
| `samples_per_epoch` | `1000` | Samples per epoch; ignored in sequential mode |
| `sampling` | `"random"` | `"random"` or `"sequential"` — see [Sampling](#sampling) |
| `size_weighting_exponent` | `0.0` | Weight volumes by how much data they hold — see [Random](#random-default) |
| `overlap` | `0` | Patch overlap in voxels, sequential mode only. Integer or per-axis list; each entry must be `< patch_size` |
| `sample_windows` | `false` | Randomize each coarser scale's patch origin — see [Multi-scale windows](#multi-scale-windows) |
| `chunk_aligned` | `false` | Keep random patches inside a single stored chunk, per axis where the chunk is at least patch-sized |
| `bbox_mode` | `"absolute"` | `bbox` in world coords, or `"relative"` to the finest-level crop center |
| `image_dtype` | `"float32"` | Output image dtype: `"float32"`, `"bfloat16"` or `"float16"` |
| `cache_bytes` | `1 << 30` | TensorStore cache size in bytes (1 GB) |
| `file_io_concurrency` | `64` | Concurrent TensorStore file reads |
| `augment_fn` | — | Dotted path to an `augment_fn`, or `{factory, kwargs}` — see [Wiring](#wiring) |

### Per-volume fields

| Field | Default | Description |
|---|---|---|
| `name` | — | Unique name for the volume |
| `path` | — | Path to the OME-NGFF zarr container |
| `image_key` | — | Group key within the zarr for image data |
| `label_key` | `None` | Optional group key for labels in the same zarr |
| `zarr_version` | `"zarr2"` | `"zarr2"` or `"zarr3"` |
| `weight` | `1.0` | Sampling weight; multiplies the size term, so `p ∝ weight * size ** size_weighting_exponent` |
| `exp_factor` | `1.0` | Divides the zarr's metadata voxel size to give the effective one, before level selection — see [Effective voxel sizes](#effective-voxel-sizes) |
| `resolutions` / `resolution_sampling` | — | Per-volume override of the global setting; set at most one |
| `normalize` | `true` | Scale images to [0, 1]: integer dtypes by their max, float dtypes left unchanged |
| `normalize_min` / `normalize_max` | `None` | Set both to clip to that range and map it linearly to [0, 1] |
| `patch_normalize` | `false` | Standardize each sample to zero mean / unit variance after `normalize`. Multi-scale: statistics come from the coarsest crop and apply to all scales |
| `bounding_box` | `None` | `[[min, max], ...]` per spatial axis in level-0 voxels. Confines every read extent at every scale, `sample_windows` patches included — not merely the patch center. Must be at least as large as the coarsest window |

## Recipe: 2D datasets

2D models are often strong baselines in 3D domains. Build one from three degenerate
`VolumeDataset`s — `patch_size` `(1,P,P)`, `(P,1,P)`, `(P,P,1)` — glued with `ConcatDataset`. Each
puts its singleton on a different axis, so a shuffled batch mixes shapes and default collate chokes;
squeeze the thin axis away instead. Runnable:
[`examples/example_2d.ipynb`](examples/example_2d.ipynb).

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

dataset = ConcatDataset([VolumeDataset(plane_config(base, ax, P)) for ax in spatial_order])

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
    bbox = batch["bbox"]              # (B, L, 2, Nd) — still 3D
    plane = batch["plane"]            # "yz" / "xz" / "xy", one per sample
```

`bbox` and `pixel_size` keep their full spatial rank, so a plane's position, orientation and
physical thickness all survive collation. Index a single scale out with `batch["img"][:, 0]` for a
`(B, C, H, W)` tensor.

**Caveats**

- `samples_per_epoch` counts per sub-dataset, so `len(ConcatDataset)` is the sum of three. Set them
  unequally to bias the orientation mix.
- Each `VolumeDataset` opens its own TensorStore cache per worker — divide `cache_bytes` by three.
- Both rotations are out, since they want a cubic `patch_size`. Flip and transpose the planes
  yourself.
- Mixing labelled and unlabelled volumes breaks collation: an empty `label` will not stack against a
  real one.
- Under `sampling: sequential`, `overlap` must be smaller than *every* `patch_size` entry, so pass a
  per-axis list with `0` on the thin axis.
- The three orientations are not equivalent on anisotropic data: at 4×4×33 nm an `xy` plane is a
  stored section, while `xz` and `yz` are mostly interpolation along `z`.
- Nearest-neighbor label resampling takes the *first* plane of a slab, so on a resampled thin axis
  the label plane can come from a different depth than the image plane, off by up to half a
  thickness. Read counts resolve from the *label* pyramid independently, so check both.

</details>

## Requirements

- Python ≥ 3.10, PyTorch ≥ 2.0, TensorStore ≥ 0.1.60
- Zarr datasets following the [OME-NGFF](https://ngff.openmicroscopy.org/latest/) spec (v2 or v3)
