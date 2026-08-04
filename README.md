# miao

[![PyPI version](https://img.shields.io/pypi/v/miao-io)](https://pypi.org/project/miao-io/)

Scalable PyTorch data loaders for OME-NGFF zarr datasets, powered by TensorStore.

## Installation

```bash
pip install miao-io
```

## Usage

### 1. Create a config file

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
    # resolutions: [[8, 8, 8], [16, 16, 16], [36, 36, 36]]   # optional per-volume override of the global below

resolutions: [[8, 8, 8], [16, 16, 16], [32, 32, 32]]  # one tuple per scale; output voxel size per spatial axis (output_axes spatial order)
output_axes: "lcxyz"            # layer, channels, X, Y, Z. Shuffle as you please!!!
patch_size: [64, 64, 64]
samples_per_epoch: 1000
cache_bytes: 1073741824         # 1 GB tensorstore cache
```

Scales are defined by **desired output resolution** (physical voxel size per axis, in the
same unit as the zarr's OME `coordinateTransformations` — e.g. nanometers), not by pyramid
level index. For each volume and each requested resolution, miao reads from the coarsest
pyramid level whose voxel size is still ≤ the target on every axis (preferring downsampling),
then resamples the patch to that resolution. If a target is finer than the finest stored
level, miao reads the finest level and upsamples instead. The same `resolutions` list applies
to every volume regardless of how each volume's pyramid is laid out; a volume may override it
with its own `resolutions`.

#### Random resolution sampling

Instead of a fixed `resolutions` list, you can have each sample draw its own resolutions from a
range — useful for training resolution-agnostic models. Set `resolution_sampling` (globally or
as a per-volume override) *instead of* `resolutions`:

```yaml
resolution_sampling:
  strategy: log_uniform     # only option for now (pluggable; gaussian etc. can be added)
  ranges: [[[8], [64]]]     # one or more [min, max] ranges; [[v],[w]] = isotropic (per-axis: [[a,b,c],[d,e,f]])
  n_scales: 3               # scales to draw per range — scalar (all ranges) or list (one per range)
  sort: true                # sort the drawn scales fine -> coarse
```

`ranges` is a list of `[min, max]` pairs. How each bound is written controls whether the drawn
resolution is isotropic:

- **Single-element bounds** `[[v], [w]]` are **isotropic**: one value is drawn per scale and
  broadcast to all axes (cubic voxel). E.g. `[[[1], [4]]]` — one range, cubic, 1 to 4.
- **Per-axis bounds** `[[a, b, c], [d, e, f]]` draw **each axis independently**, producing an
  anisotropic voxel. E.g. `[[[1, 1, 1], [4, 4, 4]]]` draws x, y, z separately in 1..4, so a
  sample might be `[1.8, 3.2, 2.5]` — this is *not* the same as the isotropic `[[[1], [4]]]`.

Multiple ranges are allowed, e.g. `[[[1], [2]], [[4], [8]]]` (two ranges). `n_scales` is the
number of scales to draw from each range — a scalar applies the same count to every range, or
pass a list (e.g. `[1, 1]`). The total number of scales (the `l` dimension) is the sum.

Each `__getitem__` draws those resolutions log-uniformly within each range, sorts them
fine→coarse, and the drawn values are reported both in the top-level `pixel_size` tensor and in
`meta["resolutions"]` (so a model can condition on them). The output is always `patch_size` per
scale. Constraints: exactly one of
`resolutions` / `resolution_sampling` may be set; `resolution_sampling` is incompatible with
`sampling: "sequential"`; and `sample_windows` requires every range to be isotropic.

### 2. Create a dataset

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
    img = batch["img"]                # (B, L, X, Y, Z) or (B, L, X, Y, Z, C) if channel present
    label = batch["label"]            # (B, L, X, Y, Z) or None
    bbox = batch["bbox"]              # (B, L, 2, Nd_spatial)
    pixel_size = batch["pixel_size"]  # (B, L, Nd_spatial) — output voxel size per level
    meta = batch["meta"]              # dict with volume name, coordinate, resolutions, source levels
```

#### `pixel_size`

Every sample returns a `pixel_size` tensor of shape `(L, Nd_spatial)` (batched to
`(B, L, Nd_spatial)`) giving the **physical output voxel size per scale level**, in the same
physical unit as the zarr's OME `coordinateTransformations` and in `output_axes` spatial order
(matching `bbox`). It is a first-class tensor at the top level of the batch (not nested under
`meta`), so the default PyTorch collate stacks it cleanly — convenient for feeding a
scale-conditioned model.

It reports the resolutions actually used for that sample, so in **random resolution sampling**
mode it reflects the freshly-drawn values on every `__getitem__` rather than a fixed list. The
same values are also mirrored in `meta["resolutions"]` (also in `output_axes` spatial order).

All spatial fields in `meta` are reported in `output_axes` spatial order: `coordinate`
(level-0 reference voxels), `resolutions`, and, in sequential mode, `grid_index`.

### 3. Create a 2D image dataset

A 2D dataset is the degenerate case of a 3D one: request a patch that is **one voxel thick**
along one spatial axis and the crop is an axis-aligned plane. Do that once per axis,
concatenate the three datasets, and every sample is a plane drawn from one of the three
orientations.

`patch_size` is in `output_axes` spatial order, so which axis goes thin depends on where the
`1` sits. With `output_axes: "lcxyz"` (spatial order `xyz`):

| `patch_size` | thin axis | plane | `img` shape |
|---|---|---|---|
| `[1, P, P]` | `x` | `yz` | `(L, C, 1, P, P)` |
| `[P, 1, P]` | `y` | `xz` | `(L, C, P, 1, P)` |
| `[P, P, 1]` | `z` | `xy` | `(L, C, P, P, 1)` |

Everything else keeps working — multi-scale reads, `resolution_sampling`, labels,
normalization and weighted volume sampling all behave exactly as they do in 3D.

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

The three sub-datasets put the singleton on different axes, so a shuffled batch mixes shapes
and the default collate fails (`stack expects each tensor to be equal size`). Squeeze the
singleton spatial axis in a `collate_fn` and every sample becomes `(L, C, P, P)`:

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
physical thickness all survive collation. With a single scale, `batch["img"][:, 0]` gives
the `(B, C, H, W)` a 2D model expects. A runnable version of all of this is in
[`examples/example_2d.ipynb`](examples/example_2d.ipynb).

#### Plane thickness

A one-voxel-thick output is still read as a whole number of stored voxels, so a plane spans
a physical extent of `ceil(target / level_voxel) × level_voxel` along its normal — rounded
up, and *not* the requested `pixel_size`. Asking for a 16 nm plane from a volume stored at
33 nm along that axis gives a 33 nm-thick plane while `pixel_size` still reports 16. Read
the true extent from `bbox`, which reports the physical footprint actually covered.
`level_voxel` here is the voxel size of the pyramid level actually chosen, not of level 0.

When more than one stored voxel is read along the normal, they are resampled to one by the
same trilinear resize used for the in-plane axes. That is a point sample at the middle of
the slab, not an average through it: an odd read count returns exactly the central stored
plane, an even one the mean of the two central planes, and the remaining voxels read are
discarded. So the value's effective support is one or two stored planes at the centre of the
slab, narrower than the extent `bbox` reports.

To read a single stored plane with no interpolation along the normal, request a thin-axis
resolution no coarser than the stored voxel size on that axis, which forces the read count
to 1. Resolutions are per-axis, so this is independent of the in-plane resolution — e.g.
`[4, 16, 16]` against a volume stored at 4 nm in `x` reads one `x` plane and resamples only
in-plane. Pinning the thin axis this way also rules out coarser pyramid levels, so the
in-plane read grows.

> **Labels:** images are resampled trilinearly, but labels use nearest-neighbour, which for
> a one-voxel output takes the *first* plane of the slab rather than the middle one. So when
> the thin axis is resampled, the returned label plane comes from a different depth than the
> image plane — by up to about half the plane's thickness (on 4×4×33 nm data, 4–8 nm inside
> a 16 nm plane and 12–16 nm inside a 32 nm one, depending on where the sampled centre
> falls). The label read count is resolved from the *label* pyramid independently of the
> image, so forcing the image read count to 1 is not sufficient: check both. When the two
> pyramids differ enough, the label plane can even fall outside the image slab.

#### Notes

- `samples_per_epoch` applies per sub-dataset, so `len(ConcatDataset)` is the sum of the
  three. Set them unequally to bias the orientation mixture.
- Each `VolumeDataset` builds its own TensorStore cache per worker, so three of them triple
  the cache footprint — divide `cache_bytes` by three to keep it constant.
- `aug_rot` is rejected, because it requires a cubic `patch_size` across `x`, `y`, `z`. Flip
  and transpose the collated planes yourself.
- Mixing labelled and unlabelled volumes in one config breaks collation either way: the
  unlabelled samples carry an empty `label` tensor, which will not stack against a real one.
- With `sampling: "sequential"`, any non-zero scalar `overlap` fails validation, since it
  must be smaller than every `patch_size` entry — use a per-axis list with `0` on the thin
  axis. The default `overlap: 0` is fine as a scalar. The grid steps one output plane's
  worth of extent along the normal, which is one stored plane only when the read count is 1;
  otherwise the stride can be shorter than the plane is thick, so planes overlap.
- On anisotropic data the three orientations are not equivalent: for a volume stored at
  4×4×33 nm, an `xy` plane is a stored section while `xz` and `yz` planes are mostly
  interpolation along `z`.

### How it works

Each sample:

1. Randomly picks one volume based on sampling weights
2. Picks a random coordinate in that volume's finest-scale (level-0) space
3. For each requested resolution, reads from the chosen pyramid level (centered at that
   coordinate) however many voxels are needed to yield `patch_size` voxels at the target
   resolution after resampling — i.e. `ceil(patch_size × target_resolution / level_voxel_size)`
   voxels — then resamples that read to `patch_size`
4. All crops have the same voxel count (`patch_size`) but cover increasing physical extents at coarser resolutions
5. Input axis order is auto-detected from OME-NGFF metadata — no need to specify it
6. Channel dimensions (if present in the image) are included automatically

### Config reference

| Field | Description |
|---|---|
| `volumes[].name` | Unique name for the volume |
| `volumes[].path` | Path to the OME-NGFF zarr container |
| `volumes[].image_key` | Group key within the zarr for image data |
| `volumes[].zarr_version` | `"zarr2"` or `"zarr3"` (default: `"zarr2"`) |
| `volumes[].resolutions` | Optional per-volume override of the global `resolutions` (same format) |
| `volumes[].label_key` | Optional group key for labels in the same zarr |
| `volumes[].weight` | Sampling probability weight (default: equal across volumes) |
| `volumes[].normalize` | Auto-normalize images to [0, 1] by dtype max (default: `true`). Also see `normalize_min` / `normalize_max` to set upper and lower normalization bounds|
| `volumes[].patch_normalize` | Standardize each returned sample to zero mean / unit standard deviation, applied after `normalize` (default: `false`). With multiple scales, the statistics are taken from the coarsest-resolution (largest physical extent) crop and applied to every scale |
| `volumes[].bounding_box` | Optional `[[min, max], ...]` per spatial axis (finest-level voxels, `output_axes` spatial order — same order as `patch_size`). Every window's read extent — at every scale, including coarser `sample_windows` patches — is kept strictly inside the box. Must be at least as large as the coarsest window, or dataset construction raises. |
| `volumes[].aug_rot` | If `true` (default: `false`), apply a random axis-aligned rotation/flip to each returned sample — one of the 48 symmetries of the cube. Requires isotropic output. See "Axis-aligned rotation/flip augmentation (`aug_rot`)" below. |
| `resolutions` | List of desired output resolutions, one tuple per scale. Each tuple is the output voxel size per spatial axis (physical units), in `output_axes` spatial order. The number of scales (the `l` dimension) is `len(resolutions)`. Mutually exclusive with `resolution_sampling` |
| `resolution_sampling` | Draw resolutions randomly per sample instead of a fixed list: `{strategy, ranges, n_scales, sort}`. See "Random resolution sampling" above. Mutually exclusive with `resolutions` |
| `output_axes` | Full tensor dim order including `l` (levels), optional `c` (channel), and spatial dims (e.g. `"lcxyz"`, `"lxyz"`) |
| `patch_size` | Voxel count per crop, in `output_axes` spatial order. An entry of `1` makes the crop a plane — see "3. Create a 2D image dataset" |
| `bbox_mode` | `"absolute"` (world coords, e.g. nm) or `"relative"` (relative to finest-level crop origin). Default: `"absolute"` |
| `samples_per_epoch` | Number of samples per epoch |
| `cache_bytes` | TensorStore cache size in bytes (default: 1 GB) |
| `sampling` | `"random"` (default) or `"sequential"` — see below |
| `overlap` | Voxels of overlap between adjacent patches in sequential mode (default: `0`). Integer (same for all axes) or list in `output_axes` spatial order, e.g. `[16, 16, 8]` |
| `sample_windows` | If `true` (default: `false`), each coarser scale's patch origin is chosen at random such that scale's crop still covers the finer scale's crop in reference-voxel space. (Requires more than one scale and `resolutions` ordered fine-to-coarse). See below |

> **Isotropic output:** there is no separate `isotropic` flag — request equal-valued
> resolution tuples (e.g. `[8, 8, 8]`) and miao downsamples/upsamples each axis to that
> common voxel size automatically.

Input axes are auto-detected from OME-NGFF metadata (`multiscales.axes`).

### Sequential sampling (inference / evaluation)

Set `sampling: "sequential"` to iterate over the entire volume in a deterministic grid instead of random sampling. Useful for dense inference and evaluation.

```yaml
sampling: "sequential"
overlap: 16              # or per-axis list e.g. [16, 16, 8]
```

```python
dataset = VolumeDataset(config)
# len(dataset) = total grid positions across all volumes

loader = DataLoader(dataset, batch_size=4, shuffle=False)  # shuffle=False required

for batch in loader:
    img  = batch["img"]
    pixel_size = batch["pixel_size"]   # (B, L, Nd_spatial) output voxel size per level
    meta = batch["meta"]
    # meta["grid_index"]: tuple e.g. (2, 0, 3) = position in the grid per axis
    #   (output_axes spatial order, matching the img/label tensor axes)
    # use grid_index to stitch patch predictions back into a full-volume output
```

In sequential mode `samples_per_epoch` and per-volume `weight` are ignored. For multiple volumes, all positions of volume 0 are yielded before volume 1.

In sequential mode the grid tiles the volume at the **first scale's** target resolution: the
stride is one output patch (minus `overlap`) worth of physical extent. This gives full
coverage of the output volume with no gaps, even when the source data is anisotropic. Patch
centers are reported in `meta["coordinate"]` (level-0 reference voxels) and the grid position
in `meta["grid_index"]`, both in `output_axes` spatial order.

### Multi-scale window sampling (`sample_windows`)

By default, every scale level uses the same center location, so finer patches are centered with coarser patches. With `sample_windows: true`, each coarser level samples its patch origin uniformly at random among valid positions that cover the previous level's patch.

```yaml
sample_windows: true
```

Sampled coarse patch locations stay strictly within any per-volume `bounding_box` (the box bounds every scale's read extent, not just the patch center).

`resolutions` must be listed from finest to coarsest (non-decreasing voxel size per axis, e.g. `[[8,8,8], [16,16,16]]`). Reordering coarser resolutions before finer ones will raise an error.

### Axis-aligned rotation/flip augmentation (`aug_rot`)

Set `aug_rot: true` on a volume to augment each returned sample with a random axis-aligned rotation/flip:

```yaml
volumes:
  - name: "raw"
    path: "/data/sample_A.zarr"
    image_key: "raw"
    label_key: "labels/seg"
    aug_rot: true            # optional, default: false
```

Each `__getitem__` draws **one** transform uniformly from the 48 symmetries of the cube — the full set of axis-aligned 3D rotations and reflections (3! axis permutations × 2³ per-axis flips = 48). This is the largest augmentation set achievable without interpolation, since every transform maps voxels onto voxels exactly.

The same transform is applied to:

- **the image and its labels** together (so they stay aligned), and
- **every requested scale** in the multi-scale stack (so the nested scales stay mutually consistent).

**Isotropic output is required.** An axis permutation mixes the spatial axes, which is only meaningful when the output voxels are cubes. Two conditions are checked:

- `patch_size` must be equal across `x`, `y`, `z` (validated when the config is loaded).
- The output resolution being read must be equal on all spatial axes (validated per sample, since `resolution_sampling` can draw a different resolution each call). With `resolution_sampling`, use isotropic (single-value) bounds.

It is fine for the **underlying stored data** to be anisotropic — e.g. `z` coarser than `x`/`y` and upsampled — as long as the requested **output** resolution is isotropic.

> **Note:** `aug_rot` reorients the returned `img` and `label` tensors only. The `bbox` and `meta.coordinate` still describe the original (pre-augmentation) read location in the source volume; `pixel_size` is unaffected because it is isotropic by construction.

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- TensorStore >= 0.1.60
- Zarr datasets must follow the [OME-NGFF](https://ngff.openmicroscopy.org/latest/) specification
- Supports both zarr v2 and zarr v3
