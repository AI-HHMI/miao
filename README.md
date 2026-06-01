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
    img = batch["img"]        # (B, L, X, Y, Z) or (B, L, X, Y, Z, C) if channel present
    label = batch["label"]    # (B, L, X, Y, Z) or None
    bbox = batch["bbox"]      # (B, L, 2, Nd_spatial)
    meta = batch["meta"]      # dict with volume name, coordinate, resolutions, source levels
```

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
| `volumes[].bounding_box` | Optional `[[min, max], ...]` per spatial axis to restrict sampling (finest-level voxels, storage axis order) |
| `resolutions` | List of desired output resolutions, one tuple per scale. Each tuple is the output voxel size per spatial axis (physical units), in `output_axes` spatial order. The number of scales (the `l` dimension) is `len(resolutions)` |
| `output_axes` | Full tensor dim order including `l` (levels), optional `c` (channel), and spatial dims (e.g. `"lcxyz"`, `"lxyz"`) |
| `patch_size` | Voxel count per crop, in `output_axes` spatial order |
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
    meta = batch["meta"]
    # meta["grid_index"]: tuple e.g. (2, 0, 3) = position in the grid per axis
    # use grid_index to stitch patch predictions back into a full-volume output
```

In sequential mode `samples_per_epoch` and per-volume `weight` are ignored. For multiple volumes, all positions of volume 0 are yielded before volume 1.

In sequential mode the grid tiles the volume at the **first scale's** target resolution: the
stride is one output patch (minus `overlap`) worth of physical extent. This gives full
coverage of the output volume with no gaps, even when the source data is anisotropic. Patch
centers are reported in `meta["coordinate"]` (level-0 reference voxels) and the grid position
in `meta["grid_index"]`.

### Multi-scale window sampling (`sample_windows`)

By default, every scale level uses the same center location, so finer patches are centered with coarser patches. With `sample_windows: true`, each coarser level samples its patch origin uniformly at random among valid positions that cover the previous level's patch.

```yaml
sample_windows: true
```

Sampled coarse patch locations will respect the valid sampling region, including any per-volume `bounding_box`.

`resolutions` must be listed from finest to coarsest (non-decreasing voxel size per axis, e.g. `[[8,8,8], [16,16,16]]`). Reordering coarser resolutions before finer ones will raise an error.

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- TensorStore >= 0.1.60
- Zarr datasets must follow the [OME-NGFF](https://ngff.openmicroscopy.org/latest/) specification
- Supports both zarr v2 and zarr v3
