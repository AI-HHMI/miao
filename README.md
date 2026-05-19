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
    scales: [0, 1, 2]
    label_key: "labels/seg"    # optional
    weight: 0.7                # optional, default: equal

  - name: "membrane"
    path: "/data/sample_B.zarr"
    image_key: "predictions"
    zarr_version: "zarr3"
    scales: [0, 1, 2]
    weight: 0.3

n_scales: 3                     # number of scales per dataset (each dataset must have this number of scales)
output_axes: "lcxyz"            # layer, channels, X, Y, Z. Shuffle as you please!!!
patch_size: [64, 64, 64]
isotropic: false                # if true, the image (linearly) and labels (nearest) will be upsampled. 
samples_per_epoch: 1000
cache_bytes: 1073741824         # 1 GB tensorstore cache
```

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
    meta = batch["meta"]      # dict with volume name, coordinate, scale levels
```

### How it works

Each sample:

1. Randomly picks one volume based on sampling weights
2. Picks a random coordinate in that volume's finest-scale space
3. Extracts `patch_size` voxels from each requested scale level, centered at that coordinate
4. All crops have the same voxel count but cover increasing physical extents at coarser scales
5. Input axis order is auto-detected from OME-NGFF metadata — no need to specify it
6. Channel dimensions (if present in the image) are included automatically

### Config reference

| Field | Description |
|---|---|
| `volumes[].name` | Unique name for the volume |
| `volumes[].path` | Path to the OME-NGFF zarr container |
| `volumes[].image_key` | Group key within the zarr for image data |
| `volumes[].zarr_version` | `"zarr2"` or `"zarr3"` (default: `"zarr2"`) |
| `volumes[].scales` | Which multiscale levels to extract (e.g. `[0, 1, 2]`) |
| `volumes[].label_key` | Optional group key for labels in the same zarr |
| `volumes[].weight` | Sampling probability weight (default: equal across volumes) |
| `volumes[].normalize` | Auto-normalize images to [0, 1] by dtype max (default: `true`). Also see `normalize_min` / `normalize_max` to set upper and lower normalization bounds|
| `volumes[].bounding_box` | Optional `[[min, max], ...]` per spatial axis to restrict sampling (finest-scale voxels, storage axis order) |
| `output_axes` | Full tensor dim order including `l` (levels), optional `c` (channel), and spatial dims (e.g. `"lcxyz"`, `"lxyz"`) |
| `patch_size` | Voxel count per crop, in `output_axes` spatial order |
| `bbox_mode` | `"absolute"` (world coords, e.g. nm) or `"relative"` (relative to finest-level crop origin). Default: `"absolute"` |
| `samples_per_epoch` | Number of samples per epoch |
| `isotropic` | Flag to determine if images and labels should be upsampled to be isotropic |
| `cache_bytes` | TensorStore cache size in bytes (default: 1 GB) |
| `sampling` | `"random"` (default) or `"sequential"` — see below |
| `overlap` | Voxels of overlap between adjacent patches in sequential mode (default: `0`). Integer (same for all axes) or list in `output_axes` spatial order, e.g. `[16, 16, 8]` |
| `sample_windows` | If `true` (default: `false`), each coarser scale's patch origin is chosen at random such that scale's crop still covers the finer scale's crop in finest-voxel space. (Requires `n_scales` > 1 and each volume's `scales` list ordered fine-to-coarse). See below |

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

### Isotropic + sequential sampling

When `isotropic: true` and `sampling: "sequential"` are used together, the grid is built in **isotropic output space** rather than storage space. This ensures full coverage of the isotropic output volume with no gaps, even when the source data is anisotropic.

For example, with voxel sizes 9×9×20 nm (Z is ~2.2× coarser), the grid produces ~2.2× more positions along Z than it would in storage space — matching the interpolated output resolution.

```yaml
isotropic: true
sampling: "sequential"
overlap: 16
```

Two coordinate fields are available in `meta`:

| Field | Space | Use case |
|---|---|---|
| `meta["coordinate"]` | Storage voxels | Debugging, cross-referencing with on-disk data |
| `meta["isotropic_coordinate"]` | Isotropic output voxels | Stitching inference predictions into a dense volume |

`isotropic_coordinate` is present whenever `isotropic: true` (both random and sequential modes) and absent when `isotropic: false`.

### Multi-scale window sampling (`sample_windows`)

By default, every scale level uses the same center location, so finer patches are centered with coarser patches. With `sample_windows: true`, each coarser level samples its patch origin uniformly at random among valid positions that cover the previous level's patch.

```yaml
sample_windows: true
```

Sampled coarse patch locations will respect the valid sampling region, including any per-volume `bounding_box`.

Each volume's `scales` must be listed from finest to coarsest (e.g. `[0, 1, 2]` with increasing voxel size). Reordering coarser levels before finer ones will raise an error.

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- TensorStore >= 0.1.60
- Zarr datasets must follow the [OME-NGFF](https://ngff.openmicroscopy.org/latest/) specification
- Supports both zarr v2 and zarr v3
