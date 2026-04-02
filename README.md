# miao

Scalable PyTorch data loaders for OME-NGFF zarr datasets, powered by TensorStore.

## Installation

```bash
pip install -e .
```

## Usage

### 1. Create a config file

```yaml
# config.yaml
volumes:
  - name: "raw"
    path: "/data/sample_A.zarr"
    image_key: "raw"
    axes: "zyx"
    zarr_version: "zarr2"      # "zarr2" or "zarr3" (default: "zarr2")
    scales: [0, 1, 2]
    label_key: "labels/seg"    # optional
    weight: 0.7                # optional, default: equal

  - name: "membrane"
    path: "/data/sample_B.zarr"
    image_key: "predictions"
    axes: "xyz"
    zarr_version: "zarr3"
    scales: [0, 1]
    weight: 0.3

output_axes: "xyz"
patch_size: [64, 64, 64]
samples_per_epoch: 1000
cache_bytes: 1073741824       # 1 GB tensorstore cache
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
    img = batch["img"]        # (B, L, X, Y, Z)
    label = batch["label"]    # (B, L, X, Y, Z) or None
    meta = batch["meta"]      # dict with volume name, coordinate, scale levels
```

### How it works

Each sample:

1. Randomly picks one volume based on sampling weights
2. Picks a random coordinate in that volume's finest-scale space
3. Extracts `patch_size` voxels from each requested scale level, centered at that coordinate
4. All crops have the same voxel count but cover increasing physical extents at coarser scales

### Config reference

| Field | Description |
|---|---|
| `volumes[].name` | Unique name for the volume |
| `volumes[].path` | Path to the OME-NGFF zarr container |
| `volumes[].image_key` | Group key within the zarr for image data |
| `volumes[].axes` | Storage axis order (e.g. `"zyx"`, `"xyz"`) |
| `volumes[].zarr_version` | `"zarr2"` or `"zarr3"` (default: `"zarr2"`) |
| `volumes[].scales` | Which multiscale levels to extract (e.g. `[0, 1, 2]`) |
| `volumes[].label_key` | Optional group key for labels in the same zarr |
| `volumes[].weight` | Sampling probability weight (default: equal across volumes) |
| `volumes[].normalize` | Auto-normalize images to [0, 1] by dtype max (default: `true`) |
| `volumes[].bounding_box` | Optional `[[min, max], ...]` per axis to restrict sampling region (finest-scale voxels, storage axis order) |
| `output_axes` | Axis order of output tensors |
| `patch_size` | Voxel count per crop, in `output_axes` order |
| `samples_per_epoch` | Number of samples per epoch |
| `cache_bytes` | TensorStore cache size in bytes (default: 1 GB) |

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- TensorStore >= 0.1.60
- Zarr datasets must follow the [OME-NGFF](https://ngff.openmicroscopy.org/latest/) specification
- Supports both zarr v2 and zarr v3
