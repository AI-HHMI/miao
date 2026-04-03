"""Test fixtures: create temporary OME-NGFF zarr containers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr
from zarr.storage import LocalStore


def _create_ome_ngff_zarr2(
    root_path: Path,
    group_key: str,
    base_shape: tuple[int, ...],
    num_scales: int = 3,
    axes: list[dict] | None = None,
    dtype: str = "float32",
    fill_value: float | None = None,
) -> None:
    """Create an OME-NGFF compliant zarr2 multiscale group.

    Each scale level is downsampled by 2x in all dimensions.
    """
    if axes is None:
        ndim = len(base_shape)
        axis_names = ["z", "y", "x"][-ndim:]
        axes = [{"name": n, "type": "space", "unit": "micrometer"} for n in axis_names]

    store = LocalStore(str(root_path))
    root = zarr.open_group(store, mode="a", zarr_format=2)

    # Create the group for this key (handle nested keys like "labels/seg")
    parts = group_key.split("/")
    grp = root
    for part in parts:
        grp = grp.create_group(part, overwrite=False) if part not in grp else grp[part]

    datasets = []
    for level in range(num_scales):
        scale_factor = 2**level
        level_shape = tuple(s // scale_factor for s in base_shape)
        scale_factors = [float(scale_factor)] * len(base_shape)

        # Create array with deterministic data
        rng = np.random.RandomState(42 + level)
        if fill_value is not None:
            data = np.full(level_shape, fill_value, dtype=dtype)
        else:
            data = rng.rand(*level_shape).astype(dtype)

        arr = grp.create_array(
            str(level),
            shape=level_shape,
            chunks=tuple(min(32, s) for s in level_shape),
            dtype=dtype,
            overwrite=True,
        )
        arr[:] = data

        datasets.append(
            {
                "path": str(level),
                "coordinateTransformations": [
                    {"type": "scale", "scale": scale_factors}
                ],
            }
        )

    # Write OME-NGFF .zattrs manually (zarr 3.x attrs API may not write to .zattrs correctly for v2)
    zattrs_path = root_path / group_key / ".zattrs"
    existing = json.loads(zattrs_path.read_text()) if zattrs_path.exists() else {}
    existing["multiscales"] = [
        {
            "version": "0.4",
            "axes": axes,
            "datasets": datasets,
        }
    ]
    zattrs_path.write_text(json.dumps(existing))


@pytest.fixture
def zarr2_volume(tmp_path: Path) -> Path:
    """Create a zarr2 OME-NGFF volume with 'raw' image and 'labels/seg' groups."""
    zarr_path = tmp_path / "test_volume.zarr"

    _create_ome_ngff_zarr2(
        zarr_path,
        group_key="raw",
        base_shape=(64, 64, 64),
        num_scales=3,
        axes=[
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ],
        dtype="float32",
    )

    _create_ome_ngff_zarr2(
        zarr_path,
        group_key="labels/seg",
        base_shape=(64, 64, 64),
        num_scales=3,
        axes=[
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ],
        dtype="uint32",
        fill_value=1,
    )

    return zarr_path


@pytest.fixture
def zarr2_volume_xyz(tmp_path: Path) -> Path:
    """Create a zarr2 OME-NGFF volume stored in xyz order."""
    zarr_path = tmp_path / "test_volume_xyz.zarr"

    _create_ome_ngff_zarr2(
        zarr_path,
        group_key="raw",
        base_shape=(64, 64, 64),
        num_scales=2,
        axes=[
            {"name": "x", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "z", "type": "space", "unit": "micrometer"},
        ],
        dtype="float32",
    )

    return zarr_path


@pytest.fixture
def sample_config(zarr2_volume: Path) -> dict:
    """Return a sample config dict for testing."""
    return {
        "volumes": [
            {
                "name": "test_raw",
                "path": str(zarr2_volume),
                "image_key": "raw",
                "scales": [0, 1, 2],
                "label_key": "labels/seg",
            }
        ],
        "n_scales": 3,
        "output_axes": "lzyx",
        "patch_size": [8, 8, 8],
        "samples_per_epoch": 10,
    }
