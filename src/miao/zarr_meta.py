"""OME-NGFF metadata reading and zarr version detection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import tensorstore as ts

ZarrVersion = Literal["zarr2", "zarr3"]


@dataclass
class ScaleMetadata:
    """Metadata for a single scale level within an OME-NGFF multiscale group."""

    path: str
    scale_factors: list[float]
    shape: list[int]
    chunks: list[int]
    dtype: np.dtype


@dataclass
class OmeMetadata:
    """Parsed OME-NGFF metadata for a multiscale group."""

    axes: list[dict]  # [{"name": "z", "type": "space", "unit": "micrometer"}, ...]
    axis_names: list[str]
    scales: dict[int, ScaleMetadata]  # level index -> metadata
    zarr_version: ZarrVersion


def detect_zarr_version(path: Path | str) -> ZarrVersion:
    """Detect whether a zarr array is v2 or v3.

    For OME-NGFF groups, descends into the first dataset to check.
    """
    path = Path(path)

    if (path / "zarr.json").exists():
        return "zarr3"
    if (path / ".zarray").exists():
        return "zarr2"

    # Check if this is an OME-NGFF group — descend into first dataset
    zattrs_path = path / ".zattrs"
    if zattrs_path.exists():
        attrs = json.loads(zattrs_path.read_text())
        if "multiscales" in attrs:
            ds_path = attrs["multiscales"][0]["datasets"][0]["path"]
            return detect_zarr_version(path / ds_path)

    raise ValueError(f"Cannot detect zarr version at {path}")


def _read_array_metadata(
    array_path: Path, zarr_version: ZarrVersion
) -> tuple[list[int], list[int], np.dtype]:
    """Read shape, chunks, dtype from a zarr array using tensorstore."""
    driver = "zarr" if zarr_version == "zarr2" else "zarr3"
    store = ts.open(
        {
            "driver": driver,
            "kvstore": {"driver": "file", "path": str(array_path)},
        },
        open=True,
    ).result()

    shape = list(store.shape)
    dtype = store.dtype.numpy_dtype

    # Get chunk shape from the spec metadata
    spec = store.spec().to_json()
    metadata = spec.get("metadata", {})
    if zarr_version == "zarr2":
        chunks = list(metadata.get("chunks", shape))
    else:
        chunk_grid = metadata.get("chunk_grid", {})
        chunks = list(chunk_grid.get("configuration", {}).get("chunk_shape", shape))

    return shape, chunks, dtype


def _read_multiscales(group_path: Path, zarr_version: ZarrVersion) -> dict:
    """Read the OME-NGFF multiscales metadata from a group directory.

    Handles both zarr2 (.zattrs with 'multiscales') and
    zarr3 (zarr.json with 'attributes.ome.multiscales' or 'attributes.multiscales').
    """
    if zarr_version == "zarr2":
        zattrs_path = group_path / ".zattrs"
        if not zattrs_path.exists():
            raise FileNotFoundError(
                f"No .zattrs found at {zattrs_path}. Is this an OME-NGFF group?"
            )
        attrs = json.loads(zattrs_path.read_text())
        if "multiscales" not in attrs:
            raise ValueError(
                f"No 'multiscales' key in .zattrs at {group_path}. "
                "Expected OME-NGFF metadata."
            )
        return attrs["multiscales"][0]

    # zarr3: metadata is in zarr.json under attributes
    zarr_json_path = group_path / "zarr.json"
    if not zarr_json_path.exists():
        raise FileNotFoundError(
            f"No zarr.json found at {zarr_json_path}. Is this an OME-NGFF group?"
        )
    data = json.loads(zarr_json_path.read_text())
    attributes = data.get("attributes", {})

    # OME-NGFF v0.5+: nested under "ome" key
    if "ome" in attributes and "multiscales" in attributes["ome"]:
        return attributes["ome"]["multiscales"][0]

    # Fallback: directly under attributes
    if "multiscales" in attributes:
        return attributes["multiscales"][0]

    raise ValueError(
        f"No multiscales metadata found in zarr.json at {group_path}. "
        "Expected OME-NGFF metadata under 'attributes.ome.multiscales' "
        "or 'attributes.multiscales'."
    )


def read_ome_metadata(
    zarr_path: Path | str,
    group_key: str,
    zarr_version: ZarrVersion,
    requested_scales: list[int] | None = None,
) -> OmeMetadata:
    """Read OME-NGFF metadata for a multiscale group within a zarr container.

    Args:
        zarr_path: Path to the zarr container root.
        group_key: Key/path within the zarr to the OME-NGFF multiscale group
                   (e.g., "raw", "labels/seg").
        zarr_version: "zarr2" or "zarr3".
        requested_scales: Which scale level indices to load metadata for.
                          None means all available scales.
    """
    zarr_path = Path(zarr_path)
    group_path = zarr_path / group_key

    multiscales = _read_multiscales(group_path, zarr_version)
    axes = multiscales.get("axes", [])
    axis_names = [ax["name"] for ax in axes]
    datasets = multiscales["datasets"]

    # Parse scale-level metadata
    if requested_scales is None:
        requested_scales = list(range(len(datasets)))

    scales: dict[int, ScaleMetadata] = {}
    for level_idx in requested_scales:
        if level_idx >= len(datasets):
            raise IndexError(
                f"Requested scale level {level_idx} but only "
                f"{len(datasets)} levels available in {group_path}"
            )
        ds = datasets[level_idx]
        ds_rel_path = ds["path"]
        array_path = group_path / ds_rel_path

        # Get scale factors from coordinateTransformations
        scale_factors = [1.0] * len(axis_names)
        for transform in ds.get("coordinateTransformations", []):
            if transform["type"] == "scale":
                scale_factors = transform["scale"]
                break

        shape, chunks, dtype = _read_array_metadata(array_path, zarr_version)

        scales[level_idx] = ScaleMetadata(
            path=ds_rel_path,
            scale_factors=scale_factors,
            shape=shape,
            chunks=chunks,
            dtype=dtype,
        )

    return OmeMetadata(
        axes=axes,
        axis_names=axis_names,
        scales=scales,
        zarr_version=zarr_version,
    )
