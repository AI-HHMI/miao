"""TensorStore handle creation and management."""

from __future__ import annotations

from pathlib import Path

import tensorstore as ts

from miao.zarr_meta import StoreFormat


def create_context(cache_bytes: int = 1 << 30, concurrency: int = 64) -> ts.Context:
    """Create a shared tensorstore Context with cache pool."""
    return ts.Context(
        {
            "cache_pool": {"total_bytes_limit": cache_bytes},
            "file_io_concurrency": {"limit": concurrency},
        }
    )


def open_store(
    path: str | Path,
    store_format: StoreFormat,
    context: ts.Context | None = None,
    fixed_index: dict[int, int] | None = None,
) -> ts.TensorStore:
    """Open a tensorstore handle for one pyramid level.

    Args:
        path: The level's array directory (zarr, N5), or for Neuroglancer precomputed the volume
            directory joined with the level's key.
        store_format: "zarr2", "zarr3", "n5" or "precomputed".
        context: Optional tensorstore context with cache settings.
        fixed_index: Optional {dimension: index} to pin; those dimensions are removed from the
            returned handle (see `zarr_meta.fix_axes`).
    """
    path = Path(path)
    if store_format == "precomputed":
        # One `info` file describes every level of a precomputed volume; a level is its key.
        spec = {
            "driver": "neuroglancer_precomputed",
            "kvstore": {"driver": "file", "path": str(path.parent)},
            "scale_metadata": {"key": path.name},
        }
    else:
        spec = {
            "driver": {"zarr2": "zarr", "zarr3": "zarr3", "n5": "n5"}[store_format],
            "kvstore": {"driver": "file", "path": str(path)},
        }

    kwargs: dict = {"open": True}
    if context is not None:
        kwargs["context"] = context

    store = ts.open(spec, **kwargs).result()
    if store_format == "precomputed":
        # Its domain starts at the level's voxel_offset, and it always has a channel dimension;
        # miao indexes from 0 and describes a single-channel volume as x, y, z (miao.formats).
        store = store[ts.d[:].translate_to[0]]
        if store.shape[-1] == 1:
            store = store[ts.d["channel"][0]]
    if fixed_index:
        dims = sorted(fixed_index)
        store = store[ts.d[tuple(dims)][tuple(fixed_index[d] for d in dims)]]
    return store
