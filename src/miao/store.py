"""TensorStore handle creation and management."""

from __future__ import annotations

from pathlib import Path

import tensorstore as ts

from miao.zarr_meta import ZarrVersion


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
    zarr_version: ZarrVersion,
    context: ts.Context | None = None,
) -> ts.TensorStore:
    """Open a tensorstore handle for a zarr array.

    Args:
        path: Path to the zarr array directory.
        zarr_version: "zarr2" or "zarr3".
        context: Optional tensorstore context with cache settings.
    """
    driver = "zarr" if zarr_version == "zarr2" else "zarr3"
    spec = {
        "driver": driver,
        "kvstore": {"driver": "file", "path": str(path)},
    }

    kwargs: dict = {"open": True}
    if context is not None:
        kwargs["context"] = context

    return ts.open(spec, **kwargs).result()
