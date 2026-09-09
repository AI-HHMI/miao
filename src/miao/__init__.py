from __future__ import annotations

from typing import TYPE_CHECKING

from miao.config import MiaoConfig, VolumeConfig, load_config, save_config

if TYPE_CHECKING:
    from miao.dataset import VolumeDataset, collate_deferred, finish_images

__all__ = [
    "MiaoConfig",
    "VolumeConfig",
    "load_config",
    "save_config",
    "VolumeDataset",
    "collate_deferred",
    "finish_images",
]

# This scheme avoids loading pytorch unless we really need it. Cuts import time from 1.5s -> 40ms.
_DEFERRED_EXPORTS = {
    "VolumeDataset",
    "collate_deferred",
    "finish_images",
}


def __getattr__(name: str):
    if name in _DEFERRED_EXPORTS:
        from miao import dataset

        val = getattr(dataset, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

