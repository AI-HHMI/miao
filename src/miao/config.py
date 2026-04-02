"""YAML config loading and pydantic validation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, field_validator, model_validator


class VolumeConfig(BaseModel):
    """Configuration for a single zarr volume."""

    name: str
    path: str
    image_key: str
    axes: str
    scales: list[int]
    zarr_version: Literal["zarr2", "zarr3"] = "zarr2"
    label_key: Optional[str] = None
    weight: float = 1.0
    normalize: bool = True  # auto-normalize images to [0, 1] based on source dtype
    bounding_box: Optional[list[list[int]]] = None  # [[min_0, max_0], [min_1, max_1], ...] in finest-scale voxels, storage axis order

    @field_validator("axes")
    @classmethod
    def validate_axes(cls, v: str) -> str:
        valid_chars = set("txyzc")
        if not set(v).issubset(valid_chars):
            raise ValueError(
                f"axes must only contain characters from {{t, x, y, z, c}}, got {v!r}"
            )
        if len(v) != len(set(v)):
            raise ValueError(f"axes must not contain duplicates, got {v!r}")
        return v

    @field_validator("weight")
    @classmethod
    def validate_weight(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"weight must be positive, got {v}")
        return v


class MiaoConfig(BaseModel):
    """Top-level configuration for miaio dataset."""

    volumes: list[VolumeConfig]
    output_axes: str
    patch_size: list[int]
    samples_per_epoch: int = 1000
    cache_bytes: int = 1 << 30  # 1 GB

    @field_validator("output_axes")
    @classmethod
    def validate_output_axes(cls, v: str) -> str:
        valid_chars = set("txyzc")
        if not set(v).issubset(valid_chars):
            raise ValueError(
                f"output_axes must only contain characters from {{t, x, y, z, c}}, got {v!r}"
            )
        if len(v) != len(set(v)):
            raise ValueError(f"output_axes must not contain duplicates, got {v!r}")
        return v

    @model_validator(mode="after")
    def validate_axes_compatibility(self) -> "MiaoConfig":
        for vol in self.volumes:
            if set(vol.axes) != set(self.output_axes):
                raise ValueError(
                    f"Volume {vol.name!r} axes {vol.axes!r} must be a "
                    f"permutation of output_axes {self.output_axes!r}"
                )
        return self

    @model_validator(mode="after")
    def validate_patch_size_dims(self) -> "MiaoConfig":
        if len(self.patch_size) != len(self.output_axes):
            raise ValueError(
                f"patch_size has {len(self.patch_size)} elements but "
                f"output_axes {self.output_axes!r} has {len(self.output_axes)} dimensions"
            )
        return self

    @model_validator(mode="after")
    def validate_unique_names(self) -> "MiaoConfig":
        names = [v.name for v in self.volumes]
        if len(names) != len(set(names)):
            raise ValueError("Volume names must be unique")
        return self


def load_config(path: str | Path) -> MiaoConfig:
    """Load and validate a YAML config file."""
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)
    return MiaoConfig(**data)
