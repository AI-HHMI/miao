"""YAML config loading and pydantic validation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Union

import yaml
from pydantic import BaseModel, field_validator, model_validator


class VolumeConfig(BaseModel):
    """Configuration for a single zarr volume."""

    name: str
    path: str
    image_key: str
    # Optional per-volume override of the global MiaoConfig.resolutions. Each inner list is the
    # desired output voxel size per spatial axis (physical units, same unit as the zarr's OME
    # coordinateTransformations scale), in output_axes spatial order — like patch_size.
    resolutions: Optional[list[list[float]]] = None
    zarr_version: Literal["zarr2", "zarr3"] = "zarr2"
    label_key: Optional[str] = None
    weight: float = 1.0
    normalize: bool = True  # scale images to [0, 1]; see normalize_min / normalize_max
    # If both set: clip to [normalize_min, normalize_max] then linear map to [0, 1].
    # If both omitted: integer images use [0, dtype_max]; float images are unchanged.
    normalize_min: Optional[float] = None
    normalize_max: Optional[float] = None
    bounding_box: Optional[list[list[int]]] = None  # [[min_0, max_0], [min_1, max_1], ...] in finest-scale voxels, storage axis order

    @field_validator("weight")
    @classmethod
    def validate_weight(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"weight must be positive, got {v}")
        return v

    @model_validator(mode="after")
    def validate_normalize_range(self) -> "VolumeConfig":
        lo, hi = self.normalize_min, self.normalize_max
        if (lo is None) ^ (hi is None):
            raise ValueError(
                "normalize_min and normalize_max must both be set or both omitted"
            )
        if lo is not None and hi is not None:
            if hi <= lo:
                raise ValueError(
                    f"normalize_max must be greater than normalize_min, got {lo} and {hi}"
                )
        return self


class MiaoConfig(BaseModel):
    """Top-level configuration for miaio dataset."""

    volumes: list[VolumeConfig]
    # Desired output resolutions, one per scale. Each inner list is the output voxel size per
    # spatial axis (physical units, same unit as the zarr's OME coordinateTransformations scale),
    # in output_axes spatial order — like patch_size. Used as the default for every volume; a
    # volume may override via VolumeConfig.resolutions. The number of scales (the stacked 'l'
    # dimension) is len(resolutions).
    resolutions: list[list[float]]
    output_axes: str
    patch_size: list[int]
    bbox_mode: Literal["absolute", "relative"] = "absolute"
    samples_per_epoch: int = 1000
    cache_bytes: int = 1 << 30  # 1 GB
    file_io_concurrency: int = 64
    sampling: Literal["random", "sequential"] = "random"
    overlap: Union[int, list[int]] = 0  # voxels; in output_axes spatial order (same as patch_size)
    # If True, each coarser scale's patch origin is sampled uniformly at random such that the patch
    # still covers the previous (finer) scale's patch in finest-index space. Requires scales in
    # each volume's `scales` list to be ordered fine-to-coarse (non-decreasing relative voxel size).
    sample_windows: bool = False

    @field_validator("output_axes")
    @classmethod
    def validate_output_axes(cls, v: str) -> str:
        valid_chars = set("ltxyzc")
        if not set(v).issubset(valid_chars):
            raise ValueError(
                f"output_axes must only contain characters from {{l, t, x, y, z, c}}, got {v!r}"
            )
        if len(v) != len(set(v)):
            raise ValueError(f"output_axes must not contain duplicates, got {v!r}")
        if "l" not in v:
            raise ValueError(
                f"output_axes must contain 'l' (scale level dimension), got {v!r}"
            )
        return v

    def resolutions_for(self, vol: VolumeConfig) -> list[list[float]]:
        """Return the effective resolutions for a volume (per-volume override or global default)."""
        return vol.resolutions if vol.resolutions is not None else self.resolutions

    @property
    def n_scales(self) -> int:
        """Number of scales (the stacked 'l' dimension), derived from resolutions."""
        return len(self.resolutions)

    @model_validator(mode="after")
    def validate_patch_size_dims(self) -> "MiaoConfig":
        from miao.axes import spatial_axes

        n_spatial = len(spatial_axes(self.output_axes))
        if len(self.patch_size) != n_spatial:
            raise ValueError(
                f"patch_size has {len(self.patch_size)} elements but "
                f"output_axes {self.output_axes!r} has {n_spatial} spatial dimensions"
            )
        return self

    @model_validator(mode="after")
    def validate_resolutions(self) -> "MiaoConfig":
        from miao.axes import spatial_axes

        n_spatial = len(spatial_axes(self.output_axes))

        def _check(res: list[list[float]], where: str) -> None:
            if len(res) == 0:
                raise ValueError(f"{where} must contain at least one resolution")
            for i, r in enumerate(res):
                if len(r) != n_spatial:
                    raise ValueError(
                        f"{where}[{i}] has {len(r)} elements but output_axes "
                        f"{self.output_axes!r} has {n_spatial} spatial dimensions"
                    )
                if any(v <= 0 for v in r):
                    raise ValueError(f"{where}[{i}]={r} must have all positive values")

        _check(self.resolutions, "resolutions")
        for vol in self.volumes:
            if vol.resolutions is not None:
                _check(vol.resolutions, f"Volume {vol.name!r} resolutions")
            eff = self.resolutions_for(vol)
            if len(eff) != self.n_scales:
                raise ValueError(
                    f"Volume {vol.name!r} has {len(eff)} resolutions "
                    f"but the global resolutions defines {self.n_scales} scales — "
                    "all volumes must define the same number of scales"
                )
        return self

    @model_validator(mode="after")
    def validate_sample_windows_ordering(self) -> "MiaoConfig":
        if not self.sample_windows:
            return self
        for vol in self.volumes:
            res = self.resolutions_for(vol)
            for i in range(1, len(res)):
                prev = res[i - 1]
                curr = res[i]
                if any(c + 1e-9 < p for c, p in zip(curr, prev)):
                    raise ValueError(
                        f"sample_windows requires resolutions ordered fine-to-coarse "
                        f"(non-decreasing per axis). For volume {vol.name!r}, resolution "
                        f"index {i - 1}->{i} is {prev} -> {curr}."
                    )
        return self

    @model_validator(mode="after")
    def validate_unique_names(self) -> "MiaoConfig":
        names = [v.name for v in self.volumes]
        if len(names) != len(set(names)):
            raise ValueError("Volume names must be unique")
        return self

    @model_validator(mode="after")
    def validate_overlap(self) -> "MiaoConfig":
        if self.sampling != "sequential":
            return self
        ov = self.overlap if isinstance(self.overlap, list) else [self.overlap] * len(self.patch_size)
        if len(ov) != len(self.patch_size):
            raise ValueError(
                f"overlap has {len(ov)} elements but patch_size has {len(self.patch_size)} — "
                "they must match (both in output_axes spatial order)"
            )
        for i, (o, p) in enumerate(zip(ov, self.patch_size)):
            if o < 0:
                raise ValueError(f"overlap[{i}]={o} must be >= 0")
            if o >= p:
                raise ValueError(
                    f"overlap[{i}]={o} must be < patch_size[{i}]={p} (stride would be <= 0)"
                )
        return self


def load_config(path: str | Path) -> MiaoConfig:
    """Load and validate a YAML config file."""
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)
    return MiaoConfig(**data)
