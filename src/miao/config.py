"""YAML config loading and pydantic validation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Union

import yaml
from pydantic import BaseModel, field_validator, model_validator


class ResolutionSampling(BaseModel):
    """Spec for randomly sampling output resolutions per __getitem__ call.

    An alternative to a fixed `resolutions` list: each sample draws `n_scales` resolutions from
    `[min, max]` (per spatial axis, in output_axes spatial order — like patch_size), then sorts
    them fine-to-coarse. The drawn resolutions are reported in meta["resolutions"].
    """

    # Sampling strategy. Only "log_uniform" is implemented; the dataset dispatches through a
    # registry, so additional strategies (e.g. gaussian) can be added without schema changes.
    strategy: Literal["log_uniform"] = "log_uniform"
    n_scales: int
    min: list[float]  # minimum output voxel size per spatial axis (output_axes spatial order)
    max: list[float]  # maximum output voxel size per spatial axis (output_axes spatial order)
    # If True, draw one scalar factor per scale and broadcast to all axes (cubic/isotropic
    # voxel). If False, draw each spatial axis independently (anisotropic voxel allowed).
    isotropic: bool = False
    sort: bool = True  # sort the sampled scales fine-to-coarse

    @field_validator("n_scales")
    @classmethod
    def validate_n_scales(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"n_scales must be >= 1, got {v}")
        return v


class VolumeConfig(BaseModel):
    """Configuration for a single zarr volume."""

    name: str
    path: str
    image_key: str
    # Optional per-volume override of the global MiaoConfig.resolutions. Each inner list is the
    # desired output voxel size per spatial axis (physical units, same unit as the zarr's OME
    # coordinateTransformations scale), in output_axes spatial order — like patch_size.
    resolutions: Optional[list[list[float]]] = None
    # Optional per-volume override of the global MiaoConfig.resolution_sampling.
    resolution_sampling: Optional[ResolutionSampling] = None
    zarr_version: Literal["zarr2", "zarr3"] = "zarr2"
    label_key: Optional[str] = None
    weight: float = 1.0
    normalize: bool = True  # scale images to [0, 1]; see normalize_min / normalize_max
    # If both set: clip to [normalize_min, normalize_max] then linear map to [0, 1].
    # If both omitted: integer images use [0, dtype_max]; float images are unchanged.
    normalize_min: Optional[float] = None
    normalize_max: Optional[float] = None
    # [[min_0, max_0], ...] in finest-scale voxels, storage axis order. Strictly contains every
    # window's read extent (all scales, including coarser sample_windows patches), not just the
    # patch center. Must be at least as large as the coarsest window.
    bounding_box: Optional[list[list[int]]] = None

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
    #
    # Exactly one of `resolutions` / `resolution_sampling` must be set (globally and for any
    # per-volume override). `resolution_sampling` draws resolutions randomly per sample instead.
    resolutions: Optional[list[list[float]]] = None
    resolution_sampling: Optional[ResolutionSampling] = None
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

    def resolutions_for(self, vol: VolumeConfig) -> Optional[list[list[float]]]:
        """Return the effective fixed resolutions for a volume (override or global), or None
        when this volume uses resolution sampling."""
        if vol.resolutions is not None:
            return vol.resolutions
        if vol.resolution_sampling is not None:
            return None  # per-volume override selects sampling; ignore global fixed list
        return self.resolutions

    def resolution_sampling_for(self, vol: VolumeConfig) -> Optional[ResolutionSampling]:
        """Return the effective resolution sampling spec for a volume (override or global),
        or None when this volume uses fixed resolutions."""
        if vol.resolution_sampling is not None:
            return vol.resolution_sampling
        if vol.resolutions is not None:
            return None  # per-volume override selects fixed resolutions; ignore global sampling
        return self.resolution_sampling

    def is_sampling(self, vol: VolumeConfig) -> bool:
        """Whether this volume draws resolutions randomly per sample."""
        return self.resolution_sampling_for(vol) is not None

    @property
    def n_scales(self) -> int:
        """Number of scales (the stacked 'l' dimension)."""
        if self.resolutions is not None:
            return len(self.resolutions)
        if self.resolution_sampling is not None:
            return self.resolution_sampling.n_scales
        # Mixed/invalid global config is caught by validators; fall back to the first volume.
        vol = self.volumes[0]
        if vol.resolutions is not None:
            return len(vol.resolutions)
        return vol.resolution_sampling.n_scales

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

        def _check_fixed(res: list[list[float]], where: str) -> None:
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

        def _check_sampling(spec: ResolutionSampling, where: str) -> None:
            if len(spec.min) != n_spatial or len(spec.max) != n_spatial:
                raise ValueError(
                    f"{where} min/max must each have {n_spatial} elements (output_axes "
                    f"{self.output_axes!r} spatial dims), got min={spec.min}, max={spec.max}"
                )
            if any(v <= 0 for v in spec.min) or any(v <= 0 for v in spec.max):
                raise ValueError(f"{where} min/max must have all positive values")
            if any(mx < mn for mn, mx in zip(spec.min, spec.max)):
                raise ValueError(
                    f"{where} requires min[i] <= max[i] per axis, got min={spec.min}, max={spec.max}"
                )

        def _exactly_one(res, samp, where: str) -> None:
            if (res is None) == (samp is None):
                raise ValueError(
                    f"{where}: exactly one of `resolutions` / `resolution_sampling` must be set"
                )

        # Global: exactly one of the two must be set.
        _exactly_one(self.resolutions, self.resolution_sampling, "config")
        if self.resolutions is not None:
            _check_fixed(self.resolutions, "resolutions")
        if self.resolution_sampling is not None:
            _check_sampling(self.resolution_sampling, "resolution_sampling")

        for vol in self.volumes:
            # A volume may not set both override fields.
            if vol.resolutions is not None and vol.resolution_sampling is not None:
                raise ValueError(
                    f"Volume {vol.name!r}: set at most one of `resolutions` / "
                    "`resolution_sampling` as an override"
                )
            if vol.resolutions is not None:
                _check_fixed(vol.resolutions, f"Volume {vol.name!r} resolutions")
            if vol.resolution_sampling is not None:
                _check_sampling(
                    vol.resolution_sampling, f"Volume {vol.name!r} resolution_sampling"
                )

            # Every volume must resolve to the same number of scales.
            res = self.resolutions_for(vol)
            samp = self.resolution_sampling_for(vol)
            n = len(res) if res is not None else samp.n_scales
            if n != self.n_scales:
                raise ValueError(
                    f"Volume {vol.name!r} defines {n} scales but the config defines "
                    f"{self.n_scales} — all volumes must define the same number of scales"
                )
        return self

    @model_validator(mode="after")
    def validate_sampling_constraints(self) -> "MiaoConfig":
        any_sampling = self.is_sampling_any()
        if any_sampling and self.sampling == "sequential":
            raise ValueError(
                "resolution_sampling is incompatible with sampling='sequential' "
                "(the sequential grid is built from a fixed first-scale resolution)"
            )
        if self.sample_windows and any_sampling:
            for vol in self.volumes:
                samp = self.resolution_sampling_for(vol)
                if samp is not None and not samp.isotropic:
                    raise ValueError(
                        f"Volume {vol.name!r}: sample_windows with resolution_sampling requires "
                        "isotropic=True (per-axis-independent draws cannot guarantee the per-axis "
                        "fine-to-coarse ordering sample_windows needs)"
                    )
        return self

    def is_sampling_any(self) -> bool:
        return any(self.is_sampling(vol) for vol in self.volumes)

    @model_validator(mode="after")
    def validate_sample_windows_ordering(self) -> "MiaoConfig":
        if not self.sample_windows:
            return self
        for vol in self.volumes:
            res = self.resolutions_for(vol)
            if res is None:
                continue  # sampling mode: ordering enforced by sort + isotropic constraint
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
