"""PyTorch Dataset for multi-scale patch sampling from OME-NGFF zarr volumes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tensorstore as ts
import torch
import torch.nn.functional as F
import torch.utils.data

from miao.axes import (
    compute_permutation,
    map_patch_size_to_input,
    reorient,
    spatial_axes,
    spatial_indices,
)
from miao.config import MiaoConfig, ResolutionSampling, VolumeConfig
from miao.store import create_context, open_store
from miao.zarr_meta import OmeMetadata, read_ome_metadata


def _random_patch_origin_covering_fine_extent(
    fine_lo: np.ndarray,
    fine_hi: np.ndarray,
    rel_curr: np.ndarray,
    eff_shape_curr: np.ndarray,
    max_origin: np.ndarray,
    *,
    fine_roi_lo: np.ndarray | None = None,
    fine_roi_hi_excl: np.ndarray | None = None,
) -> np.ndarray:
    """Sample an integer patch origin at the current scale, such that the patch covers [fine_lo, fine_hi) 
    in finest-index space and is within the valid data region.

    ``fine_*`` are per spatial axis in the same finest-coordinate frame as ``VolumeInfo.min_center``.
    ``rel_curr`` is ``relative_scale_factors`` at the current scale (voxel size ratio vs finest).
    ``max_origin`` is ``spatial_shape - eff_shape`` (largest valid origin per axis, inclusive).

    ``fine_roi_lo`` / ``fine_roi_hi_excl`` (if set) bound the patch's finest extent
    ``[origin * rel_curr, (origin + eff_shape) * rel_curr)`` to ``[fine_roi_lo, fine_roi_hi_excl)``
    — used to keep the patch strictly inside a ``bounding_box``. It is a hard constraint; the
    caller (via ``_center_bounds``) guarantees a feasible origin exists, so this does not fail in
    valid configs.
    """
    rel_curr = rel_curr.astype(np.float64)
    fine_lo = fine_lo.astype(np.float64)
    fine_hi = fine_hi.astype(np.float64)
    eff_f = eff_shape_curr.astype(np.float64)

    # Cover the finer extent AND fit the volume.
    lo = np.maximum(np.ceil(fine_hi / rel_curr - eff_f - 1e-9), 0.0)
    hi = np.minimum(np.floor(fine_lo / rel_curr + 1e-9), max_origin.astype(np.float64))
    # Stay inside the bounding_box (reference frame), if given.
    if fine_roi_lo is not None:
        lo = np.maximum(lo, np.ceil(fine_roi_lo.astype(np.float64) / rel_curr - 1e-9))
    if fine_roi_hi_excl is not None:
        hi = np.minimum(
            hi, np.floor(fine_roi_hi_excl.astype(np.float64) / rel_curr - eff_f + 1e-9)
        )
    if np.any(lo > hi):
        raise ValueError(
            "No patch origin at this scale that covers the finer-level window, fits the volume, "
            "and stays inside the bounding_box "
            f"(lo={lo.tolist()}, hi={hi.tolist()}, fine_lo={fine_lo.tolist()}, "
            f"fine_hi={fine_hi.tolist()}, rel_curr={rel_curr.tolist()}, "
            f"roi_lo={None if fine_roi_lo is None else fine_roi_lo.tolist()}, "
            f"roi_hi_excl={None if fine_roi_hi_excl is None else fine_roi_hi_excl.tolist()})"
        )

    out = np.empty(len(lo), dtype=np.int64)
    for d in range(len(lo)):
        out[d] = np.random.randint(int(lo[d]), int(hi[d]) + 1)
    return out


def _select_level_for_resolution(
    level_voxels: dict[int, np.ndarray], target: np.ndarray
) -> int:
    """Pick the pyramid level to read from for a desired output resolution.

    Prefers downsampling: among levels whose voxel size is <= ``target`` on every axis,
    returns the coarsest (largest voxel) — the least data to read while still only
    downsampling. If no level qualifies (``target`` finer than the finest level), returns
    the finest level (smallest voxel), so the caller upsamples instead.

    ``level_voxels`` maps pyramid level index -> spatial voxel size (per axis, same frame
    as ``target``).
    """
    eps = 1e-9
    candidates = [
        lvl for lvl, v in level_voxels.items() if np.all(v <= target + eps)
    ]
    if candidates:
        return max(candidates, key=lambda lvl: float(np.prod(level_voxels[lvl])))
    return min(level_voxels, key=lambda lvl: float(np.prod(level_voxels[lvl])))


def _sample_log_uniform(
    spec: ResolutionSampling, n_axes: int, rng: np.random.RandomState
) -> np.ndarray:
    """Draw resolutions log-uniformly from each range. Returns (total_n_scales, n_axes), output
    order. Isotropic ranges (single-value bounds) draw one scalar per scale and broadcast to all
    axes (cubic voxel); per-axis ranges draw each axis independently.
    """
    counts = spec.n_scales_per_range()
    chunks: list[np.ndarray] = []
    for (lo, hi), k in zip(spec.ranges, counts):
        lo_a = np.array(lo, dtype=np.float64)
        hi_a = np.array(hi, dtype=np.float64)
        if lo_a.shape[0] == 1:  # isotropic range
            u = rng.uniform(np.log(lo_a[0]), np.log(hi_a[0]), size=k)
            chunks.append(np.exp(u)[:, None].repeat(n_axes, axis=1))
        else:
            u = rng.uniform(np.log(lo_a), np.log(hi_a), size=(k, n_axes))
            chunks.append(np.exp(u))
    return np.concatenate(chunks, axis=0)


# Strategy registry — add new samplers here (e.g. "gaussian") without touching call sites.
_RESOLUTION_SAMPLERS = {
    "log_uniform": _sample_log_uniform,
}


def _sort_fine_to_coarse(res: np.ndarray) -> np.ndarray:
    """Sort sampled resolutions (n_scales, n_axes) fine-to-coarse by per-scale geometric mean."""
    keys = np.prod(res, axis=1)
    return res[np.argsort(keys, kind="stable")]


def _sample_resolutions(
    spec: ResolutionSampling, n_axes: int, rng: np.random.RandomState
) -> list[list[float]]:
    """Draw a set of resolutions for one sample, in output_axes spatial order."""
    raw = _RESOLUTION_SAMPLERS[spec.strategy](spec, n_axes, rng)
    if spec.sort:
        raw = _sort_fine_to_coarse(raw)
    return [row.tolist() for row in raw]


def _normalize_image_tensor(
    img_tensor: torch.Tensor,
    *,
    normalize: bool,
    normalize_min: float | None,
    normalize_max: float | None,
    image_dtype: np.dtype,
) -> torch.Tensor:
    """Scale image values to approximately [0, 1] for training."""
    if not normalize:
        return img_tensor
    if normalize_min is not None and normalize_max is not None:
        lo = float(normalize_min)
        hi = float(normalize_max)
        img_tensor = img_tensor.clamp(lo, hi)
        return (img_tensor - lo) / (hi - lo)
    if np.issubdtype(image_dtype, np.integer):
        return img_tensor / float(np.iinfo(image_dtype).max)
    return img_tensor


@dataclass
class ScaleResolution:
    """Per-scale resolution of a set of target resolutions to pyramid reads.

    Produced by ``VolumeDataset._resolve_scales`` — either once at init (fixed resolutions) or
    fresh per ``__getitem__`` call (resolution sampling). All lists are length ``n_scales``.
    """

    # Per-scale target output resolution (spatial dims, image storage axis order)
    resolutions: list[np.ndarray]
    # Per-scale chosen image pyramid level to read from
    chosen_levels: list[int]
    # Per-scale spatial scale factors of the chosen level relative to level-0 (voxel ratio)
    relative_scale_factors: list[np.ndarray]
    # Per-scale storage read shape (image spatial axis order); resampled to read_shape on read
    read_shapes: list[np.ndarray]
    # Per-scale chosen label pyramid level (None if no labels)
    label_chosen_levels: list[int] | None
    # Per-scale label spatial scale factors of the chosen label level relative to image level-0
    label_relative_scale_factors: list[np.ndarray] | None
    # Per-scale label storage read shape (label spatial axis order)
    label_read_shapes: list[np.ndarray] | None


@dataclass
class VolumeInfo:
    """Resolved metadata for a single volume, ready for sampling.

    Holds the resolution-independent static metadata. The per-scale resolution
    (``ScaleResolution``) is cached in ``scales`` when resolutions are fixed, or recomputed per
    sample (``scales is None`` and ``sampling_spec`` is set).
    """

    config: VolumeConfig
    image_meta: OmeMetadata
    label_meta: OmeMetadata | None
    # Full axes strings derived from OME-NGFF metadata (e.g., "zyxc", "zyx")
    img_axes: str
    img_spatial_axes: str
    lbl_axes: str | None
    lbl_spatial_axes: str | None
    # Indices of spatial dims in the OME-NGFF axis arrays
    img_spatial_idx: list[int]
    lbl_spatial_idx: list[int] | None
    # Source image dtype (for normalization)
    image_dtype: np.dtype
    # Level-0 (finest pyramid) absolute spatial voxel size (physical units). This is the
    # reference frame for center coordinates and min_center/max_center.
    finest_voxel_size: np.ndarray
    # Per-level absolute spatial voxel sizes (for on-the-fly level selection)
    img_level_voxels: dict[int, np.ndarray]
    lbl_level_voxels: dict[int, np.ndarray] | None
    # Output patch size in image spatial axis order (interpolation target for every scale)
    read_shape: list[int]
    # Valid center coordinate range (spatial dims only, in image spatial axis order, level-0 frame)
    min_center: np.ndarray
    max_center: np.ndarray
    # Cached per-scale resolution (fixed-resolution mode); None when sampling per call
    scales: ScaleResolution | None = None
    # Resolution sampling spec (sampling mode); None when resolutions are fixed
    sampling_spec: ResolutionSampling | None = None


class VolumeDataset(torch.utils.data.Dataset):
    """Map-style dataset that yields multi-scale random patches from OME-NGFF zarr volumes.

    Each __getitem__ call:
    1. Randomly picks one volume (weighted sampling).
    2. Picks a random coordinate in that volume's finest-scale space.
    3. Extracts patch_size voxels from each requested scale level.
    4. Returns {"img": Tensor, "label": Tensor, "bbox": Tensor, "meta": dict}.
       If labels are unavailable, "label" is an empty long tensor.

    Tensor shapes are (1, L, *output_axes_dims) where L = number of scale levels.
    Input axes are auto-detected from OME-NGFF metadata.
    """

    def __init__(self, config: MiaoConfig) -> None:
        self.config = config

        # Normalize sampling weights to probabilities
        weights = np.array([v.weight for v in config.volumes])
        self._probabilities = weights / weights.sum()

        # Read metadata and precompute sampling bounds for each volume
        self._volumes: list[VolumeInfo] = []
        for vol_cfg in config.volumes:
            self._volumes.append(self._resolve_volume(vol_cfg))

        # Build sequential grid before printing summary (summary reports grid size)
        self._grid: list[tuple[int, np.ndarray, tuple]] = []
        if config.sampling == "sequential":
            self._build_sequential_grid()

        # Print summary
        self._print_summary()

        # Per-worker tensorstore handle cache (populated lazily)
        self._worker_stores: dict[int, dict] = {}

    def _print_summary(self) -> None:
        """Print a summary of detected metadata for all volumes."""
        if self.config.sampling == "sequential":
            ov = self.config.overlap
            mode_str = f"sequential, {len(self._grid)} total positions, overlap={ov}"
        else:
            mode_str = f"random, {self.config.samples_per_epoch} samples/epoch"
        print(f"VolumeDataset: {len(self._volumes)} volume(s), "
              f"{mode_str}, "
              f"patch_size={self.config.patch_size}")
        for vi in self._volumes:
            finest_meta = vi.image_meta.scales[0]
            prob = self._probabilities[self._volumes.index(vi)]
            # Get axis unit from OME-NGFF metadata
            units = [ax.get("unit", "") for ax in vi.image_meta.axes]
            unit_str = units[0] if units and all(u == units[0] for u in units if u) else ""
            unit_label = f" {unit_str}" if unit_str else ""
            lines = [
                f"  [{vi.config.name}]",
                f"    image: axes={vi.img_axes!r}, shape={finest_meta.shape}, "
                f"dtype={finest_meta.dtype}",
            ]
            if vi.scales is not None:
                # Fixed: target resolution -> chosen source level (voxel size, storage read shape)
                sc = vi.scales
                for s in range(len(sc.resolutions)):
                    target = sc.resolutions[s].tolist()
                    lvl = sc.chosen_levels[s]
                    lvl_voxel = (sc.relative_scale_factors[s] * vi.finest_voxel_size).tolist()
                    read = sc.read_shapes[s].tolist()
                    lines.append(
                        f"    scale {s}: resolution={target}{unit_label} -> level {lvl} "
                        f"(voxel_size={lvl_voxel}, read {read} -> resample to {vi.read_shape})"
                    )
            else:
                # Sampling: report the spec; resolutions drawn fresh per sample.
                sp = vi.sampling_spec
                lines.append(
                    f"    resolution_sampling: {sp.strategy}, "
                    f"total_scales={sp.total_n_scales()}{unit_label}"
                )
                for (lo, hi), k in zip(sp.ranges, sp.n_scales_per_range()):
                    iso = " (isotropic)" if sp.range_is_isotropic([lo, hi]) else ""
                    lines.append(f"      range {lo}..{hi} x{k}{iso}")
            if vi.label_meta is not None:
                lbl_meta = vi.label_meta.scales[0]
                lines.append(
                    f"    label: axes={vi.lbl_axes!r}, shape={lbl_meta.shape}, "
                    f"dtype={lbl_meta.dtype}"
                )
            lines.append(f"    sampling: weight={prob:.2f}, "
                         f"center_range=[{vi.min_center.tolist()}, {vi.max_center.tolist()}]")
            if vi.config.normalize:
                if (
                    vi.config.normalize_min is not None
                    and vi.config.normalize_max is not None
                ):
                    lines.append(
                        "    normalize: "
                        f"clamp [{vi.config.normalize_min}, {vi.config.normalize_max}] -> [0, 1]"
                    )
                elif np.issubdtype(vi.image_dtype, np.integer):
                    lines.append(
                        f"    normalize: {vi.image_dtype} -> [0, 1] (divide by dtype max)"
                    )
                else:
                    lines.append(
                        "    normalize: float dtype unchanged "
                        "(set normalize_min and normalize_max to scale)"
                    )
            print("\n".join(lines))
        dims = ", ".join(c.upper() for c in self.config.output_axes)
        print(f"  output: axes={self.config.output_axes!r}, tensor_shape=({dims})")

    def _resolve_volume(self, vol_cfg: VolumeConfig) -> VolumeInfo:
        """Read OME-NGFF metadata and precompute sampling bounds for a volume.

        Each requested output resolution is resolved to a pyramid level (preferring
        downsampling: the coarsest level whose voxel size is still <= the target on every
        axis; the finest level when the target is finer than anything stored, so we
        upsample). Patches are read at that level and resampled to ``patch_size``.
        """
        # Load all pyramid levels so resolution selection can choose among them.
        image_meta = read_ome_metadata(
            vol_cfg.path, vol_cfg.image_key, vol_cfg.zarr_version, None
        )

        label_meta = None
        if vol_cfg.label_key:
            label_meta = read_ome_metadata(
                vol_cfg.path, vol_cfg.label_key, vol_cfg.zarr_version, None
            )

        # Derive axes from OME-NGFF metadata
        img_axes = "".join(image_meta.axis_names)
        img_spatial = spatial_axes(img_axes)
        img_sp_idx = spatial_indices(img_axes)
        output_spatial = spatial_axes(self.config.output_axes)

        lbl_axes = None
        lbl_spatial = None
        lbl_sp_idx = None
        if label_meta is not None:
            lbl_axes = "".join(label_meta.axis_names)
            lbl_spatial = spatial_axes(lbl_axes)
            lbl_sp_idx = spatial_indices(lbl_axes)

        # Output patch size in image spatial axis order (interpolation target for every scale)
        read_shape = map_patch_size_to_input(
            self.config.patch_size, img_spatial, output_spatial
        )

        # Reference frame: finest pyramid level (index 0) absolute spatial voxel size.
        finest_spatial_factors = np.array(
            image_meta.scales[0].scale_factors, dtype=np.float64
        )[img_sp_idx]

        # Per-level absolute spatial voxel sizes (for level selection)
        img_level_voxels: dict[int, np.ndarray] = {
            lvl: np.array(m.scale_factors, dtype=np.float64)[img_sp_idx]
            for lvl, m in image_meta.scales.items()
        }
        lbl_level_voxels: dict[int, np.ndarray] | None = None
        if label_meta is not None and lbl_sp_idx is not None:
            lbl_level_voxels = {
                lvl: np.array(m.scale_factors, dtype=np.float64)[lbl_sp_idx]
                for lvl, m in label_meta.scales.items()
            }

        # Build the static (resolution-independent) volume info first.
        vol_info = VolumeInfo(
            config=vol_cfg,
            image_meta=image_meta,
            label_meta=label_meta,
            img_axes=img_axes,
            img_spatial_axes=img_spatial,
            lbl_axes=lbl_axes,
            lbl_spatial_axes=lbl_spatial,
            img_spatial_idx=img_sp_idx,
            lbl_spatial_idx=lbl_sp_idx,
            image_dtype=image_meta.scales[0].dtype,
            finest_voxel_size=finest_spatial_factors,
            img_level_voxels=img_level_voxels,
            lbl_level_voxels=lbl_level_voxels,
            read_shape=read_shape,
            min_center=np.zeros(len(img_sp_idx), dtype=np.int64),
            max_center=np.zeros(len(img_sp_idx), dtype=np.int64),
        )

        sampling_spec = self.config.resolution_sampling_for(vol_cfg)
        if sampling_spec is not None:
            # Sampling mode: resolutions drawn per __getitem__. Bound centers using the coarsest
            # case (the per-axis max upper bound across all ranges) — the largest physical extent
            # is the binding constraint, so any sampled resolution is guaranteed to fit.
            vol_info.sampling_spec = sampling_spec
            n_axes = len(output_spatial)
            bound_res = [sampling_spec.max_resolution(n_axes)]
            scale_res = self._resolve_scales(vol_info, bound_res)
        else:
            # Fixed mode: resolve once and cache.
            effective_res = self.config.resolutions_for(vol_cfg)
            scale_res = self._resolve_scales(vol_info, effective_res)
            vol_info.scales = scale_res

        # _center_bounds folds in the volume fit and the strict bounding_box (extent-based).
        min_center, max_center = self._center_bounds(vol_info, scale_res)
        vol_info.min_center = np.ceil(min_center).astype(np.int64)
        vol_info.max_center = np.floor(max_center).astype(np.int64)
        return vol_info

    def _resolve_scales(
        self, vol_info: VolumeInfo, resolutions_output_order: list
    ) -> ScaleResolution:
        """Resolve a set of target resolutions (output_axes spatial order) to per-scale pyramid
        reads. Pure function of the volume's static metadata; called once (fixed mode) or per
        sample (sampling mode)."""
        output_spatial = spatial_axes(self.config.output_axes)
        img_spatial = vol_info.img_spatial_axes
        lbl_spatial = vol_info.lbl_spatial_axes
        finest = vol_info.finest_voxel_size
        read_shape_arr = np.array(vol_info.read_shape, dtype=np.float64)
        has_labels = vol_info.label_meta is not None and vol_info.lbl_level_voxels is not None

        resolutions: list[np.ndarray] = []
        chosen_levels: list[int] = []
        relative_scale_factors: list[np.ndarray] = []
        read_shapes: list[np.ndarray] = []
        label_chosen_levels: list[int] | None = [] if has_labels else None
        label_relative_scale_factors: list[np.ndarray] | None = [] if has_labels else None
        label_read_shapes: list[np.ndarray] | None = [] if has_labels else None

        for target_out in resolutions_output_order:
            img_target = np.array(
                map_patch_size_to_input(list(target_out), img_spatial, output_spatial),
                dtype=np.float64,
            )
            resolutions.append(img_target)
            img_lvl = _select_level_for_resolution(vol_info.img_level_voxels, img_target)
            img_voxel = vol_info.img_level_voxels[img_lvl]
            chosen_levels.append(img_lvl)
            relative_scale_factors.append(img_voxel / finest)
            # Storage voxels to read so that, after resampling to read_shape, the output patch
            # has voxel size `target`: read = patch * target / level_voxel.
            read_shapes.append(np.ceil(read_shape_arr * img_target / img_voxel).astype(np.int64))

            if has_labels:
                lbl_target = np.array(
                    map_patch_size_to_input(list(target_out), lbl_spatial, output_spatial),
                    dtype=np.float64,
                )
                lbl_lvl = _select_level_for_resolution(vol_info.lbl_level_voxels, lbl_target)
                lbl_voxel = vol_info.lbl_level_voxels[lbl_lvl]
                lbl_patch = np.array(
                    map_patch_size_to_input(self.config.patch_size, lbl_spatial, output_spatial),
                    dtype=np.float64,
                )
                label_chosen_levels.append(lbl_lvl)
                label_relative_scale_factors.append(lbl_voxel / finest)
                label_read_shapes.append(
                    np.ceil(lbl_patch * lbl_target / lbl_voxel).astype(np.int64)
                )

        return ScaleResolution(
            resolutions=resolutions,
            chosen_levels=chosen_levels,
            relative_scale_factors=relative_scale_factors,
            read_shapes=read_shapes,
            label_chosen_levels=label_chosen_levels,
            label_relative_scale_factors=label_relative_scale_factors,
            label_read_shapes=label_read_shapes,
        )

    def _center_bounds(
        self, vol_info: VolumeInfo, scale_res: ScaleResolution
    ) -> tuple[np.ndarray, np.ndarray]:
        """Valid center coordinate range (spatial dims, level-0 frame) for a ScaleResolution.

        Keeps every scale's (image and label) read extent inside the volume and, when a
        ``bounding_box`` is set, strictly inside that box. The bounding box constrains the read
        *extent* (not just the center), so a centered patch placed at any returned center lies
        fully within the box; the most restrictive (coarsest/largest) scale dominates.
        """
        img_sp_idx = vol_info.img_spatial_idx
        finest_spatial_shape = np.array(vol_info.image_meta.scales[0].shape)[img_sp_idx]
        min_center = np.zeros(len(img_sp_idx), dtype=np.float64)
        max_center = finest_spatial_shape.copy().astype(np.float64)

        bb = None
        if vol_info.config.bounding_box is not None:
            bb = np.array(vol_info.config.bounding_box, dtype=np.float64)  # ref-frame [lo, hi)

        def _apply(rel: np.ndarray, eff: np.ndarray, sp_shape: np.ndarray) -> None:
            nonlocal min_center, max_center
            eff_half = np.floor(eff / 2)
            # Volume fit (centered patch within [0, sp_shape) at this level).
            min_center = np.maximum(min_center, rel * eff_half)
            max_center = np.minimum(max_center, rel * (sp_shape - eff + eff_half))
            if bb is not None:
                # Strict bbox on the read extent. A centered patch spans (ref frame, worst-case
                # over the floor()/eff//2 quantization) [center - (eff_half + 1)*rel,
                # center + ceil(eff/2)*rel]; keep that inside [bb_lo, bb_hi).
                ceil_half = np.ceil(eff / 2)
                min_center = np.maximum(min_center, bb[:, 0] + (eff_half + 1.0) * rel)
                max_center = np.minimum(max_center, bb[:, 1] - ceil_half * rel)

        for s in range(len(scale_res.resolutions)):
            img_sp_shape = np.array(
                vol_info.image_meta.scales[scale_res.chosen_levels[s]].shape, dtype=np.float64
            )[img_sp_idx]
            _apply(
                scale_res.relative_scale_factors[s],
                scale_res.read_shapes[s].astype(np.float64),
                img_sp_shape,
            )

            if (
                vol_info.label_meta is not None
                and scale_res.label_relative_scale_factors is not None
                and scale_res.label_chosen_levels is not None
                and scale_res.label_read_shapes is not None
                and vol_info.lbl_spatial_idx is not None
            ):
                lbl_sp_shape = np.array(
                    vol_info.label_meta.scales[scale_res.label_chosen_levels[s]].shape,
                    dtype=np.float64,
                )[vol_info.lbl_spatial_idx]
                _apply(
                    scale_res.label_relative_scale_factors[s],
                    scale_res.label_read_shapes[s].astype(np.float64),
                    lbl_sp_shape,
                )

        if bb is not None and np.any(np.ceil(min_center) > np.floor(max_center)):
            raise ValueError(
                f"Volume {vol_info.config.name!r}: bounding_box "
                f"{vol_info.config.bounding_box} is too small to contain the requested "
                f"window(s) at patch_size={self.config.patch_size}. Valid center range "
                f"collapsed (min_center={np.ceil(min_center).tolist()} > "
                f"max_center={np.floor(max_center).tolist()})."
            )

        return min_center, max_center

    def _build_sequential_grid(self) -> None:
        """Precompute flat list of (vol_idx, center, grid_index) for sequential sampling.

        center is in the level-0 reference frame (image spatial axis order, matching
        min_center/max_center). grid_index is a tuple of per-axis position indices within the
        grid for that volume, useful for stitching patch predictions back into a full-volume
        output.

        The grid tiles the volume at the first scale's target resolution: the stride is one
        output patch (minus overlap) worth of physical extent, expressed in reference voxels.
        """
        from itertools import product as iproduct

        output_spatial = spatial_axes(self.config.output_axes)
        ov = self.config.overlap
        if isinstance(ov, int):
            ov = [ov] * len(self.config.patch_size)

        for vol_idx, vol_info in enumerate(self._volumes):
            # Map overlap from output_axes spatial order → input (storage) spatial order
            overlap_input = map_patch_size_to_input(
                ov, vol_info.img_spatial_axes, output_spatial
            )

            # Scale-0 output voxel size in reference (level-0) voxels per axis. Sequential mode
            # always uses fixed resolutions (sampling is rejected by config validation).
            assert vol_info.scales is not None
            ref_per_patch_voxel = vol_info.scales.resolutions[0] / vol_info.finest_voxel_size
            stride = (
                (np.array(vol_info.read_shape) - np.array(overlap_input)) * ref_per_patch_voxel
            )

            ranges: list[list[int]] = []
            for i in range(len(vol_info.read_shape)):
                lo = int(vol_info.min_center[i])
                hi = int(vol_info.max_center[i])
                step = max(1, int(round(stride[i])))
                positions = list(range(lo, hi + 1, step))
                if not positions:
                    raise ValueError(
                        f"Volume {vol_info.config.name!r}: no valid center positions "
                        f"along spatial axis {i} (min_center={lo} > max_center={hi}). "
                        f"Volume may be too small for the requested patch_size."
                    )
                if positions[-1] < hi:
                    positions.append(hi)
                ranges.append(positions)

            for grid_idx in iproduct(*[range(len(r)) for r in ranges]):
                center = np.array(
                    [ranges[ax][grid_idx[ax]] for ax in range(len(ranges))]
                )
                self._grid.append((vol_idx, center, grid_idx))

    def _get_worker_id(self) -> int:
        worker_info = torch.utils.data.get_worker_info()
        return worker_info.id if worker_info is not None else 0

    def _get_stores(self) -> dict:
        """Lazily create tensorstore handles for the current worker."""
        worker_id = self._get_worker_id()
        if worker_id not in self._worker_stores:
            ctx = create_context(self.config.cache_bytes, self.config.file_io_concurrency)
            stores: dict = {}
            for vol_info in self._volumes:
                vol_name = vol_info.config.name
                zarr_path = Path(vol_info.config.path)
                zarr_ver = vol_info.config.zarr_version
                stores[vol_name] = {"img": {}, "label": {}}

                # Which pyramid levels to open: the union of cached chosen levels in fixed mode,
                # or all levels in sampling mode (any level may be selected per call).
                if vol_info.scales is not None:
                    img_levels = sorted(set(vol_info.scales.chosen_levels))
                    lbl_levels = (
                        sorted(set(vol_info.scales.label_chosen_levels))
                        if vol_info.scales.label_chosen_levels is not None
                        else []
                    )
                else:
                    img_levels = sorted(vol_info.image_meta.scales.keys())
                    lbl_levels = (
                        sorted(vol_info.label_meta.scales.keys())
                        if vol_info.label_meta is not None
                        else []
                    )

                for level in img_levels:
                    img_array_path = (
                        zarr_path
                        / vol_info.config.image_key
                        / vol_info.image_meta.scales[level].path
                    )
                    stores[vol_name]["img"][level] = open_store(
                        img_array_path, zarr_ver, ctx
                    )

                if vol_info.config.label_key and vol_info.label_meta:
                    for level in lbl_levels:
                        lbl_array_path = (
                            zarr_path
                            / vol_info.config.label_key
                            / vol_info.label_meta.scales[level].path
                        )
                        stores[vol_name]["label"][level] = open_store(
                            lbl_array_path, zarr_ver, ctx
                        )

            self._worker_stores[worker_id] = stores
        return self._worker_stores[worker_id]

    def __len__(self) -> int:
        if self.config.sampling == "sequential":
            return len(self._grid)
        return self.config.samples_per_epoch

    def _build_img_slices(
        self, origin: np.ndarray, read_shape: np.ndarray, vol_info: VolumeInfo
    ) -> tuple:
        """Build slices for image array: spatial dims get crop, channel gets slice(None)."""
        slices = []
        sp_i = 0
        for dim_i, ax_char in enumerate(vol_info.img_axes):
            if dim_i in vol_info.img_spatial_idx:
                slices.append(slice(int(origin[sp_i]), int(origin[sp_i] + read_shape[sp_i])))
                sp_i += 1
            else:
                slices.append(slice(None))  # channel dim: take all
        return tuple(slices)

    def __getitem__(self, idx: int) -> dict:
        stores = self._get_stores()

        # Pick volume (and center in sequential mode) based on sampling strategy
        grid_index: tuple | None = None
        if self.config.sampling == "sequential":
            vol_idx, center, grid_index = self._grid[idx]
        else:
            vol_idx = np.random.choice(len(self._volumes), p=self._probabilities)
        vol_info = self._volumes[vol_idx]
        vol_stores = stores[vol_info.config.name]
        output_spatial = spatial_axes(self.config.output_axes)
        spatial_perm = compute_permutation(vol_info.img_spatial_axes, output_spatial)

        # Per-scale reads: cached (fixed resolutions) or freshly drawn (resolution sampling).
        if vol_info.scales is not None:
            scales = vol_info.scales
        else:
            sampled = _sample_resolutions(
                vol_info.sampling_spec, len(output_spatial), np.random
            )
            scales = self._resolve_scales(vol_info, sampled)

        # Determine image intermediate axes after stacking: "l" + img_axes
        # Handle channel mismatch:
        #   - img has c, output doesn't: squeeze c before stacking
        #   - img lacks c, output has c: unsqueeze c after stacking
        #   - both have or neither has c: direct permute
        has_img_channel = "c" in vol_info.img_axes
        wants_channel = "c" in self.config.output_axes
        squeeze_channel = has_img_channel and not wants_channel
        add_channel = not has_img_channel and wants_channel

        if squeeze_channel:
            img_intermediate = "l" + vol_info.img_spatial_axes
        else:
            img_intermediate = "l" + vol_info.img_axes

        # If we need to add a channel dim, we'll unsqueeze after stacking
        # and insert 'c' into the intermediate string at the end
        if add_channel:
            img_intermediate_for_perm = img_intermediate + "c"
        else:
            img_intermediate_for_perm = img_intermediate
        img_perm = compute_permutation(img_intermediate_for_perm, self.config.output_axes)

        # Label intermediate/output handling:
        # - labels are returned without channel axis
        # - if label storage has 'c', squeeze it before stacking/permuting
        lbl_output_axes = self.config.output_axes.replace("c", "")
        has_lbl_channel = vol_info.lbl_axes is not None and "c" in vol_info.lbl_axes
        if vol_info.lbl_axes is not None:
            if has_lbl_channel:
                lbl_intermediate = "l" + spatial_axes(vol_info.lbl_axes)
            else:
                lbl_intermediate = "l" + vol_info.lbl_axes
            lbl_perm = compute_permutation(lbl_intermediate, lbl_output_axes)

        # Pick center coordinate (random mode only; sequential already set center above)
        if self.config.sampling == "random":
            center = np.array(
                [
                    np.random.randint(lo, hi + 1)
                    for lo, hi in zip(vol_info.min_center, vol_info.max_center)
                ]
            )

        read_shape = np.array(vol_info.read_shape)
        target_size = tuple(int(s) for s in read_shape)  # interpolation target
        img_crops: list[np.ndarray] = []
        label_crops: list[np.ndarray] = []
        bboxes: list[np.ndarray] = []

        # When sample_windows samples a coarser scale's origin off-center, constrain its read
        # extent to the bounding_box (reference / level-0 voxels) so every window stays strictly
        # inside the box. Without a bounding_box, only covering + volume bounds apply. The box is
        # a hard constraint here; _center_bounds guarantees a feasible origin exists.
        sample_roi_lo: np.ndarray | None = None
        sample_roi_hi_excl: np.ndarray | None = None
        n_scales = len(scales.resolutions)
        if self.config.sample_windows and n_scales > 1 and vol_info.config.bounding_box is not None:
            bb = np.array(vol_info.config.bounding_box, dtype=np.float64)
            sample_roi_lo = bb[:, 0]
            sample_roi_hi_excl = bb[:, 1]

        # Phase 1: Compute slices and issue all reads concurrently via ts.Batch
        img_futures: list[ts.Future] = []
        lbl_futures: list[ts.Future | None] = []

        prev_origin: np.ndarray | None = None
        prev_eff_shape: np.ndarray | None = None
        prev_rel: np.ndarray | None = None

        with ts.Batch() as batch:
            for s in range(n_scales):
                level = scales.chosen_levels[s]
                rel_factors = scales.relative_scale_factors[s]
                eff_shape = scales.read_shapes[s]
                eff_half = eff_shape // 2

                if self.config.sample_windows and s > 0:
                    assert prev_origin is not None and prev_eff_shape is not None and prev_rel is not None
                    if not np.all(rel_factors + 1e-9 >= prev_rel):
                        raise AssertionError(
                            "sample_windows requires `resolutions` ordered from higher resolution "
                            "to lower (non-decreasing relative_scale_factors per axis). "
                            f"At scale index {s - 1} -> {s}, prev_rel={prev_rel.tolist()} but "
                            f"current rel={rel_factors.tolist()}."
                        )
                    fine_lo = prev_origin.astype(np.float64) * prev_rel
                    fine_hi = (prev_origin + prev_eff_shape).astype(np.float64) * prev_rel
                    img_sp_shape = np.array(
                        vol_info.image_meta.scales[level].shape, dtype=np.int64
                    )[vol_info.img_spatial_idx]
                    max_origin = img_sp_shape - eff_shape
                    origin = _random_patch_origin_covering_fine_extent(
                        fine_lo,
                        fine_hi,
                        rel_factors,
                        eff_shape,
                        max_origin,
                        fine_roi_lo=sample_roi_lo,
                        fine_roi_hi_excl=sample_roi_hi_excl,
                    )
                else:
                    center_at_level = np.floor(center / rel_factors).astype(np.int64)
                    origin = center_at_level - eff_half

                prev_origin = origin
                prev_eff_shape = np.asarray(eff_shape, dtype=np.int64).copy()
                prev_rel = rel_factors.astype(np.float64).copy()

                voxel_size = vol_info.finest_voxel_size
                phys_min = (origin * rel_factors * voxel_size).astype(np.float64)
                phys_max = ((origin + eff_shape) * rel_factors * voxel_size).astype(np.float64)
                bbox = np.stack(
                    [phys_min[list(spatial_perm)], phys_max[list(spatial_perm)]]
                )
                bboxes.append(bbox)

                img_slices = self._build_img_slices(origin, eff_shape, vol_info)
                img_futures.append(vol_stores["img"][level][img_slices].read(batch=batch))

                if (
                    vol_info.config.label_key
                    and scales.label_chosen_levels is not None
                    and vol_info.label_meta is not None
                    and scales.label_relative_scale_factors is not None
                    and scales.label_read_shapes is not None
                    and vol_info.lbl_axes is not None
                    and vol_info.lbl_spatial_idx is not None
                ):
                    lbl_level = scales.label_chosen_levels[s]
                    lbl_rel_factors = scales.label_relative_scale_factors[s]
                    lbl_eff_shape = scales.label_read_shapes[s]
                    lbl_eff_half = lbl_eff_shape // 2
                    if self.config.sample_windows and s > 0:
                        center_fine = (
                            origin.astype(np.float64) + eff_half.astype(np.float64)
                        ) * rel_factors
                        lbl_center = np.floor(center_fine / lbl_rel_factors).astype(
                            np.int64
                        )
                    else:
                        lbl_center = np.floor(center / lbl_rel_factors).astype(np.int64)
                    lbl_origin = lbl_center - lbl_eff_half
                    # Build label slices from label axes metadata:
                    # spatial dims get cropped; non-spatial dims (e.g. channel) take all.
                    lbl_slices = []
                    sp_i = 0
                    for dim_i, _ax_char in enumerate(vol_info.lbl_axes):
                        if dim_i in vol_info.lbl_spatial_idx:
                            lbl_slices.append(
                                slice(int(lbl_origin[sp_i]), int(lbl_origin[sp_i] + lbl_eff_shape[sp_i]))
                            )
                            sp_i += 1
                        else:
                            lbl_slices.append(slice(None))
                    lbl_futures.append(vol_stores["label"][lbl_level][tuple(lbl_slices)].read(batch=batch))
                else:
                    lbl_futures.append(None)

        # Phase 2: Collect results (all reads already completed when batch exited)
        for img_future, lbl_future in zip(img_futures, lbl_futures):
            patch = np.asarray(img_future.result())
            if has_img_channel and not wants_channel:
                c_idx = vol_info.img_axes.index("c")
                patch = np.squeeze(patch, axis=c_idx)

            # Resample the storage read to the output patch_size (downsample or upsample).
            has_channel_in_patch = has_img_channel and not squeeze_channel
            if has_channel_in_patch:
                sp_shape = tuple(patch.shape[i] for i in vol_info.img_spatial_idx)
            else:
                sp_shape = tuple(patch.shape)

            if sp_shape != target_size:
                patch_t = torch.from_numpy(patch).float()
                if has_channel_in_patch:
                    # Rearrange to (C, spatial...) so F.interpolate sees (N,C,D,H,W)
                    c_pos = vol_info.img_axes.index("c")
                    perm = [c_pos] + list(vol_info.img_spatial_idx)
                    inv_perm = [0] * len(perm)
                    for i, p in enumerate(perm):
                        inv_perm[p] = i
                    patch_t = patch_t.permute(perm)
                    patch_t = F.interpolate(
                        patch_t.unsqueeze(0),
                        size=target_size, mode="trilinear", align_corners=False,
                    ).squeeze(0)
                    patch_t = patch_t.permute(inv_perm)
                else:
                    # Spatial only — add batch + channel dims
                    patch_t = F.interpolate(
                        patch_t.unsqueeze(0).unsqueeze(0),
                        size=target_size, mode="trilinear", align_corners=False,
                    ).squeeze(0).squeeze(0)
                patch = patch_t.numpy()

            img_crops.append(patch)

            if lbl_future is not None:
                lbl = np.asarray(lbl_future.result())

                if has_lbl_channel and vol_info.lbl_axes is not None:
                    c_idx = vol_info.lbl_axes.index("c")
                    lbl = np.squeeze(lbl, axis=c_idx)

                # Resample labels with nearest-neighbor to preserve integer IDs
                if has_lbl_channel and vol_info.lbl_spatial_idx is not None and vol_info.lbl_axes is not None:
                    # After squeeze, spatial indices shift left past removed channel.
                    c_idx = vol_info.lbl_axes.index("c")
                    shifted_spatial_idx = [i if i < c_idx else i - 1 for i in vol_info.lbl_spatial_idx]
                    lbl_spatial_shape = tuple(int(lbl.shape[i]) for i in shifted_spatial_idx)
                else:
                    lbl_spatial_shape = tuple(int(s) for s in lbl.shape)
                if lbl_spatial_shape != target_size:
                    lbl_t = torch.from_numpy(lbl).float().unsqueeze(0).unsqueeze(0)
                    lbl_t = F.interpolate(lbl_t, size=target_size, mode="nearest").squeeze(0).squeeze(0)
                    lbl = lbl_t.numpy().astype(np.int64)
                label_crops.append(lbl)

        # Stack across levels → (L, *storage_axes), then permute to output_axes
        img_stacked = torch.from_numpy(np.stack(img_crops))
        if add_channel:
            img_stacked = img_stacked.unsqueeze(-1)  # add singleton C at end
        img_tensor = img_stacked.permute(img_perm).float()
        img_tensor = _normalize_image_tensor(
            img_tensor,
            normalize=vol_info.config.normalize,
            normalize_min=vol_info.config.normalize_min,
            normalize_max=vol_info.config.normalize_max,
            image_dtype=vol_info.image_dtype,
        )
        if label_crops:
            label_tensor = torch.from_numpy(np.stack(label_crops)).permute(lbl_perm).long()
        else:
            # Keep output collate-friendly for default PyTorch DataLoader.
            label_tensor = torch.empty(0, dtype=torch.long)

        # Stack bboxes: (L, 2, Nd_spatial) in output spatial order, physical units
        bbox_arr = np.stack(bboxes)
        if self.config.bbox_mode == "relative":
            # Make relative to the center of the finest-level crop
            finest_center = (bbox_arr[0, 0] + bbox_arr[0, 1]) / 2.0
            bbox_arr = bbox_arr - finest_center
        bbox_tensor = torch.from_numpy(bbox_arr).float()

        return {
            "img": img_tensor,
            "label": label_tensor,
            "bbox": bbox_tensor,
            "meta": {
                "volume": vol_info.config.name,
                "coordinate": center.tolist(),
                "resolutions": [r.tolist() for r in scales.resolutions],
                "source_levels": scales.chosen_levels,
                # grid_index only present in sequential mode; None breaks DataLoader collation
                **({"grid_index": grid_index} if grid_index is not None else {}),
            },
        }
