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
from miao.config import MiaoConfig, VolumeConfig
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

    When ``fine_roi_lo`` / ``fine_roi_hi_excl`` are set, the patch's finest extent
    ``[origin * rel_curr, (origin + eff_shape) * rel_curr)`` must lie inside the half-open ROI
    ``[fine_roi_lo, fine_roi_hi_excl)`` per axis (same frame as ``min_center`` / ``max_center``).
    """
    rel_curr = rel_curr.astype(np.float64)
    fine_lo = fine_lo.astype(np.float64)
    fine_hi = fine_hi.astype(np.float64)
    eff_f = eff_shape_curr.astype(np.float64)
    omin = np.ceil(fine_hi / rel_curr - eff_f - 1e-9)
    omax = np.floor(fine_lo / rel_curr + 1e-9)
    omin = np.maximum(omin, 0.0)
    omax = np.minimum(omax, max_origin.astype(np.float64))
    if fine_roi_lo is not None:
        roi_omin = np.ceil(fine_roi_lo.astype(np.float64) / rel_curr - 1e-9)
        omin = np.maximum(omin, roi_omin)
    if fine_roi_hi_excl is not None:
        roi_omax = np.floor(
            fine_roi_hi_excl.astype(np.float64) / rel_curr - eff_f + 1e-9
        )
        omax = np.minimum(omax, roi_omax)
    if np.any(omin > omax):
        raise ValueError(
            "No patch origin at this scale that covers the finer-level window, fits the volume, "
            "and stays inside the min_center/max_center finest ROI "
            f"(omin={omin.tolist()}, omax={omax.tolist()}, fine_lo={fine_lo.tolist()}, "
            f"fine_hi={fine_hi.tolist()}, rel_curr={rel_curr.tolist()}, "
            f"roi_lo={None if fine_roi_lo is None else fine_roi_lo.tolist()}, "
            f"roi_hi_excl={None if fine_roi_hi_excl is None else fine_roi_hi_excl.tolist()})"
        )
    out = np.empty(len(omin), dtype=np.int64)
    for d in range(len(omin)):
        lo_i = int(omin[d])
        hi_i = int(omax[d])
        out[d] = np.random.randint(lo_i, hi_i + 1)
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
class VolumeInfo:
    """Resolved metadata for a single volume, ready for sampling."""

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
    # Per-scale target output resolution (spatial dims, image storage axis order)
    resolutions: list[np.ndarray]
    # Per-scale chosen image pyramid level to read from
    chosen_levels: list[int]
    # Per-scale chosen label pyramid level (None if no labels)
    label_chosen_levels: list[int] | None
    # Per-scale spatial scale factors of the chosen level relative to level-0 (voxel ratio)
    relative_scale_factors: list[np.ndarray]
    # Per-scale label spatial scale factors of the chosen label level relative to image level-0
    label_relative_scale_factors: list[np.ndarray] | None
    # Per-scale storage read shape (image spatial axis order); resampled to read_shape on read
    read_shapes: list[np.ndarray]
    # Per-scale label storage read shape (label spatial axis order)
    label_read_shapes: list[np.ndarray] | None
    # Valid center coordinate range (spatial dims only, in image spatial axis order, level-0 frame)
    min_center: np.ndarray
    max_center: np.ndarray
    # Output patch size in image spatial axis order (interpolation target for every scale)
    read_shape: list[int]


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
            # Per-scale: target resolution -> chosen source level (voxel size, storage read shape)
            for s in range(len(vi.resolutions)):
                target = vi.resolutions[s].tolist()
                lvl = vi.chosen_levels[s]
                lvl_voxel = (vi.relative_scale_factors[s] * vi.finest_voxel_size).tolist()
                read = vi.read_shapes[s].tolist()
                lines.append(
                    f"    scale {s}: resolution={target}{unit_label} -> level {lvl} "
                    f"(voxel_size={lvl_voxel}, read {read} -> resample to {vi.read_shape})"
                )
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
        read_shape_arr = np.array(read_shape, dtype=np.float64)

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

        # Effective target resolutions for this volume, mapped to image storage spatial order.
        effective_res = self.config.resolutions_for(vol_cfg)
        resolutions: list[np.ndarray] = [
            np.array(
                map_patch_size_to_input(r, img_spatial, output_spatial), dtype=np.float64
            )
            for r in effective_res
        ]

        # Resolve each scale: chosen level, relative factors, storage read shapes.
        chosen_levels: list[int] = []
        relative_scale_factors: list[np.ndarray] = []
        read_shapes: list[np.ndarray] = []
        label_chosen_levels: list[int] | None = [] if label_meta is not None else None
        label_relative_scale_factors: list[np.ndarray] | None = (
            [] if label_meta is not None else None
        )
        label_read_shapes: list[np.ndarray] | None = (
            [] if label_meta is not None else None
        )

        for scale_i, target in enumerate(resolutions):
            img_lvl = _select_level_for_resolution(img_level_voxels, target)
            img_voxel = img_level_voxels[img_lvl]
            chosen_levels.append(img_lvl)
            relative_scale_factors.append(img_voxel / finest_spatial_factors)
            # Storage voxels to read so that, after resampling to read_shape, the output
            # patch has voxel size `target`: read = patch * target / level_voxel.
            read_shapes.append(
                np.ceil(read_shape_arr * target / img_voxel).astype(np.int64)
            )

            if (
                label_meta is not None
                and lbl_level_voxels is not None
                and lbl_sp_idx is not None
            ):
                # Map this scale's target into label spatial order (labels may store axes
                # in a different order than the image).
                lbl_target = np.array(
                    map_patch_size_to_input(
                        effective_res[scale_i], lbl_spatial, output_spatial
                    ),
                    dtype=np.float64,
                )
                lbl_lvl = _select_level_for_resolution(lbl_level_voxels, lbl_target)
                lbl_voxel = lbl_level_voxels[lbl_lvl]
                assert label_chosen_levels is not None
                assert label_relative_scale_factors is not None
                assert label_read_shapes is not None
                label_chosen_levels.append(lbl_lvl)
                label_relative_scale_factors.append(lbl_voxel / finest_spatial_factors)
                label_read_shapes.append(
                    np.ceil(
                        np.array(
                            map_patch_size_to_input(
                                self.config.patch_size, lbl_spatial, output_spatial
                            ),
                            dtype=np.float64,
                        )
                        * lbl_target
                        / lbl_voxel
                    ).astype(np.int64)
                )

        # Compute valid center coordinate range (spatial dims only, level-0 frame)
        finest_spatial_shape = np.array(image_meta.scales[0].shape)[img_sp_idx]
        min_center = np.zeros(len(img_sp_idx), dtype=np.float64)
        max_center = finest_spatial_shape.copy().astype(np.float64)

        for s in range(len(resolutions)):
            rf = relative_scale_factors[s]
            img_sp_shape = np.array(
                image_meta.scales[chosen_levels[s]].shape, dtype=np.float64
            )[img_sp_idx]
            eff_shape = read_shapes[s].astype(np.float64)
            eff_half = np.floor(eff_shape / 2)
            min_center = np.maximum(min_center, rf * eff_half)
            max_center = np.minimum(
                max_center, rf * (img_sp_shape - eff_shape + eff_half)
            )

            if (
                label_meta is not None
                and label_relative_scale_factors is not None
                and label_chosen_levels is not None
                and label_read_shapes is not None
                and lbl_sp_idx is not None
            ):
                lrf = label_relative_scale_factors[s]
                lbl_sp_shape = np.array(
                    label_meta.scales[label_chosen_levels[s]].shape, dtype=np.float64
                )[lbl_sp_idx]
                lbl_eff = label_read_shapes[s].astype(np.float64)
                lbl_half = np.floor(lbl_eff / 2)
                min_center = np.maximum(min_center, lrf * lbl_half)
                max_center = np.minimum(
                    max_center, lrf * (lbl_sp_shape - lbl_eff + lbl_half)
                )

        # Apply optional bounding box constraint
        if vol_cfg.bounding_box is not None:
            bb = np.array(vol_cfg.bounding_box)
            min_center = np.maximum(min_center, bb[:, 0].astype(np.float64))
            max_center = np.minimum(max_center, bb[:, 1].astype(np.float64))

        min_center = np.ceil(min_center).astype(np.int64)
        max_center = np.floor(max_center).astype(np.int64)

        return VolumeInfo(
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
            resolutions=resolutions,
            chosen_levels=chosen_levels,
            label_chosen_levels=label_chosen_levels,
            relative_scale_factors=relative_scale_factors,
            label_relative_scale_factors=label_relative_scale_factors,
            read_shapes=read_shapes,
            label_read_shapes=label_read_shapes,
            min_center=min_center,
            max_center=max_center,
            read_shape=read_shape,
        )

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

            # Scale-0 output voxel size in reference (level-0) voxels per axis.
            ref_per_patch_voxel = vol_info.resolutions[0] / vol_info.finest_voxel_size
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

                # Open the union of chosen pyramid levels across scales (deduped).
                for level in sorted(set(vol_info.chosen_levels)):
                    img_array_path = (
                        zarr_path
                        / vol_info.config.image_key
                        / vol_info.image_meta.scales[level].path
                    )
                    stores[vol_name]["img"][level] = open_store(
                        img_array_path, zarr_ver, ctx
                    )

                if vol_info.config.label_key and vol_info.label_meta and vol_info.label_chosen_levels:
                    for level in sorted(set(vol_info.label_chosen_levels)):
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

        # Reference-frame ROI for any valid center in [min_center, max_center] at the first
        # scale — used to clip random coarse origins when sample_windows is enabled (includes
        # optional bounding_box via min/max_center). The origins produced by
        # _random_patch_origin_covering_fine_extent are in chosen-level voxels, so the ROI is
        # expressed in reference (level-0) voxels via scale-0's relative factors.
        sample_roi_lo: np.ndarray | None = None
        sample_roi_hi_excl: np.ndarray | None = None
        n_scales = len(vol_info.resolutions)
        if self.config.sample_windows and n_scales > 1:
            rel0 = vol_info.relative_scale_factors[0]
            eff0 = vol_info.read_shapes[0]
            h0 = (eff0 // 2).astype(np.float64) * rel0
            eff0_ref = eff0.astype(np.float64) * rel0
            sample_roi_lo = vol_info.min_center.astype(np.float64) - h0
            sample_roi_hi_excl = (
                vol_info.max_center.astype(np.float64) - h0 + eff0_ref
            )

        # Phase 1: Compute slices and issue all reads concurrently via ts.Batch
        img_futures: list[ts.Future] = []
        lbl_futures: list[ts.Future | None] = []

        prev_origin: np.ndarray | None = None
        prev_eff_shape: np.ndarray | None = None
        prev_rel: np.ndarray | None = None

        with ts.Batch() as batch:
            for s in range(n_scales):
                level = vol_info.chosen_levels[s]
                rel_factors = vol_info.relative_scale_factors[s]
                eff_shape = vol_info.read_shapes[s]
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
                    and vol_info.label_chosen_levels is not None
                    and vol_info.label_meta is not None
                    and vol_info.label_relative_scale_factors is not None
                    and vol_info.label_read_shapes is not None
                    and vol_info.lbl_axes is not None
                    and vol_info.lbl_spatial_idx is not None
                ):
                    lbl_level = vol_info.label_chosen_levels[s]
                    lbl_rel_factors = vol_info.label_relative_scale_factors[s]
                    lbl_eff_shape = vol_info.label_read_shapes[s]
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
                "resolutions": [r.tolist() for r in vol_info.resolutions],
                "source_levels": vol_info.chosen_levels,
                # grid_index only present in sequential mode; None breaks DataLoader collation
                **({"grid_index": grid_index} if grid_index is not None else {}),
            },
        }
