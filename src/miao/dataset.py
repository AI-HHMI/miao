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
from miao.zarr_meta import OmeMetadata, ScaleMetadata, read_ome_metadata


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
    # Finest-scale absolute spatial voxel size (physical units, e.g., nanometers)
    finest_voxel_size: np.ndarray
    # Per-scale: spatial-only scale factors relative to image finest scale
    relative_scale_factors: dict[int, np.ndarray]
    # Per-scale: label spatial-only scale factors relative to image finest scale
    label_relative_scale_factors: dict[int, np.ndarray] | None
    # Valid center coordinate range (spatial dims only, in image spatial axis order)
    min_center: np.ndarray
    max_center: np.ndarray
    # Patch size in image spatial axis order
    read_shape: list[int]
    # Per-scale: spatial read shape for isotropic mode (None if isotropic=False)
    iso_read_shapes: dict[int, np.ndarray] | None
    # Per-axis zoom factor from storage to isotropic space (None if isotropic=False)
    # e.g., [5, 1, 1] for voxel sizes [40, 8, 8] nm (Z is 5x coarser)
    iso_zoom_factors: np.ndarray | None = None
    # Smallest safe dtypes for label data (derived from on-disk dtype)
    label_np_dtype: np.dtype = np.dtype(np.int64)
    label_torch_dtype: torch.dtype = torch.int64


def _label_dtypes(source_dtype: np.dtype) -> tuple[np.dtype, torch.dtype]:
    """Pick the smallest safe integer dtypes for label data.

    Maps the on-disk label dtype to the narrowest numpy/torch integer type
    that can represent the full value range without overflow.
    """
    if np.issubdtype(source_dtype, np.floating):
        return np.dtype(np.int64), torch.int64

    info = np.iinfo(source_dtype)
    lo, hi = info.min, info.max
    # Check signed types from narrowest to widest
    for np_dt, torch_dt in [
        (np.int8, torch.int8),
        (np.int16, torch.int16),
        (np.int32, torch.int32),
        (np.int64, torch.int64),
    ]:
        ii = np.iinfo(np_dt)
        if ii.min <= lo and ii.max >= hi:
            return np.dtype(np_dt), torch_dt
    return np.dtype(np.int64), torch.int64


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
        self._grid_iso_centers: list[np.ndarray] = []
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
            iso_tag = " (isotropic grid)" if self.config.isotropic else ""
            mode_str = f"sequential, {len(self._grid)} total positions, overlap={ov}{iso_tag}"
        else:
            mode_str = f"random, {self.config.samples_per_epoch} samples/epoch"
        print(f"VolumeDataset: {len(self._volumes)} volume(s), "
              f"{mode_str}, "
              f"patch_size={self.config.patch_size}")
        for vi in self._volumes:
            finest = min(vi.config.scales)
            finest_meta = vi.image_meta.scales[finest]
            prob = self._probabilities[self._volumes.index(vi)]
            # Get axis unit from OME-NGFF metadata
            units = [ax.get("unit", "") for ax in vi.image_meta.axes]
            unit_str = units[0] if units and all(u == units[0] for u in units if u) else ""
            lines = [
                f"  [{vi.config.name}]",
                f"    image: axes={vi.img_axes!r}, shape={finest_meta.shape}, "
                f"dtype={finest_meta.dtype}",
            ]
            # Per-level voxel resolution
            for level in vi.config.scales:
                sf = vi.image_meta.scales[level].scale_factors
                sp_sf = [sf[i] for i in vi.img_spatial_idx]
                unit_label = f" {unit_str}" if unit_str else ""
                iso_info = ""
                if vi.iso_read_shapes is not None:
                    iso_target = min(sp_sf)
                    iso_voxel = [iso_target] * len(sp_sf)
                    iso_info = f" -> iso {iso_voxel} (read {vi.iso_read_shapes[level].tolist()})"
                lines.append(f"    scale {level}: voxel_size={sp_sf}{unit_label}{iso_info}")
            if vi.label_meta is not None:
                lbl_meta = vi.label_meta.scales[finest]
                lines.append(
                    f"    label: axes={vi.lbl_axes!r}, shape={lbl_meta.shape}, "
                    f"dtype={lbl_meta.dtype} -> {vi.label_np_dtype}"
                )
            lines.append(f"    sampling: weight={prob:.2f}, "
                         f"center_range=[{vi.min_center.tolist()}, {vi.max_center.tolist()}]")
            if self.config.chunk_aligned:
                sp_chunks = [finest_meta.chunks[i] for i in vi.img_spatial_idx]
                lines.append(f"    chunk_aligned: spatial_chunks={sp_chunks}")
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
        """Read OME-NGFF metadata and precompute sampling bounds for a volume."""
        image_meta = read_ome_metadata(
            vol_cfg.path, vol_cfg.image_key, vol_cfg.zarr_version, vol_cfg.scales
        )

        label_meta = None
        if vol_cfg.label_key:
            label_meta = read_ome_metadata(
                vol_cfg.path, vol_cfg.label_key, vol_cfg.zarr_version, vol_cfg.scales
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

        # Compute patch size in image spatial axis order
        read_shape = map_patch_size_to_input(
            self.config.patch_size, img_spatial, output_spatial
        )

        # Compute spatial-only relative scale factors
        finest_scale = min(vol_cfg.scales)
        finest_all_factors = np.array(
            image_meta.scales[finest_scale].scale_factors, dtype=np.float64
        )
        finest_spatial_factors = finest_all_factors[img_sp_idx]

        relative_scale_factors: dict[int, np.ndarray] = {}
        for level in vol_cfg.scales:
            level_all = np.array(
                image_meta.scales[level].scale_factors, dtype=np.float64
            )
            relative_scale_factors[level] = level_all[img_sp_idx] / finest_spatial_factors

        # Compute label spatial-only relative scale factors
        label_relative_scale_factors: dict[int, np.ndarray] | None = None
        if label_meta is not None and lbl_sp_idx is not None:
            label_relative_scale_factors = {}
            for level in vol_cfg.scales:
                lbl_all = np.array(
                    label_meta.scales[level].scale_factors, dtype=np.float64
                )
                label_relative_scale_factors[level] = (
                    lbl_all[lbl_sp_idx] / finest_spatial_factors
                )

        # Compute per-level isotropic read shapes and zoom factors if requested
        iso_read_shapes: dict[int, np.ndarray] | None = None
        iso_zoom_factors: np.ndarray | None = None
        if self.config.isotropic:
            iso_read_shapes = {}
            for level in vol_cfg.scales:
                rf = relative_scale_factors[level]
                level_voxel = rf * finest_spatial_factors  # absolute voxel size at this level
                target_iso = level_voxel.min()
                iso_read_shapes[level] = np.ceil(
                    np.array(read_shape, dtype=np.float64) * target_iso / level_voxel
                ).astype(np.int64)
            # Zoom factor: how many isotropic voxels per storage voxel, per axis
            target_iso_voxel = finest_spatial_factors.min()
            iso_zoom_factors = finest_spatial_factors / target_iso_voxel

        # Compute valid center coordinate range (spatial dims only)
        finest_all_shape = np.array(image_meta.scales[finest_scale].shape)
        finest_spatial_shape = finest_all_shape[img_sp_idx]
        read_shape_arr = np.array(read_shape, dtype=np.float64)
        half_patch = np.floor(read_shape_arr / 2)

        min_center = np.zeros(len(img_sp_idx), dtype=np.float64)
        max_center = finest_spatial_shape.copy().astype(np.float64)

        for level in vol_cfg.scales:
            rf = relative_scale_factors[level]
            img_all_shape = np.array(
                image_meta.scales[level].shape, dtype=np.float64
            )
            img_sp_shape = img_all_shape[img_sp_idx]

            # Use per-level isotropic read shape if enabled
            if iso_read_shapes is not None:
                eff_shape = iso_read_shapes[level].astype(np.float64)
            else:
                eff_shape = read_shape_arr
            eff_half = np.floor(eff_shape / 2)

            min_center = np.maximum(min_center, rf * eff_half)
            max_center = np.minimum(
                max_center, rf * (img_sp_shape - eff_shape + eff_half)
            )

            if label_meta is not None and label_relative_scale_factors is not None and lbl_sp_idx is not None:
                lrf = label_relative_scale_factors[level]
                lbl_all_shape = np.array(
                    label_meta.scales[level].shape, dtype=np.float64
                )
                lbl_sp_shape = lbl_all_shape[lbl_sp_idx]
                min_center = np.maximum(min_center, lrf * eff_half)
                max_center = np.minimum(
                    max_center, lrf * (lbl_sp_shape - eff_shape + eff_half)
                )

        # Apply optional bounding box constraint
        if vol_cfg.bounding_box is not None:
            bb = np.array(vol_cfg.bounding_box)
            min_center = np.maximum(min_center, bb[:, 0].astype(np.float64))
            max_center = np.minimum(max_center, bb[:, 1].astype(np.float64))

        min_center = np.ceil(min_center).astype(np.int64)
        max_center = np.floor(max_center).astype(np.int64)

        # Derive smallest safe label dtypes from on-disk dtype
        if label_meta is not None:
            lbl_np_dt, lbl_torch_dt = _label_dtypes(
                label_meta.scales[finest_scale].dtype
            )
        else:
            lbl_np_dt, lbl_torch_dt = np.dtype(np.int64), torch.int64

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
            image_dtype=image_meta.scales[finest_scale].dtype,
            finest_voxel_size=finest_spatial_factors,
            relative_scale_factors=relative_scale_factors,
            label_relative_scale_factors=label_relative_scale_factors,
            min_center=min_center,
            max_center=max_center,
            read_shape=read_shape,
            iso_read_shapes=iso_read_shapes,
            iso_zoom_factors=iso_zoom_factors,
            label_np_dtype=lbl_np_dt,
            label_torch_dtype=lbl_torch_dt,
        )

    def _build_sequential_grid(self) -> None:
        """Precompute flat list of (vol_idx, center, grid_index) for sequential sampling.

        center is in image spatial axis order (matching min_center/max_center).
        grid_index is a tuple of per-axis position indices within the grid for that volume,
        useful for stitching patch predictions back into a full-volume output.

        When isotropic=True, the grid is built in isotropic output space so that
        stride and overlap semantics match the isotropic output tensor. Grid positions
        are converted back to storage coordinates for reading. Isotropic centers are
        stored in self._grid_iso_centers for inclusion in meta["isotropic_coordinate"].
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

            if vol_info.iso_zoom_factors is not None:
                # --- Isotropic mode: build grid in isotropic output space ---
                zoom = vol_info.iso_zoom_factors
                iso_min = vol_info.min_center * zoom
                iso_max = vol_info.max_center * zoom

                # patch_size mapped to input (storage) axis order — this is the
                # isotropic output size per axis (what the caller gets after interpolation)
                patch_input = map_patch_size_to_input(
                    self.config.patch_size, vol_info.img_spatial_axes, output_spatial
                )
                iso_stride = np.array(patch_input) - np.array(overlap_input)

                ranges: list[list[int]] = []
                for i in range(len(patch_input)):
                    lo = int(iso_min[i])
                    hi = int(iso_max[i])
                    positions = list(range(lo, hi + 1, int(iso_stride[i])))
                    if not positions:
                        raise ValueError(
                            f"Volume {vol_info.config.name!r}: no valid center positions "
                            f"along spatial axis {i} in isotropic space "
                            f"(iso_min={lo} > iso_max={hi}). "
                            f"Volume may be too small for the requested patch_size."
                        )
                    if positions[-1] < hi:
                        positions.append(hi)
                    ranges.append(positions)

                for grid_idx in iproduct(*[range(len(r)) for r in ranges]):
                    iso_center = np.array(
                        [ranges[ax][grid_idx[ax]] for ax in range(len(ranges))]
                    )
                    storage_center = np.round(iso_center / zoom).astype(np.int64)
                    storage_center = np.clip(
                        storage_center, vol_info.min_center, vol_info.max_center
                    )
                    self._grid.append((vol_idx, storage_center, grid_idx))
                    self._grid_iso_centers.append(iso_center)
            else:
                # --- Non-isotropic: build grid in storage space ---
                stride = np.array(vol_info.read_shape) - np.array(overlap_input)

                ranges: list[list[int]] = []
                for i in range(len(vol_info.read_shape)):
                    lo = int(vol_info.min_center[i])
                    hi = int(vol_info.max_center[i])
                    positions = list(range(lo, hi + 1, int(stride[i])))
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

                for level in vol_info.config.scales:
                    img_array_path = (
                        zarr_path
                        / vol_info.config.image_key
                        / vol_info.image_meta.scales[level].path
                    )
                    stores[vol_name]["img"][level] = open_store(
                        img_array_path, zarr_ver, ctx,
                    )

                    if vol_info.config.label_key and vol_info.label_meta:
                        lbl_array_path = (
                            zarr_path
                            / vol_info.config.label_key
                            / vol_info.label_meta.scales[level].path
                        )
                        stores[vol_name]["label"][level] = open_store(
                            lbl_array_path, zarr_ver, ctx,
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

    def _sample_chunk_aligned_center(self, vol_info: VolumeInfo) -> np.ndarray:
        """Sample a random patch center constrained to lie within a single chunk.

        For each spatial axis:
          - If chunk_size >= patch_size: pick a random chunk, then a random
            center within it such that the full patch fits inside one chunk.
          - If chunk_size < patch_size: fall back to unconstrained random.

        Returns center array in image spatial axis order.
        """
        finest_scale = min(vol_info.config.scales)
        full_chunks = vol_info.image_meta.scales[finest_scale].chunks
        spatial_chunks = [full_chunks[i] for i in vol_info.img_spatial_idx]

        center = np.empty(len(vol_info.img_spatial_idx), dtype=np.int64)

        for ax in range(len(vol_info.img_spatial_idx)):
            chunk_sz = spatial_chunks[ax]
            patch_sz = vol_info.read_shape[ax]
            lo = int(vol_info.min_center[ax])
            hi = int(vol_info.max_center[ax])

            if chunk_sz < patch_sz:
                center[ax] = np.random.randint(lo, hi + 1)
                continue

            half = patch_sz // 2
            spatial_extent = vol_info.image_meta.scales[finest_scale].shape[
                vol_info.img_spatial_idx[ax]
            ]
            n_chunks = int(np.ceil(spatial_extent / chunk_sz))

            valid_chunks: list[tuple[int, int]] = []
            for ci in range(n_chunks):
                chunk_start = ci * chunk_sz
                # Valid center range within this chunk (patch fits entirely):
                c_lo = chunk_start + half
                c_hi = chunk_start + chunk_sz - patch_sz + half
                if c_hi < c_lo:
                    continue
                # Clamp to volume's valid center range
                c_lo = max(c_lo, lo)
                c_hi = min(c_hi, hi)
                if c_lo <= c_hi:
                    valid_chunks.append((c_lo, c_hi))

            if not valid_chunks:
                center[ax] = np.random.randint(lo, hi + 1)
                continue

            # Weight chunks by valid range width for uniform spatial coverage
            widths = np.array(
                [c_hi - c_lo + 1 for c_lo, c_hi in valid_chunks], dtype=np.float64
            )
            probs = widths / widths.sum()
            chunk_idx = np.random.choice(len(valid_chunks), p=probs)
            c_lo, c_hi = valid_chunks[chunk_idx]
            center[ax] = np.random.randint(c_lo, c_hi + 1)

        return center

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
            if self.config.chunk_aligned:
                center = self._sample_chunk_aligned_center(vol_info)
            else:
                center = np.array(
                    [
                        np.random.randint(lo, hi + 1)
                        for lo, hi in zip(vol_info.min_center, vol_info.max_center)
                    ]
                )

        read_shape = np.array(vol_info.read_shape)
        half_patch = read_shape // 2
        target_size = tuple(int(s) for s in read_shape)  # interpolation target
        img_crops: list[np.ndarray] = []
        label_crops: list[np.ndarray] = []
        bboxes: list[np.ndarray] = []

        # Finest-coordinate ROI for any valid finest center in [min_center, max_center] at the
        # first (finest) scale in config order — used to clip random coarse origins when
        # sample_windows is enabled (includes optional bounding_box via min/max_center).
        sample_roi_lo: np.ndarray | None = None
        sample_roi_hi_excl: np.ndarray | None = None
        if self.config.sample_windows and len(vol_info.config.scales) > 1:
            level0 = vol_info.config.scales[0]
            if vol_info.iso_read_shapes is not None:
                eff0 = vol_info.iso_read_shapes[level0]
            else:
                eff0 = read_shape
            h0 = eff0 // 2
            sample_roi_lo = vol_info.min_center.astype(np.float64) - h0.astype(np.float64)
            sample_roi_hi_excl = (
                vol_info.max_center.astype(np.float64)
                - h0.astype(np.float64)
                + eff0.astype(np.float64)
            )

        # Phase 1: Compute slices and issue all reads concurrently via ts.Batch
        level_info: list[dict] = []
        img_futures: list[ts.Future] = []
        lbl_futures: list[ts.Future | None] = []

        prev_origin: np.ndarray | None = None
        prev_eff_shape: np.ndarray | None = None
        prev_rel: np.ndarray | None = None

        with ts.Batch() as batch:
            for level_i, level in enumerate(vol_info.config.scales):
                rel_factors = vol_info.relative_scale_factors[level]

                # Use per-level isotropic read shape if enabled
                if vol_info.iso_read_shapes is not None:
                    eff_shape = vol_info.iso_read_shapes[level]
                    eff_half = eff_shape // 2
                else:
                    eff_shape = read_shape
                    eff_half = half_patch

                if self.config.sample_windows and level_i > 0:
                    assert prev_origin is not None and prev_eff_shape is not None and prev_rel is not None
                    if not np.all(rel_factors + 1e-9 >= prev_rel):
                        raise AssertionError(
                            "sample_windows requires each volume's `scales` list to be ordered from "
                            "higher resolution to lower (non-decreasing relative_scale_factors per axis). "
                            f"At index {level_i - 1} -> {level_i}, prev_rel={prev_rel.tolist()} but "
                            f"current rel={rel_factors.tolist()} for scale level {level!r}."
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
                    and level in vol_stores["label"]
                    and vol_info.label_meta is not None
                    and vol_info.label_relative_scale_factors is not None
                    and vol_info.lbl_axes is not None
                    and vol_info.lbl_spatial_idx is not None
                ):
                    lbl_rel_factors = vol_info.label_relative_scale_factors[level]
                    if self.config.sample_windows and level_i > 0:
                        center_fine = (
                            origin.astype(np.float64) + eff_half.astype(np.float64)
                        ) * rel_factors
                        lbl_center = np.floor(center_fine / lbl_rel_factors).astype(
                            np.int64
                        )
                        lbl_origin = lbl_center - eff_half
                    else:
                        lbl_center = np.floor(center / lbl_rel_factors).astype(np.int64)
                        lbl_origin = lbl_center - eff_half
                    # Build label slices from label axes metadata:
                    # spatial dims get cropped; non-spatial dims (e.g. channel) take all.
                    lbl_slices = []
                    sp_i = 0
                    for dim_i, _ax_char in enumerate(vol_info.lbl_axes):
                        if dim_i in vol_info.lbl_spatial_idx:
                            lbl_slices.append(
                                slice(int(lbl_origin[sp_i]), int(lbl_origin[sp_i] + eff_shape[sp_i]))
                            )
                            sp_i += 1
                        else:
                            lbl_slices.append(slice(None))
                    lbl_futures.append(vol_stores["label"][level][tuple(lbl_slices)].read(batch=batch))
                else:
                    lbl_futures.append(None)

        # Phase 2: Collect results (all reads already completed when batch exited)
        for img_future, lbl_future in zip(img_futures, lbl_futures):
            patch = np.asarray(img_future.result())
            if has_img_channel and not wants_channel:
                c_idx = vol_info.img_axes.index("c")
                patch = np.squeeze(patch, axis=c_idx)

            # Interpolate to target patch_size if isotropic mode changed the read shape
            if vol_info.iso_read_shapes is not None:
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

                # Interpolate labels with nearest-neighbor to preserve integer IDs
                if vol_info.iso_read_shapes is not None:
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
                        lbl = lbl_t.numpy().astype(vol_info.label_np_dtype)
                label_crops.append(lbl)

        # Stack across levels → (L, *storage_axes), then permute to output_axes
        img_stacked = torch.from_numpy(np.stack(img_crops))
        if add_channel:
            img_stacked = img_stacked.unsqueeze(-1)  # add singleton C at end
        from miao.config import IMAGE_DTYPE_MAP
        _img_dtype = IMAGE_DTYPE_MAP[self.config.image_dtype]
        img_tensor = img_stacked.permute(img_perm).to(_img_dtype)
        img_tensor = _normalize_image_tensor(
            img_tensor,
            normalize=vol_info.config.normalize,
            normalize_min=vol_info.config.normalize_min,
            normalize_max=vol_info.config.normalize_max,
            image_dtype=vol_info.image_dtype,
        )
        if label_crops:
            label_tensor = torch.from_numpy(np.stack(label_crops)).permute(lbl_perm).to(vol_info.label_torch_dtype)
        else:
            # Keep output collate-friendly for default PyTorch DataLoader.
            label_tensor = torch.empty(0, dtype=vol_info.label_torch_dtype)

        # Stack bboxes: (L, 2, Nd_spatial) in output spatial order, physical units
        bbox_arr = np.stack(bboxes)
        if self.config.bbox_mode == "relative":
            # Make relative to the center of the finest-level crop
            finest_center = (bbox_arr[0, 0] + bbox_arr[0, 1]) / 2.0
            bbox_arr = bbox_arr - finest_center
        bbox_tensor = torch.from_numpy(bbox_arr).float()

        # Compute isotropic coordinate if applicable
        iso_coord = None
        if vol_info.iso_zoom_factors is not None:
            if self.config.sampling == "sequential" and self._grid_iso_centers:
                iso_coord = self._grid_iso_centers[idx].tolist()
            else:
                # Random mode: convert storage center to isotropic space
                iso_coord = (center.astype(np.float64) * vol_info.iso_zoom_factors).tolist()

        return {
            "img": img_tensor,
            "label": label_tensor,
            "bbox": bbox_tensor,
            "meta": {
                "volume": vol_info.config.name,
                "coordinate": center.tolist(),
                "scale_levels": vol_info.config.scales,
                **({"isotropic_coordinate": iso_coord} if iso_coord is not None else {}),
                # grid_index only present in sequential mode; None breaks DataLoader collation
                **({"grid_index": grid_index} if grid_index is not None else {}),
            },
        }
