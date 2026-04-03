"""PyTorch Dataset for multi-scale patch sampling from OME-NGFF zarr volumes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
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
    # Per-scale: spatial-only scale factors relative to image finest scale
    relative_scale_factors: dict[int, np.ndarray]
    # Per-scale: label spatial-only scale factors relative to image finest scale
    label_relative_scale_factors: dict[int, np.ndarray] | None
    # Valid center coordinate range (spatial dims only, in image spatial axis order)
    min_center: np.ndarray
    max_center: np.ndarray
    # Patch size in image spatial axis order
    read_shape: list[int]


class VolumeDataset(torch.utils.data.Dataset):
    """Map-style dataset that yields multi-scale random patches from OME-NGFF zarr volumes.

    Each __getitem__ call:
    1. Randomly picks one volume (weighted sampling).
    2. Picks a random coordinate in that volume's finest-scale space.
    3. Extracts patch_size voxels from each requested scale level.
    4. Returns {"img": Tensor, "label": Tensor | None, "bbox": Tensor, "meta": dict}.

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

        # Print summary
        self._print_summary()

        # Per-worker tensorstore handle cache (populated lazily)
        self._worker_stores: dict[int, dict] = {}

    def _print_summary(self) -> None:
        """Print a summary of detected metadata for all volumes."""
        print(f"VolumeDataset: {len(self._volumes)} volume(s), "
              f"{self.config.samples_per_epoch} samples/epoch, "
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
                lines.append(f"    scale {level}: voxel_size={sp_sf}{unit_label}")
            if vi.label_meta is not None:
                lbl_meta = vi.label_meta.scales[finest]
                lines.append(
                    f"    label: axes={vi.lbl_axes!r}, shape={lbl_meta.shape}, "
                    f"dtype={lbl_meta.dtype}"
                )
            lines.append(f"    sampling: weight={prob:.2f}, "
                         f"center_range=[{vi.min_center.tolist()}, {vi.max_center.tolist()}]")
            if vi.config.normalize:
                lines.append(f"    normalize: {vi.image_dtype} -> [0, 1]")
            print("\n".join(lines))
        output_spatial = spatial_axes(self.config.output_axes)
        has_c = "c" in self.config.output_axes
        shape_desc = f"(1, L, {', '.join(output_spatial.upper())}{''.join(', C' if has_c else '')})"
        print(f"  output: axes={self.config.output_axes!r}, tensor_shape={shape_desc}")

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
            min_center = np.maximum(min_center, rf * half_patch)
            max_center = np.minimum(
                max_center, rf * (img_sp_shape - read_shape_arr + half_patch)
            )

            if label_meta is not None and label_relative_scale_factors is not None and lbl_sp_idx is not None:
                lrf = label_relative_scale_factors[level]
                lbl_all_shape = np.array(
                    label_meta.scales[level].shape, dtype=np.float64
                )
                lbl_sp_shape = lbl_all_shape[lbl_sp_idx]
                min_center = np.maximum(min_center, lrf * half_patch)
                max_center = np.minimum(
                    max_center, lrf * (lbl_sp_shape - read_shape_arr + half_patch)
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
            image_dtype=image_meta.scales[finest_scale].dtype,
            relative_scale_factors=relative_scale_factors,
            label_relative_scale_factors=label_relative_scale_factors,
            min_center=min_center,
            max_center=max_center,
            read_shape=read_shape,
        )

    def _get_worker_id(self) -> int:
        worker_info = torch.utils.data.get_worker_info()
        return worker_info.id if worker_info is not None else 0

    def _get_stores(self) -> dict:
        """Lazily create tensorstore handles for the current worker."""
        worker_id = self._get_worker_id()
        if worker_id not in self._worker_stores:
            ctx = create_context(self.config.cache_bytes)
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
                        img_array_path, zarr_ver, ctx
                    )

                    if vol_info.config.label_key and vol_info.label_meta:
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

    def _reorient_img(self, patch: np.ndarray, vol_info: VolumeInfo) -> np.ndarray:
        """Reorient image patch to output axes order.

        If image has channel but output_axes doesn't, squeeze channel after read.
        """
        output_spatial = spatial_axes(self.config.output_axes)
        has_img_channel = "c" in vol_info.img_axes
        wants_channel = "c" in self.config.output_axes

        if has_img_channel and not wants_channel:
            # Squeeze channel dim, then reorient spatial only
            c_idx = vol_info.img_axes.index("c")
            patch = np.squeeze(patch, axis=c_idx)
            return reorient(patch, vol_info.img_spatial_axes, output_spatial)
        elif has_img_channel and wants_channel:
            # Reorient full axes including channel
            return reorient(patch, vol_info.img_axes, self.config.output_axes)
        else:
            # No channel anywhere, reorient spatial only
            return reorient(patch, vol_info.img_spatial_axes, output_spatial)

    def __getitem__(self, idx: int) -> dict:
        stores = self._get_stores()

        # Pick a volume based on sampling weights
        vol_idx = np.random.choice(len(self._volumes), p=self._probabilities)
        vol_info = self._volumes[vol_idx]
        vol_stores = stores[vol_info.config.name]
        output_spatial = spatial_axes(self.config.output_axes)

        # Pick a random center coordinate at finest scale (spatial dims, image spatial order)
        center = np.array(
            [
                np.random.randint(lo, hi + 1)
                for lo, hi in zip(vol_info.min_center, vol_info.max_center)
            ]
        )

        read_shape = np.array(vol_info.read_shape)
        half_patch = read_shape // 2
        spatial_perm = compute_permutation(vol_info.img_spatial_axes, output_spatial)
        img_crops: list[torch.Tensor] = []
        label_crops: list[torch.Tensor] = []
        bboxes: list[np.ndarray] = []

        for level in vol_info.config.scales:
            rel_factors = vol_info.relative_scale_factors[level]

            # Convert center to this scale level, then compute crop origin
            center_at_level = np.floor(center / rel_factors).astype(np.int64)
            origin = center_at_level - half_patch

            # Compute world-coordinate bbox (spatial only, in output order)
            world_min = (origin * rel_factors).astype(np.float64)
            world_max = ((origin + read_shape) * rel_factors).astype(np.float64)
            bbox = np.stack(
                [world_min[list(spatial_perm)], world_max[list(spatial_perm)]]
            )
            bboxes.append(bbox)

            # Read image crop (spatial dims get crop, channel gets slice(None))
            img_slices = self._build_img_slices(origin, read_shape, vol_info)
            patch = vol_stores["img"][level][img_slices].read().result()
            patch = self._reorient_img(np.asarray(patch), vol_info)
            img_crops.append(torch.from_numpy(patch.copy()))

            # Read label crop (labels are spatial-only)
            if vol_info.config.label_key and level in vol_stores["label"]:
                lbl_rel_factors = vol_info.label_relative_scale_factors[level]
                lbl_center = np.floor(center / lbl_rel_factors).astype(np.int64)
                lbl_origin = lbl_center - half_patch
                lbl_slices = tuple(
                    slice(int(o), int(o + s))
                    for o, s in zip(lbl_origin, read_shape)
                )
                lbl = vol_stores["label"][level][lbl_slices].read().result()
                lbl = reorient(
                    np.asarray(lbl), vol_info.lbl_spatial_axes, output_spatial
                )
                label_crops.append(torch.from_numpy(lbl.copy()))

        # Stack scales: (L, *dims) -> (1, L, *dims)
        img_tensor = torch.stack(img_crops).unsqueeze(0).float()
        if vol_info.config.normalize and np.issubdtype(vol_info.image_dtype, np.integer):
            img_tensor = img_tensor / float(np.iinfo(vol_info.image_dtype).max)
        label_tensor = (
            torch.stack(label_crops).unsqueeze(0).long() if label_crops else None
        )

        # Stack bboxes: (L, 2, Nd_spatial) in output spatial order
        bbox_tensor = torch.from_numpy(np.stack(bboxes)).float()

        return {
            "img": img_tensor,
            "label": label_tensor,
            "bbox": bbox_tensor,
            "meta": {
                "volume": vol_info.config.name,
                "coordinate": center.tolist(),
                "scale_levels": vol_info.config.scales,
            },
        }
