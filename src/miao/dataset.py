"""PyTorch Dataset for multi-scale patch sampling from OME-NGFF zarr volumes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.utils.data

from miao.axes import compute_permutation, map_patch_size_to_input, reorient
from miao.config import MiaoConfig, VolumeConfig
from miao.store import create_context, open_store
from miao.zarr_meta import OmeMetadata, ScaleMetadata, read_ome_metadata


@dataclass
class VolumeInfo:
    """Resolved metadata for a single volume, ready for sampling."""

    config: VolumeConfig
    image_meta: OmeMetadata
    label_meta: OmeMetadata | None
    # Per-scale: image scale factors relative to image finest scale
    relative_scale_factors: dict[int, np.ndarray]
    # Source image dtype (for normalization)
    image_dtype: np.dtype
    # Per-scale: label scale factors relative to image finest scale (None if no labels)
    label_relative_scale_factors: dict[int, np.ndarray] | None
    # Valid center coordinate range at finest scale (in input-axis order)
    min_center: np.ndarray
    max_center: np.ndarray
    # Patch size in input-axis (storage) order
    read_shape: list[int]


class VolumeDataset(torch.utils.data.Dataset):
    """Map-style dataset that yields multi-scale random patches from OME-NGFF zarr volumes.

    Each __getitem__ call:
    1. Randomly picks one volume (weighted sampling).
    2. Picks a random coordinate in that volume's finest-scale space.
    3. Extracts patch_size voxels from each requested scale level.
    4. Returns {"img": Tensor, "label": Tensor | None, "meta": dict}.

    Tensor shapes are (1, L, *output_axes_dims) where L = number of scale levels.
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

        # Per-worker tensorstore handle cache (populated lazily)
        self._worker_stores: dict[int, dict] = {}

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

        # Compute patch size in storage (input) axis order
        read_shape = map_patch_size_to_input(
            self.config.patch_size, vol_cfg.axes, self.config.output_axes
        )

        # Compute relative scale factors (relative to image finest requested scale)
        finest_scale = min(vol_cfg.scales)
        finest_factors = np.array(
            image_meta.scales[finest_scale].scale_factors, dtype=np.float64
        )

        relative_scale_factors: dict[int, np.ndarray] = {}
        for level in vol_cfg.scales:
            level_factors = np.array(
                image_meta.scales[level].scale_factors, dtype=np.float64
            )
            relative_scale_factors[level] = level_factors / finest_factors

        # Compute label relative scale factors (label absolute / image finest)
        label_relative_scale_factors: dict[int, np.ndarray] | None = None
        if label_meta is not None:
            label_relative_scale_factors = {}
            for level in vol_cfg.scales:
                lbl_factors = np.array(
                    label_meta.scales[level].scale_factors, dtype=np.float64
                )
                label_relative_scale_factors[level] = lbl_factors / finest_factors

        # Compute valid center coordinate range at finest image scale.
        # For each level, the crop origin at that level is:
        #   origin = floor(center / rel_factor) - read_shape // 2
        # Constraints: origin >= 0 and origin + read_shape <= shape
        # In finest-scale coords:
        #   center >= rel_factor * (read_shape // 2)
        #   center <= rel_factor * (shape - read_shape + read_shape // 2)
        finest_shape = np.array(image_meta.scales[finest_scale].shape)
        read_shape_arr = np.array(read_shape, dtype=np.float64)
        half_patch = np.floor(read_shape_arr / 2)

        min_center = np.zeros_like(finest_shape, dtype=np.float64)
        max_center = finest_shape.copy().astype(np.float64)

        for level in vol_cfg.scales:
            rf = relative_scale_factors[level]
            img_shape = np.array(
                image_meta.scales[level].shape, dtype=np.float64
            )
            min_center = np.maximum(min_center, rf * half_patch)
            max_center = np.minimum(
                max_center, rf * (img_shape - read_shape_arr + half_patch)
            )

            # Label constraint
            if label_meta is not None and label_relative_scale_factors is not None:
                lrf = label_relative_scale_factors[level]
                lbl_shape = np.array(
                    label_meta.scales[level].shape, dtype=np.float64
                )
                min_center = np.maximum(min_center, lrf * half_patch)
                max_center = np.minimum(
                    max_center, lrf * (lbl_shape - read_shape_arr + half_patch)
                )

        # Apply optional bounding box constraint (finest-scale voxels, storage axis order)
        if vol_cfg.bounding_box is not None:
            bb = np.array(vol_cfg.bounding_box)  # shape (ndim, 2)
            min_center = np.maximum(min_center, bb[:, 0].astype(np.float64))
            max_center = np.minimum(max_center, bb[:, 1].astype(np.float64))

        min_center = np.ceil(min_center).astype(np.int64)
        max_center = np.floor(max_center).astype(np.int64)

        return VolumeInfo(
            config=vol_cfg,
            image_meta=image_meta,
            image_dtype=image_meta.scales[finest_scale].dtype,
            label_meta=label_meta,
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

    def __getitem__(self, idx: int) -> dict:
        stores = self._get_stores()

        # Pick a volume based on sampling weights
        vol_idx = np.random.choice(len(self._volumes), p=self._probabilities)
        vol_info = self._volumes[vol_idx]
        vol_stores = stores[vol_info.config.name]

        # Pick a random center coordinate at finest scale (in storage axis order)
        center = np.array(
            [
                np.random.randint(lo, hi + 1)
                for lo, hi in zip(vol_info.min_center, vol_info.max_center)
            ]
        )

        read_shape = np.array(vol_info.read_shape)
        half_patch = read_shape // 2
        perm = compute_permutation(vol_info.config.axes, self.config.output_axes)
        img_crops: list[torch.Tensor] = []
        label_crops: list[torch.Tensor] = []
        bboxes: list[np.ndarray] = []  # per-level: (2, Nd) in output_axes order

        for level in vol_info.config.scales:
            rel_factors = vol_info.relative_scale_factors[level]

            # Convert center to this scale level, then compute crop origin
            center_at_level = np.floor(center / rel_factors).astype(np.int64)
            origin = center_at_level - half_patch

            # Compute world-coordinate bbox (finest-scale voxel space)
            # origin and read_shape are in storage axis order; convert to world coords
            world_min = (origin * rel_factors).astype(np.float64)
            world_max = ((origin + read_shape) * rel_factors).astype(np.float64)
            # Reorder to output_axes
            bbox = np.stack([world_min[list(perm)], world_max[list(perm)]])  # (2, Nd)
            bboxes.append(bbox)

            slices = tuple(
                slice(int(o), int(o + s))
                for o, s in zip(origin, read_shape)
            )

            # Read image crop
            patch = vol_stores["img"][level][slices].read().result()
            patch = reorient(
                np.asarray(patch), vol_info.config.axes, self.config.output_axes
            )
            img_crops.append(torch.from_numpy(patch.copy()))

            # Read label crop (labels may have different scale factors)
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
                    np.asarray(lbl), vol_info.config.axes, self.config.output_axes
                )
                label_crops.append(torch.from_numpy(lbl.copy()))

        # Stack scales: (L, *spatial) -> (1, L, *spatial)
        img_tensor = torch.stack(img_crops).unsqueeze(0).float()
        if vol_info.config.normalize and np.issubdtype(vol_info.image_dtype, np.integer):
            img_tensor = img_tensor / float(np.iinfo(vol_info.image_dtype).max)
        label_tensor = (
            torch.stack(label_crops).unsqueeze(0).long() if label_crops else None
        )

        # Stack bboxes: (L, 2, Nd) in output_axes order, finest-scale voxel coords
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
