"""PyTorch Dataset for multi-scale patch sampling from OME-NGFF zarr volumes."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorstore as ts
import torch
import torch.nn.functional as F
import torch.utils.data

from miao.axes import (
    compute_permutation,
    invert_permutation,
    map_patch_size_to_input,
    spatial_axes,
    spatial_indices,
)
from miao.config import MiaoConfig, VolumeConfig
from miao.store import create_context, open_store
from miao.zarr_meta import OmeMetadata, read_ome_metadata


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


class VolumeDataset(torch.utils.data.Dataset):
    """Map-style dataset that yields multi-scale random patches from OME-NGFF zarr volumes.

    Each __getitem__ call:
    1. Randomly picks one volume (weighted sampling).
    2. Picks a random coordinate in that volume's finest-scale space.
    3. Extracts patch_size voxels from each requested scale level.
    4. Returns {"img": Tensor, "label": Tensor | None, "bbox": Tensor, "meta": dict}.

    Tensor shape follows config.output_axes (which must contain 'l' for the scale-level dim).
    A DataLoader adds the leading batch dim, so iterating a DataLoader yields (B, *output_axes_dims).
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
        for i, vi in enumerate(self._volumes):
            finest = min(vi.config.scales)
            finest_meta = vi.image_meta.scales[finest]
            prob = self._probabilities[i]
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
                    f"dtype={lbl_meta.dtype}"
                )
            lines.append(f"    sampling: weight={prob:.2f}, "
                         f"center_range=[{vi.min_center.tolist()}, {vi.max_center.tolist()}]")
            if vi.config.normalize:
                lines.append(f"    normalize: {vi.image_dtype} -> [0, 1]")
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

        image_dtype = image_meta.scales[finest_scale].dtype
        if vol_cfg.normalize and not np.issubdtype(image_dtype, np.integer):
            warnings.warn(
                f"Volume {vol_cfg.name!r}: normalize=True but image dtype is "
                f"{image_dtype} (non-integer). Normalization is a no-op for "
                f"non-integer dtypes; set normalize=False to silence this warning.",
                stacklevel=2,
            )

        relative_scale_factors: dict[int, np.ndarray] = {}
        for level in vol_cfg.scales:
            level_all = np.array(
                image_meta.scales[level].scale_factors, dtype=np.float64
            )
            relative_scale_factors[level] = level_all[img_sp_idx] / finest_spatial_factors

        # Compute label spatial-only relative scale factors
        label_relative_scale_factors: dict[int, np.ndarray] | None = None
        if label_meta is not None and lbl_sp_idx is not None:
            # Labels must share the image's finest-scale physical voxel size,
            # since `center` is sampled in image-finest coordinates and divided
            # by label scale factors to locate label patches.
            lbl_finest_all = np.array(
                label_meta.scales[finest_scale].scale_factors, dtype=np.float64
            )
            lbl_finest_spatial = lbl_finest_all[lbl_sp_idx]
            if not np.allclose(lbl_finest_spatial, finest_spatial_factors):
                raise ValueError(
                    f"Volume {vol_cfg.name!r}: label finest-scale voxel size "
                    f"{lbl_finest_spatial.tolist()} does not match image finest-scale "
                    f"voxel size {finest_spatial_factors.tolist()}. "
                    f"Image and label pyramids must share the same finest-scale physical resolution."
                )
            label_relative_scale_factors = {}
            for level in vol_cfg.scales:
                lbl_all = np.array(
                    label_meta.scales[level].scale_factors, dtype=np.float64
                )
                label_relative_scale_factors[level] = (
                    lbl_all[lbl_sp_idx] / finest_spatial_factors
                )

        # Compute per-level isotropic read shapes if requested
        iso_read_shapes: dict[int, np.ndarray] | None = None
        if self.config.isotropic:
            iso_read_shapes = {}
            for level in vol_cfg.scales:
                rf = relative_scale_factors[level]
                level_voxel = rf * finest_spatial_factors  # absolute voxel size at this level
                target_iso = level_voxel.min()
                iso_read_shapes[level] = np.ceil(
                    np.array(read_shape, dtype=np.float64) * target_iso / level_voxel
                ).astype(np.int64)

        # Compute valid center coordinate range (spatial dims only)
        finest_all_shape = np.array(image_meta.scales[finest_scale].shape)
        finest_spatial_shape = finest_all_shape[img_sp_idx]
        read_shape_arr = np.array(read_shape, dtype=np.float64)

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
            if len(vol_cfg.bounding_box) != len(img_sp_idx):
                raise ValueError(
                    f"Volume {vol_cfg.name!r}: bounding_box has "
                    f"{len(vol_cfg.bounding_box)} entries but image has "
                    f"{len(img_sp_idx)} spatial dims ({img_spatial!r})"
                )
            bb = np.array(vol_cfg.bounding_box)
            min_center = np.maximum(min_center, bb[:, 0].astype(np.float64))
            max_center = np.minimum(max_center, bb[:, 1].astype(np.float64))

        min_center = np.ceil(min_center).astype(np.int64)
        max_center = np.floor(max_center).astype(np.int64)

        if np.any(min_center > max_center):
            raise ValueError(
                f"Volume {vol_cfg.name!r}: no valid center coordinates "
                f"(min={min_center.tolist()}, max={max_center.tolist()}). "
                f"Patch size is too large for this volume or bounding_box is too restrictive."
            )

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
            image_dtype=image_dtype,
            finest_voxel_size=finest_spatial_factors,
            relative_scale_factors=relative_scale_factors,
            label_relative_scale_factors=label_relative_scale_factors,
            min_center=min_center,
            max_center=max_center,
            read_shape=read_shape,
            iso_read_shapes=iso_read_shapes,
        )

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

    @staticmethod
    def _build_slices(
        origin: np.ndarray,
        read_shape: np.ndarray,
        axes: str,
        spatial_idx: list[int],
    ) -> tuple:
        """Build slices for an array: spatial dims get crop, non-spatial dims get slice(None)."""
        slices = []
        sp_i = 0
        for dim_i in range(len(axes)):
            if dim_i in spatial_idx:
                slices.append(slice(int(origin[sp_i]), int(origin[sp_i] + read_shape[sp_i])))
                sp_i += 1
            else:
                slices.append(slice(None))
        return tuple(slices)

    def __getitem__(self, idx: int) -> dict:
        stores = self._get_stores()

        # Pick a volume based on sampling weights
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

        # Label intermediate: "l" + lbl_axes, output = output_axes minus "c"
        lbl_output_axes = self.config.output_axes.replace("c", "")
        lbl_perm: tuple[int, ...] | None = None
        if vol_info.lbl_axes is not None:
            lbl_intermediate = "l" + vol_info.lbl_axes
            lbl_perm = compute_permutation(lbl_intermediate, lbl_output_axes)

        # Pick a random center coordinate at finest scale (spatial dims, image spatial order)
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

        # Phase 1: Compute slices and issue all reads concurrently via ts.Batch
        img_futures: list[ts.Future] = []
        lbl_futures: list[ts.Future | None] = []

        with ts.Batch() as batch:
            for level in vol_info.config.scales:
                rel_factors = vol_info.relative_scale_factors[level]
                center_at_level = np.floor(center / rel_factors).astype(np.int64)

                # Use per-level isotropic read shape if enabled
                if vol_info.iso_read_shapes is not None:
                    eff_shape = vol_info.iso_read_shapes[level]
                    eff_half = eff_shape // 2
                else:
                    eff_shape = read_shape
                    eff_half = half_patch

                origin = center_at_level - eff_half

                voxel_size = vol_info.finest_voxel_size
                if vol_info.iso_read_shapes is not None:
                    # Anchor bbox to the finest-level center so it stays symmetric
                    # across axes; integer rounding of center_at_level would otherwise
                    # shift coarse levels by up to 0.5 * level_voxel per axis.
                    level_voxel = rel_factors * voxel_size
                    target_iso = level_voxel.min()
                    phys_center = center.astype(np.float64) * voxel_size
                    iso_half_extent = np.array(read_shape, dtype=np.float64) * target_iso / 2.0
                    phys_min = (phys_center - iso_half_extent).astype(np.float64)
                    phys_max = (phys_center + iso_half_extent).astype(np.float64)
                else:
                    phys_min = (origin * rel_factors * voxel_size).astype(np.float64)
                    phys_max = ((origin + eff_shape) * rel_factors * voxel_size).astype(np.float64)
                bbox = np.stack(
                    [phys_min[list(spatial_perm)], phys_max[list(spatial_perm)]]
                )
                bboxes.append(bbox)

                img_slices = self._build_slices(
                    origin, eff_shape, vol_info.img_axes, vol_info.img_spatial_idx
                )
                img_futures.append(vol_stores["img"][level][img_slices].read(batch=batch))

                if vol_info.config.label_key and level in vol_stores["label"]:
                    lbl_rel_factors = vol_info.label_relative_scale_factors[level]
                    lbl_center = np.floor(center / lbl_rel_factors).astype(np.int64)
                    lbl_origin = lbl_center - eff_half
                    lbl_slices = self._build_slices(
                        lbl_origin, eff_shape, vol_info.lbl_axes, vol_info.lbl_spatial_idx
                    )
                    lbl_futures.append(vol_stores["label"][level][lbl_slices].read(batch=batch))
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
                        inv_perm = invert_permutation(perm)
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
                # Interpolate labels with nearest-neighbor to preserve integer IDs
                if vol_info.iso_read_shapes is not None and tuple(lbl.shape[-len(target_size):]) != target_size:
                    lbl_t = torch.from_numpy(lbl).float().unsqueeze(0).unsqueeze(0)
                    lbl_t = F.interpolate(
                        lbl_t, size=target_size, mode="nearest",
                    ).squeeze(0).squeeze(0)
                    lbl = lbl_t.numpy().astype(np.int64)
                label_crops.append(lbl)

        # Stack across levels → (L, *storage_axes), then permute to output_axes
        img_stacked = torch.from_numpy(np.stack(img_crops))
        if add_channel:
            img_stacked = img_stacked.unsqueeze(-1)  # add singleton C at end
        img_tensor = img_stacked.permute(img_perm).float()
        if vol_info.config.normalize and np.issubdtype(vol_info.image_dtype, np.integer):
            img_tensor = img_tensor / float(np.iinfo(vol_info.image_dtype).max)
        if label_crops:
            assert lbl_perm is not None, "lbl_perm must be set when labels are present"
            label_tensor = (
                torch.from_numpy(np.stack(label_crops)).permute(lbl_perm).long()
            )
        else:
            label_tensor = None

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
                "scale_levels": vol_info.config.scales,
            },
        }
