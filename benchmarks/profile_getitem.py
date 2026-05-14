"""Per-stage profiling of VolumeDataset.__getitem__().

Instruments each stage of the data loading pipeline to identify bottlenecks:
  1. Store lookup + volume/coordinate selection
  2. TensorStore batch reads (ts.Batch async IO submission)
  3. Data materialization (future.result + np.asarray — actual IO + decompress)
  4. Isotropic resize (F.interpolate trilinear/nearest)
  5. Stack + permute + normalize + return dict construction

Also measures:
  - DataLoader throughput at different worker counts
  - Worker startup cost (_get_stores)
  - Per-scale-level read sizes and IO bandwidth

Outputs:
  - Terminal summary table
  - CSV per-sample timings (benchmarks/results_YYYYMMDD.csv)
  - CSV DataLoader throughput (benchmarks/results_YYYYMMDD_dataloader.csv)

Use generate_report.py to create a PDF report from the CSV output.

Usage:
    python benchmarks/profile_getitem.py --config examples/config.yaml
    python benchmarks/profile_getitem.py --config examples/config.yaml --num_samples 200
    python benchmarks/profile_getitem.py --config examples/config.yaml --num_workers 0,4,8,16

References:
    - GitHub Issue #1: https://github.com/AI-HHMI/miao/issues/1
    - GitHub Issue #8: https://github.com/AI-HHMI/miao/issues/8 (continuous scale sampling)
"""

from __future__ import annotations

import argparse
import csv
import json
import resource
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import tensorstore as ts
import torch
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader

from miao import VolumeDataset, load_config
from miao.axes import (
    compute_permutation,
    map_patch_size_to_input,
    spatial_axes,
    spatial_indices,
)
from miao.config import MiaoConfig
from miao.dataset import VolumeInfo, _normalize_image_tensor
from miao.store import create_context, open_store

WARMUP = 5

# TensorStore metrics we snapshot before/after data materialization
_TS_METRIC_KEYS = (
    "/tensorstore/cache/hit_count",
    "/tensorstore/cache/miss_count",
    "/tensorstore/cache/evict_count",
    "/tensorstore/kvstore/file/bytes_read",
    "/tensorstore/kvstore/file/read",
    "/tensorstore/kvstore/file/read_latency_ms",
)


def _collect_ts_metrics() -> dict[str, Any]:
    """Snapshot TensorStore metrics relevant to IO profiling."""
    result: dict[str, Any] = {}
    try:
        metrics = ts.experimental_collect_matching_metrics("", include_zero_metrics=True)
    except Exception:
        return result
    for m in metrics:
        name = m.get("name", "")
        if name not in _TS_METRIC_KEYS:
            continue
        vals = m.get("values", [])
        if not vals:
            continue
        v = vals[0]
        if "count" in v:
            # Histogram-style metric (e.g., read_latency_ms)
            result[name] = {"count": v.get("count", 0), "mean": v.get("mean", 0.0),
                            "sum": v.get("count", 0) * v.get("mean", 0.0)}
        else:
            result[name] = {"value": v.get("value", 0)}
    return result


def _apply_ts_deltas(timings: "StageTimings", before: dict[str, Any],
                     after: dict[str, Any]) -> None:
    """Compute per-sample deltas between two TensorStore metric snapshots."""
    def _delta_value(key: str) -> int:
        b = before.get(key, {}).get("value", 0)
        a = after.get(key, {}).get("value", 0)
        return a - b

    timings.ts_cache_hit_delta = _delta_value("/tensorstore/cache/hit_count")
    timings.ts_cache_miss_delta = _delta_value("/tensorstore/cache/miss_count")
    timings.ts_cache_evict_delta = _delta_value("/tensorstore/cache/evict_count")
    timings.ts_file_bytes_read_delta = _delta_value("/tensorstore/kvstore/file/bytes_read")
    timings.ts_file_read_count_delta = _delta_value("/tensorstore/kvstore/file/read")

    # Read latency is a histogram: compute mean of new observations in this delta
    lat_key = "/tensorstore/kvstore/file/read_latency_ms"
    b_data = before.get(lat_key, {})
    a_data = after.get(lat_key, {})
    b_count = b_data.get("count", 0)
    a_count = a_data.get("count", 0)
    if a_count > b_count:
        b_sum = b_data.get("sum", 0.0)
        a_sum = a_data.get("sum", 0.0)
        timings.ts_file_read_latency_ms_mean = (a_sum - b_sum) / (a_count - b_count)


@dataclass
class StageTimings:
    """Timing results for a single __getitem__ call, in milliseconds."""

    store_lookup_ms: float = 0.0      # _get_stores + volume/coord selection
    ts_batch_submit_ms: float = 0.0   # ts.Batch() context (async IO submission)
    data_materialize_ms: float = 0.0  # future.result() + np.asarray (actual IO + decompress)
    iso_resize_ms: float = 0.0        # F.interpolate for isotropic (0 if disabled)
    stack_permute_norm_ms: float = 0.0 # stack + permute + normalize + dict
    total_ms: float = 0.0             # wall clock for entire __getitem__

    # Per-level read sizes (in voxels, for read-size analysis)
    read_voxels_per_level: list[int] = field(default_factory=list)
    # Per-level read bytes (for bandwidth estimation)
    read_bytes_total: int = 0

    # --- Data materialize breakdown ---
    # Approach A: per-level, per-type splitting (ms)
    img_future_result_ms_per_level: list[float] = field(default_factory=list)
    lbl_future_result_ms_per_level: list[float] = field(default_factory=list)
    np_asarray_img_ms_per_level: list[float] = field(default_factory=list)
    np_asarray_lbl_ms_per_level: list[float] = field(default_factory=list)

    # Approach B: CPU vs IO separation
    data_materialize_cpu_ms: float = 0.0       # time.process_time() delta
    data_materialize_io_wait_ms: float = 0.0   # wall_time - cpu_time
    rusage_majflt_delta: int = 0               # major page faults (NFS fetches)
    rusage_minflt_delta: int = 0               # minor page faults (cached pages)
    rusage_utime_delta_ms: float = 0.0         # user CPU time delta
    rusage_stime_delta_ms: float = 0.0         # system CPU time delta

    # Approach C: TensorStore metrics deltas (per-sample)
    ts_cache_hit_delta: int = 0
    ts_cache_miss_delta: int = 0
    ts_file_bytes_read_delta: int = 0
    ts_file_read_count_delta: int = 0
    ts_file_read_latency_ms_mean: float = 0.0
    ts_cache_evict_delta: int = 0

    # Per-sample volume name (for per-dataset analysis)
    volume_name: str = ""


def profiled_getitem(dataset: VolumeDataset, idx: int) -> tuple[dict, StageTimings]:
    """Run __getitem__ logic with per-stage timing instrumentation.

    Reimplements the __getitem__ pipeline with timing wrappers around each stage.
    Returns the same output dict as the original plus a StageTimings dataclass.
    """
    timings = StageTimings()
    t_total_start = time.perf_counter()

    # === Stage 1: Store lookup + volume/coord selection ===
    t0 = time.perf_counter()

    stores = dataset._get_stores()

    grid_index = None
    if dataset.config.sampling == "sequential":
        vol_idx, center, grid_index = dataset._grid[idx]
    else:
        vol_idx = np.random.choice(len(dataset._volumes), p=dataset._probabilities)
    vol_info = dataset._volumes[vol_idx]
    timings.volume_name = vol_info.config.name
    vol_stores = stores[vol_info.config.name]
    output_spatial = spatial_axes(dataset.config.output_axes)
    spatial_perm = compute_permutation(vol_info.img_spatial_axes, output_spatial)

    has_img_channel = "c" in vol_info.img_axes
    wants_channel = "c" in dataset.config.output_axes
    squeeze_channel = has_img_channel and not wants_channel
    add_channel = not has_img_channel and wants_channel

    if squeeze_channel:
        img_intermediate = "l" + vol_info.img_spatial_axes
    else:
        img_intermediate = "l" + vol_info.img_axes

    if add_channel:
        img_intermediate_for_perm = img_intermediate + "c"
    else:
        img_intermediate_for_perm = img_intermediate
    img_perm = compute_permutation(img_intermediate_for_perm, dataset.config.output_axes)

    lbl_output_axes = dataset.config.output_axes.replace("c", "")
    has_lbl_channel = vol_info.lbl_axes is not None and "c" in vol_info.lbl_axes
    lbl_perm = None
    if vol_info.lbl_axes is not None:
        if has_lbl_channel:
            lbl_intermediate = "l" + spatial_axes(vol_info.lbl_axes)
        else:
            lbl_intermediate = "l" + vol_info.lbl_axes
        lbl_perm = compute_permutation(lbl_intermediate, lbl_output_axes)

    if dataset.config.sampling == "random":
        center = np.array([
            np.random.randint(lo, hi + 1)
            for lo, hi in zip(vol_info.min_center, vol_info.max_center)
        ])

    read_shape = np.array(vol_info.read_shape)
    half_patch = read_shape // 2
    target_size = tuple(int(s) for s in read_shape)

    timings.store_lookup_ms = (time.perf_counter() - t0) * 1000

    # === Stage 2: TensorStore batch submission ===
    t0 = time.perf_counter()

    img_crops = []
    label_crops = []
    bboxes = []
    img_futures = []
    lbl_futures = []
    eff_shapes_per_level = []
    total_read_bytes = 0

    with ts.Batch() as batch:
        for level in vol_info.config.scales:
            rel_factors = vol_info.relative_scale_factors[level]
            center_at_level = np.floor(center / rel_factors).astype(np.int64)

            if vol_info.iso_read_shapes is not None:
                eff_shape = vol_info.iso_read_shapes[level]
                eff_half = eff_shape // 2
            else:
                eff_shape = read_shape
                eff_half = half_patch

            origin = center_at_level - eff_half
            eff_shapes_per_level.append(eff_shape)

            # Track read size per level
            voxel_count = int(np.prod(eff_shape))
            timings.read_voxels_per_level.append(voxel_count)
            total_read_bytes += voxel_count * np.dtype(vol_info.image_dtype).itemsize

            voxel_size = vol_info.finest_voxel_size
            phys_min = (origin * rel_factors * voxel_size).astype(np.float64)
            phys_max = ((origin + eff_shape) * rel_factors * voxel_size).astype(np.float64)
            bbox = np.stack([phys_min[list(spatial_perm)], phys_max[list(spatial_perm)]])
            bboxes.append(bbox)

            img_slices = dataset._build_img_slices(origin, eff_shape, vol_info)
            img_futures.append(vol_stores["img"][level][img_slices].read(batch=batch))

            if (
                vol_info.config.label_key
                and level in vol_stores["label"]
                and vol_info.label_relative_scale_factors is not None
                and vol_info.lbl_axes is not None
                and vol_info.lbl_spatial_idx is not None
            ):
                lbl_rel_factors = vol_info.label_relative_scale_factors[level]
                lbl_center = np.floor(center / lbl_rel_factors).astype(np.int64)
                lbl_origin = lbl_center - eff_half
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
                if vol_info.label_meta:
                    lbl_dtype = vol_info.label_meta.scales[level].dtype
                    total_read_bytes += voxel_count * np.dtype(lbl_dtype).itemsize
            else:
                lbl_futures.append(None)

    timings.ts_batch_submit_ms = (time.perf_counter() - t0) * 1000
    timings.read_bytes_total = total_read_bytes

    # === Stage 3a: Data materialization (future.result + np.asarray) ===
    # Approach B: capture CPU time and rusage before materialization
    cpu_t0 = time.process_time()
    rusage_before = resource.getrusage(resource.RUSAGE_SELF)
    # Approach C: snapshot TensorStore metrics before
    ts_metrics_before = _collect_ts_metrics()

    t0 = time.perf_counter()

    raw_img_patches = []
    raw_lbl_patches = []
    for img_future, lbl_future in zip(img_futures, lbl_futures):
        # Approach A: time future.result() and np.asarray() separately per level
        t_fr = time.perf_counter()
        img_result = img_future.result()
        timings.img_future_result_ms_per_level.append(
            (time.perf_counter() - t_fr) * 1000)

        t_np = time.perf_counter()
        patch = np.asarray(img_result)
        timings.np_asarray_img_ms_per_level.append(
            (time.perf_counter() - t_np) * 1000)

        if has_img_channel and not wants_channel:
            c_idx = vol_info.img_axes.index("c")
            patch = np.squeeze(patch, axis=c_idx)
        raw_img_patches.append(patch)

        if lbl_future is not None:
            t_fr = time.perf_counter()
            lbl_result = lbl_future.result()
            timings.lbl_future_result_ms_per_level.append(
                (time.perf_counter() - t_fr) * 1000)

            t_np = time.perf_counter()
            lbl = np.asarray(lbl_result)
            timings.np_asarray_lbl_ms_per_level.append(
                (time.perf_counter() - t_np) * 1000)

            if has_lbl_channel and vol_info.lbl_axes is not None:
                c_idx = vol_info.lbl_axes.index("c")
                lbl = np.squeeze(lbl, axis=c_idx)
            raw_lbl_patches.append(lbl)
        else:
            timings.lbl_future_result_ms_per_level.append(0.0)
            timings.np_asarray_lbl_ms_per_level.append(0.0)
            raw_lbl_patches.append(None)

    timings.data_materialize_ms = (time.perf_counter() - t0) * 1000

    # Approach B: compute CPU vs IO wait
    cpu_elapsed = (time.process_time() - cpu_t0) * 1000
    timings.data_materialize_cpu_ms = cpu_elapsed
    timings.data_materialize_io_wait_ms = timings.data_materialize_ms - cpu_elapsed
    rusage_after = resource.getrusage(resource.RUSAGE_SELF)
    timings.rusage_majflt_delta = rusage_after.ru_majflt - rusage_before.ru_majflt
    timings.rusage_minflt_delta = rusage_after.ru_minflt - rusage_before.ru_minflt
    timings.rusage_utime_delta_ms = (rusage_after.ru_utime - rusage_before.ru_utime) * 1000
    timings.rusage_stime_delta_ms = (rusage_after.ru_stime - rusage_before.ru_stime) * 1000

    # Approach C: compute TensorStore metric deltas
    ts_metrics_after = _collect_ts_metrics()
    _apply_ts_deltas(timings, ts_metrics_before, ts_metrics_after)

    # === Stage 3b: Isotropic resize (F.interpolate) ===
    t0 = time.perf_counter()

    for patch, lbl in zip(raw_img_patches, raw_lbl_patches):
        if vol_info.iso_read_shapes is not None:
            has_channel_in_patch = has_img_channel and not squeeze_channel
            if has_channel_in_patch:
                sp_shape = tuple(patch.shape[i] for i in vol_info.img_spatial_idx)
            else:
                sp_shape = tuple(patch.shape)

            if sp_shape != target_size:
                patch_t = torch.from_numpy(patch).float()
                if has_channel_in_patch:
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
                    patch_t = F.interpolate(
                        patch_t.unsqueeze(0).unsqueeze(0),
                        size=target_size, mode="trilinear", align_corners=False,
                    ).squeeze(0).squeeze(0)
                patch = patch_t.numpy()

        img_crops.append(patch)

        if lbl is not None:
            if vol_info.iso_read_shapes is not None:
                if has_lbl_channel and vol_info.lbl_spatial_idx is not None and vol_info.lbl_axes is not None:
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

    timings.iso_resize_ms = (time.perf_counter() - t0) * 1000

    # === Stage 4: Stack + permute + normalize + return dict ===
    t0 = time.perf_counter()

    img_stacked = torch.from_numpy(np.stack(img_crops))
    if add_channel:
        img_stacked = img_stacked.unsqueeze(-1)
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
        label_tensor = torch.empty(0, dtype=torch.long)

    bbox_arr = np.stack(bboxes)
    if dataset.config.bbox_mode == "relative":
        finest_center = (bbox_arr[0, 0] + bbox_arr[0, 1]) / 2.0
        bbox_arr = bbox_arr - finest_center
    bbox_tensor = torch.from_numpy(bbox_arr).float()

    iso_coord = None
    if vol_info.iso_zoom_factors is not None:
        if dataset.config.sampling == "sequential" and dataset._grid_iso_centers:
            iso_coord = dataset._grid_iso_centers[idx].tolist()
        else:
            iso_coord = (center.astype(np.float64) * vol_info.iso_zoom_factors).tolist()

    result = {
        "img": img_tensor,
        "label": label_tensor,
        "bbox": bbox_tensor,
        "meta": {
            "volume": vol_info.config.name,
            "coordinate": center.tolist(),
            "scale_levels": vol_info.config.scales,
            **({"isotropic_coordinate": iso_coord} if iso_coord is not None else {}),
            **({"grid_index": grid_index} if grid_index is not None else {}),
        },
    }

    timings.stack_permute_norm_ms = (time.perf_counter() - t0) * 1000
    timings.total_ms = (time.perf_counter() - t_total_start) * 1000

    return result, timings


def profile_worker_startup(dataset: VolumeDataset) -> float:
    """Measure _get_stores() cold-start cost in milliseconds."""
    dataset._worker_stores.clear()
    t0 = time.perf_counter()
    dataset._get_stores()
    return (time.perf_counter() - t0) * 1000


def run_profiling(
    dataset: VolumeDataset, num_samples: int
) -> list[StageTimings]:
    """Run profiled __getitem__ for num_samples and return all timings."""
    for i in range(WARMUP):
        profiled_getitem(dataset, i)

    timings_list = []
    for i in range(num_samples):
        _, timings = profiled_getitem(dataset, i)
        timings_list.append(timings)

    return timings_list


def benchmark_dataloader_throughput(
    dataset: VolumeDataset,
    num_samples: int,
    num_workers: int,
    batch_size: int = 4,
) -> dict[str, float]:
    """Measure DataLoader throughput (samples/sec) at a given worker count."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        multiprocessing_context="forkserver" if num_workers > 0 else None,
    )
    it = iter(loader)
    for _ in range(WARMUP):
        try:
            next(it)
        except StopIteration:
            break

    num_batches = num_samples // batch_size
    batch_latencies = []
    it = iter(loader)
    for _ in range(num_batches):
        t0 = time.perf_counter()
        try:
            next(it)
        except StopIteration:
            break
        batch_latencies.append((time.perf_counter() - t0) * 1000)

    if not batch_latencies:
        return {"mean_ms": 0, "median_ms": 0, "p95_ms": 0, "samples_per_sec": 0, "n_batches": 0}

    arr = np.array(batch_latencies)
    total_samples = len(batch_latencies) * batch_size
    total_s = float(np.sum(arr) / 1000)
    return {
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "samples_per_sec": total_samples / total_s if total_s > 0 else 0,
        "n_batches": len(batch_latencies),
    }


def print_stage_summary(timings_list: list[StageTimings], label: str = "") -> None:
    """Print summary statistics for per-stage timings."""
    n = len(timings_list)
    if n == 0:
        print("No samples collected.")
        return

    stages = {
        "store_lookup": [t.store_lookup_ms for t in timings_list],
        "ts_batch_submit": [t.ts_batch_submit_ms for t in timings_list],
        "data_materialize": [t.data_materialize_ms for t in timings_list],
        "iso_resize": [t.iso_resize_ms for t in timings_list],
        "stack_perm_norm": [t.stack_permute_norm_ms for t in timings_list],
        "total": [t.total_ms for t in timings_list],
    }

    title = f"Per-stage profiling ({n} samples)"
    if label:
        title += f" — {label}"
    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}")
    print(f"{'Stage':<20s} {'Mean':>8s} {'Median':>8s} {'P95':>8s} {'P99':>8s} {'% of total':>10s}")
    print(f"{'-' * 20} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 10}")

    total_mean = np.mean(stages["total"])
    for name, values in stages.items():
        arr = np.array(values)
        mean = np.mean(arr)
        pct = (mean / total_mean * 100) if total_mean > 0 else 0
        pct_str = f"{pct:.1f}%" if name != "total" else ""
        print(
            f"{name:<20s} "
            f"{mean:>7.2f}ms "
            f"{np.median(arr):>7.2f}ms "
            f"{np.percentile(arr, 95):>7.2f}ms "
            f"{np.percentile(arr, 99):>7.2f}ms "
            f"{pct_str:>10s}"
        )

    all_voxels = [v for t in timings_list for v in t.read_voxels_per_level]
    if all_voxels:
        n_levels = len(timings_list[0].read_voxels_per_level)
        print(f"\nRead sizes per level (voxels):")
        for level_i in range(n_levels):
            level_voxels = [t.read_voxels_per_level[level_i] for t in timings_list]
            mean_v = np.mean(level_voxels)
            print(f"  Level {level_i}: {mean_v:,.0f} voxels/read (avg)")

    all_bytes = [t.read_bytes_total for t in timings_list]
    all_materialize = [t.data_materialize_ms for t in timings_list]
    if all_bytes and all_materialize:
        avg_bytes = np.mean(all_bytes)
        avg_mat_ms = np.mean(all_materialize)
        if avg_mat_ms > 0:
            bandwidth_mbps = (avg_bytes / 1e6) / (avg_mat_ms / 1000)
            print(f"\nEstimated IO bandwidth:")
            print(f"  Avg bytes/sample: {avg_bytes / 1e6:.2f} MB (img + labels, all levels)")
            print(f"  Avg materialize time: {avg_mat_ms:.2f} ms")
            print(f"  Effective bandwidth: {bandwidth_mbps:.1f} MB/s")

    # Data materialize breakdown
    if timings_list[0].img_future_result_ms_per_level:
        n_levels = len(timings_list[0].img_future_result_ms_per_level)
        print(f"\nData materialize breakdown:")

        # Per-level future.result() times
        print(f"  Per-level future.result() (ms):")
        for level_i in range(n_levels):
            img_times = [t.img_future_result_ms_per_level[level_i] for t in timings_list]
            lbl_times = [t.lbl_future_result_ms_per_level[level_i] for t in timings_list]
            print(f"    Level {level_i}: img={np.mean(img_times):.2f}  lbl={np.mean(lbl_times):.2f}")

        # CPU vs IO (note: process_time() sums ALL threads, so CPU > wall when
        # TensorStore uses multi-threaded decompression)
        cpu_arr = np.array([t.data_materialize_cpu_ms for t in timings_list])
        io_arr = np.array([t.data_materialize_io_wait_ms for t in timings_list])
        cpu_mean = np.mean(cpu_arr)
        parallelism = cpu_mean / avg_mat_ms if avg_mat_ms > 0 else 0
        print(f"  CPU vs IO (Approach B):")
        print(f"    CPU time (all threads): {cpu_mean:.2f}ms")
        print(f"    Wall time:              {avg_mat_ms:.2f}ms")
        print(f"    Parallelism ratio:      {parallelism:.1f}x (CPU/wall; >1 = multi-threaded decompress)")
        majflt_arr = np.array([t.rusage_majflt_delta for t in timings_list])
        minflt_arr = np.array([t.rusage_minflt_delta for t in timings_list])
        print(f"    Page faults: major={np.mean(majflt_arr):.1f} (NFS fetches), minor={np.mean(minflt_arr):.0f} (cached)")

        # TensorStore cache metrics
        hits = np.array([t.ts_cache_hit_delta for t in timings_list])
        misses = np.array([t.ts_cache_miss_delta for t in timings_list])
        ts_bytes = np.array([t.ts_file_bytes_read_delta for t in timings_list])
        ts_lat = np.array([t.ts_file_read_latency_ms_mean for t in timings_list])
        print(f"  TensorStore cache:")
        print(f"    Hits: {np.mean(hits):.1f}/sample  Misses: {np.mean(misses):.1f}/sample")
        if np.mean(hits) + np.mean(misses) > 0:
            hit_rate = np.mean(hits) / (np.mean(hits) + np.mean(misses)) * 100
            print(f"    Hit rate: {hit_rate:.0f}%")
        print(f"    File bytes read: {np.mean(ts_bytes)/1e6:.2f} MB/sample")
        print(f"    File read latency: {np.mean(ts_lat):.2f} ms (TensorStore-reported)")

    print()


def _pad3(lst: list, default: float = 0.0) -> list:
    """Pad a list to length 3 for CSV output (supports up to 3 scale levels)."""
    return (lst + [default] * 3)[:3]


def save_stage_csv(timings_list: list[StageTimings], path: str, label: str = "") -> None:
    """Save per-sample stage timings to CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "label", "sample_idx", "volume_name",
            "store_lookup_ms", "ts_batch_submit_ms", "data_materialize_ms",
            "iso_resize_ms", "stack_perm_norm_ms", "total_ms",
            "read_bytes_total",
            "read_voxels_level_0", "read_voxels_level_1", "read_voxels_level_2",
            # Approach A: per-level sub-timers
            "img_fr_ms_l0", "img_fr_ms_l1", "img_fr_ms_l2",
            "lbl_fr_ms_l0", "lbl_fr_ms_l1", "lbl_fr_ms_l2",
            "np_img_ms_l0", "np_img_ms_l1", "np_img_ms_l2",
            "np_lbl_ms_l0", "np_lbl_ms_l1", "np_lbl_ms_l2",
            # Approach B: CPU vs IO
            "cpu_ms", "io_wait_ms", "majflt", "minflt", "utime_ms", "stime_ms",
            # Approach C: TensorStore metrics
            "ts_cache_hits", "ts_cache_misses", "ts_file_bytes",
            "ts_file_reads", "ts_read_lat_ms", "ts_evicts",
        ])
        for i, t in enumerate(timings_list):
            voxels = _pad3(t.read_voxels_per_level, 0)
            img_fr = _pad3(t.img_future_result_ms_per_level)
            lbl_fr = _pad3(t.lbl_future_result_ms_per_level)
            np_img = _pad3(t.np_asarray_img_ms_per_level)
            np_lbl = _pad3(t.np_asarray_lbl_ms_per_level)
            writer.writerow([
                label, i, t.volume_name,
                f"{t.store_lookup_ms:.3f}",
                f"{t.ts_batch_submit_ms:.3f}",
                f"{t.data_materialize_ms:.3f}",
                f"{t.iso_resize_ms:.3f}",
                f"{t.stack_permute_norm_ms:.3f}",
                f"{t.total_ms:.3f}",
                t.read_bytes_total,
                voxels[0], voxels[1], voxels[2],
                # Approach A
                f"{img_fr[0]:.3f}", f"{img_fr[1]:.3f}", f"{img_fr[2]:.3f}",
                f"{lbl_fr[0]:.3f}", f"{lbl_fr[1]:.3f}", f"{lbl_fr[2]:.3f}",
                f"{np_img[0]:.3f}", f"{np_img[1]:.3f}", f"{np_img[2]:.3f}",
                f"{np_lbl[0]:.3f}", f"{np_lbl[1]:.3f}", f"{np_lbl[2]:.3f}",
                # Approach B
                f"{t.data_materialize_cpu_ms:.3f}",
                f"{t.data_materialize_io_wait_ms:.3f}",
                t.rusage_majflt_delta, t.rusage_minflt_delta,
                f"{t.rusage_utime_delta_ms:.3f}",
                f"{t.rusage_stime_delta_ms:.3f}",
                # Approach C
                t.ts_cache_hit_delta, t.ts_cache_miss_delta,
                t.ts_file_bytes_read_delta, t.ts_file_read_count_delta,
                f"{t.ts_file_read_latency_ms_mean:.3f}",
                t.ts_cache_evict_delta,
            ])


def save_dataloader_csv(dl_results: dict[int, dict], path: str, batch_size: int) -> None:
    """Save DataLoader throughput results to CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "num_workers", "batch_size",
            "mean_ms", "median_ms", "p95_ms",
            "samples_per_sec", "n_batches",
        ])
        for nw in sorted(dl_results.keys()):
            stats = dl_results[nw]
            writer.writerow([
                nw, batch_size,
                f"{stats['mean_ms']:.2f}",
                f"{stats['median_ms']:.2f}",
                f"{stats['p95_ms']:.2f}",
                f"{stats['samples_per_sec']:.2f}",
                stats["n_batches"],
            ])


def save_batch_sweep_csv(bs_results: dict[int, dict], path: str, num_workers: int) -> None:
    """Save batch size sweep results to CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "batch_size", "num_workers",
            "mean_ms", "median_ms", "p95_ms",
            "samples_per_sec", "n_batches",
        ])
        for bs in sorted(bs_results.keys()):
            stats = bs_results[bs]
            writer.writerow([
                bs, num_workers,
                f"{stats['mean_ms']:.2f}",
                f"{stats['median_ms']:.2f}",
                f"{stats['p95_ms']:.2f}",
                f"{stats['samples_per_sec']:.2f}",
                stats["n_batches"],
            ])


def save_metadata_json(path: str, config: MiaoConfig, startup_ms: float,
                       num_samples: int, label: str,
                       batch_sweep: bool = False) -> None:
    """Save run metadata (config, startup cost) as JSON for the report generator."""
    meta = {
        "label": label,
        "timestamp": datetime.now().isoformat(),
        "num_samples": num_samples,
        "startup_ms": startup_ms,
        "batch_sweep": batch_sweep,
        "config": {
            "volumes": [v.name for v in config.volumes],
            "patch_size": config.patch_size,
            "n_scales": config.n_scales,
            "isotropic": config.isotropic,
            "cache_bytes": config.cache_bytes,
            "file_io_concurrency": config.file_io_concurrency,
        },
    }
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Per-stage profiling of VolumeDataset.__getitem__()",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", required=True, help="Path to miao config YAML"
    )
    parser.add_argument(
        "--num_samples", type=int, default=100,
        help="Number of samples to profile (default: 100)"
    )
    parser.add_argument(
        "--num_workers", default="0,4,8,12",
        help="Comma-separated worker counts for DataLoader throughput test (default: '0,4,8,12')"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size for DataLoader throughput test (default: 4)"
    )
    parser.add_argument(
        "--output_dir", default="benchmarks",
        help="Base output directory (default: benchmarks/). "
             "Results go into a dated subfolder: benchmarks/run_YYYYMMDD/"
    )
    parser.add_argument(
        "--label", default="",
        help="Label for output files (default: config filename stem). "
             "Results saved as results_{label}.csv, etc."
    )
    parser.add_argument(
        "--cache_comparison", action="store_true",
        help="Run profiling twice: cold cache (fresh dataset) then warm cache "
             "(populated TensorStore cache). Saves results_{label}_cold.csv and "
             "results_{label}_warm.csv"
    )
    parser.add_argument(
        "--batch_sweep", action="store_true",
        help="Sweep batch sizes instead of worker counts. Measures DataLoader "
             "throughput at each batch size with a fixed worker count. "
             "Saves results_{label}_batch_sweep.csv"
    )
    parser.add_argument(
        "--batch_sizes", default="1,2,4,8,16",
        help="Comma-separated batch sizes for batch sweep (default: '1,2,4,8,16')"
    )
    parser.add_argument(
        "--batch_sweep_workers", default="4",
        help="Comma-separated worker counts for batch sweep (default: '4'). "
             "Each worker count produces a separate latency curve."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    worker_counts = [int(w) for w in args.num_workers.split(",")]
    date_str = datetime.now().strftime("%Y%m%d")
    label = args.label or Path(args.config).stem
    output_dir = Path(args.output_dir) / f"run_{date_str}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output file paths
    stage_csv = output_dir / f"results_{label}.csv"
    dl_csv = output_dir / f"results_{label}_dataloader.csv"
    meta_json = output_dir / f"results_{label}_meta.json"

    print(f"\nmiao IO Profiling")
    print(f"{'=' * 60}")
    print(f"Config: {args.config}")
    print(f"Patch size: {config.patch_size}")
    print(f"Scales: {config.n_scales}")
    print(f"Isotropic: {config.isotropic}")
    print(f"Cache: {config.cache_bytes / 1e9:.1f} GB")
    print(f"IO concurrency: {config.file_io_concurrency}")
    print(f"Samples: {args.num_samples}")
    print(f"Workers: {worker_counts}")
    print(f"Label: {label}")
    print(f"Output: {output_dir}/results_{label}*.{{csv,json}}")

    # === Initialize dataset ===
    print(f"\nInitializing dataset...")
    dataset = VolumeDataset(config)

    # === Worker startup cost ===
    print("Measuring worker startup...")
    startup_ms = profile_worker_startup(dataset)
    print(f"  Startup: {startup_ms:.1f}ms")

    # === Per-stage profiling ===
    if args.cache_comparison:
        # Cold run: fresh dataset with empty cache
        print(f"\nRunning COLD cache profiling ({args.num_samples} samples)...")
        cold_label = f"{label}_cold"
        timings_cold = run_profiling(dataset, args.num_samples)
        print_stage_summary(timings_cold, label=cold_label)
        cold_csv = output_dir / f"results_{cold_label}.csv"
        save_stage_csv(timings_cold, str(cold_csv), label=cold_label)
        print(f"Cold cache timings saved to: {cold_csv}")
        save_metadata_json(
            str(output_dir / f"results_{cold_label}_meta.json"),
            config, startup_ms, args.num_samples, cold_label,
        )

        # Warm run: reuse same dataset (TensorStore cache populated)
        print(f"\nRunning WARM cache profiling ({args.num_samples} samples)...")
        warm_label = f"{label}_warm"
        timings_warm = run_profiling(dataset, args.num_samples)
        print_stage_summary(timings_warm, label=warm_label)
        warm_csv = output_dir / f"results_{warm_label}.csv"
        save_stage_csv(timings_warm, str(warm_csv), label=warm_label)
        print(f"Warm cache timings saved to: {warm_csv}")
        save_metadata_json(
            str(output_dir / f"results_{warm_label}_meta.json"),
            config, startup_ms, args.num_samples, warm_label,
        )

        # Use the cold run as the primary result
        timings = timings_cold
        stage_csv = cold_csv
    else:
        print(f"\nRunning per-stage profiling ({args.num_samples} samples)...")
        timings = run_profiling(dataset, args.num_samples)
        print_stage_summary(timings, label=label)

        # Save per-stage CSV
        save_stage_csv(timings, str(stage_csv), label=label)
        print(f"Per-sample timings saved to: {stage_csv}")

    # === DataLoader throughput ===
    print(f"\n{'=' * 60}")
    print(f"DataLoader throughput (batch_size={args.batch_size})")
    print(f"{'=' * 60}")
    print(f"{'Workers':<10s} {'Mean':>10s} {'Median':>10s} {'P95':>10s} {'Samp/sec':>12s}")
    print(f"{'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 12}")

    dl_results: dict[int, dict] = {}
    for nw in worker_counts:
        stats = benchmark_dataloader_throughput(
            dataset, args.num_samples, nw, args.batch_size
        )
        dl_results[nw] = stats
        print(
            f"{nw:<10d} "
            f"{stats['mean_ms']:>9.1f}ms "
            f"{stats['median_ms']:>9.1f}ms "
            f"{stats['p95_ms']:>9.1f}ms "
            f"{stats['samples_per_sec']:>11.1f}"
        )
    print()

    # Save DataLoader CSV
    save_dataloader_csv(dl_results, str(dl_csv), args.batch_size)
    print(f"DataLoader throughput saved to: {dl_csv}")

    # === Batch size sweep (optional) ===
    if args.batch_sweep:
        batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
        sweep_workers = [int(w) for w in args.batch_sweep_workers.split(",")]
        bs_csv = output_dir / f"results_{label}_batch_sweep.csv"

        # Collect results for all (worker_count, batch_size) combinations
        all_bs_results: dict[int, dict[int, dict]] = {}  # nw -> bs -> stats
        for nw in sweep_workers:
            print(f"\n{'=' * 60}")
            print(f"Batch Size Sweep (num_workers={nw})")
            print(f"{'=' * 60}")
            print(f"{'Batch Size':<12s} {'Mean':>10s} {'Median':>10s} {'P95':>10s} {'Samp/sec':>12s}")
            print(f"{'-' * 12} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 12}")

            all_bs_results[nw] = {}
            for bs in batch_sizes:
                stats = benchmark_dataloader_throughput(
                    dataset, args.num_samples, nw, batch_size=bs
                )
                all_bs_results[nw][bs] = stats
                print(
                    f"{bs:<12d} "
                    f"{stats['mean_ms']:>9.1f}ms "
                    f"{stats['median_ms']:>9.1f}ms "
                    f"{stats['p95_ms']:>9.1f}ms "
                    f"{stats['samples_per_sec']:>11.1f}"
                )
            print()

        # Save all results to a single CSV
        with open(str(bs_csv), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "batch_size", "num_workers",
                "mean_ms", "median_ms", "p95_ms",
                "samples_per_sec", "n_batches",
            ])
            for nw in sorted(all_bs_results.keys()):
                for bs in sorted(all_bs_results[nw].keys()):
                    stats = all_bs_results[nw][bs]
                    writer.writerow([
                        bs, nw,
                        f"{stats['mean_ms']:.2f}",
                        f"{stats['median_ms']:.2f}",
                        f"{stats['p95_ms']:.2f}",
                        f"{stats['samples_per_sec']:.2f}",
                        stats["n_batches"],
                    ])
        print(f"Batch size sweep saved to: {bs_csv}")

    # Save run metadata
    save_metadata_json(str(meta_json), config, startup_ms, args.num_samples, label,
                       batch_sweep=args.batch_sweep)
    print(f"Run metadata saved to: {meta_json}")

    # Hint for report generation
    print(f"\nTo generate PDF report from these results:")
    print(f"  python benchmarks/generate_report.py --results_dir {output_dir} --label {label}")
    print(f"\nTo compare multiple runs:")
    print(f"  python benchmarks/generate_report.py --compare {stage_csv} <other_results.csv>")
    print()


if __name__ == "__main__":
    main()
