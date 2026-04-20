"""Benchmark read latency of VolumeDataset.

Usage:
    python benchmarks/benchmark.py --config examples/config.yaml
    python benchmarks/benchmark.py --config examples/config.yaml --num_samples 200 --num_workers 0,4,8
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from miao import VolumeDataset, load_config

RESULTS_DIR = Path(__file__).parent / "results"

WARMUP = 5


def benchmark_single_sample(
    dataset: VolumeDataset, num_samples: int
) -> dict[str, float]:
    """Benchmark dataset[i] calls without DataLoader."""
    # Warmup
    for i in range(WARMUP):
        _ = dataset[i]

    latencies = []
    for i in range(num_samples):
        t0 = time.perf_counter()
        _ = dataset[i]
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    arr = np.array(latencies)
    return {
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "total_s": float(np.sum(arr) / 1000),
        "n": num_samples,
    }


def benchmark_dataloader(
    dataset: VolumeDataset,
    num_samples: int,
    num_workers: int,
    batch_size: int = 4,
) -> dict[str, float]:
    """Benchmark DataLoader throughput."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        multiprocessing_context="forkserver" if num_workers > 0 else None,
    )

    # Warmup + drain the prefetch queue so timed batches reflect true wait time
    # rather than shared-memory dequeue of already-prefetched batches.
    prefetch = getattr(loader, "prefetch_factor", 2) or 2
    drain = max(num_workers, 1) * prefetch
    it = iter(loader)
    for _ in range(WARMUP + drain):
        try:
            _ = next(it)
        except StopIteration:
            break

    # Benchmark
    num_batches = num_samples // batch_size
    batch_latencies = []
    for _ in range(num_batches):
        t0 = time.perf_counter()
        try:
            _ = next(it)
        except StopIteration:
            break
        batch_latencies.append((time.perf_counter() - t0) * 1000)

    if not batch_latencies:
        return {
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "samples_per_sec": 0.0,
            "n_batches": 0,
        }

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


def save_plot(
    worker_counts: list[int],
    stats_per_workers: list[dict[str, float]],
    config_path: str,
    patch_size: list[int],
    n_scales: int,
    n_volumes: int,
    batch_size: int,
    num_samples: int,
) -> Path:
    """Save a chart of workers -> batch latency under benchmarks/results/."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    mean_ms = [s["mean_ms"] for s in stats_per_workers]
    p95_ms = [s["p95_ms"] for s in stats_per_workers]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(worker_counts, mean_ms, marker="o", label="mean")
    ax.plot(worker_counts, p95_ms, marker="s", linestyle="--", label="p95")
    ax.set_xlabel("num_workers")
    ax.set_ylabel("batch latency (ms)")
    ax.set_xticks(worker_counts)
    ax.grid(True, alpha=0.3)
    ax.legend()

    config_name = Path(config_path).stem
    title = (
        f"VolumeDataset benchmark — {config_name}\n"
        f"patch={patch_size}  scales={n_scales}  volumes={n_volumes}  "
        f"batch_size={batch_size}  samples={num_samples}"
    )
    ax.set_title(title, fontsize=10)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = RESULTS_DIR / f"{config_name}_bs{batch_size}_n{num_samples}_{timestamp}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Benchmark VolumeDataset read latency")
    parser.add_argument(
        "--config", default="examples/config.yaml", help="Path to miao config YAML"
    )
    parser.add_argument(
        "--num_samples", type=int, default=100, help="Number of samples to benchmark"
    )
    parser.add_argument(
        "--num_workers",
        default="0,4,8,16",
        help="Comma-separated list of num_workers values to test",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for DataLoader benchmark"
    )
    args = parser.parse_args()

    worker_counts = [int(w) for w in args.num_workers.split(",")]

    config = load_config(args.config)
    dataset = VolumeDataset(config)

    n_volumes = len(config.volumes)
    n_scales = config.n_scales
    patch_size = config.patch_size
    print(f"\nBenchmark: VolumeDataset read latency")
    print(f"Config: {args.config}")
    print(f"Patch size: {patch_size}, Scales: {n_scales}, Volumes: {n_volumes}")

    # Single-sample benchmark
    print(f"\nSingle-sample (no DataLoader, {args.num_samples} samples, {WARMUP} warmup):")
    stats = benchmark_single_sample(dataset, args.num_samples)
    print(
        f"  mean={stats['mean_ms']:.1f}ms  "
        f"median={stats['median_ms']:.1f}ms  "
        f"p95={stats['p95_ms']:.1f}ms  "
        f"p99={stats['p99_ms']:.1f}ms  "
        f"total={stats['total_s']:.1f}s"
    )

    # DataLoader benchmark
    print(f"\nDataLoader throughput (batch_size={args.batch_size}):")
    all_stats: list[dict[str, float]] = []
    for nw in worker_counts:
        stats = benchmark_dataloader(dataset, args.num_samples, nw, args.batch_size)
        all_stats.append(stats)
        print(
            f"  workers={nw:<3d} "
            f"mean={stats['mean_ms']:.1f}ms/batch  "
            f"median={stats['median_ms']:.1f}ms/batch  "
            f"p95={stats['p95_ms']:.1f}ms/batch  "
            f"{stats['samples_per_sec']:.1f} samples/sec  "
            f"({stats['n_batches']} batches)"
        )

    plot_path = save_plot(
        worker_counts=worker_counts,
        stats_per_workers=all_stats,
        config_path=args.config,
        patch_size=patch_size,
        n_scales=n_scales,
        n_volumes=n_volumes,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
    )
    print(f"\nSaved chart: {plot_path}")


if __name__ == "__main__":
    main()
