"""Measure the throughput gain from `defer_image_ops`.

Standalone: with no arguments it writes its own synthetic corpus, so it needs nothing but miao
and a GPU. Point `--config` at a real miao YAML to measure the corpus you actually train on.

    python benchmarks/bench_defer_image_ops.py
    python benchmarks/bench_defer_image_ops.py --config path/to/corpus.yaml --patch 256

What is compared. Two arms that differ only in where the resample and the normalization happen,
each measured to the same finish line -- a normalized image tensor on the GPU in `image_dtype`,
which is what a training step consumes:

    baseline  workers resample + normalize, then          batch["img"].to(device)
    deferred  workers read only, then    finish_images(batch, device=device)

The GPU work the deferred arm adds is inside its own timing, so the number below is a net gain,
not the worker saving alone. Both arms are timed in steady state after a warmup, with a
`cuda.synchronize()` before the clock stops, because otherwise the deferred arm would be timed
having merely queued its kernels.

Both arms are also checked for equivalence on one batch before timing. A throughput comparison
between two things that compute different tensors is not worth reading.

The measured regime is input-bound: no model runs, so the loop consumes batches as fast as the
worker pool produces them and the result is the ceiling the input pipeline imposes on training.
That is the quantity the flag is meant to move. A step that spends longer in the model than in
the loader sees a smaller end-to-end gain -- up to none, if the loader was never the constraint.

On the synthetic corpus the read is served from local disk and is therefore cheaper than a real
read from /nrs. Since deferring moves the *other* work, a cheaper read makes the deferred arm
look better: treat the synthetic number as an upper bound and the `--config` number as the real
one. The script prints the read's share of worker time in both cases so the difference is visible
rather than implied.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from miao import VolumeDataset, collate_deferred, finish_images, load_config
from miao.config import MiaoConfig

# The requested resolution for the synthetic corpus; volumes are stored relative to it.
TARGET_VOXEL = 8.0

# ------------------------------------------------------------------ synthetic corpus


def build_corpus(root: Path, n_volumes: int, side: int, matched_fraction: float,
                 coarse_voxel: float, target_voxel: float) -> list[dict]:
    """Write `n_volumes` single-level uint8 OME-NGFF volumes and return miao volume entries.

    `uint8` because that is what the real corpus stores, and it is the dtype that makes the
    transfer saving real: normalization is what would widen it to float32 in the worker.

    `matched_fraction` of the volumes are stored at exactly the requested resolution and so skip
    the interpolation; the rest are stored at `coarse_voxel` and pay it. The real corpus has
    roughly 1 in 5 matched, which is the default here -- the flag's value depends on that ratio,
    so it is a parameter rather than a constant.

    `coarse_voxel` sets which way the resample goes, and it decides the transfer as well as the
    compute. Coarser than the request means the read is *smaller* than the patch, so deferring
    saves the interpolation's expansion on top of the dtype's width. Finer than the request means
    the read is *larger* -- miao picks the coarsest level that still only downsamples, so a
    volume stored at half the requested voxel is read at 8x the patch's voxels -- and a deferred
    uint8 crop can then cross the bus for *more* bytes than the finished float32 patch would.
    """
    import zarr
    from zarr.storage import LocalStore

    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    entries: list[dict] = []
    n_matched = int(round(n_volumes * matched_fraction))

    for i in range(n_volumes):
        voxel = target_voxel if i < n_matched else coarse_voxel
        path = root / f"vol_{i:03d}.zarr"
        if not path.exists():
            grp = zarr.open_group(LocalStore(str(path)), mode="a", zarr_format=2)
            g = grp.create_group("raw")
            arr = g.create_array(
                "0", shape=(side,) * 3, chunks=(128,) * 3, dtype="uint8", overwrite=True
            )
            # Written in slabs: the whole array at once would be a needless spike in a job whose
            # memory limit is shared with the worker pool this script is about to start.
            for z0 in range(0, side, 64):
                z1 = min(z0 + 64, side)
                arr[z0:z1] = rng.integers(0, 256, (z1 - z0, side, side), dtype=np.uint8)
            (path / "raw" / ".zattrs").write_text(json.dumps({"multiscales": [{
                "version": "0.4",
                "axes": [{"name": c, "type": "space", "unit": "nanometer"} for c in "zyx"],
                "datasets": [{"path": "0", "coordinateTransformations": [
                    {"type": "scale", "scale": [voxel] * 3}]}],
            }]}))
        entries.append({
            "name": f"vol_{i:03d}", "path": str(path), "image_key": "raw",
            "normalize": True,
        })
    return entries


# ------------------------------------------------------------------ measurement


def _config(base: dict, *, defer: bool) -> MiaoConfig:
    return MiaoConfig(**{**base, "defer_image_ops": defer})


def _loader(config: MiaoConfig, batch_size: int, workers: int, pin: bool) -> DataLoader:
    """`collate_deferred` for the deferred arm; the default collate cannot batch crops whose
    shapes differ, which they do until something resamples them."""
    return DataLoader(
        VolumeDataset(config),
        batch_size=batch_size,
        num_workers=workers,
        collate_fn=collate_deferred if config.defer_image_ops else None,
        pin_memory=pin,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
        drop_last=True,
    )


def _bytes_transferred(batch: dict) -> int:
    """Host-to-device bytes for one batch's image, deferred (a list of crops) or not."""
    if isinstance(batch["img"], list):
        return sum(c.numel() * c.element_size() for s in batch["img"] for c in s)
    return batch["img"].numel() * batch["img"].element_size()


def measure(
    base: dict, *, defer: bool, device: str, batch_size: int, workers: int,
    steps: int, warmup: int,
) -> dict:
    """Steady-state ms per step, both arms taken to the same finish line."""
    config = _config(base, defer=defer)
    loader = _loader(config, batch_size, workers, pin=device.startswith("cuda"))
    it = iter(loader)

    step_ms: list[float] = []
    finish_ms: list[float] = []
    nbytes = 0

    for i in range(warmup + steps):
        t0 = time.perf_counter()
        batch = next(it)
        if i >= warmup:
            nbytes = _bytes_transferred(batch)
        t_fetch = time.perf_counter()
        if defer:
            img = finish_images(batch, device=device)["img"]
        else:
            img = batch["img"].to(device, non_blocking=True)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        if i >= warmup:
            step_ms.append((t1 - t0) * 1e3)
            finish_ms.append((t1 - t_fetch) * 1e3)
        del batch, img

    del it, loader
    arr = np.asarray(step_ms)
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "samples_per_s": batch_size * 1e3 / float(arr.mean()),
        "device_ms": float(np.mean(finish_ms)),
        "h2d_mb": nbytes / 2**20,
    }


def check_equivalent(base: dict, device: str, batch_size: int) -> str:
    """The two arms must produce the same tensor, or the timing above compares nothing.

    Drawn with the RNG pinned around each `__getitem__`, since the sampler picks a volume and a
    centre per call and unpinned draws would read different crops.
    """
    plain = VolumeDataset(_config(base, defer=False))
    deferred = VolumeDataset(_config(base, defer=True))

    for i in range(batch_size):
        np.random.seed(4000 + i)
        want = plain[i]["img"]
        np.random.seed(4000 + i)
        got = finish_images(collate_deferred([deferred[i]]), device=device)["img"][0].cpu()
        if got.shape != want.shape:
            return f"FAIL shape {tuple(got.shape)} != {tuple(want.shape)}"
        err = (got.float() - want.float()).abs().max().item()
        if err > 1e-4:
            return f"FAIL max|deferred - baseline| = {err:.2e}"
    return "ok (identical to baseline within 1e-4)"


def read_share(base: dict) -> str:
    """The read's share of one worker's per-sample time, deferred -- i.e. of what is left after
    the resample and the normalization are handed to the device. Reported because it bounds the
    gain: whatever fraction the read already is, deferring cannot go below it."""
    deferred = VolumeDataset(_config(base, defer=True))
    plain = VolumeDataset(_config(base, defer=False))
    for i in range(2):  # warm the stores
        deferred[i], plain[i]

    def _time(ds, n=8):
        t0 = time.perf_counter()
        for i in range(n):
            ds[i]
        return (time.perf_counter() - t0) * 1e3 / n

    d, p = _time(deferred), _time(plain)
    return f"{d:7.1f} ms deferred / {p:7.1f} ms full  ->  worker keeps {100 * d / p:.0f}%"


# ------------------------------------------------------------------ entry point


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, help="miao YAML; omit to build a synthetic corpus")
    ap.add_argument("--patch", type=int, default=256, help="cubic patch size")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=None,
                    help="default: enough to drain the initial prefetch buffer")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--volumes", type=int, default=24, help="synthetic corpus size")
    ap.add_argument("--coarse-voxel", type=float, default=10.0,
                    help="synthetic: stored voxel size of the volumes that need resampling; "
                         f"below {8.0:g} means miao reads more voxels than the patch")
    ap.add_argument("--matched-fraction", type=float, default=0.2,
                    help="synthetic: share of volumes stored at the requested resolution")
    ap.add_argument("--root", type=Path, default=None,
                    help="where to write the synthetic corpus (default: node-local temp)")
    args = ap.parse_args()
    if args.warmup is None:
        # The workers fill `workers * prefetch_factor` batches before the loop starts, so a
        # short warmup measures a queue being drained rather than the pipeline's rate: the
        # baseline arm reports a 10 ms median next to a 130 ms mean and neither is its
        # throughput. Drain that buffer first, then measure.
        args.warmup = args.workers * 2 + 4

    tmp_root: Path | None = None
    if args.config:
        base = load_config(args.config).model_dump()
        base["defer_image_ops"] = False
        base["augment_fn"] = None  # refused alongside the flag, and not what is being measured
        # A corpus config outlives the corpus: the census configs under lmd-v0.0.1 name volumes
        # that have since been reorganized away. Dropping them keeps a benchmark from dying on
        # bookkeeping, and the count is printed so the corpus measured is never a guess.
        kept = [v for v in base["volumes"] if Path(v["path"]).exists()]
        dropped = len(base["volumes"]) - len(kept)
        base["volumes"] = kept
        label = f"{args.config.name} ({len(kept)} volumes" + (
            f", {dropped} skipped as missing)" if dropped else ")")
    else:
        tmp_root = args.root or Path(tempfile.mkdtemp(prefix="miao_defer_bench_"))
        # The array has to hold the largest read any volume needs, which is the patch scaled by
        # how much finer than the request that volume is stored -- not the patch itself.
        largest_read = int(args.patch * max(1.0, TARGET_VOXEL / args.coarse_voxel))
        side = largest_read + 64
        print(f"building {args.volumes} synthetic volumes ({side}^3 uint8) under {tmp_root} ...",
              flush=True)
        volumes = build_corpus(tmp_root, args.volumes, side, args.matched_fraction,
                               args.coarse_voxel, TARGET_VOXEL)
        base = {
            "volumes": volumes,
            "resolutions": [[TARGET_VOXEL] * 3],
            "patch_size": [args.patch] * 3,
            "output_axes": "lczyx",
            "samples_per_epoch": args.batch_size * (args.steps + args.warmup + 2),
        }
        direction = "upsampled" if args.coarse_voxel > TARGET_VOXEL else "downsampled"
        label = (f"synthetic ({args.volumes} volumes, {args.matched_fraction:.0%} at the "
                 f"requested {TARGET_VOXEL:g} nm; the rest stored at {args.coarse_voxel:g} nm, "
                 f"read {largest_read}^3 and {direction})")

    try:
        print(f"\ncorpus     {label}")
        print(f"patch      {args.patch}^3   batch {args.batch_size}   "
              f"workers {args.workers}   device {args.device}")
        if args.device == "cpu":
            print("NOTE: no GPU -- 'deferred' runs the same ops on the CPU, so it measures the\n"
                  "      cost of moving them off the workers with nowhere better to put them.")
        print(f"equivalence  {check_equivalent(base, args.device, min(4, args.batch_size))}")
        print(f"worker cost  {read_share(base)}\n")

        rows = {
            "baseline (workers resample + normalize)": measure(
                base, defer=False, device=args.device, batch_size=args.batch_size,
                workers=args.workers, steps=args.steps, warmup=args.warmup),
            "deferred (device resamples + normalizes)": measure(
                base, defer=True, device=args.device, batch_size=args.batch_size,
                workers=args.workers, steps=args.steps, warmup=args.warmup),
        }

        print(f"{'':42s} {'ms/step':>9s} {'median':>8s} {'p95':>8s} "
              f"{'samples/s':>10s} {'H2D MB':>8s} {'device ms':>10s}")
        for name, r in rows.items():
            print(f"{name:42s} {r['mean_ms']:9.1f} {r['median_ms']:8.1f} {r['p95_ms']:8.1f} "
                  f"{r['samples_per_s']:10.1f} {r['h2d_mb']:8.1f} {r['device_ms']:10.1f}")

        b, d = rows["baseline (workers resample + normalize)"], \
            rows["deferred (device resamples + normalizes)"]
        print(f"\nspeedup    {b['mean_ms'] / d['mean_ms']:.2f}x  "
              f"({b['samples_per_s']:.1f} -> {d['samples_per_s']:.1f} samples/s)")
        print(f"transfer   {b['h2d_mb'] / d['h2d_mb']:.2f}x smaller  "
              f"({b['h2d_mb']:.1f} -> {d['h2d_mb']:.1f} MB per batch)")
        print(f"added GPU  {d['device_ms']:.1f} ms/step for the resample + normalization "
              f"(vs {b['device_ms']:.1f} ms for the plain transfer)")
    finally:
        if tmp_root is not None and args.root is None:
            shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
