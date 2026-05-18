#!/bin/bash
# Run all format comparison + chunk/shard sweep benchmarks
# Output: benchmarks/run_20260518/
# Each run: 100 samples, workers 0,4 (0 for per-sample profiling, 4 for throughput)

set -e
cd /groups/scicompsoft/home/chend/temp/miao

OUTDIR="benchmarks"
SAMPLES=100
WORKERS="0,4"

echo "========================================"
echo "Format Comparison Benchmarks (4 full-dataset configs)"
echo "========================================"

echo "[1/13] zarr2 (original format)..."
pixi run python benchmarks/profile_getitem.py \
  --config examples/bench_fmt_zarr2.yaml \
  --num_samples $SAMPLES --num_workers $WORKERS \
  --label fmt_zarr2 --output_dir $OUTDIR

echo "[2/13] zarr3-nosharding..."
pixi run python benchmarks/profile_getitem.py \
  --config examples/bench_fmt_zarr3_nosharding.yaml \
  --num_samples $SAMPLES --num_workers $WORKERS \
  --label fmt_zarr3_nosharding --output_dir $OUTDIR

echo "[3/13] zarr3-sharded (256 chunk, default shard)..."
pixi run python benchmarks/profile_getitem.py \
  --config examples/bench_fmt_zarr3_sharded.yaml \
  --num_samples $SAMPLES --num_workers $WORKERS \
  --label fmt_zarr3_sharded --output_dir $OUTDIR

echo "[4/13] zarr3-sharded-128chunk..."
pixi run python benchmarks/profile_getitem.py \
  --config examples/bench_fmt_zarr3_sharded_128chunk.yaml \
  --num_samples $SAMPLES --num_workers $WORKERS \
  --label fmt_zarr3_sharded_128chunk --output_dir $OUTDIR

echo ""
echo "========================================"
echo "Chunk/Shard Sweep Benchmarks (9 bbox crop configs)"
echo "========================================"

echo "[5/13] zarr2-chunk64..."
pixi run python benchmarks/profile_getitem.py \
  --config examples/bench_sweep_zarr2_chunk64.yaml \
  --num_samples $SAMPLES --num_workers $WORKERS \
  --label sweep_zarr2_chunk64 --output_dir $OUTDIR

echo "[6/13] zarr2-chunk128..."
pixi run python benchmarks/profile_getitem.py \
  --config examples/bench_sweep_zarr2_chunk128.yaml \
  --num_samples $SAMPLES --num_workers $WORKERS \
  --label sweep_zarr2_chunk128 --output_dir $OUTDIR

echo "[7/13] zarr2-chunk256..."
pixi run python benchmarks/profile_getitem.py \
  --config examples/bench_sweep_zarr2_chunk256.yaml \
  --num_samples $SAMPLES --num_workers $WORKERS \
  --label sweep_zarr2_chunk256 --output_dir $OUTDIR

echo "[8/13] zarr3-noshard-chunk128..."
pixi run python benchmarks/profile_getitem.py \
  --config examples/bench_sweep_zarr3_noshard_chunk128.yaml \
  --num_samples $SAMPLES --num_workers $WORKERS \
  --label sweep_zarr3_noshard_chunk128 --output_dir $OUTDIR

echo "[9/13] zarr3-shard512-chunk64..."
pixi run python benchmarks/profile_getitem.py \
  --config examples/bench_sweep_zarr3_shard512_chunk64.yaml \
  --num_samples $SAMPLES --num_workers $WORKERS \
  --label sweep_zarr3_shard512_chunk64 --output_dir $OUTDIR

echo "[10/13] zarr3-shard512-chunk128..."
pixi run python benchmarks/profile_getitem.py \
  --config examples/bench_sweep_zarr3_shard512_chunk128.yaml \
  --num_samples $SAMPLES --num_workers $WORKERS \
  --label sweep_zarr3_shard512_chunk128 --output_dir $OUTDIR

echo "[11/13] zarr3-shard1024-chunk128..."
pixi run python benchmarks/profile_getitem.py \
  --config examples/bench_sweep_zarr3_shard1024_chunk128.yaml \
  --num_samples $SAMPLES --num_workers $WORKERS \
  --label sweep_zarr3_shard1024_chunk128 --output_dir $OUTDIR

echo "[12/13] zarr3-shard512-chunk256..."
pixi run python benchmarks/profile_getitem.py \
  --config examples/bench_sweep_zarr3_shard512_chunk256.yaml \
  --num_samples $SAMPLES --num_workers $WORKERS \
  --label sweep_zarr3_shard512_chunk256 --output_dir $OUTDIR

echo "[13/13] zarr3-shard1024-chunk256..."
pixi run python benchmarks/profile_getitem.py \
  --config examples/bench_sweep_zarr3_shard1024_chunk256.yaml \
  --num_samples $SAMPLES --num_workers $WORKERS \
  --label sweep_zarr3_shard1024_chunk256 --output_dir $OUTDIR

echo ""
echo "========================================"
echo "ALL 13 BENCHMARKS COMPLETE"
echo "Results in: $OUTDIR/run_20260518/"
echo "========================================"
echo ""
echo "To generate consolidated report:"
echo "  pixi run python benchmarks/generate_report.py --consolidate --results_dir $OUTDIR/run_20260518"
