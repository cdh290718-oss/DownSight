#!/usr/bin/env bash
set -eu
out="$PWD/outputs/v003_joint_error_comparison_20261128"
mkdir -p "$out"/{runtime,tmp}
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export XDG_CACHE_HOME="$out/runtime" MPLCONFIGDIR="$out/runtime" TMPDIR="$out/tmp" NUMBA_CACHE_DIR="$out/runtime"
py=$(python3 -B -c "import json;print(json.load(open('outputs/initialization_ablation_20261109/model_paths.json'))['body_python'])")
exec "$py" -Bu runs/2026-11-28_v003_joint_error_comparison/compare.py
