#!/usr/bin/env bash
set -eu
run=runs/2026-11-28_shape_standard_skeleton
runtime_root="$PWD/outputs/shape_standard_skeleton_20261128/runtime"
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export XDG_CACHE_HOME="$runtime_root/cache" MPLCONFIGDIR="$runtime_root/mpl" NUMBA_CACHE_DIR="$runtime_root/numba" TMPDIR="$runtime_root/tmp"
body_py=$(python -B -c "import json;print(json.load(open('outputs/initialization_ablation_20261109/model_paths.json'))['body_python'])")
exec "$body_py" -Bu "$run/score.py"
