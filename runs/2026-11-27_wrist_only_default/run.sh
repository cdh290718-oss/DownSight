#!/usr/bin/env bash
set -eu
r=runs/2026-11-27_wrist_only_default
o="$PWD/outputs/wrist_only_default_20261127"
mkdir -p "$o"/{export,runtime/mpl,runtime/cache,runtime/temp} "$r/evidence"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME="$o/runtime/cache" MPLCONFIGDIR="$o/runtime/mpl" TMPDIR="$o/runtime/temp" NUMBA_CACHE_DIR="$o/runtime/cache"
py=$(python3 -Bc "import json;print(json.load(open('outputs/initialization_ablation_20261109/model_paths.json'))['body_python'])")
exec "$py" -Bu "$r/wrist_only.py" "$1"
