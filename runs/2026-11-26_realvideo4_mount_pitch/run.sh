#!/usr/bin/env bash
set -eu
r=runs/2026-11-26_realvideo4_mount_pitch
o="$PWD/outputs/realvideo4_mount_pitch_20261126"
mkdir -p "$o/runtime"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=''
export XDG_CACHE_HOME="$o/runtime" MPLCONFIGDIR="$o/runtime" TMPDIR="$o/runtime" NUMBA_CACHE_DIR="$o/runtime"
py=$(python -B -c "import json;print(json.load(open('outputs/initialization_ablation_20261109/model_paths.json'))['body_python'])")
exec "$py" -Bu "$r/$1.py"
