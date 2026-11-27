#!/usr/bin/env bash
set -eu
r=runs/2026-11-27_golden_d_v5
o="$PWD/outputs/golden_d_v5_20261127"
mkdir -p "$o"/{export,mpl,cache,tmp}
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME="$o/cache" MPLCONFIGDIR="$o/mpl" TMPDIR="$o/tmp" NUMBA_CACHE_DIR="$o/cache"
py=$(python3 -Bc "import json;print(json.load(open('outputs/initialization_ablation_20261109/model_paths.json'))['body_python'])")
exec "$py" -Bu "$r/replay.py"
