#!/usr/bin/env bash
set -eu
run=runs/2026-11-26_realvideo4_head_obj
runtime_root="$PWD/outputs/realvideo4_head_obj_20261126/runtime"
mkdir -p "$runtime_root"/{cache,mpl,tmp,numba}
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export XDG_CACHE_HOME="$runtime_root/cache" MPLCONFIGDIR="$runtime_root/mpl" NUMBA_CACHE_DIR="$runtime_root/numba" TMPDIR="$runtime_root/tmp"
body_py=$(python -B -c "import json;print(json.load(open('outputs/initialization_ablation_20261109/model_paths.json'))['body_python'])")
exec "$body_py" -Bu "$run/render_obj.py" "$@"
