#!/usr/bin/env bash
set -eu
run=runs/2026-11-23_fullbody_joint_candidates
runtime_root="$PWD/outputs/fullbody_joint_candidates_20261123/runtime"
mkdir -p "$runtime_root"/{cache,mpl,numba,tmp}
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export XDG_CACHE_HOME="$runtime_root/cache" MPLCONFIGDIR="$runtime_root/mpl" TMPDIR="$runtime_root/tmp" NUMBA_CACHE_DIR="$runtime_root/numba"
body_py=$(python -B -c "import json;print(json.load(open('outputs/initialization_ablation_20261109/model_paths.json'))['body_python'])")
exec "$body_py" -Bu "$run/$1.py" "$2"
