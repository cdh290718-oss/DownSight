#!/usr/bin/env bash
set -eu
run_path=runs/2026-11-23_rw_joint_ranges
runtime="$PWD/outputs/rw_joint_ranges_20261123/runtime"
mkdir -p "$runtime"/{cache,mpl,numba,tmp}
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export PYTHONPATH="$PWD" XDG_CACHE_HOME="$runtime/cache" MPLCONFIGDIR="$runtime/mpl" TMPDIR="$runtime/tmp" NUMBA_CACHE_DIR="$runtime/numba"
body_py=$(python -B -c "import json;print(json.load(open('outputs/initialization_ablation_20261109/model_paths.json'))['body_python'])")
program=$1
shift
exec "$body_py" -Bu "$run_path/$program.py" "$@"
