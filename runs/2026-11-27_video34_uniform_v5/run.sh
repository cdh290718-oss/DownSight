#!/usr/bin/env bash
set -eu
r=runs/2026-11-27_video34_uniform_v5
o="$PWD/outputs/video34_uniform_v5_20261127"
mkdir -p "$o"/{runtime,tmp,images,seg,poses,fit,export,palm/poses}
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 XDG_CACHE_HOME="$o/runtime" MPLCONFIGDIR="$o/runtime" TMPDIR="$o/tmp" NUMBA_CACHE_DIR="$o/runtime" YOLO_CONFIG_DIR="$o/runtime"
key=body_python
case "$1" in detect|segment) key=pose_python;; esac
py=$(python -B -c "import sys,json;print(json.load(open('outputs/initialization_ablation_20261109/model_paths.json'))[sys.argv[1]])" "$key")
exec "$py" -Bu "$r/driver.py" "$@"
