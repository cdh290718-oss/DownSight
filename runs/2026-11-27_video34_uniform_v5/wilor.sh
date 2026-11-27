#!/usr/bin/env bash
set -eu
root="$PWD"
o="$root/outputs/video34_uniform_v5_20261127"
config=configs/pipeline.head01.developer.json
repo=$(python -Bc "import json;print(json.load(open('$config'))['paths']['wilor_repo'])")
py=$(python -Bc "import json;print(json.load(open('$config'))['paths']['wilor_python'])")
[ -e "$o/mano_data" ] || ln -s "$repo/mano_data" "$o/mano_data"
export PYTHONPATH="$root:$repo" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCH_HOME="$o/runtime" XDG_CACHE_HOME="$o/runtime" YOLO_CONFIG_DIR="$o/runtime" MPLCONFIGDIR="$o/runtime" TMPDIR="$o/tmp"
cd "$o"
exec "$py" -Bu "$root/runs/2026-11-27_video34_uniform_v5/${1:-wilor_batch}.py"
