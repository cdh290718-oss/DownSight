#!/usr/bin/env bash
set -eu
root=$PWD
r=runs/2026-11-27_wilor_native_hands
o="$root/outputs/wilor_native_hands_20261127"
getpath() { python3 -c 'import json,sys; print(json.load(open("configs/pipeline.head01.developer.json"))["paths"][sys.argv[1]])' "$1"; }
py=$(getpath wilor_python)
repo=$(getpath wilor_repo)
mkdir -p "$r/assets" "$r/evidence" "$o/runtime"
if [ ! -e "$o/mano_data" ]; then ln -s "$repo/mano_data" "$o/mano_data"; fi
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$root:$repo"
for var in XDG_CACHE_HOME MPLCONFIGDIR TMPDIR TORCH_HOME CUDA_CACHE_PATH YOLO_CONFIG_DIR; do
    mkdir -p "$o/runtime/$var"
    export "$var=$o/runtime/$var"
done
cd "$o"
exec "$py" -Bu "$root/$r/render_native.py"
