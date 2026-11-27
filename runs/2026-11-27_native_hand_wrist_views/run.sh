#!/usr/bin/env bash
set -eu
root=$PWD
r=$root/runs/2026-11-27_native_hand_wrist_views
o=$root/outputs/native_hand_wrist_views_20261127
mkdir -p "$o"/{export,runtime/cache,runtime/mpl,runtime/tmp} "$r"/{assets,evidence}
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME="$o/runtime/cache" TORCH_HOME="$o/runtime/cache" MPLCONFIGDIR="$o/runtime/mpl" TMPDIR="$o/runtime/tmp" NUMBA_CACHE_DIR="$o/runtime/cache" YOLO_CONFIG_DIR="$o/runtime/cache"
field() { python3 -c "import json,sys;print(json.load(open('configs/pipeline.head01.developer.json'))['paths'][sys.argv[1]])" "$1"; }
wpy=$(field wilor_python)
repo=$(field wilor_repo)
if [ ! -e "$o/mano_data" ]; then ln -s "$repo/mano_data" "$o/mano_data"; fi
(cd "$o" && PYTHONPATH="$root:$repo" "$wpy" -Bu "$r/recover.py")
bpy=$(python3 -c "import json;print(json.load(open('outputs/initialization_ablation_20261109/model_paths.json'))['body_python'])")
PYTHONPATH="$root" "$bpy" -Bu "$r/wrist_views.py"
PYTHONPATH="$root" "$bpy" -Bu "$r/native_inputs.py"
