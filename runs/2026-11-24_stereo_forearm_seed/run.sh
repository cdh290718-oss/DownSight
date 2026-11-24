#!/usr/bin/env bash
set -eu
run=runs/2026-11-24_stereo_forearm_seed
runtime="$PWD/outputs/stereo_forearm_seed_20261124/runtime"
mkdir -p "$runtime"/{cache,mpl,tmp}
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD"
export XDG_CACHE_HOME="$runtime/cache" MPLCONFIGDIR="$runtime/mpl" TMPDIR="$runtime/tmp"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
body_python=$(python -B -c "import json; print(json.load(open('outputs/initialization_ablation_20261109/model_paths.json'))['body_python'])")
"$body_python" -Bu "$run/forearm_seed.py" fit
"$body_python" -Bu "$run/forearm_seed.py" score
"$body_python" -Bu "$run/compare_views.py"
