#!/usr/bin/env bash
set -eu
run=runs/2026-11-25_wilor_wrists
out="$PWD/outputs/wilor_wrists_20261125"
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD"
export TMPDIR="$out/tmp" XDG_CACHE_HOME="$out/cache" MPLCONFIGDIR="$out/mpl"
export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
body_python=$(python -B -c "import json; print(json.load(open('configs/pipeline.head01.developer.json'))['paths']['body_python'])")
"$body_python" -Bu "$run/compare_wrists.py" fit
"$body_python" -Bu "$run/compare_wrists.py" score
