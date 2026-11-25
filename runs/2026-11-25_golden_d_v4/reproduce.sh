#!/usr/bin/env bash
set -eu
r=runs/2026-11-25_golden_d_v4
o="$PWD/outputs/golden_d_v4_20261125"
mkdir -p "$o"/{logs,cache,mpl,tmp}
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export XDG_CACHE_HOME="$o/cache" MPLCONFIGDIR="$o/mpl" TMPDIR="$o/tmp" NUMBA_CACHE_DIR="$o/cache"
py=$(python -B -c "import json;print(json.load(open('outputs/initialization_ablation_20261109/model_paths.json'))['body_python'])")
workers=8
pids=()
for ((i=0; i<workers; i++)); do
    taskset -c "$((2*i+2))-$((2*i+3))" "$py" -Bu "$r/reproduce.py" "$i" "$workers" > "$o/logs/$i.txt" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
