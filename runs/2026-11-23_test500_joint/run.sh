#!/usr/bin/env bash
set -eu
run=runs/2026-11-23_test500_joint
runtime_root="$PWD/outputs/test500_joint_20261123/runtime"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$PWD"
export XDG_CACHE_HOME="$runtime_root/cache" TORCH_HOME="$runtime_root/cache/torch" HF_HOME="$runtime_root/cache/hf" CUDA_CACHE_PATH="$runtime_root/cache/cuda" TRITON_CACHE_DIR="$runtime_root/cache/triton" MPLCONFIGDIR="$runtime_root/mpl" NUMBA_CACHE_DIR="$runtime_root/numba" TMPDIR="$runtime_root/tmp"
if [[ "$1" == detect ]]; then
  py=$(python -B -c "import json;print(json.load(open('outputs/rw_a20000_training_20261117T152700444455Z/launch.json'))['command'][0])")
else
  export CUDA_VISIBLE_DEVICES=''
  py=$(python -B -c "import json;print(json.load(open('outputs/initialization_ablation_20261109/model_paths.json'))['body_python'])")
fi
exec "$py" -Bu "$run/$1.py" "${@:2}"
