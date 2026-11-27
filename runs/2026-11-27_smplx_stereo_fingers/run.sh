#!/usr/bin/env bash
set -eu
r=runs/2026-11-27_smplx_stereo_fingers
o="$PWD/outputs/smplx_stereo_fingers_20261127"
mkdir -p "$o"/{seg,poses,palm/poses,palm/evidence,runtime/cache,runtime/tmp,runtime/mpl,runtime/cuda}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$PWD"
export XDG_CACHE_HOME="$o/runtime/cache" MPLCONFIGDIR="$o/runtime/mpl" TMPDIR="$o/runtime/tmp" CUDA_CACHE_PATH="$o/runtime/cuda" NUMBA_CACHE_DIR="$o/runtime/cache"
mkdir -p "$r/assets" "$r/evidence"
program=$1
shift
key=body_python
export CUDA_VISIBLE_DEVICES=''
py=$(python3 -B -c "import json,sys;print(json.load(open('outputs/initialization_ablation_20261109/model_paths.json'))[sys.argv[1]])" "$key")
exec "$py" -Bu "$r/$program.py" "$@"
