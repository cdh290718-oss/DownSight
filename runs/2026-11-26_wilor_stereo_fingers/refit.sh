set -eu
root=$PWD
r=runs/2026-11-26_wilor_stereo_fingers
o=$root/outputs/wilor_stereo_fingers_20261126
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
for var in XDG_CACHE_HOME MPLCONFIGDIR TMPDIR TORCH_HOME CUDA_CACHE_PATH YOLO_CONFIG_DIR; do export "$var=$o/runtime/$var"; done
getpath() { python3 -c 'import json,sys;print(json.load(open("configs/pipeline.head01.developer.json"))["paths"][sys.argv[1]])' "$1"; }
py=$(getpath wilor_python)
export PYTHONPATH="$root:$(getpath wilor_repo)"
cd "$o"
exec "$py" -Bu "$root/$r/refit.py"
