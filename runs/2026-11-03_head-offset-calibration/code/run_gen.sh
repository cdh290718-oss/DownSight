#!/usr/bin/env bash
set -euo pipefail
: "${BODY_PYTHON:?}" "${STEREO_EXPERIMENT:?}" "${EGOSMPLX_BACKEND:?}" "${DOUBLE_SPHERE_SOURCE:?}" "${CLOUD:?}" "${CALIBRATION:?}" "${RUN_OUTPUT:?}"
export PYTHONDONTWRITEBYTECODE=1
run_one() {
 local label="$1"; shift
 "$BODY_PYTHON" runs/2026-11-03_head-offset-calibration/code/fit_fixed.py --variant full --seed 0 --steps 500 --dense --cloud "$CLOUD" --fixed-center "$CALIBRATION" --perturb-cm "$@" --output "$RUN_OUTPUT/$label" > "$RUN_OUTPUT/$label.txt" 2>&1
}
run_one zero 0 0 0
for magnitude in 1 2 5 10; do
 for axis in x y z; do
  for sign in p m; do
   value="$magnitude"; if [[ "$sign" == m ]]; then value="-$magnitude"; fi
   case "$axis" in
    x) run_one "${axis}_${sign}_${magnitude}" "$value" 0 0;;
    y) run_one "${axis}_${sign}_${magnitude}" 0 "$value" 0;;
    z) run_one "${axis}_${sign}_${magnitude}" 0 0 "$value";;
   esac
  done
 done
done
