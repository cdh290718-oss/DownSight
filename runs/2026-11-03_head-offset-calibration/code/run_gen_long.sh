#!/usr/bin/env bash
set -euo pipefail
: "${BODY_PYTHON:?}" "${STEREO_EXPERIMENT:?}" "${EGOSMPLX_BACKEND:?}" "${DOUBLE_SPHERE_SOURCE:?}" "${CLOUD:?}" "${CALIBRATION:?}" "${RUN_OUTPUT:?}"
export PYTHONDONTWRITEBYTECODE=1
run_one() {
 local label="$1"; shift
 "$BODY_PYTHON" runs/2026-11-03_head-offset-calibration/code/fit_fixed.py --variant full --seed 0 --steps 1500 --dense --cloud "$CLOUD" --fixed-center "$CALIBRATION" --perturb-cm "$@" --output "$RUN_OUTPUT/$label" > "$RUN_OUTPUT/$label.txt" 2>&1
}
run_one long_zero 0 0 0
for axis in x y z; do
 for sign in p m; do
  value=5; if [[ "$sign" == m ]]; then value=-5; fi
  case "$axis" in
   x) run_one "long_${axis}_${sign}_5" "$value" 0 0;;
   y) run_one "long_${axis}_${sign}_5" 0 "$value" 0;;
   z) run_one "long_${axis}_${sign}_5" 0 0 "$value";;
  esac
 done
done
