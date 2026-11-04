#!/usr/bin/env bash
set -euo pipefail
: "${BODY_PYTHON:?}" "${EGOSMPLX_BACKEND:?}" "${DOUBLE_SPHERE_SOURCE:?}" "${CALIBRATION:?}" "${RUN_OUTPUT:?}" "${MODE:?}" "${FIRST:?}" "${LAST:?}" "${CLOUD:?}"
export PYTHONDONTWRITEBYTECODE=1
case "$MODE" in
 v0) STEREO_EXPERIMENT="$RUN_OUTPUT/observations" ;;
 wilor) STEREO_EXPERIMENT="$RUN_OUTPUT/wilor_observations" ;;
 *) exit 2 ;;
esac
export STEREO_EXPERIMENT
"$BODY_PYTHON" runs/2026-11-04_v0-expanded-evaluation/code/fit_180.py --variant full --seed 0 --steps 1500 --first "$FIRST" --last "$LAST" --dense --cloud "$CLOUD" --fixed-center "$CALIBRATION" --perturb-cm 0 0 0 --output "$RUN_OUTPUT/fits/${MODE}_${FIRST}_${LAST}" > "$RUN_OUTPUT/fit_${MODE}_${FIRST}_${LAST}.log" 2>&1
