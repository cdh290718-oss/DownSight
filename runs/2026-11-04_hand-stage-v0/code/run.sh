#!/usr/bin/env bash
set -euo pipefail
: "${BODY_PYTHON:?}" "${EGOSMPLX_BACKEND:?}" "${DOUBLE_SPHERE_SOURCE:?}" "${PREVIOUS_OUTPUT:?}" "${RUN_OUTPUT:?}" "${FIRST:?}" "${LAST:?}"
export PYTHONDONTWRITEBYTECODE=1
"$BODY_PYTHON" runs/2026-11-04_hand-stage-v0/code/fit_hands.py --v0 "$PREVIOUS_OUTPUT/fits/v0" --observations "$PREVIOUS_OUTPUT/wilor_observations/observations.npz" --calibration "$PREVIOUS_OUTPUT/observations/camera_calibration.json" --initial "$PREVIOUS_OUTPUT/network" --first "$FIRST" --last "$LAST" --output "$RUN_OUTPUT/part_${FIRST}_${LAST}" > "$RUN_OUTPUT/part_${FIRST}_${LAST}.log" 2>&1
