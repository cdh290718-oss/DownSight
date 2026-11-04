#!/usr/bin/env bash
set -euo pipefail
: "${BODY_PYTHON:?}" "${EGOSMPLX_BACKEND:?}" "${DOUBLE_SPHERE_SOURCE:?}" "${CLOUD:?}" "${CALIBRATION:?}" "${VARIANTS:?}" "${RUN_OUTPUT:?}"
export PYTHONDONTWRITEBYTECODE=1
for steps in 500 1500; do
 STEREO_EXPERIMENT="$VARIANTS/published_gate_cross_hand" "$BODY_PYTHON" runs/2026-11-03_head-offset-calibration/code/fit_fixed.py --variant full --seed 0 --steps "$steps" --dense --cloud "$CLOUD" --fixed-center "$CALIBRATION" --perturb-cm 0 0 0 --output "$RUN_OUTPUT/published_gate_cross_hand_${steps}" > "$RUN_OUTPUT/published_gate_cross_hand_${steps}.txt" 2>&1
done
