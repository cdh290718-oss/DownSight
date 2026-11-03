#!/usr/bin/env bash
set -euo pipefail
: "${BODY_PYTHON:?}" "${STEREO_EXPERIMENT:?}" "${EGOSMPLX_BACKEND:?}" "${DOUBLE_SPHERE_SOURCE:?}" "${CLOUD:?}" "${RUN_OUTPUT:?}"
export PYTHONDONTWRITEBYTECODE=1
for condition in baseline dense; do
 for seed in 0 1 2 3 4 5 6; do
  args=(); if [[ "$condition" == dense ]]; then args+=(--dense); fi
  "$BODY_PYTHON" runs/2026-11-03_torso-position-identifiability/code/fit.py --variant full --seed "$seed" --steps 500 --cloud "$CLOUD" --output "$RUN_OUTPUT/${condition}_${seed}" "${args[@]}" > "$RUN_OUTPUT/${condition}_${seed}.txt" 2>&1
 done
done
