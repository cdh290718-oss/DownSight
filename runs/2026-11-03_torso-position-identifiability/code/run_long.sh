#!/usr/bin/env bash
set -euo pipefail
: "${BODY_PYTHON:?}" "${STEREO_EXPERIMENT:?}" "${EGOSMPLX_BACKEND:?}" "${DOUBLE_SPHERE_SOURCE:?}" "${CLOUD:?}" "${RUN_OUTPUT:?}"
export PYTHONDONTWRITEBYTECODE=1
python runs/2026-11-03_torso-position-identifiability/code/select_long.py --results "$RUN_OUTPUT" --output "$RUN_OUTPUT/long_selection.json"
for condition in baseline dense; do
 for seed in $(python -c 'import json,sys;print(*json.load(open(sys.argv[1]))[sys.argv[2]]["seeds"])' "$RUN_OUTPUT/long_selection.json" "$condition"); do
  args=(); if [[ "$condition" == dense ]]; then args+=(--dense); fi
  "$BODY_PYTHON" runs/2026-11-03_torso-position-identifiability/code/fit.py --variant full --seed "$seed" --steps 1500 --cloud "$CLOUD" --output "$RUN_OUTPUT/${condition}_long_${seed}" "${args[@]}" > "$RUN_OUTPUT/${condition}_long_${seed}.txt" 2>&1
 done
done
