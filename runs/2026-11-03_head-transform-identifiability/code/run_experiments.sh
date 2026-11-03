#!/usr/bin/env bash
set -euo pipefail
: "${BODY_PYTHON:?}" "${STEREO_EXPERIMENT:?}" "${SAPIENS_DETECTIONS:?}" "${EGOSMPLX_BACKEND:?}" "${DOUBLE_SPHERE_SOURCE:?}" "${RUN_OUTPUT:?}"
export PYTHONDONTWRITEBYTECODE=1
for condition in no_prior weak_prior shoulders; do
  for seed in 0 1 2 3 4 5 6; do
    args=()
    if [[ "$condition" == weak_prior ]]; then args+=(--weak-prior); fi
    if [[ "$condition" == shoulders ]]; then args+=(--shoulders); fi
    "$BODY_PYTHON" runs/2026-11-03_head-transform-identifiability/code/fit.py --variant full --seed "$seed" --steps 500 --output "$RUN_OUTPUT/${condition}_${seed}" "${args[@]}" > "$RUN_OUTPUT/${condition}_${seed}.txt" 2>&1
  done
done
# Predeclared sensitivity sweep; no reference labels are used for fitting or selection.
for stereo_weight in 2 5; do
  "$BODY_PYTHON" runs/2026-11-03_head-transform-identifiability/code/fit.py --variant full --seed 0 --steps 500 --weak-prior --stereo-weight "$stereo_weight" --output "$RUN_OUTPUT/stereo_${stereo_weight}" > "$RUN_OUTPUT/stereo_${stereo_weight}.txt" 2>&1
done
# Longer optimization for two no-prior starts checks iteration sensitivity.
for seed in 0 1; do
  "$BODY_PYTHON" runs/2026-11-03_head-transform-identifiability/code/fit.py --variant full --seed "$seed" --steps 1500 --output "$RUN_OUTPUT/no_prior_long_${seed}" > "$RUN_OUTPUT/no_prior_long_${seed}.txt" 2>&1
done
# After the label-free 500-step results, extend the two starts with the greatest center separation.
for seed in 3 4; do
  "$BODY_PYTHON" runs/2026-11-03_head-transform-identifiability/code/fit.py --variant full --seed "$seed" --steps 1500 --output "$RUN_OUTPUT/no_prior_long_${seed}" > "$RUN_OUTPUT/no_prior_long_${seed}.txt" 2>&1
done
