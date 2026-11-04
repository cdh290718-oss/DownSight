#!/usr/bin/env bash
set -euo pipefail
: "${BODY_PYTHON:?}" "${EGOSMPLX_BACKEND:?}" "${DOUBLE_SPHERE_SOURCE:?}" "${CLOUD:?}" "${CALIBRATION:?}" "${VARIANTS:?}" "${RUN_OUTPUT:?}"
export PYTHONDONTWRITEBYTECODE=1
for steps in 500 1500; do
 for mode in wrist_only finger_2d_only finger_3d_only published_gate_cross_hand; do
  normalizer=baseline; if [[ "$mode" == published_gate_cross_hand ]]; then normalizer=published_gate; fi
  STEREO_EXPERIMENT="$VARIANTS/$mode" "$BODY_PYTHON" runs/2026-11-04_wrist-tail-gating/code/fit_fixed_denominator.py --variant full --seed 0 --steps "$steps" --dense --cloud "$CLOUD" --fixed-center "$CALIBRATION" --normalizer-observations "$VARIANTS/$normalizer/observations.npz" --perturb-cm 0 0 0 --output "$RUN_OUTPUT/${mode}_fixedden_${steps}" > "$RUN_OUTPUT/${mode}_fixedden_${steps}.txt" 2>&1
 done
done
