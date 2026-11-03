#!/usr/bin/env bash
set -euo pipefail
: "${BODY_PYTHON:?}" "${EGOSMPLX_BACKEND:?}" "${DOUBLE_SPHERE_SOURCE:?}" "${INITIAL:?}" "${FIT:?}" "${CALIBRATION:?}" "${RECTIFICATION:?}" "${RUN_OUTPUT:?}"
export PYTHONDONTWRITEBYTECODE=1
for spec in 2:0 1:40 1:60 2:40 2:60; do
 axes="${spec%:*}"; degrees="${spec#*:}"; name="long_axes${axes}_deg${degrees}_trial0"
 if "$BODY_PYTHON" runs/2026-11-03_head-offset-calibration/code/smplx_motion_sim.py --initial "$INITIAL" --fit "$FIT" --calibration "$CALIBRATION" --rectification "$RECTIFICATION" --axes "$axes" --degrees "$degrees" --trial 0 --steps 3000 --output "$RUN_OUTPUT/$name.json" > "$RUN_OUTPUT/$name.txt" 2>&1; then printf 'CASE_EXIT 0\n' >> "$RUN_OUTPUT/$name.txt"; else status=$?; printf 'CASE_EXIT %s\n' "$status" >> "$RUN_OUTPUT/$name.txt"; fi
done
