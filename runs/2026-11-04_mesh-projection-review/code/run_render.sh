#!/usr/bin/env bash
set -euo pipefail
: "${BODY_PYTHON:?}" "${EGOSMPLX_BACKEND:?}" "${DOUBLE_SPHERE_SOURCE:?}" "${RUN_OUTPUT:?}" "${PREVIOUS_OUTPUT:?}" "${HAND_OUTPUT:?}" "${GEN_IMAGES:?}" "${DFM_REFERENCE:?}" "${RENDER_OUTPUT:?}"
export PYTHONDONTWRITEBYTECODE=1
RUN_CODE=runs/2026-11-04_mesh-projection-review/code
SELECT=runs/2026-11-04_mesh-projection-review/evidence/selection.json
CAL="$PREVIOUS_OUTPUT/observations/camera_calibration.json"
"$BODY_PYTHON" "$RUN_CODE/reconstruct.py" --selection "$SELECT" --v0 "$PREVIOUS_OUTPUT/fits/v0" --stage "$HAND_OUTPUT/evaluation_fits/two_stage" --network "$PREVIOUS_OUTPUT/network" --output "$RUN_OUTPUT/meshes"
"$BODY_PYTHON" "$RUN_CODE/scene_depth.py" --selection "$SELECT" --images "$GEN_IMAGES" --calibration "$CAL" --output "$RUN_OUTPUT/scene_depth"
"$BODY_PYTHON" "$RUN_CODE/test_rasterizer.py"
"$BODY_PYTHON" "$RUN_CODE/render_comparisons.py" --selection "$SELECT" --meshes "$RUN_OUTPUT/meshes" --scene-depth "$RUN_OUTPUT/scene_depth" --images "$GEN_IMAGES" --reference "$DFM_REFERENCE" --sapiens "$PREVIOUS_OUTPUT/observations/observations.npz" --wilor "$PREVIOUS_OUTPUT/wilor_observations/observations.npz" --calibration "$CAL" --fits "$HAND_OUTPUT/evaluation_fits" --output "$RENDER_OUTPUT"
