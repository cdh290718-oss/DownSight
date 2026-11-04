#!/usr/bin/env bash
set -euo pipefail
: "${BODY_PYTHON:?}" "${EGOSMPLX_BACKEND:?}" "${DOUBLE_SPHERE_SOURCE:?}" "${INPUT_IMAGES:?}" "${VIRTUAL_CALIBRATION:?}" "${CHECKPOINT:?}" "${DETECTOR_CONFIG:?}" "${RUN_OUTPUT:?}"
export PYTHONDONTWRITEBYTECODE=1 EGOSMPLX_REPO_ROOT="$EGOSMPLX_BACKEND" PYTHONPATH="$DOUBLE_SPHERE_SOURCE"
cd "$DOUBLE_SPHERE_SOURCE"
"$BODY_PYTHON" -m egosmplx_pipeline.egosmplx_predict --image-dir "$INPUT_IMAGES" --image-glob '*.png' --fisheye-calibration "$VIRTUAL_CALIBRATION" --config-path "$EGOSMPLX_BACKEND/main/config/config_ft_egopw_sceneego_egowholebody_unrealego_egofishbody.py" --checkpoint-path "$CHECKPOINT" --det-config "$DETECTOR_CONFIG" --smplx-input-mode det_undistort_contiguous --contiguous-roi-mode aspect_pad --output-dir "$RUN_OUTPUT/network" > "$RUN_OUTPUT/network.log" 2>&1
