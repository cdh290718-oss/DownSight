#!/usr/bin/env bash
set -euo pipefail
: "${POSE_PYTHON:?}" "${EGOSMPLX_BACKEND:?}" "${GEN_IMAGES:?}" "${RUN_OUTPUT:?}"
export PYTHONDONTWRITEBYTECODE=1
"$POSE_PYTHON" runs/2026-11-03_torso-position-identifiability/code/segment.py --repo "$EGOSMPLX_BACKEND/official_baseline/sapiens2" --deps "$EGOSMPLX_BACKEND/.runtime/sapiens_deps" --checkpoint "$EGOSMPLX_BACKEND/pretrained_models/sapiens2_seg/sapiens2_1b_seg.safetensors" --images "$GEN_IMAGES" --frames "$RUN_OUTPUT/observations/observations.npz" --output "$RUN_OUTPUT/segmentation" > "$RUN_OUTPUT/segmentation.log" 2>&1
