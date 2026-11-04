#!/usr/bin/env bash
set -euo pipefail
: "${WILOR_PYTHON:?}" "${WILOR_REPO:?}" "${DOWNSIGHT_REPO:?}" "${RUN_OUTPUT:?}"
export PYTHONDONTWRITEBYTECODE=1 YOLO_CONFIG_DIR="$RUN_OUTPUT/ultralytics" PYTHONPATH="$DOWNSIGHT_REPO"
cd "$WILOR_REPO"
"$WILOR_PYTHON" -m egosmplx_pipeline.wilor_predict --manifest "$RUN_OUTPUT/wilor_images/manifest.json" --output-dir "$RUN_OUTPUT/wilor" --checkpoint "$WILOR_REPO/pretrained_models/wilor_final.ckpt" --config "$WILOR_REPO/pretrained_models/model_config.yaml" --detector "$WILOR_REPO/pretrained_models/detector.pt" > "$RUN_OUTPUT/wilor.log" 2>&1
