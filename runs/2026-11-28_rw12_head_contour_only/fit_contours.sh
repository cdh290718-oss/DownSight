#!/usr/bin/env bash
set -euo pipefail
run=runs/2026-11-28_rw12_head_contour_only
out=outputs/rw12_head_contour_only_20261128
export RW12_CONTOUR_RUN="$run" RW12_CONTOUR_OUT="$out"
seq 0 11 | xargs -n 1 -P 6 bash -c '
  bash "$RW12_CONTOUR_RUN/run.sh" head_contour "$1" > "$RW12_CONTOUR_OUT/head_$1.log" 2>&1
' bash
