#!/usr/bin/env bash
set -eu
r=runs/2026-11-25_frozen100_outline
o=outputs/frozen100_outline_20261125
workers=${1:-12}
source=${2:-archive}
pids=()
for ((i=0;i<workers;i++)); do
  taskset -c "$((2*i+2))-$((2*i+3))" bash "$r/run.sh" compare "$i" "$workers" 100 "$source" > "$o/worker${i}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
