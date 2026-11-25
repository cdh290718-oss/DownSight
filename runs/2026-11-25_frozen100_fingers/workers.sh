#!/usr/bin/env bash
set -eu
r=runs/2026-11-25_frozen100_fingers
o=outputs/frozen100_fingers_20261125
workers=${1:-12}
group=${2:-fixed}
first_cpu=2
if [ "$group" = free ]; then first_cpu=42; fi
pids=()
for ((i=0;i<workers;i++)); do
  taskset -c "$((2*i+first_cpu))-$((2*i+first_cpu+1))" bash "$r/run.sh" fit "$i" "$workers" "$group" > "$o/${group}${i}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
