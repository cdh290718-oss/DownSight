#!/usr/bin/env bash
set -eu
r=runs/2026-11-25_validation200_fingers
o=outputs/validation200_fingers_20261125
workers=${1:-16}
program=${2:-variants}
first_cpu=2

pids=()
for ((i=0;i<workers;i++)); do
  taskset -c "$((2*i+first_cpu))-$((2*i+first_cpu+1))" bash "$r/run.sh" "$program" "$i" "$workers" > "$o/${program}${i}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
