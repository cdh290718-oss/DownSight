#!/usr/bin/env bash
set -eu
r=runs/2026-11-25_floor24_foot_lock
o=outputs/floor24_foot_lock_20261125
workers=${1:-8}
program=${2:-lock}
first_cpu=2
mkdir -p "$o/poses"

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
