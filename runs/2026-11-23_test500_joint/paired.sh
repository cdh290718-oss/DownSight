#!/usr/bin/env bash
set -eu
run=runs/2026-11-23_test500_joint
out=outputs/test500_joint_20261123
bash "$run/run.sh" body prepare > "$out/prepare.log" 2>&1
# Independent frames, full 20,000-step solver per frame in each CPU process.
pids=()
for n in {0..7}; do
  (bash "$run/run.sh" body fit "$n" && bash "$run/run.sh" body search "$n") > "$out/shard_$n.log" 2>&1 &
  pids+=("$!")
done
failed=0
for p in "${pids[@]}"; do wait "$p" || failed=1; done
if [[ "$failed" == 1 ]]; then exit 1; fi
bash "$run/run.sh" score > "$out/score.log" 2>&1
