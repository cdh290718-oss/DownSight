#!/usr/bin/env bash
set -euo pipefail
run=runs/2026-11-28_rw12_headonly
out=outputs/rw12_headonly_20261128
export RW12_RUN="$run" RW12_OUT="$out"
python -B -c 'import json,sys; rows=json.load(open(sys.argv[1]))["items"];
for index,row in enumerate(rows): print(index, row["pair"])' "$out/fit/0/prepared.json" |
  xargs -n 2 -P 4 bash -c '
    index=$1
    pair=$2
    bash "$RW12_RUN/run.sh" body_refine "$index" > "$RW12_OUT/${pair}_body.log" 2>&1 &&
      bash "$RW12_RUN/run.sh" head "$pair" > "$RW12_OUT/${pair}_head.log" 2>&1
  ' bash
bash "$run/run.sh" score
