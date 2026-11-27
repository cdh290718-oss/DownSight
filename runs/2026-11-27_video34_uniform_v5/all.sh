#!/usr/bin/env bash
set -eu
r=runs/2026-11-27_video34_uniform_v5
bash "$r/run.sh" sample
python -B "$r/timing.py"
bash "$r/run.sh" detect
bash "$r/run.sh" prepare
bash "$r/run.sh" segment
bash "$r/wilor.sh"
bash "$r/wilor.sh" guided_batch
bash "$r/run.sh" fit
bash "$r/run.sh" candidates
python -B "$r/frames_job.py"
python -B "$r/package.py"
