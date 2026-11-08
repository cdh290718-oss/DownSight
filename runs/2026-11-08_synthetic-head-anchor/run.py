"""Run both views with fixed configuration, recording every process exit."""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/synthetic_head_anchor_20261108'
cfg = json.loads((ROOT / 'outputs/synthetic_original_20261108/configs/left.json').read_text())
runtime = OUT / 'runtime'
runtime.mkdir(parents=True, exist_ok=True)
(RUN / 'evidence').mkdir(exist_ok=True)
env = os.environ.copy()
env.update(PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='',
           OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
           TMPDIR=str(runtime), XDG_CACHE_HOME=str(runtime / 'cache'),
           MPLCONFIGDIR=str(runtime / 'matplotlib'))
for key in ['EGOSMPLX_V2', 'EGOSMPLX_V2_HAND_STAGE', 'EGOSMPLX_HEAD_NECK']:
    env.pop(key, None)
records = []
for side in ['left', 'right']:
    cmd = [cfg['paths']['body_python'], str(RUN / 'fit.py'), '--side', side]
    start = time.time()
    with (OUT / (side + '.log')).open('x') as log:
        result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    records.append(dict(side=side, exit_status=result.returncode, elapsed_seconds=time.time() - start,
                        command=['<BODY_PYTHON>', 'runs/2026-11-08_synthetic-head-anchor/fit.py', '--side', side],
                        settings_sha256=hashlib.sha256((RUN / 'settings.json').read_bytes()).hexdigest(),
                        fit_code_sha256=hashlib.sha256((RUN / 'fit.py').read_bytes()).hexdigest()))
    (RUN / 'evidence/execution.json').write_text(json.dumps(records, indent=2) + '\n')
    print(side, result.returncode, flush=True)
    if result.returncode:
        text = (OUT / (side + '.log')).read_text(errors='replace')
        text = text.replace(str(ROOT.parent), '<WORKSPACE>').replace('/usr/', '<SYSTEM>/')
        (RUN / 'evidence' / (side + '_failure.txt')).write_text(text.rstrip() + '\n')
