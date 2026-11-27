"""Run source-camera prediction or calibrated fitting with repository-local caches."""
import json
import os
from pathlib import Path
import subprocess
import sys
RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
OUT = ROOT/'outputs/validation200_v5_20261127'
paths = json.loads((ROOT/'configs/pipeline.head01.developer.json').read_text())['paths']
mode = sys.argv[1]
assert mode in ('predict_new','measure')
for p in [OUT/'poses',OUT/'rows',RUN/'evidence']:
    p.mkdir(exist_ok=True,parents=True)
env = dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
for name in ('XDG_CACHE_HOME','MPLCONFIGDIR','TORCH_HOME','YOLO_CONFIG_DIR','CUDA_CACHE_PATH','TMPDIR','NUMBA_CACHE_DIR'):
    p = OUT/'runtime'/name
    p.mkdir(exist_ok=True,parents=True)
    env[name] = str(p)
if mode=='predict_new':
    link = OUT/'mano_data'
    if not link.exists():
        link.symlink_to(Path(paths['wilor_repo'])/'mano_data',target_is_directory=True)
    env.update(CUDA_VISIBLE_DEVICES='0',PYTHONPATH=os.pathsep.join([str(ROOT),paths['wilor_repo']]))
    py = paths['wilor_python']
else:
    env.update(CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(ROOT))
    py = json.loads((ROOT/'outputs/initialization_ablation_20261109/model_paths.json').read_text())['body_python']
subprocess.run([py,'-Bu',str(RUN/(mode+'.py'))]+sys.argv[2:],cwd=OUT,env=env,check=True)
