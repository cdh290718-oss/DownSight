"""Use the installed WiLoR CUDA environment for both models; confine caches to outputs."""
import json
import os
import subprocess
import sys
from pathlib import Path
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
OUT=ROOT/'outputs/vidihand2_comparison_20261127'
for folder in (OUT/'poses',OUT/'export',RUN/'assets',RUN/'evidence'):
    folder.mkdir(parents=True,exist_ok=True)
p=json.loads((ROOT/'configs/pipeline.head01.developer.json').read_text())['paths']
for name in ('XDG_CACHE_HOME','MPLCONFIGDIR','TORCH_HOME','HF_HOME','YOLO_CONFIG_DIR','CUDA_CACHE_PATH','TMPDIR','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR'):
    cache=OUT/'runtime'/name
    cache.mkdir(parents=True,exist_ok=True)
    os.environ[name]=str(cache)
for link,target in [(OUT/'mano_data',Path(p['wilor_repo'])/'mano_data'),(OUT/'vendor/checkpoints/mano',Path(p['wilor_repo'])/'mano_data')]:
    link.parent.mkdir(exist_ok=True,parents=True)
    if not link.exists(): link.symlink_to(target,target_is_directory=True)
mode=sys.argv[1]
assert mode in ('wilor','vidi','compare','temporal')
if mode=='compare':
    py=json.loads((ROOT/'outputs/initialization_ablation_20261109/model_paths.json').read_text())['body_python']
    search=str(ROOT)
else:
    py=p['wilor_python']
    search=os.pathsep.join((str(OUT/'vendor') if mode=='vidi' else p['wilor_repo'],str(ROOT)))
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=search,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='' if mode=='compare' else '0')
subprocess.run([py,'-Bu',str(RUN/('compare.py' if mode=='compare' else 'predict.py')),mode],cwd=OUT,env=env,check=True)
