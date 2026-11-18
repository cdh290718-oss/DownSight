"""Run-local paths, with external sources opened read-only."""
import json,os,hashlib
from pathlib import Path
RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[1];OUT=ROOT/'outputs/real001_golden_d_four_models_20261118';EVIDENCE=RUN/'evidence'
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def setup_model():
 from egosmplx_pipeline.bootstrap import setup
 p=read(ROOT/'outputs/initialization_ablation_20261109/model_paths.json');os.environ.update(EGOSMPLX_CONFIG=p['egosmplx_config'],EGOSMPLX_REPO_ROOT=p['egosmplx_repo']);return setup(p['egosmplx_repo'])
def camera_files():return [r['row']['calibration'] for r in read(OUT/'records.json')]
