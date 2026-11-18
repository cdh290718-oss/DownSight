"""External assets are read-only; all generated assets stay in this run's output."""
import hashlib,json,os
from pathlib import Path
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
OUT=ROOT/'outputs/rw_frozen200_golden_d_20261118'
EVIDENCE=RUN/'evidence'
NAMES=['Neck','R_Shoulder','R_Elbow','R_Wrist','L_Shoulder','L_Elbow','L_Wrist','R_Hip','R_Knee','R_Ankle','R_Toe','L_Hip','L_Knee','L_Ankle','L_Toe']
MODEL_IDS=[7,9,11,13,8,10,12,2,4,6,19,1,3,5,16]
NATIVE_IDS=[12,17,19,21,16,18,20,2,5,8,65,1,4,7,62]
OBS=[i for i,j in enumerate(NATIVE_IDS) if j<22 and i!=0]
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def setup_model():
    from egosmplx_pipeline.bootstrap import setup
    p=read(ROOT/'outputs/initialization_ablation_20261109/model_paths.json')
    os.environ.update(EGOSMPLX_CONFIG=p['egosmplx_config'],EGOSMPLX_REPO_ROOT=p['egosmplx_repo'])
    return setup(p['egosmplx_repo'])
def camera_files():return [OUT/('camera_'+s+'.json') for s in ['left','right']]
