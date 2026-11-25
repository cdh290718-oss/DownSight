import hashlib
import json
from pathlib import Path

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
OUT = ROOT/'outputs/realvideo4_dv3_20261125'
DATA = ROOT/'data/realdata'
CAL = DATA/'calibration/2026-11-08_lower_fisheye'
SESSION = DATA/'session_20261108_180008'
SIDES = ['left','right']

def read(path):
    return json.loads(path.read_text())

def write(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4<<20),b''):
            h.update(chunk)
    return h.hexdigest()

def cameras():
    return [OUT/(side+'_cal.json') for side in SIDES]
