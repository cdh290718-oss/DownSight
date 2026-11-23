from pathlib import Path
import json, hashlib
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
OUT=ROOT/'outputs/test500_joint_20261123'
FIT=OUT/'fit'
BACK=ROOT.parent/'EgoSMPLX'
SOURCE=BACK/'data/UnrealEgo_RW_Test/prepared/samples.npz'
OLD=ROOT/'outputs/rw_test_completion_20261117'
def read(p): return json.loads(p.read_text())
def write(p,data): p.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):h.update(block)
    return h.hexdigest()
