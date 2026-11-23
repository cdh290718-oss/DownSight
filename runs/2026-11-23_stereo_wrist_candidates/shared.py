from pathlib import Path
import sys
import numpy as np

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
OUT = ROOT/'outputs/stereo_wrist_candidates_20261123'
VAL = OUT/'validation_fit'
VM = ROOT/'outputs/rw_a20000_dataset_20261117/validation.json'
TM = ROOT/'outputs/rw_test_completion_20261117/test_manifest.json'
sys.path.insert(0, str(ROOT/'runs/2026-11-19_rw_frozen200_free12'))
import measure as ref
import run_fit as base
from metrics import align

read, write, sha = ref.read, ref.write, ref.sha
def pair_images(manifest):
    pairs={}
    for row in read(manifest)['records']:
        pair=pairs.setdefault(row['pair_id'],{})
        assert row['view'] not in pair
        pair[row['view']]=row
    return [(v['left'],v['right']) for _,v in sorted(pairs.items())]

def mapping():
    ids=read(ROOT/'runs/2026-11-16_sapiens-unrealego-audit/evidence/audit.json')['joint_mapping']
    lookup={r['rw_index']:r['sapiens308_id'] for r in ids}
    return [lookup[j] for j in ref.common.OBS]
