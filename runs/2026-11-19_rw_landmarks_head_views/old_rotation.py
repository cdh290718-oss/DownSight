"""Run the legacy full SMPL-X solver with only its rotation weight changed."""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

run = Path(__file__).resolve().parent
root = run.parents[1]
prior = root/'runs/2026-11-15_unrealego-gt2d'
out = root/'outputs/rw_landmarks_head_views_20261119'
parser = argparse.ArgumentParser()
parser.add_argument('pair')
parser.add_argument('mode', choices=['weak', 'off'])
args = parser.parse_args()
dest = out/'old'/args.mode/args.pair
dest.mkdir(parents=True, exist_ok=False)
source = root/'outputs/unrealego_gt2d_ideal_20261115'/args.pair
for name in ['inputs.json', 'model_paths.json', 'triangulated.json', 'anchors/oracle.json']:
    (dest/name).parent.mkdir(exist_ok=True)
    shutil.copyfile(source/name, dest/name)
cfg = json.loads((prior/'settings.json').read_text())
cfg['head_rotation_weight'] *= {'weak':0.1,'off':0.0}[args.mode]
(dest/'settings.json').write_text(json.dumps(cfg))
shutil.copyfile(prior/'cases.json', dest/'cases.json')
os.environ['D_BATCH_PAIR'] = args.pair
sys.path.insert(0,str(prior))
import fit
fit.OUT = dest
fit.RUN = dest
sys.argv = [__file__, '--case','D']
fit.main()
