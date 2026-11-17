"""Paths and runtime helpers; original inputs remain local and read-only."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
OUT = ROOT / os.environ.get('D_INTERVENTION_OUTPUT', 'outputs/2d_intervention_20261117_retry1')
ARCH = ROOT.parent / 'EgoFishPose/experiments/benchmark_body_hand10_head01_20261014_v1'
EVIDENCE = RUN / 'evidence'
NAMES = ['Neck', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow',
         'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe']
MODEL_IDS = [7, 9, 11, 13, 8, 10, 12, 2, 4, 6, 19, 1, 3, 5, 16]
NATIVE_IDS = [12, 17, 19, 21, 16, 18, 20, 2, 5, 8, 65, 1, 4, 7, 62]
OBS = [i for i, j in enumerate(NATIVE_IDS) if j < 22 and i != 0]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def setup_model():
    from egosmplx_pipeline.bootstrap import setup
    paths = read(ROOT / 'outputs/initialization_ablation_20261109/model_paths.json')
    os.environ.update(EGOSMPLX_CONFIG=paths['egosmplx_config'], EGOSMPLX_REPO_ROOT=paths['egosmplx_repo'])
    return setup(paths['egosmplx_repo'])


def camera_files():
    return [ARCH / 'body_calibration_fisheye' / ('camera_' + side + '.json')
            for side in ['left', 'right']]


def sanitized(text):
    return str(text).replace(str(ROOT.parent), '<WORKSPACE>')
