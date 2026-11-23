"""Bone geometry shared by official-label statistics and SMPL-X fitting."""
import hashlib
import json
from pathlib import Path

import torch
from torch.nn.functional import normalize

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
EVIDENCE = RUN / 'evidence'
OUTPUT = ROOT / 'outputs/rw_joint_ranges_20261123'
CHAINS = [(side + '_' + a, side + '_' + b, side + '_' + c)
          for side in ('L', 'R') for a, b, c in
          [('Shoulder', 'Elbow', 'Wrist'), ('Hip', 'Knee', 'Ankle')]]
NAMES = [chain[1] for chain in CHAINS]


def features(x, lookup):
    a, b, c = [[lookup[name] for name in column] for column in zip(*CHAINS)]
    upper = normalize(x[:, b] - x[:, a], dim=-1)
    lower = normalize(x[:, c] - x[:, b], dim=-1)
    axis = normalize(x[:, lookup['L_Hip']] - x[:, lookup['R_Hip']], dim=-1)
    turn = torch.cross(upper, lower, dim=-1)
    magnitude = torch.rad2deg(torch.atan2(turn.norm(dim=-1), (upper * lower).sum(-1)))
    direction = (turn * axis[:, None]).sum(-1)
    return torch.stack((magnitude, direction), dim=-1)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
