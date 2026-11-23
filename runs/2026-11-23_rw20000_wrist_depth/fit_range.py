"""Paired baseline/soft-range fit. Construct the extra targets from pixels only."""
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn.functional import huber_loss
from check_depth import geometry, ref, ROOT, RUN

import run_fit as base

OUT = ROOT / 'outputs/rw20000_wrist_depth_20261123'
MAX_SIGMA_M = .050
RANGE_WEIGHT = .25


class RangeProblem(base.trial.TorsoOnly):
    def __init__(self, rows, cfg, model, cameras):
        super().__init__(rows, cfg, model, cameras)
        self.target = self.t([r['wrist_range'] for r in rows])
        self.sigma = self.t([r['wrist_sigma'] for r in rows])
        self.active_range = self.t([[s > 0 and r['condition'] == 'range' for s in r['wrist_sigma']] for r in rows])
        self.wrist_ids = [next(p['native_index'] for p in rows[0]['points'] if p['name'] == name)
                          for name in ['L_Wrist', 'R_Wrist']]

    def objective(self, values):
        total, feasible, terms = super().objective(values)
        distance = torch.linalg.norm(values[0][:, self.wrist_ids], dim=-1)
        residual = (distance - self.target) / self.sigma.clamp_min(.008)
        robust = huber_loss(residual, torch.zeros_like(residual), reduction='none',
                            delta=self.cfg['metric_huber_delta'])
        extra = self.cfg['metric_loss_weight'] * RANGE_WEIGHT * (robust * self.active_range).sum(1) / 2
        terms['metric/wrist_range'] = extra
        return total + extra, feasible, terms


def main():
    def protect(event, args):
        if event == 'open' and args and isinstance(args[0], str) and Path(args[0]).name in ('reference.json', 'results.json'):
            raise PermissionError('Evaluation results and labels are unavailable during fitting.')
    sys.addaudithook(protect)
    _, camera = ref.common.setup_model()
    cams = [camera(p) for p in ref.camera_files()]
    samples = ref.read(ref.OUT / 'prepared.json')['items']
    prepared = []
    for row in samples:
        if row['condition'] != 'rw20000':
            continue
        right = np.array(row['inputs']['views']['right']['camera']['left_to_view'])
        targets, sigmas = [], []
        for name in ['L_Wrist', 'R_Wrist']:
            point = next(p for p in row['points'] if p['name'] == name)
            rec = geometry.triangulate(cams, right[:3, :3], right[:3, 3], point['xy']) if all(point['valid']) else {}
            distance = sigma = 0.
            if rec.get('accepted'):
                xyz = np.array(rec['point_cam3_m'])
                unit = xyz / np.linalg.norm(xyz)
                s = float(np.sqrt(unit @ np.array(rec['covariance_cam3_m2']) @ unit))
                if s <= MAX_SIGMA_M:
                    distance, sigma = float(np.linalg.norm(xyz)), s
            targets.append(distance)
            sigmas.append(sigma)
        if any(sigmas):
            prepared.extend(dict(row, condition=c, wrist_range=targets, wrist_sigma=sigmas) for c in ['baseline','range'])
    assert len(prepared) == 18
    ref.write(OUT / 'prepared.json', dict(items=prepared))
    (OUT / 'protocol.json').write_bytes((ref.OUT / 'protocol.json').read_bytes())
    base.fit.RUN = base.fit.OUT = OUT
    base.fit.Problem = RangeProblem
    base.fit.camera_files = ref.camera_files
    sys.argv = [__file__, '--threads', '1']
    base.fit.main()


if __name__ == '__main__':
    main()
