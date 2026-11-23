"""Fit a paired copy of each RW20000 observation under the two weighting rules."""
from pathlib import Path
import sys
import torch

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
OUT = ROOT / 'outputs/rw20000_equal12_20261120'
REF = ROOT / 'runs/2026-11-19_rw_frozen200_free12'
sys.path.insert(0, str(REF))
import run_fit as base

read, write, sha = base.read, base.write, base.sha


class Paired(base.trial.TorsoOnly):
    def __init__(self, rows, cfg, model, cameras):
        super().__init__(rows, cfg, model, cameras)
        equal = torch.tensor([r['condition'] == 'equal' for r in rows])
        coefficients = torch.zeros_like(self.selected, dtype=torch.float32)
        for name, members in self.group_ids.items():
            assert (members.sum(1) > 0).all()
            # Each group keeps its original total coefficient. For the equal arm
            # each receives the mean of all observed joints after averaging eyes.
            members[equal] = self.selected[equal].float()
            weight = cfg['shoulder_weight'] if name == 'shoulder' else 1.
            coefficients += weight * members / members.sum(1, keepdim=True)
        assert torch.allclose(coefficients.sum(1),torch.full((len(rows),),2.25))
        assert torch.allclose(coefficients[equal],2.25*self.selected[equal]/self.selected[equal].sum(1,keepdim=True))
        write(RUN/'evidence/weights.json',dict(joints=[p['name'] for p in rows[0]['points']],
            rows=[dict(pair=r['pair'],condition=r['condition'],coefficients=c.tolist(),valid_views=int(v.sum()))
                  for r,c,v in zip(rows,coefficients,self.valid)],metric_loss_weight=cfg['metric_loss_weight']))


def main():
    inputs = read(REF/'evidence/inputs.json')
    for name, expected in [('prepared.json',inputs['solver_input_sha256']),('protocol.json',inputs['protocol_sha256'])]:
        assert sha(base.OUT/name)==expected
    for p in base.camera_files():
        assert sha(p)==inputs['cameras'][p.name]
    manifest=read(ROOT/'configs/golden_d.2d-v2.json')
    assert manifest['method_id']=='golden-d-2d-v2'
    for f in manifest['locked_files']:
        assert sha(ROOT/f['path'])==f['sha256']
    archive=read(base.OUT/'prepared.json')['items']
    samples=[r for r in archive if r['condition']=='rw20000']
    assert len(samples)==len({r['pair'] for r in samples})==100
    assert all(len(r['points'])==12 and not any(p['accepted'] for p in r['points']) for r in samples)
    prepared=[dict(r,condition=c) for r in samples for c in ('baseline','equal')]
    # baseline is the solver's internal reference label when building the zero-rank shape basis.
    write(OUT/'prepared.json',dict(items=prepared))
    (OUT/'protocol.json').write_bytes((base.OUT/'protocol.json').read_bytes())
    def protect(event,args):
        if event=='open' and args and isinstance(args[0],str) and Path(args[0]).name=='reference.json':
            raise PermissionError('Evaluation labels are unavailable during fitting.')
    sys.addaudithook(protect)
    base.fit.RUN=base.fit.OUT=OUT
    base.fit.Problem=Paired
    base.fit.camera_files=base.camera_files
    sys.argv=[__file__,'--threads','1']
    base.fit.main()


if __name__=='__main__':
    main()
