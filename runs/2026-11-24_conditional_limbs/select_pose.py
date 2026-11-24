"""Rerank the V3 chain proposals, including all archived complete poses."""
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from prior import RUN, ROOT, OUT, NATIVE, directions, distances

REF = ROOT/'runs/2026-11-23_fullbody_joint_candidates'
sys.path.insert(0,str(REF))
import search
from search import prev


def select(split):
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],str):
            name=Path(args[0]).name
            if name in {'reference.json','validation_truth.json','samples.npz','validation.json','test_manifest.json'} or name.endswith('_results.json'):
                raise PermissionError('Evaluation data excluded from selection')
    sys.addaudithook(guard)
    torch.set_num_threads(1)
    train = np.load(OUT/'training.npz')
    trees = [cKDTree(train['features'][:,k]) for k in range(4)]
    threshold = train['threshold']
    def penalty(xyz):
        ds = distances(trees,directions(xyz[:,NATIVE].astype(float)))
        return .01*(np.maximum(ds/threshold-1,0)**2).mean(axis=1)

    class WithPrior(prev.base.trial.TorsoOnly):
        def objective(self, values):
            raw, ok, terms = super().objective(values)
            cost = torch.as_tensor(penalty(values[0].detach().numpy()),dtype=raw.dtype)
            return raw+cost,ok,terms

    prev.base.trial.TorsoOnly = WithPrior
    prev.base.fit.camera_files = prev.ref.camera_files
    model,camera = prev.ref.common.setup_model()
    folder = prev.VAL if split=='validation' else prev.base.OUT
    cfg = prev.read(folder/'protocol.json')
    items = prev.read(folder/'prepared.json')['items']
    audit = prev.read(REF/('evidence/'+split+'_search.json'))
    def checked(path,field):
        assert prev.sha(path)==audit[field]
        return path
    checked(folder/'prepared.json','prepared_sha256')
    checked(folder/'protocol.json','protocol_sha256')
    old = dict(np.load(checked(search.OUT/(split+'_bank.npz'),'parameter_sha256')))
    choices = prev.read(checked(search.OUT/(split+'_choices.json'),'choices_sha256'))
    wrist = dict(np.load(checked(prev.OUT/(split+'_objective.npz'),'wrist_sha256')))
    windex = {p:i for i,p in enumerate(wrist['pair'])}
    poses, decisions = [], {}
    with torch.no_grad():
        for key,r in choices.items():
            offset = r['offset']
            item = items[r['index']]
            assert item['pair']==key
            w = {f:wrist[f][windex[key]] for f in search.FIELDS}
            b = {f:old[f][offset] for f in search.FIELDS}
            options,n = search.frame(item,w,b,model,camera,cfg)
            # Also include already composed original v3 poses in case of numerical ties.
            prob = WithPrior([dict(item,condition='baseline')],cfg,model,camera)
            count=len(r['names'])
            sl=slice(offset,offset+count)
            params=[torch.from_numpy(old[f][sl]).float() for f in ['body_pose','betas','global0','delta','translation']]
            xyz,_=prob.fk(*params)
            extra=penalty(xyz.numpy())
            scores=np.asarray(r['loss'])+extra
            old_pick=int(scores.argmin())
            new_pick=min(options,key=lambda c:c['loss'])
            if new_pick['loss'] < scores[old_pick]-1e-8:
                p = dict(w,**{f:new_pick[f] for f in search.FIELDS[:3]})
                name = 'recomposed/'+new_pick['name']
            else:
                p={f:old[f][offset+old_pick] for f in search.FIELDS}
                name='archived/'+r['names'][old_pick]
            poses.append(p)
            decisions[key]=dict(name=name,base=r['names'][r['best']],chain_trials=n,
                                base_penalty=float(extra[r['best']]))
            if len(poses)%25==0:
                print(split,len(poses),flush=True)
    np.savez_compressed(OUT/(split+'_selected.npz'),**{f:np.stack([p[f] for p in poses]) for f in search.FIELDS})
    prev.write(OUT/(split+'_decisions.json'),decisions)
    prev.write(RUN/('evidence/'+split+'_selection.json'),dict(count=len(poses),
        archived_inputs=audit,selected_sha256=prev.sha(OUT/(split+'_selected.npz')),
        training_sha256=prev.sha(OUT/'training.npz'),decision_sha256=prev.sha(OUT/(split+'_decisions.json')),exit_status=0))


if __name__=='__main__':
    select(sys.argv[1])
