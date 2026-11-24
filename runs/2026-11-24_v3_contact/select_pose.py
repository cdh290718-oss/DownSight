"""Single-frame collision-aware selection on the fixed neutral body."""
import itertools
from pathlib import Path
import sys
import numpy as np
import torch

from contact import Contact, EV, OUT, ROOT, TOL_MM, penalty, preserves_near
sys.path.insert(0,str(ROOT/'runs/2026-11-23_fullbody_joint_candidates'))
import search
from search import prev
sys.path.insert(0,str(ROOT/'runs/2026-11-19_rw_landmarks_head_views'))
from export import geometry


def vertices(layer,param):
    fields={k:param[n] for k,n in [('global_orient','global0'),('body_pose','body_pose'),('transl','translation'),('betas','betas')]}
    return geometry(layer,fields,param['delta'],False,None)[0]


def extra_options(item, param, affected, model, camera, cfg):
    """Keep varied low-loss chains about the v3 winning trunk, then combine their poses."""
    problem=prev.base.trial.TorsoOnly([dict(item,condition='baseline')],cfg,model,camera)
    caps=dict(zip(problem.body_ids.tolist(),problem.caps.numpy()))
    original=param['body_pose']
    def score(poses):
        body=torch.tensor(np.stack(poses),dtype=torch.float32)
        def tile(k):return torch.from_numpy(param[k]).float().reshape(1,-1).expand(len(body),-1)
        native,rh=problem.fk(body,tile('betas'),tile('global0'),tile('delta'),tile('translation'))
        problem.bodyvar=body[:,problem.body_ids]
        loss,ok,_=problem.objective((native,rh,tile('betas')))
        return native.numpy(), np.where(ok.numpy(),loss.numpy(),np.inf)
    joint=score([original])[0][0].astype(float)
    world=[search.rot(param['delta'])@search.rot(param['global0'])]
    for i in range(1,22):
        world.append(world[problem.fk.parents[i]]@search.rot(original[i-1]))
    points={p['native_index']:p for p in item['points']}
    banks={}
    for idx in sorted(set(itertools.chain.from_iterable(affected))):
        s,e,w=search.CHAINS[idx]
        poses=[original.copy()]
        if all(points[e]['valid']) and all(points[w]['valid']):
            ray=problem.cameras[0].camera2world_ray(np.array(points[w]['xy'][:1]))[0]
            local,_,_=search.chain_options(joint,world,problem.fk.parents,(s,e,w),
                [search.unit(joint[w]),search.unit(ray)],np.linalg.norm(joint[w]),[caps[s-1],caps[e-1]])
            for angles in local:
                p=original.copy()
                p[[s-1,e-1]]=angles
                poses.append(p)
        pos,loss=score(poses)
        chosen=[0]
        for i in np.argsort(loss,kind='stable'):
            if not np.isfinite(loss[i]): break
            if all(np.max(np.linalg.norm(pos[i,[e,w]]-pos[j,[e,w]],axis=1))>.012 for j in chosen):
                chosen.append(i)
            if len(chosen)==5: break
        banks[idx]=[poses[i] for i in chosen]
    candidates=[]
    for ia,ib in affected:
        a,b=search.CHAINS[ia],search.CHAINS[ib]
        for pa,pb in itertools.product(banks[ia],banks[ib]):
            p=original.copy()
            for ch,q in [(a,pa),(b,pb)]:
                ids=np.asarray(ch[:2])-1
                p[ids]=q[ids]
            candidates.append(p)
    # Palm orientations are weakly observed: try capped local wrist rotations at a fixed wrist centre.
    for idx in sorted(set(itertools.chain.from_iterable(affected))&{0,1}):
        w=search.CHAINS[idx][-1]-1
        for axis in np.eye(3):
            for degrees in [-30.,-15.,15.,30.]:
                aa=search.rv(search.rot(original[w])@search.rot(axis*np.deg2rad(degrees)))
                if np.linalg.norm(aa)<=caps[w]:
                    p=original.copy()
                    p[w]=aa
                    candidates.append(p)
    unique={p.tobytes():p for p in candidates if not np.array_equal(p,original)}
    if not unique:return []
    poses=list(unique.values())
    _,losses=score(poses)
    return [('extra'+str(i),dict(param,body_pose=p),float(loss)) for i,(p,loss) in enumerate(zip(poses,losses)) if np.isfinite(loss)]


def main(split):
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],str):
            name=Path(args[0]).name
            if name in ('reference.json','validation_truth.json','samples.npz','validation.json','test_manifest.json') or name.endswith('_results.json'):
                raise PermissionError('Evaluation labels excluded from selection')
    sys.addaudithook(guard)
    torch.set_num_threads(1)
    prev.base.fit.camera_files=prev.ref.camera_files
    model,camera=prev.ref.common.setup_model()
    layer=model.layer['neutral']
    contact=Contact(layer,np.asarray(model.face))
    folder=prev.VAL if split=='validation' else prev.base.OUT
    audit=prev.read(search.RUN/('evidence/'+split+'_search.json'))
    def checked(path, key):
        assert prev.sha(path)==audit[key]
        return path
    cfg=prev.read(checked(folder/'protocol.json','protocol_sha256'))
    items=prev.read(checked(folder/'prepared.json','prepared_sha256'))['items']
    bank=dict(np.load(checked(search.OUT/(split+'_bank.npz'),'parameter_sha256')))
    assert not bank['betas'].any()
    choices=prev.read(checked(search.OUT/(split+'_choices.json'),'choices_sha256'))
    rows,selected={},[]
    limb={'L_hand':0,'R_hand':1,'L_thigh':2,'R_thigh':3}
    with torch.no_grad():
        for pair,r in choices.items():
            item=items[r['index']]
            assert item['pair']==pair
            variants=[]
            for i,(name,loss) in enumerate(zip(r['names'],r['loss'])):
                p={k:bank[k][r['offset']+i].copy() for k in search.FIELDS}
                variants.append((name,p,loss))
            name,orig,raw=variants[r['best']]
            before=contact.measure(vertices(layer,orig))
            bestscore=raw+penalty(before)
            pick,after,method,finalraw=orig,before,'v3',raw
            affected=[tuple(limb[n] for n in key.split('/')) for key,(d,g) in before.items() if d>TOL_MM]
            n=0
            if affected:
                variants+=extra_options(item,orig,affected,model,camera,cfg)
                for tag,params,loss in sorted(variants,key=lambda v:v[2]):
                    if loss>=bestscore-1e-9:break
                    if all(np.array_equal(params[k],orig[k]) for k in search.FIELDS):continue
                    m=contact.measure(vertices(layer,params))
                    n+=1
                    total=loss+penalty(m)
                    if total<bestscore-1e-9 and preserves_near(before,m) and penalty(m)<penalty(before):
                        bestscore,pick,after,method,finalraw=total,params,m,tag,loss
            selected.append(pick)
            rows[pair]=dict(before=before,after=after,selected=method,base_name=name,mesh_alternatives=n,
                            v3_loss=raw,selected_loss=finalraw,selected_total=bestscore)
            if len(rows)%25==0:print(split,len(rows),sum(v['selected']!='v3' for v in rows.values()),flush=True)
    path=OUT/(split+'_selected.npz')
    np.savez_compressed(path,pairs=list(rows),**{k:np.stack([p[k] for p in selected]) for k in search.FIELDS})
    prev.write(EV/(split+'_selection.json'),dict(rows=rows,parameter_sha256=prev.sha(path),inputs=audit,exit_status=0))


if __name__=='__main__':
    main(sys.argv[1])
