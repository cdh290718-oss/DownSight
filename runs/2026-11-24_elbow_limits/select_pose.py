"""Select a feasible V3 pose under fixed template-based elbow limits."""
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot
from limits import Elbow, CHAINS, LOW, HIGH, legal

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
OUT = ROOT/'outputs/elbow_limits_20261124'
REF = ROOT/'runs/2026-11-23_fullbody_joint_candidates'
sys.path.insert(0,str(REF))
import search
from search import prev


def read_angles(body, axes):
    return np.stack([a.decompose(Rot.from_rotvec(body[:,e-1]).as_matrix())
                     for a,(_,e,_) in zip(axes,CHAINS)],axis=1)


def select(split):
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],str):
            if Path(args[0]).name in {'reference.json','validation_truth.json','training.npz','samples.npz','validation.json','test_manifest.json'} or Path(args[0]).name.endswith('_scores.json'):
                raise PermissionError('No pose library or scoring reference during selection')
    sys.addaudithook(guard)
    torch.set_num_threads(1)
    prev.base.fit.camera_files=prev.ref.camera_files
    model,camera=prev.ref.common.setup_model()
    layer=model.layer['neutral']
    rest=(layer.J_regressor@layer.v_template).detach().numpy().astype(float)
    axes=[Elbow(rest,k) for k in range(2)]
    tests=[a.check() for a in axes]
    # Positive extension-deviation and axial-turn conventions mirror on bilateral neutral geometry.
    mirror=np.diag([-1.,1.,1.])
    mirrored=rest@mirror
    for l,r in [(16,17),(18,19),(20,21)]:
        mirrored[[l,r]]=mirrored[[r,l]]
    mirrored_right=Elbow(mirrored,1)
    asymmetry=[]
    for a in [[80,0,0],[0,12,0],[0,0,80],[100,12,30]]:
        err=Rot.from_matrix((mirror@axes[0].compose(a)@mirror).T@axes[1].compose(a)).magnitude()
        asymmetry.append(float(np.rad2deg(err)))
        assert np.allclose(mirror@axes[0].compose(a)@mirror,mirrored_right.compose(a),atol=1e-8)
        # Sign of each recoverable rotation is correct on the actual template as well.
        mapped=axes[1].decompose(mirror@axes[0].compose(a)@mirror)
        active=np.abs(a)>0
        assert (np.sign(mapped[active])==np.sign(np.asarray(a)[active])).all()
    folder=prev.VAL if split=='validation' else prev.base.OUT
    cfg=prev.read(folder/'protocol.json')
    items=prev.read(folder/'prepared.json')['items']
    archive=prev.read(REF/('evidence/'+split+'_search.json'))
    for path,key in [(search.OUT/(split+'_bank.npz'),'parameter_sha256'),
                     (search.OUT/(split+'_choices.json'),'choices_sha256'),
                     (folder/'protocol.json','protocol_sha256'),(folder/'prepared.json','prepared_sha256')]:
        assert prev.sha(path)==archive[key]
    bank=dict(np.load(search.OUT/(split+'_bank.npz')))
    choices=prev.read(search.OUT/(split+'_choices.json'))
    records,poses={},[]
    for key,r in choices.items():
        cs=np.arange(len(r['names']))+r['offset']
        old=cs[r['best']]
        pose=bank['body_pose'][cs].copy()
        before=read_angles(pose,axes)
        valid=legal(before)
        valid_full=valid.all(1)
        newidx=r['best']
        if not valid_full[newidx]:
            item=items[r['index']]
            assert key==item['pair']
            p=prev.base.trial.TorsoOnly([dict(item,condition='baseline')],cfg,model,camera)
            def score(bodies,which):
                ids=cs[which]
                t=lambda v: torch.as_tensor(np.ascontiguousarray(v),dtype=torch.float32)
                body=t(bodies)
                xyz,rh=p.fk(body,t(bank['betas'][ids]),t(bank['global0'][ids]),t(bank['delta'][ids]),t(bank['translation'][ids]))
                p.bodyvar=body[:,p.body_ids]
                loss,ok,_=p.objective((xyz,rh,t(bank['betas'][ids])))
                return loss.detach().numpy(),ok.numpy()
            original,ok=score(pose,np.arange(len(cs)))
            assert ok.all() and np.max(abs(original-r['loss']))<2e-6
            for k,(s,e,w) in enumerate(CHAINS):
                bag,owners=[],[]
                for j in np.where(~valid[:,k])[0]:
                    body=pose[j]
                    # Postrotate the shoulder about its upper-arm bone; reverse it at the elbow.
                    # Prior to any clamping this preserves all distal positions and orientations.
                    g=Rot.from_rotvec(np.deg2rad(np.arange(-180,180,10))[:,None]*axes[k].upper).as_matrix()
                    shoulder=Rot.from_matrix(Rot.from_rotvec(body[s-1]).as_matrix()@g).as_rotvec()
                    elbow=g.transpose(0,2,1)@Rot.from_rotvec(body[e-1]).as_matrix()
                    angles=np.clip(axes[k].decompose(elbow),LOW,HIGH)
                    elbow=Rot.from_matrix(axes[k].compose(angles)).as_rotvec()
                    ok=(np.linalg.norm(shoulder,axis=1)<=cfg['absolute_pose_caps_deg'][('L','R')[k]+'_Shoulder']*np.pi/180+1e-6)&(np.linalg.norm(elbow,axis=1)<=cfg['absolute_pose_caps_deg'][('L','R')[k]+'_Elbow']*np.pi/180+1e-6)
                    options=np.repeat(body[None],ok.sum()+1,axis=0)
                    options[:-1,s-1],options[:-1,e-1]=shoulder[ok],elbow[ok]
                    options[-1,e-1]=0
                    bag.extend(options);owners.extend([j]*len(options))
                if bag:
                    owners=np.array(owners)
                    loss,good=score(bag,owners)
                    loss[~good]=np.inf
                    for j in np.unique(owners):
                        ix=np.flatnonzero(owners==j)
                        pick=ix[np.argmin(loss[ix])]
                        if np.isfinite(loss[pick]):
                            pose[j]=bag[pick]
            loss,good=score(pose,np.arange(len(cs)))
            good &= legal(read_angles(pose,axes)).all(1)
            if good.any():
                newidx=int(np.argmin(np.where(good,loss,np.inf)))
            else:
                # Explicit abstention; retain old pose only for failure-inclusive scoring.
                pose[newidx]=bank['body_pose'][old]
        b=pose[newidx]; idx=cs[newidx]
        after=read_angles(b[None],axes)[0]
        records[key]=dict(before=before[r['best']].tolist(),after=after.tolist(),
                          base_valid=bool(valid_full[r['best']]),original_pool_valid=int(valid_full.sum()),
                          feasible=bool(legal(after).all()),source=r['names'][newidx])
        poses.append({f:(b if f=='body_pose' else bank[f][idx]) for f in search.FIELDS})
    np.savez_compressed(OUT/(split+'_selected.npz'),**{f:np.stack([v[f] for v in poses]) for f in search.FIELDS})
    prev.write(OUT/(split+'_decisions.json'),records)
    prev.write(RUN/('evidence/'+split+'_limits.json'),dict(axes=tests,neutral_mirror_asymmetry_deg=asymmetry,limits=[LOW.tolist(),HIGH.tolist()],frames=records,
               selected_sha256=prev.sha(OUT/(split+'_selected.npz')),inputs=archive,exit_status=0))
    print(split,len(poses),sum(r['base_valid'] for r in records.values()),sum(r['feasible'] for r in records.values()))


if __name__=='__main__':
    with torch.no_grad():
        select(sys.argv[1])
