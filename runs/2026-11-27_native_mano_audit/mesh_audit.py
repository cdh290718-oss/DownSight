"""Paired MANO audit and factorized SMPL-X wrist deformation."""
from pathlib import Path
from collections import Counter
import hashlib,json,sys
import numpy as np
import torch
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
OUT=ROOT/'outputs/native_mano_audit_20261127'
sys.path.insert(0,str(RUN.parent/'2026-11-23_floor24_intersections'))
from inspect_obj import ref,groups,self_intersections
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
archive=read(RUN.parent/'2026-11-27_v5_hand_self_review/evidence/hands.json')
sources=read(RUN.parent/'2026-11-27_validation200_v5/evidence/selection.json')

def native():
    replay=read(RUN/'evidence/replay.json')
    results={}
    for k,row in archive.items():
        for side,h in row.items():
            if h['source'] not in ('cam3','cam4'):
                assert h['v5']==0
                continue
            key=k+'_'+side
            path=OUT/'poses'/(key+'.npz')
            assert sha(path)==replay[key]['sha256']
            ab=np.load(path)
            hits=self_intersections(ab['vertices'].astype(float),ab['faces'].astype(int))
            faces=ab['faces'].astype(int)
            initial=set(self_intersections(ab['neutral'].astype(float),faces))
            parts=ab['weights'][faces].sum(1)
            names=np.array(['palm','index','middle','pinky','ring','thumb'])
            labels=names[np.column_stack([parts[:,0],parts[:,1:].reshape(-1,5,3).sum(2)]).argmax(1)]
            regions=Counter('/'.join(sorted([labels[l],labels[r]])) for l,r in hits)
            results[key]=dict(native_regions=regions,overlap_regions=sorted(set(regions)&set(h['regions'])),neutral=len(initial),native_new_face_pairs=len(set(hits)-initial),native=len(hits),transferred_fingers=h['finger_only'],final=h['v5'],source=h['source'])
    assert set(results)==set(replay)
    (RUN/'evidence/native.json').write_text(json.dumps(results,indent=1)+'\n')

def wrist():
    from smplx.lbs import batch_rodrigues,batch_rigid_transform
    model,_=ref.common.setup_model()
    layer=model.layer['neutral'];faces=np.asarray(model.face)
    labels=groups(layer,faces);weights=layer.lbs_weights
    rest=layer.v_template.unsqueeze(0)
    restj=(layer.J_regressor@rest[0])[None]
    def skin(x,g):
        t=lambda z:torch.as_tensor(z,dtype=torch.float32).reshape(1,-1)
        params=layer(global_orient=t(g),betas=t(np.zeros(10)),body_pose=t(x[:21]),left_hand_pose=t(x[23:38]),right_hand_pose=t(x[38:]),jaw_pose=t(np.zeros(3)),leye_pose=t(np.zeros(3)),reye_pose=t(np.zeros(3)),expression=t(np.zeros(10)),return_full_pose=True)
        rot=batch_rodrigues(params.full_pose.reshape(-1,3)).reshape(1,-1,3,3)
        offset=((rot[:,1:]-torch.eye(3)).reshape(1,-1)@layer.posedirs).reshape(-1,3)
        js,A=batch_rigid_transform(rot,restj,layer.parents)
        T=(weights@A[0].reshape(-1,16)).reshape(-1,4,4)
        def apply(off):
            return (T[:,:3,:3]@(rest[0]+off)[:,:,None])[:,:,0]+T[:,:3,3]
        verts=apply(offset)
        assert torch.max(abs(verts-params.vertices[0]))<2e-6
        return verts.numpy().astype(float),offset,apply,A[0].numpy().astype(float),js[0].numpy().astype(float)
    results={}
    for k,row in archive.items():
        if not any(v['wrist_only'] for v in row.values()): continue
        pa=ROOT/'outputs/validation200_fingers_20261125/poses'/(k+'.npz')
        pb=ROOT/'outputs/validation200_v5_20261127/rows'/(k+'.npz')
        assert sha(pa)==sources[k]['seed_sha256'] and sha(pb)==sources[k]['sha256']
        z=np.load(pa);a=z['fixed_full'];b=a.copy();b[19:21]=np.load(pb)['v5'][19:21]
        va,oa,fa,A,ja=skin(a,z['global0'])
        vb,ob,fb,B,jb=skin(b,z['global0'])
        for i,(side,h) in enumerate(row.items()):
            if not h['wrist_only']:continue
            w=20+i
            rigid=(va-ja[w])@(B[w,:3,:3]@A[w,:3,:3].T).T+jb[w]
            f=faces[labels==('L_hand','R_hand')[i]]
            hits={n:self_intersections(v,f) for n,v in [('v4',va),('rigid',rigid),('weights_only',fb(oa).numpy().astype(float)),('correctives_only',fa(ob).numpy().astype(float)),('both',vb)]}
            assert not hits['v4'] and not hits['rigid'] and len(hits['both'])==h['wrist_only']
            idx=25+15*i
            partweights=weights.numpy()[f].sum(1)
            parts=np.array(['palm','index','middle','pinky','ring','thumb'])[np.column_stack([partweights[:,w],partweights[:,idx:idx+15].reshape(-1,5,3).sum(2)]).argmax(1)]
            ids=np.unique(f[np.unique(np.array(hits['both']).ravel())]).astype(int)
            outside=1.-weights[ids,w].numpy()-weights[ids,idx:idx+15].sum(1).numpy()
            distance=1000*np.linalg.norm(vb[ids]-jb[w],axis=1)
            results[k+'_'+side]=dict(outside_hand_chain_weight_range=[float(outside.min()),float(outside.max())],dist_from_wrist_mm=[float(distance.min()),float(distance.max())],counts={n:len(v) for n,v in hits.items()},parts=Counter('/'.join(sorted([parts[l],parts[r]])) for l,r in hits['both']),max_vertex_difference_from_rigid_mm=float(1000*np.linalg.norm(vb[ids]-rigid[ids],axis=1).max()))
    assert len(results)==10
    (RUN/'evidence/wrist.json').write_text(json.dumps(results,indent=1)+'\n')
if __name__=='__main__':
    torch.set_num_threads(1)
    with torch.no_grad(): {'wrist':wrist,'native':native}[sys.argv[1]]()
