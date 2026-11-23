"""Generate local SMPL-X arm solutions and stereo patch evidence without scoring labels."""
import sys
import cv2
import torch
from scipy.spatial.transform import Rotation
from shared import *

OFFSETS_MM = [-300,-200,-120,-80,-40,-20,0,20,40,80,120,200,300]
ARMS = [(16,18,20),(17,19,21)]

def rot(v):
    return Rotation.from_rotvec(v).as_matrix()

def rv(m):
    return Rotation.from_matrix(m).as_rotvec()

def unit(x):
    x=np.asarray(x,dtype=np.float64)
    return x / max(np.linalg.norm(x),1e-12)

def turn(a,b):
    a,b=unit(a),unit(b)
    axis=np.cross(a,b)
    sine=np.linalg.norm(axis)
    if sine<1e-12:
        if a@b>0: return np.eye(3)
        axis=np.cross(a,np.eye(3)[np.argmin(abs(a))])
    return rot(unit(axis)*np.arctan2(sine,a@b))

def arm_ik(joint,world,parents,arm,target):
    s,e,w=arm
    S,E,W=joint[[s,e,w]]
    a,b=np.linalg.norm(E-S),np.linalg.norm(W-E)
    dist=np.linalg.norm(target-S)
    if not abs(a-b)+1e-6<dist<a+b-1e-6: return None
    axis=(target-S)/dist
    along=(a*a-b*b+dist*dist)/(2*dist)
    radial=E-S-axis*((E-S)@axis)
    if np.linalg.norm(radial)<1e-7: return None
    elbow=S+along*axis+unit(radial)*np.sqrt(max(0,a*a-along*along))
    shoulder_rot=turn(E-S,elbow-S)
    new_s=shoulder_rot@world[s]
    new_e=turn(shoulder_rot@(W-E),target-elbow)@shoulder_rot@world[e]
    return np.stack([rv(world[parents[s]].T@new_s),rv(new_s.T@new_e)])

def patch_cost(images,uv):
    # An oriented wrist patch plus an elongated forearm patch in each original image.
    # Sample subpixel BGR values; centered and normalized intensities reduce exposure bias.
    desc=[]; textures=[]
    for im,(w,e) in zip(images,uv):
        length=np.linalg.norm(e-w)
        if length<8: return None
        axis=(e-w)/length; normal=np.array([-axis[1],axis[0]])
        blocks=[]
        for center,extent,halfwidth in [(w,8,8),(w+.22*(e-w),min(24,.18*length),6)]:
            x,y=np.meshgrid(np.linspace(-extent,extent,9),np.linspace(-halfwidth,halfwidth,7))
            grid=center+x[:,:,None]*axis+y[:,:,None]*normal
            if (grid.min((0,1))<1).any() or (grid.max((0,1))>=np.array(im.shape[1::-1])-2).any():
                return None
            patch=cv2.remap(im,grid[:,:,0].astype('f'),grid[:,:,1].astype('f'),cv2.INTER_LINEAR).astype(float)
            mono=patch.mean(2)
            textures.append(float(mono.std()))
            vec=(patch-patch.mean((0,1))).ravel()
            blocks.append(vec/max(np.linalg.norm(vec),1e-8))
        desc.append(blocks)
    return float(1-np.mean([a@b for a,b in zip(*desc)])),float(min(textures))

def frame_candidates(item,params,i,model,camera,images,adapters,cfg):
    p=base.trial.TorsoOnly([dict(item,condition='baseline')],cfg,model,camera)
    pose=params['body_pose'][i].copy()
    seed,delta,tr=(torch.from_numpy(params[k][i:i+1]).float() for k in ('global0','delta','translation'))
    def eval_body(poses):
        b=torch.from_numpy(np.array(poses)).float()
        n=len(b); betas=torch.zeros(n,10)
        native,rh=p.fk(b,betas,seed.expand(n,3),delta.expand(n,3),tr.expand(n,3))
        p.bodyvar=b[:,p.body_ids]
        loss,ok,_=p.objective((native,rh,betas))
        return native.numpy(),loss.numpy(),ok.numpy()
    native,_,_=eval_body([pose]); joint=native[0].astype(float)
    full=ref.full_joints(params,model.layer['neutral'],i)
    assert np.max(abs(full[:22]-joint))<5e-5
    parents=p.fk.parents; world=[rot(delta[0].numpy())@rot(seed[0].numpy())]
    for k in range(1,22): world.append(world[parents[k]]@rot(pose[k-1]))
    cameras=p.cameras
    output=[]
    for arm in ARMS:
        s,e,w=arm
        pt=next(z for z in item['points'] if z['native_index']==w)
        elbow=next(z for z in item['points'] if z['native_index']==e)
        poses=[pose]; targets=[joint[w]]; enabled=all(pt['valid']) and all(elbow['valid'])
        if enabled:
            left_ray=unit(cameras[0].camera2world_ray(np.array(pt['xy'][:1]))[0])
            for ray in [unit(joint[w]),left_ray]:
                for shift in OFFSETS_MM:
                    target=(np.linalg.norm(joint[w])+shift/1000)*ray
                    if target[2]<=0 or any(np.linalg.norm(target-old)<.002 for old in targets): continue
                    local=arm_ik(joint,world,parents,arm,target)
                    if local is None: continue
                    if any(np.linalg.norm(q)>np.deg2rad(cfg['absolute_pose_caps_deg'][name]) for q,name in zip(local,[pt['name'].replace('Wrist','Shoulder'),elbow['name']])): continue
                    new=pose.copy();new[[s-1,e-1]]=local
                    poses.append(new); targets.append(target)
        xyz,loss,ok=eval_body(poses)
        assert ok[0] and np.isclose(loss[0],params['loss'][i],rtol=1e-4,atol=2e-6)
        error=np.max(np.linalg.norm(xyz[:,w]-targets,axis=1))
        assert error<5e-5, (item["pair"],w,error)
        observations=[]
        for v,side in enumerate(['left','right']):
            T=np.asarray(item['inputs']['views'][side]['camera']['left_to_view'])
            q=xyz[:,[w,e]].reshape(-1,3)@T[:3,:3].T+T[:3,3]
            uv=(cameras[v].world2camera(q)-adapters[v]['offset'])/adapters[v]['scale']
            observations.append(uv.reshape(-1,2,2))
        choices=[]
        for j in np.where(ok)[0]:
            uv=[v[j] for v in observations]
            photo=patch_cost(images,uv) if enabled else None
            choices.append(dict(loss=float(loss[j]),photo=photo,depth=float(np.linalg.norm(xyz[j,w])),
                 pose=poses[j][[s-1,e-1]].tolist(),wrist=xyz[j,w].tolist(),elbow=xyz[j,e].tolist()))
        output.append(choices)
    return output

def make(split):
    torch.set_num_threads(1);cv2.setNumThreads(1)
    if split=='test': assert (RUN/'evidence/frozen_rule.json').is_file()
    folder,manifest,condition=(VAL,VM,'baseline') if split=='validation' else (base.OUT,TM,'rw20000')
    cfg=read(folder/'protocol.json')
    items=read(folder/'prepared.json')['items']
    pairs=pair_images(manifest)
    ids=read(folder/'fit_complete.json')['ids']
    selected=[i for i,(_,c,_) in enumerate(ids) if c==condition]
    assert len(selected)==len(pairs)
    params=dict(np.load(folder/'budget_20000.npz'))
    if split=='test':
        assert sha(folder/'budget_20000.npz')==read(base.RUN/'evidence/execution.json')['params_sha256']
        # Validate ordering by both IDs from original fitting provenance.
        pmap={r['pair']:r['image_ids'] for r in read(base.GOLD/'evidence/preparation.json')['pairs']}
    model,camera=ref.common.setup_model()
    base.fit.camera_files=ref.camera_files
    adapters=[read(p)['input_adapter'] for p in ref.camera_files()]
    result={}
    failures=[]
    with torch.no_grad():
        for i,pair in zip(selected,pairs):
            item=items[i]; key=item['pair']
            assert ids[i]==[key,condition,'2d']
            if split=='test': assert [r['id'] for r in pair]==pmap[key]
            if params['selected_step'][i]<0:
                assert split=='validation'
                failures.append(key)
                continue
            images=[]
            for r in pair:
                f=manifest.parent/r['image']
                assert sha(f)==r['image_sha256']
                im=cv2.imread(str(f)); assert list(im.shape[1::-1])==r['size']
                images.append(im)
            result[key]=dict(index=i,subject=pair[0]['subject'],arms=frame_candidates(
                item,params,i,model,camera,images,adapters,cfg))
    write(OUT/(split+'_candidates.json'),result)
    write(RUN/('evidence/'+split+'_coverage.json'),dict(pairs=len(pairs),feasible=len(result),
          excluded_no_baseline=failures,candidate_sha256=sha(OUT/(split+'_candidates.json')),
          baseline_sha256=sha(folder/'budget_20000.npz')))

if __name__=='__main__':
    def protect(event,args):
        if event=='open' and args and isinstance(args[0],str) and Path(args[0]).name in {'reference.json','validation_truth.json','samples.npz'}:
            raise PermissionError('evaluation labels are excluded from candidate generation')
    sys.addaudithook(protect)
    make(sys.argv[1])
