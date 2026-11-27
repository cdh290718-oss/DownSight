"""Transfer native effective rotations, preserving a fixed SMPL-X body, shape and wrist centers."""
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
OUT=ROOT/'outputs/hand_pose_transfer_20261127'
BASE=RUN.parent/'2026-11-27_smplx_stereo_fingers'
BASE_OUT=ROOT/'outputs/smplx_stereo_fingers_20261127'
NATIVE=ROOT/'outputs/wilor_stereo_fingers_20261126'
sys.path.insert(0,str(BASE))
from compare_fit import ref,read,write,sha,paired,DIGITS,TIPS,SIDES,JOINT_NAMES
from gallery import geometry
from calibrate import projection

def mat(r):
    return Rotation.from_rotvec(r).as_matrix()
def rv(m):
    return Rotation.from_matrix(m).as_rotvec()
def unit(x):
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def angle(x,y):
    return np.rad2deg(np.arccos(np.clip((unit(x)*unit(y)).sum(-1),-1,1)))
def palm_basis(j):
    up=unit(j[9]-j[0])
    across=j[5]-j[17]
    across=unit(across-up*(up@across))
    return np.column_stack((across,up,np.cross(across,up)))
def bones(j):
    return np.diff(j[1:].reshape(5,4,3),axis=1).reshape(15,3)
def opening(j):
    return float(np.linalg.norm(j[4]-j[8])/np.linalg.norm(j[5]-j[17]))
def rms(v):
    return float(np.sqrt(np.mean(np.square(v))))

def mesh(layer,x,p):
    return geometry(layer,dict(global_orient=p['global0'],body_pose=x[:21],transl=x[22],betas=p['betas'],
                               left_hand_pose=x[23:38],right_hand_pose=x[38:]),x[21],False,None)[:2]

def main():
    torch.set_num_threads(1)
    model,_=ref.prev.ref.common.setup_model()
    layer=model.layer['neutral'].eval()
    assert not layer.use_pca
    means=layer.pose_mean.detach().numpy().reshape(-1,3)
    assert means.shape==(55,3) and not means[:25].any()
    rotations=[d+str(k) for d in ('index','middle','pinky','ring','thumb') for k in (1,2,3)]
    names=['wrist']+[d+s for d in DIGITS for s in ('1','2','3','')]
    ids=np.array([[JOINT_NAMES.index(side+'_'+n) for n in names] for side in SIDES])
    for i,side in enumerate(SIDES):
        assert list(JOINT_NAMES[25+15*i:40+15*i])==[side+'_'+n for n in rotations]
    native=read(OUT/'native.json')
    receipts=read(RUN.parent/'2026-11-27_wilor_native_hands/evidence/results.json')['cases']
    cams=list(map(read,ref.cameras()))
    assert [sha(p) for p in ref.cameras()]==read(ref.PRIOR/'evidence/calibration.json')['converted']
    summaries={}
    for row in read(ref.OLD/'fit/0/prepared.json')['items'][2:4]:
        key=row['pair']
        rec=read(BASE/'evidence'/(key+'.json'))['results']['B']
        file=BASE_OUT/'poses'/(key+'_B.npz')
        assert sha(file)==rec['pose_sha256']
        pack=dict(np.load(file))
        seed=pack['locked']
        _,j0=mesh(layer,seed,pack)
        neutral=np.zeros_like(seed)
        neutral[23:]=-means[25:]
        _,rest=mesh(layer,neutral,dict(pack,global0=np.zeros(3)))
        world=[mat(seed[21])@mat(pack['global0'].reshape(3))]
        for n in range(1,22):
            world.append(world[int(layer.parents[n])]@mat(seed[n-1]))
        transforms=[np.array(row['inputs']['views'][eye]['camera']['left_to_view']) for eye in SIDES]
        def project(points):
            return np.stack([projection(points@t[:3,:3].T+t[:3,3],cam) for cam,t in zip(cams,transforms)])
        results={}
        for source,eye in enumerate(SIDES):
            tag='cam'+str(3+source)
            x=seed.copy()
            tags={}
            source_joints={}
            for i,side in enumerate(SIDES):
                label=key+'_'+eye+'_'+side[0].upper()
                path=NATIVE/(label+'.npz')
                assert sha(path)==receipts[label]['input_pack_sha256']
                p=np.load(path)
                mirror=np.diag([-1,1,1]) if i==0 else np.eye(3)
                q=palm_basis(rest[ids[i]])@palm_basis(np.array(native[label]['rest'])).T
                orient=transforms[source][:3,:3].T@mirror@p['global_orient'][0]@mirror@q.T
                e,w=18+i,20+i
                x[w-1]=rv(world[e].T@orient)
                local=q@mirror@p['hand_pose']@mirror@q.T
                s=slice(23+15*i,38+15*i)
                mean=means[25+15*i:40+15*i]
                x[s]=rv(local)-mean
                residual=np.max(np.abs(mat(x[s]+mean)-local))
                assert residual<1e-4
                sj=np.array(native[label]['posed'])@transforms[source][:3,:3]
                source_joints[side]=sj
                tags[side]=dict(effective_rotation_max_abs=float(residual),rest_alignment_deg=float(np.rad2deg(np.linalg.norm(rv(q)))),
                                mean_norm_rad=float(np.linalg.norm(mean)),source_pack_sha256=sha(path))
            for suffix in ('', '_wrist', '_B_wrist'):
                mode=tag+suffix
                trial=x.copy()
                if suffix=='_B_wrist':
                    trial[19:21]=seed[19:21]
                if suffix=='_wrist':
                    _,jp=mesh(layer,x,pack)
                    for i,side in enumerate(SIDES):
                        e,w=18+i,20+i
                        pivot=jp[w]
                        base=world[e]@mat(x[w-1])
                        local=(jp[ids[i,[5,17]]]-pivot)@base
                        obs=np.array(rec['hands'][side]['observation'])[:,[5,17]]
                        def loss(r):
                            pos=local@(world[e]@mat(r)).T+pivot
                            return np.r_[(project(pos)-obs).ravel()/6,rv(mat(x[w-1]).T@mat(r))/.65]
                        sol=least_squares(loss,x[w-1],loss='soft_l1',max_nfev=180)
                        trial[w-1]=rv(mat(sol.x))
                _,j=mesh(layer,trial,pack)
                drift=float(1000*np.linalg.norm(j[:22]-j0[:22],axis=1).max())
                assert drift<.001 and np.array_equal(trial[23:],x[23:])
                dest=OUT/'poses'/(key+'_'+mode+'.npz')
                np.savez_compressed(dest,**dict(pack,locked=trial))
                hands={}
                for i,side in enumerate(SIDES):
                    mapped=j[ids[i]]
                    src=source_joints[side]
                    src_local=src@palm_basis(src)
                    map_local=(mapped-mapped[:1])@palm_basis(mapped)
                    dirs=angle(bones(src_local),bones(map_local))
                    proj=project(mapped)
                    old=rec['hands'][side]
                    obs=np.asarray(old['observation'])
                    inside=np.asarray(old['inside'])
                    error=np.linalg.norm(proj-obs,axis=-1)
                    refs=paired.palm.triangulate(obs[:,[0,5,17]],cams,transforms[1])
                    pn=lambda a:np.cross(a[1]-a[0],a[2]-a[0])
                    check=dict(tags[side],local_bone_mean_deg=float(dirs.mean()),local_bone_max_deg=float(dirs.max()),
                               source_opening=opening(src),mapped_opening=opening(mapped),
                               source_mapped_normal_deg=float(angle(pn(src[[0,5,17]]),pn(mapped[[0,5,17]]))),
                               stereo_normal_deg=float(angle(pn(refs),pn(mapped[[0,5,17]]))),
                               wrist_local_deg=float(np.rad2deg(np.linalg.norm(trial[19+i]))),
                               thumb_side_matches=bool(src_local[1,0]*map_local[1,0]>0))
                    assert check['thumb_side_matches']
                    hands[side]=dict(uv=proj.tolist(),observation=obs.tolist(),inside=inside.tolist(),transfer=check,
                        tip_px=rms(error[:,TIPS][inside[:,TIPS]]),wrist_px=rms(error[:,0]),root_px=rms(error[:,[5,17]]),
                        thumb_px=rms(error[:,4][inside[:,4]]),tip_values=error[:,TIPS][inside[:,TIPS]].tolist())
                results[mode]=dict(pose_sha256=sha(dest),hands=hands,body_drift_mm=drift)
        results['B']=rec
        write(RUN/'evidence'/(key+'.json'),dict(results=results,input_pose_sha256=sha(file),flat_hand_mean=bool(layer.flat_hand_mean),exit_status=0))
        summaries[key]=results
    summary={}
    for mode in summaries['clip003']:
        pool=[]
        checks=[]
        for results in summaries.values():
            for hand in results[mode]['hands'].values():
                mask=np.array(hand['inside'])[:,TIPS]
                gap=np.linalg.norm(np.array(hand['uv'])-hand['observation'],axis=-1)
                pool.extend(gap[:,TIPS][mask])
                if 'transfer' in hand: checks.append(hand['transfer'])
        assert len(pool)==33
        summary[mode]=dict(tips=rms(pool),count=len(pool))
        if checks:
            summary[mode].update(mean_bone_deg=float(np.mean([k['local_bone_mean_deg'] for k in checks])),
                mean_normal_deg=float(np.mean([k['stereo_normal_deg'] for k in checks])))
    write(RUN/'evidence/summary.json',summary)
    print(summary)
if __name__=='__main__': main()
