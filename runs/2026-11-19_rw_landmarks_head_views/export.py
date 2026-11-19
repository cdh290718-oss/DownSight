"""Evaluate only completed fits. All meshes and individual 3D coordinates stay in the local output."""
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

from study import RUN, OUT, ROOT, OLD, read, write, sha, common
from evaluate import align

OLD_GROUP=['old_history','old_weak','old_off']
CONTROL=['baseline','15history','12camera','15camera','12torso','15torso']
CAM_GROUP=['camera_lock','torso_caminit']
AUTO=['auto12','auto15']
AUTO_HEAD=['auto12_camera','auto12_torso','auto15_camera','auto15_torso']
LABEL={'auto12_camera':'RW20000 | 12 | camera lock','auto15_camera':'RW20000 | 15 | camera lock',
       'auto12_torso':'RW20000 | 12 | torso lock','auto15_torso':'RW20000 | 15 | torso lock','auto12':'RW20000 | 12 points','auto15':'RW20000 | 15 points','baseline':'12 | historical','15history':'15 | historical',
       '12camera':'12 | camera prior','15camera':'15 | camera prior',
       '12torso':'12 | torso lock','15torso':'15 | torso lock',
       'camera_lock':'15 | camera lock','torso_caminit':'15 | torso lock (cam init)',
       'old_history':'Old 15 | historical','old_weak':'Old 15 | weak (0.1x)','old_off':'Old 15 | off'}
assert set(LABEL)==set(OLD_GROUP+CONTROL+CAM_GROUP+AUTO+AUTO_HEAD)


def rot(aa):
    return cv2.Rodrigues(np.asarray(aa,dtype=float).reshape(3))[0]


def angle(r1,r2):
    u,_,vt=np.linalg.svd(r1.T@r2)
    r=u@vt
    return float(np.rad2deg(np.arctan2(np.linalg.norm([r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1]])/2,(np.trace(r)-1)/2)))


def geometry(layer, p, correction_delta, cam_lock, camera_rot):
    body=np.array(p['body_pose']).reshape(21,3)
    g=np.asarray(p['global_orient']).reshape(3)
    to_t=lambda a:torch.tensor(a,dtype=torch.float32).reshape(1,-1)
    fields={k:to_t(p.get(k,np.zeros(dim))) for k,dim in [('left_hand_pose',45),('right_hand_pose',45),
           ('expression',10),('jaw_pose',3),('leye_pose',3),('reye_pose',3)]}
    with torch.no_grad():
        full=layer(global_orient=to_t(g),body_pose=to_t(body),betas=to_t(p['betas']),**fields)
    j=full.joints[0].numpy()
    v=full.vertices[0].numpy()
    d=rot(correction_delta)
    pivot=j[0].copy()
    tr=np.asarray(p['transl']).reshape(3)
    v=(v-pivot)@d.T+pivot+tr
    j=(j-pivot)@d.T+pivot+tr
    chest=d@rot(g)
    for k in [3,6,9]:
        chest=chest@rot(body[k-1])
    head=chest@rot(body[11])@rot(body[14])
    if cam_lock:
        corr=camera_rot@head.T
        v=(v-j[15])@corr.T+j[15]
        j=(j-j[15])@corr.T+j[15]
        head,chest=corr@head,corr@chest
    return v,j,head,chest


def modes(family):
    if family=='old':
        for i in range(1,5):
            pair='body%02d'%i
            for label in OLD_GROUP:
                path=(OLD if label=='old_history' else OUT/'old'/label.split('_')[1])/pair/'D_left'
                opt=read(path/'optimization.json')
                p=dict(np.load(path/'final.npz'))
                assert sha(path/'final.npz')==opt['final']['sha256']
                yield pair,label,p,np.zeros(3),opt['best_step'],opt['convergence']['diagnostic_pass'],sha(path/'final.npz')
        return
    d={'camera':OUT/'camera_only','automatic':OUT/'automatic','current':OUT,'automatic_head':OUT/'automatic_head'}[family]
    complete=read(d/'fit_complete.json')
    diag=read(d/'diagnostic_20000.json')
    assert complete['exit_status']==0 and all(diag['selected_feasible'])
    fn=d/'budget_20000.npz'
    with np.load(fn) as z:
        for k,(pair,key,_) in enumerate(complete['ids']):
            label={'baseline':'camera_lock','torso':'torso_caminit'}[key] if family=='camera' else key
            if family=='automatic_head': label='auto12_camera' if key=='baseline' else key
            if family=='automatic': label={'baseline':'auto12','15history':'auto15'}[key]
            p={key:z[name][k] for key,name in [('global_orient','global0'),('body_pose','body_pose'),('betas','betas'),('transl','translation')]}
            yield pair,label,p,z['delta'][k],int(z['selected_step'][k]),diag['convergence_pass'][k],sha(fn)


def main(family):
    model,camera_cls=common.setup_model()
    layer=model.layer['neutral'].cpu().eval()
    layer.requires_grad_(False)
    gt=np.load(common.ARCH/'body_gt_official.npz')['joints_m']
    cams=[camera_cls(f) for f in common.camera_files()]
    cal=read(RUN/'evidence/calibration.json')
    camR=np.array(cal['camera_R_head_proxy'])
    ks=cal['ratios']
    faces=np.asarray(model.face,dtype=np.int32)
    rows=[]
    for pair,label,p,delta,step,converged,source_hash in modes(family):
        v,j,h,c=geometry(layer,p,delta,label in ['camera_lock','auto12_camera','auto15_camera'],camR)
        assert np.isfinite(v).all() and np.isfinite(j).all()
        if 'smplx_mesh_cam' in p:
            assert np.max(np.abs(v-p['smplx_mesh_cam']))<3e-6
        if 'torso' in label: assert angle(h,c)<.002
        if label in ['camera_lock','auto12_camera','auto15_camera']: assert angle(h,camR)<.002
        native=j[common.NATIVE_IDS]
        mapped=native.copy()
        for idx,a,b,ratio in zip([0,10,14],[15,8,7],[12,11,10],ks):
            mapped[idx]=j[a]+ratio*(j[b]-j[a])
        base=OLD/pair
        anchor=read(base/'anchors/oracle.json')
        obs=read(base/'observations_local.json')
        transforms=read(base/'inputs.json')['views']
        g=gt[2*(int(pair[-2:])-1)].astype(float)
        original=np.linalg.norm(native-g,axis=1)*1000
        semantic=np.linalg.norm(mapped-g,axis=1)*1000
        px=[]
        for i,side in enumerate(['left','right']):
            T=np.array(transforms[side]['camera']['left_to_view'])
            uv=cams[i].world2camera(mapped@T[:3,:3].T+T[:3,3])
            px.append(np.linalg.norm(uv-np.array(obs['views'][i])[:,:2],axis=1))
        dest=OUT/'objects'/pair
        dest.mkdir(parents=True,exist_ok=True)
        path=dest/(pair+'__'+label+'__left_cv_m.obj')
        with path.open('w') as fh:
            fh.write('# fitted SMPL-X surface; left camera CV; meters\n')
            np.savetxt(fh,v,fmt='v %.9f %.9f %.9f')
            np.savetxt(fh,faces+1,fmt='f %d %d %d')
        lines=path.read_text().splitlines()
        rv=np.array([s.split()[1:] for s in lines if s.startswith('v ')],float)
        rf=np.array([s.split()[1:] for s in lines if s.startswith('f ')],int)-1
        assert np.max(np.abs(rv-v))<1e-8 and np.array_equal(rf,faces)
        assert rf.min()==0 and rf.max()==len(rv)-1
        rows.append(dict(pair=pair,condition=label,source_sha256=source_hash,
          obj=path.name,obj_sha256=sha(path),vertices=len(rv),faces=len(rf),selected_step=step,converged=bool(converged),
          native15_nonpa_mm=original.mean(),native15_pa_mm=float(np.linalg.norm(align(native,g)-g,axis=1).mean()*1000),
          mapped15_nonpa_mm=semantic.mean(),mapped15_pa_mm=float(np.linalg.norm(align(mapped,g)-g,axis=1).mean()*1000),
          common12_mm=original[common.OBS].mean(),native_errors_mm=original.tolist(),mapped_errors_mm=semantic.tolist(),
          matched12_reproj_px=np.array(px)[:,common.OBS].mean(),mapped15_reproj_px=np.mean(px),
          head_position_mm=float(np.linalg.norm(j[15]-anchor['position'])*1000),head_chest_deg=angle(h,c),
          head_history_deg=angle(h,np.asarray(anchor['rotation'])),head_camera_proxy_deg=angle(h,camR)))
    write(RUN/'evidence'/('results_'+family+'.json'),dict(family=family,joints=common.NAMES,rows=rows))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('family',choices=['current','camera','old','automatic','automatic_head'])
    main(parser.parse_args().family)
