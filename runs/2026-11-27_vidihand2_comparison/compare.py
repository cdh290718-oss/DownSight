"""Compare native projection and the two methods on a shared, fixed B wrist."""
from pathlib import Path
import sys
import cv2
import numpy as np
import torch
import open3d as o3d

RUN=Path(__file__).resolve().parent
sys.path.insert(0,str(RUN.parent/'2026-11-27_stereo_hand_selection'))
from choose import CASES,SIDES,DIGITS,TIPS,JOINT_NAMES,old,read,write,sha
from compare_fit import FINGER_JOINTS
from render_selection import stereo,full_body,rot,title,groups,free_render
OUT=RUN.parents[1]/'outputs/vidihand2_comparison_20261127'
MODES=('wilor','vidi')
COLORS={'wilor':(40,167,255),'vidi':(90,220,80),'obs':(255,230,30)}

def errors(uv,target,mask):
    dist=np.linalg.norm(uv-target,axis=-1)
    return dist[...,TIPS][mask[...,TIPS]].tolist()

def draw(im,points,color):
    xy=np.rint(points).astype(int)
    for k in (1,5,9,13,17):
        for a,b in zip([0,k,k+1,k+2],range(k,k+4)):
            cv2.line(im,tuple(xy[a]),tuple(xy[b]),color,1,cv2.LINE_AA)
    for i in [0]+TIPS:
        cv2.circle(im,tuple(xy[i]),3,color,-1,cv2.LINE_AA)

def local_mesh(models,joints,ids,faces,parts):
    panels=[]
    for i,side in enumerate(SIDES):
        hand=faces[parts==side[0].upper()+'_hand']
        across,up,normal=old.palm_basis(joints[ids[i]]).T
        row=[]
        for angle in np.deg2rad([12,78]):
            axes=np.column_stack([across*np.cos(angle)+normal*np.sin(angle),up,across*np.sin(angle)-normal*np.cos(angle)])
            points=np.vstack([v[np.unique(hand)]@axes for v in models.values()])
            center=(points.min(0)+points.max(0))*.5
            scale=226/np.ptp(points[:,:2],axis=0).max()
            winding=hand[:,::-1] if np.linalg.det(axes)<0 else hand
            for name,v in models.items():
                im=free_render(v,winding,axes,center,scale)[112:368,47:303].copy()
                im[np.all(im==im[0,0],axis=2)]=(36,36,33)
                row.append(title(im,f'{side} | {name} | {np.rad2deg(angle):.0f} deg'))
        panels.append(np.hstack(row))
    return np.vstack(panels)

def main():
    torch.set_num_threads(1)
    model,Camera=old.ref.prev.ref.common.setup_model()
    layer=model.layer['neutral'].eval()
    means=layer.pose_mean.detach().numpy().reshape(-1,3)[25:]
    faces=np.asarray(model.face,dtype=np.int32)
    part_labels=groups(layer,faces)
    yy,xx=np.indices((720,1280))
    xy=np.stack((xx,yy),axis=-1).reshape(-1,2)
    eye_rays=[]
    for calibration in old.ref.cameras():
        rays=np.zeros((720,1280,6),np.float32)
        rays[:,:,3:]=Camera(calibration).camera2world_ray(xy).reshape(720,1280,3)
        eye_rays.append(o3d.core.Tensor(rays))
    labels=['wrist']+[d+s for d in DIGITS for s in ('1','2','3','')]
    ids=np.asarray([[JOINT_NAMES.index(side+'_'+n) for n in labels] for side in SIDES])
    rows={r['pair']:r for r in read(old.ref.OLD/'fit/0/prepared.json')['items']}
    cams=list(map(read,old.ref.cameras()))
    assert [sha(p) for p in old.ref.cameras()]==read(old.ref.PRIOR/'evidence/calibration.json')['converted']
    inputs={m:read(RUN/'evidence'/(m+'.json'))['views'] for m in MODES}
    temporal=read(RUN/'evidence/temporal.json')['views']
    coverage={}
    for name,view in temporal.items():
        counts=np.asarray(view['boxes_by_frame'])
        found=counts>0
        valid=np.asarray(inputs['vidi'][name]['valid_by_frame'],dtype=bool)
        assert valid.shape==found.shape==(81,2)
        coverage[name]=dict(side_frames=int(valid.size),wilor=int(found.sum()),vidi=int(valid.sum()),
                            only_vidi=int((valid&~found).sum()),only_wilor=int((found&~valid).sum()),
                            both=int((found&valid).sum()),duplicate_side_frames=int((counts>1).sum()))
    frames=read(old.ref.PRIOR/'evidence/frames.json')
    refs=read(RUN.parent/'2026-11-27_stereo_hand_selection/evidence/metrics.json')
    stereo.RUN,stereo.OUT=RUN,OUT
    images,records,pools={}, {}, {}
    for key in CASES:
        b=read(old.BASE/'evidence'/(key+'.json'))['results']['B']
        bp=old.BASE_OUT/'poses'/(key+'_B.npz')
        assert sha(bp)==b['pose_sha256']
        p=dict(np.load(bp))
        seed=p['locked']
        bmesh,j0=old.mesh(layer,seed,p)
        neutral=np.zeros_like(seed)
        neutral[23:]=-means
        _,rest=old.mesh(layer,neutral,dict(p,global0=np.zeros(3)))
        transforms=[np.array(rows[key]['inputs']['views'][eye]['camera']['left_to_view']) for eye in SIDES]
        def project(x):
            return np.stack([old.projection(x@t[:3,:3].T+t[:3,3],cam) for t,cam in zip(transforms,cams)])
        meshes={}
        rec={}
        records[key]=rec
        for mode in MODES:
            x=seed.copy()
            report={}
            rec[mode]=report
            for i,side in enumerate(SIDES):
                h=b['hands'][side]
                obs=np.array(h['observation'])
                used=np.array(h['used'])
                inside=np.array(h['inside'])
                assert h['point_indices']==FINGER_JOINTS
                candidates,poses={},{}
                sl=slice(23+15*i,38+15*i)
                mirror=np.diag([-1,1,1]) if i==0 else np.eye(3)
                for n,eye in enumerate(SIDES):
                    src=key+'_'+eye
                    source='cam'+str(n+3)
                    if side not in inputs[mode][src]['sha256']: continue
                    path=OUT/'poses'/(mode+'_'+src+'_'+side+'.npz')
                    assert sha(path)==inputs[mode][src]['sha256'][side]
                    vals=np.load(path)
                    if mode=='vidi':
                        native=old.projection(vals['joint'],cams[n])
                        mano_native=old.projection(vals['mano_joint'],cams[n])
                    else:
                        homogeneous=np.column_stack([vals['uv'],np.ones(21)])
                        native=old.projection(homogeneous@np.linalg.inv(vals['K']).T,cams[n])
                        mano_native=native
                    q=old.palm_basis(rest[ids[i]])@old.palm_basis(vals['rest']).T
                    local=q@mirror@vals['pose']@mirror@q.T
                    poses[source]=old.rv(local)-means[15*i:15*(i+1)]
                    assert np.max(abs(old.mat(poses[source]+means[15*i:15*(i+1)])-local))<1e-4
                    trial=seed.copy()
                    trial[sl]=poses[source]
                    _,j=old.mesh(layer,trial,p)
                    uv=project(j[ids[i]])
                    dist=np.linalg.norm(uv[:,FINGER_JOINTS]-obs[:,FINGER_JOINTS],axis=-1)
                    robust=2*(np.sqrt(1+(dist/6)**2)-1)
                    score=float(np.mean([a[m].mean() for a,m in zip(robust,used)]))
                    candidates[source]=dict(score=score,native_uv=native.tolist(),native_tips=errors(native,obs[n],inside[n]),
                                             mano_tips=errors(mano_native,obs[n],inside[n]),
                                             native_wrist_px=float(np.linalg.norm(native[0]-obs[n,0])),uv=uv.tolist())
                hand=dict(candidates=candidates)
                if candidates:
                    winner=min(candidates,key=lambda s:candidates[s]['score'])
                    x[sl]=poses[winner]
                    hand['winner']=winner
                report[side]=hand
            v,j=old.mesh(layer,x,p)
            drift=float(np.linalg.norm(j[:22]-j0[:22],axis=1).max()*1000)
            assert drift<.001 and np.array_equal(x[:23],seed[:23])
            report['drift_mm']=drift
            for i,side in enumerate(SIDES):
                hand=report[side]
                if 'winner' not in hand: continue
                uv=project(j[ids[i]])
                assert np.max(abs(uv-hand['candidates'][hand['winner']]['uv']))<.05
                h=b['hands'][side]
                hand.update(uv=uv.tolist(),fixed_tips=errors(uv,np.array(h['observation']),np.array(h['inside'])))
            if all('winner' in report[s] for s in SIDES):
                v,obj=stereo.export(key,mode,v,faces)
                meshes[mode]=v
                stereo.save(images,key+'_'+mode+'_body',full_body(v,faces,rot(x[21])@rot(p['global0'])))
        if meshes:
            stereo.save(images,key+'_local_OBJ',local_mesh(dict(B=bmesh,**meshes),j0,ids,faces,part_labels))
        closeups=[]
        for n,eye in enumerate(SIDES):
            path=old.ref.OLD/'images'/(key+'_'+eye+'.png')
            assert sha(path)==frames[int(key[-1])-1]['views'][eye]['frame']
            photo=cv2.imread(str(path))
            source='cam'+str(3+n)
            full=[]
            for mode in MODES:
                im=photo.copy()
                for side in SIDES:
                    native=rec[mode][side]['candidates'].get(source)
                    if native: draw(im,np.asarray(native['native_uv']),COLORS[mode])
                full.append(title(im,mode+' native | '+source))
                if mode in meshes:
                    t=transforms[n]
                    parts,visible,shade=stereo.raycast(meshes[mode]@t[:3,:3].T+t[:3,3],faces,part_labels,eye_rays[n])
                    overlay=photo.copy()
                    overlay[visible]=(.5*overlay[visible]+.5*shade[visible,None]).astype('uint8')
                    stereo.edge(overlay,parts>0,COLORS[mode])
                    full.append(title(overlay,mode+' B wrist | '+source))
            stereo.save(images,key+'_'+source,np.vstack(full))
            for side in SIDES:
                h=b['hands'][side]
                obs=np.asarray(h['observation'])[n]
                inside=np.asarray(h['inside'])[n]
                tiles=[]
                observed=photo.copy()
                for pt in obs[inside]:
                    cv2.drawMarker(observed,tuple(np.rint(pt).astype(int)),COLORS['obs'],cv2.MARKER_TILTED_CROSS,7,1)
                tiles.append((observed,'Auto points'))
                for subset in ('native','fixed'):
                    for mode in MODES:
                        d=rec[mode][side]
                        cand=d['candidates'].get(source)
                        uv=(cand['native_uv'] if cand else None) if subset=='native' else (d['uv'][n] if 'uv' in d else None)
                        im=photo.copy()
                        if uv is not None: draw(im,np.array(uv),COLORS[mode])
                        tiles.append((im,mode+' '+subset+(' (miss)' if uv is None else '')))
                # Select a common crop around the target hand using the same wrist-distance filter as B.
                radius=np.median(np.linalg.norm(obs[[5,9,13,17]]-obs[0],axis=1))
                pts=obs[inside&(np.linalg.norm(obs-obs[0],axis=1)<3.2*radius)]
                size=int(np.clip(np.ptp(pts,axis=0).max()+72,170,330))
                left,top=np.clip(((pts.min(0)+pts.max(0)-size)*.5).astype(int),0,[1280-size,720-size])
                closeups.append(np.hstack([title(cv2.resize(im[top:top+size,left:left+size],(235,235)),f'{3+n} {side[0].upper()} | '+name) for im,name in tiles]))
                common=all(source in rec[m][side]['candidates'] for m in MODES)
                for m in MODES:
                    candidate=rec[m][side]['candidates'].get(source)
                    if candidate:
                        for field in ('native_tips','mano_tips'):
                            pools.setdefault(m+'_'+field+'_all',[]).extend(candidate[field])
                            if common: pools.setdefault(m+'_'+field+'_common',[]).extend(candidate[field])
        stereo.save(images,key+'_hands',np.vstack(closeups))
        for side in SIDES:
            common=all('winner' in rec[m][side] for m in MODES)
            for m in MODES:
                if 'winner' in rec[m][side]:
                    e=rec[m][side]['fixed_tips']
                    pools.setdefault(m+'_fixed_all',[]).extend(e)
                    if common: pools.setdefault(m+'_fixed_common',[]).extend(e)
            if common:
                h=refs[key]['hands'][side]
                pools.setdefault('archived_guided_fixed_common',[]).extend(errors(np.array(h['uv']),np.array(h['observation']),np.array(h['inside'])))
    summary={k:dict(n=len(v),rmse_px=old.rms(v)) for k,v in pools.items() if v}
    write(RUN/'evidence/metrics.json',dict(cases=records,pooled=summary,coverage=coverage))
    write(RUN/'evidence/results.json',dict(images=images,exit_status=0))
    print(summary)
if __name__=='__main__': main()
