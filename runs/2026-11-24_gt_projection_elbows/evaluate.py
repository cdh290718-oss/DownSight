"""Score the locked poses and render two mesh comparisons in the gallery coordinates."""
from pathlib import Path
import sys
import cv2
from scipy.optimize import least_squares
from pixel_intervention import CASES, MODES, FIT, OUT, ROOT, RUN, key, np, prev, torch
sys.path.insert(0,str(ROOT/'runs/2026-11-20_rw20000_floor12'))
import render_gallery as render
sys.path.insert(0,str(ROOT/'runs/2026-11-23_floor24_intersections'))
import inspect_obj as surface


def evaluate():
    torch.set_num_threads(1)
    items={v['pair']:v for v in prev.read(FIT/'prepared.json')['items']}
    cfg=prev.read(RUN/'evidence/validation_limits.json')
    assert prev.sha(OUT/'validation_selected.npz')==cfg['selected_sha256']
    bank=dict(np.load(OUT/'validation_bank.npz'))
    limit=dict(np.load(OUT/'validation_selected.npz'))
    ch=prev.read(OUT/'validation_choices.json')
    keys=list(prev.read(OUT/'validation_decisions.json'))
    model,cam=prev.ref.common.setup_model()
    layer=model.layer['neutral']
    cameras=[cam(p) for p in prev.ref.camera_files()]
    adapters=[prev.read(p)['input_adapter'] for p in prev.ref.camera_files()]
    truth=prev.read(prev.OUT/'validation_truth.json')
    ims=prev.pair_images(prev.VM)
    obs=prev.ref.common.OBS
    names=[prev.ref.common.NAMES[i] for i in obs]
    native=np.asarray(prev.ref.common.NATIVE_IDS)[obs]
    faces=np.asarray(model.face)
    labels=surface.groups(layer,faces)
    rows,views,tri={}, {}, {}
    previous=prev.read(ROOT/'runs/2026-11-24_elbow_limits/evidence/validation_scores.json')['rows']
    for frame in CASES:
        gt=np.asarray(truth[frame]['gt'])[obs]
        im=ims[int(frame[-3:])-1]
        outputs={}
        tri[frame]={}
        rows[frame]={}
        for mode in MODES:
            k=key(frame,mode)
            item=items[k]
            assert truth[frame]['subject']==im[0]['subject']
            # Separate unregularized two-view triangulation of observed points, with no body model.
            if mode in ['detector','projected_all']:
                T=np.array(item['inputs']['views']['right']['camera']['left_to_view'])
                estimates={}; positions={}
                for j,p in enumerate(item['points']):
                    if not all(p['valid']):continue
                    uv=np.asarray(p['xy'])
                    rays=[c.camera2world_ray(a[None])[0] for c,a in zip(cameras,uv)]
                    rays=[v/np.linalg.norm(v) for v in rays]
                    origin=-T[:3,:3].T@T[:3,3]
                    direction=T[:3,:3].T@rays[1]
                    depths=np.linalg.lstsq(np.column_stack([rays[0],-direction]),origin,rcond=None)[0]
                    start=(depths[0]*rays[0]+origin+depths[1]*direction)/2
                    def residual(x):
                        return np.concatenate([(c.world2camera(y[None])[0]-a)/d['scale']
                                for c,y,a,d in zip(cameras,[x,T[:3,:3]@x+T[:3,3]],uv,adapters)])
                    sol=least_squares(residual,start)
                    assert sol.success
                    positions[p['name']]=sol.x
                    estimates[p['name']]=dict(error_mm=float(1000*np.linalg.norm(sol.x-gt[j])),
                        reprojection_rmse_px=float(np.linalg.norm(sol.fun)/np.sqrt(2)))
                directions={}
                for side in ['L','R']:
                    e,w=[side+'_'+p for p in ['Elbow','Wrist']]
                    if e in positions and w in positions:
                        directions[side]=surface.angle(positions[w]-positions[e],gt[names.index(w)]-gt[names.index(e)])
                tri[frame][mode]=dict(joints=estimates,forearm_direction_error_deg=directions)
            r=ch[k]; row=rows[frame][mode]={}
            for tag,pack,i in [('v3',bank,r['offset']+r['best']),('limited',limit,keys.index(k))]:
                x=prev.ref.full_joints(pack,layer,i)[native]
                err=1000*np.linalg.norm(x-gt,axis=1)
                pix=prev.ref.pixel_errors(item,x,im,prev.mapping(),cameras,adapters)
                def px(field):
                    return float(np.sqrt(sum(v['sum_sq_px2'][field] for v in pix)/sum(v['n'] for v in pix)))
                angles={}
                for side in ['L','R']:
                    e,w=[names.index(side+'_'+j) for j in ['Elbow','Wrist']]
                    angles[side]=surface.angle(x[w]-x[e],gt[w]-gt[e])
                result=dict(non_pa_mm=float(err.mean()),pa_mm=float(1000*np.linalg.norm(prev.align(x,gt)-gt,axis=1).mean()),
                            per_joint_mm=dict(zip(names,err.tolist())),forearm_direction_error_deg=angles,
                            official_2d_rmse_px=px('fit_vs_gt'),input_2d_rmse_px=px('fit_vs_detection'))
                if mode=='detector':
                    original=previous[frame][tag]
                    result['abs_difference_from_archived_non_pa_mm']=float(abs(original['mpjpe']-result['non_pa_mm']))
                row[tag]=result
                if tag=='limited' and mode in ['detector','projected_all']:
                    params={n:pack[f][i] for n,f in [('body_pose','body_pose'),('global_orient','global0'),('transl','translation'),('betas','betas')]}
                    verts=render.g.geometry(layer,params,pack['delta'][i],False,None)[0]
                    outputs[mode]=verts
                    result['upperarm_forearm_crossing_face_pairs']={}
                    for side in ['L','R']:
                        mask=np.isin(labels,[side+'_upperarm',side+'_forearm'])
                        parts=labels[mask]
                        hits=surface.self_intersections(verts.astype(float),faces[mask])
                        result['upperarm_forearm_crossing_face_pairs'][side]=sum(int(parts[a]!=parts[b]) for a,b in hits)
        source=prev.read(ROOT/'outputs/fixed_floor24_20261123'/(frame+'.json'))
        up=np.array(source['ground']['up']); forward=np.array(source['front_forward_left_cv'])
        z=render.floor.unit(forward-up*(up@forward)); x=render.floor.unit(np.cross(up,z))
        bases=[np.column_stack([x,up,z]),np.column_stack([z,up,-x])]
        ext=[np.concatenate(list(outputs.values()))@b for b in bases]
        scale=min(min(313/np.ptp(v[:,0]),438/np.ptp(v[:,1])) for v in ext)
        panels=[]
        for mode,title in [('detector','Detector'),('projected_all','GT projection')]:
            for b,v,side in zip(bases,ext,['front','side']):
                img=render.g.free_render(outputs[mode],faces,b,(v.min(0)+v.max(0))/2,scale)
                img=cv2.copyMakeBorder(img,38,2,0,0,cv2.BORDER_CONSTANT,value=(245,245,245))
                cv2.putText(img,title+' / '+side,(12,24),cv2.FONT_HERSHEY_SIMPLEX,.66,(40,40,40),1,cv2.LINE_AA)
                panels.append(img)
        path=RUN/'assets'/frame/'comparison.jpg'
        render.g.save(path,np.concatenate(panels,axis=1))
        views[frame]=prev.sha(path)
    prev.write(RUN/'evidence/scores.json',dict(rows=rows,triangulation=tri,exit_status=0))
    prev.write(RUN/'evidence/views.json',views)


if __name__=='__main__':
    evaluate()
