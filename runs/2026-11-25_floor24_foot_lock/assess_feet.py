"""Independent mesh reconstruction and RW toe-base scoring on the sealed gallery."""
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from lock import RUN, OUT, ROOT, PREVIOUS, FEET, prev

sys.path.insert(0,str(RUN.parent/'2026-11-20_rw20000_floor12'))
from render_gallery import g

MODES=('current','reset','locked')


def foot_views(meshes,faces,foot_vertices,entry):
    up=np.array(entry['ground']['up'])
    forward=np.array(entry['front_forward_left_cv'])
    forward-=up*(forward@up)
    forward/=np.linalg.norm(forward)
    across=np.cross(up,forward)
    columns=[[] for _ in meshes]
    for basis in (np.column_stack((across,up,forward)),np.column_stack((forward,up,-across))):
        coords=np.concatenate([v[foot_vertices]@basis for v in meshes])
        lo,hi=coords.min(0),coords.max(0)
        center=(lo+hi)/2
        scale=min(305/max(hi[0]-lo[0],.35),235/max(hi[1]-lo[1],.26))
        for col,v in zip(columns,meshes):
            im=g.free_render(v,faces,basis,center,scale)
            col.append(im[95:385])
    return np.hstack([np.vstack(col) for col in columns])


def main():
    torch.set_num_threads(1)
    results=prev.read(PREVIOUS/'evidence/results.json')
    gallery={r['pair']:r for r in prev.read(ROOT/'runs/2026-11-23_fixed_floor24/evidence/gallery.json')['rows']}
    assert len(results['rows'])==24
    for row in results['rows']:
        path=OUT/'poses'/(row['pair']+'.npz')
        assert prev.sha(path)==prev.read(path.with_suffix('.json'))['sha256']
    model,_=g.common.setup_model()
    layer=model.layer['neutral'].cpu().eval()
    faces=np.asarray(model.face,np.int32)
    foot_vertices=layer.lbs_weights.numpy()[:,[7,8,10,11]].sum(1)>.65
    rows={}
    for row in results['rows']:
        key=row['pair']
        test=row['split']=='test'
        folder='frozen100_fingers' if test else 'validation200_fingers'
        path=ROOT/'outputs'/(folder+'_20261125')/'poses'/(key+'.npz')
        assert prev.sha(path)==row['pose_sha256']
        source=prev.read(RUN.parent/('2026-11-25_'+folder)/'evidence/results.json')
        truth_path=g.DATA/'reference.json' if test else prev.OUT/'validation_truth.json'
        assert prev.sha(truth_path)==source['reference_sha256']
        gt=np.array(prev.read(truth_path)[key]['gt'])
        cfg=prev.read((prev.base.OUT if test else prev.VAL)/'protocol.json')
        ratios=np.array(cfg['landmark_ratios'][1:])
        with np.load(path) as z:cur,global0=z['fixed_full'],z['global0']
        newpath=OUT/'poses'/(key+'.npz')
        with np.load(newpath) as z:
            assert np.array_equal(global0,z['global0'])
            locked=z['locked']
        reset=cur.copy()
        reset[FEET]=0.
        assert not locked[FEET].any()
        meshes=[]
        metrics={}
        for mode,x in zip(MODES,(cur,reset,locked)):
            assert not x[23:].any()
            body=dict(global_orient=global0,body_pose=x[:21],transl=x[22],betas=np.zeros(10),
                      left_hand_pose=x[23:38],right_hand_pose=x[38:])
            v,j,_,_=g.geometry(layer,body,x[21],False,None)
            meshes.append(v)
            pred=j[g.common.NATIVE_IDS][g.common.OBS]
            target=gt[g.common.OBS]
            ankles=j[[8,7]]
            ray=j[[11,10]]-ankles
            target_ray=gt[[10,14]]-gt[[9,13]]
            cosine=(ray*target_ray).sum(1)/(np.linalg.norm(ray,axis=1)*np.linalg.norm(target_ray,axis=1))
            metrics[mode]=dict(pa_mm=float(1000*np.linalg.norm(g.align(pred,target)-target,axis=1).mean()),
                        nonpa_mm=float(1000*np.linalg.norm(pred-target,axis=1).mean()),
                        toe_mm=float(1000*np.linalg.norm(ankles+ratios[:,None]*ray-gt[[10,14]],axis=1).mean()),
                        toe_direction_deg=float(np.degrees(np.arccos(cosine.clip(-1,1))).mean()),
                        local_rotation_deg=np.rad2deg(np.linalg.norm(x[FEET],axis=1)).tolist())
        assert max(abs(metrics['current'][m]-row[m]) for m in ('pa_mm','nonpa_mm'))<.001
        assert max(abs(metrics['reset'][m]-metrics['current'][m]) for m in ('pa_mm','nonpa_mm'))<.001
        path_img=RUN/'assets'/(key+'.jpg')
        image=foot_views(meshes,faces,foot_vertices,gallery[key])
        g.save(path_img,image)
        rows[key]=dict(split=row['split'],metrics=metrics,prior_sha256=row['pose_sha256'],
                       locked_sha256=prev.sha(newpath),views_sha256=prev.sha(path_img))
    summary={m:{k:float(np.mean([r['metrics'][m][k] for r in rows.values()])) for k in
                ('pa_mm','nonpa_mm','toe_mm','toe_direction_deg')} for m in MODES}
    improved={m:sum(r['metrics'][m]['toe_direction_deg']<r['metrics']['current']['toe_direction_deg'] for r in rows.values())
              for m in ('reset','locked')}
    angles=np.array([r['metrics']['current']['local_rotation_deg'] for r in rows.values()])
    result=dict(summary=summary,direction_improved_frames=improved,current_max_ankle_deg=angles[:,:2].max(),
                current_max_forefoot_deg=angles[:,2:].max(),base_commit='13bbff9a4fcae8ea394de454e3853ab76410c814',
                dirty_worktree=True,prior_gallery_sha256=prev.sha(PREVIOUS/'evidence/results.json'),
                commands=['bash <RUN>/workers.sh 8 lock','bash <RUN>/run.sh assess_feet'],exit_status=0,rows=rows)
    prev.write(RUN/'evidence/results.json',result)
    print(summary,improved)

if __name__=='__main__':main()
