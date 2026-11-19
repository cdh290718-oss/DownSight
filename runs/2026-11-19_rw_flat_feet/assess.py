"""Verify the final SMPL-X sole geometry, stereo measurements and OBJ package."""
import hashlib
import json
import sys
import zipfile
import numpy as np
from experiment import (RUN,OUT,ROOT,common,read,write,sha,KEYS as LABELS,
                        FlatProblem,sole_frame,world_ankles,flat_targets)
from plane import angle
import cv2
import torch
import export
import render
from evaluate import align

E=RUN/'evidence'
TAG='rw-s9-flat-feet-20261119'
ARCHIVE='rw_s9_flat_feet_OBJ.zip'
GROUPS={'auto':['free12','flat12','free15','flat15']}
LEGEND={'free12':'RW20000 | 12 | free feet','flat12':'RW20000 | 12 | flat feet',
        'free15':'RW20000 | 15 | free feet','flat15':'RW20000 | 15 | flat feet'}

def assess():
    comp=read(OUT/'fit_complete.json')
    diag=read(OUT/'diagnostic_20000.json')
    assert comp['exit_status']==0 and len(comp['ids'])==16 and all(diag['selected_feasible'])
    cfg=read(OUT/'protocol.json')
    model,cam_cls=common.setup_model()
    layer=model.layer['neutral'].cpu().eval()
    faces=np.asarray(model.face)
    cams=[cam_cls(p) for p in common.camera_files()]
    gt=np.load(common.ARCH/'body_gt_official.npz')['joints_m']
    items=read(OUT/'prepared.json')['items']
    problem=FlatProblem(items,cfg,model,cam_cls)
    floor_rows={r['pair']:r for r in read(E/'floor.json')['rows']}
    patches=cfg['sole_patches']
    results=[]
    with np.load(OUT/'budget_20000.npz') as z:
        for i,(pair,k,_) in enumerate(comp['ids']):
            item=items[i]
            assert (pair,k)==(item['pair'],item['condition'])
            label=LABELS[k]
            p={a:z[b][i] for a,b in [('global_orient','global0'),('body_pose','body_pose'),('betas','betas'),('transl','translation')]}
            assert not np.any(p['body_pose'].reshape(21,3)[[11,14]]) and not np.any(p['betas'])
            normal=np.array(item['floor_up'])
            aa=p['body_pose'].reshape(21,3).copy()
            bt=lambda a: torch.tensor(np.array(a),dtype=torch.float32)[None]
            knees,world=world_ankles(bt(aa),bt(p['global_orient'].reshape(3)),bt(z['delta'][i].reshape(3)))
            if item['flat']:
                target=flat_targets(world,problem.frames,bt(normal))
                local=(knees.transpose(-1,-2)@target)[0].numpy()
                aa[[6,7]]=np.stack([cv2.Rodrigues(rot.astype(float))[0].reshape(3) for rot in local])
                p['body_pose']=aa
            _,actual=world_ankles(bt(aa),bt(p['global_orient'].reshape(3)),bt(z['delta'][i].reshape(3)))
            frame_angles=[angle(u,normal) for u in (actual@problem.frames)[0,:,:,1].numpy()]
            if item['flat']:assert max(frame_angles)<.04
            v,j,head,chest=export.geometry(layer,p,z['delta'][i],False,None)
            assert np.allclose(head,chest,atol=1e-6) and np.isfinite(v).all()
            g=gt[2*(int(pair[-2:])-1)]
            m=j[common.NATIVE_IDS].copy()
            for index,a,b,ratio in zip([0,10,14],[15,8,7],[12,11,10],cfg['landmark_ratios']):
                m[index]=j[a]+ratio*(j[b]-j[a])
            err=np.linalg.norm(m-g,axis=1)*1000
            obs=np.array(read(export.OLD/pair/'observations_local.json')['views'])
            px=[]
            for eye,side in enumerate(['left','right']):
                T=np.asarray(item['inputs']['views'][side]['camera']['left_to_view'])
                xy=cams[eye].world2camera(m@T[:3,:3].T+T[:3,3])
                px.append(np.linalg.norm(xy[common.OBS]-obs[eye,common.OBS,:2],axis=1))
            path=OUT/'objects'/pair/(pair+'__'+label+'__left_cv_m.obj')
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('w') as f:
                f.write('# left camera CV, meters\n')
                np.savetxt(f,v,fmt='v %.9f %.9f %.9f')
                np.savetxt(f,faces+1,fmt='f %d %d %d')
            vv,ff=render.load_obj(path)
            assert np.max(np.abs(vv-v))<1e-8 and np.array_equal(ff,faces)
            sole_tilts=[angle(sole_frame(vv,gp)[:,1],normal) for gp in patches]
            # Check that the full exported body's forefoot matches the geometry optimized by the solver.
            if item['flat']:
                toe_fk=j[[7,8]]+(target[0]@problem.offset[...,None])[...,0].numpy()
                assert np.max(np.abs(toe_fk-j[[10,11]]))<3e-5
            sole_height=[(np.mean([vv[idx].mean(0) for idx in gp],axis=0)@normal+
                          floor_rows[pair]['plane_distance_m'])*1000 for gp in patches]
            results.append(dict(sole_tilt_deg=sole_tilts,mean_sole_tilt_deg=np.mean(sole_tilts),
                  ankle_frame_tilt_deg=frame_angles,sole_signed_height_mm=sole_height,
                  effective_local_ankle_deg=np.rad2deg(np.linalg.norm(aa[[6,7]],axis=1)).tolist(),pair=pair,condition=label,obj=path.name,obj_sha256=sha(path),
                  common12_mm=err[common.OBS].mean(),mapped15_nonpa_mm=err.mean(),
                  mapped15_pa_mm=np.linalg.norm(align(m,g)-g,axis=1).mean()*1000,
                  matched12_reproj_px=np.mean(px),head_position_mm=np.linalg.norm(j[15]-item['head_position'])*1000,
                  head_chest_deg=export.angle(head,chest),root_change_deg=export.angle(np.eye(3),export.rot(z['delta'][i])),
                  converged=diag['convergence_pass'][i],selected_step=int(z['selected_step'][i])))
    by={}
    for group,keys in GROUPS.items():
        rows=[r for r in results if r['condition'] in keys]
        write(E/('results_'+group+'.json'),dict(rows=rows))
        for k in keys:by[k]=[r for r in rows if r['condition']==k]
    fields=['common12_mm','mapped15_nonpa_mm','mapped15_pa_mm','matched12_reproj_px','head_position_mm','head_chest_deg','mean_sole_tilt_deg']
    assert all(len(rows)==4 for rows in by.values())
    means={k:{f:np.mean([r[f] for r in rows]) for f in fields} for k,rows in by.items()}
    reference=read(ROOT/'runs/2026-11-19_rw_auto_history/evidence/summary.json')['means']['body']
    regression=abs(means['free12']['mapped15_nonpa_mm']-reference['mapped15_nonpa_mm'])
    assert regression<.1
    write(E/'summary.json',dict(means=means,free12_reference_delta_mm=regression,
          max_mesh_tilt_flat_deg=max(max(r['sole_tilt_deg']) for r in results if 'flat' in r['condition'])))
    write(E/'execution.json',dict(base='f8fc5ae32df369f92bc269cb9c7cb538d1f7ef4d',worktree='new run plus unrelated existing changes',complete=comp,
        diagnostics=diag,convergence_thresholds=cfg['convergence'],config_sha256=sha(OUT/'protocol.json'),
        checkpoint_sha256=sha(OUT/'budget_20000.npz'),exit_prepare_and_fit=int((OUT/'fit.exit').read_text()),
        camera_sha256={p.name:sha(p) for p in common.camera_files()},
        fit_script_sha256=sha(ROOT/'runs/2026-11-17_newdetector_d10/fit.py'),
        commands=['<BODY_PYTHON> -Bu '+str((RUN/name).relative_to(ROOT)) for name in ['floor.py','plane.py','experiment.py']]+
                 ['<BODY_PYTHON> -Bu '+str((RUN/'assess.py').relative_to(ROOT))+' '+s for s in ['evaluate','render','package']]))


def pictures():
    render.RUN=RUN
    render.OUT=OUT
    render.ASSETS=RUN/'assets'
    render.GROUPS=GROUPS
    render.LABEL.update(LEGEND)
    for group in GROUPS:
        render.main(group)


def package():
    manifest={}
    for group in GROUPS:
        rows=read(E/('results_'+group+'.json'))['rows']
        images=read(E/('renders_'+group+'.json'))['rows']
        assert len(rows)==len(images)==16
        for row in rows:
            f=OUT/'objects'/row['pair']/row['obj']
            r=next(i for i in images if i['pair']==row['pair'] and i['mode']==row['condition'])
            assert row['obj_sha256']==r['source_obj_sha256']==sha(f)
            for n,h in r['images'].items():
                assert sha(RUN/'assets'/row['pair']/n)==h
            manifest[str(f.relative_to(OUT/'objects'))]=sha(f)
    view_hashes=read(E/'floor_views.json')['images']
    assert len(view_hashes)==4 and all(sha(RUN/'assets'/name)==h for name,h in view_hashes.items())
    path=OUT/ARCHIVE
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for n in manifest:z.write(OUT/'objects'/n,n)
        z.writestr('manifest.json',json.dumps(manifest,indent=2))
        z.writestr('README.txt','Left camera CV (X right, Y down, Z forward), meters. No alignment.\n'+json.dumps(LEGEND,indent=2))
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None and len(z.namelist())==18
        for n,h in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
    write(E/'attachment_audit.json',dict(archive=ARCHIVE,tag=TAG,files_sha256={ARCHIVE:sha(path)},bytes=path.stat().st_size,
        counts=dict(meshes=len(manifest),jpeg=len(list((RUN/'assets').rglob('*.jpg')))),members_verified=True,
        side_view_retry={'initial_exit':1,'cause':'Visualization imported a similarly named previous assessment module and selected missing conditions (StopIteration). Switched to the current condition keys. No solver inputs or fitted outputs affected.','command':'<BODY_PYTHON> -Bu '+str((RUN/'side_view.py').relative_to(ROOT))},
        exits={k:int((OUT/(k+'.exit')).read_text()) for k in ['fit','evaluate','render','side_view']}))

if __name__=='__main__':
    {'evaluate':assess,'render':pictures,'package':package}[sys.argv[1]]()
