"""Evaluate the completed position-only experiment, then render and package its fitted surfaces."""
import hashlib
import json
import sys
import zipfile
import numpy as np
from trial import RUN,OUT,PREV,ROOT,common,read,write,sha,LABELS
import export
import render
from evaluate import align

E=RUN/'evidence'
TAG='rw-s9-position-only-20261119'
ARCHIVE='rw_s9_position_only_OBJ.zip'
GROUPS={'auto':['auto_fixed','auto_body'],'gt':['gt_fixed','gt_body']}
LEGEND={'auto_fixed':'RW20000 12 | fixed start','auto_body':'RW20000 12 | shoulder/hip start',
        'gt_fixed':'Official 15 | fixed start','gt_body':'Official 15 | shoulder/hip start'}


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
    results=[]
    with np.load(OUT/'budget_20000.npz') as z:
        for i,(pair,k,_) in enumerate(comp['ids']):
            item=items[i]
            assert (pair,k)==(item['pair'],item['condition'])
            label=LABELS[k]
            p={a:z[b][i] for a,b in [('global_orient','global0'),('body_pose','body_pose'),('betas','betas'),('transl','translation')]}
            assert not np.any(p['body_pose'].reshape(21,3)[[11,14]]) and not np.any(p['betas'])
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
            results.append(dict(pair=pair,condition=label,obj=path.name,obj_sha256=sha(path),
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
    for stage,k,name in [('automatic_head','auto12_torso','auto_proxy'),('current','15torso','gt_history'),('camera','torso_caminit','gt_proxy')]:
        by[name]=[r for r in read(PREV/'evidence'/('results_'+stage+'.json'))['rows'] if r['condition']==k]
    fields=['common12_mm','mapped15_nonpa_mm','mapped15_pa_mm','matched12_reproj_px','head_position_mm','head_chest_deg']
    assert all(len(rows)==4 for rows in by.values())
    means={k:{f:np.mean([r[f] for r in rows]) for f in fields} for k,rows in by.items()}
    write(E/'summary.json',dict(means=means,three_D_mm_by_pair={k:{r['pair']:r['mapped15_nonpa_mm'] for r in rows} for k,rows in by.items()}))
    write(E/'execution.json',dict(base='78748243914c8727ed79ab43e07e2bbe46693fa3',worktree='new run plus unrelated existing changes',complete=comp,
        diagnostics=diag,convergence_thresholds=cfg['convergence'],config_sha256=sha(OUT/'protocol.json'),
        checkpoint_sha256=sha(OUT/'budget_20000.npz'),exit_prepare_and_fit=int((OUT/'pipeline.exit').read_text()),
        camera_sha256={p.name:sha(p) for p in common.camera_files()},
        fit_script_sha256=sha(ROOT/'runs/2026-11-17_newdetector_d10/fit.py'),
        commands=['<BODY_PYTHON> -Bu '+str((RUN/'trial.py').relative_to(ROOT))+' '+s for s in ['prepare','fit']]+
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
        assert len(rows)==len(images)==8
        for row in rows:
            f=OUT/'objects'/row['pair']/row['obj']
            r=next(i for i in images if i['pair']==row['pair'] and i['mode']==row['condition'])
            assert row['obj_sha256']==r['source_obj_sha256']==sha(f)
            for n,h in r['images'].items():
                assert sha(RUN/'assets'/row['pair']/n)==h
            manifest[str(f.relative_to(OUT/'objects'))]=sha(f)
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
        exits={k:int((OUT/(k+'.exit')).read_text()) for k in ['pipeline','evaluate','render']}))

if __name__=='__main__':
    {'evaluate':assess,'render':pictures,'package':package}[sys.argv[1]]()
