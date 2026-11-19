"""Summarize paired diagnostics; package/verify only the requested fitted surfaces."""
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np

from study import ROOT,RUN,OUT,OLD,common,read,write,sha
from export import OLD_GROUP,CONTROL,CAM_GROUP,AUTO,AUTO_HEAD,LABEL

FAMILIES=['current','camera','automatic','old','automatic_head']
TAG='rw-s9-landmarks-head-20261119'
ARCHIVE='rw_s9_body01-04_landmarks_head_OBJ.zip'


def main():
    records=[]
    renders=[]
    for family in FAMILIES:
        records.extend(read(RUN/'evidence'/('results_'+family+'.json'))['rows'])
        renders.extend(read(RUN/'evidence'/('renders_'+family+'.json'))['rows'])
    keys=OLD_GROUP+CONTROL+CAM_GROUP+AUTO+AUTO_HEAD
    assert len(records)==len(set((r['pair'],r['condition']) for r in records))==4*len(keys)
    by={m:{} for m in keys}
    metrics=['common12_mm','native15_nonpa_mm','native15_pa_mm','mapped15_nonpa_mm',
             'mapped15_pa_mm','matched12_reproj_px','mapped15_reproj_px',
             'head_position_mm','head_chest_deg','head_history_deg','head_camera_proxy_deg']
    for m in keys:
        rows=[r for r in records if r['condition']==m]
        assert len(rows)==4
        by[m]={k:float(np.mean([r[k] for r in rows])) for k in metrics}
        by[m]['converged']=sum(r['converged'] for r in rows)
        by[m]['mapped_individual_mm']=dict(zip(common.NAMES,np.mean([r['mapped_errors_mm'] for r in rows],axis=0).tolist()))
    checks=[]
    original=read(ROOT/'runs/2026-11-15_unrealego-gt2d/evidence/results.json')['records']
    replay=read(ROOT/'runs/2026-11-19_rw_head_neck_oracle/evidence/results.json')['records']
    for k in range(4):
        pair='body%02d'%(k+1)
        def row(m):return next(r for r in records if r['pair']==pair and r['condition']==m)
        legacy_error=abs(row('old_history')['native15_nonpa_mm']-original[k]['absolute15_mean_mm'])
        checks.append(dict(pair=pair,old_original_nonpa_difference_mm=legacy_error,
            current12_history_nonpa_difference_mm=abs(row('baseline')['native15_nonpa_mm']-replay[k]['absolute_mean_mm'])))
        assert legacy_error<.01
        source_obj=ROOT/'data/obj'/('UnrealEgoRW_GT2D_'+pair+'__D_fit__leftcam_m.obj')
        rebuilt=OUT/'objects'/pair/row('old_history')['obj']
        def mesh(path):
            lines=path.read_text().splitlines()
            verts=np.loadtxt(io.StringIO('\n'.join(v[2:] for v in lines if v.startswith('v '))))
            faces=[v.split()[1:] for v in lines if v.startswith('f ')]
            return verts,faces
        orig_v,orig_faces=mesh(source_obj)
        rebuilt_v,rebuilt_faces=mesh(rebuilt)
        diff=float(np.abs(orig_v-rebuilt_v).max()*1000)
        assert orig_faces==rebuilt_faces and diff<.003
        checks[-1].update(archive_obj_sha256=sha(source_obj),original_obj_max_coordinate_difference_mm=diff,original_obj_faces_identical=True)
        for mode in ['weak','off']:
            new=OUT/'old'/mode/pair
            old=OLD/pair
            for f in ['inputs.json','model_paths.json','triangulated.json','anchors/oracle.json']:
                assert sha(new/f)==sha(old/f)
            with np.load(old/'D_left/initial.npz') as a,np.load(new/'D_left/initial.npz') as b:
                assert a.files==b.files
                for name in a.files:
                    assert np.array_equal(a[name],b[name])
    rc={p.name:int(p.read_text()) for p in OUT.glob('*.exit')}
    required={'prepare','fit','camera_only','automatic','automatic_head','point_checks'}
    required.update(stage+'_'+family for stage in ['export','render'] for family in FAMILIES)
    required.update('old_'+mode+'_body%02d'%i for mode in ['off','weak'] for i in range(1,5))
    assert all(rc.get(k+'.exit')==0 for k in required)
    detector=read(RUN/'evidence/automatic_inputs.json')
    auto_error={j:float(np.mean([r['original_px'] for r in detector['errors'] if r['joint']==j])) for j in common.NAMES}
    comparisons={}
    for a,b in [('baseline','15history'),('12camera','15camera'),('12torso','15torso'),('auto12','auto15'),('auto12_camera','auto15_camera'),('auto12_torso','auto15_torso'),('old_history','old_off'),('old_history','old_weak')]:
        paired=[]
        for k in range(4):
            pair='body%02d'%(k+1)
            x,y=[next(v for v in records if v['pair']==pair and v['condition']==m) for m in [a,b]]
            paired.append(dict(pair=pair,common12_delta_mm=y['common12_mm']-x['common12_mm'],
                               mapped15_delta_mm=y['mapped15_nonpa_mm']-x['mapped15_nonpa_mm']))
        comparisons[a+'_to_'+b]=paired
    write(RUN/'evidence/summary.json',dict(means=by,pairs=checks,differences=comparisons,
         automatic_original_mean_px=auto_error,source_records=len(records),views_per_mesh=4))
    hashes={}
    manifest=[]
    archive=OUT/ARCHIVE
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for r in records:
            p=OUT/'objects'/r['pair']/r['obj']
            digest=sha(p)
            assert digest==r['obj_sha256']
            vis=next(v for v in renders if v['pair']==r['pair'] and v['mode']==r['condition'])
            assert vis['source_obj_sha256']==digest
            for name,h in vis['images'].items():
                assert sha(RUN/'assets'/r['pair']/name)==h
            relative=r['pair']+'/'+r['obj']
            hashes[relative]=digest
            manifest.append(dict(file=relative,sha256=digest,vertices=r['vertices'],faces=r['faces'],
                                 pair=r['pair'],condition=r['condition']))
            z.write(p,relative)
        z.writestr('manifest.json',json.dumps(manifest,indent=2))
        readme='Four S9 poses. Each OBJ is a shared stereo fit in LEFT CAMERA CV (+X right, +Y down, +Z forward), meters. Same topology in every file. No PA alignment or scale changes. No texture, skinning rig or weights.\n'
        readme+='\n'.join(k+' = '+LABEL[k] for k in keys)
        z.writestr('README.txt',readme+'\nDetails and matched images: https://github.com/cdh290718-oss/DownSight/tree/'+TAG+'/runs/'+RUN.name+'\n')
    with zipfile.ZipFile(archive) as z:
        assert not z.testzip()
        assert set(z.namelist())==set(hashes)|{'manifest.json','README.txt'}
        for path,expected in hashes.items():
            assert hashlib.sha256(z.read(path)).hexdigest()==expected
    write(RUN/'evidence/attachment_audit.json',dict(archive=ARCHIVE,tag=TAG,files_sha256={ARCHIVE:sha(archive)},
          bytes=archive.stat().st_size,mesh_count=len(records),members=len(hashes)+2,exit_status=0,
          checks='OBJ topology, reread coordinates, image-source mesh identity, archive member hashes; no body model, raw photos, predictions or weights.'))
    write(RUN/'evidence/execution.json',dict(base='736ee9d7ca77b54571a91f0a07648c5ad189bd1d',
          worktree='New research run plus unrelated existing local changes. Output isolated and CPU-only.',
          exits=rc,commands=['<BODY_PYTHON> -Bu '+str((RUN/f).relative_to(ROOT))+' '+args for f,args in
          [('old_rotation.py','<body01..04> <weak|off>'),('study.py','prepare'),('study.py','fit'),
           ('automatic.py',''),('automatic_head.py',''),('point_checks.py',''),('camera_only.py',''),('export.py','<family>'),('render.py','<family>'),('package.py','')]],
          actual_protocols={d.name:sha(d/'protocol.json') for d in [OUT,OUT/'automatic',OUT/'camera_only',OUT/'automatic_head']},
          fit_settings={k:read(OUT/'protocol.json')[k] for k in ['pose_prior_weight','metric_loss_weight','shoulder_weight','head_position_sigma_m','head_rotation_sigma_deg','head_position_weight','head_rotation_weight','sigma_px_fixed','budgets','selection']},
          calibration_files_sha256={p.name:sha(p) for p in common.camera_files()},
          prepared_shape_ranks={d.name:{k:v['rank'] for k,v in read(d/'initialization.json')['initialization'].items()}
                                for d in [OUT,OUT/'automatic',OUT/'camera_only',OUT/'automatic_head']},
          old_cfg_sha256=sha(ROOT/'runs/2026-11-15_unrealego-gt2d/settings.json'),
          prior_golden_files={f:sha(ROOT/'runs/2026-11-17_newdetector_d10'/f) for f in ['fit.py','geometry.py','common.py','protocol.json']},
          diagnostics={d.name:{k:v for k,v in read(d/'diagnostic_20000.json').items() if k!='ids'}
                       for d in [OUT,OUT/'automatic',OUT/'camera_only',OUT/'automatic_head']},
          convergence_thresholds=read(OUT/'protocol.json')['convergence'],
          cpu_complete={d.name:read(d/'fit_complete.json') for d in [OUT,OUT/'automatic',OUT/'camera_only',OUT/'automatic_head']},
          photos_sha256={p.name:sha(p) for p in sorted((common.ARCH/'body_input').glob('body0[1-4]_*.png'))}))
    return by


if __name__=='__main__':
    main()
