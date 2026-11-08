"""Export authoritative parameters and freshly rebuilt caches, keeping raw fits."""
import hashlib,json,os
import numpy as np
from common import ROOT,RUN,OUT,OLD,read,project_numpy
from egosmplx_pipeline.bootstrap import setup
from egosmplx_pipeline.io import load
cfg=read(OLD/'configs/left.json');paths=cfg['paths'];os.environ.update(EGOSMPLX_CONFIG=paths['egosmplx_config'],EGOSMPLX_REPO_ROOT=paths['egosmplx_repo']);model,_=setup(paths['egosmplx_repo'])
from egosmplx_pipeline import fusion
fusion.initialize(model);inputs=read(OUT/'inputs.json');records=[]
for q in read(RUN/'evidence/execution.json'):
    folder=OUT/(q['case']+'_'+q['seed']);source=folder/'final.npz'
    with np.load(source) as z:
        p={k:z[k].copy() for k in fusion.PARAMS+['transl']}
        old_caches={k:z[k].copy() for k in ['smplx_joint_cam','smplx_joint_img','smplx_body_joints_2d_render_cam'] if k in z}
    mesh,joints=fusion.geometry(p);uv=project_numpy(joints,inputs['views']['left']['camera'])
    p.update(coordinate_frame=np.array('cam3 CV'),smplx_mesh_cam=mesh,smplx_joint_cam=joints,smplx_joint_cam_root_relative=joints-joints[:1],smplx_body_joints_3d_local=joints[:25]-p['transl'].reshape(3),smplx_body_joints_3d_render_cam=joints[:25])
    dst=folder/'parameters.npz';np.savez_compressed(dst,**p)
    rebuilt=dict(smplx_joint_cam=joints,smplx_joint_img=uv,smplx_body_joints_2d_render_cam=uv[:25])
    mismatch={k:float(np.nanmax(abs(v-rebuilt[k]))) for k,v in old_caches.items()}
    load(dst)  # Canonical parameter files remain compatible with strict finite I/O.
    with np.load(source) as old:mesh_error=float(np.max(abs(mesh-old['smplx_mesh_cam'])))
    assert mesh_error<1e-5
    projection_file=folder/'projections.npz';np.savez_compressed(projection_file,uv=uv,valid=np.isfinite(uv).all(1),coordinate_frame=np.array('cam3 CV'))
    records.append(dict(case=q['case'],seed=q['seed'],source_final_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),canonical_export_sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),all_parameter_arrays_unchanged=True,reconstructed_mesh_max_m=mesh_error,strict_parameter_load_passed=True,invalid_projection_joint_count=int((~np.isfinite(uv).all(1)).sum()),legacy_cache_max_difference=mismatch))
(RUN/'evidence/exports.json').write_text(json.dumps(dict(exit_status=0,command='<BODY_PYTHON> runs/2026-11-08_synthetic-stereo-abcd/export.py',scope='Raw final.npz copies include inherited network joint/projection caches. Metrics reconstruct authoritative parameters, never use those caches. parameters.npz is the finite canonical export with parameters and rebuilt 3D caches; optional projections.npz explicitly masks behind-camera points. Fitted parameters unchanged. Original fit outputs retained.',records=records),indent=2)+'\n');print('exported 8 canonical parameter files with rebuilt caches')
