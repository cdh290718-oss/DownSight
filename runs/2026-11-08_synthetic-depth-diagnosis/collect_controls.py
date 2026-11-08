"""Publish numeric checks and optimizer traces, without local paths or raw outputs."""
import json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[2];D=R/'outputs/synthetic_depth_diagnosis_20261108';O=R/'outputs/synthetic_original_20261108';E=Path(__file__).resolve().parent/'evidence'
def read(p):return json.loads(p.read_text())
def delta(a,b):
    with np.load(a,allow_pickle=False) as x,np.load(b,allow_pickle=False) as y:
        return {k:float(np.max(np.abs(x[k]-y[k]))) for k in ['body_pose','global_orient','transl','smplx_mesh_cam']}
body={}
for scope in ['all','non_head_neck','oracle']:
    r=read(D/scope/'automatic/refinement_report.json');body[scope]={k:r[k] for k in ['best_step','best_objective','minimum_vertex_z_m','target_fit_before','target_fit_after','history']}
rigid={}
for scope in ['all','non_head_neck','none']:
    r=read(D/('rigid_'+scope)/'optimization_report.json');rigid[scope]={k:r[k] for k in ['history','raw_to_sapiens2','optimized_to_sapiens2','optimized_cam_trans_m','global_orient_delta_deg']}
result={'body_controls':body,'rigid_controls':rigid,'parameter_max_abs_differences':{'body_original_vs_control_all':delta(O/'fits/left/frames/left/body/params.npz',D/'all/automatic/refined_params.npz'),'body_all_vs_non_head_neck':delta(D/'all/automatic/refined_params.npz',D/'non_head_neck/automatic/refined_params.npz'),'rigid_original_vs_control_all':delta(O/'fits/left/frames/left/rigid/optimized_params.npz',D/'rigid_all/optimized_params.npz'),'rigid_all_vs_no_depth_penalty':delta(D/'rigid_all/optimized_params.npz',D/'rigid_none/optimized_params.npz')}}
(E/'controls.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
