"""Collect compact reproducibility and validation facts, never full provenance."""
import hashlib,json,re,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
RUN=Path(__file__).resolve().parent
OUT=ROOT/'outputs/hand_stage_ablation_20261104'
read=lambda p:json.loads(p.read_text())
def redact(value):
    if isinstance(value,str):
        return value.replace(str(ROOT.parent),'<WORKSPACE>').replace(str(Path.home()),'<HOME>')
    if isinstance(value,list):return [redact(v) for v in value]
    if isinstance(value,dict):return {redact(k):redact(v) for k,v in value.items()}
    return value
records=[];hashsets=[];refs={};frozen={}
for group in ['A','B','C']:
    refs[group]=read(OUT/group/'inputs.json')['records']
    if group!='A':
        batch=read(OUT/group/'batch/execution.json');assert len(batch)==6 and all(q['exit_status']==0 for q in batch)
        hashes=read(OUT/group/'batch/code_hashes.json');hashsets.append(hashes)
        for name,digest in hashes.items():
            executed=subprocess.check_output(['git','show','5ee8502bf403028adeadd9c3174348b7045930c6:'+name])
            assert hashlib.sha256(executed).hexdigest()==digest
    for rec in refs[group]:
        cfg=read(Path(rec['config']));root=Path(cfg['output']);frame=root/'frames'/rec['id']
        assert read(root/'run_state.json')['state']=='complete'
        prov=[read(p) for p in (root/'provenance').glob('*.json')]
        completed=[p for p in prov if p['state']=='complete']
        assert len(completed)==1
        p=completed[0];rigid=read(frame/'rigid/optimization_report.json');pose=read(frame/'body_fit/automatic/refinement_report.json');val=read(frame/'validation.json')
        val={k:v for k,v in val.items() if k in ['passed','pipeline_profile','matched_hands','body_reload_max_abs_m','fusion_reload_max_abs_m','obj_reload_max_abs_m','local_surface_accepted','surface_audit_scope','selected_body_candidate','minimum_fused_vertex_z_m','body_rmse_px','hand_rmse_px','failure_reason','failures','errors','warnings']}
        refinement={}
        for stage in ['wrist/frames','wrist/active_frames','contour/active_frames','contour/strong_frames','contour/selected_frames']:
            path=root/'reference'/stage/rec['id']/rec['camera']/'refinement.json'
            if path.exists():
                rr=read(path);refinement[stage]={k:rr.get(k) for k in ['steps','loss','rejected_steps','cpu_feasibility_step_fraction','selected_candidate','selection_scores','selection_diagnostics']}
        records.append(dict(group=group,frame=rec['id'],reference_refinement=refinement,attempts=[{k:a.get(k) for k in ['state','resume','started_utc','finished_utc']} for a in prov],config_sha256=p['config_sha256'],pipeline_git={k:p['pipeline_git'].get(k) for k in ['commit','dirty','diff_sha256']},resources_sha256=p['integrity']['resources_sha256'],backend_contract_sha256=p['integrity']['backend']['contract_sha256'],backend_source_tree_sha256=p['integrity']['backend']['source_tree']['tree_sha256'],input_sha256=p['input_sha256'],numerical_source_sha256=p['pipeline_files_sha256'],rigid_hand_stage=rigid.get('hand_stage','full'),rigid_selected_names=[q['name'] for q in rigid['selected']],pose_limits=pose['joints_at_increment_limit'],pose_head_reference=pose['head_neck'],validation=val))
assert hashsets[0]==hashsets[1]
for i in range(6):
    trio=[next(q for q in records if q['group']==g and q['frame']==refs[g][i]['id']) for g in ['A','B','C']]
    assert trio[0]['resources_sha256']==trio[1]['resources_sha256']==trio[2]['resources_sha256']
    assert trio[0]['input_sha256']==trio[1]['input_sha256']==trio[2]['input_sha256']
    assert trio[0]['backend_contract_sha256']==trio[1]['backend_contract_sha256']==trio[2]['backend_contract_sha256']
    assert trio[0]['backend_source_tree_sha256']==trio[1]['backend_source_tree_sha256']==trio[2]['backend_source_tree_sha256']
    assert trio[0]['rigid_selected_names']==trio[1]['rigid_selected_names']==trio[2]['rigid_selected_names']
    arrays=[]
    for g in ['B','C']:
        rec=refs[g][i];cfg=read(Path(rec['config']));arrays.append(np.load(Path(cfg['output'])/'frames'/rec['id']/'rigid/optimized_params.npz'))
    assert set(arrays[0].files)==set(arrays[1].files)
    assert all(np.array_equal(arrays[0][k],arrays[1][k]) for k in arrays[0].files)
    rec=refs['A'][i];paths=dict(rec['row']['predictions'])
    segment=Path(paths.pop('segmentation'))
    paths.update(segment_labels=str(segment/'labels.npy'),segment_probabilities=str(segment/'probabilities.npz'),stereo_context=rec['context'])
    frozen[rec['id']]={k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in paths.items()}
    for g in ['B','C']:
        assert refs[g][i]['row']==rec['row'] and refs[g][i]['context']==rec['context']
result=redact(dict(records=records,frozen_observation_sha256=frozen,B_C_rigid_parameters_bitwise_equal=True,all_resource_and_input_hashes_equal=True,B_C_numerical_source_equal=True,execution={g:read(OUT/g/'batch/execution.json') for g in ['B','C']}))
(RUN/'evidence/execution_and_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print('PASS: 12 completed fits; fixed resources/inputs; identical B/C rigid parameters')
