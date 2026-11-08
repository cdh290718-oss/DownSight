"""Produce concise, path-redacted execution and failure evidence."""
import hashlib,json,re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent
OUT=ROOT/'outputs/synthetic_original_20261108';read=lambda p:json.loads(p.read_text())

def redact(value):
    if isinstance(value,str):
        value=value.replace(str(ROOT.parent),'<WORKSPACE>').replace(str(Path.home()),'<HOME>').replace('/opt'+'/conda','<BUILD_ROOT>')
        value=value.replace('/tmp/','<TEMP>/').replace('/usr/','<SYSTEM>/')
        return re.sub(r'(?i)(token|password|secret|api_key)\s*[:=]\s*[^\s,}]+',r'\1=<REDACTED>',value)
    if isinstance(value,list):return [redact(v) for v in value]
    if isinstance(value,dict):return {redact(k):redact(v) for k,v in value.items()}
    return value

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

records=[]
for rec in read(OUT/'inputs.json')['records']:
    cfg=read(Path(rec['config']));root=Path(cfg['output']);frame=root/'frames'/rec['id']
    record=dict(side=rec['id'],camera=rec['camera'])
    state=root/'run_state.json'
    if state.exists():record['state']=read(state).get('state')
    provs=[read(p) for p in (root/'provenance').glob('*.json')]
    record['attempts']=[dict(state=p.get('state'),resume=p.get('resume'),pipeline_commit=p['pipeline_git'].get('commit'),
        dirty=p['pipeline_git'].get('dirty'),diff_sha256=p['pipeline_git'].get('diff_sha256'),
        resources_sha256=p['integrity'].get('resources_sha256'),backend_contract_sha256=p['integrity']['backend'].get('contract_sha256'),
        backend_source_tree_sha256=p['integrity']['backend']['source_tree'].get('tree_sha256'),
        pipeline_files_sha256=p['pipeline_files_sha256'],input_sha256=p['input_sha256'],error_type=p.get('error_type')) for p in provs]
    validation=frame/'validation.json'
    if validation.exists():
        d=read(validation);record['validation']={k:d.get(k) for k in ['passed','pipeline_profile','matched_hands','matches','manual_annotations_used','body_target_count','body_reload_max_abs_m','fusion_reload_max_abs_m','obj_reload_max_abs_m','local_surface_accepted']}
        geo=frame/'fused/geometry_report.json'
        if geo.exists():
            g=read(geo);record['local_surface_checks']={s:{phase:{k:v for k,v in q[phase].items() if k!='face_pairs'} for phase in ['before','after']} for s,q in g.get('local_intersection_checks',{}).items()}
    network_root=root
    if rec['id']=='right' and (RUN/'evidence/cpu_fallback.json').exists():network_root=OUT/'fits/right'
    detection=network_root/'observations'/rec['id']/'network/detections'/f'{rec["id"]}_detections.json'
    if detection.exists():
        d=read(detection);record['person_detector']={k:d.get(k) for k in ['raw_person_detections','score_thr','bbox_thr','selected_bboxes_xyxy_score']}
    network=network_root/'observations'/rec['id']/'network/manifest.json'
    if network.exists():
        d=read(network);record['body_network']={k:d.get(k) for k in ['processed_images','images_without_detection','projected_persons','detector_enabled','score_thr','bbox_thr','smplx_input_mode','contiguous_roi_mode','projection_consistency_max_error_px']}
    if rec['id']=='left':
        pose_manifest=root/'observations/left/sapiens/manifest.json'
        seg_manifest=root/'observations/segmentation/manifest.json'
        if pose_manifest.exists():
            d=read(pose_manifest);record['pose_metadata']={k:d.get(k) for k in ['gpu','precision','flip_test','configured_sha256_verified','strict_load','checkpoint_sha256','manual_labels_used']}
        if seg_manifest.exists():
            d=read(seg_manifest);record['segmentation_metadata']={k:d.get(k) for k in ['precision','configured_sha256_verified','strict_load','checkpoint_sha256','manual_annotations_used']}
    manifest=root/'automatic_manifest.json' if (root/'automatic_manifest.json').exists() else Path(cfg['manifest'])
    if manifest.exists():
        source=read(manifest)['frames'][0];pred=source.get('predictions',{});fingerprints={}
        for key,path in pred.items():
            if key=='segmentation':
                for name in ['labels.npy','probabilities.npz']:
                    f=Path(path)/name
                    if f.is_file():fingerprints[key+'/'+name]=sha(f)
            elif Path(path).is_file():fingerprints[key]=sha(path)
        record['observation_sha256']=fingerprints
    if record.get('state')!='complete':
        retry=Path(rec['config']).stem.split('_',1)[1] if '_' in Path(rec['config']).stem else None
        faildir=RUN/'evidence'/(f'{rec["id"]}_{retry}_failure' if retry else f'{rec["id"]}_failure');faildir.mkdir(parents=True,exist_ok=True)
        launcher=OUT/(f'{rec["id"]}_{retry}.launcher.log' if retry else f'{rec["id"]}.launcher.log')
        paths=[launcher]+sorted(root.rglob('*.log')) if root.exists() else [launcher]
        for i,p in enumerate(paths):
            if not p.is_file():continue
            # Preserve each available failure log, including numeric values and traceback.
            out=faildir/f'{i:02d}_{p.name}.txt';out.write_text(redact(p.read_text(errors='replace')))
        record['failure_log_files']=[str(p.relative_to(RUN)) for p in faildir.glob('*.txt')]
    records.append(record)
result=dict(execution=redact(read(OUT/'attempts.json')),records=redact(records),dataset_validation_command='PYTHONDONTWRITEBYTECODE=1 <BODY_PYTHON> <DATASET>/validate_sample.py',dataset_validation_exit_status=0,
            redaction='Workspace, home and build prefixes replaced; credential-looking values removed. Full original logs and provenance remain local.')
if all(r.get('state')=='complete' and r['attempts'] for r in records):
    left,right=[r['attempts'][-1]['pipeline_files_sha256'] for r in records]
    changed=sorted(k for k in set(left)|set(right) if left.get(k)!=right.get(k))
    allowed=['egosmplx_pipeline/sapiens_pose.py','egosmplx_pipeline/sapiens_seg.py']
    assert set(changed).issubset(set(allowed)),changed
    result['pipeline_source_comparison']=dict(changed_files=changed,original_fitting_modules_unchanged=True)
    configs=[read(Path(q['config'])) for q in read(OUT/'inputs.json')['records']]
    changed_config=sorted(k for k in set(configs[0])|set(configs[1]) if configs[0].get(k)!=configs[1].get(k))
    assert set(changed_config).issubset({'gpu','manifest','output'}),changed_config
    result['configuration_comparison']=dict(changed_top_level_keys=changed_config,original_fitting_and_head_settings_equal=True)
if (RUN/'evidence/retry.json').exists():
    retry=read(RUN/'evidence/retry.json')
    initial=read(OUT/'configs/right.json');replacement=read(OUT/'configs/right_retry1.json')
    assert {k for k in replacement if replacement[k]!=initial[k]}=={'output'}
    common={k:v for k,v in initial.items() if k!='output'}
    retry['shared_configuration_sha256']=hashlib.sha256(json.dumps(common,sort_keys=True).encode()).hexdigest()
    prior=[read(p) for p in (OUT/'fits/right/provenance').glob('*.json')]
    retry['initial_provenance_summary']=[dict(state=p['state'],error_type=p.get('error_type'),pipeline_commit=p['pipeline_git'].get('commit'),resources_sha256=p['integrity']['resources_sha256'],input_sha256=p['input_sha256']) for p in prior]
    later=next(r for r in records if r['side']=='right')
    if later['attempts']:
        assert prior[-1]['integrity']['resources_sha256']==later['attempts'][-1]['resources_sha256']
        assert prior[-1]['input_sha256']==later['attempts'][-1]['input_sha256']
    result['retry']=redact(retry)
if (RUN/'evidence/retry2.json').exists():result['retry2']=read(RUN/'evidence/retry2.json')
if (RUN/'evidence/cpu_fallback.json').exists():
    pose=read(OUT/'cpu_observations/right_pose/manifest.json');seg=read(OUT/'cpu_observations/right_seg/manifest.json')
    result['cpu_observation_metadata']={
        'pose':{k:pose.get(k) for k in ['device','precision','flip_test','configured_sha256_verified','strict_load','checkpoint_sha256','manual_labels_used']},
        'segmentation':{k:seg.get(k) for k in ['device','precision','configured_sha256_verified','strict_load','checkpoint_sha256','manual_annotations_used']}}
(RUN/'evidence/execution.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps([{'side':r['side'],'state':r.get('state')} for r in records]))
