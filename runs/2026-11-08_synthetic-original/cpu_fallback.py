"""Run the same fresh observations and original fitting on CPU after GPU faults."""
import argparse,hashlib,json,os,re,subprocess,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent
OUT=ROOT/'outputs/synthetic_original_20261108';DATA=OUT/'dataset/samples/000001';read=lambda p:json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def redact(text):
    text=text.replace(str(ROOT.parent),'<WORKSPACE>').replace(str(Path.home()),'<HOME>').replace('/opt'+'/conda','<BUILD_ROOT>').replace('/tmp/','<TEMP>/').replace('/usr/','<SYSTEM>/')
    return re.sub(r'(?i)(token|password|secret|api_key)\s*[:=]\s*[^\s,}]+',r'\1=<REDACTED>',text)

parser=argparse.ArgumentParser();parser.add_argument('--reproduce',action='store_true',help='Run CPU comparison after a completed original right body prediction')
args=parser.parse_args()
attempts=read(OUT/'attempts.json')
if not args.reproduce:
    assert attempts[-1]['attempt']=='retry2' and attempts[-1]['exit_status']!=0
    failed=OUT/'fits/right_retry2';logdir=RUN/'evidence/right_retry2_failure';logdir.mkdir(exist_ok=False)
    for i,p in enumerate([OUT/'right_retry2.launcher.log']+sorted(failed.rglob('*.log'))):
        if p.is_file():(logdir/f'{i:02d}_{p.name}.txt').write_text(redact(p.read_text(errors='replace')))
cfg=read(OUT/'configs/right.json');old=Path(cfg['output'])
source=old/'observations/right/network/smplx/right_00_smplx.npz';meta=read(old/'observations/right/network/manifest.json')
assert source.is_file() and meta['processed_images']==1 and meta['images_without_detection']==0 and meta['projected_persons']==1
assert sha(old/'observations/images/right.png')==sha(DATA/'right.png')
assert sha(Path(meta['fisheye_calibration']))==sha(OUT/'dataset/calibration/cam4_right_lower.egosmplx.json')
CPU=OUT/'cpu_observations';CPU.mkdir(exist_ok=False)
manifest=CPU/'right_manifest.json';manifest.write_text(json.dumps({'frames':[{'id':'right','camera':'cam4','image':str(DATA/'right.png')}]},indent=2)+'\n')
runtime=OUT/'runtime';env=os.environ.copy();env.update(PYTHONPATH=os.pathsep.join([str(ROOT),cfg['paths']['wilor_repo']]),
    PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
    TMPDIR=str(runtime),XDG_CACHE_HOME=str(runtime/'cache'),XDG_CONFIG_HOME=str(runtime/'config'),MPLCONFIGDIR=str(runtime/'matplotlib'),
    YOLO_CONFIG_DIR=str(runtime/'ultralytics'),TORCH_HOME=str(runtime/'torch'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',YOLO_AUTOINSTALL='false')
paths=cfg['paths'];pose_common=['--repo-root',paths['sapiens_repo'],'--dependencies',paths['sapiens_dependencies'],'--checkpoint',paths['pose_checkpoint'],'--checkpoint-sha256',cfg['checkpoint_sha256']['pose'],'--device','cpu']
jobs={
 'right_pose':([paths['pose_python'],'-m','egosmplx_pipeline.sapiens_pose',*pose_common,'--image',str(DATA/'right.png'),'--output-dir',str(CPU/'right_pose')],ROOT),
 'right_seg':([paths['pose_python'],'-m','egosmplx_pipeline.sapiens_seg','--repo-root',paths['sapiens_repo'],'--dependencies',paths['sapiens_dependencies'],'--checkpoint',paths['seg_checkpoint'],'--checkpoint-sha256',cfg['checkpoint_sha256']['seg'],'--manifest',str(manifest),'--output-dir',str(CPU/'right_seg'),'--device','cpu'],ROOT),
 'right_wilor':([paths['wilor_python'],'-m','egosmplx_pipeline.wilor_predict','--manifest',str(manifest),'--output-dir',str(CPU/'right_wilor'),'--checkpoint',paths['wilor_checkpoint'],'--config',paths['wilor_config'],'--detector',paths['wilor_detector']],Path(paths['wilor_repo'])),
 'left_pose_control':([paths['pose_python'],'-m','egosmplx_pipeline.sapiens_pose',*pose_common,'--image',str(DATA/'left.png'),'--output-dir',str(CPU/'left_pose_control')],ROOT),
}
def execute(item):
    name,(cmd,cwd)=item;start=time.time()
    with (CPU/(name+'.log')).open('w') as f:r=subprocess.run(cmd,env=env,cwd=cwd,stdout=f,stderr=subprocess.STDOUT)
    return dict(name=name,exit_status=r.returncode,elapsed_seconds=time.time()-start,command=[redact(str(x)) for x in cmd])
results=[]
with ThreadPoolExecutor(max_workers=3) as pool:
    for future in as_completed([pool.submit(execute,item) for item in jobs.items()]):
        r=future.result();results.append(r);(CPU/'workers.json').write_text(json.dumps(results,indent=2)+'\n');print(r['name'],'exit',r['exit_status'],'seconds',round(r['elapsed_seconds'],1),flush=True)
        if r['exit_status']:
            (RUN/'evidence'/(r['name']+'_failure.txt')).write_text(redact((CPU/(r['name']+'.log')).read_text(errors='replace')))
(RUN/'evidence/cpu_workers.json').write_text(json.dumps(results,indent=2)+'\n')
required={'right_pose','right_seg','right_wilor'}
if any(r['exit_status'] for r in results if r['name'] in required):
    attempts.append(dict(side='right',attempt='cpu_observations',exit_status=1,reason='one or more CPU observation workers failed'))
    (OUT/'attempts.json').write_text(json.dumps(attempts,indent=2)+'\n');raise SystemExit(1)
gpu_pose=read(old.parent/'left/observations/left/sapiens/keypoints.json')
control=next(r for r in results if r['name']=='left_pose_control')
if control['exit_status']==0:
    import math
    cpu_pose=read(CPU/'left_pose_control/keypoints.json');a={q['name']:q for q in gpu_pose['keypoints']};b={q['name']:q for q in cpu_pose['keypoints']}
    assert gpu_pose['image_sha256']==cpu_pose['image_sha256'] and set(a)==set(b)
    names=['nose','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle']
    points=[dict(name=n,position_delta_px=math.hypot(a[n]['x']-b[n]['x'],a[n]['y']-b[n]['y']),gpu_score=a[n]['score'],cpu_score=b[n]['score'],same_score_threshold=(a[n]['score']>=.3)==(b[n]['score']>=.3)) for n in names]
    (RUN/'evidence/device_control.json').write_text(json.dumps(dict(scope='Same left input and Sapiens2 checkpoint, original GPU BF16 versus CPU BF16; no fitting and no ground truth',points=points,
        mean_position_delta_px=sum(q['position_delta_px'] for q in points)/len(points),max_position_delta_px=max(q['position_delta_px'] for q in points),score_threshold_disagreements=sum(not q['same_score_threshold'] for q in points)),indent=2)+'\n')
row=read(OUT/'manifests/right.json')['frames'][0];row['predictions']={'egosmplx':str(source),'sapiens':str(CPU/'right_pose/keypoints.json'),
    'wilor':str(CPU/'right_wilor/npz/right_wilor.npz'),'segmentation':str(CPU/'right_seg/frames/right')}
cached_manifest=CPU/'right_cached_manifest.json';cached_manifest.write_text(json.dumps(dict(frames=[row]),indent=2)+'\n')
new=dict(cfg);new.update(manifest=str(cached_manifest),output=str(OUT/'fits/right_cpu'),gpu='')
assert {k for k in new if new[k]!=cfg.get(k)}=={'manifest','output','gpu'}
config=OUT/'configs/right_cpu.json';config.write_text(json.dumps(new,indent=2)+'\n')
inputs=read(OUT/'inputs.json');next(r for r in inputs['records'] if r['id']=='right')['config']=str(config);(OUT/'inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
details=dict(reason=('Optional CPU reproduction' if args.reproduce else 'Three original GPU attempts failed with CUDA unspecified launch failure; no GPU reset or other process was altered'),
    reused_body_prediction_sha256=sha(source),reused_body_predictor=('right initial attempt, successfully completed' if args.reproduce else 'right initial attempt, successfully completed prior to failed Sapiens2'),
    image_sha256=sha(DATA/'right.png'),calibration_sha256=sha(OUT/'dataset/calibration/cam4_right_lower.egosmplx.json'),
    new_observations=['right_pose','right_seg','right_wilor'],new_device='cpu',cuda_visible_devices='',precision='Sapiens autocast bfloat16, WiLoR float32',
    original_model_weights_preprocessing_and_fit_parameters_unchanged=True,config_changes=['manifest','output','gpu'],
    cached_predictions='All observations produced from this same input in this run; no ground truth used')
(RUN/'evidence/cpu_fallback.json').write_text(json.dumps(details,indent=2)+'\n')
cmd=[paths['body_python'],'-m','egosmplx_pipeline','run','--config',str(config),'--cached-predictions'];start=time.time()
with (OUT/'right_cpu.launcher.log').open('w') as f:r=subprocess.run(cmd,env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
attempts.append(dict(side='right',attempt='cpu_fitting',command=cmd,exit_status=r.returncode,elapsed_seconds=time.time()-start))
(OUT/'attempts.json').write_text(json.dumps(attempts,indent=2)+'\n');print('right CPU original fit exit',r.returncode,flush=True)
raise SystemExit(r.returncode)
