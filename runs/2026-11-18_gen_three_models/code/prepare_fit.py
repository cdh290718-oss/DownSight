import json,copy,hashlib,cv2,numpy as np
from pathlib import Path
root=Path.cwd();r=root/'runs/2026-11-18_gen_three_models';out=root/'outputs/gen_three_models_20261118';data=root.parent/'gen_egosmplx_run';dfm=np.load(data/'dfm_reference_virtual.npz');frames=list(map(int,dfm['frames']));rig=json.loads((root/'experiments/gen_stereo_analysis_20261030_20261103/first_stereo_20261031/transform.json').read_text());T=np.eye(4);T[:3,:3]=rig['R_cam4_from_cam3'];T[:3,3]=rig['t_cam4_from_cam3_m'];template=json.loads((root/'outputs/newdetector_d10_20261117/prepared.json').read_text())['items'][0];pred={v:{x['id']:x for x in json.loads((out/(v+'.json')).read_text())} for v in ['rw500','rw5000','rw20000']};indices={p['name']:p['native_index'] for p in template['points']};rows=[];sources=[]
for frame in frames:
 ident=f'gen_{frame:05d}_cam3';raw=[]
 for part in ('outputs_tangent_retry2','outputs_resume'):
  path=root/'runs/2026-11-16_gen-body-only-20'/part/ident/'fit/raw/params.npz'
  if path.exists():raw.append(path)
 if raw:
  z=np.load(raw[0]);source='historical canonical network output'
 else:
  paths=list((out/'head_network'/ident/'smplx').glob('*_smplx.npz'));assert len(paths)==1,(ident,paths);z=np.load(paths[0]);source='new network output'
 body=np.asarray(z['body_pose']).reshape(21,3);g=np.asarray(z['global_orient_network'] if 'global_orient_network' in z else z['global_orient']).reshape(3);R=cv2.Rodrigues(g)[0]
 for j in [3,6,9,12,15]:R=R@cv2.Rodrigues(body[j-1])[0]
 sources.append(dict(frame=frame,source=source,sha256=hashlib.sha256(Path(z.fid.name).read_bytes()).hexdigest()))
 for version in pred:
  item=copy.deepcopy(template);item['pair']=f'gen_{frame:05d}';item['sequence']='Gen single recording';item['subject']='Gen person';item['condition']=version;item['anchor']={'position':dfm['cam3_body_xyz'][frames.index(frame),15].tolist(),'rotation':R.tolist()};item['inputs']['views']['left']['camera']['left_to_view']=np.eye(4).tolist();item['inputs']['views']['right']['camera']['left_to_view']=T.tolist()
  for p in item['points']:
   name=p['name'];mapped=('right_' if name.startswith('R_') else 'left_')+p['part'];p['xy']=[];p['valid']=[];p['scores']=[]
   for view in ['cam3','cam4']:
    rec=pred[version][f'gen_{frame:05d}_{view}'];q=next(x for x in rec['points'] if x['name']==mapped);p['xy'].append(q['prediction']);p['valid'].append(q['reference_in_image']);p['scores'].append(q['score'])
   p['accepted']=False;p.pop('xyz',None);p.pop('covariance',None)
  rows.append(item)
assert len(rows)==60
(out/'prepared.json').open('x').write(json.dumps({'items':rows,'protocol_sha256':hashlib.sha256((r/'protocol.json').read_bytes()).hexdigest()},indent=2));(r/'evidence/preparation.json').write_text(json.dumps(dict(command='prepare_fit.py',exit_status=0,pairs=20,conditions=3,head_position='DFM model reference; not independently verified',head_orientation='EgoSMPLX estimated head rotation, network',relative_camera='rigid alignment of DFM body reference points; reference information leaks into camera calibration',mask='DFM reference-in-frame; not actual visibility; no confidence threshold',sources=sources),indent=2)+'\n')
