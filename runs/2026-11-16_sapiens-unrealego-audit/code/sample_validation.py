import argparse, io, json, hashlib, tarfile, zipfile
from pathlib import Path
import cv2, numpy as np
p=argparse.ArgumentParser(); p.add_argument('--archive',required=True);p.add_argument('--prepared',required=True);p.add_argument('--splits',required=True);p.add_argument('--out',required=True);p.add_argument('--evidence',required=True);a=p.parse_args()
out=Path(a.out); out.mkdir(parents=True,exist_ok=False); (out/'images').mkdir(); (out/'overlays').mkdir()
seq='studio/2023_01_19/S5/seq_4-7-0'
with zipfile.ZipFile(a.splits) as z:
 splits={s:set(z.read('UnrealEgoData_rw/'+s+'.txt').decode().splitlines()) for s in ['train','validation','test']}
assert seq in splits['validation'] and seq not in splits['train'] and seq not in splits['test']
npz=np.load(a.prepared,allow_pickle=False); arrays={k:npz[k] for k in ['source_sequence','frame_index','camera_side','image_path','keypoints2d']}; inds=np.flatnonzero(arrays['source_sequence']==seq); frame_ids=sorted(set(int(x) for x in arrays['frame_index'][inds])); selected=[frame_ids[i] for i in np.linspace(0,len(frame_ids)-1,50,dtype=int)]; wanted={}; prefix='UnrealEgoData_rw_train_val_rgb/'
for i in inds:
 frame=int(arrays['frame_index'][i]); side=str(arrays['camera_side'][i]); path=str(arrays['image_path'][i]).split(prefix,1)[1]
 if frame in selected: wanted[path]=int(i)
assert len(wanted)==100
names=['Neck','R_Shoulder','R_Elbow','R_Wrist','L_Shoulder','L_Elbow','L_Wrist','R_Hip','R_Knee','R_Ankle','R_Toe','L_Hip','L_Knee','L_Ankle','L_Toe'];records=[]
with tarfile.open(a.archive,'r|gz') as tf:
 for member in tf:
  if member.name not in wanted or not member.isfile(): continue
  idx=wanted[member.name]; b=tf.extractfile(member).read(); im=cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_COLOR)
  if im is None: raise ValueError('image decode failed')
  h,w=im.shape[:2]; points=np.asarray(arrays['keypoints2d'][idx]); frame=int(arrays['frame_index'][idx]); side=str(arrays['camera_side'][idx]); ident=f'{frame:05d}_{side}'; (out/'images'/f'{ident}.png').write_bytes(b); overlay=im.copy(); vals=[]
  for j,name in enumerate(names):
   x,y=map(float,points[j,:2]); inside=bool(np.isfinite([x,y]).all() and 0<=x<w and 0<=y<h); vals.append(dict(name=name,xy=[x,y],inside=inside,visibility='unknown',validity='source supplies coordinates; not independently checked'))
   if inside:
    pt=(round(x),round(y)); cv2.circle(overlay,pt,5,(30,220,30) if name.startswith('L_') else (30,150,255),-1); cv2.putText(overlay,name,(pt[0]+5,pt[1]-5),cv2.FONT_HERSHEY_SIMPLEX,.38,(255,255,255),1,cv2.LINE_AA)
  cv2.imwrite(str(out/'overlays'/f'{ident}.jpg'),overlay)
  records.append(dict(id=ident,sequence=seq,subject='S5',frame=frame,view=side,split='validation',pair_id=f'{seq}/{frame}',size=[w,h],image='images/'+ident+'.png',overlay='overlays/'+ident+'.jpg',image_sha256=hashlib.sha256(b).hexdigest(),points=vals,nose=dict(valid=False,visibility='unknown',reason='not supplied'),source_image=member.name))
  if len(records)==100: break
records.sort(key=lambda x:(x['frame'],x['view'])); ids={x['id'] for x in records}; paired=sum({f'{f:05d}_left',f'{f:05d}_right'} <= ids for f in selected)
with (out/'manifest.json').open('x') as f: json.dump(dict(source='official RW prepared 2D coordinates, not manually verified',records=records),f,indent=2)
summary=dict(command='python runs/2026-11-16_sapiens-unrealego-audit/code/sample_validation.py --archive <RGB_ARCHIVE_PART> --prepared <RW_PREPARED_NPZ> --splits <OFFICIAL_SPLITS_ZIP> --out <RUN_OUTPUT> --evidence <EVIDENCE_JSON>',exit_status=0,sequence=seq,split='validation',sample_count=len(records),pairs=paired,selected_frames=selected,source_joint_order=names,nose_present=False,visibility='unknown; prepared input assigns 1 to all joints without visibility check',dimensions=sorted({tuple(x['size']) for x in records}),inside_counts={name:sum(p['inside'] for x in records for p in x['points'] if p['name']==name) for name in names},raw_images_and_full_coordinates='local output only',manual_visual_review='not yet completed',partial_archive_stop='stopped after all selected images; full archive completeness not verified')
Path(a.evidence).write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))
