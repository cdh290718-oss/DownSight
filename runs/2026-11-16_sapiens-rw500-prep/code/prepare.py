"""Extract selected paired RW images without modifying external archives."""
import argparse, hashlib, json, tarfile, zipfile
from pathlib import Path
import cv2
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
run=Path(__file__).resolve().parents[1]; old=run.parent/'2026-11-16_sapiens-unrealego-audit'; backend=a.workspace/'EgoSMPLX'; data=backend/'data/UnrealEgo_RW_Train'
index=json.loads((run/'evidence/archive_index.json').read_text()); mapping=json.loads((old/'evidence/audit.json').read_text())['joint_mapping']
# Exclude final sequence because the archive ends within it.
seqs=[r['sequence'] for r in index['sequences'] if r['split']=='train'][:-1]
with zipfile.ZipFile(data/'official_splits/UnrealEgoData_rw.zip') as z:
 splits={s:set(z.read('UnrealEgoData_rw/'+s+'.txt').decode().splitlines()) for s in ('train','validation','test')}
n=np.load(data/'prepared/samples.npz',allow_pickle=False); arrays={k:n[k] for k in ('source_sequence','frame_index','camera_side','image_path','keypoints2d')}
frames={s:sorted(set(map(int,arrays['frame_index'][arrays['source_sequence']==s]))) for s in seqs}
quota={s:0 for s in seqs}
while sum(quota.values())<250:
 for s in seqs:
  if quota[s]<len(frames[s]) and sum(quota.values())<250:quota[s]+=1
selected={s:set(np.array(frames[s])[np.linspace(0,len(frames[s])-1,quota[s],dtype=int)].tolist()) for s in seqs}
wanted={}
for i,s in enumerate(arrays['source_sequence']):
 if s in selected and int(arrays['frame_index'][i]) in selected[s]:
  assert s in splits['train'] and s not in splits['validation']|splits['test']
  wanted[str(arrays['image_path'][i]).split('UnrealEgoData_rw_train_val_rgb/',1)[1]]=i
assert len(wanted)==500
out=a.output;out.mkdir(exist_ok=False);(out/'images').mkdir();(out/'overlays').mkdir();records=[]
with tarfile.open(backend/'.unrealego_rw_repair_cache/UnrealEgoData_rw_train_val_rgb/studio.tar.gz000','r|gz') as tar:
 for member in tar:
  if member.name not in wanted:continue
  i=wanted[member.name];raw=tar.extractfile(member).read();im=cv2.imdecode(np.frombuffer(raw,np.uint8),1);assert im is not None
  h,w=im.shape[:2];s=str(arrays['source_sequence'][i]);frame=int(arrays['frame_index'][i]);side=str(arrays['camera_side'][i]);ident=s.replace('/','_')+f'_{frame:05d}_{side}'
  (out/'images'/f'{ident}.png').write_bytes(raw);xy=np.zeros((308,2),np.float32);mask=np.zeros(308,np.float32);overlay=im.copy()
  for m in mapping:
   if m['rw_index'] is None:continue
   v=arrays['keypoints2d'][i,m['rw_index'],:2];j=m['sapiens308_id'];xy[j]=v;mask[j]=int(np.isfinite(v).all() and 0<=v[0]<w and 0<=v[1]<h)
   if mask[j]:
    pt=tuple(np.round(v).astype(int));cv2.circle(overlay,pt,5,(0,220,0) if m['name'].startswith('left') else (0,140,255),-1);cv2.putText(overlay,m['name'],pt,cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),1)
  cv2.imwrite(str(out/'overlays'/f'{ident}.jpg'),overlay)
  records.append(dict(id=ident,image='images/'+ident+'.png',image_sha256=hashlib.sha256(raw).hexdigest(),sequence=s,subject=s.split('/')[2],frame=frame,view=side,pair_id=s+f'/{frame}',split='train',size=[w,h],crop_xyxy=[0,0,w,h],keypoints308_xy=xy.tolist(),keypoints308_supervision_mask=mask.tolist(),visibility='unknown; mask means finite in-frame projected annotation, not visible',source_image=member.name,source_index=i,original_points=arrays['keypoints2d'][i].tolist()))
  if len(records)==500:break
assert len(records)==500
records.sort(key=lambda r:r['id']);(out/'train.json').write_text(json.dumps(dict(status='candidate projected anatomical-joint supervision; review required',records=records),indent=2))
pairs={r['pair_id'] for r in records};assert len(pairs)==250
for pair in pairs:assert {r['view'] for r in records if r['pair_id']==pair}=={'left','right'}
# Fixed prior validation images stay in their original location and are never copied into train.
val=Path('outputs/sapiens_unrealego_audit_20261116/validation_s5_retry/candidate_308.json')
vr=json.loads(val.read_text())['records'];assert not {r['sequence'] for r in vr}&set(seqs)
summary=dict(command='prepare.py --workspace <WORKSPACE> --output <LOCAL_OUTPUT>/dataset',exit_status=0,train_images=len(records),train_pairs=len(pairs),subjects=sorted({r['subject'] for r in records}),sequence_images={s:sum(r['sequence']==s for r in records) for s in seqs},validation_images=len(vr),validation_subjects=sorted({r['subject'] for r in vr}),sequence_overlap=0,pair_overlap=0,nose_supervised=False,supervised_counts={m['name']:sum(r['keypoints308_supervision_mask'][m['sapiens308_id']] for r in records) for m in mapping},manifest_sha256=hashlib.sha256((out/'train.json').read_bytes()).hexdigest(),source='prepared samples.npz; original pose JSON unavailable',visibility='unknown',last_archive_sequence_excluded=True)
(run/'evidence/dataset.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
