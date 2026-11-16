import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--mapping',required=True);p.add_argument('--output',required=True);a=p.parse_args();data=json.loads(Path(a.input).read_text());mapping=json.loads(Path(a.mapping).read_text())['joint_mapping']; rows=[]
for r in data['records']:
 points={p['name']:p for p in r['points']}; xy=[[0.0,0.0] for _ in range(308)]; supervised=[0]*308; present=[0]*308
 for m in mapping:
  if m['rw_name']:
   pt=points[m['rw_name']]; xy[m['sapiens308_id']]=pt['xy']; present[m['sapiens308_id']]=int(pt['inside'])
 rows.append(dict(id=r['id'],image=r['image'],image_sha256=r['image_sha256'],dataset='UnrealEgo-RW',subject=r['subject'],sequence=r['sequence'],frame=r['frame'],view=r['view'],pair_id=r['pair_id'],split=r['split'],size=r['size'],crop_xyxy=[0,0,*r['size']],keypoints308_xy=xy,keypoints308_candidate_in_image=present,keypoints308_supervision_mask=supervised,visibility='unknown',source_image=r['source_image'],source_2d='RW prepared samples.npz; original pose JSON unavailable',source_calibration='RW prepared samples.npz calib_board_pose; original source unavailable',original_points=r['points']))
Path(a.output).open('x').write(json.dumps(dict(status='inspection only; all masks zero until coordinate, semantics and visibility review; validation samples must not train',records=rows),indent=2)+'\n')
print('records',len(rows),'all supervision masks zero')
