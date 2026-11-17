"""Materialize only approved selected RGB/labels into a new repository output."""
import collections
import hashlib
import io
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]
BACKEND = ROOT.parent/'EgoSMPLX'
OUT = ROOT/'outputs/rw_a20000_dataset_20261117'


def read(p):
    return json.loads(p.read_text())


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, v):
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    start=time.time()
    cfg=read(RUN/'config.json')
    selections={'train':ROOT/'outputs/rw_a20000_selection_20261117/train_selection.json',
                'validation':ROOT/'outputs/rw_a_budget_20261117/validation_selection.json'}
    for split,p in selections.items():
        assert digest(p)==cfg[('training' if split=='train' else 'validation')+'_selection_sha256']
    OUT.mkdir(exist_ok=False);(OUT/'images').mkdir();(OUT/'overlays').mkdir()
    cache=BACKEND/'data/image_cache/egosmplx_training_safe_20260726'
    shards=read(cache/'manifest.json')['shards'];assert not (cache/'.building').exists()
    source=BACKEND/'data/UnrealEgo_RW_Train/prepared/samples.npz'
    prior=read(ROOT/'runs/2026-11-17_rw-a-scale-prep/evidence/inventory.json')
    assert digest(source)==prior['samples_sha256']
    with np.load(source,allow_pickle=False) as z:
        xy=z['keypoints2d'];sequences=z['source_sequence'];frames=z['frame_index'];sides=z['camera_side'];paths=z['image_path']
    mapping=[m for m in read(ROOT/'runs/2026-11-16_sapiens-unrealego-audit/evidence/audit.json')['joint_mapping'] if m['rw_index'] is not None]
    connections={};allrows={};summary={};probes=[]
    for split,sel in selections.items():
        records=read(sel)['records'];assert len(records)==cfg[split+'_images']
        rows=[];supervision=collections.Counter()
        for n,r in enumerate(records):
            i=r['source_index'];key=r['original_cache_key']
            assert str(paths[i])==key and str(sequences[i])==r['sequence'] and int(frames[i])==r['frame'] and str(sides[i])==r['view']
            k=int(hashlib.md5(key.encode()).hexdigest(),16)%len(shards)
            if k not in connections:connections[k]=sqlite3.connect((cache/shards[k]).as_uri()+'?mode=ro',uri=True)
            found=connections[k].execute('SELECT data FROM images WHERE key=?',(key,)).fetchone();assert found is not None
            raw=found[0]
            with Image.open(io.BytesIO(raw)) as im:
                im.load();w,h=im.size;assert im.mode=='RGB'
            points=np.zeros((308,2),np.float32);mask=np.zeros(308,np.int32)
            for m in mapping:
                j=m['sapiens308_id'];v=xy[i,m['rw_index'],:2]
                assert np.isfinite(v).all(),(split,i,m['name'])
                points[j]=v;mask[j]=int(0<=v[0]<w and 0<=v[1]<h);supervision[m['name']]+=int(mask[j])
            assert mask.sum()>0 and mask[0]==0
            ident=r['sequence'].replace('/','_')+'_%06d_%s'%(r['frame'],r['view'])
            name='images/'+ident+'.png';p=OUT/name;assert not p.exists();p.write_bytes(raw)
            rows.append(dict(id=ident,image=name,image_sha256=hashlib.sha256(raw).hexdigest(),
                sequence=r['sequence'],subject=r['subject'],frame=r['frame'],view=r['view'],pair_id=r['pair_id'],
                split=split,size=[w,h],keypoints308_xy=points.tolist(),keypoints308_supervision_mask=mask.tolist(),
                visibility='unknown; finite in-frame projected anatomical labels, not visibility',source_index=i))
            if (n+1)%500==0: print(split,n+1,'decoded',round(time.time()-start,1),'seconds',flush=True)
        write(OUT/(split+'.json'),dict(records=rows));allrows[split]=rows
        summary[split]=dict(images=len(rows),pairs=len({r['pair_id'] for r in rows}),
            subjects=dict(collections.Counter(r['subject'] for r in rows)),
            sequences=len({r['sequence'] for r in rows}),supervised_counts=dict(supervision),
            manifest_sha256=digest(OUT/(split+'.json')),selection_sha256=digest(sel))
        # Two poses per subject: first pair and most widely spread projected joints.
        for subject in sorted({r['subject'] for r in rows}):
            subjectrows=[r for r in rows if r['subject']==subject]
            def spread(r):
                pts=np.array(r['keypoints308_xy'])[np.array(r['keypoints308_supervision_mask'],bool)]
                return float(np.ptp(pts,axis=0).sum())
            chosen=list(dict.fromkeys([subjectrows[0]['pair_id'],max(subjectrows,key=spread)['pair_id']]))
            sheet=Image.new('RGB',(1024,560),(30,30,30));d=ImageDraw.Draw(sheet)
            d.text((8,4),split+' '+subject+' | L/R labels; projection may be occluded',fill='white')
            for row_index,pair in enumerate(chosen):
                for col,eye in enumerate(['left','right']):
                    r=next(r for r in subjectrows if r['pair_id']==pair and r['view']==eye)
                    with Image.open(OUT/r['image']) as im:
                        im=im.convert('RGB');draw=ImageDraw.Draw(im)
                        for m in mapping:
                            j=m['sapiens308_id']
                            if not r['keypoints308_supervision_mask'][j]:continue
                            x,y=r['keypoints308_xy'][j];color=(30,255,50) if m['name'].startswith('left') else (255,140,20)
                            draw.ellipse((x-5,y-5,x+5,y+5),fill=color)
                            draw.text((x+6,y),m['name'].replace('left_','L_').replace('right_','R_'),fill=color)
                        im.thumbnail((500,240));sheet.paste(im,(col*512,row_index*270+30))
                    d.text((col*512+5,row_index*270+18),r['view']+' '+r['sequence'].split('/')[-1]+' frame '+str(r['frame']),fill='white')
                    probes.append({k:r[k] for k in ['split','subject','sequence','frame','view','image_sha256']})
            sheet.save(OUT/'overlays'/(split+'_'+subject+'.jpg'))
    for c in connections.values():c.close()
    assert not {r['image_sha256'] for r in allrows['train']} & {r['image_sha256'] for r in allrows['validation']}
    assert not {r['sequence'] for r in allrows['train']} & {r['sequence'] for r in allrows['validation']}
    for split,rows in allrows.items():
        pairs=collections.defaultdict(list)
        for r in rows:pairs[r['pair_id']].append(r['view'])
        assert all(sorted(v)==['left','right'] for v in pairs.values())
    write(RUN/'evidence/dataset.json',dict(command='python code/prepare.py',exit_status=0,
        source_npz_sha256=digest(source),cache_manifest_sha256=digest(cache/'manifest.json'),
        config_sha256=digest(RUN/'config.json'),summary=summary,decode_failures=0,
        cross_split_sequence_overlap=0,cross_split_image_hash_overlap=0,
        overlays=probes,elapsed_seconds=time.time()-start,
        scope='All selected images decoded and hashed; finite projected labels checked; overlays require human/agent inspection; no visibility certification'))
    print('DONE',json.dumps(summary),flush=True)


if __name__=='__main__':main()
