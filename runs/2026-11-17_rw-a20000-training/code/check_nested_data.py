"""Check nested labels, fixed validation/test and heldout isolation before training."""
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw

RUN=Path(__file__).resolve().parents[1];ROOT=RUN.parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    out=ROOT/'outputs/rw_a20000_dataset_20261117'
    previous=ROOT/'outputs/rw_a5000_dataset_20261117'
    rows=read(out/'train.json')['records'];old=read(previous/'train.json')['records']
    byid={r['id']:r for r in rows};assert len(byid)==20000
    assert len(old)==5000 and all(r==byid[r['id']] for r in old)
    assert sha(out/'validation.json')==sha(previous/'validation.json')
    test=ROOT/'outputs/rw_a5000_evaluation_20261117/test_manifest.json'
    same=ROOT/'outputs/sapiens_s6_heldout_20261117/manifest.json'
    frozen=read(ROOT/'runs/2026-11-17_rw-a5000-evaluation/evidence/2d_summary.json')['manifest_sha256']
    assert sha(test)==frozen['test_available'] and sha(same)==frozen['same_person']
    sequences={r['sequence'] for r in rows};hashes={r['image_sha256'] for r in rows}
    for p in [test,same]:
        rr=read(p)['records']
        assert not sequences & {r['sequence'] for r in rr}
        assert not hashes & {r['image_sha256'] for r in rr}
        assert all(sha(p.parent/r['image'])==r['image_sha256'] for r in rr)
    oldids={r['id'] for r in old}
    mapping=[m for m in read(ROOT/'runs/2026-11-16_sapiens-unrealego-audit/evidence/audit.json')['joint_mapping'] if m['rw_index'] is not None]
    probes=[]
    for subject in sorted({r['subject'] for r in rows}):
        first=next(r for r in rows if r['subject']==subject and r['id'] not in oldids)
        chosen=[r for r in rows if r['pair_id']==first['pair_id']];assert len(chosen)==2
        sheet=Image.new('RGB',(1024,540),(20,20,20));draw=ImageDraw.Draw(sheet)
        for col,r in enumerate(chosen):
            with Image.open(out/r['image']) as im:
                im=im.convert('RGB');d=ImageDraw.Draw(im)
                for m in mapping:
                    j=m['sapiens308_id']
                    if not r['keypoints308_supervision_mask'][j]:continue
                    x,y=r['keypoints308_xy'][j];color='lime' if m['name'].startswith('left') else 'orange'
                    d.ellipse((x-5,y-5,x+5,y+5),fill=color);d.text((x+6,y),m['name'],fill=color)
                im.thumbnail((500,500));sheet.paste(im,(col*512,30))
            draw.text((col*512+5,5),subject+' '+r['view']+' frame '+str(r['frame']),fill='white')
            probes.append({k:r[k] for k in ['id','subject','image_sha256']})
        sheet.save(out/'overlays'/('new_'+subject+'.jpg'))
    result=dict(command='python code/check_nested_data.py',exit_status=0,nested_old_images_unchanged=5000,
        validation_manifest_identical=True,validation_sha256=sha(out/'validation.json'),
        test_manifest_unchanged=True,test_sha256=sha(test),test_images=46,
        same_person_manifest_unchanged=True,same_person_sha256=sha(same),
        train_heldout_sequence_and_image_hash_overlap=0,new_image_overlay_probes=probes)
    (RUN/'evidence/nested_data.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Nested data and frozen evaluation checks passed; new-image overlays ready.')

if __name__=='__main__':main()
