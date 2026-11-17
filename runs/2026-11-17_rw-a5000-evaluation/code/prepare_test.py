"""Recover only frozen test candidates from a read-only partial local archive."""
import hashlib
import io
import json
import tarfile
import time
from pathlib import Path
import numpy as np
from PIL import Image

RUN=Path(__file__).resolve().parents[1];ROOT=RUN.parents[1]
OUT=ROOT/'outputs/rw_a5000_evaluation_20261117';BACK=ROOT.parent/'EgoSMPLX'
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
    start=time.time();OUT.mkdir(exist_ok=False);(OUT/'test_images').mkdir();(OUT/'tmp').mkdir();(OUT/'cache').mkdir()
    selection=ROOT/'outputs/rw_a_budget_20261117/test_selection.json'
    frozen=json.loads((ROOT/'runs/2026-11-17_rw-a-budget/evidence/selection.json').read_text())
    assert hashlib.sha256(selection.read_bytes()).hexdigest()==frozen['selection']['test']['sha256']
    rows=json.loads(selection.read_text())['records']
    wanted={r['original_cache_key'].split('UnrealEgoData_rw_test_rgb/',1)[1]:r for r in rows}
    source=BACK/'data/UnrealEgo_RW_Test/prepared/samples.npz'
    with np.load(source,allow_pickle=False) as z:
        xy=z['keypoints2d'];seq=z['source_sequence'];frame=z['frame_index'];side=z['camera_side']
    mapping=[m for m in json.loads((ROOT/'runs/2026-11-16_sapiens-unrealego-audit/evidence/audit.json').read_text())['joint_mapping'] if m['rw_index'] is not None]
    archive=BACK/'.unrealego_rw_repair_cache/UnrealEgoData_rw_test_rgb/studio.tar.gz000'
    found={};error=None;members=0;counts={}
    try:
        with tarfile.open(archive,'r|gz') as tar:
            for member in tar:
                members+=1
                if '/fisheye_final_image/' in member.name and member.isfile():
                    subject=member.name.split('/')[2];counts[subject]=counts.get(subject,0)+1
                if members%2000==0:print('archive members',members,'selected images',len(found),flush=True)
                if member.name not in wanted:continue
                r=wanted[member.name];i=r['source_index']
                assert str(seq[i])==r['sequence'] and int(frame[i])==r['frame'] and str(side[i])==r['view']
                raw=tar.extractfile(member).read()
                with Image.open(io.BytesIO(raw)) as im:im.load();w,h=im.size
                pts=np.zeros((308,2),np.float32);mask=np.zeros(308,np.int32)
                for m in mapping:
                    p=xy[i,m['rw_index'],:2];assert np.isfinite(p).all();j=m['sapiens308_id'];pts[j]=p
                    mask[j]=int(0<=p[0]<w and 0<=p[1]<h)
                name=r['sequence'].replace('/','_')+'_%06d_%s.png'%(r['frame'],r['view'])
                target=OUT/'test_images'/name;assert not target.exists();target.write_bytes(raw)
                found[member.name]=dict(id=name,image='test_images/'+name,image_sha256=hashlib.sha256(raw).hexdigest(),
                    sequence=r['sequence'],subject=r['subject'],frame=r['frame'],view=r['view'],pair_id=r['pair_id'],
                    split='test',size=[w,h],keypoints308_xy=pts.tolist(),keypoints308_supervision_mask=mask.tolist())
    except (tarfile.TarError,EOFError,OSError) as e:error=type(e).__name__+': '+str(e)
    pairs={}
    for r in found.values():pairs.setdefault(r['pair_id'],[]).append(r)
    complete=[r for vv in pairs.values() if {r['view'] for r in vv}=={'left','right'} for r in vv]
    complete.sort(key=lambda r:r['id'])
    write(OUT/'test_manifest.json',dict(records=complete))
    write(RUN/'evidence/test_availability.json',dict(command='python code/prepare_test.py',exit_status=0,
        requested_images=len(rows),recovered_candidate_images=len(found),paired_evaluated_images=len(complete),
        paired_evaluated_pairs=len(complete)//2,subjects=sorted({r['subject'] for r in complete}),
        sequences=sorted({r['sequence'] for r in complete}),archive_members_read=members,
        archived_image_headers_by_subject=counts,archive_bytes=archive.stat().st_size,
        archive_error=error,archive_complete=error is None,
        source_npz_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256((OUT/'test_manifest.json').read_bytes()).hexdigest(),
        frozen_selection_sha256=hashlib.sha256(selection.read_bytes()).hexdigest(),
        elapsed_seconds=time.time()-start,
        scope='Availability-selected subset of previously frozen candidates; not full 200-image test; old S9 diagnostic sequences excluded; historical use outside this round not exhaustively audited.'))
    print('DONE',len(complete),'paired images; error:',error,flush=True)
if __name__=='__main__':main()
