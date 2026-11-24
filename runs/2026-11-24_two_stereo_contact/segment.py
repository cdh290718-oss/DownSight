"""Segment the four hash-verified original frames using the existing local installation."""
import json, hashlib, os, subprocess
from pathlib import Path
root=Path.cwd()
out=root/'outputs/two_stereo_contact_20261124'
config=json.loads((root/'configs/pipeline.head01.developer.json').read_text())
paths=config['paths']
records=json.loads((root/'runs/2026-11-23_fixed_floor24/evidence/gallery.json').read_text())['rows']
frames=[]
for pair in ['pair055','pair068']:
    record=next(row for row in records if row['pair']==pair)
    for side,expected in zip(['left','right'],record['image_sha256']):
        image=root/'outputs/fixed_floor24_20261123/gallery'/pair/(side+'.png')
        assert hashlib.sha256(image.read_bytes()).hexdigest()==expected
        frames.append(dict(id=pair+'_'+side,camera=side,image=str(image)))
manifest=out/'images.json'
manifest.write_text(json.dumps(dict(frames=frames)))
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(out/'runtime/mpl'),
         XDG_CACHE_HOME=str(out/'runtime/cache'),TMPDIR=str(out/'runtime/tmp'),
         CUDA_CACHE_PATH=str(out/'runtime/cuda'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
for name in ['mpl','cache','tmp','cuda']:
    (out/'runtime'/name).mkdir(parents=True,exist_ok=True)
subprocess.run([paths['pose_python'],'-B','egosmplx_pipeline/sapiens_seg.py','--repo-root',paths['sapiens_repo'],
    '--dependencies',paths['sapiens_dependencies'],'--checkpoint',paths['seg_checkpoint'],'--checkpoint-sha256',
    config['checkpoint_sha256']['seg'],'--manifest',str(manifest),'--output-dir',str(out/'seg'),'--device','cuda'],env=env,check=True)
