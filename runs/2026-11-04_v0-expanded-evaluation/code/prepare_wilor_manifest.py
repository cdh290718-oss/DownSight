"""Stage unique symlinks to existing input images; create a WiLoR manifest."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--images',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);rows=[]
for cam in (2,3):
 for source in sorted((a.images/f'cam{cam}').glob('*.jpg')):
  target=a.output/f'gen_{source.stem}_cam{cam}.jpg';target.symlink_to(source.resolve());rows.append({'id':target.stem,'image':str(target.absolute())})
assert len(rows)==360
(a.output/'manifest.json').write_text(json.dumps({'frames':rows},indent=2)+'\n')
