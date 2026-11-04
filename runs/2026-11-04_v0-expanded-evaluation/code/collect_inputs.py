"""Hash input resources without serializing workstation paths or private data."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--images',type=Path,required=True);p.add_argument('--detections',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True);p.add_argument('--code',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
files=[p for cam in ('cam2','cam3') for p in sorted((a.images/cam).glob('*.jpg'))];h=hashlib.sha256()
for f in files:h.update((f.parent.name+'/'+f.name+' '+sha(f)+'\n').encode())
out={'baseline_commit':'e6021e5','resources':{'native_images':{'count':len(files),'sha256_of_sorted_relative_name_and_digest':h.hexdigest()},'sapiens_detections':sha(a.detections),'dfm_evaluation_reference':sha(a.reference),'image_estimated_stereo_calibration':sha(a.calibration)},'run_code_sha256':{p.name:sha(p) for p in sorted(a.code.iterdir()) if p.suffix in ('.py','.sh')},'checkpoint_sha256':{'egosmplx':'c7d30f04d13ed80f1941226458a43dfd50fc0ab0e5902701b6c9895e1f92a3b3','sapiens_segmentation':'4b73c44963b377e93fcb4c4053f72a189836a22d05e12c30383046b9cd3c5bd4','wilor':'22fb2660f6ac3f611919c9abd5c412372a212309d04fcf3697b291da25a29456','wilor_detector':'5ef3df44e42d2db52d4ffe91f83a22ce9925e2acc9abebf453f2c5d22e380033'},'checkpoint_hash_source':'existing recorded project configuration; segmentation also rehashed during inference','asset_access':'local restricted assets; not included in repository','simulated_calibration':'DFM head of ten prior training frames, explicitly disclosed exception'}
a.output.write_text(json.dumps(out,indent=2)+'\n')
