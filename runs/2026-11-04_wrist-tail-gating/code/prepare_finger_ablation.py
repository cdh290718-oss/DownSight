"""Post-hoc diagnostic: isolate 2D and stereo fingertip contributions."""
import argparse,json,os
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);a=p.parse_args();source=a.inputs/'baseline';d=np.load(source/'observations.npz',allow_pickle=False);ids=[i for i,n in enumerate(d['names'].tolist()) if 'wrist' not in n and 'elbow' not in n]
for mode in ('finger_2d_only','finger_3d_only'):
 x={k:d[k].copy() for k in d.files}
 if mode=='finger_2d_only':x['tri_valid'][:,ids]=False
 else:x['valid'][:,:,ids]=False
 target=a.inputs/mode;target.mkdir(exist_ok=False);(target/'camera_calibration.json').write_text((source/'camera_calibration.json').read_text());os.symlink((source/'initial').resolve(),target/'initial',target_is_directory=True)
 with (target/'observations.npz').open('xb') as f:np.savez_compressed(f,**x)
 print(mode,'test 2d finger',int(x['valid'][10:,0,ids].sum()),'test stereo finger',int(x['tri_valid'][10:,ids].sum()))
