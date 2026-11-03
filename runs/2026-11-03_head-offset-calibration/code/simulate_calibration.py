"""Produce a single simulated calibration from training DFM head joints only."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--observations',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
ref=np.load(a.reference,allow_pickle=False);obs=np.load(a.observations,allow_pickle=False);idx={int(f):i for i,f in enumerate(ref['frames'])};frames=obs['frames'][:10];indices=[idx[int(f)] for f in frames];head=ref['cam2_body_xyz'][indices,15];valid=ref['cam2_body_valid'][indices,15]&np.isfinite(head).all(1);assert valid.any();center=np.median(head[valid],axis=0)
with a.output.open('x') as f:json.dump({'description':'One simulated fixed head center; derived from training DFM only, not a physical calibration','training_frame_ids':frames.tolist(),'valid_training_count':int(valid.sum()),'head_center_cam2_m':center.tolist()},f,indent=2);f.write('\n')
