"""Reconstruct archived v4 head surface, native joint and named mouth landmarks."""
import sys
from pathlib import Path
import numpy as np
import torch
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
OUT=ROOT/'outputs/realvideo4_head_mount_20261126'
V4=RUN.parent/'2026-11-25_realvideo4_dv4'
sys.path.insert(0,str(V4))
import gallery
from smplx.joint_names import JOINT_NAMES

def main():
    torch.set_num_threads(1)
    model,_=gallery.v3.ref.common.setup_model()
    layer=model.layer['neutral'].eval()
    faces=np.array(model.face)
    region=layer.lbs_weights.argmax(1).numpy()
    # Include face, jaw and eyes, exclude neck and hair/clothing.
    use=np.isin(region,[15,22,23,24])[faces].all(1)
    names=[n for n in JOINT_NAMES if 'mouth' in n or 'lip' in n]
    refitted = sys.argv[1]=='refitted'
    base = OUT if refitted else gallery.OUT
    receipt = OUT if refitted else V4
    dest = OUT/'refitted' if refitted else OUT
    dest.mkdir(exist_ok=True)
    for k in sys.argv[2:] if refitted else sys.argv[1:]:
        src=base/'poses'/(k+'.npz')
        assert gallery.sha(src)==gallery.read(receipt/'evidence'/(k+'.json'))['pose_hash']
        with np.load(src) as p:
            x=p['locked']
            params=dict(global_orient=p['global0'],body_pose=x[:21],betas=p['betas'],transl=x[22],
                        left_hand_pose=x[23:38],right_hand_pose=x[38:])
        v,j,_,_=gallery.geometry(layer,params,x[21],False,None)
        np.savez(dest/(k+'.npz'),body_joints=j[:22],v=v,f=faces,head_faces=use,head=j[15],names=names,lm=j[[JOINT_NAMES.index(n) for n in names]])
if __name__=='__main__':main()
