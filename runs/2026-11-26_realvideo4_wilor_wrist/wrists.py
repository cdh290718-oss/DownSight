"""Recover the unchanged metric SMPL-X wrist centers from the common start."""
import numpy as np
import torch
from context_run import RUN, OUT, PREVIOUS, ref, palm, read, write, sha, SIDES
from smplx.joint_names import JOINT_NAMES

def main():
    model,_ = ref.prev.ref.common.setup_model()
    layer = model.layer['neutral'].eval()
    records = {}
    indices = [JOINT_NAMES.index(side+'_wrist') for side in SIDES]
    for row in read(ref.OLD/'fit/0/prepared.json')['items']:
        key = row['pair']
        baseline = read(PREVIOUS/'evidence'/(key+'.json'))
        path = palm.OUT/'poses'/(key+'.npz')
        assert sha(path) == baseline['start_sha256']
        pack = dict(np.load(path))
        x = torch.from_numpy(pack['locked'])
        with torch.no_grad():
            j = layer(global_orient=torch.from_numpy(pack['global0']).reshape(1,3),
                      body_pose=x[:21].reshape(1,63),betas=torch.from_numpy(pack['betas']).reshape(1,10),
                       jaw_pose=x.new_zeros((1,3)),leye_pose=x.new_zeros((1,3)),reye_pose=x.new_zeros((1,3)),expression=x.new_zeros((1,10)),
                      left_hand_pose=x[23:38].reshape(1,45),right_hand_pose=x[38:].reshape(1,45)).joints[0].numpy()
        world = (j[indices]-j[:1])@palm.rot(x[21].numpy()).T+j[:1]+x[22].numpy()
        for i,(eye,camera) in enumerate(zip(SIDES,ref.cameras())):
            T = np.asarray(row['inputs']['views'][eye]['camera']['left_to_view'])
            xyz = world@T[:3,:3].T+T[:3,3]
            uv = palm.projection(xyz,read(camera))
            for n,side in enumerate(SIDES):
                cached = baseline['results']['start']['hands'][side]['uv'][i][0]
                error = float(np.linalg.norm(uv[n]-cached))
                assert error < .001
                records[key+'_'+eye+'_'+side[0].upper()] = dict(xyz_m=xyz[n].tolist(),uv=uv[n].tolist(),replay_px=error)
    write(RUN/'evidence/wrists.json',records)

if __name__=='__main__':
    main()
