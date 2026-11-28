import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/fixed_skeleton_shape_20261128'
sys.path.insert(0, str(ROOT / 'runs/2026-11-25_frozen100_fingers'))
sys.path.insert(0, str(ROOT / 'runs/2026-11-20_rw20000_floor12'))
from geometry_inputs import GridEye, prev, search
import compare
from common_run import groups, rotation
from render_gallery import g, render_views
from smplx.lbs import batch_rodrigues, batch_rigid_transform

def main():
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    model, camera = g.common.setup_model()
    layer = model.layer['neutral'].cpu().eval()
    faces = np.asarray(model.face, np.int32)
    labels = groups(layer, faces)
    gallery = g.read(ROOT / 'runs/2026-11-23_fixed_floor24/evidence/gallery.json')['rows']
    basis = layer.shapedirs[:, :, :10].detach()
    weights = layer.lbs_weights.detach()
    rest_joints = (layer.J_regressor @ layer.v_template).detach()
    output = []
    for entry in gallery:
        pair, split = entry['pair'], entry['split']
        folder = prev.base.OUT if split == 'test' else prev.VAL
        compare.OUT = ROOT / ('outputs/frozen100_outline_20261125' if split == 'test' else 'outputs/validation200_fingers_20261125')
        choices = g.read(search.OUT / (split + '_choices.json'))
        record = choices[pair]
        with np.load(search.OUT / (split + '_bank.npz')) as bank:
            idx = record['offset'] + record['fixed']
            param = {k: bank[k][idx].copy() for k in search.FIELDS}
        item = g.read(folder / 'prepared.json')['items'][record['index']]
        cameras = [camera(path) for path in prev.ref.camera_files()]
        eyes = [GridEye(item, side, cam, g.read(path)['input_adapter'], labels, faces)
                for side, cam, path in zip(('left','right'), cameras, prev.ref.camera_files())]
        body = dict(global_orient=param['global0'], body_pose=param['body_pose'],
                    transl=param['translation'], betas=param['betas'])
        vertices, joints, _, _ = g.geometry(layer, body, param['delta'], False, None)
        zero = torch.zeros(1, 10)
        with torch.no_grad():
            result = layer(global_orient=torch.tensor(param['global0']).reshape(1,3),
                           body_pose=torch.tensor(param['body_pose']).reshape(1,63),
                           betas=zero, left_hand_pose=torch.zeros(1,45), right_hand_pose=torch.zeros(1,45),
                           jaw_pose=torch.zeros(1,3), leye_pose=torch.zeros(1,3), reye_pose=torch.zeros(1,3),
                           expression=zero, return_full_pose=True)
            rotations = batch_rodrigues(result.full_pose.reshape(-1,3)).reshape(1,-1,3,3)
            _, transforms = batch_rigid_transform(rotations, rest_joints.unsqueeze(0), layer.parents)
            vertex_rotations = torch.einsum('vj,jab->vab', weights, transforms[0, :, :3, :3])
            delta_rot = rotation(torch.tensor(param['delta'], dtype=torch.float32).reshape(3)).detach().float()
            shape_basis = torch.einsum('ab,vbc,vck->vak', delta_rot, vertex_rotations, basis)
        base = torch.tensor(vertices, dtype=torch.float32)
        coeff = torch.nn.Parameter(torch.zeros(10))
        opt = torch.optim.Adam([coeff], lr=0.05)
        best = None
        for step in range(31):
            opt.zero_grad()
            v = base + torch.einsum('vck,k->vc', shape_basis, coeff)
            loss = sum(eye.compare(v, True)[0] for eye in eyes) / 2
            objective = loss + 0.03 * coeff.square().mean()
            if best is None or objective.item() < best[0]:
                best = (objective.item(), coeff.detach().clone(), loss.item(), step)
            if step < 30:
                objective.backward()
                opt.step()
                with torch.no_grad():
                    coeff.clamp_(-3, 3)
        with torch.no_grad():
            fitted = (base + torch.einsum('vck,k->vc', shape_basis, best[1])).numpy()
            start_loss = sum(eye.compare(base)[0].item() for eye in eyes) / 2
            end_loss = sum(eye.compare(torch.tensor(fitted))[0].item() for eye in eyes) / 2
        dest = OUT / pair
        dest.mkdir(parents=True, exist_ok=False)
        obj = dest / (pair + '.obj')
        with obj.open('w') as file:
            np.savetxt(file, fitted, fmt='v %.9f %.9f %.9f')
            np.savetxt(file, faces + 1, fmt='f %d %d %d')
        loaded, loaded_faces = g.load_obj(obj)
        assert np.array_equal(faces, loaded_faces) and np.max(np.abs(loaded-fitted)) < 1e-8
        asset = RUN / 'assets' / pair
        asset.mkdir(parents=True, exist_ok=False)
        for side, image in zip(('front','side'), render_views(loaded, faces, np.asarray(entry['ground']['up']), np.asarray(entry['front_forward_left_cv']))):
            g.save(asset / (side + '.jpg'), image)
        truthpath = g.DATA / 'reference.json' if split == 'test' else ROOT/'outputs/stereo_wrist_candidates_20261123/validation_truth.json'
        target = np.asarray(g.read(truthpath)[pair]['gt'])[g.common.OBS]
        pred = joints[g.common.NATIVE_IDS][g.common.OBS]
        pa = float(np.linalg.norm(g.align(pred,target)-target, axis=1).mean()*1000)
        nonpa = float(np.linalg.norm(pred-target, axis=1).mean()*1000)
        row = dict(pair=pair, subject=entry['subject'], split=split, pa_mm=pa, nonpa_mm=nonpa,
                   contour_before_px=start_loss*12, contour_after_px=end_loss*12,
                   coefficients=best[1].tolist(), selected_step=best[3], joint_displacement_mm=0.0,
                   ground_accepted=entry['ground']['accepted'])
        output.append(row)
        (RUN/'evidence'/'progress.json').write_text(json.dumps(output, indent=2))
        print(pair, round(start_loss*12,2), round(end_loss*12,2), flush=True)
    (RUN/'evidence'/'results.json').write_text(json.dumps({'rows':output,'exit_status':0}, indent=2))
if __name__ == '__main__':
    main()
