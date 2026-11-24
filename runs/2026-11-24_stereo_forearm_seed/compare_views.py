"""Render the two locked predictions with identical floor-relative cameras and crop."""
import sys
import cv2
from forearm_seed import BRANCHES, FRAMES, OUT, ROOT, RUN, joint, np, prev, torch

sys.path.insert(0, str(ROOT/'runs/2026-11-20_rw20000_floor12'))
import render_gallery as render


def main():
    torch.set_num_threads(1)
    ev = prev.read(RUN/'evidence/selection.json')
    assert ev['checkpoints_sha256'] == prev.sha(OUT/'checkpoints.npz')
    archive = dict(np.load(joint.OUT/'validation_bank.npz'))
    choices = prev.read(joint.OUT/'validation_choices.json')
    pack = dict(np.load(OUT/'checkpoints.npz'))
    model = prev.ref.common.setup_model()[0]
    for j, frame in enumerate(FRAMES):
        r = choices[frame]
        baseline = (archive, r['offset'] + r['best'])
        winner = ev['frames'][frame]['winner']
        new = baseline if winner == 'archive' else (pack, j*len(BRANCHES)+BRANCHES.index(winner))
        meshes = []
        for data, i in (baseline, new):
            params = {k:data[field][i] for k,field in [('body_pose','body_pose'),('betas','betas'),
                                                        ('global_orient','global0'),('transl','translation')]}
            meshes.append(render.g.geometry(model.layer['neutral'],params,data['delta'][i],False,None)[0])
        ground = prev.read(ROOT/'outputs/fixed_floor24_20261123'/(frame+'.json'))
        up, forward = np.array(ground['ground']['up']), np.array(ground['front_forward_left_cv'])
        z = render.floor.unit(forward-up*(up@forward))
        x = render.floor.unit(np.cross(up,z))
        axes = [np.column_stack([x,up,z]), np.column_stack([z,up,-x])]
        bounds = [np.concatenate(meshes)@b for b in axes]
        scale = min(min(313/np.ptp(v[:,0]),438/np.ptp(v[:,1])) for v in bounds)
        panels = []
        for b, v, view in zip(axes, bounds, ('front','side')):
            for vertices, tag in zip(meshes, ('v3', 'selected')):
                img = render.g.free_render(vertices, np.asarray(model.face), b, (v.min(0)+v.max(0))/2, scale)
                img = cv2.copyMakeBorder(img,38,2,0,0,cv2.BORDER_CONSTANT,value=(245,245,245))
                cv2.putText(img,frame+' '+tag+' / '+view,(10,24),cv2.FONT_HERSHEY_SIMPLEX,
                            .63,(40,40,40),1,cv2.LINE_AA)
                panels.append(img)
        render.g.save(RUN/'assets'/(frame+'.jpg'), np.concatenate(panels,axis=1))


if __name__ == '__main__':
    main()
