"""Render full-body front and side views of the verified output-time foot reset."""
from pathlib import Path
import subprocess
import sys
import cv2
import numpy as np
import torch

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
sys.path.insert(0, str(RUN.parent/'2026-11-20_rw20000_floor12'))
from render_gallery import g, render_views


def main():
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    model, _ = g.common.setup_model()
    layer = model.layer['neutral'].cpu().eval()
    faces = np.asarray(model.face, np.int32)
    floor_path = RUN.parent/'2026-11-23_fixed_floor24/evidence/gallery.json'
    reset_path = RUN.parent/'2026-11-25_floor24_foot_lock/evidence/results.json'
    floor_rows = g.read(floor_path)['rows']
    reset_rows = g.read(reset_path)['rows']
    assert len(floor_rows)==len(reset_rows)==24
    assert {r['pair'] for r in floor_rows}==set(reset_rows)
    page = ['# 脚部复位后的 24 对全身渲染','', '左为正面，右为侧面。','']
    rows = {}
    for frame in floor_rows:
        pair = frame['pair']
        archived = reset_rows[pair]
        assert archived['split']==frame['split']
        test = frame['split']=='test'
        source = 'frozen100_fingers' if test else 'validation200_fingers'
        folder = ROOT/'outputs'/(source+'_20261125')/'poses'
        pose_path = folder/(pair+'.npz')
        assert g.sha(pose_path)==archived['prior_sha256']
        v3_path = ROOT/'outputs/frozen100_outline_20261125/poses'/(pair+'.npz') if test else pose_path
        with np.load(pose_path) as pose, np.load(v3_path) as original:
            x = pose['fixed_full'].copy()
            indices = [6,7,9,10]
            x[indices] = original['v3'][indices]
            assert not x[indices].any() and not x[23:].any()
            assert np.array_equal(np.delete(x,indices,axis=0), np.delete(pose['fixed_full'],indices,axis=0))
            body = dict(global_orient=pose['global0'], body_pose=x[:21], transl=x[22], betas=np.zeros(10),
                        left_hand_pose=x[23:38], right_hand_pose=x[38:])
        vertices, _, _, _ = g.geometry(layer,body,x[21],False,None)
        assert np.isfinite(vertices).all()
        views = list(render_views(vertices,faces,np.array(frame['ground']['up']),np.array(frame['front_forward_left_cv'])))
        assert len(views)==2 and all(im.shape==(480,350,3) and (im[:,:,0]<235).sum()>300 for im in views)
        image = RUN/'assets'/(pair+'.jpg')
        g.save(image,np.hstack(views))
        assert cv2.imread(str(image)).shape==(480,700,3)
        path = image.relative_to(RUN)
        page.extend([f"### {frame['subject']} / {pair}",'',f'[![{pair} 全身正面与侧面]({path})]({path})',''])
        rows[pair] = dict(split=frame['split'],pose_sha256=g.sha(pose_path),v3_pose_sha256=g.sha(v3_path),
                          preview=str(path),preview_sha256=g.sha(image))
    (RUN/'gallery.md').write_text('\n'.join(page))
    g.write(RUN/'evidence/render.json',dict(method='fixed_full / output-time reset',rows=rows,
            base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),dirty_worktree=True,
            input_sha256={str(p.relative_to(ROOT)):g.sha(p) for p in (reset_path,floor_path)},
            views_per_pair=2,command='bash <RUN>/run.sh',exit_status=0))


if __name__=='__main__':
    main()
