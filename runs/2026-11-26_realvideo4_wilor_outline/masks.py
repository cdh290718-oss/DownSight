"""Render each monocular WiLoR MANO prediction at its native image placement."""
from pathlib import Path
import os
import sys
import subprocess
import json
import hashlib

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
OUT = ROOT / 'outputs/realvideo4_wilor_outline_20261126'
PREV = ROOT / 'outputs/realvideo4_palm_20261126'
OLD = ROOT / 'outputs/realvideo4_dv3_20261125'
read = lambda path: json.loads(path.read_text())
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()


def infer(cfg):
    import cv2
    import numpy as np
    import torch
    from egosmplx_pipeline import wilor_predict as w
    paths = cfg['paths']
    guide = read(PREV / 'guided.json')['selected']
    frames = read(RUN.parent / '2026-11-25_realvideo4_dv3/evidence/frames.json')
    assert sha(PREV / 'guided.json') == read(RUN.parent / '2026-11-26_realvideo4_palm/evidence/clip001.json')['guided_hash']
    assert sha(Path(paths['wilor_checkpoint'])) == cfg['checkpoint_sha256']['wilor']
    model, config = w.load_wilor(paths['wilor_checkpoint'], paths['wilor_config'])
    model.cuda().eval()
    faces = np.asarray(model.mano.faces)
    records = {}
    for row in frames:
        for eye in ('left', 'right'):
            name = row['pair'] + '_' + eye
            path = OLD / 'images' / (name + '.png')
            assert sha(path) == row['views'][eye]['frame']
            image = cv2.imread(str(path))
            centers = np.array([guide[name][side]['crop_center'] for side in ('L','R')])
            radius = image.shape[1] / 16
            boxes = np.column_stack((centers-radius, centers+radius))
            dataset = w.ViTDetDataset(config, image, boxes, np.array([0,1]), rescale_factor=2.)
            batch = w.recursive_to(next(iter(torch.utils.data.DataLoader(dataset,batch_size=2))), 'cuda')
            with torch.inference_mode():
                pred = model(batch)
                cam = pred['pred_cam'].clone()
                sign = 2 * batch['right'] - 1
                cam[:,1] *= sign
                sizes = batch['img_size'].float()
                focal = config.EXTRA.FOCAL_LENGTH / config.MODEL.IMAGE_SIZE * sizes.max()
                trans = w.cam_crop_to_full(cam,batch['box_center'].float(),batch['box_size'].float(),sizes,focal)
                joints = pred['pred_keypoints_3d'].clone()
                verts = pred['pred_vertices'].clone()
                joints[:,:,0] *= sign[:,None]
                verts[:,:,0] *= sign[:,None]
            for i,side in enumerate(('L','R')):
                def project(points):
                    return w.project(points[i].cpu().numpy(),trans[i].cpu().numpy(),float(focal),sizes[i].cpu().numpy())
                replay = np.max(np.abs(project(joints)-guide[name][side]['uv']))
                assert replay < .1
                uv = project(verts)
                assert torch.all(verts[i,:,2]+trans[i,2] > 0) and np.isfinite(uv).all()
                mask = np.zeros(image.shape[:2],np.uint8)
                for triangle in np.rint(uv[faces]*16).astype(np.int32):
                    cv2.fillConvexPoly(mask,triangle,255,shift=4)
                label = name + '_' + side
                target = OUT/'masks'/(label+'.png')
                assert cv2.imwrite(str(target),mask)
                records[label] = dict(mask_sha256=sha(target),area_pixels=int((mask>0).sum()),replay_max_px=float(replay))
    (RUN/'evidence/masks.json').write_text(json.dumps(dict(checkpoint_sha256=cfg['checkpoint_sha256']['wilor'],views=records,exit_status=0),indent=2))


if __name__ == '__main__':
    cfg = read(ROOT/'configs/pipeline.head01.developer.json')
    if sys.argv[1:] == ['infer']:
        infer(cfg)
    else:
        mano = OUT/'mano_data'
        if not mano.exists():
            mano.symlink_to(Path(cfg['paths']['wilor_repo'])/'mano_data',target_is_directory=True)
        env = dict(os.environ,PYTHONPATH=os.pathsep.join((str(ROOT),cfg['paths']['wilor_repo'])),PYTHONDONTWRITEBYTECODE='1')
        for name in ('XDG_CACHE_HOME','MPLCONFIGDIR','TMPDIR','TORCH_HOME','CUDA_CACHE_PATH','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','YOLO_CONFIG_DIR'):
            dest=OUT/'runtime'/name
            dest.mkdir(exist_ok=True)
            env[name]=str(dest)
        subprocess.run([cfg['paths']['wilor_python'],'-Bu',str(Path(__file__).resolve()),'infer'],cwd=OUT,env=env,check=True)
