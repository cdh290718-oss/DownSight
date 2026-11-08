"""Oracle Head-position experiment. No other GT values enter the objective.

The two controls share the original raw initialization. anchor_only_rigid adds
one position loss to the original rigid objective. requested_rigid changes the
observations/depth rule too, and is followed by alternating pose/rigid blocks.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from egosmplx_pipeline.bootstrap import setup
from egosmplx_pipeline.body_common import (
    BODY_POSE_NAMES, JOINT_MAPPING, MAX_DELTA_DEG, axis_angle_rotation,
)
from egosmplx_pipeline.io import read, load

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/synthetic_head_anchor_20261108'
OLD = ROOT / 'outputs/synthetic_original_20261108'


def shoulder_excess(residual, tolerance_px):
    """Radial dead zone, preserving residual direction outside the zone."""
    radius = residual.norm(dim=-1, keepdim=True)
    return residual * (1. - tolerance_px / radius.clamp_min(1e-6)).clamp_min(0.)


def weighted_mean(values, weights):
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def positive_projection_points(points, minimum_z):
    """Numerical projection guard; no in-place autograd view modification."""
    return torch.stack([points[:, 0], points[:, 1], points[:, 2].clamp_min(minimum_z)], dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--side', choices=['left', 'right'], required=True)
    a = ap.parse_args()
    dest = OUT / a.side
    dest.mkdir(parents=True, exist_ok=False)
    cfg = read(OLD / 'configs' / ('left.json' if a.side == 'left' else 'right_cpu.json'))
    settings = read(RUN / 'settings.json')
    os.environ.update(EGOSMPLX_CONFIG=cfg['paths']['egosmplx_config'],
                      EGOSMPLX_REPO_ROOT=cfg['paths']['egosmplx_repo'])
    model, Camera = setup(cfg['paths']['egosmplx_repo'])
    from egosmplx_pipeline import fusion
    fusion.initialize(model)
    layer = model.layer['neutral'].cpu().eval()
    layer.requires_grad_(False)
    t = lambda x: torch.as_tensor(x, dtype=torch.float32)
    old_frame = Path(cfg['output']) / 'frames' / a.side
    raw_path = old_frame / 'raw/params.npz'
    raw = load(raw_path)
    old_job = read(Path(cfg['output']) / 'jobs' / (a.side + '.json'))
    obs_path = Path(old_job['row']['predictions']['sapiens'])
    obs = read(obs_path)
    image_path = OLD / 'dataset/samples/000001' / (a.side + '.png')
    assert obs['image_sha256'] == hashlib.sha256(image_path.read_bytes()).hexdigest()
    assert obs['coordinate_space'] == 'original_image_pixels'
    assert [obs['image_width'], obs['image_height']] == [1280, 720]
    cam = Camera(old_job['row']['calibration'])
    lookup = {q['name']: q for q in obs['keypoints']}
    selected = [n for n in JOINT_MAPPING if n in lookup and
                np.isfinite([lookup[n]['x'], lookup[n]['y'], lookup[n]['score']]).all()
                and lookup[n]['score'] >= .3 and
                0 <= lookup[n]['x'] < 1280 and 0 <= lookup[n]['y'] < 720]
    # Head is a native SMPL-X joint, absent from the 65-joint output ordering.
    head_index = model.orig_joints_name.index('Head')
    assert head_index == 15
    gt_path = OLD / 'dataset/samples/000001' / (a.side + '_vertices.npz')
    # Isolate the sole GT input before running the optimizer. Other GT arrays
    # are read only by the separate evaluation program, never here.
    with np.load(gt_path, allow_pickle=False) as source:
        head_target = t(source['joints_camera_cv_m'][head_index].copy())
    assert torch.isfinite(head_target).all()
    fixed_keys = ['betas', 'left_hand_pose', 'right_hand_pose', 'jaw_pose',
                  'leye_pose', 'reye_pose', 'expression']
    fixed = {k: t(raw[k]).reshape(1, -1) for k in fixed_keys}
    body0 = t(raw['body_pose']).reshape(21, 3)
    glob0 = t(raw['global_orient']).reshape(1, 3)
    cam0 = t(raw['transl']).reshape(3)
    active = list(MAX_DELTA_DEG)
    active_ids = torch.tensor([BODY_POSE_NAMES.index(n) for n in active])
    caps = t([np.deg2rad(MAX_DELTA_DEG[n]) for n in active])
    head_ids = np.flatnonzero(layer.lbs_weights.detach().numpy().argmax(1) == head_index)
    config_record = dict(settings=settings, side=a.side,
                         head_target_camera_m=head_target.tolist(), head_joint_index=head_index,
                         raw_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                         observations_sha256=hashlib.sha256(obs_path.read_bytes()).hexdigest(),
                         head_source_sha256=hashlib.sha256(gt_path.read_bytes()).hexdigest(),
                         selected_before_nose_removal=selected, fixed_parameters=fixed_keys,
                         no_other_gt_used=True, camera='original polynomial calibration',
                         anchor_only='original rigid losses plus measured-position oracle; retains nose and original all-vertex soft penalty',
                         requested='no nose, relaxed shoulder group, no all-vertex depth constraint, positive observed joints only; no old rigid-derived head prior',
                         hand_scope='SMPL-X hand parameters fixed; no MANO replacement or downstream refinement in this body-stage diagnostic')
    (dest / 'inputs.json').write_text(json.dumps(config_record, indent=2) + '\n')
    summaries = {}

    def fit_rigid(mode):
        requested = mode == 'requested_rigid'
        names = [n for n in selected if not (requested and n == 'nose')]
        indices = torch.tensor([model.joints_name.index(JOINT_MAPPING[n]) for n in names])
        target = t([[lookup[n]['x'], lookup[n]['y']] for n in names])
        weights = t([min(1., lookup[n]['score']) ** 2 for n in names])
        groups = {
            'body': torch.tensor([i for i, n in enumerate(names) if not n.endswith(('_shoulder', '_wrist'))], dtype=torch.long),
            'wrist': torch.tensor([i for i, n in enumerate(names) if n.endswith('_wrist')], dtype=torch.long),
            'shoulder': torch.tensor([i for i, n in enumerate(names) if n.endswith('_shoulder')], dtype=torch.long),
        }
        scale = t([1280, 720])
        pose_delta = torch.zeros((len(active), 3), requires_grad=True)
        rotation_delta = torch.zeros(3, requires_grad=True)
        translation = cam0.clone().requires_grad_(True)
        reference = None
        wrist_floor = None

        def local_geometry():
            body = body0.clone()
            body[active_ids] += pose_delta
            result = layer(body_pose=body.reshape(1, -1), global_orient=glob0, **fixed)
            return result.vertices[0], result.joints[0], result.joints[0, 0]

        def forward(local=None):
            vertices, allj, pivot = local_geometry() if local is None else local
            rotation = axis_angle_rotation(rotation_delta)
            mesh = (vertices - pivot) @ rotation.T + pivot + translation
            native = (allj - pivot) @ rotation.T + pivot + translation
            joints = native[list(model.joint_idx)]
            # No full-mesh projection in the optimizer. Only selected observed
            # joints need positive z. Head anchor is a 3D position only.
            projected_joints = joints[indices]
            if requested:
                safe = positive_projection_points(projected_joints, settings['selected_joint_minimum_z_m'])
                pixels = cam.world2camera_pytorch(safe)
            else:
                pixels = cam.world2camera_pytorch(projected_joints)
            return mesh, native, joints, pixels

        def objective(mesh, native, joints, pixels):
            residual = pixels - target
            if requested:
                terms = {}
                for name, ids in groups.items():
                    r = residual[ids]
                    if name == 'shoulder':
                        r = shoulder_excess(r, settings['shoulder_tolerance_px'])
                    loss = F.smooth_l1_loss(r / scale, torch.zeros_like(r), beta=.005, reduction='none').mean(1)
                    terms[name] = weighted_mean(loss, weights[ids])
                rep = (settings['other_body_group_weight'] * terms['body'] +
                       settings['wrist_group_weight'] * terms['wrist'] +
                       settings['shoulder_group_weight'] * terms['shoulder'])
                selected_z = joints[indices, 2]
                positive = .01 * F.relu((settings['selected_joint_minimum_z_m'] - selected_z) / .01).square().mean()
                depth = mesh.new_zeros(())
            else:
                robust = F.smooth_l1_loss(residual / scale, torch.zeros_like(residual), beta=.005, reduction='none').mean(1)
                rep = weighted_mean(robust, weights)
                terms = {'original_weighted_2d': rep}
                positive = mesh.new_zeros(())
                depth = F.relu(.02 - mesh[:, 2].min()).square()
            anchor = settings['anchor']['loss_weight'] * ((native[head_index] - head_target) / settings['anchor']['sigma_m']).square().sum()
            if reference is None:
                prior = 1e-4 * ((translation - cam0) / t([.25, .25, .5])).square().mean() + 2e-4 * rotation_delta.square().mean()
                wrist_guard = mesh.new_zeros(())
            else:
                prior = (settings['pose_prior_weight'] * (pose_delta / caps[:, None]).square().mean() +
                         settings['camera_prior_weight'] * ((translation - reference['translation']) / t([.10, .10, .18])).square().mean() +
                         settings['orient_prior_weight'] * ((rotation_delta - reference['rotation']) / np.deg2rad(8.)).square().mean())
                wrist_guard = settings['wrist_guard_weight'] * F.relu((wrist_floor - joints[[12, 13]].norm(dim=1)) / .05).square().mean()
            loss = rep + anchor + prior + depth + positive + wrist_guard
            feasible = (not requested or bool((joints[indices, 2] >= settings['selected_joint_minimum_z_m']).all()))
            if wrist_floor is not None:
                feasible = feasible and bool((joints[[12, 13]].norm(dim=1) >= wrist_floor - 1e-6).all())
            breakdown = {k: float(v.detach()) for k, v in dict(terms, anchor=anchor, prior=prior, depth=depth, positive_joint_guard=positive, wrist_guard=wrist_guard).items()}
            return loss, feasible, breakdown

        def snapshot():
            return tuple(p.detach().clone() for p in (pose_delta, rotation_delta, translation))

        def restore(state):
            with torch.no_grad():
                for p, v in zip((pose_delta, rotation_delta, translation), state):
                    p.copy_(v)

        def optimize_block(label, kind, steps):
            start = time.time()
            if kind == 'pose':
                params = [{'params': [pose_delta], 'lr': settings['pose_lr']}]
                local = None
                frozen_before = (rotation_delta.detach().clone(), translation.detach().clone())
            else:
                params = [{'params': [rotation_delta], 'lr': .002 if reference is None else settings['post_rigid_orient_lr']},
                          {'params': [translation], 'lr': .005 if reference is None else settings['post_rigid_camera_lr']}]
                with torch.no_grad():
                    local = tuple(x.detach() for x in local_geometry())
                frozen_before = pose_delta.detach().clone()
            opt = torch.optim.Adam(params)
            trace, best_loss, best = [], float('inf'), None
            for step in range(steps + 1):
                for p in (pose_delta, rotation_delta, translation):
                    p.grad = None
                mesh, native, joints, pixels = forward(local)
                loss, feasible, terms = objective(mesh, native, joints, pixels)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite objective in ' + label)
                value = float(loss.detach())
                if feasible and value < best_loss:
                    best_loss, best = value, snapshot()
                if step % 100 == 0 or step == steps:
                    row = dict(step=step, loss=value, feasible=feasible, terms=terms,
                               head_error_mm=float((native[head_index] - head_target).norm().detach() * 1000),
                               pelvis_z_m=float(joints[0, 2].detach()), minimum_mesh_z_m=float(mesh[:, 2].min().detach()))
                    trace.append(row)
                if step == steps:
                    break
                loss.backward()
                if not all(p.grad is not None and torch.isfinite(p.grad).all() for group in params for p in group['params']):
                    raise RuntimeError('Nonfinite gradient in ' + label)
                opt.step()
                with torch.no_grad():
                    if kind == 'pose':
                        pose_delta.mul_(torch.minimum(torch.ones_like(caps), caps / pose_delta.norm(dim=1).clamp_min(1e-8))[:, None])
                    else:
                        if reference is None:
                            rotation_delta.mul_(torch.clamp(t(np.deg2rad(60.)) / rotation_delta.norm().clamp_min(1e-8), max=1.))
                        else:
                            gd = rotation_delta - reference['rotation']
                            limit = np.deg2rad(settings['post_rigid_orientation_increment_limit_deg'])
                            rotation_delta.copy_(reference['rotation'] + gd * torch.clamp(t(limit) / gd.norm().clamp_min(1e-8), max=1.))
                        translation[2].clamp_(.05, 3.)
            if best is None:
                raise RuntimeError('No feasible state for ' + label)
            restore(best)
            if kind == 'pose':
                frozen_equal = torch.equal(rotation_delta, frozen_before[0]) and torch.equal(translation, frozen_before[1])
            else:
                frozen_equal = torch.equal(pose_delta, frozen_before)
            assert frozen_equal
            with torch.no_grad():
                mesh, native, joints, pixels = forward()
                final_loss, feasible, terms = objective(mesh, native, joints, pixels)
                assert feasible and float(final_loss) <= trace[0]['loss'] + 2e-6
            saved = {k: v.copy() for k, v in raw.items()}
            body = body0.detach().clone()
            body[active_ids] += pose_delta.detach()
            rotation = axis_angle_rotation(rotation_delta).detach().numpy() @ cv2.Rodrigues(raw['global_orient'].astype(float).reshape(3))[0]
            saved.update(body_pose=body.numpy().reshape(raw['body_pose'].shape),
                         global_orient=cv2.Rodrigues(rotation.astype(float))[0].astype(np.float32).reshape(1, 3),
                         transl=translation.detach().numpy().reshape(1, 3))
            mv, jj = fusion.geometry(saved)
            geometry_error = float(np.max(np.abs(mv - mesh.detach().numpy())))
            # Behind-camera or unobserved joints can have unstable polynomial
            # extrapolations; only actually constrained joints are pixel-tested.
            uv_reload = cam.world2camera(jj[indices.numpy()].astype(float))
            pixel_error = float(np.max(np.abs(uv_reload - pixels.detach().numpy())))
            if geometry_error > 1e-5 or pixel_error > .05:
                raise RuntimeError('Regeneration error: %g m; %g selected px' % (geometry_error, pixel_error))
            saved.update(smplx_mesh_cam=mv, smplx_body_joints_3d_render_cam=jj[:25],
                         smplx_body_joints_3d_local=jj[:25] - saved['transl'].reshape(3),
                         smplx_body_joints_2d_render_cam=cam.world2camera(jj[:25].astype(float)),
                         smplx_joint_cam_root_relative=jj - jj[:1])
            for key in fixed_keys:
                assert np.array_equal(saved[key], raw[key]), key
            stage_dir = dest / label
            stage_dir.mkdir(exist_ok=False)
            np.savez_compressed(stage_dir / 'params.npz', **saved)
            record = dict(label=label, kind=kind, steps=steps, elapsed_seconds=time.time() - start,
                          initial_loss=trace[0]['loss'], selected_best_loss=float(final_loss), terms=terms,
                          selected_points=names, trace=trace, frozen_block_equal=frozen_equal,
                          regenerated_mesh_max_abs_m=geometry_error, regenerated_selected_pixel_max_abs_px=pixel_error,
                          head_error_mm=float((native[head_index] - head_target).norm() * 1000),
                          head_position_m=native[head_index].tolist(), pelvis_z_m=float(joints[0, 2]),
                          minimum_vertex_z_m=float(mesh[:, 2].min()),
                          wrist_distances_m=joints[[12, 13]].norm(dim=1).tolist(),
                          wrist_floor_m=wrist_floor.tolist() if wrist_floor is not None else None,
                          pose_increment_deg={n: float(v) for n, v in zip(active, np.rad2deg(pose_delta.detach().norm(dim=1).numpy()))},
                          selected_target_error_px={n: float(v) for n, v in zip(names, (pixels - target).norm(dim=1))},
                          joint_positive_guard=requested, all_vertex_near_guard=not requested,
                          head_anchor_is_oracle=True, parameter_sha256=hashlib.sha256((stage_dir / 'params.npz').read_bytes()).hexdigest())
            (stage_dir / 'stage.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
            print(a.side, label, 'loss', round(float(final_loss), 6), 'head_mm', round(record['head_error_mm'], 3), flush=True)
            summaries[label] = record
            (dest / 'stages.json').write_text(json.dumps(summaries, indent=2, allow_nan=False) + '\n')

        optimize_block(mode, 'rigid', settings['rigid_steps'])
        if not requested:
            return
        reference = {'translation': translation.detach().clone(), 'rotation': rotation_delta.detach().clone()}
        with torch.no_grad():
            wrist_floor = forward()[2][[12, 13]].norm(dim=1) * settings['wrist_floor_fraction']
        for cycle in range(1, settings['alternating_cycles'] + 1):
            optimize_block('cycle%d_pose' % cycle, 'pose', settings['pose_steps_per_cycle'])
            optimize_block('cycle%d_rigid' % cycle, 'rigid', settings['rigid_steps_per_cycle'])

    fit_rigid('anchor_only_rigid')
    fit_rigid('requested_rigid')


if __name__ == '__main__':
    main()
