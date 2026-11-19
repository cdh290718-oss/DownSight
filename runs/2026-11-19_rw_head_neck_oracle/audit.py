"""Read body GT and both selected checkpoints only after completing the oracle fits."""
from pathlib import Path
import json

import numpy as np
import torch

from run import HERE, OLD, OUT, ROOT, common, read, sha, write
from geometry import rotation


def distance(a, b):
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)) * 1000)


def main():
    assert read(OUT/'fit_complete.json')['exit_status'] == 0
    results = read(HERE/'evidence/results.json')['records']
    model, _ = common.setup_model()
    layer = model.layer['neutral'].cpu().eval()
    rest = (layer.J_regressor @ layer.v_template).detach().numpy()
    assert model.orig_joints_name[12] == 'Neck' and model.orig_joints_name[15] == 'Head'
    assert model.joint_idx[7] == 12 and model.joints_name[7] == 'Neck'
    assert layer.parents[15].item() == 12
    rest_length = distance(rest[15], rest[12])
    cf = read(ROOT/'outputs/initialization_ablation_20261109/model_paths.json')
    backend = Path(cf['egosmplx_repo'])
    ds = backend/'data/UnrealEgo_RW_Test/prepared'
    meta = read(ds/'metadata.json')
    assert meta['unrealego_fit_bones'][:2] == meta['fit_joint_names'][:2] == ['Head','Neck']
    assert meta['train_joint_names'][0] == meta['unrealego_train_bones'][0] == 'Neck'
    with np.load(ds/'samples.npz') as z:
        body = z['keypoints3d']
        annotation = z['keypoints3d_fit'].astype(float)
        assert np.array_equal(annotation[:,1:], body)
        subject = np.array([s.split('/')[2] for s in z['source_sequence']])
        eyes = z['camera_side']
        xy_all = z['keypoints2d']
    lengths = np.linalg.norm(annotation[:,0]-annotation[:,1], axis=-1)*1000
    by_subject = {}
    for name in sorted(set(subject)):
        vals = lengths[subject==name]
        by_subject[name] = dict(records=len(vals), min_mm=vals.min(), median_mm=np.median(vals), max_mm=vals.max())
    items = read(OUT/'prepared.json')['items']
    rank = read(OUT/'initialization.json')['initialization']
    ref_results = read(ROOT/'runs/2026-11-17_2d-intervention/evidence/results.json')['records']
    prior = {r['pair']:r for r in ref_results if r['condition']=='all_full' and r['budget']==20000 and r['method']=='2d'}
    previous = read(ROOT/'runs/2026-11-15_unrealego-gt2d/evidence/results.json')['records']
    direct = read(ROOT/'runs/2026-11-15_unrealego-gt16-four/evidence/results.json')['records']
    source_samples = read(common.ARCH/'body_samples.json')['samples']
    adapters = [read(p)['input_adapter'] for p in common.camera_files()]
    gt = np.load(common.ARCH/'body_gt_official.npz')['joints_m']
    pose_checkpoint = OUT/'budget_20000.npz'
    fitted = np.load(pose_checkpoint)
    tensor = lambda key: torch.tensor(fitted[key],dtype=torch.float32)
    assert not np.any(fitted['betas'])
    with torch.no_grad():
        b = len(items)
        fixed = {name:torch.zeros(b,width) for name,width in
                 [('left_hand_pose',45),('right_hand_pose',45),('jaw_pose',3),
                  ('leye_pose',3),('reye_pose',3),('expression',10)]}
        full = layer(body_pose=tensor('body_pose').reshape(b,-1),betas=tensor('betas'),
                     global_orient=tensor('global0'),**fixed)
        pivot = full.joints[:,:1]
        rd = rotation(tensor('delta')).transpose(1,2)
        native = ((full.joints-pivot)@rd + pivot + tensor('translation')[:,None]).numpy()
        mesh = ((full.vertices-pivot)@rd + pivot + tensor('translation')[:,None]).numpy()
    rows = []
    for k, (item, r) in enumerate(zip(items, results)):
        pair = item['pair']
        assert pair == r['pair'] == previous[k]['pair'] and r['selected_feasible'] and rank[pair]['rank']==0
        for v, eye in enumerate(['left','right']):
            source = source_samples[2*k+v]
            idx = source['sample_index']
            assert source['id'] == pair+'_'+eye and eyes[idx]==eye
            assert np.array_equal(gt[2*k+v], body[idx])
            uv = xy_all[idx,:,:2].astype(float) * adapters[v]['scale'] + adapters[v]['offset']
            assert np.array_equal(uv[common.OBS], [p['xy'][v] for p in item['points']])
        idx = source_samples[2*k]['sample_index']
        head, neck = annotation[idx,:2]
        assert np.array_equal(item['anchor']['position'],head)
        pred = native[k]
        body_err = np.linalg.norm(pred[common.NATIVE_IDS]-body[idx].astype(float),axis=1)*1000
        np.testing.assert_allclose(r['absolute_per_joint_mm'],body_err,atol=.01)
        e_head = distance(head,pred[15])
        source_len = distance(head,neck)
        model_len = distance(pred[15],pred[12])
        assert abs(model_len-rest_length)<.001
        lower = max(0.,abs(rest_length-source_len)-e_head)
        assert body_err[0] + .001 >= lower
        op = OLD/pair
        legacy = np.load(op/'D_left/final.npz')['smplx_joint_cam']
        assert sha(op/'D_left/final.npz')==previous[k]['final_sha256']
        xyz = read(op/'triangulated.json')['points']
        assert [p['name'] for p in xyz] == common.NAMES
        err = np.linalg.norm(np.array([p['xyz'] for p in xyz])-body[idx],axis=1)*1000
        direct_r = next(a for a in direct if a['method']=='D' and a['pair']==pair)
        rows.append(dict(pair=pair,head_error_mm=e_head,neck_error_mm=body_err[0],
            rw_head_neck_mm=source_len,smplx_head_neck_mm=model_len,
            neck_error_lower_bound_mm=lower,legacy_gt2d_neck_mm=previous[k]['absolute15_per_joint_mm'][0],
            legacy_gt2d_head_mm=previous[k]['head_position_error_mm'],
            legacy_gt2d_anchor_to_fitted_neck_mm=distance(head,legacy[7]),
            legacy_gt3d_neck_mm=direct_r['absolute15_per_joint_mm'][0],
            legacy_triangulated_gt3d_max_mm=err.max(),old_all_full_nonpa_mm=prior[pair]['absolute_mean_mm'],
            new_nonpa_mm=r['absolute_mean_mm'],new_pa_mm=r['pa_mean_mm'],
            converged=r['convergence_pass'],selected_step=r['selected_step'],
            **{opt+'_to_shoulders_midpoint_mm':distance(src,center)
               for opt,src,center in [('rw_head',head,body[idx,[1,4]].mean(0)),
                                      ('rw_neck',neck,body[idx,[1,4]].mean(0)),
                                      ('smplx_head',pred[15],pred[[16,17]].mean(0)),
                                      ('smplx_neck',pred[12],pred[[16,17]].mean(0))]}))
    binary_sources = {p.name:sha(p) for p in [ds/'samples.npz',ds/'metadata.json',backend/'tools/prepare_unrealego_rw_train.py',
                      backend/'common/utils/human_models.py',common.ARCH/'prepare_body_archive.py',pose_checkpoint]}
    out = dict(source_hashes=binary_sources,rest_length_mm=rest_length,shape_rank=0,beta_abs_max=float(np.abs(fitted['betas']).max()),
               official_scale_to_m=float(meta['unit_scale_to_meters']),native_parent_of_15=layer.parents[15].item(),
               rw_fit_head_neck_indices=[0,1],rw_gt15_neck_index=0,smplx_output_neck_index=7,
               smplx_native_neck_head=[12,15],subject_lengths_mm=by_subject,pairs=rows,
               mean_neck_mm=np.mean([v['neck_error_mm'] for v in rows]),
               mean_head_mm=np.mean([v['head_error_mm'] for v in rows]),
               mean_nonpa_mm=np.mean([r['absolute_mean_mm'] for r in results]),
               mean_pa_mm=np.mean([r['pa_mean_mm'] for r in results]),
               fixed_shape_length_gap_mm=rest_length-np.mean([v['rw_head_neck_mm'] for v in rows]))
    write(common.EVIDENCE/'audit.json',out)
    np.savez_compressed(OUT/'inspection_local.npz',mesh=mesh,faces=model.face,
                        neck_head=native[:,[12,15]],gt_neck_head=annotation[[s['sample_index'] for s in source_samples[:8:2]],:2][:,::-1])


if __name__ == '__main__':
    main()
