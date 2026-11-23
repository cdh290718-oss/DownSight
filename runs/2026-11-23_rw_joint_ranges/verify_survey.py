"""Cross-check magnitudes from 15-point labels by triangle side lengths."""
import json
from pathlib import Path
import numpy as np
from angles import CHAINS, EVIDENCE, ROOT, sha, write


def main():
    backend = ROOT.parent/'EgoSMPLX'
    tables = json.loads((EVIDENCE/'survey.json').read_text())
    order = tables['percentile_order']
    all_angles, rows = [], {}
    for collection in ('Train','Test'):
        prepared = backend/'data'/('UnrealEgo_RW_'+collection)/'prepared'
        meta = json.loads((prepared/'metadata.json').read_text())
        names = meta['train_joint_names']
        with np.load(prepared/'samples.npz') as z:
            labels, with_head = z['keypoints3d'], z['keypoints3d_fit']
            assert np.array_equal(labels,with_head[:,[meta['fit_joint_names'].index(n) for n in names]])
            points = labels[z['camera_side']=='left'].astype(float)
        bends = []
        for chain in CHAINS:
            a,b,c = [points[:,names.index(n)] for n in chain]
            ab,bc = np.linalg.norm(b-a,axis=1),np.linalg.norm(c-b,axis=1)
            ac_sq = np.square(c-a).sum(axis=1)
            bends.append(np.rad2deg(np.arccos(np.clip((ac_sq-ab**2-bc**2)/(2*ab*bc),-1,1))))
        all_angles.append(np.stack(bends,axis=1))
        rows[collection] = dict(raw_pose_dir_available=Path(meta['pose_root']).is_dir(),
            labels15_equal_corresponding_labels16=True, raw_to_m=meta['unit_scale_to_meters'],
            official_joint_names=meta['unrealego_fit_bones'])
    measured = np.percentile(np.concatenate(all_angles),order,axis=0)
    expected = np.array(tables['splits']['all_descriptive']['percentiles'])[:,:,0]
    delta = float(abs(measured-expected).max())
    assert delta < 1e-6
    files = {'json_preparation':backend/'tools/prepare_unrealego_rw_train.py',
             'model_setup':backend/'common/utils/human_models.py',
             'neutral_smplx':backend/'common/utils/human_model_files/smplx/SMPLX_NEUTRAL.npz'}
    write(EVIDENCE/'source_check.json',dict(command='bash run.sh verify_survey', exit_status=0,
        baseline='c0c225eef7d39c344a7a61fd1e87566e3fd8a47e', dirty_worktree_at_execution=True,
        sources=rows, checked_angle_percentiles=order,
        max_percentile_difference_deg=delta, external_sha256={k:sha(p) for k,p in files.items()},
        official_source_url='https://unrealego.mpi-inf.mpg.de/', source='Prepared official per-view 3D annotations. Original JSON roots unavailable at recorded locations.'))


if __name__=='__main__':
    main()
