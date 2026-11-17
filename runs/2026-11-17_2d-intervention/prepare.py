"""Coordinate-only interventions; never read non-Head body 3D ground truth."""
import copy
import sys

import numpy as np

from common import ARCH, EVIDENCE, MODEL_IDS, NAMES, OBS, OUT, ROOT, RUN, camera_files, read, sha, write


def main():
    cfg = read(RUN / 'protocol.json')
    assert not (OUT / 'prepared.json').exists(), 'Use a new output directory rather than overwrite preparation.'
    sys.path.insert(0, str(ROOT / 'runs/2026-11-04_session-hand6-v2/code'))
    from prepare import Camera, triangulate
    samples = read(ARCH / 'body_samples.json')['samples']
    datafile = ROOT.parent / 'EgoSMPLX/data/UnrealEgo_RW_Test/prepared/samples.npz'
    # Only the 2D array is extracted. Head anchors already exist from prior runs.
    with np.load(datafile) as data:
        reference2d = data['keypoints2d']
    rig = read(ROOT / 'runs/2026-11-15_unrealego-gt2d/evidence/preparation_ideal.json')
    r, t = np.array(rig['rig_rotation']), np.array(rig['rig_translation_m'])
    cams = [Camera(p) for p in camera_files()]
    adapters = [read(p)['input_adapter'] for p in camera_files()]
    items, public = [], []
    for pair_idx, name in enumerate(cfg['pairs']):
        recs = samples[2*pair_idx:2*pair_idx+2]
        assert recs[0]['id'] == name + '_left' and recs[1]['id'] == name + '_right'
        baseline = ROOT / 'outputs/unrealego_d_batch_20261115' / name
        tri = read(baseline / 'triangulated.json')['points']
        archived = {q['index']: q for q in tri if 'index' in q}
        targets, detections, detfiles = [], [], []
        for v, rec in enumerate(recs):
            gt = reference2d[rec['sample_index']].astype(float)
            gt[:, :2] = gt[:, :2] * adapters[v]['scale'] + np.array(adapters[v]['offset'])
            targets.append(gt[OBS])
            detfile = ARCH/'body_work_full_image'/rec['id']/'observations'/rec['id']/'sapiens/keypoints.json'
            detfiles.append(detfile)
            detections.append({q['name']: q for q in read(detfile)['keypoints']})
        reference = np.array(targets)
        points = []
        for slot, i in enumerate(OBS):
            part = NAMES[i].split('_')[-1].lower()
            detname = ('right_' if NAMES[i].startswith('R_') else 'left_') + part
            xy, valid, scores = [], [], []
            for v in range(2):
                p = detections[v].get(detname)
                uv = np.array([p['x'], p['y']], float) if p else np.array([0., 0.])
                gt = reference[v, slot]
                ok = bool(p and np.isfinite(uv).all() and np.isfinite(gt).all() and gt[2] > 0
                          and 280 <= gt[0] < 1000 and 0 <= gt[1] < 720)
                xy.append(uv.tolist()); valid.append(ok); scores.append(float(p['score']) if p else None)
            old = archived.get(MODEL_IDS[i], {})
            accepted = bool(old.get('accepted', False) and all(valid))
            point = dict(name=NAMES[i], index=MODEL_IDS[i], native_index=[12,17,19,21,16,18,20,2,5,8,65,1,4,7,62][i],
                         group='shoulder' if part == 'shoulder' else 'wrist' if part == 'wrist' else 'body',
                         part=part, xy=xy, valid=valid, scores=scores,
                         sigma_px=cfg['sigma_px_fixed'], accepted=accepted, source='sapiens')
            if accepted:
                point.update(xyz=old['xyz'], covariance=old['covariance'])
            points.append(point)
        input_rows = []
        for condition in cfg['conditions']:
            changed = []
            alpha = 0.5 if condition == 'all_half' else 0. if condition == 'baseline' else 1.
            new = copy.deepcopy(points)
            for j, p in enumerate(new):
                change = condition.startswith('all_') or p['part'] == condition
                if change:
                    changed.append(p['name'])
                    p['xy'] = [((1-alpha)*np.array(p['xy'][v]) + alpha*reference[v,j,:2]).tolist()
                               if p['valid'][v] else p['xy'][v] for v in range(2)]
                if p['accepted'] and change:
                    q = triangulate(cams, r, t, np.array(p['xy']))
                    if 'point_cam3_m' not in q or not np.isfinite(q['point_cam3_m']).all():
                        raise RuntimeError(name + '/' + condition + ': fixed-route triangulation unavailable')
                    p['xyz'] = q['point_cam3_m']
                # Identical masks, scores, covariance and group normalizers.
                assert {k:v for k,v in p.items() if k not in ['xy','xyz']} == {
                    k:v for k,v in points[j].items() if k not in ['xy','xyz']}
                if not change:
                    assert p == points[j]
            errors = [float(np.linalg.norm(np.array(p['xy'][v])-reference[v,j,:2]))
                      for j,p in enumerate(new) for v in range(2) if p['valid'][v]]
            base_errors = [float(np.linalg.norm(np.array(p['xy'][v])-reference[v,j,:2]))
                           for j,p in enumerate(points) for v in range(2) if p['valid'][v]]
            if condition == 'all_half':
                assert np.allclose(errors, np.array(base_errors)*0.5, atol=1e-10)
            if condition == 'all_full':
                assert max(errors) < 1e-10
            item = dict(pair=name, sequence=recs[0]['sequence'], subject='S9', condition=condition,
                        changed_joints=changed, points=new,
                        anchor=read(baseline/'anchors/oracle.json'), inputs=read(baseline/'inputs.json'))
            items.append(item)
            input_rows.append(dict(condition=condition, changed_joints=changed,
                                   mean_2d_error_px=float(np.mean(errors)),
                                   per_view_joint_error_px=errors, valid_view_points=len(errors)))
        public.append(dict(pair=name, sequence=recs[0]['sequence'], subject='S9',
                           sample_indices=[r['sample_index'] for r in recs],
                           baseline_accepted_3d=[p['name'] for p in points if p['accepted']],
                           baseline_valid_views={p['name']:p['valid'] for p in points},
                           input_metrics=input_rows,
                           source_sha256={str(p.relative_to(ROOT.parent)):sha(p) for p in
                                          detfiles+[baseline/'anchors/oracle.json', baseline/'triangulated.json']}))
    assert len(set(p['sequence'] for p in public)) == len(public)
    write(OUT/'prepared.json', dict(items=items, protocol_sha256=sha(RUN/'protocol.json')))
    write(EVIDENCE/'preparation.json', dict(
        command='<BODY_PYTHON> runs/2026-11-17_2d-intervention/prepare.py', exit_status=0,
        dataset_sha256=sha(datafile), dataset_arrays_read=['keypoints2d'],
        calibration_sha256=[sha(p) for p in camera_files()],
        input_policy=cfg['observation_policy'], pairs=public,
        protocol_sha256=sha(RUN/'protocol.json'), prepared_local_sha256=sha(OUT/'prepared.json'),
        controls_verified=['valid masks', 'score fields', 'group and sigma', 'covariance and route',
                           'unchanged points exact', 'all_half halves each input error', 'all_full zero reference distance'],
        note='Scores retained but not thresholded in coordinate-only diagnostic; fixed mixed retains original strict baseline 3D route; Head and calibration annotation-dependent.'))
    print('Prepared', len(items), 'pair/condition combinations', flush=True)


if __name__ == '__main__':
    main()
