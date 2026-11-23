"""Audit paired views and fit empirical bounds on official training people only."""
import json
import zipfile
import numpy as np
import torch
from angles import CHAINS, NAMES, ROOT, EVIDENCE, features, sha, write

PERCENTILES = [0, .5, 1, 5, 50, 95, 99, 99.5, 100]
SPLITS = ('train', 'validation', 'test')


def summarize(values, bounds):
    outside = (values < bounds[0]) | (values > bounds[1])
    return dict(frames=len(values), percentiles=np.percentile(values, PERCENTILES, axis=0).tolist(),
                outside_per_feature=outside.sum(0).tolist(), frames_any_outside=int(outside.any((1, 2)).sum()))


def main():
    torch.set_num_threads(1)
    data = ROOT.parent / 'EgoSMPLX/data'
    splitfile = data / 'UnrealEgo_RW_Train/official_splits/UnrealEgoData_rw.zip'
    with zipfile.ZipFile(splitfile) as archive:
        lists = {s: set(archive.read('UnrealEgoData_rw/' + s + '.txt').decode().splitlines()) for s in SPLITS}
    values, people, coverage, sources = {}, {}, {}, {}
    for collection, partitions in [('Train', SPLITS[:2]), ('Test', SPLITS[2:])]:
        folder = data / ('UnrealEgo_RW_' + collection) / 'prepared'
        meta = json.loads((folder / 'metadata.json').read_text())
        names = meta['fit_joint_names']
        lookup = {n: names.index(n) for chain in CHAINS for n in chain}
        with np.load(folder / 'samples.npz') as z:
            seq, frames, sides = (z[k] for k in ('source_sequence', 'frame_index', 'camera_side'))
            points = z['keypoints3d_fit']
            fields = list(z.files)
        assert set(seq) == set.union(*(lists[s] for s in partitions))
        assert len(points) == meta['sample_count'] == 2 * meta['pose_json_count']
        assert set(sides) == {'left', 'right'} and np.isfinite(points).all()
        sources[collection] = dict(samples_sha256=sha(folder/'samples.npz'), fields=fields,
            meta_sha256=sha(folder/'metadata.json'), count=len(seq), joint_order=names,
            invalid_pose=meta['invalid_pose'], missing_rgb=meta['missing_rgb'])
        for s in partitions:
            eyes = []
            identities = []
            for eye in ['left', 'right']:
                ids = np.flatnonzero(np.isin(seq, list(lists[s])) & (sides == eye))
                ids = ids[np.lexsort((frames[ids], seq[ids]))]
                keys = list(zip(seq[ids], frames[ids]))
                assert len(keys) == len(set(keys))
                identities.append(keys)
                x = torch.from_numpy(points[ids].astype(float))
                segments = [(a,b) for a,b,c in CHAINS] + [(b,c) for a,b,c in CHAINS] + [('L_Hip','R_Hip')]
                assert all((x[:,lookup[a]]-x[:,lookup[b]]).norm(dim=-1).min() > 1e-6 for a,b in segments)
                eyes.append(features(x, lookup).numpy())
            assert identities[0] == identities[1]
            diff = np.abs(eyes[0] - eyes[1]).max(axis=(0, 1))
            assert diff[0] < .01 and diff[1] < .001, diff
            values[s] = eyes[0]
            people[s] = np.array([k[0].split('/')[2] for k in identities[0]])
            coverage[s] = dict(sequences=len(lists[s]), frames=len(eyes[0]), view_records=2*len(eyes[0]),
                               subjects=sorted(set(people[s])), stereo_max_abs_difference=diff.tolist())
    assert sum(len(set(p)) for p in people.values()) == len(set(np.concatenate(list(people.values()))))
    qlo, qhi = np.percentile(values['train'], [.5, 99.5], axis=0)
    bounds = np.stack([np.maximum(qlo - [5., .05], [0., -1.]), np.minimum(qhi + [5., .05], [180., 1.])])
    write(EVIDENCE/'prior.json', dict(joints=NAMES, features=['bend_deg', 'signed_cross_hip_axis'],
         training_people=coverage['train']['subjects'], lower=bounds[0].tolist(), upper=bounds[1].tolist(),
         scales=[10., .15], weight=.01, split_sha256=sha(splitfile), train_source_sha256=sources['Train']['samples_sha256']))
    tables = {s: summarize(values[s], bounds) for s in SPLITS}
    tables['all_descriptive'] = summarize(np.concatenate(list(values.values())), bounds)
    persons = {s: {p: summarize(values[s][people[s] == p], bounds) for p in coverage[s]['subjects']} for s in SPLITS}
    write(EVIDENCE/'survey.json', dict(command='bash run.sh survey', exit_status=0, partitions=coverage,
         sources=sources, joint_order=NAMES, percentile_order=PERCENTILES, splits=tables, people=persons))


if __name__ == '__main__':
    main()
