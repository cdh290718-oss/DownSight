"""Check reference reprojection and the reliability of the three additional detector channels."""
import runpy
import numpy as np
from study import OUT, OLD, ROOT, RUN, common, read, write
from automatic import meta_path


def summaries(x):
    x=np.asarray(x,float)
    return dict(n=len(x),mean_px=float(x.mean()),median_px=float(np.median(x)),p90_px=float(np.percentile(x,90)))


def main():
    g=np.load(common.ARCH/'body_gt_official.npz')['joints_m']
    _,camera_cls=common.setup_model()
    cams=[camera_cls(p) for p in common.camera_files()]
    checks=[]
    for i in range(4):
        pair='body%02d'%(i+1)
        obs=read(OLD/pair/'observations_local.json')
        assert obs['names']==common.NAMES
        transforms=read(OLD/pair/'inputs.json')['views']
        for eye,side in enumerate(['left','right']):
            t=np.array(transforms[side]['camera']['left_to_view'])
            joints=g[2*i]@t[:3,:3].T+t[:3,3]
            pred=cams[eye].world2camera(joints)
            ref=np.array(obs['views'][eye])[:,:2]
            errors=np.linalg.norm(pred-ref,axis=1)
            checks.append(dict(pair=pair,eye=side,errors_px=errors.tolist()))
    train=ROOT.parent/'EgoSMPLX/data/UnrealEgo_RW_Train/prepared'
    names=read(train/'metadata.json')['train_joint_names']
    with np.load(train/'samples.npz') as z:
        ref=z['keypoints2d']
    indices={v['name']:i for i,v in runpy.run_path(str(meta_path))['dataset_info']['keypoint_info'].items()}
    mapping={m['name']:m for m in read(ROOT/'runs/2026-11-16_sapiens-unrealego-audit/evidence/audit.json')['joint_mapping']}
    val={x['id']:x for x in read(ROOT/'outputs/rw_a20000_dataset_20261117/validation.json')['records']}
    frac=read(RUN/'evidence/automatic_inputs.json')['foot_fraction']
    samples={k:[] for k in names}
    for row in read(ROOT/'outputs/rw_a20000_evaluation_20261117/rw20000_validation_predictions.json'):
        r=val[row['id']]
        assert r['subject']!='S9' and row['image_sha256']==r['image_sha256']
        xy=np.asarray(row['xy'])
        gt=ref[r['source_index']]
        for j,n in enumerate(names):
            if not ((gt[j,:2]>=0)&(gt[j,:2]<872)).all():
                continue
            if n=='Neck':
                p=xy[indices['neck']]
            else:
                side,part=n.split('_')
                prefix='left_' if side=='L' else 'right_'
                if part=='Toe':
                    heel=xy[indices[prefix+'heel']]
                    tips=xy[[indices[prefix+'big_toe'],indices[prefix+'small_toe']]].mean(0)
                    p=heel+frac*(tips-heel)
                else:
                    p=xy[mapping[prefix+part.lower()]['sapiens308_id']]
            samples[n].append(np.linalg.norm(p-gt[j,:2]))
    e=np.array([r['errors_px'] for r in checks])
    write(RUN/'evidence/point_checks.json',dict(names=names,gt3d_reprojection_vs_gt2d=checks,
          projection_reference_mean_px=e.mean(),projection_reference_max_px=e.max(),
          validation_original_px={k:summaries(v) for k,v in samples.items()},
          common12_validation=summaries([v for j in common.OBS for v in samples[names[j]]]),
          note='Reference reprojection is in the adapted image. Detector errors are in original RW pixels. The validation set was used to fit the foot scalar.'))
if __name__=='__main__':
    main()
