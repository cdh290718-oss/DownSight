"""Control for image sample count and accepted spans using official labels in the same slots."""
from calibrate import load_labels, sparse_median, stereo_targets
from data_shape import *
import torch


def score(pred, sequences, values):
    rows = [np.abs(values[s]-pred).mean(0)*1000 for s in sequences]
    prop = [np.abs(ratios(values[s])-ratios(pred)).mean(0) for s in sequences]
    return dict(mae_mm=np.mean(rows),proportion_mae_pp=np.mean(prop))


def main():
    torch.set_num_threads(1)
    values,_,_ = load_labels()
    cal=read(EV/'calibration.json')
    parts=cal['split']
    model,cam=ref.common.setup_model()
    basis=ShapeModel(model.layer['neutral'])
    stereo,used=stereo_targets(parts,[cam(p) for p in ref.camera_files()])
    matched={}
    for split,collection in [('validation','Train'),('test','Test')]:
        folder=ROOT.parent/'EgoSMPLX/data'/('UnrealEgo_RW_'+collection)/'prepared'
        with np.load(folder/'samples.npz') as z:
            xyz,seq,frame,eye=[z[k] for k in ['keypoints3d_fit','source_sequence','frame_index','camera_side']]
        names=read(folder/'metadata.json')['fit_joint_names']
        index={n:i for i,n in enumerate(names)}
        for pair,_,im,_ in inputs(split):
            meta=im[0]
            s=meta['sequence']
            if pair not in used.get(s,[]):
                continue
            i=meta['source_index']
            assert (seq[i],int(frame[i]),eye[i])==(s,meta['frame'],'left')
            lengths=sizes(xyz[i].astype(float),index)
            at=used[s].index(pair)
            lengths[~np.isfinite(stereo[s][at])]=np.nan
            if s not in matched:
                matched[s]=np.full((len(used[s]),len(SPANS)),np.nan)
            matched[s][at]=lengths
    records={}
    for p,part in parts.items():
        # Diagnostic uniform scale fitted by least squares to all 10 official spans.
        target=np.median([np.median(values[s],axis=0) for s in part['calibration']],axis=0)
        template=basis.predict(np.zeros(10))
        scale=(template@target)/(template@template)
        entries={'uniform_scale':score(scale*template,part['holdout'],values)}
        seqs=[s for s in part['calibration'] if s in matched]
        if seqs:
            for arm,source in [('matched_official',matched),('stereo12',stereo)]:
                frames=np.concatenate([source[s] for s in seqs])
                median=sparse_median(np.array([sparse_median(np.array(source[s]),1) for s in seqs]),2)
                median[np.isfinite(frames).sum(0)<3]=np.nan
                assert np.isfinite(median).tolist()==cal['records'][p]['stereo']['valid_spans']
                entries[arm]=score(basis.predict(basis.solve(median,.025)),part['holdout'],values)
            assert abs(entries['stereo12']['mae_mm']-cal['records'][p]['scores']['stereo12']['mae_mm'])<1e-8
        records[p]=entries
    summary={k:{m:np.mean([r[k][m] for r in records.values() if k in r]) for m in
                ['mae_mm','proportion_mae_pp']} for k in ['uniform_scale','matched_official','stereo12']}
    write(EV/'controls.json',dict(exit_status=0,records=records,summary=summary))
    print(summary,flush=True)

if __name__=='__main__':
    main()
