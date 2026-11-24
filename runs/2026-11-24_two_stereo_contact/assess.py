"""Evaluate the sealed selections and cross-check every intersection using the archived checker."""
import hashlib, subprocess, sys
from pathlib import Path
import numpy as np
import torch
import open3d as o3d
from common_run import ROOT,RUN,OUT,EV,PAIRS,inputs,prev,rotation,summary
from fit_probe import Frame
from inspect_obj import self_intersections


def gap(v,faces,labels,names):
    def directed(source,target):
        dst=faces[labels==target]
        src=faces[labels==source]
        samples=np.vstack((v[np.unique(src)],v[src].mean(1)))
        scene=o3d.t.geometry.RaycastingScene(nthreads=1)
        scene.add_triangles(o3d.core.Tensor(v),o3d.core.Tensor(dst.astype(np.uint32)))
        return float(scene.compute_distance(o3d.core.Tensor(samples),nthreads=1).numpy().min()*1000)
    return min(directed(*names),directed(*names[::-1]))


def main():
    torch.set_num_threads(1)
    model,cam,cfg,data=inputs()
    baseline=prev.read(ROOT/'runs/2026-11-24_v3_contact/evidence/test_evaluation.json')
    truthfile=prev.ref.SOURCE/'reference.json'
    assert prev.sha(truthfile)==baseline['truth_sha256']
    truths=prev.read(truthfile)
    obs=prev.ref.common.OBS
    natives=np.array(prev.ref.common.NATIVE_IDS)[obs]
    records={}
    for pair,(item,orig) in data.items():
        frame=Frame(pair,item,orig,model,cam,cfg)
        archived=prev.read(EV/(pair+'_final.json'))
        poses=OUT/(pair+'_final.npz')
        assert prev.sha(poses)==archived['final_sha256']
        target=np.asarray(truths[pair]['gt'])[obs]
        modes={}
        with np.load(poses) as states,torch.no_grad():
            for mode,x in states.items():
                x=torch.from_numpy(x)
                v,native,_,_=frame.state(x)
                # Check the full SMPL-X regressed joints against the solver's analytic FK.
                zeros=frame.beta[:,:3]
                full=frame.layer(global_orient=frame.g,body_pose=x[:21].reshape(1,63),betas=frame.beta,
                                 left_hand_pose=x[23:38].reshape(1,45),right_hand_pose=x[38:].reshape(1,45),
                                 jaw_pose=zeros,leye_pose=zeros,reye_pose=zeros,expression=frame.beta)
                joints=full.joints[0]
                pred=((joints-joints[:1])@rotation(x[21]).T+joints[:1]+x[22]).numpy()[natives]
                assert np.max(abs(pred-native[0,natives].numpy()))<1e-5
                row=archived['rows'][mode]['result']
                hits=np.array(self_intersections(v.numpy().astype(float),frame.faces),dtype=int).reshape(-1,2)
                assert summary(hits,frame.labels)==row['crossings']
                parts=frame.target.split('/')
                intersects=row['crossings'].get('/'.join(sorted(parts)),0)
                rec=dict(pa_mm=float(np.linalg.norm(prev.align(pred,target)-target,axis=1).mean()*1000),
                         nonpa_mm=float(np.linalg.norm(pred-target,axis=1).mean()*1000),
                         target_crossings=intersects,raw_crossings=len(hits),active_crossings=row['active_triangles'],
                         target_sample_depth_mm=row['target_depth_gap_mm'][0],
                         target_sample_gap_mm=0. if intersects else gap(v.numpy(),frame.faces,frame.labels,parts),
                         outline_px=row['outline_loss']*12,
                         official_target_used_only_after_selection=True)
                if mode=='v3':
                    reference=baseline['rows'][pair]['before']
                    assert abs(rec['pa_mm']-reference['pa'])<.001 and abs(rec['nonpa_mm']-reference['nonpa'])<.001
                modes[mode]=rec
        records[pair]=modes
    manifest=prev.read(OUT/'seg/manifest.json')
    seg={k:manifest[k] for k in ['checkpoint_sha256','config_sha256','strict_load','device','precision','preprocessing','postprocessing']}
    seg['frames']=[{k:r[k] for k in ['id','image_sha256','original_hw','parts']} for r in manifest['frames']]
    seg['prediction_hashes']={r['id']:prev.sha(OUT/'seg/frames'/r['id']/'probabilities.npz') for r in manifest['frames']}
    prev.write(EV/'evaluation.json',dict(results=records,segmentation=seg,reference_sha256=prev.sha(truthfile),
               truth_manifest_sha256=baseline['manifest_sha256'],exit_status=0))
    print(records)


if __name__=='__main__':
    main()
