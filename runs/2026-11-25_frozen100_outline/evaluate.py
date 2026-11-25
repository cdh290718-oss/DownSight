"""Paired evaluation on official held-out joints after sealing all 100 sets of fitted parameters."""
from collections import Counter
import json
import numpy as np
import torch
from compare import GridEye,fit_probe
from shared_run import OUT,EV,ROOT,prev,search,Frame
from common_run import rotation,summary
from contact import PAIRS
from runpy import run_path
gap=run_path(str(ROOT/'runs/2026-11-24_two_stereo_contact/assess.py'))['gap']

METHODS=('v3','matched','outline')
KEYS=['pa_mm','nonpa_mm','outline_px','max_target_depth_mm','crossings_outside_template']


def summarize(rows,keys):
    result={}
    for m in METHODS:
        rr=[rows[k]['methods'][m] for k in keys]
        mean={f:float(np.mean([r[f] for r in rr])) for f in KEYS}
        result[m]=dict(mean,frames=len(rr),strict_target_frames=sum(r['target_crossings']>0 for r in rr),
                       depth_over_2mm_frames=sum(r['max_target_depth_mm']>2 for r in rr),
                       outside_template_frames=sum(r['crossings_outside_template']>0 for r in rr),
                       joint_mm=dict(zip(rr[0]['joint_mm'],np.mean([list(r['joint_mm'].values()) for r in rr],axis=0).tolist())),
                       gt_px=float(np.sqrt(sum(r['gt_sq_px'] for r in rr)/sum(r['n_px'] for r in rr))))
    return result


def paired(rows,keys,first,last):
    out={}
    for metric in KEYS[:3]:
        d=np.array([rows[k]['methods'][last][metric]-rows[k]['methods'][first][metric] for k in keys])
        out[metric]=dict(mean=float(d.mean()),median=float(np.median(d)),improved=int((d<-.01).sum()),
                         worse=int((d>.01).sum()),tied=int((abs(d)<=.01).sum()),
                         largest_increase=[keys[d.argmax()],float(d.max())],largest_decrease=[keys[d.argmin()],float(d.min())])
    flags=[(rows[k]['methods'][first]['target_crossings']>0,rows[k]['methods'][last]['target_crossings']>0) for k in keys]
    out['target_frame_transitions']=dict(new=sum(not a and b for a,b in flags),cleared=sum(a and not b for a,b in flags),
                                        persistent=sum(a and b for a,b in flags))
    return out


def grade():
    torch.set_num_threads(1)
    fit_probe.Eye=GridEye
    prev.base.fit.camera_files=prev.ref.camera_files
    selections={p.stem:prev.read(p) for p in sorted((OUT/'poses').glob('pair*.json'))}
    assert list(selections)==['pair%03d'%n for n in range(1,101)]
    assert all(prev.sha(OUT/'poses'/(k+'.npz'))==r['pose_sha256'] for k,r in selections.items())
    old_path=ROOT/'runs/2026-11-24_v3_contact/evidence'
    old=prev.read(old_path/'test_evaluation.json')
    geo=prev.read(old_path/'verification.json')['splits']['test']
    historical=prev.read(ROOT/'runs/2026-11-24_two_stereo_contact/evidence/evaluation.json')['results']
    target_file=prev.ref.SOURCE/'reference.json'
    assert prev.sha(target_file)==old['truth_sha256'] and prev.sha(prev.TM)==old['manifest_sha256']
    truths=prev.read(target_file)
    images=prev.pair_images(prev.TM)
    model,cam=prev.ref.common.setup_model()
    cfg=prev.read(prev.base.OUT/'protocol.json')
    items={r['pair']:r for r in prev.read(prev.base.OUT/'prepared.json')['items'] if r['condition']=='rw20000'}
    obs=prev.ref.common.OBS
    natives=np.array(prev.ref.common.NATIVE_IDS)[obs]
    adapters=[prev.read(p)['input_adapter'] for p in prev.ref.camera_files()]
    ids=prev.mapping()
    rows={}
    examples={}
    for num,(key,selection) in enumerate(selections.items()):
        item=items[key]; z=dict(np.load(OUT/'poses'/(key+'.npz')))
        x=z['v3']
        orig=dict(body_pose=x[:21],delta=x[21],translation=x[22],global0=z['global0'])
        frame=Frame(key,item,orig,model,cam,cfg)
        target=np.array(truths[key]['gt'])[obs]
        im=images[num]
        assert im[0]['subject']==truths[key]['subject']==selection['subject']
        evaluated={}
        for m in METHODS:
            x=torch.from_numpy(z[m])
            with torch.no_grad():
                v,n,_,ok=frame.state(x)
                assert ok
                full=frame.layer(global_orient=frame.g,body_pose=x[:21].reshape(1,63),betas=frame.beta,
                                 left_hand_pose=x[23:38].reshape(1,45),right_hand_pose=x[38:].reshape(1,45),
                                 jaw_pose=frame.beta[:,:3],leye_pose=frame.beta[:,:3],reye_pose=frame.beta[:,:3],expression=frame.beta)
                joints=full.joints[0]
                pred=((joints-joints[:1])@rotation(x[21]).T+joints[:1]+x[22]).numpy()[natives]
                assert np.max(abs(pred-n[0,natives].numpy()))<1e-5
                edges,per_eye=frame.observed(v)
                hits,active=frame.gate.check(v.numpy())
                counts=summary(hits,frame.labels)
                contact=frame.contact.measure(v.numpy())
                hand_hits={'/'.join(p):counts.get('/'.join(sorted(p)),0) for p in PAIRS}
                views=prev.ref.pixel_errors(item,pred,im,ids,frame.p.cameras,adapters)
                rec=dict(pa_mm=float(np.linalg.norm(prev.align(pred,target)-target,axis=1).mean()*1000),
                         nonpa_mm=float(np.linalg.norm(pred-target,axis=1).mean()*1000),
                         outline_px=edges.item()*12,per_eye_outline_px=per_eye,
                         joint_mm=dict(zip([prev.ref.common.NAMES[j] for j in obs],(1000*np.linalg.norm(pred-target,axis=1)).tolist())),
                         target_crossings=sum(hand_hits.values()),target_counts=hand_hits,
                         max_target_depth_mm=max(d for d,g in contact.values()),sampled_contact_mm=contact,
                         raw_crossings=len(hits),crossings_outside_template=len(active),
                         gt_sq_px=sum(a['sum_sq_px2']['fit_vs_gt'] for a in views),n_px=sum(a['n'] for a in views))
                if key in historical:
                    parts=frame.target.split('/')
                    rec['example_gap_mm']=0. if hand_hits[frame.target] else gap(v.numpy(),frame.faces,frame.labels,parts)
                evaluated[m]=rec
        b=evaluated['v3']; prior=old['rows'][key]['before']
        assert abs(b['pa_mm']-prior['pa'])<.001 and abs(b['nonpa_mm']-prior['nonpa'])<.001
        assert b['target_counts']==geo['rows'][key]['before']
        rows[key]=dict(subject=selection['subject'],methods=evaluated)
        if key in historical:
            now=evaluated['outline'];then=historical[key]['outline']
            segdiff={}
            for side in ['left','right']:
                name=key+'_'+side
                with np.load(ROOT/'outputs/two_stereo_contact_20261124/seg/frames'/name/'probabilities.npz') as pack:
                    delta=np.load(OUT/'seg'/(name+'.npy')).astype(float)-pack['probabilities'][:,2::4,2::4]
                segdiff[side]=float(np.abs(delta).max())
            examples[key]=dict(previous_outline=then,current_outline={k:v for k,v in now.items() if k not in ('per_eye_outline_px','sampled_contact_mm')},
                               sampled_probability_max_difference=segdiff)
        if (num+1)%20==0:print('graded',num+1,flush=True)
    keys=list(rows)
    subjects=sorted({r['subject'] for r in rows.values()})
    prev.write(EV/'results.json',dict(summary=summarize(rows,keys),subjects={s:summarize(rows,[k for k in keys if rows[k]['subject']==s]) for s in subjects},
          paired={k:paired(rows,keys,k,'outline') for k in ('v3','matched')},examples=examples,rows=rows,
          excluding_pair003_pair004=summarize(rows,[k for k in keys if k not in ('pair003','pair004')]),
          previous_collision_only=dict(summary=old['summary']['after'],strict_frames=geo['strict_crossing_frames']['after']),
          reference_sha256=prev.sha(target_file),manifest_sha256=prev.sha(prev.TM),exit_status=0))
    prev.write(EV/'selections.json',dict(rows=selections,exit_status=0))


if __name__=='__main__':grade()
