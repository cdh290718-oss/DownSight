"""Evaluate the sealed validation fits using the same full contour and official 12-joint reference."""
from collections import Counter
import numpy as np
import torch
from inputs import EdgeFrame, EV, OUT, ROOT, prev, search
from common_run import rotation, summary
from contact import PAIRS
from inspect_obj import self_intersections

MODES = ('v3', 'fixed_full', 'palm')
DISTANCES = ('pa_mm', 'nonpa_mm', 'full_outline_px')
SPOT = {'val001', 'val031', 'val055', 'val100', 'val200'}


def aggregate(rows):
    result = {}
    for mode in MODES:
        values = [r['methods'][mode] for r in rows]
        result[mode] = {m: float(np.mean([v[m] for v in values])) for m in DISTANCES}
        result[mode].update(
            frames=len(values),
            self_frames=sum(bool(v['left_self'] or v['right_self']) for v in values),
            target_frames=sum(any(v['target_counts'].values()) for v in values),
            by_region_frames={p: sum(v['target_counts'][p] > 0 for v in values)
                              for p in values[0]['target_counts']},
            finger_max_delta_rad=max(v['finger_max_delta_rad'] for v in values))
    return result


def changes(rows, mode, ref):
    out = {}
    for name in DISTANCES:
        diff = np.array([r['methods'][mode][name] - r['methods'][ref][name] for r in rows])
        out[name] = dict(mean=float(diff.mean()), median=float(np.median(diff)),
                         improved=int((diff < -.01).sum()), worsened=int((diff > .01).sum()),
                         within_0p01=int((np.abs(diff) <= .01).sum()))
    transitions = Counter()
    for r in rows:
        a, b = [any(r['methods'][m]['target_counts'].values()) for m in (ref, mode)]
        transitions[{(True, True):'persisted', (True,False):'resolved',
                     (False,True):'new', (False,False):'clear_both'}[a,b]] += 1
    out['crossings'] = dict(transitions)
    return out


def main():
    torch.set_num_threads(1)
    choices = prev.read(search.OUT/'validation_choices.json')
    keys = sorted(choices)
    seg = prev.read(EV/'segmentation.json')['eyes']
    selections = {}
    for key in keys:
        path = OUT/'poses'/(key+'.npz')
        rec = prev.read(path.with_suffix('.json'))
        assert rec['exit_status'] == 0 and rec['sha256'] == prev.sha(path)
        for side in ('left','right'):
            name = key+'_'+side
            assert rec['seg_sha256'][name] == seg[name]['sampled_sha256'] == prev.sha(OUT/'seg'/(name+'.npy'))
        selections[key] = rec
    assert len(keys) == 200 and len(seg) == 400
    prev.write(EV/'selections.json', selections)
    previous = ROOT/'runs/2026-11-24_v3_contact/evidence'
    old = prev.read(previous/'validation_evaluation.json')
    geom = prev.read(previous/'verification.json')['splits']['validation']['rows']
    target_path = prev.OUT/'validation_truth.json'
    assert prev.sha(target_path) == old['truth_sha256']
    truth = prev.read(target_path)
    model, cam = prev.ref.common.setup_model()
    items = prev.read(prev.VAL/'prepared.json')['items']
    cfg = prev.read(prev.VAL/'protocol.json')
    obs = prev.ref.common.OBS
    natives = np.asarray(prev.ref.common.NATIVE_IDS)[obs]
    rows, checks = {}, {}
    max_delta = {'pa_mm':0., 'nonpa_mm':0.}
    counts_disagree = {}
    support = {m: Counter() for m in MODES[1:]}
    for key in keys:
        z = dict(np.load(OUT/'poses'/(key+'.npz')))
        baseline = z['v3']
        orig = dict(body_pose=baseline[:21], delta=baseline[21], translation=baseline[22], global0=z['global0'])
        frame = EdgeFrame(key,items[choices[key]['index']],orig,model,cam,cfg)
        assert np.array_equal(frame.v3.numpy(),baseline)
        target = np.asarray(truth[key]['gt'])[obs]
        assert old['rows'][key]['subject'] == truth[key]['subject']
        row = dict(subject=truth[key]['subject'],methods={})
        for mode in MODES:
            x = torch.from_numpy(z[mode])
            assert torch.equal(x[23:],frame.v3[23:])
            with torch.no_grad():
                v,n,_,ok = frame.state(x)
                assert ok
                zeros = frame.beta[:,:3]
                layer = frame.layer(global_orient=frame.g,betas=frame.beta,body_pose=x[:21].reshape(1,63),
                                    left_hand_pose=x[23:38].reshape(1,45),right_hand_pose=x[38:].reshape(1,45),
                                    jaw_pose=zeros,leye_pose=zeros,reye_pose=zeros,expression=frame.beta)
                joints = layer.joints[0]
                pred = ((joints-joints[:1])@rotation(x[21]).T+joints[:1]+x[22]).numpy()[natives]
                assert np.max(np.abs(pred-n[0,natives].numpy())) < 1e-5
                hits,active = frame.gate.check(v.numpy())
                counts = summary(hits,frame.labels)
                hands = {'/'.join(p):counts.get('/'.join(sorted(p)),0) for p in PAIRS}
                measured = dict(pa_mm=float(1000*np.linalg.norm(prev.align(pred,target)-target,axis=1).mean()),
                                nonpa_mm=float(1000*np.linalg.norm(pred-target,axis=1).mean()),
                                full_outline_px=frame.observed(v)[0].item()*12, target_counts=hands,
                                left_self=counts.get('L_hand/L_hand',0),right_self=counts.get('R_hand/R_hand',0),
                                outside_template_face_pairs=len(active),finger_max_delta_rad=float(abs(x[23:]-frame.v3[23:]).max()))
                if key in SPOT:
                    independent = {p:len(self_intersections(v.numpy().astype(float),frame.faces[frame.labels==p]))
                                   for p in ('L_hand','R_hand')}
                    assert all(independent[p] == counts.get(p+'/'+p,0) for p in independent)
                    checks.setdefault(key,{})[mode] = independent
            row['methods'][mode] = measured
            if mode == 'v3':
                for metric in max_delta:
                    hist = old['rows'][key]['before'][metric.replace('_mm','')]
                    max_delta[metric] = max(max_delta[metric],abs(hist-measured[metric]))
                if hands != geom[key]['before']:
                    counts_disagree[key] = dict(rebuilt=hands, archive=geom[key]['before'])
            else:
                for eye in selections[key]['methods'][mode]['support']:
                    hands = [eye.get(p,(0,0)) for p in ('L_hand','R_hand')]
                    support[mode].update(eyes_with_hand=int(any(b>=6 for a,b in hands)),eyes=1,
                                         retained_hand_samples=sum(b for a,b in hands if b>=6),
                                         original_hand_samples=sum(a for a,b in hands))
        rows[key] = row
        if len(rows)%20 == 0: print('graded',len(rows),flush=True)
    result = dict(summary=aggregate(rows.values()),
                  subjects={s:aggregate([r for r in rows.values() if r['subject']==s])
                            for s in sorted({r['subject'] for r in rows.values()})},
                  paired={m+'_vs_'+b:changes(rows.values(),m,b)
                          for m,b in [('fixed_full','v3'),('palm','v3'),('palm','fixed_full')]},
                  support=support,reference_sha256=prev.sha(target_path),baseline_max_error_difference_mm=max_delta,
                  baseline_crossing_count_disagreements=counts_disagree,independent_hand_checks=checks,rows=rows,exit_status=0)
    prev.write(EV/'results.json',result)
    assert max(max_delta.values()) < .001 and not counts_disagree, (max_delta,counts_disagree)


if __name__ == '__main__':
    main()
