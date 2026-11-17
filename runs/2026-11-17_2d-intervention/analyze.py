"""Paired joint effects and descriptive sequence-cluster resampling."""
import csv

import numpy as np

from common import EVIDENCE, NAMES, OUT, RUN, read, sha, write


def main():
    cfg=read(RUN/'protocol.json');stats=cfg['statistics'];pairs=cfg['pairs']
    raw=read(EVIDENCE/'results.json')['records']
    index={(r['method'],r['budget'],r['condition'],r['pair']):r for r in raw}
    assert len(index)==len(raw)==len(cfg['methods'])*len(cfg['budgets'])*len(cfg['conditions'])*len(pairs)
    failed=[r for r in raw if r['status']!='ok']
    assert not failed, 'Do not silently drop failures from paired inference.'
    conditions=cfg['conditions'][1:]
    rng=np.random.default_rng(stats['bootstrap_seed'])
    # Resample whole sequence rows, preserving all joints and interventions together.
    draws=rng.integers(0,len(pairs),(stats['bootstrap_draws'],len(pairs)))
    summaries=[];sensitivity=[]
    for method in cfg['methods']:
        for budget in cfg['budgets']:
            for metric in ['absolute','pa']:
                baseline=np.array([index[method,budget,'baseline',p][metric+'_per_joint_mm'] for p in pairs])
                corrected=np.array([[index[method,budget,c,p][metric+'_per_joint_mm'] for p in pairs] for c in conditions])
                effects=baseline[None]-corrected
                means=effects.mean(1)
                boot=effects[:,draws,:].mean(2).transpose(1,0,2)
                lower,upper=np.quantile(boot,[.025,.975],axis=0)
                # Centered bootstrap max-absolute-error band across 8 x 15 cells.
                radius=float(np.quantile(np.abs(boot-means[None]).max(axis=(1,2)),stats['coverage']))
                for k,condition in enumerate(conditions):
                    rows=[index[method,budget,condition,p] for p in pairs]
                    base_rows=[index[method,budget,'baseline',p] for p in pairs]
                    q=dict(method=method,budget=budget,condition=condition,metric=metric,
                           baseline_mean_mm=float(baseline.mean()),corrected_mean_mm=float(corrected[k].mean()),
                           mean_improvement_mm=float(effects[k].mean()),
                           per_joint_baseline_mm=baseline.mean(0).tolist(),
                           per_joint_corrected_mm=corrected[k].mean(0).tolist(),
                           per_joint_improvement_mm=means[k].tolist(),
                           per_sequence_improvement_mm=effects[k].tolist(),
                           mean_joint_improved_count=int((means[k]>stats['practical_change_mm']).sum()),
                           mean_joint_worsened_count=int((means[k]<-stats['practical_change_mm']).sum()),
                           strict_mean_joint_improved_count=int((means[k]>0).sum()),
                           per_joint_sample_improved_count=(effects[k]>stats['practical_change_mm']).sum(0).tolist(),
                           per_joint_sample_worsened_count=(effects[k]<-stats['practical_change_mm']).sum(0).tolist(),
                           per_joint_pointwise_ci95_lower_mm=lower[k].tolist(),
                           per_joint_pointwise_ci95_upper_mm=upper[k].tolist(),
                           simultaneous_radius_mm=radius,
                           per_joint_simultaneous_ci95_lower_mm=(means[k]-radius).tolist(),
                           per_joint_simultaneous_ci95_upper_mm=(means[k]+radius).tolist(),
                           all_joints_practically_improve_with_simultaneous_support=bool((means[k]-radius>stats['practical_change_mm']).all()),
                           every_sample_every_joint_practically_improves=bool((effects[k]>stats['practical_change_mm']).all()),
                           joint_sample_improved_count=int((effects[k]>stats['practical_change_mm']).sum()),
                           joint_sample_worsened_count=int((effects[k]<-stats['practical_change_mm']).sum()),
                           maximum_joint_sample_worsening_mm=float(max(0.,-effects[k].min())),
                           maximum_joint_sample_improvement_mm=float(effects[k].max()),
                           baseline_convergence_pass=sum(r['convergence_pass'] for r in base_rows),
                           corrected_convergence_pass=sum(r['convergence_pass'] for r in rows))
                    summaries.append(q)
        for condition in conditions:
            a=next(q for q in summaries if q['method']==method and q['condition']==condition and q['metric']=='absolute' and q['budget']==cfg['budgets'][0])
            b=next(q for q in summaries if q['method']==method and q['condition']==condition and q['metric']=='absolute' and q['budget']==cfg['budgets'][1])
            sensitivity.append(dict(method=method,condition=condition,
                                    standard_effect_mm=a['per_joint_improvement_mm'],extended_effect_mm=b['per_joint_improvement_mm'],
                                    joint_effect_sign_changes=[NAMES[j] for j in range(15)
                                        if np.sign(a['per_joint_improvement_mm'][j])!=np.sign(b['per_joint_improvement_mm'][j])],
                                    mean_improvement_standard_mm=a['mean_improvement_mm'],
                                    mean_improvement_extended_mm=b['mean_improvement_mm']))
    write(EVIDENCE/'analysis.json',dict(command='<BODY_PYTHON> runs/2026-11-17_2d-intervention/analyze.py',exit_status=0,
                                      source_sha256=sha(EVIDENCE/'results.json'),joint_names=NAMES,
                                      pair_order=pairs,statistics=stats,summaries=summaries,budget_sensitivity=sensitivity))
    for method in cfg['methods']:
        for budget in cfg['budgets']:
            rows=[q for q in summaries if q['method']==method and q['budget']==budget and q['metric']=='absolute']
            path=EVIDENCE/('effects_'+method+'_'+str(budget)+'.csv')
            with path.open('w',newline='') as f:
                writer=csv.writer(f,lineterminator='\n');writer.writerow(['2d_intervention']+NAMES)
                for q in rows:writer.writerow([q['condition']]+q['per_joint_improvement_mm'])
    plot(summaries,cfg)
    for q in summaries:
        if q['metric']=='absolute':
            print(q['method'],q['budget'],q['condition'],
                  round(q['baseline_mean_mm'],2),'->',round(q['corrected_mean_mm'],2),
                  'improved/worsened joints',q['mean_joint_improved_count'],q['mean_joint_worsened_count'],flush=True)


def plot(summaries,cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    assets=RUN/'assets';assets.mkdir(exist_ok=True)
    selected=[q for q in summaries if q['metric']=='absolute' and q['budget']==cfg['budgets'][0]]
    maximum=max(abs(v) for q in selected for v in q['per_joint_improvement_mm'])
    fig,axes=plt.subplots(2,1,figsize=(15,9),layout='constrained')
    for ax,method in zip(axes,cfg['methods']):
        rows=[q for q in selected if q['method']==method]
        values=np.array([q['per_joint_improvement_mm'] for q in rows])
        image=ax.imshow(values,cmap='RdBu',norm=TwoSlopeNorm(vmin=-maximum,vcenter=0,vmax=maximum),aspect='auto')
        ax.set_xticks(range(15),NAMES,rotation=40,ha='right')
        ax.set_yticks(range(len(rows)),[q['condition'] for q in rows])
        ax.set_title(method+' | baseline error minus corrected error (mm); positive = better')
        ax.set_ylabel('2D coordinates corrected')
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j,i,f'{values[i,j]:+.0f}',ha='center',va='center',fontsize=8,
                        color='white' if abs(values[i,j])>maximum*.55 else 'black')
    fig.colorbar(image,ax=axes,label='Mean unaligned 3D improvement (mm)',shrink=.8)
    fig.suptitle('10 fixed poses / 10 sequences / one subject S9 | 20,000-step budget\nLow-confidence detections retained; routing, masks, weights and initialization fixed')
    fig.savefig(assets/'joint_effects.png',dpi=170)
    plt.close(fig)


if __name__=='__main__':
    main()
