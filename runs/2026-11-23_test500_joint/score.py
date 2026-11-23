"""Score the same 12 official left-camera landmarks after all choices have been written."""
from body import RUN,ROOT,FIT,OUT,SOURCE,SHARDS,shard,ref,read,write,sha,np,torch
METHODS=('baseline','fixed','joint')

def summarize(rows):
    return {k:{metric:float(np.mean([r['scores'][k][metric] for r in rows]))
               for metric in ('nonpa_mm','pa_mm')} for k in METHODS}

def main():
    torch.set_num_threads(1)
    meta={r['pair']:r for r in read(OUT/'manifest.json')}
    truth=np.load(SOURCE)['keypoints3d']
    model,_=ref.common.setup_model();layer=model.layer['neutral'].cpu().eval()
    obs=ref.common.OBS;ids=np.array(ref.common.NATIVE_IDS)[obs]
    records={};failures=[];hashes={};fit_times=[]
    for n in range(SHARDS):
        dest=shard(n)
        cfg=read(dest/'fit_complete.json');assert cfg['steps']==20000
        fit_times.append(cfg['elapsed_seconds'])
        pack=dict(np.load(dest/'selected.npz'));choices=read(dest/'selected.json')
        hashes[str(n)]={p:sha(dest/p) for p in ('budget_20000.npz','selected.npz','selected.json')}
        for c in choices:
            key=c['pair']
            if 'failed' in c:failures.append(c);continue
            m=meta[key];gt=truth[m['indices'][0],obs].astype(float)
            scores={}
            for j,k in enumerate(METHODS):
                pred=ref.full_joints(pack,layer,c['offset']+j)[ids]
                absolute,pa=ref.joint_errors(pred,gt)
                assert np.isfinite([absolute,pa]).all()
                scores[k]=dict(nonpa_mm=absolute,pa_mm=pa)
            assert c['losses'][2]<=c['losses'][1]<=c['losses'][0]
            assert key not in records
            records[key]=dict(subject=m['subject'],scores=scores,selected=c['selected'],losses=c['losses'])
    assert len(records)+len(failures)==500 and set(records)|{c['pair'] for c in failures}==set(meta)
    groups={s:[r for r in records.values() if r['subject']==s] for s in ('S1','S9','S11')}
    summary=summarize(list(records.values()))
    counts={}
    for m,other in [('fixed','baseline'),('joint','fixed')]:
        counts[m]={}
        for metric in ('nonpa_mm','pa_mm'):
            ds=np.array([r['scores'][m][metric]-r['scores'][other][metric] for r in records.values()])
            counts[m][metric]=dict(improved=int(sum(ds<-.01)),worsened=int(sum(ds>.01)),tied=int(sum(abs(ds)<=.01)),
                largest_improvement_mm=float(-min(ds)),largest_worsening_mm=float(max(ds)),mean_delta_mm=float(ds.mean()),
                median_delta_mm=float(np.median(ds)),mean_delta_excluding_largest_gain_mm=float(np.delete(ds,ds.argmin()).mean()))
    write(RUN/'evidence/results.json',dict(n_requested=500,n_scored=len(records),failures=failures,summary=summary,
          subjects={s:dict(n=len(rr),summary=summarize(rr)) for s,rr in groups.items()},counts=counts,
          names=[ref.common.NAMES[i] for i in obs],shard_hashes=hashes,fit_seconds_by_shard=fit_times,
          rows=records,exit_status=0))
    print(summary,counts,flush=True)
if __name__=='__main__':main()
