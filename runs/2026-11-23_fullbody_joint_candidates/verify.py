"""Re-pool public frame statistics without weights, data or third-party packages."""
import hashlib
import json
from math import sqrt, isclose
from pathlib import Path

RUN = Path(__file__).resolve().parent
EVIDENCE = RUN/'evidence'
def read(path):
    return json.loads(path.read_text())

def verify():
    for split,expected in [('test',100),('validation',200)]:
        search,results,ablation = [read(EVIDENCE/(split+'_'+suffix+'.json')) for suffix in ('search','results','ablation')]
        rows = results['rows']
        assert len(rows)==expected==search['pairs']==len(ablation['rows'])
        assert search['choices_sha256']==results['choices_sha256']
        assert search['source_sha256']==hashlib.sha256((RUN/'search.py').read_bytes()).hexdigest()
        for method in list(results['summary']['all'])+['expanded_arms']:
            if method=='expanded_arms':
                frames = [dict(v,n_px=rows[k]['scores']['baseline']['n_px']) for k,v in ablation['rows'].items()]
                totals = ablation[method]
            else:
                frames = [r['scores'][method] for r in rows.values()]
                totals = results['summary']['all'][method]
            for metric in ('mpjpe','rmse','pa','wrist_rmse','other10','px'):
                if metric=='px':
                    pooled = sqrt(sum(r['n_px']*r[metric]**2 for r in frames)/sum(r['n_px'] for r in frames))
                elif metric in ('rmse','wrist_rmse'):
                    pooled = sqrt(sum(r[metric]**2 for r in frames)/expected)
                else:
                    pooled = sum(r[metric] for r in frames)/expected
                assert isclose(pooled,totals[metric],rel_tol=1e-10), (method,metric,pooled,totals[metric])
        for r in rows.values():
            s = r['scores']
            assert s['joint']['loss']<=min(s[k]['loss'] for k in ('baseline','wrist','fixed'))
            assert s['oracle']['mpjpe'] <= s['joint']['mpjpe']+1e-8
        print(split,expected,'paired results verified')

if __name__=='__main__':
    verify()
