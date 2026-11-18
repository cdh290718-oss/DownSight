"""Check paired denominators, published metrics, and PA transform convention."""
import ast
import numpy as np
from common import ROOT,RUN,OUT,EVIDENCE,read,write,sha
from evaluate import align

def main():
    for p in RUN.glob('*.py'):ast.parse(p.read_text())
    items=read(OUT/'prepared.json')['items'];groups={}
    for item in items:groups.setdefault(item['base_pair'],[]).append(item)
    assert len(groups)==100 and len(items)==400
    for group in groups.values():
        assert len(group)==4
        for item in group:
            assert item['anchor']==group[0]['anchor'] and item['inputs']==group[0]['inputs']
            assert [p['valid'] for p in item['points']]==[p['valid'] for p in group[0]['points']]
            assert not any(p['accepted'] for p in item['points'])
    # Known proper rotation, scale and translation must disappear under PA.
    x=np.random.RandomState(18).normal(size=(15,3))
    R=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
    y=1.7*x@R.T+np.array([.2,-.4,2.])
    pa_error=float(np.abs(align(x,y)-y).max());assert pa_error<1e-12
    records=read(EVIDENCE/'results.json')['records'];summary=read(EVIDENCE/'summary.json')
    assert len(records)==400
    bymodel={m:{r['pair']:r for r in records if r['model']==m and r['status']=='ok'} for m in summary['summary']}
    common=set.intersection(*[set(v) for v in bymodel.values()]);assert len(common)==summary['common_success_pairs']
    for m,d in bymodel.items():
        assert len(d)==summary['summary'][m]['success']
        for k,v in summary['paired'][m].items():assert abs(v-np.mean([d[p][k] for p in common]))<1e-9
        for rec in d.values():
            assert abs(np.mean(rec['absolute_per_joint_mm'])-rec['absolute_mpjpe_mm'])<1e-9
            assert abs(np.mean(rec['pa_per_joint_mm'])-rec['pa_mpjpe_mm'])<1e-9
    reference_fit=ROOT/'runs/2026-11-17_newdetector_d10/fit.py'
    expected=reference_fit.read_text().replace("'oracle_shape.json']", "'oracle_shape.json', 'reference.json', 'unrealego_rw_test_snapshot11_root_cam.npz']")
    assert (RUN/'fit.py').read_text()==expected
    assert sha(RUN/'geometry.py')==sha(ROOT/'runs/2026-11-17_newdetector_d10/geometry.py')
    write(EVIDENCE/'verification.json',dict(command='<BODY_PYTHON> verify.py',exit_status=0,shared_input_pairs=len(groups),total_records=len(records),common_success_pairs=len(common),pa_known_similarity_max_error=pa_error,reference_fit_only_change='deny opening reference.json and head network cache',geometry_matches_reference=True,all_aggregate_metrics_recomputed=True,all_failure_records_preserved=True,syntax_valid=True))
    print('Verification passed')

if __name__=='__main__':main()
