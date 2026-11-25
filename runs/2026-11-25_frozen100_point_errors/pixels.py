"""Audit per-pair RW20000 detection residuals in original image pixels."""
import numpy as np
from refine import RUN,ROOT,prev

CN={'Shoulder':'肩','Elbow':'肘','Wrist':'腕','Hip':'髋','Knee':'膝','Ankle':'踝'}
def name(j):
    s,k=j.split('_')
    return ('左' if s=='L' else '右')+CN[k]

def main():
    ref=prev.ref
    archived=ref.read(ROOT/'runs/2026-11-20_rw20000_joint_rmse/evidence/results.json')
    path=ref.OUT/'prepared.json'
    files=[path,*ref.camera_files()]
    assert all(ref.sha(f)==archived['input_sha256'][str(f.relative_to(ROOT))] for f in files)
    rows={v['pair']:v for v in ref.read(path)['items'] if v['condition']=='rw20000'}
    images,ids=ref.reference_images()
    joints=[ref.common.NAMES[i] for i in ref.common.OBS]
    assert set(images)==set(rows) and len(rows)==100
    adapters=[ref.read(c)['input_adapter'] for c in ref.camera_files()]
    mat=np.full((100,12,2),np.nan)
    cases={};pairs=sorted(rows)
    for n,k in enumerate(pairs):
        pts=rows[k]['points']
        assert [p['name'] for p in pts]==joints
        for eye,(im,ad) in enumerate(zip(images[k],adapters)):
            assert im['view']==('left','right')[eye]
            uv=(np.array([p['xy'][eye] for p in pts])-ad['offset'])/ad['scale']
            gt=np.array(im['keypoints308_xy'])[ids]
            mask=np.array(im['keypoints308_supervision_mask'])[ids].astype(bool)
            assert np.array_equal(mask,[p['valid'][eye] for p in pts])
            mat[n,mask,eye]=np.linalg.norm(uv[mask]-gt[mask],axis=1)
        x=mat[n];m=x>20
        cases[k]=dict(subject=images[k][0]['subject'],rmse=float(np.sqrt(np.nanmean(x*x))),
            any20=int(m.any(1).sum()),both20=int(m.all(1).sum()),
            errors={j:[float(v) if np.isfinite(v) else None for v in u] for j,u in zip(joints,x)})
    stats={}
    for i,j in enumerate(joints):
        x=mat[:,i];m=x>20
        idx,eye=np.unravel_index(np.nanargmax(x),x.shape)
        rmse=float(np.sqrt(np.nanmean(x*x)))
        assert abs(rmse-archived['joints'][j]['input_rmse_px'])<1e-6
        stats[j]=dict(rmse=rmse,median=float(np.nanmedian(x)),obs=int(np.isfinite(x).sum()),any20=int(m.any(1).sum()),
                      both20=int(m.all(1).sum()),over20=int(m.sum()),max_px=float(x[idx,eye]),max_pair=pairs[idx],max_eye=['左目','右目'][eye])
    exceed=mat>20
    others=[i for i,j in enumerate(joints) if j.split('_')[1] not in ['Elbow','Wrist']]
    overall=dict(rmse=float(np.sqrt(np.nanmean(mat**2))),valid=int(np.isfinite(mat).sum()),obs20=int(exceed.sum()),
        any20=int(exceed.any((1,2)).sum()),both20=int(exceed.all(2).any(1).sum()),
        other_any=int(exceed[:,others,:].any((1,2)).sum()),other_only=int((exceed[:,others,:].any((1,2)) & ~exceed[:,[i for i in range(12) if i not in others],:].any((1,2))).sum()))
    assert overall['valid']==2398 and abs(overall['rmse']-archived['overall']['input_rmse_px'])<1e-6
    prev.write(RUN/'evidence/pixels.json',dict(overall=overall,joints=stats,cases=cases,manifest_sha256=archived['manifest_sha256'],
              inputs={f.name:ref.sha(f) for f in files},exit_status=0))
    lines=['# 冻结 100 对逐帧关节偏差','', '按大偏差关节数降序，同数量按全身二维 RMSE 降序。括号内是原图左目 / 右目像素距离。列出至少一眼大于 20 px 的全部关节。右腕 pair073 双眼均无有效标注。全部有效观测见[数值](evidence/pixels.json)。','',
           '| 帧 | 人物 | 全身 RMSE px | 超线关节数 | 两眼同超数 | 关节（左目 / 右目 px） |','|---|---|---:|---:|---:|---|']
    for k,v in sorted(cases.items(),key=lambda kv:(-kv[1]['any20'],-kv[1]['rmse'])):
        hits=sorted([(j,a,b) for j,(a,b) in v['errors'].items() if a is not None and max(a,b)>20],key=lambda u:-max(u[1:]))
        lst='、'.join(f'{name(j)} {a:.1f}/{b:.1f}' for j,a,b in hits) or '—'
        lines.append(f"| {k} | {v['subject']} | {v['rmse']:.2f} | {v['any20']} | {v['both20']} | {lst} |")
    (RUN/'pairs.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    main()
