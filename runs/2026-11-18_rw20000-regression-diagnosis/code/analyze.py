"""Paired descriptive analysis; no training or detector inference."""
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

RUN=Path(__file__).resolve().parents[1];ROOT=RUN.parents[1]
OUT=ROOT/'outputs/rw20000_regression_diagnosis_20261118'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def summary(rows,names):
    a=np.array([r['old'] for r in rows],float);b=np.array([r['new'] for r in rows],float)
    assert np.array_equal(np.isnan(a),np.isnan(b))
    return dict(images=len(rows),old_mean_px=float(np.nanmean(a)),new_mean_px=float(np.nanmean(b)),
        improved_images=int((np.nanmean(b,axis=1)<np.nanmean(a,axis=1)).sum()),
        worsened_images=int((np.nanmean(b,axis=1)>np.nanmean(a,axis=1)).sum()),
        joints={n:dict(old_px=float(np.nanmean(a[:,j])),new_px=float(np.nanmean(b[:,j]))) for j,n in enumerate(names)})

def overlay(path,old,new,gt,ids,title,target):
    with Image.open(path) as im:im=im.convert('RGB');d=ImageDraw.Draw(im)
    # Comparison in original pixel coordinates. Cross: reference; circle: 5k; square: 20k.
    for name,j in ids.items():
        for xy,color,shape in [(gt[j],'yellow','cross'),(old[j],'cyan','circle'),(new[j],'magenta','square')]:
            x,y=xy
            if shape=='cross':d.line((x-7,y,x+7,y),fill=color,width=2);d.line((x,y-7,x,y+7),fill=color,width=2)
            elif shape=='circle':d.ellipse((x-6,y-6,x+6,y+6),outline=color,width=2)
            else:d.rectangle((x-5,y-5,x+5,y+5),outline=color,width=2)
        x,y=gt[j];d.text((x+8,y+7),name,fill='yellow')
    canvas=Image.new('RGB',(im.width,im.height+45));canvas.paste(im,(0,45));dd=ImageDraw.Draw(canvas)
    dd.text((8,5),title,fill='white');dd.text((8,22),'Yellow + reference; cyan circle RW5000; magenta square RW20000',fill='white')
    canvas.save(target)

def main():
    ep=ROOT/'runs/2026-11-18_rw-frozen200-evaluation/evidence/2d_errors.json'
    npth=ROOT/'runs/2026-11-18_rw20000_frozen200/evidence/2d_errors.json'
    old=read(ep);new=read(npth);assert old['joint_ids']==new['joint_ids']
    names=old['joint_names'];ids=old['joint_ids'];a={r['id']:r for r in old['records'] if r['version']=='rw5000'}
    b={r['id']:r for r in new['records'] if r['version']=='rw20000'};assert set(a)==set(b) and len(a)==200
    rows=[]
    for ident in a:
        assert [x is None for x in a[ident]['per_joint_error_px']]==[x is None for x in b[ident]['per_joint_error_px']]
        row={k:a[ident][k] for k in ['id','subject','sequence','pair_id']}
        row.update(old=a[ident]['per_joint_error_px'],new=b[ident]['per_joint_error_px']);rows.append(row)
    bysubject={u:summary([r for r in rows if r['subject']==u],names) for u in sorted({r['subject'] for r in rows})}
    byseq={u:summary([r for r in rows if r['sequence']==u],names) for u in sorted({r['sequence'] for r in rows})}
    def delta(r):return float(np.nanmean(np.array(r['new'],float))-np.nanmean(np.array(r['old'],float)))
    worst=sorted(rows,key=delta,reverse=True)[:8]
    write(RUN/'evidence/frozen200_analysis.json',dict(command='python code/analyze.py',exit_status=0,
        sources_sha256={str(ep.relative_to(ROOT)):sha(ep),str(npth.relative_to(ROOT)):sha(npth)},
        overall=summary(rows,names),by_subject=bysubject,by_sequence=byseq,
        worst_image_regressions=[dict(id=r['id'],subject=r['subject'],sequence=r['sequence'],mean_increase_px=delta(r),old=r['old'],new=r['new']) for r in worst],
        scope='Descriptive paired analysis of already evaluated correlated images; no model selection.'))
    prep=read(ROOT/'outputs/rw_a20000_evaluation_20261118/d_fit/prepared.json')['items']
    reference=read(ROOT/'outputs/2d_intervention_20261117_retry1/prepared.json')['items']
    gt={r['pair']:{p['name']:p for p in r['points']} for r in reference if r['condition']=='all_full'}
    result=[]
    for row in prep:
        if row['condition'] not in ['rw5000','rw20000']:continue
        points={}
        for p in row['points']:
            error=np.linalg.norm(np.array(p['xy'])-np.array(gt[row['pair']][p['name']]['xy']),axis=-1)/(720/872)
            points[p['name']]=dict(error_px=error.tolist(),scores=p['scores'],valid=p['valid'])
        result.append(dict(pair=row['pair'],version=row['condition'],mean_px=float(np.mean([e for p in points.values() for e,v in zip(p['error_px'],p['valid']) if v])),points=points))
    write(RUN/'evidence/diagnostic_2d.json',dict(command='python code/analyze.py',exit_status=0,pixel_space='original 872-wide image',records=result,scope='Archived official projected 2D reference; analysis only, never fed to coordinate-swap fits.'))
    archive=ROOT.parent/'EgoFishPose/experiments/benchmark_body_hand10_head01_20261014_v1'
    pred={v:{x['id']:x for x in read(ROOT/('outputs/rw_a20000_evaluation_20261117/'+v+'_diagnostic_predictions.json'))} for v in ['rw5000','rw20000']}
    mapping={'R_Shoulder':6,'R_Elbow':8,'R_Wrist':41,'L_Shoulder':5,'L_Elbow':7,'L_Wrist':62,'R_Hip':10,'R_Knee':12,'R_Ankle':14,'L_Hip':9,'L_Knee':11,'L_Ankle':13}
    visuals=[]
    for eye,i in [('left',0),('right',1)]:
        key='body07_'+eye;path=archive/'body_original'/(key+'.png');assert sha(path)==pred['rw5000'][key]['image_sha256']==pred['rw20000'][key]['image_sha256']
        xy=np.zeros((308,2));
        for name,j in mapping.items():xy[j]=(np.array(gt['body07'][name]['xy'][i])-[279.91284403669727,-0.08715596330275227])/(720/872)
        dst=OUT/'visuals'/(key+'.jpg');overlay(path,pred['rw5000'][key]['xy'],pred['rw20000'][key]['xy'],xy,mapping,key,dst)
        visuals.append(dict(file=dst.name,sha256=sha(dst),source_image_sha256=sha(path)))
    manifest=ROOT/'outputs/rw_test_completion_20261117/test_manifest.json';m={x['id']:x for x in read(manifest)['records']}
    pa={x['id']:x for x in read(ROOT/'outputs/rw_test_completion_20261117/rw5000_test_frozen_predictions.json')}
    pb={x['id']:x for x in read(ROOT/'outputs/rw20000_frozen200_20261118/rw20000_test_frozen_predictions.json')}
    lw=names.index('left_wrist');chosen=max((r for r in rows if r['subject']=='S11'),key=lambda r:r['new'][lw]-r['old'][lw])
    pair=chosen['pair_id']
    for rec in [x for x in m.values() if x['pair_id']==pair]:
        key=rec['id'];path=manifest.parent/rec['image'];assert sha(path)==rec['image_sha256']
        dst=OUT/'visuals'/('S11_wrist_'+rec['view']+'.jpg')
        overlay(path,pa[key]['xy'],pb[key]['xy'],rec['keypoints308_xy'],{'L_Wrist':62,'R_Wrist':41,'L_Hip':9,'R_Hip':10},'S11 left-wrist regression '+rec['view'],dst)
        visuals.append(dict(file=dst.name,sha256=sha(dst),source_image_sha256=sha(path),id=key))
    write(RUN/'evidence/visual_manifest.json',dict(command='python code/analyze.py',exit_status=0,visuals=visuals,selection='body07 both views and S11 pair containing largest left-wrist regression; selected after observing errors.',images_published=False))
    print('Frozen200 analysis, diagnostic errors and four local overlays written.')

if __name__=='__main__':main()
