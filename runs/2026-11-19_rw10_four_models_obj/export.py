"""Export a metric-independent stratified selection from the evaluated fits."""
import csv,hashlib,json,sys,zipfile,time
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent
OUT=ROOT/'outputs/rw10_four_models_obj_20261119'
PRIOR=ROOT/'runs/2026-11-18_rw_frozen200_golden_d'
OLD=ROOT/'outputs/rw_frozen200_golden_d_20261118'
sys.path.insert(0,str(PRIOR))
from common import setup_model,read,write,sha,NATIVE_IDS
from geometry import rotation

def main():
    start=time.time();prep=read(PRIOR/'evidence/preparation.json')
    assert sha(OLD/'prepared.json')==prep['prepared_local_sha256']
    result=read(PRIOR/'evidence/results.json');assert sha(OLD/'budget_20000.npz')==result['parameter_local_sha256']
    items=read(OLD/'prepared.json')['items'];done=read(OLD/'fit_complete.json');assert done['exit_status']==0
    params=dict(np.load(OLD/'budget_20000.npz',allow_pickle=False))
    previous=dict(np.load(OLD/'predicted_gt15.npz',allow_pickle=False))
    assert list(previous['ids'])==[i['pair'] for i in items]==[i[0] for i in done['ids']]
    # Deterministic independent-of-performance choice; prefer distinct sequences.
    selection=[];seed='rw10-four-models-20261119'
    for subject,count in [('S1',4),('S9',3),('S11',3)]:
        candidates=[p for p in prep['pairs'] if p['subject']==subject]
        candidates.sort(key=lambda p:hashlib.sha256((seed+'|'+p['sequence']+'|'+str(p['frame'])).encode()).hexdigest())
        chosen=[];seen=set()
        for p in candidates:
            if p['sequence'] not in seen:chosen.append(p);seen.add(p['sequence'])
            if len(chosen)==count:break
        assert len(chosen)==count
        selection.extend(sorted(chosen,key=lambda p:(p['sequence'],p['frame'])))
    assert len(selection)==10
    write(RUN/'evidence/selection.json',dict(seed=seed,rule='SHA256(seed|sequence|frame) rank per subject, select distinct sequences, quotas S1=4 S9=3 S11=3; no metric or convergence read for selection',pairs=selection,source_preparation_sha256=sha(PRIOR/'evidence/preparation.json')))
    bundle=OUT/'UnrealEgoRW_test10pairs_4models_OBJ';bundle.mkdir(exist_ok=False)
    model,_=setup_model();torch.set_num_threads(2)
    layer=model.layer['neutral'].cpu().eval();layer.requires_grad_(False)
    faces=np.asarray(model.face,dtype=np.int64);entries=[];cameras={}
    models=['original','rw500','rw5000','rw20000']
    lookup={(i['base_pair'],i['detector']):k for k,i in enumerate(items)}
    diagnostics={(r['pair'],r['model']):r for r in result['records']}
    for n,pair in enumerate(selection,1):
        name='%02d_%s_%s_frame%06d_%s'%(n,pair['subject'],pair['sequence'].split('/')[-1],pair['frame'],pair['pair'])
        folder=bundle/name;folder.mkdir()
        for detector in models:
            k=lookup[(pair['pair'],detector)];item=items[k];rec=diagnostics[(pair['pair'],detector)]
            assert rec['status']=='ok' and params['selected_step'][k]>=0
            t=lambda key:torch.tensor(params[key][k:k+1],dtype=torch.float32)
            fixed={key:torch.zeros(1,n) for key,n in [('left_hand_pose',45),('right_hand_pose',45),('jaw_pose',3),('leye_pose',3),('reye_pose',3),('expression',10)]}
            with torch.no_grad():
                full=layer(body_pose=t('body_pose').reshape(1,-1),betas=t('betas'),global_orient=t('global0'),**fixed)
                rd=rotation(t('delta'));pivot=full.joints[:,:1]
                vertices=((full.vertices-pivot)@rd.transpose(1,2)+pivot+t('translation')[:,None])[0].numpy().astype(float)
                joints=((full.joints-pivot)@rd.transpose(1,2)+pivot+t('translation')[:,None])[0].numpy()
            difference=float(np.max(np.abs(joints[NATIVE_IDS]-previous['joints_m'][k])))
            assert difference<3e-6,difference
            assert vertices.shape==(10475,3) and np.isfinite(vertices).all()
            filename=name+'__'+detector+'__leftcam_m.obj';path=folder/filename
            with path.open('w') as f:
                f.write('# Fitted UnrealEgo-RW SMPL-X mesh; left-camera CV coordinates; meters; no PA alignment\n')
                np.savetxt(f,vertices,fmt='v %.9f %.9f %.9f');np.savetxt(f,faces+1,fmt='f %d %d %d')
            rv=[];rf=[]
            for line in path.read_text().splitlines():
                parts=line.split()
                if parts[0]=='v':rv.append([float(v) for v in parts[1:]])
                if parts[0]=='f':rf.append([int(v)-1 for v in parts[1:]])
            rv=np.array(rv);rf=np.array(rf)
            error=float(np.max(np.abs(rv-vertices)))
            assert error<1e-8 and np.array_equal(rf,faces) and rf.min()>=0 and rf.max()<len(rv)
            entry=dict(pair=pair['pair'],subject=pair['subject'],sequence=pair['sequence'],frame=pair['frame'],model=detector,file=str(path.relative_to(bundle)),sha256=sha(path),bytes=path.stat().st_size,vertices=len(vertices),faces=len(faces),coordinate_frame='left camera CV',units='meters',source_row=k,selected_step=int(params['selected_step'][k]),convergence_pass=rec['convergence_pass'],reconstruction_vs_previous_joints_max_m=difference,obj_roundtrip_max_m=error)
            entries.append(entry)
            cameras[name]=dict(left_to_right=item['inputs']['views']['right']['camera']['left_to_view'],formula='x_right=R @ x_left+t',units='meters')
        print('Exported',name,'4 models',flush=True)
    assert len(entries)==40
    columns=['pair','subject','sequence','frame','model','file','coordinate_frame','units','selected_step','convergence_pass','sha256']
    with (bundle/'index.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(entries)
    write(bundle/'manifest.json',dict(files=entries,selection_seed=seed,parameter_source_sha256=sha(OLD/'budget_20000.npz')))
    write(bundle/'left_to_right_by_sample.json',cameras)
    for side in ['left','right']:write(bundle/('camera_'+side+'_1280x720.json'),read(OLD/('camera_'+side+'.json')))
    (bundle/'README_zh.md').write_text('''# UnrealEgo-RW 测试集：10 对 × 4 模型 OBJ

每个样本文件夹有 original（原模型）、rw500（500 张）、rw5000（5000 张）、rw20000（20000 张）四份 OBJ，共 40 份。每对双目共同拟合一个人体，因此每模型每对只有一个网格，不是左右眼各拟合一个。样本来自此前冻结测试 100 对；S1 抽 4 对、S9/S11 各 3 对，覆盖 10 条不同序列。固定哈希抽样，没有按误差或外观挑选。

文件名：编号_人物_序列_frame帧号_原评测pair编号__模型__leftcam_m.obj。index.csv 列出完整原始序列、帧号、模型与 SHA-256。

坐标：该帧左相机 CV 坐标，单位米，+X 向右、+Y 向下、+Z 向前。四模型同一帧可直接叠加；不同帧的相机坐标不是统一世界坐标。未做 PA 对齐、居中、缩放或坐标翻转。查看器可能使用 Y-up / Z-up，若显示方向不习惯，可统一旋转查看，勿覆盖原文件。left_to_right_by_sample.json 可将同一人体转换到右相机。

相机 JSON 对应原生 872×872 图像均匀缩放到 720×720、水平填充到 1280×720 的既有适配器；不能直接当作原生图标定，具体尺度及像素中心偏移见 input_adapter。没有附带数据集原图。

网格直接重建自上一轮黄金 D 的已选检查点，没有重新优化。它们是拟合结果，不是真值或原始模板。保持原来的固定手姿态，没有 MANO 手部替换或接缝融合；头部位置与观测掩码存在标注辅助，Head 朝向为共享网络估计。原优化未普遍达到严格收敛标准，manifest/index 保留每份状态。OBJ 仅有变形后的顶点和三角面，没有贴图、骨骼绑定或可编辑 SMPL-X 姿态系数。

原测试报告：https://github.com/cdh290718-oss/DownSight/blob/main/runs/2026-11-18_rw20000_frozen200/report.md
''')
    with (bundle/'SHA256SUMS.txt').open('w') as f:
        for p in sorted(bundle.rglob('*')):
            if p.is_file() and p.name!='SHA256SUMS.txt':f.write(sha(p)+'  '+str(p.relative_to(bundle))+'\n')
    archive=OUT/'UnrealEgoRW_test10pairs_original_rw500_rw5000_rw20000_OBJ.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():z.write(p,str(p.relative_to(OUT)))
    write(RUN/'evidence/exports.json',dict(command='PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= <BODY_PYTHON> export.py',exit_status=0,elapsed_seconds=time.time()-start,source_run=PRIOR.name,source_parameter_sha256=sha(OLD/'budget_20000.npz'),source_prepared_sha256=sha(OLD/'prepared.json'),source_previous_predictions_sha256=sha(OLD/'predicted_gt15.npz'),files=entries,archive=dict(file=archive.name,bytes=archive.stat().st_size,sha256=sha(archive)),fit_unchanged=True,mano_fusion=False))
    print('DONE',len(entries),'OBJ;',archive.stat().st_size,'ZIP bytes',flush=True)

if __name__=='__main__':main()
