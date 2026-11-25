"""Render the user's fixed 24-frame sample from the archived fixed_full fits."""
import csv
import sys
from pathlib import Path
import cv2
import numpy as np
import torch

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
FLOOR = RUN.parent/'2026-11-23_fixed_floor24'
sys.path.insert(0, str(RUN.parent/'2026-11-20_rw20000_floor12'))
from render_gallery import g, render_views

SPLITS = {'test': 'frozen100_fingers', 'validation': 'validation200_fingers'}
BASE = '58387c62eeea1800489ef09b8efd4ee48e4b95f4'


def main():
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    model, _ = g.common.setup_model()
    layer = model.layer['neutral'].cpu().eval()
    faces = np.asarray(model.face, np.int32)
    source = FLOOR/'evidence/gallery.json'
    sample = g.read(FLOOR/'evidence/sample.json')['rows']
    gallery = g.read(source)['rows']
    sample_by_id = {r['pair']:r for r in sample}
    assert len(sample)==len(sample_by_id)==len(gallery)
    for entry in gallery:
        assert all(entry[k]==v for k,v in sample_by_id[entry['pair']].items())
    assert len(set(r['pair'] for r in gallery)) == 24
    rows = []
    hashes = {str(source.relative_to(ROOT)): g.sha(source)}
    for split, name in SPLITS.items():
        evidence = RUN.parent/('2026-11-25_'+name)/'evidence'
        results = g.read(evidence/'results.json')
        selections = g.read(evidence/'selections.json')
        reference = (g.DATA/'reference.json' if split=='test' else
                     ROOT/'outputs/stereo_wrist_candidates_20261123/validation_truth.json')
        assert g.sha(reference) == results['reference_sha256']
        truth = g.read(reference)
        for path in (reference, evidence/'results.json', evidence/'selections.json'):
            hashes[str(path.relative_to(ROOT))] = g.sha(path)
        for entry in gallery:
            if entry['split'] != split:
                continue
            key = entry['pair']
            record = selections[key]['poses'] if split=='test' else selections[key]
            path = ROOT/'outputs'/(name+'_20261125')/'poses'/(key+'.npz')
            assert g.sha(path) == record['sha256']
            with np.load(path) as pack:
                x, root = pack['fixed_full'], pack['global0']
            assert x.shape==(53,3) and not x[23:].any()
            body = dict(body_pose=x[:21], global_orient=root, transl=x[22], betas=np.zeros(10),
                        left_hand_pose=x[23:38], right_hand_pose=x[38:])
            vertices, joints, _, _ = g.geometry(layer, body, x[21], False, None)
            ids = g.common.OBS
            pred = joints[g.common.NATIVE_IDS][ids]
            target = np.asarray(truth[key]['gt'])[ids]
            assert truth[key]['subject']==entry['subject']==results['rows'][key]['subject']
            archived = results['rows'][key]['methods']['fixed_full']
            pa = float(np.linalg.norm(g.align(pred,target)-target,axis=1).mean()*1000)
            nonpa = float(np.linalg.norm(pred-target,axis=1).mean()*1000)
            assert max(abs(pa-archived['pa_mm']),abs(nonpa-archived['nonpa_mm'])) < .001
            views = render_views(vertices,faces,np.asarray(entry['ground']['up']),
                                 np.asarray(entry['front_forward_left_cv']))
            assets = {}
            for side in ('left','right'):
                asset = FLOOR/'assets'/key/(side+'.jpg')
                assert g.sha(asset)==entry['assets'][asset.name]
                assets[side]=dict(path='../'+str(asset.relative_to(RUN.parent)),sha256=g.sha(asset))
            for side, image in zip(('front','side'), views):
                asset = RUN/'assets'/key/(side+'.jpg')
                asset.parent.mkdir(parents=True,exist_ok=True)
                assert np.count_nonzero(image[:,:,0]<235)>300
                g.save(asset,image)
                assert cv2.imread(str(asset)).shape==image.shape
                assets[side]=dict(path=str(asset.relative_to(RUN)),sha256=g.sha(asset))
            assert len(assets)==4
            rows.append(dict(pair=key,subject=entry['subject'],split=split,ground_accepted=entry['ground']['accepted'],
                             pa_mm=archived['pa_mm'],nonpa_mm=archived['nonpa_mm'],rebuilt_pa_mm=pa,rebuilt_nonpa_mm=nonpa,
                             pose_sha256=record['sha256'],choice=record['methods']['fixed_full']['chosen'],assets=assets))
    assert [r['pair'] for r in rows]==[r['pair'] for r in gallery]
    summary = {}
    for label in ('all','test','validation'):
        sub = [r for r in rows if label=='all' or r['split']==label]
        assert len(sub)==(24 if label=='all' else 12)
        summary[label] = dict(frames=len(sub),**{m:float(np.mean([r[m] for r in sub])) for m in ('pa_mm','nonpa_mm')})
    g.write(RUN/'evidence/results.json',dict(summary=summary,rows=rows,method='fixed_full',input_sha256=hashes,
            base_commit=BASE,dirty_worktree=True,source_sha256=g.sha(Path(__file__)),command='bash <RUN>/run.sh',exit_status=0))
    with (RUN/'evidence/metrics.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=['pair','subject','split','pa_mm','nonpa_mm'],extrasaction='ignore',lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    report = ['# 手指固定、完整双目轮廓 · 24 对渲染与关节误差','',
              '沿用指定图库的 24 对及原排列。每行显示同一时刻的双目图、最新方法拟合的 SMPL-X 网格正侧面，以及该姿态的关节误差。点图可放大。','',
              '| 样本 | 对数 | PA-MPJPE（mm） | 非 PA MPJPE（mm） |','|---|---:|---:|---:|']
    for key,label in [('all','**全部**'),('test','测试图库子集'),('validation','验证图库子集')]:
        a=summary[key]
        report.append(f"| {label} | {a['frames']} | {a['pa_mm']:.2f} | {a['nonpa_mm']:.2f} |")
    report.extend(['','两项均评估双侧肩、肘、腕、髋、膝、踝的 12 个关节。非 PA 在左相机坐标下直接与官方三维标注比较，PA 先逐帧对齐平移、旋转和统一尺度。先求每帧 12 点的平均距离，再对样本等权平均。[逐帧 CSV](evidence/metrics.csv) · [完整精度与渲染核查](evidence/results.json)',''])
    for split,title in [('test','测试图库的 12 对'),('validation','验证图库的 12 对')]:
        report.extend(['## '+title,'','| 人物 / 样本 | 左目 | 右目 | 最新正面 | 最新侧面 | PA（mm） | 非 PA（mm） |',
                       '|---|---|---|---|---|---:|---:|'])
        for row in rows:
            if row['split'] != split:continue
            cells=[row['subject']+' / '+row['pair']+('（地面暂定）' if not row['ground_accepted'] else '')]
            for key,label in [('left','左目'),('right','右目'),('front','正面'),('side','侧面')]:
                path=row['assets'][key]['path']
                cells.append(f'[![{row["pair"]} {label}]({path})]({path})')
            cells.extend(f'{row[k]:.2f}' for k in ('pa_mm','nonpa_mm'))
            report.append('| '+' | '.join(cells)+' |')
        report.append('')
    report.extend(['## 评分与取景','',
      '每帧按 SHA-256 核对已封存的最终姿态文件，读取 fixed_full 分支，使用标准体型和固定手指参数重新生成网格、关节点及两张渲染图。所有逐帧重算误差与原评分的差异均小于 0.001 mm。两张原图引用同一帧已公开的预览。正侧视图沿用原图库该帧的地面法向和观看方向，同尺度分别居中。地面暂定只影响看图时对身体倾斜的判断。[样本与取景来源](../2026-11-23_fixed_floor24/report.md) · [测试拟合](../2026-11-25_frozen100_fingers/report.md) · [验证拟合](../2026-11-25_validation200_fingers/report.md)','',
      f'执行基线为 `{BASE}`。工作树含本轮文件及此前已有的无关修改。此轮复用已拟合参数，在 CPU 上重建与渲染。开发机复现命令为 `bash runs/2026-11-25_fixed_full_floor24/run.sh`。',
      '推送前核对工作树、暂存区、待推提交与远端，仅发布本轮代码、文档、数值证据和渲染预览。检查单文件低于 50 MB、图像元数据、凭据、环境文件、机器绝对路径和禁发资产。公开路径为仓库相对路径或占位符。[发布自检](evidence/prepush.json)'])
    (RUN/'report.md').write_text('\n'.join(report)+'\n')
    print(summary)

if __name__=='__main__':
    main()
