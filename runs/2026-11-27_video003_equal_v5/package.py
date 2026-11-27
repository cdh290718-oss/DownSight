from pathlib import Path
import json,hashlib,zipfile
import cv2,numpy as np
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
OUT=ROOT/'outputs/video003_equal_v5_20261127'
read=lambda p:json.loads(p.read_text())
frames=read(RUN/'evidence/frames.json')
rows=[];images=[];all_errors={'1':{'elbow':[],'wrist':[],'tips':[]},'4':{'elbow':[],'wrist':[],'tips':[]}}
changed=[];members={}
for frame in frames:
    key=frame['pair']
    render=read(RUN/'evidence'/(key+'_render.json'))
    comp=read(RUN/'evidence'/(key+'_comparison.json'))
    members[render['obj']]=render['obj_sha256']
    for weight in ['1','4']:
        for side in ['left','right']:
            h=comp[weight]['hands'][side]
            for measure in all_errors[weight]:all_errors[weight][measure].extend(h[measure])
    for side in ['left','right']:
        if comp['1']['hands'][side]['source']!=comp['4']['hands'][side]['source']:changed.append(key+' '+side)
    e=render['errors']['tips'];tip=float(np.sqrt(np.mean(np.square(e)))) if e else 0
    def link(mode,title):return f'[{title}](assets/{key}_{mode}.jpg)'
    rows.append('| '+' | '.join([key,f"{frame['relative_s']:.2f}",link('mesh','网格'),link('points','点位'),link('both','叠加'),link('obj','正侧面'),link('hands','原图放大'),link('handobj','双角度'),link('side','4∶1 / 1∶1'),f'{tip:.1f}'])+' |')
    images.append(cv2.resize(cv2.imread(str(RUN/'assets'/(key+'_overview.jpg'))),None,fx=.65,fy=.65,interpolation=cv2.INTER_AREA))
assert cv2.imwrite(str(RUN/'assets/video003_all.jpg'),np.vstack([np.hstack(images[i:i+2]) for i in range(0,10,2)]),[cv2.IMWRITE_JPEG_QUALITY,92])
archive=OUT/'video003_equal_v5_OBJ.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for name,h in members.items():
        assert hashlib.sha256((OUT/'export'/name).read_bytes()).hexdigest()==h
        z.write(OUT/'export'/name,name)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None and len(z.namelist())==10
    for name,h in members.items():assert hashlib.sha256(z.read(name)).hexdigest()==h
stats={w:{k:dict(rmse=float(np.sqrt(np.mean(np.square(v)))) if v else None,n=len(v)) for k,v in metrics.items()} for w,metrics in all_errors.items()}
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
summary=dict(stats=stats,changed_hand_sources=changed,archive_sha256=digest(archive),montage_sha256=digest(RUN/'assets/video003_all.jpg'),exit_status=0)
(RUN/'evidence/summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# Video 003 · 腕肘同权 Dv5 逐帧图集','','按拍摄时间排列，每行两对。总览中同一对从左至右为 cam3、cam4、OBJ 正面、侧面。青叉为自动检测，橙圈为 SMPL-X 投影。按表中链接打开原尺寸图片。','','[![十对总览](assets/video003_all.jpg)](assets/video003_all.jpg)','','| 样本 | 秒 | 双目 mesh | 自动与投影点 | 网格加点 | 全身 OBJ | 双目双手 | 手部 OBJ | 权重对照 | 指尖 RMSE（px） |','|---|---:|---|---|---|---|---|---|---|---:|']+rows
lines+=['','## 同帧误差对比','','原图像素 RMSE 按两眼、两手所有有效观测合并。自动检测和遮挡会影响这些指标。','','| 腕∶肘权重 | 肘点 RMSE | 腕点 RMSE | 留出五指尖 RMSE |','|---|---:|---:|---:|']
for w in ['4','1']:
    s=stats[w];lines.append(f"| {w}∶1 | {s['elbow']['rmse']:.2f} | {s['wrist']['rmse']:.2f} | {s['tips']['rmse']:.2f} |")
lines+=['',f"候选来源发生变化的手共 {len(changed)} 只。逐帧细节见数值证据。",'', '双目帧、身体 v4 起点、紫线平移、检测点及手形候选沿用原实验。肩肘校正将腕点权重调为与肘点相同，再独立运行逐手候选选择和腕旋转拟合。原 4∶1 结果逐值复现，双目渲染和侧视都使用实际导出的 OBJ。该实验未改变默认配置。[原 4∶1 图集](../2026-11-27_video34_uniform_v5/video003.md) · [逐帧证据](evidence/summary.json)','','执行基线 3bfd86c06a8b339ca71d8c0aac085747df796fb9，工作树含本轮和既有无关改动。运行仓库内 run.sh 对索引 0 至 9，随后运行 package.py。OBJ 回读及 ZIP 哈希已核对，原始网格位于开发机输出目录。公开前检查文件大小、凭据、机器路径和 JPEG 元数据。']
(RUN/'report.md').write_text('\n'.join(lines)+'\n')
(RUN/'review.md').write_text('# 待审核\n\n核对十对的同权拟合和原 4∶1 逐值复现，检查候选变化和双目图集。\n')
