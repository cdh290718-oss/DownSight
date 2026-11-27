from pathlib import Path
import hashlib,json,zipfile
import numpy as np,cv2
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
OUT=ROOT/'outputs/video003_strict_v5_20261127'
PRIOR=RUN.parent/'2026-11-27_video34_uniform_v5'
read=lambda p:json.loads(p.read_text())
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
frames=read(RUN/'evidence/frames.json')
rows=[];images=[];body=[];tips=[];fallback=[];members={}
for frame in frames:
    key=frame['pair']
    rec=read(RUN/'evidence'/(key+'_render.json'))
    assert rec['exit_status']==0 and rec['drift_mm']<.001
    for name,h in rec['hashes'].items():assert digest(RUN/'assets'/name)==h
    members[rec['obj']]=rec['obj_sha256']
    body.extend(rec['errors']['body']);tips.extend(rec['errors']['tips'])
    for s,h in rec['hands'].items():
        if h['source'] not in ('cam3','cam4'):fallback.append(dict(pair=key,side=s,reason=h['source']))
    e=rec['errors']['tips']
    score=f'{np.sqrt(np.mean(np.square(e))):.1f}' if e else '—'
    def link(mode,title):return f'[{title}](assets/{key}_{mode}.jpg)'
    status='、'.join(('左' if s=='left' else '右')+'手回退' for s,h in rec['hands'].items() if h['source'] not in ('cam3','cam4')) or '双手选解'
    rows.append('| '+' | '.join([key,f"{frame['relative_s']:.2f}",link('mesh','网格'),link('points','点位'),link('both','叠加'),link('obj','正侧面'),link('hands','原图放大'),link('handobj','双角度'),score,status])+' |')
    images.append(cv2.resize(cv2.imread(str(RUN/'assets'/(key+'_overview.jpg'))),None,fx=.65,fy=.65,interpolation=cv2.INTER_AREA))
assert cv2.imwrite(str(RUN/'assets/video003_all.jpg'),np.vstack([np.hstack(images[i:i+2]) for i in range(0,10,2)]),[cv2.IMWRITE_JPEG_QUALITY,92])
archive=OUT/'video003_direct_v4_v5_OBJ.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for name,h in members.items():
        assert digest(OUT/'export'/name)==h
        z.write(OUT/'export'/name,name)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None and len(z.namelist())==10
    for name,h in members.items():assert hashlib.sha256(z.read(name)).hexdigest()==h
summary=dict(body_rmse_px=float(np.sqrt(np.mean(np.square(body)))),body_n=len(body),tips_rmse_px=float(np.sqrt(np.mean(np.square(tips)))),tips_n=len(tips),fallback=fallback,members=members,archive_sha256=digest(archive),montage_sha256=digest(RUN/'assets/video003_all.jpg'),exit_status=0)
(RUN/'evidence/summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# Video 003 · v4 身体直接进入 Dv5 逐帧图集','','按拍摄时间排列，每行两对。总览中同一对从左至右为 cam3、cam4、OBJ 正面、侧面。青叉为自动检测，橙圈为 SMPL-X 投影。按表中链接打开原尺寸图片。','','[![十对总览](assets/video003_all.jpg)](assets/video003_all.jpg)','','| 样本 | 秒 | 双目 mesh | 自动与投影点 | 网格加点 | 全身 OBJ | 双目双手 | 手部 OBJ | 指尖 RMSE（px） | 手部状态 |','|---|---:|---|---|---|---|---|---|---:|---|']+rows
lines+=['','## 同帧结果','',f"身体自动点 RMSE 为 **{summary['body_rmse_px']:.2f} px**（{len(body)} 个观测），画内五指尖 RMSE 为 **{summary['tips_rmse_px']:.2f} px**（{len(tips)} 个观测）。误差使用两眼自动检测，遮挡和错误检测会影响数值。[逐帧详情](evidence/summary.json)",'', '每对使用同一批同步图像和哈希核验的 v4 原身体选姿，直接执行冻结 v5 的透视 WiLoR 逐手候选选择与双目腕旋转。固定肩、肘和腕中心，身体主关节最大位移低于 0.001 mm。十份网格重新导出并回读验证。此处理沿用 v4 原有头部目标；003、004 紫线平移及其肩肘校正未应用。[原视频实验图集](../2026-11-27_video34_uniform_v5/video003.md) · [同权肩肘校正图集](../2026-11-27_video003_equal_v5/report.md)','','基线 90bf2404173e319f6a633d31c89af493c5671235。工作树含本轮新增文件和既有无关改动。运行本目录 run.sh，索引依次为 0 至 9，再运行 package.py。输入、网格、候选及图像指纹见证据；OBJ 压缩包保存在开发机输出目录。推送前核对工作区、暂存区、增量提交及远端，检查文件大小、JPEG 元数据、凭据及机器路径。']
(RUN/'report.md').write_text('\n'.join(lines)+'\n')
(RUN/'review.md').write_text('# 待审核\n\n核对各帧 v4 原姿态、v5 手部候选与双目图像哈希，检查关节和腕中心位移及候选回退。\n')
