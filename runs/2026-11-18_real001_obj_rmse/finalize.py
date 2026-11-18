"""Audit OBJ archive and make the public numerical comparison."""
import json,hashlib,zipfile,math,ast,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent;OUT=ROOT/'outputs/real001_obj_rmse_20261118';read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
e=read(RUN/'evidence/exports.json');rmse=read(RUN/'evidence/rmse.json');points=read(RUN/'evidence/point_residuals.json');summary=rmse['summary'];archive=OUT/e['archive']['file'];assert sha(archive)==e['archive']['sha256']
for model,rows in points.items():
 assert len(rows)==24
 for scope in ['all_12_joints_per_view','common_fit_mask']:
  for camera in ['cam3','cam4','stereo']:
   selected=[p for p in rows if (scope=='all_12_joints_per_view' or p['fit_mask']) and (camera=='stereo' or p['camera']==camera)];v=summary[model][scope][camera]
   sq=sum(p['dx_px']**2+p['dy_px']**2 for p in selected);assert len(selected)==v['points'];assert abs(math.sqrt(sq/len(selected))-v['radial_rmse_px'])<1e-9
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None;names=z.namelist();assert sum(n.endswith('.obj') for n in names)==8
 for name in names:
  assert not name.startswith('/') and '..' not in Path(name).parts;assert name.endswith(('.obj','.json','.md'));raw=z.read(name);assert len(raw)<50*1024*1024;s=raw.decode();assert not re.search(r'/(?:vepfs[^/\s]*|peter|root|home)/|github_pat_[A-Za-z0-9_]+|gh[pousr]_[A-Za-z0-9]{20,}',s)
  if name.endswith('.json'):json.loads(s)
  if name.endswith('.obj'):
   entry=next(x for x in e['files'] if x['relative_path']==name);assert hashlib.sha256(raw).hexdigest()==entry['sha256'];lines=s.splitlines();assert sum(x.startswith('v ') for x in lines)==entry['vertices'];assert sum(x.startswith('f ') for x in lines)==entry['faces'];assert all(x.startswith(('#','v ','f ')) for x in lines)
(RUN/'evidence/attachment_audit.json').write_text(json.dumps(dict(exit_status=0,archive=e['archive'],files_sha256={archive.name:sha(archive)},obj_files=8,archive_members=len(names),no_model_weights_dataset_original_images=True,no_machine_paths_credentials=True,allowed_payload='User-requested fitted OBJ meshes, camera config, manifest, README',rmse_independently_recomputed=True),indent=2)+'\n')
for p in RUN.glob('*.py'):ast.parse(p.read_text())
(RUN/'evidence/execution.json').write_text(json.dumps(dict(base_commit='60f2c92c282d595cfadc7a5a424a158ae1c4e10c',worktree='new uncommitted run code; unrelated existing changes untouched',export_exit_status=0,validation_exit_status=0,parameters_reused_without_fitting=True,source_run='2026-11-18_real001_golden_d_four_models',code_sha256={p.name:sha(p) for p in RUN.glob('*.py')},projection_code_sha256=sha(ROOT/'egosmplx_pipeline/camera.py'),source_detection_sha256=rmse['source_detections_sha256'],source_prepared_sha256=rmse['source_prepared_sha256'],command='PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. <EGOSMPLX_ENV>/bin/python export_evaluate.py; python finalize.py'),indent=2)+'\n')
url='https://github.com/cdh290718-oss/DownSight/releases/download/real001-obj-rmse-20261118/'+archive.name
lines=['# 真实样本001：四模型 OBJ 与自动点二维 RMSE','',f'**[下载四模型 OBJ 包]({url})**','', '包内 cam3_cv 是四模型的主要网格；cam4_cv 是相同四个人体变换到另一相机后的坐标版本，共 8 个 OBJ，不是 8 次拟合。文件名包含样本、29秒、检测器和坐标系。单位米，CV 坐标 +X 向右、+Y 向下、+Z 向前；不做中心化、缩放或额外翻转。','', 'OBJ 保存已变形的顶点与三角面，不保存 SMPL-X 形状/姿态参数或骨骼绑定。需要修改姿态时请使用[上一轮 NPZ/JSON 参数包](https://github.com/cdh290718-oss/DownSight/releases/tag/real001-goldend2d-four-models-20261118)。网格继承上一轮固定手型，没有新增 MANO 手部拼接。','', '## RMSE 定义与范围','','使用原图像素坐标下的径向二维 RMSE：','', '`RMSE = sqrt(sum((u_mesh_joint-u_auto)^2 + (v_mesh_joint-v_auto)^2) / N)`','', '范围是左右肩、肘、腕、髋、膝、踝共12个身体关节，不含手指，不能衡量手掌翻转或 MANO 拼接质量。这里投影的是从同一套 SMPL-X 参数还原的身体关节，不是寻找网格最近顶点。双目合并汇总两目的平方距离再开根号，不直接平均两目的 RMSE。若用 x/y 分量作为 2N 个样本，结果会再除以 sqrt(2)；evidence/rmse.json 同时保留该定义，正文统一用径向 RMSE。','', '**每个模型对照它自身的自动检测点，因此不是共同独立真值，不能用此表评判检测器或真实三维准确率。**','', '## 全部自动点：每目12点，合并24点','','包含之前未进入共同拟合掩码的 cam3 左肩、cam4 右肩；这些自动预测可在画面外，仍参与本表，不按分数或误差删除。','', '| 检测器 | cam3 RMSE px | cam4 RMSE px | 双目合并 RMSE px |','|---|---:|---:|---:|']
labels={'original':'原模型','rw500':'500 张','rw5000':'5000 张','rw20000':'20000 张'}
for model,label in labels.items():
 v=summary[model]['all_12_joints_per_view'];lines.append('| '+label+' | '+' | '.join(f"{v[k]['radial_rmse_px']:.2f}" for k in ['cam3','cam4','stereo'])+' |')
lines+=['','## 共同拟合掩码：每目11点，合并22点','','该表仅评估上一轮实际使用的同一组观测掩码；不是重新挑选小误差点。','', '| 检测器 | cam3 RMSE px | cam4 RMSE px | 双目合并 RMSE px |','|---|---:|---:|---:|']
for model,label in labels.items():
 v=summary[model]['common_fit_mask'];lines.append('| '+label+' | '+' | '.join(f"{v[k]['radial_rmse_px']:.2f}" for k in ['cam3','cam4','stereo'])+' |')
lines+=['','两种口径下，500 张模型的拟合结果与自身自动点距离最小。这是本次单个真实双目样本上的自一致性，不代表其检测最准；不同模型的目标点本来就不同。','', '## 导出与复现','','执行基线 `60f2c92c282d595cfadc7a5a424a158ae1c4e10c`，上轮 review 待审核。本轮代码运行时未提交，原有工作树改动保留。读取上一轮四份发布参数，逐个核对其 SHA-256，再用相同 SMPL-X 层重建。相机、Head、体型、手型及拟合状态均未修改，也未重跑检测或优化。上一轮四组均未通过末段稳定性诊断，本次导出不会改变该限制。','', 'OBJ 使用一基三角面索引，保留完整表面，无贴图。每个文件写出后重新读取，检查顶点数、面数、索引范围、拓扑一致及顶点精度；相机标定与变换、文件 SHA-256 放在 ZIP 中。未包含训练权重、SMPL-X 模型权重、原图、完整原始流水线结果。','', '逐点二维残差：[point_residuals.json](evidence/point_residuals.json)；完整 RMSE 与分母：[rmse.json](evidence/rmse.json)；OBJ 校验与哈希：[exports.json](evidence/exports.json)；运行记录：[execution.json](evidence/execution.json)。原始坐标和完整本地路径保留本机。','', '## 发布自检','','用户明确请求 OBJ 打包并沿用 GitHub 下载交付。拟合网格作为独立 Release 附件，Git 只保存代码、报告、精简证据；没有 NPY/NPZ/OBJ/ZIP、权重、数据集、视频、凭据、.env 或完整 provenance/patch 入库。单文件小于 50 MB。附件内容/哈希自检见 evidence/attachment_audit.json，工作区、暂存、待推提交和远端检查见 evidence/prepush.json；历史不在本轮审计范围。']
(RUN/'report.md').write_text('\n'.join(lines)+'\n');print('Archive and RMSE verified; report generated')
