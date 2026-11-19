"""Validate the user-requested archive and generate its download documentation."""
import ast,hashlib,json,re,subprocess,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent
OUT=ROOT/'outputs/rw10_four_models_obj_20261119'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
e=read(RUN/'evidence/exports.json');selection=read(RUN/'evidence/selection.json');archive=OUT/e['archive']['file']
assert sha(archive)==e['archive']['sha256']
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    names=z.namelist();assert len(names)==len(set(names))
    assert sum(n.endswith('.obj') for n in names)==40
    root=names[0].split('/')[0];checks=z.read(root+'/SHA256SUMS.txt').decode().splitlines()
    for line in checks:
        digest,name=line.split('  ',1);assert hashlib.sha256(z.read(root+'/'+name)).hexdigest()==digest
    for name in names:
        assert not name.startswith('/') and '..' not in Path(name).parts
        assert Path(name).suffix in ['.obj','.json','.csv','.md','.txt']
        raw=z.read(name);assert len(raw)<50*1024*1024;text=raw.decode('utf-8-sig')
        assert not re.search(r'/(?:vepfs[^/\s]*|peter|root|home)/|github_pat_[A-Za-z0-9_]+|gh[pousr]_[A-Za-z0-9]{20,}',text)
        if name.endswith('.json'):json.loads(text)
        if name.endswith('.obj'):
            entry=next(x for x in e['files'] if root+'/'+x['file']==name)
            assert hashlib.sha256(raw).hexdigest()==entry['sha256']
            lines=text.splitlines();assert all(l.startswith(('#','v ','f ')) for l in lines)
            assert sum(l.startswith('v ') for l in lines)==entry['vertices']
            assert sum(l.startswith('f ') for l in lines)==entry['faces']
write(RUN/'evidence/attachment_audit.json',dict(exit_status=0,files_sha256={archive.name:sha(archive)},archive=e['archive'],archive_members=len(names),obj_files=40,samples=10,models=['original','rw500','rw5000','rw20000'],all_member_checksums_verified=True,no_weights_original_dataset_images=True,no_machine_paths_credentials=True,allowed_payload='User-requested fitted OBJ meshes, sample index, calibrated coordinate transforms, documentation and checksums'))
for p in RUN.glob('*.py'):ast.parse(p.read_text())
base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
write(RUN/'evidence/execution.json',dict(baseline_commit=base,worktree_dirty=True,unrelated_changes_preserved=True,source_run='2026-11-18_rw_frozen200_golden_d',source_parameter_sha256=e['source_parameter_sha256'],export_exit_status=0,archive_validation_exit_status=0,command=e['command'],source_sha256={p.name:sha(p) for p in RUN.glob('*.py')},elapsed_seconds=e['elapsed_seconds'],refit=False,coordinate_frame='left camera CV, meters, no PA alignment',full_local_output='<WORKSPACE>/DownSight/outputs/rw10_four_models_obj_20261119'))
url='https://github.com/cdh290718-oss/DownSight/releases/download/unrealego-rw-test10-four-models-obj-20261119/'+archive.name
rows=['| 文件夹编号 | 人物 | 原评测样本 | 序列 | 帧号 |','|---|---|---|---|---:|']
for i,p in enumerate(selection['pairs'],1):rows.append('| %02d | %s | %s | %s | %d |'%(i,p['subject'],p['pair'],p['sequence'].split('/')[-1],p['frame']))
max_joint=max(x['reconstruction_vs_previous_joints_max_m'] for x in e['files']);max_roundtrip=max(x['obj_roundtrip_max_m'] for x in e['files'])
report=f'''# UnrealEgo-RW 测试集：10 对 × 4 模型 OBJ

**[下载全部 40 个 OBJ（ZIP，{archive.stat().st_size/1024/1024:.2f} MiB）]({url})**

从上一轮冻结 100 对测试样本中固定抽取 10 对：S1 4 对，S9/S11 各 3 对，覆盖 10 条不同序列。按固定 SHA-256 排序取样，不依据误差、外观或收敛状态挑选。四个模型使用相同样本。每个样本文件夹有 `original`、`rw500`、`rw5000`、`rw20000` 四份网格；一对双目共同拟合一个人体，因此总数为 40 个 OBJ。

文件名格式：`编号_人物_序列_frame帧号_原评测pair编号__模型__leftcam_m.obj`。包内有 `index.csv`、`manifest.json`、`README_zh.md`、相机坐标变换和 `SHA256SUMS.txt`。

{chr(10).join(rows)}

单位米，保持原左相机 CV 坐标（X 向右、Y 向下、Z 向前），未做 PA 对齐、居中、缩放或翻转。同一帧四模型可直接叠加；不同帧不是共同世界坐标。相机文件对应已有 1280×720 适配画布，原生 872×872 的缩放/偏移见包内说明。

直接从上一轮已评价参数重建，没有重新检测或优化；保留固定手姿态，**没有 MANO 手部拼接**。OBJ 含变形顶点与三角面，没有贴图、骨骼绑定或可编辑姿态系数。原实验的标注辅助和收敛限制仍适用，每份状态见索引。

## 验证与执行

执行基线 `{base}`；工作树包含本轮新增代码及既有修改，既有文件改动保留。核对源参数与输入 SHA-256，按原评价的 SMPL-X 层及根旋转轴心变换重建。与原评价关节坐标最大差异 {max_joint:.9g} m；OBJ 回读顶点最大差异 {max_roundtrip:.9g} m，拓扑、索引范围、40 个文件及 ZIP 全部校验通过。证据：[抽样](evidence/selection.json)、[导出](evidence/exports.json)、[附件自检](evidence/attachment_audit.json)、[执行](evidence/execution.json)。没有更改原 100 对的指标。

ZIP SHA-256：`{sha(archive)}`。

## 发布自检

用户要求命名打包并沿用 GitHub 下载交付。拟合 OBJ 只作为独立 Release 附件，Git 仅保存代码、报告和精简脱敏证据；不包含权重、数据集、原图、视频、NPY/NPZ/OBJ/ZIP、完整原始输出、完整 provenance/patch、凭据或 .env。每文件小于 50 MB，机器路径替换为占位符。工作区、暂存、待推提交和远端增量审计见 [prepush.json](evidence/prepush.json)，不将本轮检查作为历史审计。

[原测试报告](../2026-11-18_rw20000_frozen200/report.md) · [黄金 D 方法和指标](../2026-11-18_rw_frozen200_golden_d/report.md)
'''
(RUN/'report.md').write_text(report)
print('Archive verified; report ready; ZIP bytes',archive.stat().st_size)
