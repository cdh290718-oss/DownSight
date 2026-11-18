"""Build concise original-report update and a separate evidence-backed methods report."""
import platform,subprocess
from common import ROOT,RUN,OUT,EVIDENCE,read,write,sha

def main():
    result=read(EVIDENCE/'summary.json');prep=read(EVIDENCE/'preparation.json')
    done=read(OUT/'fit_complete.json');status=read(OUT/'fit_status.json');assert status['exit_status']==0
    summary=result['summary'];labels={'original':'原模型','rw500':'500 张模型','rw5000':'5000 张模型','rw20000':'20000 张模型'}
    common=result['common_success_pairs'];assert common>0
    table=['| 模型 | 非 PA MPJPE ↓（mm） | PA-MPJPE ↓（mm） | 成功 / 总数 | 收敛诊断通过 |','|---|---:|---:|---:|---:|']
    for m,s in summary.items():
        paired=result['paired'][m]
        table.append('| %s | %.2f | %.2f | %d / %d | %d / %d |'%(labels[m],paired['absolute_mpjpe_mm'],paired['pa_mpjpe_mm'],s['success'],s['total'],s['converged'],s['total']))
    table='\n'.join(table)
    rawlog=(OUT/'fit.log').read_text()
    steplog=[l for l in rawlog.splitlines() if l.startswith(('step ','DONE','WARNING: You are using a SMPL'))]
    (EVIDENCE/'fit_excerpt.txt').write_text('\n'.join(steplog)+'\n')
    changed=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).splitlines()
    baseline=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    paths=read(ROOT/'outputs/initialization_ablation_20261109/model_paths.json')
    backend_probe=subprocess.run(['git','rev-parse','HEAD'],cwd=paths['egosmplx_repo'],text=True,capture_output=True)
    backend=backend_probe.stdout.strip() if backend_probe.returncode==0 else 'external backend is not a Git checkout; configuration SHA-256 recorded, no commit available'
    versions={p.name:sha(p) for p in RUN.glob('*.py')}
    write(EVIDENCE/'execution.json',dict(baseline_commit=baseline,worktree_dirty=True,preexisting_tracked_change='runs/2026-11-16_sapiens-rw500-prep/code/train.py',unrelated_changes_preserved=True,backend_commit=backend,backend_config_sha256=sha(paths['egosmplx_config']),source_sha256=versions,commands=[{'command':'PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. <BODY_PYTHON> prepare.py','exit_status':0},{'command':'PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 <BODY_PYTHON> fit.py --threads 1','exit_status':status['exit_status'],'elapsed_seconds':status['elapsed_seconds']},{'command':'PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= <BODY_PYTHON> evaluate.py','exit_status':result['exit_status']}],fit_steps=done['steps'],fit_elapsed_seconds=done['elapsed_seconds'],device=done['device'],threads=done['threads'],prepared_local_sha256=sha(OUT/'prepared.json'),protocol_sha256=sha(RUN/'protocol.json'),fit_log_local_sha256=sha(OUT/'fit.log'),artifacts={'local_output':'<WORKSPACE>/DownSight/outputs/rw_frozen200_golden_d_20261118','access':'requires development-machine storage and licensed external SMPL-X/backend assets; public evidence alone is not an independent reproduction'},redaction='Machine paths replaced by <WORKSPACE>/<BODY_PYTHON>; raw coordinates, parameters, trace and full logs stay local. Metrics and failures preserved.'))
    subset=['| 模型 | 12 点非 PA mm | 12 点 PA mm | 15 点髋中心相对 MPJPE mm |','|---|---:|---:|---:|']
    for m,s in result['paired'].items():subset.append('| %s | %.2f | %.2f | %.2f |'%(labels[m],s['absolute_12_mpjpe_mm'],s['pa_12_mpjpe_mm'],s['root_relative_mpjpe_mm']))
    subject_table=['| 人物 | 模型 | 成功对数 | 非 PA mm | PA mm |','|---|---|---:|---:|---:|']
    for person in ['S1','S9','S11']:
        for m,s in summary.items():
            v=s['subjects'][person]
            subject_table.append('| %s | %s | %d | %.2f | %.2f |'%(person,labels[m],v['count'],v['absolute_mpjpe_mm'],v['pa_mpjpe_mm']))
    all_converged=sum(s['converged'] for s in summary.values())
    report=f'''# RW 冻结 200 张：黄金 D 四模型三维比较

对原报告的同一 200 张（100 对双目，S1/S9/S11）补充当前 `golden-d-2d-v1` 三维拟合，四个检测器共 400 次独立优化。没有重选样本、重新训练或按三维误差选检查点。原二维预测的 ID、图片 SHA-256、掩码逐一核对，重算结果与原报告一致。证据见 [输入核查](evidence/preparation.json)。

{table}

表中误差在四模型共同成功的 **{common} 对**上计算；成功、失败与收敛数按全部 100 对分别统计。非 PA 是左相机原始坐标下的平均关节欧氏距离，不做平移、旋转或尺度对齐。PA 每对独立做带正旋转、平移和尺度的相似变换。评价 15 点，沿用旧 RW D 报告的 Toe→heel 顶点代理映射；该代理不是解剖学上完全等价的脚趾。逐对、逐关节误差与失败均保留在 [results.json](evidence/results.json)，聚合见 [summary.json](evidence/summary.json)。

## 固定条件与辅助信息

- 使用已有原模型、RW500、RW5000、RW20000 的 308 点预测；只使用左右肩、肘、腕、髋、膝、踝这 12 点。固定参考画内掩码，不按置信度或三维结果筛选。2398 个原生图像观测点；鼻点无可靠标签，不用 Head 代替鼻点。
- 保持参考 D 的损失、学习率、姿态约束与 20000 步预算，按最低可行训练目标选检查点；全二维重投影，无身体三维残差、无 MANO/WiLoR 手部损失；均值体型、模板身体姿态、局部手姿固定。
- 将原生 872×872 的检测坐标按已有适配器缩到 720 并水平填充到 1280×720，同时使用对应标定，保留参考 D 的 8 px 损失尺度。表中旧二维误差仍以原生 872 图为单位，两个像素尺度不可混淆。
- Head 位置使用该帧官方 Head 三维标注；朝向使用同一历史 snapshot-11 左眼网络输出的运动链旋转，**不是 Head 朝向真值**。四个检测器共享同一锚点，网络身体姿态、平移和体型不作为拟合初值。缓存和初始网络的标识、SHA-256 见输入核查。
- 按 S1、S9、S11 各拍摄会话恢复双目刚性外参，来自未评估帧的官方双目三维标注。S1/S9 使用与冻结集不同的序列；S11 所有序列均在冻结集内，故使用同序列另一未评估帧（frame 1），明确保留序列重叠。这是**标注辅助的校准**，不是新的官方外参文件。用评测点仅检查坐标一致性，不调整外参；各会话最大刚性残差小于 0.001 mm，投影最大误差小于 0.001 px，仅说明与数据标注自洽，不代表传感器绝对精度。

## 结果口径与限制

为区分脚趾代理和绝对位置影响，附同一检查点的 12 个身体点指标（PA 在这 12 点上重新求解），以及 15 点双髋中心相对 MPJPE：

{chr(10).join(subset)}

分人物结果（各模型自身成功样本）：

{chr(10).join(subject_table)}

全部优化预算为 20000 步，CPU 单线程向量化运行约 {done['elapsed_seconds']/60:.2f} 分钟。400 次拟合中 {all_converged} 次通过末段稳定性诊断（最后 1000 步目标相对变化 <0.001 且最大关节移动 <0.1 mm）；完成预算不代表全部充分收敛或姿态正确。[拟合日志摘录](evidence/fit_excerpt.txt)保留全程定期进度，失败与稳定性量保留在逐对证据中。精确 FK 与完整 SMPL-X 的一致性在评价时断言 <0.003 mm。

该冻结集是官方测试划分中的 3 人子集，并非完整官方 benchmark；序列/帧相关，部分测试序列此前用于诊断，不能声称完全未接触。Head 位置、掩码及外参使用标注辅助，因此不能与没有相同辅助输入的论文数字直接排名。Head 估计、关节语义、均值体型和未充分收敛都仍可能限制结果。二维更准是否带来三维改善应以本表为准。

## 执行、异常与公开范围

实际执行代码基线 `{baseline}`，工作树有既有修改及本轮代码；既有训练脚本修改与未跟踪实验均保留。后端版本、源文件/输入摘要、命令和退出状态见 [execution.json](evidence/execution.json)。拟合脚本只增加禁止读取评价 reference/网络缓存文件的保护，目标函数与参考相同；评价在拟合结束后单独运行，未使用三维误差选取拟合结果。

初次准备因 S11 无独立序列标定候选而退出，未启动拟合；采用未评估帧且明确披露序列重叠后重试成功。该异常保留在 preparation.json。原始坐标、参数、数据、权重、完整日志和 trace 留在本地输出；公开版本只含代码、协议、指标、资源哈希及脱敏精简证据。不能仅凭公开证据声称已独立复现。

## 推送前自检

工作区、暂存内容、待推提交和远端状态按本轮增量检查，见 [prepush.json](evidence/prepush.json)。不包含权重、数据集、视频、NPY/NPZ、完整原始输出、完整 provenance/patch、凭据或 .env；机器绝对路径已替换，每文件小于 50 MB。既有未跟踪和其他修改不加入提交；此自检不追溯认证历史文件。

[原报告](../2026-11-18_rw20000_frozen200/report.md)
'''
    (RUN/'report.md').write_text(report)
    original=ROOT/'runs/2026-11-18_rw20000_frozen200/report.md'
    text=original.read_text();marker='## 补充：同一冻结集的黄金 D 三维结果'
    assert marker not in text,'Do not overwrite a prior update'
    text=text.replace('此次只做二维比较，未运行三角化或 D 优化。','最初这一轮仅做二维比较；同一批样本的黄金 D 三维补充结果见文末。')
    text+=f'''\n{marker}

2026-11-18 补充：沿用上述 200 张 / 100 对和四模型的同一二维预测，统一全二维黄金 D、20000 步预算。误差统计四模型共同成功的 {common} 对。

{table}

单位 mm，越低越好。非 PA 是左相机绝对坐标 MPJPE，不对齐；PA 为逐对相似变换后的 MPJPE。使用 15 点口径，两个 Toe 沿用 heel 代理；另有不含颈/脚趾的 12 点结果。**这是标注辅助诊断**：Head 位置和画内掩码来自标注，朝向为共享网络估计；每个拍摄会话的外参由未评估帧标注恢复，S11 标定帧与测试存在序列重叠。完成预算不等于全部达到收敛标准，不能直接作为无标注辅助的官方 benchmark 排名。

[方法、分人物和收敛说明](../2026-11-18_rw_frozen200_golden_d/report.md) · [汇总证据](../2026-11-18_rw_frozen200_golden_d/evidence/summary.json) · [逐对误差](../2026-11-18_rw_frozen200_golden_d/evidence/results.json) · [本次发布检查](../2026-11-18_rw_frozen200_golden_d/evidence/prepush.json)。
'''
    original.write_text(text)
    print(table)

if __name__=='__main__':main()
