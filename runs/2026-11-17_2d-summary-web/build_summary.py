"""Build a sourced Markdown summary and a self-contained static research page."""
import hashlib
import html
import json
from pathlib import Path

RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
SOURCE=ROOT/'runs/2026-11-17_2d-intervention'
COMMIT='42c647b479623486192959ddea88e57c615c2af1'
BASE='https://github.com/cdh290718-oss/DownSight/blob/'+COMMIT+'/runs/2026-11-17_2d-intervention/'
LABELS={'baseline':'原预测','all_half':'全身半修正','all_full':'全身全修正','shoulder':'双肩','elbow':'双肘',
        'wrist':'双腕','hip':'双髋','knee':'双膝','ankle':'双踝'}
JOINTS=['颈','右肩','右肘','右腕','左肩','左肘','左腕','右髋','右膝','右踝','右脚跟*','左髋','左膝','左踝','左脚跟*']


def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    analysis=read(SOURCE/'evidence/analysis.json');results=read(SOURCE/'evidence/results.json')
    prep=read(SOURCE/'evidence/preparation.json');cfg=read(SOURCE/'protocol.json')
    literature=read(RUN/'evidence/literature_baselines.json')
    summaries=analysis['summaries']
    def get(c='all_full',m='2d',b=20000,metric='absolute'):
        return next(r for r in summaries if (r['condition'],r['method'],r['budget'],r['metric'])==(c,m,b,metric))
    def record(c,p='body02'):
        return next(r for r in results['records'] if (r['condition'],r['pair'],r['method'],r['budget'])==(c,p,'2d',20000))
    selected_keys=['method','budget','condition','metric','baseline_mean_mm','corrected_mean_mm',
                   'mean_improvement_mm','per_joint_baseline_mm','per_joint_corrected_mm','per_joint_improvement_mm',
                   'per_joint_sample_improved_count','per_joint_sample_worsened_count',
                   'per_joint_simultaneous_ci95_lower_mm','per_joint_simultaneous_ci95_upper_mm',
                   'mean_joint_improved_count','mean_joint_worsened_count','strict_mean_joint_improved_count',
                   'baseline_convergence_pass','corrected_convergence_pass',
                   'all_joints_practically_improve_with_simultaneous_support',
                   'every_sample_every_joint_practically_improves']
    bundle=dict(experiment_commit=COMMIT,pair_order=cfg['pairs'],joint_names=analysis['joint_names'],joint_labels=JOINTS,
                labels=LABELS,conditions=cfg['conditions'][1:],methods=cfg['methods'],budgets=cfg['budgets'],
                summaries=[{k:r[k] for k in selected_keys} for r in summaries],
                records=[{k:r[k] for k in ['method','budget','condition','pair','absolute_per_joint_mm','pa_per_joint_mm']}
                         for r in results['records']],
                mean_input_2d_px={c:sum(next(r['mean_2d_error_px'] for r in p['input_metrics'] if r['condition']==c)
                                      for p in prep['pairs'])/len(prep['pairs']) for c in cfg['conditions']},
                literature=literature,source_urls=dict(report=BASE+'report.md',results=BASE+'evidence/results.json',
                analysis=BASE+'evidence/analysis.json',protocol=BASE+'protocol.json',failures=BASE+'evidence/first_attempt_failure.json',
                verification=BASE+'evidence/verification.json',
                summary='https://github.com/cdh290718-oss/DownSight/blob/main/runs/2026-11-17_2d-summary-web/summary.md'))
    write(RUN/'evidence/summary_data.json',bundle)
    primary,half,pa=get(),get('all_half'),get(metric='pa')
    original,corrected=record('baseline'),record('all_full')
    rows=['# 二维更准，三维是否全身都更好？', '',
          '在线阅读：[交互研究网页](https://cdh290718-oss.github.io/egofishpose-demo/2d-to-3d-summary-20261117/)。', '',
          '本汇总整理既有实验与后续澄清，不新增拟合。实验快照 `'+COMMIT+'`；论文参考核查日期见 evidence/literature_baselines.json。', '',
          '## 可以说什么', '',
          '**本组样本中，全身十二个二维身体关节同时修正后，十五个评测关节的三维误差均值均下降；局部修正可以带来远处收益，也可能让其他关节退化。** 这不等于每个样本每个关节都改善，也不建立任意二维修正的普遍保证。', '',
          '## 数据、方法与指标', '',
          '- 数据：UnrealEgo-RW 真实数据，固定 body01–body10；S9 一人、十段不同序列各一帧，十对双目图像。不是完整测试集，也不是跨人实验。',
          '- 本轮主实验：直接二维重投影拟合。辅助实验：固定路由的三维／二维混合拟合。两者复用 D 的部分初始化与约束，但都不是原样 D。原 D 的观测路径为二维点 → 三角化 → 三维目标拟合。',
          '- 十二个输入关节为左右肩、肘、腕、髋、膝、踝。十五点评测另外包含 Neck 和两个脚部代理点，不包含 Head。ToeBase 用 heel proxy，对应语义未独立验证。',
          '- 保留低置信度原预测，固定全部观测覆盖、相机、Head、权重、协方差、体型空间、初始化、预算与选点。体型基底全部为零秩，实际固定均值体型。',
          '- 不对齐误差是相机坐标下的逐点三维欧氏距离均值；PA 先对全部评测骨架统一调整平移、旋转和尺度，再算误差。不是对每个关节独立对齐。', '',
          '## 二维改善怎样确认', '',
          '逐关节、逐相机计算像素欧氏距离 `e = ||p_prediction - p_reference||₂`。修正坐标为 `p_new = (1-α)p_prediction + α p_reference`；α=0.5 时每个有效点误差恰好减半，α=1 时到参考标注的距离为零。输入关系已经逐点检查。官方参考标注及关节定义仍有误差可能。', '',
          '| 全身输入条件 | 输入二维参考距离 px | 最终不对齐三维 mm | 最终 PA mm |',
          '|---|---:|---:|---:|']
    for c in ['baseline','all_half','all_full']:
        q=get('all_full' if c=='baseline' else c);p=get('all_full' if c=='baseline' else c,metric='pa')
        rows.append(f"| {LABELS[c]} | {bundle['mean_input_2d_px'][c]:.2f} | {(q['baseline_mean_mm'] if c=='baseline' else q['corrected_mean_mm']):.2f} | {(p['baseline_mean_mm'] if c=='baseline' else p['corrected_mean_mm']):.2f} |")
    rows += ['', '上表为直接二维、标准预算结果。保留低置信度点的诊断基线不是部署流程精度；参考点替换的改善幅度不是实际微调收益承诺。', '',
             '## 全身修正的逐关节结果', '',
             '| 关节 | 原预测 mm | 全身参考二维 mm | 改善量 mm | 改善／退化样本数 |',
             '|---|---:|---:|---:|---:|']
    for i,name in enumerate(analysis['joint_names']):
        rows.append(f"| {name} | {primary['per_joint_baseline_mm'][i]:.2f} | {primary['per_joint_corrected_mm'][i]:.2f} | {primary['per_joint_improvement_mm'][i]:+.2f} | {primary['per_joint_sample_improved_count'][i]}/{primary['per_joint_sample_worsened_count'][i]} |")
    rows += ['', '正改善量表示误差降低。样本计数按超过 +1 mm／低于 −1 mm 统计，其余为变化小；这是描述门槛，不代表统计显著性。', '',
             '## 为什么全身均值改善，不等于每个样本都改善', '',
             f"body02 的左腕，不对齐误差为 {original['absolute_per_joint_mm'][6]:.2f} → {corrected['absolute_per_joint_mm'][6]:.2f} mm；PA 后为 {original['pa_per_joint_mm'][6]:.2f} → {corrected['pa_per_joint_mm'][6]:.2f} mm。两种指标的变化方向可以不同，因为 PA 对齐针对整个骨架。PA 有残余误差不等于 PA 算错。", '',
             '## 局部修正', '', '| 修正的二维部位（左右同时） | 全身三维均值 mm | 平均改善 mm | 均值改善／退化关节数 | 均值退化关节 |', '|---|---:|---:|---:|---|']
    for c in cfg['conditions'][3:]:
        q=get(c);worse=', '.join(n for n,v in zip(analysis['joint_names'],q['per_joint_improvement_mm']) if v < -1) or '无（按描述门槛）'
        rows.append(f"| {LABELS[c]} | {q['corrected_mean_mm']:.2f} | {q['mean_improvement_mm']:+.2f} | {q['mean_joint_improved_count']}/{q['mean_joint_worsened_count']} | {worse} |")
    rows += ['', '## 优化预算与结论边界', '',
             '两个方法分别检查标准与延长预算。全身整体修正的关节均值收益方向保留，但局部效果会随方法与预算变化。全修正组没有通过严格末段稳定性诊断；同时区间也不足以支持全部关节稳定受益。单人便利样本的序列重采样只反映该组样本的敏感性。', '',
             'Head 位置使用标注、朝向取历史估计，外参来自另一段标注配准；官方二维和三维同源。评测身体三维真值在全部拟合结束后读取。初次保存检查点发生错误，已修复重跑且保留失败；成功运行的标准预算输出与保留快照逐位相同。', '',
             '## 论文参考：只列公开报告值，不与本地结果排名', '',
             '官方页面按十六个身体关节、设备相对协议提供测试结果；本地仅十对、十五点，且有参考二维和标注 Head 输入。训练数据、时序、附加深度、关节定义与划分均需统一后才能公平比较。下列方法未在本地复现。', '',
             '| 官方榜单方法 | 训练数据 | MPJPE mm | PA-MPJPE mm |', '|---|---|---:|---:|']
    for r in literature['official']:rows.append(f"| {r['method']} | {r['training']} | {r['mpjpe']:.2f} | {r['pa_mpjpe']:.2f} |")
    rows += ['', '来源：[UnrealEgo 官方榜单](https://unrealego.mpi-inf.mpg.de/)。CVPR 2024 的 T=5 方法使用场景深度信息。', '',
             '| 近期预印本：作者本地评测 | 训练数据 | MPJPE mm | PA-MPJPE mm |', '|---|---|---:|---:|']
    for r in literature['preprint']:rows.append(f"| {r['method']} | {r['training']} | {r['mpjpe']:.2f} | {r['pa_mpjpe']:.2f} |")
    rows += ['', '来源：[EgoRayFormer v1，表 1](https://arxiv.org/html/2610.26641v1)。未提交官方榜单；作者注明旧榜单的划分版本未核实，并披露开发阶段曾进行初步测试集评测。这些数字仅供文献参考，不作为同条件排名。原始表行、来源摘要与核查日期见 [literature_baselines.json](evidence/literature_baselines.json)。', '',
             '## 当前判断与下一步', '',
             '给定参考二维仍有明显残差，意味着应继续检查三维拟合与评测对应；固定体型、Head／相机语义、关节对应及优化状态都是待隔离因素，尚不能确定各自贡献。86.61 mm 不能当作 D 的结果或准确二维观测的性能上限。', '',
             '下一步应在同一批样本、同一关节对应和相同指标下运行原样 D、直接二维及可访问的论文基线，再分别评测真实微调检测器的坐标改善与端到端覆盖变化。此项属于待决定的后续实验，本轮未新增运行。', '',
             '## 原始证据', '']
    for name,url in bundle['source_urls'].items():rows.append(f'- [{name}]({url})')
    rows += ['', '本轮汇总的数据、源码与发布记录在本目录 evidence/。原始图像、坐标、权重、完整本地输出、完整 provenance 和 patch 未发布。', '']
    (RUN/'summary.md').write_text('\n'.join(rows))
    baseline_rows=''.join(f'<tr><td>{html.escape(r["method"])}</td><td>{html.escape(r["training"])}</td><td>{r["mpjpe"]:.2f}</td><td>{r["pa_mpjpe"]:.2f}</td></tr>' for r in literature['official'])
    preprint_rows=''.join(f'<tr><td>{html.escape(r["method"])}</td><td>{html.escape(r["training"])}</td><td>{r["mpjpe"]:.2f}</td><td>{r["pa_mpjpe"]:.2f}</td></tr>' for r in literature['preprint'])
    joint_rows=''.join(f'<tr><th scope="row">{JOINTS[i]}</th><td>{primary["per_joint_baseline_mm"][i]:.2f}</td><td>{primary["per_joint_corrected_mm"][i]:.2f}</td><td class="positive">{primary["per_joint_improvement_mm"][i]:+.2f}</td><td>{primary["per_joint_sample_improved_count"][i]}/{primary["per_joint_sample_worsened_count"][i]}</td></tr>' for i in range(15))
    json_text=json.dumps(bundle,ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('<','\\u003c')
    template=(RUN/'site.template.html').read_text()
    substitutions={'{{DATA}}':json_text,'{{SCRIPT}}':(RUN/'site.js').read_text(),'{{JOINT_ROWS}}':joint_rows,'{{BASELINE_ROWS}}':baseline_rows,'{{PREPRINT_ROWS}}':preprint_rows,
                   '{{REPORT}}':bundle['source_urls']['report'],'{{SUMMARY}}':bundle['source_urls']['summary'],
                   '{{RESULTS}}':bundle['source_urls']['results'],'{{ANALYSIS}}':bundle['source_urls']['analysis'],
                   '{{PROTOCOL}}':bundle['source_urls']['protocol'],'{{FAILURES}}':bundle['source_urls']['failures']}
    for key,value in substitutions.items():template=template.replace(key,value)
    assert '{{' not in template,'Unfilled template placeholder'
    dist=RUN/'site/dist';dist.mkdir(parents=True,exist_ok=True)
    (dist/'index.html').write_text(template)
    write(RUN/'evidence/build.json',dict(command='python runs/2026-11-17_2d-summary-web/build_summary.py',exit_status=0,
        experiment_commit=COMMIT,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [SOURCE/'evidence/analysis.json',SOURCE/'evidence/results.json',SOURCE/'evidence/preparation.json',SOURCE/'protocol.json',RUN/'evidence/literature_baselines.json']},
        html_sha256=sha(dist/'index.html'),html_bytes=(dist/'index.html').stat().st_size,
        summary_sha256=sha(RUN/'summary.md'),template_sha256=sha(RUN/'site.template.html'),
        no_new_fit=True,no_dataset_images_or_coordinates=True))
    print('Built sourced summary and self-contained static page.')


if __name__=='__main__':main()
