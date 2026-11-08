"""Write a concise Chinese report and side-by-side GitHub projection page."""
import json
from pathlib import Path

RUN=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text())
d=read(RUN/'evidence/metrics.json');e=read(RUN/'evidence/execution.json');inputs=read(RUN/'evidence/inputs.json')
fmt=lambda x,unit='':('—' if x is None else f'{x:.2f}{unit}')
parts=['# 原流程：双目合成真值实验','',
       '数据是一个姿态的左右两目，不是两个独立姿态。原始图像、合成网格、模型权重和完整输出未提交到 GitHub；完整文件留在开发机，投影图是用于审阅的派生结果。','',
       '[投影图并排查看](projection.md) · [数据来源及真值定义](https://github.com/cdh290718-oss/DownSightSmplxdata/tree/192671271a74b99f41b9eb033d8f88b1dac77ada)','',
       '基线 `e49ec2a283a5cab2409aba880ffac3fd339df1d9`；执行时工作树状态及数值代码摘要见 `evidence/execution.json`。每视角单独使用自动网络推理和原流程，无真值输入。原头部参考 0.1、优化步数、匹配和接缝保持配置，未启用 v2。','',
       '## 结果','',
       '| 视角 | 状态 | 匹配手数 | 身体25点绝对均值 mm | 身体25点骨盆对齐均值 mm | 身体入画真值点的二维均值 px |',
       '|---|---|---:|---:|---:|---:|']
for r in d['views']:
    if r['execution_exit']:
        parts.append(f'| {r["camera"]} / {r["side"]} | 失败（退出 {r["execution_exit"]}） | — | — | — | — |');continue
    g=r['stages']['final']['groups']['body25'];n=r['validation']['matched_hands']
    parts.append(f'| {r["camera"]} / {r["side"]} | 完成 | {n} | {fmt(g["absolute_mm"]["mean"])} | {fmt(g["pelvis_aligned_mm"]["mean"])} | {fmt(g["projection_brown_camera_px"]["mean"])}（{g["projection_brown_camera_px"]["count"]}/{g["ground_truth_in_frame"]} 点） |')
parts+=['','身体二维误差使用渲染所用 Brown–Conrady 相机对预测三维关节重投影，再与真值比较；只纳入真值入画且预测在镜头前、投影有限的点。预测在画面外仍计误差。逐点计数、原流程多项式投影误差、手指及各阶段结果见 `evidence/metrics.json`。25 点的三维绝对和骨盆对齐误差均包含未入画点；其中骨盆对齐仅平移，不作旋转或尺度对齐。','',
        '| 视角 / 阶段 | 身体绝对均值 mm | 骨盆对齐均值 mm | 入画点二维均值 px |','|---|---:|---:|---:|']
for r in d['views']:
    if r['execution_exit']:continue
    for stage in ['raw','rigid','body','final']:
        g=r['stages'][stage]['groups']['body25']
        parts.append(f'| {r["camera"]} / {stage} | {fmt(g["absolute_mm"]["mean"])} | {fmt(g["pelvis_aligned_mm"]["mean"])} | {fmt(g["projection_brown_camera_px"]["mean"])}（{g["projection_brown_camera_px"]["count"]}/{g["ground_truth_in_frame"]}） |')
parts+=['',
        '| 视角 / 手 | 使用 WiLoR 匹配 | 手指20点绝对均值 mm | 腕对齐均值 mm | 腕深度误差 mm | 入画手指二维均值 px |',
        '|---|---|---:|---:|---:|---:|']
for r in d['views']:
    if r['execution_exit']:continue
    for hand in ['left','right']:
        g=r['stages']['final']['groups'][hand+'_fingers20'];matched=hand in (r['validation']['matches'] or {})
        parts.append(f'| {r["camera"]} / {hand} | {matched} | {fmt(g["absolute_mm"]["mean"])} | {fmt(g["wrist_aligned_mm"]["mean"])} | {fmt(g["wrist_depth_error_mm"])} | {fmt(g["projection_brown_camera_px"]["mean"])}（{g["projection_brown_camera_px"]["count"]}/{g["ground_truth_in_frame"]} 点） |')
if d['topology_exactly_matches_backend']:
    parts+=['','真值与后端 SMPL-X 的面拓扑和顶点顺序一致；融合网格沿用顶点索引。以下全顶点误差包含画外、被遮挡和镜头后的顶点：','',
            '| 视角 | 顶点数 | 绝对误差均值 mm | 骨盆对齐误差均值 mm |','|---|---:|---:|---:|']
    for r in d['views']:
        if r['execution_exit']:continue
        q=r['stages']['final'];a=q.get('vertex_all_absolute_mm');b=q.get('vertex_all_pelvis_aligned_mm')
        if a and b:parts.append(f'| {r["camera"]} | {a["count"]} | {fmt(a["mean"])} | {fmt(b["mean"])} |')
parts+=['','二维括号内为有效点数／真值入画点数；腕对齐只减去该侧腕关节的三维平移，用于观察手指形状与方向误差；腕深度误差为预测 z 减真值 z。未匹配到 WiLoR 的一侧仍评估原流程保留的 SMPL-X 手，不代表已完成 MANO 替换。','',
        'GT 的入画标记只检查位于相机前方和画幅内，没有遮挡测试；不能解释为实际可见。合成背景、光照、人体外观、安装位姿及标定与真实设备可能不同，一对图不足以推断真实数据精度。','',
        '渲染采用 Brown–Conrady，原流程采用数据集提供的多项式拟合。以入画真值关节作为同一三维输入，两模型的像素差如下：','',
        '| 视角 | 有效点数 | 平均差 px | 最大差 px |','|---|---:|---:|---:|']
for r in d['views']:
    q=r['camera_model_difference_px'];parts.append(f'| {r["camera"]} | {q["count"]} | {fmt(q["mean"])} | {fmt(q["max"])} |')
parts+=['','真值相机主点零基换算仍是数据集提供者的假设；模型差包含该约定，不把它直接归因于拟合。合成网格、坐标变换和真值投影的独立自检结果见 `evidence/dataset_validation.json`。','',
        '原流程身体姿态阶段对全部顶点使用 0.02 m 的近深度限制，而该真值网格本身有一部分顶点位于限制以内或镜头后方：','',
        '| 视角 | 真值顶点数 | 在镜头后方 | 深度小于 0.02 m | 最小深度 m |','|---|---:|---:|---:|---:|']
for r in d['views']:
    q=r['truth_mesh_depth'];parts.append(f'| {r["camera"]} | {q["vertex_count"]} | {q["behind_camera_count"]} | {q["below_2cm_count"]} | {q["minimum_z_m"]:.4f} |')
if d['topology_exactly_matches_backend']:
    parts+=['','以 SMPL-X 权重中 Head 主导的顶点作为头部，统计位于镜头前且投影入画的数量（未检查场景或身体遮挡）：','',
            '| 视角 | 真值头部顶点入画数 | 预测头部顶点入画数 |','|---|---:|---:|']
    for r in d['views']:
        prediction=r['stages']['final']['head_vertices_in_frame'] if not r['execution_exit'] else '—'
        parts.append(f'| {r["camera"]} | {r["truth_head_vertices_in_frame"]} | {prediction} |')
parts+=['','这说明原流程的身体近深度限制与该真值几何并不完全一致；不能仅凭这些计数断言它对本次误差的贡献。此处没有修改原规则。','',
        '## 执行与验证','',
        '每个视角独立输出；只使用原始渲染图和标定进行推理，真值仅在完成后用于评价。网络、权重、输入和后端指纹以及实际退出状态见 `evidence/inputs.json`、`evidence/resources.json` 和 `evidence/execution.json`。','']
for r in d['views']:
    if r['execution_exit']:
        parts.append(f'- {r["side"]}：原流程未完成，未伪造预测；完整脱敏失败日志见执行证据中的 `failure_log_files`。')
    else:
        v=r['validation'];parts.append(f'- {r["side"]}：流程退出 0，匹配手 {v["matched_hands"]}；重载验证 {v["passed"]}，局部表面检查 {v["local_surface_accepted"]}。重载或局部检查不代表姿态正确或整个网格无自相交。')
parts+=['','数据集自带的独立 25 点评测脚本与两目的重算结果一致，见 `evidence/crosscheck.json`；相关 20 项测试通过，见 `evidence/tests.json`。','',
        '单视角数字均可在 `evidence/metrics.json` 查到。绝对路径以 `<WORKSPACE>`、`<HOME>`、`<BUILD_ROOT>` 等占位符脱敏；完整配置、原始日志和 NPZ 仅留在开发机。','']
if (RUN/'evidence/retry.json').exists():
    parts+=['右图首次运行在 Sapiens2 阶段出现 CUDA `unspecified launch failure`，退出状态及脱敏完整日志见 `evidence/retry.json` 和 `evidence/right_failure/`；随后只更换输出目录，用相同输入、模型及原流程重试。小规模 CUDA 检查成功并不保证完整模型执行成功。','']
if (RUN/'evidence/retry2.json').exists():
    parts+=['相同配置的第二次尝试仍出现 CUDA 错误，日志见 `evidence/right_retry1_failure/`；第三次只增加 `CUDA_LAUNCH_BLOCKING=1` 以同步执行并定位问题，模型、输入和拟合参数不变。','']
if (RUN/'evidence/cpu_fallback.json').exists():
    parts+=['第三次尝试仍遇到 CUDA 错误，日志见 `evidence/right_retry2_failure/`。因此右目复用了首次尝试已成功生成的原始身体网络预测，并对同一图像在 CPU 新运行 Sapiens2 姿态、分割和 WiLoR，再以这些本轮自动观测运行原拟合；没有使用真值生成观测。左目使用原 GPU 运行。','',
            'CPU 模式仅新增 Sapiens2 两个工作器的设备选项；默认仍为 CUDA。网络结构、权重、预处理、姿态翻转测试、BF16 自动混合精度、WiLoR 模型和拟合参数保持一致，但 CPU 与 GPU 的浮点实现可能不同。运行命令和状态见 `evidence/cpu_workers.json`、`evidence/cpu_fallback.json`。','']
    control=RUN/'evidence/device_control.json'
    if control.exists():
        q=read(control);parts+= [f'同一左图的 Sapiens2 CPU/GPU 对照：13 个身体候选点位置差均值 {q["mean_position_delta_px"]:.2f} px、最大 {q["max_position_delta_px"]:.2f} px，置信度及入画筛选不一致 {q["selection_disagreements"]} 点。见 `evidence/device_control.json`；这是设备数值差，非对真值误差，也不能保证右图结果相同。该对照不覆盖分割、WiLoR 或后续优化的 CPU/GPU 差异。','']
    else:
        parts+=['左图 CPU/GPU 姿态对照未完成，设备误差无法量化；具体工作器状态和失败日志见 `evidence/cpu_workers.json`。','']
elif (RUN/'evidence/cpu_workers.json').exists():
    parts+=['CPU 备选观测运行未全部成功，具体状态和完整脱敏日志见 `evidence/cpu_workers.json` 及相应失败日志；未用其他数据补造结果。','']
parts+= [
        '首次评测尝试使用严格的预测 NPZ 读取器，因真值中镜头后方顶点的二维坐标按数据定义为 NaN 而退出；保留 `evidence/evaluation_attempt_01.txt`。首次独立复核误读右图 GPU 失败目录，因文件不存在而退出；保留 `evidence/crosscheck_attempt_01.txt`，修正为最终 CPU 输出后重新复核。两次均为评测脚本问题，没有改动已完成的拟合结果。','',
        '## 复现与推送检查','',
        '在具备原模型资源和外部后端的开发机上，检出记录的代码，将数据仓库固定提交克隆到 `outputs/synthetic_original_20261108/dataset`，并准备本机路径配置；用 `prepare.py --template <本机配置>`，然后运行 `run_views.py`。复现本轮右目 CPU 流程需先有成功的右目原始身体网络预测，再运行 `cpu_fallback.py --reproduce` 及 `device_control.py`；最后运行 `evaluate.py`、`resources.py --phase after`、`collect.py`、`crosscheck.py` 和 `build_report.py`。`resources.py --phase during` 应在右目 CPU 流程开始前运行。相机和图像按数据集原文件读取，权重按配置中的 SHA-256 复验；SMPL-X、SMPL 和 MANO 模型须从各自提供方取得及遵守授权。无法访问外部资源者只能复核公开证据。','',
        '每次推送前检查工作区、暂存及待推送提交、远端状态；不包含权重、数据集原图、NPZ/NPY、视频、完整原始输出、provenance、patch 或凭据，单文件小于 50 MB。投影 JPG 为指定样例的派生图。检查结果见 `evidence/prepush.json`；历史文件不在本次增量审计范围内。']
if (RUN/'evidence/prepush_attempt_01.txt').exists():
    parts+=['','首次暂存检查发现失败日志的文件末尾存在多余空行，未执行提交或推送；移除空行后重新检查。记录见 `evidence/prepush_attempt_01.txt`。']
(RUN/'report.md').write_text('\n'.join(parts)+'\n')
projection=['# 合成真值与原流程投影','',
            '同一姿态、左右相机。左列为合成真值网格（绿色）；右列为原流程预测网格（青色）。绿色点为真值，红色点为预测；点只显示部分上肢和手指关节。真值投影采用渲染的 Brown–Conrady 模型，预测网格采用原流程多项式模型。网格采用近似逐像素深度测试处理自身遮挡，未模拟环境遮挡；跨镜头或投影过长的三角形被过滤。入画关节点不等于无遮挡。','']
for r in d['views']:
    projection+=['## '+r['camera']+' / '+r['side'],'']
    if r['execution_exit']:
        projection += [f'原流程未完成（退出 {r["execution_exit"]}），无预测图。', '',f'![真值投影]({r["truth_image"]})','']
    else:
        projection += ['| 合成真值 | 原流程预测 |','|---|---|',f'| ![真值投影]({r["truth_image"]}) | ![预测投影]({r["prediction_image"]}) |','']
projection+=['[返回报告](report.md)','']
(RUN/'projection.md').write_text('\n'.join(projection))
print('Report and GitHub projection page written')
