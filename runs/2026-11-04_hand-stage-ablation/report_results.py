"""Append auditable tables to the method report once all fits are complete."""
import json
from pathlib import Path
RUN=Path(__file__).resolve().parent
d=json.loads((RUN/'evidence/metrics.json').read_text())
v=json.loads((RUN/'evidence/execution_and_validation.json').read_text())
parts=['## 完整结果','',
       '下表是固定新三对的全部六视角；手部均值包含左右手。骨盆深度沿相机 z 轴，肩宽为投影肩关节间距。二维误差单位 px，三维手误差单位 mm。','',
       '| 阶段 | 组别 | 肩误差 px | 骨盆深度 m | 投影肩宽 px | 手 2D px | 手 3D mm |',
       '|---|---|---:|---:|---:|---:|---:|']
for stage in ['rigid','body_pose','wrist','contour','guided','fused']:
    for group in ['A','B','C']:
        q=d['summary'][group][stage]
        parts.append(f'| {stage} | {group} | {q["shoulder_rmse_px"]:.4f} | {q["pelvis_z_m"]:.6f} | {q["projected_shoulder_width_px"]:.4f} | {q["hand_rmse_2d_px"]:.4f} | {q["hand_rmse_3d_m"]*1000:.4f} |')
a=d['summary']['A']['rigid'];b=d['summary']['B']['rigid']
parts+=['','### 刚体假设','',f'A→B：平均骨盆深度变化 {(b["pelvis_z_m"]-a["pelvis_z_m"])*100:.4f} cm，肩误差变化 {b["shoulder_rmse_px"]-a["shoulder_rmse_px"]:+.4f} px，肩宽变化 {b["projected_shoulder_width_px"]-a["projected_shoulder_width_px"]:+.4f} px。B/C 刚体所有保存参数逐位一致。','',
        '在这批数据和当前权重下，移除刚体手部项使身体变近，支持存在“刚体手部约束推远整体”的效应。它不能证明三角化错误：本轮同时移除了 WiLoR 二维及三角化三维项，也不能证明该效应是肩部偏差的唯一原因。','',
        '| 视角 | B−A 刚体深度 cm | B−A 刚体肩误差 px | B−A 刚体肩宽 px |','|---|---:|---:|---:|']
for r in d['frames']:
    a=r['methods']['A']['stages']['rigid'];b=r['methods']['B']['stages']['rigid']
    parts.append(f'| {r["pair"]}/{r["camera"]} | {(b["pelvis_z_m"]-a["pelvis_z_m"])*100:+.4f} | {b["shoulder_rmse_px"]-a["shoulder_rmse_px"]:+.4f} | {b["projected_shoulder_width_px"]-a["projected_shoulder_width_px"]:+.4f} |')
parts+=['','### 最终手部退化检查','',
        '门槛事先写入 brief：融合手二维均值增加不超过 1 px，任一手／视角不超过 3 px；三维均值增加不超过 5 mm，任一手／视角不超过 15 mm。需全部满足。','',
        '| 相对 A | 2D 均值变化 px | 2D 最差变化 px | 3D 均值变化 mm | 3D 最差变化 mm | 通过 |',
        '|---|---:|---:|---:|---:|---|']
for group,c in d['comparisons'].items():
    parts.append(f'| {group} | {c["mean_delta_2d_px"]:+.4f} | {c["max_delta_2d_px"]:+.4f} | {c["mean_delta_3d_m"]*1000:+.4f} | {c["max_delta_3d_m"]*1000:+.4f} | {c["hand_noninferiority_passed"]} |')
for group,c in d['comparisons'].items():
    for field,unit,scale in [('delta_2d_px','px',1),('delta_3d_m','mm',1000)]:
        worst=max(c['changes'],key=lambda x:x[field]);parts.append(f'\n{group} 的 {unit} 最大退化：{worst["pair"]}/{worst["camera"]}/{worst["side"]}，{worst[field]*scale:+.4f} {unit}。\n')
parts+=['','### 重载与其他观察','']
for group in ['A','B','C']:
    rows=[r for r in v['records'] if r['group']==group]
    passed=sum(bool(r['validation'].get('passed')) for r in rows)
    surface=sum(bool(r['validation'].get('local_surface_accepted')) for r in rows)
    parts.append(f'- {group}：最终验证 {passed}/{len(rows)}，局部表面优化接受 {surface}/{len(rows)}；最终头部入画顶点计数合计 {d["summary"][group]["fused"]["head_visible_vertices"]}。局部接受不代表全网格无相交。')
parts+=['','B/C 全部视角的最终完成尝试退出状态均为 0；中断尝试的退出状态未知，另行保留。输入／资源哈希保持一致，详见 `evidence/execution_and_validation.json`。A 的肩、手二维及三维最终误差与上一轮公开证据一致，评估未更换基准。','',
        '### C 的解释边界','',
        'C 的直接手部梯度隔离已经验证，但身体／头部目标及候选接受规则仍会影响整体。因此，C 的骨盆深度不要求保持刚体值。不能仅凭整体仍有移动认定隔离失效，也不能宣称本实验冻结了整体。比较 B→C 衡量的是这套梯度分配策略的效果。','']
path=RUN/'report.md';original=path.read_text().replace('本轮正在执行；数值结论将在完整评估后追加。','本轮已完成固定新三对的 B/C 对照和逐阶段评估。')
original=original.split('\n## 完整结果')[0]
path.write_text(original.rstrip()+'\n\n'+'\n'.join(parts)+'\n')
print('Appended complete stage and validation tables')
