"""Small static A/B/C comparison with complete stage metrics and per-hand deltas."""
import html,json
from pathlib import Path
RUN=Path(__file__).resolve().parent
d=json.loads((RUN/'evidence/metrics.json').read_text())
names={'A':'A · 当前 v2','B':'B · 刚体不用手约束','C':'C · B + 手部梯度仅到手臂'}
stages={'raw':'网络初值','rigid':'刚体结束','body_pose':'姿态结束','wrist':'腕部细化','contour':'轮廓细化','guided':'Guided 身体','fused':'最终融合'}
parts=['''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>新三对：手部约束阶段消融</title><style>
body{font:16px/1.6 system-ui,sans-serif;max-width:1500px;margin:32px auto;padding:0 20px;color:#182433;background:#f5f7fa}h1,h2{line-height:1.3}p{max-width:1100px}.table{overflow-x:auto}table{border-collapse:collapse;background:white;width:100%;margin:12px 0 28px}td,th{padding:9px 12px;text-align:right;border-bottom:1px solid #dce2e8}td:first-child,th:first-child{text-align:left}th{background:#e8eef5}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0;background:white;padding:8px}img{width:100%;height:auto}figcaption{font-size:14px}.card{background:white;border:1px solid #dce2e8;padding:18px;margin:24px 0}.bad{color:#a52b24}.ok{color:#14633d}a{color:#1558a6}summary{cursor:pointer}small{color:#536274}@media(max-width:850px){.grid{grid-template-columns:1fr}body{padding:0 12px}}
</style><h1>新三对：手部约束阶段消融</h1><p>A 是恢复肩点后的 v2；B 仅移除刚体阶段的 WiLoR 二维和三角化三维手约束；C 在 B 上，将后续手部、腕环和前臂轮廓目标的直接梯度限制到锁骨、肩、肘、腕。Sapiens 身体腕点保留，身体和头部观测仍可调整整体。</p><p>六视角、固定自动观测，非人工真值。身体数值是六视角 RMSE 的均值，手部是十二个手／视角 RMSE 的均值。第 7 项未实施，原 MANO 替换与接缝融合保留。SMPL-X 手误差与融合手误差分别列出。</p><p><a href="metrics.json">完整数值 JSON</a> · <a href="https://github.com/cdh290718-oss/DownSight/tree/main/runs/2026-11-04_hand-stage-ablation">代码与报告</a></p>''']
parts.append('<h2>刚体与最终结果</h2><div class="table"><table><tr><th>阶段 / 组别</th><th>肩 RMSE px</th><th>骨盆深度 m</th><th>投影肩宽 px</th><th>手 2D RMSE px</th><th>手 3D RMSE mm</th></tr>')
for stage in ['rigid','fused']:
    for group in names:
        s=d['summary'][group][stage]
        parts.append(f'<tr><td>{stages[stage]} / {group}</td><td>{s["shoulder_rmse_px"]:.2f}</td><td>{s["pelvis_z_m"]:.3f}</td><td>{s["projected_shoulder_width_px"]:.1f}</td><td>{s["hand_rmse_2d_px"]:.2f}</td><td>{s["hand_rmse_3d_m"]*1000:.2f}</td></tr>')
parts.append('</table></div><details><summary>展开全部阶段均值</summary><div class="table"><table><tr><th>阶段 / 组别</th><th>肩 RMSE px</th><th>骨盆深度 m</th><th>投影肩宽 px</th><th>手 2D RMSE px</th><th>手 3D RMSE mm</th></tr>')
for stage,label in stages.items():
    for group in names:
        s=d['summary'][group][stage]
        parts.append(f'<tr><td>{label} / {group}</td><td>{s["shoulder_rmse_px"]:.2f}</td><td>{s["pelvis_z_m"]:.3f}</td><td>{s["projected_shoulder_width_px"]:.1f}</td><td>{s["hand_rmse_2d_px"]:.2f}</td><td>{s["hand_rmse_3d_m"]*1000:.2f}</td></tr>')
parts.append('</table></div></details><p><small>最终融合行使用替换后的 MANO 网格；其余行使用 SMPL-X 网格。三维腕点采用身体骨骼腕中心，其余手点由网格回归。C 隔离直接梯度，但候选评分、回退及身体／头部目标仍会间接影响整体位姿。</small></p>')
parts.append('<h2>手部是否明显变差</h2><p>预先约定的实用门槛：相对 A，融合手二维均值增加 ≤ 1 px、最差手／视角 ≤ 3 px；三维均值增加 ≤ 5 mm、最差手／视角 ≤ 15 mm。这不是统计显著性检验。</p>')
for group,q in d['comparisons'].items():
    passed=q['hand_noninferiority_passed'];parts.append(f'<p class="{"ok" if passed else "bad"}"><b>{group}：{"通过" if passed else "未通过"}</b>。二维均值变化 {q["mean_delta_2d_px"]:+.2f} px，最差 {q["max_delta_2d_px"]:+.2f} px；三维均值变化 {q["mean_delta_3d_m"]*1000:+.2f} mm，最差 {q["max_delta_3d_m"]*1000:+.2f} mm。</p>')
parts.append('<h2>其他检查</h2><p>最终头部入画顶点合计：A 1615，B 2，C 3。局部表面检查通过视角：A 6/6、B 5/6、C 2/6。B 的 new_001 / cam4 左侧前臂／接缝局部检查发现 47 对相交面，因此 B 是有收益的候选方案，尚不能视为几何全面验收通过。<a href="surface_checks.json">局部检查证据及范围</a>。</p>')
parts.append('<h2>逐视角最终投影</h2>')
for r in d['frames']:
    parts.append(f'<section class="card"><h3>{html.escape(r["pair"])} · {html.escape(r["camera"])}</h3><div class="grid">')
    for group,label in names.items():
        m=r['methods'][group];s=m['stages']['fused']
        hands=' / '.join(f'{side} {s["hands"][side]["rmse_2d_px"]:.2f}' for side in ['left','right'])
        surface=m['validation']['local_surface_accepted']
        parts.append(f'<figure><a href="{m["image"]}"><img loading="lazy" src="{m["image"]}" alt="{r["pair"]} {r["camera"]} {label}"></a><figcaption><b>{label}</b><br>肩 {s["shoulder_rmse_px"]:.2f} px · 深度 {s["pelvis_z_m"]:.3f} m · 肩宽 {s["projected_shoulder_width_px"]:.1f} px<br>融合手 2D：{hands} px<br><span class="{"ok" if surface else "bad"}">局部表面检查：{"通过" if surface else "未通过"}</span></figcaption></figure>')
    parts.append('</div><details><summary>该视角各阶段及左右手数值</summary><div class="table"><table><tr><th>阶段 / 组别</th><th>肩 px</th><th>深度 m</th><th>肩宽 px</th><th>左手 2D px</th><th>右手 2D px</th><th>左手 3D mm</th><th>右手 3D mm</th></tr>')
    for stage,label in stages.items():
        for group in names:
            s=r['methods'][group]['stages'][stage];l=s['hands']['left'];rr=s['hands']['right']
            parts.append(f'<tr><td>{label} / {group}</td><td>{s["shoulder_rmse_px"]:.2f}</td><td>{s["pelvis_z_m"]:.3f}</td><td>{s["projected_shoulder_width_px"]:.1f}</td><td>{l["rmse_2d_px"]:.2f}</td><td>{rr["rmse_2d_px"]:.2f}</td><td>{l["rmse_3d_m"]*1000:.2f}</td><td>{rr["rmse_3d_m"]*1000:.2f}</td></tr>')
    parts.append('</table></div></details></section>')
(RUN/'site/index.html').write_text('\n'.join(parts)+'\n</html>\n')
print('Built 18-image comparison site')
