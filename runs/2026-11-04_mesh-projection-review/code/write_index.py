"""Create frame selection and metric index for the visualization-only run."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--selection',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();d=json.loads(a.selection.read_text());rows=d['selected'];s=['# 双目网格投影索引','','每张图上半是 cam2、下半是 cam3；每一目中左侧为 v0，右侧为二阶段，并各有左右手相同区域的放大图。青色是可见网格、洋红色是可见头部；绿色是 DFM 手臂和 21 点骨架（模型参考，不是真值）；橙色圈是自动 Sapiens 腕点，蓝点是已匹配并通过上轮二维筛选的 WiLoR 手点。WiLoR 点在 v0 图上也显示，便于对照；v0 拟合并未使用它们。','', 'cam3 单目蓝点仅表示自动观测被选中；上一轮手部阶段使用的是 cam2 二维和有效双目三维，只有 cam3 点并不等于该手参与拟合。放大图以 DFM 手部参考为中心；画外或严重偏离的预测可能不在局部裁剪内，应同时看全图。','', '相机的网格自身遮挡用逐像素射线与三角面交点深度缓冲处理；场景遮挡仅使用部分通过左右一致性检查的双目深度估计（4 cm 保护裕量），并不覆盖所有像素，也可能误判。DFM 骨架与自动点作为参考标记不进行遮挡剔除。','', '腕为与 DFM 的三维欧氏距离；手部为相对于各自腕部的 21 点三维平均距离；掌面角为与 DFM 的角度差。单位分别为 cm、cm、度。数值来自上一轮的逐帧证据，本轮不重算指标或改变拟合。','', '| 帧 | 选择理由 | 腕 L / R cm | 手部 v0 L / R → 二阶段 L / R cm | 掌面角 v0 L / R → 二阶段 L / R ° |','|---|---|---:|---:|---:|']
for x in rows:
 f=x['frame'];h=x['hands'];w=x['wrists'];hfmt=lambda mode,key:f"{h[mode]['left'][key]:.2f} / {h[mode]['right'][key]:.2f}";s.append(f"| [{f:05d}](comparisons/frame_{f:05d}.jpg) | {'；'.join(x['reasons'])} | {w['left']:.2f} / {w['right']:.2f} | {hfmt('v0','mean_relative_21_cm')} → {hfmt('two_stage','mean_relative_21_cm')} | {hfmt('v0','palm_angle_deg')} → {hfmt('two_stage','palm_angle_deg')} |")
s+=['','## 图像','']
for x in rows:
 f=x['frame'];s += [f"### {f:05d}：{'；'.join(x['reasons'])}",'',f'![{f:05d} 双目网格与手部对照](comparisons/frame_{f:05d}.jpg)','']
a.output.write_text('\n'.join(s),encoding='utf-8')
