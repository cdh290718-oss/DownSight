"""Small buildless GitHub Pages comparison; all numbers from fixed metrics."""
import argparse,html,json,shutil,hashlib
from pathlib import Path

def fmt(x,scale=1):return '—' if x is None else f'{x*scale:.1f}'
def main():
 p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);assets=a.output/'assets';assets.mkdir();m=json.loads((a.evaluation/'metrics.json').read_text());shutil.copyfile(a.evaluation/'metrics.json',a.output/'metrics.json');cards=[]
 labels={'new_001':'29.0 秒 · 新样本 001','new_022':'31.1 秒 · 新样本 022','new_100':'38.9 秒 · 新样本 100','old_011':'旧样本 11','old_012':'旧样本 12'}
 order=list(labels)
 for pair in order:
  rows=[r for r in m['frames'] if r['pair']==pair];parts=[]
  for r in rows:
   methods=r['methods'];fig=[]
   for method,label in [('original','历史原流程'),('v2','本次 v2 · 第 7 项未做')]:
    q=methods.get(method,{})
    if q.get('failed') or 'image' not in q:
     fig.append('<figure><figcaption>'+label+'</figcaption><div class="missing">本次运行失败，保留该样本</div></figure>');continue
    name=q['image'];shutil.copyfile(a.evaluation/name,assets/name)
    fig.append(f'<figure><figcaption>{label}</figcaption><a href="assets/{name}" target="_blank" rel="noopener"><img src="assets/{name}" width="1280" height="720" loading="lazy" alt="{labels[pair]} {r["camera"]} {label}"></a></figure>')
   metrics=[]
   for label,key in [('身体（原全部点）','body_all'),('身体（v2 保留点）','body_retained')]:
    b=methods['original'][key];v=methods.get('v2',{}).get(key,{})
    metrics.append(f'<tr><th>{label}</th><td>{fmt(b.get("rmse"))}</td><td>{fmt(v.get("rmse"))}</td><td>{b["count"]} 点</td></tr>')
   for side,cn in [('left','左手'),('right','右手')]:
    b=methods['original']['hands'].get(side);v=methods.get('v2',{}).get('hands',{}).get(side)
    for label,key in [('原全部点','all'),('v2 保留点','retained')]:
     metrics.append(f'<tr><th>{cn}（{label}）</th><td>{fmt(b[key]["rmse"] if b else None)}</td><td>{fmt(v[key]["rmse"] if v else None)}</td><td>{b[key]["count"] if b else 0} 点</td></tr>')
   detail=[]
   for side,cn in [('left','左手'),('right','右手')]:
    b=methods['original']['hands'].get(side);v=methods.get('v2',{}).get('hands',{}).get(side)
    detail.append(f'<p>{cn}局部相交面配对：{b["proper_crossings"] if b else "—"} → {v["proper_crossings"] if v else "—"}；三角化腕点差：{fmt(b.get("triangulated_wrist_m") if b else None,100)} → {fmt(v.get("triangulated_wrist_m") if v else None,100)} cm。</p>')
   b=methods['original'];v=methods.get('v2',{});detail.append(f'<p>本目满足“前方且入画”的头顶点：{b["head_visible"]} → {v.get("head_visible","—")}。此计数不考虑遮挡。</p>')
   parts.append(f'<div class="view"><h3>{r["camera"]}</h3><div class="comparison">{"".join(fig)}</div><div class="numbers"><table><caption>同一自动观测下的二维 RMSE / px，越低越贴近观测</caption><thead><tr><th>部位</th><th>原流程</th><th>v2</th><th>评测点数</th></tr></thead><tbody>{"".join(metrics)}</tbody></table><details><summary>三维一致性与接缝检查</summary>{"".join(detail)}<p>局部检查不覆盖共面接触或整身自交；三角化腕点差是拟合目标的一致性，不是真值误差。</p></details></div></div>')
  dt=rows[0]['time_difference_seconds']*1000;note='双目手点有效；外参仍未经本序列验证。' if pair.startswith('new') else '缺检 / 身份冲突使双目手部 3D 不可用；仍运行并展示全部结果。'
  cards.append(f'<section class="pair" id="{pair}"><div class="pair-title"><h2>{labels[pair]}</h2><span>两目时间差 {dt:.3f} ms</span></div><p class="note">{note}</p>{"".join(parts)}</section>')
 page='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DownSight · 原流程与 v2</title><style>
:root{color-scheme:light;--ink:#192633;--muted:#536579;--line:#d9e2e8;--accent:#176c8d}*{box-sizing:border-box}body{margin:0;background:#f3f6f7;color:var(--ink);font:16px/1.65 system-ui,-apple-system,"Noto Sans SC",sans-serif}main{max-width:1500px;margin:auto;padding:32px 24px}header{max-width:1000px;margin:12px 0 30px}h1{font-size:clamp(30px,4vw,48px);line-height:1.2;margin:12px 0 20px}h2{font-size:25px;margin:0}h3{margin:8px 0}p{margin:10px 0}a{color:var(--accent)}.eyebrow{font-size:13px;font-weight:700;letter-spacing:.1em;color:var(--accent)}.chips{display:flex;gap:8px;flex-wrap:wrap}.chips span,nav a{padding:6px 12px;border:1px solid var(--line);border-radius:6px;background:white}.warning{border-left:4px solid #aa7324;background:#fff5e6;padding:12px 16px;color:#694916}nav{display:flex;flex-wrap:wrap;gap:9px;margin:24px 0}nav a{text-decoration:none}.pair{background:white;border:1px solid var(--line);border-radius:12px;margin:26px 0;padding:22px;scroll-margin-top:12px}.pair-title{display:flex;align-items:baseline;justify-content:space-between;gap:12px;flex-wrap:wrap}.pair-title span,.note{color:var(--muted);font-size:14px}.view{margin-top:22px}.comparison{display:grid;grid-template-columns:1fr 1fr;gap:12px}figure{margin:0;min-width:0}figcaption{font-weight:650;padding:7px 11px;background:#eaf0f4;border-radius:6px 6px 0 0}figure:nth-child(2) figcaption{background:#dceef3;color:#145d78}img{display:block;width:100%;height:auto;background:#dfe4e8}.numbers{display:grid;grid-template-columns:1.15fr 1fr;gap:24px;margin-top:16px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px}caption{text-align:left;color:var(--muted);font-size:13px;margin:0 0 7px}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}tbody th{font-weight:500}details{font-size:14px;color:var(--muted)}summary{cursor:pointer;color:var(--ink);padding:6px 0}.missing{padding:70px 15px;background:#fff0ef;color:#9a322c}.method{max-width:1000px;margin:32px 0}footer{border-top:1px solid var(--line);padding:20px 0;color:var(--muted);font-size:14px}@media(max-width:760px){main{padding:20px 12px}.pair{padding:14px}.comparison,.numbers{grid-template-columns:1fr}.numbers{gap:10px}h2{font-size:21px}}
</style></head><body><main><header><div class="eyebrow">DOWNSIGHT / FIXED FIVE-PAIR COMPARISON / V2</div><h1>原流程与 v2<br>固定五对图，逐视角对照</h1><p>左侧是原实验保存的完整结果，右侧是本次重新拟合的 v2。两侧使用相同渲染方式；点击图片打开原分辨率查看。</p><div class="chips"><span>5 对 / 10 个视角</span><span>第 7 项暂缓</span><span>保留原 MANO 抬升与接缝</span><span>不重新运行网络</span></div><p class="warning">探索结果：使用历史固定外参，未证明该外参在本序列准确；旧两对约有 0.10 秒时间差。二维误差与三角化一致性都不等于真实三维精度，不能只据数值下降判断采用。</p></header><nav>__NAV__</nav>__CARDS__<section class="method"><h2>本次修改与比较口径</h2><p>深度保护排除头颈；删除“腕到相机距离不得低于初始值 × 0.85”；加入 WiLoR 双目腕 / 手点三角化及各向异性不确定度；剔除边缘 40 px 内观测（髋、膝、踝、鼻仍按原规则）；加入两目头部入画软惩罚；按身体、腕、左右手及二维 / 三维分别归一化。</p><p>沿用原刚体 → 身体 → 腕点 → 轮廓 → MANO 辅助身体 → 原有界接缝顺序。每个视角独立拟合，共用转换到本目坐标的三角化观测；没有改成一份共享网格，没有新增 MANO 三维放置或腕朝向对齐阶段。保留原 head reference 0.1。</p><p>“原全部点”和“v2 保留点”均用同一份冻结自动观测分别评测两种结果，避免靠删除困难点造成表面改善。缺失项显示“—”。新样本 022 的 cam4 原结果包含历史手部恢复；本次沿用对应缓存和匹配。历史成品与当前重跑之间还可能存在历史实现差异，结果不能当作单因素消融。</p><p>网格渲染处理模型自身遮挡；不处理真实物体的遮挡。近相机或过长的投影三角形会被裁剪，两侧使用同一规则。</p><p><a href="metrics.json">下载逐帧评测 JSON</a> · <a href="https://github.com/cdh290718-oss/DownSight/tree/main/runs/2026-11-04_session-hand6-v2">代码、报告与证据</a> · <a href="../session-hand6-stereo-joint-20261025/">查看原实验网页</a></p></section><footer>DownSight · 固定样本探索对照 · 所有历史成品保留</footer></main></body></html>'''
 aggregate=[]
 for subset,label in [('all','全部五对'),('new','新三对：有手部三角化'),('old','旧两对：无手部三角化')]:
  rows=[r for r in m['frames'] if subset=='all' or r['pair'].startswith(subset)];cells=[]
  for method in ['original','v2']:
   valid=[r['methods'][method] for r in rows if not r['methods'][method].get('failed')]
   body=[q['body_all']['rmse'] for q in valid if q['body_all']['rmse'] is not None]
   hands=[h['all']['rmse'] for q in valid for h in q['hands'].values() if h and h['all']['rmse'] is not None]
   cells.append((sum(body)/len(body) if body else None,sum(hands)/len(hands) if hands else None))
  aggregate.append(f'<tr><th>{label}</th><td>{fmt(cells[0][0])} → {fmt(cells[1][0])}</td><td>{fmt(cells[0][1])} → {fmt(cells[1][1])}</td></tr>')
 overview='<section class="method"><h2>全部原有效点的误差</h2><p>原流程 → v2；单位 px。身体按视角、手按侧别与视角先计算 RMSE，再取平均。下方逐图同时列出筛选后误差。</p><table><thead><tr><th>样本</th><th>身体</th><th>手部</th></tr></thead><tbody>'+''.join(aggregate)+'</tbody></table></section>'
 page=page.replace('__NAV__',''.join(f'<a href="#{x}">{labels[x]}</a>' for x in order)).replace('__CARDS__',overview+''.join(cards));(a.output/'index.html').write_text(page)
 files={str(p.relative_to(a.output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in a.output.rglob('*') if p.is_file()};(a.output.parent/'site_manifest.json').write_text(json.dumps(dict(files=files),indent=2)+'\n');print('SITE',len(files),'files')
if __name__=='__main__':main()
