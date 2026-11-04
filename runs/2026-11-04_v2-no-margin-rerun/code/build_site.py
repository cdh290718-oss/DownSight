"""Three-way fixed-sample comparison, with no build or remote asset dependency."""
import argparse
import hashlib
import html
import json
import shutil
from pathlib import Path

METHODS = [('original', '历史原流程'), ('edge40', '上一版 v2 · 边缘筛选 40 px'),
           ('no_margin', '本次 v2 · 取消边缘筛选')]
PAIRS = [('new_001', '新样本 001 · 29.0 秒'), ('new_022', '新样本 022 · 31.1 秒'),
         ('new_100', '新样本 100 · 38.9 秒'), ('old_011', '旧样本 11'), ('old_012', '旧样本 12')]


def number(value, scale=1):
    return '—' if value is None else f'{value*scale:.1f}'


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values)/len(values) if values else None


def row(label, values):
    return '<tr><th>'+label+'</th>'+''.join('<td>'+str(v)+'</td>' for v in values)+'</tr>'


def table(rows, caption=''):
    return '<table><caption>'+caption+'</caption><thead><tr><th>指标</th>'+''.join(
        '<th>'+label+'</th>' for _, label in METHODS)+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table>'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--previous-site', type=Path, required=True)
    parser.add_argument('--point-selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    assets = args.output/'assets'; assets.mkdir()
    metrics = json.loads((args.evaluation/'metrics.json').read_text())
    selections = {r['frame']: r for r in json.loads(args.point_selection.read_text())['frames']}
    assert len(metrics['frames']) == 10
    overview = []
    for prefix, label in [('', '全部五对'), ('new', '新三对'), ('old', '旧两对')]:
        selected = [r for r in metrics['frames'] if r['pair'].startswith(prefix)]
        for field, title in [('body_all', '身体全部原有效点'), ('body_shoulders', '肩点')]:
            overview.append(row(label+' · '+title, [number(mean([r['methods'][m][field]['rmse'] for r in selected])) for m, _ in METHODS]))
        overview.append(row(label+' · 手部', [number(mean([h['all']['rmse'] for r in selected for h in r['methods'][m]['hands'].values() if h])) for m, _ in METHODS]))
    sections = []
    translations = {'left_shoulder':'左肩', 'right_shoulder':'右肩', 'right_elbow':'右肘'}
    for pair, title in PAIRS:
        views = []
        for r in [q for q in metrics['frames'] if q['pair'] == pair]:
            figures = []
            for method, label in METHODS:
                q = r['methods'][method]
                assert not q.get('failed')
                source = (args.previous_site/'assets' if method == 'edge40' else args.evaluation)/q['image']
                name = pair+'-'+r['camera']+'-'+method+'.jpg'
                shutil.copyfile(source, assets/name)
                q['image'] = 'assets/'+name
                figures.append(f'<figure><figcaption>{label}</figcaption><a href="assets/{name}" target="_blank" rel="noopener"><img src="assets/{name}" width="1280" height="720" loading="lazy" alt="{title} {r["camera"]} {label}"></a></figure>')
            methods = [r['methods'][m] for m, _ in METHODS]
            rows = [row('身体全部原有效点 RMSE / px', [number(q['body_all']['rmse']) for q in methods]),
                    row('肩点 RMSE / px', [number(q['body_shoulders']['rmse']) for q in methods])]
            for side, side_name in [('left','左手'), ('right','右手')]:
                hands = [q['hands'].get(side) for q in methods]
                rows.append(row(side_name+' RMSE / px', [number(h['all']['rmse']) if h else '—' for h in hands]))
                rows.append(row(side_name+'三角化腕点差 / cm', [number(h['triangulated_wrist_m'],100) if h else '—' for h in hands]))
            rows.append(row('头部前方且入画的顶点数', [q['head_visible'] for q in methods]))
            rows.append(row('局部相交面配对数', [sum(h['proper_crossings'] for h in q['hands'].values() if h) for q in methods]))
            rows.append(row('投影双肩间距 / px', [number(q['projected_shoulder_distance_px']) for q in methods]))
            rows.append(row('骨盆深度 / cm', [number(q['pelvis_z_m'],100) for q in methods]))
            selection = selections[r['frame']]
            restored = '、'.join(translations.get(n,n) for n in selection['restored_points'])
            point_note = f'身体拟合点：上一版 {len(selection["previous_v2_points"])} → 本次 {len(selection["current_v2_points"])}；恢复：{restored}。'
            views.append(f'<article><h3>{r["camera"]}</h3><p class="note">{point_note}</p><div class="comparison">'+''.join(figures)+'</div><details><summary>展开逐项指标与体型投影诊断</summary>'+table(rows)+'<p class="note">肩间距与骨盆深度只用于解释投影变化，没有“越大/越小越好”的方向。头部计数不考虑遮挡；局部相交检查不覆盖整身自交或共面接触。</p></details></article>')
        note = '两手三角化目标有效。' if pair.startswith('new') else '旧两对缺少通过匹配门控的同侧双目手点；仍有单目手部二维约束。'
        sections.append(f'<section id="{pair}" class="pair"><h2>{title}</h2><p class="note">{note}</p>'+''.join(views)+'</section>')
    nav = ''.join(f'<a href="#{pair}">{label}</a>' for pair,label in PAIRS)
    page = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DownSight · 取消边缘筛选后的 v2</title><style>
*{box-sizing:border-box}body{margin:0;background:#f1f5f7;color:#203141;font:16px/1.65 system-ui,-apple-system,"Noto Sans SC",sans-serif}main{max-width:1740px;margin:auto;padding:30px 24px}header{max-width:1080px}h1{font-size:clamp(30px,4vw,48px);line-height:1.2;margin:12px 0 22px}h2{margin:0 0 12px;font-size:25px}h3{margin:22px 0 4px}a{color:#176b8d}.eyebrow{font-size:13px;font-weight:700;letter-spacing:.08em;color:#176b8d}.notice{border-left:4px solid #bf8530;background:#fff4df;padding:12px 16px}.note{font-size:14px;color:#596c7c}.pair,.overview{background:white;border:1px solid #d5e0e6;border-radius:10px;margin:25px 0;padding:20px;scroll-margin-top:15px}nav{display:flex;flex-wrap:wrap;gap:10px;margin:24px 0}nav a{padding:6px 12px;background:white;border:1px solid #d5e0e6;border-radius:5px;text-decoration:none}.comparison{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}figure{margin:0;min-width:0}figcaption{font-size:14px;font-weight:650;background:#edf1f4;padding:8px 10px;border-radius:5px 5px 0 0}figure:last-child figcaption{background:#d6eee5;color:#175f49}img{display:block;width:100%;height:auto}details{margin-top:14px;overflow-x:auto}summary{cursor:pointer;font-weight:600;padding:5px 0}table{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}caption{text-align:left;color:#596c7c;padding:10px 0}th,td{text-align:left;padding:8px 10px;border-bottom:1px solid #dbe4e9}tbody th{font-weight:500}td:last-child{background:#f0faf5}footer{font-size:14px;color:#596c7c;margin:30px 0}.overview{overflow-x:auto}@media(max-width:850px){main{padding:20px 12px}.comparison{grid-template-columns:1fr}.pair{padding:12px}table{min-width:650px}}
</style></head><body><main><header><div class="eyebrow">DOWNSIGHT / SAME FIVE PAIRS / NO BORDER EXCLUSION</div><h1>恢复肩点后，v2 表现如何？</h1><p>同样五对图、十个视角。依次比较历史原流程、上一版带边缘筛选的 v2，以及本次取消边缘筛选后的完整重跑。点击图片查看原分辨率。</p><p>取消的是整条 40 px 边缘规则，恢复满足原置信度和画内条件的肩点及其他边缘点。其他 v2 目标保持不变，第 7 项仍未实施。</p><p class="notice">指标参照自动检测与派生三角化目标，不是真实三维精度。历史外参未在本序列独立验证，旧两对约有 0.10 秒时间差。所有样本均保留，包括退化和局部穿插。</p></header><nav>__NAV__</nav><section class="overview"><h2>同一套原有效点的二维误差</h2>__OVERVIEW__<p class="note">身体/肩部先按视角计算 RMSE，手部先按侧别与视角计算 RMSE，再取平均。三列使用相同观测，单位 px；越低表示越贴近这些观测。</p></section>__SECTIONS__<footer><p>网格渲染处理自身遮挡，不处理真实物体遮挡；三列采用相同投影与裁剪规则。每目仍独立拟合，三角化目标转换到各目坐标；原 MANO 抬升和接缝保留。</p><p><a href="metrics.json">逐视角指标 JSON</a> · <a href="https://github.com/cdh290718-oss/DownSight/tree/main/runs/2026-11-04_v2-no-margin-rerun">报告与证据</a> · <a href="../session-hand6-original-v2-20261104/">上一轮对比页</a></p></footer></main></body></html>'''
    page = page.replace('__NAV__',nav).replace('__OVERVIEW__',table(overview)).replace('__SECTIONS__',''.join(sections))
    (args.output/'index.html').write_text(page)
    (args.output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    manifest = {p.relative_to(args.output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.rglob('*') if p.is_file()}
    (args.output.parent/'site_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('SITE',len(manifest),'files')


if __name__ == '__main__':
    main()
