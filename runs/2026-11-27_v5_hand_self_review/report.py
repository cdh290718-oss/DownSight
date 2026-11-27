"""Build the gallery and frame index from the archived selection and geometry replay."""
from collections import Counter
from pathlib import Path
import json
R=Path(__file__).resolve().parent
hands=json.loads((R/'evidence/hands.json').read_text())
cases=json.loads((R.parent/'2026-11-27_validation200_v5/evidence/results.json').read_text())['rows']
def summary(key):
    h=[v for row in hands.values() for v in row.values()]
    return sum(any(v[key] for v in row.values()) for row in hands.values()),sum(v[key]>0 for v in h),sum(v[key] for v in h)
sides=('left','right')
selected=[k for k in sorted(hands) if any(v['archived'] for v in hands[k].values())]
h=[v for row in hands.values() for v in row.values()]
counts=Counter((v['source'],bool(v['archived'])) for v in h)
diff=[(k,s,v['archived'],v['v5']) for k,row in hands.items() for s,v in row.items() if v['archived']!=v['v5']]
regions=Counter()
for v in h:regions.update(v['regions'])
def num(k,s):return hands[k][s]['archived']
table=['| 帧 | 人物 | 左手相交面片对 | 左手来源 | 右手相交面片对 | 右手来源 |','|---|---|---:|---|---:|---|']
for k in selected:
    l,r=hands[k].values()
    table.append(f"| {k} | {cases[k]['subject']} | {l['archived']} | {l['source']} | {r['archived']} | {r['source']} |")
(R/'frames.md').write_text('# 全部 100 个手内自交帧\n\ncam3、cam4 分别是从左、右目 WiLoR 得到的候选，no candidate 表示沿用 v4。左右手按人物解剖侧。\n\n'+'\n'.join(table)+'\n')
lines=['# 黄金 D v5 手内自交：同一只手的 v4、v5 对照','','[全部 100 帧，逐帧标出左右手、相交面片对数与候选来源](frames.md) · [400 只手的旋转替换实验](evidence/hands.json)','',
'原评估中仅左手 15 帧、仅右手 55 帧、双手各自发生自交 30 帧。130 只自交手都实际采用了 WiLoR：左目候选 64 只，右目候选 66 只。采用 WiLoR 共 280 只，另外 120 只保留 v4，均无自交。这里的自交均发生在同一只手的网格内部。','',
'## 单独替换指姿或腕旋转','','固定同一份 SMPL-X、身体和体型，只交换已封存参数，重建网格。每只手检查无共享顶点的三角面之间的严格内部相交。','',
'| 参数来源 | 自交帧 | 自交手 | 相交三角面片对合计 |','|---|---:|---:|---:|']
for key,label in [('v4','v4'),('wrist_only','仅用 v5 腕旋转'),('finger_only','仅用 v5 指姿'),('v5','完整 v5')]:
    lines.append('| '+label+' | '+' | '.join(map(str,summary(key)))+' |')
assert not diff
overlap=sum(any(h['finger_only'] for h in hands[k].values()) for k in selected)
lines+=['',f'逐手重建与归档的 400 个计数全部一致。仅换指姿已在原有 100 帧中的 {overlap} 帧重现自交。最常见的区域组合是食指–中指（{regions["index/middle"]} 对），其次是中指–无名指（{regions["middle/ring"]} 对）。',
'WiLoR 候选提供的手指旋转迁移到 SMPL-X 后固定，最终两目选解使用二维手点误差；该评分没有手内自交项。上述消融检验迁移后的指姿和腕旋转，未对原生 MANO 网格评估相交。','',
'## 渲染','','左列 v4，右列 v5。同一只手使用相同正交视角、中心和比例；每只手的两排相差 90°。红色标记参与严格相交的三角面，遮挡可能盖住部分红面。图中 pairs 是相交面片对数，不能换算成穿入深度。解剖部位按面片顶点的蒙皮权重近似归类。','']
cn=dict(index='食指',middle='中指',pinky='小指',ring='无名指',thumb='拇指',palm='掌部')
for k in ('val003','val001','val044','val184'):
    lines+=['### '+k,'']
    for s in sides:
        a=hands[k][s]
        if a['archived']:
            lines+=['人物'+('左手' if s=='left' else '右手')+f"共 {a['archived']} 对，来源 {a['source']}。主要相交部位 "+'、'.join('–'.join(cn[name] for name in pair.split('/'))+f'（{n}）' for pair,n in Counter(a['regions']).most_common(3))+'。','']
    lines+=['![v4 和 v5 手部局部网格](assets/'+k+'.jpg)','']
lines+=['选样包括最少相交的并列帧之一、接近中位数的单手与双手案例，以及总面片对数最多的一帧。完整统计使用全部 200 对。','','## 执行与发布','','基线 a887345，加本目录新增文件；工作树有无关改动。运行 bash '+str(R.relative_to(R.parents[1])/'run.sh')+' 后运行本目录 report.py。需要受限 SMPL-X 模型与前轮验证姿态，逐帧核验 v4 和 v5 哈希。原始数据、数组与网格留在开发机。公开增量逐项检查路径、凭据、文件类型与尺寸，核对工作区、暂存、待推送提交及远端。']
(R/'report.md').write_text('\n'.join(lines)+'\n')
print(lines[10:19])
