"""Build the browsable report and a separate offline pack containing byte-exact originals."""
import hashlib
import zipfile
from collections import Counter
from PIL import Image
from sample import RUN, ROOT, OLD, OUT, PRIOR, MANIFEST, grouped, read, write, sha

def package():
    sample=read(RUN/'evidence/sample.json')['rows']
    rows=sorted((read(OUT/(s['pair']+'.json')) for s in sample),key=lambda r:(r['split'],r['pair']))
    old={r['pair']:r for r in read(OLD/'evidence/renders_floor.json')['rows']}
    maps={s:grouped(s) for s in MANIFEST}
    archive=OUT/'fixed_trunk_floor24.zip'
    local=OUT/'gallery'
    title='固定躯干、四肢联合候选 · 双目原图与地面视角'
    doc=[f'# {title}','',
         '每行是同一帧，依次为左目、右目、OBJ 地面正视图、OBJ 地面侧视图。点击图片可单独查看。左右目保留完整画幅，无检测点或网格叠加；在线提供缩小的 JPEG 预览。开发机离线包附逐字节一致的双目原始 PNG 和各帧 OBJ。','',
         '[旧图库的 12 对](#旧图库的-12-对) · [验证集新抽的 12 对](#验证集新抽的-12-对) · [原地面图库](../2026-11-20_rw20000_floor12/report.md) · [候选方法](../2026-11-23_fullbody_joint_candidates/report.md)','']
    html=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>',
          '<style>body{font:16px system-ui;margin:2em auto;padding:0 1em;max-width:1300px;color:#181e23}article{padding:.7em 0 1.4em;border-bottom:1px solid #ccc}figure{margin:0}section{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}img{width:100%;height:auto}a{color:#245a77}@media(max-width:700px){section{grid-template-columns:repeat(2,minmax(0,1fr))}}</style>',
          '<h1>'+title+'</h1><p>点击左、右目可打开完整原始 PNG；每对可单独下载 OBJ。</p>']
    names=['左目原图','右目原图','地面正视','地面侧视']
    checks={}
    for split,heading in [('test','旧图库的 12 对'),('validation','验证集新抽的 12 对')]:
        group=[r for r in rows if r['split']==split]
        assert len(group)==12
        accepted=sum(r['ground']['accepted'] for r in group)
        changed=[r['pair'] for r in group if r['changed_from_golden']]
        tentative=[r['pair'] for r in group if not r['ground']['accepted']]
        intro=('逐帧沿用原报告的地面法向与骨盆前向。' if split=='test' else
               '从 200 对中按源样本标识的固定哈希顺序抽取。S3、S5、S13、S15 各三对，每人来自三个不同序列；抽样不使用姿态误差、是否改姿或地面质量。')
        status=f'地面方向通过内部检查 {accepted}/12；暂定帧为 '+ '、'.join(tentative)+'。'
        delta='相对原版黄金 D v2，固定躯干候选选出了新姿态的样本为 '+('、'.join(changed) if changed else '零')+'。'
        doc+=['## '+heading,'',intro+status+delta,'','| 样本 · 地面方向 | 左目原图 | 右目原图 | OBJ 正面 | OBJ 侧面 |','|---|---|---|---|---|']
        html+=['<h2>'+heading+'</h2><p>'+intro+status+delta+'</p>']
        for r in group:
            key=r['pair'];flag='通过' if r['ground']['accepted'] else '暂定'
            label=r['subject']+' / '+key
            paths=[f'assets/{key}/{k}.jpg' for k in ['left','right','front','side']]
            thumbs=[f'[![{n}]({p})]({p})' for n,p in zip(names,paths)]
            doc+=['| '+' | '.join([label+' · '+flag]+thumbs)+' |']
            html+=[f'<article><h3>{label} · 地面{flag}</h3><p><a href="{key}/{key}.obj" download>下载 OBJ</a></p><section>']
            for name,file in zip(names,['left.png','right.png','front.jpg','side.jpg']):
                html+=[f'<figure><figcaption>{name}</figcaption><a href="{key}/{file}"><img loading="lazy" alt="{name}" src="{key}/{file}"></a></figure>']
            html+=['</section></article>']
            for side,h in zip(['left','right'],r['image_sha256']):
                path=local/key/(side+'.png')
                assert sha(path)==h==sha(MANIFEST[split].parent/maps[split][key][side]['image'])
                with Image.open(path) as im:
                    assert im.format=='PNG' and im.size==(872,872)
            for p in paths:
                with Image.open(RUN/p) as im:
                    assert im.format=='JPEG' and not im.getexif()
            assert sha(local/key/(key+'.obj'))==r['obj_sha256']
            assert all(sha(RUN/'assets'/key/k)==h for k,h in r['assets'].items())
            if split=='test':
                same=all(r['assets'][view]==old[key]['images'][view] for view in ['front.jpg','side.jpg'])
                assert r['ground']['up']==old[key]['up'] and same==(not r['changed_from_golden'])
                r['both_renders_byte_identical_to_old']=same
        doc+=['']
        checks[split]=dict(pairs=len(group),passed=accepted,tentative=tentative,changed_from_golden=changed)
    doc+=['## 如何读图','',
          '旧组固定该帧地面法向和正侧视方位后重新渲染，保留原姿态的样本已逐张与旧正、侧图核对文件哈希。新组单独在左目圈选可见地面区域，结合右目和会话双目标定，由地面纹理匹配与三角化估计法向；稀疏支持不足时补做极线纹理匹配。两者都无法提供足够对应点时，用双眼纹理做平面扫描并标记暂定。地面方向仅用于取景；两视图同尺度、分别居中，画面底边不表示足底接地点。暂定帧的整体倾角有不确定性。[每帧地面拟合与诊断](evidence/gallery.json)','',
          '每帧复用上轮按黄金 D v2 原损失选出的固定躯干解。核对参数库与选择表指纹，检查根位姿、体型及躯干参数；从完整 SMPL-X 生成网格，将 OBJ 写入左相机米制坐标，再读回文件绘图。以官方关节点核对上轮对应帧的全身 MPJPE，容差 0.001 mm。去重同时检查源序列与帧号、双目图像哈希。[样本清单](evidence/sample.json) · [生成脚本](build_views.py)','',
          '## 文件与复现','',
          '开发机的离线图库为 `outputs/fixed_floor24_20261123/fixed_trunk_floor24.zip`。解压打开 `index.html` 即可按样本浏览原始 PNG、两张地面视图和 OBJ。文件摘要见[打包核验](evidence/gallery.json)。复跑需获准访问原图、已冻结的姿态参数、会话标定以及原 SMPL-X 环境，在仓库根目录依次运行本轮 sample.py、run.sh test、run.sh validation、package.py。','',
          '启动时的执行基线为 `c0c225eef7d39c344a7a61fd1e87566e3fd8a47e`。工作期间另一轮独立提交将共享工作树和远端更新至 `0a1c771`；核对差异仅涉及另一轮 runs 目录，本轮依赖与输入指纹一致。在这一新基线上完成发布自检。工作树包含本轮新增文件及既有无关改动。输入直接取自已冻结的两组结果。首轮 val017 的稀疏和极线方法均因对应点不足而退出，重试纳入稠密纹理估计并保留暂定标记。公开证据将机器路径改成仓库相对资源标识。[运行记录](evidence/execution.json)','',
          '推送前自检覆盖工作区、暂存文件、待推提交及远端基线。仅发布本轮源码、脱敏报告、数值摘要及四栏浏览图；检查了文件类型、体积、图像元数据、机器路径和凭据。原始 PNG、OBJ、离线 ZIP、完整输出和既有无关文件留在开发机。']
    (RUN/'report.md').write_text('\n'.join(doc)+'\n')
    (local/'index.html').write_text('\n'.join(html+['</html>']))
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for p in sorted(local.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(local))
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist())==121 and z.testzip() is None
        for name in z.namelist():assert hashlib.sha256(z.read(name)).hexdigest()==sha(local/name)
    checks['offline_files']=dict(Counter(p.suffix for p in local.rglob('*') if p.is_file()))
    write(RUN/'evidence/gallery.json',dict(groups=checks,archive=dict(filename=archive.name,bytes=archive.stat().st_size,sha256=sha(archive)),
          fixed_search_inputs={s:read(PRIOR/('evidence/'+s+'_search.json')) for s in MANIFEST},
          original_floor_sha256=sha(OLD/'evidence/renders_floor.json'),rows=rows,exit_status=0))
if __name__=='__main__':
    package()
