"""Create a non-overwriting review of the saved measurements."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('analysis',OUT/'analyze.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
data=json.loads((OUT/'measurements.json').read_text())
frames=data['frames']
bone=list(csv.DictReader((OUT/'bones.csv').open()))
CN={'left_shoulder':'左肩','right_shoulder':'右肩','left_elbow':'左肘','right_elbow':'右肘',
    'left_wrist':'左腕','right_wrist':'右腕','left_hip':'左髋','right_hip':'右髋',
    'left_knee':'左膝','right_knee':'右膝','left_ankle':'左踝','right_ankle':'右踝','nose':'鼻'}
BN={'left_upper_arm':'左肩—肘','right_upper_arm':'右肩—肘','left_forearm':'左肘—腕',
    'right_forearm':'右肘—腕','shoulder_width':'双肩宽'}
def fmt(x,n=1):
    if x is None or x=='':return '—'
    return ('%.'+str(n)+'f')%float(x)


def draw_segmentation():
    fig,axes=plt.subplots(2,2,figsize=(13,11))
    first=frames[0];rows={x['joint']:x for x in first['rows']}
    for side,axrow in zip(('left','right'),axes):
        rec=rows[side+'_elbow']
        for k,ax in enumerate(axrow):
            cam='cam%d'%(k+3)
            impath=a.ROOT/'images'/cam/'gen_00000.png'
            image=cv2.cvtColor(cv2.imread(str(impath)),cv2.COLOR_BGR2RGB)
            labels=np.load(a.C/'observations/segmentation/frames'/('gen_00000_'+cam)/'labels.npy')
            mask23=labels==23;maskarms=np.isin(labels,[7,11,16,20])
            overlay=image.copy().astype(float)
            overlay[mask23]=.55*overlay[mask23]+.45*np.array([255,170,0])
            overlay[maskarms]=.55*overlay[maskarms]+.45*np.array([0,255,0])
            ax.imshow(overlay.astype(np.uint8))
            s=np.array(rec['uv_'+cam]);d=np.array(rec['dfm_uv_'+cam]);c=np.array(rec['c_uv_'+cam])
            ax.scatter(*s,c='cyan',edgecolor='black',s=95,zorder=3,label='Sapiens')
            ax.scatter(*d,c='white',edgecolor='black',s=80,marker='D',zorder=3,label='DFM projected')
            ax.scatter(*c,c='magenta',s=95,marker='x',linewidth=2,zorder=3,label='C projected')
            points=np.stack([s,d,c]);lo=points.min(0)-115;hi=points.max(0)+115
            ax.set_xlim(max(0,lo[0]),min(1280,hi[0]));ax.set_ylim(min(720,hi[1]),max(0,lo[1]));ax.axis('off')
            ax.set_title('%s elbow / %s | amber: Upper_Clothing, green: arm labels'%(side,cam))
            ax.legend(loc='lower left',fontsize=8)
    fig.tight_layout()
    with (OUT/'elbow_segmentation_00000.png').open('xb') as f:fig.savefig(f,format='png',dpi=140,bbox_inches='tight')
    plt.close(fig)


def verify():
    p1=np.c_[np.eye(3),np.zeros(3)];p2=np.c_[a.R,a.t]
    values={}
    for fr in frames:
        idx=fr['dfm_index'];uv3=a.dfm['cam3_body_uv'][idx];uv4=a.dfm['cam4_body_uv'][idx]
        rays3=np.array([a.ray(x,a.cal[0]) for x in uv3]);rays4=np.array([a.ray(x,a.cal[1]) for x in uv4])
        xyz4=cv2.triangulatePoints(p1,p2,(rays3[:,:2]/rays3[:,2,None]).T,(rays4[:,:2]/rays4[:,2,None]).T)
        reconstructed=(xyz4[:3]/xyz4[3]).T
        err=np.linalg.norm(reconstructed-a.dfm['cam3_body_xyz'][idx],axis=1)
        assert np.max(err)<1e-8
        # Reproduce the old report's independently saved rounded triangle ranges.
        if fr['frame']==0:
            old={r['joint']:r for r in csv.DictReader((a.PREV/'joints.csv').open())}
            diffs=[abs(r['tri_range_cm']/100-float(old[r['joint']]['range_m'])) for r in fr['rows'] if r['joint'] in old]
            assert max(diffs)<.0006
        values[str(fr['frame'])]=dict(independent_cv2_dlt_reference_max_error_m=float(err.max()),
             original_dfm_uv_reprojection_max_error_px=fr['projection_max_error_px'],
             image_hashes_match_sapiens=True)
    pp=np.array([a.project(p,a.cal[0]) for p in a.cj]);ss=a.body['smplx_joint_img'][:,:2]
    maxerr=float(np.linalg.norm(pp-ss,axis=1).max());assert maxerr<.001
    values['C_max_reproject_difference_vs_saved_px']=maxerr
    values['limitations']='Synthetic consistency checks reuse DFM-derived relative pose and do not independently validate calibration or ground truth.'
    with (OUT/'validation.json').open('x') as f:json.dump(values,f,indent=2);f.write('\n')
    return values


def report():
    s=[]
    def line(x=''):s.append(x)
    line('# Gen 双目三角化与 DFM / 实验 C 对照')
    line()
    line('## 数据范围和结论')
    line('使用同一视频的 `gen_00000`、`gen_00270` 两个既定时刻，每个时刻各两张双目图。预先选择 00270 为源清单中 00000 后的下一时刻；只为其两张图运行既有 Sapiens2-1B。原结果、标定与流程代码均未改动。搜索到的其他 Sapiens 缓存均是 00000 的重复：两侧各 7 份，同侧关节坐标和分数完全一致。另有 18 个时刻仅找到图像和 DFM，没有已有 Sapiens 结果，本次未批量推理。')
    line()
    line('实验 C 只有 00000/cam3 的身体姿态拟合；00270 没有该时刻的 C 结果。两时刻的肩肘腕骨长可以与 00000 的 C 身体骨长作条件性参考，但肩宽随锁骨姿态变化，头到手的距离随动作变化，不能当作第二时刻的 C 实测值。')
    line()
    line('最明显的反例在 00270：左右肘在 cam3 相距 10.4 px、cam4 相距 12.9 px，三角化位置相距 0.9 cm；左右腕分别相距 2.9 / 3.3 px，三维相距 1.8 cm。它们高度疑似将同一条可见手臂当成左右两臂。右肘与右腕的对极残差为 5.7 / 1.8 px，右前臂三角长度 23.8 cm 与 C 的 24.0 cm 也接近。**因此，对极通过、深度合理、单根骨长合理均不能保证身份或位置正确。**')
    line()
    line('对 00000，双腕三维坐标与 DFM 相距 1.2 / 1.1 cm；双肘相距 8.4 / 10.0 cm。两肘各视角自动点均落在衣物投影上；左右图中 Sapiens 与 DFM 肘投影相隔 21.3 / 26.6 px（左）及 47.8 / 43.9 px（右）。两种位置的肘部 21×21 邻域均被 Sapiens 分割为 Upper_Clothing，不能区分袖子表面与被衣物遮挡的真实关节，也无法判定是 Sapiens 深度偏近、DFM 肘不准，还是两者都有偏差。')
    line()
    line('00000 的左髋、右髋、左膝和左踝三角距离为 35.0、45.9、52.8、44.8 cm，与 DFM 的三维差为 34.6、26.4、51.9、98.5 cm；对应点在图中落在上衣或袖子投影附近，且分割也预测为 Upper_Clothing。这支持“疑似不可见而不宜直接信任”，但仅有二维图像与射线解，不能证明三维位置确实落在托盘或衣物表面。')
    line()
    line('DFM 文件自己的说明只建议参考手和腕；本报告把身体其它关节作为对照，不当成真值。不同来源的人体关节点定义也可能存在差异。00270 的 DFM 右肘与右腕投影在画面外，且 DFM 身体点没有独立标注验证。')
    line()
    line('## 逐关节比较')
    line('所有长度单位 cm；距离均为 cam3 相机中心至点的欧氏距离，而非 z 深度。三维差是两组三维坐标的欧氏距离，范围差是“三角化距离−DFM距离”。鼻无 DFM 身体点映射。9 px 对极门槛只是上一轮基于每坐标 3 px 噪声的演示，未经误报率标定；小于门槛不代表正确。')
    for fr in frames:
        line();line('### 帧 %05d'%fr['frame']);line()
        line('|关节|对极 px|三角距离|DFM距离|距离差|三维差|三角点到 DFM 头|DFM 点到 DFM 头|C 点到 C 头|')
        line('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
        for r in fr['rows']:
            line('|%s|%s|%s|%s|%s|%s|%s|%s|%s|'%(
                CN[r['joint']],fmt(r['epi_sampson_px']),fmt(r['tri_range_cm']),fmt(r['dfm_range_cm']),
                fmt(r['tri_minus_dfm_range_cm']),fmt(r['tri_dfm_3d_error_cm']),fmt(r['tri_to_dfm_head_cm']),
                fmt(r['dfm_to_dfm_head_cm']),fmt(r['c_to_c_head_cm'])))
    line();line('头距离比较时三角点使用同帧 DFM 头作为参考；C 使用自身头关节。00000 两个头点相差 1.69 cm。DFM 头在画面外，仍是模型参考，不能当成独立的头部真值。')
    line();line('## 肩肘腕骨长')
    line('单位 cm；C 列始终是 00000 的 C 身体结果。00270 的 C 列只用于展示同人体型的参考，不代表该姿态已做 C 拟合。肩宽不是严格固定骨长。')
    line();line('|帧|线段|三角化|DFM|C（00000参考）|三角−C|端点对极 >9 px|')
    line('|---|---|---:|---:|---:|---:|:---:|')
    for x in bone:
        line('|%05d|%s|%s|%s|%s|%s|%s|'%(int(x['frame']),BN[x['bone']],fmt(x['tri_cm']),fmt(x['dfm_cm']),fmt(x['c_frame0_reference_cm']),fmt(x['tri_minus_c_cm']), '是' if x['one_or_both_endpoints_epi_gt9']=='True' else '否'))
    line();line('辅助检查：DFM 自身20时刻的左前臂长度约 22.4–25.9 cm、右前臂约 19.4–24.6 cm；它们是 DFM 输出间的变化，不构成人体真值。')
    line();line('## 肘部细查')
    line('[局部叠加图](elbow_evidence.png)：青点 Sapiens，橙十字 DFM 投影，紫叉是仅在00000提供的 C 投影；绿色线是现有分割预测的手臂类别边界。00270 未跑分割。')
    line()
    line('[00000 的分割叠加](elbow_segmentation_00000.png)：橙色为 Upper_Clothing，绿色为手臂类别；青点、白菱形、紫叉分别为 Sapiens、DFM、C。第一帧四个视角/肘位置上，Sapiens 点和 DFM 投影点的 21×21 像素邻域均为 Upper_Clothing。分割来自另一个自动模型，不能作为关节中心或深度真值。')
    line()
    line('00000 左肘三角/DFM 距离 38.3/46.5 cm；右肘 38.7/48.0 cm。三角范围对两图每坐标独立 σ=3 px 的线性化随机敏感度分别约 2.17/2.26 cm，不能解释成已测量的真实误差，也未包含标定、同步、检测系统偏差或 DFM 不确定性。两侧可见衣袖说明有图像证据，但关节中心被衣物覆盖；左肘两种投影都在袖附近，右肘 DFM 投影靠近躯干衣物边缘。现有证据不能确定哪一个三维肘点更准确。')
    line()
    line('00270 左肘三角/DFM 范围 38.4/38.7 cm，三维差 1.3 cm；但这只是与 DFM 接近，不是独立真值。00270 右肘对应两条近似重合的左右袖点，DFM 右肘投影在画外；应先标记疑似身份混淆/不可见，不能把 58.9 cm 的参考差归咎于 DFM。')
    line();line('## 方法和局限')
    line('用源文件 `sourceParams` 的 Double Sphere 模型反投影到单位射线，用 cam3 射线上的两射线最近解三角化；正深度之外不给有效距离。相对 R/t 是上一轮由 DFM 两路身体坐标刚体配准得到的 5.566 cm 基线，20帧内部残差近机器精度。这说明坐标数据互相一致，**不能独立证明相机标定或 DFM 关节准确**。')
    line()
    line('DFM NPZ 仅含 22 点数组，未保存名称；报告按 SMPL 常见22点顺序映射肩肘腕/髋膝踝和头，与上一轮腕肘对照一致，但要向数据生成源核实关节定义。Sapiens、DFM、SMPL-X 对关节中心的定义和衣物遮挡可能不完全一致。')
    line()
    line('距离敏感度是像素独立零均值高斯 σ=3 px 的条件性线性化，不包括点身份混淆、系统误差、遮挡、同步与标定误差；因此不能把“落在几倍 σ 内”用作准确率保证。')
    line()
    line('可以将对极残差、图像内可见性、跨侧重复、射线夹角、骨长范围、二维图像证据和参考差异一起作为**可疑点提示**；这些样本尚不足以设定通用阈值或证明可可靠分类。进一步区分肘点成因需要独立标注/其他视角/可靠深度或经验证的三维真值。')
    line();line('## 文件')
    line('- [逐关节明细 CSV](joints.csv)：包含二维坐标误差、三维距离与随机敏感度。')
    line('- [骨长 CSV](bones.csv)；[完整测量 JSON](measurements.json)，包括每个点的坐标和分割邻域。')
    line('- [00000 全点叠加](all_joints_00000.png)；[00270 全点叠加](all_joints_00270.png)；[肘部局部图](elbow_evidence.png)；[00000 肘部分割图](elbow_segmentation_00000.png)。')
    line('- [验证结果](validation.json)、[分析脚本](analyze.py)、[报告脚本](make_report.py)、[原始三角诊断](../gen_egosmplx_two_frames_20261030/stereo_diagnostic_20261031_rkk9qawi/report.md)。')
    with (OUT/'report.md').open('x') as f:f.write('\n'.join(s)+'\n')


if __name__=='__main__':
    verify();draw_segmentation();report()
    print('Wrote validation.json, elbow_segmentation_00000.png and report.md')
