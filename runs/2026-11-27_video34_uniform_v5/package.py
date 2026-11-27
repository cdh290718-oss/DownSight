from base import *
from zipfile import ZipFile,ZIP_DEFLATED
import cv2,numpy as np

def main():
    frames=read(RUN/'evidence/frames.json')
    pool={}; members={};sheets={}; fallback_hands=[]; per_pair={}
    for clip in [3,4]:
        table=[];thumb=[];body=[];tips=[];sources={};fallback=0
        for f in [f for f in frames if f['clip']==clip]:
            k=f['pair']
            result=read(RUN/'evidence'/(k+'_render.json'))
            assert result['exit_status']==0
            for name,digest in result['hashes'].items():assert sha(RUN/'assets'/name)==digest
            members[result['obj']]=result['obj_sha256']
            for side in SIDES:
                source=result['hands'][side]['source']
                sources[source]=sources.get(source,0)+1
                if source not in ['cam3','cam4']:fallback_hands.append(dict(pair=k,side=side,reason=source))
            fallback+=bool(result['source']['palm_fallback'])
            body+=result['errors']['body'];tips+=result['errors']['tips']
            frame_tips=result['errors']['tips']
            tip_rmse=float(np.sqrt(np.mean(np.square(frame_tips)))) if frame_tips else None
            per_pair[k]=dict(tips_px=tip_rmse,tips_n=len(frame_tips))
            tip_label='—' if tip_rmse is None else f'{tip_rmse:.1f}'
            notes=[('左手' if s=='left' else '右手')+'回退' for s,h in result['hands'].items() if h['source'] not in ['cam3','cam4']]
            status='、'.join(notes) or '双手选解'
            def link(mode,text):return f'[{text}](assets/{k}_{mode}.jpg)'
            table.append('| '+' | '.join([k,f"{f['relative_s']:.2f}",link('mesh','网格'),link('points','点位'),link('both','叠加'),link('obj','正侧面'),link('hands','原图放大'),link('handobj','双角度'),tip_label,status])+' |')
            img=cv2.imread(str(RUN/'assets'/(k+'_overview.jpg')))
            thumb.append(cv2.resize(img,None,fx=.65,fy=.65,interpolation=cv2.INTER_AREA))
        canvas=np.vstack([np.hstack(thumb[i:i+2]) for i in range(0,10,2)])
        name=f'video{clip:03}_all.jpg'
        assert cv2.imwrite(str(RUN/'assets'/name),canvas,[cv2.IMWRITE_JPEG_QUALITY,92])
        sheets[name]=sha(RUN/'assets'/name)
        pool[str(clip)]=dict(samples=10,sources=sources,palm_fallback_frames=fallback,body_n=len(body),tips_n=len(tips),
            body_px=float(np.sqrt(np.mean(np.square(body)))),tips_px=float(np.sqrt(np.mean(np.square(tips)))))
        (RUN/f'video{clip:03}.md').write_text(f'# Video {clip:03} · Dv5 逐帧图集\n\n按拍摄时间排列，每行两对。总览中同一对从左至右为 cam3、cam4、OBJ 正面、侧面。青叉为自动检测，橙圈为 SMPL-X 投影。按表中链接打开原尺寸图片。\n\n[![十对总览](assets/{name})](assets/{name})\n\n| 样本 | 秒 | 双目 mesh | 自动与投影点 | 网格加点 | 全身 OBJ | 双目双手 | 手部 OBJ | 指尖 RMSE（px） | 手部状态 |\n|---|---:|---|---|---|---|---|---|---:|---|\n'+'\n'.join(table)+'\n')
    bundle=OUT/'twenty_Dv5_cam3_m_OBJ.zip'
    with ZipFile(bundle,'w',ZIP_DEFLATED) as z:
        for n,h in members.items():
            p=OUT/'export'/n;assert sha(p)==h;z.write(p,n)
    with ZipFile(bundle) as z:
        assert len(z.namelist())==20 and z.testzip() is None
        assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in members.items())
    write(RUN/'evidence/summary.json',dict(segments=pool,frames=per_pair,fallback_hands=fallback_hands,overview_sha256=sheets,bundle_sha256=sha(bundle),members=members,
        maximum_stereo_delta_ms=max(f['delta_ms'] for f in frames),exit_status=0))
if __name__=='__main__':main()
