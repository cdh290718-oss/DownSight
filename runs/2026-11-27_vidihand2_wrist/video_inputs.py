"""Make a calibrated perspective clip per original eye, centred at the archived stereo frames."""
from pathlib import Path
import subprocess
import sys
import cv2
import numpy as np
RUN=Path(__file__).resolve().parent
sys.path.insert(0,str(RUN.parent/'2026-11-25_realvideo4_dv3'))
from fileset import ROOT,OUT as OLD,SESSION,SIDES,cameras,read,write,sha
from frames import timestamps
from calibrate import projection,rays

def main():
    cv2.setNumThreads(1)
    out=ROOT/'outputs/vidihand2_wrist_20261127/video'
    out.mkdir(parents=True,exist_ok=True)
    ffmpeg=Path(read(ROOT/'outputs/initialization_ablation_20261109/model_paths.json')['body_python']).with_name('ffmpeg')
    meta=read(RUN.parent/'2026-11-25_realvideo4_dv3/evidence/frames.json')
    det=read(OLD/'predictions.json')
    size=(640,480)
    focal=size[0]/(2*np.tan(np.deg2rad(125/2)))
    K=np.array([[focal,0,(size[0]-1)/2],[0,focal,(size[1]-1)/2],[0,0,1.]])
    y,x=np.indices(size[::-1])
    unit=np.stack([x,y,np.ones_like(x)],axis=-1).reshape(-1,3)@np.linalg.inv(K).T
    info={}
    for eye,calib in zip(SIDES,cameras()):
        cam=read(calib)
        uv=projection(unit,cam)
        remap=uv.reshape(size[::-1]+(2,)).astype('f')
        for record in meta[2:4]:
            key=record['pair']
            ref=record['views'][eye]
            src=next((SESSION/'videos'/(eye+'_lower')).glob('*_'+key[-3:]+'.mp4'))
            csv=src.with_name(src.stem+'_时间戳.csv')
            assert sha(src)==ref['clip'] and sha(csv)==ref['csv']
            center=ref['frame_one_based']-1
            indices=center+np.arange(-40,41)*2
            lo,hi=indices[[0,-1]]
            assert lo>=0 and hi<ref['csv_frame_count']
            filter_=f"select=between(n\\,{lo}\\,{hi})*not(mod(n-{lo}\\,2))"
            cmd=[str(ffmpeg),'-v','error','-threads','2','-i',str(src),'-vf',filter_,'-vsync','0','-frames:v','81','-threads','1','-f','rawvideo','-pix_fmt','bgr24','pipe:1']
            raw=subprocess.check_output(cmd)
            frames=np.frombuffer(raw,np.uint8).reshape(len(indices),720,1280,3)
            orig=OLD/'images'/(key+'_'+eye+'.png')
            assert sha(orig)==ref['frame']
            center_diff=int(np.abs(frames[40].astype('i2')-cv2.imread(str(orig))).max())
            assert center_diff<=2,center_diff
            dest=out/(key+'_'+eye+'.npy')
            rgb=np.lib.format.open_memmap(dest,mode='w+',dtype='uint8',shape=(len(indices),size[1],size[0],3))
            for n,frame in enumerate(frames):
                rgb[n]=cv2.remap(frame,remap,None,cv2.INTER_LINEAR)[:,:,::-1]
            del rgb
            ts=timestamps(csv)[indices]/1e7
            name=key+'_'+eye
            target=np.asarray(det[name+'.png']['xy'])[21:63]
            good=((target>=0)&(target<[1280,720])).all(-1)
            rv=rays(target,cam)@K.T
            pp=rv[:,:2]/rv[:,2:]
            visible=((pp>=0)&(pp<size)).all(-1)&(rv[:,2]>0)
            info[name]=dict(frames=len(indices),center_index=40,source_one_based=int(center+1),source_sha256=ref['clip'],
                            center_pixel_max_diff=center_diff,fps=float((len(indices)-1)/(ts[-1]-ts[0])),K=K.tolist(),size=list(size),
                            detector_inframe_native=int(good.sum()),also_in_virtual=int((good&visible).sum()),file_sha256=sha(dest))
    write(RUN/'evidence/video.json',info)
if __name__=='__main__':
    main()
