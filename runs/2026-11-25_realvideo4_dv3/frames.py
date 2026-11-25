import csv
import subprocess
from pathlib import Path
import cv2
import numpy as np
from fileset import ROOT,RUN,OUT,SESSION,SIDES,sha,write

def read_tool_binary():
    from fileset import read
    return read(ROOT/'outputs/initialization_ablation_20261109/model_paths.json')['body_python']

def timestamps(path):
    with path.open(encoding='utf-8-sig') as f:
        records=list(csv.DictReader(line for line in f if not line.startswith('#')))
    t=np.array([int(x['system_relative_time_ticks']) for x in records])
    assert np.all(np.diff(t)>0) and all(int(x['recording_frame_index'])==i+1 for i,x in enumerate(records))
    return t

def main():
    (OUT/'images').mkdir(parents=True,exist_ok=True)
    captures=[]
    for k in range(1,5):
        key=f'clip{k:03d}'
        sources=[next((SESSION/'videos'/(s+'_lower')).glob(f'*_{k:03d}.mp4')) for s in SIDES]
        csvs=[p.with_name(p.stem+'_时间戳.csv') for p in sources]
        clocks=list(map(timestamps,csvs))
        start=max(c[0] for c in clocks)
        stop=min(c[-1] for c in clocks)
        assert stop>start
        middle=(start+stop)/2
        li=int(np.argmin(abs(clocks[0]-middle)))
        ri=int(np.argmin(abs(clocks[1]-clocks[0][li])))
        record=dict(pair=key,delta_ms=float(abs(clocks[1][ri]-clocks[0][li])/10000),overlap_duration_s=float((stop-start)/1e7),views={})
        for p,c,t,ix,s in zip(sources,csvs,clocks,(li,ri),SIDES):
            tools=Path(read_tool_binary()).parent
            raw=subprocess.check_output([str(tools/'ffprobe'),'-v','error','-select_streams','v:0','-show_entries','packet=pts_time','-of','csv=p=0',str(p)])
            pts=np.sort(np.fromstring(raw.decode(),sep='\n'))
            assert len(pts)>=len(t)
            window=slice(ix-100,ix+101)
            residual=(pts[window]-pts[ix])-(t[window]-t[ix])/1e7
            assert np.max(abs(residual))<.0001
            dest=OUT/'images'/(key+'_'+s+'.png')
            subprocess.run([str(tools/'ffmpeg'),'-v','error','-y','-threads','2','-i',str(p),'-vf',f'select=eq(n\\,{ix})','-vsync','0','-frames:v','1',str(dest)],check=True)
            assert cv2.imread(str(dest)).shape==(720,1280,3)
            record['views'][s]=dict(frame_one_based=ix+1,csv_frame_count=len(t),container_packets=len(pts),pts_s=float(pts[ix]),max_timing_discrepancy_ms=float(np.max(abs(residual))*1000),seconds_from_stream_start=float((t[ix]-t[0])/1e7),
                                     ticks=int(t[ix]),clip=sha(p),csv=sha(c),frame=sha(dest))
        captures.append(record)
    write(RUN/'evidence/frames.json',captures)

if __name__=='__main__':
    main()
