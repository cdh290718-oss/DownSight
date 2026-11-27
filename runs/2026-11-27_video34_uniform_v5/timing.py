"""Verify sampled video presentation indices against recording clocks."""
from base import *
from frames import timestamps
import numpy as np,subprocess
tool=Path(read(ROOT/'outputs/initialization_ablation_20261109/model_paths.json')['body_python']).parent/'ffprobe'
rows=read(RUN/'evidence/frames.json')
info={}
for clip in (3,4):
    for side in SIDES:
        p=next((SESSION/'videos'/(side+'_lower')).glob(f'*_{clip:03}.mp4'))
        csv=p.with_name(p.stem+'_时间戳.csv')
        t=timestamps(csv)
        raw=subprocess.check_output([str(tool),'-v','error','-select_streams','v:0','-show_entries','packet=pts_time','-of','csv=p=0',str(p)])
        pts=np.sort(np.fromstring(raw.decode(),sep='\n'))
        ids=np.array([r['views'][side]['index_zero_based'] for r in rows if r['clip']==clip])
        assert len(ids)==len(set(ids))==10 and np.all(np.diff(ids)>0)
        diff=(pts[ids]-pts[ids[0]])-(t[ids]-t[ids[0]])/1e7
        assert abs(diff).max()<.001
        info[f'{clip}_{side}']=dict(video_sha256=sha(p),csv_sha256=sha(csv),presentation_drift_ms=(diff*1000).tolist(),frames=len(t))
write(RUN/'evidence/timing.json',dict(streams=info,exit_status=0))
