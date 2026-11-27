from base import *
import numpy as np
import cv2
mode=sys.argv[1]
if mode=='sample':
    from frames import timestamps
    rows=[]
    for clip in (3,4):
        paths=[next((SESSION/'videos'/(s+'_lower')).glob(f'*_{clip:03}.mp4')) for s in SIDES]
        clocks=[timestamps(p.with_name(p.stem+'_时间戳.csv')) for p in paths]
        videos=[cv2.VideoCapture(str(p)) for p in paths]
        a=max(c[0] for c in clocks); b=min(c[-1] for c in clocks)
        for n in range(10):
            key=f'v{clip:03}_{n+1:02}'
            goal=a+(b-a)*(n+.5)/10
            left=int(np.argmin(abs(clocks[0]-goal)))
            right=int(np.argmin(abs(clocks[1]-clocks[0][left])))
            row=dict(pair=key,clip=clip,fraction=(n+.5)/10,relative_s=(int(clocks[0][left])-a)/1e7,delta_ms=abs(int(clocks[1][right]-clocks[0][left]))/1e4,views={})
            for side,cap,i in zip(SIDES,videos,[left,right]):
                cap.set(cv2.CAP_PROP_POS_FRAMES,i)
                ok,im=cap.read(); assert ok and cap.get(cv2.CAP_PROP_POS_FRAMES)==i+1
                p=OUT/'images'/(key+'_'+side+'.png')
                assert cv2.imwrite(str(p),im)
                row['views'][side]=dict(frame=sha(p),index_zero_based=i)
            rows.append(row)
        for cap in videos:cap.release()
    write(RUN/'evidence/frames.json',rows)
elif mode=='detect':
    import detect
    detect.OUT,detect.RUN=OUT,RUN
    detect.main()
elif mode=='segment':
    from segment_batch import main
    main()
elif mode in ('prepare','fit','candidates','v4'):
    import torch,solve
    torch.set_num_threads(1);cv2.setNumThreads(1)
    v=solve.v3
    solve.RUN,solve.OUT=RUN,OUT
    v.RUN,v.OUT,v.FIT=RUN,OUT,OUT/'fit'
    v.ref.camera_files=cameras
    if mode=='prepare':solve.prepare()
    elif mode in ('fit','candidates'):
        sys.addaudithook(v.base.fit.forbid_gt)
        getattr(v,mode)(0)
        if mode=='candidates':write(RUN/'evidence/results.json',dict(selected_hash=sha(OUT/'fit/0/selected.npz')))
    else:
        from body4 import refine
        refine(int(sys.argv[2]))
else:
    import importlib
    importlib.import_module(mode).main(*sys.argv[2:])
