"""Compare before and after on an identical pixel window in each eye."""
import cv2
import numpy as np
from pathlib import Path
import json
RUN = Path(__file__).resolve().parent
OLD = RUN.parent/'2026-11-26_realvideo34_purple_hand_a'
gallery = json.loads((RUN/'evidence/gallery.json').read_text())['rows']
for key,row in gallery.items():
    tiles=[]
    for cam,view in row['views'].items():
        for side,hand in view['hands'].items():
            pts = np.array([point['det'] for point in hand])
            # Body/hand combined render has a 32-pixel title band.
            center = np.rint((pts.max(0)+pts.min(0))/2).astype(int)
            size = int(max(160,np.ptp(pts,axis=0).max()+125))
            x,y = np.clip(center-size//2,[0,0],[1280-size,720-size])
            pair=[]
            for folder,label in [(OLD,'Purple A'),(RUN,'Arm + A')]:
                image=cv2.imread(str(folder/'assets'/(key+'_'+cam+'_combined.jpg')))
                crop=cv2.resize(image[32+y:32+y+size,x:x+size],(250,250))
                strip=np.zeros((26,250,3),np.uint8)
                cv2.putText(strip,cam+' '+side+' | '+label,(5,18),0,.49,(255,255,255),1,cv2.LINE_AA)
                pair.append(np.vstack([strip,crop]))
            tiles.append(np.hstack(pair))
    dest=RUN/'assets'/(key+'_comparison.jpg')
    assert cv2.imwrite(str(dest),np.vstack(tiles))
