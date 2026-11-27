"""Compare old and rectified native WiLoR landmarks in identical raw camera windows."""
from wrist_views import RUN, ROOT, NATIVE, SIDES, TIPS, old, read, write, sha, stereo, title
import cv2
import numpy as np

def draw(photo, uv, color):
    im = photo.copy()
    points = np.rint(uv).astype(int)
    for start in (1,5,9,13,17):
        ids = [0]+list(range(start,start+4))
        cv2.polylines(im,[points[ids].astype('int32')],False,color,1,cv2.LINE_AA)
        for idx in ids:
            cv2.circle(im,tuple(points[idx]),2,color,-1,cv2.LINE_AA)
    return im

def main():
    stereo.RUN = RUN
    bundle = read(RUN/'evidence/results.json')
    native = read(NATIVE/'evidence/results.json')['cases']
    newer = read(RUN.parent/'2026-11-27_vidihand2_comparison/evidence/metrics.json')['cases']
    stats = {}
    pools = {'old':[],'new':[]}
    for key,data in newer.items():
        base = read(old.RUN/'evidence'/(key+'.json'))['results']['B']['hands']
        panels = []
        for i,view in enumerate(SIDES):
            eye = f'cam{3+i}'
            path = old.ref.OLD/'images'/f'{key}_{view}.png'
            photo = cv2.imread(str(path))
            for side in SIDES:
                letter = side[0].upper()
                tag = f'{key}_{view}_{letter}'
                assert sha(path)==native[tag]['photo_sha256']
                pack = old.NATIVE/(tag+'.npz')
                assert sha(pack)==native[tag]['input_pack_sha256']
                uv = np.load(pack)['uv']
                cand = data['wilor'][side]['candidates'][eye]
                uv_new = np.array(cand['native_uv'])
                obs = np.array(base[side]['observation'][i])
                mask = np.array(base[side]['inside'][i])[TIPS]
                x,y,w,h = native[tag]['crop_xywh']
                tiles = []
                for label,im in [('Photo',photo),('Old native',draw(photo,uv,(25,175,245))),
                                 ('New native',draw(photo,uv_new,(25,175,245)))]:
                    tiles.append(title(cv2.resize(im[y:y+h,x:x+w],(256,256)),f'{eye} {letter} | {label}'))
                panels.append(np.hstack(tiles))
                stats[tag] = {}
                for name,pts in [('old',uv),('new',uv_new)]:
                    err = np.linalg.norm(pts-obs,axis=1)[TIPS][mask]
                    if name=='new':
                        assert np.allclose(err,cand['native_tips'],atol=.001)
                    pools[name].extend(err.tolist())
                    stats[tag][name] = dict(tips_px=err.tolist(),wrist_px=float(np.linalg.norm(pts[0]-obs[0])))
        stereo.save(bundle['images'],key+'_inputs',np.vstack(panels))
    stats['pooled'] = {k:dict(count=len(v),rmse_px=old.rms(v)) for k,v in pools.items()}
    bundle['independent_native'] = stats
    write(RUN/'evidence/results.json',bundle)

if __name__=='__main__':
    main()
