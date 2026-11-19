"""Local side views for the four paired poses; green = RW neck, red = model neck."""
from pathlib import Path
import cv2
import numpy as np

root = Path(__file__).resolve().parents[2]
out = root/'outputs/rw_head_neck_oracle_20261119'
old = root/'outputs/unrealego_gt2d_ideal_20261115'
raw = np.load(out/'inspection_local.npz')
meshes, necks = [], []
for k in range(4):
    with np.load(old/('body%02d' % (k+1))/'D_left/final.npz') as z:
        meshes.append(z['smplx_mesh_cam'])
        necks.append(z['smplx_joint_cam'][7])
meshes.extend(raw['mesh'])
necks.extend(raw['neck_head'][:,0])
target = raw['gt_neck_head'][:,0]
faces = raw['faces']
# One orthographic camera direction for all panels, chosen using the first old pose.
with np.load(old/'body01/D_left/final.npz') as z:
    axis = cv2.Rodrigues(z['global_orient'].reshape(3))[0]
basis = axis @ cv2.Rodrigues(np.array([0., np.pi/2, 0.]))[0]
meshes = np.array(meshes) @ basis
width, height = 244, 355
canvas = np.full((2*height+83, 4*width, 3), 247, np.uint8)
clouds = [np.vstack([meshes[k], meshes[k+4]]) for k in range(4)]
scale = min(min((width-16)/np.ptp(c[:,0]), (height-20)/np.ptp(c[:,1])) for c in clouds)
for k, cloud in enumerate(clouds):
    center = (cloud.max(0)+cloud.min(0))/2
    cv2.putText(canvas, 'body%02d' % (k+1), (k*width+79,23),
                cv2.FONT_HERSHEY_SIMPLEX,.63,(35,35,35),1,cv2.LINE_AA)
    for row in range(2):
        i = row*4+k
        v = meshes[i]-center
        pix = lambda q: np.rint(q[...,:2]*[scale,-scale]+[width/2,height/2]).astype(np.int32)
        xy = pix(v)
        tri = v[faces]
        normal = np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        normal /= np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-12)
        shade = (129+96*np.maximum(0,normal@[-.35,.55,.75])).astype(int)
        tile = np.full((height,width,3),247,np.uint8)
        for fi in np.argsort(tri[:,:,2].mean(1)):
            c = int(shade[fi])
            cv2.fillConvexPoly(tile,xy[faces[fi]],(c,c,c))
        cv2.circle(tile,pix(target[k]@basis-center),4,(60,153,25),-1,cv2.LINE_AA)
        cv2.drawMarker(tile,pix(necks[i]@basis-center),(48,51,194),cv2.MARKER_TILTED_CROSS,10,2)
        top = 32+row*height
        canvas[top:top+height,k*width:(k+1)*width] = tile
cv2.putText(canvas,'Top: prior GT2D triangulated fit.  Bottom: current full-2D fit.  Dot: RW neck.  Cross: fitted neck.',
            (20,2*height+60),cv2.FONT_HERSHEY_SIMPLEX,.57,(30,30,30),1,cv2.LINE_AA)
assert cv2.imwrite(str(out/'neck_comparison.jpg'),canvas)
