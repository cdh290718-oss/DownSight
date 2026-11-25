"""Convert the new MATLAB model and check it against independently detected board corners."""
import numpy as np
import cv2
from numpy.polynomial import Polynomial
from fileset import CAL,SESSION,OUT,RUN,SIDES,cameras,read,write,sha

def convert(source, transpose=False):
    a0,a2,a3,a4=source['mapping_coefficients_a0_a2_a3_a4']
    poly=np.array([a0,0.,a2,a3,a4])
    S=np.asarray(source['stretch_matrix'])
    if transpose:
        S=S.T
    center=np.asarray(source['distortion_center_xy'])-1.0
    radius=np.linspace(0,1150,10001)
    z=np.polynomial.polynomial.polyval(radius,poly)
    theta=np.arctan2(-z,radius)
    assert np.all(np.diff(theta)>0)
    inv=Polynomial.fit(theta,radius,14).convert()
    return dict(size=[1280,720],image_center=center.tolist(),affine=[float(S[0,0]),float(S[0,1]),float(S[1,0])],
                polynomialC2W=(-poly).tolist(),polynomialW2C=inv.coef.tolist(),input_adapter=dict(scale=1.,offset=[0.,0.]))

def rays(p,cam):
    c,d,e=cam['affine']
    v=(p-np.array(cam['image_center']))@np.linalg.inv([[c,d],[e,1.]]).T
    z=-np.polynomial.polynomial.polyval(np.linalg.norm(v,axis=1),cam['polynomialC2W'])
    return np.column_stack((v,z))

def projection(q,cam):
    rho=np.linalg.norm(q[:,:2],axis=1)
    th=np.arctan2(-q[:,2],rho)
    r=np.polynomial.polynomial.polyval(th,cam['polynomialW2C'])
    v=q[:,:2]*(r/np.maximum(rho,1e-12))[:,None]
    c,d,e=cam['affine']
    return v@np.array([[c,d],[e,1]]).T+cam['image_center']

def board_error(corners,cams,T):
    # Distance in native pixels to the local epipolar curve, evaluated on the right image.
    left,right=corners
    ra=rays(left,cams[0])@T[:3,:3].T
    normals=np.cross(T[:3,3],ra)
    def residual(q):
        return np.sum(normals*rays(q,cams[1]),axis=1)
    ds=[(residual(right+dg)-residual(right-dg))/.02 for dg in [[.01,0.],[0.,.01]]]
    return float(np.mean(abs(residual(right))/np.maximum(np.linalg.norm(ds,axis=0),1e-12)))

def main():
    source=read(CAL/'fisheye_intrinsics.json')
    ext=read(CAL/'stereo_extrinsics.json')
    T=np.array(ext['T_left_to_right_column_vectors'])
    T[:3,3]/=1000
    assert np.allclose(T[:3,:3].T,ext['R_left_to_right'])
    cameras_by_convention=[[convert(source[s+'_lower'],flip) for s in SIDES] for flip in [False,True]]
    check={}
    for idx in [5,15,25,30,51,60,68,69,71,73,74,75,76]:
        corners=[]
        for side in SIDES:
            path=SESSION/(side+'_lower')/(side+f'_lower_{idx:03d}.png')
            ok,xy=cv2.findChessboardCornersSB(cv2.imread(str(path),cv2.IMREAD_GRAYSCALE),(11,8))
            if not ok:
                break
            corners.append(xy.reshape(-1,2).astype(float))
        if len(corners)==2:
            check[str(idx)]=[min(board_error((corners[0],rhs),cams,T) for rhs in (corners[1],corners[1][::-1])) for cams in cameras_by_convention]
    assert len(check)>=2,check
    means=np.mean(list(check.values()),axis=0)
    which=int(np.argmin(means))
    assert means[which]<1 and means[1-which]>2*means[which],(means,check)
    cams=cameras_by_convention[which]
    grid=np.stack(np.meshgrid(np.linspace(0,1279,100),np.linspace(0,719,70)),-1).reshape(-1,2)
    errors=[]
    for path,cam in zip(cameras(),cams):
        err=np.linalg.norm(projection(rays(grid,cam),cam)-grid,axis=1)
        assert err.max()<.05,err.max()
        errors.append(float(err.max()))
        write(path,cam)
    write(RUN/'evidence/calibration.json',dict(intrinsics=sha(CAL/'fisheye_intrinsics.json'),extrinsics=sha(CAL/'stereo_extrinsics.json'),
        converted=[sha(p) for p in cameras()],board_right_epipolar_mean_px_direct_transposed=check,stretch_transposed=bool(which),
        matlab_to_native_pixel_shift=[-1,-1],native_grid_max_roundtrip_px=errors,T=T.tolist(),checker_square_assumed_mm=15,exit_status=0))
if __name__=='__main__':
    main()
