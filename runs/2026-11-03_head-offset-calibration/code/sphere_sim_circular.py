"""Optimistic neck-position oracle: test sphere-center geometry without annotations."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--trials',type=int,default=20);a=p.parse_args()
rng=np.random.default_rng(20261103);length=.10;noise=.01;n=20;htrue=np.array([.01,.14,-.03]);base=np.array([0.,length,0.]);directions=np.vstack([np.zeros(3),np.eye(3),-np.eye(3)])*.15
conditions=[]
for axes in (1,2):
 for degrees in (0,10,20,40,60):
  phase=np.linspace(0,2*np.pi,n,endpoint=False);v=np.deg2rad(degrees);rv=np.zeros((n,3));rv[:,0]=v*np.sin(phase)
  if axes==2:rv[:,2]=v*np.cos(phase)
  truth=htrue+Rotation.from_rotvec(rv).apply(np.tile(base,(n,1)))
  centered=truth-truth.mean(axis=0);s=np.linalg.svd(centered,compute_uv=False)
  for known in (True,False):
   trials=[]
   for rep in range(a.trials):
    measured=truth+rng.normal(0,noise,size=(n,3));pred=[];loss=[];status=[]
    for shift in directions:
     x0=np.r_[htrue+shift] if known else np.r_[htrue+shift,length]
     def fun(x):return (np.linalg.norm(measured-x[:3],axis=1)-(length if known else x[3]))/noise
     lower=np.r_[htrue-.5] if known else np.r_[htrue-.5,.05];upper=np.r_[htrue+.5] if known else np.r_[htrue+.5,.3]
     fit=least_squares(fun,x0,bounds=(lower,upper),max_nfev=250,ftol=1e-9,xtol=1e-9,gtol=1e-9);pred.append(fit.x[:3]);loss.append(float(np.mean(fun(fit.x)**2)));status.append(bool(fit.success))
    pred=np.array(pred);err=np.linalg.norm(pred-htrue,axis=1)*100;spread=max(np.linalg.norm(x-y) for x in pred for y in pred)*100
    trials.append({'trial':rep,'positions_m':pred.tolist(),'error_cm':err.tolist(),'max_pairwise_cm':float(spread),'loss':loss,'converged':status})
   spreads=[t['max_pairwise_cm'] for t in trials];errors=[max(t['error_cm']) for t in trials]
   conditions.append({'axes':axes,'amplitude_deg':degrees,'radius_mode':'known' if known else 'estimated','noiseless_neck_spread_singular_values_m':s.tolist(),'spread_cm_median':float(np.median(spreads)),'spread_cm_p90':float(np.percentile(spreads,90)),'worst_initialization_error_cm_median':float(np.median(errors)),'trials':trials})
out={'setup':{'frames':n,'trials':a.trials,'neck_head_distance_m':length,'per_axis_neck_position_noise_std_m':noise,'noise_model':'independent Gaussian 3D neck-coordinate error; optimistic neck-position oracle','starts_cm':(directions*100).tolist(),'bounds_m':'center within 0.5 m of synthetic truth; unknown radius constrained to [0.05,0.30] m','motion':'one axis x; two axes x and z; rotations vary sinusoidally with maximum rotation-vector component indicated','seed':20261103,'caveat':'Direct 3D neck coordinates and known radius in the known case are oracle observations, not supplied by an image-only device; shape, soft tissue and correspondence errors omitted.'},'conditions':conditions}
with a.output.open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
