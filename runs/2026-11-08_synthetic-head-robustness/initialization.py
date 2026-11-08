"""Observation-only shape observability and weighted rigid initialization."""
import numpy as np
import cv2
import torch

def initialize(layer,model,raw,points,threshold):
 points=[p for p in points if p['accepted'] and p['source']=='sapiens'];lookup={p['index']:p for p in points}
 # Rest joints are affine in beta; metric lengths give pose-independent shape sensitivity.
 J0=(layer.J_regressor@layer.v_template).detach().numpy();S=torch.einsum('jv,vck->jck',layer.J_regressor,layer.shapedirs[:,:,:10]).detach().numpy()
 pairs=[(8,9),(1,2),(8,10),(9,11),(10,12),(11,13),(1,3),(2,4),(3,5),(4,6)]
 mapping=list(model.joint_idx);rows=[];information=np.zeros((10,10))
 for a,b in pairs:
  if a not in lookup or b not in lookup:continue
  pa,pb=lookup[a],lookup[b];difference=np.array(pa['xyz'])-np.array(pb['xyz']);unit=difference/np.linalg.norm(difference);variance=float(unit@(np.array(pa['covariance'])+np.array(pb['covariance']))@unit)
  nativea,nativeb=mapping[a],mapping[b];d=J0[nativea]-J0[nativeb];derivative=d/np.linalg.norm(d)@(S[nativea]-S[nativeb]);information+=np.outer(derivative,derivative)/variance
  rows.append(dict(indices=[a,b],observed_length_m=float(np.linalg.norm(difference)),length_sigma_m=float(np.sqrt(variance)),derivative_m=derivative.tolist()))
 eig,V=np.linalg.eigh(information);keep=eig>=threshold;basis=V[:,keep];beta=basis@(basis.T@raw['betas'].reshape(-1));fixed={k:torch.tensor(raw[k],dtype=torch.float32).reshape(1,-1) for k in ['body_pose','global_orient','left_hand_pose','right_hand_pose','jaw_pose','leye_pose','reye_pose','expression']}
 with torch.no_grad():out=layer(betas=torch.tensor(beta,dtype=torch.float32).reshape(1,-1),**fixed)
 native=out.joints[0].numpy();src=native[[mapping[p['index']] for p in points]];dst=np.array([p['xyz'] for p in points]);w=np.array([1/np.trace(p['covariance']) for p in points]);w/=w.sum();x=(src*w[:,None]).sum(0);y=(dst*w[:,None]).sum(0);U,sv,Vt=np.linalg.svd((src-x).T@(w[:,None]*(dst-y)));R=Vt.T@np.diag([1,1,np.linalg.det(Vt.T@U.T)])@U.T;t=y-R@x
 # SMPL-X rotates around the rest pelvis. Convert rigid alignment to its transl convention.
 global0=cv2.Rodrigues(R@cv2.Rodrigues(raw['global_orient'].reshape(3).astype(float))[0])[0].reshape(3);pivot=native[0];translation=R@pivot+t-pivot
 return beta,basis,global0,translation,dict(shape_information_eigenvalues=eig.tolist(),retained_rank=int(keep.sum()),basis=basis.tolist(),bone_information=rows,alignment_rms_m=float(np.sqrt((w*np.sum((src@R.T+t-dst)**2,axis=1)).sum())),alignment_singular_values=sv.tolist(),beta_initial=beta.tolist(),global_initial=global0.tolist(),translation_initial=translation.tolist(),uses_gt=False)
