"""Shared observation-only shape basis and Head-aligned root; independent of network root."""
import numpy as np
import torch
import cv2
from common import head_rotation_numpy

def shape_basis(layer,model,points,threshold):
 lookup={p['index']:p for p in points if p['accepted'] and p['source']=='sapiens'}
 J0=(layer.J_regressor@layer.v_template).detach().numpy();S=torch.einsum('jv,vck->jck',layer.J_regressor,layer.shapedirs[:,:,:10]).detach().numpy();mapping=list(model.joint_idx);information=np.zeros((10,10));rows=[]
 for a,b in [(8,9),(1,2),(8,10),(9,11),(10,12),(11,13),(1,3),(2,4),(3,5),(4,6)]:
  if a not in lookup or b not in lookup:continue
  pa,pb=lookup[a],lookup[b];difference=np.array(pa['xyz'])-np.array(pb['xyz']);unit=difference/np.linalg.norm(difference);variance=float(unit@(np.array(pa['covariance'])+np.array(pb['covariance']))@unit)
  d=J0[mapping[a]]-J0[mapping[b]];derivative=d/np.linalg.norm(d)@(S[mapping[a]]-S[mapping[b]]);information+=np.outer(derivative,derivative)/variance
  rows.append(dict(indices=[a,b],observed_length_m=float(np.linalg.norm(difference)),length_sigma_m=float(np.sqrt(variance)),derivative_m=derivative.tolist()))
 eig,V=np.linalg.eigh(information);basis=V[:,eig>=threshold]
 return basis,dict(shape_information_eigenvalues=eig.tolist(),basis=basis.tolist(),retained_rank=basis.shape[1],bone_information=rows)

def root_from_head(layer,body,beta,fixed,anchor):
 p=dict(global_orient=np.zeros(3),body_pose=np.array(body));relative=head_rotation_numpy(p)
 rotation=np.array(anchor['rotation'])@relative.T;g=cv2.Rodrigues(rotation)[0].reshape(3)
 with torch.no_grad():native=layer(body_pose=torch.tensor(body,dtype=torch.float32).reshape(1,-1),global_orient=torch.tensor(g,dtype=torch.float32).reshape(1,3),betas=torch.tensor(beta,dtype=torch.float32).reshape(1,-1),**fixed).joints[0].numpy()
 tr=np.array(anchor['position'])-native[15]
 return g,tr
