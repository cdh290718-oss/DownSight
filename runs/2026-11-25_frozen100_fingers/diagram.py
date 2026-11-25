"""Display which observed hand boundaries survive the approximate finger exclusion."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
from geometry_inputs import RUN,OUT,EV,PRIOR,prev,Frame,GridEye,fit_probe
from hand_edges import MaskedEye,HANDS

def main():
    torch.set_num_threads(1);fit_probe.Eye=GridEye
    prev.base.fit.camera_files=prev.ref.camera_files
    model,cam=prev.ref.common.setup_model();cfg=prev.read(prev.base.OUT/'protocol.json')
    rows=prev.read(prev.base.OUT/'prepared.json')['items']
    fig,axes=plt.subplots(2,4,figsize=(8.5,4.6))
    for i,pair in enumerate(['pair055','pair068']):
        p=dict(np.load(PRIOR/'poses'/(pair+'.npz')));x=p['v3']
        item=next(r for r in rows if r['pair']==pair and r['condition']=='rw20000')
        f=Frame(pair,item,dict(body_pose=x[:21],delta=x[21],translation=x[22],global0=p['global0']),model,cam,cfg)
        weights=f.layer.lbs_weights.numpy();faces=f.faces
        digit=np.isin(f.labels,HANDS) & (weights[:,25:].sum(1)[faces].sum(1)>weights[:,[20,21]].sum(1)[faces].sum(1))
        with torch.no_grad():v=f.state(f.v3)[0]
        for j,(side,e) in enumerate(zip(['left','right'],f.eyes)):
            lab=np.load(PRIOR/'seg'/(pair+'_'+side+'.npy')).argmax(0)
            masked=MaskedEye(e,'palm',lab,digit,v)
            for hand,name in enumerate(HANDS):
                ax=axes[i,2*j+hand]
                original=e.targets.get(name,np.empty((0,2)))
                kept=masked.targets.get(name,np.empty((0,2)))
                ax.scatter(*original.T,s=8,color='#9c9da2')
                ax.scatter(*kept.T,s=10,color='#147b67')
                ax.invert_yaxis();ax.set_aspect('equal');ax.axis('off')
                if i==0:ax.set_title(side.capitalize()+' eye / '+('L' if hand==0 else 'R')+' hand',fontsize=10)
                ax.text(.5,-.04,pair+'   '+str(len(kept))+'/'+str(len(original)),transform=ax.transAxes,ha='center',fontsize=9)
    fig.suptitle('Observed hand edges used by the palm-only approximation')
    fig.legend(handles=[Line2D([0],[0],marker='o',linestyle='',color=c,label=label) for c,label in
        [('#147b67','Kept'),('#9c9da2','Excluded')]],loc='lower center',ncol=2,frameon=False)
    fig.tight_layout(rect=(0,.08,1,.92))
    fig.savefig(EV/'hand_edges.png',dpi=130)

if __name__=='__main__':main()
