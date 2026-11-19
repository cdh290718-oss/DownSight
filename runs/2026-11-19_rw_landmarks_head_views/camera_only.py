"""Two operational 15-point variants using no historical rotation, even to initialize."""
import sys

import torch

from study import OUT, read, write
import fit

DEST=OUT/'camera_only'


def prepare():
    cfg=read(OUT/'protocol.json')
    cfg['conditions']=['baseline','torso']
    items=[]
    for item in read(OUT/'prepared.json')['items']:
        if item['condition'] not in ['15camera','15torso']:
            continue
        item['condition']='baseline' if item['head_mode']=='camera' else 'torso'
        item['anchor']['rotation']=cfg['camera_R_head_proxy']
        items.append(item)
    write(DEST/'protocol.json',cfg)
    write(DEST/'prepared.json',dict(items=items))


def rotate_about_head(x, head, correction):
    return (x-head)@correction.transpose(-1,-2)+head


class CameraProblem(fit.Problem):
    def __init__(self, prepared,cfg,model,camera_cls):
        super().__init__(prepared,cfg,model,camera_cls)
        self.lock=torch.tensor([x['head_mode']=='camera' for x in self.items])
        self.movable=torch.ones(len(self.items),21,1)
        for joint in [12,15]:
            self.movable[~self.lock,joint-1]=0
        self.ratios=cfg['landmark_ratios']

    def parameters(self):
        body,beta=super().parameters()
        return body*self.movable,beta

    def forward(self):
        native,rh,beta=super().forward()
        corr=torch.where(self.lock[:,None,None],self.head_rot@rh.transpose(-1,-2),torch.eye(3))
        native=rotate_about_head(native,native[:,15:16],corr)
        points=[native[:,a]+k*(native[:,b]-native[:,a])
                for a,b,k in zip([15,8,7],[12,11,10],self.ratios)]
        return torch.cat([native,torch.stack(points,dim=1)],dim=1),corr@rh,beta

    def objective(self,values):
        native,_,beta=values
        # Camera orientation enforced exactly by parameterization; torso pose by locked local rotations.
        return super().objective((native,self.head_rot,beta))


if __name__=='__main__':
    prepare()
    fit.Problem=CameraProblem
    fit.RUN=fit.OUT=DEST
    sys.argv=[__file__,'--threads','1']
    fit.main()
