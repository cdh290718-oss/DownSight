"""Replay the same automatic observations with no historical head orientation."""
import copy
import sys
from study import OUT,read,write
from camera_only import CameraProblem
import fit

dest=OUT/'automatic_head'
cfg=read(OUT/'automatic/protocol.json')
items=[]
for base in read(OUT/'automatic/prepared.json')['items']:
    count=12 if base['condition']=='baseline' else 15
    for orientation in ['camera','torso']:
        item=copy.deepcopy(base)
        item['anchor']['rotation']=cfg['camera_R_head_proxy']
        item['head_mode']=orientation
        item['condition']='auto%d_%s'%(count,orientation)
        items.append(item)
# The base solver obtains shared first-pass initialization from the condition named baseline.
for item in items:
    if item['condition']=='auto12_camera':
        item['condition']='baseline'
cfg['conditions']=['baseline','auto12_torso','auto15_camera','auto15_torso']
write(dest/'protocol.json',cfg)
write(dest/'prepared.json',dict(items=items))
fit.Problem=CameraProblem
fit.RUN=fit.OUT=dest
sys.argv=[__file__,'--threads','1']
fit.main()
