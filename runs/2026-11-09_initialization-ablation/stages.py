"""Predeclared evaluation checkpoints; no GT-based selection."""
from common import ROOT,RUN,OUT,read

def stages():
 rows=[dict(label='original',folder=ROOT/'outputs/synthetic_body_target_oracle_20261109/control_left',file='final',case='original',budget=3400)]
 for c in read(RUN/'cases.json'):
  folder=OUT/(c['case']+'_left')
  for budget in [0]+read(RUN/'settings.json')['budget_steps']:
   rows.append(dict(label=c['case']+'_'+str(budget),folder=folder,file='initial' if budget==0 else 'budget_'+str(budget),case=c['case'],budget=budget))
 return rows
