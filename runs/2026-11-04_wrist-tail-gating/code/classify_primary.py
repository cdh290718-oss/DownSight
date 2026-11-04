"""Evidence-backed case descriptions for every flagged primary sample."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--metrics',type=Path,required=True);p.add_argument('--identity',type=Path,required=True);p.add_argument('--cross',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();m=json.loads(a.metrics.read_text());identity=json.loads(a.identity.read_text());cross=json.loads(a.cross.read_text());desc={
(2700,'left'):('gating-worsened fit / unresolved','Wrist triangulation is close to DFM; published gating raises 500-step error. 1500 steps and no-fingertip ablation reduce it; no unique mechanism or proven detector misidentity.'),
(2970,'left'):('cross-hand 2D ambiguity / fit interaction','Several left fingertip detections nearly overlap right fingertips; published gate rejects inconsistent left fingertips. Wrist stereo is close to DFM; both gating and no-fingertip fit reduce the tail.'),
(3240,'left'):('cross-hand fingertip duplication / fit conflict','Four left fingertips lie close to same-named right fingertips in both views and in 3D; published gate accepts them. Cross-hand removal and no-fingertip ablation both reduce wrist discrepancy.'),
(3780,'left'):('cross-hand fingertip duplication / fit conflict','Four left fingertip points lie closer to the opposite wrist; several nearly duplicate right fingertips. Published gate accepts them. Cross-hand removal and no-fingertip ablation reduce wrist discrepancy.'),
(4320,'left'):('cross-hand fingertip plausibility conflict','Several left fingertip 3D candidates are much closer to the opposite wrist; the cross-hand filter removes four. Images show hands and object overlap, but exact identity and occlusion cannot be independently established.'),
(4860,'right'):('fingertip-fit interaction / unresolved','Right wrist stereo is near DFM, while baseline fitting is farther away. No-fingertip and longer fits improve the reference difference; there is no confirmed cross-hand duplicate for this wrist.')}
rows=[]
for frame in m['per_frame']:
 for side,v in frame['wrists'].items():
  if not v['bad_frame']:continue
  k=(frame['frame'],side);assert k in desc;cat,explain=desc[k];chosen=[x for x in cross['decisions'] if x['frame']==k[0] and x['joint'].startswith(side+'_') and x['dropped']];ids=[x for x in identity['points'] if x['frame']==k[0]]
  rows.append({'frame':k[0],'side':side,'category':cat,'explanation':explain,'baseline_stereo_wrist_vs_dfm_cm':v['observation_vs_dfm_cm'],'baseline_fit_to_stereo_cm':v['fits']['baseline_500']['vs_observation_cm'],'fit_vs_dfm_cm':{name:v['fits'][name]['vs_dfm_cm'] for name in ('baseline_500','published_gate_500','published_gate_cross_hand_500','wrist_only_500','baseline_1500','published_gate_cross_hand_1500')},'cross_hand_dropped':[x['joint'] for x in chosen],'identity_diagnostics':ids,'overlays':{'wrist_500':f'wrist_crops/frame_{k[0]:05d}.jpg','wrist_1500':f'wrist_crops_1500/frame_{k[0]:05d}.jpg','identity':f'identity_overlays/frame_{k[0]:05d}.jpg'}})
assert set(desc)=={(r['frame'],r['side']) for r in rows}
with a.output.open('x') as f:json.dump({'rule':m['bad_frame_rule'],'cases':rows,'note':'Causal descriptions are hypotheses supported by ablations and image observations; DFM is not independent anatomy ground truth.'},f,indent=2,allow_nan=False);f.write('\n')
