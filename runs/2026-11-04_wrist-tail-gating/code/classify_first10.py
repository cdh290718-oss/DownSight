"""Describe each flagged supplementary wrist using metrics and image-backed diagnostics."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--metrics',type=Path,required=True);p.add_argument('--identity',type=Path,required=True);p.add_argument('--cross',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();m=json.loads(a.metrics.read_text());ident=json.loads(a.identity.read_text());cross=json.loads(a.cross.read_text());desc={
(0,'left'):('cross-hand duplicate retained / unresolved','Multiple left and right fingertip detections coincide in both views. The rule removes none for this side because wrist-distance margin is insufficient; both fits remain above threshold.'),
(270,'left'):('fit conflict / nearby cross-hand ambiguity','Wrist triangulation is close to DFM and the candidate reduces, but does not conclusively isolate, the source of the baseline difference; the opposite wrist has a serious detection ambiguity.'),
(270,'right'):('outside-view and wrong-side observation / insufficient constraint','DFM projects outside cam2. The detector right wrist in cam2 is near its left wrist, stereo fails and published gating discards the right wrist; both models have large DFM difference.'),
(810,'right'):('wrist fit vs observation conflict / unresolved','Stereo wrist differs from DFM by about 3 cm and both fits remain about 5 cm from DFM; no confirmed cross-hand identity error or unique root cause.'),
(1080,'left'):('cross-hand 2D mismatch / original gate removes conflicting points','Several left fingertips coincide with right fingertips in one view but not the other; original gate rejects them for stereo inconsistency and the combined candidate sharply reduces the wrist difference.'),
(2160,'left'):('overlap / suspected cross-hand fingertip conflict','Hands overlap in the image; the candidate removes a left fingertip closer to the opposite wrist. The combined gate reduces wrist difference, but other gate changes prevent attribution to this one point alone.')}
out=[]
for r in m['per_frame']:
 for side,v in r['wrists'].items():
  if not v['bad_frame']:continue
  key=(r['frame'],side);assert key in desc;category,explanation=desc[key];out.append({'frame':key[0],'side':side,'category':category,'explanation':explanation,'wrist_stereo_vs_dfm_cm':v['observation_vs_dfm_cm'],'fit_vs_dfm_cm':{k:z['vs_dfm_cm'] for k,z in v['fits'].items()},'cross_hand_dropped':[z['joint'] for z in cross['decisions'] if z['frame']==key[0] and z['joint'].startswith(side+'_') and z['dropped']],'identity_diagnostics':[z for z in ident['points'] if z['frame']==key[0]],'overlays':{'wrist':f'first10_crops/frame_{key[0]:05d}.jpg','identity':f'first10_identity_overlays/frame_{key[0]:05d}.jpg'}})
assert set(desc)=={(r['frame'],r['side']) for r in out}
with a.output.open('x') as f:json.dump({'rule':'Either baseline or candidate 500/1500 Euclidean DFM difference >= 5cm','cases':out,'note':'Descriptions use image and ablation evidence; DFM is an unvalidated model reference.'},f,indent=2,allow_nan=False);f.write('\n')
