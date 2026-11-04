"""Read completed reports only; no model execution or optimization."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/session_hand6_v2_no_margin_rerun_20261104/prepared/full_cpu_v1'
sources = {}

def read(path):
    sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())

def mean(values):
    return sum(values) / len(values)

def shoulder_rmse(errors):
    return math.sqrt(mean([v*v for k, v in errors.items() if k.endswith('_shoulder')]))

metrics = read(ROOT / 'runs/2026-11-04_v2-no-margin-rerun/evidence/metrics.json')
rows = []
for r in metrics['frames']:
    if not r['pair'].startswith('new'):
        continue
    base = OUT / r['frame'] / 'frames' / r['frame']
    rigid = read(base / 'rigid/optimization_report.json')
    pose = read(base / 'body_fit/automatic/refinement_report.json')
    guided = read(base / 'guided_body/optimization.json')
    scores = {q['name']:q['score'] for q in rigid['selected']}
    denom = max(1., sum(min(1., s)**2 for n,s in scores.items() if not n.endswith('_wrist')))
    final_errors = guided['after']['body_errors_px']
    shoulders = {}
    for name in ('left_shoulder', 'right_shoulder'):
        e = final_errors[name]
        shoulders[name] = dict(score=scores[name], error_px=e,
            fraction_of_body_group_weight=scores[name]**2/denom,
            robust_radial_derivative=2*e/(400+e*e),
            derivative_relative_to_20px=(2*e/(400+e*e))/.05,
            observation_radial_derivative=.01*scores[name]**2/denom*2*e/(400+e*e))
    h = pose['head_neck']
    rows.append(dict(pair=r['pair'],camera=r['camera'],frame=r['frame'],
        selected_body_names=list(scores),shoulders=shoulders,
        shoulder_rmse_by_stage_px={
            'rigid':shoulder_rmse(pose['target_fit_before']['per_joint_px']),
            'body_pose':shoulder_rmse(pose['target_fit_after']['per_joint_px']),
            'before_guided':shoulder_rmse(guided['initial']['body_errors_px']),
            'final':shoulder_rmse(final_errors)},
        body_pose_limits=pose['joints_at_increment_limit'],
        body_pose_increments_deg=pose['pose_increment_deg'],
        rigid_head_reference_radius_m=h['rigid_head_reference_radius_m'],
        body_pose_head_displacement_rmse_m=h['rigid_head_vertex_rmse_m'],
        body_pose_head_reference_loss=h['settings']['rigid_head_weight']*(h['rigid_head_vertex_rmse_m']/h['rigid_head_reference_radius_m'])**2,
        body_pose_objective=pose['best_objective'],
        historical_and_final={m:{k:r['methods'][m][k] for k in ['body_shoulders','pelvis_z_m','projected_shoulder_distance_px']} for m in ['original','edge40','no_margin']}))

result = dict(reference='Automatic Sapiens/WiLoR observations, not ground truth',
    scope='Six new views; no refitting. Pixel radial derivatives are not full parameter gradients.',
    rows=rows,
    mean_shoulder_rmse_by_stage_px={s:mean([r['shoulder_rmse_by_stage_px'][s] for r in rows]) for s in ['rigid','body_pose','before_guided','final']},
    mean_shoulder_rmse_by_method_px={m:mean([r['historical_and_final'][m]['body_shoulders']['rmse'] for r in rows]) for m in ['original','edge40','no_margin']},
    mean_shoulder_width_px={m:mean([r['historical_and_final'][m]['projected_shoulder_distance_px'] for r in rows]) for m in ['original','edge40','no_margin']},
    mean_pelvis_z_m={m:mean([r['historical_and_final'][m]['pelvis_z_m'] for r in rows]) for m in ['original','edge40','no_margin']},
    joint_limit_view_counts={n:sum(n in r['body_pose_limits'] for r in rows) for n in ['L_Collar','R_Collar','L_Shoulder','R_Shoulder','L_Elbow','R_Elbow']},
    selected_shoulder_observations=sum(len(r['shoulders']) for r in rows),
    source_sha256=sources)
(RUN/'evidence').mkdir(exist_ok=True)
(RUN/'evidence/diagnosis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['rows','source_sha256']},ensure_ascii=False,indent=2))
