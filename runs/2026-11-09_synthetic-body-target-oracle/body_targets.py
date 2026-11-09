"""Replace only selected Sapiens target xyz after automatic initialization."""
import copy

def replace_targets(automatic,override):
    points=copy.deepcopy(automatic);changes=[]
    permitted={p['name'] for p in automatic if p['source']=='sapiens' and p['accepted']}
    expected=set()
    if override is not None:
        assert override['coordinate_frame']=='cam3 CV metres'
        expected={'left_hip','right_hip'} if override['scope']=='hips' else permitted
        assert override['scope'] in ['hips','body']
        assert len(override['targets'])==len(expected)
        assert {p['name'] for p in override['targets']}==expected
        for target in override['targets']:
            point=next(p for p in points if p['name']==target['name'] and p['source']=='sapiens')
            assert point['index']==target['index'] and point['accepted']
            changes.append(dict(name=point['name'],index=point['index'],before_xyz_m=point['xyz'],after_xyz_m=target['xyz'],only_changed_field='xyz'))
            point['xyz']=list(target['xyz'])
    for old,new in zip(automatic,points):
        assert {k:v for k,v in old.items() if k!='xyz'}=={k:v for k,v in new.items() if k!='xyz'}
        if old['source']!='sapiens' or old['name'] not in expected:assert old==new
    assert len(changes)==len(expected)
    return points,dict(changes=changes,initialization_uses_automatic_targets_only=True,covariances_weights_masks_unchanged=True,metadata_note='Gating/reprojection/covariance describe unchanged original observations; only loss target xyz changes. No new targets added.')
