"""Change only two loss target coordinates. Call strictly after initialization."""
import copy

def replace_targets(automatic,override):
    points=copy.deepcopy(automatic);changes=[]
    if override is not None:
        assert override['coordinate_frame']=='cam3 CV metres'
        assert {p['name'] for p in override['targets']}=={'left_hip','right_hip'}
        for target in override['targets']:
            point=next(p for p in points if p['name']==target['name'] and p['source']=='sapiens')
            assert point['index']==target['index'] and point['accepted']
            changes.append(dict(name=point['name'],index=point['index'],before_xyz_m=point['xyz'],after_xyz_m=target['xyz'],only_changed_field='xyz'))
            point['xyz']=list(target['xyz'])
    for old,new in zip(automatic,points):
        assert {k:v for k,v in old.items() if k!='xyz'}=={k:v for k,v in new.items() if k!='xyz'}
        if old['name'] not in {'left_hip','right_hip'}:assert old==new
    assert len(changes)==(2 if override is not None else 0)
    return points,dict(changes=changes,initialization_uses_automatic_targets_only=True,covariances_weights_masks_unchanged=True,metadata_note='Reprojection, pixel covariance and gating metadata describe the original automatic detections; only loss xyz is overridden.')
