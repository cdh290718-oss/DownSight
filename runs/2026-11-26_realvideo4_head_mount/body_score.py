"""Score the full-body result on the same RW20000 observations as the original report."""
from measure import ROOT,RUN,OUT,V4,PRIOR,T,cams,projection,read,write,sha,np

def main():
    folder=OUT/'fit/0'
    original=read(V4/'evidence/results.json')['cases']
    changed=read(RUN/'evidence/refitted.json')['rows']
    target=np.array(read(RUN/'evidence/fit003.json')['anchor_candidate_cam3_m'])
    prior_items=read(ROOT/'outputs/realvideo4_dv3_20261125/fit/0/prepared.json')['items']
    rows={}
    for i,item in enumerate(read(folder/'prepared.json')['items']):
        assert all(item[f]==value for f,value in prior_items[i].items() if f!='head_position')
        assert np.array_equal(item['head_position'],target)
        k=item['pair']
        data=np.load(OUT/'refitted'/(k+'.npz'))
        errors=[]
        for e in range(2):
            joints=data['body_joints'][[p['native_index'] for p in item['points']]]
            uv=projection(joints if e==0 else joints@T[:3,:3].T+T[:3,3],cams[e])
            for p,pt in zip(item['points'],uv):
                if p['valid'][e]:
                    errors.append(np.sum((pt-np.array(p['xy'][e]))**2))
        assert len(errors)==original[k]['valid_observations']
        selection=read(OUT/'evidence'/(k+'.json'))
        path=OUT/'poses'/(k+'.npz')
        assert sha(path)==selection['pose_hash']
        rows[k]=dict(rmse_before_px=original[k]['combined_rmse_px'],rmse_after_px=float(np.sqrt(np.mean(errors))),
                     v4_pose_sha256=sha(path),selected=selection['selected'],valid=len(errors),
                     head_target_error_mm=float(np.linalg.norm(data['head']-target)*1000),
                     contour_after_px=12*selection['choice_scores'][selection['selected']]['contour'],
                     contour_before_px=12*original[k]['contour_v4'])
    write(RUN/'evidence/body.json',dict(rows=rows,steps=read(folder/'fit_complete.json')['steps'],
          prepared_sha256=sha(folder/'prepared.json'),protocol_sha256=sha(folder/'protocol.json'),
          diagnostics=read(folder/'diagnostic_20000.json'),exit_status=0))
    for k,r in rows.items():print(k,r['rmse_before_px'],r['rmse_after_px'])
if __name__=='__main__':main()
