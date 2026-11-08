"""Run original body-only pose optimizer, changing only near-depth scope."""
import argparse, runpy, sys
p=argparse.ArgumentParser();p.add_argument('--scope',choices=['all','non_head_neck'],required=True);args,rest=p.parse_known_args()
if args.scope=='non_head_neck':
    from egosmplx_pipeline import v2_policy
    def min_z(vertices):
        ids,_,_=v2_policy.vertex_indices()
        return vertices[ids,2].min()
    v2_policy.min_z=min_z
sys.argv=['egosmplx_pipeline.body_pose']+rest
runpy.run_module('egosmplx_pipeline.body_pose',run_name='__main__')
