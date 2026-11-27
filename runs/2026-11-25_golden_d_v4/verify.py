"""Verify both pinned method identities and the 24 reference selections after replay."""
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
BASE = '4a32d40765230fcf8db903cc102b1caef54484b1'
CFG = 'configs/golden_d.2d-v4.json'


def git_file(commit, name):
    return subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT)


def read(path):
    return json.loads((ROOT/path).read_bytes())


def main():
    manifest = read(CFG)
    old = manifest['previous_manifest']
    assert (ROOT/old).read_bytes()==git_file(BASE,'configs/golden_d.current.json')
    previous = read(old)
    assert (manifest['method_id'],previous['method_id'])==('golden-d-2d-v4','golden-d-2d-v3')
    for version in (previous,manifest):
        files = version['locked_files']
        assert len(files)==len({f['path'] for f in files})
        for f in files:
            data = (ROOT/f['path']).read_bytes()
            assert hashlib.sha256(data).hexdigest()==f['sha256'] and data==git_file(version['source_commit'],f['path'])
    assert manifest['implementation'].endswith('/lock.py') and manifest['reference_result_key']=='summary.locked'
    assert manifest['refinement']['output_array']=='locked' and manifest['refinement']['foot_body_pose_indices_zero_based']==[6,7,9,10]
    results = read(manifest['reference_results'])
    keys = set(results['rows'])
    replay = ROOT/'outputs/golden_d_v4_20261125/check/poses'
    logs = ROOT/'outputs/golden_d_v4_20261125/logs'
    exact = [line[6:] for log in logs.glob('*.txt') for line in log.read_text().splitlines() if line.startswith('exact ')]
    assert len(exact)==len(keys)==24 and set(exact)==keys
    for key in keys:
        assert not any(results['rows'][key]['metrics']['locked']['local_rotation_deg'])
        rec = read(replay/(key+'.json'))
        assert hashlib.sha256((replay/(key+'.npz')).read_bytes()).hexdigest()==rec['sha256']
    summary = results['summary']['locked']
    for metric,value in summary.items():
        assert abs(statistics.mean(row['metrics']['locked'][metric] for row in results['rows'].values())-value)<1e-8
    evidence = dict(base_commit=BASE,dirty_worktree=True,method=manifest['method_id'],implementation_commit=manifest['source_commit'],
                    manifest_sha256=hashlib.sha256((ROOT/CFG).read_bytes()).hexdigest(),
                    prior_manifest_sha256=hashlib.sha256((ROOT/old).read_bytes()).hexdigest(),
                    locked_files=len(manifest['locked_files']),reference_key=manifest['reference_result_key'],
                    reference_summary=summary,exact_pose_and_selection_matches=sorted(exact),
                    commands=[dict(command='bash <RUN>/reproduce.sh',exit_status=0),dict(command='python -B <RUN>/verify.py',exit_status=0)])
    (RUN/'evidence/verification.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print('Verified',manifest['method_id'],len(keys),'exact replays')


if __name__=='__main__':
    main()
