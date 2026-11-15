#!/usr/bin/env python3
"""Check golden D archive identity and public file integrity; does not run a fit."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / 'experiments/synthetic_template_head_stereo_20261109/golden_d.json'
ARCHIVE = 'd26b9ee292204506492fb9e1d221d683531465ea'
SOURCE = '485fc71ca37d3c68a74bf99338fe247c197bf6eb'


def main():
    lock = json.loads(LOCK.read_text(encoding='utf-8'))
    manifest = json.loads((LOCK.parent / 'manifest.json').read_text(encoding='utf-8'))
    case = json.loads((LOCK.parent / 'case.json').read_text(encoding='utf-8'))
    expected_case = {'case': 'D', 'seed': 'left', 'shape_seed': 'mean', 'pose_seed': 'template'}
    if lock['archive_commit'] != ARCHIVE or lock['source_publication_commit'] != SOURCE:
        raise ValueError('Unexpected archive or source commit')
    if lock['case'] != expected_case or case != expected_case or manifest['primary_case'] != expected_case:
        raise ValueError('Golden D case definition mismatch')
    for key, manifest_key in [('budget_steps', 'primary_budget_steps'), ('selected_training_step', 'selected_training_step'), ('dataset', 'dataset')]:
        if lock[key] != manifest[manifest_key]:
            raise ValueError('Golden D metadata mismatch: ' + key)
    if lock['external_inputs'] != manifest['external_inputs'] or lock['external_outputs'] != manifest['external_outputs']:
        raise ValueError('External resource records differ from archived manifest')
    resource_paths = [item['path'] for item in lock['tracked_resources']]
    required = ['experiments/synthetic_template_head_stereo_20261109/manifest.json',
                'experiments/synthetic_template_head_stereo_20261109/case.json',
                'runs/2026-11-09_initialization-ablation/fit.py',
                'runs/2026-11-09_initialization-ablation/assets/left_D_20000.jpg',
                'runs/2026-11-09_initialization-ablation/assets/right_D_20000.jpg']
    if len(set(resource_paths)) != len(resource_paths) or not set(required).issubset(resource_paths):
        raise ValueError('Missing or duplicate locked resource')
    for item in lock['tracked_resources']:
        rel = Path(item['path'])
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Unsafe resource path')
        current = (ROOT / rel).read_bytes()
        original = subprocess.run(['git', 'show', ARCHIVE + ':' + rel.as_posix()], cwd=ROOT,
                                  capture_output=True, check=True).stdout
        if current != original or len(current) != item['bytes'] or hashlib.sha256(current).hexdigest() != item['sha256']:
            raise ValueError('Golden D resource mismatch: ' + rel.as_posix())
    print('PASS: golden D identity, manifest and %d public files match pinned archive.' % len(resource_paths))
    print('External assets, repeated runs, numerical tolerance and scientific accuracy were not verified.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print('FAIL: ' + str(exc), file=sys.stderr)
        sys.exit(1)
