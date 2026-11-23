"""Fit both objectives from template without reading past fits or evaluation labels."""
import json
from pathlib import Path
import sys
import torch

from angles import EVIDENCE, OUTPUT, ROOT, features, sha, write

sys.path.insert(0, str(ROOT/'runs/2026-11-19_rw_frozen200_free12'))
import run_fit as base


class EmpiricalProblem(base.trial.TorsoOnly):
    def __init__(self, prepared, cfg, model, camera_cls):
        super().__init__(prepared, cfg, model, camera_cls)
        prior = json.loads((EVIDENCE/'prior.json').read_text())
        self.lower = self.t(prior['lower'])
        self.upper = self.t(prior['upper'])
        self.scales = self.t(prior['scales'])
        self.weight = prior['weight']
        self.lookup = {p['name']: p['native_index'] for p in prepared[0]['points']}
        self.apply_prior = self.t([i['condition'] == 'rw_prior' for i in self.items])

    def objective(self, values):
        original, feasible, terms = super().objective(values)
        f = features(values[0], self.lookup)
        violation = (self.lower-f).relu() + (f-self.upper).relu()
        extra = self.weight * (violation/self.scales).square().mean((1, 2)) * self.apply_prior
        terms['rw_rom'] = extra
        return original + extra, feasible, terms


def main(split):
    if split not in ('validation', 'test'):
        raise ValueError(split)
    def guard(event, args):
        if event == 'open' and args and isinstance(args[0], (str, Path)):
            path = Path(args[0])
            blocked_names = {'samples.npz', 'reference.json', 'validation_truth.json', 'results.json', 'survey.json'}
            if path.name in blocked_names or ('pseudo-gt' in str(path).lower()) or (path.name.startswith('budget_') and path.parent != OUTPUT/split):
                raise PermissionError(path.name)
    sys.addaudithook(guard)
    previous = ROOT/'outputs/stereo_wrist_candidates_20261123/validation_fit' if split == 'validation' else base.OUT
    folder = OUTPUT/split
    folder.mkdir(parents=True, exist_ok=False)
    settings = previous/'protocol.json'
    audit_path = ROOT/'runs/2026-11-23_stereo_wrist_candidates/evidence/splits.json'
    expected = (base.read(audit_path)['hashes']['validation_prepared'] if split == 'validation'
                else base.read(base.RUN/'evidence/inputs.json')['solver_input_sha256'])
    assert sha(previous/'prepared.json') == expected
    source = base.read(previous/'prepared.json')['items']
    condition = 'baseline' if split == 'validation' else 'rw20000'
    inputs = [r for r in source if r['condition'] == condition]
    assert len(inputs) == len({r['pair'] for r in inputs}) == (200 if split == 'validation' else 100)
    rows = [dict(row, condition=arm) for row in inputs for arm in ('baseline', 'rw_prior')]
    write(folder/'prepared.json', dict(items=rows))
    (folder/'protocol.json').write_bytes(settings.read_bytes())
    base.fit.RUN = base.fit.OUT = folder
    base.fit.camera_files = base.camera_files
    base.fit.Problem = EmpiricalProblem
    sys.argv = [__file__, '--threads', '1']
    base.fit.main()
    done = base.read(folder/'fit_complete.json')
    write(EVIDENCE/(split+'_fit.json'), dict(command='bash run.sh fit_prior '+split,
        exit_status=done['exit_status'], elapsed_seconds=done['elapsed_seconds'], steps=done['steps'],
        source_input_sha256=expected, protocol_sha256=sha(settings), prior_sha256=sha(EVIDENCE/'prior.json'),
        camera_sha256=[sha(p) for p in base.camera_files()], selected_fit_sha256=sha(folder/'budget_20000.npz'),
        pair_count=len(inputs), conditions=['baseline','rw_prior']))


if __name__ == '__main__':
    main(sys.argv[1])
