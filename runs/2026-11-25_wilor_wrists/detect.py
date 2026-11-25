"""Run the repository WiLoR entry point on four SHA-verified original images."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
OUT = ROOT/'outputs/wilor_wrists_20261125'
PHOTOS = ROOT/'outputs/gt_projection_elbows_20261124/image_inputs.json'
IMAGES = ROOT/'outputs/rw_a20000_dataset_20261117'
FRAMES = ('val055', 'val031')
EYES = ('left','right')


def read(path):
    return json.loads(path.read_text())


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    config = read(ROOT/'configs/pipeline.head01.developer.json')
    paths = config['paths']
    photos = read(PHOTOS)
    rows = []
    for frame in FRAMES:
        for eye, entry in zip(EYES, photos[frame]):
            path = IMAGES/entry['image']
            assert path.stem.endswith('_'+eye) and digest(path) == entry['sha256']
            rows.append(dict(image=str(path)))
    write(OUT/'images.json',dict(frames=rows))
    repo = Path(paths['wilor_repo'])
    asset_link = OUT/'mano_data'
    if not asset_link.exists():
        asset_link.symlink_to(repo/'mano_data', target_is_directory=True)
    assert asset_link.resolve() == (repo/'mano_data').resolve()
    opts = [('checkpoint','wilor_checkpoint'),('config','wilor_config'),('detector','wilor_detector')]
    hashes = {key:digest(Path(paths[key])) for _,key in opts}
    assert hashes['wilor_checkpoint'] == config['checkpoint_sha256']['wilor']
    assert hashes['wilor_detector'] == config['checkpoint_sha256']['detector']
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(ROOT),str(repo))), PYTHONDONTWRITEBYTECODE='1',
               YOLO_CONFIG_DIR=str(OUT/'yolo'), XDG_CACHE_HOME=str(OUT/'cache'), MPLCONFIGDIR=str(OUT/'mpl'),
               TORCH_HOME=str(OUT/'torch'), TORCHINDUCTOR_CACHE_DIR=str(OUT/'inductor'),
               TRITON_CACHE_DIR=str(OUT/'triton'), CUDA_CACHE_PATH=str(OUT/'cuda'), TMPDIR=str(OUT/'tmp'))
    for key in ('TMPDIR','YOLO_CONFIG_DIR','MPLCONFIGDIR'):
        Path(env[key]).mkdir(exist_ok=True)
    cmd = [paths['wilor_python'],'-Bu','-m','egosmplx_pipeline.wilor_predict','--manifest',str(OUT/'images.json'),
           '--output-dir',str(OUT/'observations'),'--confidence','0.20','--rescale-factor','2.0']
    for flag,key in opts:
        cmd += ['--'+flag, paths[key]]
    if sys.argv[1:] != ["--summarize"]:
        subprocess.run(cmd,cwd=OUT,env=env,check=True)
    rows = read(OUT/'observations/wilor_report.json')['frames']
    selected = {}
    for row in rows:
        stem = Path(row['image']).stem
        assert row['image'] in {r['image'] for r in read(OUT/'images.json')['frames']}
        selected[stem] = {}
        for side in ('L','R'):
            hits = [h for h in row['detections'] if h['is_right'] == (side=='R')]
            best = max(hits,key=lambda h:h['confidence']) if hits else None
            selected[stem][side] = {'count':len(hits),'score': best['confidence'] if best else None,
                                    'uv':best['joints_2d'][0] if best else None}
    assert len(selected)==4
    write(RUN/'evidence/detections.json',dict(selected=selected,weights_sha256=hashes,
          input_sha256={str(Path(entry['image']).stem):entry['sha256'] for f in FRAMES for entry in photos[f]},
          backend_source_sha256={str(q.relative_to(repo)):digest(q) for q in sorted((repo/'wilor').rglob('*.py'))},
          report_sha256=digest(OUT/'observations/wilor_report.json'),exit_status=0))


if __name__ == '__main__':
    main()
