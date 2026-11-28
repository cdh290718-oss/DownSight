import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
GALLERY = ROOT / 'runs/2026-11-23_fixed_floor24/evidence/gallery.json'
PREPARATION = ROOT / 'runs/2026-11-18_rw_frozen200_golden_d/evidence/preparation.json'
SEGMENTATION = ROOT / 'runs/2026-11-25_frozen100_outline/evidence/segmentation.json'
PROBS = ROOT / 'outputs/frozen100_outline_20261125/seg'
MANIFEST = ROOT / 'outputs/rw_test_completion_20261117/test_manifest.json'
STRIDE = 4
CLASSES = [
    ('foreground', None, (255, 255, 255)),
    ('left_hand', 6, (255, 255, 0)),
    ('right_hand', 15, (255, 255, 0)),
    ('left_forearm', 7, (0, 255, 255)),
    ('right_forearm', 16, (0, 255, 255)),
    ('left_upperarm', 11, (0, 150, 255)),
    ('right_upperarm', 20, (0, 150, 255)),
    ('left_thigh', 12, (40, 220, 40)),
    ('right_thigh', 21, (40, 220, 40)),
]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def boundary(mask):
    edge = mask & (cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) == 0)
    edge[[0, -1], :] = False
    edge[:, [0, -1]] = False
    return edge


def main():
    gallery = [row for row in read(GALLERY)['rows'] if row['split'] == 'test']
    prep = read(PREPARATION)
    assert sha(MANIFEST) == prep['manifest_sha256']
    pairs = {row['pair']: row for row in prep['pairs']}
    images = {row['id']: row for row in read(MANIFEST)['records']}
    seg = read(SEGMENTATION)
    assert len(gallery) == 12
    result = []
    for row in gallery:
        pair = row['pair']
        original = pairs[pair]
        entry = {'pair': pair, 'subject': row['subject'], 'eyes': {}}
        for side, image_id, image_hash in zip(('left', 'right'), original['image_ids'], original['image_sha256']):
            image_path = MANIFEST.parent / images[image_id]['image']
            prob_path = PROBS / (pair + '_' + side + '.npy')
            assert images[image_id]['view'] == side
            assert sha(image_path) == image_hash == seg['eyes'][pair + '_' + side]['image_sha256']
            assert sha(prob_path) == seg['eyes'][pair + '_' + side]['sampled_sha256']
            image = cv2.imread(str(image_path))
            assert image is not None and image.shape[:2] == (872, 872)
            prob = np.load(prob_path).astype(float)
            assert prob.shape == (29, 218, 218) and np.isfinite(prob).all()
            label = prob.argmax(axis=0)
            confidence = prob.max(axis=0)
            counts = {}
            overlay = image.copy()
            for name, category, color in CLASSES:
                mask = label != 0 if category is None else label == category
                sure = (1 - prob[0] if category is None else confidence) >= 0.55
                edge = boundary(mask) & sure
                area_count = int((mask & sure).sum())
                edge_count = int(edge.sum())
                accepted = area_count >= 12 and edge_count >= 6
                counts[name] = {'area': area_count, 'boundary': edge_count, 'used': accepted}
                if accepted:
                    yy, xx = np.nonzero(edge)
                    for x, y in zip(xx, yy):
                        cv2.circle(overlay, (int(x * STRIDE + 2), int(y * STRIDE + 2)), 2, color, -1)
            asset = RUN / 'assets' / (pair + '_' + side + '.jpg')
            assert not asset.exists()
            assert cv2.imwrite(str(asset), overlay, [cv2.IMWRITE_JPEG_QUALITY, 91])
            check = cv2.imread(str(asset))
            assert check is not None and check.shape == image.shape
            entry['eyes'][side] = {'image_sha256': image_hash, 'probability_sha256': sha(prob_path),
                                   'render_sha256': sha(asset), 'channels': counts}
        result.append(entry)
        print(pair, flush=True)
    evidence = {'base_commit': 'bbc60839d3635be7da15690f48c1f89e3dbb3cb5',
                'dirty_worktree': True, 'command': 'python -B <RUN>/render.py',
                'preparation_sha256': sha(PREPARATION), 'segmentation_sha256': sha(SEGMENTATION),
                'gallery_sha256': sha(GALLERY), 'frames': 12, 'views': 24, 'rows': result,
                'exit_status': 0}
    (RUN / 'evidence/results.json').write_text(json.dumps(evidence, indent=2) + '\n')


if __name__ == '__main__':
    main()
