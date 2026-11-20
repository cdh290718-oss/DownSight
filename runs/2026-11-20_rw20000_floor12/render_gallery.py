"""Fit one carpet plane per stereo exposure, then render the archived mesh."""
import sys
from pathlib import Path

import cv2
import numpy as np

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
sys.path.insert(0, str(ROOT / 'runs/2026-11-20_rw20000_floor3'))
import assess as floor
from assess import g

# Bare carpet in the raw left photographs (half resolution); original three masks retained.
REGIONS = dict(floor.REGIONS, **{
    'pair001': [[(104,219),(132,224),(153,260),(173,325),(178,365),(203,384),(166,379),(138,351),(116,307)],
                [(232,348),(253,295),(282,271),(332,282),(352,283),(375,275),(373,298),(350,323),(312,346),(291,377),(232,389),(213,379)]],
    'pair022': [[(122,240),(152,256),(182,283),(230,293),(261,286),(266,304),(263,354),(233,374),(191,367),(162,337),(139,312)],
                [(273,246),(292,200),(310,161),(326,168),(338,199),(365,201),(384,198),(404,195),(411,225),(403,253),(371,261),(337,249),(308,270)]],
    'pair028': [[(38,237),(60,222),(77,232),(97,255),(148,281),(190,295),(240,283),(275,275),(314,260),(317,280),(298,311),(277,349),(233,385),(172,383),(161,344),(140,307),(77,287),(44,301)]],
    'pair034': [[(109,243),(125,261),(148,267),(166,245),(167,219),(191,210),(219,214),(243,233),(263,230),(286,212),(320,201),(341,210),(352,235),(330,260),(296,285),(241,322),(192,346),(151,345),(136,312),(114,274)]],
    'pair036': [[(79,230),(99,224),(136,238),(182,255),(216,249),(228,219),(253,217),(273,206),(275,174),(300,151),(324,157),(329,173),(311,184),(295,240),(290,277),(301,298),(273,304),(229,314),(200,343),(151,330),(126,315),(137,290),(118,277),(82,273)]],
    'pair037': [[(94,241),(117,253),(151,272),(166,263),(160,248),(190,238),(220,240),(246,236),(284,222),(301,219),(302,238),(272,264),(260,278),(273,295),(291,292),(284,306),(241,315),(174,312),(128,300),(103,276)]],
    'pair068': [[(287,147),(306,165),(331,169),(349,152),(359,179),(358,204),(348,234),(340,251),(327,260),(308,280),(275,296),(243,297),(243,276),(264,255),(277,227)],
                [(123,277),(138,270),(163,265),(184,277),(201,291),(203,310),(184,328),(155,322),(131,307),(118,299)]],
    'pair094': [[(133,259),(157,235),(182,237),(209,247),(238,258),(267,250),(289,239),(299,264),(330,273),(359,252),(372,255),(367,291),(350,323),(325,346),(303,369),(254,386),(205,382),(177,360),(145,346),(126,321),(125,287)]],
    'pair088': [[(91,247),(109,248),(124,268),(146,283),(166,275),(174,257),(192,252),(216,269),(236,275),(254,264),(268,244),(310,223),(316,210),(334,229),(338,261),(329,285),(308,303),(288,325),(255,347),(216,363),(176,366),(137,352),(119,324),(101,307)]],
})


RETRY = {'pair001','pair022','pair028','pair034','pair068','pair094'}


# Wider exposed carpet and flat tape, with raised props and limbs excluded.
REVISED = {
    'pair001': REGIONS['pair001'] + [
        [(257,89),(296,90),(317,111),(322,123),(310,134),(285,116),(253,103)],
        [(313,207),(334,250),(315,255),(294,254),(275,263),(287,236)],
        [(270,286),(346,278),(373,263),(399,252),(394,286),(376,308),(351,323),(326,331),(294,346),(254,355)]],
    'pair034': [[(104,260),(142,273),(164,250),(165,218),(212,215),(241,237),(265,231),
                 (276,212),(314,188),(337,200),(350,232),(376,251),(354,274),(312,306),
                 (270,330),(234,349),(196,360),(175,363),(148,345),(135,315),(119,291)],
                [(91,210),(110,209),(128,205),(124,226),(117,250),(123,261),(109,272),(98,254)]],
    'pair068': [
        [(292,88),(305,91),(341,102),(348,124),(343,142),(332,153),(309,153),(301,140),
         (318,145),(337,138),(331,123),(318,122),(306,133),(290,127)],
        [(288,151),(311,151),(328,156),(348,141),(358,157),(361,209),(354,242),
         (344,250),(329,248),(316,268),(310,283),(285,287),(267,293),(242,296),
         (240,282),(263,249),(279,220)],
        [(115,269),(143,257),(158,251),(181,262),(205,277),(209,299),(226,314),
         (221,329),(177,329),(143,325),(120,311),(104,294),(112,281),(132,290),
         (153,302),(167,286),(160,271),(148,270),(133,279)],
        [(181,196),(232,197),(228,225),(218,251),(206,244),(194,229)]]
}


def render_views(vertices, faces, up, forward):
    z = floor.unit(forward - up * (up @ forward))
    x = floor.unit(np.cross(up, z))
    # Same side as in the original three-frame assessment.
    bases = [np.column_stack([x, up, z]), np.column_stack([z, up, -x])]
    coordinates = [vertices @ basis for basis in bases]
    scale = min(min(313 / np.ptp(v[:, 0]), 438 / np.ptp(v[:, 1])) for v in coordinates)
    for basis, v in zip(bases, coordinates):
        assert np.allclose(basis.T @ basis, np.eye(3)) and np.isclose(np.linalg.det(basis), 1)
        yield g.free_render(vertices, faces, basis, (v.max(0) + v.min(0)) / 2, scale)


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('pairs', nargs='*')
    parser.add_argument('--initial',action='store_true',help='Repeat the original SIFT-only plane estimates')
    args = parser.parse_args()
    selected = args.pairs
    if args.initial and selected:
        parser.error('The initial pass covers all twelve pairs')
    out = ROOT / 'outputs/rw20000_floor12_20261120'
    source = g.read(g.RUN / 'evidence/renders_rw20000.json')['rows']
    provenance = g.read(ROOT / 'runs/2026-11-18_rw_frozen200_golden_d/evidence/preparation.json')
    pairs = {r['pair']:r for r in provenance['pairs']}
    assert g.sha(g.MANIFEST) == provenance['manifest_sha256']
    images = {r['id']:r for r in g.read(g.MANIFEST)['records']}
    inp = g.read(g.REF / 'inputs.json')
    assert g.sha(g.SOURCE / 'prepared.json') == inp['solver_input_sha256']
    checkpoint = g.SOURCE / 'budget_20000.npz'
    assert g.sha(checkpoint) == g.read(g.REF / 'execution.json')['params_sha256']
    items = g.read(g.SOURCE / 'prepared.json')['items']
    model, cam = g.common.setup_model()
    calfiles = [g.DATA / ('camera_' + side + '.json') for side in ['left','right']]
    for path in calfiles:
        assert g.sha(path) == inp['cameras'][path.name]
    cams = [floor.RawCamera(cam(p), g.read(p)['input_adapter']) for p in calfiles]
    old = {r['pair']:r for r in g.read(floor.RUN / 'evidence/planes.json')['rows']}
    initial = {}
    with np.load(checkpoint) as params:
        for entry in source:
            pair = entry['pair']
            if selected and pair not in selected:
                continue
            p = pairs[pair]
            i = next(i for i,it in enumerate(items) if (it['pair'], it['condition']) == (pair,'rw20000'))
            views = []
            for idx, key in enumerate(p['image_ids']):
                path = g.MANIFEST.parent / images[key]['image']
                assert g.sha(path) == p['image_sha256'][idx]
                views.append(cv2.imread(str(path)))
            T = np.array(items[i]['inputs']['views']['right']['camera']['left_to_view'])
            polys = [np.asarray(poly,np.int32)*2 for poly in REGIONS[pair]]
            method = 'sift'
            if pair in RETRY and not args.initial:
                from patch_match import estimate
                method = 'epipolar_patch'
                if pair in REVISED:
                    mask = np.zeros(views[0].shape[:2], np.uint8)
                    for poly in REVISED[pair]:
                        cv2.fillPoly(mask,[np.asarray(poly,np.int32)*2],255)
                    polys = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2]
                up, d, quality, _ = estimate(views, cams, T, polys)
                if pair in REVISED and not quality['accepted']:
                    # Use the better-supported provisional estimate if neither method passes.
                    candidate = floor.estimate(views, cams, T, polys)
                    other = candidate[2]
                    first = quality.copy()
                    def rank(q):
                        return (q['accepted'], q['ransac_inliers'] >= 8, -max(q['checker_angle_deg']/5,
                                q['checker_distance_diff_mm']/50,q['sparse_median_residual_mm']/50,q['sparse_vs_dense_deg']/10))
                    if rank(other) > rank(quality):
                        up, d, quality, _ = candidate
                        method = 'sift_revised'
                    quality['revised_patch_check'] = first
                    quality['revised_sift_check'] = other.copy() if method != 'sift_revised' else dict(
                        (k,v) for k,v in other.items() if k not in ('revised_patch_check','revised_sift_check'))

            else:
                up, d, quality, _ = floor.estimate(views, cams, T, polys)
            if args.initial:
                initial[pair] = dict(up=up.tolist(),height_m=d,validation=quality)
                continue
            obj = g.OUT / 'objects' / entry['subject'] / entry['obj']
            assert g.sha(obj) == entry['source_obj_sha256']
            vertices, faces = g.load_obj(obj)
            forward = (g.rot(params['delta'][i]) @ g.rot(params['global0'][i]))[:,2]
            hashes = {}
            for name, im in zip(['front','side'], render_views(vertices, faces, up, forward)):
                path = RUN / 'assets' / pair / (name + '.jpg')
                g.save(path, im)
                hashes[path.name] = g.sha(path)
            check = None
            if pair in old:
                check = dict(angle_deg=floor.angle(up,np.array(old[pair]['up_left_cv'])),
                             offset_change_mm=float(1000*(d-old[pair]['camera_height_m'])))
            record = dict(pair=pair, method=method, subject=entry['subject'], obj_sha256=g.sha(obj), up=up.tolist(),
                          height_m=d, validation=quality, prior_repeat=check, image_sha256=p['image_sha256'], images=hashes)
            g.write(out / (pair+'.json'),record)
            print(pair, quality, flush=True)
    if args.initial:
        g.write(RUN / 'evidence/initial.json',initial)
        return
    records = [g.read(out / (r['pair']+'.json')) for r in source if (out / (r['pair']+'.json')).exists()]
    if len(records) == len(source) and all(r.get('method') in ('sift','sift_revised','epipolar_patch') for r in records):
        g.write(RUN / 'evidence/renders_floor.json',dict(rows=records,base='093e213bd2c632c777f8213748917fec9f0d55a0',
            command='<BODY_PYTHON> -Bu render_gallery.py',exit_status=0,
            checkpoint_sha256=g.sha(checkpoint), solver_input_sha256=inp['solver_input_sha256'],
            calibration_sha256=inp['cameras'],manifest_sha256=provenance['manifest_sha256']))


if __name__ == '__main__':
    main()
