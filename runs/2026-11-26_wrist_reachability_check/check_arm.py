"""Check wrist-aligned RMSE and 3D arm reachability for purple translation."""
import json, sys, numpy as np, torch
sys.path.insert(0, 'runs/2026-11-26_realvideo4_wilor_outline')
import paired

print("=== Part 1: Wrist-aligned RMSE ===")
for clip in ['clip003', 'clip004']:
    purp = json.load(open(f'runs/2026-11-26_realvideo34_purple_hand_a/evidence/{clip}.json'))['results']['control']['hands']
    orig = json.load(open(f'runs/2026-11-26_realvideo4_wilor_outline/evidence/{clip}.json'))['results']['control']['hands']
    for name, h in [('orig_A', orig), ('purple_A', purp)]:
        w_errs, unaligned, aligned = [], [], []
        for side in ['left', 'right']:
            uv = np.array(h[side]['uv'])
            tg = np.array(h[side]['target'])
            dw = uv[:, 0:1, :] - tg[:, 0:1, :]
            w_errs.append(np.sum(dw[:, 0]**2, axis=-1))
            unaligned.append(np.sum((uv[:, 5:10] - tg[:, 5:10])**2, axis=-1))
            aligned.append(np.sum(((uv[:, 5:10] - dw) - tg[:, 5:10])**2, axis=-1))
        print(f"  {clip} {name:10s}  wrist={np.sqrt(np.mean(w_errs)):.2f}  tips={np.sqrt(np.mean(unaligned)):.2f}  wrist_aligned_tips={np.sqrt(np.mean(aligned)):.2f}")

print("\n=== Part 2: 3D arm reachability ===")
ref = paired.ref
ref.prev.ref.camera_files = ref.prev.base.fit.camera_files = ref.cameras
model, cam = ref.prev.ref.common.setup_model()
cfg = ref.read(ref.OLD / 'fit/0/protocol.json')
prep = ref.read(ref.OLD / 'fit/0/prepared.json')['items']
render = json.load(open('runs/2026-11-26_realvideo4_pitch_obj/evidence/render.json'))['rows']

from smplx.joint_names import JOINT_NAMES
idx_map = {n: JOINT_NAMES.index(n) for n in [
    'head', 'left_collar', 'right_collar',
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist']}

for idx, clip in [(2, 'clip003'), (3, 'clip004')]:
    pack = dict(np.load(f'outputs/realvideo4_wilor_outline_20261126/poses/{clip}_control.npz'))
    x0 = torch.from_numpy(pack['locked'])
    start = dict(body_pose=x0[:21].numpy(), delta=x0[21].numpy(),
                 translation=x0[22].numpy(), betas=pack['betas'], global0=pack['global0'])
    frame = ref.FixedFeet(clip, prep[idx], start, model, cam, cfg)
    with torch.no_grad():
        _, j_orig = paired.body(frame, x0)
    t_purp = torch.tensor(render[clip]['Purple']['translation_m'], dtype=x0.dtype)
    j_purp = j_orig + t_purp
    print(f"\n  {clip}  |t_purp|={t_purp.norm().item()*100:.2f} cm  t={[round(x,4) for x in t_purp.tolist()]}")
    for side in ['left', 'right']:
        sh_o = j_orig[idx_map[side + '_shoulder']]
        el_o = j_orig[idx_map[side + '_elbow']]
        wr_o = j_orig[idx_map[side + '_wrist']]
        sh_p = j_purp[idx_map[side + '_shoulder']]
        u_len = (el_o - sh_o).norm().item()
        f_len = (wr_o - el_o).norm().item()
        d_orig = (wr_o - sh_o).norm().item()
        d_req = (wr_o - sh_p).norm().item()
        print(f"    {side:5s}: upper={u_len*100:.2f}cm  fore={f_len*100:.2f}cm  sum={( u_len+f_len)*100:.2f}cm | "
              f"orig_sh->wr={d_orig*100:.2f}cm  purp_sh->orig_wr={d_req*100:.2f}cm  margin={( u_len+f_len-d_req)*100:+.2f}cm")

print("\nDone.")
