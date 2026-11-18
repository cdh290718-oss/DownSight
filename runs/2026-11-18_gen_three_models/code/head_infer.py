import json,os,subprocess,sys
from pathlib import Path
root=Path.cwd();work=root.parent;back=work/'EgoSMPLX';data=work/'gen_egosmplx_run';output=root/'outputs/gen_three_models_20261118/head_network';output.mkdir(exist_ok=False);frames=json.loads((data/'frames.json').read_text())['frames'];existing=set(json.loads((root/'runs/2026-11-16_gen-body-only-20/evidence/selected_manifest.json').read_text())['selected_frame_ids']);cfg=back/'main/config/config_ft_egopw_sceneego_egowholebody_unrealego_egofishbody.py';ckpt=back/'output/train_5datasets_8gpu_camtrans_w10_12ep_fixed_20260809_20260809_163018/model_dump/snapshot_11.pth.tar';env=dict(os.environ,PYTHONPATH=str(root),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='0',EGOSMPLX_REPO_ROOT=str(back),EGOSMPLX_CONFIG=str(cfg),TMPDIR=str(output),XDG_CACHE_HOME=str(output));python=work/'miniconda3/envs/egosmplx/bin/python3.8'
for row in frames:
 if row['camera']!='cam3' or int(row['id'].split('_')[1]) in existing:continue
 image=data/row['image'];dest=output/row['id'];log=output/(row['id']+'.log');cmd=[str(python),'-m','egosmplx_pipeline.egosmplx_predict','--image-dir',str(image.parent),'--image-glob',image.name,'--fisheye-calibration',str(data/row['calibration']),'--config-path',str(cfg),'--checkpoint-path',str(ckpt),'--output-dir',str(dest),'--smplx-input-mode','det_tangent','--input-crop-mode','none'];print('starting',row['id'],flush=True)
 with log.open('x') as f:proc=subprocess.run(cmd,cwd=back,env=env,stdout=f,stderr=subprocess.STDOUT)
 if proc.returncode:raise RuntimeError(f'{row["id"]}: exit {proc.returncode}; see local log')
 print('done',row['id'],flush=True)
