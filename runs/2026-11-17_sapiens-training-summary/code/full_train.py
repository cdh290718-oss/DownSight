"""Local sparse-label adaptation; use official Sapiens architecture and transforms.
No external checkout or environment is modified. --check performs CPU data checks only.
"""
import argparse, hashlib, json, math, os, random, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
import torch

RUN=Path(__file__).resolve().parents[1]
ROOT=RUN.parents[1]

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def write(p,v):
    p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--check',action='store_true');p.add_argument('--confirmed',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
    if not a.check and not a.confirmed: p.error('Training requires separate user confirmation and --confirmed')
    c=json.loads((RUN/'config.json').read_text());backend=a.workspace.resolve()/'EgoSMPLX';repo=backend/'official_baseline/sapiens2';checkpoint=backend/'pretrained_models/external_pose/sapiens2_1b_pose.safetensors'
    sys.path[:0]=[str(backend/'.runtime/sapiens_deps'),str(repo)]
    import cv2
    from sapiens.engine.config import Config
    from sapiens.engine.datasets import Compose
    from sapiens.registry import MODELS
    from sapiens.pose.datasets import parse_pose_metainfo
    from sapiens.pose.datasets.codecs.udp_heatmap import UDPHeatmap
    from safetensors.torch import load_file, save_file
    cfg=Config.fromfile(repo/'sapiens/pose/configs/keypoints308/shutterstock_goliath_3po/sapiens2_1b_keypoints308_shutterstock_goliath_3po-1024x768.py')
    meta=parse_pose_metainfo(dict(from_file=str(repo/'sapiens/pose/configs/_base_/keypoints308.py')))
    codec=UDPHeatmap(**{k:v for k,v in cfg.codec.items() if k!='type'})
    transform=Compose(cfg.test_pipeline[:-1]+[dict(type='PoseGenerateTarget',encoder=cfg.codec),cfg.test_pipeline[-1]])
    pp=MODELS.build(cfg.data_preprocessor)
    mapping=json.loads((RUN.parent/'2026-11-16_sapiens-unrealego-audit/evidence/audit.json').read_text())['joint_mapping']
    joint_ids=[m['sapiens308_id'] for m in mapping if m['rw_index'] is not None]
    manifests={s:ROOT/c[k] for s,k in [('train','train_manifest'),('validation','validation_manifest')]}
    rows={s:json.loads(f.read_text())['records'] for s,f in manifests.items()}
    for s,rr in rows.items():
        assert len(rr)==c[s+'_images']
        for r in rr:
            assert r['split']==s
            if s=='validation':
                # Original inspection manifest stays immutable; activate only mapped in-image GT here.
                r['keypoints308_supervision_mask']=[int(j in joint_ids and v) for j,v in enumerate(r['keypoints308_candidate_in_image'])]
            assert r['keypoints308_supervision_mask'][0]==0
            assert all(not v or j in joint_ids for j,v in enumerate(r['keypoints308_supervision_mask']))
    assert not {r['sequence'] for r in rows['train']}&{r['sequence'] for r in rows['validation']}
    assert not {r['pair_id'] for r in rows['train']}&{r['pair_id'] for r in rows['validation']}
    assert not {r['image_sha256'] for r in rows['train']}&{r['image_sha256'] for r in rows['validation']}

    def sample(s,r):
        im=cv2.imread(str(manifests[s].parent/r['image'])); assert im is not None
        w,h=r['size'];assert im.shape[:2]==(h,w)
        return transform(dict(img=im,bbox=np.array([[0,0,w,h]],np.float32),bbox_score=np.ones(1,np.float32),keypoints=np.array([r['keypoints308_xy']],np.float32),keypoints_visible=np.array([r['keypoints308_supervision_mask']],np.float32)))

    if a.check:
        inspected=[]
        for s,rr in rows.items():
            for r in rr:assert digest(manifests[s].parent/r['image'])==r['image_sha256']
            # Exercise both cameras in every sequence through the exact CPU target pipeline.
            seen=set()
            for r in rr:
                key=(r['sequence'],r['view'])
                if key in seen:continue
                seen.add(key);d=pp(sample(s,r));ds=d['data_samples'];weight=ds['keypoint_weights'];hm=ds['heatmaps']
                assert tuple(d['inputs'].shape)==(1,3,1024,768)
                assert tuple(hm.shape)==(308,256,192)
                assert float(weight[...,0].sum())==0 and float(weight.sum())>0
                assert bool(torch.isfinite(d['inputs']).all())
                # No model, optimizer, backward pass, CUDA allocation, or training step.
                inspected.append(dict(split=s,sequence=r['sequence'],view=r['view'],valid_targets=int(weight.sum()),input_shape=list(d['inputs'].shape),heatmap_shape=list(hm.shape)))
        write(RUN/'evidence/preflight.json',dict(command='train.py --workspace <WORKSPACE> --check',exit_status=0,images_hash_verified=sum(map(len,rows.values())),cpu_pipeline_samples=inspected,training_started=False,gpu_model_executed=False,torch=torch.__version__,config_sha256=digest(RUN/'config.json')))
        print('CPU checks passed; no training executed',flush=True);return

    assert a.output is not None
    pre=json.loads((RUN/'evidence/preflight.json').read_text());assert pre['config_sha256']==digest(RUN/'config.json')
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==c['sapiens_commit']
    assert digest(checkpoint)==c['checkpoint_sha256']
    for s,rr in rows.items():
        for r in rr:assert digest(manifests[s].parent/r['image'])==r['image_sha256']
    out=a.output.resolve();assert out.is_relative_to(ROOT/'outputs');out.mkdir(exist_ok=False,parents=True)
    write(out/'config.json',c)
    random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed'])
    cfg.model['backbone'].pop('init_cfg',None)
    cfg.model['backbone']['frozen_stages']=-1
    model=MODELS.build(cfg.model);model.load_state_dict(load_file(str(checkpoint)),strict=True)
    assert len(model.backbone.blocks)==40
    for param in model.parameters():param.requires_grad_(True)
    # All frozen blocks run in eval mode, head and final blocks train.
    trainable=[n for n,p in model.named_parameters() if p.requires_grad]
    write(out/'trainable.json',dict(names=trainable,parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),total_parameters=sum(p.numel() for p in model.parameters())))
    model.cuda();pp.cuda()
    optimizer=torch.optim.AdamW([dict(params=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('decode_head.')],lr=c['head_lr']),dict(params=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('backbone.')],lr=c['backbone_lr'])],weight_decay=c['weight_decay'])
    steps_per_epoch=math.ceil(len(rows['train'])/c['gradient_accumulation']);total_steps=steps_per_epoch*c['epochs']
    def multiplier(step):return min(1.,(step+1)/c['warmup_steps'])*max(0.,(total_steps-step)/max(1,total_steps-c['warmup_steps']))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,multiplier)
    t0=time.time();step=0
    def evaluate(epoch):
        model.eval();errs={j:[] for j in joint_ids};correct=[];fail=0;raw=[]
        with torch.inference_mode():
            for r in rows['validation']:
                d=pp(sample('validation',r));x=d['inputs']
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    hm=model(x)
                    if cfg.val_cfg.get('flip_test',False):hm=(hm+model(x.flip(-1)).flip(-1)[:,meta['flip_indices']])/2
                xy,score=codec.decode(hm.float().cpu().numpy()[0]);m=d['data_samples']['meta']
                cv=lambda v:v.detach().cpu().numpy() if torch.is_tensor(v) else np.array(v)
                xy=(xy/cv(m['input_size'])*cv(m['bbox_scale'])+cv(m['bbox_center'])-.5*cv(m['bbox_scale']))[0]
                gt=np.array(r['keypoints308_xy']);mask=np.array(r['keypoints308_supervision_mask'],bool);e=np.linalg.norm(xy-gt,axis=-1);finite=np.isfinite(e)
                fail+=int(not finite[mask].all());correct.append(int(((e<=c['correct_threshold_px'])&mask&finite).sum()))
                for j in joint_ids:
                    if mask[j] and finite[j]:errs[j].append(float(e[j]))
                raw.append(dict(id=r['id'],xy=xy.tolist(),scores=score.tolist(),mask=mask.tolist()))
        allerr=[v for vv in errs.values() for v in vv]
        result=dict(epoch=epoch,mean_px=float(np.mean(allerr)) if allerr else None,per_joint_px={str(j):float(np.mean(v)) if v else None for j,v in errs.items()},mean_correct_points=float(np.mean(correct)),threshold_px=c['correct_threshold_px'],failed_images=fail,images=len(rows['validation']),valid_evaluated_points=len(allerr),elapsed_seconds=time.time()-t0)
        # Raw predictions stay only in ignored local outputs.
        write(out/f'eval_{epoch:02d}.json',result);write(out/f'predictions_{epoch:02d}.json',raw)
        print(json.dumps(result),flush=True);return result
    try:
        base=evaluate(0);best=base['mean_px'] if base['failed_images']==0 and base['mean_px'] is not None else math.inf;best_epoch=0
        for epoch in range(1,c['epochs']+1):
            model.train()
            order=list(rows['train']);random.shuffle(order);optimizer.zero_grad(set_to_none=True);loss_sum=0.
            for i,r in enumerate(order):
                d=pp(sample('train',r));ds=d['data_samples']
                target=torch.as_tensor(ds['heatmaps'],device=d['inputs'].device,dtype=torch.float32)[None]
                weight=torch.as_tensor(ds['keypoint_weights'],device=d['inputs'].device,dtype=torch.float32).reshape(1,308,1,1)
                group_size=min(c['gradient_accumulation'],len(order)-(i//c['gradient_accumulation'])*c['gradient_accumulation'])
                with torch.autocast('cuda',dtype=torch.bfloat16):hm=model(d['inputs'])
                loss=((hm.float()-target.float()).square()*weight).sum()/(weight.sum()*hm.shape[-1]*hm.shape[-2]).clamp_min(1.)
                if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
                (loss/group_size).backward();loss_sum+=float(loss.detach())
                if (i+1)%c['gradient_accumulation']==0 or i+1==len(order):
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],c['clip_grad_norm'],error_if_nonfinite=True)
                    optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);step+=1
                    if step%10==0:
                        status=dict(state='training',epoch=epoch,step=step,total_steps=total_steps,loss=float(loss.detach()),elapsed_seconds=time.time()-t0,peak_gpu_bytes=torch.cuda.max_memory_allocated())
                        write(out/'status.json',status);print(json.dumps(status),flush=True)
            metric=evaluate(epoch)
            # Epoch files unique; saves only trainable parameter delta, loaded on exact base checkpoint.
            adapter={n:p.detach().cpu().contiguous() for n,p in model.named_parameters() if p.requires_grad}
            adapter.update({n:b.detach().cpu().contiguous() for n,b in model.named_buffers() if n.startswith('decode_head.')})
            save_file(adapter,str(out/f'adapter_{epoch:02d}.safetensors'))
            del adapter
            if metric['failed_images']==0 and metric['mean_px'] is not None and metric['mean_px']<best:best=metric['mean_px'];best_epoch=epoch
            write(out/'best.json',dict(epoch=best_epoch,mean_px=best if math.isfinite(best) else None,base_checkpoint_sha256=c['checkpoint_sha256'],note='epoch 0 means original checkpoint wins; adapter contains replacement parameters and head buffers; load exact base strictly then overlay adapter with strict=False'))
            torch.save(dict(optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,head_buffers={n:b.detach().cpu() for n,b in model.named_buffers() if n.startswith('decode_head.')}),out/'last_optimizer.pt')
        write(out/'status.json',dict(state='completed',best_epoch=best_epoch,elapsed_seconds=time.time()-t0,step=step))
    except BaseException as e:
        write(out/'status.json',dict(state='failed',error=repr(e),elapsed_seconds=time.time()-t0,step=step));raise

if __name__=='__main__':main()
