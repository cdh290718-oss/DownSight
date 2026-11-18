"""Read-only split provenance and exact-byte fine-tuning overlap check."""
import os,json,hashlib,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent;OUT=ROOT/'outputs/cross_dataset100_20261118'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
root=Path(os.environ['SCENEEGO_ROOT']);out={}
with zipfile.ZipFile(root/'SceneEgo_test.zip') as z:
 for seq in ['jian1','jian2','diogo1','diogo2']:
  key=seq+'/annotation.pkl';b=z.read(key);out[seq]={'test_zip_annotation_matches_local':b==(root/key).read_bytes(),'sha256':hashlib.sha256(b).hexdigest()};assert out[seq]['test_zip_annotation_matches_local']
p=Path(os.environ['EGOPW_ROOT']);test=set((p/'test_sequences.txt').read_text().split());train=set((p/'training_seqences.txt').read_text().split());assert not train&test
out['egopw']={'test_directories':sorted(test),'train_test_directory_intersection':sorted(train&test),'test_list_sha256':sha(p/'test_sequences.txt'),'train_list_sha256':sha(p/'training_seqences.txt')}
(RUN/'evidence/split_checks.json').write_text(json.dumps(out,indent=2)+'\n')
trainfiles={'rw500':ROOT/'outputs/sapiens_rw500_prep_20261116/dataset/train.json','rw5000':ROOT/'outputs/rw_a5000_dataset_20261117/train.json','rw20000':ROOT/'outputs/rw_a20000_dataset_20261117/train.json'}
evals={d:json.loads((OUT/(d+'_manifest.json')).read_text()) for d in ['egopw','sceneego','unrealego2']};record={}
for model,p in trainfiles.items():
 d=json.loads(p.read_text())['records'];h={r['image_sha256'] for r in d};record[model]={'training_images':len(d),'train_manifest_sha256':sha(p),'evaluation_image_hash_overlap':{name:sum(r['image_sha256'] in h for r in rows) for name,rows in evals.items()}}
 assert not any(record[model]['evaluation_image_hash_overlap'].values())
(RUN/'evidence/finetune_overlap.json').write_text(json.dumps({'command':'EGOPW_ROOT=<DATA_DISK>/EgoPW SCENEEGO_ROOT=<DATA_DISK>/SceneEgo python runs/2026-11-18_cross_dataset100/check_inputs.py','exit_status':0,'checks':record,'scope':'exact encoded image hashes in local fine-tuning manifests; not a perceptual-duplicate or upstream-pretraining audit'},indent=2)+'\n');print('Split provenance and fine-tuning image hash checks passed')
