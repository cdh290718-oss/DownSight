import argparse, collections, json, tarfile, zipfile
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--archive',required=True); p.add_argument('--splits',required=True); p.add_argument('--output',required=True); a=p.parse_args()
with zipfile.ZipFile(a.splits) as z:
    splits={k:set(z.read('UnrealEgoData_rw/'+k+'.txt').decode().splitlines()) for k in ('train','validation','test')}
counts=collections.Counter(); examples={}; error=None
try:
    with tarfile.open(a.archive,'r|gz') as tar:
        for m in tar:
            if m.isfile() and '/fisheye_final_image/' in m.name:
                seq=m.name.split('/fisheye_final_image/')[0]
                counts[seq]+=1
                examples.setdefault(seq,m.name)
except (EOFError,tarfile.ReadError,OSError) as e:
    error=type(e).__name__+': '+str(e)
rows=[dict(sequence=s,split=next((k for k,v in splits.items() if s in v),'unknown'),images=n,example=examples[s]) for s,n in counts.items()]
result=dict(command='index_archive.py --archive <RGB_PART> --splits <OFFICIAL_SPLITS> --output <EVIDENCE>',exit_status=0,archive_complete=error is None,archive_error=error,sequences=rows,images_by_split={k:sum(r['images'] for r in rows if r['split']==k) for k in ('train','validation','test','unknown')})
Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
