"""Verify the canonical Pages URL and retain all propagation attempts."""
import hashlib
import json
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

RUN=Path(__file__).resolve().parent
URL='https://cdh290718-oss.github.io/egofishpose-demo/2d-to-3d-summary-20261117/'


def main():
    evidence=RUN/'evidence';history=evidence/'online_attempts.json'
    attempts=json.loads(history.read_text()) if history.exists() else [
        json.loads((evidence/p).read_text()) for p in ['online_initial.json','online_check.json']
        if (evidence/p).exists()]
    request=urllib.request.Request(URL,headers={'User-Agent':'Mozilla/5.0','Cache-Control':'no-cache'})
    current=dict(command='python runs/2026-11-17_2d-summary-web/verify_online.py',url=URL,
                 checked_at=datetime.now(timezone(timedelta(hours=8))).isoformat())
    try:
        with urllib.request.urlopen(request,timeout=30) as r:raw=r.read();status=r.status
        current.update(http_status=status,body_sha256=hashlib.sha256(raw).hexdigest(),
                       matches_local=raw==(RUN/'site/dist/index.html').read_bytes(),
                       title_present='2D 更准，3D 会全身更好吗？' in raw.decode())
        current['exit_status']=0 if status==200 and current['matches_local'] and current['title_present'] else 1
    except (urllib.error.HTTPError,urllib.error.URLError,TimeoutError) as error:
        current.update(http_status=getattr(error,'code',None),error=str(error),exit_status=1)
    attempts.append(current);history.write_text(json.dumps(attempts,ensure_ascii=False,indent=2)+'\n')
    (evidence/'online_verified.json').write_text(json.dumps(current,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(current,ensure_ascii=False))
    raise SystemExit(current['exit_status'])


if __name__=='__main__':main()
