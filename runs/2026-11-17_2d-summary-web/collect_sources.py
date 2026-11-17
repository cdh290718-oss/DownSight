"""Capture primary-source table facts without publishing complete web pages."""
import hashlib
import json
import re
import urllib.request
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from html import unescape
from pathlib import Path

RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]


class Tables(HTMLParser):
    def __init__(self):
        super().__init__();self.rows=[];self.row=None;self.cell=None
    def handle_starttag(self,tag,attrs):
        if tag=='tr':self.row=[]
        if tag in ['td','th'] and self.row is not None:self.cell=[]
    def handle_data(self,data):
        if self.cell is not None:self.cell.append(data)
    def handle_endtag(self,tag):
        if tag in ['td','th'] and self.cell is not None:
            self.row.append(' '.join(''.join(self.cell).split()));self.cell=None
        if tag=='tr' and self.row is not None:
            self.rows.append(self.row);self.row=None


def main():
    local=ROOT/'outputs/2d_summary_web_20261117';local.mkdir(exist_ok=True)
    urls={'official':'https://unrealego.mpi-inf.mpg.de/',
          'preprint':'https://arxiv.org/html/2610.26641v1'}
    sources={}
    for key,url in urls.items():
        cached=local/(key+'.html')
        if cached.exists():raw=cached.read_bytes();status=200
        else:
            request=urllib.request.Request(url,headers={'User-Agent':'DownSight-research-summary/1.0'})
            with urllib.request.urlopen(request,timeout=40) as response:
                raw=response.read();status=response.status
            cached.write_bytes(raw)
        text=raw.decode()
        # The official site builds some tables from JavaScript HTML strings.
        # Extract row fragments as well as ordinary rendered-source tables.
        rows=[]
        for fragment in re.findall(r'<tr\b[^>]*>(.*?)</tr>',text,re.S):
            cells=re.findall(r'<t[dh]\b[^>]*>(.*?)</t[dh]>',fragment,re.S)
            rows.append([' '.join(unescape(re.sub(r'<[^>]*>','',c)).replace('\\n',' ').split()) for c in cells])
        sources[key]={'url':url,'http_status':status,'body_sha256':hashlib.sha256(raw).hexdigest(),
                      'rows':rows}
    official=[
        dict(method='Zhao et al. · 3DV 2021',training='UnrealEgo2 + RW',mpjpe=97.86,pa_mpjpe=69.92),
        dict(method='UnrealEgo · ECCV 2022',training='UnrealEgo2 + RW',mpjpe=92.48,pa_mpjpe=67.15),
        dict(method='Akada et al. · CVPR 2024 · T=5',training='RW only',mpjpe=104.14,pa_mpjpe=82.18),
        dict(method='Akada et al. · CVPR 2024 · T=5',training='UnrealEgo2 + RW',mpjpe=72.89,pa_mpjpe=56.19),
    ]
    for row in official:
        matches=[r for r in sources['official']['rows']
                 if f"{row['mpjpe']:.2f}" in r and f"{row['pa_mpjpe']:.2f}" in r]
        assert len(matches)==1,(row,matches)
        row.update(source=urls['official'],source_row=matches[0],status='official leaderboard',
                   locally_reproduced=False,
                   input_note='CVPR 2024 T=5 uses scene depth; inference/training differs across methods.')
    preprints=[
        dict(method='EgoRayFormer · 2026 preprint',training='RW only',mpjpe=49.55,pa_mpjpe=40.59),
        dict(method='EgoRayFormer · 2026 preprint',training='UnrealEgo2 + RW',mpjpe=47.57,pa_mpjpe=37.21),
    ]
    matches=[r for r in sources['preprint']['rows'] if r and 'EgoRayFormer' in r[0] and any('49.55' in cell for cell in r)
             and any('47.57' in cell for cell in r)]
    assert len(matches)==1,matches
    for row in preprints:
        assert any(f"{row['pa_mpjpe']:.2f}" in cell for cell in matches[0])
        row.update(source=urls['preprint'],source_row=matches[0],status='preprint; author local evaluation',
                   locally_reproduced=False,submitted_to_leaderboard=False,
                   caveats=['Older leaderboard split version is unverified against current test split.',
                            'Authors disclose preliminary test-set evaluation during development.',
                            'Calibration and causal temporal images are used; no ground-truth input at inference.'])
    evidence={'command':'python runs/2026-11-17_2d-summary-web/collect_sources.py','exit_status':0,
              'checked_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
              'sources':{k:{a:b for a,b in v.items() if a!='rows'} for k,v in sources.items()},
              'official':official,'preprint':preprints,
              'scope':'Reported literature numbers, not local reproduction or a fair ranking against the ten-pose diagnostic.'}
    dest=RUN/'evidence';dest.mkdir(exist_ok=True)
    (dest/'literature_baselines.json').write_text(json.dumps(evidence,indent=2,ensure_ascii=False)+'\n')
    print('Verified primary-source rows for',len(official)+len(preprints),'literature references.')


if __name__=='__main__':main()
