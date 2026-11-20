"""Build the visual index and a checked archive of the individual rendered views."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import hashlib
import json
import zipfile

RUN = Path(__file__).resolve().parent
OUT = RUN.parents[1]/'outputs/rw20000_floor12_20261120'
ARCHIVE = 'RW20000_floor12_front_side.zip'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    results = json.loads((RUN/'evidence/renders_floor.json').read_text())
    sheet = Image.new('RGB',(1400,3*284),(244,244,244))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype('DejaVuSans.ttf',18)
    members = {}
    with zipfile.ZipFile(OUT/ARCHIVE,'w',zipfile.ZIP_DEFLATED) as arch:
        for row in results['rows']:
            p = row['pair']
            for name, expected in row['images'].items():
                raw = (RUN/'assets'/p/name).read_bytes()
                assert digest(raw) == expected
                path = row['subject']+'/'+p+'/'+name
                arch.writestr(path,raw)
                members[path]=expected
        for i,subject in enumerate(['S1','S9','S11']):
            rows=sorted((r for r in results['rows'] if r['subject']==subject),key=lambda r:r['pair'])
            assert len(rows)==4
            for k,r in enumerate(rows):
                x,y=k*350,i*284
                text=subject+' / '+r['pair']
                if not r['validation']['accepted']:
                    text+='  (tentative)'
                draw.text((x+8,y+7),text,fill=(52,52,52),font=font)
                for n,side in enumerate(['front','side']):
                    with Image.open(RUN/'assets'/r['pair']/(side+'.jpg')) as im:
                        sheet.paste(im.resize((175,240),Image.Resampling.LANCZOS),(x+n*175,y+38))
    sheet.save(RUN/'assets/overview.jpg',quality=92)
    with zipfile.ZipFile(OUT/ARCHIVE) as arch:
        assert len(members)==24 and len(arch.namelist())==24 and arch.testzip() is None
        assert all(digest(arch.read(k))==v for k,v in members.items())
    f=OUT/ARCHIVE
    record=dict(archive=ARCHIVE,tag='rw20000-floor12-views-20261120',files_sha256={ARCHIVE:digest(f.read_bytes())},
                bytes=f.stat().st_size,images=members,overview_sha256=digest((RUN/'assets/overview.jpg').read_bytes()),
                command='python -B package_views.py',exit_status=0)
    (RUN/'evidence/attachment_audit.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':
    main()
