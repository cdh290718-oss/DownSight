import struct,json,zstandard
from pathlib import Path
p=Path('data/0d7da991acf44b46ba41710a8960209c.mcap');out=Path('outputs/gen_mcap_head_audit_20261118');out.mkdir(exist_ok=False);f=p.open('rb');f.seek(-37,2);s=struct.unpack_from('<Q',f.read(37),9)[0];f.seek(s)
def st(b,i):
 n=struct.unpack_from('<I',b,i)[0];return b[i+4:i+4+n].decode(),i+4+n
while f.tell()<p.stat().st_size-37:
 h=f.read(9)
 if len(h)<9:break
 op,n=struct.unpack('<BQ',h);b=f.read(n)
 if op==3:
  i=struct.unpack_from('<H',b)[0];name,k=st(b,2);enc,k=st(b,k);size=struct.unpack_from('<I',b,k)[0];(out/f'schema_{i}.bin').write_bytes(b[k+4:k+4+size])
f.seek(8);wanted={2,6,11,12,14};found={};chunks=0
while f.tell()<s and len(found)<len(wanted):
 h=f.read(9)
 if len(h)<9:break
 op,n=struct.unpack('<BQ',h);b=f.read(n)
 if op!=6:continue
 length=struct.unpack_from('<I',b,28)[0];comp=b[32:32+length].decode();k=32+length;size=struct.unpack_from('<Q',b,k)[0];z=b[k+8:k+8+size];plain=zstandard.ZstdDecompressor().decompress(z,max_output_size=struct.unpack_from('<Q',b,16)[0]) if comp=='zstd' else z;chunks+=1;i=0
 while i+9<=len(plain):
  typ,L=struct.unpack_from('<BQ',plain,i);payload=plain[i+9:i+9+L];i+=9+L
  if typ!=5:continue
  channel,seq,logtime,pubtime=struct.unpack_from('<HIQQ',payload);raw=payload[22:]
  if channel in wanted and channel not in found:
   (out/f'message_{channel}.bin').write_bytes(raw);found[channel]={'log_time_ns':logtime,'publish_time_ns':pubtime,'length':len(raw),'sequence':seq}
print(json.dumps({'chunks_read':chunks,'selected':found},indent=2));(out/'sample_metadata.json').write_text(json.dumps(found,indent=2))
