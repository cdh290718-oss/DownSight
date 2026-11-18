import struct,json,hashlib
from pathlib import Path
p=Path('data/0d7da991acf44b46ba41710a8960209c.mcap');f=p.open('rb');f.seek(-37,2);v=f.read(37);assert v[0]==2 and struct.unpack_from('<Q',v,1)[0]==20;start,end,crc=struct.unpack_from('<QQI',v,9);f.seek(start);schemas={};channels={};ops={}
def string(b,i):
 n=struct.unpack_from('<I',b,i)[0];return b[i+4:i+4+n].decode('utf8','replace'),i+4+n
while f.tell()<p.stat().st_size-37:
 hdr=f.read(9)
 if len(hdr)<9:break
 op,n=struct.unpack('<BQ',hdr);b=f.read(n);ops[str(op)]=ops.get(str(op),0)+1
 if op==3:
  id=struct.unpack_from('<H',b)[0];name,i=string(b,2);encoding,i=string(b,i);size=struct.unpack_from('<I',b,i)[0];schema=b[i+4:i+4+size];schemas[id]={'name':name,'encoding':encoding,'schema':schema.decode('utf8','replace')}
 if op==4:
  id,sid=struct.unpack_from('<HH',b);topic,i=string(b,4);encoding,i=string(b,i);channels[id]={'schema_id':sid,'topic':topic,'message_encoding':encoding}
print(json.dumps({'file_bytes':p.stat().st_size,'summary_start':start,'summary_offset_start':end,'operations':ops,'channels':channels,'schemas':schemas},ensure_ascii=False,indent=2))
