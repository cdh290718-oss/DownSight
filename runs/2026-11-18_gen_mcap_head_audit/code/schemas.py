import struct,json
from pathlib import Path
from google.protobuf import descriptor_pb2
p=Path('data/0d7da991acf44b46ba41710a8960209c.mcap');f=p.open('rb');f.seek(-37,2);v=f.read(37);start=struct.unpack_from('<Q',v,9)[0];f.seek(start);res={}
def string(b,i):
 n=struct.unpack_from('<I',b,i)[0];return b[i+4:i+4+n].decode(),i+4+n
def msg(m,prefix=''):
 return {'name':prefix+m.name,'fields':[{'name':x.name,'type':x.type,'type_name':x.type_name,'label':x.label} for x in m.field],'nested':[msg(x,prefix+m.name+'.') for x in m.nested_type]}
while f.tell()<p.stat().st_size-37:
 h=f.read(9)
 if len(h)<9:break
 op,n=struct.unpack('<BQ',h);b=f.read(n)
 if op!=3:continue
 id=struct.unpack_from('<H',b)[0];name,i=string(b,2);encoding,i=string(b,i);size=struct.unpack_from('<I',b,i)[0];s=b[i+4:i+4+size];ds=descriptor_pb2.FileDescriptorSet();ds.ParseFromString(s);res[str(id)]={'schema_name':name,'encoding':encoding,'files':[{'name':d.name,'package':d.package,'messages':[msg(m) for m in d.message_type]} for d in ds.file]}
Path('outputs/gen_mcap_schemas_20261118.json').write_text(json.dumps(res,indent=2));
for i in ['2','4','9','10','11']:
 print('\n',i,res[i]['schema_name']);
 for f in res[i]['files']:
  for m in f['messages']:
   print(m['name'],[(x['name'],x['type_name']) for x in m['fields']])
