"""Read embedded protobuf schemas and all relevant MCAP messages; no third-party MCAP needed."""
import ctypes, ctypes.util, hashlib, io, json, struct, zlib
from collections import Counter, defaultdict
from pathlib import Path
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory, json_format

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/gen_head_camera_alignment_20261118'
RUN=Path(__file__).resolve().parents[1]
SRC=ROOT/'data/0d7da991acf44b46ba41710a8960209c.mcap'

def string(b,i):
    n=struct.unpack_from('<I',b,i)[0]
    return b[i+4:i+4+n].decode(),i+4+n

def records(f,end):
    while f.tell()<end:
        h=f.read(9)
        assert len(h)==9
        op,n=struct.unpack('<BQ',h);b=f.read(n)
        assert len(b)==n
        yield op,b

lib=ctypes.CDLL(ctypes.util.find_library('zstd'))
lib.ZSTD_decompress.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t]
lib.ZSTD_decompress.restype=ctypes.c_size_t
lib.ZSTD_isError.argtypes=[ctypes.c_size_t];lib.ZSTD_isError.restype=ctypes.c_uint

def unpack(b):
    size,crc=struct.unpack_from('<QI',b,16);comp,k=string(b,28)
    n=struct.unpack_from('<Q',b,k)[0];z=b[k+8:k+8+n]
    if comp=='zstd':
        dst=ctypes.create_string_buffer(size);n=lib.ZSTD_decompress(dst,size,z,len(z));assert not lib.ZSTD_isError(n) and n==size
        plain=dst.raw
    else:
        assert comp=='';plain=z
    assert len(plain)==size
    if crc:assert zlib.crc32(plain)==crc
    return plain

def main():
    classes={};channels={};rows=defaultdict(list);counts=Counter();chunks=0
    with SRC.open('rb') as f:
        assert f.read(8)==b'\x89MCAP0\r\n'
        f.seek(-37,2);footer=f.read(37);start=struct.unpack_from('<Q',footer,9)[0];f.seek(start)
        for op,b in records(f,SRC.stat().st_size-37):
            if op==3:
                sid=struct.unpack_from('<H',b)[0];name,k=string(b,2);enc,k=string(b,k);assert enc=='protobuf'
                n=struct.unpack_from('<I',b,k)[0];ds=descriptor_pb2.FileDescriptorSet();ds.ParseFromString(b[k+4:k+4+n]);pool=descriptor_pool.DescriptorPool();pending=list(ds.file)
                while pending:
                    remain=[]
                    for desc in pending:
                        try:pool.Add(desc)
                        except Exception:remain.append(desc)
                    assert len(remain)<len(pending);pending=remain
                classes[sid]=message_factory.GetMessageClass(pool.FindMessageTypeByName(name))
            elif op==4:
                cid,sid=struct.unpack_from('<HH',b);topic,k=string(b,4);channels[cid]={'schema_id':sid,'topic':topic}
        f.seek(8)
        def message(b):
            cid,seq,log,pub=struct.unpack_from('<HIQQ',b);idx=counts[cid];counts[cid]+=1
            sid=channels[cid]['schema_id']
            if sid not in (1,2,4,6,7,9,8,11):return
            o=classes[sid]();o.ParseFromString(b[22:])
            if sid==1 and cid not in (1,13):return
            entry={'index':idx,'sequence':seq,'log_ns':log,'publish_ns':pub}
            if sid==1:
                entry.update(timestamp_ns=o.timestamp.seconds*10**9+o.timestamp.nanos,frame_id=o.frame_id,format=o.format)
                if idx in (0,270,5130):
                    (OUT/f'image_{cid}_{idx:05d}.{o.format}').write_bytes(o.data)
            else:entry['message']=json_format.MessageToDict(o,preserving_proto_field_name=True)
            rows[cid].append(entry)
        for op,b in records(f,start):
            if op==6:
                plain=unpack(b);chunks+=1
                for t,p in records(io.BytesIO(plain),len(plain)):
                    if t==5:message(p)
            elif op==5:message(b)
    for cid,items in rows.items():(OUT/f'channel_{cid}.json').write_text(json.dumps(items))
    report={'source_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'source_bytes':SRC.stat().st_size,'chunks_read_and_crc_checked_if_present':chunks,'channels':{str(cid):dict(v,count=counts[cid]) for cid,v in channels.items()},'command':'<PYTHON_ENV>/bin/python runs/2026-11-18_gen_head_camera_alignment/code/extract.py','exit_status':0}
    (RUN/'evidence/extraction.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
