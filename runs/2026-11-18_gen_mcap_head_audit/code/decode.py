import json,math
from pathlib import Path
from google.protobuf import descriptor_pb2,descriptor_pool,message_factory
p=Path('outputs/gen_mcap_head_audit_20261118');result={}
for ch,sid in [(2,2),(14,2),(6,4),(11,9),(12,9)]:
 ds=descriptor_pb2.FileDescriptorSet();ds.ParseFromString((p/f'schema_{sid}.bin').read_bytes());pool=descriptor_pool.DescriptorPool();remaining=list(ds.file)
 while remaining:
  pending=[];progress=False
  for desc in remaining:
   try:pool.Add(desc);progress=True
   except Exception:pending.append(desc)
  assert progress,[x.name for x in pending];remaining=pending
 name='foxglove.PosesInFrame' if ch==6 else 'foxglove.CameraCalibration' if ch in (2,14) else 'foxglove.PoseInFrame';cls=message_factory.GetMessageClass(pool.FindMessageTypeByName(name));obj=cls();obj.ParseFromString((p/f'message_{ch}.bin').read_bytes())
 if ch==6:
  poses=[]
  for i,x in enumerate(obj.poses):
   q=x.orientation;pos=x.position;poses.append({'index':i,'position':[pos.x,pos.y,pos.z],'orientation':[q.x,q.y,q.z,q.w],'quaternion_norm':math.sqrt(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w),'orientation_present':x.HasField('orientation')})
  result[str(ch)]={'type':name,'frame_id':obj.frame_id,'poses':poses}
 elif ch in (11,12):
  x=obj.pose;q=x.orientation;v=x.position;result[str(ch)]={'type':name,'frame_id':obj.frame_id,'position':[v.x,v.y,v.z],'orientation':[q.x,q.y,q.z,q.w],'orientation_present':x.HasField('orientation'),'quaternion_norm':math.sqrt(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w)}
 else:result[str(ch)]={'type':name,'frame_id':obj.frame_id,'T_b_c':list(obj.T_b_c),'width':obj.width,'height':obj.height}
(p/'decoded.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
