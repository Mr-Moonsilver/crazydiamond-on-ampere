"""Real CPU BF16 snapshots/byte oracle; original completion lease contract stays mandatory."""
from dataclasses import replace
import hashlib,os,sys,types,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'patches/s5_engine' if (ROOT/'patches/s5_engine').is_dir() else Path(__file__).resolve().parent/'e3_builder'))
import native_mm as mm
from e3_types import TransactionId
class Multi(unittest.TestCase):
 def setUp(self):
  import torch
  self.torch=torch;self.flag=patch.dict(os.environ,FLASH_S5_NATIVE_MM='1');self.flag.start();self.addCleanup(self.flag.stop)
 def request(self,n=4):
  t=self.torch;features=[]
  for i in range(n):
   tensor=t.full((1024,2560),float(i+1),dtype=t.bfloat16);grid=t.tensor([1,64,64],dtype=t.int64)
   features.append(types.SimpleNamespace(modality='image',identifier='image-'+str(i),mm_position=types.SimpleNamespace(offset=1536+i*1026,length=1024,is_embed=None),data={'image_embeds':types.SimpleNamespace(data=tensor),'image_grid_thw':types.SimpleNamespace(data=grid)}))
  return types.SimpleNamespace(request_id='four-images',num_prompt_tokens=max(7000,1536+n*1026+8),mm_features=features)
 def test_four_images_two_chunk_overlap_bit_equal_and_legacy_single_identity(self):
  r=self.request();plan=mm.request_plan(r);self.assertEqual(len(plan.positions),4096);self.assertEqual(len(plan.data),20*2**20)
  expected={f.mm_position.offset+i:f.data['image_embeds'].data[i].view(self.torch.uint8).numpy().tobytes() for f in r.mm_features for i in range(1024)}
  reconstructed={};maximum=0
  for seq,start in enumerate(range(0,7000,2048),1):
   tx=TransactionId(seq,r.request_id,1,start,min(start+2048,7000),1);chunk=plan.chunk(tx,1)
   if chunk is None:continue
   maximum=max(maximum,len(chunk.rows));self.assertLessEqual(len(chunk.rows),2048)
   for i,pos in enumerate(chunk.rows):reconstructed[start+pos]=chunk.data[i*mm.ROW_BYTES:(i+1)*mm.ROW_BYTES]
  self.assertEqual(reconstructed,expected);self.assertGreater(maximum,1024)
  single=self.request(1);value=mm.request_plan(single);f=single.mm_features[0];p=f.mm_position;data=f.data
  old=(single.request_id,id(f),id(data),id(data['image_embeds'].data),id(data['image_grid_thw'].data),f.identifier,p.offset,p.length,id(p.is_embed))
  self.assertEqual(value.signature,old);self.assertEqual(value.data,data['image_embeds'].data.view(self.torch.uint8).numpy().tobytes())
  self.assertIs(mm.request_plan(single),value);self.assertFalse(self.torch.cuda.is_initialized())
 def test_fifth_overlap_reorder_bad_grid_and_mutation_refuse(self):
  with self.assertRaises(mm.NativeMMError):mm.request_plan(self.request(9))
  for which in ('overlap','reorder','grid','mask'):
   r=self.request(2)
   if which=='overlap':r.mm_features[1].mm_position.offset=1600
   if which=='reorder':r.mm_features.reverse()
   if which=='grid':r.mm_features[1].data['image_grid_thw'].data=self.torch.tensor([1,2,2])
   if which=='mask':r.mm_features[1].mm_position.is_embed=self.torch.zeros(1024,dtype=self.torch.bool)
   with self.assertRaises(mm.NativeMMError,msg=which):mm.request_plan(r)
  r=self.request(2);mm.request_plan(r);r.mm_features[1].identifier='changed'
  with self.assertRaisesRegex(mm.NativeMMError,'metadata changed'):mm.request_plan(r)
 def test_eight_image_plan_keeps8192_row_cache_and_per_chunk2048_limit(self):
  request=self.request(8);plan=mm.request_plan(request)
  self.assertEqual(len(plan.positions),8192);self.assertEqual(len(plan.data),40*2**20);self.assertEqual(mm.CACHE_ROWS,8192)
  for start in range(0,request.num_prompt_tokens,2048):
   tx=TransactionId(1,request.request_id,1,start,min(start+2048,request.num_prompt_tokens),1);chunk=plan.chunk(tx,1)
   if chunk is not None:self.assertLessEqual(len(chunk.rows),2048)

 def test_2048_row_payload_fits_the_pinned16MiB_broadcast_guard(self):
  from executor_hooks import BoundedSender,Backpressure
  tx=TransactionId(1,'wire-max-mm',1,0,2048,1);data=b'\0'*(2048*mm.ROW_BYTES)
  chunk=mm.MMChunk(tx,1,tuple(range(2048)),data,hashlib.sha256(data).hexdigest())
  queue=types.SimpleNamespace(enqueue=lambda x:None,buffer=types.SimpleNamespace(max_chunk_bytes=16*2**20))
  sender=BoundedSender(queue);envelope=('flash_s5_submit',dict(scopes={r:chunk for r in range(4)}))
  sender._check_size(envelope)
  import pickle
  self.assertLess(len(pickle.dumps(envelope,protocol=pickle.HIGHEST_PROTOCOL)),11*2**20)
  queue.buffer.max_chunk_bytes=5*2**20
  with self.assertRaises(Backpressure):sender._check_size(envelope)

 def test_boundary_receipt_hashes_request_and_reports_physical_memory_without_sync(self):
  tx=TransactionId(1,'private-request-uid',1,0,2048,1);data=b'\0'*mm.ROW_BYTES
  chunk=mm.MMChunk(tx,1,(0,),data,hashlib.sha256(data).hexdigest())
  admission=types.SimpleNamespace(mm=chunk,transaction=tx,state_epoch=1)
  cuda=types.SimpleNamespace(mem_get_info=lambda device:(80*2**20,24576*2**20))
  with patch.object(mm,'inventory',return_value=dict(cache_storage_bytes=0,tower_registered_cuda_bytes=0)),self.assertLogs(mm.__name__,level='INFO') as logs:
   mm.note_prefix(admission,0,runner=types.SimpleNamespace(device='cuda:0'),worker_rank=1,torch_module=types.SimpleNamespace(cuda=cuda))
  import json
  raw=next(x.split('flash-s5 native-mm-boundary ',1)[1] for x in logs.output if 'flash-s5 native-mm-boundary ' in x);row=json.loads(raw)
  self.assertEqual(row['worker_rank'],1);self.assertEqual(row['free_bytes'],80*2**20);self.assertNotIn('private-request-uid',raw)
  self.assertEqual(row['request_sha256'],hashlib.sha256(b'private-request-uid').hexdigest())

 def test_oversized_chunk_and_stale_lease_never_release(self):
  tx=TransactionId(1,'multi-negative',1,0,4096,1);data=b'\x00'*((2048+1)*mm.ROW_BYTES)
  with self.assertRaises(mm.NativeMMError):mm.MMChunk(tx,1,tuple(range(2049)),data,hashlib.sha256(data).hexdigest())
  tx=replace(tx,chunk_end=2048);data=b'\x00'*(2048*mm.ROW_BYTES);chunk=mm.MMChunk(tx,1,tuple(range(2048)),data,hashlib.sha256(data).hexdigest())
  lease=mm.SpliceLease(chunk);lease.keepalive.append(object());event=types.SimpleNamespace(query=lambda:False);lease.bind(event)
  with self.assertRaises(mm.NativeMMError):lease.release()
  self.assertTrue(lease.keepalive);event.query=lambda:True;lease.release();self.assertEqual(lease.keepalive,[])
if __name__=='__main__':unittest.main()
