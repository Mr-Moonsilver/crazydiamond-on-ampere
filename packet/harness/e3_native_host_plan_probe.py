#!/usr/bin/env python3
"""Actual QSA metadata class and original guard prefix; CPU only, no GPU owners."""
import argparse,ast,copy,hashlib,importlib,inspect,json,traceback
from pathlib import Path
from types import SimpleNamespace
from e3_native_import_probe import check

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--graph',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 result=check(json.loads(a.graph.read_text()));result['schema']='native11-real-qsa-host-plan/1'
 try:
  import torch
  qsa=importlib.import_module('vllm.models.qwen4_exp.nvidia.qsa')
  lc=importlib.import_module('vllm.v1.flash_e3.lane_chunk')
  et=importlib.import_module('vllm.v1.flash_e3.e3_types')
  from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
  assert qsa.FlashAttentionMetadata is FlashAttentionMetadata
  n=8;end=32
  def metadata():
   return FlashAttentionMetadata(num_actual_tokens=n,max_query_len=n,
     query_start_loc=torch.tensor([0,n],dtype=torch.int32,device='cpu'),
     max_seq_len=end,seq_lens=torch.tensor([end],dtype=torch.int32,device='cpu'),
     block_table=torch.tensor([[1,2,3]],dtype=torch.int32,device='cpu'),
     slot_mapping=torch.arange(n,dtype=torch.int64,device='cpu'),
     use_cascade=False,common_prefix_len=0,cu_prefix_query_lens=None,
     prefix_kv_lens=None,suffix_kv_lens=None)
  source=Path(qsa.__file__).read_text();tree=ast.parse(source)
  fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_flash_s5_sparse_staged_operation')
  stop=next(i for i,x in enumerate(fn.body) if isinstance(x,ast.ImportFrom) and x.module=='ops.qsa')
  projected=copy.deepcopy(fn);projected.name='_actual_guard_prefix';projected.body=projected.body[:stop]+[ast.Return(value=ast.Tuple(elts=[ast.Name(id='host',ctx=ast.Load()),ast.Name(id='start',ctx=ast.Load()),ast.Name(id='end',ctx=ast.Load())],ctx=ast.Load()))]
  mod=ast.fix_missing_locations(ast.Module(body=[projected],type_ignores=[]))
  assert not any(isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr in ('tolist','item','cpu','numpy') for x in ast.walk(mod))
  ns=dict(qsa.__dict__);exec(compile(mod,str(qsa.__file__),'exec'),ns)
  # Only the already-observed operation preconditions are input data; no model,
  # capability/readiness owner or module/class is constructed or replaced.
  layer=SimpleNamespace(_qsa_kv_offload=True,_qsa_stage_min_rows=1)
  def guard(md):return ns['_actual_guard_prefix'](None,layer,None,None,None,md,n,True,None)
  old=metadata()
  try:guard(old)
  except RuntimeError as exc:assert str(exc)=='S5 operation requires CPU host metadata; no device readback'
  else:raise AssertionError('native10 metadata shape not refused')
  admission=lc.ChunkAdmission(et.TransactionId(1,'native11-cpu',1,24,32,1),33,tuple(range(n)),(1,2),((1,2,3),),state_epoch=1)
  md=metadata();other=metadata();starts=torch.tensor([0,n],dtype=torch.int32,device='cpu');lengths=torch.tensor([end],dtype=torch.int32,device='cpu')
  qsl=md.query_start_loc;sl=md.seq_lens
  host=lc.attach_qsa_host_metadata({'qsa':md,'other':other},('qsa',),starts,lengths,admission)
  assert md._qsa_host_meta is host and guard(md)==(host,0,n)
  assert md.query_start_loc is qsl and md.seq_lens is sl and not hasattr(other,'_qsa_host_meta')
  starts.fill_(0);lengths.fill_(0)
  assert host==([0,n],[end]) and guard(md)==(host,0,n)
  refused=0
  for qs,ls in (([0,n-1],[end]),([0,n],[end-1])):
   try:lc.attach_qsa_host_metadata({'qsa':metadata()},('qsa',),torch.tensor(qs,device='cpu'),torch.tensor(ls,device='cpu'),admission)
   except lc.LaneChunkError:refused+=1
   else:raise AssertionError('wrong chunk CPU plan admitted')
  stale=metadata();stale._qsa_host_meta=([0,n],[end-1])
  try:lc.attach_qsa_host_metadata({'qsa':stale},('qsa',),torch.tensor([0,n],device='cpu'),torch.tensor([end],device='cpu'),admission)
  except lc.LaneChunkError:refused+=1
  else:raise AssertionError('stale metadata overwritten')
  assert refused==3
  result['host_plan']=dict(verdict='PASS',actual_metadata_class=type(md).__module__+'.'+type(md).__name__,metadata_source_sha256=sha(inspect.getfile(type(md))),lane_source_sha256=sha(lc.__file__),qsa_source_sha256=sha(qsa.__file__),native10_negative_rejected=True,actual_guard_prefix_pass=True,guard_has_no_device_readback=True,shared_plan_object=True,cpu_buffer_reuse_preserved=True,device_fields_unchanged=True,unrelated_metadata_untouched=True,refusals=refused,scope='actual metadata class and generated guard-prefix projection plus real lane binding helper; operation config is CPU input data from observed preconditions; no model/GPU/lease qualification')
 except Exception as exc:
  result['failures'].append('host_plan:'+type(exc).__name__);result['host_plan']=dict(verdict='FAIL',error_type=type(exc).__name__,error_message=str(exc)[:512],stack=[dict(file=Path(f.filename).name,function=f.name,line=f.lineno) for f in traceback.extract_tb(exc.__traceback__)[-7:]])
 import torch
 result['cuda_initialized']=torch.cuda.is_initialized()
 if result['cuda_initialized']:result['failures'].append('CUDA-initialized')
 result['verdict']='FAIL' if result['failures'] else 'INCONCLUSIVE' if result['missing'] else 'PASS';result['rc']=1 if result['failures'] else 2 if result['missing'] else 0
 result['probe_sha256']=sha(__file__);result['graph_sha256']=sha(a.graph)
 with a.out.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
 print(json.dumps({k:result[k] for k in ('verdict','rc','counts','failures','missing','cuda_initialized')}));return result['rc']
if __name__=='__main__':raise SystemExit(main())
