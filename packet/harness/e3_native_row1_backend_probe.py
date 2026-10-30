#!/usr/bin/env python3
"""Actual generated-overlay row1 backend contract, constructor and CPU tensors.

No builder import path, GPU/model/allocator owner, mocks or readiness flags.
The frozen E2 backend is retained as the exact negative control.
"""
import argparse,hashlib,importlib,inspect,json,traceback
from pathlib import Path
from e3_native_import_probe import check
REQUIRED=('snapshot','equal','nbytes','differing_bytes','finite','gather_rows','gather_reference')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--graph',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 result=check(json.loads(a.graph.read_text()));result['schema']='native10-real-overlay-row1-backend/1'
 try:
  import torch
  su=importlib.import_module('vllm.v1.flash_e3.s5_startup')
  ro=importlib.import_module('vllm.v1.flash_e3.row1_oracle')
  et=importlib.import_module('vllm.v1.flash_e3.e3_types')
  oldmod=importlib.import_module('vllm.v1.flash_e2.exactness')
  typ,qualifier_type=su.row1_qualification_types();backend=typ()
  assert typ.__module__=='vllm.v1.flash_e3.row1_exactness'
  assert qualifier_type is ro.Row1Qualifier
  assert all(callable(getattr(backend,n,None)) for n in REQUIRED)
  kwargs=dict(transaction=et.TransactionId(1,'native10-cpu',1,0,4,1),
              lane=et.LaneId(0,et.LaneRole.PREFILL,1,0),
              epochs=lambda:dict(worker_incarnation='cpu-only',engine_epoch=1,route_generation=1))
  store=ro.Row1RecordStore(capacity=64);qualified=qualifier_type(store,backend=backend,**kwargs)
  assert qualified.backend is backend and qualified.store is store
  assert sha(oldmod.__file__)=='baf79d18af1b1c97f5b61472c7bac93d9d937d9f72adcbe517212718c1f10545'
  old=oldmod.TorchExactBackend();missing=[n for n in REQUIRED if not callable(getattr(old,n,None))]
  assert missing==['gather_rows']
  try:qualifier_type(ro.Row1RecordStore(capacity=64),backend=old,**kwargs)
  except ro.Row1OracleError as exc:assert str(exc)=='qualifier backend lacks gather_rows'
  else:raise AssertionError('native9 frozen-E2 constructor negative not rejected')
  src=torch.arange(96,dtype=torch.float32,device='cpu').reshape(8,1,4,3)
  prior=src.clone();table=torch.tensor([5,2],dtype=torch.int32,device='cpu')
  out=torch.empty((2,1,4,3),dtype=src.dtype,device='cpu')
  backend.gather_reference(src,table,0,2,out)
  assert torch.equal(out,torch.index_select(src,0,table.to(torch.int64)))
  rows=torch.full((5,1,4,3),-99.0,dtype=src.dtype,device='cpu')
  backend.gather_rows(src,5,7,rows,3)
  flat=rows.reshape(-1,3)
  assert torch.equal(flat[3:10],src.reshape(-1,3)[5:12])
  assert bool((flat[:3]==-99).all()) and bool((flat[10:]==-99).all())
  snap=backend.snapshot(out);same=snap.clone();different=snap.clone();different.reshape(-1)[0]+=1
  assert backend.equal(snap,same) and not backend.equal(snap,different)
  assert backend.nbytes(snap)==snap.numel()*snap.element_size()
  assert backend.differing_bytes(snap,same)==0 and backend.differing_bytes(snap,different)>0
  assert backend.finite(snap) and not backend.finite(torch.tensor([float('nan')],device='cpu'))
  assert torch.equal(src,prior)
  result['row1_backend']=dict(verdict='PASS',required=list(REQUIRED),all_seven_callable=True,
      all_seven_exercised_on_cpu=True,actual_qualifier_constructed=True,
      backend_owner=typ.__module__+'.'+typ.__name__,backend_origin=inspect.getfile(typ),
      backend_sha256=sha(inspect.getfile(typ)),qualifier_origin=inspect.getfile(qualifier_type),
      qualifier_sha256=sha(inspect.getfile(qualifier_type)),frozen_e2_origin=oldmod.__file__,
      frozen_e2_sha256=sha(oldmod.__file__),frozen_e2_missing=missing,
      native9_constructor_negative_rejected=True,dense_gather_exact=True,sparse_gather_exact=True,
      sparse_outside_rows_unchanged=True,source_unchanged=True,
      gpu_owners_constructed=False,scope='actual mounted E2/E3 files, real Torch CPU tensors and actual Row1Qualifier constructor; no native GPU/lease/lifecycle qualification')
 except Exception as exc:
  result['failures'].append('row1_backend:'+type(exc).__name__)
  result['row1_backend']=dict(verdict='FAIL',error_type=type(exc).__name__,error_message=str(exc)[:512],stack=[dict(file=Path(f.filename).name,function=f.name,line=f.lineno) for f in traceback.extract_tb(exc.__traceback__)[-7:]])
 import torch
 result['cuda_initialized']=torch.cuda.is_initialized()
 if result['cuda_initialized']:result['failures'].append('CUDA-context-in-CPU-proof')
 result['verdict']='FAIL' if result['failures'] else 'INCONCLUSIVE' if result['missing'] else 'PASS';result['rc']=1 if result['failures'] else 2 if result['missing'] else 0
 result['graph_sha256']=sha(a.graph);result['probe_sha256']=sha(__file__)
 with a.out.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
 print(json.dumps({k:result[k] for k in ('verdict','rc','counts','failures','missing','cuda_initialized')}));return result['rc']
if __name__=='__main__':raise SystemExit(main())
