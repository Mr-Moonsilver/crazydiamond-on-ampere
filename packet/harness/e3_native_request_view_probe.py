#!/usr/bin/env python3
"""CPU-only actual-image Request/token-view and mounted import proof.

The metadata method is pure: its self argument supplies only the two explicit
PLE configuration values. No allocator/provider/GPU owner is constructed and
no readiness flag, class or module is replaced.
"""
import argparse
import ast
from dataclasses import dataclass
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import traceback
from e3_native_import_probe import check

@dataclass(frozen=True)
class PleContract:
    contract_len: int = 2
    eos_token_id: int = 248044


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result=check(json.loads(a.graph.read_text()))
    result['schema']='native9-real-request-view-proof/1'
    try:
        from collections.abc import Sequence
        from vllm.v1.request import Request
        from vllm.v1.utils import ConstantList
        from vllm.sampling_params import SamplingParams
        am=importlib.import_module('vllm.v1.flash_e3.admission_metadata')
        contract=PleContract()
        request=Request(request_id='native9-cpu-view',prompt_token_ids=list(range(98312)),
                        sampling_params=SamplingParams(max_tokens=1),pooling_params=None)
        view=request.all_token_ids
        assert isinstance(view,ConstantList) and isinstance(view,Sequence)
        assert not isinstance(view,(list,tuple))
        assert len(view)==98312
        # Exact native8 method bytes, with its actual runtime exception owner.
        legacy_path=Path('/proof/native8-admission-metadata.py')
        expected=json.loads(Path('/proof/native8-counterexample.json').read_text())['sha256']
        assert hashlib.sha256(legacy_path.read_bytes()).hexdigest()==expected
        tree=ast.parse(legacy_path.read_text())
        cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='NativeAdmissionMetadata')
        fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_ngram_context')
        ns=dict(am.__dict__);exec(compile(ast.Module(body=[fn],type_ignores=[]),str(legacy_path),'exec'),ns)
        refused=0;cases=[]
        for start in (0,1,4,1024,2048,98304):
            request.num_computed_tokens=start
            wanted=tuple([contract.eos_token_id]*max(0,2-start)+list(view[max(0,start-2):start]))
            actual=am.NativeAdmissionMetadata._ngram_context(contract,request,start)
            assert actual==(wanted,max(0,2-start))
            try:ns['_ngram_context'](contract,request,start)
            except am.E3ContractError as exc:
                assert 'tokens do not cover' in str(exc);refused+=1
            else:raise AssertionError('native8 ConstantList counterexample not rejected')
            cases.append(dict(start=start,available=len(view),context_exact=True,padding=actual[1]))
        try:am.NativeAdmissionMetadata._ngram_context(contract,request,len(view)+1)
        except am.E3ContractError as exc:assert 'do not cover' in str(exc)
        else:raise AssertionError('short token view accepted')
        try:view.append(1)
        except TypeError:pass
        else:raise AssertionError('real view is mutable')
        rejected=0
        for bad in ('1234',b'1234',bytearray(b'1234'),{0:1},iter(range(4)),
                    ConstantList([0,True,2,3]),ConstantList([0,-1,2,3])):
            request.all_token_ids=bad
            try:am.NativeAdmissionMetadata._ngram_context(contract,request,2)
            except am.E3ContractError:rejected+=1
            else:raise AssertionError('malformed token view accepted')
        request.all_token_ids=view
        result['request_view']=dict(verdict='PASS',cases=cases,legacy_refused=refused,
            malformed_refused=rejected,short_view_refused=True,source_unchanged=len(view)==98312,
            actual_request_type=Request.__module__+'.'+Request.__name__,
            actual_view_type=type(view).__module__+'.'+type(view).__name__,
            request_source_sha256=hashlib.sha256(Path(inspect.getfile(Request)).read_bytes()).hexdigest(),
            view_source_sha256=hashlib.sha256(Path(inspect.getfile(ConstantList)).read_bytes()).hexdigest(),
            metadata_provider_constructed=False,allocator_constructed=False,
            scope='actual real Request/ConstantList; unbound pure metadata method with explicit PLE contract values; no allocator/lifecycle proof')
    except Exception as exc:
        result['failures'].append('request_view:'+type(exc).__name__)
        result['request_view']=dict(verdict='FAIL',error_type=type(exc).__name__,error_message=str(exc)[:512],
            stack=[dict(file=Path(f.filename).name,function=f.name,line=f.lineno) for f in traceback.extract_tb(exc.__traceback__)[-6:]])
    import torch
    result['cuda_initialized']=torch.cuda.is_initialized()
    if result['cuda_initialized']:result['failures'].append('CUDA-context-in-CPU-proof')
    result['verdict']='FAIL' if result['failures'] else 'INCONCLUSIVE' if result['missing'] else 'PASS'
    result['rc']=1 if result['failures'] else 2 if result['missing'] else 0
    result['probe_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with a.out.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({k:result[k] for k in ('verdict','rc','counts','failures','missing','cuda_initialized')}))
    return result['rc']
if __name__=='__main__':raise SystemExit(main())
