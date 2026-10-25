"""Opt-in same-input E2 byte oracle. Debug only; GPU-unqualified offline.

Only counters/shapes are persisted, never tensors, request IDs or addresses.
The baseline gather reuses the SAME arena. Snapshots live on CPU, not in a
second VRAM window. Any mismatch writes an immutable primary failure.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path

from retention_contract import ContractError


class ExactnessError(ContractError):
    pass


class TorchExactBackend:
    def __init__(self):
        import torch
        self.torch = torch

    def snapshot(self,tensor):
        # .contiguous() AFTER .cpu(): no temporary contiguous GPU allocation.
        return tensor.detach().cpu().contiguous()

    def integers(self,tensor):
        return self.snapshot(tensor).reshape(-1).tolist()

    def equal(self,a,b):
        return (a.shape==b.shape and a.dtype==b.dtype and
                self.torch.equal(a.view(self.torch.uint8),b.view(self.torch.uint8)))

    def nbytes(self,a):
        return a.numel()*a.element_size()

    def finite(self,a):
        # Small CPU chunks bound temporary bool storage for output checks.
        flat=a.reshape(-1)
        return all(bool(self.torch.isfinite(flat[i:i+262144]).all()) for i in range(0,flat.numel(),262144))

    def differing_bytes(self,a,b):
        if a.shape!=b.shape or a.dtype!=b.dtype:
            return None
        x=a.view(self.torch.uint8).reshape(-1);y=b.view(self.torch.uint8).reshape(-1)
        return sum(int((x[i:i+1048576]!=y[i:i+1048576]).count_nonzero()) for i in range(0,x.numel(),1048576))

    def gather_reference(self,kv,table,lo,hi,staged):
        physical=table[lo:hi].to(self.torch.int64)
        self.torch.index_select(kv,0,physical,out=staged)


class Exactness:
    def __init__(self,directory,rank,layers,backend,emit,*,page_tokens=4048,arena_pages=24):
        if rank not in ('PP0_TP0','PP0_TP1','PP1_TP0','PP1_TP1'):
            raise ContractError('invalid exactness worker rank')
        self.root=Path(directory)
        if not self.root.is_dir():
            raise ContractError('exactness directory must be prepared by the slot owner')
        self.rank,self.backend,self.emit=rank,backend,emit
        self.page_tokens,self.arena_pages=page_tokens,arena_pages
        self.progress=self.root/(rank+'.exact.json')
        self.primary=self.root/(rank+'.failure.json')
        self.tmp=self.root/(rank+'.exact.next')
        self.active=None
        self.active_buckets=0
        self.failed=False
        self.stats={name:dict(calls=0,dense_dma=0,sparse_uva=0,empty=0,buckets=0,bucket_bytes=0,
                             output_bytes=0,max_length=0,prefix_hit_calls=0,
                             dense_max_length=0,dense_multibucket_calls=0) for name in layers}
        if self.primary.exists() or self.progress.exists() or self.tmp.exists():
            raise ContractError('exactness worker artifact already exists')
        self._write(initial=True)

    def _record(self):
        return dict(schema=1,rank=self.rank,failed=self.failed,active=self.active is not None,
                    page_tokens=self.page_tokens,arena_pages=self.arena_pages,
                    layers=self.stats,scope='same-input QSA bytes before MoE; not model-level equivalence')

    def _write(self,initial=False):
        if initial:
            with self.progress.open('x') as f:json.dump(self._record(),f,sort_keys=True)
        else:
            with self.tmp.open('x') as f:json.dump(self._record(),f,sort_keys=True)
            os.replace(self.tmp,self.progress)

    def fail(self,kind,**evidence):
        self.failed=True
        record=dict(schema=1,rank=self.rank,fatal_correctness=True,kind=kind,evidence=evidence)
        try:
            with self.primary.open('x') as f:json.dump(record,f,sort_keys=True)
        except FileExistsError:
            pass  # First failure is immutable; subsequent errors cannot clear it.
        self.emit('FLASH_E2_EXACT_FAIL '+json.dumps(record,sort_keys=True))
        raise ExactnessError('E2 exactness failed: '+kind)

    @contextmanager
    def layer(self,lifecycle,layer,best,row,args,gpu_metadata):
        if self.failed or self.active is not None:
            raise ExactnessError('exactness oracle failed or active')
        b=self.backend
        try:
            # Publish the in-flight state BEFORE any GPU read or candidate work.
            # A worker killed mid-check cannot leave an earlier PASS-shaped
            # progress file as the apparent final result.
            self.active=(layer,args,row)
            self.active_buckets=0
            self._write()
            if gpu_metadata is None or len(gpu_metadata)!=2:
                self.fail('missing_gpu_metadata',layer=layer)
            starts,lengths=gpu_metadata
            if (b.integers(starts[best:best+2])!=[row.start,row.end] or
                    b.integers(lengths[best:best+1])!=[row.length]):
                self.fail('gpu_sequence_metadata',layer=layer)
            mirror=lifecycle.mirror
            physical=mirror.current[row.request_id].tables[mirror.layer_groups[layer]][:args[5]+1]
            if b.integers(args[3][:args[5]+1])!=list(physical):
                self.fail('gpu_block_table',layer=layer,pages=len(physical))
            if b.integers(args[4])!=list(range(row.start,row.end)):
                self.fail('gpu_query_rows',layer=layer,rows=row.end-row.start)
            yield
        except ExactnessError:
            raise
        except BaseException as exc:
            self.fail('oracle_or_candidate_exception',layer=layer,error_type=type(exc).__name__)
        finally:
            if not self.failed:self.active=None

    def bucket(self,session,staged,lo,hi,table):
        if self.active is None or self.active[0]!=session.layer:
            self.fail('bucket_outside_layer')
        layer,args,_=self.active
        b=self.backend
        candidate=b.snapshot(staged)
        b.gather_reference(args[1],table,lo,hi,staged)
        baseline=b.snapshot(staged)
        if not b.equal(candidate,baseline):
            self.fail('arena_bytes',layer=layer,lo=lo,hi=hi,
                      compared_bytes=b.nbytes(candidate),different_bytes=b.differing_bytes(candidate,baseline))
        self.stats[layer]['buckets']+=1
        self.stats[layer]['bucket_bytes']+=b.nbytes(candidate)
        self.active_buckets+=1

    def output(self,lifecycle,original,route):
        if self.active is None or route not in ('dense_dma','sparse_uva','empty'):
            self.fail('invalid_output_check')
        layer,args,row=self.active
        b=self.backend
        expected=((args[5]+1+lifecycle.mirror.arena.arena_pages-1)//lifecycle.mirror.arena.arena_pages
                  if route=='dense_dma' else 0)
        if self.active_buckets!=expected:
            self.fail('bucket_coverage',layer=layer,route=route,expected=expected,observed=self.active_buckets)
        candidate=b.snapshot(args[8][row.start:row.end])
        original(*args)
        baseline=b.snapshot(args[8][row.start:row.end])
        if not b.equal(candidate,baseline):
            self.fail('qsa_output_bytes',layer=layer,route=route,rows=row.end-row.start,
                      compared_bytes=b.nbytes(candidate),different_bytes=b.differing_bytes(candidate,baseline))
        if not b.finite(candidate) or not b.finite(baseline):
            self.fail('nonfinite_qsa_output',layer=layer,route=route)
        stats=self.stats[layer]
        stats['calls']+=1;stats[route]+=1;stats['output_bytes']+=b.nbytes(candidate)
        stats['max_length']=max(stats['max_length'],row.length)
        if route=='dense_dma':
            stats['dense_max_length']=max(stats['dense_max_length'],row.length)
            stats['dense_multibucket_calls']+=int(self.active_buckets>1)
        f=lifecycle.frame
        if row.request_id in f.restarted and lifecycle.mirror.current[row.request_id].computed>0:
            stats['prefix_hit_calls']+=1

    def completed(self):
        if self.failed or self.active is not None:
            raise ExactnessError('cannot publish incomplete exactness evidence')
        self._write()
