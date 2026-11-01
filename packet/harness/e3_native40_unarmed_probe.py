#!/usr/bin/env python3
"""Connected CPU gate: mounted AsyncScheduler/Core batch step, real route/bridge/driver.
Device resources, one-token sampling and process transport are declared leaves.
No scheduling predicate, graph validator or transition acknowledgement is patched.
"""
import argparse
import ast
from collections import deque
from concurrent.futures import Future
import contextlib
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace as NS
from unittest import mock
import e3_native37_admission_probe as base


def overlay_aliases():
    package=importlib.import_module('vllm.v1.flash_e3')
    root=Path(package.__file__).parent
    failures={}
    for path in sorted(root.glob('*.py')):
        if path.stem=='__init__':continue
        assert path.stem not in sys.modules, path.stem
        try:sys.modules[path.stem]=importlib.import_module('vllm.v1.flash_e3.'+path.stem)
        except Exception as exc:failures[path.stem]=type(exc).__name__
    return root,failures


class Connected:
    def __init__(self, stack):
        import test_e3_r4_worker_bridge as bridge
        import test_e3_native_storage_replay as replay
        self.stack=stack
        assert bridge.ed is stack.ed and bridge.sh is stack.sh
        S=stack.scheduler
        S._flash_s5_driver=None
        self.paused=False
        self.engine=NS(is_scheduler_paused=lambda:self.paused,scheduler=S)
        self.sender=bridge.RoutingSender()
        self.transport=bridge.DirectTransport(self.sender)
        self.executor=bridge.FakeExecutor(self.sender,self.transport)
        self.driver=stack.ed.NativeSegmentDriver(stack.controller,executor=self.executor,scheduler=S,
            engine=self.engine,layer_count=4,layer_split=2,stage_of=((0,0),(1,0),(2,1),(3,1)),
            reply_timeout_s=30,stop_budget=64,stop_tick_budget=8).install()
        holder=NS(workers={},driver=self.driver)
        holder.worker=lambda rank:holder.workers[rank]
        with mock.patch.object(bridge,'route_objects',replay.native_route_objects):
            for rank,stage in ((0,0),(1,0),(2,1),(3,1)):bridge.Stack._add_worker(holder,rank,stage)
        self.workers=holder.workers
        self.transport.network=NS(workers=self.workers)
        for rank in (2,3):self.workers[rank].channel.ready=True
        stack.driver=self.driver
        # Execute the entire actual generated Core batch method, no extracted
        # replacement branch: only its executor/sampler/context leaves are CPU.
        from vllm.v1.engine.core import EngineCore
        self.core_step=EngineCore.step_with_batch_queue
        self.outputs=[]
        self.engine.batch_queue=deque()
        self.engine.batch_queue_size=1
        self.engine._should_throttle_prefills=lambda:False
        self.engine.log_error_detail=lambda out:contextlib.nullcontext()
        self.engine.capture_iteration_details=lambda out:contextlib.nullcontext(None)
        self.engine._attach_iteration_details=lambda *args:None
        self.engine._process_aborts_queue=lambda:None
        self.engine.is_ec_consumer=True
        self.engine.is_pooling_model=False
        self.engine.model_executor=NS(execute_model=self.execute,sample_tokens=lambda *a,**k:self.future)
        self.blocked_event=None

    def complete(self):
        for worker in self.workers.values():worker.bus.complete_all(True)
        if self.blocked_event is not None:self.blocked_event.complete=False

    def control(self,command):
        reply=self.driver.submit_control(command)
        for _ in range(500):
            if not hasattr(reply,'done') or reply.done():return reply.result() if hasattr(reply,'result') else reply
            sys.modules['executor_hooks'].engine_idle_tick(self.engine)
            self.complete()
        raise AssertionError('control did not settle')

    def execute(self,out,**kwargs):
        from vllm.v1.outputs import ModelRunnerOutput
        import test_e3_native_storage_replay as replay
        self.outputs.append(dict(out.num_scheduled_tokens))
        ids=list(out.num_scheduled_tokens)
        prefill=any(self.stack.scheduler.requests[rid].is_prefill_chunk or count>1
                    for rid,count in out.num_scheduled_tokens.items())
        if ids:
            desc=replay._Vllm(tokens=sum(out.num_scheduled_tokens.values()),reqs=len(ids))
            if prefill:desc.cg_mode=replay._Mode('NONE')
            for worker in self.workers.values():
                runner=replay.NS(_flash_e3_graph_router=worker.router)
                sys.modules['graph_hooks'].check_route(runner,out,desc,len(ids),desc.num_tokens,
                                                       None if prefill else 1)
        self.future=Future()
        self.future.set_result(ModelRunnerOutput(req_ids=ids,req_id_to_index={k:i for i,k in enumerate(ids)},
            sampled_token_ids=[[7] if not self.stack.scheduler.requests[k].is_prefill_chunk else [] for k in ids]))
        return self.future

    def step(self):
        before=len(self.outputs)
        result=self.core_step(self.engine)
        self.complete()
        return self.outputs[-1] if len(self.outputs)>before else None,result


def run():
    root,failures=overlay_aliases()
    stack=base.Stack()
    conn=Connected(stack)
    clock=[time.monotonic_ns()]
    stack.controller._now_ns=lambda:clock[0]
    conn.driver._monotonic_ns=lambda:clock[0]
    from e3_native_cow_lease_probe import engine_request,host_preprocess
    from vllm.v1.request import RequestStatus
    import test_e3_native_storage_replay as replay
    checks=[]
    def check(name,ok,**details):
        checks.append(dict(name=name,ok=bool(ok),**details))
        assert ok,name
    def req(name,n,outputs=32):
        r=host_preprocess(stack.core,engine_request(name,[(i*17+3)%251 for i in range(n)],name,outputs))
        stack.scheduler.add_request(r);return r
    conn.paused=True
    conn.control({'action':'bootstrap_engine_owner'})
    conn.paused=False
    # Exact box pre-row1 scenario: zero decoders, no native admission intent,
    # and an ordinary API-shaped prompt after the applied split ACK.
    conn.paused=True
    conn.control({'action':'enter_split'})
    conn.paused=False
    empty=req('empty-split-plain',256,outputs=1)
    for i in range(3):
        scheduled,_=conn.step()
        check('empty-split-plain-held-%d'%i,empty.num_computed_tokens==0
              and (not scheduled or empty.request_id not in scheduled)
              and conn.driver.admission_intent() is None)
    clock[0]+=31_000_000_000
    for _ in range(500):
        conn.step()
        if empty.request_id not in stack.scheduler.requests:break
    check('empty-split-plain-completes-after-full',not conn.driver._split_ready
          and all(w.controller.route=='full' for w in conn.workers.values())
          and list(empty.output_token_ids)==[7]
          and stack.controller.last_split_hold_timeout['admission_armed'] is False)
    decode=req('live-decode',40)
    conn.step()
    running=req('running-prefill',10000)
    conn.step()
    frontier=running.num_computed_tokens
    conn.paused=True
    conn.control({'action':'enter_split'})
    conn.paused=False
    check('real-all-rank-unarmed-split',conn.driver._split_ready and conn.driver.admission_intent() is None
          and all(w.controller.route=='split' for w in conn.workers.values()))
    fresh=req('foreign-fresh',50)
    blocks=stack.free_blocks()
    for i in range(3):
        scheduled,_=conn.step()
        check('decode-progress-prefills-held-%s'%i, scheduled.get(decode.request_id)==1
              and fresh.request_id not in scheduled and running.request_id not in scheduled
              and fresh.num_computed_tokens==0 and running.num_computed_tokens==frontier
              and stack.free_blocks()==blocks,scheduled=scheduled)
    # A real native replay lease on rank1. Return must observe its event,
    # retry the identical transition and not stock-schedule until all acks.
    w=conn.workers[1]
    desc=replay.ExecutionDescriptor.from_vllm(replay._Vllm())
    bank=w.router.banks['decode-split']
    lease=w.router.native_replay(desc,observed_context=bank.context.context_id,
                                observed_stream=bank.context.stream_token)
    event=w.bus.events[-1]
    event.complete=False
    conn.blocked_event=event
    clock[0]+=31_000_000_000
    for _ in range(8):
        scheduled,_=conn.step()
        check('pending-ack-no-stock',scheduled is None)
    check('one-rank-pending-keeps-split',conn.driver._split_ready and conn.driver._transition is not None
          and conn.workers[1].controller.route=='split')
    conn.blocked_event=None
    event.complete=True
    for _ in range(500):
        conn.step()
        if not conn.driver._split_ready:break
    check('all-rank-full-before-prefill',not conn.driver._split_ready
          and all(w.controller.route=='full' for w in conn.workers.values()))
    scheduled,_=conn.step()
    check('running-prefill-resumes-on-full',running.request_id in scheduled
          and running.num_computed_tokens>frontier,scheduled=scheduled)
    # Complete held request and decoder via actual schedule/update. No hold
    # mutates tokens; synthetic output is identical to the FULL-alone sampler.
    for _ in range(160):
        conn.step()
        if not stack.scheduler.requests:break
    check('all-requests-completed-no-leases',not stack.scheduler.requests
          and not stack.leases.held_blocks() and stack.controller.parked is None)
    expected={r.request_id:list(r.output_token_ids) for r in (decode,running,fresh)}
    for original in (decode,running,fresh):
        solo=req(original.request_id+'-solo',len(original.prompt_token_ids))
        for _ in range(160):
            conn.step()
            if solo.request_id not in stack.scheduler.requests:break
        check('full-alone-synthetic-continuation-'+original.request_id,
              list(solo.output_token_ids)==expected[original.request_id] and len(solo.output_token_ids)==32)
    # Positive-frontier streaming resume via the real add_request/session path.
    import e3_native39_trunk_probe as oldprobe
    stream=oldprobe.streaming_request(stack,'stream-resume',list(range(256))*8)
    for _ in range(8):
        conn.step()
        if stream.status==RequestStatus.WAITING_FOR_STREAMING_REQ:break
    check('stream-full-input-finished',stream.status==RequestStatus.WAITING_FOR_STREAMING_REQ)
    oldprobe.streaming_request(stack,'stream-resume',list(range(74)))
    saved=stream.num_computed_tokens
    conn.paused=True;conn.control({'action':'enter_split'});conn.paused=False
    scheduled,_=conn.step()
    check('stream-positive-frontier-held-unarmed',stream.request_id not in scheduled and stream.num_computed_tokens==saved)
    stack.abort(stream)
    conn.paused=True;conn.control({'action':'leave_split'});conn.paused=False
    # A real stock preemption after an output token: reset/recompute must be
    # held before allocation just like a fresh request, even without intent.
    recompute=req('recompute',40)
    conn.step()
    check('preemption-has-output',recompute.num_output_tokens>0)
    stack.scheduler.running.remove(recompute)
    stack.scheduler._preempt_request(recompute,time.monotonic())
    check('real-preemption-zeroes-frontier',recompute.num_computed_tokens==0 and recompute.num_output_tokens>0)
    conn.paused=True;conn.control({'action':'enter_split'});conn.paused=False
    blocks=stack.free_blocks()
    scheduled,_=conn.step()
    check('preempted-output-held-unarmed',recompute.request_id not in scheduled and recompute.num_computed_tokens==0 and stack.free_blocks()==blocks)
    stack.abort(recompute)
    conn.paused=True;conn.control({'action':'leave_split'});conn.paused=False
    # Production hybrid prefix cache, including the stock final-token case.
    cache_ids=[(i*13+3)%251 for i in range(177)]
    cached=oldprobe.salted_request(stack,'cache-source',cache_ids,'n40-cache')
    for _ in range(8):
        conn.step()
        if cached.request_id not in stack.scheduler.requests:break
    hit=oldprobe.salted_request(stack,'cache-final-token',cache_ids,'n40-cache')
    computed,n_cached,*_=stack.scheduler._get_local_prefix_cache_hit(hit)
    check('actual-full-prefix-hit',n_cached==176,hit=n_cached)
    conn.paused=True;conn.control({'action':'enter_split'});conn.paused=False
    blocks=stack.free_blocks()
    scheduled,_=conn.step()
    check('cached-final-token-held-unarmed',hit.request_id not in scheduled and hit.num_computed_tokens==0 and stack.free_blocks()==blocks)
    conn.paused=True;conn.control({'action':'leave_split'});conn.paused=False
    conn.step()
    check('cached-final-token-completes-on-full',hit.request_id not in stack.scheduler.requests and list(hit.output_token_ids)==[7])
    check('final-ownership-clean',not stack.scheduler.requests and not stack.leases.held_blocks() and stack.controller.parked is None)
    import torch
    assert not torch.cuda.is_initialized()
    return dict(verdict='PASS',checks=checks,geometry=stack.geometry,aliases_root=str(root),
                alias_import_failures=failures,cuda_initialized=False,core_step='EngineCore.step_with_batch_queue',
                last_hold=stack.controller.native_admission_record().get('last_split_hold_timeout'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:record=run();rc=0
    except Exception as exc:
        record=dict(verdict='FAIL',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc());rc=1
    record.update(schema='native40-connected-cpu/1',gpu_execution=False,box_evidence=False,
                  probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with a.out.open('x') as f:json.dump(record,f,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps({k:record.get(k) for k in ('verdict','error_type','error')}));return rc
if __name__=='__main__':raise SystemExit(main())
