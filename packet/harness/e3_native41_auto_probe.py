#!/usr/bin/env python3
"""Plain API-shaped requests through actual startup/Core/scheduler/driver/bridges.
Only CPU construction/device/process/model leaves, listed in native41_fixture_leaves.json.
"""
import argparse
from collections import deque
from concurrent.futures import Future
import contextlib
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import tempfile
import traceback
from types import SimpleNamespace as NS
from unittest import mock
import e3_native37_admission_probe as base
from e3_native40_unarmed_probe import overlay_aliases


class Connected:
    def __init__(self, switch):
        import test_e3_r4_worker_bridge as bridge
        import test_e3_native_storage_replay as replay
        import test_e3_native_overlap as native
        import s5_startup as startup
        import executor_hooks as eh
        import native_backend as nb
        import native_resources as nr
        import async_handoff as ah
        from vllm.v1.engine.core import EngineCore, EngineCoreProc
        import vllm.distributed as distributed
        init=distributed.init_distributed_environment
        rendezvous=Path(tempfile.mkdtemp(prefix='native41-cpu-',dir='/tmp'))/'rendezvous'
        def cpu_init(**kwargs):
            kwargs['distributed_init_method']='file://'+str(rendezvous)
            return init(**kwargs)
        with mock.patch.object(distributed,'init_distributed_environment',cpu_init):
            stack=base.Stack()
        self.stack=stack
        self.scheduler=S=stack.scheduler
        assert bridge.sh is stack.sh and bridge.ed is stack.ed
        # Discard only unused constructor objects, before any request/lease/RPC.
        assert not S.requests and not stack.leases.held_blocks() and stack.controller.parked is None
        S._flash_s5_driver=S._flash_s5_admission=S._flash_s5_leases=None
        assert S._flash_s5_admission_metadata._scope is None
        S._flash_s5_admission_metadata=None
        S.kv_cache_manager.block_pool._flash_s5_leases=None
        self.engine=NS(scheduler=S,vllm_config=stack.host,batch_queue=deque(),batch_queue_size=2)
        self.engine.is_scheduler_paused=lambda:EngineCore.is_scheduler_paused(self.engine)
        self.sender=bridge.RoutingSender();self.transport=bridge.DirectTransport(self.sender)
        self.executor=bridge.FakeExecutor(self.sender,self.transport)
        self.executor.response_mqs={rank:object() for rank in range(4)}
        self.executor.shutdown=lambda:None
        self.engine.model_executor=self.executor
        self.workers={}
        self.external_controls=0
        real_driver=stack.ed.NativeSegmentDriver
        conn=self
        class DriverWithCPUWorkers(real_driver):
            def install(self):
                result=super().install()
                holder=NS(driver=self,workers={})
                holder.worker=lambda rank:holder.workers[rank]
                with mock.patch.object(bridge,'route_objects',replay.native_route_objects):
                    for rank,stage in ((0,0),(1,0),(2,1),(3,1)):bridge.Stack._add_worker(holder,rank,stage)
                conn.workers=holder.workers
                conn.transport.network=NS(workers=conn.workers)
                for rank,w in conn.workers.items():
                    if rank in (2,3):w.channel.ready=True
                    runner=native.attach_runner(w)
                    runner.max_num_reqs=8
                    runner.cudagraph_manager=conn.dispatch_manager(w)
                    resource=w.dispatcher.native
                    resource.runner=runner;resource.router=w.router
                    resource.decode_owner=w.state.decode_owner
                    resource.inventory=lambda:dict(cpu_resource_leaf=True)
                return result
        class CPUComposition(ah.Composition):
            def __init__(self):
                self.scheduler_leases=stack.sh.SchedulerLeases()
                self.frontier=ah.CommittedFrontier();self.ledger=ah.LifetimeLedger()
                self.dispatcher=NS(cpu_unused_handoff=True)
        composition=CPUComposition()
        env={'FLASH_S5_OVERLAP':'1','FLASH_S5_AUTO_ARM':switch}
        # Production install_engine, actual provider/controller/driver/startup
        # fence/policy install. CPU resource constructors alone are substituted.
        with mock.patch.object(startup,'require_native_ready',lambda:None), \
             mock.patch.object(startup,'_native_execution_shape',lambda *a,**k:(4,2,((0,0),(1,0),(2,1),(3,1)))), \
             mock.patch.object(startup,'build_composition',lambda *a,**k:composition), \
             mock.patch.object(eh,'install',lambda *a,**k:conn.transport), \
             mock.patch.object(nb,'RemoteCompletionBackend',lambda *a,**k:NS(cpu_unused_backend=True)), \
             mock.patch.object(nr,'response_success',lambda:object()), \
             mock.patch.object(stack.ed,'NativeSegmentDriver',DriverWithCPUWorkers):
            startup.install_engine(self.engine,env=env)
        self.driver=self.engine._flash_s5_driver
        stack.driver=self.driver;stack.controller=self.driver.controller
        stack.leases=self.driver.controller.leases
        self.controller=stack.controller
        # Remaining real Core fields are constructed after the startup hook.
        self.engine.batch_queue=deque();self.engine.batch_queue_size=2
        self.engine._should_throttle_prefills=lambda:False
        self.engine.log_error_detail=lambda out:contextlib.nullcontext()
        self.engine.capture_iteration_details=lambda out:contextlib.nullcontext(None)
        self.engine._attach_iteration_details=lambda *args:None
        self.engine._process_aborts_queue=lambda:None
        self.engine.is_ec_consumer=True;self.engine.is_pooling_model=False
        self.executor.execute_model=self.execute
        self.executor.sample_tokens=lambda *a,**k:self.future
        self.engine.step_fn=lambda:EngineCore.step_with_batch_queue(self.engine)
        self.engine.post_step=lambda executed:None
        self.engine.output_queue=NS(put_nowait=lambda output:None)
        self.process_step=lambda:EngineCoreProc._process_engine_step.__wrapped__(self.engine) if hasattr(EngineCoreProc._process_engine_step,'__wrapped__') else EngineCoreProc._process_engine_step(self.engine)
        self.has_work=lambda:EngineCoreProc.has_work(self.engine)
        self.engine.engines_running=False
        self.traces=[];self.native=[];self.seen_tx=set();self.ticks=0;self.blocked_event=None;self.skew=None

    def dispatch_manager(self,worker):
        from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager, BatchExecutionDescriptor
        from vllm.config import CUDAGraphMode
        m=ModelCudaGraphManager.__new__(ModelCudaGraphManager)
        m._graphs_captured=True;m._lora_dispatch_map={};m._max_lora_case=0
        bank=worker.router.banks['decode-split']
        descs=[BatchExecutionDescriptor(cg_mode=CUDAGraphMode.FULL,num_tokens=d.num_tokens,
            num_reqs=d.num_reqs,uniform_token_count=d.uniform_token_count,max_query_len=d.max_query_len)
            for d in bank.descriptors]
        m._candidates={(n,0):[d for d in descs if d.num_tokens>=n] for n in (1,2)}
        return m

    def request(self,name,n,outputs=8):
        from e3_native_cow_lease_probe import engine_request,host_preprocess
        request=host_preprocess(self.stack.core,engine_request(name,[(i*13+3)%251 for i in range(n)],name,outputs))
        self.scheduler.add_request(request)
        return request

    def sync_rows(self):
        # Leaf for GPU runner's ordinary req-state host mirrors (no validator
        # patched): only requests actually dispatched on the ordinary path.
        import native_overlap_worker as nw
        for w in self.workers.values():
            r=w.model_runner
            live=[req for req in self.scheduler.running if req.request_id in self.dispatched]
            r.req_states=NS(req_id_to_index={req.request_id:i for i,req in enumerate(live)},
                prefill_len=NS(np=[req.num_prompt_tokens for req in live]),
                num_computed_prefill_tokens=[min(req.num_computed_tokens,req.num_prompt_tokens) for req in live])
            assert nw.live_census(r,r._flash_s5_native_census)['available']

    def execute(self,out,**kwargs):
        from vllm.v1.outputs import ModelRunnerOutput
        import test_e3_native_storage_replay as replay
        import graph_hooks
        import native_overlap_worker as nw
        ids=list(out.num_scheduled_tokens)
        self.dispatched.update(ids)
        self.sync_rows()
        prefill=any(count>1 or self.scheduler.requests[rid].is_prefill_chunk for rid,count in out.num_scheduled_tokens.items())
        if ids:
            desc=replay._Vllm(tokens=sum(out.num_scheduled_tokens.values()),reqs=len(ids))
            if prefill:desc.cg_mode=replay._Mode('NONE')
            for w in self.workers.values():
                nw.install_census(w.model_runner).observe_scheduler(out)
                graph_hooks.check_route(NS(_flash_e3_graph_router=w.router),out,desc,len(ids),desc.num_tokens,None if prefill else 1)
        self.traces.append(dict(scheduled=dict(out.num_scheduled_tokens),split=self.driver._split_ready,
                               prefill=prefill,generation=self.driver.route_generation))
        self.future=Future()
        self.future.set_result(ModelRunnerOutput(req_ids=ids,req_id_to_index={k:i for i,k in enumerate(ids)},
            sampled_token_ids=[[7] if not self.scheduler.requests[k].is_prefill_chunk else [] for k in ids]))
        return self.future

    def step(self):
        if not hasattr(self,'dispatched'):self.dispatched=set()
        if self.scheduler.requests or getattr(getattr(self.driver, 'auto_arm', None), 'active', False):
            assert self.has_work(), 'paused automatic cycle lost Core wakeup'
        self.process_step()
        parked=self.controller.parked
        if parked is not None and parked.transaction not in self.seen_tx:
            tx=parked.transaction;self.seen_tx.add(tx)
            self.native.append(dict(uid=tx.request_uid,start=tx.chunk_start,end=tx.chunk_end,
                                    terminal=parked.admission.terminal,split=self.driver._split_ready))
        if self.skew is not None:self.skew.tick()
        else:
            for w in self.workers.values():w.bus.complete_all(True)
        if self.blocked_event is not None:self.blocked_event.complete=False
        self.ticks+=1


class Skew:
    """tp-skew leaf: asynchronous per-rank GPU completion (fixture leaf
    tp-skew-completion). In-order partial completion per rank and Core tick,
    progress between the per-rank deliveries of one round, and an event
    recorded on an idle stream (every earlier event done) completing at once."""
    RATES={0:1.0,1:0.3,2:0.6,3:0.3}
    def __init__(self,conn,seed=4141):
        self.conn=conn;self.rnd=random.Random(seed)
        self.finalize=0;self.mixed=0;self.refused=[]
        # Completion guard: per transaction, the ranks whose own state event
        # has completed; a worker release of that transaction, or the next
        # transaction's open (its PREPARE copy follows), needs all four.
        self.done={};self.opened=[];self.violations=[];self.opens_checked=0;self.releases_checked=0
    def observe(self):
        for rank,w in self.conn.workers.items():
            b=w._flash_s5_bridge;tx=b.transaction;ev=b.dispatcher.state_event
            if tx is not None and ev is not None and ev.complete:
                self.done.setdefault(tx.txn_seq,set()).add(rank)
    def progress(self,rank):
        events=[e for e in self.conn.workers[rank].bus.events if not e.complete]
        count=sum(self.rnd.random()<self.RATES[rank] for _ in events)
        for e in events[:count]:e.complete=True
    def tick(self):
        for rank in sorted(self.conn.workers):self.progress(rank)
        self.observe()
    @contextlib.contextmanager
    def installed(self):
        import test_e3_r4_worker_bridge as r4
        import wire_v1 as wire
        skew=self;conn=self.conn
        original=r4.ControllableEvent.record
        def record(event,stream=None):
            original(event,stream)
            earlier=event.bus.events[:event.bus.events.index(event)]
            if all(e.complete for e in earlier) and skew.rnd.random()<0.5:event.complete=True
        ranks=set(conn.workers)
        for rank,w in conn.workers.items():
            bridge=w._flash_s5_bridge
            real_open=w.flash_s5_open;real_release=w.flash_s5_release
            def guarded_open(message,_real=real_open,_rank=rank,_bridge=bridge):
                skew.observe();before={s:set(v) for s,v in skew.done.items()};prior=list(skew.opened)
                result=_real(message);tx=_bridge.transaction
                if tx is not None:
                    earlier=[s for s in prior if s<tx.txn_seq]
                    if earlier:
                        skew.opens_checked+=1;missing=ranks-before.get(max(earlier),set())
                        if missing:skew.violations.append(('open-before-prior-state',_rank,tx.txn_seq,sorted(missing)))
                    if tx.txn_seq not in skew.opened:skew.opened.append(tx.txn_seq)
                return result
            def guarded_release(message,_real=real_release,_rank=rank,_bridge=bridge):
                skew.observe();tx=_bridge.transaction
                if tx is not None:
                    skew.releases_checked+=1;missing=ranks-skew.done.get(tx.txn_seq,set())
                    if missing:skew.violations.append(('release-before-state',_rank,tx.txn_seq,sorted(missing)))
                return _real(message)
            w.flash_s5_open=guarded_open;w.flash_s5_release=guarded_release
        for w in conn.workers.values():
            for name in ('flash_s5_submit','flash_s5_poll','flash_s5_release'):
                real=getattr(w,name)
                def wrapped(message,_real=real):
                    for rank in sorted(conn.workers):
                        if skew.rnd.random()<0.5:skew.progress(rank)
                    return _real(message)
                setattr(w,name,wrapped)
        driver=conn.driver;real_apply=driver._apply_segments
        def apply(values):
            values=tuple(values);command=driver._command
            if command.kind is wire.SegmentKind.STATE_FINALIZE:
                phases={v.cursor.phase for v in values if v.accepted}
                skew.finalize+=1;skew.mixed+=len(phases)>1
            try:return real_apply(values)
            except Exception as exc:
                skew.refused.append(str(exc)[:300]);raise
        driver._apply_segments=apply
        with mock.patch.object(r4.ControllableEvent,'record',record):
            yield self


def run(mode):
    root,failures=overlay_aliases()
    conn=Connected('0' if mode=='off' else '1')
    conn.dispatched=set()
    with contextlib.ExitStack() as scope:
        if mode=='tp-skew':
            conn.skew=scope.enter_context(Skew(conn).installed())
        return _run(mode,conn,root,failures)


def _run(mode,conn,root,failures):
    checks=[]
    def check(name,ok,**details):
        checks.append(dict(name=name,ok=bool(ok),**details));assert ok,name
    check('startup-full-owner-bound',conn.driver._bootstrapped and not conn.driver._split_ready
          and conn.driver.admission_intent() is None and conn.engine._flash_s5_startup_ready is not None)
    cohort=conn.request('cohort',65537 if mode=='running-prefill' else 40,outputs=2048)
    for _ in range(4):conn.step()
    before=cohort.num_output_tokens
    req=conn.request('x'*257 if mode=='long-id' else 'long',197315 if mode=='refuse' else (32721 if mode=='below-threshold' else 32801),outputs=8)
    cohort_prefilling=cohort.num_computed_tokens<cohort.num_prompt_tokens
    injected=False
    pending_checks=0
    reuse=None
    for _ in range(2500):
        policy=getattr(conn.driver,'auto_arm',None)
        phase=getattr(policy,'phase',None)
        if not injected and mode in ('abort-before','reuse') and phase=='drain':
            from vllm.v1.engine.core import EngineCore
            EngineCore.abort_requests(conn.engine,[req.request_id])
            if mode=='reuse':reuse=conn.request(req.request_id,50,outputs=1)
            injected=True
        if not injected and mode=='abort-live' and conn.controller.active:
            from vllm.v1.engine.core import EngineCore
            EngineCore.abort_requests(conn.engine,[req.request_id]);injected=True
        if not injected and mode=='foreign' and phase=='native':
            reuse=conn.request('foreign',50,outputs=1);injected=True
        if not injected and mode=='bad-graph' and phase=='barrier':
            conn.workers[1].model_runner.cudagraph_manager._graphs_captured=False;injected=True
        if not injected and mode=='pending-split' and phase=='split':
            conn.transport.hold=True;injected=True
            pending_checks=len(conn.traces)
        if injected and mode=='pending-split' and conn.transport.hold:
            check('pending-split-no-stock-%d'%conn.ticks,len(conn.traces)==pending_checks and not conn.native)
            if sum(c['name'].startswith('pending-split-no-stock') for c in checks)>=5:conn.transport.deliver_held()
        if not injected and mode=='pending-full' and phase=='native' and conn.driver._split_ready:
            import test_e3_native_storage_replay as replay
            w=conn.workers[1];bank=w.router.banks['decode-split']
            w.router.native_replay(replay.ExecutionDescriptor.from_vllm(replay._Vllm()),
                observed_context=bank.context.context_id,observed_stream=bank.context.stream_token)
            conn.blocked_event=w.bus.events[-1];conn.blocked_event.complete=False;injected=True
        if mode=='pending-full' and conn.blocked_event is not None and conn.driver._transition is not None:
            if pending_checks==0:pending_checks=len(conn.traces)
            check('pending-full-no-stock-%d'%conn.ticks,len(conn.traces)==pending_checks)
            if sum(c['name'].startswith('pending-full-no-stock') for c in checks)>=5:
                conn.blocked_event.complete=True;conn.blocked_event=None
        conn.step()
        policy=getattr(conn.driver,'auto_arm',None)
        if (req.num_output_tokens or (injected and mode in ('abort-before','abort-live','reuse'))) and not conn.driver.active and not getattr(policy,'active',False):break
    if mode=='tp-skew':
        skew=conn.skew
        check('tp-skew-no-refusal',not skew.refused,refused=skew.refused)
        check('tp-skew-finalize-commit-beside-state-finalize',skew.mixed>=1,
              finalize_agreements=skew.finalize,mixed=skew.mixed)
        # Every worker release, and every next chunk's open (its PREPARE copy
        # follows the open), waited for all four ranks' own state events.
        check('tp-skew-no-completion-dependent-action-early',
              not skew.violations and skew.opens_checked>=4*(len(skew.opened)-1)>0
              and skew.releases_checked>=4*len(skew.opened),
              violations=skew.violations[:8],opens_checked=skew.opens_checked,
              releases_checked=skew.releases_checked,transactions=len(skew.opened))
    if mode in ('auto','foreign','pending-split','pending-full','tp-skew'):
        check('ordinary-requests-trigger-native-cycle',len(conn.native)>1 and all(r['uid']==req.request_id for r in conn.native))
        check('native-covers-0-through-n-minus-1',conn.native[0]['start']==0 and conn.native[-1]['end']==32800
              and all(a['end']==b['start'] for a,b in zip(conn.native,conn.native[1:])))
        check('terminal-only-last',sum(r['terminal'] for r in conn.native)==1 and conn.native[-1]['terminal'])
        check('all-rank-full-handback',not conn.driver._split_ready and all(w.controller.route=='full' for w in conn.workers.values()))
        check('decode-continued',cohort.num_output_tokens>before+4)
        check('no-stock-prefill-on-split',not any(row['split'] and (row['prefill'] or req.request_id in row['scheduled']) for row in conn.traces))
        check('one-native-request-per-cycle',getattr(policy,'completed',0)==1 and conn.driver.admission_intent() is None)
        check('no-native-control-client',conn.external_controls==0)
    elif mode=='running-prefill':
        # native_auto_arm._cohort refuses a cohort whose member is still prefilling
        # (num_computed_tokens < num_prompt_tokens), so stock starts the long head
        # beside it: never selected and never on split. The engine records no
        # reason field; the rule's precondition is checked at the head's arrival.
        snap=policy.snapshot()
        check('running-prefill-head-arrived-during-cohort-prefill',cohort_prefilling)
        check('running-prefill-cohort-not-selected',snap['selection_seq']==0 and not conn.native)
        check('running-prefill-never-split',not conn.driver._split_ready and all(not r['split'] for r in conn.traces))
        check('running-prefill-decoder-continues',cohort.num_output_tokens>before)
    elif mode=='off':
        check('default-off-no-auto-or-native',getattr(conn.driver,'auto_arm',None) is None and not conn.native)
        check('default-off-full',not conn.driver._split_ready and all(not r['split'] for r in conn.traces))
    elif mode in ('abort-before','abort-live','reuse','bad-graph','refuse','below-threshold','long-id'):
        if mode not in ('refuse','below-threshold','long-id'):check('failure-case-injected',injected)
        if mode in ('abort-before','reuse','bad-graph','refuse','below-threshold','long-id'):check('no-native-admission',not conn.native)
        check('failure-recovers-full',not conn.driver._split_ready and not conn.driver.active and not policy.active)
        check('failure-decoder-continues',cohort.num_output_tokens>before)
    if mode not in ('abort-before','abort-live','reuse'):
        check('stock-first-token-produced',req.num_output_tokens>0 and list(req.output_token_ids)==[7]*req.num_output_tokens)
    # Cleanup through the real scheduler + lease/free hooks.
    if req.request_id in conn.scheduler.requests:conn.stack.abort(conn.scheduler.requests[req.request_id])
    if reuse is not None and reuse.request_id in conn.scheduler.requests:conn.stack.abort(reuse)
    conn.stack.abort(cohort)
    for _ in range(8):conn.step()
    check('clean-owned-state',not conn.scheduler.requests and not conn.controller.active and not conn.stack.leases.held_blocks() and not conn.driver.active)
    import torch
    check('no-cuda-initialized',not torch.cuda.is_initialized())
    return dict(verdict='PASS',checks=checks,mode=mode,native=conn.native,ticks=conn.ticks,
        trace=conn.traces,cohort_tokens=cohort.num_output_tokens,request_tokens=list(req.output_token_ids),
        rpc_methods=[e[0] for e in conn.sender.envelopes],geometry=conn.stack.geometry,
        alias_failures=failures,cuda_initialized=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=('auto','off','abort-before','abort-live','reuse','foreign','bad-graph','refuse','pending-split','pending-full','running-prefill','below-threshold','long-id','tp-skew'),default='auto');a=p.parse_args()
    try:r=run(a.mode);rc=0
    except Exception as exc:r=dict(verdict='FAIL',mode=a.mode,error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc());rc=1
    r.update(schema='native41-connected-cpu/1',gpu_execution=False,box_evidence=False,probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with a.out.open('x') as f:json.dump(r,f,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps({k:r.get(k) for k in ('verdict','mode','error_type','error')}));return rc
if __name__=='__main__':raise SystemExit(main())
