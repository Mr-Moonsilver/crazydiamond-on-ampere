"""E2 CPU binding/source tests. Injected waits are not CUDA completion evidence."""
import ast
from collections import deque
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle
import sys
import textwrap
import threading
from types import MethodType, SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))
from block_mirror import AllocationJournal, CpuTables, WorkerMirror
from retention_contract import ContractError, Pending
import mirror_hooks

LAYERS = {f'qsa{i}': 0 for i in range(6)}
SOURCES = json.loads((ROOT / 'patches/s5_engine/mirror_source.json').read_text())['modules']
BP = 'v1/core/block_pool.py'
SC = 'v1/core/sched/scheduler.py'
BT = 'v1/worker/gpu/block_table.py'
MP = 'v1/executor/multiproc_executor.py'


def load(rel, method, namespace=None):
    ns = {} if namespace is None else namespace
    code = 'from __future__ import annotations\n' + SOURCES[rel]['methods'][method]['source']
    exec(compile(code, rel + ':' + method, 'exec'), ns)
    return ns[method]


def output(new=(), cached=(), scheduled=None, updates=None, copies=(), retired=(), preempted=()):
    return NS(scheduled_new_reqs=[NS(req_id=r, block_ids=b, num_computed_tokens=n) for r,b,n in new],
              scheduled_cached_reqs=NS(req_ids=[r for r,n,b in cached],
                                      num_computed_tokens=[n for r,n,b in cached],
                                      new_block_ids=[b for r,n,b in cached]),
              num_scheduled_tokens=scheduled or {}, block_table_updates=updates,
              kv_cache_block_copies=[NS(src_block_id=s, dst_block_id=d) for s,d in copies],
              finished_req_ids=set(retired), preempted_req_ids=set(preempted))


def setup():
    journal = AllocationJournal(1, 100)
    mirror = WorkerMirror(1, 100, LAYERS, page_tokens=4, arena_pages=2)
    tables = CpuTables([1], 100)
    return journal, mirror, tables


def first(j, m, t, count=3):
    j.allocated([1]); t.append(0, ([1],), True)
    f = j.frame(output(new=[('a', ([1],), 0)], scheduled={'a':count}), lambda _: ([1],))
    m.enter(f, {'a':0}, t)
    return f


def consume(m, rid='a', layer='qsa0'):
    m.producer_ready(layer, m.sequence, lambda: None)
    a = m.acquire(rid, layer)
    spans = []
    while a.next_page < len(a.sources):
        b, c = a.begin_fill(); spans.append((b.lo,b.hi))
        a.complete_fill(c); a.complete_attention(a.begin_attention(b))
    a.release()
    return spans


class BindingTests(unittest.TestCase):
    def test_frame_frozen_before_future_schedule_and_pickle(self):
        j,m,t = setup(); j.allocated([1]); rows=([1],)
        o=output(new=[('a', rows, 0)], scheduled={'a':3})
        frame=j.frame(o, lambda _: rows)
        rows[0][0]=9; o.num_scheduled_tokens['a']=10; j.allocated([1])
        self.assertEqual(frame.rows[0].tables, ((1,),))
        self.assertEqual(frame.allocations, ((1,1),))
        self.assertEqual(pickle.loads(pickle.dumps(frame)), frame)
        t.append(0, ([1],), True); m.enter(frame, {'a':0}, t)
        self.assertEqual(consume(m), [(0,1)])

    def test_append_replace_fragmentation_and_empty_append(self):
        t=CpuTables([1,2], 10)
        t.append(0, ([4,2],[3]), True)
        t.append(0, ([],[1]), False)
        self.assertEqual(t.rows[0], ((4,2),(6,7,2,3)))
        t.append(0, ([7],[]), True)
        self.assertEqual(t.rows[0], ((7,),()))
        before=t.rows[0]
        with self.assertRaises(ContractError): t.append(0, (list(range(11)),[]), False)
        self.assertEqual(t.rows[0],before)

    def test_worker_mapping_must_equal_authoritative_snapshot(self):
        j,m,t=setup(); j.allocated([1,2]); t.append(0, ([2],), True)
        f=j.frame(output(new=[('a',([1],),0)],scheduled={'a':2}),lambda _:([1],))
        with self.assertRaisesRegex(ContractError,'differs'):m.enter(f,{'a':0},t)
        self.assertTrue(m.failed)
        with self.assertRaises(Pending):m.enter(f,{'a':0},t)

    def test_unobserved_or_null_prefix_cannot_become_source(self):
        j,m,t=setup()
        with self.assertRaises(ContractError):j.frame(output(new=[('a',([1],),0)],scheduled={'a':2}),lambda _:([1],))
        t.append(0,([0],),True)
        f=j.frame(output(new=[('a',([0],),0)],scheduled={'a':2}),lambda _:([0],))
        with self.assertRaises(ContractError):m.enter(f,{'a':0},t)

    def test_lazy_decode_writes_wait_for_actual_producer_fence(self):
        j,m,t=setup(); first(j,m,t)
        with self.assertRaises(Pending):m.acquire('a','qsa0')
        m.leave()  # ordinary graph decode need not add a CPU synchronization
        f=j.frame(output(cached=[('a',3,None)],scheduled={'a':1}),lambda _:([1],))
        m.enter(f,{'a':0},t)
        with self.assertRaises(Pending):m.acquire('a','qsa0')
        p=m.pool.pages[('qsa0',1)]
        self.assertEqual((p.valid,p.version,p.writer),(4,2,(2,'qsa0')))
        observed=[]
        m.producer_ready('qsa0',2,lambda:observed.append('event synchronized'))
        self.assertEqual(observed,['event synchronized']);self.assertIsNone(p.writer)
        consume(m);m.leave()

    def test_shared_prefix_waits_for_prior_enqueued_write(self):
        j,m,t=setup();first(j,m,t,4);m.leave()
        j.allocated([2]);t.append(1,([1,2],),True)
        f=j.frame(output(new=[('b',([1,2],),4)],scheduled={'b':1}),lambda _:([1,2],))
        m.enter(f,{'a':0,'b':1},t)
        with self.assertRaises(Pending):m.acquire('b','qsa0')
        self.assertEqual(consume(m,'b'),[(0,2)])
        self.assertEqual(m.pool.pages[('qsa0',1)].version,1)
        self.assertEqual(m.pool.held_count(),0)

    def test_cow_private_destination_versions_and_count_line(self):
        j,m,t=setup();first(j,m,t);m.leave()
        j.allocated([2]);t.append(1,([2],),True)
        f=j.frame(output(new=[('b',([2],),3)],scheduled={'b':1},copies=[(1,2)]),lambda _:([2],))
        m.enter(f,{'a':0,'b':1},t)
        for layer in LAYERS:
            self.assertEqual(m.pool.pages[(layer,1)].valid,3)
            self.assertEqual(m.pool.pages[(layer,2)].valid,4)
            self.assertEqual(m.pool.pages[(layer,2)].version,2)
        self.assertEqual(m.cow_line(),'FLASH_E2_COW seq=2 pairs=1 views=6 cumulative_pairs=1 cumulative_views=6')
        consume(m,'b');m.leave()

    def test_zero_token_cow_and_retirement_are_not_lost(self):
        j,m,t=setup();first(j,m,t);m.leave()
        j.allocated([2]);t.append(1,([2],),True)
        f=j.frame(output(new=[('b',([2],),3)],copies=[(1,2)],retired=['a']),lambda _:None)
        m.enter(f,{'b':1},t);self.assertEqual(m.sequence,2)
        self.assertEqual(m.cow_pairs,1);self.assertNotIn('a',m.requests)
        self.assertEqual(m.pool.pages[('qsa0',2)].valid,3)
        m.leave()
        f=j.frame(output(cached=[('b',3,None)],scheduled={'b':1}),lambda _:([2],))
        m.enter(f,{'b':1},t);consume(m,'b');m.leave()

    def test_allocation_reuse_increments_incarnation_without_touch_refs(self):
        j,m,t=setup();first(j,m,t);consume(m);old=m.pool.pages[('qsa0',1)].key;m.leave()
        j.allocated([1]); t.append(0,([1],),True)
        f=j.frame(output(new=[('b',([1],),0)],scheduled={'b':1},retired=['a']),lambda _:([1],))
        m.enter(f,{'b':0},t);new=m.pool.pages[('qsa0',1)].key
        self.assertEqual(new.incarnation,old.incarnation+1)
        self.assertEqual(m.requests['b'].generation,2)
        self.assertEqual(m.pool.pages[('qsa0',1)].valid,1)
        with self.assertRaises(ContractError):m.pool.page(old)

    def test_cancellation_preemption_and_same_id_resume(self):
        for field in ('retired','preempted'):
            with self.subTest(field=field):
                j,m,t=setup();first(j,m,t);consume(m);key=m.requests['a'];m.leave()
                f=j.frame(output(**{field:['a']}),lambda _:None)
                m.enter(f,{},t);m.leave();self.assertFalse(m.requests)
                j.allocated([1]);t.append(0,([1],),True)
                f=j.frame(output(new=[('a',([1],),0)],scheduled={'a':1}),lambda _:([1],))
                m.enter(f,{'a':0},t)
                self.assertGreater(m.requests['a'].generation,key.generation)

    def test_pending_fill_or_consumer_forbids_rpc_return_and_next_reuse(self):
        for kind in ('copy','consume'):
            j,m,t=setup();first(j,m,t);m.producer_ready('qsa0',1,lambda:None)
            a=m.acquire('a','qsa0');b,c=a.begin_fill()
            if kind=='consume':a.complete_fill(c);c=a.begin_attention(b)
            with self.assertRaises(Pending):m.leave()
            with self.assertRaises(Pending):m.enter(replace(j.frame(output(),lambda _:None),sequence=2),{},t)
            (a.complete_fill if kind=='copy' else a.complete_attention)(c)
            a.cancel(m.requests['a']);a.release();m.leave()

    def test_fence_failure_is_quarantined_and_shutdown_cannot_fake_clean(self):
        j,m,t=setup();first(j,m,t)
        def failed():raise RuntimeError('injected device loss')
        with self.assertRaises(RuntimeError):m.producer_ready('qsa0',1,failed)
        self.assertTrue(m.failed)
        with self.assertRaises(Pending):m.shutdown(lambda:None)

    def test_shutdown_drains_metadata_and_preserves_fixed_storage(self):
        j,m,t=setup();first(j,m,t);consume(m);m.leave()
        states=m.shutdown(lambda:None)
        for name in ('active_owners','source_leases','pending_dma','pending_consumers',
                     'registered_requests','valid_arena_pages','pending_producer_pages'):
            self.assertEqual(states[name],0)
        self.assertEqual(states['incremental_kv_bytes'],0)
        self.assertEqual(states['reused_arena_bytes'],2*4*1024)

    def test_reordered_requests_keep_exact_slot_mapping(self):
        j,m,t=setup();j.allocated([1,2]);t.append(0,([1],),True);t.append(1,([2],),True)
        f=j.frame(output(new=[('a',([1],),0),('b',([2],),0)],scheduled={'b':2,'a':3}),
                  lambda rid:([1 if rid=='a' else 2],))
        m.enter(f,{'a':0,'b':1},t)
        consume(m,'b');consume(m,'a');m.leave()

    def test_cow_filter_matches_full_updates_without_double_append(self):
        j,m,t=setup();first(j,m,t);m.leave();j.allocated([2])
        t.append(0,([2],),True)
        o=output(cached=[('a',3,([2],))],scheduled={'a':1},updates={'a':([2],)},copies=[(1,2)])
        f=j.frame(o,lambda _:([2],));m.enter(f,{'a':0},t)
        self.assertEqual(f.copy_destinations,((2,),));self.assertEqual(m.cow_pairs,1)
        consume(m);m.leave()

    def test_sequence_epoch_and_expansion_rejected(self):
        for change in ({'sequence':2},{'epoch':2}):
            j,m,t=setup();j.allocated([1]);t.append(0,([1],),True)
            f=j.frame(output(new=[('a',([1],),0)],scheduled={'a':1}),lambda _:([1],))
            with self.assertRaises(ContractError):m.enter(replace(f,**change),{'a':0},t)
        j,m,t=setup();first(j,m,t);m.leave()
        with self.assertRaises(ContractError):m.enter(j.frame(output(),lambda _:None),{},CpuTables([2],10))


class TableScopeTests(unittest.TestCase):
    """q16f (25 Oct 16:32): exact-dense died on the 8th RPC with 'worker CPU table differs from scheduler
    snapshot'. The engine has 39 cache groups (37 MambaSpec, one circular buffer, one QSA group 38); the pinned
    Mamba align-mode manager retires an older state block to NULL in the scheduler's req_to_blocks only, while the
    worker's block table stays append-only. E2 reads only its QSA source groups: those remain exact, any other
    group may diverge and is counted and reported once per group, without block IDs."""
    def two_groups(self):
        j = AllocationJournal(1, 100); m = WorkerMirror(1, 100, {k: 1 for k in LAYERS}, page_tokens=4, arena_pages=2)
        t = CpuTables([1, 1], 100); j.allocated([5, 1]); t.append(0, ([5], [1]), True)
        f = j.frame(output(new=[('a', ([5], [1]), 0)], scheduled={'a': 3}), lambda _: ([5], [1]))
        m.enter(f, {'a': 0}, t); m.leave()
        return j, m, t

    def test_non_source_group_retired_to_null_is_tolerated_counted_and_noted_once(self):
        j, m, t = self.two_groups()
        f = j.frame(output(cached=[('a', 3, None)], scheduled={'a': 1}), lambda _: ([0], [1]))
        m.enter(f, {'a': 0}, t)
        self.assertFalse(m.failed)
        self.assertEqual(consume(m, 'a', 'qsa0'), [(0, 1)]); m.leave()
        c = m.counters()
        self.assertEqual((c['table_divergences'], c['divergent_groups']), (1, [0]))
        notes = m.take_divergence_notes()
        self.assertEqual(notes, [dict(sequence=2, slot=0, groups=[0], total_groups=1,
                                      new=[dict(group=0, worker=1, scheduler=1, differ=1, scheduler_null=1, first=0)])])
        self.assertEqual(m.take_divergence_notes(), [])
        j.allocated([2]); t.append(0, ([], [2]), False)
        f = j.frame(output(cached=[('a', 4, ([], [2]))], scheduled={'a': 1}), lambda _: ([0], [1, 2]))
        m.enter(f, {'a': 0}, t); m.leave()
        self.assertEqual(m.counters()['table_divergences'], 2)
        self.assertEqual(m.take_divergence_notes(), [])
        self.assertNotIn('5', json.dumps(notes))

    def test_source_group_divergence_is_still_a_hard_failure_with_metadata_only(self):
        j, m, t = self.two_groups(); j.allocated([7])
        f = j.frame(output(cached=[('a', 3, None)], scheduled={'a': 1}), lambda _: ([5], [7]))
        with self.assertRaisesRegex(ContractError, 'differs.*QSA group') as cm: m.enter(f, {'a': 0}, t)
        self.assertTrue(m.failed)
        delta = json.loads(str(cm.exception).split('QSA group ', 1)[1])
        self.assertEqual(delta, dict(group=1, worker=1, scheduler=1, differ=1, scheduler_null=0, first=0))
        self.assertEqual(m.counters()['table_divergences'], 0)
        with self.assertRaises(Pending): m.enter(f, {'a': 0}, t)

    def test_group_count_and_missing_slot_rejected(self):
        j, m, t = self.two_groups()
        f = j.frame(output(cached=[('a', 3, None)], scheduled={'a': 1}), lambda _: ([5], [1], ()))
        with self.assertRaisesRegex(ContractError, 'differs.*2 vs 3 groups'): m.enter(f, {'a': 0}, t)
        j, m, t = self.two_groups()
        f = j.frame(output(cached=[('a', 3, None)], scheduled={'a': 1}), lambda _: ([5], [1]))
        with self.assertRaisesRegex(ContractError, 'differs.*no vs 2 groups'): m.enter(f, {'a': 1}, t)


class SourceTests(unittest.TestCase):
    def test_exact_pinned_source_excerpts(self):
        for rel, entry in SOURCES.items():
            for method in entry['methods'].values():
                self.assertEqual(hashlib.sha256(method['source'].encode()).hexdigest(),method['sha256'])
            path=ROOT/'reference/vllm-src'/rel
            if not path.exists():continue
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),entry['sha256'])
            cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name==entry['class'])
            for name,method in entry['methods'].items():
                node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
                excerpt=textwrap.dedent(''.join(path.read_text().splitlines(keepends=True)[node.lineno-1:node.end_lineno]))
                self.assertEqual(excerpt,method['source'])

    def test_actual_writable_predicate_why_an_added_pin_is_wrong(self):
        fn=load(BP,'is_block_writable');b=NS(is_null=False,ref_cnt=1,block_hash=None)
        self.assertTrue(fn(None,b));b.ref_cnt+=1;self.assertFalse(fn(None,b))
        b.ref_cnt=1;b.block_hash='cached';self.assertFalse(fn(None,b))

    def test_actual_allocation_refcounts_identical_with_journal_on_or_off(self):
        class Queue:
            def __init__(self,blocks):self.blocks=list(blocks)
            def popleft_n(self,n):out=self.blocks[:n];self.blocks=self.blocks[n:];return out
        src='class BlockPool:\n'+textwrap.indent(SOURCES[BP]['methods']['get_new_blocks']['source'],'    ')
        # Exercise the exact transformed method without importing vLLM/CUDA.
        src=mirror_hooks.replace_method(src,'BlockPool','get_new_blocks',lambda s:mirror_hooks.once(s,
            '        return ret\n',
            '        if self._flash_e2_journal is not None:\n'
            '            self._flash_e2_journal.allocated(b.block_id for b in ret)\n'
            '        return ret\n'))
        ns={};exec('from __future__ import annotations\n'+src,ns)
        observed=[]
        for enabled in (False,True):
            blocks=[NS(block_id=i,ref_cnt=0) for i in range(1,4)]
            pool=NS(free_block_queue=Queue(blocks),_reuse_watchers={},enable_caching=True,
                    _evict_first=set(),metrics_collector=None,
                    _maybe_evict_cached_block=lambda _:None,
                    _flash_e2_journal=AllocationJournal(1,10) if enabled else None)
            pool.get_num_free_blocks=lambda:len(pool.free_block_queue.blocks)
            ret=ns['BlockPool'].get_new_blocks(pool,2)
            observed.append([(b.block_id,b.ref_cnt) for b in ret])
            if enabled:self.assertEqual(pool._flash_e2_journal.pending,{1:1,2:1})
        self.assertEqual(observed,[[(1,1),(2,1)]]*2)

    def test_cow_invariant_checker_does_not_grade_copy_frequency(self):
        import cow_invariants
        inv=dict(src_mutated=0,controls_changed=0,src_changed_by_copy=0,prefix_unknown=0,
                 src_rechecks=2,controls=2,src_reallocated=1)
        for count in (1,10000):
            lines=[f'{rank} QSA host-page CoW verify copies {count}, after 0 (equal), invariants {inv}'
                   for rank in sorted(cow_invariants.RANKS)]
            self.assertTrue(cow_invariants.check(lines)[0])

    def test_actual_deferred_free_needs_future_nonempty_fence(self):
        released=[];s=NS(defer_block_free=True,processed_step_seq=0,sched_step_seq=0,
                         deferred_frees=deque(),kv_cache_manager=NS(block_pool=NS(free_blocks=lambda bs:released.extend(bs))))
        defer=load(SC,'_free_cow_retained_blocks');drain=load(SC,'_drain_deferred_frees')
        defer(s,[1,2],1);drain(s);self.assertFalse(released)
        # A zero-token output does not advance processed_step_seq in the image.
        drain(s);self.assertFalse(released)
        s.processed_step_seq=1;drain(s);self.assertEqual(released,[1,2])

    def test_actual_request_free_default_and_deferred_routes(self):
        freed=[];popped=[]
        mgr=NS(free=lambda r:freed.append(r),pop_blocks_for_free=lambda r:popped.append(r) or [1,2])
        s=NS(defer_block_free=False,processed_step_seq=0,sched_step_seq=3,deferred_frees=deque(),kv_cache_manager=mgr)
        s._request_blocks_can_be_freed=MethodType(load(SC,'_request_blocks_can_be_freed'),s)
        fn=load(SC,'_free_request_blocks');r=NS(last_sched_seq=3)
        fn(s,r);self.assertEqual(freed,[r]);self.assertFalse(popped)
        s.defer_block_free=True;fn(s,r);self.assertEqual(list(s.deferred_frees),[(3,[1,2])])

    def test_actual_block_table_writes_match_cpu_mirror(self):
        # Minimal CPU doubles for staged writer and ndarray two-index access.
        class Array:
            def __init__(self):self.data={}
            def __getitem__(self,k):return self.data.get(k,0)
            def __setitem__(self,k,v):self.data[k]=v
        class Writer:
            def __init__(self):self.gpu=NS(shape=(2,20));self.rows={}
            def stage_write(self,slot,start,ids):
                self.rows.setdefault(slot,[])
                row=self.rows[slot]
                row[start:start+len(ids)]=ids
        writers=[Writer(),Writer()]
        runner=NS(num_kv_cache_groups=2,num_blocks=NS(np=Array()),blocks_per_kv_block=[1,2],block_tables=writers)
        fn=load(BT,'append_block_ids');mirror=CpuTables([1,2],20)
        for blocks,overwrite in ((([4,1],[5]),True), (([3],[2]),False), (([7],[]),True)):
            fn(runner,0,blocks,overwrite);mirror.append(0,blocks,overwrite)
            actual=tuple(tuple(w.rows[0][:runner.num_blocks.np[g,0]]) for g,w in enumerate(writers))
            self.assertEqual(actual,mirror.rows[0])

    def test_exact_worker_rpc_cannot_execute_next_reuse_until_dma_returns(self):
        entered=threading.Event();finish=threading.Event();overwritten=threading.Event();calls=[]
        def dma():
            entered.set()
            if not finish.wait(2):raise RuntimeError('test timeout')
            calls.append('DMA and attention completed')
        def reuse():overwritten.set();calls.append('later physical overwrite')
        worker=NS(rank=0,worker=NS(dma=dma,reuse=reuse),handle_output=lambda _:None)
        fn=load(MP,'_execute_worker_rpc',{'logger':NS(exception=lambda _:None)})
        def run():
            fn(worker,('dma',(),{},0));fn(worker,('reuse',(),{},0))
        thread=threading.Thread(target=run);thread.start()
        try:
            self.assertTrue(entered.wait(1));self.assertFalse(overwritten.is_set())
        finally:
            finish.set();thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(calls,['DMA and attention completed','later physical overwrite'])

    def test_metadata_transforms_compile_and_preserve_refcount_methods(self):
        transform={BP:mirror_hooks.block_pool,SC:mirror_hooks.scheduler,
                   'v1/core/sched/output.py':mirror_hooks.scheduler_output,BT:mirror_hooks.block_tables}
        for rel,fn in transform.items():
            path=ROOT/'reference/vllm-src'/rel
            if not path.exists():self.skipTest('optional full reference source absent')
            before=path.read_text();after=fn(before);compile(after,rel,'exec')
            self.assertNotEqual(before,after)
            if rel==BP:
                for name in ('is_block_writable','touch','free_blocks','unpin_blocks'):
                    def method(src):
                        cls=next(n for n in ast.parse(src).body if isinstance(n,ast.ClassDef) and n.name=='BlockPool')
                        n=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
                        return ast.get_source_segment(src,n)
                    self.assertEqual(method(before),method(after))
            if rel==SC:
                self.assertEqual(after.count('self.defer_block_free = True'),before.count('self.defer_block_free = True'))


if __name__=='__main__':unittest.main()
