"""CPU runner seam tests. No real runner, CUDA or source-install claim."""
from dataclasses import replace
import ast
import json
from pathlib import Path
import sys
import textwrap
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))
from runner_lifecycle import RunnerLifecycle, execute_with_e2
from retention_contract import ContractError, Pending
import runner_hooks
from test_e2_mirror import setup, output, LAYERS
from test_e2_arena_dma import Backend, Buffer, ArenaDMA, TensorFacts


class CpuSequence(list):
    device = 'cpu'


def batch(ids=('a',), counts=(4,), computed=(0,), slots=(0,)):
    offsets = [0]
    for c in counts:
        offsets.append(offsets[-1] + c)
    return NS(req_ids=list(ids), num_reqs=len(ids), idx_mapping_np=list(slots),
              num_scheduled_tokens=list(counts), num_computed_tokens_np=list(computed),
              query_start_loc_np=offsets,
              seq_lens_cpu_upper_bound=CpuSequence(c+s for c,s in zip(computed, counts)))


def fixture(enabled=True):
    j,m,t = setup(); m.arena.arena_pages = 23
    backend = Backend()
    hosts = {name: Buffer((100,1,4,512),'cpu',1_000_000*(i+1)) for i,name in enumerate(LAYERS)}
    dma = ArenaDMA(m, hosts, backend)
    lines = []; switch = [enabled]
    bridge = RunnerLifecycle(m, dma, lambda:switch[0], lines.append)
    j.allocated([1]); t.append(0,([1],),True)
    o = output(new=[('a',([1],),0)],scheduled={'a':4})
    o.flash_e2_frame = j.frame(o,lambda _:([1],))
    arena = Buffer((23,1,4,512),backend.device,8_000_000)
    table = NS(facts=TensorFacts(backend.device,(100,),(1,),'torch.int32',4,False,True,9_000_000))
    args = (object(),object(),object(),table,object(),0,arena,False,object())
    return j,m,t,backend,dma,lines,switch,bridge,o,args


def candidate(*args, e2):
    e2.route('dense_dma',1)
    e2.fill(args[6].first(1),0,1,args[3])
    e2.consume_begin();e2.consume_end()


class LifecycleTests(unittest.TestCase):
    def test_real_frame_all_layers_drained_and_cow_logged_on_success(self):
        j,m,t,b,d,lines,switch,x,o,args = fixture()
        with x.rpc(o):
            x.updated({'a':0},t);x.batch(batch())
            for name in LAYERS:
                x.staged(name,0,0,4,4,args,lambda *a:self.fail('off called'),candidate)
            self.assertFalse(lines)
        self.assertEqual(d.stats['dense_dma'],6)
        self.assertEqual(m.pool.held_count(),0);self.assertFalse(m.active)
        self.assertIsNone(x.frame)
        self.assertEqual(lines,['FLASH_E2_COW seq=1 pairs=0 views=0 cumulative_pairs=0 cumulative_views=0'])

    def test_off_has_same_metadata_and_cow_without_any_gpu_fence(self):
        j,m,t,b,d,lines,switch,x,o,args = fixture(False)
        calls = []
        with x.rpc(o):
            x.updated({'a':0},t);x.batch(batch())
            for name in LAYERS:
                x.staged(name,0,0,4,4,args,lambda *a:calls.append(a),candidate)
        self.assertEqual(calls,[args]*6);self.assertFalse(b.events)
        self.assertTrue(all(v==1 for v in x.off_calls.values()))
        self.assertIn('pairs=0 views=0',lines[0])
        self.assertEqual(m.sequence,1)

    def test_switch_sampled_once_not_between_layers(self):
        j,m,t,b,d,lines,switch,x,o,args = fixture()
        with x.rpc(o):
            x.updated({'a':0},t);x.batch(batch());switch[0]=False
            x.staged('qsa0',0,0,4,4,args,lambda *a:self.fail('mid-RPC switch'),candidate)
        self.assertEqual(d.stats['dense_dma'],1)

    def test_prepared_batch_reorder_is_valid_but_wrong_slot_or_length_fails(self):
        for defect in ('slot','length','computed','count','offset','ids','gpu-seq'):
            with self.subTest(defect=defect):
                j,m,t,b,d,lines,switch,x,o,args = fixture()
                v=batch()
                if defect=='slot':v.idx_mapping_np=[1]
                elif defect=='length':v.seq_lens_cpu_upper_bound[0]=5
                elif defect=='computed':v.num_computed_tokens_np=[1]
                elif defect=='count':v.num_scheduled_tokens=[3]
                elif defect=='offset':v.query_start_loc_np=[0,3]
                elif defect=='ids':v.req_ids=['not-a']
                # The device test uses a list-like object, never a GPU tensor.
                if defect=='gpu-seq':
                    v.seq_lens_cpu_upper_bound=CpuSequence([4]);v.seq_lens_cpu_upper_bound.device='cuda:0'
                with self.assertRaises(ContractError):
                    with x.rpc(o):x.updated({'a':0},t);x.batch(v)
                self.assertTrue(m.failed);self.assertFalse(lines)
        j,m,t,b,d,lines,switch,x,o,args = fixture(False)
        j.allocated([2]);t.append(1,([2],),True)
        # Include both allocations in the first immutable frame for this worker.
        f=o.flash_e2_frame
        second=replace(f.rows[0],request_id='b',tables=((2,),))
        o.flash_e2_frame=replace(f,allocations=((1,1),(2,1)),rows=(f.rows[0],second),restarted=('a','b'))
        with x.rpc(o):
            x.updated({'a':0,'b':1},t)
            x.batch(batch(('b','a'),(4,4),(0,0),(1,0)))
            self.assertEqual([r.request_id for r in x.batch_rows],['b','a'])

    def test_cow_observation_matches_real_filtered_pairs_in_both_switch_states(self):
        for enabled in (False,True):
            j,m,t,b,d,lines,switch,x,o,args=fixture(enabled)
            with x.rpc(o):x.updated({'a':0},t);x.batch(batch())
            j.allocated([2]);t.append(1,([2],),True)
            o=output(new=[('b',([2],),3)],scheduled={'b':1},copies=[(1,2)])
            o.flash_e2_frame=j.frame(o,lambda _:([2],))
            with x.rpc(o):
                x.actual_cow(1,6)
                x.updated({'a':0,'b':1},t);x.batch(batch(('b',),(1,),(3,),(1,)))
            self.assertEqual(lines[-1],'FLASH_E2_COW seq=2 pairs=1 views=6 cumulative_pairs=1 cumulative_views=6')

    def test_missing_or_duplicate_cow_observation_does_not_pass(self):
        for actual in ('extra','duplicate','missing'):
            j,m,t,b,d,lines,switch,x,o,args=fixture()
            if actual=='missing':
                with x.rpc(o):x.updated({'a':0},t);x.batch(batch())
                j.allocated([2]);t.append(1,([2],),True)
                o=output(new=[('b',([2],),3)],scheduled={'b':1},copies=[(1,2)])
                o.flash_e2_frame=j.frame(o,lambda _:([2],))
            with self.assertRaises(ContractError):
                with x.rpc(o):
                    if actual=='extra':x.actual_cow(1,6)
                    if actual=='duplicate':x.actual_cow(0,6);x.actual_cow(0,6)
                    x.updated({'a':0,'b':1} if actual=='missing' else {'a':0},t)
            self.assertTrue(m.failed)

    def test_zero_token_rpc_still_consumes_frame_and_retirement(self):
        j,m,t,b,d,lines,switch,x,o,args=fixture()
        with x.rpc(o):x.updated({'a':0},t);x.batch(batch())
        o=output(retired=['a']);o.flash_e2_frame=j.frame(o,lambda _:())
        with x.rpc(o):x.updated({},t)
        self.assertEqual(m.sequence,2);self.assertFalse(m.requests)
        self.assertIn('seq=2 pairs=0',lines[-1]);self.assertFalse(b.events)

    def test_rpc_exception_quarantines_and_cannot_report_completed(self):
        j,m,t,b,d,lines,switch,x,o,args=fixture()
        with self.assertRaisesRegex(RuntimeError,'forward'):
            with x.rpc(o):
                x.updated({'a':0},t);x.batch(batch());raise RuntimeError('forward')
        self.assertTrue(m.failed);self.assertIsNotNone(x.frame);self.assertFalse(lines)
        with self.assertRaises(Pending):
            with x.rpc(o):pass

    def test_early_return_or_nested_rpc_cannot_release_ownership(self):
        for action in ('no-update','no-batch','nested'):
            j,m,t,b,d,lines,switch,x,o,args=fixture()
            with self.assertRaises((ContractError,Pending)):
                with x.rpc(o):
                    if action=='no-update':pass
                    elif action=='no-batch':x.updated({'a':0},t)
                    else:
                        with x.rpc(o):pass
            self.assertFalse(lines)

    def test_staged_metadata_mismatch_rejected_before_any_dma(self):
        for best,start,end,length in ((1,0,4,4),(0,1,4,4),(0,0,3,4),(0,0,4,5)):
            j,m,t,b,d,lines,switch,x,o,args=fixture()
            with self.assertRaises(ContractError):
                with x.rpc(o):
                    x.updated({'a':0},t);x.batch(batch())
                    x.staged('qsa0',best,start,end,length,args,lambda *a:None,candidate)
            self.assertFalse(b.events)

    def test_shutdown_waits_before_retirement_no_gpu_storage_is_freed(self):
        j,m,t,b,d,lines,switch,x,o,args=fixture(False)
        with x.rpc(o):x.updated({'a':0},t);x.batch(batch())
        waited=[]
        def wait():
            self.assertTrue(m.requests);waited.append(True)
        result=x.shutdown(wait)
        self.assertEqual(waited,[True]);self.assertEqual(result['pending_producer_pages'],0)
        self.assertFalse(m.requests);self.assertEqual(len(d.hosts),6)
        self.assertTrue(lines[-1].startswith('FLASH_E2_DRAIN '))

    def test_decorator_off_and_dummy_forward_every_argument_once(self):
        calls=[]
        @execute_with_e2
        def method(*args):calls.append(args);return 'same-result'
        runner=NS()
        self.assertEqual(method(runner,'output','pp',False,False,False,0,False),'same-result')
        runner._flash_e2_lifecycle=NS(rpc=lambda _:self.fail('dummy entered lease'))
        self.assertEqual(method(runner,'dummy','pp',True,True,True,22,True),'same-result')
        self.assertEqual(len(calls),2)
        self.assertEqual(calls[1][1:],('dummy','pp',True,True,True,22,True))

    def test_decorator_real_executes_in_rpc_and_preserves_return(self):
        j,m,t,b,d,lines,switch,x,o,args=fixture(False)
        runner=NS(_flash_e2_lifecycle=x)
        sentinel=object()
        @execute_with_e2
        def method(self,output,intermediate,*rest):
            self._flash_e2_lifecycle.updated({'a':0},t)
            self._flash_e2_lifecycle.batch(batch())
            self.assert_unused = intermediate
            return sentinel
        self.assertIs(method(runner,o,'pp'),sentinel)
        self.assertEqual(runner.assert_unused,'pp');self.assertEqual(m.sequence,1)
        self.assertEqual(len(lines),1)


class SourceHookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources=json.loads((ROOT/'patches/s5_engine/runner_source.json').read_text())
        cls.runner=cls.sources['v1/worker/gpu/model_runner.py']['methods']
        cls.synthetic='class GPUModelRunner(LoRAModelRunnerMixin):\n'+''.join(cls.runner.values())

    def test_source_hooks_reverse_to_original_byte_for_byte(self):
        s=runner_hooks.runner_execution(self.synthetic)
        ast.parse(s)
        restored=s.replace(runner_hooks.IMPORT,'').replace('    @execute_with_e2\n','')
        for insertion in (runner_hooks.UPDATED,runner_hooks.BATCH,runner_hooks.COW,runner_hooks.SHUTDOWN):
            self.assertEqual(restored.count(insertion),1);restored=restored.replace(insertion,'')
        self.assertEqual(restored,self.synthetic)
        qsa=self.sources['models/qwen4_exp/nvidia/qsa.py']['methods']['_qsa_staged_prefill']
        q=runner_hooks.qsa_staged(qsa);ast.parse(q)
        self.assertEqual(q.replace(runner_hooks.STAGED,''),qsa)
        with self.assertRaises(ValueError):runner_hooks.runner_execution(s)
        with self.assertRaises(ValueError):runner_hooks.qsa_staged(q)

    def method(self,name):
        source=runner_hooks.runner_execution(self.synthetic)
        node=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name==name)
        method='\n'.join(source.splitlines()[node.lineno-1:node.end_lineno])
        ns={}
        exec('from __future__ import annotations\n'+textwrap.dedent(method),ns)
        return ns[name]

    def test_actual_pinned_cow_filter_observed_before_verify_branch(self):
        calls=[]
        runner=NS(_qsa_host_views=[NS(shape=(100,1,4,512)) for _ in range(6)],
                  _qsa_host_groups=[0],_qsa_cow_verify=True,_qsa_srcs=set(),
                  kv_cache_config=NS(num_blocks=100),
                  _flash_e2_lifecycle=NS(actual_cow=lambda p,v:calls.append(('observed',p,v))),
                  _qsa_verify_step=lambda views,ids,pairs,*rest:calls.append(('copied',[(p.src_block_id,p.dst_block_id) for p in pairs])))
        o=output(new=[('a',([2],),3)],copies=[(1,2),(8,9)])
        self.method('_qsa_copy_host_pages')(runner,o)
        self.assertEqual(calls,[('observed',1,6),('copied',[(1,2)])])

    def test_actual_zero_token_runner_path_calls_hooks_without_pp_change(self):
        j,m,t,b,d,lines,switch,x,o,args=fixture(False)
        # No token forward: use an empty first scheduler frame, including the
        # observed allocation delta; the real runner still performs updates.
        o.flash_e2_frame=replace(o.flash_e2_frame,rows=(),restarted=())
        o.total_num_scheduled_tokens=0
        events=[];sentinel=object()
        runner=NS(_flash_e2_lifecycle=x,req_states=NS(req_id_to_index={}),
                  update_pp_decode_requests=lambda:events.append('existing-pp-update'),
                  finish_requests=lambda o:events.append('finish'),
                  free_states=lambda o:events.append('free'),
                  add_requests=lambda o:events.append('add'),
                  update_requests=lambda o:events.append('update'),
                  block_tables=NS(_flash_e2_tables=t,apply_staged_writes=lambda:events.append('apply')),
                  kv_connector=NS(no_forward=lambda o:sentinel),
                  _merge_ec_connector_no_forward=lambda o,result:result)
        method=execute_with_e2(self.method('execute_model'))
        self.assertIs(method(runner,o),sentinel)
        self.assertEqual(events,['existing-pp-update','finish','free','add','update','apply'])
        self.assertEqual(m.sequence,1);self.assertFalse(m.active)
        self.assertEqual(len(lines),1);self.assertFalse(b.events)


if __name__=='__main__':unittest.main()
