"""Opt-in junction yield: stock geometry, real commit, connected bulk reopen."""
from pathlib import Path
import inspect, json, os, sys, textwrap, types, unittest, weakref
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'harness'),str(ROOT/'patches/s5_engine')]
import scheduler_hooks as sh
import engine_driver as ed
import test_append_yield as a1
import test_coverage_residue as residue
import test_e3_admission_native as adm
import test_native39_trunk as queues


def candidate():
    request=a1.Req('junction-append',12000)
    request.shared_prefix_boundary=0;request.block_hashes=()
    manager=residue.JunctionManager(4048,4576)
    lookup=manager.get_computed_blocks
    def blocks(request):
        _,start,junction=lookup(request)
        return types.SimpleNamespace(blocks=()),start,junction
    manager.get_computed_blocks=blocks
    manager.coordinator.get_num_blocks_to_allocate=lambda **kw:kw['num_tokens']-kw['total_computed_tokens']
    manager.watermark_blocks=0;manager.block_pool.get_num_free_blocks=lambda:1_000_000
    scheduler=adm.Scheduler(manager,[request])
    scheduler.requests={request.request_id:request}
    scheduler.waiting=queues.IterQueue([request]);scheduler.skipped_waiting=queues.IterQueue([])
    for name,value in residue.split_host(max_num_scheduled_tokens=1024).items():
        setattr(scheduler,name,value)
    scheduler._mamba_block_aligned_split=types.MethodType(residue.SPLIT,scheduler)
    c=adm.make_controller(scheduler,chunk=2048)
    c.minimum_stage_rows=64;c.native_hold_s=600;c.native_residue=1;c.junction_yield=True
    c.append_yield=sh.AppendYieldConfig(8192,4,16,4,enabled=True)
    return c,scheduler,manager,request


class JunctionPlanning(unittest.TestCase):
    def setUp(self):
        self.logger=mock.Mock()
        patch=mock.patch.object(ed,'_progress_logger',return_value=self.logger)
        patch.start();self.addCleanup(patch.stop)

    def plan(self,c,s,r):
        return c._append_yield_plan(s,r,s.requests,c.append_yield)

    def test_plan_is_readonly_and_first_actual_commit_publishes_at_the_junction(self):
        c,s,m,r=candidate()
        plan=self.plan(c,s,r)
        self.assertIsNotNone(plan)
        self.assertEqual(plan[0],528)
        self.assertEqual(sum(plan),c.native_end(s,r)-4048)
        self.assertEqual(r.shared_prefix_boundary,0)
        self.assertEqual(r.num_computed_tokens,0)
        self.assertEqual(m.alloc_calls,[])
        record=c.admit(s)
        self.assertEqual((record.transaction.chunk_start,record.transaction.chunk_end),(4048,4576))
        self.assertEqual(r.shared_prefix_boundary,4576)
        residue.commit(c,record)
        self.assertEqual(r.num_computed_tokens,4576)
        self.assertEqual(m.cache_calls,[(r.request_id,4576)])
        self.assertEqual(c.residue_totals['junction'],1)

    def test_switch_off_residue_off_and_hold_off_keep_the_original_refusal(self):
        for field,value in (('junction_yield',False),('native_residue',0),('native_hold_s',0)):
            c,s,m,r=candidate();setattr(c,field,value)
            with self.subTest(field=field):
                self.assertIsNone(self.plan(c,s,r))
                self.assertEqual(c.yield_refusals,{'shared-junction':1})
                self.assertEqual(m.alloc_calls,[])

    def test_changed_prefix_is_relooked_up_at_admission_not_taken_from_the_scan(self):
        c,s,m,r=candidate();self.assertEqual(self.plan(c,s,r)[0],528)
        m.junction=4752
        record=c.admit(s)
        self.assertEqual((record.transaction.chunk_start,record.transaction.chunk_end),(4048,4752))
        self.assertEqual(r.shared_prefix_boundary,4752)

    def test_junction_does_not_bypass_new_row_free_block_or_complete_plan_guards(self):
        for why in ('max-new','blocks','no-plan'):
            c,s,m,r=candidate()
            if why=='max-new':c.append_yield=sh.AppendYieldConfig(100,4,16,4,enabled=True)
            if why=='blocks':m.block_pool.get_num_free_blocks=lambda:0
            if why=='no-plan':s._mamba_block_aligned_split=lambda *a,**kw:0
            with self.subTest(why=why):
                self.assertIsNone(self.plan(c,s,r))
                self.assertEqual(c.yield_refusals.get(why),1)
                self.assertEqual(m.alloc_calls,[])
                self.assertEqual(r.shared_prefix_boundary,0)

    def test_refusal_receipt_has_uid_effective_hit_and_actual_queue_position(self):
        c,s,m,r=candidate();c.append_yield=sh.AppendYieldConfig(100,4,16,4,enabled=True)
        s.waiting.items.insert(0,a1.Req('ahead',13000))
        r.arrival_time=123.5
        self.assertIsNone(self.plan(c,s,r))
        self.assertIsNone(self.plan(c,s,r))
        self.assertEqual(self.logger.info.call_count,1)
        row=json.loads(self.logger.info.call_args.args[1])
        self.assertEqual({k:row[k] for k in ('request_uid','reason','hit_end','junction','queue','queue_position')},
                         dict(request_uid=r.request_id,reason='max-new',hit_end=4048,junction=4576,
                              queue='waiting',queue_position=1))
        self.assertEqual(row['prompt_tokens'],12000)
        self.assertEqual(row['arrival_wall_s'],123.5)
        self.assertEqual(c.yield_receipts['duplicate'],1)
        self.assertEqual(c.yield_receipts['errors'],0)

    def test_receipt_failure_and_limit_never_change_the_refusal_or_drop_a_lease(self):
        c,s,m,r=candidate();c.append_yield=sh.AppendYieldConfig(100,4,16,4,enabled=True)
        self.logger.info.side_effect=OSError('declared log fault')
        self.assertIsNone(self.plan(c,s,r));self.assertEqual(c.yield_receipts['errors'],1)
        c._yield_receipt_seen.clear();c.yield_receipts['written']=4096
        self.assertIsNone(self.plan(c,s,r));self.assertEqual(c.yield_receipts['dropped'],1)
        self.assertEqual(m.alloc_calls,[])
        counters=c.native_admission_record()['queue_policy']['yield_lookup_receipts']
        self.assertEqual((counters['errors'],counters['dropped']),(1,1))

    def test_invalid_switch_value_is_refused_at_construction(self):
        with mock.patch.dict(os.environ,FLASH_S5_JUNCTION_YIELD='maybe'):
            with self.assertRaises(sh.E3ContractError):candidate()


class JunctionYieldReopen(unittest.TestCase):
    def test_real_admission_uses_the_auto_policy_nonhead_request_without_moving_its_peer(self):
        rig=a1.Rig(cached={});c,s,d,p=rig.c,rig.s,rig.d,rig.p
        target=rig.add_held('selected-nonhead',9)
        s.skipped_waiting.items.remove(target);s.waiting.items.append(target)
        c.shortest_bulk=True
        p._internal=True
        s.set_pause_state(a1.Pause.PAUSED_ALL)
        try:
            d._disarm_admission('test-reselect')
            p.request_ref=weakref.ref(target)
            d._arm_admission(a1.auto.PHASE)
        finally:
            p._internal=False
            s.set_pause_state(a1.Pause.UNPAUSED)
        self.assertIs(s.waiting.peek_request(),rig.bulk)
        transaction=rig.admit()
        self.assertIsNotNone(transaction)
        self.assertEqual(transaction.request_uid,target.request_id)
        self.assertIs(c.parked.request,target)
        self.assertIn(rig.bulk,s.waiting.items)
        self.assertEqual(rig.bulk.num_computed_tokens,0)

    def test_both_drivers_close_bulk_publish_junction_and_reopen_survivor_from_its_frontier(self):
        for kind in ('native','staged'):
            with self.subTest(driver=kind),mock.patch.object(ed,'_progress_logger',return_value=mock.Mock()):
                rig=a1.Rig(cached={},max_new=8,driver=kind)
                c,s=rig.c,rig.s
                c.junction_yield=True;c.native_residue=1;c.native_hold_s=600
                for name,value in residue.split_host().items():setattr(s,name,value)
                s.cache_config=types.SimpleNamespace(block_size=8);s.hash_block_size=2;s.max_num_scheduled_tokens=4
                s.mamba_prefill_checkpoint_alignment=8
                s._mamba_block_aligned_split=types.MethodType(residue.SPLIT,s)
                rig.bulk.shared_prefix_boundary=0;rig.bulk.block_hashes=()
                original=rig.admit();self.assertIsNotNone(original)
                append=rig.add_held('junction-returner',12,cached=4)
                append.shared_prefix_boundary=0;append.block_hashes=()
                lookup=s.kv_cache_manager.get_computed_blocks
                def junction(request):
                    blocks,start,shared=lookup(request)
                    return blocks,start,6 if request is append else shared
                s.kv_cache_manager.get_computed_blocks=junction
                bulk_frontier=original.chunk_end
                rig.finish_chunk()
                self.assertEqual(rig.d.append_yield_phase(),'closing')
                self.assertIsNone(rig.admit(),'no append open while bulk row is owned')
                a1.to_append(rig)
                transactions=a1.run_append(rig)
                self.assertEqual(transactions[0].chunk_end,6)
                self.assertEqual(transactions[-1].chunk_end,11)
                self.assertEqual(append.shared_prefix_boundary,6)
                self.assertEqual(rig.bulk.num_computed_tokens,bulk_frontier)
                self.assertFalse(set(rig.rows(append.request_id))&set(rig.rows(a1.BULK_UID)))
                a1.to_full(rig);rig.stock_handback(append);a1.to_resume(rig)
                reopened=rig.admit()
                self.assertEqual((reopened.request_uid,reopened.chunk_start),(a1.BULK_UID,bulk_frontier))
                self.assertEqual(reopened.request_generation,original.request_generation)
                self.assertNotEqual(reopened.txn_seq,original.txn_seq)


class Mutants(unittest.TestCase):
    def caught(self, method, old, new, check):
        original=getattr(sh.PrefillAdmissionController,method)
        source=textwrap.dedent(inspect.getsource(original))
        self.assertEqual(source.count(old),1)
        namespace=dict(vars(sh))
        exec(compile(source.replace(old,new,1),'<junction-yield-mutant>','exec'),namespace)
        with mock.patch.object(sh.PrefillAdmissionController,method,namespace[method]):
            result=unittest.TestResult();check.run(result)
        self.assertTrue(result.failures,'mutant must fail an assertion, not only raise')
        self.assertEqual(result.errors,[])

    def test_old_shared_junction_refusal_is_caught(self):
        self.caught('_append_yield_plan','if shared and not allow_junction:', 'if shared:',
                    JunctionPlanning('test_plan_is_readonly_and_first_actual_commit_publishes_at_the_junction'))

    def test_missing_boundary_on_the_readonly_view_is_caught(self):
        self.caught('_append_yield_plan','shared_prefix_boundary=shared) if allow_junction else request)',
                    'shared_prefix_boundary=0) if allow_junction else request)',
                    JunctionPlanning('test_plan_is_readonly_and_first_actual_commit_publishes_at_the_junction'))

    def test_mutating_the_real_request_during_scan_is_caught(self):
        self.caught('_append_yield_plan','prompt_end = request.num_tokens',
                    'request.shared_prefix_boundary = 4576; prompt_end = request.num_tokens',
                    JunctionPlanning('test_plan_is_readonly_and_first_actual_commit_publishes_at_the_junction'))

    def test_missing_nonhead_admission_adapter_is_caught(self):
        self.caught('_admit_observed','self.shortest_bulk and policy is not None and policy.active',
                    'False and policy is not None and policy.active',
                    JunctionYieldReopen('test_real_admission_uses_the_auto_policy_nonhead_request_without_moving_its_peer'))


if __name__ == '__main__':
    unittest.main()
