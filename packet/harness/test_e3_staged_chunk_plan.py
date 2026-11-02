"""Native15 CPU admission planning; no GPU or engine evidence.

Real controller allocation/parking checks use explicit scheduler/device fixtures.
The pure planner only calls the stock-compatible alignment callback; helper guard
checks execute the generated source with the unchanged pinned minimum64.
"""
import ast
from pathlib import Path
import sys
import types
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'patches/s5_engine'),str(ROOT/'harness')]
import scheduler_hooks as sh
import test_e3_admission_native as admission
import test_e3_native_measurement as measurement
import test_e3_sparse_history_dma as sparse


class PlannerTests(unittest.TestCase):
    def plan(self, total, cap=1024, minimum=64, split=None, **kw):
        fn=split or (lambda start,limit:limit)
        plan=sh.staged_chunk_plan(0,total,cap,minimum,fn,**kw)
        if plan:
            self.assertEqual(sum(plan),total)
            position=0
            for size in plan:
                self.assertLessEqual(size,cap)
                self.assertGreaterEqual(size,minimum)
                self.assertEqual(fn(position,size),size)
                position+=size
        return plan

    def test_frozen_request_tail_is_redistributed_without_cap_change(self):
        plan=self.plan(98311)
        self.assertEqual(plan[:-2],(1024,)*95)
        self.assertEqual(plan[-2:],(967,64))

    def test_minimum_boundaries_and_terminal_one_token_is_not_absorbed(self):
        for rows in (1,7,63):
            self.assertEqual(self.plan(rows),())
        for rows in (64,65,1024):
            self.assertEqual(self.plan(rows),(rows,))
        for tail in (1,7,63,64,65):
            plan=self.plan(4096+tail)
            self.assertTrue(plan)
            self.assertGreaterEqual(plan[-1],64)

    def test_mandatory_boundaries_are_never_crossed(self):
        stops=(4048,8096,10000)
        def split(start,limit):
            return min(limit,next(x for x in stops if x>start)-start)
        plan=self.plan(10000,split=split)
        position=0
        ends=[]
        for size in plan:
            position+=size;ends.append(position)
        self.assertTrue(set(stops)<=set(ends))

    def test_impossible_mandatory_short_first_slice_stays_stock(self):
        self.assertEqual(self.plan(4096,split=lambda start,n:min(n,7-start) if start<7 else n),())

    def test_impossible_late_stop_backtracks_without_invented_chunk(self):
        stops=(4096,4103)
        def split(start,n):return min(n,next(x for x in stops if x>start)-start)
        self.assertEqual(self.plan(4103,split=split,query_limit=2000),())

    def test_exact_block_alignment_and_budget(self):
        def split(start,n):return n//16*16
        self.assertTrue(self.plan(4096,cap=256,split=split))
        self.assertEqual(self.plan(4097,cap=256,split=split,query_limit=2000),())
        self.assertEqual(self.plan(100,cap=63),())

    def test_search_is_bounded(self):
        calls=[]
        def split(start,n):calls.append((start,n));return n
        self.assertEqual(self.plan(98311,split=split,query_limit=3),())
        self.assertEqual(len(calls),3)

    def test_split_contract_is_validated(self):
        for value in (-1,True,1.0,2048):
            with self.assertRaises(sh.E3ContractError):
                self.plan(2048,split=lambda *args:value)

    def test_request_projection_is_read_only(self):
        real=types.SimpleNamespace(num_computed_tokens=0,num_tokens=98312,shared_prefix_boundary=10)
        view=sh._ProjectedPrefillRequest(real,100)
        self.assertEqual(view.num_computed_tokens,100)
        self.assertEqual(view.shared_prefix_boundary,10)
        with self.assertRaises(sh.E3ContractError):view.num_computed_tokens=99
        self.assertEqual(real.num_computed_tokens,0)


class AdmissionTests(unittest.TestCase):
    def make(self,tokens,cached=0):
        manager=admission.Manager(cached=cached)
        request=admission.Request(tokens=tokens)
        scheduler=admission.Scheduler(manager,[request])
        controller=sh.PrefillAdmissionController(scheduler,leases=sh.SchedulerLeases(),
            frontier=admission.ah.CommittedFrontier(),engine_epoch=7,
            chunk_tokens=1024,ranks=(0,1),minimum_stage_rows=64).attach()
        return manager,request,scheduler,controller

    def test_before_any_allocation_rejects_unplannable_request(self):
        manager,request,scheduler,controller=self.make(64)
        self.assertIsNone(controller.admit(scheduler))
        self.assertEqual(manager.alloc_calls,[])
        self.assertIs(scheduler.waiting.peek_request(),request)
        self.assertIsNone(controller.parked)

    def test_budget_refusal_is_before_allocation(self):
        manager,request,scheduler,controller=self.make(98312)
        scheduler.max_num_scheduled_tokens=63
        self.assertIsNone(controller.admit(scheduler))
        self.assertEqual(manager.alloc_calls,[])

    def test_cached_frontier_plan_keeps_original_request_unchanged(self):
        manager,request,scheduler,controller=self.make(98312,cached=97280)
        record=controller.admit(scheduler)
        self.assertEqual(record.transaction.chunk_start,97280)
        self.assertEqual(record.chunk_tokens,967)
        self.assertEqual(request.num_computed_tokens,0)
        self.assertEqual(manager.alloc_calls[0][1],967)

    def test_retained_row_never_falls_into_stock_when_new_plan_is_impossible(self):
        manager,request,scheduler,controller=self.make(98312)
        txn=types.SimpleNamespace(request_uid=request.request_id,request_generation=1)
        controller._request_identities[request.request_id]=(request,1)
        scheduler._flash_s5_driver=types.SimpleNamespace(retained_request=lambda:txn)
        scheduler.max_num_scheduled_tokens=63
        with self.assertRaisesRegex(sh.E3ContractError,'retained native request'):
            controller.admit(scheduler)
        self.assertEqual(manager.alloc_calls,[])
        self.assertIs(scheduler.waiting.peek_request(),request)

    def test_production_constructor_binds_pinned_minimum(self):
        source=(ROOT/'reference/vllm-src/models/qwen4_exp/nvidia/qsa.py').read_text()
        value=next(ast.literal_eval(n.value) for n in ast.parse(source).body
            if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_QSA_STAGE_MIN_ROWS' for t in n.targets))
        self.assertEqual(sh.QSA_STAGE_MIN_ROWS,value)
        startup=ast.parse((ROOT/'patches/s5_engine/s5_startup.py').read_text())
        # C1 (FLASH_S5_STAGE_OVERLAP) constructs through ``controller_class``, which is bound to the frozen
        # PrefillAdmissionController unless the switch selects its subclass; either form is the one construction site.
        defaults=[n for n in ast.walk(startup) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Name)
            and n.value.id=='PrefillAdmissionController' and [ast.unparse(t) for t in n.targets]==['controller_class']]
        names=('PrefillAdmissionController','controller_class') if len(defaults)==1 else ('PrefillAdmissionController',)
        calls=[n for n in ast.walk(startup) if isinstance(n,ast.Call)
            and isinstance(n.func,ast.Name) and n.func.id in names]
        self.assertEqual(len(calls),1)
        self.assertEqual(ast.unparse(next(k.value for k in calls[0].keywords if k.arg=='minimum_stage_rows')),'QSA_STAGE_MIN_ROWS')

    def test_real_source_driver_runs_balanced_terminal_chunk_and_all_rank_close(self):
        s=measurement.make_long_stack(tokens=98312,layers=48,chunk=1024,manager_cls=measurement.ChunkBlocks)
        # Initial1024 chunk was admitted by fixture construction. Enable the
        # production minimum before subsequent admissions; it is already valid.
        s.controller.minimum_stage_rows=64
        entries=measurement.run_journal(s)
        opens=[e for e in entries if e['kind']=='chunk_open']
        self.assertEqual(len(opens),97)
        self.assertTrue(all(64<=e['chunk_end']-e['chunk_start']<=1024 for e in opens))
        self.assertEqual([(e['chunk_start'],e['chunk_end']) for e in opens[-2:]],[(97280,98247),(98247,98311)])
        self.assertEqual(s.driver.retained_row(),None)
        self.assertEqual(s.controller.commits,97)
        census=measurement.census_for(s)
        for entry in entries:census.ingest(entry)
        start,end=measurement.window_of(entries)
        evidence=census.evidence(start,end,now_ns=end+1)
        self.assertTrue(evidence['completeness']['complete'],evidence['reasons'])


if __name__=='__main__':unittest.main()
