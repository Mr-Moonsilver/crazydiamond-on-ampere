"""Fit floor: real Core/FIFO path, synthetic CUDA counters, no device execution."""
import ast
from concurrent.futures import Future
import enum
import inspect
import io
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace as NS
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'patches/s5_engine'),str(ROOT/'tools')]
import fit_memory as fit
import fit_8x200k_source as source


class Pause(str,enum.Enum):
    UNPAUSED='UNPAUSED'
    PAUSED_ALL='PAUSED_ALL'


def peers(floor,low=None):
    return [dict(rank=r,sequence=floor.sequence,role=floor.selected,free_bytes=fit.FLOOR-1 if r==low else 128*2**20,
                 native_present=floor.selected!='stock',ooms=0,startup=dict(verdict='PASS')) for r in range(4)]


def stock():
    scheduler=NS(pause_state=Pause.UNPAUSED,requests={})
    scheduler.requests={key:NS(num_prompt_tokens=8,num_computed_tokens=8) for key in ('a','b')}
    scheduler.set_pause_state=lambda state:setattr(scheduler,'pause_state',state)
    engine=NS(scheduler=scheduler,batch_queue=[])
    floor=fit.FitFloor(engine,'stock',writer=lambda row:None)
    def rpc(name,*,args,non_block=False,timeout=30):
        assert name=='flash_s5_fit_memory' and args[0]['sequence']==floor.sequence
        value=peers(floor)
        if non_block:
            future=Future();future.set_result(value);return future
        return value
    engine.model_executor=NS(collective_rpc=rpc)
    return floor,engine


class Contract(unittest.TestCase):
    def test_actual_worker_host_fields_reach_all_rank_peak_report(self):
        cuda=NS(mem_get_info=lambda device:(128*2**20,24*2**30),memory_stats=lambda device:{},
                memory_allocated=lambda device:10,memory_reserved=lambda device:20)
        floor,_=stock();floor.sequence=1;rows=[]
        with mock.patch.dict(os.environ,{fit.ENV:'stock','FLASH_S5_NATIVE_MM':'0'}):
            for rank in range(4):
                worker=NS(rank=rank,device=NS(index=rank),model_runner=NS(),_flash_fit_startup=dict(verdict='PASS'))
                row=fit.worker_measure(worker,dict(sequence=1,role='stock'),torch_module=NS(cuda=cuda))
                self.assertEqual(set(row['host_rss_bytes']),{'VmRSS','VmHWM','RssAnon','RssShmem'})
                rows.append(row)
            floor.observe(rows)
        self.assertEqual(set(floor.host_rss_peaks),set(map(str,range(4))))
        self.assertTrue(all(row['VmHWM']>=row['VmRSS'] for row in floor.host_rss_peaks.values()))

    def test_eight_resident_frontiers_are_simultaneous_not_accumulated(self):
        floor,engine=stock()
        engine.scheduler.requests={str(r):NS(num_computed_tokens=190000,num_tokens=198192) for r in range(7)}
        floor.sample_resident();self.assertFalse(floor.resident_peak['eight_at_190k_seen'])
        engine.scheduler.requests['7']=NS(num_computed_tokens=189999,num_tokens=198192)
        floor.sample_resident();self.assertFalse(floor.resident_peak['eight_at_190k_seen'])
        engine.scheduler.requests['7'].num_computed_tokens=190000
        floor.sample_resident();self.assertTrue(floor.resident_peak['eight_at_190k_seen'])
        self.assertEqual(floor.resident_peak['tokens'],1520000)

    def test_flag_off_does_not_touch_the_engine_or_create_cuda_objects(self):
        with mock.patch.dict(os.environ,{},clear=True):
            engine=NS();fit.install(engine)
            self.assertFalse(hasattr(engine,'_flash_fit_memory'))
            self.assertIsNone(fit.before_step(engine));self.assertFalse(fit.pending(engine))

    def test_stock_rounds_all_peers_and_clean_pause(self):
        floor,engine=stock();self.assertFalse(floor.tick())
        floor.note_schedule(NS(num_scheduled_tokens={'a':1,'b':1}))
        self.assertFalse(floor.tick());self.assertEqual(floor.coverage['decode_checked'],1)
        floor.finishing=True;self.assertTrue(floor.tick());self.assertTrue(floor.settled)
        self.assertEqual(engine.scheduler.pause_state,Pause.PAUSED_ALL)
        self.assertEqual(len(floor.minima),4)

    def test_every_low_peer_holds_and_queued_stock_is_applied_before_settled(self):
        for rank in range(4):
            floor,engine=stock();floor.sequence=1
            floor.observe(peers(floor,rank));self.assertIsNotNone(floor.hold)
            engine.batch_queue=[object()]
            self.assertFalse(floor.tick());self.assertFalse(floor.settled)
            engine.batch_queue.clear();self.assertTrue(floor.tick());self.assertTrue(floor.settled)

    def test_missing_duplicate_foreign_or_stale_snapshot_refuses(self):
        for change in (lambda rows:rows.pop(),lambda rows:rows[3].update(rank=2),
                       lambda rows:rows[2].update(sequence=99),lambda rows:rows[1].update(role='e6')):
            floor,_=stock();floor.sequence=1;rows=peers(floor);change(rows)
            floor.observe(rows);self.assertIsNotNone(floor.hold)

    def test_floor_mutants(self):
        original=inspect.getsource(fit.FitFloor.observe)
        for old,new in (("v<FLOOR","v<0"),("len(rows)!=4","False")):
            namespace=dict(fit.__dict__);exec(original.lstrip().replace(old,new),namespace)
            floor,_=stock();floor.sequence=1
            values=peers(floor,1) if old=='v<FLOOR' else peers(floor)[:-1]
            # The missing-peer mutant still needs to retain a map with rank3 so
            # the later explicit rank3 receipt is not what catches the loss.
            if old!='v<FLOOR':values=peers(floor)[1:]
            namespace['observe'](floor,values)
            self.assertIsNone(floor.hold)
            actual,_=stock();actual.sequence=1;actual.observe(values);self.assertIsNotNone(actual.hold)

    def test_settled_native_stop_never_runs_another_stock_step(self):
        from graph_event_floor import Floor
        engine=NS(_flash_s5_driver=NS(),scheduler=NS())
        floor=fit.FitFloor(engine,'e6',writer=lambda row:None)
        def settled(owner):owner.settled=True;return False
        with mock.patch.object(Floor,'stop_tick',settled):
            self.assertTrue(floor.stop_tick())
        engine._flash_fit_memory=floor;floor.hold='floor-stop'
        self.assertEqual(fit.before_step(engine),({},False))

    def test_stock_pending_round_receives_without_background_reply_reader(self):
        floor,engine=stock();floor.tick();engine.batch_queue=[object()]
        class Ordered:
            def done(self):return False
            def result(self):self.read=True;return peers(floor)
            read=False
        pending=Ordered();floor.future=pending;floor.sequence+=1;floor.in_query=[('decode',None)]
        self.assertFalse(floor.tick());self.assertFalse(pending.read)
        floor.owed=[[('decode',None)],[('decode',None)]]
        floor.tick();self.assertTrue(pending.read)
        self.assertEqual(floor.coverage['decode_checked'],1)

    def test_real_receipt_writer_accepts_fit_and_still_refuses_unknown_prefix(self):
        import native_measurement as nm
        rows=[]
        # The raw frozen module has the same explicit prefix transform as the
        # worker package; no logger-only fake may hide a rejected write.
        prefixes=nm.RECEIPT_JSON_PREFIXES
        if source.EXPORTED.is_dir():self.assertIn('flash-s5 fit-memory ',prefixes)
        else:prefixes=prefixes+('flash-s5 fit-memory ','flash-s5 graph-events ','flash-s5 startup-vram ')
        writer=nm.ReceiptWriter(fd=1,write=lambda fd,data:rows.append(data) or len(data))
        with mock.patch.object(nm,'RECEIPT_JSON_PREFIXES',prefixes),mock.patch.object(nm,'_RECEIPT_WRITER',writer):
            fit.emit(dict(kind='fixture-check',sequence=1,free_bytes={str(r):128*2**20 for r in range(4)}))
            self.assertFalse(writer.write_lines(('unregistered-prefix {}',)))
        self.assertTrue(any(b'flash-s5 fit-memory ' in row for row in rows));self.assertEqual(writer.faults,1)
        with mock.patch.object(nm,'RECEIPT_JSON_PREFIXES',tuple(p for p in prefixes if p!='flash-s5 fit-memory ')),mock.patch.object(nm,'_RECEIPT_WRITER',writer):
            with self.assertRaises(RuntimeError):fit.emit(dict(kind='reverted-prefix-mutant'))

    def test_frozen_core_worker_and_router_transform_parse_and_pin(self):
        for rel in ('e3_sources/core.py','e3_sources/worker.py','e3_sources/api_routers.py','e3/scheduler_hooks.py','e3/native_measurement.py','e3/native_auto_arm.py'):
            before=(source.PARENT/rel).read_text();after,pin=source.transform(rel,before)
            self.assertNotEqual(pin['before'],pin['after']);ast.parse(after)
            with self.assertRaises(ValueError):source.transform(rel,after)


class RealCore(unittest.TestCase):
    def attach(self,r,floor,low=None):
        original=r.core.model_executor.collective_rpc
        def rpc(name,*,args,non_block=False,timeout=30,**kwargs):
            if name!='flash_s5_fit_memory':return original(name,args=args,non_block=non_block,**kwargs)
            self.assertEqual(args[0]['sequence'],floor.sequence)
            self.assertFalse(non_block)
            return peers(floor,low)
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(r.core.model_executor,'collective_rpc',side_effect=rpc).start()

    def admission_methods(self):
        import test_append_yield as ay
        text,_=source.transform('e3/scheduler_hooks.py',(source.PARENT/'e3/scheduler_hooks.py').read_text())
        cls=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='PrefillAdmissionController')
        body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('admit','_admit_observed')]
        tree=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+body,type_ignores=[])
        scope=dict(ay.sh.__dict__);exec(compile(ast.fix_missing_locations(tree),'fit/admission.py','exec'),scope)
        return scope

    def check_w21_order(self):
        import test_append_yield as ay
        import hashlib,json
        fixture=Path(__file__).with_name('fit_w21_auto_tick.py.src')
        pin=json.loads(fixture.with_name('fit_w21_repro.json').read_text())
        self.assertEqual(hashlib.sha256(fixture.read_bytes()).hexdigest(),pin['tick_sha256'])
        scope=dict(ay.auto.__dict__);exec(compile(fixture.read_text(),str(fixture),'exec'),scope)
        methods=self.admission_methods()
        with mock.patch.object(ay.auto.AutoArmPolicy,'tick',scope['tick']),mock.patch.object(ay.sh.PrefillAdmissionController,'admit',methods['admit']),mock.patch.object(ay.sh.PrefillAdmissionController,'_admit_observed',methods['_admit_observed']):
            r=ay.Rig(driver='staged');r.c.native_residue=1;r.c.native_hold_s=600
            floor=fit.FitFloor(r.core,'e6',writer=lambda row:None)
            r.d.graph_event_floor=floor;r.core._flash_fit_memory=floor;self.attach(r,floor)
            r.p.schedule_before=r.c._schedule_seq;r.p.admitted_before=r.c.admitted
            admitted=r.admit()
            # The missing interleave in b's test: policy runs BEFORE another admission.
            r.p.tick()
            self.assertFalse(r.p.rollbacks)
            self.assertIsNotNone(admitted);self.assertTrue(r.d.active)
            self.assertEqual(floor.coverage['chunk_checked'],1)
            self.assertIsNone(floor.pending_chunk)

    def test_w21_frozen_b_tick_zero_http_then_revert_mutant(self):
        self.check_w21_order()
        def old_pending(owner,key):
            owner.pending_chunk=key;owner.checked=None
            owner.coverage['chunks']+=1;owner.owed.append([('chunk',key)])
            return False
        with mock.patch.object(fit.FitFloor,'before',old_pending):
            with self.assertRaises(AssertionError):self.check_w21_order()

    def test_precommit_joins_an_earlier_decode_census_before_new_permit(self):
        floor,engine=stock();floor.tick()
        order=[]
        class Pending:
            def done(self):return False
            def result(self):order.append('decode-reply');return peers(floor)
        floor.sequence+=1;floor.future=Pending();floor.in_query=[('decode',None)]
        original=engine.model_executor.collective_rpc
        def rpc(*args,**kwargs):
            self.assertEqual(order,['decode-reply']);order.append('permit')
            return original(*args,**kwargs)
        with mock.patch.object(engine.model_executor,'collective_rpc',side_effect=rpc):
            self.assertTrue(floor.before_arm())
        self.assertEqual(order,['decode-reply','permit'])
        self.assertEqual(floor.coverage['decode_checked'],1)
        self.assertIsNone(floor.future)

    def test_prearm_permit_precedes_both_real_arm_paths_and_low_peer_never_arms(self):
        import test_append_yield as ay
        text,_=source.transform('e3/native_auto_arm.py',(source.PARENT/'e3/native_auto_arm.py').read_text())
        cls=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='AutoArmPolicy')
        methods={}
        for name in ('tick','_o3_arm'):
            node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
            scope=dict(ay.auto.__dict__);exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source.PARENT/'e3/native_auto_arm.py'),'exec'),scope)
            methods[name]=scope[name]
        for name in ('tick','_o3_arm'):
            for low in (None,1):
                with self.subTest(path=name,low=low):
                    r=ay.Rig(driver='staged');r.d._disarm_admission('fixture-rearm')
                    floor=fit.FitFloor(r.core,'e6',writer=lambda row:None);r.d.graph_event_floor=floor
                    self.attach(r,floor,low)
                    r.p.plan=r.p._plan(r.bulk);r.p.phase='split';r.p._pause(ay.Pause.PAUSED_ALL)
                    result=dict(typed_acks=True,route_after='split',transition_acks=[{}]*4)
                    r.p.pending=Future();r.p.pending.set_result(result)
                    order=[];original=r.d._arm_admission
                    def arm(phase):
                        self.assertEqual(floor.coverage['checks'],2) # startup then arm
                        self.assertEqual(set(floor.minima),set(map(str,range(4))))
                        order.append('arm');return original(phase)
                    with mock.patch.object(r.d,'_arm_admission',side_effect=arm):methods[name](r.p)
                    self.assertEqual(bool(order),low is None)
                    self.assertEqual(r.d.admission_intent() is not None,low is None)
                    self.assertEqual(floor.hold is not None,low is not None)
                    if low is not None:
                        for _ in range(400):
                            floor.stop_tick();r.stack.complete_everything()
                            if floor.settled:break
                        self.assertTrue(floor.settled)
                        self.assertFalse(r.d._split_ready)
                        self.assertIsNone(r.d.admission_intent())

    def test_real_native_admission_waits_for_exact_peer_permit_then_stops_active_pass(self):
        import test_append_yield as ay
        text,_=source.transform('e3/scheduler_hooks.py',(source.PARENT/'e3/scheduler_hooks.py').read_text())
        owner=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='PrefillAdmissionController')
        methods=[n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name in ('admit','_admit_observed')]
        tree=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+methods,type_ignores=[])
        scope=dict(ay.sh.__dict__);exec(compile(ast.fix_missing_locations(tree),'fit/admission.py','exec'),scope)
        with mock.patch.object(ay.sh.PrefillAdmissionController,'admit',scope['admit']),mock.patch.object(ay.sh.PrefillAdmissionController,'_admit_observed',scope['_admit_observed']):
            r=ay.Rig(driver='staged');floor=fit.FitFloor(r.core,'e6',writer=lambda row:None)
            r.d.graph_event_floor=floor;r.core._flash_fit_memory=floor
            self.attach(r,floor)
            self.assertIsNotNone(r.admit());self.assertIsNone(floor.checked)
            floor.trip('fixture-low-peer-after-admission')
            for _ in range(400):
                floor.stop_tick();r.stack.complete_everything()
                if floor.settled:break
            self.assertTrue(floor.settled)
            self.assertFalse(any(w._flash_s5_bridge.active for w in r.stack.workers.values()))
            self.assertEqual(floor.coverage['chunks'],floor.coverage['chunk_checked'])

    def test_low_peer_after_real_queued_sample_cleanly_drains_and_stops(self):
        original=Path.read_text
        def reference(path,*args,**kwargs):
            root=ROOT/'reference'
            if path.is_relative_to(root) and not path.exists():
                path=Path('<hub-dir>/agents/flash-next/fn-main/reference')/path.relative_to(root)
            return original(path,*args,**kwargs)
        patch=mock.patch.object(Path,'read_text',reference)
        patch.start();self.addCleanup(patch.stop)
        import test_control_inwait as c
        from typing import cast
        case=c.RealCoreOrder('test_real_four_worker_fifo_query_overlaps_decode_and_switch_follows_applied_cut')
        case.setUp();self.addCleanup(case.doCleanups)
        r=case.r;events=[]
        floor=fit.FitFloor(r.core,'e6',writer=events.append)
        r.core._flash_fit_memory=floor;r.d.graph_event_floor=floor
        for rank,worker in r.stack.workers.items():
            worker.flash_s5_fit_memory=lambda command,rank=rank:dict(rank=rank,sequence=command['sequence'],role='e6',
                free_bytes=fit.FLOOR-1 if rank==1 else 128*2**20,native_present=True,ooms=0,startup=dict(verdict='PASS'))
        # Executed body is the box-shaped Core after the exact fit transformation.
        text,_=source.transform('e3_sources/core.py',(source.PARENT/'e3_sources/core.py').read_text())
        owner=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='EngineCore')
        method=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='step_with_batch_queue')
        tree=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),method],type_ignores=[])
        scope=dict(_flash_s5_hooks=c.eh,_flash_fit=fit,cast=cast,Future=Future)
        exec(compile(ast.fix_missing_locations(tree),'fit/core.py','exec'),scope)
        step=scope['step_with_batch_queue']
        self.assertEqual(len(r.core.batch_queue),1)
        step(r.core)
        self.assertEqual([x[0] for x in case.wire.writes[:3]],['execute_model','sample_tokens','flash_s5_fit_memory'])
        self.assertFalse(floor.settled);case.wire.sample_gate.set()
        deadline=time.monotonic()+5
        while not floor.settled and time.monotonic()<deadline:
            step(r.core);time.sleep(.001)
        self.assertTrue(floor.settled);self.assertTrue(case.applied);self.assertFalse(r.core.batch_queue)
        self.assertIsNone(r.d.admission_intent());self.assertFalse(r.d._split_ready)
        self.assertTrue(all(not w._flash_s5_bridge.active for w in r.stack.workers.values()))
        for order in case.wire.order:
            self.assertLess(order.index('sample_tokens'),order.index('flash_s5_fit_memory'))
        self.assertFalse(case.wire.errors)


if __name__=='__main__':unittest.main()
