"""Bounded gap receipts: actual Core admission, four-worker control and commit; no device evidence."""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS,MethodType
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'patches/s5_engine'),str(ROOT/'harness')]
import engine_driver as ed
import executor_hooks as eh
import native_measurement as nm
import receipt_lines as rl
import test_e3_native_overlap as native
import test_e3_r4_worker_bridge as bridge
import test_compose5_declines as declines


def rows_of(lines):
    return [json.loads(x[len('flash-s5 native-gap '):]) for x in lines if x.startswith('flash-s5 native-gap ')]


class GapReceipts(unittest.TestCase):
    def recorder(self):
        d=NS(engine_epoch=17,route_generation=3,clock_basis='CLOCK_MONOTONIC',tx=None,auto_arm=None)
        d.progress=ed.NativeProgress('trace',NS(),driver=d)
        d._gap_event=MethodType(ed.NativeSegmentDriver._gap_event,d)
        return d

    def test_actual_generated_core_admission_logs_after_scheduler_acceptance(self):
        source=(ROOT/'reference/vllm-src/v1/engine/core.py').read_text()
        generated=eh.generate_core(source)
        self.assertEqual(eh.revert_core(generated),source)
        cls=next(n for n in ast.parse(generated).body if isinstance(n,ast.ClassDef) and n.name=='EngineCore')
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='add_request')
        ns=dict(Request=object,_flash_s5_hooks=eh)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<generated-Core-add_request>','exec'),ns)
        d=self.recorder();order=[];lines=[]
        core=NS(_flash_s5_driver=d,scheduler=NS(add_request=lambda req:order.append('accepted')))
        req=NS(request_id='test-request',num_prompt_tokens=123,pooling_params=None,kv_transfer_params=None,
               ec_transfer_params=None,abort_immediately=False)
        with patch.object(ed,'emit_receipt_lines',lambda xs,_:lines.extend(xs) or order.append('logged') or True):
            ns['add_request'](core,req)
        self.assertEqual(order,['accepted','logged'])
        (row,)=rows_of(lines)
        self.assertEqual(row['event'],'request_admitted')
        self.assertEqual(row['request_key'],hashlib.sha256(b'test-request').hexdigest()[:16])
        self.assertEqual((row['engine_epoch'],row['route_generation'],row['prompt_tokens']),(17,3,123))
        self.assertNotIn(req.request_id,lines[0])
        core.scheduler.add_request=lambda req: (_ for _ in ()).throw(ValueError('rejected'))
        with self.assertRaisesRegex(ValueError,'rejected'),patch.object(ed,'emit_receipt_lines') as emit:
            ns['add_request'](core,req)
        emit.assert_not_called()

    def test_real_four_worker_control_has_send_and_receive_without_open_chunk(self):
        s=native.idle_stack();native.core_for(s,paused=True);d=s.driver
        d.progress=ed.NativeProgress('trace',s.executor,driver=d)
        lines=[]
        with patch.object(ed,'emit_receipt_lines',lambda xs,_:lines.extend(xs) or True):
            f=d.submit_control(dict(action='census',engine_epoch=d.engine_epoch))
            for _ in range(500):
                if f.done():break
                d.control_tick();s.complete_everything()
            self.assertTrue(f.done());result=f.result()
        rows=rows_of(lines)
        self.assertEqual([r['event'] for r in rows],['control_submit_begin','control_sent','control_reply'])
        self.assertEqual(rows[1]['sequence'],rows[2]['sequence'])
        self.assertTrue(all(not r['recording'] for r in rows))
        self.assertEqual(sorted(r['observed_ns'] for r in rows),[r['observed_ns'] for r in rows])
        self.assertEqual(len(result['workers']),4)

    def run_chunk(self,record,broken=False):
        s=bridge.Stack();request=s.controller.parked.request
        if record:s.driver.progress=ed.NativeProgress('trace',s.executor,driver=s.driver)
        lines=[]
        def emit(xs,_):
            if broken:raise OSError('fixture sink unavailable')
            lines.extend(xs);return True
        with patch.object(ed,'emit_receipt_lines',emit):
            for _ in range(500):
                s.advance();s.complete_everything()
                if s.cache_calls():break
        self.assertTrue(s.cache_calls())
        return s,(request.num_computed_tokens,s.cache_calls(),list(s.sender.envelopes)),rows_of(lines)

    def test_real_four_worker_chunk_open_commit_and_failing_sink_do_not_change_work(self):
        a,ref,_=self.run_chunk(False)
        b,got,rows=self.run_chunk(True)
        c,broken,_=self.run_chunk(True,broken=True)
        self.assertEqual(got,ref);self.assertEqual(broken,ref)
        events=[r['event'] for r in rows]
        self.assertIn('open_submit_begin',events);self.assertIn('open_sent',events)
        self.assertIn('chunk_open',events);self.assertIn('chunk_commit',events);self.assertIn('source_hold',events)
        commits=[r for r in rows if r['event']=='chunk_commit']
        self.assertEqual(len(commits),1)
        self.assertEqual(commits[0]['transaction'],a.record.transaction.txn_seq)
        self.assertGreater(c.driver.progress.gap_errors,0)

    def test_real_policy_selection_pause_and_drain_empty_are_visible(self):
        s=declines.Stage(declines.req());d=s.driver
        d.route_generation=0;d.clock_basis='CLOCK_MONOTONIC';d.tx=None
        d.progress=ed.NativeProgress('trace',NS(),driver=d)
        d._gap_event=MethodType(ed.NativeSegmentDriver._gap_event,d)
        lines=[]
        with patch.object(ed,'emit_receipt_lines',lambda xs,_:lines.extend(xs) or True):
            self.assertTrue(s.policy._select())
            s.policy.tick()
        events=[r['event'] for r in rows_of(lines)]
        self.assertIn('select_begin',events);self.assertIn('pause',events);self.assertIn('drain_empty',events)
        self.assertEqual(s.policy.phase,'barrier')

    def test_budget_has_one_explicit_coverage_marker_and_no_unbounded_logging(self):
        d=self.recorder();d.progress.GAP_LIMIT=2;lines=[]
        with patch.object(ed,'emit_receipt_lines',lambda xs,_:lines.extend(xs) or True):
            for _ in range(100):d._gap_event('request_admitted',request_uid='r',prompt_tokens=99)
        rows=rows_of(lines)
        self.assertEqual(len(rows),3)
        self.assertEqual(rows[-1]['event'],'coverage-incomplete')
        self.assertFalse(rows[-1]['coverage_complete'])
        self.assertEqual(d.progress.gap_dropped,98)
        self.assertLess(max(len(s) for s in lines),1024)
        self.assertEqual(d.progress.snapshot()['gap_dropped'],98)

    def test_real_atomic_writer_and_integrity_reader_accept_the_new_prefix(self):
        d=self.recorder();read_fd,write_fd=os.pipe()
        try:
            with os.fdopen(write_fd,'wb') as out:
                writer=nm.ReceiptWriter(fd=out.fileno())
                with patch.object(nm,'_RECEIPT_WRITER',writer):
                    d._gap_event('request_admitted',request_uid='r',prompt_tokens=99)
            data=os.read(read_fd,8192).decode()
        finally:
            os.close(read_fd)
        lines=data.splitlines()
        self.assertEqual(len(lines),2)
        self.assertEqual(rl.verdict(rl.check(lines))['verdict'],'OK')
        row=json.loads(lines[1].split(' ',2)[2])
        self.assertEqual(row['gap_seq'],1)
        self.assertEqual(lines[1].count('"seq":'),1)
        self.assertEqual(row['schema'],'s5-native-gap/1')
        self.assertEqual(d.progress.gap_errors,0)
        self.assertLessEqual(len(lines[1].encode())+1,nm.RECEIPT_RECORD_BYTES)

    def test_absent_recorder_and_broken_callback_never_change_the_call(self):
        eh.engine_request_admitted(NS(),NS())
        s=declines.Stage(declines.req())
        s.driver._gap_event=lambda *a,**k: (_ for _ in ()).throw(OSError('fixture sink'))
        self.assertTrue(s.policy._select())
        self.assertEqual(s.policy.phase,'drain')


if __name__=='__main__':unittest.main()
