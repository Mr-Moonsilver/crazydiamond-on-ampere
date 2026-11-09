"""Passive fused phase ports; CUDA overhead/numerics remain an owner device check."""
import ast
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'patches/s5_engine'),str(ROOT/'harness')]
import native_measurement as nm
import qsa_operation as qo
import qsa_driver
import receipt_lines as rl
import test_n3_indexer_lifetime as life
import test_e3_native_qsa_fused as fused
import test_e3_sparse_history_dma as sparse
import test_alpha_next_receipts as atomic


class Event:
    def __init__(self,device,stream):self.device=device;self.stream=stream
    def elapsed_time(self,other):
        assert self.device.reading,'elapsed read in launch path'
        self.device.reads+=1
        return .25


class Device:
    compute='compute';copy='copy';native_fused_phase_every=1
    def __init__(self):self.records=[];self.reads=0;self.queries=0;self.reading=False;self.ready=True;self.fail=False
    def timing_event(self,stream):
        if self.fail:raise OSError('fixture event failure')
        event=Event(self,stream);self.records.append(event);return event
    def query(self,event):
        assert self.reading,'query in launch path'
        self.queries+=1;return self.ready
    def stream_token(self,stream):return {'compute':101,'copy':202}[stream]


def recorder():
    d=Device();tx=sparse.make_transaction(chunk_end=201261)
    op=NS(device=d,transaction=tx,lane=sparse.make_lane(),_cancelled=None,stats={})
    run=NS(layer_name='qsa.0',valid_tokens=201261,pages=tuple(range(50)),buckets=(0,))
    op._layers={'qsa.0':run}
    op._current_run=lambda:run
    trace=nm.FusedPhaseSpans(op);trace.group=(0,2);trace.stage=0;trace.fused_layers.add('qsa.0')
    return trace,d,op


def rows(trace,device):
    texts=[];device.reading=True
    with patch.object(nm,'emit_receipt_lines',side_effect=lambda lines,logger:texts.extend(lines)):
        trace.emit()
    return [json.loads(s[len(nm.FusedPhaseSpans.PREFIX):]) for s in texts
            if s.startswith(nm.FusedPhaseSpans.PREFIX)]


def all_phases(trace):
    previous=object();trace.consumer(previous,'earlier',0)
    trace.bucket(0,previous,'previous_layer')
    for phase in trace.PHASES:
        trace.end(trace.begin(phase,'copy' if phase in ('copy_wait','copy',
            'wait_current_producer','wait_previous_arena_consumer') else 'compute',
            bucket=0 if phase not in ('score','topk','indexer','canonical_selection','kv_producer_update') else None))


class BoundsAndScope(unittest.TestCase):
    def test_startup_interval_and_operation_sampling(self):
        self.assertEqual(qo.fused_phase_every({}),0)
        for n in ('1','16','4096'):self.assertEqual(qo.fused_phase_every({'FLASH_S5_FUSED_PHASE_EVERY':n}),int(n))
        for n in ('-1','4097','01','yes','16.0',''):
            with self.assertRaises(qo.OperationError):qo.fused_phase_every({'FLASH_S5_FUSED_PHASE_EVERY':n})
        _,device,op=recorder();device.native_fused_phase_every=16
        for seq in (1,2,16,17):
            real=qo.QsaPrefillOperation(replace(op.transaction,txn_seq=seq),op.lane,device=device)
            self.assertEqual(real.phase_trace is not None,seq in (1,17))
            real.transport_policy='dense';self.assertIsNone(real.fused_refusal())
            real.trace=object();self.assertEqual(real.fused_refusal(),'diagnostic-readback-attached')

    def test_event_reads_deferred_and_caps_fail_visible(self):
        trace,d,op=recorder();all_phases(trace)
        self.assertEqual((d.queries,d.reads),(0,0))
        data=rows(trace,d);self.assertTrue(data[-1]['complete'])
        self.assertEqual(data[0]['phases']['score']['count'],1)
        self.assertEqual(data[0]['phases']['copy']['stream'],'copy')
        self.assertEqual((d.queries,d.reads),(len(trace.PHASES),len(trace.PHASES)))
        trace,d,op=recorder()
        for _ in range(300):trace.end(trace.begin('score','compute'))
        self.assertEqual(len(d.records),512);self.assertEqual(trace.markers,512);self.assertEqual(trace.dropped,44)
        data=rows(trace,d);self.assertFalse(data[-1]['complete']);self.assertGreater(data[-1]['incomplete'],0)

    def test_pending_missing_failed_and_cancelled_coverage(self):
        for condition in ('pending','not-ready','event-error','cancelled','missing-phase','not-fused'):
            with self.subTest(condition=condition):
                trace,d,op=recorder();all_phases(trace)
                if condition=='pending':trace.begin('score','compute')
                if condition=='not-ready':d.ready=False
                if condition=='event-error':d.fail=True;trace.begin('score','compute')
                if condition=='cancelled':op._cancelled='cancelled'
                if condition=='missing-phase':trace.pairs=[p for p in trace.pairs if p[1]!='score']
                if condition=='not-fused':trace.rows['qsa.0']['fused']=False
                data=rows(trace,d);self.assertFalse(data[-1]['complete'])
                if condition=='not-ready':self.assertEqual(d.reads,0)

    def test_context_restored_on_nested_scope_and_exception(self):
        self.assertIsNone(nm.INDEXER_PHASE_CONTEXT.get())
        with nm.indexer_phase_scope('outer'):
            with self.assertRaisesRegex(ValueError,'fixture'):
                with nm.indexer_phase_scope('inner'):raise ValueError('fixture')
            self.assertEqual(nm.INDEXER_PHASE_CONTEXT.get(),'outer')
            with nm.indexer_phase_scope(None):self.assertIsNone(nm.INDEXER_PHASE_CONTEXT.get())
        self.assertIsNone(nm.INDEXER_PHASE_CONTEXT.get())

    def test_real_atomic_writer_integrity_and_budget(self):
        trace,d,op=recorder();all_phases(trace);d.reading=True
        writer,cap=atomic.capture_writer()
        try:
            with patch.object(nm,'_RECEIPT_WRITER',writer):trace.emit()
            lines=atomic.lines_from(cap.writes)
            self.assertEqual(rl.verdict(rl.check(lines))['verdict'],'OK')
            self.assertEqual(writer.faults,0)
            self.assertEqual(writer.oversize,0)
            self.assertTrue(all(len(data)<=3968 for _,data in cap.writes))
            self.assertEqual(len(lines),4)  # path, layer, bucket witness, summary
        finally:atomic.close_pipe(writer)

    def test_layer_bound(self):
        trace,d,op=recorder();name=['qsa.0']
        op._current_run=lambda:NS(layer_name=name[0],valid_tokens=201261)
        for i in range(32):
            name[0]='qsa.'+str(i);trace.end(trace.begin('score','compute'))
        self.assertEqual(len(trace.rows),24);self.assertEqual(trace.dropped,8)


class IndexerCPU(unittest.TestCase):
    def test_actual_generated_indexer_sampled_and_unsampled_are_bit_equal(self):
        import torch
        helper=life.CpuLifetime();helper.torch=torch;torch.set_num_threads(4)
        text=life.build.compose_indexer(life.source())
        for history,count in ((97152,1024),(201261,1352)):
            with self.subTest(history=history,count=count):
                expected,a=helper.run_case(life.source(),count,history)
                trace,d,op=recorder()
                with nm.indexer_phase_scope(trace):got,b=helper.run_case(text,count,history)
                self.assertTrue(torch.equal(expected,got));self.assertEqual(a.calls,b.calls)
                self.assertEqual(b.ledger.peak_slices,1)
                tiles=sum(c[0]=='score' for c in b.calls)
                self.assertEqual(len(trace.pairs),2*tiles)
                self.assertEqual({p[1] for p in trace.pairs},{'score','topk'})
                self.assertEqual((d.queries,d.reads),(0,0))
        self.assertIsNone(nm.INDEXER_PHASE_CONTEXT.get())
        self.assertFalse(torch.cuda.is_initialized())


class RealFused(fused.FusedCase):
    def run_network(self,sample,sink_fail=False):
        network=fused.FusedNetwork(self)
        for stack in network.qsa.values():
            stack.device.native_fused_phase_every=1 if sample else 0
            stack.phase_events=[]
            def timing(stream,stack=stack):
                event=stack.device.record(stream);event.elapsed_time=lambda other:.25
                stack.phase_events.append(event);return event
            stack.device.timing_event=timing
            stack.device.stream_token=lambda stream:{'compute':101,'copy':202}[stream]
        network.run_until(lambda:network.layer_accepted(0) and network.layer_accepted(1),ready=())
        for stack in network.qsa.values():
            self.assertFalse(any(e.ready for e in stack.phase_events))
            self.assertEqual(stack.device.synchronized,0)
            self.assertEqual(stack.device.host_reads,[])
            self.assertEqual(stack.operation.phase_trace is not None,sample)
            if sample:
                self.assertEqual(len(stack.operation.phase_trace.pairs),18) # three bucket paths and three producer brackets
                self.assertTrue(stack.operation.phase_trace.rows[stack.attention.layer_name]['fused'])
        # The fixture indexer is a kernel leaf; it does not forge score/top-k timing.
        captured=[]
        def sink(lines,logger):
            if sink_fail=='refused':return False
            if sink_fail:raise OSError('fixture sink')
            captured.extend(lines)
        with patch.object(nm,'emit_receipt_lines',side_effect=sink):
            network.finish()
        for stack in network.qsa.values():self.assertTrue(stack.operation._released)
        evidence=[json.loads(s[len(nm.FusedPhaseSpans.PREFIX):]) for s in captured if s.startswith(nm.FusedPhaseSpans.PREFIX)]
        if sink_fail:
            self.assertTrue(all(s.operation.stats['phase_emit_errors']==1 for s in network.qsa.values()))
        if sample and not sink_fail:
            self.assertTrue(evidence)
            self.assertTrue(all(not r['complete'] for r in evidence),'fake indexer cannot certify score coverage')
            layers=[r for r in evidence if r.get('kind')!='summary']
            self.assertTrue(all(r['history']==fused.HISTORY+64 and r['pages']==50 for r in layers))
            self.assertTrue(all({'indexer','canonical_selection','kv_producer_update'} <= r['phases'].keys() for r in layers))
            buckets=[json.loads(s[len(nm.FusedPhaseSpans.WITNESS_PREFIX):]) for s in captured
                     if s.startswith(nm.FusedPhaseSpans.WITNESS_PREFIX)]
            self.assertEqual(len(buckets),6)
            self.assertTrue(all(r['consumer_kind']==('none' if r['bucket']==0 else 'previous_bucket') for r in buckets))
        return ([(e.kind,e.descriptor.layer_start) for e in network.sent()],
                [(s.device.page_copies,s.operation.stats['bytes'],len(s.device.waits),s.operation.stats['sparse_source_released']) for s in network.qsa.values()])

    def test_same_real_fused_commands_copies_holds_and_dependencies(self):
        self.assertEqual(self.run_network(False),self.run_network(True))

    def test_phase_sink_failure_cannot_quarantine_or_strand_release(self):
        for mode in (True,'refused'):
            with self.subTest(mode=mode):self.run_network(True,sink_fail=mode)


if __name__=='__main__':unittest.main()
