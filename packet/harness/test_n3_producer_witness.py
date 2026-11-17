"""Real native dependencies over a declared logical-stream device, never GPU evidence."""
import json
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'harness'),str(ROOT/'patches/s5_engine')]
import native_measurement as nm
import qsa_operation as qo
import test_e3_sparse_history_dma as sparse
import test_n3_fused_phases as phases
import test_alpha_next_receipts as atomic
import receipt_lines as rl


class TimingEvent:
    def __init__(self,port,stream):
        self.port,self.stream,self.at=port,stream,port.clock[stream]
    def query(self):
        assert self.port.reading,'timing event queried before release'
        return True
    def elapsed_time(self,other):
        assert self.port.reading and self.stream==other.stream,'cross-stream or eager elapsed read'
        return other.at-self.at


class StreamDevice(sparse.FakeSparseDevice):
    """CUDA port: two logical clocks, real event edges, deterministic synthetic milliseconds."""
    def __init__(self):
        super().__init__(auto_ready=True)
        self.clock={'compute':0.,'copy':0.};self.reading=False;self.timing=[];self.fail_timing=False
    def record(self,stream):
        event=super().record(stream);event.at=self.clock[stream];return event
    def wait_event(self,stream,event):
        super().wait_event(stream,event)
        self.clock[stream]=max(self.clock[stream],event.at)
    def timing_event(self,stream):
        if self.fail_timing:raise OSError('fixture timing allocation')
        event=TimingEvent(self,stream);self.timing.append(event);return event
    def stream_token(self,stream):return {'compute':101,'copy':202}[stream]
    def copy_pages(self,source,arena,source_page,arena_page,pages):
        super().copy_pages(source,arena,source_page,arena_page,pages)
        self.clock['copy']+=pages/4


class DependencyOrder(sparse._GeneratedHarness,unittest.TestCase):
    for _name in ('make_operation','hold_for','bind','prepare','open_layer'):
        locals()[_name]=getattr(sparse.GeneratedSparseTests,_name)

    def execute(self,sampled,fail_timing=False):
        self.device=StreamDevice();self.device.native_fused_phase_every=int(sampled)
        self.device.native_qsa_policy='dense';self.device.fail_timing=fail_timing
        mapping=tuple(range(50));op,lane,tx,sources=self.prepare({'qsa.0':mapping,'qsa.1':mapping})
        for li,(name,source) in enumerate(sources.items()):
            self.device.clock['compute']+=10. # leaf: intervening native model/producer work
            self.open_layer(op,lane,tx,name,source,mapping,min_pages=24)
            run=op._current_run()
            if op.phase_trace is not None:op.phase_trace.group=(li*2,li*2+2)
            for index,bucket in enumerate(run.buckets):
                old=op._last_arena_consumer;before=len(self.device.waits)
                producer=run.producer.event
                prior=old if index==0 else run.consumed[-1].event
                op.fill_bucket(self.arena[:bucket.pages],bucket.lo,bucket.hi,self.table)
                actual=self.device.waits[before:]
                wanted=[(self.device.copy,producer)] if index==0 else []
                if prior is not None:wanted.append((self.device.copy,prior))
                self.assertEqual(actual[:-1],wanted,'native producer/consumer order changed')
                self.assertEqual(actual[-1][0],self.device.compute)
                self.assertEqual(actual[-1][1].token,self.device.copy,'compute lost its copy-ready edge')
                self.device.clock['compute']+=3. # leaf: native attention execution
                op.after_attention_bucket(source,self.arena)
            op.end_layer(name)
        op.end_forward();self.device.ready_all();self.assertTrue(op.drain())
        self.assertFalse(self.device.synchronized);self.assertEqual(self.device.host_reads,[])
        captured=[];self.device.reading=True
        with patch.object(nm,'emit_receipt_lines',side_effect=lambda rows,logger:captured.extend(rows)):
            op.release()
        signature=([(stream,self.device.events.index(event)) for stream,event in self.device.waits],
                   self.device.page_copies,self.device.clock,op.stats['sparse_source_released'])
        return signature,captured,op

    def test_two_layers_three_buckets_preserve_dependencies_off_on(self):
        off,_,_=self.execute(False);on,texts,op=self.execute(True)
        self.assertEqual(off,on)
        rows=[json.loads(s[len(nm.FusedPhaseSpans.WITNESS_PREFIX):]) for s in texts if s.startswith(nm.FusedPhaseSpans.WITNESS_PREFIX)]
        self.assertEqual(len(rows),6)
        by={(r['layer'],r['bucket']):r for r in rows}
        first=by['qsa.0',0];following=by['qsa.0',1];next_layer=by['qsa.1',0]
        self.assertEqual(first['consumer_kind'],'none')
        self.assertNotIn('wait_previous_arena_consumer',first['phases'])
        self.assertEqual(first['phases']['wait_current_producer']['ms'],10.)
        self.assertEqual(first['phases']['compute_wait_for_copy']['ms'],6.)
        self.assertEqual((following['consumer_kind'],following['consumer_layer'],following['consumer_bucket']),('previous_bucket','qsa.0',0))
        self.assertEqual(following['phases']['wait_previous_arena_consumer']['ms'],3.)
        self.assertEqual((next_layer['consumer_kind'],next_layer['consumer_layer'],next_layer['consumer_bucket']),('previous_layer','qsa.0',2))
        self.assertEqual(next_layer['phases']['wait_current_producer']['ms'],13.)
        self.assertEqual(next_layer['phases']['wait_previous_arena_consumer']['ms'],0.)
        self.assertTrue(all(r['copy_stream']==202 and r['producer_stream']==101 for r in rows))
        self.assertTrue(all(v['host_ns']>=0 for r in rows for v in r['phases'].values()))
        # Direct component fixture has no indexer or fused execution: never forge a complete box witness.
        summary=json.loads(texts[-1][len(nm.FusedPhaseSpans.PREFIX):])
        self.assertFalse(summary['complete']);self.assertEqual(summary['bucket_rows'],6)
        layer_rows=[json.loads(s[len(nm.FusedPhaseSpans.PREFIX):]) for s in texts[:-1] if s.startswith(nm.FusedPhaseSpans.PREFIX)]
        self.assertTrue(all(r['history']==50*4048 and r['pages']==50 and r['buckets']==3 for r in layer_rows))

    def test_timing_allocation_failure_preserves_execution(self):
        off,_,_=self.execute(False);on,texts,_=self.execute(True,fail_timing=True)
        self.assertEqual(off,on)
        summary=json.loads(texts[-1][len(nm.FusedPhaseSpans.PREFIX):])
        self.assertFalse(summary['complete']);self.assertGreater(summary['errors'],0)


class BoundsAndRecords(unittest.TestCase):
    def test_consumer_stream_is_snapshotted_at_its_record(self):
        trace,d,op=phases.recorder();event=object()
        trace.consumer(event,'qsa.0',0)
        with patch.object(d,'stream_token',side_effect=lambda stream:303 if stream==d.compute else 202):
            trace.bucket(1,event,'previous_bucket')
        row=trace.buckets['qsa.0',1]
        self.assertEqual(row['consumer_stream'],101)
        self.assertEqual(row['producer_stream'],303)

    def test_history_refreshed_after_initial_zero_latch(self):
        trace,d,op=phases.recorder();run=op._current_run();run.valid_tokens=0
        phases.all_phases(trace);self.assertEqual(trace.rows['qsa.0']['history'],0)
        run.valid_tokens=201261
        data=phases.rows(trace,d)
        self.assertTrue(data[-1]['complete']);self.assertEqual(data[0]['history'],201261)
        self.assertEqual(data[0]['pages'],50)

    def test_bucket_and_consumer_metadata_are_bounded(self):
        trace,d,op=phases.recorder()
        events=[object() for _ in range(140)]
        for i,event in enumerate(events):
            trace.consumer(event,'qsa.0',i);trace.bucket(i,None,'none')
        self.assertEqual(len(trace.buckets),128);self.assertEqual(len(trace.consumers),128)
        self.assertGreater(trace.dropped,0)
        self.assertFalse(phases.rows(trace,d)[-1]['complete'])

    def test_atomic_writer_new_prefix_and_long_fields_fit(self):
        trace,d,op=phases.recorder();phases.all_phases(trace);d.reading=True
        op.transaction=replace(op.transaction,engine_epoch=2**63-1,txn_seq=2**63-1,request_generation=2**63-1)
        trace.route_generation=2**63-1
        for b in trace.buckets.values():b.update(producer_stream=2**63-1,copy_stream=2**63-2,consumer_stream=2**63-1)
        writer,cap=atomic.capture_writer()
        try:
            with patch.object(nm,'_RECEIPT_WRITER',writer),patch.object(phases.Event,'elapsed_time',return_value=123456789.12345678):trace.emit()
            lines=atomic.lines_from(cap.writes)
            self.assertEqual(rl.verdict(rl.check(lines))['verdict'],'OK')
            self.assertEqual(writer.oversize,0);self.assertEqual(writer.faults,0)
            self.assertTrue(any(nm.FusedPhaseSpans.WITNESS_PREFIX in line for line in lines))
            self.assertTrue(all(len(data)<=3968 for _,data in cap.writes))
        finally:atomic.close_pipe(writer)

    def test_worst_pinned_geometry_fits_existing_event_budget(self):
        # Six local QSA layers, max startup2048 rows,262144 history,64MiB FP32 score cap,24-page arena.
        width=((262144//4+63)//64)*64;tiles=-(-2048//(64*2**20//(width*4)))
        pages=(262144+4047)//4048;buckets=(pages+23)//24
        self.assertEqual(tiles,8);self.assertEqual(buckets,3)
        markers=0
        for layer in range(6):
            pairs=2*tiles+3+4*buckets+1+(buckets-1)+int(layer>0)
            markers+=2*pairs
        self.assertEqual(markers,418);self.assertLess(markers,nm.FusedPhaseSpans.MAX_MARKERS)
        trace,d,op=phases.recorder();last=None
        for layer in range(6):
            name='qsa.'+str(layer);run=NS(layer_name=name,valid_tokens=262144,
                pages=tuple(range(pages)),buckets=tuple(range(buckets)))
            op._layers[name]=run;op._current_run=lambda run=run:run
            trace.group=(4*layer+2,4*layer+4);trace.fused_layers.add(name)
            whole=trace.begin('indexer',d.compute)
            for _ in range(tiles):
                for phase in ('score','topk'):trace.end(trace.begin(phase,d.compute))
            trace.end(whole)
            for phase in ('canonical_selection','kv_producer_update'):trace.end(trace.begin(phase,d.compute))
            for bucket in range(buckets):
                trace.bucket(bucket,last,'none' if last is None else 'previous_layer' if bucket==0 else 'previous_bucket')
                waiting=trace.begin('copy_wait',d.copy,bucket=bucket)
                if bucket==0:trace.end(trace.begin('wait_current_producer',d.copy,bucket=bucket))
                if last is not None:trace.end(trace.begin('wait_previous_arena_consumer',d.copy,bucket=bucket))
                trace.end(waiting)
                for phase,stream in [('copy',d.copy),('compute_wait_for_copy',d.compute),('attention',d.compute)]:
                    trace.end(trace.begin(phase,stream,bucket=bucket))
                last=object();trace.consumer(last,name,bucket)
        data=phases.rows(trace,d)
        self.assertEqual(trace.markers,markers);self.assertEqual(len(d.records),markers)
        self.assertTrue(data[-1]['complete']);self.assertEqual(data[-1]['bucket_rows'],18)
        self.assertEqual(trace.dropped,0)


if __name__=='__main__':unittest.main()
