"""Immutable whole-descriptor lookahead, real operation; explicit CPU stream/tensor leaves."""
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'harness'), str(ROOT/'patches/s5_engine')]
import qsa_operation as qo
import native_measurement as nm
import test_e3_sparse_history_dma as s
import test_n3_producer_witness as w
import test_e3_native_qsa_fused as fused
import test_e3_native_submission_credit as c
import qsa_driver
import sparse_history_dma as shd
import e3_types as et
import lane_chunk
import test_n3_fused_phases as phases
import test_alpha_next_receipts as atomic
import receipt_lines as rl


class Device(w.StreamDevice):
    """Two logical clocks; snapshots of page identities stand in for GPU values."""
    def __init__(self, enabled=True):
        super().__init__()
        self.native_next_layer_prefetch = enabled
        self.native_qsa_policy = 'dense'
        self.arena_pages = {}
        self.order = []
    def copy_pages(self, source, arena, source_page, arena_page, pages):
        super().copy_pages(source, arena, source_page, arena_page, pages)
        for i in range(pages):
            self.arena_pages[arena_page+i] = (source.address, source_page+i)
        self.order.append(('copy', self.clock['copy'], source_page, arena_page, pages))
    def wait_event(self, stream, event):
        super().wait_event(stream, event)
        self.order.append(('wait', stream, event.token, event.at))


class PrefixCase(s._GeneratedHarness, unittest.TestCase):
    def setup_operation(self, start=201261, width=2024, enabled=True, mapping=None, blocked=(), cow=()):
        self.device = Device(enabled)
        tx=s.make_transaction(chunk_start=start,chunk_end=start+width)
        lane=s.make_lane()
        pages=(tx.chunk_end+4047)//4048
        mapping=tuple(mapping if mapping is not None else range(1,pages+1))
        op=qo.QsaPrefillOperation(tx,lane,device=self.device,plan_provider=qo.metadata_plan_provider(
            s.Snapshot([s.Row()],range(260)),layer_tables=(('qsa.0',mapping),('qsa.1',mapping))))
        op.state_epoch=1
        frame=lane_chunk.ChunkAdmission(tx,tx.chunk_end+1,(1,)*width,(),(mapping,),
                 new_block_ids_to_zero=blocked,cow_pairs=cow,state_epoch=1)
        if enabled:op.bind_prefetch_frame(frame)
        self.sources={}
        for i,name in enumerate(('qsa.0','qsa.1')):
            source=self.make_source(max(mapping)+1)
            source.address=0x10000000*(i+1)
            op.register_source(name,source,versions=tuple(range(max(mapping)+2)))
            hold=shd.SparseSourceHold(s.make_owner(lane,name,mapping,tx.chunk_end,transaction=tx),allocator=s.FakeAllocator())
            op.attach_sparse_source(name,hold)
            self.sources[name]=source
        op.begin_forward(layers=tuple(self.sources))
        return op,mapping,frame

    def layer(self,op,name,mapping):
        self.device.clock['compute']+=10
        op.begin_layer(name,self.sources[name])
        op.producer_submitted(name,self.sources[name])
        op.bind_plan(name,self.table,len(mapping),op.transaction.chunk_end,self.arena)
        run=op._current_run()
        # Same immutable dense profile as the real four-worker grant.
        run.schedule_grant=NS(fused=True)
        values=[]
        for bucket in run.buckets:
            op.fill_bucket(self.arena[:bucket.pages],bucket.lo,bucket.hi,self.table)
            # CPU values are a page-identity shadow, not BF16 attention numerics.
            values.append(tuple(self.device.arena_pages[i] for i in range(bucket.pages)))
            self.device.clock['compute']+=3
            op.after_attention_bucket(self.sources[name],self.arena)
        op.end_layer(name)
        return values

    def execute(self,enabled,start=201261,width=2024,mapping=None):
        op,mapping,frame=self.setup_operation(start,width,enabled,mapping)
        result=[self.layer(op,'qsa.0',mapping)]
        before=self.device.clock['compute']
        if enabled:self.assertTrue(op.prefetch_next('qsa.1',1,qo.MAX_BUCKET_PAGES*qo.PAGE_BYTES,(2,4)))
        result.append(self.layer(op,'qsa.1',mapping))
        inventory=op.prefetch_inventory()
        op.end_forward();self.device.ready_all();self.assertTrue(op.drain());op.release()
        return result,self.device.page_copies,self.device.clock['compute']-before,inventory,op

    def test_same_bytes_descriptors_and_order_with_a_real_overlap_window(self):
        off=self.execute(False);on=self.execute(True)
        self.assertEqual(off[:2],on[:2])
        self.assertLess(on[2],off[2], 'lookahead lost its overlap with producer work')
        self.assertEqual((on[3]['rows'][0][3:9]),[1,1,0,24*qo.PAGE_BYTES,24*qo.PAGE_BYTES,0])
        self.assertEqual(on[3]['extra_tensor_bytes'],0)
        self.assertEqual(on[3]['extra_arena_bytes'],0)
        self.assertEqual(on[4]._prefix_copies,[])
        self.assertTrue(all(r.sparse_hold.released for r in on[4]._layers.values()))

    def test_geometry_matrix_including_fragmented_maps_and_partial_descriptors(self):
        for start in (52623,97151,97152,133679,201261):
            for width in (1024,1352,2024):
                with self.subTest(start=start,width=width):
                    count=(start+width+4047)//4048
                    mapping=tuple(i+1+(i//7)*3 for i in range(count))
                    off=self.execute(False,start,width,mapping)
                    on=self.execute(True,start,width,mapping)
                    self.assertEqual(off[:2],on[:2])
                    self.assertTrue(all(n<=4 for _,_,n in on[1]))

    def test_mutable_boundary_keeps_the_whole_crossing_descriptor_cold(self):
        op,mapping,_=self.setup_operation(start=5*4048+100,width=1024)
        self.layer(op,'qsa.0',mapping)
        self.assertTrue(op.prefetch_next('qsa.1',1,24*qo.PAGE_BYTES))
        self.assertEqual(op._prefix_pending.pieces,((1,0,4),))
        self.assertEqual(op._prefix_rows['qsa.1']['issued_bytes'],4*qo.PAGE_BYTES)

    def test_first_layer_fresh_short_and_zero_or_cow_pages_stay_cold(self):
        for start,blocked,cow,mapping in [(0,(),(),None),(1024,(),(),None),
                (201261,(1,),(),None),(201261,(),((1,90),),None),
                (201261,(),(),tuple(range(51)))]:
            with self.subTest(start=start,blocked=blocked,cow=cow):
                op,mapping,_=self.setup_operation(start=start,width=2024,blocked=blocked,cow=cow,mapping=mapping)
                self.layer(op,'qsa.0',mapping)
                before=list(self.device.page_copies)
                self.assertFalse(op.prefetch_next('qsa.1',1,24*qo.PAGE_BYTES))
                self.assertEqual(before,self.device.page_copies)

    def test_waits_for_last_consumer_not_first_and_adoption_keeps_producer_edge(self):
        op,mapping,_=self.setup_operation()
        self.layer(op,'qsa.0',mapping)
        run=op._current_run();last=run.consumed[-1].event
        self.assertGreater(last.at,run.consumed[0].event.at)
        before=len(self.device.waits)
        self.assertTrue(op.prefetch_next('qsa.1',1,24*qo.PAGE_BYTES))
        self.assertEqual(self.device.waits[before:],[('copy',last)])
        prefix=op._prefix_pending.event
        self.assertGreaterEqual(prefix.at,last.at+6)
        self.layer(op,'qsa.1',mapping)
        producer=op._layers['qsa.1'].producer.event
        self.assertIn(('copy',producer),self.device.waits)
        self.assertEqual(op._prefix_rows['qsa.1']['used'],1)

    def test_foreign_frame_stale_version_or_wrong_arena_refuse_before_copy(self):
        for case in ('frame','version','arena','consumer','source'):
            with self.subTest(case=case):
                op,mapping,_=self.setup_operation();self.layer(op,'qsa.0',mapping)
                run=op._layers['qsa.1']
                if case=='frame':op._prefix_frame=replace(op._prefix_frame,state_epoch=2)
                if case=='version':op.source_versions['qsa.1']=(999,)
                if case=='arena':op._current_run().arena.address+=128
                if case=='consumer':op._last_arena_consumer=op._current_run().consumed[0].event
                if case=='source':run.source.address+=128
                count=len(self.device.page_copies)
                self.assertFalse(op.prefetch_next('qsa.1',1,24*qo.PAGE_BYTES))
                self.assertEqual(count,len(self.device.page_copies))

    def test_stale_adoption_fails_closed_without_a_silent_recopy(self):
        for case in ('epoch','pages','address','used'):
            with self.subTest(case=case):
                op,mapping,_=self.setup_operation();self.layer(op,'qsa.0',mapping)
                self.assertTrue(op.prefetch_next('qsa.1',1,24*qo.PAGE_BYTES))
                prefix=op._prefix_pending
                if case=='epoch':op.state_epoch+=1
                if case=='pages':prefix.pages=tuple(reversed(prefix.pages))
                if case=='address':self.sources['qsa.1'].address+=128
                if case=='used':prefix.used=True
                with self.assertRaises((qo.OperationError,shd.SourceRefused)):
                    self.layer(op,'qsa.1',mapping)
                self.assertFalse(op._released)

    def test_each_adoption_identity_component_is_a_fail_closed_boundary(self):
        for case in ('index','layer','identity','pages','source','hold','source-facts','arena','pieces','used','event'):
            with self.subTest(case=case):
                op,mapping,_=self.setup_operation();self.layer(op,'qsa.0',mapping)
                self.assertTrue(op.prefetch_next('qsa.1',1,24*qo.PAGE_BYTES))
                op.begin_layer('qsa.1',self.sources['qsa.1'])
                op.producer_submitted('qsa.1',self.sources['qsa.1'])
                op.bind_plan('qsa.1',self.table,len(mapping),op.transaction.chunk_end,self.arena)
                run=op._current_run();record=op._prefix_pending;pieces=run.buckets[0].pieces;index=0
                if case=='index':index=1
                if case=='layer':record.layer_name='foreign'
                if case=='identity':record.identity=()
                if case=='pages':record.pages=tuple(reversed(record.pages))
                if case=='source':record.owners=(object(),*record.owners[1:])
                if case=='hold':record.owners=(*record.owners[:2],object())
                if case=='source-facts':record.source_facts=replace(record.source_facts,address=99)
                if case=='arena':record.arena_facts=replace(record.arena_facts,address=99)
                if case=='pieces':pieces=tuple(reversed(pieces))
                if case=='used':record.used=True
                if case=='event':record.event=None
                with self.assertRaisesRegex(qo.OperationError,'prefetched prefix differs'):
                    op._adopt_prefix(run,index,pieces)
                self.assertTrue(op._failed)


    def test_cancel_retains_prefetch_until_its_own_completion_and_reports_waste(self):
        op,mapping,_=self.setup_operation();self.layer(op,'qsa.0',mapping)
        self.assertTrue(op.prefetch_next('qsa.1',1,24*qo.PAGE_BYTES))
        prefix=op._prefix_pending
        self.device.ready_all();prefix.event.ready=False
        op.cancel('test-cancel')
        self.assertFalse(op.drain())
        with self.assertRaises(et.E3Pending):op.release()
        self.assertTrue(prefix.owners)
        self.assertEqual(op.prefetch_inventory()['rows'][0][3:6],[1,0,1])
        prefix.event.ready=True
        self.assertTrue(op.drain());op.release()
        self.assertEqual(op._prefix_copies,[])
        self.assertTrue(all(r.sparse_hold.released for r in op._layers.values()))

    def test_partial_enqueue_failure_keeps_owners_and_quarantines(self):
        op,mapping,_=self.setup_operation();self.layer(op,'qsa.0',mapping)
        copy=self.device.copy_pages;calls=[]
        def fail(*args):
            calls.append(1)
            if len(calls)==2:raise RuntimeError('fixture partial DMA')
            return copy(*args)
        with patch.object(self.device,'copy_pages',side_effect=fail):
            with self.assertRaisesRegex(RuntimeError,'partial DMA'):
                op.prefetch_next('qsa.1',1,24*qo.PAGE_BYTES)
        self.assertTrue(op._failed);self.assertFalse(op.drain())
        self.assertTrue(op._prefix_pending.owners)
        with self.assertRaises(et.E3Pending):op.release()

    def test_second_chunk_has_new_identity_no_cross_chunk_cached_values(self):
        first=self.execute(True)
        second=self.execute(True,start=203285,width=1352)
        self.assertNotEqual(first[4].transaction,second[4].transaction)
        self.assertEqual(first[4]._prefix_copies,second[4]._prefix_copies)
        self.assertEqual(second[3]['rows'][0][3:6],[1,1,0])

    def test_only_one_pending_prefix_and_at_most_six_local_layers(self):
        op,mapping,_=self.setup_operation();self.layer(op,'qsa.0',mapping)
        self.assertTrue(op.prefetch_next('qsa.1',1,24*qo.PAGE_BYTES))
        before=list(self.device.page_copies)
        self.assertFalse(op.prefetch_next('qsa.1',1,24*qo.PAGE_BYTES))
        self.assertEqual(before,self.device.page_copies)
        op,mapping,_=self.setup_operation();self.layer(op,'qsa.0',mapping)
        for i in range(2,7):op._layers['extra.'+str(i)]=op._layers['qsa.1']
        self.assertFalse(op.prefetch_next('qsa.1',1,24*qo.PAGE_BYTES))


    def test_forecast_refuses_out_of_pool_and_missing_page_versions_before_dma(self):
        for case in ('source','versions'):
            op,mapping,_=self.setup_operation();self.layer(op,'qsa.0',mapping)
            run=op._layers['qsa.1']
            if case=='source':
                run.source.shape=(2,1,4048,512)
                run.source_facts=self.device.facts(run.source)
            else:
                run.source_versions=(1,2);op.source_versions['qsa.1']=(1,2)
            copies=list(self.device.page_copies)
            self.assertFalse(op.prefetch_next('qsa.1',1,24*qo.PAGE_BYTES))
            self.assertEqual(copies,self.device.page_copies)

    def test_only_the_exact_frozen_admission_can_seed_a_prefetch_frame(self):
        op,mapping,frame=self.setup_operation(enabled=False)
        fresh=qo.QsaPrefillOperation(op.transaction,op.lane,device=self.device)
        with self.assertRaisesRegex(qo.OperationError,'prefetch frame differs'):
            fresh.bind_prefetch_frame(NS(transaction=frame.transaction,state_epoch=1))


    def test_lost_lease_never_treats_allocation_versions_as_a_ready_certificate(self):
        op,mapping,_=self.setup_operation();self.layer(op,'qsa.0',mapping)
        hold=op._layers['qsa.1'].sparse_hold
        hold._allocator.refs.clear()
        copies=list(self.device.page_copies)
        with self.assertRaises((RuntimeError,shd.SourceRefused)):
            op.prefetch_next('qsa.1',1,24*qo.PAGE_BYTES)
        self.assertEqual(copies,self.device.page_copies)

    def test_real_atomic_alloc_receipt_reports_per_layer_counts_and_bytes(self):
        op,mapping,_=self.setup_operation();self.layer(op,'qsa.0',mapping)
        self.assertTrue(op.prefetch_next('qsa.1',1,24*qo.PAGE_BYTES,(2,4)))
        self.layer(op,'qsa.1',mapping);op.end_forward();self.device.ready_all();self.assertTrue(op.drain())
        op.memory_receipts=True
        writer,cap=atomic.capture_writer()
        try:
            with patch.object(nm,'_RECEIPT_WRITER',writer):op.release()
            lines=atomic.lines_from(cap.writes)
            self.assertEqual(rl.verdict(rl.check(lines))['verdict'],'OK')
            line=next(x for x in lines if x.startswith('flash-s5 native-alloc '))
            text=line.split(' prefix_prefetch=',1)[1].split(' pid=',1)[0]
            # Receipt suffix is independently hashed by the real writer.
            data=json.JSONDecoder().raw_decode(text)[0]
            self.assertEqual(data['rows'][0][3:9],[1,1,0,24*qo.PAGE_BYTES,24*qo.PAGE_BYTES,0])
            self.assertEqual((writer.oversize,writer.faults),(0,0))
            self.assertTrue(all(len(b)<=3968 for _,b in cap.writes))
        finally:atomic.close_pipe(writer)


    def test_full_source_hold_is_validated_and_no_device_reads_or_waits_on_host(self):
        op,mapping,_=self.setup_operation();self.layer(op,'qsa.0',mapping)
        hold=op._layers['qsa.1'].sparse_hold
        with patch.object(hold,'validate',wraps=hold.validate) as validate:
            self.assertTrue(op.prefetch_next('qsa.1',1,24*qo.PAGE_BYTES))
        self.assertEqual(validate.call_args.args,(hold.owner,'qsa.1',mapping))
        self.assertEqual(self.device.host_reads,[]);self.assertEqual(self.device.synchronized,0)


class TwoLayerNetwork(fused.FusedNetwork):
    """Four typed workers; stage0 owns two QSA layers sharing the real helper arena."""
    def __init__(self,harness,enabled=True,cold_rank=None):
        super().__init__(harness)
        for rank,stack in self.qsa.items():
            stack.device.native_next_layer_prefetch=enabled
            second=c.FakeAttention(stack,stack.kv_cache)
            second.layer_name='qsa.1'
            second._qsa_host_kv=s.T((51,1,4048,512),device='cpu',dtype='torch.bfloat16',pinned=True)
            stack.model.layers[1]=c.FakeQsaLayer(second,stack.tokens)
            stack.attn_metadata['qsa.1']=stack.main
            # The same side metadata and selected indices, independent source allocation.
            sources={'qsa.0':stack.source,'qsa.1':second._qsa_host_kv}
            def make(transaction,stack=stack,sources=sources):
                op=qo.QsaPrefillOperation(transaction,stack.lane,device=stack.device,
                    plan_provider=qo.metadata_plan_provider(s.Snapshot([s.Row()],range(260)),
                        layer_tables=tuple((name,stack.mapping) for name in sources)))
                op.state_epoch=1
                return op
            stack.make_operation=make
            def begin(runner,marker,stack=stack,sources=sources,rank=rank):
                op=stack.server.operation
                for name,source in sources.items():
                    op.register_source(name,source,versions=stack.versions)
                    hold=shd.SparseSourceHold(s.make_owner(stack.lane,name,stack.mapping,
                        stack.transaction.chunk_end,transaction=op.transaction),allocator=s.FakeAllocator())
                    op.attach_sparse_source(name,hold);stack.holds.append(hold)
                op.begin_forward(layers=tuple(sources))
                if rank==cold_rank:op._prefix_frame=None
            stack.server.begin_prefill_operation=begin
            def factory(operation,module,context,stack=stack):
                stack.driver=qsa_driver.QsaLayerDriver(operation,staged_operation=harness.helper_dense,
                    sparse_staged_operation=harness.helper_sparse,forward_context=stack._forward_scope,
                    torch_module=stack.torch,clock=lambda:0)
                return stack.driver
            stack.dispatcher.driver_factory=factory
        self.install_tp()

    def finish(self,ready=(0,1)):
        super().finish(ready)
        self.driver.stop('cancel')
        self.pump_stop(ready)


class Connected(fused.FusedCase):
    def assert_closed(self,network):
        # Normal terminal completion precedes stop; no cancellation closure is fabricated.
        self.assertIsNone(network.driver.error)
        self.assertFalse(network.driver.active)
        self.assertFalse(network.driver._inflight)
        self.assertIsNone(network.driver.tx)
        self.assertEqual(network.transport._calls,{})
        self.assertTrue(network.controller.leases.shutdown()['clean'])
        for stack in network.qsa.values():
            self.assertIsNone(stack.bridge.transaction)
            self.assertIsNone(stack.dispatcher.transaction)
            self.assertTrue(stack.operation._released)
            self.assertTrue(all(h.released for h in stack.holds))
    def test_real_four_worker_fused_driver_issues_and_adopts_next_layer_without_new_profile(self):
        results={}
        for enabled in (False,True):
            network=TwoLayerNetwork(self,enabled)
            network.finish();self.assert_closed(network)
            results[enabled]=[]
            for rank,stack in network.qsa.items():
                op=stack.operation
                self.assertEqual(op.prefetch_enabled,enabled)
                self.assertEqual(op._prefix_frame,None) # cleared only by real drain/release
                self.assertEqual(op._prefix_rows.get('qsa.1',{}).get('used',0),int(enabled))
                self.assertEqual(stack.device.synchronized,0)
                self.assertEqual(stack.device.host_reads,[])
                self.assertEqual(stack.attention.pre_calls,1)
                self.assertEqual(stack.model.layers[1].self_attn.pre_calls,1)
                results[enabled].append((stack.device.page_copies,
                    tuple((k,op.stats[k]) for k in ('producers','buckets','pieces','bytes','sparse_source_released')),
                    len(network.profile_polls()),len(network.sent())))
        self.assertEqual(results[False],results[True])

    def test_one_tp_peer_can_take_cold_path_without_changing_collective_or_profile_order(self):
        network=TwoLayerNetwork(self,True,cold_rank=1)
        network.finish();self.assert_closed(network)
        self.assertEqual(network.qsa[0].operation._prefix_rows['qsa.1']['used'],1)
        self.assertEqual(network.qsa[1].operation._prefix_rows['qsa.1']['used'],0)
        self.assertEqual(network.qsa[0].device.page_copies,network.qsa[1].device.page_copies)
        self.assertEqual(network.tp[0].pynccl_comm.calls,network.tp[1].pynccl_comm.calls)


class SwitchAndBudget(unittest.TestCase):
    def test_startup_switch_is_strict_and_default_off(self):
        for value,want in [(None,False),('0',False),('1',True)]:
            self.assertIs(qo.next_layer_prefetch({} if value is None else {'FLASH_S5_NEXT_LAYER_PREFETCH':value}),want)
        for value in ('true','2','',1):
            with self.assertRaises(qo.OperationError):qo.next_layer_prefetch({'FLASH_S5_NEXT_LAYER_PREFETCH':value})
    def test_future_layer_sampler_refreshes_history_fused_and_actual_group(self):
        trace,d,op=phases.recorder()
        trace.fused_layers.clear()
        op._layer=lambda name:op._layers[name]
        op._prefix_rows={'qsa.0':dict(issued=1,group=(14,16))}
        token=trace.begin('prefetch_wait_for_arena',d.copy,layer_name='qsa.0',group=(0,2));trace.end(token)
        trace.end(trace.begin('prefetch_copy',d.copy,layer_name='qsa.0',group=(0,2)))
        self.assertFalse(trace.rows['qsa.0']['fused'])
        trace.fused_layers.add('qsa.0');phases.all_phases(trace)
        data=phases.rows(trace,d)
        self.assertTrue(data[-1]['complete'])
        self.assertEqual(data[0]['group'],[14,16])
        self.assertEqual(data[0]['phases']['prefetch_copy']['count'],1)


    def test_sampled_event_budget_and_receipt_schema_are_bounded(self):
        self.assertLessEqual(418+5*4,nm.FusedPhaseSpans.MAX_MARKERS)
        op=NS(_prefix_rows={f'qsa.{i}':dict(layer=i,group=(i*2,i*2+2),issued=1,used=1,
            issued_bytes=24*qo.PAGE_BYTES,used_bytes=24*qo.PAGE_BYTES,descriptors=6,why='used',plan_ns=10**9,enqueue_ns=10**9) for i in range(6)},
            _prefix_copies=[],_prefix_overflow=0)
        record=qo.QsaPrefillOperation.prefetch_inventory(op)
        self.assertLess(len(json.dumps(record,separators=(',',':'))),1800)
        self.assertEqual(record['overflow'],0)


if __name__=='__main__':unittest.main()
