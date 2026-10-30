"""Native31 host consensus/order and certified error-publication regressions."""
from pathlib import Path
import queue
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'patches/s5_engine'))
import lane_comm as lc
from engine_driver import NativeSegmentDriver,DriverError

class Matrix:
    shape=(8,7);dtype='int32';device='cuda:0'
    def __init__(self,rows):self.rows=[list(r) for r in rows]
    def is_contiguous(self):return True

class Selection(unittest.TestCase):
    def test_in_place_leader_and_current_lane_stream(self):
        stream=object();calls=[];wire={}
        torch=NS(int32='int32',cuda=NS(current_stream=lambda dev:stream))
        class Comm:
            disabled=False;world_size=2
            def __init__(self,rank):self.rank=rank
            def broadcast(self,tensor,src,stream):
                calls.append((self.rank,src,stream))
                if self.rank==src:wire['rows']=[r[:] for r in tensor.rows]
                else:tensor.rows[:]=[r[:] for r in wire['rows']]
        leader=Matrix([[i for i in range(7)] for _ in range(8)])
        peer=Matrix([[i+1 for i in range(7)] for _ in range(8)])
        old_leader=[r[:] for r in leader.rows];old_peer=[r[:] for r in peer.rows]
        for rank,tensor in enumerate((leader,peer)):
            group=NS(world_size=2,device_communicator=NS(pynccl_comm=Comm(rank)))
            lc.canonical_selection(tensor,group=group,torch_module=torch)
        self.assertEqual(leader.rows,old_leader);self.assertNotEqual(peer.rows,old_peer)
        self.assertEqual(peer.rows,old_leader);self.assertEqual(calls,[(0,0,stream),(1,0,stream)])

    def test_no_allocation_or_stream_lookup_tp1_and_fail_closed_tp2(self):
        lc.canonical_selection(object(),group=NS(world_size=1))
        with self.assertRaises(lc.LaneCommError):
            lc.canonical_selection(Matrix([[0]*7]*8),group=NS(world_size=2,device_communicator=None),torch_module=NS(int32='int32'))

    def test_interleaved_native_and_decode_collective_order_matches_both_peers(self):
        # The worker RPC FIFO serializes host submission, not GPU completion.
        # Each TP peer sees the same segment/decode command order. Distinct
        # streams/comms allow overlap; no reverse cross-stream wait is added.
        traces=[]
        for rank in (0,1):
            trace=[];current=['decode'];streams={'native':object(),'decode':object()}
            class Comm:
                disabled=False;world_size=2
                def __init__(self,lane):self.lane=lane
                def broadcast(self,tensor,src,stream):
                    self.assert_stream(stream)
                    trace.append((self.lane,'selection',src))
                def assert_stream(self,stream):
                    if stream is not streams[self.lane]:raise AssertionError('wrong lane stream')
                def all_reduce(self,tensor,stream):
                    self.assert_stream(stream);trace.append((self.lane,'all_reduce'));return tensor
            torch=NS(int32='int32',cuda=NS(current_stream=lambda device:streams[current[0]]))
            ordinary=NS(pynccl_comm=Comm('decode'))
            native=lc.PrefillTpCommunicator(Comm('native'),'cuda:0',torch_module=torch)
            group=NS(world_size=2,device_communicator=ordinary)
            def decode_step():
                current[0]='decode'
                lc.canonical_selection(Matrix([[0]*7]*8),group=group,torch_module=torch)
                ordinary.pynccl_comm.all_reduce(None,stream=streams['decode'])
            for chunk in range(4):
                current[0]='native'
                with lc.tp_scope(group,native):
                    lc.canonical_selection(Matrix([[0]*7]*8),group=group,torch_module=torch)
                self.assertIs(group.device_communicator,ordinary)
                decode_step()  # after native _pre
                for bucket in range(3):decode_step()  # sparse DMA/attention yields
                current[0]='native'
                with lc.tp_scope(group,native):native.all_reduce(Matrix([[0]*7]*8))
                self.assertIs(group.device_communicator,ordinary)
                decode_step()  # after native _tail's o_proj all-reduce
            traces.append(trace)
        self.assertEqual(traces[0],traces[1])
        self.assertEqual(len(traces[0]),48)

    def test_actual_sparse_planner_reproduces_e_wave_boundary(self):
        from sparse_history_dma import make_sparse_plan,stream_sparse_plan,PAGE_ROWS,ARENA_PAGES,MAX_DESCRIPTORS
        rows=[];cursor=0
        for i in range(3072):
            n=24 if i<1836 else 16
            if cursor%PAGE_ROWS+n>PAGE_ROWS:cursor=(cursor//PAGE_ROWS+1)*PAGE_ROWS
            rows.extend(range(cursor,cursor+n));cursor+=n+8
        mapping=tuple(range(1,(98144+PAGE_ROWS-1)//PAGE_ROWS+1));profiles=[]
        for values in (rows,rows+list(range(cursor,cursor+8))):
            p=make_sparse_plan(tuple(values),mapping,tuple(range(1,len(mapping)+2)),1,page_rows=PAGE_ROWS,arena_pages=ARENA_PAGES,max_descriptors=MAX_DESCRIPTORS,valid_tokens=98144,null_page=0)
            if p.refused:p=stream_sparse_plan(p)
            self.assertFalse(p.refused)
            profiles.append((len(values),tuple((b.descriptors+255)//256 for b in p.buckets)))
        self.assertEqual(profiles,[(63840,(12,)),(63848,(13,))])

class Publication(unittest.TestCase):
    def make_driver(self):
        d=NativeSegmentDriver.__new__(NativeSegmentDriver)
        d._profile_failure=(NS(request_uid='request',txn_seq=97),'unsupported-native-schedule-profile')
        d._profile_failed_request=NS(request_id='request',client_index=0)
        d.tx=None;d.controller=NS(active=False);d._retained=None
        d._return_pending='cancel';d._last_native_closure={'complete':True}
        d.engine=NS(output_queue=queue.Queue());self.finishes=[];self.records=[]
        d.scheduler=NS(finish_requests=lambda *a:self.finishes.append(a))
        d._record=lambda *a,**kw:self.records.append((a,kw))
        return d

    def setUp(self):
        self.patch=patch.dict(sys.modules,{'vllm.v1.request':NS(RequestStatus=NS(FINISHED_ERROR='error')),'vllm.v1.engine':NS(EngineCoreOutput=lambda **kw:NS(**kw),EngineCoreOutputs=lambda **kw:NS(**kw),FinishReason=NS(ERROR='error'))})
        self.patch.start();self.addCleanup(self.patch.stop)

    def test_error_published_while_full_return_waits_and_exactly_once(self):
        d=self.make_driver();self.assertTrue(d._publish_profile_failure())
        client,packet=d.engine.output_queue.get_nowait()
        self.assertEqual(client,0);self.assertEqual(packet.outputs[0].finish_reason,'error')
        self.assertEqual(packet.outputs[0].new_token_ids,[]);self.assertEqual(d._return_pending,'cancel')
        self.assertFalse(d._publish_profile_failure());self.assertEqual(len(self.finishes),1)

    def test_no_early_error_before_closed_native_ownership(self):
        d=self.make_driver();d.tx=object();self.assertFalse(d._publish_profile_failure())
        d.tx=None;d._last_native_closure={'complete':False}
        with self.assertRaises(DriverError):d._publish_profile_failure()
        self.assertTrue(d.engine.output_queue.empty());self.assertEqual(self.finishes,[])

    def test_full_output_queue_preserves_packet_and_retries_without_finishing_twice(self):
        d=self.make_driver();d.engine.output_queue=queue.Queue(maxsize=1);d.engine.output_queue.put('busy')
        self.assertFalse(d._publish_profile_failure());self.assertIsNotNone(d._profile_failure)
        self.assertEqual(d.engine.output_queue.get_nowait(),'busy');self.assertTrue(d._publish_profile_failure())
        self.assertEqual(len(self.finishes),1)

class ConnectedPublication(unittest.TestCase):
    def test_actual_all_rank_profile_abort_publishes_before_full_ack(self):
        import test_e3_native_schedule_profile as fixtures
        fixtures.ScheduleProfile.setUpClass()
        h=fixtures.ScheduleProfile();h.selection_counts=(4097,97153);h.setUp()
        self.addCleanup(h.doCleanups)
        n=fixtures.ProfileNetwork(h);d=n.driver
        d.engine.output_queue=queue.Queue()
        d.controller.request().client_index=0
        n.scheduler.finish_requests=lambda *args:None
        published=False
        with patch.dict(sys.modules,{'vllm.v1.request':NS(RequestStatus=NS(FINISHED_ERROR='error')),'vllm.v1.engine':NS(EngineCoreOutput=lambda **kw:NS(**kw),EngineCoreOutputs=lambda **kw:NS(**kw),FinishReason=NS(ERROR='error'))}):
            for tick in range(100):
                n.complete_everything();d.advance(deliver_outputs=False)
                if not d.engine.output_queue.empty():
                    self.assertTrue(d._last_native_closure['complete'])
                    self.assertIsNone(d.tx);self.assertFalse(d.controller.active)
                    self.assertIsNotNone(d._return_pending)
                    published=True;break
        self.assertTrue(published)
        self.assertEqual(d.engine.output_queue.get_nowait()[1].outputs[0].stop_reason,'unsupported-native-schedule-profile')
        self.assertIsNone(d._profile_failure)

if __name__=='__main__':unittest.main()
