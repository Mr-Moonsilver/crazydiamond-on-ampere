"""Real private/driver/workspace paths; CPU model/CUDA leaves are explicit."""
import ast
import contextlib
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'patches/s5_engine'), str(ROOT/'harness')]
import lane_private as private
import native_resources
from lane_state import prefill_scratch_specs
from test_e3_lane_private import PrivateBindingsTests, Tensor, FakeStream
from startup_vram_budget import startup_phase, wrap_headroom_capture, VramBudgetRefused

REMOVED = {'prefill_hc_hidden', 'prefill_hc_block_output', 'prefill_hc_injection',
           'prefill_qsa_accum', 'prefill_pp_handoff'}


class ContractTests(unittest.TestCase):
    def test_poison_removed_owners_never_allocated(self):
        calls = []
        class Device:
            def allocate(self, size, name):
                if name in REMOVED:
                    raise AssertionError('poison owner '+name)
                calls.append(name)
                return types.SimpleNamespace(address=4096+len(calls)*4096, name=name, nbytes=size)
        bindings, inventory, buffers = native_resources._allocate_scratch(Device(), 2048, compact=True)
        full = prefill_scratch_specs(2048)
        self.assertEqual(sum(s.nbytes for s in full if s.name in REMOVED), 59875328)
        self.assertEqual(set(calls), {s.name for s in full}-REMOVED)
        self.assertIn('prefill_qsa_topk', calls)
        self.assertEqual(len(bindings), len(buffers))
        self.assertEqual(sum(row[1] for row in inventory), sum(s.nbytes for s in full)-59875328)

    def test_package_qualifies_optional_startup_helper(self):
        import make_s5_overlap
        for name in ('native_startup','lane_private','split_bank'):
            text=make_s5_overlap.package((ROOT/'patches/s5_engine'/(name+'.py')).read_text())
            imports=[n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.ImportFrom) and n.module=='startup_vram_budget']
            self.assertTrue(imports,name)
            self.assertTrue(all(n.level==1 for n in imports),name)

    def test_flag_off_retains_identical_contract(self):
        self.assertFalse(private.headroom_enabled({}))
        self.assertEqual(prefill_scratch_specs(2048), prefill_scratch_specs(2048, compact=False))
        with self.assertRaises(private.LanePrivateError):private.headroom_enabled({'FLASH_S5_G4_HEADROOM':'yes'})


class ScorePathTests(unittest.TestCase):
    discover = PrivateBindingsTests.discover
    make_chunk = PrivateBindingsTests.make_chunk
    make_lane = PrivateBindingsTests.make_lane
    chunk_tokens, width, req_index = 8, 4, 2

    def setUp(self):
        PrivateBindingsTests.setUp(self)
        self.torch.float32 = 'float32'
        raw = Tensor((200*2**20,), 'uint8')
        self.ws.manager._current_workspaces[1] = raw
        self.bindings._workspace_buffer = raw
        self.bindings._workspace_bytes = 200*2**20
        self.ws.manager._locked = True
        self.bindings.headroom = True
        self.bindings.score_workspace = private.NativeScoreWorkspace(self.bindings)
        self.runner.worker_rank=0
        self.runner._flash_s5_native = types.SimpleNamespace(split=types.SimpleNamespace(prefill_stream=FakeStream(700)))

    @contextlib.contextmanager
    def scoped(self):
        with self.torch.cuda.stream(FakeStream(700)), self.bindings.chunk_scope(self.make_chunk()).scope(0):
            with private.native_score_scope() as owner:yield owner

    def test_real_lane_driver_uses_score_then_model_and_restores_decode(self):
        old = self.attn.indexer.__class__.__call__
        calls = []
        def indexer(indexer, *args):
            view = private.private_score_tensor(self.bindings.topk, 8, 64)
            calls.append(view.data_ptr())
            self.assertIs(view.untyped_storage(), self.bindings._workspace_buffer.untyped_storage())
            self.assertEqual(view.dtype, 'float32')
            return old(indexer, *args)
        with patch.object(type(self.attn.indexer), '__call__', indexer):
            lane, operation = self.make_lane(self.make_chunk(), bindings=self.bindings, pending=0)
            # Actual LaneForward/private scope/QsaLayerDriver, model and CUDA are fixture leaves.
            lane.begin()
            for _ in range(10):
                record=lane.layer(0)
                if record.kind=='layer':break
            else:self.fail('native layer did not finish')
            lane.finish()
        self.assertTrue(calls)
        self.assertIs(self.attn.topk_indices_buffer, self.original)
        self.assertIsNone(private._SCORE_OWNER.get())
        self.assertIsNone(private._SCORE_LEASE.get())
        self.assertIsNone(private.private_score_tensor(self.bindings.topk,8,64))

    def test_router_built_commands_reach_both_worker_private_paths(self):
        from engine_driver import NativeSegmentDriver
        from wire_v1 import SegmentKind, OpSeqTracker
        from lane_dispatch import SegmentCommand
        import test_e3_lane_dispatch as worker
        peer=ScorePathTests('runTest');peer.setUp()
        self.torch.inference_mode=lambda:contextlib.nullcontext()
        peer.torch.inference_mode=lambda:contextlib.nullcontext()
        contexts=(self,peer);workers=[];calls=[]
        original=self.attn.indexer.__class__.__call__
        def indexer(indexer,*args):
            context=next(c for c in contexts if c.attn.indexer is indexer)
            view=private.private_score_tensor(context.bindings.topk,8,64)
            calls.append((id(context),view.data_ptr()))
            return original(indexer,*args)
        for context in contexts:
            d,_,_,_,admission,operation=worker.make(True,context.torch)
            chunk=context.make_chunk()
            lane,_=context.make_lane(chunk,bindings=context.bindings,pending=0)
            d.factory.admit=lambda _,chunk=chunk:chunk
            d.native.split.prefill_stream=FakeStream(700)
            d.native.lane_forward=lambda *_,lane=lane,**__:lane
            d.driver_factory=lambda *_,lane=lane:lane.attention_driver
            d.open(worker.TXN,admission,operation)
            d.submit(worker.cmd('prepare',1));d.submit(worker.cmd('prefix',2));workers.append(d)
        driver=types.SimpleNamespace(tx=worker.TXN,stage_ranks={0:(0,1)},tickets={},
            trackers={r:OpSeqTracker(start=2) for r in (0,1)},_profile_grants={},
            _tail_boundary=None,_tail_submission=None,_tx_fused=False,_tx_group=1)
        acks=[]
        def deliver(command,targets):
            acks[:]=[workers[r].submit(SegmentCommand.from_descriptor(command)) for r in targets]
            for r in targets:driver.trackers[r].accepted(command.op_seq,command.descriptor)
        driver._send_segment=deliver
        with patch.object(type(self.attn.indexer),'__call__',indexer):
            for _ in range(10):
                NativeSegmentDriver._send_plan_entry(driver,dict(kind=SegmentKind.LAYER,stage=0,layer=0))
                if all(not ack.get('paused') for ack in acks):break
            else:self.fail('router-built native layer did not finish')
        self.assertEqual({row[0] for row in calls},{id(c) for c in contexts})
        self.assertTrue(all(d.quarantined is None for d in workers))
        self.assertTrue(all(c.attn.topk_indices_buffer is c.original for c in contexts))

    def test_fixed_view_reused_at_real_histories_and_quanta(self):
        addresses = set()
        with self.scoped() as owner:
            for history in (97152, 133679, 201261):
                width = ((history+3)//4+63)//64*64
                for quantum in (1024,1352,2024,2048):
                    rows = min(quantum, private.SCORE_BYTES//(width*4))
                    view = private.private_score_tensor(self.bindings.topk,rows,width)
                    self.assertEqual(view.shape, (rows,width))
                    self.assertTrue(view.is_contiguous());addresses.add(view.data_ptr())
            self.assertEqual(owner.uses,12)
        self.assertEqual(len(addresses),1)

    def test_wrong_workspace_lane_refuses(self):
        with self.scoped(), self.ws.use_workspace_lane(0):
            with self.assertRaisesRegex(private.LanePrivateError,'wrong workspace lane'):
                private.private_score_tensor(self.bindings.topk,8,64)

    def test_wrong_compute_stream_refuses(self):
        with self.scoped(), self.torch.cuda.stream(FakeStream(99)):
            with self.assertRaisesRegex(private.LanePrivateError,'wrong compute stream'):
                private.private_score_tensor(self.bindings.topk,8,64)

    def test_stale_generation_and_outside_lease_refuse(self):
        with self.scoped() as owner:generation=owner.generation
        with self.scoped() as current:
            with self.assertRaisesRegex(private.LanePrivateError,'stale'):
                current.view(self.bindings.topk,8,64,generation)
        with self.assertRaises(private.LanePrivateError):owner.view(self.bindings.topk,8,64,generation)

    def test_reordered_model_body_refuses_before_overwrite(self):
        with self.scoped():
            with self.assertRaisesRegex(private.LanePrivateError,'reordered'):
                private.score_model_boundary()

    def test_partial_indexer_error_retains_and_quarantines(self):
        raw = self.bindings._workspace_buffer
        with self.assertRaisesRegex(RuntimeError,'partial enqueue'):
            with self.scoped():raise RuntimeError('partial enqueue')
        self.assertTrue(self.bindings.score_workspace.failed)
        self.assertIs(self.bindings._workspace_buffer,raw)
        with self.assertRaises(private.LanePrivateError):
            with self.scoped():pass
        self.assertIsNone(private._SCORE_LEASE.get())

    def test_query_alias_refuses_before_score_write(self):
        with self.scoped():
            with self.assertRaisesRegex(private.LanePrivateError,'input aliases'):
                private.private_score_tensor(self.bindings._workspace_buffer,8,64)

    def test_nested_scores_and_oversize_refuse(self):
        with self.scoped():
            with self.assertRaisesRegex(private.LanePrivateError,'overlaps'):
                with private.native_score_scope():pass
            with self.assertRaisesRegex(private.LanePrivateError,'geometry'):
                private.private_score_tensor(self.bindings.topk,2048,50000)

    def test_full_tail_and_graph_call_have_no_native_score_lease(self):
        self.assertIsNone(private.private_score_tensor(self.bindings.topk,8,64))
        self.bindings.headroom=False
        with self.torch.cuda.stream(FakeStream(1)):
            with self.bindings.chunk_scope(self.make_chunk()).scope(0):
                with private.native_score_scope():
                    self.assertIsNone(private.private_score_tensor(self.bindings.topk,8,64))


class PhaseTests(unittest.TestCase):
    def setUp(self):
        self.rows=[];self.calls=[]
        self.runner=types.SimpleNamespace(device=0,vllm_config=types.SimpleNamespace(parallel_config=types.SimpleNamespace(rank=2)))
        self.torch=types.SimpleNamespace(cuda=types.SimpleNamespace(is_current_stream_capturing=lambda:False))
    def sample(self,torch,device):
        self.calls.append('sync/sample')
        return dict(device_index=device,total_bytes=1000,free_bytes=600-len(self.calls),allocated_bytes=100,reserved_bytes=200,observed_ns=len(self.calls))
    def test_phases_are_outside_capture_and_preserve_original_order(self):
        ca=types.SimpleNamespace(disabled=False,register_graph_buffers=lambda:self.calls.append('register'))
        tp=types.SimpleNamespace(device_communicator=types.SimpleNamespace(ca_comm=ca))
        def capture():self.calls.append('capture');ca.register_graph_buffers();return 7
        self.runner.cudagraph_manager=types.SimpleNamespace(capture=capture)
        with patch('startup_vram_budget.sample',self.sample),patch('startup_vram_budget.emit',self.rows.append):
            # Default writer is bound in the function body, not as a stale default argument.
            wrap_headroom_capture(self.runner,tp,self.torch)
            self.assertEqual(self.runner.cudagraph_manager.capture(),7)
        self.assertEqual(self.calls,['sync/sample','capture','sync/sample','register','sync/sample','sync/sample'])
        self.assertEqual([(r['label'],r['phase']) for r in self.rows],
            [('full-capture','before'),('custom-ar-registration','before'),('custom-ar-registration','after'),('full-capture','after')])
        self.assertTrue(all(r['rank']==2 for r in self.rows))
    def test_real_sampler_synchronizes_before_each_snapshot(self):
        self.torch.accelerator=types.SimpleNamespace(
            synchronize=lambda:self.calls.append('sync'),
            get_memory_info=lambda:(self.calls.append('memory') or (600,1000)))
        self.torch.cuda.memory_allocated=lambda device:100
        self.torch.cuda.memory_reserved=lambda device:200
        with startup_phase(self.runner,'real-sample',torch_module=self.torch,enabled=True,writer=self.rows.append):
            self.calls.append('work')
        self.assertEqual(self.calls,['sync','memory','work','sync','memory'])
        self.assertEqual(self.rows[1]['delta']['outside_cache_bytes'],0)

    def test_off_has_no_sampling_and_capture_refuses(self):
        with patch('startup_vram_budget.sample',side_effect=AssertionError('off sampled')):
            with startup_phase(self.runner,'test',enabled=False):pass
        self.torch.cuda.is_current_stream_capturing=lambda:True
        with self.assertRaisesRegex(VramBudgetRefused,'inside CUDA capture'):
            with startup_phase(self.runner,'test',enabled=True,torch_module=self.torch):pass



class ScoreBytesTests(unittest.TestCase):
    def test_generated_score_function_same_kernel_args_strides_and_bytes(self):
        import torch
        import make_s5_overlap
        from startup_vram_budget import facts
        source=(ROOT/'reference/vllm-src'/make_s5_overlap.INDEXER_PATH).read_text()
        generated=make_s5_overlap.compose_indexer(source)
        raw=torch.empty(200*2**20,dtype=torch.uint8)
        stream=types.SimpleNamespace(cuda_stream=77)
        cuda=types.SimpleNamespace(current_stream=lambda *_:stream,is_current_stream_capturing=lambda:False)
        manager=types.SimpleNamespace(_locked=True,_ensure_workspace_size=lambda _:raw)
        b=types.SimpleNamespace(_workspace_buffer=raw,_workspace_bytes=raw.numel(),_active=True,_failed=False,
            _require_workspace=lambda:None,_workspace_manager=manager,
            torch=types.SimpleNamespace(uint8=torch.uint8,float32=torch.float32,cuda=cuda),
            workspace_lane=1,runner=types.SimpleNamespace(worker_rank=0,device='cpu',_flash_s5_native=types.SimpleNamespace(split=types.SimpleNamespace(prefill_stream=stream))))
        owner=private.NativeScoreWorkspace(b)
        kernel_calls=[]
        class Kernel:
            def __getitem__(self,grid):
                def launch(*args,**kwargs):
                    logits=args[5]
                    kernel_calls.append((grid,args[6:],kwargs,tuple(logits.shape),tuple(logits.stride())))
                    # Only the GPU kernel is a CPU leaf. Both generated callers use this identical leaf.
                    logits.copy_(torch.arange(logits.shape[1],dtype=torch.float32).expand(logits.shape))
                return launch
        namespace=dict(torch=torch,triton=types.SimpleNamespace(cdiv=lambda a,b:-(-a//b)),
            _qsa_mqa_paged_prefill_kernel=Kernel(),private_score_tensor=private.private_score_tensor)
        def method(text):
            node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='_prefill_logits')
            scope=dict(namespace)
            exec(compile('from __future__ import annotations\n'+ast.get_source_segment(text,node),'actual-generated-prefill','exec'),scope)
            return scope['_prefill_logits']
        old,new=method(source),method(generated)
        for history in (97152,133679,201261):
            width=((history+3)//4+63)//64*64
            for quantum in (1024,1352,2024,2048):
                rows=min(quantum,private.SCORE_BYTES//(width*4))
                q=torch.ones(rows,1,4);kv=torch.ones(1,64,4);table=torch.zeros(1,(width+63)//64,dtype=torch.int32)
                starts=torch.tensor([0,rows]);visible=torch.full((rows,),width)
                args=(q,kv,table,starts,visible,rows,width,0,rows)
                expected=old(*args)
                token=private._SCORE_OWNER.set(owner)
                try:
                    with private.native_score_scope():actual=new(*args)
                finally:private._SCORE_OWNER.reset(token)
                self.assertEqual(actual.data_ptr(),raw.data_ptr()+136*2**20)
                self.assertTrue(torch.equal(expected,actual))
                self.assertEqual(kernel_calls[-1],kernel_calls[-2])
        # Full tiled host path and top-k CPU port, using the same pinned-function fixture as the lifetime proof.
        import test_n3_indexer_lifetime as lifetime
        import make_s5_overlap
        parent=source  # Pinned stock host functions, same CPU kernel/top-k ports on both paths.
        for history in (97152,133679,201261):
            for quantum in (1024,1352):
                pages=(history+4047)//4048+1
                query=(torch.arange(quantum*4).reshape(quantum,1,4).remainder(67).float()/32-1).to(torch.float32)
                cache=(torch.arange(pages*1012*4).reshape(pages,1012,1,4).remainder(103).float()/64-1).to(torch.float32)
                table=torch.arange(pages,dtype=torch.int32)[None,:]
                starts=torch.tensor([0,quantum],dtype=torch.int32)
                visible=(torch.arange(history-quantum+1,history+1,dtype=torch.int32)+3)//4
                outputs=[]
                for text,borrow in ((parent,False),(generated,True)):
                    ports=lifetime.Ports(torch);call=lifetime.function(text,ports)
                    call.__globals__['private_score_tensor']=private.private_score_tensor
                    out=torch.empty((quantum,512),dtype=torch.int32)
                    context=private._SCORE_OWNER.set(owner) if borrow else None
                    try:
                        with (private.native_score_scope() if borrow else contextlib.nullcontext()):
                            with torch.inference_mode():call(query,cache,table,starts,visible,2048,4,quantum,out,history)
                    finally:
                        if borrow:private._SCORE_OWNER.reset(context)
                    outputs.append(out)
                self.assertTrue(torch.equal(*outputs))
        # FULL/decode allocator path remains the pinned function, without the private lease.
        full=new(*args)
        self.assertNotEqual(full.untyped_storage().data_ptr(),raw.data_ptr())

if __name__=='__main__':unittest.main()
