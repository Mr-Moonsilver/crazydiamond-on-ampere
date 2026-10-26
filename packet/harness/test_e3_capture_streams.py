"""CPU ownership/source/wiring tests. No CUDA readiness claim."""
import contextlib
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace as N
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'patches/s5_engine'))
import capture_streams as cs
from green_context import GreenSplit,GreenContextError
from test_e3_green_native import FakeBinding,FakeStream,fake_torch,fake_runner,install_fake_vllm,remove_fake_vllm

class Binding(FakeBinding):
    def __init__(self):
        super().__init__();self.tokens=set();self.destroyed=[];self.wrong=False
    def aux_create(self,h):
        n=1000+len(self.tokens);self.tokens.add(n);return n
    def aux_destroy(self,h,n):
        if n not in self.tokens:return 1
        self.tokens.remove(n);self.destroyed.append(n);return 0
    def decode_stream_check(self,h,n):return int(self.wrong)

class Owners(unittest.TestCase):
    def setUp(self):
        self.torch=fake_torch();self.binding=Binding();self.split=GreenSplit(self.binding,0,44,torch_module=self.torch).create()
        old=FakeStream(90);self.a=N(_stream=old);self.b=N(_stream=old);self.p=N(_prefetch_stream=FakeStream(91))
        self.owner=cs.CaptureStreams(self.split,[(self.a,'_stream','shared'),(self.b,'_stream','shared'),(self.p,'_prefetch_stream','ple')])
    def test_sharing_restoration_and_graph_lease(self):
        old=self.a._stream;ple=self.p._prefetch_stream;self.owner.prepare()
        self.assertEqual(len(self.owner.streams),2)
        with self.owner.capture():
            self.assertIs(self.a._stream,self.b._stream);self.assertIsNot(self.a._stream,old)
            self.assertIsNot(self.a._stream,self.p._prefetch_stream)
            with self.assertRaises(cs.CaptureStreamError):
                with self.owner.capture():pass
        self.assertIs(self.a._stream,old);self.assertIs(self.p._prefetch_stream,ple)
        graph=object();self.owner.attach_graph(graph)
        with self.assertRaises(cs.CaptureStreamError):self.owner.close_after_graphs()
        with self.assertRaises(GreenContextError):self.split.close()
        with self.assertRaises(GreenContextError):self.split.destroy_decode_aux(self.owner.streams[0])
        self.owner.retire_graph(graph);self.owner.close_after_graphs();self.split.close()
        self.assertEqual(self.binding.destroyed,[])
        self.assertTrue(self.split.native_retained)
    def test_late_tensor_free_after_logical_bank_shutdown(self):
        # Fake allocator retains a stream use until tensor destruction, exactly
        # the obligation that graph-only leases failed to cover in job7.
        self.owner.prepare()
        with self.owner.capture():
            used = self.a._stream.cuda_stream
        graph=object();self.owner.attach_graph(graph)
        self.owner.retire_graph(graph);self.owner.close_after_graphs()
        self.split.close()
        self.assertIn(used,self.binding.tokens)  # event insertion at late free
        self.assertFalse(self.binding.destroyed)
        self.assertTrue(self.split.native_retained)
        from green_context import _PROCESS_RETAINED_SPLITS
        self.assertIn(self.split,_PROCESS_RETAINED_SPLITS)
        with self.assertRaises(GreenContextError):self.split.create_decode_aux()
        with self.assertRaises(GreenContextError):self.split.close()

    def test_exception_restores_but_retains_resources_until_cleanup(self):
        old=self.a._stream;self.owner.prepare()
        with self.assertRaisesRegex(RuntimeError,'capture'):
            with self.owner.capture():raise RuntimeError('capture')
        self.assertIs(self.a._stream,old);self.assertEqual(self.owner.state,'failed')
        with self.assertRaises(GreenContextError):self.split.close()
        with self.assertRaises(cs.CaptureStreamError):
            with self.owner.capture():pass
        self.owner.close_after_graphs();self.split.close()
    def test_wrong_context_refused_before_owner_mutation(self):
        self.owner.prepare();old=self.a._stream;self.binding.wrong=True
        with self.assertRaises(GreenContextError):
            with self.owner.capture():pass
        self.assertIs(self.a._stream,old);self.assertFalse(self.owner.retained)
        self.binding.wrong=False;self.owner.close_after_graphs();self.split.close()
    def test_owner_changed_between_prepare_and_capture(self):
        self.owner.prepare();self.a._stream=FakeStream(99)
        with self.assertRaises(cs.CaptureStreamError):
            with self.owner.capture():pass
        self.owner.close_after_graphs();self.split.close()
    def test_duplicate_unsupported_and_null_owner_refused(self):
        for owners in ([(self.a,'_stream','shared')]*2,[(self.a,'_stream','ple')],[(N(_stream=None),'_stream','shared')]):
            with self.assertRaises(cs.CaptureStreamError):cs.CaptureStreams(self.split,owners)
        self.split.close()
    def test_pinned_real_fork_join_methods_use_replaced_owners(self):
        import ast
        from types import MethodType
        events=[];current=[FakeStream(700)]
        @contextlib.contextmanager
        def stream_scope(stream):
            old=current[0];current[0]=stream
            try:yield
            finally:current[0]=old
        class Event:
            def record(self,stream):events.append(('record',stream.cuda_stream))
            def wait(self,stream):events.append(('wait',stream.cuda_stream))
        env={'torch':N(Tensor=object,cuda=N(stream=stream_scope,current_stream=lambda:current[0])),
             'current_stream':lambda:current[0], 'SharedExpertsOrder':N(MULTI_STREAM_OVERLAPPED=3)}
        def method(path,cls,name):
            tree=ast.parse((ROOT/'reference/vllm-src'/path).read_text())
            c=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
            f=next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name==name)
            f.decorator_list=[]
            module=ast.Module(body=[f],type_ignores=[])
            exec(compile(module,path,'exec'),env)
            return env[name]
        shared_path=next(iter(cs.PINS))
        self.a._determine_shared_experts_order=lambda x:3;self.a._output_idx=0
        self.a._output=[None,None];self.a._input_ready_event=[Event()];self.a._output_ready_event=[Event()]
        self.a._layer=lambda x:events.append(('layer',current[0].cuda_stream)) or x
        self.a.maybe_forward_async=MethodType(method(shared_path,'SharedExperts','maybe_forward_async'),self.a)
        self.a.wait=MethodType(method(shared_path,'SharedExperts','wait'),self.a)
        ple_path=list(cs.PINS)[1]
        self.p.etp_data_parallel_size=1;self.p._prefetch_buffer=[object()]
        self.p._get_dp_gather_slot=MethodType(method(ple_path,'Qwen4ExpPLEEmbedding','_get_dp_gather_slot'),self.p)
        self.p._gather_dp_ids=MethodType(method(ple_path,'Qwen4ExpPLEEmbedding','_gather_dp_ids'),self.p)
        self.p._lookup=lambda ids,output:events.append(('lookup',current[0].cuda_stream))
        self.p.start_prefetch=MethodType(method(ple_path,'Qwen4ExpPLEPinnedHostEmbedding','start_prefetch'),self.p)
        ids=N(shape=(1,1),record_stream=lambda stream:events.append(('ids',stream.cuda_stream)))
        # The fake stream implements only the real method's wait primitive.
        with patch.object(FakeStream,'wait_stream',lambda stream,other:events.append(('stream_wait',stream.cuda_stream)),create=True):
            self.a.maybe_forward_async(ids);self.a.wait();self.a._output[0]=None
            self.p.start_prefetch(None,ids)
            self.assertIn(('layer',90),events);self.assertIn(('lookup',91),events)
            events.clear();self.owner.prepare()
            with self.owner.capture():
                self.a.maybe_forward_async(ids);self.a.wait();self.a._output[0]=None
                self.p.start_prefetch(None,ids)
                self.assertIn(('layer',self.a._stream.cuda_stream),events)
                self.assertIn(('lookup',self.p._prefetch_stream.cuda_stream),events)
                self.assertNotIn(('layer',90),events);self.assertNotIn(('lookup',91),events)
        self.owner.close_after_graphs();self.split.close()

    def test_source_pins_match_reference_and_generator_closure(self):
        import make_s5_overlap as maker
        for path,h in cs.PINS.items():
            self.assertEqual(hashlib.sha256((ROOT/'reference/vllm-src'/path).read_bytes()).hexdigest(),h)
            self.assertEqual(maker.REFERENCE_INPUTS[path],h)
        self.assertIn('capture_streams',maker.E3_MODULES)
        self.split.close()
    def test_real_bank_capture_restores_and_drains_side_owners(self):
        import split_bank
        markers=[];install_fake_vllm(markers)
        try:
            runner=fake_runner(self.torch)
            bank=split_bank.SplitDecodeBank(runner,self.split,torch_module=self.torch)
            self.owner.prepare()
            with patch('capture_streams.for_runner',return_value=self.owner):bank.capture()
            controller=split_bank.RouteController(runner,bank,torch_module=self.torch)
            for stream in self.owner.consumer_streams():self.assertIn(stream,controller.consumer_streams())
            runner._flash_s5_split_bank=bank;runner._flash_s5_split=self.split
            split_bank.shutdown(runner)
            self.assertEqual(self.owner.state,'closed');self.assertEqual(self.split.state,'closed')
        finally:remove_fake_vllm()
    def test_discovery_exact_types_and_local_layer_counts(self):
        class Shared:pass
        class Ple:pass
        Model=type('Qwen4ExpModel',(),{'__module__':'vllm.models.qwen4_exp.nvidia.model'})
        model=Model();model.start_layer=0;model.end_layer=2;model.config=N(ple_layer_ids=[2])
        a,b=Shared(),Shared();a._stream=b._stream=FakeStream(42);p=Ple();p._prefetch_stream=FakeStream(43)
        model.modules=lambda:[model,a,b,p]
        with patch('capture_streams.pinned_types',return_value=(Shared,Ple)):
            owner=cs.for_runner(N(model=model),self.split);self.assertEqual(len(owner.streams),2);owner.close_after_graphs()
            model.end_layer=3
            with self.assertRaises(cs.CaptureStreamError):cs.for_runner(N(model=model),self.split)
        self.split.close()

if __name__=='__main__':unittest.main()
