"""CPU conformance for native33 diagnostic-only wrappers and captured pointers."""
import ast
import contextlib
import gc
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import weakref

import native33_diag as d


class Tensor:
    def __init__(self, ptr):self.ptr=ptr;self.shape=(4,);self.dtype='int32';self.device='cuda:1'
    def data_ptr(self):return self.ptr
    def item(self):raise AssertionError('device readback')
    def to(self,*a,**k):raise AssertionError('device allocation')


class Diagnostic(unittest.TestCase):
    def setUp(self):
        self.rows=[];self.syncs=[];self.real_emit=d.emit
        self.enterContext(patch.dict(os.environ,FLASH_NATIVE33_DIAG='1'))
        self.enterContext(patch.object(d,'emit',lambda event,**kw:self.rows.append(dict(event=event,**kw))))
        self.enterContext(patch.object(d,'fence',lambda:self.syncs.append('fence')))
        old=getattr(d.host._TLS,'context',None)
        d.host._TLS.context=dict(state=dict(rank=1,active=True),fields=dict(rank=1,label='LAYER',op_seq=5,step=8))
        self.addCleanup(setattr,d.host._TLS,'context',old)
        for name,value in [('capturing',None),('capture_key',None),('forward_roots',{}),('replay_key',None)]:
            before=getattr(d._TLS,name,None);setattr(d._TLS,name,value);self.addCleanup(setattr,d._TLS,name,before)

    def test_exact_result_and_flushed_order(self):
        result=object();calls=[]
        def original(x,*,n):calls.append((x,n));return result
        self.assertIs(d.launch(original,'test.kernel','triton-jit',(2,),dict(n=3)),result)
        self.assertEqual(calls,[(2,3)]);self.assertEqual(self.syncs,['fence'])
        self.assertEqual([r['event'] for r in self.rows],['launch-begin','launch-end'])
        self.assertEqual(self.rows[0]['kernel_name'],'test.kernel')

    def test_inactive_other_rank_capture_and_pp_never_fence(self):
        ctx=d.host._TLS.context
        for label,rank,active,capturing in [('PP_SEND',1,True,None),('PP_RECEIVE',1,True,None),
                                          ('LAYER',0,True,None),('DECODE_STEP',1,False,None),
                                          ('LAYER',1,True,object())]:
            ctx['fields']['label']=label;ctx['state'].update(rank=rank,active=active);d._TLS.capturing=capturing
            self.assertEqual(d.launch(lambda:7,'kernel','triton-jit',(),{}),7)
        ctx['fields']['label']='DECODE_STEP';ctx['state'].update(rank=1,active=True);d._TLS.capturing=None
        for name in ('c10d::send','c10d::recv'):
            self.assertEqual(d.launch(lambda:7,name,'torch-operator',(),{}),7)
        self.assertEqual(self.rows,[]);self.assertEqual(self.syncs,[])

    def test_fence_error_preserves_exception_and_records_named_launch(self):
        error=RuntimeError('PRIVATE_ERROR_TEXT')
        with patch.object(d,'fence',side_effect=error):
            with self.assertRaises(RuntimeError) as caught:d.launch(lambda:None,'kernel','triton-jit',(),{})
        self.assertIs(caught.exception,error)
        self.assertEqual(self.rows[-1]['event'],'launch-exception')
        self.assertEqual(self.rows[-1]['kernel_name'],'kernel')
        self.assertNotIn('PRIVATE_ERROR_TEXT',json.dumps(self.rows))
        self.assertTrue(self.rows[-1]['frames'])

    def test_nested_first_error_names_inner_launch(self):
        def bad():raise ValueError('fixture')
        def outer():return d.launch(bad,'inner.kernel','triton-jit',(),{})
        with self.assertRaises(ValueError):d.launch(outer,'outer.operator','torch-operator',(),{})
        errors=[r for r in self.rows if r['event']=='launch-exception']
        self.assertEqual([r['operator'] for r in errors],['inner.kernel','outer.operator'])

    def test_pointer_snapshot_does_not_retain_tensor_or_read_values(self):
        value=Tensor(123);ref=weakref.ref(value)
        snapshot=d.pointers(dict(input_buffers=NS(input_ids=value)))
        self.assertEqual(snapshot['values']['input_buffers.input_ids']['data_ptr'],'0x7b')
        del value;gc.collect();self.assertIsNone(ref())

    def test_pointer_change_and_missing_capture_are_reported(self):
        graph=NS # use weak-referenceable object, not SimpleNamespace
        class Graph:pass
        graph=Graph();runner=NS # same reason for runner weakref
        class Runner:pass
        runner=Runner();runner.input_buffers=NS(input_ids=Tensor(100));runner.block_tables=NS(num_blocks=NS(gpu=Tensor(200)))
        with patch.object(d,'_RUNNER',weakref.ref(runner)):
            snap=d.pointers(d.runner_roots())
            d._GRAPHS[graph]=dict(capture_id=1,key=dict(mode='FULL',num_tokens=4),snapshot=snap,replay_seen=False)
            runner.block_tables.num_blocks.gpu=Tensor(201)
            d.compare_graph(graph)
        changed=[r for r in self.rows if r['event']=='graph-pointer' and r['pointer_changed']]
        self.assertEqual(len(changed),1);self.assertEqual(changed[0]['captured']['data_ptr'],'0xc8')
        self.assertEqual(changed[0]['live']['data_ptr'],'0xc9')
        d.compare_graph(Graph());self.assertFalse(self.rows[-1]['capture_inventory_available'])

    def test_actual_pinned_host_capture_body_registers_graph_and_descriptor(self):
        root=Path(__file__).resolve().parent
        source=root/'native30_overlay/e3_sources/cudagraph_utils.py'
        if not source.exists():source=Path('<hub-dir>/agents/flash-next/tmp/native32-held-c/q18a-native32-bundle-b50aea5737fb/native30_overlay/e3_sources/cudagraph_utils.py')
        tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CudaGraphManager')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='capture');method.decorator_list=[]
        mode=NS(PIECEWISE='PIECEWISE',FULL='FULL',NONE='NONE')
        class Desc:
            cg_mode=NS(name='FULL');num_tokens=4;num_reqs=4;num_ubatches=1;max_query_len=1;num_active_loras=0;uniform_token_count=1
        desc=Desc();desc.cg_mode=mode.FULL
        class Graph:
            def __init__(self,keep_graph=False):self.kept=keep_graph;self.instantiated=False
            def instantiate(self):self.instantiated=True
        @contextlib.contextmanager
        def graph_context(graph,*a,**k):
            d.capture_begin(lambda *a,**k:None,(graph,),{})
            try:yield
            finally:d.capture_end(lambda *a,**k:None,(graph,),{})
        class Runner:pass
        runner=Runner();runner.input_buffers=NS(input_ids=Tensor(123))
        owner=NS(device='cpu-leaf',_capture_descs={'FULL':[desc]},_max_full_descs_to_capture=None,
                 use_breakable_cg=False,graphs={},pool='cpu-pool-leaf',_capture_mem_samples=None)
        stream=NS(cuda_stream=11)
        owner._flash_s5_capture_context=lambda:contextlib.nullcontext()
        owner._flash_s5_capture_owner=NS(state='open',decode_stream_token=11,decode_stream=stream)
        fake_torch=NS(cuda=NS(CUDAGraph=Graph,graph=graph_context,current_stream=lambda device:stream))
        sys.path.insert(0,str(root.parent/'patches/s5_engine'))
        from worker_hooks import capture_context,capture_graph,checked_capture_stream,finish_graph
        # Desc mode must expose .name as well as compare to the exact enum key.
        class Mode(str):
            @property
            def name(self):return str(self)
        mode.FULL=Mode('FULL');mode.PIECEWISE=Mode('PIECEWISE');mode.NONE=Mode('NONE');desc.cg_mode=mode.FULL
        scope=dict(torch=fake_torch,CUDAGraphMode=mode,CreateForwardFn=object,
                   graph_capture=lambda **k:contextlib.nullcontext(),_flash_s5_capture_context=capture_context,
                   _flash_s5_capture_graph=lambda manager:capture_graph(manager,torch_module=fake_torch),
                   _flash_s5_checked_capture_stream=lambda manager,stream,desc:checked_capture_stream(manager,stream,desc,torch_module=fake_torch),
                   _flash_s5_finish_graph=finish_graph,current_stream=lambda:stream,is_global_first_rank=lambda:False,
                   logger=NS(debug=lambda *a:None),get_offloader=lambda:NS(sync_prev_onload=lambda:None,join_after_forward=lambda:None),
                   set_graph_pool_id=lambda p:None,compilation_counter=NS(num_cudagraph_captured=0))
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),scope)
        calls=[]
        def factory(desc,warmup):
            model_inputs=dict(input_ids=runner.input_buffers.input_ids)
            def forward(mode):calls.append((mode,model_inputs))
            return forward
        with patch.object(d,'_RANK',1),patch.object(d,'_RUNNER',weakref.ref(runner)),patch.dict(os.environ,FLASH_S5_OVERLAP='1',FLASH_S5_REPLAY_OBSERVER='1'):
            d.capture_factory(scope['capture'],(owner,factory),{})
        self.assertEqual(len(calls),2);self.assertTrue(owner._graphs_captured)
        graph=owner.graphs[desc];record=d._GRAPHS[graph]
        self.assertTrue(graph.kept);self.assertTrue(graph.instantiated);self.assertEqual(len(owner._flash_s5_capture_stream_checks),1)
        self.assertEqual(record['key']['mode'],'FULL');self.assertEqual(record['key']['num_tokens'],4)
        self.assertEqual(record['snapshot']['values']['model_inputs.input_ids']['data_ptr'],'0x7b')

    def test_compile_only_triton_warmup_is_not_a_launch(self):
        fn=NS(fn=lambda:None)
        self.assertEqual(d.triton_launch(lambda *a,**k:9,(fn,),dict(warmup=True)),9)
        self.assertEqual(self.rows,[]);self.assertEqual(self.syncs,[])

    def test_real_emitter_budget_and_boundary_frame_path(self):
        class Runner:pass
        runner=Runner();state=d.host.state(runner,1);state['active']=True;emitted=[]
        with patch.object(d,'emit',self.real_emit),patch.object(d,'_RUNNER',weakref.ref(runner)), \
             patch.object(d,'_RECORDS',0),patch.object(d,'_BYTES',0),patch.object(d,'_LIMIT',False), \
             patch.object(d.localize,'emit',lambda event,**fields:emitted.append(dict(event=event,**fields))):
            try:raise ValueError('PRIVATE_ERROR_TEXT')
            except ValueError:d.exception_emit(lambda *a,**k:None,'exception',rank=1,label='DECODE_STEP')
            self.assertEqual(emitted[0]['rank'],1);self.assertEqual(emitted[0]['event'],'native33-boundary-exception')
            self.assertNotIn('PRIVATE_ERROR_TEXT',json.dumps(emitted))
            with patch.object(d,'MAX_LAUNCH_RECORDS',1):
                self.real_emit('launch-begin');self.real_emit('launch-begin')
            self.assertEqual(sum(r['event']=='native33-receipt-limit' for r in emitted),1)
            self.assertTrue(all(len(json.dumps(r))<4096 for r in emitted))

    def test_capture_exception_restores_state(self):
        d._TLS.capturing=object()
        with self.assertRaises(ValueError):d.capture_end(lambda:(_ for _ in ()).throw(ValueError('fixture')),(),{})
        self.assertIsNone(d._TLS.capturing)

    def test_opaque_graph_fault_never_invents_kernel_name(self):
        class Graph:pass
        graph=Graph()
        def bad(g):raise ValueError('fixture')
        with self.assertRaises(ValueError):d.graph_replay(bad,(graph,),{})
        row=[r for r in self.rows if r['event']=='launch-exception'][0]
        self.assertIsNone(row['kernel_name']);self.assertEqual(row['launch_kind'],'cuda-graph-replay')


class Queue(unittest.TestCase):
    def test_queue_changes_only_names_and_diagnostic_grade_and_switch(self):
        root=Path(__file__).parent/'runs'
        before=(root/'q18a_native32.draft.sh').read_text()
        after=(root/'q18a_native33.draft.sh').read_text().replace('q18a-native33','q18a-native32').replace('q18a_native33','q18a_native32')
        after='\n'.join(line for line in after.split('\n') if 'native33_grade.py' not in line and 'record native33-diagnostic' not in line)
        self.assertEqual(before,after)
        before=(root/'q18a_native32_settings.sh').read_text()
        after=(root/'q18a_native33_settings.sh').read_text().replace('q18a-native33','q18a-native32').replace('FLASH_NATIVE33_DIAG=1 ','')
        self.assertEqual(before,after)
        self.assertNotIn('CUDA_LAUNCH_BLOCKING=1',after)


class Grading(unittest.TestCase):
    def test_named_and_opaque_attributions_stay_distinct(self):
        import tempfile
        from native33_grade import grade
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'log'
            for kind,name,expected in [('triton-jit','kernel','NAMED_TRITON_KERNEL'),('cuda-graph-replay',None,'OPAQUE_GRAPH_REPLAY')]:
                path.write_text('NATIVE30 '+json.dumps(dict(event='native33-launch-exception',launch_kind=kind,kernel_name=name,monotonic_ns=1)))
                self.assertEqual(grade([path])['attribution'],expected)

    def test_missing_fault_or_receipt_limit_cannot_pass(self):
        import tempfile
        from native33_grade import grade
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'log';path.write_text('NATIVE30 '+json.dumps(dict(event='native33-receipt-limit')))
            r=grade([path]);self.assertEqual(r['verdict'],'INCONCLUSIVE');self.assertEqual(r['classification'],'HARNESS')


if __name__=='__main__':unittest.main()
