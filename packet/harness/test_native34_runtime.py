"""Real host protocol/stream owners with explicit CUDA event and launch leaves."""
import ast
import contextlib
import os
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace as NS,ModuleType
import unittest
from unittest.mock import patch
import weakref

PROTOCOL_RECEIPTS=[]

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'patches/s5_engine'))
import native34_worker as w
import native34_diag as d
from native34_control import Arm
from split_bank import SplitDecodeBank,RouteController
from capture_streams import CaptureStreams
from model_hooks import PrefillStreamOwners
from native_startup import WorkerNative
from green_context import GreenSplit
import test_e3_native_qualification as fixture
from test_native34_indexer import host_body,inputs,Tensor as MetadataTensor


class Stream:
    def __init__(self,token,trace):self.cuda_stream=token;self.trace=trace
    def synchronize(self):self.trace.append(('aux-sync',self.cuda_stream))


class Cuda:
    def __init__(self):
        self.trace=[];self.allowed=True;self.current=Stream(88,self.trace)
        self.memory=NS(_snapshot=lambda:dict(segments=[],device_traces=[[]]),
                       _record_memory_history=lambda **kw:self.trace.append(('history',kw)))
    def current_stream(self,*a):return self.current
    def get_device_properties(self,*a):return NS(multi_processor_count=82)
    def ExternalStream(self,token,device=None):return Stream(token,self.trace)
    def Event(self,**kwargs):
        cuda=self
        class Event:
            complete=False
            def record(self,stream):self.stream=stream;cuda.trace.append(('record',stream.cuda_stream))
            def query(self):return self.complete
            def synchronize(self):
                if not cuda.allowed:raise RuntimeError('device-leaf-unposted-peer')
                self.complete=True;cuda.trace.append(('event-sync',self.stream.cuda_stream))
        return Event()


def runner(cuda,rank=1):
    r=fixture.make_runner(rank)
    binding=NS(create=lambda *a:(object(),11,88,44,38,100,200),close=lambda *a:0,probe=lambda *a:None,
        prefill_aux_create=lambda *a:16,prefill_aux_destroy=lambda *a:0,prefill_stream_check=lambda *a:0,
        aux_create=lambda *a:15,aux_destroy=lambda *a:0,decode_stream_check=lambda *a:0)
    split=GreenSplit(binding,0,44,torch_module=NS(cuda=cuda)).create()
    r.main_stream=split.decode_stream;r.output_copy_stream=Stream(12,cuda.trace)
    r.pp_handler=NS(main_stream=Stream(13,cuda.trace),broadcast_stream=Stream(14,cuda.trace))
    bank=object.__new__(SplitDecodeBank)
    bank.capture_streams=CaptureStreams(split,[(NS(_stream=Stream(17,cuda.trace)),'_stream','shared')])
    bank.capture_streams.prepare()
    controller=RouteController(r,bank,torch_module=NS(cuda=cuda))
    controller.route='split';controller.route_generation=1
    r._controller=controller;r.native.controller=controller
    owner=PrefillStreamOwners(split,[(NS(_stream=Stream(18,cuda.trace)),'_stream')]).prepare()
    r.native.prefill_streams=owner
    r.native.split=split
    r.native.comms.pp=NS(stream=Stream(89,cuda.trace))
    r._flash_s5_bridge.dispatcher.operation=NS(device=NS(_copy_stream=Stream(90,cuda.trace)))
    return r


class Runtime(unittest.TestCase):
    def setUp(self):
        self.cuda=Cuda();self.rows=[];self.r=runner(self.cuda)
        self.patches=[patch.dict(sys.modules,torch=NS(cuda=self.cuda)),
                      patch.object(w,'emit',lambda s,event,**fields:self.rows.append(dict(event=event,**fields)))]
        for p in self.patches:p.start();self.addCleanup(p.stop)
        self.s=w.state(self.r,1);self.s.update(mode='ARMED',arm='I',epoch=3)
        self.addCleanup(lambda:setattr(w.TLS,'phase_context',None))

    def test_actual_provider_drains_every_owned_stream(self):
        proof=w.drain_all(self.r,self.s)
        self.assertTrue(proof['complete']);self.assertEqual(proof['decode_consumer_streams'],6)
        self.assertEqual(proof['owned_prefill_auxiliary_streams'],1)
        self.assertEqual({row['stream'] for row in proof['decode_stream_inventory']},{11,12,13,14,15,17})
        self.assertEqual({row['stream'] for row in proof['native_streams']},{88,89,90})
        self.assertEqual({row[1] for row in self.cuda.trace if row[0]=='event-sync'},{11,12,13,14,15,17,88,89,90})
        token=MetadataTensor((4,),4)
        w.TLS.phase_context=dict(runner=self.r,state=self.s,output_zero_id=id(token))
        for arm in ('O','S'):
            self.s.update(arm=arm,phases=[])
            before=len([row for row in self.cuda.trace if row[0]=='event-sync'])
            calls=[]
            def original(*args,**kwargs):calls.append((args,kwargs));return token
            for name in sorted(w.PHASES):
                self.assertIs(w.phase(original,(token,),dict(flag=7),name),token)
            self.assertEqual(calls,[((token,),dict(flag=7))]*8)
            waits=len([row for row in self.cuda.trace if row[0]=='event-sync'])-before
            self.assertEqual(waits,8 if arm=='S' else 0)
            self.assertEqual({row['name'] for row in self.s['phases']},w.PHASES)
            self.assertEqual({row['complete'] for row in self.s['phases']},{arm=='S'})
            self.assertTrue(self.s['providers'])

    def target(self):
        fn,calls=host_body();fn(*inputs(),compress_ratio=4,mrope_section=None,rope_pos_offset=8)
        bound=calls[0]['bound']
        jit=NS(fn=NS(__name__='_qsa_pre_indexer_kernel'),arg_names=list(bound))
        w.TLS.phase_context=dict(runner=self.r,state=self.s)
        return (jit,*bound.values()),dict(grid=(10,),warmup=False)

    def test_I_prior_failure_never_launches_target(self):
        a,k=self.target();self.cuda.allowed=False;called=[]
        with self.assertRaisesRegex(RuntimeError,'device-leaf-unposted-peer'):
            w.indexer_launch(lambda *a,**k:called.append(1),a,k)
        self.assertEqual(called,[])
        self.assertEqual(self.rows[-1]['event'],'indexer-prior-work-exception')
        self.assertFalse(self.rows[-1]['target_launched'])

    def test_I_target_failure_preserves_original_after_real_provider_drains(self):
        a,k=self.target();fault=ValueError('private-fixture-text');called=[]
        def original(*args,**kwargs):called.append(1);raise fault
        with self.assertRaises(ValueError) as caught:w.indexer_launch(original,a,k)
        self.assertIs(caught.exception,fault);self.assertEqual(called,[1])
        self.assertIn('indexer-prior-work-drain-complete',[r['event'] for r in self.rows])
        row=[r for r in self.rows if r['event']=='indexer-launch-inputs'][0]
        self.assertEqual(len(row['inputs']),15);self.assertEqual(row['missing_arguments'],[])
        self.assertEqual(self.rows[-1]['event'],'indexer-kernel-exception')
        self.assertNotIn('private-fixture-text',str(self.rows))

    def test_O_target_call_is_unchanged_and_unfenced(self):
        self.s['arm']='O';a,k=self.target();token=object();calls=[]
        result=w.indexer_launch(lambda *a,**k:(calls.append((a,k)),token)[1],a,k)
        self.assertIs(result,token);self.assertEqual(calls,[(a,k)])
        self.assertFalse(any(r[0]=='event-sync' for r in self.cuda.trace))
        self.assertNotIn('indexer-prior-work-drain-begin',[r['event'] for r in self.rows])

    def test_control_a_requires_two_actual_descriptor_cohorts_and_completion(self):
        key=dict(bank='decode-split',num_tokens=4,num_reqs=4,mode='FULL')
        for i,cohort in enumerate(sorted(w.COHORTS)):
            event=self.cuda.Event();event.record(self.cuda.current)
            self.s['graph_replays'].append(dict(event=event,key=key,cohort=cohort,graph_id=str(i),post_prefix=False))
        self.assertEqual(w.cohort_proof(self.s,post_prefix=False)['verdict'],'INCONCLUSIVE')
        for row in self.s['graph_replays']:row['event'].synchronize()
        self.assertEqual(w.cohort_proof(self.s,post_prefix=False)['verdict'],'PASS')
        self.s['graph_replays'][1]['key']=dict(key,num_reqs=8)
        self.assertEqual(w.cohort_proof(self.s,post_prefix=False)['verdict'],'INCONCLUSIVE')

    def test_real_qualification_method_gets_original_exact_schema(self):
        self.s.update(mode='OFF',arm=None,epoch=None)
        original=fixture.qualification.handle_worker
        # Configuration needs FULL and an empty worker, just as on the engine.
        self.r.controller.route='full'
        original(self.r,fixture.configure_command())
        self.r.controller.route='split'
        command=fixture.evidence_command(native34=dict(action='A_START'))
        with self.assertRaises(fixture.qualification.NativeQualificationUnavailable):original(self.r,command)
        result=w.qualification(original,(self.r,command),{})
        self.assertEqual(result['ack'],'flash_s5_teacher_evidence')
        self.assertEqual(result['native34']['verdict'],'STARTED')
        self.assertTrue(self.s['history_available'])

    def test_patch_preserves_instance_static_and_class_descriptors(self):
        calls=[]
        class Owner:
            def method(self,x):return self,x
            @staticmethod
            def static(x):return x
            @classmethod
            def factory(cls,x):return cls,x
        def callback(old,a,k):calls.append(a);return old(*a,**k)
        for name in ('method','static','factory'):w.patch(Owner,name,callback)
        owner=Owner();token=object()
        self.assertEqual(owner.method(token),(owner,token));self.assertIs(owner.static(token),token)
        self.assertEqual(owner.factory(token),(Owner,token));self.assertEqual(len(calls),3)

    def test_arm_installs_real_host_hooks_after_slot_and_role_validation(self):
        import model_hooks,lane_comm,qsa_driver,qsa_operation
        class Tensor:
            def zero_(self):return self
        class Indexer:
            def forward(self,*a,**k):return a
        class Impl:
            @staticmethod
            def do_kv_cache_update(*a):return a
        class Attention:
            def _flash_e3_attention_pre(self,*a):return a
        class Layer:
            def _flash_e3_layer_pre(self,*a):return a
        layer=Layer();layer.self_attn=Attention();layer.self_attn.indexer=Indexer();layer.self_attn.impl=Impl()
        backbone=NS(layers=[None,None,None,layer],start_layer=0,end_layer=4)
        self.r.model=NS(language_model=NS(model=backbone))
        self.r.req_states.max_num_reqs=10;self.r.req_states.req_id_to_index={str(i):i for i in range(8)}
        self.s.update(mode='A_DONE',control_a=dict(verdict='PASS'))
        root=ModuleType('vllm');root.__path__=[]
        v1=ModuleType('vllm.v1');v1.__path__=[]
        package=ModuleType('vllm.v1.flash_e3');package.__path__=[]
        modules=dict(model_hooks=model_hooks,lane_comm=lane_comm,qsa_driver=qsa_driver,qsa_operation=qsa_operation)
        aliases={'vllm':root,'vllm.v1':v1,'vllm.v1.flash_e3':package}
        for name,module in modules.items():setattr(package,name,module);aliases['vllm.v1.flash_e3.'+name]=module
        originals=[(qsa_driver.QsaLayerDriver,'_pre'),(lane_comm,'canonical_selection'),
            (qsa_operation.QsaPrefillOperation,'producer_submitted'),(qsa_operation.QsaPrefillOperation,'attach_continuation')]
        saved=[(owner,name,inspect.getattr_static(owner,name)) for owner,name in originals]
        sys.modules['torch'].Tensor=Tensor
        try:
            with patch.dict(sys.modules,aliases),patch.dict(os.environ,FLASH_NATIVE34_ARM='I'):
                command=fixture.evidence_command(native34=dict(action='ARM',arm='I'))
                result=w.qualification(fixture.qualification.handle_worker,(self.r,command),{})
                self.assertTrue(result['native34']['slot_contract']['matches'])
                self.assertTrue(getattr(type(layer.self_attn.indexer).forward,'_native34_wrapped',False))
                self.assertTrue(getattr(qsa_driver.QsaLayerDriver._pre,'_native34_wrapped',False))
                token=object();self.assertEqual(layer.self_attn.impl.do_kv_cache_update(token),(token,))
                self.assertEqual(layer.self_attn.indexer.forward(token),(token,))
        finally:
            for owner,name,value in saved:setattr(owner,name,value)

    def test_second_post_prefix_tick_is_allowed_before_checkpoint(self):
        a=Arm('I',3);a.stage='POST_DECODE'
        driver=NS(_native34_arm=a,stop_kind=None,control_pending=lambda:False,_engine_paused=lambda:False)
        core=NS(_flash_s5_driver=driver)
        calls=[]
        def original(value):calls.append(value);return None
        self.assertIsNone(d.before_stock(original,(core,),{}))
        self.assertIsNone(d.before_stock(original,(core,),{}))
        self.assertEqual(d.before_stock(original,(core,),{}),({},False))
        self.assertEqual(a.stock_ticks,2);self.assertEqual(len(calls),3)

    def test_unpublished_next_bucket_is_held_and_cleanup_remains_available(self):
        a=Arm('O',3);a.stage='POST_DECODE'
        driver=NS(_native34_arm=a,stop_kind=None,control_pending=lambda:False,_engine_paused=lambda:False)
        calls=[]
        original=lambda value:(calls.append(value),'unchanged')[1]
        self.assertIsNone(d.drive(original,(driver,),{}));self.assertEqual(calls,[])
        driver.stop_kind='abort'
        self.assertEqual(d.drive(original,(driver,),{}),'unchanged');self.assertEqual(calls,[driver])

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
            w.graph_capture_begin(lambda *a,**k:None,(graph,),{})
            try:yield
            finally:w.graph_capture_end(lambda *a,**k:None,(graph,),{})
        class Runner:pass
        runner=Runner();runner.input_buffers=NS(input_ids=MetadataTensor((4,),4))
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
        with patch.object(w,'RUNNER',weakref.ref(runner)),patch.dict(os.environ,FLASH_S5_OVERLAP='1',FLASH_S5_REPLAY_OBSERVER='1'):
            w.graph_capture_factory(scope['capture'],(owner,factory),{})
        self.assertEqual(len(calls),2);self.assertTrue(owner._graphs_captured)
        graph=owner.graphs[desc];record=w.GRAPHS[graph]
        self.assertTrue(graph.kept);self.assertTrue(graph.instantiated);self.assertEqual(len(owner._flash_s5_capture_stream_checks),1)
        self.assertEqual(record['key']['mode'],'FULL');self.assertEqual(record['key']['num_tokens'],4)
        self.assertEqual(record['key']['bank'],'decode-split')
        self.assertEqual(record['storage']['storages'][0]['data_ptr'],runner.input_buffers.input_ids.data_ptr())

    def test_real_driver_qualification_and_checkpoint_through_real_worker_entry(self):
        import pickle,time,queue
        import test_e3_native_overlap as connected
        import native_overlap_worker as nw
        import worker_hooks
        from reply_transport import ReplyTransport
        from executor_hooks import BoundedSender
        stack=connected.idle_stack(rows={str(i):i for i in range(8)})
        core,_=connected.core_for(stack,paused=True)
        driver=stack.driver;epoch=driver.engine_epoch
        import native_qualification as nq
        for rank,worker in stack.workers.items():
            r=runner(self.cuda,rank);worker.model_runner=r
            r.native.state.engine_epoch=epoch
            r.controller.route_generation=driver.route_generation
            r.controller.route='full'
            r.req_states.max_num_reqs=10
            r.req_states.prefill_len=NS(np=[8]*10)
            r.req_states.num_computed_prefill_tokens=[8]*10
            nw.install_census(r)
            w.state(r,rank).update(mode='OFF')
        class Reader:
            def __init__(self):self.queue=queue.Queue();self.reads=0
            def dequeue(self,timeout):self.reads+=1;return self.queue.get(timeout=timeout)
        readers={r:Reader() for r in range(4)}
        transport=ReplyTransport(readers,'ok',handoff_idle=True)
        calls=[];published=[]
        class Broadcast:
            current_idx=0;n_remote_reader=0
            buffer=NS(max_chunk_bytes=131072,n_reader=4,
                      get_metadata=lambda i:contextlib.nullcontext([0,0,0,0,0]))
            def enqueue(self,envelope):
                # One atomic publication seam: every peer receives the same
                # serialized command before any worker consumes it.
                published.append(pickle.loads(pickle.dumps(envelope)))
                packets={r:pickle.loads(pickle.dumps(envelope)) for r in range(4)}
                for rank,worker in stack.workers.items():
                    name,args,kwargs,_=packets[rank];calls.append((rank,name))
                    value=worker_hooks.native_overlap_rpc(worker,*args,**kwargs)
                    readers[rank].queue.put(('ok',value))
        sender=BoundedSender(Broadcast())
        driver.executor._flash_s5_transport=transport
        driver.executor._flash_s5_bounded_send=sender
        # Real frontend marker producer -> real driver's command builder,
        # submit/control state machine -> real bounded FIFO -> outer checkpoint
        # -> real worker handler -> inner qualification. No marker is fabricated.
        from e3_native34_causal import Native34PlacementRunner
        path=Path(__file__).parent/'native34_protocol_before.txt'
        scope=dict(vars(w));exec(compile(path.read_text(),str(path),'exec'),scope)
        old_checkpoint=scope['checkpoint'];captured=[]
        original_handle=nw.handle;original_qualification=nq.handle_worker
        driver_type=type(driver)
        saved=[(owner,name,inspect.getattr_static(owner,name)) for owner,name in
               ((driver_type,'advance'),(driver_type,'_apply_native_overlap'),(nw,'handle'),(nq,'handle_worker'))]
        class ClientDriver:
            def _worker(client,phase,action,payload):
                intent=dict(action='qualification',engine_epoch=epoch,
                            worker_action=action,payload=payload)
                command=driver._worker_command(dict(action='qualification',command=intent))
                captured.append(command)
                if command.get('native34',{}).get('action')=='A_START':
                    refused=[]
                    for rank,worker in stack.workers.items():
                        try:old_checkpoint(original_handle,(worker,command),{})
                        except ValueError as exc:
                            if str(exc)!='native34-checkpoint-marker-invalid':raise
                            refused.append(rank)
                    self.assertEqual(refused,list(range(4)))
                future=driver.submit_control(intent)
                deadline=time.monotonic()+3
                while not future.done() and time.monotonic()<deadline:
                    driver.advance();time.sleep(.001)
                self.assertTrue(future.done())
                return future.result()
        try:
            w.patch(nw,'handle',w.checkpoint)
            w.patch(nq,'handle_worker',w.qualification)
            w.patch(driver_type,'advance',d.advance)
            w.patch(driver_type,'_apply_native_overlap',d.checkpoint_reply)
            client=ClientDriver()
            client._worker('row4','teacher_configure',dict(enabled=True,snapshot_max_bytes=64*1024**2))
            for worker in stack.workers.values():worker.model_runner.controller.route='split'
            proofs=Native34PlacementRunner.marker(object.__new__(Native34PlacementRunner),client,'A_START',epoch)
            self.assertEqual(set(proofs),set(range(4)))
            self.assertTrue(all(row['verdict']=='STARTED' for row in proofs.values()))
            self.assertEqual(captured[-1]['action'],'teacher_evidence')
            self.assertEqual(captured[-1]['native34'],dict(action='A_START'))
            core.paused=False
            for rank,worker in stack.workers.items():
                r=worker.model_runner;r.req_states.req_id_to_index={str(i):i for i in range(8)}
                w.state(r,rank).update(mode='PREFIX',arm='S',epoch=epoch)
            arm=Arm('S',epoch);arm.stage='BEFORE_DRAIN';driver._native34_arm=arm
            with patch.object(d.localize,'emit',lambda *a,**k:None):
                driver.advance()
                deadline=time.monotonic()+3
                while arm.stage!='PREFIX' and time.monotonic()<deadline:
                    driver.advance();time.sleep(.001)
            self.assertEqual(arm.stage,'PREFIX');self.assertEqual(sender.sent,3)
            self.assertEqual(calls,[(r,'flash_s5_native_overlap') for _ in range(3) for r in range(4)])
            self.assertEqual([r.reads for r in readers.values()],[3]*4)
            self.assertFalse(driver._inflight);self.assertFalse(transport._calls)
            self.assertEqual(len(arm.records[0]['workers']),4)
            census=published[-1][1][0]
            self.assertEqual(census['action'],'census')
            self.assertEqual(census['native34'],dict(action='CHECKPOINT',arm='S',phase='before-prefix'))
            self.assertEqual(census['engine_epoch'],epoch)
            bad=pickle.loads(pickle.dumps(census));bad['native34']['phase']='invalid'
            with self.assertRaisesRegex(ValueError,'native34-checkpoint-marker-invalid'):
                worker_hooks.native_overlap_rpc(stack.workers[0],bad)
            PROTOCOL_RECEIPTS.append(dict(census_command=census,verdict='PASS',ranks=list(range(4)),
                real_driver_advance=True,real_worker_entry=True,hand_built_marker=False,
                qualification_marker=captured[-1]['native34'],
                census_phase=arm.records[0]['phase'],legacy_refusal_all_ranks=True,
                sender_calls=sender.sent,reply_reads=[r.reads for r in readers.values()],
                device_execution=False))
        finally:
            for owner,name,value in saved:setattr(owner,name,value)
            self.assertTrue(transport.close(1)['clean'])


if __name__=='__main__':unittest.main()
