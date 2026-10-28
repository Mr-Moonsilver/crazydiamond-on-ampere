#!/usr/bin/env python3
"""Connected single-worker lifecycle over real mounted classes, not an engine.

The small real Qwen model and TP1/PP1 topology are explicit fixture leaves;
no four-rank certificate, production accuracy or performance is claimed.
"""
import argparse,dataclasses,hashlib,importlib,inspect,json,tempfile,time,traceback
from pathlib import Path
from types import SimpleNamespace


_PROGRESS=None


def exception_chain(exc):
    """Retain both Python links, every frame/message, and cycle-safe references."""
    pending=[exc];seen={};rows=[]
    while pending:
        value=pending.pop(0)
        if id(value) in seen:continue
        seen[id(value)]=len(rows)
        rows.append(dict(type=type(value).__name__,module=type(value).__module__,message=str(value),
            stack=[dict(file=f.filename,function=f.name,line=f.lineno) for f in traceback.extract_tb(value.__traceback__)],
            suppress_context=bool(value.__suppress_context__),
            _cause=value.__cause__,_context=value.__context__))
        for other in (value.__cause__,value.__context__):
            if other is not None and id(other) not in seen:pending.append(other)
    for row in rows:
        for field in ('cause','context'):
            value=row.pop('_'+field);row[field]=None if value is None else seen[id(value)]
    return rows


def checkpoint(name,**fields):
    if _PROGRESS is not None:
        row=dict(step=name,observed_ns=time.monotonic_ns(),**fields)
        _PROGRESS.write(json.dumps(row,default=str)+'\n');_PROGRESS.flush()
    print('LIFECYCLE '+name,flush=True)


def admission_shape(record):
    return dict(transaction=dataclasses.asdict(record.transaction),prompt_end=record.prompt_end,
        input_ids_count=len(record.input_ids),ngram_context_count=len(record.ngram_context),
        block_table_rows=[len(row) for row in record.block_tables],
        new_block_ids_to_zero=list(record.new_block_ids_to_zero),cow_pairs=list(record.cow_pairs),
        state_epoch=record.state_epoch)


def cpu_zeroer_diagnostic(cfg,admission):
    """Real constructor's partial CPU object, not __new__/a runner shim."""
    import torch
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.flash_e3.lane_chunk import LaneRequest,PinnedPorts,LaneChunkError
    try:GPUModelRunner(cfg,torch.device('cpu'))
    except Exception as exc:
        tb=exc.__traceback__;partial=None
        while tb is not None:
            candidate=tb.tb_frame.f_locals.get('self')
            if type(candidate) is GPUModelRunner:partial=candidate
            tb=tb.tb_next
        assert partial is not None and hasattr(partial,'kv_block_zeroer'),exception_chain(exc)
        constructor_boundary=exception_chain(exc)
    else:raise AssertionError('GPU runner fully constructed on CPU; extend host execution')
    assert partial.kv_block_zeroer is None
    assert admission.new_block_ids_to_zero,'zeroer diagnostic requires actual allocation zero obligations'
    request=LaneRequest(partial,PinnedPorts.load(),torch_module=torch)
    try:request._apply_allocated_pages(admission)
    except LaneChunkError as exc:
        assert str(exc)=='runner has no KV block zeroer for the new blocks',str(exc)
        guard=exception_chain(exc)
    else:raise AssertionError('expected uninitialized zeroer guard did not fire')
    return dict(verdict='PASS',actual_partial_runner_type=type(partial).__module__+'.'+type(partial).__name__,
        actual_constructor_entered=True,runner_constructed_fully=False,runner_fields_assigned_by_fixture=False,
        constructor_boundary=constructor_boundary,actual_lane_request_method=True,
        new_block_ids_to_zero_count=len(admission.new_block_ids_to_zero),kv_block_zeroer_is_none=True,
        guard=guard,methods_patched=False,box_evidence=False)


_WORKSPACE_BOOTSTRAP_OWNERS=[]


def provision_local_workspace(device):
    """Declared pre-model empty-manager geometry fixture; never replace views."""
    from vllm.v1.worker.workspace import current_workspace_manager,init_workspace_manager
    before=current_workspace_manager()
    assert before._num_lanes==1 and not before._locked
    assert all(value is None for value in before._current_workspaces), 'workspace already owns allocated views'
    _WORKSPACE_BOOTSTRAP_OWNERS.append(before)
    init_workspace_manager(device,num_lanes=2)
    after=current_workspace_manager()
    assert after._num_lanes==2 and all(value is None for value in after._current_workspaces)
    checkpoint('empty-workspace-provisioned',before_lanes=1,after_lanes=2,
        allocated_views_replaced=0,old_empty_owner_retained=True,fixture_scope=True)
    return dict(before_lanes=1,after_lanes=2,allocated_views_replaced=0,
        old_empty_owner_retained=True,box_evidence=False)


def config(device):
    from vllm.config import (ModelConfig,VllmConfig,CacheConfig,SchedulerConfig,
        ParallelConfig,DeviceConfig,CompilationConfig)
    from vllm.models.qwen4_exp.config import Qwen4ExpTextConfig,Qwen4ExpConfig
    path=Path(tempfile.mkdtemp(dir='/tmp',prefix='matrix-qwen-'))
    text=Qwen4ExpTextConfig(vocab_size=256,hidden_size=256,intermediate_size=512,
        num_hidden_layers=3,num_attention_heads=12,num_key_value_heads=1,head_dim=256,
        layer_types=['linear_attention','full_attention','linear_attention'],
        linear_num_key_heads=4,linear_num_value_heads=4,linear_key_head_dim=64,linear_value_head_dim=64,
        num_experts=0,shared_expert_intermediate_size=0,max_position_embeddings=131072,
        hc_count=4,hc_lowrank=16,ple_layer_ids=[1],ple_embed_dim=64,heads_per_ngram=4,
        ngram_vocab_size_base=1024,make_ngram_vocab_size_divisible_by=128,
        indexer_n_heads=4,indexer_kv_heads=1,indexer_head_dim=64,indexer_budget=2048,indexer_compress_ratio=4,
        partial_rotary_factor=.25,eos_token_id=255)
    Qwen4ExpConfig(text_config=text,architectures=['Qwen4ExpForConditionalGeneration']).save_pretrained(path)
    model=ModelConfig(model=str(path),skip_tokenizer_init=True,language_model_only=True,
        dtype='bfloat16',max_model_len=98320,enforce_eager=True)
    return VllmConfig(model_config=model,cache_config=CacheConfig(block_size=4048,mamba_block_size=4048,
        num_gpu_blocks_override=64,gpu_memory_utilization=0.08,enable_prefix_caching=True,prefix_match_unit=176,mamba_cache_mode='align'),
        scheduler_config=SchedulerConfig(max_model_len=98320,is_encoder_decoder=False,
            max_num_batched_tokens=1024,max_num_seqs=10,scheduler_reserve_full_isl=False),
        parallel_config=ParallelConfig(),device_config=DeviceConfig(device=device),
        compilation_config=CompilationConfig(mode=0))


def collect():
    """Execute every CPU-executable bootstrap leaf with no platform shims."""
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.distributed import init_distributed_environment,initialize_model_parallel
    from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpForConditionalGeneration
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.flash_e3 import (lane_chunk,lane_dispatch,worker_bridge,lane_hooks,
        native_resources,native_startup,native_backend,lane_state,lane_private,qsa_driver)
    from e3_native_transitions_matrix_probe import sampler_host_constructor_path
    import ast
    # Execute deferred import statements from all GPU helpers as well. The
    # first GPU function import must not hide a missing name until owner-run.
    namespace={}
    for fn in (worker_components,execute_chunk,gpu_scenarios,cache_configuration,controller_components):
        tree=ast.parse(inspect.getsource(fn))
        for node in ast.walk(tree):
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual-gpu-branch-import>','exec'),namespace)
    host=sampler_host_constructor_path(cpu_device_probe=True)
    cfg=config('cpu')
    classes=(GPUModelRunner,lane_chunk.LaneChunkFactory,lane_dispatch.LaneDispatcher,
        worker_bridge.NativeWorkerBridge,lane_hooks.LaneRuntime,
        native_resources.WorkerHandoffServer,native_startup.WorkerNative,
        native_backend.NativeDevice,native_backend.EventCompletionBackend,
        lane_state.WorkerState,lane_private.LanePrivateBindings,qsa_driver.QsaLayerDriver)
    index=[]
    for cls in classes:
        path=Path(inspect.getfile(cls))
        index.append(dict(name=cls.__module__+'.'+cls.__name__,signature=str(inspect.signature(cls)),
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    with set_current_vllm_config(cfg):
        init_distributed_environment(world_size=1,rank=0,local_rank=0,
            distributed_init_method='file:///tmp/matrix-cpu-dist',backend='gloo')
        initialize_model_parallel(1,1,backend='gloo')
        from vllm.v1.worker.workspace import init_workspace_manager,current_workspace_manager
        init_workspace_manager(torch.device('cpu'),num_lanes=1)
        workspace_provision=provision_local_workspace(torch.device('cpu'))
        workspace=current_workspace_manager().get_simultaneous(((128,),torch.uint8))
        assert workspace[0].shape==(128,) and workspace[0].device.type=='cpu'
        import sys
        observed=set()
        def host_profile(frame,event,arg):
            if event=='call':observed.add((frame.f_globals.get('__name__',''),frame.f_code.co_qualname))
            elif event=='c_call':observed.add(('builtin',getattr(arg,'__qualname__',getattr(arg,'__name__',''))))
        previous=sys.getprofile();sys.setprofile(host_profile)
        try:source_preflight=source_owner_preflight(cfg)
        finally:sys.setprofile(previous)
        sequence_audit=stock_sequence_audit(observed)
        from vllm.v1.flash_e3 import qsa_operation,e3_types
        qsa=importlib.import_module('vllm.models.qwen4_exp.nvidia.qsa')
        getter=importlib.import_module('vllm.forward_context').get_forward_context
        # These real constructors are host-only until streams/data bind.
        device_port=qsa_operation.NativeQsaDevice('cuda:0')
        tx=e3_types.TransactionId(1,'shape-only',1,0,64,1)
        lane=e3_types.LaneId(0,e3_types.LaneRole.PREFILL,1,0)
        operation=qsa_operation.QsaPrefillOperation(tx,lane,device=device_port)
        driver=driver_factory(operation,qsa,getter)
        assert type(driver) is qsa_driver.QsaLayerDriver
        inventory=operation.inventory()
        assert inventory['sparse_descriptor_cap']==256 and inventory['max_bucket_pages']==24
        from vllm.v1.flash_e3 import wire_v1 as w
        shape_record=w.AdmissionRecord(transaction=tx,prompt_end=65,state_epoch=1,
            committed_end=0,block_tables=((1,),),slot_mappings=(),new_block_ids_to_zero=(),
            cow_pairs=(),input_ids=(1,)*64,ngram_context=(),sparse_active=False)
        w.OpenRequest(shape_record,(0,),1,16)
        w.PollRequest(tx,1,8)
        w.ReleaseRequest(tx,1,((0,0),),True)
        for kind in w.SegmentKind:
            stage=1 if kind is w.SegmentKind.PP_RECEIVE else 0
            descriptor=w.SegmentDescriptor(tx,kind,stage,
                layer_start=1 if kind is w.SegmentKind.LAYER else None,
                layer_end=2 if kind is w.SegmentKind.LAYER else None,
                seq=0 if kind in (w.SegmentKind.PP_SEND,w.SegmentKind.PP_RECEIVE) else None)
            w.SegmentCommand(descriptor,0,(0,))
        try:
            with torch.device('cpu'):Qwen4ExpForConditionalGeneration(vllm_config=cfg)
        except TypeError as exc:
            frames=traceback.extract_tb(exc.__traceback__)
            assert frames[-1].filename.endswith('qwen_gdn_linear_attn.py')
            assert 'current_platform.current_device()' in (frames[-1].line or '')
            boundary=dict(exception=type(exc).__name__,message=str(exc),
                file=frames[-1].filename,line=frames[-1].lineno,
                reason='actual GDN constructor requires the CUDA platform device; no platform/constructor patch')
        else:raise AssertionError('CPU model construction passed; extend CPU execution')
    assert not torch.cuda.is_initialized()
    return dict(verdict='PASS',constructor_index=index,host_constructor=host,
        gpu_branch_import_statements_executed=True,actual_operation_and_driver_constructed=True,
        source_owner_preflight=source_preflight,stock_startup_sequence_audit=sequence_audit,
        predicate_selftest=predicate_selftest(w,e3_types),
        actual_config_constructed=True,actual_model_constructor_entered=True,
        empty_workspace_provision=workspace_provision,
        cuda_constructor_boundary=boundary,cuda_initialized=False,
        cpu_executed_gpu_host_steps=['config','register_all_kvcache_specs',
            'get_supported_kv_cache_layouts','resolve_kv_cache_layout','record_kv_cache_layout',
            'get_kv_cache_configs','QSAKeyStateCache','QSACompressedKeyCache',
            'generate_scheduler_kv_cache_config','update_kv_cache_capacity','validate_block_size','get_scheduler_cls',
            'resolve_kv_cache_block_sizes','Scheduler','StructuredOutputManager','SchedulerLeases.attach',
            'CommittedFrontier','PrefillAdmissionController.attach','install_admission_metadata',
            'init_none_hash','get_request_block_hasher','Request','Scheduler.add_request',
            'PrefillAdmissionController.admit','opened_source','SourceDescriptor.validate_frame',
            'typed-wire-constructors','QsaPrefillOperation','QsaLayerDriver','predicate_selftest'],
        deferred_gpu_steps=['NCCL initialization','GPUModelRunner CUDA constructor',
            'load_model CUDA allocation and dummy weights','get_kv_cache_spec from GPU model object graph',
            'initialize_kv_cache CUDA/UVA allocation','worker_components CUDA streams/private bindings',
            'execute_chunk CUDA prefix/layers/finalize/query/release',
            'stock execute_model/sample_tokens/update_from_output'],
        full_gpu_constructors_executed=False,fixtures_patched=False)


def stock_sequence_audit(observed):
    """Fail on any stock call absent from executed evidence or named deferrals.

    The actual EngineCore is inspected, never constructed or patched. The
    extracted spans are semantic AST statement anchors, not a copied step list.
    """
    import ast,textwrap
    from vllm.v1.engine.core import EngineCore
    def call_name(node):return ast.unparse(node.func)
    extracted=[]
    spans=((EngineCore._initialize_kv_caches,'resolve_kv_cache_layout','vllm_config.validate_block_size'),
           (EngineCore.__init__,'self._initialize_kv_caches','Scheduler'))
    for fn,start,stop in spans:
        lines,first=inspect.getsourcelines(fn);text=textwrap.dedent(''.join(lines));tree=ast.parse(text)
        body=tree.body[0].body
        first_stmt=next(i for i,node in enumerate(body) if any(isinstance(n,ast.Call) and call_name(n)==start for n in ast.walk(node)))
        last_stmt=next(i for i,node in enumerate(body) if i>=first_stmt and any(isinstance(n,ast.Call) and call_name(n)==stop for n in ast.walk(node)))
        for statement in body[first_stmt:last_stmt+1]:
            for node in sorted((n for n in ast.walk(statement) if isinstance(n,ast.Call)),key=lambda n:(n.lineno,n.col_offset)):
                extracted.append(dict(method=fn.__name__,line=first+node.lineno-1,call=call_name(node)))
    # Map the stock variable spelling to actual Python/C call observations.
    expected={
        'resolve_kv_cache_layout':('vllm.v1.attention.backends.utils','resolve_kv_cache_layout'),
        'get_kv_cache_configs':('vllm.v1.core.kv_cache_utils','get_kv_cache_configs'),
        'generate_scheduler_kv_cache_config':('vllm.v1.core.kv_cache_utils','generate_scheduler_kv_cache_config'),
        'update_kv_cache_capacity':('vllm.v1.core.kv_cache_utils','update_kv_cache_capacity'),
        'vllm_config.validate_block_size':('vllm.config.vllm','VllmConfig.validate_block_size'),
        'StructuredOutputManager':('vllm.v1.structured_output','StructuredOutputManager.__init__'),
        'vllm_config.scheduler_config.get_scheduler_cls':('vllm.config.scheduler','SchedulerConfig.get_scheduler_cls'),
        'resolve_kv_cache_block_sizes':('vllm.v1.core.kv_cache_utils','resolve_kv_cache_block_sizes'),
        'Scheduler':('vllm.v1.core.sched.scheduler','Scheduler.__init__'),
        'len':('builtin','len'),'any':('builtin','any'),'min':('builtin','min'),
        'specs.values':('builtin','dict.values'),
    }
    deferred={
        'self.model_executor.get_supported_kv_cache_layouts':'no executor/FIFO on hub; real underlying backend-declaration function executed',
        'self.model_executor.set_kv_cache_layout':'no executor/FIFO on hub; real underlying record_kv_cache_layout adoption executed',
        'self.model_executor.determine_available_memory':'GPU profiling deferred; explicit 256MiB reduced-model fixture budget, not production memory evidence',
        'self.collective_rpc':'conditional auto-fit update RPC absent because fixed max_model_len; no executor or rank ACK simulated',
        'self._initialize_kv_caches':'EngineCore/executor and GPU profiling intentionally not constructed; extracted host functions executed by shared helper',
        'logger.warning':'empty-cache branch not taken: actual hybrid cache groups are nonempty',
    }
    for row in extracted:
        name=row['call']
        if name in expected:
            witness=expected[name]
            assert witness in observed,('stock-call-not-executed',name,witness)
            row.update(status='executed',observed_module=witness[0],observed_function=witness[1])
        elif name in deferred:row.update(status='deferred',reason=deferred[name])
        else:raise AssertionError('unclassified-stock-startup-call:'+name)
    from vllm.v1.worker.gpu_worker import Worker
    worker_rows=[]
    for method in ('__init__','init_device','load_model','initialize_from_config'):
        fn=getattr(Worker,method)
        lines,first=inspect.getsourcelines(fn);text=textwrap.dedent(''.join(lines));tree=ast.parse(text)
        entered=(Worker.__module__,'Worker.'+method) in observed
        # CPU execution has no CUDA platform/device. The GPU proof now invokes
        # every actual public method below; these are not hand-copied bodies.
        for node in sorted((n for stmt in tree.body[0].body for n in ast.walk(stmt) if isinstance(n,ast.Call)),key=lambda n:(n.lineno,n.col_offset)):
            worker_rows.append(dict(method=method,line=first+node.lineno-1,call=call_name(node),
                cpu_method_entered=entered,status='deferred',
                reason=('actual CPU constructor completed; individual branch/subcall is not inferred from method completion' if method=='__init__' else
                        'actual CPU init_device stops at unsupported CPU device; full unmodified method and following load/cache methods require CUDA'),
                gpu_execution='real Worker.'+method+' invoked directly by the device scenario'))
    assert {row['method'] for row in worker_rows}=={'__init__','init_device','load_model','initialize_from_config'}
    assert (Worker.__module__,'Worker.__init__') in observed
    assert (Worker.__module__,'Worker.init_device') in observed
    path=Path(inspect.getsourcefile(EngineCore))
    assert extracted
    return dict(verdict='PASS',actual_core_source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        callsites=extracted,worker_callsites=worker_rows,
        worker_source_sha256=hashlib.sha256(Path(inspect.getsourcefile(Worker)).read_bytes()).hexdigest(),
        worker_methods_indexed=['__init__','init_device','load_model','initialize_from_config'],
        missing=[],profiled_actual_calls=True,engine_core_constructed=False,
        fixture_scope='CPU host metadata projection; actual source-derived call coverage, not GPU platform or RPC evidence')


def wait(fn,seconds=60):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        value=fn()
        if value:return value
        time.sleep(.0005)
    raise TimeoutError('queried lifecycle deadline')


def driver_factory(operation,qsa_module,forward_context_getter):
    from vllm.v1.flash_e3.qsa_driver import QsaLayerDriver
    return QsaLayerDriver(operation,staged_operation=qsa_module._flash_s5_staged_operation,
        sparse_staged_operation=getattr(qsa_module,'_flash_s5_sparse_staged_operation',None),
        source_holds=getattr(operation,'native_source_holds',None),
        forward_context=forward_context_getter)


def opened_source(controller,record,lane):
    controller.note_source_owners(record.transaction,{lane.worker_rank:lane},3)
    return controller.source_hold(record.transaction).descriptor()


def cache_configuration(cfg,specs,backends,*,legacy_negative=False):
    """Stock EngineCore host sequence, shared by CPU-shape and GPU execution."""
    from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
    from vllm.v1.attention.backends.utils import (get_supported_kv_cache_layouts,
        resolve_kv_cache_layout,record_kv_cache_layout)
    from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
    register_all_kvcache_specs(cfg)
    supported=[layout.name for layout in get_supported_kv_cache_layouts(backends)]
    unresolved_negative=None
    if legacy_negative:
        assert cfg.cache_config.kv_cache_layout is None
        try:get_kv_cache_configs(cfg,[specs],[256*1024**2])
        except ValueError as exc:
            assert 'KV cache layout has not been resolved yet' in str(exc), str(exc)
            unresolved_negative=str(exc)
        else:raise AssertionError('native15-matrix2 unresolved cache negative passed')
    layout=resolve_kv_cache_layout(cfg,[supported],list(specs.values()))
    # This is the stock worker adoption method's implementation, not a chosen
    # layout value: resolve above derives it from actual backend declarations.
    record_kv_cache_layout(cfg.cache_config,layout.name)
    kv=get_kv_cache_configs(cfg,[specs],[256*1024**2])[0]
    kv.kv_cache_layout=cfg.cache_config.kv_cache_layout
    cfg.cache_config.num_gpu_blocks=kv.num_blocks
    assert cfg.cache_config.get_resolved_kv_cache_layout()==layout
    return kv,dict(verdict='PASS',layout=layout.name,supported=supported,
        actual_resolver=True,manual_layout_assignment=False,
        legacy_unresolved_negative=unresolved_negative,
        spec_types={name:type(spec).__name__ for name,spec in specs.items()},
        num_blocks=kv.num_blocks,groups=len(kv.kv_cache_groups))


def controller_components(cfg,worker_kv):
    """The stock worker->scheduler projection, capacity and block-size path."""
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.structured_output import StructuredOutputManager
    from vllm.v1.core.kv_cache_utils import (generate_scheduler_kv_cache_config,
        update_kv_cache_capacity,resolve_kv_cache_block_sizes)
    from vllm.v1.flash_e3 import scheduler_hooks as sh,async_handoff as ah,admission_metadata as am
    kv=generate_scheduler_kv_cache_config([worker_kv])
    cfg.cache_config.num_gpu_blocks=kv.num_blocks
    groups=kv.kv_cache_groups
    participating=[g.kv_cache_spec.block_size for g in groups if g.kv_cache_spec.prefix_cacheable]
    cfg.cache_config.block_size=min(participating or [g.kv_cache_spec.block_size for g in groups])
    update_kv_cache_capacity(cfg,kv)
    cfg.validate_block_size()
    block_size,hash_block_size=resolve_kv_cache_block_sizes(kv,cfg)
    assert block_size==4048 and hash_block_size==176,(block_size,hash_block_size)
    Scheduler=cfg.scheduler_config.get_scheduler_cls()
    scheduler=Scheduler(cfg,kv,StructuredOutputManager(cfg),block_size=block_size,hash_block_size=hash_block_size)
    leases=sh.SchedulerLeases().attach(scheduler,scheduler.kv_cache_manager.block_pool)
    controller=sh.PrefillAdmissionController(scheduler,leases=leases,frontier=ah.CommittedFrontier(),
        engine_epoch=1,chunk_tokens=1024,ranks=(0,),minimum_stage_rows=64).attach()
    am.install_admission_metadata(scheduler)
    return scheduler,controller


def source_owner_preflight(cfg):
    """Real CPU allocation -> shared source-binding path; no worker/proof invented.

    The typed lane is a predicate input only, not an actual opened GPU lane.
    No ticket is bound, no completion is supplied and no commit is attempted.
    """
    import torch
    from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpForConditionalGeneration as Model
    from vllm.v1.kv_cache_interface import KVCacheConfig,KVCacheGroupSpec,MambaSpec,FullAttentionSpec
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum as Kind
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.structured_output import StructuredOutputManager
    from vllm.v1.request import Request,RequestStatus
    from vllm.sampling_params import SamplingParams
    from vllm.v1.flash_e3 import scheduler_hooks as sh,async_handoff as ah,admission_metadata as am,e3_types as et,worker_bridge as wb
    from vllm.utils.hashing import get_hash_fn_by_name
    from vllm.v1.core.kv_cache_utils import init_none_hash,get_request_block_hasher
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionBackend
    from vllm.v1.attention.backends.short_conv_attn import PleShortConvAttentionBackend
    from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAFlashAttentionBackend
    from vllm.models.qwen4_exp.common.qsa_cache import QSAKeyStateCache,QSACompressedKeyCache,QSAStateBackend
    specs=Model.get_mamba_specs_from_config(cfg)
    assert len(specs)==2
    gdn,ple=specs
    def state_spec(spec,kind):
        return MambaSpec(block_size=4048,shapes=tuple(spec.shapes),dtypes=tuple(spec.dtypes),
            mamba_type=kind,mamba_cache_mode='align')
    # Real model-derived state specs; reduced-model names/attention geometry are
    # explicit CPU-shape leaves, not the GPU model's registered object graph.
    cache_specs={'shape.gdn0':state_spec(gdn,Kind.GDN_ATTN),
        'shape.gdn2':state_spec(gdn,Kind.GDN_ATTN),
        'shape.ple':state_spec(ple,Kind.SHORT_CONV),
        'shape.qsa':Qwen4ExpQSAFlashAttentionBackend.customize_spec(FullAttentionSpec(
            block_size=4048,num_kv_heads=1,head_size=256,head_size_v=256,dtype=torch.bfloat16,
            num_head_slots=1,state_content_bytes=2))}
    # The actual CPU platform auto-disables HMA, unlike the GPU platform.
    # This is an explicit host-metadata fixture projection, not a platform shim
    # or a ready flag. Actual GPU execution uses its unmodified real config.
    import copy
    from vllm.config import SchedulerConfig
    host_cfg=copy.copy(cfg)
    host_cfg.scheduler_config=SchedulerConfig(max_model_len=98320,is_encoder_decoder=False,
        max_num_batched_tokens=1024,max_num_seqs=10,scheduler_reserve_full_isl=False,
        disable_hybrid_kv_cache_manager=False)
    # These real side-cache constructors allocate empty CPU tensors, register
    # their actual classes, and derive their exact circular/compressed specs.
    side_caches=[QSAKeyStateCache(head_size=64,dtype=torch.bfloat16,cache_config=host_cfg.cache_config,
        prefix='shape.indexer.raw_key_cache',vllm_config=host_cfg,compress_ratio=4,
        cache_rope_positions=host_cfg.model_config.uses_mrope),
        QSACompressedKeyCache(head_size=64,dtype=torch.bfloat16,cache_config=host_cfg.cache_config,
        prefix='shape.indexer.compressed_key_cache',vllm_config=host_cfg,compress_ratio=4)]
    for owner in side_caches:cache_specs[owner.prefix]=owner.get_kv_cache_spec(host_cfg)
    kv,cache_proof=cache_configuration(host_cfg,cache_specs,
        (GDNAttentionBackend,PleShortConvAttentionBackend,Qwen4ExpQSAFlashAttentionBackend,QSAStateBackend),legacy_negative=True)
    worker_spec_types=[type(g.kv_cache_spec).__name__ for g in kv.kv_cache_groups]
    assert 'UniformTypeKVCacheSpecs' in worker_spec_types,worker_spec_types
    try:Scheduler(host_cfg,kv,StructuredOutputManager(host_cfg),block_size=4048,hash_block_size=176)
    except AssertionError as exc:
        assert 'No manager registered' in str(exc) and 'UniformTypeKVCacheSpecs' in str(exc),str(exc)
        scheduler_negative=str(exc)
    else:raise AssertionError('matrix2B worker-config-as-scheduler negative passed')
    scheduler,controller=controller_components(host_cfg,kv)
    from vllm.v1.worker.gpu_worker import Worker,ensure_kv_transfer_initialized
    worker=Worker(cfg,local_rank=0,rank=0,distributed_init_method='file:///tmp/matrix-cpu-worker',is_driver_worker=True)
    assert worker.use_v2_model_runner is True
    try:worker.init_device()
    except RuntimeError as exc:
        assert 'Unsupported device type' in str(exc),str(exc)
        worker_boundary=exception_chain(exc)
    else:raise AssertionError('CPU worker init_device completed; extend execution')
    ensure_kv_transfer_initialized(host_cfg,kv)
    hash_fn=get_hash_fn_by_name('sha256');init_none_hash(hash_fn)
    request=Request(request_id='lifecycle-cold-a',prompt_token_ids=[i%200+1 for i in range(98312)],
        sampling_params=SamplingParams(max_tokens=1),pooling_params=None,
        block_hasher=get_request_block_hasher(176,hash_fn))
    scheduler.add_request(request)
    record=controller.admit(scheduler);assert record is not None
    unpublished=tuple(controller.parked.capture.details['unpublished_blocks'])
    assert unpublished
    lane=et.LaneId(0,et.LaneRole.PREFILL,1,0)
    source=opened_source(controller,record,lane)
    checkpoint('cpu-admission-contract',shape=admission_shape(record))
    chunk_admission=wb.admission_to_chunk(record)
    source.validate_frame(chunk_admission,(0,))
    zero_diagnostic=cpu_zeroer_diagnostic(cfg,chunk_admission)
    assert controller.parked.bound is False
    assert controller.commits==0 and not controller.frontier.committed_blocks()
    controller.cancel(record.transaction,never_submitted=True)
    scheduler.finish_requests([request.request_id],RequestStatus.FINISHED_ABORTED)
    assert controller.commits==0 and not controller.frontier.committed_blocks()
    return dict(verdict='PASS',actual_scheduler_constructor=True,actual_allocator_capture=True,
        actual_shared_source_binding=True,rank_lane_map_validated=True,
        cache_configuration=cache_proof,worker_spec_types=worker_spec_types,
        scheduler_spec_types=[type(g.kv_cache_spec).__name__ for g in scheduler.kv_cache_manager.kv_cache_config.kv_cache_groups],
        legacy_worker_config_negative=scheduler_negative,cpu_hma_metadata_projection=True,
        cpu_platform_hma_auto_disabled=cfg.scheduler_config.disable_hybrid_kv_cache_manager,
        admission_contract=admission_shape(record),zeroer_diagnostic=zero_diagnostic,
        actual_worker_constructor=True,worker_device_boundary=worker_boundary,actual_transfer_precondition=True,
        unpublished_count=len(unpublished),no_ticket_bound=True,no_publication=True,
        synthetic_lane_input=True,box_evidence=False)


def worker_components(runner):
    """Actual primitive composition; public TP2/PP2 startup is not simulated."""
    import torch
    from vllm.v1.worker import workspace
    from vllm.v1.flash_e3 import (native_resources as nr,native_backend as nb,
        lane_state as ls,lane_hooks as lh,lane_chunk as lc,lane_dispatch as ld,
        native_startup as startup,lane_comm,lane_private,qsa_driver as qd,worker_bridge as wb)
    device=nb.NativeDevice(device=runner.device)
    forward=nr._forward_context_module()
    uid=device.allocate_i32(runner.max_num_reqs,name='flash_s5_prefill_uid')
    bindings,inventory,buffers=nr._allocate_scratch(device,1024)
    nr._ensure_workspace_lanes(workspace,device,1)
    state=ls.WorkerState(0,engine_epoch=0,engine_bound=False,weights=runner.model,
        weights_bytes=sum(t.numel()*t.element_size() for t in runner.model.parameters()),max_outstanding=16)
    # The normal binding method derives engine_bound; no ready/verified flag is assigned.
    state.bind_engine_owner(1,incarnation=state.incarnation)
    runtime=lh.LaneRuntime(state,forward_context=nr._ForwardContextAdapter(forward),
        workspace=nr._WorkspaceLaneAdapter(workspace,1),submit_backend=nb.EventCompletionBackend(device),
        prefill_forward_context=nr._prefill_context(forward,runner),prefill_workspace=1,
        uid_namespace=forward,prefill_uid_buffer=uid.tensor)
    runtime._native_owned_buffers=(uid,*buffers)
    runner._flash_e3_lane=runtime
    prefill=torch.cuda.Stream();copy=torch.cuda.Stream()
    op_device=nr.NativeQsaDevice(runner.device);op_device.bind_streams(prefill,copy)
    server=nr.WorkerHandoffServer(runtime,bindings=bindings,operation_device=op_device)
    # Single-rank ordinary GEMMs do not consume the Marlin workspace. This
    # allocation is a declared geometry fixture, not a production high-water fact.
    manager=workspace.current_workspace_manager()
    manager.get_simultaneous(((128,),torch.uint8))
    private=lane_private.LanePrivateBindings.discover(runner,runtime,chunk_tokens=1024,torch_module=torch)
    private.prepare_workspace()
    comms=lane_comm.PrefillCommunicators(None,None)
    native=startup.WorkerNative(runner,SimpleNamespace(prefill_stream=prefill),None,None,None,
        comms,None,None,1024,None,torch,state=state,private_bindings=private)
    native.copy_stream=copy
    factory=lc.LaneChunkFactory(runner)
    qsa=importlib.import_module('vllm.models.qwen4_exp.nvidia.qsa')
    getter=importlib.import_module('vllm.forward_context').get_forward_context
    dispatcher=ld.LaneDispatcher(runner,native,factory,server=server,qsa_module=qsa,
        driver_factory=driver_factory,forward_context_getter=getter,torch_module=torch)
    bridge=wb.NativeWorkerBridge(dispatcher,server=server,rank=0)
    runner._flash_s5_bridge=bridge
    return SimpleNamespace(runtime=runtime,server=server,native=native,factory=factory,
        dispatcher=dispatcher,bridge=bridge,state=state)


def execute_chunk(stack,controller,record):
    """Wire commands -> real bridge/dispatcher -> real model -> queried release."""
    from vllm.v1.flash_e3 import wire_v1 as w,e3_types as et
    bridge=stack.bridge
    checkpoint('bridge-open-begin',admission=admission_shape(record),raw_admission=dataclasses.asdict(record))
    opened=bridge.open(w.OpenRequest(record,(0,),record.state_epoch,16))
    checkpoint('bridge-open-end',ack=dataclasses.asdict(opened))
    source=opened_source(controller,record,opened.lane)
    seq=0;accepted=[];bound=None
    def command(kind,layer=None):
        nonlocal seq,bound
        descriptor=w.SegmentDescriptor(record.transaction,kind,0,layer_start=layer,
            layer_end=None if layer is None else layer+1)
        cmd=w.SegmentCommand(descriptor,seq,(0,),source if kind is w.SegmentKind.PREPARE else None)
        def submit():
            nonlocal seq,bound
            checkpoint('submit-begin',kind=kind.value,op_seq=seq,layer=layer)
            ack=bridge.submit(cmd)
            checkpoint('submit-end',kind=kind.value,op_seq=seq,ack=dataclasses.asdict(ack))
            if ack.pending:return None
            assert ack.accepted
            accepted.append(seq)
            if bound is None:
                ticket=et.OperationTicket(record.transaction,0,opened.lane,seq,et.DependencyRole.PRODUCER)
                controller.bind(ticket);bound=ticket
            seq+=1
            return ack
        return wait(submit)
    command(w.SegmentKind.PREPARE);command(w.SegmentKind.PREFIX)
    for layer in range(3):
        while command(w.SegmentKind.LAYER,layer).paused:pass
    command(w.SegmentKind.SUFFIX);command(w.SegmentKind.STATE_FINALIZE)
    poll=w.PollRequest(record.transaction,record.state_epoch,8)
    def complete():
        report=bridge.poll(poll)
        controller.note_report(report)
        return report if report.state_written() and not report.pending_op_seqs else None
    report=wait(complete)
    operation=stack.server.operation
    inventory=operation.inventory()
    routes=[]
    for name,run in operation._layers.items():
        sparse=run.sparse
        if sparse is not None:
            plan=sparse.plan
            routes.append(dict(layer=name,route=sparse.route,
                bucket_types=sorted({type(b).__name__ for b in plan.buckets}) if plan is not None else []))
    producer=next(p for p in report.proofs if p.ticket==bound and p.kind is et.CompletionKind.GPU_EVENT)
    release=w.ReleaseRequest(record.transaction,record.state_epoch,((0,report.completed_sequence),),record.terminal)
    ack=wait(lambda: (a if (a:=bridge.release(release)).released else None))
    assert type(report) is w.WorkerCompletionReport and type(ack) is w.ReleaseAck
    controller.note_worker_release(ack)
    assert controller.ready_to_commit(record.transaction)
    controller.commit(record.transaction,producer)
    assert ack.request_closed is record.terminal
    assert stack.factory.request.req_id == (None if record.terminal else record.transaction.request_uid)
    return dict(start=record.transaction.chunk_start,end=record.transaction.chunk_end,
        rows=record.chunk_tokens,terminal=record.terminal,accepted_ops=len(accepted),
        proofs=len(report.proofs),request_closed=ack.request_closed,
        operation_inventory=inventory,sparse_routes=routes,
        report=dataclasses.asdict(report),release=dataclasses.asdict(ack))


def gpu_scenarios():
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.distributed import init_distributed_environment,initialize_model_parallel
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.core.kv_cache_utils import get_kv_cache_configs,init_none_hash,get_request_block_hasher
    from vllm.utils.hashing import get_hash_fn_by_name
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.structured_output import StructuredOutputManager
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request
    from vllm.v1.flash_e3 import scheduler_hooks as sh,admission_metadata as am,async_handoff as ah
    from vllm.v1.worker.workspace import init_workspace_manager
    cfg=config('cuda');checkpoint('config-constructed')
    with set_current_vllm_config(cfg):
        from vllm.v1.worker.gpu_worker import Worker
        checkpoint('worker-constructor-begin')
        worker=Worker(cfg,local_rank=0,rank=0,distributed_init_method='file:///tmp/matrix-gpu-dist',is_driver_worker=True)
        checkpoint('worker-constructed',runner_v2=worker.use_v2_model_runner)
        assert worker.use_v2_model_runner is True
        worker.init_device();runner=worker.model_runner
        assert type(runner) is GPUModelRunner
        checkpoint('worker-init-device-complete')
        provision_local_workspace(runner.device)
        worker.load_model(load_dummy_weights=True);checkpoint('worker-model-loaded')
        specs=runner.get_kv_cache_spec()
        from vllm.distributed.kv_transfer.kv_connector.utils import get_current_attn_backends
        kv,cache_proof=cache_configuration(cfg,specs,get_current_attn_backends(cfg))
        checkpoint('cache-configured',proof=cache_proof)
        worker.initialize_from_config(kv)
        checkpoint('worker-cache-initialized',needs_kv_cache_zeroing=kv.needs_kv_cache_zeroing,
            zeroer_type=type(runner.kv_block_zeroer).__name__)
        assert not kv.needs_kv_cache_zeroing or runner.kv_block_zeroer is not None
        stack=worker_components(runner);checkpoint('worker-components-constructed')
        scheduler,controller=controller_components(cfg,kv);checkpoint('scheduler-constructed')
        h=get_hash_fn_by_name('sha256');init_none_hash(h)
        hasher=get_request_block_hasher(176,h)
        requests=[];previous_physical=set()
        plans=(('cold-a',98312,0),('cached-inside-page',98312,0),
               ('cold-b-recycle',98312,11),('warm-page',4049,47),('cached-page',4049,47))
        for name,length,shift in plans:
            request=Request(request_id='lifecycle-'+name,
                prompt_token_ids=[(i+shift)%200+1 for i in range(length)],
                sampling_params=SamplingParams(max_tokens=1,temperature=0),pooling_params=None,block_hasher=hasher)
            scheduler.add_request(request);checkpoint('request-added',request_name=name,prompt_tokens=length)
            _,hit,_=scheduler.kv_cache_manager.get_computed_blocks(request)
            chunks=[];physical=set()
            while request.num_computed_tokens<request.num_tokens-1:
                record=controller.admit(scheduler)
                if record is None:break
                # Actual allocator-capture obligations, not cached page touches
                # or a convenient growing-ID allocation fixture.
                unpublished=tuple(controller.parked.capture.details['unpublished_blocks'])
                physical.update(unpublished)
                checkpoint('admission-created',shape=admission_shape(record))
                row=execute_chunk(stack,controller,record)
                checkpoint('chunk-completed',receipt=row)
                row['unpublished_blocks']=list(unpublished)
                chunks.append(row)
            assert stack.factory.request.req_id is None
            ordinary=scheduler.schedule()
            assert ordinary.num_scheduled_tokens[request.request_id]==1
            output=runner.execute_model(ordinary)
            if output is None:output=runner.sample_tokens(None)
            if callable(getattr(output,'get_output',None)):output=output.get_output()
            assert output is not None
            scheduler.update_from_output(ordinary,output)
            assert request.num_output_tokens==1 and request.is_finished()
            runner.execute_model(scheduler.schedule())
            assert request.request_id not in runner.req_states.req_id_to_index
            requests.append(dict(name=name,prompt_tokens=length,chunk_count=len(chunks),chunks=chunks,cache_hit=hit,
                hit_inside_page=bool(hit and hit%4048),hit_page_edge=bool(hit and hit%4048==0),
                physical_reallocated=sorted(physical&previous_physical),
                tail_rows=[r['rows'] for r in chunks[-3:]],ordinary_tokens=1,
                ordinary_output_tokens=request.num_output_tokens,worker_row_closed=True))
            previous_physical.update(physical)
        assert requests[1]['cache_hit']>0
        assert requests[-1]['hit_page_edge']
        return dict(verdict='PASS',requests=requests,
            real_model_class=type(runner.model).__module__+'.'+type(runner.model).__name__,
            cache_configuration=cache_proof,
            typed_single_worker_releases=True,all_rank_full_certificate=False,
            sampling_after_handoff=True,engine_qualified=False)


def mounted_origins(graph_path):
    import sys
    graph=json.loads(graph_path.read_text());assert graph['verdict']=='PASS'
    rows=[]
    for spec in graph['modules']:
        module=sys.modules.get(spec['module'])
        if not spec['active_mount'] or module is None:continue
        path=Path(module.__file__).resolve()
        assert path==Path(spec['destination']).resolve(),spec['module']
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest==spec['sha256'],spec['module']
        rows.append(dict(module=spec['module'],origin=str(path),sha256=digest))
    assert any(r['module']=='vllm.v1.worker.gpu.model_runner' for r in rows)
    return rows


def grade_lifecycle(record):
    """Pure scoped structural regrade; synthetic unit rows are not GPU facts."""
    failures=[];missing=[]
    try:
        if record.get('all_rank_full_certificate') is not False:failures.append('scope:full-certificate')
        if record.get('engine_qualified') is not False:failures.append('scope:engine-qualified')
        if record.get('sampling_after_handoff') is not True:missing.append('stock-sampling')
        rows=record['requests']
        expected={'cold-a','cached-inside-page','cold-b-recycle','warm-page','cached-page'}
        if {r['name'] for r in rows}!=expected or len(rows)!=5:missing.append('request-coverage')
        for request in rows:
            name=request['name'];chunks=request['chunks']
            expected_length=4049 if name in ('warm-page','cached-page') else 98312
            if request['prompt_tokens']!=expected_length:failures.append(name+':prompt-shape')
            if request['ordinary_tokens']!=1 or request['ordinary_output_tokens']!=1:failures.append(name+':handoff-token')
            if request['worker_row_closed'] is not True:missing.append(name+':worker-row')
            position=request['cache_hit']
            for i,chunk in enumerate(chunks):
                if chunk['start']!=position or chunk['rows']!=chunk['end']-chunk['start']:failures.append(name+':frontier-gap')
                position=chunk['end']
                if not 64<=chunk['rows']<=1024:failures.append(name+':threshold')
                report=chunk['report'];release=chunk['release']
                if report['pending_op_seqs'] or report.get('failed_code'):failures.append(name+':pending-or-failed')
                if report['completed_op_seqs']!=list(range(chunk['accepted_ops'])):failures.append(name+':accepted-set')
                proofs=report['proofs']
                state_proofs=[p for p in proofs if p['kind']=='state_write']
                if not state_proofs:missing.append(name+':state-write')
                if len(state_proofs)>1:failures.append(name+':duplicate-state-write')
                for proof in proofs:
                    ticket=proof['ticket']
                    if (ticket['transaction']!=report['transaction'] or ticket['worker_rank']!=0
                            or ticket['lane']['worker_rank']!=0 or ticket['lane']['role']!='prefill'
                            or not 0<=proof['completion_seq']<=report['completed_sequence']):
                        failures.append(name+':proof-identity')
                    if proof['kind']=='state_write' and (
                            ticket['dependency_role']!='commit' or ticket['op_seq']!=chunk['accepted_ops']-1):
                        failures.append(name+':state-write-role')
                gpu_rows=[p for p in proofs if p['kind']=='gpu_event']
                gpu={p['ticket']['op_seq'] for p in gpu_rows}
                if len(gpu_rows)!=len(gpu):failures.append(name+':duplicate-gpu-proof')
                if gpu!=set(report['completed_op_seqs']):missing.append(name+':gpu-events')
                terminal=i==len(chunks)-1
                if chunk['terminal'] is not terminal:failures.append(name+':terminal')
                if (report['worker_rank']!=0 or release['worker_rank']!=report['worker_rank']
                        or release['transaction']!=report['transaction'] or release['state_epoch']!=report['state_epoch']
                        or release['completed_sequence']!=report['completed_sequence']):failures.append(name+':release-identity')
                if release['released'] is not True or release['request_closed'] is not terminal:failures.append(name+':release-closure')
            if position!=request['prompt_tokens']-1:missing.append(name+':native-prefix-or-cache-coverage')
        inside=next(r for r in rows if r['name']=='cached-inside-page')
        edge=next(r for r in rows if r['name']=='cached-page')
        if not inside['hit_inside_page']:missing.append('inside-page-cache-hit')
        if not edge['hit_page_edge']:missing.append('page-edge-cache-hit')
        if not any(r['physical_reallocated'] for r in rows if r['name']=='cold-b-recycle'):
            missing.append('completed-request-allocation-reuse-not-exercised')
    except (KeyError,TypeError,ValueError,StopIteration) as e:
        missing.append('receipt-shape:'+type(e).__name__)
    return dict(verdict='FAIL' if failures else 'INCONCLUSIVE' if missing else 'PASS',
        failures=sorted(set(failures)),missing=sorted(set(missing)),box_go=False)


def predicate_receipt(w,et):
    """Synthetic typed rows solely for predicate execution, never GPU facts."""
    scenarios=[];sequence=0
    for name,length,hit in (('cold-a',98312,0),('cached-inside-page',98312,98208),
        ('cold-b-recycle',98312,0),('warm-page',4049,0),('cached-page',4049,4048)):
        position=hit;chunks=[]
        while position<length-1:
            count=min(1024,length-1-position)
            if 0<length-1-position-count<64:count-=64-(length-1-position-count)
            end=position+count;sequence+=1
            tx=et.TransactionId(1,name,1,position,end,sequence)
            lane=et.LaneId(0,et.LaneRole.PREFILL,sequence,0)
            tickets=[et.OperationTicket(tx,0,lane,i,
                et.DependencyRole.COMMIT if i==6 else et.DependencyRole.PRODUCER) for i in range(7)]
            proofs=tuple(et.CompletionProof(t,et.CompletionKind.GPU_EVENT,i) for i,t in enumerate(tickets))
            proofs+=(et.CompletionProof(tickets[-1],et.CompletionKind.STATE_WRITE,7),)
            report=w.WorkerCompletionReport(tx,0,tuple(range(7)),(),7,proofs,w.CompletionCursor('commit',3,0),sequence)
            release=w.ReleaseAck(tx,0,sequence,7,True,end==length-1)
            chunks.append(dict(start=position,end=end,rows=count,terminal=end==length-1,
                accepted_ops=7,proofs=8,request_closed=end==length-1,
                report=dataclasses.asdict(report),release=dataclasses.asdict(release)))
            position=end
        scenarios.append(dict(name=name,prompt_tokens=length,chunk_count=len(chunks),chunks=chunks,
            cache_hit=hit,hit_inside_page=bool(hit and hit%4048),hit_page_edge=bool(hit and hit%4048==0),
            physical_reallocated=[1] if name=='cold-b-recycle' else [],tail_rows=[r['rows'] for r in chunks[-3:]],
            ordinary_tokens=1,ordinary_output_tokens=1,worker_row_closed=True))
    return json.loads(json.dumps(dict(verdict='PASS',requests=scenarios,sampling_after_handoff=True,
        all_rank_full_certificate=False,engine_qualified=False),default=str))


def predicate_selftest(w,et):
    import copy
    raw=predicate_receipt(w,et)
    positive=grade_lifecycle(raw);assert positive['verdict']=='PASS',positive
    mutation=copy.deepcopy(raw);mutation['requests'][0]['chunks'][0]['release']['request_closed']=True
    negative=grade_lifecycle(mutation);assert negative['verdict']=='FAIL',negative
    return dict(verdict='PASS',synthetic_receipt=True,box_evidence=False,
        actual_wire_dataclass_constructors=True,positive=positive,negative=negative)


def main():
    p=argparse.ArgumentParser();p.add_argument('--graph',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=('cpu-shape','gpu-all'),default='cpu-shape')
    a=p.parse_args();start=time.monotonic()
    global _PROGRESS
    _PROGRESS=(a.out.parent/'progress.jsonl').open('x')
    result=dict(schema='native15-connected-lifecycle/1',verdict='INCONCLUSIVE',rc=2,
        failure_class='NONE',failures=[],missing=[],matrix_complete=False,box_go=False,engine_constructed=False)
    try:
        result['fixture_leaves']=json.loads(Path(__file__).with_name('fixture-leaves.json').read_text())
        assert result['fixture_leaves']['clears_box_hold'] is False
        if a.mode=='cpu-shape':result['collect']=collect()
        else:
            import torch
            assert torch.cuda.device_count()==1
            torch.cuda.set_per_process_memory_fraction((2*1024**3)/torch.cuda.get_device_properties(0).total_memory,0)
            from e3_native_transitions_matrix_probe import sampler_host_constructor_path
            result['host_constructor']=sampler_host_constructor_path(cpu_device_probe=False)
            result['scenarios']=gpu_scenarios()
            result['structural_regrade']=grade_lifecycle(json.loads(json.dumps(result['scenarios'],default=str)))
            result['allocator_peak_allocated']=int(torch.cuda.max_memory_allocated())
            result['allocator_peak_reserved']=int(torch.cuda.max_memory_reserved())
            assert result['structural_regrade']['verdict']!='FAIL',result['structural_regrade']
            if result['structural_regrade']['verdict']=='INCONCLUSIVE':
                result['missing']=result['structural_regrade']['missing']
        result['mounted_origins']=mounted_origins(a.graph)
        result.update(verdict='PASS',rc=0)
        if result['missing']:result.update(verdict='INCONCLUSIVE',rc=2)
    except Exception as e:
        result.update(verdict='FAIL',rc=1,failure_class='OPEN',failures=['OPEN:'+type(e).__name__],
            exception_chain=exception_chain(e),
            exception={'type':type(e).__name__,'message':str(e)[:1024],
                'stack':[{'file':Path(f.filename).name,'function':f.name,'line':f.lineno}
                    for f in traceback.extract_tb(e.__traceback__)[-15:]]})
    _PROGRESS.close();_PROGRESS=None
    result['wall_s']=time.monotonic()-start
    with a.out.open('x') as f:json.dump(result,f,indent=2,default=str)
    print(json.dumps({k:result[k] for k in ('schema','verdict','rc','failure_class','failures','wall_s')}),flush=True)
    return result['rc']


if __name__=='__main__':raise SystemExit(main())
