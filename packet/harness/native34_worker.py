"""Native34 worker-side causal receipts; numerical functions stay unchanged."""
import functools
import inspect
import os
import threading
import weakref

import native30_host as host
import native30_localize as localize
from native34_storage import AllocationWindow,MAX_HISTORY,compact,compare_manifests
from native34_indexer import operands,direct_bounds

STATES=weakref.WeakKeyDictionary()
GRAPHS=weakref.WeakKeyDictionary()
TLS=threading.local()
RUNNER=None
MAX_BYTES=8*1024**2
COHORTS={(0,1,2,3),(4,5,6,7)}
PHASES={'LAYER_PRE','ATTENTION_PRE','INDEXER','TP_SELECTION','KV_WRITE',
        'PRODUCER_EVENT','OUTPUT_ZERO','CONTINUATION_ATTACH'}


def state(runner,rank=None):
    value=STATES.get(runner)
    if value is None:
        value=dict(rank=rank,step=0,records=0,bytes=0,limit=False,mode='OFF',
            arm=None,epoch=None,idx=None,graph_replays=[],phases=[],prefix_issued=False,
            history=None,history_available=False,metadata_complete=True,providers={})
        STATES[runner]=value
    if rank is not None:value['rank']=rank
    return value


def emit(s,event,**fields):
    try:
        if s['bytes']>=MAX_BYTES:
            if not s['limit']:
                localize.emit('native34-receipt-limit',rank=s['rank'],coverage_complete=False)
                s['limit']=True
            return
        host.emit(s,'native34-'+event,**fields)
    except Exception:s['metadata_complete']=False


def observe(s,where,fn):
    try:return fn()
    except Exception as exc:
        s['metadata_complete']=False
        emit(s,'unavailable',where=where,error_type=type(exc).__name__,coverage_complete=False)
        return None


def patch(owner,name,callback):
    descriptor=inspect.getattr_static(owner,name) if inspect.isclass(owner) else None
    kind=staticmethod if isinstance(descriptor,staticmethod) else classmethod if isinstance(descriptor,classmethod) else None
    old=descriptor.__func__ if kind else getattr(owner,name)
    if getattr(old,'_native34_wrapped',False):return
    @functools.wraps(old)
    def wrapped(*a,**k):return callback(old,a,k)
    wrapped._native34_wrapped=True
    setattr(owner,name,kind(wrapped) if kind else wrapped)


class Inventory(dict):complete=True


def tensors(roots,limit=64):
    result=Inventory();seen=set();nodes=0
    def visit(value,path,depth):
        nonlocal nodes
        nodes+=1
        if nodes>4096 or len(result)>=limit or depth>7:
            result.complete=False;return
        if value is None or type(value) in (str,bytes,int,float,bool):return
        if callable(getattr(value,'untyped_storage',None)):
            result[path]=value;return
        if id(value) in seen:return
        seen.add(id(value))
        if type(value) is dict:
            if len(value)>256:result.complete=False
            for key,item in list(value.items())[:256]:
                if type(key) in (str,int):visit(item,path+'.'+str(key),depth+1)
        elif type(value) in (tuple,list):
            if len(value)>256:result.complete=False
            for i,item in enumerate(value[:256]):visit(item,path+f'[{i}]',depth+1)
        elif type(value).__module__.startswith('vllm.v1.worker.gpu'):
            for name,item in vars(value).items():
                if name not in ('model','runner','vllm_config') and not callable(item):visit(item,path+'.'+name,depth+1)
        elif hasattr(value,'__dict__') and not callable(value):result.complete=False
    for name,value in roots.items():visit(value,name,0)
    return result


def roots(runner):
    return {name:getattr(runner,name,None) for name in
            ('input_buffers','block_tables','req_states','model_state')}


def slot_contract(runner,with_native=False):
    requests=runner.req_states
    indices=sorted(int(value) for value in requests.req_id_to_index.values())
    count=9 if with_native else 8
    observed=host.state(runner)
    native=observed.get('native_index')
    return dict(max_num_reqs=int(requests.max_num_reqs),occupied_indices=indices[:16],
        native_request_index=native,native_idx_mapping=observed.get('native_idx'),
        matches=int(requests.max_num_reqs)==10 and indices==list(range(count))
                and (not with_native or native==8),
        scope='host request slots only; no device content readback')


def update_history(runner,s):
    if not s['history_available']:return
    import torch
    observe(s,'allocator-snapshot',lambda:s['history'].update(torch.cuda.memory._snapshot()))


def start_history(runner,s):
    import torch
    snapshot=torch.cuda.memory._snapshot()
    if any(snapshot.get('device_traces',())):
        emit(s,'allocation-history-unavailable',reason='preexisting-history-not-reconfigured',coverage_complete=False)
        return
    torch.cuda.memory._record_memory_history(enabled='all',context=None,stacks='python',
                                             max_entries=MAX_HISTORY,clear_history=False)
    s['history']=AllocationWindow();s['history'].start(torch.cuda.memory._snapshot())
    s['history_available']=True
    emit(s,'allocation-history-start',max_entries=MAX_HISTORY,stack_traces=False,
         baseline_creation='unknown',scope='bounded window; no tensor retention or device readback')


def poll_phases(s):
    # Query failures are device evidence and must propagate as the original
    # exception. They are not metadata-encoding failures.
    for phase in s['phases']:
        try:ready=not phase['complete'] and phase['event'].query()
        except BaseException as exc:
            emit(s,'phase-completion-exception',phase=phase['name'],event_id=phase['id'],
                 stream=phase['stream'],error_type=type(exc).__name__,
                 scope='event-query reporting site; prior work or another stream may be responsible')
            raise
        if ready:
            phase['complete']=True
            emit(s,'phase-device-complete',phase=phase['name'],event_id=phase['id'],
                 stream=phase['stream'],scope='native current-stream tail; auxiliary drain separate')


def drain_all(runner,s):
    import torch
    native=runner._flash_s5_native
    # These are actual provider methods, including capture/output/PP consumers.
    consumer_streams=[dict(name=f'decode-consumer-{i}',stream=int(stream.cuda_stream))
                      for i,stream in enumerate(native.controller.consumer_streams())]
    consumers=native.controller.drain()
    auxiliaries=native.drain_prefill_streams()
    streams=[('native-main',native.split.prefill_stream)]
    pp=getattr(native.comms,'pp',None)
    if pp is not None:streams.append(('native-pp',pp.stream))
    bridge=getattr(runner,'_flash_s5_bridge',None)
    dispatcher=getattr(bridge,'dispatcher',None)
    operation=getattr(dispatcher,'operation',None)
    copy_stream=getattr(getattr(operation,'device',None),'_copy_stream',None)
    if copy_stream is not None:streams.append(('native-copy',copy_stream))
    seen=set();records=[]
    for name,stream in streams:
        token=int(stream.cuda_stream)
        if token in seen:continue
        seen.add(token);event=torch.cuda.Event(enable_timing=False)
        event.record(stream);event.synchronize()
        records.append(dict(name=name,stream=token,complete=bool(event.query())))
    if type(consumers) is not int or consumers<=0 or type(auxiliaries) is not int or auxiliaries<0:
        raise RuntimeError('native34-provider-drain-unproved')
    poll_phases(s)
    return dict(decode_consumer_streams=consumers,decode_stream_inventory=consumer_streams,
        owned_prefill_auxiliary_streams=auxiliaries,
        native_streams=records,complete=all(row['complete'] for row in records),
        basis='all-rank FIFO checkpoint after matching prior commands; no new decode issuance')


def cohort_proof(s,*,post_prefix):
    rows=[]
    for row in s['graph_replays']:
        if bool(row['post_prefix'])!=post_prefix:continue
        complete=bool(row['event'].query())
        key=row['key'] or {}
        rows.append(dict(cohort=list(row['cohort']),capture_key=key,
            graph_id=row['graph_id'],decode_step=row.get('decode_step'),complete=complete))
    qualified=[row for row in rows if row['complete'] and row['capture_key'].get('bank')=='decode-split'
               and row['capture_key'].get('num_tokens')==4 and row['capture_key'].get('num_reqs')==4]
    got={tuple(row['cohort']) for row in qualified}
    return dict(verdict='PASS' if COHORTS<=got and not s['limit'] else 'INCONCLUSIVE',
        required_cohorts=[list(row) for row in sorted(COHORTS)],replays=rows,
        post_prefix=post_prefix,scope='graph-tail events plus separately drained downstream consumers')


def qualification(original,args,kwargs):
    runner,command=args
    marker=command.get('native34')
    if marker is None:return original(*args,**kwargs)
    if (command.get('action')!='teacher_evidence' or type(marker) is not dict
            or set(marker)-{'action','arm'} or marker.get('action') not in ('A_START','A_END','ARM')):
        raise ValueError('native34-qualification-marker-invalid')
    s=state(runner)
    plain={key:value for key,value in command.items() if key!='native34'}
    if marker['action']=='A_END':
        # This utility is issued only after the global keep-pause; all matching
        # ordinary PP/TP work is already posted. Retire graph tail events before
        # the real qualification method performs its strict replay census.
        drain_all(runner,s)
    result=original(runner,plain,*args[2:],**kwargs)
    s['epoch']=command['engine_epoch']
    action=marker['action']
    if action=='A_START':
        if s['mode']!='OFF':raise ValueError('native34-control-a-start-state')
        s['mode']='A';s['graph_replays']=[]
        observe(s,'allocation-history-start',lambda:start_history(runner,s))
        proof=dict(action=action,engine_epoch=s['epoch'],verdict='STARTED')
    elif action=='A_END':
        if s['mode']!='A':raise ValueError('native34-control-a-end-state')
        proof=cohort_proof(s,post_prefix=False);proof.update(action=action,engine_epoch=s['epoch'])
        s['control_a']=proof;s['mode']='A_DONE'
        emit(s,'control-a-complete',**proof)
    else:
        if s['mode']!='A_DONE' or s.get('control_a',{}).get('verdict')!='PASS' or marker.get('arm') not in ('S','O','I'):
            raise ValueError('native34-arm-requires-control-a')
        if marker['arm']!=os.environ.get('FLASH_NATIVE34_ARM'):
            raise ValueError('native34-worker-arm-differs-from-frozen-role')
        geometry=slot_contract(runner)
        if not geometry['matches']:raise ValueError('native34-live-eight-slot-contract')
        s['mode']='ARMED';s['arm']=marker['arm'];s['graph_replays']=[]
        install_model(runner)
        proof=dict(action=action,arm=s['arm'],engine_epoch=s['epoch'],control_a=s['control_a'],slot_contract=geometry)
    return dict(result,native34=proof)


def checkpoint(original,args,kwargs):
    worker,command=args;marker=command.get('native34')
    # Qualification markers belong to qualification(), reached through the
    # real native_overlap_worker handler; only census owns CHECKPOINT.
    if marker is None or command.get('action')=='teacher_evidence':
        return original(*args,**kwargs)
    if (command.get('action')!='census' or type(marker) is not dict
            or set(marker)!={'action','arm','phase'} or marker['action']!='CHECKPOINT'
            or marker['phase'] not in ('before-prefix','after-prefix','post-decode')):
        raise ValueError('native34-checkpoint-marker-invalid')
    runner=worker.model_runner;s=state(runner,worker.rank)
    if marker['arm']!=s['arm'] or command['engine_epoch']!=s['epoch']:
        raise ValueError('native34-checkpoint-identity')
    emit(s,'checkpoint-begin',phase=marker['phase'],arm=s['arm'])
    drain=drain_all(runner,s);update_history(runner,s)
    phase_names={p['name'] for p in s['phases'] if p['complete']}
    cohorts=cohort_proof(s,post_prefix=True) if marker['phase']=='post-decode' else None
    proof=dict(phase=marker['phase'],arm=s['arm'],engine_epoch=s['epoch'],
        drain_complete=drain['complete'],drain=drain,
        prefix_complete=PHASES<=phase_names if s['rank'] in (0,1) else None,
        completed_phases=sorted(phase_names),
        both_post_prefix_cohorts_complete=cohorts is not None and cohorts['verdict']=='PASS',
        cohorts=cohorts,metadata_complete=s['metadata_complete'])
    plain={key:value for key,value in command.items() if key!='native34'}
    result=original(worker,plain,*args[2:],**kwargs)
    result=dict(result,qualification=dict(native34=proof))
    emit(s,'checkpoint-complete',**proof)
    return result


def graph_capture_begin(original,args,kwargs):
    result=original(*args,**kwargs)
    TLS.graph=weakref.ref(args[0]);return result


def graph_capture_end(original,args,kwargs):
    try:return original(*args,**kwargs)
    finally:TLS.graph=None


def graph_capture_factory(original,args,kwargs):
    manager=args[0];factory=args[1] if len(args)>1 else kwargs['create_forward_fn']
    def tracked(desc,*a,**k):
        fn=factory(desc,*a,**k)
        @functools.wraps(fn)
        def forward(*fa,**fk):
            ref=getattr(TLS,'graph',None);graph=ref() if ref else None
            runner=RUNNER() if RUNNER else None
            if graph is not None and runner is not None and getattr(desc,'num_tokens',None)==4:
                s=state(runner)
                def record():
                    key={name:int(getattr(desc,name)) if getattr(desc,name,None) is not None else None
                         for name in ('num_tokens','num_reqs','num_ubatches','max_query_len')}
                    key.update(mode=getattr(getattr(desc,'cg_mode',None),'name',None),
                        bank='decode-split' if getattr(manager,'_flash_s5_capture_owner',None) is not None else 'stock-or-profile',
                        manager_id=hex(id(manager)))
                    closure=inspect.getclosurevars(fn).nonlocals
                    named=roots(runner);named.update({n:closure[n] for n in ('model_inputs','attn_metadata','slot_mappings') if n in closure})
                    GRAPHS[graph]=dict(key=key,storage=compact(tensors(named),captured=key))
                observe(s,'capture-storage',record)
            return fn(*fa,**fk)
        return forward
    if len(args)>1:return original(manager,tracked,*args[2:],**kwargs)
    return original(*args,**dict(kwargs,create_forward_fn=tracked))


def graph_replay(original,args,kwargs):
    result=original(*args,**kwargs)
    runner=RUNNER() if RUNNER else None
    if runner is None:return result
    s=state(runner)
    if s['mode'] not in ('A','PREFIX','ARMED') or len(s['graph_replays'])>=32:return result
    import torch
    info=GRAPHS.get(args[0]) or {};event=torch.cuda.Event(enable_timing=False)
    event.record(torch.cuda.current_stream())
    row=dict(event=event,key=info.get('key'),cohort=s['idx'] or (),
             graph_id=hex(id(args[0])),post_prefix=s['prefix_issued'],decode_step=s['step'])
    s['graph_replays'].append(row)
    emit(s,'graph-replay-enqueued',capture_key=row['key'],cohort=list(row['cohort']),
         graph_id=row['graph_id'],post_prefix=row['post_prefix'])
    if info and not info.get('manifest_emitted'):
        info['manifest_emitted']=True
        emit(s,'graph-captured-storage',graph_id=row['graph_id'],manifest=info['storage'])
        def live_manifest():
            update_history(runner,s)
            live=compact(tensors(roots(runner)),window=s['history'],captured=row['key'])
            emit(s,'graph-live-storage',graph_id=row['graph_id'],manifest=live,
                 comparison=compare_manifests(info['storage'],live),
                 scope='named runner roots only; opaque graph read set and external storage remain incomplete')
        observe(s,'graph-live-storage',live_manifest)
    return result


def batch(original,args,kwargs):
    result=original(*args,**kwargs)
    s=state(args[0])
    if s['mode'] in ('A','PREFIX','ARMED') and result[0] is not None:
        def collect():
            values=result[0].idx_mapping_np
            if type(values) not in (tuple,list) and not type(values).__module__.startswith('numpy'):
                raise ValueError('native34-not-cpu-index-map')
            if len(values)>10:raise ValueError('native34-index-map-capacity')
            s['idx']=tuple(int(v) for v in values)
            context=getattr(host._TLS,'context',None) or {}
            s['step']=(context.get('fields') or {}).get('step',0)
        observe(s,'ordered-cohort',collect)
    return result


def submitted(original,args,kwargs):
    worker,message=args[:2]
    result=original(*args,**kwargs)
    s=state(worker.model_runner,worker.rank)
    descriptor=getattr(message,'descriptor',None)
    tx=getattr(descriptor,'transaction',None)
    if (s['mode'] in ('ARMED','PREFIX') and tx is not None and tx.txn_seq==100
            and tx.engine_epoch==s['epoch']
            and tx.chunk_start==0 and tx.chunk_end==1024
            and getattr(getattr(descriptor,'kind',None),'value',None)=='layer'
            and descriptor.stage==0 and descriptor.layer_start==3
            and getattr(message,'profile',None) is None):
        accepted=(getattr(result,'accepted',False) and not getattr(result,'pending',False)) if worker.rank in (0,1) else getattr(result,'skipped',False)
        # A retry can let one peer finish earlier. Only graphs after this
        # latest local acceptance/skip belong to the cut; earlier attempts
        # cannot certify post-prefix decode completion.
        s['prefix_issued']=accepted;s['mode']='PREFIX';s['graph_replays']=[]
        emit(s,'prefix-host-accepted',transaction=100,op_seq=message.op_seq,
             layer_start=3,layer_end=4,active_stage=worker.rank in (0,1),
             accepted=accepted,scope='host return only; device completion separately queried')
    return result


def phase(original,args,kwargs,name):
    ctx=getattr(TLS,'phase_context',None)
    if ctx is None:return original(*args,**kwargs)
    runner,s=ctx['runner'],ctx['state']
    if name=='OUTPUT_ZERO' and id(args[0])!=ctx.get('output_zero_id'):
        return original(*args,**kwargs)
    poll_phases(s)
    emit(s,'phase-begin',phase=name,transaction=100,layer=3,arm=s['arm'])
    def inputs():
        update_history(runner,s)
        import torch
        stream=int(torch.cuda.current_stream().cuda_stream)
        native=runner._flash_s5_native
        manifest=compact(tensors(dict(args=args,kwargs=kwargs)),window=s['history'],
            writer=dict(stream=stream,context=native.split.prefill_context_id),
            reader='same native phase or continuation; exact consumer source catalogue applies',
            peer=dict(provider='native PyNccl TP' if name=='TP_SELECTION' else 'not-assumed',
                      extent='unavailable for external NCCL/IPC allocations'))
        emit(s,'phase-storage',phase=name,position='before',manifest=manifest)
    observe(s,'phase-input-storage',inputs)
    try:
        result=original(*args,**kwargs)
        if name=='ATTENTION_PRE' and type(result) is dict and 'attn_output' in result:
            ctx['output_zero_id']=id(result['attn_output'])
        if len(s['phases'])>=32:
            if not s['limit']:
                localize.emit('native34-receipt-limit',rank=s['rank'],coverage_complete=False,limit='phase-ring')
            s['limit']=True;s['metadata_complete']=False
            return result
        import torch
        stream=torch.cuda.current_stream()
        event=torch.cuda.Event(enable_timing=False);event.record(stream)
        row=dict(name=name,event=event,id=len(s['phases'])+1,stream=int(stream.cuda_stream),complete=False)
        s['phases'].append(row)
        # S is explicitly a serialized diagnostic. Every phase's TP calls are
        # self-contained, both PP0 peers receive the same command, and no new
        # native PP send/receive exists in this prefix. O never waits here.
        if s['arm']=='S':event.synchronize()
        poll_phases(s)
    except BaseException as exc:
        emit(s,'phase-exception',phase=name,error_type=type(exc).__name__,
             device_operation='original phase or its completion event; victim not writer proof')
        raise
    def outputs():
        manifest=compact(tensors(dict(result=result)),window=s['history'],
            writer=dict(stream=row['stream'],context=runner._flash_s5_native.split.prefill_context_id),
            reader='subsequent original native operation',event=row['id'])
        for view in manifest['storages']:
            if 'storage_impl' not in view:continue
            if len(s['providers'])>=128:break
            s['providers'][(view['storage_impl'],view['allocation_base'])]=dict(
                observed_host_provider=name,stream=row['stream'],event_id=row['id'],
                allocation_incarnation=view['allocation_incarnation'],
                last_device_writer='UNPROVEN: returned tensor may be a view or graph output')
        emit(s,'phase-storage',phase=name,position='after',manifest=manifest)
    observe(s,'phase-output-storage',outputs)
    emit(s,'phase-enqueue-end',phase=name,event_id=row['id'],stream=row['stream'],
         synchronized=s['arm']=='S')
    return result


def qsa_pre(original,args,kwargs):
    runner=RUNNER() if RUNNER else None
    s=state(runner) if runner is not None else None
    layer=args[2] if len(args)>2 else kwargs.get('layer_idx')
    if s is None or s['mode'] not in ('ARMED','PREFIX') or layer!=3:
        return original(*args,**kwargs)
    context=getattr(host._TLS,'context',None)
    fields=context.get('fields',{}) if context else {}
    if fields.get('transaction')!=100:return original(*args,**kwargs)
    previous=getattr(TLS,'phase_context',None)
    TLS.phase_context=dict(runner=runner,state=s)
    try:return original(*args,**kwargs)
    finally:TLS.phase_context=previous


def install_model(runner):
    from vllm.v1.flash_e3.model_hooks import local_text_model
    model=local_text_model(runner.model)
    if not model.start_layer<=3<model.end_layer:return
    layer=model.layers[3];attn=layer.self_attn
    owners=[(type(layer),'_flash_e3_layer_pre','LAYER_PRE'),
            (type(attn),'_flash_e3_attention_pre','ATTENTION_PRE'),
            (type(attn.indexer),'forward','INDEXER'),
            (attn.impl if inspect.isclass(attn.impl) else type(attn.impl),'do_kv_cache_update','KV_WRITE')]
    for owner,method,label in owners:
        patch(owner,method,lambda old,a,k,label=label:phase(old,a,k,label))
    # Install shared-method receipts only after original captures and control A.
    # No diagnostic wrapper participates in the ordinary model compilation.
    import torch
    from vllm.v1.flash_e3 import lane_comm,qsa_driver,qsa_operation
    patch(torch.Tensor,'zero_',lambda old,a,k:phase(old,a,k,'OUTPUT_ZERO'))
    patch(qsa_driver.QsaLayerDriver,'_pre',qsa_pre)
    patch(lane_comm,'canonical_selection',lambda old,a,k:phase(old,a,k,'TP_SELECTION'))
    patch(qsa_operation.QsaPrefillOperation,'producer_submitted',lambda old,a,k:phase(old,a,k,'PRODUCER_EVENT'))
    patch(qsa_operation.QsaPrefillOperation,'attach_continuation',lambda old,a,k:phase(old,a,k,'CONTINUATION_ATTACH'))


def load_model(original,args,kwargs):
    global RUNNER
    worker=args[0];runner=getattr(worker,'model_runner',None)
    if runner is not None:RUNNER=weakref.ref(runner);state(runner,worker.rank)
    result=original(*args,**kwargs)
    runner=worker.model_runner;RUNNER=weakref.ref(runner);s=state(runner,worker.rank)
    emit(s,'worker-installed',rank_identity=worker.rank,arm_environment='frozen runner',
         phase_window='txn100 chunk[0,1024) first QSA pre',device_readback=False)
    return result


def instrument(module):
    name=module.__name__
    if name.endswith('.gpu_worker'):
        patch(module.Worker,'load_model',load_model)
        patch(module.Worker,'flash_s5_submit',submitted)
    elif name.endswith('.model_runner'):patch(module.GPUModelRunner,'gather_batch_req_state',batch)
    elif name.endswith('.native_qualification'):patch(module,'handle_worker',qualification)
    elif name.endswith('.native_overlap_worker'):patch(module,'handle',checkpoint)
    elif name.endswith('.cudagraph_utils'):patch(module.CudaGraphManager,'capture',graph_capture_factory)


def indexer_launch(original,args,kwargs):
    jit=args[0];fn=jit.fn
    ctx=getattr(TLS,'phase_context',None)
    if (ctx is None or kwargs.get('warmup')
            or getattr(fn,'__name__',None)!='_qsa_pre_indexer_kernel'):
        return original(*args,**kwargs)
    runner,s=ctx['runner'],ctx['state']
    import torch
    if s['rank']==1 and s['arm'] in ('S','I'):
        emit(s,'indexer-prior-work-drain-begin',arm=s['arm'],kernel='_qsa_pre_indexer_kernel',
             scope='all registered rank1 local consumers; peer remote accesses not certified')
        try:
            drain=drain_all(runner,s)
        except BaseException as exc:
            emit(s,'indexer-prior-work-exception',error_type=type(exc).__name__,
                 target_launched=False,classification='ENGINE',
                 discriminator='prior rank1 work surfaced an error before the indexer launch')
            raise
        emit(s,'indexer-prior-work-drain-complete',drain=drain,target_launched=False)
    def metadata():
        update_history(runner,s)
        names=tuple(jit.arg_names)
        bound=dict(zip(names,args[1:]));bound.update({name:kwargs[name] for name in names if name in kwargs})
        stream=int(torch.cuda.current_stream().cuda_stream)
        rows=operands(bound,window=s['history'],stream=stream,
                      context=runner._flash_s5_native.split.prefill_context_id)
        for view in rows.values():
            view['last_observed_host_provider']=s['providers'].get(
                (view.get('storage_impl'),view.get('allocation_base')),
                dict(observed_host_provider=None,last_device_writer='UNPROVEN: producer predates phase window or external'))
        scalars={name:value for name,value in bound.items() if type(value) in (int,float,bool)}
        emit(s,'indexer-launch-inputs',kernel='_qsa_pre_indexer_kernel',
            inputs=rows,scalars=scalars,missing_arguments=[name for name in names if name not in bound],
            bounds=direct_bounds(bound,rows),grid=list(kwargs.get('grid',())),
            stream=stream,context=runner._flash_s5_native.split.prefill_context_id,
            slot_contract=slot_contract(runner,with_native=True),
            arm=s['arm'],allocation_owner_scope='actual storage metadata plus source-declared owners; last device writer not presumed')
    observe(s,'indexer-inputs',metadata)
    emit(s,'indexer-kernel-begin',kernel='_qsa_pre_indexer_kernel',arm=s['arm'],rank1_priors_drained=s['rank']==1 and s['arm'] in ('S','I'))
    try:
        result=original(*args,**kwargs)
        if s['rank']==1 and s['arm'] in ('S','I'):
            torch.cuda.current_stream().synchronize()
    except BaseException as exc:
        emit(s,'indexer-kernel-exception',kernel='_qsa_pre_indexer_kernel',
            error_type=type(exc).__name__,classification='ENGINE',
            discriminator='first bracketed indexer launch/fence after registered local streams drained' if s['arm'] in ('S','I') and s['rank']==1 else 'unfenced overlapped launch; sticky errors remain possible',
            peer_remote_access_coverage='unproved; no access-PC claim')
        raise
    emit(s,'indexer-kernel-end',kernel='_qsa_pre_indexer_kernel',
         rank1_post_fence=s['rank']==1 and s['arm'] in ('S','I'))
    return result
