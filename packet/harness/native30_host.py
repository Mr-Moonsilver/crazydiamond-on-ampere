"""Bounded passive host receipts. Never dereference a device tensor or allocate a CUDA object."""
import contextlib
import functools
import hashlib
import json
import threading
import weakref

MAX_POOLS=512
MAX_ELEMENTS=262144
MAX_RECORD_BYTES=128*1024
MAX_BYTES_PER_RANK=16*1024*1024
MAX_RECORDS_PER_RANK=20000
_TLS=threading.local()
_POOLS=weakref.WeakKeyDictionary()
_RUNNERS=weakref.WeakKeyDictionary()
_EMIT=None


def safe(fn,*args,**kwargs):
    try:return fn(*args,**kwargs)
    except Exception:return None  # Diagnostics must not replace an engine result/exception.


def state(runner,rank=None):
    if runner is None:return None
    value=_RUNNERS.get(runner)
    if value is None:
        value=dict(rank=rank,active=False,step=0,native_index=None,admission=None,
                   pools=[],last_step_counts={},bytes=0,records=0,limit=False,
                   last_idx=None,native_idx=None,table_writers={})
        _RUNNERS[runner]=value
    if rank is not None:value['rank']=rank
    return value


def pool_state(pool,constructed=False):
    value=_POOLS.get(pool)
    if value is None:
        value=dict(rotations=0,attempts=0,constructor_observed=constructed,events={})
        _POOLS[pool]=value
    return value


def ints(values):
    # Accept only explicit CPU arrays/lists; no tensor.tolist(), item(), cpu() or numpy().
    if type(values) not in (list,tuple):
        if not type(values).__module__.startswith('numpy') or not hasattr(values,'shape'):
            return dict(available=False,reason='not-host-array')
        if len(values.shape)!=1:return dict(available=False,shape=list(values.shape))
    n=len(values)
    if n>MAX_ELEMENTS:return dict(available=False,length=n,reason='host-array-bound')
    if n==0:return dict(available=True,length=0,min=None,max=None)
    return dict(available=True,length=n,min=int(min(values)),max=int(max(values)))


def shape(value):return [int(x) for x in value.shape]


def discover(runner,s):
    found=[];seen=set();nodes=0
    def visit(obj,path,depth):
        nonlocal nodes
        if id(obj) in seen:return
        seen.add(id(obj));nodes+=1
        if nodes>4096 or len(found)>=MAX_POOLS:return
        cls=type(obj)
        if cls.__name__=='UvaBufferPool' and cls.__module__=='vllm.v1.worker.gpu.buffer_utils':
            found.append((path,weakref.ref(obj)));pool_state(obj);return
        if depth<=0:return
        if type(obj) in (list,tuple):
            for i,x in enumerate(obj[:256]):visit(x,f'{path}[{i}]',depth-1)
        elif type(obj) is dict:
            for name,x in list(obj.items())[:256]:visit(x,f'{path}.{name}',depth-1)
        elif cls.__module__.startswith('vllm.v1.worker.gpu'):
            for name,x in sorted(vars(obj).items()):
                if hasattr(x,'__dict__') or type(x) in (list,tuple,dict):
                    visit(x,path+'.'+name,depth-1)
    for name in ('req_states','block_tables','model_state','sampler'):
        obj=getattr(runner,name,None)
        if obj is not None:visit(obj,name,6)
    s['pools']=found
    s['inventory_complete']=nodes<=4096 and len(found)<MAX_POOLS
    tables=getattr(getattr(runner,'block_tables',None),'block_tables',())
    s['table_writers']={id(t):i for i,t in enumerate(tables[:256])}


def table_summary(runner):
    tables=getattr(runner,'block_tables',None)
    if tables is None:return dict(available=False)
    counts=getattr(getattr(tables,'num_blocks',None),'np',None)
    rows=[]
    for i,t in enumerate(tables.block_tables[:256]):
        dims=shape(t.gpu);bpk=int(tables.blocks_per_kv_block[i])
        row=dict(group=i,table_shape=dims,request_row_capacity=dims[0],block_capacity=dims[1],
                 blocks_per_kv_block=bpk)
        if counts is not None:
            vals=ints(counts[i]);row['num_blocks']=vals
            row['count_exceeds_row_capacity']=bool(vals.get('max') is not None and (vals['max']>dims[1] or vals['min']<0))
            row['num_blocks_rows_match']=bool(counts.shape[1]==dims[0])
        rows.append(row)
    return dict(available=True,num_blocks_shape=shape(counts) if counts is not None else None,
                groups=rows,kv_num_blocks=int(runner.kv_cache_config.num_blocks)
                if getattr(runner,'kv_cache_config',None) is not None else None)


def snapshot(runner,s):
    discover(runner,s)
    pools=[]
    for name,ref in s['pools']:
        pool=ref()
        if pool is None:continue
        p=pool_state(pool);previous=s['last_step_counts'].get(name)
        depth=int(pool.max_concurrency);delta=p['rotations']-previous if previous is not None else None
        pools.append(dict(name=name,slot=int(pool._curr),depth=depth,rotations=p['rotations'],copy_attempts=p['attempts'],
                          rotations_since_decode_begin=delta,possible_step_wrap=(delta>=depth) if delta is not None else None,
                          constructor_observed=p['constructor_observed']))
    return dict(idx_mapping_np=s['last_idx'],native_idx_mapping_np=s['native_idx'],max_num_reqs=int(runner.req_states.max_num_reqs),
                native_index_violation=(not 0<=s['native_index']<runner.req_states.max_num_reqs) if s['native_index'] is not None else None,
                native_request_index=s['native_index'],admission_plan=s['admission'],
                tables=table_summary(runner),pools=pools,
                pool_inventory_complete=s['inventory_complete'],reader_coverage='coverage-incomplete')


def emit(s,event,**fields):
    if s is None or _EMIT is None or s['limit']:return
    record=dict(rank=s['rank'],event=event,decode_step_window=s['step'],**fields)
    encoded=json.dumps(record,sort_keys=True,separators=(',',':'))
    charge=len(encoded.encode('utf8'))+512  # Includes the emitter's fixed metadata overhead.
    if (charge>MAX_RECORD_BYTES or s['bytes']+charge>MAX_BYTES_PER_RANK
            or s['records']>=MAX_RECORDS_PER_RANK):
        if not s['limit']:
            s['limit']=True
            _EMIT('host-receipt-limit',rank=s['rank'],record_bytes=len(encoded),
                  bytes_emitted=s['bytes'],records_emitted=s['records'],coverage_complete=False)
        return
    s['bytes']+=charge;s['records']+=1
    _EMIT(event,**{k:v for k,v in record.items() if k!='event'})


@contextlib.contextmanager
def boundary(obj,fields):
    runner=getattr(obj,'model_runner',None) or getattr(obj,'runner',None)
    prior=getattr(_TLS,'context',None)
    s=safe(state,runner,fields.get('rank'))
    context=dict(runner=runner,state=s,fields=fields,dispatcher=obj if hasattr(obj,'events') else None,
                 descriptor=False,uses=[])
    _TLS.context=context
    if s is not None and s['active']:
        if fields['label']=='DECODE_STEP':
            s['step']=fields.get('step',0)
            s['last_idx']=None
            safe(discover,runner,s)
            s['last_step_counts']={name:pool_state(ref())['rotations'] for name,ref in s['pools'] if ref() is not None}
        safe(emit,s,'host-boundary',phase='before',**fields_without_rank(fields),snapshot=safe(snapshot,runner,s))
    try:yield
    finally:
        if s is not None and s['active']:
            safe(emit,s,'host-boundary',phase='after',**fields_without_rank(fields),snapshot=safe(snapshot,runner,s))
        dispatcher=context['dispatcher']
        if dispatcher is not None and context['uses']:
            uses=getattr(dispatcher,'_native30_descriptor_uses',{})
            if len(uses)<32:
                uses[fields.get('op_seq')]=context['uses']
                dispatcher._native30_descriptor_uses=uses
        _TLS.context=prior


def fields_without_rank(fields):return {k:v for k,v in fields.items() if k!='rank'}


def plan(admission):
    # Fixed CPU admission fields only; never repr()/serialize arbitrary engine objects.
    tx=admission.transaction
    value=dict(transaction={k:getattr(tx,k) for k in ('engine_epoch','request_uid','request_generation','txn_seq','chunk_start','chunk_end')},
               **{k:getattr(admission,k) for k in ('prompt_end','input_ids','ngram_context','block_tables','new_block_ids_to_zero','cow_pairs','state_epoch')})
    total=sum(len(row) for row in admission.block_tables)+len(admission.input_ids)+len(admission.ngram_context)
    if total>MAX_ELEMENTS:return dict(available=False,reason='admission-size-bound')
    payload=json.dumps(value,sort_keys=True,separators=(',',':')).encode()
    return dict(available=True,sha256=hashlib.sha256(payload).hexdigest(),
                chunk_start=tx.chunk_start,chunk_end=tx.chunk_end,transaction=tx.txn_seq,
                scope='canonical ChunkAdmission CPU fields; no raw request/token/block payload')


def attach_event(dispatcher,op_seq):
    uses=getattr(dispatcher,'_native30_descriptor_uses',{}).pop(op_seq,())
    event=next((r.event for r in reversed(dispatcher.events) if r.op_seq==op_seq),None)
    if event is None:return
    ref=weakref.ref(event)  # Do not extend a device event's resource lifetime.
    for pool_ref,slot in uses:
        pool=pool_ref()
        if pool is not None:pool_state(pool)['events'][slot]=(ref,op_seq)


def patch_method(owner,name,callback):
    original=getattr(owner,name)
    if getattr(original,'_native30_host_wrapped',False):return
    @functools.wraps(original)
    def wrapped(self,*args,**kwargs):return callback(original,self,args,kwargs)
    wrapped._native30_host_wrapped=True
    setattr(owner,name,wrapped)


def pool_init(original,pool,args,kwargs):
    result=original(pool,*args,**kwargs);safe(pool_state,pool,True);return result


def before_pool_copy(pool):
    p=pool_state(pool);ctx=getattr(_TLS,'context',None)
    p['attempts']+=1
    target=(pool._curr+1)%pool.max_concurrency
    previous=p['events'].get(target);pending=None;event_seq=None
    if previous is not None:
        ref,event_seq=previous;event=ref()
        if event is not None:pending=safe(lambda:not bool(event.query()))
    if ctx and ctx['state'] and ctx['state']['active']:
        s=ctx['state'];name=next((name for name,ref in s['pools'] if ref() is pool),None)
        emit(s,'host-pool-rotation',name=name,target_slot_predicted=target,current_slot_before=int(pool._curr),depth=pool.max_concurrency,
             prior_rotations=p['rotations'],previous_native_descriptor_event=event_seq,
             previous_native_descriptor_pending=pending,reader_coverage='coverage-incomplete',slot_observation='predicted-before-call; concurrent writers may change selection',
             **fields_without_rank(ctx['fields']))
    return p,ctx,target


def after_pool_copy(pool,trace,result):
    if trace is None:return
    p,ctx,target=trace;p['rotations']+=1
    # Tensor pointer metadata only; no data readback, and no addresses are emitted.
    actual=next((i for i,b in enumerate(pool._uva_bufs) if b.uva.data_ptr()==result.data_ptr()),None)
    if actual is not None:target=actual
    if actual is not None and ctx and ctx['descriptor'] and len(ctx['uses'])<MAX_POOLS:
        ctx['uses'].append((weakref.ref(pool),target))


def pool_copy(original,pool,args,kwargs):
    trace=safe(before_pool_copy,pool)
    result=original(pool,*args,**kwargs)
    safe(after_pool_copy,pool,trace,result)
    return result


def observe_writer(writer,args,kwargs,ctx):
    s=ctx['state'];runner=ctx['runner']
    if s is None or not s['active']:return
    tensors=(args[0] if args else kwargs['tensors']) if type(writer).__name__=='FusedStagedWriter' else (writer,)
    staged=[]
    for tensor in tensors[:256]:
        group=s['table_writers'].get(id(tensor));row=dict(group=group,
            indices=ints(tensor._staged_write_indices),starts=ints(tensor._staged_write_starts),
            cu_lens=ints(tensor._staged_write_cu_lens),contents_length=len(tensor._staged_write_contents))
        if group is not None:
            values=ints(tensor._staged_write_contents)
            blocks=int(runner.kv_cache_config.num_blocks) if getattr(runner,'kv_cache_config',None) else None
            expansion=int(runner.block_tables.blocks_per_kv_block[group])
            bound=blocks*expansion if blocks is not None else None
            row.update(expanded_block_ids=values,kv_num_blocks=blocks,expanded_block_bound=bound,
                       block_id_violation=bool(bound is not None and values.get('max') is not None and (values['max']>=bound or values['min']<0)))
        staged.append(row)
    emit(s,'host-staged-writes',writes=staged,**fields_without_rank(ctx['fields']))


def writer_apply(original,writer,args,kwargs):
    ctx=getattr(_TLS,'context',None)
    prior=ctx['descriptor'] if ctx else False
    if ctx:
        ctx['descriptor']=True
        safe(observe_writer,writer,args,kwargs,ctx)
    try:return original(writer,*args,**kwargs)
    finally:
        if ctx:ctx['descriptor']=prior


def worker_after(original,worker,args,kwargs):
    result=original(worker,*args,**kwargs)
    runner=getattr(worker,'model_runner',None)
    s=safe(lambda:state(runner,worker.rank))
    if s is not None:
        safe(emit,s,'host-startup-baseline',snapshot=safe(snapshot,runner,s))
    return result


def batch_after(original,runner,args,kwargs):
    result=original(runner,*args,**kwargs)
    def observe():
        s=state(runner)
        if s['active'] and result[0] is not None:
            s['last_idx']=ints(result[0].idx_mapping_np)
            x=s['last_idx'];limit=int(runner.req_states.max_num_reqs)
            s['last_idx']['decode_step']=s['step']
            s['last_idx']['bound_violation']=bool(x.get('max') is not None and (x['min']<0 or x['max']>=limit or x['length']>limit))
            emit(s,'host-decode-batch',step=s['step'],snapshot=snapshot(runner,s))
    safe(observe)
    return result


def native_admit(original,factory,args,kwargs):
    def observe():
        s=state(factory.runner);s['admission']=plan(args[0]);discover(factory.runner,s)
    safe(observe)
    result=original(factory,*args,**kwargs)
    safe(lambda:state(factory.runner).update(native_idx=ints(result.batch.idx_mapping_np)))
    return result


def request_added(original,states,args,kwargs):
    result=original(states,*args,**kwargs)
    ctx=getattr(_TLS,'context',None)
    def observe():
        if ctx and ctx['state'] and ctx['fields']['label'] not in ('DECODE_STEP','DECODE_SAMPLE'):
            req_id=kwargs.get('req_id',args[0] if args else None)
            ctx['state']['native_index']=states.req_id_to_index.get(req_id)
    safe(observe)
    return result


def submit_after(original,dispatcher,args,kwargs):
    result=original(dispatcher,*args,**kwargs)
    safe(lambda:attach_event(dispatcher,(args[0] if args else kwargs['command']).op_seq))
    return result


def qualification(original,runner,command):
    result=original(runner,command)
    if command.get('action')=='teacher_configure':
        def observe():
            s=state(runner);s['active']=command['enabled']
            if s['active']:
                discover(runner,s)
                s['last_step_counts']={name:pool_state(ref())['rotations'] for name,ref in s['pools'] if ref() is not None}
                emit(s,'host-placement-baseline',snapshot=snapshot(runner,s))
        safe(observe)
    return result


HOOKS={
 'vllm.v1.worker.gpu.buffer_utils':{'UvaBufferPool':('__init__','copy_to_uva'),'StagedWriteTensor':('apply_write',),'FusedStagedWriter':('apply',)},
 'vllm.v1.worker.gpu.model_runner':{'GPUModelRunner':('gather_batch_req_state',)},
 'vllm.v1.worker.gpu.states':{'RequestState':('add_request',)},
 'vllm.v1.flash_e3.lane_chunk':{'LaneChunkFactory':('admit',)},
}


def install(module,emit_function):
    global _EMIT
    _EMIT=emit_function;name=module.__name__
    if name.endswith('.buffer_utils'):
        for cls,method,callback in ((module.UvaBufferPool,'__init__',pool_init),(module.UvaBufferPool,'copy_to_uva',pool_copy),
                                    (module.StagedWriteTensor,'apply_write',writer_apply),(module.FusedStagedWriter,'apply',writer_apply)):
            patch_method(cls,method,callback)
    elif name.endswith('.model_runner'):patch_method(module.GPUModelRunner,'gather_batch_req_state',batch_after)
    elif name.endswith('.states'):patch_method(module.RequestState,'add_request',request_added)
    elif name.endswith('.lane_chunk'):patch_method(module.LaneChunkFactory,'admit',native_admit)
    elif name.endswith('.lane_dispatch'):patch_method(module.LaneDispatcher,'submit',submit_after)
    elif name.endswith('.gpu_worker'):
        patch_method(module.Worker,'init_device',worker_after);patch_method(module.Worker,'load_model',worker_after)
    elif name.endswith('.native_qualification'):
        original=module.handle_worker
        if not getattr(original,'_native30_host_wrapped',False):
            @functools.wraps(original)
            def observed(runner,command):return qualification(original,runner,command)
            observed._native30_host_wrapped=True;module.handle_worker=observed
