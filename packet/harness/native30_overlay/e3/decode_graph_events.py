"""Timing nodes in original graphs only; no model replay or tensor allocation."""
from contextlib import contextmanager
from functools import wraps
import json
import logging
import math
import os

PROBE='FLASH_S5_GRAPH_EVENTS_PROBE'
ENV='FLASH_S5_GRAPH_EVENTS'
KEY='_flash_s5_ge'
PREFIX='flash-s5 graph-events '
POINTS=52
ROWS=4


def enabled(env=None):
    env=os.environ if env is None else env
    for key in (PROBE,ENV):
        if env.get(key,'0') not in ('0','1'):raise ValueError('invalid-boolean:'+key)
    if env.get(ENV,'0')=='1' and env.get(PROBE,'0')!='1':raise ValueError('graph-events-needs-probe')
    if env.get(PROBE,'0')=='1' and any(env.get(k,'0')!='1' for k in ('FLASH_S5_OVERLAP','FLASH_S5_AUTO_ARM','FLASH_S5_HANDLER_ANATOMY')):
        raise ValueError('graph-events-needs-owned-native-and-anatomy')
    if env.get(PROBE,'0')=='1' and (env.get('FLASH_S5_DECODE_LAYERS','0')!='0' or env.get('FLASH_S5_REPLAY_OBSERVER','0')!='0'):
        raise ValueError('graph-events-refuses-extra-captures-or-observer')
    return env.get(PROBE,'0')=='1'


def emit(row):
    from .native_measurement import emit_receipt_lines
    line=PREFIX+json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False)
    if len(line.encode())>3400:raise ValueError('graph-events-receipt-too-large')
    ok=emit_receipt_lines((line,),logging.getLogger(__name__))
    if ok is False and not getattr(emit,'warned',False):
        emit.warned=True
        logging.getLogger(__name__).warning('graph-events receipt capacity exhausted; evidence is incomplete')
    return ok


def target(desc):
    from .graph_router import ExecutionDescriptor
    d=ExecutionDescriptor.from_vllm(desc)
    return (d.cg_mode=='full' and d.num_tokens==ROWS and d.num_reqs==ROWS
            and (d.uniform_token_count==1 or d.max_query_len==1)
            and d.num_active_loras==0 and d.num_ubatches==1)


class Events:
    def __init__(self,manager,model,torch,rank,device,on,writer=emit):
        from .model_hooks import local_text_model
        self.manager,self.model,self.torch=manager,local_text_model(model),torch
        self.rank,self.device,self.on,self.writer=rank,device,on,writer
        self.bank='split' if getattr(manager,'_flash_s5_capture_owner',None) is not None else 'full'
        split=getattr(manager,'_flash_s5_capture_owner',None)
        self.sms=split.decode_sms if split is not None else int(torch.cuda.get_device_properties(device).multi_processor_count)
        self.start,self.end=self.model.start_layer,self.model.end_layer
        if type(rank) is not int or rank not in range(4) or (self.start,self.end)!=(rank//2*24,rank//2*24+24):
            raise ValueError('graph-events-model-rank-census')
        self.events=[];self.completion=None;self.descriptor=None;self.stream=None
        self.tag=None;self.pending=None;self.generation=0;self.captures=0;self.closed=False
        self.counts=dict(selected=0,read=0,stale_reuse=0,incomplete=0,replays=0,receipt_dropped=0)

    def report(self,kind,**fields):
        ok=self.writer(dict(schema='decode-graph-events/1',kind=kind,rank=self.rank,stage=self.rank//2,
                         bank=self.bank,on=self.on,rows=ROWS,bank_sms=self.sms,**fields))
        if ok is False:self.counts['receipt_dropped']+=1

    @contextmanager
    def capture(self,desc,stream=None):
        if not target(desc):yield;return
        if self.captures:raise ValueError('graph-events-duplicate-original-capture')
        self.captures+=1;self.descriptor=desc
        if not self.on:yield;return
        torch=self.torch;self.stream=stream if stream is not None else torch.cuda.current_stream()
        # Use the exact stream argument of the original torch.cuda.graph, not its caller's stream.
        with torch.cuda.stream(self.stream):
            self.events=[torch.cuda.Event(enable_timing=True,external=True) for _ in range(POINTS)]
            self.completion=torch.cuda.Event(enable_timing=False)
            for event in self.events+[self.completion]:event.record(self.stream)
            self.stream.synchronize()
        touched=[];order=[]
        def point(index):
            if int(torch.cuda.current_stream().cuda_stream)!=int(self.stream.cuda_stream):
                raise ValueError('graph-events-foreign-capture-stream')
            self.events[index].record(self.stream);order.append(index)
        def install(obj,lo,hi,root=False):
            original=obj.forward;own='forward' in vars(obj)
            @wraps(original)
            def measured(*args,**kwargs):
                point(lo)
                if root:point(1);point(2)
                result=original(*args,**kwargs)
                point(hi)
                return result
            touched.append((obj,original,own));obj.forward=measured
        try:
            install(self.model,0,51,True)
            for offset,index in enumerate(range(self.start,self.end)):
                install(self.model.layers[index],3+2*offset,4+2*offset)
            yield
            if order!=list(range(POINTS)):raise ValueError('graph-events-capture-order-or-count')
        finally:
            for obj,original,own in reversed(touched):
                if own:obj.forward=original
                else:delattr(obj,'forward')

    def install_replay(self):
        if self.captures!=1:raise ValueError('graph-events-original-four-row-capture-missing')
        original=self.manager.run_fullgraph
        def replay(desc):
            self.poll()
            selected=target(desc)
            if selected:
                if self.pending is not None:
                    self.counts['stale_reuse']+=1;self.pending=None
                self.generation+=1;self.counts['replays']+=1
            result=original(desc)
            if selected and self.on and self.tag is not None:
                if self.tag.get('rows')!=ROWS:raise ValueError('graph-events-engine-grant-width')
                if int(self.torch.cuda.current_stream().cuda_stream)!=int(self.stream.cuda_stream):
                    raise ValueError('graph-events-foreign-replay-stream')
                self.completion.record(self.stream)
                self.pending=dict(tag=dict(self.tag),generation=self.generation)
                self.counts['selected']+=1
            return result
        self.manager.run_fullgraph=replay

    def poll(self):
        p=self.pending
        if p is None:return
        if p['generation']!=self.generation:
            self.counts['stale_reuse']+=1;self.pending=None;return
        # An outside-graph completion record proves this execution really passed every node.
        if not self.completion.query() or any(not event.query() for event in self.events):return
        times=[float(self.events[0].elapsed_time(e)) for e in self.events]
        if any(not math.isfinite(x) or x<0 for x in times):raise ValueError('graph-events-invalid-device-time')
        self.counts['read']+=1;self.pending=None
        layers=[times[4+2*i]-times[3+2*i] for i in range(24)]
        if any(x<0 for x in layers):raise ValueError('graph-events-layer-time-regressed')
        self.report('sample',execute_seq=p['tag']['seq'],metadata=p['tag'],generation=self.generation,
                    layer_start=self.start,layer_ms=layers,backbone_ms=times[51],marker_pair_ms=times[2]-times[1],
                    counts=dict(self.counts))


@contextmanager
def model_capture(manager,model):
    if not enabled():yield;return
    if getattr(manager,'_max_full_descs_to_capture',None) is not None:
        yield;return  # Ephemeral graph-memory profiling is not an original runtime bank.
    if hasattr(manager,'_flash_s5_graph_events'):raise ValueError('graph-events-capture-repeated')
    import torch
    from vllm.distributed.parallel_state import get_world_group
    from .startup_vram_budget import sample
    rank=get_world_group().rank;device=manager.device.index
    before=sample(torch,device)
    owner=Events(manager,model,torch,rank,device,os.environ.get(ENV,'0')=='1')
    manager._flash_s5_graph_events=owner
    yield
    owner.install_replay()
    after=sample(torch,device)
    # No per-process incremental-fit claim: the paired reader subtracts the normal off capture.
    owner.report('capture',before=before,after=after,events=len(owner.events),completion_events=int(owner.completion is not None),
                 original_captures=owner.captures,added_graph_executables=0,added_graph_replays=0)


@contextmanager
def graph_capture(manager,desc,stream=None):
    owner=getattr(manager,'_flash_s5_graph_events',None)
    if owner is None:yield;return
    with owner.capture(desc,stream):yield


def owners(runner):
    managers=[getattr(runner,'cudagraph_manager',None),getattr(getattr(runner,'_flash_s5_split_bank',None),'manager',None)]
    return [m._flash_s5_graph_events for m in managers if getattr(m,'_flash_s5_graph_events',None) is not None]


def worker_begin(proc,method,kwargs):
    tag=kwargs.pop(KEY,None)
    if tag is not None:
        fields={'seq','engine_epoch','route_generation','engine_send_ns','rows','uid_digest','history_tokens','native'}
        if (method!='execute_model' or type(tag) is not dict or set(tag)!=fields
                or any(type(tag[k]) is not int or tag[k]<0 for k in ('seq','engine_epoch','route_generation','engine_send_ns','rows'))
                or tag['seq']<1 or tag['rows']!=ROWS
                or type(tag['uid_digest']) is not str or len(tag['uid_digest'])!=64
                or any(c not in '0123456789abcdef' for c in tag['uid_digest'])
                or type(tag['history_tokens']) is not list or len(tag['history_tokens'])!=ROWS
                or any(type(n) is not int or n<0 for n in tag['history_tokens'])):
            raise ValueError('graph-events-invalid-engine-grant')
    runner=getattr(getattr(proc,'worker',None),'model_runner',None)
    for owner in owners(runner):
        owner.poll();owner.tag=tag if method=='execute_model' else None
        if owner.tag is not None:
            bridge=getattr(runner,'_flash_s5_bridge',None);route=getattr(runner,'_flash_s5_route',None)
            if (type(tag) is not dict or tag.get('rows')!=ROWS or tag.get('engine_epoch')!=getattr(bridge,'engine_epoch',None)
                    or tag.get('route_generation')!=getattr(route,'route_generation',None)):
                raise ValueError('graph-events-stale-engine-grant')
    return tag['seq'] if tag is not None else None


def worker_returned(proc):
    runner=getattr(getattr(proc,'worker',None),'model_runner',None)
    for owner in owners(runner):owner.poll();owner.tag=None


def engine_install(engine):
    from decode_layer_runtime import EngineSampler
    if hasattr(engine.model_executor,'_flash_s5_graph_events'):raise ValueError('graph-events-engine-installed-twice')
    class Sampler(EngineSampler):
        def wire(self,method,sequence,args):
            from decode_layer_runtime import _metadata
            if method=='execute_model' and args:
                requests=getattr(getattr(self.engine,'scheduler',None),'requests',{})
                ids=getattr(args[0],'num_scheduled_tokens',{})
                for uid in ids:
                    request=requests.get(uid)
                    computed=getattr(request,'num_computed_tokens',None)
                    prompt=getattr(request,'num_prompt_tokens',None)
                    if (type(computed) is not int or type(prompt) is not int or computed<prompt
                            or getattr(request,'is_prefill_chunk',True)):
                        return None
                metadata=_metadata(args[0])
                if metadata is not None and metadata['history_tokens'] is None:
                    return None
            return super().wire(method,sequence,args)
    engine.model_executor._flash_s5_graph_events=Sampler(engine)


def shutdown(runner):
    for owner in owners(runner):
        if owner.closed:continue
        owner.poll()
        if owner.pending is not None:
            owner.counts['incomplete']+=1;owner.pending=None
        owner.report('coverage',reason='shutdown',counts=dict(owner.counts))
        owner.closed=True
