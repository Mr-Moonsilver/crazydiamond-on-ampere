"""Opt-in native30 diagnostic plugin; the mounted engine source stays unchanged."""
import functools
import contextlib
import itertools
import importlib.abc
import importlib.machinery
import json
import os
import sys
import threading
import time

PREFIX='NATIVE30 '
_LOCK=threading.Lock()
_INSTALLED=False
_CALL_IDS=itertools.count(1)
_TARGETS={'vllm.v1.flash_e3.lane_dispatch','vllm.v1.flash_e3.worker_bridge','vllm.v1.worker.gpu_worker'}


def emit(event,**fields):
    row=dict(schema='native30-localize/1',event=event,pid=os.getpid(),monotonic_ns=time.monotonic_ns(),**fields)
    # One locked write, flushed before launch. Never log argv, environment or exception text.
    with _LOCK:
        sys.stderr.write(PREFIX+json.dumps(row,sort_keys=True,separators=(',',':'))+'\n')
        sys.stderr.flush()


def mode():
    value=os.environ.get('FLASH_NATIVE30_SYNC','both')
    if value not in ('none','native','decode','both'):raise ValueError('invalid native30 sync mode')
    return value


def rank(obj):
    value=getattr(obj,'rank',None)
    if value is None:value=getattr(getattr(getattr(obj,'native',None),'state',None),'worker_rank',None)
    if type(value) is not int or value not in (0,1,2,3):raise RuntimeError('native30 worker rank unavailable')
    return value


def memory(torch,device):
    sample=dict(reserved_bytes=int(torch.cuda.memory_reserved(device)))
    try:
        free,total=torch.cuda.mem_get_info(device)
        sample.update(free_bytes=int(free),total_bytes=int(total))
    except BaseException as exc:
        sample.update(free_bytes=None,total_bytes=None,error_type=type(exc).__name__)
        return sample,exc
    return sample,None


def sync(obj,side,fields):
    if mode() not in (side,'both'):return
    import torch
    driver=getattr(obj,'dispatcher',obj)
    device=getattr(getattr(driver,'runner',None),'device',None) or getattr(getattr(obj,'model_runner',None),'device',None)
    if side=='decode':
        from vllm.v1.flash_e3.worker_hooks import decode_scope
        scope=decode_scope(obj)
    else:
        scope=driver._scope() if driver is not obj else contextlib.nullcontext()
    with scope:
        emit('sync-begin',side=side,**fields)
        sample,error=memory(torch,device);obj._native30_memory=sample
        emit('barrier-memory',side=side,position='before',memory=sample,**fields)
        if error is not None:raise error
        torch.cuda.synchronize(device)
        sample,error=memory(torch,device);obj._native30_memory=sample
        emit('barrier-memory',side=side,position='after',memory=sample,**fields)
        if error is not None:raise error
        emit('sync-end',side=side,**fields)


def wrap(cls,name,kind):
    original=getattr(cls,name)
    if getattr(original,'_native30_wrapped',False):return
    @functools.wraps(original)
    def observed(self,*args,**kwargs):
        fields=dict(rank=rank(self),label=kind,call_id=next(_CALL_IDS))
        side=None
        if kind=='FIRST_ADMISSION':
            side='native'
            record=args[0].admission
            txn=record.transaction
            fields.update(chunk_start=txn.chunk_start,chunk_end=txn.chunk_end,transaction=txn.txn_seq,first_admission=txn.chunk_start==0)
        elif kind.startswith('DECODE'):
            if name=='execute_model':self._native30_step=getattr(self,'_native30_step',0)+1
            fields.update(step=getattr(self,'_native30_step',0),phase=name)
            side='decode'
        else:
            command=args[0]
            fields.update(op_seq=command.op_seq,layer_start=command.layer_start,layer_end=command.layer_end,
                chunk_start=command.transaction.chunk_start,chunk_end=command.transaction.chunk_end,
                transaction=command.transaction.txn_seq)
            # PP send/receive may wait for a peer operation not yet submitted.
            # Label them, but never add a device-wide barrier at that boundary.
            if kind not in ('PP_SEND','PP_RECEIVE'):side='native'
        emit('begin',**fields)
        try:
            result=original(self,*args,**kwargs)
            if side:sync(self,side,fields)
        except BaseException as exc:
            pending=type(exc).__name__=='E3Pending'
            emit('pending' if pending else 'exception',classification=None if pending else 'ENGINE',error_type=type(exc).__name__,last_memory=getattr(self,'_native30_memory',None),**fields)
            raise
        emit('end',last_memory=getattr(self,'_native30_memory',None),**fields)
        return result
    observed._native30_wrapped=True
    setattr(cls,name,observed)


def instrument(module):
    name=module.__name__
    if name.endswith('.lane_dispatch'):
        for n in ('prepare','prefix','layer','suffix','state_finalize','pp_send','pp_receive'):
            wrap(module.LaneDispatcher,'_do_'+n,n.upper())
    elif name.endswith('.worker_bridge'):wrap(module.NativeWorkerBridge,'open','FIRST_ADMISSION')
    elif name.endswith('.gpu_worker'):
        wrap(module.Worker,'execute_model','DECODE_STEP')
        wrap(module.Worker,'sample_tokens','DECODE_SAMPLE')
    emit('module-installed',module=name,sync_mode=mode())


class Loader(importlib.abc.Loader):
    def __init__(self,delegate):self.delegate=delegate
    def create_module(self,spec):
        fn=getattr(self.delegate,'create_module',None)
        return fn(spec) if fn else None
    def exec_module(self,module):
        self.delegate.exec_module(module)
        instrument(module)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname not in _TARGETS:return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path,target)
        if spec is None or spec.loader is None:raise ImportError('native30 target module missing: '+fullname)
        spec.loader=Loader(spec.loader)
        return spec


def install():
    global _INSTALLED
    if _INSTALLED:return
    if os.environ.get('FLASH_NATIVE30_LOCALIZE')!='1':return
    mode()
    sys.meta_path.insert(0,Finder())
    for name in sorted(_TARGETS):
        if name in sys.modules:instrument(sys.modules[name])
    _INSTALLED=True
    emit('plugin-installed',sync_mode=mode(),cuda_launch_blocking=os.environ.get('CUDA_LAUNCH_BLOCKING','unset'))
