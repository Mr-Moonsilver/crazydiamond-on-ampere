"""Startup-only physical/allocator budget. Sharing tensor owners does not make graphs free."""
import hashlib
import json
import logging
import time
from contextlib import contextmanager
from functools import wraps

PREFIX = 'flash-s5 startup-vram '
SCHEMA = 'startup-peer-vram/1'
MIB = 2**20
# Owner's native-phase targets minus 2 MiB observed warmup growth; startup limits stay below them.
DECODE_LAYER_BUDGET = {
    str(rank): dict(total_delta_bytes=64*MIB, outside_delta_bytes=40*MIB,
                   outside_absolute_bytes=limit*MIB)
    for rank,limit in enumerate((1259,1371,1197,1309))
}


class VramBudgetRefused(RuntimeError):pass


def budget_id(budget):
    if set(budget)!=set(map(str,range(4))):raise ValueError('vram-budget-needs-four-ranks')
    fields={'total_delta_bytes','outside_delta_bytes','outside_absolute_bytes'}
    if any(set(row)!=fields or any(type(v) is not int or v<0 for v in row.values()) for row in budget.values()):
        raise ValueError('vram-budget-invalid-limit')
    return hashlib.sha256(json.dumps(budget,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def sample(torch,device):
    # Startup only: explicit synchronization is part of this receipt's measurement, never a forward hook.
    torch.accelerator.synchronize()
    free,total=torch.accelerator.get_memory_info()
    return dict(device_index=device,total_bytes=int(total),free_bytes=int(free),
                allocated_bytes=int(torch.cuda.memory_allocated(device)),
                reserved_bytes=int(torch.cuda.memory_reserved(device)),observed_ns=time.monotonic_ns())


def facts(snapshot):
    keys={'device_index','total_bytes','free_bytes','allocated_bytes','reserved_bytes','observed_ns'}
    if type(snapshot) is not dict or set(snapshot)!=keys or any(type(v) is not int or v<0 for v in snapshot.values()):
        raise ValueError('unknown-outside-cache:invalid-snapshot')
    t,f,a,r=(snapshot[k] for k in ('total_bytes','free_bytes','allocated_bytes','reserved_bytes'))
    if not 0<=a<=r<=t or not 0<=f<=t or t-f<r:raise ValueError('unknown-outside-cache:inconsistent-snapshot')
    return dict(device_used_bytes=t-f,outside_cache_bytes=t-f-r)


def assess(before,after,limits):
    try:
        b,a=facts(before),facts(after)
        if before['device_index']!=after['device_index'] or before['total_bytes']!=after['total_bytes'] or before['observed_ns']>after['observed_ns']:
            raise ValueError('unknown-outside-cache:measurement-identity')
        delta=dict(total_device_bytes=a['device_used_bytes']-b['device_used_bytes'],
                   allocated_bytes=after['allocated_bytes']-before['allocated_bytes'],
                   reserved_bytes=after['reserved_bytes']-before['reserved_bytes'],
                   outside_cache_bytes=a['outside_cache_bytes']-b['outside_cache_bytes'])
        violations=[]
        if delta['total_device_bytes']>limits['total_delta_bytes']:violations.append('total-device-delta')
        if delta['outside_cache_bytes']>limits['outside_delta_bytes']:violations.append('outside-cache-delta')
        if a['outside_cache_bytes']>limits['outside_absolute_bytes']:violations.append('outside-cache-absolute')
        return dict(status='FAIL' if violations else 'PASS',violations=violations,delta=delta,
                    before_facts=b,after_facts=a)
    except (ValueError,KeyError,TypeError) as exc:
        return dict(status='HOLD',reason=str(exc),error_type=type(exc).__name__)


def emit(row):
    from native_measurement import emit_receipt_lines
    line=PREFIX+json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False)
    if len(line.encode())>3500:raise VramBudgetRefused('startup-vram-receipt-oversize')
    if emit_receipt_lines((line,),logging.getLogger(__name__)) is False:
        raise VramBudgetRefused('startup-vram-receipt-write-failed')


class StartupBudget:
    def __init__(self,torch,device,rank,*,feature,budget=DECODE_LAYER_BUDGET,writer=emit,sampler=None):
        self.id=budget_id(budget)
        if type(rank) is not int or rank not in range(4):raise VramBudgetRefused('startup-vram-rank-invalid')
        self.torch,self.device,self.rank=torch,device,rank
        self.writer,self.sampler=writer,sample if sampler is None else sampler
        self.common=dict(schema=SCHEMA,feature=feature,rank=rank,stage=rank//2,
                         budget_id=self.id,limits=dict(budget[str(rank)]),scope='synchronized-startup-only; net footprint, not peak')
        self.before=None

    def begin(self):
        if self.before is not None:raise VramBudgetRefused('startup-vram-begin-twice')
        try:
            self.before=self.sampler(self.torch,self.device);facts(self.before)
        except Exception as exc:
            self.writer(dict(self.common,phase='before',status='HOLD',reason=str(exc)[:512],error_type=type(exc).__name__))
            raise VramBudgetRefused('startup-vram-baseline-unknown') from exc
        self.writer(dict(self.common,phase='before',status='BEFORE',snapshot=self.before))

    def finish(self):
        if self.before is None:raise VramBudgetRefused('startup-vram-finish-without-begin')
        try:after=self.sampler(self.torch,self.device)
        except Exception as exc:
            self.writer(dict(self.common,phase='after',status='HOLD',reason=str(exc)[:512],error_type=type(exc).__name__))
            raise VramBudgetRefused('startup-vram-final-unknown') from exc
        result=assess(self.before,after,self.common['limits'])
        self.writer(dict(self.common,phase='after',before=self.before,after=after,**result))
        if result['status']!='PASS':raise VramBudgetRefused('startup-vram-budget-'+result['status'].lower()+':'+','.join(result.get('violations',[])))
        return result


@contextmanager
def startup_phase(runner, name, *, torch_module=None, enabled=None, writer=None):
    """A local-device synchronized startup bracket; never a forward/capture-node hook."""
    from lane_private import headroom_enabled
    if not (headroom_enabled() if enabled is None else enabled):
        yield
        return
    if torch_module is None:
        import torch as torch_module
    if torch_module.cuda.is_current_stream_capturing():
        raise VramBudgetRefused('headroom receipt cannot run inside CUDA capture')
    parallel = runner.vllm_config.parallel_config
    rank = parallel.rank
    if type(rank) is not int or rank not in range(4):
        raise VramBudgetRefused('headroom receipt needs exact global rank')
    writer = emit if writer is None else writer
    device = runner.device if type(runner.device) is int else runner.device.index
    common = dict(schema='g4-startup-phase/1', feature='g4-headroom', rank=rank,
                  stage=rank//2, label=name, scope='synchronized-startup-net; no allocation attribution')
    before = sample(torch_module, device)
    facts(before)
    writer(dict(common, phase='before', snapshot=before))
    try:
        yield
    except BaseException as exc:
        writer(dict(common, phase='error', error_type=type(exc).__name__, reason=str(exc)[:256]))
        raise
    else:
        after = sample(torch_module, device)
        a, b = facts(after), facts(before)
        writer(dict(common, phase='after', snapshot=after, delta=dict(
            device_used_bytes=a['device_used_bytes']-b['device_used_bytes'],
            outside_cache_bytes=a['outside_cache_bytes']-b['outside_cache_bytes'],
            allocated_bytes=after['allocated_bytes']-before['allocated_bytes'],
            reserved_bytes=after['reserved_bytes']-before['reserved_bytes'])))


def wrap_headroom_capture(runner, tp_group, torch):
    """Retain each original bound method; bracket only startup capture/IPC registration."""
    manager = runner.cudagraph_manager
    if getattr(manager, '_flash_s5_headroom_wrapped', False):
        raise VramBudgetRefused('headroom capture wrapper installed twice')
    original = manager.capture
    @wraps(original)
    def capture(*args, **kwargs):
        with startup_phase(runner, 'full-capture', torch_module=torch, enabled=True):
            return original(*args, **kwargs)
    manager.capture = capture
    manager._flash_s5_headroom_wrapped = True
    ca = getattr(getattr(tp_group, 'device_communicator', None), 'ca_comm', None)
    if ca is None or getattr(ca, 'disabled', False):
        with startup_phase(runner, 'custom-ar-disabled', torch_module=torch, enabled=True):
            pass
        return
    register = ca.register_graph_buffers
    @wraps(register)
    def registration(*args, **kwargs):
        with startup_phase(runner, 'custom-ar-registration', torch_module=torch, enabled=True):
            return register(*args, **kwargs)
    ca.register_graph_buffers = registration
