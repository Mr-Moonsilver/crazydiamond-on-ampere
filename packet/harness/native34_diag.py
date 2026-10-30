"""Opt-in native34 causal interventions over unchanged native32 engine bytes."""
import functools
import os
import sys

import native30_localize as localize
import native30_policy
import native34_worker as worker
from native34_control import Arm,ProtocolRefused

INSTALLED=False
TARGETS={
 'vllm.v1.flash_e3.engine_driver','vllm.v1.flash_e3.executor_hooks',
 'vllm.v1.flash_e3.native_overlap_worker','vllm.v1.flash_e3.native_qualification',
 'vllm.v1.worker.gpu_worker','vllm.v1.worker.gpu.model_runner',
 'vllm.v1.worker.gpu.cudagraph_utils',
}


def armed(driver):return getattr(driver,'_native34_arm',None)


def settling(original,args,kwargs):
    driver,control,reply=args
    marker=((control.get('command') or {}).get('payload') or {}).get('native34')
    if marker and marker.get('action')=='ARM':
        if marker.get('arm')!=os.environ.get('FLASH_NATIVE34_ARM'):
            raise ProtocolRefused('native34-arm-differs-from-frozen-role')
        rows=driver._validate_worker_rows(reply,'qualification')
        for row in rows:
            proof=(row['qualification'] or {}).get('native34') or {}
            if (proof.get('arm')!=marker['arm'] or proof.get('engine_epoch')!=driver.engine_epoch
                    or proof.get('control_a',{}).get('verdict')!='PASS'):
                raise ProtocolRefused('native34-all-rank-control-a-unproved')
        driver._native34_arm=Arm(marker['arm'],driver.engine_epoch,driver.ranks)
    return original(*args,**kwargs)


def cleanup(driver):
    return (driver.stop_kind is not None or driver.control_pending()
            or driver._engine_paused())


def advance(original,args,kwargs):
    driver=args[0];arm=armed(driver)
    if arm is None or cleanup(driver):return original(*args,**kwargs)
    if arm.stage=='ARMED' and driver.controller.active:
        arm.admission(driver.controller.admission())
    needed=arm.needed_checkpoint()
    if needed and not driver._inflight:
        # One atomic all-rank FIFO broadcast. Existing transport owns every
        # response. No competing MQ reader, executor RPC or host wait is added.
        marker=dict(action='CHECKPOINT',arm=arm.name,phase=needed)
        command=dict(action='census',engine_epoch=driver.engine_epoch,
                     route_generation=driver.route_generation,native34=marker)
        seq=driver._submit('native_overlap',('flash_s5_native_overlap',(command,),{},None),driver.ranks)
        arm.submitted(seq,needed)
    return original(*args,**kwargs)


def drive(original,args,kwargs):
    driver=args[0];arm=armed(driver)
    if arm is not None and not cleanup(driver) and arm.blocks_native():return None
    return original(*args,**kwargs)


def segments(original,args,kwargs):
    driver,values=args;command=driver._command
    result=original(*args,**kwargs)
    arm=armed(driver)
    if arm is not None and not cleanup(driver):arm.accepted_prefix(command,values)
    return result


def checkpoint_reply(original,args,kwargs):
    driver,values,sequence=args
    arm=armed(driver)
    if arm is None or arm.sequence!=sequence:return original(*args,**kwargs)
    rows=driver._validate_worker_rows(values,'census')
    arm.completed(sequence,rows)
    localize.emit('native34-engine-checkpoint',arm=arm.name,stage=arm.stage,
                  engine_epoch=driver.engine_epoch,sequence=sequence)
    return None


def before_stock(original,args,kwargs):
    engine=args[0];result=original(*args,**kwargs)
    driver=getattr(engine,'_flash_s5_driver',None)
    arm=armed(driver) if driver is not None else None
    if result is not None or arm is None or cleanup(driver):return result
    if arm.blocks_stock():return ({},False)
    # Count an allowed scheduling tick. Even the second tick is allowed to
    # execute after it changes the NEXT tick's state to POST_DRAIN. The actual
    # worker graph events, not this count, certify both cohorts.
    arm.ordinary_tick()
    return result


def status(original,args,kwargs):
    result=original(*args,**kwargs);arm=armed(args[0])
    if arm is not None:result=dict(result,native34=arm.record())
    return result


def instrument(module):
    worker.instrument(module)
    if module.__name__.endswith('.engine_driver'):
        cls=module.NativeSegmentDriver
        for name,callback in (('_settle_worker_control',settling),('advance',advance),
            ('_drive',drive),('_apply_segments',segments),('_apply_native_overlap',checkpoint_reply),
            ('_status_record',status)):
            worker.patch(cls,name,callback)
    elif module.__name__.endswith('.executor_hooks'):
        worker.patch(module,'before_step_with_batch_queue',before_stock)


def install():
    global INSTALLED
    if INSTALLED or os.environ.get('FLASH_NATIVE34_DIAG')!='1':return
    if os.environ.get('FLASH_NATIVE34_ARM') not in ('S','O','I'):
        raise ValueError('native34-frozen-arm-required')
    if os.environ.get('FLASH_NATIVE33_DIAG')=='1' or os.environ.get('FLASH_NATIVE30_SYNC','none')!='none' or os.environ.get('CUDA_LAUNCH_BLOCKING')=='1':
        raise ValueError('native34-unregistered-synchronization-refused')
    native30_policy.install()
    import torch
    from triton.runtime.jit import JITFunction
    worker.patch(torch.cuda.CUDAGraph,'capture_begin',worker.graph_capture_begin)
    worker.patch(torch.cuda.CUDAGraph,'capture_end',worker.graph_capture_end)
    worker.patch(torch.cuda.CUDAGraph,'replay',worker.graph_replay)
    worker.patch(JITFunction,'run',worker.indexer_launch)
    previous=localize.instrument
    @functools.wraps(previous)
    def combined(module):previous(module);instrument(module)
    localize.instrument=combined;localize._TARGETS.update(TARGETS)
    for name in sorted(TARGETS):
        if name in sys.modules:instrument(sys.modules[name])
    INSTALLED=True
    localize.emit('native34-installed',arm=os.environ['FLASH_NATIVE34_ARM'],
        ordinary_capture='shared model-method hooks deferred until after control A',
        interventions='S serialized prefix; O original overlap; I rank1 pre-indexer local-stream drain',
        storage_values_read=False)
