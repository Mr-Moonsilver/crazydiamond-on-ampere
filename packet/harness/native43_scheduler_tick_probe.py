#!/usr/bin/env python3
"""Pinned-image real AsyncScheduler ticks through the Core bypass seam.

The checkpoint and pending worker-stop handshake are declared leaves: the
separate native43 four-bridge test proves the real release/cancel/FULL chain.
This probe keeps the real scheduler, KV allocator, controller/admission and
Core hooks, so a None result really schedules and reserves source history.
"""
import argparse
import json
import types
import traceback
from pathlib import Path


def run():
    import torch
    import e3_native37_admission_probe as base
    base.mounted()  # Same generated-scheduler import/namespace order as native39 probes.
    from vllm.v1.flash_e3 import executor_hooks as hooks
    stack=base.Stack()
    driver=stack.driver
    # Empty owned reply channel: no submitted work or device callbacks here.
    driver.executor._flash_s5_transport=types.SimpleNamespace(poll=lambda budget:None,_calls={})
    calls=0
    def core_tick():
        nonlocal calls
        result=hooks.engine_idle_tick(stack.engine)
        if result is None:
            result=hooks.before_step_with_batch_queue(stack.engine)
        if result is None:
            calls+=1
            return stack.scheduler.schedule()
        return result
    # One REAL empty scheduler tick between the driver advances.
    core_tick()
    assert calls==1
    stack.arm('split-prefill-solo')
    request=stack.request('native43-pending-next',197270)
    # Project the already-proved committed checkpoint/retained-worker state.
    # Request token frontier is fresh in this CPU allocation-only test, so an
    # erroneous next schedule allocates the first 1024-row native frame. It is
    # an admission hazard probe, not simulated 101200-token model execution.
    frame=stack.controller._next_transaction(request,100224,101200)
    generation=frame.request_generation
    driver._retained={'transaction':types.SimpleNamespace(request_uid=request.request_id,
                        request_generation=generation)}
    driver._native43=dict(stop_at=101200,checkpoint=dict(chunk_end=101200,exact=True),cancel_started=False)
    stops=[]
    def stop(kind):
        stops.append(kind);driver.stop_kind=kind
    driver.stop=stop
    driver._drive_stop=lambda:None  # Worker completion remains pending for 3 ticks.
    before=stack.controller.admitted
    for _ in range(3):core_tick()
    result=dict(schema='native43-real-scheduler-cutoff/1',
        verdict='PASS' if calls==1 and stack.controller.admitted==before else 'FAIL',
        real_scheduler=type(stack.scheduler).__module__+'.'+type(stack.scheduler).__name__,
        baseline_scheduler_ticks=1,unexpected_scheduler_ticks=calls-1,
        admitted_after_cutoff=stack.controller.admitted-before,
        source_leases_clean=stack.leases.shutdown()['clean'],stop_calls=stops,
        cuda_initialized=torch.cuda.is_initialized(),geometry=stack.geometry)
    assert stops==['cancel']
    assert not result['cuda_initialized']
    return result


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args(argv)
    try:result=run()
    except Exception as error:
        result=dict(verdict='FAIL',failure_class='HARNESS',error_type=type(error).__name__,cause=str(error)[:240],traceback=traceback.format_exc()[-4000:])
    with a.out.open('x') as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps(result,sort_keys=True))
    return 0 if result['verdict']=='PASS' else 1

if __name__=='__main__':raise SystemExit(main())
