#!/usr/bin/env python3
"""CPU-full negative startup-boundary evidence; never a hardware handoff proof."""
import argparse,ast,hashlib,inspect,json,sys,textwrap,time,traceback
from pathlib import Path


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(graph):
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.distributed import init_distributed_environment,initialize_model_parallel,get_tp_group,get_pp_group
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.v1.flash_e3 import lane_comm,native_startup,lane_hooks,graph_hooks,graph_router,s5_startup,engine_driver,worker_bridge
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from e3_native_transitions_lifecycle_probe import config,exception_chain
    cfg=config('cpu')
    result=dict(schema='native15-handoff-startup-boundary/1',verdict='INCONCLUSIVE',rc=2,
        hardware_handoff_proved=False,completed_request_reuse_proved=False,matrix_complete=False,
        box_qualified=False,mocks_used=False,methods_patched=False,flags_assigned=False)
    with set_current_vllm_config(cfg):
        init_distributed_environment(world_size=1,rank=0,local_rank=0,
            distributed_init_method='file:///tmp/handoff-boundary-dist',backend='gloo')
        initialize_model_parallel(1,1,backend='gloo')
        tp,pp=get_tp_group(),get_pp_group()
        nccl=PyNcclCommunicator(group=tp.cpu_group,device=torch.device('cuda:0'))
        assert nccl.world_size==1 and nccl.disabled is True and nccl.available is False
        try:lane_comm.PrefillCommunicators.build(tp,pp,torch.device('cuda:0'),None,
                max_tokens=1024,hidden_width=1024,dtype=torch.bfloat16)
        except lane_comm.LaneCommError as exc:
            assert str(exc)=='prefill TP communicator needs an enabled PyNccl communicator'
            comm_error=exception_chain(exc)
        else:raise AssertionError('default real TP1 startup unexpectedly accepted; investigate')
        try:GPUModelRunner(cfg,torch.device('cpu'))
        except Exception as exc:
            partial=None;tb=exc.__traceback__
            while tb is not None:
                obj=tb.tb_frame.f_locals.get('self')
                if type(obj) is GPUModelRunner:partial=obj
                tb=tb.tb_next
            assert partial is not None and not hasattr(partial,'_flash_s5_split')
        else:raise AssertionError('CUDA runner completed on CPU; extend execution')
        try:native_startup.install_worker_native(partial,chunk_tokens=1024,torch_module=torch)
        except native_startup.NativeStartupError as exc:
            assert str(exc)=='worker-native startup needs the open green split from worker init'
            split_error=exception_chain(exc)
        else:raise AssertionError('ordinary partial runner accepted as native green startup')
    names=('native_startup','lane_comm','lane_hooks','graph_hooks','graph_router','s5_startup','engine_driver','worker_bridge')
    modules={name:sys.modules['vllm.v1.flash_e3.'+name] for name in names}
    pins={row['module']:row for row in graph['modules'] if row['active_mount']}
    source=[]
    for name,module in modules.items():
        path=Path(inspect.getsourcefile(module));pin=pins[module.__name__]
        assert path.resolve()==Path(pin['destination']).resolve() and digest(path)==pin['sha256']
        source.append(dict(module=module.__name__,file=str(path),sha256=digest(path)))
    # Exact source identity plus checked constructor ordering; not simulated ACKs.
    fn=native_startup.install_worker_native
    lines,start=inspect.getsourcelines(fn);text=textwrap.dedent(''.join(lines));tree=ast.parse(text)
    marker=next(node for node in ast.walk(tree) if isinstance(node,ast.Assign) and any(
        isinstance(t,ast.Attribute) and t.attr=='_flash_s5_native_decode' for t in node.targets))
    assert isinstance(marker.value,ast.Constant) and marker.value.value is True
    comm=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=='PrefillCommunicators.build')
    assert comm.lineno<marker.lineno
    assignments=[]
    for row in graph['modules']:
        if not row['active_mount'] or not row['module'].startswith('vllm.v1.flash_e3.'):continue
        path=Path(row['destination']);body=ast.parse(path.read_text())
        for node in ast.walk(body):
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='_flash_s5_native_decode' for t in node.targets):
                assignments.append(dict(module=row['module'],line=node.lineno,value=ast.unparse(node.value)))
    assert assignments==[dict(module=native_startup.__name__,line=start+marker.lineno-1,value='True')],assignments
    functions=(lane_hooks.execute_with_e3,lane_hooks.LaneRuntime.execute_native_decode,
        graph_router.GraphRouter.native_decode_route,graph_hooks.replay_fullgraph,
        s5_startup.install_builder,lane_hooks.init_lane,worker_bridge.NativeWorkerBridge.transition,
        engine_driver.NativeSegmentDriver._apply_release,engine_driver.NativeSegmentDriver._drive_route)
    function_index=[]
    for fn in functions:
        src,first=inspect.getsourcelines(fn)
        function_index.append(dict(function=fn.__qualname__,module=fn.__module__,line=first,
            source_sha256=hashlib.sha256(''.join(src).encode()).hexdigest()))
    assert not torch.cuda.is_initialized()
    result.update(verdict='PASS',rc=0,default_tp1=dict(world_size=nccl.world_size,
        communicator_disabled=nccl.disabled,communicator_available=nccl.available,
        actual_prefill_startup_refusal=comm_error),green_split_required=split_error,
        marker_assignments=assignments,communicator_build_before_marker=True,
        source_pins=source,production_function_index=function_index,cuda_initialized=False,
        scope='actual CPU startup refusals and source-bound production routing audit; no native decode route constructed or bypassed')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--graph',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=('cpu-full','cpu-shape'),default='cpu-full');a=p.parse_args();started=time.monotonic()
    try:record=check(json.loads(a.graph.read_text()))
    except Exception as exc:
        record=dict(schema='native15-handoff-startup-boundary/1',verdict='FAIL',rc=1,failure_class='OPEN',
            error_type=type(exc).__name__,error_message=str(exc),stack=traceback.format_exc(),hardware_handoff_proved=False)
    record['failure_class']=record.get('failure_class','NONE');record['wall_s']=time.monotonic()-started
    record['probe_sha256']=digest(__file__);record['graph_sha256']=digest(a.graph)
    with a.out.open('x') as f:json.dump(record,f,indent=2)
    print(json.dumps({k:record[k] for k in ('verdict','rc','failure_class','wall_s','hardware_handoff_proved')}),flush=True)
    return record['rc']


if __name__=='__main__':raise SystemExit(main())
