"""GPU-only diagnostics. No engine control. Separate fresh process per case."""
import ctypes
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import traceback
import torch
import torch.distributed as dist
import s5_segment_gpu as base
from s5_clock import build_command, DeviceClock, Inconclusive
from s5_runtime import Progress, Watchdog
from s5_segment import plan, write_json
from s5_onepair import audit_grade


def build(path,copy_kernels=False):
    source=Path(__file__).with_name('s5_graph_copy.cu' if copy_kernels else 's5_graph_audit.cu')
    if path.exists():raise Inconclusive('new helper path required')
    command=build_command(source,path,torch.cuda.get_device_capability())
    p=subprocess.run(command,capture_output=True,text=True,timeout=180)
    evidence=dict(command=command,rc=p.returncode,log=(p.stdout+p.stderr)[-6000:])
    if p.returncode:raise Inconclusive('graph audit helper build failed',evidence)
    evidence['sha256']=base.digest(path)
    return evidence


def audit(graph,stream,library,path):
    lib=ctypes.CDLL(str(library));lib.s5_audit.argtypes=[ctypes.c_uint64,ctypes.c_uint64,ctypes.c_char_p]
    lib.s5_audit.restype=ctypes.c_int
    rc=lib.s5_audit(graph.raw_cuda_graph(),stream.cuda_stream,str(path).encode())
    if rc:raise Inconclusive('node audit API failed',dict(cuda_rc=rc))
    rows=[json.loads(s) for s in path.read_text().splitlines()]
    return dict(audit_grade(rows),nodes=rows)


def capture(lane,stream):
    lane.reset();torch.cuda.synchronize()
    with torch.cuda.stream(stream):lane.forward(base.Trace())
    stream.synchronize();lane.reset();torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph(keep_graph=True);trace=base.Trace()
    with lane.reduce.capture():
        with torch.cuda.graph(graph,stream=stream):lane.forward(trace)
    graph.instantiate();torch.cuda.synchronize()
    return base.DecodeGraph(graph,lane,lane.last_output),trace


def wait(event):
    while not event.query():time.sleep(.0001)


def timed(fn,stream,progress,lane,step):
    progress.event(lane,'step-enter',step=step)
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):a.record();fn();b.record()
    progress.event(lane,'step-issued',step=step)
    wait(b)
    progress.event(lane,'step-complete',step=step)
    return a.elapsed_time(b)


def collectives(rd,rp,sd,sp,progress,device,library,output,rank):
    inputs={k:torch.full((n,2560),float(rank+1),device='cuda',dtype=torch.bfloat16)
            for k,n in [('decode',4),('prefill',1024)]}
    outputs={};torch.cuda.synchronize()
    # Warm both before capture, then capture decode only (9 reductions/slice).
    for lane,reducer,stream in [('decode',rd,sd),('prefill',rp,sp)]:
        with torch.cuda.stream(stream):outputs[lane]=reducer(inputs[lane])
        stream.synchronize()
    graph=torch.cuda.CUDAGraph(keep_graph=True)
    with rd.capture():
        with torch.cuda.graph(graph,stream=sd):
            for _ in range(9):outputs['decode']=rd(inputs['decode'])
    graph.instantiate();torch.cuda.synchronize()
    node=audit(graph,sd,library,Path(str(output)+f'.rank{rank}.collective-nodes.jsonl'))
    barrier=threading.Barrier(2);rows={'decode':[],'prefill':[]};errors=[]
    def lane(name,reducer,stream):
        try:
            torch.cuda.set_device(device)
            for step in range(30):
                reducer.step=step
                barrier.wait(timeout=120)
                def execute():
                    if name=='decode':graph.replay()
                    else:
                        for _ in range(9):outputs[name]=reducer(inputs[name])
                rows[name].append(timed(execute,stream,progress,name,step))
        except BaseException as e:errors.append(e);barrier.abort()
    threads=[threading.Thread(target=lane,args=x,daemon=True) for x in [('decode',rd,sd),('prefill',rp,sp)]]
    for t in threads:t.start()
    for t in threads:t.join()
    if errors:raise errors[0]
    torch.cuda.synchronize()
    for value in outputs.values():base.checked(torch.equal(value,torch.full_like(value,3.)),'collectives-only output mismatch')
    graph.reset()
    return rows,node


def matrix(green,library,output,rank):
    """Positive/negative control: capture context, not launch-stream name."""
    primary=torch.cuda.Stream();streams={'primary':primary,'decode':green.sd,'prefill':green.sp}
    memberships,complete=green.membership()
    if not complete:raise Inconclusive('incomplete SM membership sampling')
    sets={'primary':set.union(*memberships),'decode':memberships[0],'prefill':memberships[1]}
    records=[]
    for source,stream in streams.items():
        outs=[torch.full((green.nsm*32,),-1,device='cuda',dtype=torch.int32) for _ in range(2)]
        torch.cuda.synchronize();graph=torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph,stream=stream):
            for out in outs:green.probe(out,stream,100000)
        graph.instantiate()
        node=audit(graph,stream,library,Path(str(output)+f'.rank{rank}.matrix-{source}.jsonl'))
        for dest,launch in streams.items():
            for out in outs:out.fill_(-1)
            torch.cuda.synchronize()
            with torch.cuda.stream(launch):graph.replay()
            launch.synchronize()
            observed=[set(out.cpu().tolist()) for out in outs]
            ok=all(s and -1 not in s and s<=sets[source] for s in observed)
            # Primary negative control must demonstrate escape from decode's set.
            control=source!='primary' or dest!='decode' or any(s-sets['decode'] for s in observed)
            records.append(dict(capture=source,replay=dest,observed=[sorted(s) for s in observed],
                                expected=sorted(sets[source]),contained=ok,negative_control=bool(control),audit=node['verdict']))
        graph.reset()
    return dict(records=records,pass_sampled_nodes=all(r['contained'] and r['negative_control'] for r in records),
                scope='both probe kernel nodes sampled; production node contexts audited separately')


def run(args):
    rank=int(os.environ['RANK']);device=int(os.environ['LOCAL_RANK'])
    progress=Progress(args.out,rank);wd=Watchdog(progress,args.out,rank)
    result=dict(rank=rank,case=args.case,context=args.context,
                max_connections=os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS','unset(default8)'),
                nccl_max_channels=os.environ.get('NCCL_MAX_NCHANNELS','unset'),durations_ms={})
    green=None;graph=None
    def phase(name,seconds=None):wd.arm(args.phase_timeout if seconds is None else seconds,name)
    try:
        phase('setup',300)
        if os.environ.get('WORLD_SIZE')!='2' or os.environ.get('CUDA_VISIBLE_DEVICES')!='1,3' or os.environ.get('CUDA_DEVICE_ORDER')!='PCI_BUS_ID':
            raise Inconclusive('requires isolated pair1,3 with two ranks in PCI order')
        torch.cuda.set_device(device);dist.init_process_group('gloo')
        ctl=dist.new_group([0,1],backend='gloo');dg=dist.new_group([0,1],backend='gloo');pg=dist.new_group([0,1],backend='gloo')
        library=args.out.parent/'audit.so';compiled=[None]
        if rank==0:compiled[0]=build(library)
        dist.broadcast_object_list(compiled,src=0);result['build']=compiled[0]
        clock=DeviceClock(library);result['pci_bus_id']=clock.pci_bus_id(device)
        manifest=json.loads(Path(__file__).with_name('s5_source_manifest.json').read_text())
        config_bytes=args.config.read_bytes()
        if hashlib.sha256(config_bytes).hexdigest()!=manifest['model_config_sha256']:raise Inconclusive('config identity mismatch')
        result['source_sha256']={m:base.digest(importlib.import_module(m).__file__) for m in manifest['modules']}
        if result['source_sha256']!=manifest['modules']:raise Inconclusive('source identity mismatch')
        result['overlay_ops_sha256']=args.overlay_sha
        if args.context=='green24':
            green=base.Green(library,24,device);sd,sp=green.sd,green.sp
            result['sm_split']=green.split
        else:sd,sp=torch.cuda.Stream(priority=-1),torch.cuda.Stream()
        phase('reducers')
        rd=base.Reducer(dg,device,'decode',progress,backend=args.decode_backend)
        rp=base.Reducer(pg,device,'prefill',progress,backend='pynccl');base.refault()
        if args.case=='collectives':
            phase('collectives-only-overlap')
            result['durations_ms'],result['audit']=collectives(rd,rp,sd,sp,progress,device,library,args.out,rank)
        else:
            phase('model-slice-setup',300);torch.manual_seed(871)
            spec=plan(json.loads(config_bytes),layers=4,decode_rows=4)
            weights=base.weights(spec,rank)
            want_pre=args.case in ('prefill-alone','prio-overlap')
            dec=base.Slice(spec,rank,rd,False,weights,'packed') if args.case!='prefill-alone' else None
            pre=base.Slice(spec,rank,rp,True,weights,'packed') if want_pre else None
            if args.case=='prefill-alone':
                phase('prefill-alone-green-warmup',300)
                pre.reset();torch.cuda.synchronize()
                timed(lambda:pre.forward(base.Trace()),sp,progress,'prefill','warmup')
                phase('prefill-alone-green-measured')
                for step in range(30):
                    rp.step=step
                    result['durations_ms'].setdefault('prefill',[]).append(timed(lambda:pre.forward(base.Trace()),sp,progress,'prefill',step))
                base.checked(all(bool(torch.isfinite(t).all()) for t in base.snapshot(pre)),'prefill solo nonfinite')
            else:
                phase('decode-capture',300)
                mem_before=dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),free=torch.cuda.mem_get_info()[0])
                graph,trace=capture(dec,sd)
                result['graph_memory']=dict(before=mem_before,after=dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),free=torch.cuda.mem_get_info()[0]))
                result['audit']=audit(graph.graph,sd,library,Path(str(args.out)+f'.rank{rank}.nodes.jsonl'))
                phase('decode-alone-equality')
                dec.reset();torch.cuda.synchronize()
                with torch.cuda.stream(sd):dec.forward(base.Trace())
                sd.synchronize();reference=base.snapshot(dec)
                dec.reset();torch.cuda.synchronize()
                with torch.cuda.stream(sd):graph.replay()
                sd.synchronize();base.checked(base.state_equal(base.snapshot(dec),reference),'decode graph differs from solo eager')
                if args.case=='node-audit':
                    phase('capture-replay-matrix');result['matrix']=matrix(green,library,args.out,rank)
                    if not result['matrix']['pass_sampled_nodes']:result['audit']['verdict']='INCONCLUSIVE'
                elif args.case=='prio-overlap':
                    phase('clock-validation');clock.evidence=clock.validate()
                    phase('prio-overlap')
                    value=base.run_mode('prio',dec,pre,graph,trace,sd,sp,ctl,30,.0095*4/24,clock)
                    result['overlap']=value
                    result['durations_ms']={k:[r['cuda_event_ms'] for r in value['raw_'+k]][:30] for k in ('decode','prefill')}
                    # This diagnostic's prefill tail has variable number of tickets.
                    result['durations_ms'].pop('prefill')
                else:
                    phase('decode-alone-green-measured')
                    for step in range(30):result['durations_ms'].setdefault('decode',[]).append(timed(graph.replay,sd,progress,'decode',step))
        phase('cleanup')
        torch.cuda.synchronize()
        if graph:graph.reset()
        if green:green.close()
        result['completed']=True;all_ranks=[None,None];dist.all_gather_object(all_ranks,result)
        if rank==0:write_json(args.out,dict(case=args.case,pair='1,3',ranks=all_ranks))
        dist.destroy_process_group();progress.update('completed');wd.close();return 0
    except BaseException as e:
        record=dict(fatal_correctness=isinstance(e,base.CorrectnessError),error=type(e).__name__+': '+str(e),
                    partial_rank=result,error_evidence=getattr(e,'evidence',{}),traceback=traceback.format_exc())
        target=Path(str(args.out)+f'.rank{rank}.error.json')
        if not target.exists():write_json(target,record)
        wd.close();return 1 if record['fatal_correctness'] else 2
