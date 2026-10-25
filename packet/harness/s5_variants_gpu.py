"""Unarmed representative-slice frontier and graph-bank probes, not an engine."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
import s5_segment_gpu as b
from s5_onepair_gpu import build,audit,capture,wait,timed
from s5_segment import plan,write_json
from s5_clock import DeviceClock,Inconclusive
from s5_runtime import Progress,Watchdog


def memory():
    return dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),free=torch.cuda.mem_get_info()[0])


def run(a):
    rank=int(os.environ['RANK']);device=int(os.environ['LOCAL_RANK']);output=a.out
    progress=Progress(output,rank);wd=Watchdog(progress,output,rank)
    result=dict(rank=rank,rows=a.rows,requested_split=a.split,modes={});contexts=[];graphs=[]
    def phase(name,timeout=180):wd.arm(timeout,name)
    try:
        phase('setup',300)
        if os.environ.get('CUDA_VISIBLE_DEVICES')!=a.pair or os.environ.get('WORLD_SIZE')!='2':raise Inconclusive('pair mismatch')
        torch.cuda.set_device(device);dist.init_process_group('gloo')
        ctl=dist.new_group([0,1],backend='gloo');dg=dist.new_group([0,1],backend='gloo');pg=dist.new_group([0,1],backend='gloo')
        library=output.parent/'audit.so';built=[None]
        if rank==0:built[0]=build(library)
        dist.broadcast_object_list(built,src=0);result['build']=built[0]
        manifest=json.loads(Path(__file__).with_name('s5_source_manifest.json').read_text())
        result['sources']={m:b.digest(importlib.import_module(m).__file__) for m in manifest['modules']}
        if result['sources']!=manifest['modules'] or b.digest(a.config)!=manifest['model_config_sha256']:raise Inconclusive('source/config mismatch')
        clock=DeviceClock(library);result['pci_bus_id']=clock.pci_bus_id(device)
        if a.rung!='model-terminal':clock.validate()
        spec=plan(json.loads(a.config.read_text()),layers=4,decode_rows=4)
        # Diagnostic extension of the same shapes to actual cohort loads1..5.
        spec.update(decode_rows=a.rows,streams=2*a.rows,gdn_state=[a.rows,24,128,128])
        result['spec']=spec;result['max_connections']=os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS','unset')
        torch.manual_seed(871);phase('weights-and-lanes',300)
        backend=os.environ.get('S5_DECODE_BACKEND','custom')
        if backend not in ('custom','pynccl'):raise Inconclusive('unknown decode backend')
        result['decode_backend']=backend
        rd=b.Reducer(dg,device,'decode',progress,backend=backend);rp=b.Reducer(pg,device,'prefill',progress,backend='pynccl');b.refault()
        weights=b.weights(spec,rank);dec=b.Slice(spec,rank,rd,False,weights,'packed');pre=b.Slice(spec,rank,rp,True,weights,'packed')
        sd,sp=torch.cuda.Stream(),torch.cuda.Stream()
        if a.rung=='model-terminal':
            from s5_terminal_mode import NullTrace
            b.Trace=NullTrace
        if a.rung!='model-terminal':
            phase('baseline-capture',300);basegraph,basetrace=capture(dec,sd);graphs.append(basegraph)
        def measure(name,g,trace,ds,ps,green=None,steps=200):
            phase(name);sampler=b.PcieSampler(result['pci_bus_id']);sampler.start();v=None
            torch.cuda.nvtx.range_push(name);host_start=time.time()
            try:v=b.run_mode(name if name in ('decode','prefill','serial') else 'green'+str(a.split),dec,pre,g,trace,ds,ps,ctl,steps,.0095*4/24,clock,green)
            finally:
                torch.cuda.nvtx.range_pop();counters=sampler.finish(v['host_window'] if v else None)
            v['pcie']=counters;v['phase_wall_seconds']=[host_start,time.time()];return v
        if a.rung not in ('bank','model-terminal'):
            result['modes']['decode']=measure('decode',basegraph,basetrace,sd,sp)
            result['modes']['prefill']=measure('prefill',basegraph,basetrace,sd,sp)
        phase('create-splits')
        bank={};result['bank_memory']=[dict(phase='before',**memory())]
        for split in ((24,34,44) if a.rung=='bank' else (a.split,)):
            phase('capture-green'+str(split),300)
            green=b.Green(library,split,device);contexts.append(green)
            members,complete=green.membership()
            if not complete:raise Inconclusive('incomplete membership')
            g,trace=capture(dec,green.sd);graphs.append(g)
            evidence=audit(g.graph,green.sd,library,Path(str(output)+f'.rank{rank}.split{split}.nodes.jsonl'))
            if evidence['verdict']!='PASS':raise Inconclusive('node placement not established',evidence)
            bank[split]=(green,g,trace);result['bank_memory'].append(dict(split=split,**memory()))
            if torch.cuda.memory_reserved()>18*2**30:raise Inconclusive('graph bank exceeds18GiB slice cap')
        if a.rung=='model-terminal':
            from s5_terminal_mode import run as terminal_run
            green,g,trace=bank[a.split];result['audit']=evidence
            phase('model-overlap-terminal-events')
            result['terminal_ms']=terminal_run(dec,pre,g,green.sd,green.sp,ctl,progress)
            phase('terminal-correctness',300);b.verify_mode(dec,pre,g,green.sd,green.sp)
        elif a.rung=='bank':
            result['switches']=[]
            # Contexts/graphs all exist before any timed switch. One warmup per
            # split, then each directed transition once with no pending work and
            # once with an actual prefill pending. No graph creation in timing.
            for green,g,trace in bank.values():
                phase('bank-warmup',300)
                b.verify_mode(dec,pre,g,green.sd,green.sp)
            for pending in (False,True):
                for old,new in ((24,24),(34,34),(44,44),(24,34),(34,44),(44,24),(24,44),(44,34),(34,24)):
                    phase(f'switch-{old}-{new}-pending{pending}')
                    gc,gg,tt=bank[old];nc,ng,nt=bank[new]
                    dist.barrier(group=ctl);torch.cuda.synchronize()
                    timed(gg.replay,gc.sd,progress,'decode','old')
                    terminal=torch.cuda.Event();issue=time.perf_counter()
                    if pending:
                        with torch.cuda.stream(gc.sp):pre.forward(b.Trace());terminal.record()
                    else:terminal.record(gc.sp)
                    requested=time.perf_counter();wait(terminal)
                    # Old decode already fenced; old prefill now fenced too.
                    drained=time.perf_counter()
                    gpu_ms=timed(ng.replay,nc.sd,progress,'decode','new')
                    end=time.perf_counter()
                    result['switches'].append(dict(old=old,new=new,prefill_pending=pending,
                        prefill_issue_ms=(requested-issue)*1000,drain_ms=(drained-requested)*1000,
                        dispatch_and_completion_ms=(end-drained)*1000,total_ms=(end-requested)*1000,gpu_ms=gpu_ms))
        else:
            green,g,trace=bank[a.split];result['audit']=evidence;result['actual_split']=green.split
            # Solo on the restricted streams, then the mixed run; same captured graph.
            result['modes']['decode_split']=measure('decode',g,trace,green.sd,green.sp)
            result['modes']['prefill_split']=measure('prefill',g,trace,green.sd,green.sp)
            phase('equality',300);b.verify_mode(dec,pre,g,green.sd,green.sp)
            result['modes']['overlap']=measure('overlap',g,trace,green.sd,green.sp,green)
            if a.rung=='bandwidth':
                # One controlled domain change: prefill history and PLE read
                # from VRAM. Same bytes/indices/output, not a deployable cache.
                phase('resident-prefill-ablation',300)
                result['resident_before']=memory()
                pre.kv=pre.host.to('cuda');pre.ple_uva=pre.ple_host.to('cuda')
                torch.cuda.synchronize();result['resident_after']=memory()
                if torch.cuda.memory_reserved()>18*2**30:raise Inconclusive('resident ablation exceeds18GiB')
                b.verify_mode(dec,pre,g,green.sd,green.sp)
                result['modes']['resident_overlap']=measure('resident-overlap',g,trace,green.sd,green.sp,green)
                result['bandwidth_scope']='prefill PCIe-source ablation; NVML PCIe only. DRAM attribution requires sampled GPU metrics; no claim from source byte counts.'
        phase('cleanup');torch.cuda.synchronize()
        for g in graphs:g.reset()
        for gc in contexts:gc.close()
        observed={'decode':set(),'prefill':set()}
        for line in progress.events.read_text().splitlines():
            event=json.loads(line)
            if event.get('lane') in observed and isinstance(event.get('stream'),int):observed[event['lane']].add(event['stream'])
        result['streams']={k:dict(observed_collective_stream_ids=sorted(v),count=len(v)) for k,v in observed.items()}
        result['streams']['scope']='host-issued collective streams only; graph-internal/library-private streams not enumerated'
        result['completed']=True;rows=[None,None];dist.all_gather_object(rows,result)
        if rank==0:write_json(output,dict(rung=a.rung,pair=a.pair,ranks=rows))
        dist.destroy_process_group();progress.update('completed');wd.close();return 0
    except BaseException as e:
        path=Path(str(output)+f'.rank{rank}.error.json')
        if not path.exists():write_json(path,dict(fatal_correctness=isinstance(e,b.CorrectnessError),error=str(e),evidence=getattr(e,'evidence',{}),partial_rank=result))
        wd.close();return 1 if isinstance(e,b.CorrectnessError) else 2
