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
    value=os.environ.get('S5_VALUE_PROBE','0')=='1'
    noop=os.environ.get('S5_PREFILL_NOOP_PROBE','0')=='1'
    resident=os.environ.get('S5_PREFILL_RESIDENT_PROBE','0')=='1'
    copy_kernels=os.environ.get('S5_GRAPH_KERNEL_COPIES','0')=='1'
    resident_kind=os.environ.get('S5_RESIDENT_KIND','legacy')
    ple_itemsize=int(os.environ.get('S5_PLE_ITEMSIZE','2'))
    resident_dma=os.environ.get('S5_RESIDENT_DMA','0')=='1'
    def phase(name,timeout=180):
        if value:
            setup=(name in ('setup','weights-and-lanes','baseline-capture','create-splits','terminal-prefill-warmup','terminal-resident-substitution','terminal-resident-source-trace') or name.startswith('capture-green'))
            timeout=int(os.environ.get('S5_VALUE_SETUP_TIMEOUT','300') if setup else os.environ.get('S5_VALUE_PHASE_TIMEOUT','120'))
            if timeout not in (120,300):raise Inconclusive('unreviewed value phase timeout')
        wd.arm(timeout,name)
    try:
        phase('setup',300)
        blas=os.environ.get('S5_CUBLAS_DETERMINISTIC','0')
        if blas not in ('0','1'):raise Inconclusive('unknown cuBLAS control')
        if blas=='1':
            if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise Inconclusive('cuBLAS workspace control missing before process start')
            torch.use_deterministic_algorithms(True)
        result['cublas_control']=dict(requested=blas=='1',
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
        if (value or noop or resident or copy_kernels) and a.rung!='profile-terminal':raise Inconclusive('value/control requires profile-terminal')
        if noop and (not value or a.split!=24):raise Inconclusive('no-op control scoped to value green24')
        from s5_resident import KINDS
        if resident_kind not in KINDS or ple_itemsize not in (1,2):raise Inconclusive('unknown resident source mode')
        if resident_kind!='legacy' and not (value and resident):raise Inconclusive('resident matrix requires value + resident')
        if ple_itemsize==1 and resident_kind=='legacy':raise Inconclusive('FP8 transport scoped to resident matrix')
        if resident_dma and resident_kind not in ('kv','kv_ple_prefetch'):raise Inconclusive('DMA refill requires KV residency')
        result['value_probe']=dict(enabled=value,prefill_noop=noop,resident_prefill=resident,graph_kernel_copies=copy_kernels,
                                  sparse_prefill=os.environ.get('VLLM_QSA_KVO_SPARSE_PREFILL'),resident_kind=resident_kind,ple_itemsize=ple_itemsize,resident_dma=resident_dma)
        result['value_probe'].update(source_trials=int(os.environ.get('S5_SOURCE_TRIALS','3')),
                                     source_trace_on_fail=os.environ.get('S5_SOURCE_TRACE_ON_FAIL','0')=='1')
        if os.environ.get('CUDA_VISIBLE_DEVICES')!=a.pair or os.environ.get('WORLD_SIZE')!='2':raise Inconclusive('pair mismatch')
        torch.cuda.set_device(device);dist.init_process_group('gloo')
        ctl=dist.new_group([0,1],backend='gloo');dg=dist.new_group([0,1],backend='gloo');pg=dist.new_group([0,1],backend='gloo')
        library=output.parent/'audit.so';built=[None]
        if rank==0:built[0]=build(library,copy_kernels)
        dist.broadcast_object_list(built,src=0);result['build']=built[0]
        manifest=json.loads(Path(__file__).with_name('s5_source_manifest.json').read_text())
        result['sources']={m:b.digest(importlib.import_module(m).__file__) for m in manifest['modules']}
        if result['sources']!=manifest['modules'] or b.digest(a.config)!=manifest['model_config_sha256']:raise Inconclusive('source/config mismatch')
        clock=DeviceClock(library);result['pci_bus_id']=clock.pci_bus_id(device)
        terminal=a.rung in ('model-terminal','profile-terminal')
        if not terminal:clock.validate()
        spec=plan(json.loads(a.config.read_text()),layers=4,decode_rows=4)
        # Diagnostic extension of the same shapes to actual cohort loads1..5.
        spec.update(decode_rows=a.rows,streams=2*a.rows,gdn_state=[a.rows,24,128,128])
        kind=os.environ.get('S5_LAYER_KIND','mixed');moe=os.environ.get('S5_MOE','on')
        cap=os.environ.get('FLASH_MARLIN_GREEN_CAP','0')
        if cap not in ('0','1'):raise Inconclusive('invalid Marlin cap flag')
        if a.rung!='model-terminal' and not value and (kind!='mixed' or moe!='on' or cap!='0'):raise Inconclusive('ablation only in terminal diagnostic')
        from s5_ablation import configure,apply
        configure(spec,kind)
        result['ablation']=dict(layer_kind=kind,moe=moe,marlin_cap=cap=='1')
        result['spec']=spec;result['max_connections']=os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS','unset')
        torch.manual_seed(871);phase('weights-and-lanes',300)
        backend=os.environ.get('S5_DECODE_BACKEND','custom')
        if backend not in ('custom','pynccl'):raise Inconclusive('unknown decode backend')
        result['decode_backend']=backend
        rd=b.Reducer(dg,device,'decode',progress,backend=backend);rp=b.Reducer(pg,device,'prefill',progress,backend='pynccl');b.refault()
        weights=b.weights(spec,rank);dec=b.Slice(spec,rank,rd,False,weights,'packed');pre=b.Slice(spec,rank,rp,True,weights,'packed')
        apply(dec,pre,moe)
        transport=decode_transport=None
        if resident_kind!='legacy':
            from s5_resident import PLETransport
            decode_transport=PLETransport(dec,ple_itemsize)
            transport=PLETransport(pre,ple_itemsize)
            result['ple_transport_scope']='matched byte-view FP8/2-byte BF16 input transport; BF16 slice collective; no engine FP8 equivalence claim'
        cap_evidence=None
        if cap=='1':
            if moe!='on':raise Inconclusive('cap with MoE off has no meaning')
            from s5_marlin_cap import install
            cap_evidence=install(progress)
        sd,sp=torch.cuda.Stream(),torch.cuda.Stream()
        if terminal:
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
        if a.rung not in ('bank','model-terminal','profile-terminal'):
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
            result['memory_nodes']=[x for x in evidence['nodes'] if x['type'] in ('memcpy','memset')]
            for node in result['memory_nodes']:
                print('S5_COPY '+json.dumps(dict(rank=rank,**node)),flush=True)
            if copy_kernels:
                from s5_copy_graph import KernelCopyGraph
                candidate=KernelCopyGraph(g,green.sd,library);graphs.append(candidate)
                converted=audit(candidate,green.sd,library,Path(str(output)+f'.rank{rank}.copy-nodes.jsonl'))
                if converted['verdict']!='PASS' or any(x['type'] in ('memcpy','memset') for x in converted['nodes']):raise Inconclusive('copy conversion audit failed',converted)
                expected={k:sum(x['type']==k for x in result['memory_nodes']) for k in ('memcpy','memset')}
                if candidate.counts!=expected or not expected['memcpy']:raise Inconclusive('replacement count mismatch',candidate.counts)
                # Original and converted trajectories must be bit-identical,
                # independent of the later concurrent-vs-solo noise control.
                snaps=[]
                for runner in (g,candidate):
                    dec.reset();torch.cuda.synchronize()
                    with torch.cuda.stream(green.sd):
                        for _ in range(4):runner.replay()
                    green.sd.synchronize();snaps.append(b.snapshot(dec)+[x.clone() for rc in dec.rowcaches.values() for x in (rc[1],rc[2])])
                b.checked(b.state_equal(*snaps),'kernel-copy graph differs from original graph')
                result['copy_graph']=dict(counts=candidate.counts,equality=True,audit=converted['verdict'])
                g=candidate
            bank[split]=(green,g,trace);result['bank_memory'].append(dict(split=split,**memory()))
            if torch.cuda.memory_reserved()>18*2**30:raise Inconclusive('graph bank exceeds18GiB slice cap')
        if terminal:
            from s5_terminal_mode import run as terminal_run
            from s5_terminal_correctness import verify as terminal_verify
            green,g,trace=bank[a.split];result['audit']=evidence
            result['actual_split']=green.split
            steps=int(os.environ.get('S5_TERMINAL_STEPS','200'))
            if steps not in (30,200,400):raise Inconclusive('unreviewed terminal step count')
            result['steps']=steps
            # Cold-cache JIT is a setup phase, never a throughput sample.
            phase('terminal-prefill-warmup',300)
            with torch.cuda.stream(green.sp):pre.forward(b.Trace())
            green.sp.synchronize()
            if a.rung=='profile-terminal':
                for name,gg,ds,ps,mode in (
                    ('decode',basegraph,sd,sp,'decode'),('prefill',basegraph,sd,sp,'prefill'),
                    ('decode_split',g,green.sd,green.sp,'decode'),
                    ('prefill_split',g,green.sd,green.sp,'prefill'),
                    ('serial',basegraph,sd,sp,'serial')):
                    if value and name in ('serial','prefill_split'):continue
                    phase('terminal-'+name,300)
                    result['modes'][name]=terminal_run(dec,pre,gg,ds,ps,ctl,progress,steps,mode,False)
            phase('model-overlap-terminal-events')
            result['modes']['overlap']=terminal_run(dec,pre,g,green.sd,green.sp,ctl,progress,steps,'overlap',False)
            phase('terminal-correctness',300)
            overlap=result['modes']['overlap']
            result['correctness']=terminal_verify(dec,pre,g,green.sd,green.sp,
                (len(overlap['raw_decode']),len(overlap['raw_prefill'])),progress)
            result['correctness_verified']=result['correctness']['passed']
            if noop:
                from s5_collective_control import local_prefill_collectives
                # The real communicator remains allocated: only calls are
                # bypassed. Neither decode graph nor its CA path changes.
                with local_prefill_collectives(pre,lambda:torch.cuda.current_stream().cuda_stream) as control:
                    phase('terminal-noop-warmup')
                    pre.reset();torch.cuda.synchronize()
                    with torch.cuda.stream(green.sp):pre.forward(b.Trace())
                    green.sp.synchronize()
                    phase('terminal-noop-prefill-solo')
                    result['modes']['noop_prefill']=terminal_run(dec,pre,g,sd,sp,ctl,progress,steps,'prefill',False)
                    before=control.collective
                    phase('terminal-noop-overlap')
                    m=terminal_run(dec,pre,g,green.sd,green.sp,ctl,progress,steps,'overlap',False)
                    result['modes']['overlap_noop']=m
                    result['noop_collective_calls']=control.collective-before
                    phase('terminal-noop-correctness')
                    result['noop_correctness']=terminal_verify(dec,pre,g,green.sd,green.sp,
                        (len(m['raw_decode']),len(m['raw_prefill'])),progress)
                result['noop_scope']='local-return prefill collective ablation; own solo oracle; not equivalent model or a link-only attribution'
            if resident and resident_kind!='legacy':
                from s5_resident import run_control
                run_control(dec,pre,transport,resident_kind,g,green,sd,sp,ctl,progress,steps,phase,memory,result,resident_dma)
            elif resident:
                phase('terminal-resident-setup')
                original_kv,original_ple=pre.kv,pre.ple_uva
                result['resident_before']=memory()
                pre.kv=pre.host.to('cuda');pre.ple_uva=pre.ple_host.to('cuda')
                torch.cuda.synchronize();result['resident_after']=memory()
                result['resident_bytes']=pre.host.numel()*pre.host.element_size()+pre.ple_host.numel()*pre.ple_host.element_size()
                if torch.cuda.memory_reserved()>18*2**30:raise Inconclusive('resident control exceeds18GiB')
                try:
                    # Prove substitution itself against the original host path.
                    snapshots=[]
                    for kv,ple in ((original_kv,original_ple),(pre.kv,pre.ple_uva)):
                        pre.kv,pre.ple_uva=kv,ple;pre.reset();torch.cuda.synchronize()
                        with torch.cuda.stream(green.sp):pre.forward(b.Trace())
                        green.sp.synchronize();snapshots.append(b.snapshot(pre))
                    b.checked(b.state_equal(*snapshots),'resident prefill differs from host-input prefill')
                    phase('terminal-resident-prefill-solo')
                    result['modes']['resident_prefill']=terminal_run(dec,pre,g,sd,sp,ctl,progress,steps,'prefill',False)
                    phase('terminal-resident-overlap')
                    m=terminal_run(dec,pre,g,green.sd,green.sp,ctl,progress,steps,'overlap',False)
                    result['modes']['overlap_resident']=m
                    phase('terminal-resident-correctness')
                    result['resident_correctness']=terminal_verify(dec,pre,g,green.sd,green.sp,(len(m['raw_decode']),len(m['raw_prefill'])),progress)
                finally:pre.kv,pre.ple_uva=original_kv,original_ple
            if cap_evidence:
                result['marlin_cap']=cap_evidence()
                by_stream={r['stream']:r for r in result['marlin_cap']['streams']}
                for stream,count in ((green.sd,green.split[0]),(green.sp,green.split[1])):
                    r=by_stream.get(stream.cuda_stream,{})
                    if r.get('cap')!=count or r.get('intercepted_queries',0)<1:raise Inconclusive('missing actual lane cap evidence',r)
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
        phase('cleanup')
        if transport is not None:
            from s5_resident import close_resident_resources
            result['resident_cleanup']=close_resident_resources(
                graphs,contexts,(dec,pre),(decode_transport,transport),progress)
        else:
            torch.cuda.synchronize()
            for g in reversed(graphs):g.reset()
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
        from s5_failure import worker_failure
        path=Path(str(output)+f'.rank{rank}.error.json')
        return worker_failure(path,e,b.CorrectnessError,result,wd.close)
