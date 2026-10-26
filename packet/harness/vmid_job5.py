#!/usr/bin/env python3
"""VM-only fixed observer capacity census: seven bound graphs, short/long windows."""
import argparse
import ctypes as C
import json
from pathlib import Path
import sys
import time
import vmid_support as s
from vmid_job2 import GraphBinding

BANK=7
PRESSURE=8
MIX={'kernel':1386,'memcpy':26,'memset':7}
NODES=sum(MIX.values())
WINDOWS={'short':40,'long':1200}
RESOURCE_FLOOR=189796
LIMITS=dict(buffer_bytes=1048576,buffer_count=32,max_context_records=128,
            max_copy_records=32768,max_graph_events=262144,max_graphs=64,
            max_kernel_records=131072,max_links=65536,max_nodes_per_graph=131072,
            max_nodes_total=131072)
OBSERVER_SOURCE='1dd7deae552a9994fbea439f8a140d93c52140d273c378808d5971b260d02e5b'
GREEN_SOURCE='731d31d10c69bd8f4e9cb302e46d8dc65564d974fe9480c2e1c290e0f385c598'


def occupancy(window):
    """Exact reported stored counts except resource callbacks (upper bound).

    Native event_count is not exported. resource_events counts all callbacks,
    including context callbacks that need no EventRec; use it conservatively.
    Never present this upper bound as exact EventRec occupancy.
    """
    counts=window.get('counts',{});mapping=window.get('mapping',{})
    spec={
        'aliases':(mapping.get('node_aliases'),LIMITS['max_links'],'exact joined aliases'),
        'node_keys':(mapping.get('node_keys'),LIMITS['max_nodes_total'],'exact bound node keys'),
        'resource_events':(counts.get('resource_events'),LIMITS['max_graph_events'],'callback upper bound on stored EventRec count'),
        'kernel_records':(counts.get('kernel_records'),LIMITS['max_kernel_records'],'stored kernel records'),
        'copy_records':(counts.get('copy_records'),LIMITS['max_copy_records'],'stored memcpy and memset records'),
    }
    return {key:dict(reported_count=n,capacity=cap,remaining_lower_bound=cap-n if type(n) is int else None,basis=basis)
            for key,(n,cap,basis) in spec.items()}


def phase_grade(r,replays):
    f=list(r.get('failures',[]));m=list(r.get('missing',[]))
    for key in ('output_exact','cleanup_ok','native_schema_valid'):
        if r.get(key) is False:f.append(key)
        elif r.get(key) is not True:m.append(key)
    if r.get('limits') is None:m.append('observer_limits')
    elif r['limits']!=LIMITS:f.append('observer_limits_changed')
    if r.get('replays')!=replays:m.append('replay_census')
    if r.get('bound_graphs')!=BANK or r.get('startup_pressure_graphs')!=PRESSURE:m.append('bank_capture_census')
    if r.get('split') is None:m.append('split')
    elif r['split']!=dict(decode=44,prefill=38,total=82):f.append('split_changed')
    graphs=r.get('graphs',[])
    if [(x.get('role'),x.get('index')) for x in graphs] != [('startup_pressure',i) for i in range(PRESSURE)]+[('bound',i) for i in range(BANK)]:m.append('graph_inventory_census')
    for x in graphs:
        if x.get('native_node_inventory') is None:m.append('native_node_inventory')
        elif x['native_node_inventory']!=[1386,1,24,1,7,0,0,0]:f.append('native_node_mix')
        if x.get('role')=='bound':
            if x.get('serialized_stable') is False:f.append('graph_definition_changed')
            elif x.get('serialized_stable') is not True:m.append('graph_serializer')
            if type(x.get('serialized_bytes')) is not int or x['serialized_bytes']<=0:m.append('graph_bytes')
    static=r.get('static',{});window=r.get('window',{})
    if static.get('complete') is not True:m.append('static_incomplete')
    if window.get('complete') is not True:m.append('observation_incomplete')
    for doc in (static,window):
        if doc.get('limits') is None:m.append('native_limits')
        elif doc['limits']!=LIMITS:f.append('native_limits_changed')
        slots=doc.get('graph_slots',[])
        if [x.get('slot') for x in slots]!=list(range(BANK)):m.append('seven_native_slots')
        for x in slots:
            if x.get('node_count')!=NODES or x.get('required')!=MIX:f.append('static_node_mix')
    counts=window.get('counts',{});mapping=window.get('mapping',{});flags=window.get('flags',{})
    rings=counts.get('ring_overflows',{})
    if set(rings)!=set(('aliases','resource','kernel','copy','context')):m.append('ring_loss_evidence')
    for key,n in rings.items():
        if type(n) is not int or n<0:m.append('malformed_ring_counter')
        elif n:f.append('capacity_overflow:'+key)
    for key in ('dropped_records_total','buffers_refused','drop_query_errors','parse_errors','callback_errors','unknown_node_records',
                'ambiguous_node_records','kind_mismatch_records','known_wrong_context_records','unknown_activity_records','unknown_resource_events'):
        n=counts.get(key)
        if type(n) is not int or n<0:m.append('counter:'+key)
        elif n:f.append('loss_or_mapping:'+key)
    if any(v is True for v in flags.values()) or window.get('reasons'):m.append('native_flags_or_reasons')
    if mapping.get('node_keys')!=BANK*NODES:m.append('bound_node_key_census')
    if mapping.get('node_aliases')!=BANK*NODES:m.append('bound_alias_census')
    if mapping.get('ambiguous_graph_ids') is not False or mapping.get('node_overlap') is not False:m.append('mapping_ambiguity')
    if mapping.get('alias_unknown') is None:m.append('unknown_alias_count')
    elif mapping['alias_unknown']!=0:f.append('unknown_alias')
    measured=occupancy(window)
    if r.get('occupancy')!=measured:m.append('occupancy_receipt')
    for key,x in measured.items():
        n=x['reported_count']
        if type(n) is not int or n<0:m.append('occupancy:'+key)
        elif n>x['capacity']:f.append('capacity_exceeded:'+key)
    resource=counts.get('resource_events')
    if type(resource) is not int or resource<RESOURCE_FLOOR:m.append('startup_resource_volume_below_full8')
    ctx=window.get('expected_context',{})
    if ctx.get('is_green') is False or ctx.get('contradictory') or ctx.get('sm_mismatched',0):f.append('wrong_context')
    if ctx.get('is_green') is not True or ctx.get('sm_matching',0)<1:m.append('green_context_evidence')
    slots=window.get('graph_slots',[])
    for x in slots:
        n=replays if x.get('slot')==0 else 0
        if x.get('exercised') is not bool(n):f.append('wrong_graph_exercised')
        if n and x.get('complete') is not True:m.append('active_slot_incomplete')
        for kind,key in (('kernel','kernel_executions'),('memcpy','copy_executions'),('memset','memset_executions')):
            if x.get(key)!=MIX[kind]*n:m.append('execution_count:'+kind)
            if x.get('node_execution_bounds',{}).get(kind)!={'min':n,'max':n}:m.append('per_node_multiplicity:'+kind)
    expected_kernel=MIX['kernel']*replays;expected_copy=(MIX['memcpy']+MIX['memset'])*replays
    if counts.get('kernel_records')!=expected_kernel:m.append('kernel_ring_count')
    if counts.get('copy_records')!=expected_copy:m.append('copy_ring_count')
    if r.get('expected_records')!=dict(kernel=expected_kernel,copy=expected_copy):m.append('expected_record_census')
    peak=r.get('allocator_peak_reserved')
    if type(peak) is not int or peak<0:m.append('allocator_peak')
    elif peak>64*1024**2:f.append('unexpected_allocator_growth')
    return s.verdict(f,m)


def grade(r):
    phases={name:phase_grade(r.get('phases',{}).get(name,{}),n) for name,n in WINDOWS.items()}
    combined=s.combine(list(phases.values()))
    result=s.verdict(combined['failures']+r.get('failures',[]),combined['missing']+r.get('missing',[]))
    result['phases']=phases
    return result


def phase(torch,ro,gc,native,receipt,replays,out,name):
    r=dict(replays=0,bound_graphs=0,startup_pressure_graphs=0,graphs=[],failures=[],missing=[],
           expected_records=dict(kernel=1386*replays,copy=33*replays),started_wall_s=time.time())
    session=split=None;owners=[];window=None;graphlib=None;before={}
    try:
        binding=ro.ReplayBinding(str(native/'flash_s5_replay_observer.so'),receipt['files']['flash_s5_replay_observer.so'],
            receipt['files']['flash_s5_replay_observer.cpp'],s.CUPTI,
            cupti_library_path='/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib/libcupti.so.13')
        limits=ro.ObserverLimits()
        if limits.as_record()!=LIMITS:raise RuntimeError('observer_limits_changed')
        session=ro.ObserverSession(binding,limits);session.start()
        r['limits']=limits.as_record();r['observer_host_reserved_bytes']=session._native_reserved
        split=gc.open_split(0,env={'FLASH_S5_DECODE_SMS':'44'},torch_module=torch,
                            binding=gc.DriverBinding(str(native/'flash_s5_green.so')))
        r['split']=dict(decode=split.decode_sms,prefill=split.prefill_sms,total=split.total_sms)
        ordinary=torch.cuda.Stream()
        src=torch.full((1024,),7,dtype=torch.int32,pin_memory=True)
        dst=torch.empty((1024,),dtype=torch.int32,pin_memory=True)
        a,b,z=[torch.empty((1024,),dtype=torch.int32,device='cuda') for _ in range(3)]
        torch.cuda.synchronize()
        graphlib=GraphBinding(native/'vmid_graph_probe.so')
        lib,token=binding._lib,session._token
        s.checked(lib.s5ro_bind_context(token,split.decode_context_id,44),'bind_context')
        # Eight unreplayed primary-context graphs approximate full8's startup
        # callback volume in addition to the seven bound green-bank graphs.
        # Keep every graph/exec alive to avoid recycled-id or destruction noise.
        for role,count,stream in (('startup_pressure',PRESSURE,ordinary.cuda_stream),('bound',BANK,split.decode_stream_token)):
            for i in range(count):
                own,graph=C.c_uint64(),C.c_uint64()
                s.checked(graphlib.lib.vmg_create(stream,a.data_ptr(),b.data_ptr(),z.data_ptr(),src.data_ptr(),dst.data_ptr(),
                          1024,1386,26,7,0,C.byref(own),C.byref(graph)),'graph_create')
                owners.append((role,i,own.value,graph.value))
                inv=(C.c_int*8)();s.checked(graphlib.lib.vmg_inventory(own.value,inv,8),'graph_inventory')
                item=dict(role=role,index=i,native_node_inventory=list(inv))
                if role=='bound':
                    count_out=C.c_uint32()
                    s.checked(lib.s5ro_bind_graph(token,graph.value,i,C.byref(count_out)),'bind_graph')
                    raw=s.graph_bytes(lib,graph.value);before[i]=raw
                    item.update(serialized_bytes=len(raw),serialized_stable=raw==s.graph_bytes(lib,graph.value),bound_nodes=count_out.value)
                    r['bound_graphs']+=1
                else:r['startup_pressure_graphs']+=1
                r['graphs'].append(item)
                s.write(out/(name+'-progress.json'),r)
        r['static']=session._report_call(lib.s5ro_static,(token,));ro._validate_static(r['static'])
        s.write(out/(name+'-startup.json'),r)
        target=next(owner for role,i,owner,_ in owners if role=='bound' and i==0)
        start=C.c_uint64();s.checked(lib.s5ro_begin_window(token,1,C.byref(start)),'begin_window');window=start.value
        for _ in range(replays):
            s.checked(graphlib.lib.vmg_replay(target),'replay');r['replays']+=1
        split.decode_stream.synchronize()
        r['window']=session._report_call(lib.s5ro_end_window,(token,window,1),retry=False);window=None
        # Save the raw receipt BEFORE validation: even an overflow or malformed
        # report is retained, not converted into an unexplained empty result.
        s.write(out/(name+'-window.json'),r['window'])
        r['occupancy']=occupancy(r['window'])
        ro._validate_window(r['window'],limits,1);r['native_schema_valid']=True
        for role,i,_,graph in owners:
            if role=='bound':
                item=next(x for x in r['graphs'] if x['role']==role and x['index']==i)
                item['serialized_stable'] &= before[i]==s.graph_bytes(lib,graph)
        r['output_exact']=bool(torch.all(dst==1393).item() and torch.all(b==1393).item() and torch.all(z==0).item())
        r['allocator_peak_reserved']=torch.cuda.max_memory_reserved(0)
        time.sleep(1)
    except Exception as exc:
        r['error_type']=type(exc).__name__;r['error_code']=getattr(exc,'code',None)
        if isinstance(exc,torch.cuda.OutOfMemoryError):r['failures'].append('cuda_out_of_memory')
        else:r['missing'].append('phase_exception')
        import traceback
        r['error_stack']=[dict(file=Path(x.filename).name,line=x.lineno,function=x.name) for x in traceback.extract_tb(exc.__traceback__)]
    finally:
        errors=[];drained=False
        try:
            torch.cuda.synchronize();drained=True
            if session and window is not None:
                pending=session._report_call(session.binding._lib.s5ro_end_window,(session._token,window,1),retry=False)
                s.write(out/(name+'-cleanup-window.json'),pending)
        except Exception:errors.append('drain_window')
        # Unsubscribe before graph destruction. This is normal session teardown,
        # never loss-counter clearing or a retry of a failed measurement.
        try:
            if session:session.close()
        except Exception:errors.append('observer_close')
        if drained:
            for _,_,owner,_ in reversed(owners):
                try:s.checked(graphlib.lib.vmg_close(owner),'graph_close')
                except Exception:errors.append('graph_close')
            try:
                if split:split.close()
            except Exception:errors.append('split_close')
        r['cleanup_ok']=not errors;r['cleanup_errors']=errors;r['ended_wall_s']=time.time()
    return r


def probe(modules,native,out):
    r=dict(schema='vm<vm-id>-job5/1',image=s.IMAGE,phases={},failures=[],missing=[],
           scope='synthetic seven-graph bank plus eight unreplayed startup-pressure graphs; pinned observer limits; no engine, collective or throughput claim',
           long_window_basis='1200 whole-graph replays as requested; not a measurement of teacher-token-to-replay equivalence',
           capacity_prediction=dict(kernel_long=1663200,kernel_limit=131072,copy_long=39600,copy_limit=32768,
                                    long_window_loss_expected=True))
    try:
        import torch
        torch.cuda.set_per_process_memory_fraction(.125,0)
        if torch.cuda.device_count()!=1 or torch.cuda.get_device_capability(0)!=(8,6):raise RuntimeError('one_sm86_device_required')
        sys.path.insert(0,str(modules));import replay_observer as ro;import green_context as gc
        receipt=json.loads((native/'q17-helper-build.json').read_text())
        if receipt['image']!=s.IMAGE or receipt['cupti_library_sha256']!=s.CUPTI:raise RuntimeError('identity')
        for filename,digest in receipt['files'].items():
            if s.sha(native/filename)!=digest:raise RuntimeError('pin')
        for filename in ('flash_s5_green.cu','flash_s5_replay_observer.cpp','flash_s5_graph_bytes.cpp'):
            if s.sha(modules/'native'/filename)!=receipt['files'][filename]:raise RuntimeError('source_pin')
        if receipt['files']['flash_s5_replay_observer.cpp']!=OBSERVER_SOURCE or receipt['files']['flash_s5_green.cu']!=GREEN_SOURCE:raise RuntimeError('full10_source_changed')
        r['build']=receipt;r['software']=dict(torch=str(torch.__version__),cuda=torch.version.cuda)
        for name,n in WINDOWS.items():
            # Independent observer + graph lifetime per declared phase. Preserve
            # the short verdict before deliberately stressing the long window.
            item=phase(torch,ro,gc,native,receipt,n,out,name)
            r['phases'][name]=item;item['grade']=phase_grade(item,n)
            s.write(out/(name+'.json'),item)
            if not item.get('cleanup_ok'):break
    except Exception as exc:
        r['error_type']=type(exc).__name__
        if isinstance(exc,RuntimeError) and exc.args and str(exc.args[0]) in ('identity','pin','source_pin','full10_source_changed'):r['failures'].append(str(exc.args[0]))
        else:r['missing'].append('setup_exception')
    r['grade']=grade(r);s.write(out/'probe.json',r)
    return r['grade']['rc']


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('probe','monitor','finalize'),required=True)
    p.add_argument('--modules',type=Path);p.add_argument('--native',type=Path)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--process-rc',type=int,default=2)
    a=p.parse_args()
    if a.mode=='monitor':return s.monitor(a.out,'flashnext-job5')
    if a.mode=='finalize':return s.finalize(a.out,a.process_rc,grade,'job5')
    return probe(a.modules,a.native,a.out)

if __name__=='__main__':sys.exit(main())
