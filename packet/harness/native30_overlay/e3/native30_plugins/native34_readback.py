"""Opt-in I4/S4 post-drain indirect-value diagnostic. Never repairs an operand.

CPU copies only, bounded to the first txn100/chunk0/layer3 pre-indexer on PP0.
Writer history names observed host-issued operations, not a device instruction PC.
Native35 R (arm O) takes the same copies without any drain or fence; the record
then says after_registered_drain=False and the values are evidence only.
"""
import hashlib
import json
import os
import re
import time

MAX_VALUES=4096
MAX_WRITERS=64
MAX_HISTORY=4
KEYS=('k_work_metadata_ptr','query_start_loc_ptr','logical_positions_ptr',
      'pos_ptr','state_table_ptr','state_slots_ptr','compressed_slots_ptr')
DESTINATIONS=('token_to_req_ptr','logical_positions_ptr','visible_blocks_ptr',
              'slot_mapping_ptr','k_work_metadata_ptr')


def enabled():return os.environ.get('FLASH_NATIVE34_READBACK')=='1'


def flatten(value):
    if isinstance(value,(tuple,list)):
        return [item for row in value for item in flatten(row)]
    return [int(value)]


def digest(values):return hashlib.sha256(json.dumps(values,separators=(',',':')).encode()).hexdigest()


def cpu_values(tensor):
    count=int(tensor.numel())
    if count>MAX_VALUES:raise ValueError('native34-readback-element-limit')
    # No CUDA arithmetic, gather, index_select, dtype cast or device allocation.
    return flatten(tensor.detach().cpu().tolist())


def cpu_plan(value):
    if value is None or str(getattr(value,'device',None))!='cpu':return None
    if int(value.numel())>32:return None
    return flatten(value.tolist())


def pointer(tensor):
    return dict(data_ptr=int(tensor.data_ptr()),shape=[int(v) for v in tensor.shape],
                stride=[int(v) for v in tensor.stride()],dtype=str(tensor.dtype),
                storage_extent=int(tensor.untyped_storage().nbytes()),
                storage_impl=int(tensor.untyped_storage()._cdata),
                allocation_base=int(tensor.untyped_storage().data_ptr()))


def remember(s,tensor,record,event=None):
    key=(int(tensor.untyped_storage()._cdata),int(tensor.untyped_storage().data_ptr()))
    history=s.setdefault('readback_writers',{})
    if key not in history and len(history)>=MAX_WRITERS:
        s['readback_writer_limit']=True;return
    row=dict(record,destination=pointer(tensor),event=event,host_issued_monotonic_ns=time.monotonic_ns())
    history.setdefault(key,[]).append(row)
    history[key]=history[key][-MAX_HISTORY:]


def writer_chain(s,tensor):
    key=(int(tensor.untyped_storage()._cdata),int(tensor.untyped_storage().data_ptr()))
    rows=[]
    for old in s.get('readback_writers',{}).get(key,()):
        row={k:v for k,v in old.items() if k!='event'}
        event=old.get('event');row['completion_observed']=bool(event.query()) if event is not None else False
        rows.append(row)
    return dict(observed=bool(rows),history=rows,last_catalogued_writer=rows[-1] if rows else None,
        coverage_incomplete=True,history_overflow=s.get('readback_writer_limit',False),
        scope='host-issued source catalogue and current-stream completion events; unregistered/peer writes and access-PC unproved')


def builder_context(builder,common,s,stream):
    if not any(re.search(r'(?:^|\.)layers\.3\.',n) for n in builder.layer_names):return None
    s['readback_sequence']=s.get('readback_sequence',0)+1
    return dict(sequence=s['readback_sequence'],provider='QSAMetadataBuilder.build',
        builder_id=hex(id(builder)),layers=list(builder.layer_names)[:64],
        query_start_loc_cpu=cpu_plan(common.query_start_loc_cpu),
        seq_lens_cpu_upper_bound=cpu_plan(common.seq_lens_cpu_upper_bound),
        seq_lens_exact=False,  # ordinary async decode only supplies an upper bound
        num_tokens=int(common.num_actual_tokens),num_reqs=int(common.num_reqs),
        storage_block_size=int(builder.storage_block_size),compress_ratio=int(builder.compress_ratio),
        circular_buffer_size=int(builder.kv_cache_spec.block_size) if builder.is_circular_buffer else 0,
        stream=stream,step=s['step'],mode=s['mode'])


def metadata_written(s,bound,context,event):
    for name in DESTINATIONS:
        tensor=bound.get(name)
        if tensor is None:continue
        remember(s,tensor,dict(context,kernel='_build_qsa_metadata_kernel',argument=name,
            writer_kind='actual JIT launch returned; completion event follows on same stream',
            source='models/qwen4_exp/common/qsa_cache.py:258-381'),event)


def chunk_written(chunk,s,stream,event):
    admission=chunk.admission
    if admission.transaction.txn_seq!=100 or admission.chunk_start!=0:return
    s['readback_sequence']=s.get('readback_sequence',0)+1
    record=dict(sequence=s['readback_sequence'],provider='LaneChunk.__init__',
        stream=stream,transaction=100,chunk=[admission.chunk_start,admission.chunk_end],
        request_index=int(chunk.req_index),source='lane_chunk.py:473-487',
        writer_kind='original arange/tensor/copy_/zero_ operations; constructor-tail event')
    for name in ('positions','query_start_loc'):
        remember(s,getattr(chunk,name),dict(record,argument=name),event)
    for group,tensor in enumerate(chunk.block_tables):
        remember(s,tensor,dict(record,argument='block_tables',group=group),event)


def expected(runner,bound):
    from vllm.v1.flash_e3.model_hooks import local_text_model
    chunk=runner._flash_s5_bridge.dispatcher.chunk;admission=chunk.admission
    if (admission.transaction.txn_seq,admission.chunk_start,admission.chunk_end)!=(100,0,1024):
        raise ValueError('native34-readback-wrong-chunk')
    indexer=local_text_model(runner.model).layers[3].self_attn.indexer
    groups=runner.kv_cache_config.kv_cache_groups
    def group(prefix):
        found=[i for i,g in enumerate(groups) if prefix in g.layer_names]
        if len(found)!=1:raise ValueError('native34-readback-cache-group-ambiguous')
        i=found[0];return i,chunk.tables.expanded(i,admission.block_tables[i])
    raw_gid,raw=group(indexer.raw_key_cache.prefix)
    comp_gid,comp=group(indexer.compressed_key_cache.prefix)
    n=int(bound['num_tokens']);start=admission.chunk_start;end=admission.chunk_end
    ratio=int(bound['COMPRESS_RATIO']);size=int(bound['STATE_SIZE']);page=int(bound['COMP_PAGE_SIZE'])
    positions=list(range(start,end));capacity=int(bound['state_table_ptr'].shape[1])
    if n!=end-start or not raw or len(raw)>capacity:raise ValueError('native34-readback-host-plan-shape')
    slots=[raw[0]*size+p%size if t+size>=n else -1 for t,p in enumerate(positions)]
    compressed=[]
    for p in positions:
        logical=p//ratio;column=logical//page
        valid=(p+1)%ratio==0 and column<len(comp) and chunk.tables.slot_mapping_enabled[comp_gid]
        compressed.append(comp[column]*page+logical%page if valid else -1)
    count=max(end//ratio-start//ratio,1);work=int(bound['num_k_work'])
    return dict(k_work_metadata_ptr=flatten([[0,i] if i<count else [-1,-1] for i in range(work)]),
        query_start_loc_ptr=[0,n],logical_positions_ptr=positions,pos_ptr=positions,
        state_table_ptr=raw+[0]*(capacity-len(raw)),state_slots_ptr=slots,compressed_slots_ptr=compressed),dict(
        transaction=100,chunk=[start,end],request_index=int(chunk.req_index),raw_group=raw_gid,
        compressed_group=comp_gid,raw_blocks=raw,compressed_blocks=comp,compress_ratio=ratio,
        state_size=size,compressed_page_size=page,source='actual ChunkAdmission + LaneTables.expanded + cache owner prefixes',
        qsa_metadata_request_index=0,runner_request_index=int(chunk.req_index),
        note='metadata uses compact batch request0; runner slot8 is a different index space')


def compare(name,actual,intended):
    mismatches=[dict(index=i,observed=a,expected=b) for i,(a,b) in enumerate(zip(actual,intended)) if a!=b]
    same_length=len(actual)==len(intended)
    return dict(argument=name,verdict='PASS' if same_length and not mismatches else 'VIOLATION',
        observed_count=len(actual),expected_count=len(intended),mismatch_count=len(mismatches),
        equal_lengths=same_length,first_mismatches=mismatches[:16],observed_sha256=digest(actual),expected_sha256=digest(intended))


def requirements(bound,values,cached=None):
    n=int(bound['num_tokens']);ratio=int(bound['COMPRESS_RATIO']);size=int(bound['STATE_SIZE'])
    page=int(bound['COMP_PAGE_SIZE']);blocks=int(bound['num_state_blocks']);comp=int(bound['num_compressed_blocks'])
    rows=int(bound['state_table_ptr'].shape[0]);stride=int(bound['state_table_ptr'].shape[1])
    rope_rows=int(bound['cos_sin_ptr'].shape[0]);q=values['query_start_loc_ptr'];logical=values['logical_positions_ptr']
    checks=[];tails=set();k_positions=[]
    def check(name,valid,**fields):checks.append(dict(requirement=name,verdict='PASS' if valid else 'VIOLATION',**fields))
    check('query-starts',len(q)>=2 and q[0]>=0 and all(0<=a<=b<=n for a,b in zip(q,q[1:])))
    check('Q-RoPE-positions',all(0<=p<rope_rows for p in values['pos_ptr']),rows=rope_rows)
    check('logical-position-nonnegative',len(logical)==n and all(p>=0 for p in logical))
    check('state-slots',all(v==-1 or 0<=v<blocks*size for v in values['state_slots_ptr']),kernel_masks_invalid=True)
    check('compressed-slots',all(v==-1 or 0<=v<comp*page for v in values['compressed_slots_ptr']),kernel_masks_invalid=True)
    check('state-table-blocks',all(v==-1 or 0<=v<blocks for v in values['state_table_ptr']),kernel_masks_invalid=True)
    work=values['k_work_metadata_ptr'];bad_requests=[];bad_ranges=[];bad_work=[]
    for i in range(0,len(work)-1,2):
        req,offset=work[i:i+2]
        if req<0:continue
        if not (0<=req<min(rows,len(q)-1)):
            bad_requests.append([i//2,req]);continue
        lo,hi=q[req:req+2]
        if not 0<=lo<hi<=min(n,len(logical)):
            bad_ranges.append([i//2,lo,hi]);continue
        end=logical[hi-1];start=end-(hi-lo)+1;groups=(end+1)//ratio-start//ratio
        if not 0<=offset<max(groups,1):bad_work.append([i//2,offset,groups])
        if offset>=groups:continue
        boundary=((start+ratio)//ratio)*ratio-1+offset*ratio
        first=boundary-ratio+1
        if bound['CACHE_HAS_ROPE_POS']:
            token=lo+first-start
            if first>=start:
                if lo<=token<hi and token<n:
                    axes=3 if bound['IS_2D_POSITIONS'] else 1
                    k_positions.extend(values['pos_ptr'][axis*n+token] for axis in range(axes))
                else:k_positions.append(0)
            else:
                block=values['state_table_ptr'][req*stride]
                if 0<=block<blocks:
                    key=(block,first%size);tails.add(key)
                    if cached is not None and key in cached:k_positions.extend(cached[key])
                else:k_positions.append(0)
        else:k_positions.append(first)
    check('k-work-request-index',not bad_requests,first_bad=bad_requests[:16],unmasked_device_load=True)
    check('active-query-range',not bad_ranges,first_bad=bad_ranges[:16],unmasked_device_load=True)
    check('work-in-request',not bad_work,first_bad=bad_work[:16])
    check('K-RoPE-positions',all(0<=p<rope_rows for p in k_positions),rows=rope_rows)
    unresolved=tails-set(cached or {})
    if unresolved:checks.append(dict(requirement='cached-RoPE-tail',verdict='UNPROVEN',count=len(unresolved)))
    return checks,sorted(tails)


def snapshot(runner,s,bound,emit,drained=True):
    wanted,plan=expected(runner,bound)
    values={}
    for name in KEYS:
        emit(s,'indexer-readback-copy-begin',argument=name,operand=pointer(bound[name]),target_launched=False)
        values[name]=cpu_values(bound[name])
        emit(s,'indexer-readback-copy-end',argument=name,count=len(values[name]),sha256=digest(values[name]),target_launched=False)
    checks,tails=requirements(bound,values)
    cached={}
    if len(tails)>64:raise ValueError('native34-readback-tail-limit')
    if tails:
        import torch
        d=int(bound['D']);cache=bound['state_cache_ptr']
        for block,row in tails:
            # Indices are host-checked before slicing. Only the integer RoPE
            # tail is copied; no embedding/key values or whole cache readback.
            if not (0<=block<int(cache.shape[0]) and 0<=row<int(cache.shape[1]) and d+12<=int(cache.shape[2])):
                raise ValueError('native34-readback-tail-layout')
            cpu=cache[block,row,d:d+12].detach().cpu()
            cached[(block,row)]=[int(x) for x in cpu.view(torch.int64).tolist()]
        checks,_=requirements(bound,values,cached)
    comparisons=[]
    for name in KEYS:
        result=compare(name,values[name],wanted[name]);comparisons.append(result)
        fields=dict(result,observed=values[name],expected=wanted[name],operand=pointer(bound[name]),
            writer_chain=writer_chain(s,bound[name]),device_values_observed=True)
        if len(json.dumps(fields))>110*1024:
            fields['observed']=values[name][:16];fields['expected']=wanted[name][:16];fields['full_arrays_emitted']=False
        else:fields['full_arrays_emitted']=True
        emit(s,'indexer-indirect-value',**fields)
    violations=any(r['verdict']=='VIOLATION' for r in comparisons+checks)
    result=dict(verdict='VIOLATION' if violations else 'UNPROVEN' if any(r['verdict']=='UNPROVEN' for r in checks) else 'PASS',
        device_values_observed=True,plan=plan,requirements=checks,arguments=len(comparisons),
        cached_rope_tails=[dict(block=b,row=r,positions=v,writer_chain=writer_chain(s,bound['state_cache_ptr'])) for (b,r),v in cached.items()],
        after_registered_drain=bool(drained),before_indexer_launch=True,diagnostic_only=True,
        coverage='all seven bounded indirect operands; metadata/constructor writer catalogue, peer and unregistered writers unproved',
        timing_perturbed=True,no_operand_repair=True)
    emit(s,'indexer-readback-complete',**result)
    return result
