"""Source-bound QSA pre-indexer operands and direct-address range checks.

Indirect device values remain unobserved. No check below certifies their value
bounds or the absence of a writer on a peer device.
"""
from native34_storage import tensor_view

ORIGINS={
 'q_ptr':('q','native attention-pre projected_qk view'),
 'k_ptr':('k','native attention-pre projected_qk view'),
 'pos_ptr':('positions','LaneChunk private positions'),
 'cos_sin_ptr':('cos_sin_cache','indexer rotary_emb.cos_sin_cache'),
 'q_norm_weight_ptr':('q_norm_weight','indexer q_layernorm.weight'),
 'k_norm_weight_ptr':('k_norm_weight','indexer k_layernorm.weight'),
 'q_out_ptr':('q_out','indexer.forward new_empty output'),
 'state_cache_ptr':('state_cache','indexer raw_key_cache.kv_cache, request-persistent'),
 'state_slots_ptr':('state_slots','raw_metadata.slot_mapping'),
 'state_table_ptr':('state_block_table','raw_metadata.block_table'),
 'query_start_loc_ptr':('query_start_loc','raw_metadata.query_start_loc'),
 'logical_positions_ptr':('logical_positions','raw_metadata.logical_positions'),
 'compressed_slots_ptr':('compressed_slots','compressed_metadata.slot_mapping'),
 'k_work_metadata_ptr':('k_work_metadata','compressed_metadata.k_work_metadata'),
 'compressed_cache_ptr':('compressed_cache','indexer compressed_key_cache.kv_cache, request-persistent'),
}


def operands(arguments,window=None,stream=None,context=None):
    rows={}
    for kernel_name,(name,origin) in ORIGINS.items():
        tensor=arguments.get(kernel_name)
        if tensor is None:
            rows[kernel_name]=dict(available=False,reason='operand-not-bound');continue
        try:
            view=tensor_view(tensor)
            view.update(available=True,argument=name,declared_owner=origin,
                allocation_incarnation=window.identify(view) if window else dict(available=False,reason='history-unavailable'),
                intended_access='write' if kernel_name=='q_out_ptr' else 'read/write' if kernel_name in ('state_cache_ptr','compressed_cache_ptr') else 'read',
                consumer_stream=stream,consumer_context=context,
                last_writer=dict(observed=False,declared_provider=origin,
                    scope='source ownership; no device write-trace or value readback'))
            rows[kernel_name]=view
        except Exception as exc:rows[kernel_name]=dict(available=False,error_type=type(exc).__name__)
    return rows


def direct_bounds(arguments,rows):
    checks=[]
    def access(name,last_element,extra_bytes=0):
        row=rows.get(name,{})
        if not row.get('available'):
            checks.append(dict(argument=name,verdict='UNKNOWN',reason='storage-unavailable'));return
        if last_element<0:
            checks.append(dict(argument=name,verdict='NO_ACCESS',reason='empty launch extent'));return
        last=row['data_ptr']+(last_element+1)*row['element_size']+extra_bytes
        valid=(row['allocation_base']<=row['data_ptr']<=last<=row['allocation_base']+row['storage_extent']
               and last<=row['view_end'])
        checks.append(dict(argument=name,verdict='PASS' if valid else 'VIOLATION',
                           max_access_end=last,allocation_end=row['allocation_base']+row['storage_extent']))
    try:
        n=int(arguments['num_tokens']);d=int(arguments['D']);heads=int(arguments['HQ'])
        access('q_ptr',(n-1)*int(arguments['q_stride_token'])+heads*d-1)
        access('k_ptr',(n-1)*int(arguments['k_stride_token'])+d-1)
        access('q_out_ptr',(n-1)*int(arguments['q_out_stride_token'])+(heads-1)*int(arguments['q_out_stride_head'])+d-1)
        axis=2*int(arguments['pos_stride_axis']) if arguments['IS_2D_POSITIONS'] else 0
        access('pos_ptr',axis+(n-1)*int(arguments['pos_stride_token']))
        access('q_norm_weight_ptr',d-1);access('k_norm_weight_ptr',d-1)
        access('cos_sin_ptr',d//2-1)
        access('state_table_ptr',(rows['state_table_ptr']['shape'][0]-1)*int(arguments['state_table_stride_req']))
        access('query_start_loc_ptr',rows['query_start_loc_ptr']['shape'][0]-1)
        access('state_slots_ptr',n-1);access('compressed_slots_ptr',n-1)
        access('logical_positions_ptr',n-1)
        access('k_work_metadata_ptr',int(arguments['num_k_work'])*2-1)
        cache_end=(int(arguments['num_state_blocks'])-1)*int(arguments['state_cache_stride_block'])+(int(arguments['STATE_SIZE'])-1)*int(arguments['state_cache_stride_token'])+d-1
        access('state_cache_ptr',cache_end,24 if arguments['CACHE_HAS_ROPE_POS'] else 0)
        comp_end=(int(arguments['num_compressed_blocks'])-1)*int(arguments['compressed_cache_stride_block'])+(int(arguments['COMP_PAGE_SIZE'])-1)*int(arguments['compressed_cache_stride_token'])+d-1
        access('compressed_cache_ptr',comp_end)
    except (KeyError,TypeError,ValueError) as exc:
        checks.append(dict(argument='launch-scalar-contract',verdict='UNKNOWN',error_type=type(exc).__name__))
    indirect=[dict(argument='k_work_metadata_ptr',required='each active request index in [0, min(state table request rows, query_start_loc length-1))'),
        dict(argument='query_start_loc_ptr',required='0 <= query_start < query_end <= num_tokens for each active work item'),
        dict(argument='logical_positions_ptr',required='end position coherent with the active request/chunk'),
        dict(argument='pos_ptr',required='selected RoPE positions within cos_sin table row capacity'),
        dict(argument='state_table_ptr',required='state block validity checked by kernel; position tail and layout still match'),
        dict(argument='compressed_slots_ptr',required='kernel masks to configured compressed block/page bounds')]
    return dict(direct_checks=checks,indirect_requirements=indirect,
        device_values_observed=False,
        verdict='VIOLATION' if any(row['verdict']=='VIOLATION' for row in checks) else 'UNPROVEN',
        scope='direct address arithmetic only; PASS rows do not certify indirect indices or causal safety')
