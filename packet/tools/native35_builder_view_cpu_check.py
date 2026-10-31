"""Real pinned QSAMetadataBuilder on CPU through the shipped native-private group view.

Constructs the actual image QSAMetadataBuilder (raw circular-buffer and
compressed MLA specs, real dataclasses) on torch.device('cpu') with a
VllmConfig stand-in, clones it with the shipped lane_chunk.private_attention_groups
and checks disjoint storage, the byte ledger, fail-closed classification and the
pinned CPU metadata kernel writing native rows into the private view while an
ordinary four-request build writes the stock view. CPU only; never box evidence.
"""
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace as NS

PIN='c6651a0cf27e2ffb23b90eafe21e044df5609ce976f52ebffa7a4fbea688fc04'
TENSORS=('token_to_req_buffer','slot_mapping_buffer','logical_positions_buffer','visible_blocks_buffer','k_work_metadata_buffer')


def _pinned_cpu_builder(torch):
    path=Path('/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/common/qsa_cache.py')
    source=path.read_bytes();assert hashlib.sha256(source).hexdigest()==PIN
    names={'_logical_positions','_logical_to_physical_qsa_slots','circular_qsa_slot_mapping','compressed_qsa_slot_mapping','_build_qsa_metadata_torch'}
    tree=ast.parse(source);body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names];assert len(body)==len(names)
    for fn in body:
        fn.returns=None
        for arg in fn.args.args+fn.args.kwonlyargs:arg.annotation=None
    scope=dict(torch=torch,PAD_SLOT_ID=-1)
    exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(path),'exec'),scope)
    return scope['_build_qsa_metadata_torch']


def run():
    import torch
    from vllm.models.qwen4_exp.common.qsa_cache import QSAMetadataBuilder
    from vllm.v1.kv_cache_interface import CircularBufferSpec,MLAAttentionSpec
    from vllm.v1.worker.utils import AttentionGroup
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from lane_chunk import private_attention_groups,LaneChunkError
    record=dict(schema='native35-builder-view/1',device_execution=False,real_builder_class=QSAMetadataBuilder.__module__+'.QSAMetadataBuilder',
        vllm_config='stand-in: scheduler_config(max_num_batched_tokens=1024,max_num_seqs=8), speculative_config=None, parallel_config(decode_context_parallel_size=1)')
    empty=torch.empty(0,2,dtype=torch.int32);record['empty_tensor_clone_aliases_data_ptr']=bool(empty.clone().data_ptr()==empty.data_ptr())
    cfg=NS(scheduler_config=NS(max_num_batched_tokens=1024,max_num_seqs=8),speculative_config=None,parallel_config=NS(decode_context_parallel_size=1))
    raw_spec=CircularBufferSpec(block_size=4,num_kv_heads=1,head_size=8,dtype=torch.bfloat16)
    comp_spec=MLAAttentionSpec(block_size=64,num_kv_heads=1,head_size=8,dtype=torch.bfloat16,tokens_per_state=4)
    device=torch.device('cpu')
    raw=QSAMetadataBuilder(raw_spec,['model.layers.3.self_attn.indexer.raw_key_cache'],cfg,device)
    comp=QSAMetadataBuilder(comp_spec,['model.layers.3.self_attn.indexer.compressed_key_cache'],cfg,device)
    assert raw.is_circular_buffer and raw.compress_ratio==1 and raw.storage_block_size==4 and raw.k_work_metadata_buffer.numel()==0
    assert not comp.is_circular_buffer and comp.compress_ratio==4 and comp.k_work_metadata_buffer.shape==(262,2)
    for builder in (raw,comp):
        builder.mamba_aligned_state_indices=None  # the lane assigns this field on the view
    stock=[[AttentionGroup(backend=object,layer_names=list(raw.layer_names),kv_cache_spec=raw_spec,kv_cache_group_id=0,metadata_builders=[raw])],
           [AttentionGroup(backend=object,layer_names=list(comp.layer_names),kv_cache_spec=comp_spec,kv_cache_group_id=1,metadata_builders=[comp])]]
    record['real_builder_fields']=sorted(vars(comp))
    try:
        private,nbytes,count,inventory=private_attention_groups(stock,torch_module=torch)
    except LaneChunkError as exc:
        record.update(verdict='FAIL',classification='ENGINE',error='private_attention_groups refused the real pinned builder: '+str(exc))
        return record
    classes={}
    for row in inventory:
        for field in row['fields']:classes.setdefault(field['classification'],[]).append(field['name'])
    record.update(private_builder_bytes=nbytes,private_builder_tensor_count=count,field_classes={k:sorted(set(v)) for k,v in classes.items()})
    expected_bytes=sum(int(getattr(b,n).numel())*int(getattr(b,n).element_size()) for b in (raw,comp) for n in TENSORS)
    assert nbytes==expected_bytes and count==len(TENSORS)*2,(nbytes,expected_bytes,count)
    views=[(raw,private[0][0].get_metadata_builder(0)),(comp,private[1][0].get_metadata_builder(0))]
    for s,p in views:
        assert type(p) is QSAMetadataBuilder and p is not s and p.layer_names==s.layer_names and p.layer_names is not s.layer_names
        for name in TENSORS:
            a,b=getattr(s,name),getattr(p,name)
            assert a.shape==b.shape and a.dtype==b.dtype and (a.numel()==0 or a.data_ptr()!=b.data_ptr()),name
        assert p.kv_cache_spec is s.kv_cache_spec and p.vllm_config is s.vllm_config and p.device==s.device
        assert (p.compress_ratio,p.storage_block_size,p.request_capacity,p.reorder_batch_threshold)==(s.compress_ratio,s.storage_block_size,s.request_capacity,s.reorder_batch_threshold)
        p.mamba_aligned_state_indices='native-only';assert s.mamba_aligned_state_indices is None
    assert set(classes)<={'private-tensor','private-container','shared-read-only'},set(classes)
    assert set(TENSORS)<=set(classes['private-tensor']) and 'layer_names' in classes['private-container']
    # Fail closed: an unclassifiable mutable object on the real builder refuses the view.
    comp.unclassified_native35_probe=object()
    try:
        private_attention_groups(stock,torch_module=torch);record['fail_closed']='MISSING';raise AssertionError('unclassified field accepted')
    except LaneChunkError as exc:record['fail_closed']='PASS: '+str(exc)
    finally:del comp.unclassified_native35_probe
    # Pinned CPU metadata kernel: native rows into the private view, ordinary rows into stock.
    build=_pinned_cpu_builder(torch)
    def common(q,seq,table):
        query=torch.tensor(q,dtype=torch.int32);lens=torch.tensor(seq,dtype=torch.int32);n=q[-1]
        owner=CommonAttentionMetadata(query,query,lens,len(seq),n,max(y-x for x,y in zip(q,q[1:])),max(seq),torch.tensor(table,dtype=torch.int32),torch.zeros(n,dtype=torch.int64))
        owner._token_to_req_indices_cache=torch.repeat_interleave(torch.arange(len(seq),dtype=torch.int32),torch.diff(query).long())
        return owner
    _,praw=views[0];_,pcomp=views[1]
    native=common([0,1024],[1024],[list(range(64))]);decode=common([0,1,2,3,4],[64]*4,[[11],[12],[13],[14]])
    _,plogical,_,pslots=build(native,pcomp.token_to_req_buffer,pcomp.logical_positions_buffer,pcomp.visible_blocks_buffer,pcomp.slot_mapping_buffer,
        storage_block_size=pcomp.storage_block_size,compress_ratio=pcomp.compress_ratio,k_work_metadata_buffer=pcomp.k_work_metadata_buffer[:256])
    before=(plogical.tolist(),pslots.tolist(),pcomp.k_work_metadata_buffer[:256].tolist())
    build(decode,comp.token_to_req_buffer,comp.logical_positions_buffer,comp.visible_blocks_buffer,comp.slot_mapping_buffer,
        storage_block_size=comp.storage_block_size,compress_ratio=comp.compress_ratio,k_work_metadata_buffer=comp.k_work_metadata_buffer[:4])
    after=(plogical.tolist(),pslots.tolist(),pcomp.k_work_metadata_buffer[:256].tolist())
    assert before==after and after[2][:4]==[[0,0],[0,1],[0,2],[0,3]] and comp.k_work_metadata_buffer[:4,0].tolist()==[0,1,2,3]
    assert plogical.tolist()==list(range(1024)) and not torch.cuda.is_initialized()
    record.update(verdict='PASS',classification=None,native_view_intact_after_ordinary_build=True,ordinary_stock_request_ids=[0,1,2,3],
        scope='real pinned builder class and fields on CPU; shipped private_attention_groups; pinned CPU metadata kernel; no device allocation or kernel launch')
    return record
