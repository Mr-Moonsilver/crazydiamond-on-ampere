"""Actual pinned CPU Scheduler factory, copied from the native17 host proof.

CPU HMA metadata configuration and model-spec names are explicit geometry
leaves, not CUDA/TP/PP initialization. All stock configuration/allocator calls
execute; no production class or method is patched.
"""
import copy
from e3_native_cow_lease_probe import config

def with_actual_scheduler(run):
    import torch
    from vllm.config import set_current_vllm_config,SchedulerConfig
    from vllm.distributed import init_distributed_environment,initialize_model_parallel
    from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpForConditionalGeneration as Model
    from vllm.v1.kv_cache_interface import MambaSpec,FullAttentionSpec
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum as Kind
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionBackend
    from vllm.v1.attention.backends.short_conv_attn import PleShortConvAttentionBackend
    from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAFlashAttentionBackend
    from vllm.models.qwen4_exp.common.qsa_cache import QSAKeyStateCache,QSACompressedKeyCache,QSAStateBackend
    from vllm.v1.core.kv_cache_utils import generate_scheduler_kv_cache_config,resolve_kv_cache_block_sizes,update_kv_cache_capacity,init_none_hash,get_request_block_hasher
    from vllm.v1.structured_output import StructuredOutputManager
    from vllm.v1.request import Request
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import get_hash_fn_by_name
    from vllm.v1.flash_e3 import scheduler_hooks as hooks
    from e3_native_transitions_lifecycle_probe import cache_configuration
    cfg=config()
    host=copy.copy(cfg)
    host.scheduler_config=SchedulerConfig(max_model_len=512,is_encoder_decoder=False,max_num_batched_tokens=128,
        max_num_seqs=8,async_scheduling=True,disable_hybrid_kv_cache_manager=False)
    with set_current_vllm_config(host):
        init_distributed_environment(world_size=1,rank=0,local_rank=0,distributed_init_method='file:///tmp/native18-cpu-dist',backend='gloo')
        initialize_model_parallel(1,1,backend='gloo')
        gdn,ple=Model.get_mamba_specs_from_config(host)
        def state(s,k):return MambaSpec(block_size=4048,shapes=tuple(s.shapes),dtypes=tuple(s.dtypes),mamba_type=k,mamba_cache_mode='align',tp_replicated=s.tp_replicated)
        specs={'shape.gdn0':state(gdn,Kind.GDN_ATTN),'shape.gdn2':state(gdn,Kind.GDN_ATTN),
            'shape.ple':state(ple,Kind.SHORT_CONV),
            'shape.qsa':Qwen4ExpQSAFlashAttentionBackend.customize_spec(FullAttentionSpec(block_size=4048,
                num_kv_heads=1,head_size=256,head_size_v=256,dtype=torch.bfloat16))}
        caches=[QSAKeyStateCache(head_size=64,dtype=torch.bfloat16,cache_config=host.cache_config,
            prefix='shape.indexer.raw_key_cache',vllm_config=host,compress_ratio=4,cache_rope_positions=host.model_config.uses_mrope),
            QSACompressedKeyCache(head_size=64,dtype=torch.bfloat16,cache_config=host.cache_config,
                prefix='shape.indexer.compressed_key_cache',vllm_config=host,compress_ratio=4)]
        for owner in caches:specs[owner.prefix]=owner.get_kv_cache_spec(host)
        worker_kv,cache_record=cache_configuration(host,specs,(GDNAttentionBackend,PleShortConvAttentionBackend,Qwen4ExpQSAFlashAttentionBackend,QSAStateBackend))
        kv=generate_scheduler_kv_cache_config([worker_kv]);host.cache_config.num_gpu_blocks=kv.num_blocks
        host.cache_config.block_size=min(g.kv_cache_spec.block_size for g in kv.kv_cache_groups if g.kv_cache_spec.prefix_cacheable)
        update_kv_cache_capacity(host,kv);host.validate_block_size();block,hash_block=resolve_kv_cache_block_sizes(kv,host)
        Scheduler=host.scheduler_config.get_scheduler_cls();scheduler=Scheduler(host,kv,StructuredOutputManager(host),block_size=block,hash_block_size=hash_block)
        guard=hooks.SchedulerLeases().attach(scheduler,scheduler.kv_cache_manager.block_pool)
        hash_fn=get_hash_fn_by_name('sha256');init_none_hash(hash_fn);hasher=get_request_block_hasher(hash_block,hash_fn)
        from types import SimpleNamespace
        cpu_core=SimpleNamespace(mm_receiver_cache=None,request_block_hasher=hasher,hash_block_size=hash_block)
        return run(host, scheduler, guard, cpu_core, cache_record)
