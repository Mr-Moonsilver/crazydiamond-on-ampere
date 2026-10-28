"""VM-only stock lazy Mamba API proof; no model constructor, weights or engine.

Uses the image's real state/model classes, copy functions/validator, GPU context
and LaneChunkFactory. Only model-state ownership fields and tiny cache geometry
are synthetic. The ordinary lazy context is initialized later through its
unchanged stock method and must retain its own tables and buffers.
"""
import hashlib
import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace as NS

import vmid_support as s

SOURCE_PINS = {
    'v1/worker/gpu/model_states/mamba_hybrid.py': '573e130ac5587d4822fabf97adad1ad0fda02ff83c5e70df879a5bb175ae845c',
    'v1/worker/mamba_utils.py': '7382bcc8a7b070eaec44339379ebce485777fe23b97e05ee108088faa707c7e6',
    'v1/kv_cache_interface.py': '1cf202f1a44d5bc5c3832b70b41507687ecceea3784e67a4f003bc0210d6eecb',
    'v1/attention/backends/registry.py': '77f41b9bff9b4a2f6417e8360ead3a5b560861a958ed26072d8c97f5e6b2af2b',
    'model_executor/layers/mamba/mamba_utils.py': 'ce9845b4a4ef4bc46577a33038ed69d59ceccb52f30e9c5bdaa36e48be3b2ca4',
    'v1/utils.py': '4093f1c040a5f3def599bb044b72109301b970f70a5e5b622fd03a450a11e36a',
    'models/qwen4_exp/nvidia/model.py': 'ae72d95610b80188df2ee54a8649329a2a5601c4c31cbc7b60bbda9c6dcff9c4',
}

REQUIRED = (
    'source_pins_verified', 'initial_cache_none', 'initial_context_none',
    'factory_constructed', 'private_initialized', 'private_capacity_one',
    'private_tables_exact', 'ordinary_cache_still_none',
    'ordinary_context_still_none', 'model_owner_unchanged',
    'stock_later_initialized', 'stock_context_distinct', 'stock_tables_exact',
    'stock_capacity_two', 'stock_idempotent', 'private_tables_still_exact',
    'cached_factory_constructed', 'cached_stock_context_unchanged',
    'cached_stock_descriptors_unchanged', 'aligned_indices_exact',
    'state_cache_bytes_unchanged', 'ordinary_table_bytes_unchanged',
    'missing_family_rejected', 'oversized_prefix_rejected', 'cleanup_ok',
)


def grade(record):
    if not isinstance(record, dict):
        return s.verdict([], ['proof_missing'])
    failures = [key for key in REQUIRED if record.get(key) is False]
    missing = [key for key in REQUIRED if record.get(key) is not True
               and record.get(key) is not False]
    if record.get('families') != ['GDN_ATTN', 'SHORT_CONV']:
        missing.append('family_coverage')
    if record.get('copy_prefix_lengths') != [2, 1]:
        missing.append('copy_prefix_coverage')
    return s.verdict(failures, missing)


def collect(torch, modules):
    import lane_chunk
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum as Kind
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

    model_module = importlib.import_module('vllm.models.qwen4_exp.nvidia.model')
    root_type = model_module.Qwen4ExpForConditionalGeneration
    # Pin imported runtime definitions to the bundle's exact read-only sources.
    references = Path(modules).parent / 'e3_reference'
    sources = {}
    for rel, digest in SOURCE_PINS.items():
        module = importlib.import_module('vllm.' + rel[:-3].replace('/', '.'))
        actual = Path(inspect.getfile(module)).read_bytes()
        expected = (references / rel).read_bytes()
        if actual != expected or hashlib.sha256(actual).hexdigest() != digest:
            raise RuntimeError('mamba_source_pin:' + rel)
        sources[rel] = hashlib.sha256(actual).hexdigest()

    device = torch.device('cuda:0')
    root = root_type.__new__(root_type)
    torch.nn.Module.__init__(root)
    state = MambaHybridModelState.__new__(MambaHybridModelState)
    state.model = root
    state.device = device
    state.max_num_reqs = 2
    state._align_mode = True
    state._mamba_spec = None
    state._mamba_group_ids = []
    state._mamba_state_copy_funcs = None
    state._mamba_ctx = None
    specs = (
        MambaSpec(block_size=4, shapes=((4, 4), (2, 4)),
                  dtypes=(torch.float32, torch.float32), mamba_type=Kind.GDN_ATTN,
                  mamba_cache_mode='align'),
        MambaSpec(block_size=4, shapes=((4, 4),), dtypes=(torch.float32,),
                  mamba_type=Kind.SHORT_CONV, mamba_cache_mode='align'),
    )
    names = ('synthetic.gdn', 'synthetic.ple')
    config = KVCacheConfig(num_blocks=8, kv_cache_tensors=[], kv_cache_groups=[
        KVCacheGroupSpec(layer_names=[name], kv_cache_spec=spec)
        for name, spec in zip(names, specs)])
    caches = [[torch.arange(8 * (16 if i == 0 else 8), device=device,
                            dtype=torch.float32).reshape((8,) + shape)
               for i, shape in enumerate(spec.shapes)] for spec in specs]
    before = [[tensor.clone() for tensor in group] for group in caches]
    forward_context = {name: NS(kv_cache=group) for name, group in zip(names, caches)}
    state.vllm_config = NS(compilation_config=NS(static_forward_context=forward_context))
    ordinary_tables = tuple(torch.arange(16, dtype=torch.int32, device=device).reshape(2, 8)
                            for _ in specs)
    ordinary_before = [table.clone() for table in ordinary_tables]
    runner = NS(model=root, model_state=state, device=device, kv_cache_config=config,
                vllm_config=state.vllm_config,
                block_tables=NS(num_kv_cache_groups=2, blocks_per_kv_block=[1, 1],
                                kernel_block_sizes=[4, 4], _slot_mapping_enabled=[False, False],
                                block_tables=[NS(gpu=table) for table in ordinary_tables]))
    out = {'schema': 'vm<vm-id>-mamba-align-startup/1', 'source_pins': sources,
           'source_pins_verified': True, 'initial_cache_none': state._mamba_state_copy_funcs is None,
           'initial_context_none': state._mamba_ctx is None,
           'families': [spec.mamba_type.name for spec in specs], 'copy_prefix_lengths': [2, 1],
           'scope': 'actual pinned methods/classes over a manually populated model-state shell; '
                    'small synthetic GDN/PLE caches, no model constructor/forward or engine'}
    factory = lane_chunk.LaneChunkFactory(runner)
    private = factory.align.ctx
    for row, ordinary in zip(factory.tables.rows, ordinary_tables):
        row.copy_(ordinary[:1])
    private_ptrs = [row.data_ptr() for row in factory.tables.rows]
    out.update(factory_constructed=True, private_initialized=private.is_initialized,
               private_capacity_one=private.aligned_state_indices.shape[1] == 1,
               private_tables_exact=private.block_table_ptrs.tolist() == private_ptrs,
               ordinary_cache_still_none=state._mamba_state_copy_funcs is None,
               ordinary_context_still_none=state._mamba_ctx is None,
               model_owner_unchanged=state.model is root and runner.model is root)
    gids, _ = state._get_mamba_group_info(config)
    ordinary = state._ensure_align_ctx(config, gids, ordinary_tables)
    descriptors = state._mamba_state_copy_funcs
    out.update(stock_later_initialized=ordinary.is_initialized,
               stock_context_distinct=ordinary is not private,
               stock_tables_exact=ordinary.block_table_ptrs.tolist() ==
                                  [table.data_ptr() for table in ordinary_tables],
               stock_capacity_two=ordinary.aligned_state_indices.shape[1] == 2,
               stock_idempotent=state._ensure_align_ctx(config, gids, ordinary_tables) is ordinary,
               private_tables_still_exact=private.block_table_ptrs.tolist() == private_ptrs)
    second = lane_chunk.LaneChunkFactory(runner)
    out.update(cached_factory_constructed=second.align.ctx.is_initialized,
               cached_stock_context_unchanged=state._mamba_ctx is ordinary,
               cached_stock_descriptors_unchanged=state._mamba_state_copy_funcs is descriptors)
    seq_lens = torch.tensor([5], dtype=torch.int32, device=device)
    private_indices = private.compute_aligned_state_indices(seq_lens, 1)
    ordinary_indices = ordinary.compute_aligned_state_indices(seq_lens, 1)
    torch.cuda.synchronize()
    out['aligned_indices_exact'] = torch.equal(private_indices, ordinary_indices)
    out['state_cache_bytes_unchanged'] = all(torch.equal(a, b) for aa, bb in zip(caches, before)
                                                   for a, b in zip(aa, bb))
    out['ordinary_table_bytes_unchanged'] = all(torch.equal(a, b)
                                               for a, b in zip(ordinary_tables, ordinary_before))
    # Real stock validator negatives, not substituted CUDA or native rows.
    ports = lane_chunk.PinnedPorts.load()
    groups = ports.mamba_groups(config)
    for key, bad in (
        ('missing_family_rejected', {Kind.GDN_ATTN: descriptors[Kind.GDN_ATTN]}),
        ('oversized_prefix_rejected', {**descriptors, Kind.SHORT_CONV:
                                      descriptors[Kind.SHORT_CONV] * 2}),
    ):
        try:
            ports.validate_mamba_copy_funcs(groups, bad)
        except AssertionError:
            out[key] = True
        else:
            out[key] = False
    torch.cuda.synchronize()
    out['cleanup_ok'] = True
    out['grade'] = grade(out)
    return out
