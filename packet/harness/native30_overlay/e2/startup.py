"""Discover the pinned ordinary-engine E2 binding without allocating a KV window.

No serving actions. install() is called after the real init_kv_cache completes.
Feature off and profiling return before backend/stream construction. Config
rejection is intentionally narrow; E3/green overlap is a separate design.
"""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import weakref

from .arena_dma import ArenaDMA, TorchArenaBackend
from .block_mirror import CpuTables, WorkerMirror
from .retention_contract import ContractError
from .runner_lifecycle import RunnerLifecycle


PAGE_TOKENS = 4048
ROW_BYTES = 1024


def flag(env, name, default='0'):
    value = env.get(name, default)
    if value not in ('0','1'):
        raise ContractError('invalid boolean setting: ' + name)
    return value == '1'


@dataclass(frozen=True, repr=False)
class Discovery:
    layer_groups: dict
    layers: dict
    hosts: dict
    uva: dict
    arena_pages: int
    arena_budget_bytes: int
    num_blocks: int
    pp: int


def check_config(runner, env):
    p = runner.parallel_config
    if (p.tensor_parallel_size, p.pipeline_parallel_size, p.data_parallel_size,
            p.decode_context_parallel_size) != (2,2,1,1):
        raise ContractError('E2 requires TP2 PP2 DP1 DCP1')
    if env.get('VLLM_PP_LAYER_PARTITION') != '24,24':
        raise ContractError('E2 requires explicit PP24/24')
    for name in ('speculator','speculative_config','adaptive_verification','pcp_manager',
                 'ubatch_runner','fast_prefill','lora_config'):
        if getattr(runner, name, None) is not None:
            raise ContractError('unsupported E2 runner path: ' + name)
    cfg = runner.vllm_config
    for name in ('kv_transfer_config','ec_transfer_config'):
        if getattr(cfg, name, None) is not None:
            raise ContractError('E2 disallows external cache writers/connectors: ' + name)
    if runner.is_encoder_decoder or runner.is_pooling_model:
        raise ContractError('E2 requires ordinary causal language-model generation')
    mm = getattr(runner.model_config, 'multimodal_config', None)
    if mm is not None and not mm.language_model_only:
        raise ContractError('first E2 candidate requires language-model-only')
    if runner.decode_query_len != 1:
        raise ContractError('E2 requires one decode token, no speculation')
    if runner.scheduler_config.max_num_batched_tokens != 1024:
        raise ContractError('first E2 candidate requires 1024-token chunk budget')
    cc = runner.compilation_config
    if int(cc.mode) != 0 or cc.cudagraph_mode.name not in ('NONE','FULL_DECODE_ONLY'):
        raise ContractError('E2 requires eager prefill; ordinary decode graphs may stay enabled')
    for key in ('QSA_RETAIN_HISTORY','FLASH_S4_SEGMENT','VLLM_USE_BREAKABLE_CUDAGRAPH','FLASH_PP_GROUPING'):
        if flag(env, key):
            raise ContractError('incompatible E2 experiment: ' + key)
    if env.get('CUDA_MPS_PIPE_DIRECTORY'):
        raise ContractError('MPS is outside the serial E2 candidate')
    if not flag(env, 'VLLM_QSA_KV_OFFLOAD'):
        raise ContractError('E2 requires host QSA KV')
    if runner.cache_config.kv_sharing_fast_prefill:
        raise ContractError('E2 rejects fast-prefill cache sharing')
    if not isinstance(getattr(runner.block_tables, '_flash_e2_tables', None), CpuTables):
        raise ContractError('CPU table hook is not installed')
    if runner.block_tables._flash_e2_tables.rows:
        raise ContractError('E2 cannot attach to a running worker')


def discover(runner, env):
    check_config(runner, env)
    if (runner.is_first_pp_rank, runner.is_last_pp_rank) == (True,False):
        pp = 0
    elif (runner.is_first_pp_rank, runner.is_last_pp_rank) == (False,True):
        pp = 1
    else:
        raise ContractError('invalid two-stage PP placement')
    expected = set(range(3+24*pp,24+24*pp,4))
    ctx = runner.compilation_config.static_forward_context
    names = {n for n,l in ctx.items() if getattr(l,'_qsa_host_kv',None) is not None}
    if len(names) != 6:
        raise ContractError('expected exactly six local QSA host histories')
    groups = {}; indices = set(); layers = {}; hosts = {}; uva = {}; budgets = set()
    tables = runner.block_tables._flash_e2_tables
    for g,group in enumerate(runner.kv_cache_config.kv_cache_groups):
        for name in group.layer_names:
            if name not in names:
                continue
            if name in groups:
                raise ContractError('QSA layer appears in multiple cache groups')
            matches = re.findall(r'(?:^|\.)layers\.(\d+)\.', name)
            if len(matches) != 1:
                raise ContractError('unrecognized QSA layer name structure')
            index = int(matches[0]); indices.add(index)
            layer = ctx[name]
            if index not in expected or getattr(layer,'layer_name',None) != name:
                raise ContractError('loaded QSA layer does not match PP24/24 placement')
            if getattr(layer,'_flash_e2_lifecycle_ref',None) is not None:
                raise ContractError('QSA layer is already bound to E2')
            if not layer._qsa_kv_offload or layer._qsa_stage_min_rows <= 0:
                raise ContractError('host KV/staged prefill is disabled on a QSA layer')
            if g >= len(tables.bpk) or tables.bpk[g] != 1:
                raise ContractError('expanded QSA block tables are unsupported')
            if group.kv_cache_spec.block_size != PAGE_TOKENS:
                raise ContractError('QSA scheduler page size changed')
            groups[name] = g; layers[name] = layer
            hosts[name] = layer._qsa_host_kv; uva[name] = layer.kv_cache
            budget = layer._qsa_stage_arena_bytes
            if type(budget) is not int or budget <= 0:
                raise ContractError('invalid baseline arena budget')
            budgets.add(budget)
    if set(groups) != names or indices != expected or len(budgets) != 1:
        raise ContractError('incomplete/aliased QSA placement or inconsistent arena budget')
    budget = budgets.pop()
    # This is EXACTLY qsa_staged_arena_pages, not the slice's hard-coded 23.
    # 100663296 // (4048*1024) == 24; a 50-page engine prompt uses 24/24/2.
    pages = max(1,budget//(PAGE_TOKENS*ROW_BYTES))
    return Discovery(groups,layers,hosts,uva,pages,budget,runner.kv_cache_config.num_blocks,pp)


class Switch:
    """A/B convention used by retain_ab: env enables, an owned off file disables.

    Read once per RPC. This class only reads existence; it never creates,
    removes or changes any file. The slot owner owns the named switch file.
    """
    def __init__(self, env):
        self.enabled = flag(env,'FLASH_E2_DMA')
        self.path = env.get('FLASH_E2_OFF_FILE','/work/e2-arena.off')
        if self.path != '/work/e2-arena.off':
            raise ContractError('E2 off-file must be the declared /work/e2-arena.off')

    def __call__(self):
        return self.enabled and not Path(self.path).exists()


def install(runner, is_profiling=False, *, env=None, backend_factory=TorchArenaBackend, emit=None,
            exact_backend_factory=None, tp_rank=None):
    env = os.environ if env is None else env
    if not flag(env,'FLASH_E2_ARENA') or is_profiling:
        return None
    if getattr(runner,'_flash_e2_lifecycle',None) is not None:
        raise ContractError('E2 runner is already initialized')
    found = discover(runner, env)
    switch = Switch(env)
    verify = flag(env,'FLASH_E2_VERIFY')
    memory_stats = flag(env,'FLASH_E2_MEMORY_STATS')
    if memory_stats and not verify:
        raise ContractError('E2 allocator diagnostics require FLASH_E2_VERIFY=1')
    directory = env.get('FLASH_E2_REPORT_DIR','')
    if verify and (not directory.startswith('/work/logs/e2/') or '..' in Path(directory).parts):
        raise ContractError('exactness requires an owned directory under /work/logs/e2/')
    backend = backend_factory(runner.device)
    mirror = WorkerMirror(1,found.num_blocks,found.layer_groups,PAGE_TOKENS,found.arena_pages)
    dma = ArenaDMA(mirror,found.hosts,backend)
    # The actual baseline UVA view must describe the same loaded history shape.
    # We do not assume host/device pointer values are numerically identical.
    for name, tensor in found.uva.items():
        facts = backend.facts(tensor); host = dma.host_facts[name]
        if (facts.device != backend.device or facts.shape != host.shape or
                facts.strides != host.strides or facts.dtype != host.dtype or
                facts.item_bytes != host.item_bytes or not facts.contiguous):
            raise ContractError('baseline UVA binding differs from registered host source')
    if emit is None:
        from vllm.logger import init_logger
        logger = init_logger(__name__)
        emit = lambda line: logger.info('%s',line)
    lifecycle = RunnerLifecycle(mirror,dma,switch,emit)
    lifecycle.uva_facts = {name:backend.facts(tensor) for name,tensor in found.uva.items()}
    if verify:
        from .exactness import Exactness, TorchExactBackend
        if tp_rank is None:
            from vllm.distributed import get_tp_group
            tp_rank = get_tp_group().rank_in_group
        factory = TorchExactBackend if exact_backend_factory is None else exact_backend_factory
        oracle = Exactness(directory,f'PP{found.pp}_TP{tp_rank}',found.layers,factory(),emit,
                           page_tokens=PAGE_TOKENS,arena_pages=found.arena_pages,
                           memory_stats=memory_stats)
        lifecycle.exactness = oracle
        dma.verify_bucket = oracle.bucket
    # No strong reference from layers back to the lifecycle/host tensors.
    ref = weakref.ref(lifecycle)
    for name,layer in found.layers.items():
        layer._flash_e2_lifecycle_ref = ref
        layer._flash_e2_layer_name = name
    runner._flash_e2_lifecycle = lifecycle
    emit('FLASH_E2_STARTUP ' + json.dumps(dict(
        pp=found.pp,local_qsa_layers=sorted(found.layer_groups),
        local_qsa_groups=found.layer_groups,pool_blocks=found.num_blocks,
        page_tokens=PAGE_TOKENS,arena_pages=found.arena_pages,
        arena_budget_bytes=found.arena_budget_bytes,baseline_arena_bytes=mirror.arena.storage_bytes,
        incremental_kv_bytes=0,host_sources_bytes=sum(f.nbytes for f in dma.host_facts.values()),
        source='existing baseline arena; no pool/row-cache/arena cuts',
        dma_env_enabled=switch.enabled,off_file=switch.path,
        exactness_enabled=verify,
        memory_stats_enabled=memory_stats,
        reserve_bytes=536870912,ledger_status='reserve requires simultaneous peak measurement',
        additional_resources='one copy stream, completion events, bounded CPU metadata; measure runtime overhead'
    ),sort_keys=True))
    return lifecycle
