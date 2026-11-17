#!/usr/bin/env python3
"""CPU cost witness for the stock V2 decode-step attention-metadata build.

Runs INSIDE the pinned image (sha256:ad067db5...), CPU only, no GPU, no network.
It executes the real ``MambaHybridModelState.prepare_attn`` (the anatomy seam
``prep_attn_meta``), which calls the real ``attn_utils.build_attn_metadata``,
which calls the real metadata builders (GDN, PLE short-conv, Qwen4Exp QSA flash,
QSA state ring/compressed) built by the real ``AttentionGroup.create_metadata_builders``
from a real ``VllmConfig`` made from the real model config.

Every fixture leaf (what is not the box's real object) is listed in LEAVES with
the real object it replaces and why it is equivalent for this measurement; the
record carries the list.  Two passes:

* ``cpu``  : tensors that live on the GPU on the box live on the CPU here;
             gives host wall time per call / per builder and real values.
* ``meta`` : the same tensors on PyTorch's meta device; every aten op on a meta
             tensor is an op the box issues to the GPU (a kernel or memcpy
             launch), every op on a CPU tensor is host-only; a device->host
             read (``.item()``/``.tolist()`` on a device tensor, i.e. a sync on
             the box) raises on meta and is recorded.  Triton launches are
             recorded by a launcher stub (no Triton on CPU).
"""
import argparse
import ast
import cProfile
import gc
import hashlib
import io
import json
import os
import platform
import pstats
import statistics
import sys
import tempfile
import time
import traceback
from pathlib import Path

VLLM = Path('/usr/local/lib/python3.12/dist-packages/vllm')
SOURCES = [
    'v1/worker/gpu/model_states/mamba_hybrid.py',
    'v1/worker/gpu/model_states/default.py',
    'v1/worker/gpu/attn_utils.py',
    'v1/worker/utils.py',
    'v1/worker/mamba_utils.py',
    'v1/attention/backend.py',
    'v1/attention/backends/utils.py',
    'v1/attention/backends/gdn_attn.py',
    'v1/attention/backends/mamba_attn.py',
    'v1/attention/backends/short_conv_attn.py',
    'v1/attention/backends/flash_attn.py',
    'models/qwen4_exp/common/qsa_cache.py',
    'models/qwen4_exp/nvidia/qsa.py',
    'models/qwen4_exp/nvidia/ple_layer.py',
    'models/qwen4_exp/nvidia/model_state.py',
    'v1/kv_cache_interface.py',
    'v1/core/kv_cache_utils.py',
    'platforms/cuda.py',
]

# Box facts (results/box/q18a-stage1-sweep, p0-d44 engine.log + identity):
BOX = dict(
    max_model_len=262144, max_num_seqs=10, max_num_batched_tokens=1024,
    block_size=4048, ring_block_size=4, compress_ratio=4, num_blocks=1340,
    cudagraph_capture_sizes=[1, 2, 3, 4, 6, 8, 9, 12, 15, 18, 21, 24],
    max_cudagraph_capture_size=24, cudagraph_mode='FULL_DECODE_ONLY',
    mamba_cache_mode='align', speculative_config=None, tp=2, pp=2,
    pp_partition=(24, 24), num_kv_cache_groups=39,
    group_sizes='[4048]*37 + [4] + [4048]', host_kv_group=38,
)

LEAVES = [
    dict(leaf='device', real='CUDA tensors on the box (cuda:N, sm86)',
         fixture="pass 'cpu': torch.device('cpu'); pass 'meta': torch.device('meta')",
         why='Builder code is device-agnostic Python + aten ops; host cost of the Python and the dispatcher is the same code. '
             'The CUDA launch cost per device op is NOT measured here; it is counted (meta pass) and modelled separately.'),
    dict(leaf='triton launches', real='Triton JIT launcher for _build_qsa_metadata_kernel (and any other @triton.jit on the path)',
         fixture='launcher stub that records (kernel, grid) and returns; build_qsa_metadata forced to build_qsa_metadata_triton (the box resolves HAS_TRITON=True on CUDA)',
         why='No Triton backend on CPU. The Python before/after the launch is real; the launcher cost itself is modelled separately.'),
    dict(leaf='VllmConfig', real="engine VllmConfig from the box's launch args",
         fixture='real VllmConfig/ModelConfig/CacheConfig/SchedulerConfig/CompilationConfig built from reference/model-config.json (quantization_config dropped) with the box values in BOX; builder attributes that derive from the platform (use_full_cuda_graph, decode_cudagraph_max_bs, max_cudagraph_size, aot_schedule) asserted/forced to the box value and recorded',
         why='Builders read only scheduler sizes, cudagraph mode/capture size, cache mode, speculative config and head counts from the config; all are set to the box values. Quantization does not reach the builders.'),
    dict(leaf='KV cache groups / attention groups', real="the box's 39-group KVCacheConfig (kv_cache_utils._get_packed_kv_cache_groups over 48 layers) projected per PP worker, and runner.attn_groups from attn_utils.init_attn_backend",
         fixture='per-stage KVCacheConfig with 39 groups built directly: 36 one-layer GDN MambaSpec groups (18 local), 1 PLE MambaSpec group (stage 0 only), 1 CircularBufferSpec ring group (block 4, 6 local layers), 1 attention group (UniformTypeKVCacheSpecs: 6 FullAttentionSpec + 6 MLAAttentionSpec compressed); AttentionGroups made with the real class, keyed like init_attn_backend (backend, per-layer spec), builders made by the real create_metadata_builders with kernel block sizes from the real prepare_kernel_block_sizes',
         why='Matches the box log (39 groups; sizes 37x4048, 4, 4048; host KV views in group 38) and the packing source. Spec fields the builders read (block_size, num_speculative_blocks, cache mode, tokens_per_state/num_states) carry box values; state shapes/dtypes are not read by any decode build() and are declared, not derived.'),
    dict(leaf='model_state', real='Qwen4ExpModelState instance on the worker',
         fixture='object.__new__(Qwen4ExpModelState) with the attributes prepare_attn reads (vllm_config, max_model_len, cache_config, _align_mode, num_accepted_tokens_gpu, recoverssm=None, mamba group caches)',
         why='prepare_attn is inherited unchanged from MambaHybridModelState; the constructor only builds rope/ngram state that prepare_attn never reads.'),
    dict(leaf='InputBatch / block tables / slot mappings', real='runner.prepare_inputs + runner.prepare_attn outputs for 8 decode requests in a FULL decode step',
         fixture='real InputBatch dataclass with 8 one-token requests (padded size 8, a capture size), seq_lens from the history set; block tables: per-group int32 [max_num_seqs, width] with distinct ids, sliced [:8] like gather_block_tables; slot mappings [39, 1024] int64 sliced [:, :8]',
         why='prepare_attn and the builders read only shapes, query_start_loc/seq_lens values and the block-table tensors; their contents only change which ids are gathered.'),
    dict(leaf='platform.is_arch_support_pdl', real='CudaPlatform.is_arch_support_pdl: torch.cuda.current_device() + get_device_capability() per QSA build',
         fixture='the CPU platform default (returns False)',
         why='Same return value on sm86 (major 8 < 9); the box pays two small CUDA runtime queries per QSA build (~2 builders/step), not measured here.'),
]

LEAVES.append(dict(leaf='cudagraph_mode', real='FULL_DECODE_ONLY on the box (engine.log config line)',
    fixture='CompilationConfig resolved NONE on the CPU platform; set back to FULL_DECODE_ONLY on the VllmConfig before any builder is made',
    why='Builders read it at __init__ (use_full_cuda_graph) and BaseMamba reads it inside build() to stage state_indices_tensor_d; w1 missed that staging (kept as a failed attempt).'))
LEAVES.append(dict(leaf='Qwen4ExpQSAMetadataBuilder class text', real='the box overlay qsa.py (stage1-sweep build r6 final/qsa.py), whose class adds build(): super().build() + an _QSA_HOST_META branch',
    fixture="the overlay's class source compiled in the image module's namespace with _QSA_HOST_META=False",
    why='The image class lacks that build(); the overlay module imports the e3 lane, which is not mounted. The box sets no QSA_HOST_META, so the branch is not taken there either.'))

TRITON_LAUNCHES = []
TRITON_HOOK = [None]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def copy_sources(out):
    rows = {}
    dst = out / 'sources'
    dst.mkdir(exist_ok=True)
    for rel in SOURCES:
        p = VLLM / rel
        if p.exists():
            data = p.read_bytes()
            (dst / rel.replace('/', '__')).write_bytes(data)
            rows[rel] = hashlib.sha256(data).hexdigest()
        else:
            rows[rel] = None
    return rows


def class_source(path, name):
    text = Path(path).read_text()
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return ast.get_source_segment(text, node)
    return None


def install_overlay_builder(path):
    """Run the box overlay's Qwen4ExpQSAMetadataBuilder class text (the image's lacks its build()).

    The overlay module itself imports the e3 lane (not mounted); only this class is
    compiled, in the image module's namespace, with _QSA_HOST_META=False (the box sets
    no QSA_HOST_META / QSA_HOST_META_VERIFY: engine identity record).
    """
    import vllm.models.qwen4_exp.nvidia.model  # noqa: F401  (qsa <-> model import cycle: model first)
    import vllm.models.qwen4_exp.nvidia.qsa as q
    src = class_source(path, 'Qwen4ExpQSAMetadataBuilder')
    ns = q.__dict__
    ns.setdefault('_QSA_HOST_META', False)
    assert ns['_QSA_HOST_META'] is False
    before = q.Qwen4ExpQSAMetadataBuilder
    exec(compile(src, str(path) + ':Qwen4ExpQSAMetadataBuilder', 'exec'), ns)
    assert q.Qwen4ExpQSAMetadataBuilder is not before
    assert q.Qwen4ExpQSAFlashAttentionBackend.get_builder_cls() is q.Qwen4ExpQSAMetadataBuilder
    return dict(installed=True, sha256=hashlib.sha256(src.encode()).hexdigest(),
                has_build=('def build' in src))


def install_triton_stub():
    """Record every Triton kernel launch instead of running it."""
    import triton
    from triton.runtime import jit as tjit

    def getitem(self, grid):
        name = getattr(getattr(self, 'fn', None), '__name__', None) or getattr(self, '__name__', repr(self))

        def launch(*args, **kwargs):
            TRITON_LAUNCHES.append(name)
            if TRITON_HOOK[0] is not None:
                TRITON_HOOK[0](name, args, kwargs)
            return None
        return launch

    tjit.KernelInterface.__getitem__ = getitem
    return triton.__version__


def build_config(model_json, work, torch):
    from vllm.config import (CacheConfig, CompilationConfig, DeviceConfig, ModelConfig,
                             ParallelConfig, SchedulerConfig, VllmConfig)
    import vllm.models.qwen4_exp.config as qcfg
    raw = json.loads(Path(model_json).read_text())
    raw.pop('quantization_config', None)
    model_path = Path(tempfile.mkdtemp(prefix='attn-meta-model-', dir=work))
    (model_path / 'config.json').write_text(json.dumps(raw))
    notes = {}
    model = ModelConfig(model=str(model_path), skip_tokenizer_init=True, language_model_only=True,
                        dtype='bfloat16', max_model_len=BOX['max_model_len'], enforce_eager=False)
    try:
        cache = CacheConfig(block_size=BOX['block_size'], mamba_cache_mode=BOX['mamba_cache_mode'],
                            enable_prefix_caching=True)
    except Exception as exc:  # recorded, the builders take block sizes from their specs
        notes['cache_config_fallback'] = repr(exc)[:300]
        cache = CacheConfig(block_size=16, mamba_cache_mode=BOX['mamba_cache_mode'])
    sched = SchedulerConfig(max_model_len=BOX['max_model_len'],
                            max_num_batched_tokens=BOX['max_num_batched_tokens'],
                            max_num_seqs=BOX['max_num_seqs'], is_encoder_decoder=False,
                            enable_chunked_prefill=True)
    comp = CompilationConfig(mode=0, cudagraph_mode=BOX['cudagraph_mode'],
                             cudagraph_capture_sizes=list(BOX['cudagraph_capture_sizes']),
                             max_cudagraph_capture_size=BOX['max_cudagraph_capture_size'])
    try:
        par = ParallelConfig(tensor_parallel_size=BOX['tp'], pipeline_parallel_size=BOX['pp'])
    except Exception as exc:
        notes['parallel_config_fallback'] = repr(exc)[:300]
        par = ParallelConfig()
    try:
        cfg = VllmConfig(model_config=model, cache_config=cache, scheduler_config=sched,
                         parallel_config=par, device_config=DeviceConfig(device='cpu'),
                         compilation_config=comp)
    except Exception as exc:
        notes['vllm_config_parallel_fallback'] = repr(exc)[:300]
        cfg = VllmConfig(model_config=model, cache_config=cache, scheduler_config=sched,
                         parallel_config=ParallelConfig(), device_config=DeviceConfig(device='cpu'),
                         compilation_config=comp)
    from vllm.config.compilation import CUDAGraphMode
    notes['platform_resolved_cudagraph_mode'] = str(cfg.compilation_config.cudagraph_mode)
    if cfg.compilation_config.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY:
        # The CPU platform resolves NONE; the box resolves FULL_DECODE_ONLY (engine.log
        # config line). Builders read it at __init__ AND BaseMamba reads it inside build().
        cfg.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        notes['cudagraph_mode_set_to_box_value'] = True
    notes['resolved'] = dict(
        cudagraph_mode=str(cfg.compilation_config.cudagraph_mode),
        max_cudagraph_capture_size=cfg.compilation_config.max_cudagraph_capture_size,
        mamba_cache_mode=cfg.cache_config.mamba_cache_mode,
        cache_block_size=cfg.cache_config.block_size,
        num_speculative_tokens=cfg.num_speculative_tokens,
        speculative_config=repr(cfg.speculative_config),
        tp=cfg.parallel_config.tensor_parallel_size, pp=cfg.parallel_config.pipeline_parallel_size,
        max_num_seqs=cfg.scheduler_config.max_num_seqs,
        max_num_batched_tokens=cfg.scheduler_config.max_num_batched_tokens,
        max_model_len=cfg.model_config.max_model_len,
        architectures=list(getattr(cfg.model_config.hf_config, 'architectures', []) or []),
    )
    return cfg, notes


def stage_layout(stage, text_cfg):
    """Global layer -> kind, and this stage's layer list (PP partition 24/24)."""
    types = text_cfg['layer_types']
    lo, hi = (0, 24) if stage == 0 else (24, 48)
    gdn_global = [i for i, t in enumerate(types) if t == 'linear_attention']
    full_global = [i for i, t in enumerate(types) if t == 'full_attention']
    return dict(lo=lo, hi=hi, gdn_global=gdn_global, full_global=full_global,
                gdn_local=[i for i in gdn_global if lo <= i < hi],
                full_local=[i for i in full_global if lo <= i < hi],
                ple_local=(stage == 0))


def make_groups(stage, cfg, text_cfg, torch, device, notes):
    from vllm.v1.kv_cache_interface import (CircularBufferSpec, FullAttentionSpec, KVCacheConfig,
                                            KVCacheGroupSpec, MambaSpec, MLAAttentionSpec,
                                            UniformTypeKVCacheSpecs)
    from vllm.v1.worker.utils import AttentionGroup, prepare_kernel_block_sizes
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionBackend
    from vllm.models.qwen4_exp.common.qsa_cache import QSAStateBackend
    try:
        from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
        gdn_type = MambaAttentionBackendEnum.GDN_ATTN
    except Exception as exc:
        notes['gdn_enum'] = repr(exc)[:200]
        gdn_type = None
    try:
        from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAFlashAttentionBackend as FA
        notes['fa_backend'] = 'Qwen4ExpQSAFlashAttentionBackend'
    except Exception as exc:
        notes['fa_backend_import'] = repr(exc)[:400]
        raise
    ple_backend = None
    try:
        from vllm.models.qwen4_exp.nvidia.ple_layer import PleShortConvAttentionBackend as ple_backend
        notes['ple_backend'] = 'ple_layer.PleShortConvAttentionBackend'
    except Exception as exc:
        notes['ple_backend_import'] = repr(exc)[:400]
        from vllm.v1.attention.backends.short_conv_attn import PleShortConvAttentionBackend as ple_backend
        notes['ple_backend'] = 'short_conv_attn.PleShortConvAttentionBackend'
    ple_type = getattr(ple_backend, 'mamba_type', None)
    lay = stage_layout(stage, text_cfg)
    bs = BOX['block_size']
    bf16 = torch.bfloat16
    mkw = dict(block_size=bs, mamba_cache_mode=BOX['mamba_cache_mode'], num_speculative_blocks=0)
    # Shapes/dtypes are declared (not read by any decode build()); TP=2 GDN: conv (10240/2, 3), ssm (24,128,128).
    gdn_spec = MambaSpec(shapes=((3, 5120), (24, 128, 128)), dtypes=(bf16, torch.float32),
                         **({'mamba_type': gdn_type} if gdn_type is not None else {}), **mkw)
    ple_kw = {}
    try:
        from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum as E
        for name in ('PLE_SHORT_CONV', 'SHORT_CONV'):
            if hasattr(E, name):
                ple_kw['mamba_type'] = getattr(E, name)
                break
    except Exception:
        pass
    ple_spec = MambaSpec(shapes=((3, 2560),), dtypes=(bf16,), **ple_kw, **mkw)
    notes['ple_mamba_type'] = repr(ple_kw.get('mamba_type'))
    ring_spec = CircularBufferSpec(block_size=BOX['ring_block_size'], num_kv_heads=1, head_size=140,
                                   head_size_v=0, dtype=bf16)
    fa_spec = FullAttentionSpec(block_size=bs, num_kv_heads=1, head_size=256, head_size_v=256, dtype=bf16,
                                num_head_slots=1, state_content_bytes=2)
    mla_spec = MLAAttentionSpec(block_size=bs, num_kv_heads=1, head_size=128, dtype=bf16,
                                tokens_per_state=BOX['compress_ratio'])
    P = 'language_model.model.layers.'
    groups = []      # KVCacheGroupSpec, global order
    attn = []        # list[list[AttentionGroup]]
    for gi, layer in enumerate(lay['gdn_global']):                       # groups 0..35
        names = [f'{P}{layer}.linear_attn'] if lay['lo'] <= layer < lay['hi'] else []
        groups.append(KVCacheGroupSpec(names, gdn_spec))
        attn.append([AttentionGroup(GDNAttentionBackend, list(names), gdn_spec, gi)] if names else [])
    ple_names = [f'{P}1.ple_short_conv'] if lay['ple_local'] else []        # group 36
    groups.append(KVCacheGroupSpec(ple_names, ple_spec))
    attn.append([AttentionGroup(ple_backend, list(ple_names), ple_spec, 36)] if ple_names else [])
    ring_names = [f'{P}{i}.self_attn.indexer.raw_key_cache' for i in lay['full_local']]   # group 37
    groups.append(KVCacheGroupSpec(ring_names, ring_spec))
    attn.append([AttentionGroup(QSAStateBackend, list(ring_names), ring_spec, 37)])
    fa_names = [f'{P}{i}.self_attn.attn' for i in lay['full_local']]                       # group 38
    mla_names = [f'{P}{i}.self_attn.indexer.compressed_key_cache' for i in lay['full_local']]
    specs = {**{n: fa_spec for n in fa_names}, **{n: mla_spec for n in mla_names}}
    try:
        uni = UniformTypeKVCacheSpecs(block_size=bs, kv_cache_specs=specs)
    except Exception as exc:
        notes['uniform_spec'] = repr(exc)[:300]
        uni = fa_spec
    groups.append(KVCacheGroupSpec(fa_names + mla_names, uni))
    attn.append([AttentionGroup(FA, list(fa_names), fa_spec, 38),
                 AttentionGroup(QSAStateBackend, list(mla_names), mla_spec, 38)])
    kv_cfg = KVCacheConfig(num_blocks=BOX['num_blocks'], kv_cache_tensors=[], kv_cache_groups=groups)
    try:
        kbs = prepare_kernel_block_sizes(kv_cfg, attn)
        notes['kernel_block_sizes_source'] = 'prepare_kernel_block_sizes'
    except Exception as exc:
        notes['kernel_block_sizes_error'] = repr(exc)[:300]
        kbs = [None] * len(attn)
    for gid, gl in enumerate(attn):
        for g in gl:
            g.create_metadata_builders(vllm_config=cfg, device=device,
                                       kernel_block_size=kbs[gid] if gid < len(kbs) else None,
                                       num_metadata_builders=1)
    return kv_cfg, attn, lay, kbs


def force_box_builder_attrs(attn, notes):
    forced = []
    for gl in attn:
        for g in gl:
            b = g.get_metadata_builder(0)
            cls = type(b).__name__
            for name, want in (('use_full_cuda_graph', True), ('decode_cudagraph_max_bs', 10),
                               ('max_cudagraph_size', 24), ('aot_schedule', False)):
                if hasattr(b, name) and getattr(b, name) != want:
                    forced.append((cls, name, repr(getattr(b, name)), repr(want)))
                    setattr(b, name, want)
    notes['forced_builder_attrs'] = sorted(set(forced))


def builders_summary(attn):
    rows = []
    for gid, gl in enumerate(attn):
        for g in gl:
            b = g.get_metadata_builder(0)
            rows.append(dict(group=gid, builder=type(b).__module__ + '.' + type(b).__name__,
                             layers=len(g.layer_names), spec=type(g.kv_cache_spec).__name__,
                             block_size=getattr(b.kv_cache_spec, 'block_size', None)))
    return rows


def make_batch(hist, torch, np, device):
    from vllm.v1.worker.gpu.input_batch import InputBatch
    n = len(hist)
    seq = np.asarray([h + 1 for h in hist], dtype=np.int32)
    qsl = np.arange(n + 1, dtype=np.int32)
    ones = np.ones(n, dtype=np.int32)
    seq_buf = torch.zeros(BOX['max_num_seqs'], dtype=torch.int32, device=device)
    seq_buf[:n] = torch.from_numpy(seq).to(device)
    qsl_buf = torch.zeros(BOX['max_num_seqs'] + 1, dtype=torch.int32, device=device)
    qsl_buf[:n + 1] = torch.from_numpy(qsl).to(device)
    pos = torch.from_numpy(seq.astype(np.int64) - 1).to(device)
    kw = dict(
        req_ids=[f'r{i}' for i in range(n)], num_reqs=n, num_reqs_after_padding=n,
        idx_mapping=torch.arange(n, dtype=torch.int32, device=device), idx_mapping_np=np.arange(n),
        expanded_idx_mapping=torch.arange(n, dtype=torch.int32, device=device),
        expanded_local_pos=torch.zeros(n, dtype=torch.int32, device=device),
        num_scheduled_tokens=ones, num_tokens=n, num_tokens_after_padding=n,
        num_draft_tokens=0, num_draft_tokens_per_req=None,
        query_start_loc=qsl_buf[:n + 1], query_start_loc_np=qsl,
        seq_lens=seq_buf[:n], seq_lens_cpu_upper_bound=torch.from_numpy(seq.copy()),
        dcp_local_seq_lens=None, num_computed_tokens_np=seq - 1,
        prefill_len_np=np.asarray(hist, dtype=np.int32), num_computed_prefill_tokens_np=np.asarray(hist, dtype=np.int32),
        is_prefilling_np=np.zeros(n, dtype=np.bool_), has_prefill=False,
        input_ids=torch.zeros(n, dtype=torch.int32, device=device), positions=pos,
        is_padding=torch.zeros(n, dtype=torch.bool, device=device),
        logits_indices=torch.arange(n, dtype=torch.int32, device=device),
        cu_num_logits=torch.arange(n + 1, dtype=torch.int32, device=device),
        cu_num_logits_np=np.arange(n + 1, dtype=np.int32), has_structured_output_reqs=False,
        prompt_lens=torch.from_numpy(np.asarray(hist, dtype=np.int32)).to(device),
    )
    import dataclasses
    fields = {f.name for f in dataclasses.fields(InputBatch)}
    return InputBatch(**{k: v for k, v in kw.items() if k in fields}), sorted(fields - set(kw))


def make_tables(kv_cfg, cfg, n, torch, device):
    tables = []
    for gid, g in enumerate(kv_cfg.kv_cache_groups):
        spec = g.kv_cache_spec
        bs = spec.block_size
        width = -(-BOX['max_model_len'] // bs) if bs >= 16 else 1
        base = 1 + gid * 4096
        t = (torch.arange(BOX['max_num_seqs'] * width, dtype=torch.int32) + base).reshape(BOX['max_num_seqs'], width)
        tables.append(t.to(device)[:n])
    slots = torch.zeros(len(tables), BOX['max_num_batched_tokens'], dtype=torch.int64, device=device)[:, :n]
    return tuple(tables), slots


def make_mstate(cfg, torch, device, notes):
    try:
        from vllm.models.qwen4_exp.nvidia.model_state import Qwen4ExpModelState as cls
    except Exception as exc:
        notes['model_state_import'] = repr(exc)[:300]
        from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState as cls
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
    assert cls.prepare_attn is MambaHybridModelState.prepare_attn
    m = object.__new__(cls)
    m.vllm_config = cfg
    m.model_config = cfg.model_config
    m.cache_config = cfg.cache_config
    m.max_model_len = cfg.model_config.max_model_len
    m.max_num_reqs = BOX['max_num_seqs']
    m.device = device
    m.num_accepted_tokens_gpu = torch.ones(BOX['max_num_seqs'], dtype=torch.int32, device=device)
    m._align_mode = cfg.cache_config.mamba_cache_mode == 'align'
    m.recoverssm = None
    m._mamba_spec = None
    m._mamba_group_ids = []
    m._mamba_ctx = None
    m._mamba_state_copy_funcs = None
    notes['model_state_class'] = cls.__module__ + '.' + cls.__name__
    return m


class Fixture:
    def __init__(self, stage, device, cfg, text_cfg, torch, np, hist, notes):
        from vllm.config.compilation import CUDAGraphMode
        self.kv_cfg, self.attn, self.lay, self.kbs = make_groups(stage, cfg, text_cfg, torch, device, notes)
        force_box_builder_attrs(self.attn, notes)
        self.batch, self.unset_fields = make_batch(hist, torch, np, device)
        self.tables, self.slots = make_tables(self.kv_cfg, cfg, len(hist), torch, device)
        self.mstate = make_mstate(cfg, torch, device, notes)
        self.mode = CUDAGraphMode.FULL

    def call(self):
        return self.mstate.prepare_attn(self.batch, self.mode, self.tables, self.slots, self.attn,
                                        self.kv_cfg, for_capture=False)


def time_calls(fn, n, warm):
    for _ in range(warm):
        fn()
    out = []
    pc = time.perf_counter_ns
    for _ in range(n):
        t0 = pc()
        fn()
        out.append(pc() - t0)
    return out


def time_cold(fn, n, warm, np):
    """Each timed call preceded (untimed) by touching 48 MiB and churning Python objects."""
    junk = np.ones(48 * 1024 * 1024 // 8)
    objs = [dict(a=i, b=str(i)) for i in range(20000)]
    for _ in range(warm):
        fn()
    out = []
    pc = time.perf_counter_ns
    for _ in range(n):
        junk += 1.0
        sum(len(o) for o in objs)
        t0 = pc()
        fn()
        out.append(pc() - t0)
    return out


def op_micro(torch):
    pc = time.perf_counter_ns
    x = torch.arange(8, dtype=torch.int32)
    y = torch.empty(10, dtype=torch.int32)
    bt = torch.arange(650, dtype=torch.int32).reshape(10, 65)[:8]
    idx = torch.zeros(8, 1, dtype=torch.int64)
    cases = dict(add=lambda: x + 1, floordiv=lambda: x // 4048, copy_=lambda: y[:8].copy_(x),
                 gather=lambda: torch.gather(bt, 1, idx), arange=lambda: torch.arange(1, dtype=torch.int32),
                 slice_view=lambda: y[:8], select_view=lambda: bt[:, 0], empty_fill=lambda: y[10:].fill_(0))
    res = {}
    for k, f in cases.items():
        for _ in range(2000):
            f()
        t = []
        for _ in range(20):
            t0 = pc()
            for _ in range(1000):
                f()
            t.append((pc() - t0) / 1000)
        res[k] = round(min(t) / 1e3, 3)
    return res


def summ(ns):
    s = sorted(ns)
    return dict(n=len(s), mean_us=round(statistics.fmean(s) / 1e3, 2), p50_us=round(s[len(s) // 2] / 1e3, 2),
                p10_us=round(s[len(s) // 10] / 1e3, 2), p90_us=round(s[(9 * len(s)) // 10] / 1e3, 2),
                min_us=round(s[0] / 1e3, 2))


def per_builder(fx, n, warm):
    """Wrap each builder's build() (instance attribute) to time it; returns per-class totals per call."""
    pc = time.perf_counter_ns
    acc = {}
    wrapped = []
    for gid, gl in enumerate(fx.attn):
        for g in gl:
            b = g.get_metadata_builder(0)
            key = type(b).__name__ + ('[ring]' if getattr(b, 'is_circular_buffer', False) else
                                      '[compressed]' if getattr(b, 'compress_ratio', 1) != 1 else '')
            orig = b.build

            def make(orig, key):
                def build(*a, **k):
                    t0 = pc()
                    r = orig(*a, **k)
                    acc.setdefault(key, []).append(pc() - t0)
                    return r
                return build
            b.build = make(orig, key)
            wrapped.append(b)
    calls = time_calls(fx.call, n, warm)
    for b in wrapped:
        del b.build
    # empty-wrapper overhead
    def noop(*a, **k):
        return None
    def wrap(f):
        def w(*a, **k):
            t0 = pc(); r = f(*a, **k); acc.setdefault('_noop', []).append(pc() - t0); return r
        return w
    w = wrap(noop)
    ov = time_calls(lambda: w(0, x=1), 20000, 100)
    total = sum(calls)
    rows = {}
    for key, v in acc.items():
        if key == '_noop':
            continue
        per_call = len(v) / (n + warm)
        tail = v[-int(per_call * n):]
        rows[key] = dict(builds_per_step=per_call, mean_us_per_build=round(statistics.fmean(tail) / 1e3, 2),
                         us_per_step=round(sum(tail) / n / 1e3, 2))
    return dict(step=summ(calls), builders=rows, wrapper_overhead_us=round(statistics.fmean(ov) / 1e3, 3),
                builders_sum_us_per_step=round(sum(r['us_per_step'] for r in rows.values()), 2))


def builder_buffer_ids(fx, torch):
    """id() of every tensor a builder owns (persistent buffers a captured graph reads)."""
    ids = {}
    for gl in fx.attn:
        for g in gl:
            b = g.get_metadata_builder(0)
            for k, v in vars(b).items():
                if isinstance(v, torch.Tensor):
                    ids[id(v)] = type(b).__name__ + '.' + k
    return ids


def dataflow(fx, torch):
    """Meta pass 2: which device ops feed a write into a builder-owned persistent buffer.

    Every aten op is logged with the ids of its tensor inputs/outputs (all kept alive
    for the call).  An op whose mutated/out argument is (a view of) a builder buffer
    is a GRAPH WRITE (the FULL graph reads that buffer at replay).  Backward
    reachability from the graph writes (and from Triton launches, whose outputs are
    builder buffers too) marks the ops the graph needs; every other op only feeds the
    returned metadata objects, which FULL replay never reads.
    """
    from torch.utils._python_dispatch import TorchDispatchMode
    bufs = builder_buffer_ids(fx, torch)
    log = []
    keep = []

    def base_id(t):
        while getattr(t, '_base', None) is not None:
            t = t._base
        return id(t)

    current = ['framework']

    class Flow(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            out = func(*args, **kwargs)
            ins = [a for a in torch.utils._pytree.tree_leaves((args, kwargs)) if isinstance(a, torch.Tensor)]
            outs = [a for a in torch.utils._pytree.tree_leaves(out) if isinstance(a, torch.Tensor)]
            keep.append((ins, outs))
            schema = func._schema
            mutated = [args[i] for i, arg in enumerate(schema.arguments[:len(args)])
                       if arg.alias_info is not None and arg.alias_info.is_write and isinstance(args[i], torch.Tensor)]
            mutated += [v for k, v in kwargs.items() if k == 'out' and isinstance(v, torch.Tensor)]
            dev = any(t.device.type == 'meta' for t in ins + outs)
            write = [bufs.get(id(t)) or bufs.get(base_id(t)) for t in mutated]
            write = [w for w in write if w]
            log.append(dict(op=str(func.overloadpacket.__name__), where=current[0], device=dev,
                            ins={id(t) for t in ins} | {base_id(t) for t in ins},
                            outs={id(t) for t in outs} | {base_id(t) for t in mutated},
                            graph_write=write))
            return out

    wrapped = []
    for gl in fx.attn:
        for g in gl:
            b = g.get_metadata_builder(0)
            key = type(b).__name__
            orig = b.build

            def make(orig, key):
                def build(*a, **k):
                    prev = current[0]; current[0] = key
                    try:
                        return orig(*a, **k)
                    finally:
                        current[0] = prev
                return build
            b.build = make(orig, key)
            wrapped.append(b)
    def hook(name, args, kwargs):
        ts = [a for a in list(args) + list(kwargs.values()) if isinstance(a, torch.Tensor)]
        keep.append((ts, []))
        log.append(dict(op='triton:' + name, where=current[0], device=True,
                        ins={id(t) for t in ts} | {base_id(t) for t in ts}, outs=set(),
                        graph_write=['triton-args']))
    TRITON_HOOK[0] = hook
    try:
        with Flow():
            fx.call()
    finally:
        TRITON_HOOK[0] = None
    for b in wrapped:
        del b.build
    need = set()
    needed = [False] * len(log)
    for i in range(len(log) - 1, -1, -1):
        r = log[i]
        if r['graph_write'] or (r['outs'] & need):
            needed[i] = True
            need |= r['ins']
    summary = {}
    for r, nd in zip(log, needed):
        s = summary.setdefault(r['where'], dict(device_ops_needed=0, device_ops_dead=0, host_ops_needed=0,
                                                host_ops_dead=0, graph_writes=[]))
        k = ('device' if r['device'] else 'host') + '_ops_' + ('needed' if nd else 'dead')
        s[k] += 1
        if r['graph_write']:
            s['graph_writes'].append(r['op'] + '->' + ','.join(r['graph_write']))
    return summary


def op_census(fx, torch):
    """Meta pass: classify every aten op inside one prepare_attn call by builder."""
    from torch.utils._python_dispatch import TorchDispatchMode
    current = ['framework']
    rows = {}

    def dev_of(x):
        return x.device.type if isinstance(x, torch.Tensor) else None

    class Census(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            out = func(*args, **kwargs)
            flat_in = [a for a in torch.utils._pytree.tree_leaves((args, kwargs)) if isinstance(a, torch.Tensor)]
            flat_out = [a for a in torch.utils._pytree.tree_leaves(out) if isinstance(a, torch.Tensor)]
            devs = {dev_of(a) for a in flat_in + flat_out}
            name = str(func.overloadpacket.__name__) if hasattr(func, 'overloadpacket') else str(func)
            is_view = bool(getattr(func, 'is_view', False)) or name in (
                'view', 'slice', 'select', 'unsqueeze', 'squeeze', 'expand', 'as_strided', 'alias', 't',
                'transpose', 'permute', '_unsafe_view', 'detach', 'lift_fresh')
            if 'meta' in devs:
                if name in ('copy_', '_to_copy') and 'cpu' in devs:
                    kind = 'h2d_or_d2h'
                elif is_view:
                    kind = 'device_view'
                else:
                    kind = 'device_op'
            else:
                kind = 'host_view' if is_view else 'host_op'
            r = rows.setdefault(current[0], {})
            r.setdefault(kind, {}).setdefault(name, 0)
            r[kind][name] += 1
            return out

    wrapped = []
    for gl in fx.attn:
        for g in gl:
            b = g.get_metadata_builder(0)
            key = type(b).__name__ + ('[ring]' if getattr(b, 'is_circular_buffer', False) else
                                      '[compressed]' if getattr(b, 'compress_ratio', 1) != 1 else '')
            orig = b.build

            def make(orig, key):
                def build(*a, **k):
                    prev = current[0]; current[0] = key
                    n0 = len(TRITON_LAUNCHES)
                    try:
                        return orig(*a, **k)
                    finally:
                        rows.setdefault(key, {}).setdefault('triton', {}).setdefault('launches', 0)
                        rows[key]['triton']['launches'] += len(TRITON_LAUNCHES) - n0
                        rows[key].setdefault('builds', 0)
                        rows[key]['builds'] += 1
                        current[0] = prev
                return build
            b.build = make(orig, key)
            wrapped.append(b)
    syncs = []
    n0 = len(TRITON_LAUNCHES)
    with Census():
        try:
            fx.call()
        except Exception as exc:
            syncs.append(dict(where=current[0], error=repr(exc)[:400], tb=traceback.format_exc()[-1500:]))
    for b in wrapped:
        del b.build
    return dict(rows=rows, failures=syncs, triton_kernels=TRITON_LAUNCHES[n0:])


def snapshot_values(meta, torch):
    """Values the FULL graph would read: per builder, its persistent buffers + the metadata tensors."""
    out = {}
    for name, md in sorted(meta.items()):
        rec = {}
        for k, v in vars(md).items() if hasattr(md, '__dict__') else []:
            if isinstance(v, torch.Tensor) and v.device.type == 'cpu':
                rec[k] = v.tolist()
            elif isinstance(v, (int, float, bool, type(None))):
                rec[k] = v
        out[name] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--model-config', required=True)
    ap.add_argument('--overlay-qsa', default=None)
    ap.add_argument('--iters', type=int, default=3000)
    ap.add_argument('--warm', type=int, default=300)
    a = ap.parse_args()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    rec = dict(schema='attn-meta-witness/2', started=time.time(), box=BOX, leaves=LEAVES,
               python=sys.version, host=platform.processor() or platform.machine(),
               cpu_count=os.cpu_count(), affinity=len(os.sched_getaffinity(0)))
    try:
        rec['cpu_model'] = next((l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')), None)
    except Exception:
        rec['cpu_model'] = None
    rec['sources'] = copy_sources(out)
    if a.overlay_qsa:
        img = class_source(VLLM / 'models/qwen4_exp/nvidia/qsa.py', 'Qwen4ExpQSAMetadataBuilder')
        ovl = class_source(a.overlay_qsa, 'Qwen4ExpQSAMetadataBuilder')
        rec['overlay_qsa_builder_class_identical'] = (img is not None and img == ovl)
        rec['overlay_qsa_sha256'] = sha(a.overlay_qsa)
    notes = {}
    rec['notes'] = notes
    try:
        import numpy as np
        import torch
        rec['torch'] = torch.__version__
        rec['torch_threads'] = torch.get_num_threads()
        rec['triton'] = install_triton_stub()
        import vllm.models.qwen4_exp.common.qsa_cache as qc
        notes['qsa_HAS_TRITON'] = bool(qc.HAS_TRITON)
        notes['qsa_build_resolved'] = qc.build_qsa_metadata.__name__
        qc.build_qsa_metadata = qc.build_qsa_metadata_triton
        if a.overlay_qsa:
            notes['overlay_builder'] = install_overlay_builder(a.overlay_qsa)
        text_cfg = json.loads(Path(a.model_config).read_text())['text_config']
        work = Path(tempfile.mkdtemp(prefix='attn-meta-', dir='/tmp'))
        cfg, cnotes = build_config(a.model_config, work, torch)
        notes['config'] = cnotes
        hists = {
            'h65k': [65000 + 997 * i for i in range(8)],
            'h110k': [110000 + 997 * i for i in range(8)],
            'mix': [65000 + 997 * i for i in range(4)] + [110000 + 997 * i for i in range(4)],
        }
        rec['histories'] = hists
        results = {}
        for stage in (0, 1):
            for hname, hist in hists.items():
                key = f'stage{stage}-{hname}'
                gc.collect()
                fx = Fixture(stage, torch.device('cpu'), cfg, text_cfg, torch, np, hist, notes.setdefault(key, {}))
                if stage == 0 and hname == 'h65k':
                    rec['builders_stage0'] = builders_summary(fx.attn)
                    rec['kernel_block_sizes'] = [k for k in fx.kbs]
                    rec['input_batch_unset_fields'] = fx.unset_fields
                if stage == 1 and hname == 'h65k':
                    rec['builders_stage1'] = builders_summary(fx.attn)
                md = fx.call()
                r = dict(layers_with_metadata=len(md), distinct_metadata_objects=len({id(v) for v in md.values()}))
                r['step'] = summ(time_calls(fx.call, a.iters, a.warm))
                r['per_builder'] = per_builder(fx, a.iters // 2, a.warm)
                gc.disable()
                r['step_gc_off'] = summ(time_calls(fx.call, a.iters // 2, a.warm))
                gc.enable()
                r['step_cold'] = summ(time_cold(fx.call, a.iters // 3, a.warm, np))
                results[key] = r
                if hname == 'mix':
                    vals = snapshot_values(fx.call(), torch)
                    (out / f'values-stage{stage}-mix.json').write_text(json.dumps(vals, sort_keys=True))
                print(key, json.dumps(r['step']), flush=True)
        rec['cpu'] = results
        rec['op_micro_us_cpu'] = op_micro(torch)
        # cProfile one configuration
        fx = Fixture(1, torch.device('cpu'), cfg, text_cfg, torch, np, hists['mix'], notes.setdefault('profile', {}))
        time_calls(fx.call, 0, 200)
        prof = cProfile.Profile()
        prof.enable()
        for _ in range(1000):
            fx.call()
        prof.disable()
        s = io.StringIO()
        pstats.Stats(prof, stream=s).sort_stats('tottime').print_stats(45)
        (out / 'profile-stage1-mix.txt').write_text(s.getvalue())
        # meta census + meta timing
        census = {}
        meta_timing = {}
        for stage in (0, 1):
            TRITON_LAUNCHES.clear()
            fxm = Fixture(stage, torch.device('meta'), cfg, text_cfg, torch, np, hists['mix'], notes.setdefault(f'meta-stage{stage}', {}))
            census[f'stage{stage}'] = op_census(fxm, torch)
            try:
                census[f'stage{stage}']['dataflow'] = dataflow(fxm, torch)
            except Exception as exc:
                census[f'stage{stage}']['dataflow'] = dict(error=repr(exc)[:400], tb=traceback.format_exc()[-1500:])
            try:
                meta_timing[f'stage{stage}'] = summ(time_calls(fxm.call, a.iters // 2, a.warm))
            except Exception as exc:
                meta_timing[f'stage{stage}'] = dict(error=repr(exc)[:400])
        rec['meta_census'] = census
        rec['meta_timing'] = meta_timing
        rec['verdict'] = 'PASS'
        rc = 0
    except Exception as exc:
        rec['verdict'] = 'FAIL'
        rec['error'] = repr(exc)[:1000]
        rec['traceback'] = traceback.format_exc()[-6000:]
        rc = 1
    rec['finished'] = time.time()
    (out / 'witness.json').write_text(json.dumps(rec, indent=1, sort_keys=True, default=str) + '\n')
    print(json.dumps(dict(verdict=rec['verdict'], error=rec.get('error'))), flush=True)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
