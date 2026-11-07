"""Generated native call sites for the bounded QSA prefill operation.

Three fail-closed transforms, applied in this order by ``make_s5_overlap``:

1. ``generate_ops`` on the E2-composed ``ops/qsa.py``: appends
   ``flash_s5_arena_attention``, a byte-preserving copy of the pinned/E2
   staged-attention function except that the dense ``torch.index_select`` is
   replaced by ``e3op.fill_bucket(...)`` (bounded copies) and the consumer
   event is recorded by ``e3op.after_attention_bucket(...)`` after the kernel.
2. ``generate_layer`` on the E2-composed ``qsa.py``: branches
   ``_qsa_staged_prefill`` to a module-level dense helper driven by
   ``layer._flash_s5_operation`` and wraps ``index_and_store`` with the
   producer boundary in ``_run_qsa``.
3. ``generate_runner`` installs an explicit refusal at the old whole-forward
   seam. The native prefix/layer/suffix dispatcher must replace it before use.

The attention function and helper are generators. Own them with
QsaPrefillOperation.attach_continuation and advance one bucket per call under
the correct lane context. Stock forward cannot consume a generator safely,
so its installed-operation branch refuses instead of falling through to UVA.

Every input is pinned by full-file SHA-256.  Each transform has an exact
inverse checked in CPU tests, so the default-off E2 path is untouched.  The
sparse-UVA compaction branch, the retained-history shortcut and any missing
CPU host metadata are refused rather than approximated.
"""
import ast
import hashlib
from pathlib import Path

import qsa_operation
from resident_attention import function

E2_QSA_SHA256 = '3dc7e447d174eaa0463865edfe978af08e75d1fd8bc5de86810c4aa914281ad7'
E2_OPS_SHA256 = 'aab7f8d83426f9c7c097347330102dbef991fd5403916107d52cd4d883b8f190'
# lane-o3 (7 Nov 2026): was 0f5b40fd506beb42a46a70a97bb7784e538aa97b0227e7a64ba41366fe9b4535; the composition gains exactly the
# three lines of graph_hooks' handback_uniform call after stock's gather_batch_req_state (inert unless FLASH_S5_O3=1).
S5_RUNNER_SHA256 = '966ffbf182c7d352737845272fca139c7b9af19e4758a6977afa33b1e471bdd4'

# The fixed-capacity union kernels are copied verbatim from the reviewed
# ``patches/sparse_stage/kernels.py``; its bytes are pinned here.  The old
# ``_s6_gather`` UVA gather in that file is deliberately not emitted: the
# sparse route uses only the qualified compact-row DMA below.
SPARSE_STAGE_KERNELS = ('_s6_mark', '_s6_scan', '_s6_offsets', '_s6_pack',
                        '_s6_remap')
SPARSE_STAGE_SHA256 = '10d675e158465971cb7a7726055a4f55aa42c18a22baddd1a451104f3f827660'
UNION_SOURCE_SHA256 = '87ab16989a02626d5e380e4607d0e411feb621d2435ff9c9a59d76b90d75057e'

OPS_NAME = 'flash_s5_arena_attention'
SPARSE_OPS_NAME = 'flash_s5_sparse_arena_attention'
LAYER_HELPER = '_flash_s5_staged_operation'
SPARSE_LAYER_HELPER = '_flash_s5_sparse_staged_operation'
RUNNER_SEAM = qsa_operation.RUNNER_SEAM
MARKERS = ('_flash_s5_operation', 'flash_s5_arena_attention',
           'flash_s5_sparse_arena_attention', '_flash_s5_sparse_staged_operation',
           '_s6_union', RUNNER_SEAM)


def _sha(source):
    return hashlib.sha256(source.encode()).hexdigest()


def _once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('S5 operation source anchor changed: '
                         + old[:80].replace('\n', '\\n'))
    return text.replace(old, new, 1)


def _parsed(text):
    try:
        ast.parse(text)
    except SyntaxError as exc:
        raise ValueError('generated S5 operation source is invalid') from exc
    return text


def _union_text(source):
    """Five pinned kernels plus the fixed-capacity launch wrapper."""
    if _sha(source) != SPARSE_STAGE_SHA256:
        raise ValueError('unreviewed sparse-stage union kernels')
    # resident_attention.function intentionally starts at ``def`` for ordinary
    # wrappers. A Triton kernel must retain its decorator to be subscriptable.
    # Preserve the pinned kernel's actual decorator/source bytes, not a guessed
    # replacement, and refuse any unreviewed decorator shape.
    nodes = {node.name: node for node in ast.parse(source).body
             if isinstance(node, ast.FunctionDef)}
    lines = source.splitlines(keepends=True)
    parts = []
    for name in SPARSE_STAGE_KERNELS:
        node = nodes[name]
        if ([ast.unparse(item) for item in node.decorator_list] != ['triton.jit']):
            raise ValueError('sparse union kernel lost its pinned Triton decorator')
        start = node.decorator_list[0].lineno - 1
        parts.append(''.join(lines[start:node.end_lineno]).rstrip('\n'))
    if '_s6_gather' in '\n\n'.join(parts):
        raise ValueError('the sparse union overlay must not emit the UVA gather')
    text = '\n\n'.join(parts) + '\n\n' + _S6_UNION.rstrip('\n')
    if UNION_SOURCE_SHA256 and _sha(text) != UNION_SOURCE_SHA256:
        raise ValueError('sparse union overlay bytes changed')
    return text


_S6_UNION = '''
def _s6_union(logical_indices, rows, num_tokens, scratch):
    # Fixed-capacity sorted union and inverse on the device.  Every buffer is
    # the operation's explicit scratch, sized from host-known geometry; no
    # device value selects a shape and no host scalar is read.  Sorted packing
    # keeps the pinned sorted-distinct order; entries in [0, N) are selected,
    # every other entry is padding (the pinned kernel contract).
    b = 1024
    groups = triton.cdiv(num_tokens, b)
    flags = scratch.flags
    ranks = scratch.ranks
    tokens = scratch.tokens
    totals = scratch.totals
    offsets = scratch.offsets
    info = scratch.info
    width = logical_indices.shape[1] - 1
    _s6_mark[(triton.cdiv(rows.numel() * width, b),)](
        logical_indices, rows, flags, logical_indices.stride(0), width,
        num_tokens, rows.numel(), b)
    _s6_scan[(groups,)](flags, ranks, totals, num_tokens, b)
    _s6_offsets[(1,)](totals, offsets, info, num_tokens, groups,
                      triton.next_power_of_2(groups))
    _s6_pack[(groups,)](flags, ranks, offsets, tokens, num_tokens, b)
    remapped = scratch.remapped
    _s6_remap[(triton.cdiv(rows.numel() * width, b),)](
        logical_indices, rows, ranks, info, remapped,
        logical_indices.stride(0), remapped.stride(0), width, rows.numel(),
        num_tokens, b)
    return tokens, info, remapped
'''


def _union_overlay():
    """Pinned union text, read from the reviewed source file at build time."""
    path = Path(__file__).resolve().parents[1] / 'sparse_stage' / 'kernels.py'
    try:
        source = path.read_text()
    except OSError as exc:
        raise ValueError('pinned sparse-stage kernels are unavailable') from exc
    return _union_text(source)


# --- attention function -----------------------------------------------------

_OPS_FORWARD = (
    ('def flash_e2_arena_attention(', 'def flash_s5_arena_attention('),
    ('    *, e2,\n', '    *, e3op,\n'),
    ('        raise RuntimeError("E2 arena rejects retained-bucket shortcut")\n',
     '        raise RuntimeError("S5 operation rejects retained-bucket shortcut")\n'),
    ('    if num_rows == 0:\n        e2.route("empty", 0)\n        return\n',
     '    if num_rows == 0:\n'
     '        raise RuntimeError("S5 operation received no attention rows")\n'),
    ('        if unique_valid.numel() == 0:\n            e2.route("empty", 0)\n            out[rows] = 0\n',
     '        if unique_valid.numel() == 0:\n'
     '            raise RuntimeError("S5 operation received no selected rows")\n'),
    ('    e2.route("sparse_uva" if unique_tokens is not None else "dense_dma", num_pages)\n',
     '    if unique_tokens is not None:\n'
     '        raise RuntimeError("S5 operation does not support sparse-UVA compaction")\n'
     '    yield e3op.schedule_boundary()\n'),
    ('            e2.fill(staged, page_lo, page_hi, block_table_row)\n',
     '            e3op.fill_bucket(staged, page_lo, page_hi, block_table_row)\n'),
    ('        if unique_tokens is None:\n            e2.consume_begin()\n', ''),
    ('        if unique_tokens is None:\n            e2.consume_end()\n',
     '        yield e3op.after_attention_bucket(\n'
     '            staged, block_table_row, q, logical_indices, rows, acc, max_state, norm_state, out)\n'),
    ('    _t = _STAGE_TIMER.begin(num_rows, num_pages) if _STAGE_TIMER is not None else None\n'
     '    if _STAGE_TIMER is not None and _STAGE_TIMER.pre_retain is not None:\n'
     '        if _t is not None:\n'
     '            _t["ret"] = _STAGE_TIMER.pre_retain\n'
     '        _STAGE_TIMER.pre_retain = None\n',
     '    _t = None  # lane-local timing comes from the operation server\n'),
    ('    if (retained is None and num_pages > int(os.environ.get("QSA_SPARSE_MIN_PAGES", pages_in_arena))\n'
     '            and os.environ.get("VLLM_QSA_KVO_SPARSE_PREFILL", "1") == "1"):\n',
     '    if False:  # declared dense transport; never mutate process environment\n'),
    ('        _qsa_sparse_staged_gqa_kernel[(num_rows, num_kv_heads)](\n',
     '        e3op.retain_attention(q, logical_indices, rows, acc, max_state, norm_state, out)\n'
     '        _qsa_sparse_staged_gqa_kernel[(num_rows, num_kv_heads)](\n'),
)


_SPARSE_PRELUDE_OLD = '''    if (retained is None and num_pages > int(os.environ.get("QSA_SPARSE_MIN_PAGES", pages_in_arena))
            and os.environ.get("VLLM_QSA_KVO_SPARSE_PREFILL", "1") == "1"):
        selected_rows = logical_indices.index_select(0, rows)
        unique, inverse = torch.unique(
            selected_rows[:, :-1].reshape(-1), sorted=True, return_inverse=True
        )
        has_padding = unique.numel() > 0 and int(unique[0]) == -1
        unique_valid = unique[1:] if has_padding else unique
        if unique_valid.numel() == 0:
            e2.route("empty", 0)
            out[rows] = 0
            return
        if _t is not None:
            _t["coverage"] = unique_valid.numel() / max(1, num_pages * page_size)
        if unique_valid.numel() < 0.8 * num_pages * page_size:
            remapped = logical_indices.clone()
            compact = inverse.reshape(num_rows, -1).to(torch.int32)
            if has_padding:
                compact -= 1
            remapped[rows, :-1] = compact
            logical_indices = remapped
            unique_tokens = unique_valid
            num_pages = triton.cdiv(unique_tokens.numel(), page_size)
'''

_SPARSE_PRELUDE_NEW = '''    if retained is None and e3op.sparse_selection_eligible(num_pages, pages_in_arena):
        # Fixed-capacity sorted union and exact inverse, entirely on the device.
        # Capacity, row count and width are host-known; no CUDA scalar, list or
        # shape-dependent allocation is read or created here.
        capacity = num_pages * page_size
        union = e3op.sparse_union_scratch(capacity, logical_indices, rows)
        unique_tokens, union_info, remapped = _s6_union(
            logical_indices, rows, capacity, union)
        # Bounded D2H into pinned scratch on the copy stream, behind a compute
        # event.  The generator yields until that copy event is host-queried;
        # the count and selection prefix are read only from completed pinned
        # memory.  Host slicing by an already-read count is not a device read.
        _selection = e3op.stage_sparse_selection(unique_tokens, union_info)
        yield e3op.schedule_boundary(_selection)
        while not e3op.sparse_selection_ready(_selection):
            yield e3op.sparse_selection_pending(_selection)
        selected_count = e3op.sparse_selection_count(_selection)
        if selected_count == 0:
            e3op.sparse_empty()
            out[rows] = 0
            return
        # Integer form of the pinned ``count < 0.8 * capacity`` threshold; it
        # is exact for every integer count and capacity.
        if selected_count * 5 < capacity * 4:
            logical_indices = remapped
            num_pages = triton.cdiv(selected_count, page_size)
        else:
            unique_tokens = None
'''

_SPARSE_FORWARD = (
    ('def flash_e2_arena_attention(', 'def flash_s5_sparse_arena_attention('),
    ('    *, e2,\n', '    *, e3op,\n'),
    ('        raise RuntimeError("E2 arena rejects retained-bucket shortcut")\n',
     '        raise RuntimeError("S5 sparse operation rejects retained-bucket shortcut")\n'),
    ('    if num_rows == 0:\n        e2.route("empty", 0)\n        return\n',
     '    if num_rows == 0:\n'
     '        raise RuntimeError("S5 sparse operation received no attention rows")\n'),
    (_SPARSE_PRELUDE_OLD, _SPARSE_PRELUDE_NEW),
    ('    e2.route("sparse_uva" if unique_tokens is not None else "dense_dma", num_pages)\n',
     '    e3op.declare_sparse_route("sparse_dma" if unique_tokens is not None else "dense_dma")\n'
     '    if e3op._current_run().schedule_boundary is None:\n'
     '        yield e3op.schedule_boundary()\n'),
    ('            e2.fill(staged, page_lo, page_hi, block_table_row)\n',
     '            e3op.fill_bucket(staged, page_lo, page_hi, block_table_row)\n'),
    ('            selected = unique_tokens[page_lo * page_size:page_hi * page_size]\n'
     '            _qsa_gather_selected_host_rows[(selected.numel(), num_kv_heads)](\n'
     '                kv_cache, selected, block_table_row, staged,\n'
     '                kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),\n'
     '                staged.stride(0), staged.stride(1), staged.stride(2),\n'
     '                selected.numel(), PAGE_SIZE=page_size, ROW_WIDTH=2 * head_dim,\n'
     '                BLOCK_D=triton.next_power_of_2(2 * head_dim), num_warps=4,\n'
     '            )\n',
     '            selected = unique_tokens[\n'
     '                page_lo * page_size:min(page_hi * page_size, selected_count)]\n'
     '            while True:\n'
     '                _copy = e3op.fill_sparse_bucket(\n'
     '                    staged, selected, page_lo * page_size, block_table_row)\n'
     '                if not getattr(_copy, "sparse_copy_pending", False):\n'
     '                    break\n'
     '                yield _copy\n'),
    ('        if unique_tokens is None:\n            e2.consume_begin()\n', ''),
    ('        if unique_tokens is None:\n            e2.consume_end()\n',
     '        yield e3op.after_attention_bucket(\n'
     '            staged, block_table_row, q, logical_indices, rows, acc, max_state, norm_state, out)\n'),
    ('    _t = _STAGE_TIMER.begin(num_rows, num_pages) if _STAGE_TIMER is not None else None\n'
     '    if _STAGE_TIMER is not None and _STAGE_TIMER.pre_retain is not None:\n'
     '        if _t is not None:\n'
     '            _t["ret"] = _STAGE_TIMER.pre_retain\n'
     '        _STAGE_TIMER.pre_retain = None\n',
     '    _t = None  # lane-local timing comes from the operation server\n'),
)


def generate_ops(source):
    """Append the dense and sparse ``flash_s5_*`` attention overlays."""
    if _sha(source) != E2_OPS_SHA256:
        raise ValueError('unreviewed E2 attention source for the S5 operation')
    if 'flash_s5_arena_attention' in source:
        raise ValueError('attention module already has the S5 operation')
    dense = function(source, 'flash_e2_arena_attention')
    for old, new in _OPS_FORWARD:
        dense = _once(dense, old, new)
    sparse = function(source, 'flash_e2_arena_attention')
    for old, new in _SPARSE_FORWARD:
        sparse = _once(sparse, old, new)
    union = _union_overlay()
    if 'torch.unique' in sparse or 'unique[' in sparse:
        raise ValueError('sparse operation must not read the device union on the host')
    if '_s6_gather' in union:
        raise ValueError('sparse operation must not emit the UVA gather')
    return _parsed(source + '\n\n' + dense + '\n\n' + sparse + '\n\n' + union + '\n')


def revert_ops(source):
    """Recover the E2 attention module bytes; refuse anything else."""
    if 'flash_s5_arena_attention' not in source:
        raise ValueError('attention module has no S5 operation')
    if 'flash_s5_sparse_arena_attention' not in source:
        raise ValueError('attention module has no sparse S5 operation')
    union = _union_overlay()
    suffix = '\n\n' + union + '\n'
    if not source.endswith(suffix):
        raise ValueError('S5 union kernels are not the pinned appended overlay')
    head = source[:-len(suffix)]
    sparse_index = head.rfind('\n\ndef flash_s5_sparse_arena_attention(')
    if sparse_index < 0:
        raise ValueError('S5 sparse attention function is not an appended overlay')
    dense_index = head.rfind('\n\ndef flash_s5_arena_attention(')
    if dense_index < 0 or dense_index > sparse_index:
        raise ValueError('S5 dense attention function is not an appended overlay')
    sparse_candidate = head[sparse_index + 2:].rstrip('\n') + '\n'
    dense_candidate = head[dense_index + 2:sparse_index].rstrip('\n') + '\n'
    recovered_sparse = sparse_candidate
    for old, new in reversed(_SPARSE_FORWARD):
        if new:
            recovered_sparse = _once(recovered_sparse, new, old)
        else:
            kernel = '        _qsa_sparse_staged_gqa_kernel[(num_rows, num_kv_heads)](\n'
            recovered_sparse = _once(recovered_sparse, kernel, old + kernel)
    recovered_dense = dense_candidate
    for old, new in reversed(_OPS_FORWARD):
        if new:
            recovered_dense = _once(recovered_dense, new, old)
        else:
            kernel = '        _qsa_sparse_staged_gqa_kernel[(num_rows, num_kv_heads)](\n'
            recovered_dense = _once(recovered_dense, kernel, old + kernel)
    recovered = head[:dense_index]
    if _sha(recovered) != E2_OPS_SHA256:
        raise ValueError('S5 attention revert did not recover the E2 bytes')
    baseline = function(recovered, 'flash_e2_arena_attention')
    if recovered_dense != baseline:
        raise ValueError('S5 attention arithmetic was not recovered')
    if recovered_sparse != baseline:
        raise ValueError('S5 sparse attention arithmetic was not recovered')
    return recovered


# --- layer / runner call sites ---------------------------------------------

_LAYER_BRANCH = '''    _s5_op = getattr(layer, "_flash_s5_operation", None)
    if _s5_op is not None:
        raise RuntimeError("S5 requires the bounded layer dispatcher; stock forward refused")
'''

_LAYER_PRODUCER = '''        _s5_op = getattr(self, "_flash_s5_operation", None)
        if _s5_op is not None:
            _s5_op.begin_layer(self.layer_name, self._qsa_host_kv)
'''

_LAYER_STORED = '''        _s5_op = getattr(self, "_flash_s5_operation", None)
        if _s5_op is not None:
            _s5_op.producer_submitted(self.layer_name, key, value, self.kv_cache)
'''

_LAYER_CONSUMED = '''        _s5_op = getattr(self, "_flash_s5_operation", None)
        if _s5_op is not None:
            _s5_op.end_layer(self.layer_name)
'''

_HELPER = '''def _flash_s5_staged_operation(
    operation,
    layer,
    query,
    kv_cache,
    logical_indices,
    attn_metadata,
    num_tokens,
    use_prefill_config,
    output,
):
    """Dense staged rows driven by the worker's bounded prefill operation.

    Installed only for a real S5 prefill transaction.  The CPU host metadata is
    mandatory; a missing/fragmented plan, sparse compaction or retained-bucket
    shortcut raises instead of silently reverting to the whole-batch path.
    """
    if not getattr(layer, "_qsa_kv_offload", False):
        raise RuntimeError("S5 operation requires the QSA host-KV path")
    min_rows = getattr(layer, "_qsa_stage_min_rows", 0)
    _observe_shape = getattr(operation, "observe_chunk_shape", None)
    if callable(_observe_shape):
        _observe_shape(num_tokens, min_rows, "dense_staged")
    if min_rows <= 0 or num_tokens < min_rows:
        raise RuntimeError("S5 operation requires the staged dense row threshold")
    host = getattr(attn_metadata, "_qsa_host_meta", None)
    starts = getattr(attn_metadata, "query_start_loc", None)
    seq_lens = getattr(attn_metadata, "seq_lens", None)
    if host is None or starts is None or seq_lens is None:
        raise RuntimeError("S5 operation requires CPU host metadata; no device readback")
    num_requests = int(seq_lens.shape[0])
    if num_requests != 1 or starts.shape[0] != 2:
        raise RuntimeError("S5 operation received invalid request metadata")
    if len(host[0]) < num_requests + 1 or len(host[1]) < num_requests:
        raise RuntimeError("S5 operation requires the complete CPU host metadata")
    offsets = host[0][: num_requests + 1]
    lengths = host[1][: num_requests]
    best = 0  # one admitted request; no mixed-batch UVA remainder
    start, end = offsets[best], offsets[best + 1]
    if start != 0 or end - start < min_rows or end != num_tokens:
        raise RuntimeError("S5 operation cannot stage this request's rows")
    from .ops.qsa import flash_s5_arena_attention, qsa_get_staging_arena
    arena = qsa_get_staging_arena(
        kv_cache, getattr(layer, "_qsa_stage_arena_bytes", 0)
    )
    if arena is None:
        raise RuntimeError("S5 operation has no staging arena")
    page_size = kv_cache.shape[2]
    max_logical_page = (max(int(lengths[best]), 1) - 1) // page_size
    if max_logical_page >= attn_metadata.block_table.shape[1]:
        raise RuntimeError("S5 operation page plan exceeds the block table")
    rows = torch.arange(start, end, device=query.device, dtype=torch.int64)
    operation.bind_plan(
        layer.layer_name,
        attn_metadata.block_table[best],
        max_logical_page + 1,
        int(lengths[best]),
        arena,
    )
    yield from flash_s5_arena_attention(
        query[:num_tokens],
        kv_cache,
        logical_indices,
        attn_metadata.block_table[best],
        rows,
        max_logical_page,
        arena,
        use_prefill_config,
        output[:num_tokens],
        e3op=operation,
    )
    return start, end
'''

_SPARSE_HELPER_ANCHOR = '''    operation.bind_plan(
        layer.layer_name,
        attn_metadata.block_table[best],
        max_logical_page + 1,
        int(lengths[best]),
        arena,
    )
    yield from flash_s5_arena_attention(
'''


def _sparse_helper_text():
    """Derive the compact-row helper from the dense one with exact anchors."""
    text = _HELPER
    for old, new in (
        ('def _flash_s5_staged_operation(', 'def _flash_s5_sparse_staged_operation('),
        ('_observe_shape(num_tokens, min_rows, "dense_staged")',
         '_observe_shape(num_tokens, min_rows, "sparse_staged")'),
        ('"""Dense staged rows driven by the worker\'s bounded prefill operation.',
         '"""Compact-row staged rows driven by the worker\'s bounded prefill operation.'),
        ('a missing/fragmented plan, sparse compaction or retained-bucket\n'
         '    shortcut raises',
         'a missing/fragmented plan, an unadmitted sparse route or a\n'
         '    retained-bucket shortcut raises'),
        (_SPARSE_HELPER_ANCHOR, _SPARSE_HELPER_ANCHOR.replace(
            '    yield from flash_s5_arena_attention(\n',
            '''    operation.bind_sparse_context(
        layer.layer_name,
        arena,
        attn_metadata.block_table[best],
        min_pages=arena.shape[0],
    )
    # Native row1 qualification binds the original UVA view and the original
    # attention function; both are no-ops unless qualification is enabled.
    operation.bind_row1_uva(layer.layer_name, kv_cache)
    from .ops.qsa import flash_e2_arena_attention
    operation.bind_row1_reference(layer.layer_name, flash_e2_arena_attention)
    yield from flash_s5_arena_attention(
''')),
        ('    return start, end\n',
         '''    operation.row1_output(
        layer.layer_name,
        query[:num_tokens],
        kv_cache,
        logical_indices,
        attn_metadata.block_table[best],
        rows,
        max_logical_page,
        arena,
        use_prefill_config,
        output[:num_tokens],
    )
    return start, end
'''),
    ):
        if text.count(old) != 1:
            raise ValueError('sparse helper anchor changed: ' + old[:60])
        text = text.replace(old, new, 1)
    if text.count('flash_s5_arena_attention') != 2:
        raise ValueError('sparse helper call-site anchor changed')
    return text.replace('flash_s5_arena_attention', 'flash_s5_sparse_arena_attention')


SPARSE_HELPER = _sparse_helper_text()

_STORED_ANCHOR = '''        impl.do_kv_cache_update(
            self,
            key,
            value,
            self.kv_cache,
            main_metadata.slot_mapping,
        )
'''

_FORWARD_QSA_ANCHOR = '''        impl.forward_qsa(
            self,
            query,
            key,
            value,
            self.kv_cache,
            main_metadata,
            output,
            token_to_req=side_metadata.token_to_req,
            use_prefill_config=main_metadata.max_query_len > self._max_decode_query_len,
        )
'''


_SELECTION_IMPORT = "from vllm.v1.flash_e3.lane_comm import register_selection_op\n_flash_s5_selection = register_selection_op()\n\n"
_SELECTION_ANCHOR = '            raise RuntimeError("QSA indexer returned an invalid selection shape")\n'
_SELECTION_CALL = '        _flash_s5_selection(selected)\n'


def generate_layer(source):
    """Add the operation branch and producer/consumer call sites to QSA."""
    if _sha(source) != E2_QSA_SHA256:
        raise ValueError('unreviewed E2 QSA source for the S5 operation')
    if '_flash_s5_operation' in source:
        raise ValueError('QSA module already carries the S5 operation')
    text = _once(source,
                 '    if not getattr(layer, "_qsa_kv_offload", False):\n        return None\n',
                 '    if not getattr(layer, "_qsa_kv_offload", False):\n        return None\n'
                 + _LAYER_BRANCH)
    text = _once(text, '        selected = self.indexer(\n',
                 _LAYER_PRODUCER + '        selected = self.indexer(\n')
    text = _once(text, _SELECTION_ANCHOR, _SELECTION_ANCHOR + _SELECTION_CALL)
    text = _once(text, '__all__ = [', _SELECTION_IMPORT + '__all__ = [')
    text = _once(text, _STORED_ANCHOR, _STORED_ANCHOR + _LAYER_STORED)
    text = _once(text, _FORWARD_QSA_ANCHOR, _FORWARD_QSA_ANCHOR + _LAYER_CONSUMED)
    text = _once(text, '__all__ = [',
                 _HELPER + '\n\n' + SPARSE_HELPER + '\n\n__all__ = [')
    return _parsed(text)


def revert_layer(source):
    """Recover the E2 QSA bytes; every inserted site must be present exactly once."""
    if '_flash_s5_operation' not in source:
        raise ValueError('QSA module has no S5 operation')
    if '_flash_s5_sparse_staged_operation' not in source:
        raise ValueError('QSA module has no sparse S5 operation')
    text = _once(source, _SELECTION_IMPORT, '')
    text = _once(text, _SELECTION_CALL, '')
    text = _once(text, _LAYER_BRANCH, '')
    text = _once(text, _LAYER_PRODUCER, '')
    text = _once(text, _LAYER_STORED, '')
    text = _once(text, _LAYER_CONSUMED, '')
    text = _once(text, _HELPER + '\n\n' + SPARSE_HELPER + '\n\n', '')
    if _sha(text) != E2_QSA_SHA256:
        raise ValueError('S5 QSA revert did not recover the E2 bytes')
    return text


_SEAM = f'''    @torch.inference_mode()
    def {RUNNER_SEAM}(self, scheduler_output):
        """Refuse stock execution until the bounded layer dispatcher is wired.

        A whole execute_model call touches ordinary batch/PP state and cannot
        suspend at an attention bucket. Never hide it behind an operation name.
        """
        raise RuntimeError("S5 bounded layer dispatcher is not installed; stock forward refused")

'''

_RUNNER_ANCHOR = '    @torch.inference_mode()\n    def execute_model('


def generate_runner(source):
    """Add the worker-server seam to the S5-composed runner, before the wrapper."""
    if _sha(source) != S5_RUNNER_SHA256:
        raise ValueError('unreviewed S5 model-runner source for the S5 operation')
    if RUNNER_SEAM in source:
        raise ValueError('model runner already has the S5 operation seam')
    return _parsed(_once(source, _RUNNER_ANCHOR, _SEAM + _RUNNER_ANCHOR))


def revert_runner(source):
    """Recover the S5-composed runner bytes."""
    if RUNNER_SEAM not in source:
        raise ValueError('model runner has no S5 operation seam')
    text = _once(source, _SEAM, '')
    if _sha(text) != S5_RUNNER_SHA256:
        raise ValueError('S5 runner revert did not recover the S5-composed bytes')
    return text


__all__ = [
    'E2_OPS_SHA256', 'E2_QSA_SHA256', 'LAYER_HELPER', 'MARKERS', 'OPS_NAME',
    'RUNNER_SEAM', 'S5_RUNNER_SHA256', 'SPARSE_HELPER', 'SPARSE_LAYER_HELPER',
    'SPARSE_OPS_NAME', 'SPARSE_STAGE_SHA256', 'UNION_SOURCE_SHA256',
    'generate_layer', 'generate_ops', 'generate_runner', 'revert_layer',
    'revert_ops', 'revert_runner',
]
