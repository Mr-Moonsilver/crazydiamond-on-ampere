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

import qsa_operation
from resident_attention import function

E2_QSA_SHA256 = '3dc7e447d174eaa0463865edfe978af08e75d1fd8bc5de86810c4aa914281ad7'
E2_OPS_SHA256 = 'aab7f8d83426f9c7c097347330102dbef991fd5403916107d52cd4d883b8f190'
S5_RUNNER_SHA256 = 'a9dc60073613e1beaf2832a4048baad55b09bccb0bcee57c59c00e18b4bdfdd4'

OPS_NAME = 'flash_s5_arena_attention'
LAYER_HELPER = '_flash_s5_staged_operation'
RUNNER_SEAM = qsa_operation.RUNNER_SEAM
MARKERS = ('_flash_s5_operation', 'flash_s5_arena_attention', RUNNER_SEAM)


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
     '        raise RuntimeError("S5 operation does not support sparse-UVA compaction")\n'),
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


def generate_ops(source):
    """Append ``flash_s5_arena_attention`` to the E2-composed attention module."""
    if _sha(source) != E2_OPS_SHA256:
        raise ValueError('unreviewed E2 attention source for the S5 operation')
    if 'flash_s5_arena_attention' in source:
        raise ValueError('attention module already has the S5 operation')
    candidate = function(source, 'flash_e2_arena_attention')
    for old, new in _OPS_FORWARD:
        candidate = _once(candidate, old, new)
    return _parsed(source + '\n\n' + candidate + '\n')


def revert_ops(source):
    """Recover the E2 attention module bytes; refuse anything else."""
    if 'flash_s5_arena_attention' not in source:
        raise ValueError('attention module has no S5 operation')
    index = source.rfind('\n\ndef flash_s5_arena_attention(')
    if index < 0:
        raise ValueError('S5 attention function is not an appended overlay')
    candidate = source[index + 2:].rstrip('\n') + '\n'
    original = candidate
    for old, new in reversed(_OPS_FORWARD):
        if new:
            original = _once(original, new, old)
        else:
            kernel = '        _qsa_sparse_staged_gqa_kernel[(num_rows, num_kv_heads)](\n'
            original = _once(original, kernel, old + kernel)
    recovered = source[:index]
    if _sha(recovered) != E2_OPS_SHA256:
        raise ValueError('S5 attention revert did not recover the E2 bytes')
    if function(recovered, 'flash_e2_arena_attention') != original:
        raise ValueError('S5 attention arithmetic was not recovered')
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
    text = _once(text, _STORED_ANCHOR, _STORED_ANCHOR + _LAYER_STORED)
    text = _once(text, _FORWARD_QSA_ANCHOR, _FORWARD_QSA_ANCHOR + _LAYER_CONSUMED)
    text = _once(text, '__all__ = [', _HELPER + '\n\n__all__ = [')
    return _parsed(text)


def revert_layer(source):
    """Recover the E2 QSA bytes; every inserted site must be present exactly once."""
    if '_flash_s5_operation' not in source:
        raise ValueError('QSA module has no S5 operation')
    text = _once(source, _LAYER_BRANCH, '')
    text = _once(text, _LAYER_PRODUCER, '')
    text = _once(text, _LAYER_STORED, '')
    text = _once(text, _LAYER_CONSUMED, '')
    text = _once(text, _HELPER + '\n\n', '')
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
    'RUNNER_SEAM', 'S5_RUNNER_SHA256', 'generate_layer', 'generate_ops',
    'generate_runner', 'revert_layer', 'revert_ops', 'revert_runner',
]
