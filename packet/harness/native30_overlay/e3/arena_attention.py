"""Pinned staged-attention delta: dense CPU DMA, unchanged sparse-UVA fallback.

Append a private entry point, leaving the original function byte-identical.
No standalone deployable overlay: callers must bind the serial owner/fences.
"""
import ast
import hashlib

from resident_attention import BASE_SHA, NAME, function


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('attention source anchor changed')
    return text.replace(old, new, 1)


def generate(source):
    if hashlib.sha256(source.encode()).hexdigest() != BASE_SHA:
        raise ValueError('unreviewed staged-attention source')
    candidate = function(source, NAME)
    candidate = once(candidate, 'def ' + NAME + '(', 'def flash_e2_arena_attention(')
    candidate = once(candidate, '    retained_copied: int = 0,\n',
                     '    retained_copied: int = 0,\n    *, e2,\n')
    candidate = once(candidate, '    num_rows = rows.numel()\n',
                     '    if retained is not None or retained_copied != 0:\n'
                     '        raise RuntimeError("E2 arena rejects retained-bucket shortcut")\n'
                     '    num_rows = rows.numel()\n')
    candidate = once(candidate, '    if num_rows == 0:\n        return\n',
                     '    if num_rows == 0:\n        e2.route("empty", 0)\n        return\n')
    candidate = once(candidate, '        if unique_valid.numel() == 0:\n            out[rows] = 0\n',
                     '        if unique_valid.numel() == 0:\n'
                     '            e2.route("empty", 0)\n            out[rows] = 0\n')
    candidate = once(candidate, '    first_hi = retained.shape[0] if retained is not None else 0\n',
                     '    e2.route("sparse_uva" if unique_tokens is not None else "dense_dma", num_pages)\n'
                     '    first_hi = retained.shape[0] if retained is not None else 0\n')
    candidate = once(candidate,
                     '            physical = block_table_row[page_lo:page_hi].to(torch.int64)\n'
                     '            torch.index_select(kv_cache, 0, physical, out=staged)\n',
                     '            e2.fill(staged, page_lo, page_hi, block_table_row)\n')
    candidate = once(candidate,
                     '        _qsa_sparse_staged_gqa_kernel[(num_rows, num_kv_heads)](\n',
                     '        if unique_tokens is None:\n            e2.consume_begin()\n'
                     '        _qsa_sparse_staged_gqa_kernel[(num_rows, num_kv_heads)](\n')
    candidate = once(candidate, '            num_stages=2,\n        )\n',
                     '            num_stages=2,\n        )\n'
                     '        if unique_tokens is None:\n            e2.consume_end()\n')
    result = source + '\n\n' + candidate + '\n'
    ast.parse(result)
    return result
