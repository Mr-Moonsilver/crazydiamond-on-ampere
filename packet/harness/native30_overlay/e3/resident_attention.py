"""Pinned-source E2 staged-attention helper generator, NOT an engine overlay.

Stock functions remain byte-for-byte present. The added private entry point
changes only the source tensor/table, after a caller proves the resident lease
ready. Sparse union/remap, kernel calls and bucket arithmetic stay unchanged.
"""
import ast
import hashlib

BASE_SHA = '5a0d962022ec4e47809fa072d2b13745c84413de894048e67f10c3c10a45d51b'
NAME = 'qsa_sparse_paged_attention_staged'


def function(source, name):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ''.join(source.splitlines(keepends=True)[node.lineno-1:node.end_lineno])


def generate(source):
    if hashlib.sha256(source.encode()).hexdigest() != BASE_SHA:
        raise ValueError('unreviewed staged-attention source')
    original = function(source, NAME)
    candidate = original.replace('def '+NAME+'(', 'def flash_e2_resident_attention(', 1)
    candidate = candidate.replace('    retained_copied: int = 0,\n',
                                  '    retained_copied: int = 0,\n    *, resident: torch.Tensor,\n', 1)
    # kv_cache's original physical indices have no meaning in the logical slab.
    # Preserve dtype/shape and all baseline arithmetic; only the gather source
    # and its mapping become a GPU allocation plus the identity page mapping.
    anchor = '    num_rows = rows.numel()\n'
    prologue = '''    if retained is not None or retained_copied != 0:
        raise RuntimeError("E2 cannot use the legacy retained-bucket shortcut")
    if (resident.device != q.device or resident.dtype != kv_cache.dtype or
            resident.shape[1:] != kv_cache.shape[1:] or
            resident.shape[0] < max_logical_page + 1 or not resident.is_contiguous()):
        raise RuntimeError("invalid E2 resident source")
    kv_cache = resident
    block_table_row = torch.arange(max_logical_page + 1, device=q.device, dtype=block_table_row.dtype)
'''
    if candidate.count(anchor) != 1:
        raise ValueError('ambiguous insertion anchor')
    candidate = candidate.replace(anchor, prologue+anchor, 1)
    result = source+'\n\n'+candidate+'\n'
    ast.parse(result)
    return result
