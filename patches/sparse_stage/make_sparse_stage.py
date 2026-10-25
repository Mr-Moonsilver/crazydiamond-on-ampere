#!/usr/bin/env python3
"""Generate a new overlay-v16h from a checked v16g directory; never overwrite."""
import argparse
import ast
import hashlib
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
BASE_SHA='5a0d962022ec4e47809fa072d2b13745c84413de894048e67f10c3c10a45d51b'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def sub(s,a,b):
    assert s.count(a)==1,(a[:80],s.count(a))
    return s.replace(a,b)
def function(src,name):
    node=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name==name)
    lines=src.splitlines(keepends=True)
    start=min([node.lineno]+[d.lineno for d in node.decorator_list])-1
    return ''.join(lines[start:node.end_lineno])

def generate(src):
    assert hashlib.sha256(src.encode()).hexdigest()==BASE_SHA,'unreviewed ops source'
    original=function(src,'qsa_sparse_paged_attention_staged')
    async_fn=original.replace('def qsa_sparse_paged_attention_staged(', 'def _s6_staged_async(',1)
    start=async_fn.index('    unique_tokens = None\n');end=async_fn.index('    group_size = ',start)
    async_fn=async_fn[:start]+'''    unique_tokens, union_info, logical_indices = _s6_union(logical_indices, rows, num_pages * page_size)
    # Host launch bounds stay fixed; kernels read count/branch and skip unused
    # buckets. Timer v8 assumes CPU-known counts, so use wall/step metrics here.
    spans = [(lo, min(lo + pages_in_arena, num_pages)) for lo in range(0, num_pages, pages_in_arena)]
    num_buckets = len(spans)
    _t = None

'''+async_fn[end:]
    start=async_fn.index('        from_retained = ');end=async_fn.index('        k_arena, v_arena = ',start)
    async_fn=async_fn[:start]+'''        staged = arena[:page_hi - page_lo]
        _s6_gather[((page_hi-page_lo)*page_size,num_kv_heads)](
            kv_cache,unique_tokens,block_table_row,staged,union_info,
            kv_cache.stride(0),kv_cache.stride(1),kv_cache.stride(2),
            staged.stride(0),staged.stride(1),staged.stride(2),
            page_lo,page_hi-page_lo,num_pages*page_size,page_size,2*head_dim,
            triton.next_power_of_2(2*head_dim),num_warps=4)
'''+async_fn[end:]
    async_fn=sub(async_fn,'_qsa_sparse_staged_gqa_kernel[','_s6_staged_kernel[')
    async_fn=sub(async_fn,'            out,\n','            out,\n            union_info,\n            num_pages * page_size,\n')
    async_fn=sub(async_fn,'            LAST_BUCKET=bucket == num_buckets - 1,','            LAST_BUCKET=bucket == num_buckets - 1,\n            ARENA_PAGES=pages_in_arena,')
    kernel=function(src,'_qsa_sparse_staged_gqa_kernel')
    kernel=sub(kernel,'def _qsa_sparse_staged_gqa_kernel(','def _s6_staged_kernel(')
    kernel=sub(kernel,'    output_ptr,\n','    output_ptr,\n    union_info,\n    total_tokens,\n')
    kernel=sub(kernel,'    LAST_BUCKET: tl.constexpr,','    LAST_BUCKET: tl.constexpr,\n    ARENA_PAGES: tl.constexpr,')
    body=kernel.index('    valid_count = ')
    prefix=kernel[:body]
    tail=kernel[body:]
    tail=sub(tail,'(logical_token >= 0) & (arena_page >= 0)',
             '(logical_token >= 0) & (logical_token < tl.where(sparse, count, total_tokens)) & (arena_page >= 0)')
    tail=sub(tail,'    if LAST_BUCKET:', '    if page_lo + ARENA_PAGES >= effective_pages:')
    # Entire original arithmetic executes only for an active bucket. FIRST_BUCKET
    # still means page_lo==0; the device count determines the LAST active bucket.
    kernel=prefix+'''    count = tl.load(union_info)
    sparse = tl.load(union_info + 1) != 0
    effective_pages = tl.cdiv(tl.where(sparse,count,total_tokens),PAGE_SIZE)
    if effective_pages == 0:
        if FIRST_BUCKET:
            h = tl.arange(0,BLOCK_M)
            d = tl.arange(0,HEAD_DIM)
            tl.store(output_ptr + row*stride_output_row + (kv_head*GROUP_SIZE+h[:,None])*stride_output_head + d[None,:],
                     0.,h[:,None]<GROUP_SIZE)
    elif page_lo < effective_pages:
        pages_in_arena = tl.minimum(pages_in_arena,effective_pages-page_lo)
'''+''.join('    '+line if line.strip() else line for line in tail.splitlines(keepends=True))
    signature=original[:original.index('    """')]
    wrapper=signature+'''    enabled = os.environ.get("FLASH_QSA_SPARSE_ASYNC", "0") == "1"
    eligible = (retained is None and rows.numel() > 0 and
        max_logical_page + 1 > int(os.environ.get("QSA_SPARSE_MIN_PAGES", arena.shape[0])) and
        os.environ.get("VLLM_QSA_KVO_SPARSE_PREFILL", "1") == "1")
    args = (q,kv_cache,block_table_row,logical_indices,rows,max_logical_page,arena,use_prefill_config,out,retained,retained_copied)
    if not enabled or not eligible:
        return _s6_staged_stock(*args)
    _s6_staged_async(*args)
    if os.environ.get("FLASH_QSA_SPARSE_VERIFY", "0") == "1":
        # Explicit diagnostic path only: compare exact same inputs on the same
        # stream, synchronise to report byte equality. Never enabled for speed.
        reference = torch.empty_like(out)
        _s6_staged_stock(*args[:8],reference,*args[9:])
        actual=out.index_select(0,rows).contiguous().view(torch.uint8)
        expected=reference.index_select(0,rows).contiguous().view(torch.uint8)
        equal=torch.equal(actual,expected)
        from vllm.logger import init_logger
        init_logger("vllm.qsa.s6").info("S6_VERIFY rows=%d equal=%d", rows.numel(),int(equal))
        if not equal: raise RuntimeError("S6 staged output byte mismatch")
'''
    code=src.replace(original,original.replace('def qsa_sparse_paged_attention_staged(','def _s6_staged_stock(',1)+'\n\n'+(HERE/'kernels.py').read_text()+'\n\n'+kernel+'\n\n'+async_fn+'\n\n'+wrapper)
    ast.parse(code);return code

def build(base,out):
    assert not out.exists(),'new output directory required'
    for folder in (base,base/'core'):
        for line in (folder/'SHA256SUMS').read_text().splitlines():
            digest,name=line.split(maxsplit=1);p=folder/name
            assert p.resolve().is_relative_to(folder.resolve()) and sha(p)==digest,'base manifest mismatch'
    result=generate((base/'ops_qsa.py').read_text())
    shutil.copytree(base,out)
    (out/'ops_qsa.py').write_text(result)
    lines=[]
    for line in (out/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split(maxsplit=1)
        lines.append(sha(out/name)+'  '+name)
    (out/'SHA256SUMS').write_text('\n'.join(lines)+'\n')
    return sha(out/'ops_qsa.py')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('base',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
    print('ops_qsa.py sha256='+build(a.base,a.out))
