"""Pinned QSA gather-only substitution and serial bucket ownership for VM job4."""
import ast
import hashlib

SOURCE_SHA='5a0d962022ec4e47809fa072d2b13745c84413de894048e67f10c3c10a45d51b'
NAME='qsa_sparse_paged_attention_staged'
GATHER='''            _qsa_gather_selected_host_rows[(selected.numel(), num_kv_heads)](
                kv_cache, selected, block_table_row, staged,
                kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
                staged.stride(0), staged.stride(1), staged.stride(2),
                selected.numel(), PAGE_SIZE=page_size, ROW_WIDTH=2 * head_dim,
                BLOCK_D=triton.next_power_of_2(2 * head_dim), num_warps=4,
            )'''


def generate(source):
    if hashlib.sha256(source.encode()).hexdigest()!=SOURCE_SHA:
        raise ValueError('unpinned_qsa_source')
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==NAME)
    original=''.join(source.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
    if original.count(GATHER)!=1:raise ValueError('sparse_gather_anchor')
    result=original.replace('def '+NAME+'(', 'def dma_attention(',1)
    result=result.replace('    retained_copied: int = 0,\n', '    retained_copied: int = 0,\n    *, dma_fill,\n',1)
    result=result.replace(GATHER,'            dma_fill(staged, selected, page_lo * page_size)',1)
    ast.parse(result)
    return result


class Cancelled(RuntimeError):
    pass


class BucketLease:
    """Local test ownership. A partial fill never grants consumer permission.

    Events are host-queried; an abort requires draining its last submitted copy
    before reuse. This is not an engine scheduler lease or async cancellation.
    """
    def __init__(self):
        self.state='free';self.event=None;self.published=0

    def begin(self):
        if self.state!='free':raise RuntimeError('arena_not_reusable')
        self.state='filling'

    def publish(self,event):
        if self.state!='filling' or not event.query():raise RuntimeError('copy_not_complete')
        self.state='ready';self.published+=1

    def consume(self):
        if self.state!='ready':raise RuntimeError('partial_bucket_not_visible')
        self.state='consuming'

    def complete(self,event):
        if self.state!='consuming' or not event.query():raise RuntimeError('consumer_not_complete')
        self.state='free'

    def abort(self,event):
        if self.state!='filling':raise RuntimeError('abort_not_filling')
        self.state='aborted';self.event=event

    def drain(self):
        if self.state!='aborted' or not self.event.query():raise RuntimeError('abort_not_drained')
        self.event=None;self.state='free'
