#!/usr/bin/env python3
"""Decoders keep the QSA row cache in mixed steps (goal item 3). On top of make_hostmeta.py (qsa.py.v6 -> v7).

In a batch that mixes prefill and decode rows, use_prefill_config is true and every row that is not staged goes
through qsa_sparse_paged_attention with the prefill configuration and rowcache=None: the decoders of that step read
all their selected rows from host instead of the VRAM row cache. PD alternation keeps such steps rare, but short-first
fallbacks, paused alternation and any S5 variant that batches decode with prefill make them common.
QSA_ROWCACHE_MIXED=1 (needs QSA_HOST_META=1 for the CPU offsets, no sync): the leading run of requests with exactly
one query row (decodes; the batch is checked, not assumed, to start with them) is served first with the decode
configuration and the row cache, exactly as in a decode-only step; the remaining rows take the existing path. If the
batch does not start with decode rows, nothing changes. The row-cache ownership update above runs for every step
already, so the cache state the decoders see is the same as in a decode step. Leak, lifetime and canary gates are
required before this is used (the row cache has a leak history, 23 Oct).
Unset, the patched file behaves exactly like its input.
usage: make_rowmix.py qsa.py.v6 qsa.py.v7
"""
import difflib, sys

src = orig = open(sys.argv[1]).read()


def sub(old, new):
    global src
    assert src.count(old) == 1, (old[:80], src.count(old))
    src = src.replace(old, new)


sub('''_QSA_HOST_META_STATS = [0, 0]  # verified calls, mismatches
''', '''_QSA_HOST_META_STATS = [0, 0]  # verified calls, mismatches
_QSA_ROWCACHE_MIXED = os.environ.get("QSA_ROWCACHE_MIXED", "0") == "1"
_QSA_ROWMIX_STATS = [0, 0]  # mixed steps seen (first QSA layer), of which served decode rows from the row cache


def _qsa_leading_decode_rows(attn_metadata):
    """Rows [0, n) of the batch that are single-row requests (decodes), from the CPU offsets; 0 if unknown."""
    host = getattr(attn_metadata, "_qsa_host_meta", None)
    if host is None:
        return 0
    offs = host[0]
    n = 0
    while n + 1 < len(offs) and offs[n + 1] - offs[n] == 1:
        n += 1
    return n
''')

sub('''        if staged is None:
            qsa_sparse_paged_attention(
                query[:num_tokens],
                key_cache,
                value_cache,
                logical_indices,
                attn_metadata.block_table,
                token_to_req,
                use_prefill_config,
                output[:num_tokens],
                rowcache=None if use_prefill_config else rowcache,
            )
            return output
        staged_start, staged_end = staged
        for low, high in ((0, staged_start), (staged_end, num_tokens)):''', '''        lead = 0
        if _QSA_ROWCACHE_MIXED and use_prefill_config and rowcache is not None:
            # make_rowmix.py: the batch's leading decode rows keep the row cache in a mixed step
            lead = min(_qsa_leading_decode_rows(attn_metadata), num_tokens,
                       staged[0] if staged is not None else num_tokens)
            if layer.layer_name.split(".")[-3] in ("3", "27"):
                _QSA_ROWMIX_STATS[0] += 1
                _QSA_ROWMIX_STATS[1] += lead > 0
                if _QSA_ROWMIX_STATS[0] % 2000 == 0:
                    from vllm.logger import init_logger
                    init_logger(__name__).info("QSA rowcache mixed steps: %d seen, %d with cached decode rows",
                                               *_QSA_ROWMIX_STATS)
            if lead:
                qsa_sparse_paged_attention(
                    query[:lead],
                    key_cache,
                    value_cache,
                    logical_indices[:lead],
                    attn_metadata.block_table,
                    token_to_req[:lead],
                    False,
                    output[:lead],
                    rowcache=rowcache,
                )
        if staged is None:
            if lead >= num_tokens:
                return output
            qsa_sparse_paged_attention(
                query[lead:num_tokens],
                key_cache,
                value_cache,
                logical_indices[lead:],
                attn_metadata.block_table,
                token_to_req[lead:],
                use_prefill_config,
                output[lead:num_tokens],
                rowcache=None if use_prefill_config else rowcache,
            )
            return output
        staged_start, staged_end = staged
        for low, high in ((lead, staged_start), (staged_end, num_tokens)):''')
open(sys.argv[2], "w").write(src)
sys.stdout.writelines(difflib.unified_diff(orig.splitlines(True), src.splitlines(True), "qsa.py.v6", "qsa.py.v7"))
