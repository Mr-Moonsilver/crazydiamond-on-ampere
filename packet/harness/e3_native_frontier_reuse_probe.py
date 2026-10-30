#!/usr/bin/env python3
"""Pinned-image CPU allocator proof; no GPU/model, patched methods or kernels."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import sys

from e3_native_import_probe import check, error


def collect():
    import torch
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.single_type_kv_cache_manager import MambaManager
    from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
    from vllm.v1.kv_cache_interface import MambaSpec
    from vllm.v1.flash_e3 import scheduler_hooks as sh, async_handoff as ah
    from vllm.v1.flash_e3.e3_types import TransactionId, E3ContractError, E3Pending

    def world(size=5):
        pool = BlockPool(size, True, 1024)
        scheduler = NS()  # Explicit owner holder, not a fake allocator/manager.
        leases = sh.SchedulerLeases().attach(scheduler, pool)
        frontier = ah.CommittedFrontier()
        controller = sh.PrefillAdmissionController(
            scheduler, leases=leases, frontier=frontier, engine_epoch=1,
            chunk_tokens=1024, ranks=(0, 1, 2, 3)).attach()
        return pool, scheduler, leases, frontier, controller

    def tx(seq):
        return TransactionId(1, 'cpu-proof', 1, (seq-1)*1024, seq*1024, seq)

    def publish(frontier, seq, ids):
        frontier.reserve(tx(seq), ids)
        frontier.publish(tx(seq))

    def refuses(fn, kind, text):
        try:
            fn()
        except kind as exc:
            assert text in str(exc), str(exc)
            return str(exc)
        raise AssertionError('required refusal absent: '+text)

    result = dict(methods_patched=False, kernels_patched=False,
                  synthetic_scheduler_owner=True, allocator_mocked=False,
                  manager_mocked=False, worker_reports_constructed=False)
    pool, scheduler, leases, frontier, controller = world(4)
    pages = pool.get_new_blocks(3)
    ids = tuple(b.block_id for b in pages)
    publish(frontier, 1, ids)
    key = bytes(range(32))
    pool._insert_block_hash(make_block_hash_with_group_id(key, 0), pages[0], 1024)
    pool.free_blocks(pages)
    assert frontier.shareable(ids) == ids
    assert pool.get_cached_block(key, [0]) == [pages[0]]
    pool.touch([pages[0]])
    assert frontier.shareable(ids) == ids
    pool.free_blocks([pages[0]])
    new_pages = pool.get_new_blocks(3)
    assert set(b.block_id for b in new_pages) == set(ids)
    assert frontier.committed_blocks() == ()
    assert pool.get_cached_block(key, [0]) is None
    publish(frontier, 2, tuple(b.block_id for b in new_pages))
    result['pool_reuse'] = dict(pages=list(ids), cached_free_preserved=True,
        cache_hit_touch_preserved=True, real_reallocation_retired=True,
        real_cache_evicted=True, republished=True)

    pool, scheduler, leases, frontier, controller = world(5)
    spec = MambaSpec(block_size=1024, shapes=((2, 2),),
                     dtypes=(torch.float32,), mamba_cache_mode='align')
    manager = MambaManager(spec, pool, enable_caching=True,
        kv_cache_group_id=0, scheduler_block_size=1024)
    seen = set()
    rows = []
    for seq in range(1, 98):
        start, end = (seq-1)*1024, seq*1024
        manager.remove_skipped_blocks('cpu-proof', start, 98312)
        allocated = manager.allocate_new_blocks('cpu-proof', end, end)
        ids = tuple(b.block_id for b in allocated if not b.is_null)
        assert ids
        reused = sorted(set(ids) & seen)
        publish(frontier, seq, ids)
        rows.append(dict(chunk=seq, allocated=list(ids), reused=reused,
                         free=pool.get_num_free_blocks()))
        seen.update(ids)
        assert len(frontier._published) <= 4
    assert any(r['reused'] for r in rows)
    manager.free('cpu-proof')
    result['mamba_recycling'] = dict(chunks=rows, distinct_pages=len(seen),
        actual_manager=type(manager).__module__+'.'+type(manager).__name__,
        no_manual_frontier_retirement=True, free_after=pool.get_num_free_blocks())

    # Real free/allocate calls with actual lease guards; the last check invokes
    # the same installed callback directly to test its whole-batch atomicity.
    pool, scheduler, leases, frontier, controller = world(3)
    pages = pool.get_new_blocks(2)
    ids = tuple(b.block_id for b in pages)
    publish(frontier, 1, ids)
    leases.reserve_blocks(tx(2), ah.LeaseCategory.SOURCE_HISTORY, (ids[1],),
                          blocks=(pages[1],))
    pool.free_blocks(pages)
    assert pool.get_num_free_blocks() == 1
    message = refuses(lambda: sh.note_allocated(pool, pages), E3ContractError,
                      'leased block')
    assert frontier.committed_blocks() == tuple(sorted(ids))
    result['held_batch'] = dict(refusal=message, publications_unchanged=True,
                               actual_free_deferred=True)

    pool, scheduler, leases, frontier, controller = world(3)
    pages = pool.get_new_blocks(2)
    ids = tuple(b.block_id for b in pages)
    publish(frontier, 1, (ids[0],))
    frontier.reserve(tx(2), (ids[1],))
    pool.free_blocks(pages)
    message = refuses(lambda: pool.get_new_blocks(2), E3Pending, 'unpublished')
    assert frontier.committed_blocks() == (ids[0],)
    assert frontier._epoch[tx(2)] == (ids[1],)
    result['unpublished_batch'] = dict(refusal=message,
        publications_unchanged=True, allocator_mutated_before_refusal=True,
        isolated_refused_pool_never_reused=True)

    # Run the frozen old method, unchanged, on real owner/allocator objects.
    # This is a direct legacy callback counterexample, not a patched live hook.
    old_path = Path('/proof/native12_scheduler_hooks.py')
    name = 'vllm.v1.flash_e3._native12_scheduler_hooks'
    spec = importlib.util.spec_from_file_location(name, old_path)
    old = importlib.util.module_from_spec(spec)
    sys.modules[name] = old
    spec.loader.exec_module(old)
    pool, scheduler, leases, frontier, controller = world(2)
    pages = pool.get_new_blocks(1)
    publish(frontier, 1, (pages[0].block_id,))
    old.SchedulerLeases.note_allocated(leases, pages)
    message = refuses(lambda: frontier.reserve(tx(2), (pages[0].block_id,)),
                      ah.HandoffError, 'explicit frontier retirement')
    result['native12_negative'] = dict(refused=True, message=message,
        scope='unchanged legacy callback invoked directly on real owner/page',
        source_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest())
    assert not torch.cuda.is_initialized()
    result['cuda_initialized'] = False
    result['verdict'] = 'PASS'
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    try:
        j = check(json.loads(a.graph.read_text()))
        if j['verdict'] == 'PASS':
            j['frontier_reuse'] = collect()
    except Exception as exc:
        j = locals().get('j', dict(failures=[], missing=[], counts={}))
        j.update(verdict='FAIL', rc=1, exception=error(exc))
        j['failures'].append('frontier-reuse-proof')
    j.update(graph_sha256=hashlib.sha256(a.graph.read_bytes()).hexdigest(),
             probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             hardware_qualified=False, engine_qualified=False,
             scope='real mounted allocator/Mamba manager and frontier on CPU; '
                   'synthetic scheduler owner, no workers/model/performance')
    with a.out.open('x') as f:
        json.dump(j, f, indent=2, sort_keys=True)
        f.write('\n')
    print(json.dumps({k:j.get(k) for k in ('verdict','rc','failures','missing')}))
    return j['rc']


if __name__ == '__main__':
    raise SystemExit(main())
