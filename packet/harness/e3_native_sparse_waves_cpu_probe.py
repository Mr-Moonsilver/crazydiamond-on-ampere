#!/usr/bin/env python3
"""Actual mounted planner with CPU tensor copies; GPU/event execution unproved."""
import argparse
import hashlib
import json
from pathlib import Path
from e3_native_import_probe import check, error


def collect():
    import torch
    from vllm.v1.flash_e3 import sparse_history_dma as shd
    assert not torch.cuda.is_initialized()
    cases = []
    for count, pages in ((256,25),(257,25),(4097,25),(49000,25),(97153,128)):
        # Actual tensor bytes, synthetic fragmented selection/map. No device port,
        # allocator, worker or completion facts are mocked or constructed.
        mapping = tuple(reversed(range(pages)))
        selected = tuple(range(0, count*2, 2))
        versions = (1,)*pages
        strict = shd.make_sparse_plan(selected, mapping, versions, 1)
        assert strict.refused == (count>256)
        if strict.refused:
            assert strict.fallback.reason=='descriptor_cap' and strict.fallback.descriptors==257
            plan = shd.stream_sparse_plan(strict)
        else:
            plan = strict
        source = torch.arange(pages*shd.PAGE_ROWS, dtype=torch.int64)
        arena = torch.full((shd.ARENA_ROWS,), -1, dtype=torch.int64)
        high = total = 0
        bucket_rows=[]
        for bucket in plan.buckets:
            waves=bucket.waves() if isinstance(bucket,shd.SparseStreamBucket) else (bucket.pieces,)
            cursor=0
            for wave in waves:
                assert 0<len(wave)<=256
                high=max(high,len(wave));total+=1
                for piece in wave:
                    assert piece.arena_row==cursor
                    assert piece.source_pages<=4 and piece.rows<=4*shd.PAGE_ROWS
                    arena[piece.arena_row:piece.arena_end].copy_(source[piece.source_row:piece.source_row+piece.rows])
                    cursor=piece.arena_end
            assert cursor==bucket.rows
            ids=selected[bucket.compact_start:bucket.compact_end]
            expected=torch.tensor([mapping[x//shd.PAGE_ROWS]*shd.PAGE_ROWS+x%shd.PAGE_ROWS for x in ids])
            assert torch.equal(arena[:bucket.rows],expected)
            bucket_rows.append(bucket.rows)
        assert plan.selected==selected and plan.descriptor_high_water<=256
        try: shd.validate_plan_source(plan,mapping,versions,2)
        except shd.SourceRefused: pass
        else: raise AssertionError('stale epoch accepted')
        cases.append(dict(count=count,physical_pages=pages,strict_native13_refused=strict.refused,
            waves=total,descriptor_high_water=high,rows_exact=True,bucket_rows=bucket_rows,
            compact_order_unchanged=True,stale_epoch_refused=True))
    bad=shd.make_sparse_plan((0,), (0,), (1,), 1, null_page=0)
    try: shd.stream_sparse_plan(bad)
    except shd.SparsePlanError: pass
    else: raise AssertionError('unsupported map streamed')
    assert not torch.cuda.is_initialized()
    return dict(verdict='PASS',cases=cases,unsupported_mapping_refused=True,
        methods_patched=False,kernels_patched=False,mocks_used=False,
        synthetic_selection_and_map=True,cpu_tensor_rows=True,
        gpu_execution=False,event_sequencing_proved=False,
        source_sha256=hashlib.sha256(Path(shd.__file__).read_bytes()).hexdigest())


def main():
    p=argparse.ArgumentParser();p.add_argument('--graph',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    try:
        j=check(json.loads(a.graph.read_text()))
        if j['verdict']=='PASS': j['sparse_copy_waves']=collect()
    except Exception as exc:
        j=locals().get('j',dict(failures=[],missing=[],counts={}))
        j.update(verdict='FAIL',rc=1,exception=error(exc));j['failures'].append('sparse-waves-cpu-proof')
    j.update(graph_sha256=hashlib.sha256(a.graph.read_bytes()).hexdigest(),
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        hardware_qualified=False,engine_qualified=False,
        scope='actual mounted planner and CPU byte/order checks; no model, DMA, kernels or event proof')
    with a.out.open('x') as f: json.dump(j,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({k:j.get(k) for k in ('verdict','rc','failures','missing')}))
    return j['rc']


if __name__=='__main__':raise SystemExit(main())
