#!/usr/bin/env python3
"""Additional CPU stress on frozen native14. No GPU/engine/box action."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import struct
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'harness'))
import test_e3_sparse_history_dma as f
import test_e3_sparse_copy_waves as waves
from test_e3_native_measurement import make_long_stack


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_frozen(allow_validation_hoist=False):
    bundle=Path('<hub-dir>/agents/flash-next/tmp/q18a-native14-bundle-2bfadb912407')
    for name in ('sparse_history_dma','qsa_operation','qsa_driver','operation_source','engine_driver','native_measurement'):
        assert sha(ROOT/'patches/s5_engine'/(name+'.py'))==sha(bundle/'e3_builder'/(name+'.py'))
    old=(bundle/'e3_builder/wire_v1.py').read_text()
    current=(ROOT/'patches/s5_engine/wire_v1.py').read_text()
    if allow_validation_hoist:
        expected=old.replace('        last = -1\n        for entry in self.host_submit_ns:',
            '        accepted_ops = done.union(self.pending_op_seqs)\n        last = -1\n        for entry in self.host_submit_ns:').replace(
            'if op_seq not in done.union(self.pending_op_seqs):','if op_seq not in accepted_ops:')
        assert current==expected
    else:assert current==old
    return dict(bundle=str(bundle),manifest=sha(bundle/'SHA256SUMS'),
                device_and_wave_runtime_bytes_equal=True,
                report_validation_hoist_only=allow_validation_hoist,
                wire_sha256=sha(ROOT/'patches/s5_engine/wire_v1.py'))


def multibucket(cancel_second=False):
    cls=waves.OperationTests
    cls.setUpClass()
    test=cls('runTest');test.setUp()
    try:
        mapping=tuple((i*17+3)%131 for i in range(128))
        operation,lane,tx,sources=test.prepare({'qsa.0':mapping})
        hold=test.open_layer(operation,lane,tx,'qsa.0',sources['qsa.0'],mapping,min_pages=24)
        rows,width=120,2052
        count=rows*(width-1)
        selected=tuple(range(0,count*2,2))
        values=[]
        for row in range(rows):values.extend(selected[row*(width-1):(row+1)*(width-1)]+(-1,))
        logical=f.T((rows,width),dtype='torch.int32',values=values)
        query=f.T((rows,f.QUERY_HEADS,f.HEAD_DIM))
        output=f.T(query.shape)
        rids=f.T((rows,),dtype='torch.int64',values=list(range(rows)))
        snapshots=[]
        def kernel(grid,*args,**kwargs):
            index=len(snapshots);start=index*f.shd.ARENA_ROWS
            ids=selected[start:start+f.shd.ARENA_ROWS]
            actual=test.arena.storage.rows[:len(ids)]
            expected=[struct.pack('<Q',mapping[t//f.PAGE_TOKENS]*f.PAGE_TOKENS+t%f.PAGE_TOKENS)*4 for t in ids]
            assert actual==expected
            snapshots.append(dict(rows=len(ids),digest=hashlib.sha256(b''.join(actual)).hexdigest(),
                first=kwargs['FIRST_BUCKET'],last=kwargs['LAST_BUCKET']))
        test.kernel._launch=kernel  # Explicit CPU kernel-boundary simulator.
        operation.attach_continuation(test.attention(query,sources['qsa.0'],logical,test.table,
            rids,len(mapping)-1,test.arena,True,output,e3op=operation))
        max_wave=0
        for _ in range(2000):
            before=len(test.device.copies)
            try:value=operation.advance()
            except StopIteration:break
            max_wave=max(max_wave,len(test.device.copies)-before)
            assert max_wave<=256
            if (cancel_second and len(snapshots)==1 and getattr(value,'sparse_copy_pending',False)):
                assert operation._layers['qsa.0'].sparse.next_bucket==1
                assert hold.stats['published']==1
                operation.cancel('second bucket prefix')
                assert operation.drain()
                assert hold.stats['consumed']==1 and hold.stats['abandoned']==1
                assert len(snapshots)==1 and len(test.kernel.calls)==1
                operation.release()
                assert hold.released
                return dict(status='PASS',cancel_second_bucket=True,complete_buckets=1,
                    abandoned_prefix=1,partial_published=False,source_released=True,
                    max_descriptors_per_advance=max_wave,record_count=len(operation._records))
        else:raise AssertionError('continuation exceeded bounded loop')
        assert not cancel_second,'second bucket cancellation was not exercised'
        assert [x['rows'] for x in snapshots]==[97152,97152,51816]
        assert [x['first'] for x in snapshots]==[True,False,False]
        assert [x['last'] for x in snapshots]==[False,False,True]
        assert len(test.device.copies)==count
        assert operation.stats['copy_events']==963
        assert len(operation._records)==8
        operation.end_layer('qsa.0');operation.end_forward()
        assert operation.drain();operation.release()
        assert hold.stats['consumed']==3 and hold.released
        return dict(status='PASS',selected_rows=count,buckets=snapshots,copy_events=963,
            max_descriptors_per_advance=max_wave,record_count=8,source_released=True)
    finally:test.doCleanups()


def expanded_reports():
    s=make_long_stack(tokens=2,layers=48)
    start=int(s.driver._monotonic_ns())
    s.drive(lambda:s.driver._split_ready)
    for worker in s.workers.values():
        worker.lane.pause_counts={layer:180 for layer in worker.lane.layers}
    s.drive(lambda:s.driver.phase=='finalize',limit=60000)
    s.complete_everything();s.drive(lambda:s.driver.tx is None)
    end=int(s.driver._monotonic_ns())+1
    evidence=s.driver.measurement_census.evidence(start,end,now_ns=end+1)
    assert evidence['completeness']['complete'],evidence['reasons']
    assert evidence['segments'][0]['segment_count']==8698
    assert evidence['segments'][0]['stage_segment_counts']=={'0':4349,'1':4349}
    reports=[row for row in s.driver._journal if row['kind']=='report']
    paged=[row for row in reports if isinstance(row['proofs'],dict)]
    assert any(row['proofs']['levels']==2 for row in paged)
    assert s.driver.error is None
    return dict(status='PASS',layers=48,pauses_per_layer=180,accepted_segments=8698,
        stage_segments={'0':4349,'1':4349},proof_counts=[len(r['proofs']) for r in evidence['completions']],
        complete=True,raw_journal_dropped=s.driver._journal_dropped,
        actual_two_level_paged_proof=True,retained_host_bytes=s.driver.measurement_census.host_bytes_estimate(),
        aggregate_limits=s.driver.measurement_census.stats()['capacity'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--allow-validation-hoist',action='store_true');a=p.parse_args()
    assert not a.out.exists()
    result=dict(schema='native14-postdelivery-stress/1',frozen=check_frozen(a.allow_validation_hoist),cases={},scope='CPU fixture/device boundaries; no new GPU, full-model, TP/PP hardware or numerical proof',runtime_changed=a.allow_validation_hoist,bundle_changed=False)
    for name,fn in [('multibucket',multibucket),('cancel_second_bucket',lambda:multibucket(True)),('expanded_reports',expanded_reports)]:
        print('BEGIN '+name,flush=True);start=time.monotonic()
        result['cases'][name]=fn();result['cases'][name]['wall_s']=time.monotonic()-start
        print('END '+name+' PASS',flush=True)
    assert check_frozen(a.allow_validation_hoist)==result['frozen']
    result['status']='PASS'
    with a.out.open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
    print('POST_DELIVERY_STRESS PASS',flush=True)


if __name__=='__main__':main()
