#!/usr/bin/env python3
"""Physical query samples inside the regraded native fresh-pressure window.

512 MiB is a hard reserve. Allocator peaks are separately queried per worker;
reserved allocator memory is never substituted for physical free memory.
Sampling proves observed points, not continuous free-memory availability.
"""
import argparse
import json
import math
from pathlib import Path

from e3_native_performance import native_window


def number(value):
    return type(value) in (int,float) and math.isfinite(value)


def grade(performance,samples):
    failures,missing=[],[]
    row=((performance.get('lanes') or {}).get('overlap') or {}).get('fresh') or {}
    proof=native_window(row,'fresh',performance.get('engine_epoch'))
    if proof.get('complete') is not True:
        missing.append('native-fresh-pressure-window-unproved')
    group=proof.get('group') or {}
    start,end=group.get('start_ns'),group.get('end_ns')
    anchor,mono=performance.get('wall_anchor_s'),performance.get('wall_anchor_ns')
    uncertainty=performance.get('wall_anchor_uncertainty_ns')
    endwall,endmono=performance.get('wall_end_s'),performance.get('wall_end_ns')
    clocks=(all(number(x) for x in (start,end,anchor,mono,uncertainty,endwall,endmono))
            and end>start and endmono>mono and 0<=uncertainty<=50_000_000
            and abs((endwall-anchor)-(endmono-mono)/1e9)<=.1)
    bounds=None
    if not clocks:
        missing.append('wall-monotonic-binding-unproved')
    else:
        bounds={'start_s':anchor+(start-mono+uncertainty)/1e9,
                'end_s':anchor+(end-mono-uncertainty)/1e9,
                'basis':'entire physical query bracket inside source-joined fresh overlap interval'}
    retained=[];minima={}
    for index,sample in enumerate(samples):
        if type(sample) is not dict or bounds is None:continue
        a,b=sample.get('start_s'),sample.get('end_s')
        if not (number(a) and number(b) and bounds['start_s']<=a<=b<=bounds['end_s']):continue
        if sample.get('error') or b-a>.5:
            missing.append('physical-query-unproved:'+str(index));continue
        gpus=sample.get('gpus')
        if type(gpus) is not dict or set(gpus)!={'0','1','2','3'}:
            missing.append('four-physical-gpus-unproved:'+str(index));continue
        values={}
        for rank in range(4):
            value=gpus[str(rank)]
            used,total=value.get('used_mib'),value.get('total_mib')
            if not (number(used) and number(total) and 0<=used<=total and total>0):
                missing.append('physical-value-unproved:'+str(index));continue
            free=total-used;values[rank]=free
            minima[rank]=min(minima.get(rank,float('inf')),free)
            if free<512:failures.append('512MiB_reserve_breached:gpu%d:%.1f'%(rank,free))
        retained.append({'source_row':index,'start_s':a,'end_s':b,'free_mib':values})
    if not retained or set(minima)!=set(range(4)):
        missing.append('fresh-pressure-physical-samples-missing')
    rows=((row.get('drain') or {}).get('barrier') or {}).get('workers')
    allocator={}
    if type(rows) is not list or len(rows)!=4:
        missing.append('four-queried-allocator-peaks-missing')
    else:
        for worker in rows:
            rank=worker.get('worker_rank');value=worker.get('allocator') or {}
            if (type(rank) is not int or rank not in range(4) or rank in allocator
                    or worker.get('engine_epoch')!=performance.get('engine_epoch')
                    or worker.get('engine_epoch_ok') is not True
                    or value.get('basis')!='torch-cuda-allocator-queried-v1'
                    or any(type(value.get(key)) is not int or value[key]<0
                           for key in ('allocated_peak_bytes','reserved_peak_bytes'))):
                missing.append('worker-allocator-identity-unproved');continue
            allocator[rank]=value
        if set(allocator)!=set(range(4)):missing.append('four-queried-allocator-peaks-missing')
    return {'schema':'s5-native-memory-grade/1',
            'verdict':'FAIL' if failures else 'INCONCLUSIVE' if missing else 'PASS',
            'failures':sorted(set(failures)),'missing':sorted(set(missing)),
            'reserve_mib':512,'min_free_mib_under_pressure':minima,
            'pressure_window':bounds,'physical_samples':retained,'allocator_peaks':allocator,
            'scope':'observed physical query points under fresh native pressure; lifetime allocator peaks; no continuous reserve guarantee'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ('performance','samples','out'):p.add_argument('--'+name,required=True)
    a=p.parse_args(argv)
    if Path(a.out).exists():return 2
    try:
        performance=json.loads(Path(a.performance).read_text())
        samples=[json.loads(line) for line in Path(a.samples).read_text().splitlines()]
        record=grade(performance,samples)
    except Exception as exc:
        record={'verdict':'INCONCLUSIVE','failures':[],'missing':['memory-input-unproved:'+type(exc).__name__]}
    with open(a.out,'x') as handle:json.dump(record,handle,sort_keys=True)
    print('NATIVE_MEMORY '+record['verdict'])
    return {'PASS':0,'FAIL':1}.get(record['verdict'],2)


if __name__=='__main__':raise SystemExit(main())
