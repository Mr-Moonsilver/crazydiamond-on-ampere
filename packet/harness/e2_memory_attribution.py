#!/usr/bin/env python3
"""Diagnostic allocator/physical-memory observations; never a reserve grader.

sample writes the original 3-column GPU CSV AND UTC query brackets in JSONL.
report pairs allocator observations with nearby physical samples, retaining
skew and both surrounding samples. Separate observations are NOT simultaneous
and used-minus-reserved is an estimate, not proven native allocation.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import signal
import subprocess
import time

RANKS = tuple(f'PP{pp}_TP{tp}' for pp in (0, 1) for tp in (0, 1))
MIB = 1024**2


def parse_gpu(text):
    result = {}
    for fields in csv.reader(text.splitlines()):
        if len(fields) != 3:
            raise ValueError('GPU CSV needs index, used MiB, total MiB')
        i, used, total = map(int, fields)
        if i < 0 or not 0 <= used <= total or total <= 0 or str(i) in result:
            raise ValueError('invalid or duplicate GPU sample')
        result[str(i)] = dict(used_mib=used, total_mib=total)
    if set(result) != {'0', '1', '2', '3'}:
        raise ValueError('four physical GPUs required')
    return result


def sample(csv_path, timed_path, interval=2):
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, stop)
    with csv_path.open('x', buffering=1) as raw, timed_path.open('x', buffering=1) as timed:
        while not stopping:
            begin, mono = time.time(), time.monotonic()
            row = dict(clock='unix_realtime_seconds', start_s=begin)
            try:
                p = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total',
                                    '--format=csv,noheader,nounits'],
                                   capture_output=True, text=True, timeout=10, check=True)
                row['gpus'] = parse_gpu(p.stdout)
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                row['error'] = type(exc).__name__  # No raw command output.
            row.update(end_s=time.time(), elapsed_s=time.monotonic()-mono)
            if abs((row['end_s']-begin)-row['elapsed_s']) > 0.1:
                row['error'] = 'ClockStep'
            if 'gpus' in row:
                for i, values in sorted(row['gpus'].items()):
                    raw.write(f"{i}, {values['used_mib']}, {values['total_mib']}\n")
            timed.write(json.dumps(row, sort_keys=True)+'\n')
            # Same cadence as the old query; sleep 2 s AFTER each query.
            end = time.monotonic()+interval
            while not stopping and time.monotonic() < end:
                time.sleep(min(0.1, max(0, end-time.monotonic())))
        timed.write(json.dumps(dict(kind='stopped',wall_s=time.time(),
                                    clock='unix_realtime_seconds'))+'\n')


def finite(x):
    return type(x) in (int, float) and math.isfinite(x)


def nearest_pair(record, samples, gpu, max_skew):
    when = record['wall_s']
    usable = [s for s in samples if 'error' not in s and str(gpu) in s.get('gpus', {})
              and finite(s.get('start_s')) and finite(s.get('end_s'))
              and s['start_s'] <= s['end_s'] and s['end_s']-s['start_s'] <= 0.5]
    if not usable:
        return dict(paired=False, why='no valid short GPU query brackets')
    def middle(s): return (s['start_s']+s['end_s'])/2
    s = min(usable, key=lambda s: abs(middle(s)-when))
    skew = middle(s)-when
    def point(s):
        return dict(start_s=s['start_s'],end_s=s['end_s'],**s['gpus'][str(gpu)])
    before = [q for q in usable if q['end_s'] <= when]
    after = [q for q in usable if q['start_s'] >= when]
    pair = dict(paired=abs(skew)<=max_skew, skew_s=skew, nearest=point(s),
                before=point(max(before,key=lambda q:q['end_s'])) if before else None,
                after=point(min(after,key=lambda q:q['start_s'])) if after else None)
    if pair['paired']:
        # Deliberately neither clamp negative residuals nor use peak_reserved.
        pair['non_allocator_used_estimate_bytes'] = (
            s['gpus'][str(gpu)]['used_mib']*MIB-record['counters']['reserved_bytes.all.current'])
    else:
        pair['why'] = 'nearest GPU sample exceeds skew bound'
    return pair


def report(lines, samples, ledger, sidecars, drive=None, baseline=None, max_skew=1.25):
    missing, observations = [], []
    mapping = {rank:ledger.get('per_rank',{}).get('Worker_'+rank,{}).get('gpu') for rank in RANKS}
    mapping_ok = all(type(i) is int for i in mapping.values()) and set(mapping.values())=={0,1,2,3}
    if not mapping_ok:
        missing.append('missing/non-unique ledger rank-to-physical-GPU mapping')
    for line in lines:
        if 'FLASH_E2_MEMORY ' not in line:
            continue
        try:
            r=json.loads(line.split('FLASH_E2_MEMORY ',1)[1])
            if r['rank'] not in RANKS or not finite(r['wall_s']): raise ValueError()
            c=r['counters']
            for k in ('reserved_bytes.all.current','allocated_bytes.all.current'):
                if type(c.get(k)) is not int or c[k]<0: raise ValueError()
            item={k:r[k] for k in ('rank','rpc_sequence','layer','boundary','wall_s')}
            item.update(counters=c,reserved_minus_allocated_bytes=c['reserved_bytes.all.current']-c['allocated_bytes.all.current'])
            active=c.get('active_bytes.all.current')
            item['reserved_minus_active_bytes']=(c['reserved_bytes.all.current']-active if type(active) is int else None)
            item['active_minus_allocated_bytes']=(active-c['allocated_bytes.all.current'] if type(active) is int else None)
            item['physical']=nearest_pair(r,samples,mapping[r['rank']],max_skew) if mapping_ok else dict(paired=False)
            observations.append(item)
        except (ValueError,KeyError,TypeError):
            missing.append('malformed allocator observation')
    workers={}
    for rank in RANKS:
        rows=[r for r in observations if r['rank']==rank]
        sc=sidecars.get(rank,{})
        mem=sc.get('memory',{})
        counts=sc.get('layers',{})
        expected=sum(4*v.get('calls',0)+2*v.get('buckets',0) for v in counts.values())
        if not rows or not mem.get('samples') or mem.get('missing'):
            missing.append(rank+': incomplete allocator observations/sidecar')
        if sc.get('rank')!=rank or sc.get('failed') is not False or sc.get('active') is not False:
            missing.append(rank+': sidecar is not a completed same-rank exactness record')
        if mem.get('samples')!=expected:
            missing.append(rank+': observation count does not match completed byte checks')
        paired=[r for r in rows if r['physical']['paired']]
        if not paired: missing.append(rank+': no time-near physical samples')
        workers[rank]=dict(gpu=mapping[rank],memory_stats_calls=mem.get('samples'),expected_calls=expected,
            emitted_records=len(rows),paired_records=len(paired),highwater=mem.get('highwater'),
            max_observed_reserved_minus_allocated_bytes=max((r['reserved_minus_allocated_bytes'] for r in rows),default=None))
    timing={}
    if drive is not None and baseline is not None:
        for part in ('fresh','append'):
            a,b=drive.get(part,{}),baseline.get(part,{})
            if (not finite(a.get('seconds')) or not finite(b.get('seconds')) or b['seconds']<=0 or
                any(a.get(k)!=b.get(k) for k in ('prompt_tokens','completion_tokens','cached_tokens'))):
                missing.append(part+': timing comparison has missing/unmatched workload evidence')
            else:
                timing[part]=dict(seconds=a['seconds'],q16h_seconds=b['seconds'],
                                  delta_s=a['seconds']-b['seconds'],ratio=a['seconds']/b['seconds'])
    return dict(verdict='INCONCLUSIVE' if missing else 'OBSERVED',missing=sorted(set(missing)),
        max_skew_s=max_skew,workers=workers,pairs=observations,timing_vs_q16h=timing,
        physical_query_errors=sum('error' in s for s in samples),
        scope='diagnostic only; UTC realtime clock on CT and worker; separate samples are not simultaneous; '
              'used-minus-current-reserved is a skew/rounding-sensitive estimate, not proven native allocation; '
              'no reserve verdict; emits bounded by monotonic high-water updates, not RPC count; '
              'verification wall-time ratios show observed difference, not isolated telemetry overhead')


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    s=sub.add_parser('sample');s.add_argument('csv',type=Path);s.add_argument('timed',type=Path)
    r=sub.add_parser('report');r.add_argument('engine',type=Path);r.add_argument('timed',type=Path)
    r.add_argument('ledger',type=Path);r.add_argument('sidecars',type=Path);r.add_argument('out',type=Path)
    r.add_argument('--drive',type=Path);r.add_argument('--baseline',type=Path)
    a=p.parse_args()
    if a.mode=='sample':sample(a.csv,a.timed);return 0
    try:
        load=lambda path:json.loads(path.read_text())
        value=report(a.engine.read_text().splitlines(),[json.loads(x) for x in a.timed.read_text().splitlines()],
                     load(a.ledger),{rank:load(a.sidecars/(rank+'.exact.json')) for rank in RANKS},
                     load(a.drive) if a.drive else None,load(a.baseline) if a.baseline else None)
    except (OSError,ValueError,KeyError,TypeError,AttributeError) as exc:
        value=dict(verdict='INCONCLUSIVE',missing=['unreadable diagnostic input: '+type(exc).__name__])
    with a.out.open('x') as f:json.dump(value,f,sort_keys=True)
    print(value['verdict'],'allocator/physical pairing; see JSON for skew and limits')
    return 0 if value['verdict']=='OBSERVED' else 2


if __name__=='__main__':raise SystemExit(main())
