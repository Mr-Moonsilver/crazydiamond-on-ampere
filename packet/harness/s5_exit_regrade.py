#!/usr/bin/env python3
"""Additive, offline measurement salvage; NEVER a clean-exit/release PASS.

Input is a profile-terminal case directory and its owner's idle handoff. Original
grades/logs stay unchanged. QUALIFIED_MEASUREMENT (exit 0) means only the raw
measurement passed and a narrowly identified allocator teardown failure occurred
after both ranks entered cleanup. Tail, correctness and rank-error gates remain.
"""
import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
from s5_value import grade


WARNING = re.compile(r'^\[W(\d{4}) (\d{2}:\d{2}:\d{2})\.(\d{9}) CachingHostAllocator\.cpp:26\] '
                     r'Warning: Exception in pinned allocator free\(\), rethrowing \(function free\)$', re.M)
RANK_EXIT = re.compile(r'^\s*rank\s*:\s*(\d+) \(local_rank: (\d+)\)\s*\n'
                      r'\s*exitcode\s*:\s*(-?\d+) \(pid: (\d+)\)', re.M)


def ns(value):
    value=Decimal(str(value))
    if not value.is_finite():raise ValueError('nonfinite timestamp')
    return int(value*10**9)


def warning_times(log, near_s):
    """C10 MMDD UTC clock, full nine fractional digits; no float truncation.

    JSON wall_s has its original ~sub-microsecond precision. Choosing the nearest
    year handles midnight/New Year without treating stderr merge order as time.
    """
    near=ns(near_s);year=datetime.fromtimestamp(float(near_s),timezone.utc).year
    times=[]
    for mmdd,clock,fraction in WARNING.findall(log):
        candidates=[]
        for y in (year-1,year,year+1):
            try:dt=datetime.strptime(f'{y}{mmdd} {clock}','%Y%m%d %H:%M:%S').replace(tzinfo=timezone.utc)
            except ValueError:continue
            candidates.append(int(dt.timestamp())*10**9+int(fraction))
        if not candidates:raise ValueError('invalid allocator warning timestamp')
        times.append(min(candidates,key=lambda t:abs(t-near)))
    return times


def check_log(log, phases):
    warnings=warning_times(log,phases[0]['cleanup_wall_s'])
    if len(warnings)!=2:raise ValueError('expected exactly two known allocator warnings')
    start=max(ns(p['cleanup_wall_s']) for p in phases)
    end=max(ns(p['completed_wall_s']) for p in phases)
    if not all(start<=t<=end+60*10**9 for t in warnings):
        raise ValueError('allocator warning predates both cleanups or is not near teardown')
    # No rank attribution is inferred from interleaved native stderr. Require
    # BOTH workers to have reached cleanup before the EARLIEST warning.
    for required in ('Exception raised from record at ', 'CUDAEvent.h:152',
                     'PyGC_Collect', 'Py_FinalizeEx', 'Fatal Python error:',
                     'Aborted', 'c10::AcceleratorError', 'Signal 6 (SIGABRT)'):
        if required not in log:raise ValueError('missing native allocator/finalization signature')
    cuda_errors=re.findall(r'CUDA error:\s*([^\n]+)',log)
    # q11s resident34/dma34: the SAME CUDAEvent.h:152 + GC/finalize failure
    # reports invalid-device-context on the second worker. Accept only these
    # two spellings under all the existing chronology/stack/rank-exit checks.
    cuda_errors=[x.strip() for x in cuda_errors]
    if len(cuda_errors)!=2 or any(x not in ('context is destroyed','invalid device context') for x in cuda_errors):
        raise ValueError('unrecognized CUDA error')
    if re.search(r'(?i)out of memory|illegal memory|device.side assert|watchdog|segmentation fault|'
                 r'nccl(?:Unhandled|System|Internal|Invalid|Remote)|NCCL WARN|'
                 r'correctness.*(?:fail|differ)|(?:fail|mismatch).*correctness|timed out',log):
        raise ValueError('additional failure signature')
    errors=re.findall(r'^([\w.]+(?:Error|Exception)):',log,re.M)
    if any(e!='torch.distributed.elastic.multiprocessing.errors.ChildFailedError' for e in errors):
        raise ValueError('additional Python error')
    exits=RANK_EXIT.findall(log)
    expected={(str(p['rank']),str(p['rank']),'-6',str(p['pid'])) for p in phases}
    if len(exits)!=2 or set(exits)!=expected:raise ValueError('both rank/PID SIGABRT exits required')
    return dict(cuda_errors=cuda_errors,warning_utc_ns=warnings,latest_cleanup_utc_ns=start,
                latest_completed_utc_ns=end,
                warnings_after_all_completed=all(t>=end for t in warnings),
                warning_offset_ms_from_latest_cleanup=[(t-start)/1e6 for t in warnings],
                chronology='warnings are unranked; both ranks entered cleanup before any warning')


def check_correctness(raw):
    """Keep terminal_verify's exact solo-set oracle, NOT first-solo equality.

    Marlin can yield two distinct solo outputs. A concurrent output must match
    one observed solo output bit-for-bit; recurrent states must always agree.
    This permits no numerical tolerance and no novel concurrent-only output.
    """
    for rank in raw['ranks']:
        probe=rank['value_probe']
        names=['correctness']+(['noop_correctness'] if probe['prefill_noop'] else [])
        if probe.get('resident_prefill'):names.append('resident_correctness')
        for name in names:
            c=rank[name]
            if c['passed'] is not True:raise ValueError('correctness failed')
            for lane in ('decode','prefill'):
                v=c[lane];comparisons=v['comparisons'];n=v['solo_runs']
                if n!=4 or len(comparisons)!=n or [x['solo'] for x in comparisons]!=list(range(n)):
                    raise ValueError('incomplete solo oracle')
                if not all(v[k] is True for k in ('output_matches_solo','solo_states_identical','concurrent_states_match','finite')):
                    raise ValueError('solo-set/state/finite oracle failed')
                if not any(x['output_bitwise'] is True for x in comparisons) or v['bitwise_first'] is not comparisons[0]['output_bitwise']:
                    raise ValueError('no exact observed solo match')
                if type(v['distinct_solo_outputs']) is not int or not 1<=v['distinct_solo_outputs']<=n:
                    raise ValueError('invalid solo diversity')


def regrade(folder,handoff):
    folder=Path(folder);handoff=Path(handoff);hashes={}
    out=dict(verdict='INCONCLUSIVE',measurement_accepted=False,measurement_verdict='INCONCLUSIVE',
             process_exit_clean=False,original_verdict=None,qualification=None,
             release_verdict='NOT_EVALUATED',scope='measurement salvage only; not liveness, leak or release clearance')
    def read(path):
        data=path.read_bytes();hashes[str(path)]=hashlib.sha256(data).hexdigest()
        return data.decode('utf-8')
    def load(path):return json.loads(read(path))
    try:
        stamped=load(folder/'grade.json');out['original_verdict']=stamped.get('verdict')
        if stamped.get('verdict')=='FAIL':out['verdict']='FAIL'
        errors=list(folder.glob('result.json.rank*.error.json'))
        if errors:
            for p in errors:
                if load(p).get('fatal_correctness') is True:out['verdict']='FAIL'
            raise ValueError('rank error artifact forbids salvage')
        raw=load(folder/'result.json')
        measured=grade(raw);out['measurement_verdict']=measured['verdict']
        if stamped.get('verdict')=='FAIL' or measured['verdict']=='FAIL':out['verdict']='FAIL'
        if measured['verdict']!='PASS' or out['verdict']=='FAIL':raise ValueError('raw/recorded measurement gate did not pass')
        if raw.get('rung')!='profile-terminal':raise ValueError('only terminal value measurements supported')
        if not measured['ranks']:raise ValueError('measurement rows required')
        check_correctness(raw)
        # Do not turn any other INCONCLUSIVE into a measurement pass. This must
        # be exactly the queue's launcher-only downgrade of the SAME raw grade.
        launch=int(read(folder/'launch-rc').strip())
        expected=dict(measured,verdict='INCONCLUSIVE',launcher_rc=1,reason='process did not exit cleanly')
        if launch!=1 or stamped!=expected:raise ValueError('not the exact launcher-only downgrade')
        h=load(handoff)
        if h.get('own_container_stopped')!=1 or h.get('gpu_idle')!=1 or h.get('engine_untouched') is not True:
            raise ValueError('owner cleanup/idle handoff missing')
        phases=[]
        for rank in (0,1):
            final=load(folder/f'result.json.rank{rank}.progress.json')
            rows=[json.loads(s) for s in read(folder/f'result.json.rank{rank}.progress.jsonl').splitlines()]
            if final.get('rank')!=rank or final.get('phase')!='completed' or type(final.get('pid')) is not int:
                raise ValueError('completed rank/PID snapshot required')
            if not rows or any(r.get('rank')!=rank for r in rows):raise ValueError('mixed/missing rank history')
            if any(type(r.get('seq')) is not int for r in rows) or any(a['seq']>=b['seq'] for a,b in zip(rows,rows[1:])):
                raise ValueError('invalid progress sequence')
            cleanup=[(i,r) for i,r in enumerate(rows) if r.get('kind')=='phase' and r.get('phase')=='cleanup']
            if len(cleanup)!=1:raise ValueError('one cleanup entry required')
            pos,c=cleanup[0];last=rows[-1]
            if any(r.get('kind')!='phase' or not (r.get('phase','').startswith('cleanup') or r.get('phase')=='completed') for r in rows[pos:]):
                raise ValueError('work continued after cleanup began')
            if last.get('kind')!='phase' or any(last.get(k)!=final.get(k) for k in ('rank','pid','phase','wall_s')):
                raise ValueError('snapshot/history disagree')
            if c.get('pid')!=final['pid'] or not ns(c['wall_s'])<=ns(final['wall_s'])<=ns(c['wall_s'])+60*10**9:
                raise ValueError('cleanup completion not bounded/ordered')
            phases.append(dict(rank=rank,pid=final['pid'],cleanup_wall_s=c['wall_s'],completed_wall_s=final['wall_s']))
        timing=check_log(read(folder/'run.log'),phases)
        out.update(verdict='QUALIFIED_MEASUREMENT',measurement_accepted=True,
                   qualification='ALLOCATOR_TEARDOWN_FAILURE',measurement=measured,
                   phases=phases,timing=timing,recorded_external_cleanup=h,
                   cleanup_scope='container stopped and GPU idle; in-process allocator cleanup FAILED')
    except (OSError,ValueError,KeyError,TypeError,AttributeError,IndexError,ArithmeticError):
        # No raw log lines, payloads, stack traces or exception values in reports.
        out['reason']='required evidence missing, inconsistent, or outside the narrow allocator-teardown case'
    out['evidence_sha256']=hashes
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('case',type=Path,help='profile-terminal directory with result.json and run.log')
    p.add_argument('--handoff',required=True,type=Path,help='that case owner wrapper handoff.json')
    p.add_argument('--out',required=True,type=Path,help='NEW report path; never overwrite a grade')
    a=p.parse_args();r=regrade(a.case,a.handoff)
    with a.out.open('x') as f:f.write(json.dumps(r,allow_nan=False,indent=2)+'\n')
    print(f"S5_EXIT_REGRADE {r['verdict']} measurement={r['measurement_verdict']} process_exit_clean=false")
    return {'QUALIFIED_MEASUREMENT':0,'FAIL':1,'INCONCLUSIVE':2}[r['verdict']]


if __name__=='__main__':raise SystemExit(main())
