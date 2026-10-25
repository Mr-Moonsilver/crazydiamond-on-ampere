"""Fail-closed PP grouping release evidence. CPU-only imports; no serving calls."""
import ast
import json
import re
from pathlib import Path
import sys
import cow_invariants
import leakgrade


def result(fails=(), missing=(), **extra):
    return dict(verdict='FAIL' if fails else ('INCONCLUSIVE' if missing else 'PASS'),
                fails=list(fails), missing=list(missing), **extra)


def leak(d, n, rounds, boundary, cancel=False, partial=True):
    rows = d.get('results', [])
    fails = list(d.get('errors', []))
    if (d.get('N'), d.get('ROUNDS'), d.get('B'), d.get('MODE')) != (n,rounds,boundary,'inside'):
        fails.append('wrong leak workload')
    if len(rows) != n*rounds or len({(r['round'],r['session']) for r in rows}) != n*rounds:
        fails.append('missing/duplicate session results')
    codes = [r['code'] for r in rows]
    lockers = [r['locker'] for r in rows if 'locker' in r]
    if len(set(codes)) != len(codes) or len(set(lockers)) != n*rounds:
        fails.append('duplicate/missing identifiers')
    cases = []
    for r in rows:
        if leakgrade.foreign_in(r.get('gen1'), [c for c in codes+lockers if c not in (r['code'],r.get('locker'))]):
            fails.append('foreign identifier in first output')
        for key, cold, cached, own, ids in (('answer','cold','cached',r['code'],codes),
                                         ('answer3','cold3','cached3',r.get('locker'),lockers)):
            if key not in r or own is None:
                fails.append('missing chained-copy answer'); continue
            foreign = [c for c in ids if c != own]
            first = leakgrade.first_verdict(r[key],own,foreign,r.get(cold))
            settled = leakgrade.settle(r.get(key+'_retries',[]),own,foreign,r.get(cached)) if first=='SUSPECT' else None
            cases.append(dict(first=first, settled=settled, partial=bool(r.get(cached)) and r[cached]%4048!=0))
    v = leakgrade.run_verdict(cases,n*rounds*2,fails,
         restarted=not d.get('identity_start') or d['identity_start'] != d.get('identity_end'), need_partial=partial)
    if cancel and not d.get('run',{}).get('cancelled',0):
        v['inconclusive'].append('cancellation not exercised')
        if v['verdict']=='PASS': v['verdict']='INCONCLUSIVE'
    return v


def invariant_rows(text):
    rows={}
    for line in text.splitlines():
        if 'QSA host-page CoW verify' not in line: continue
        w=re.search(r'Worker_PP\d+_TP\d+',line); inv=re.search(r'invariants (\{.*\})',line)
        if w and inv: rows[w[0]]=ast.literal_eval(inv[1])
    return rows


def pool(text):
    rows=[]
    for line in text.splitlines():
        if 'FLASH_POOL_STATS ' in line:
            rows.append(json.loads(line.split('FLASH_POOL_STATS ',1)[1]))
    return rows[-1]['groups'] if rows else {}


def lifetime(before, after, pressure, interval, verify):
    fails, missing=[],[]
    if pressure.get('errors'): fails.append('pressure stream error')
    n=pressure.get('streams',0)
    first,last=pressure.get('first_tokens',[]),pressure.get('last_tokens',[])
    if n <= 0 or len(first)!=n or len(last)!=n or not all(
            isinstance(a,(int,float)) and isinstance(b,(int,float)) and a <= interval[0] and b >= interval[1]
            for a,b in zip(first,last)):
        missing.append('pressure tokens did not bracket the leak interval on every stream')
    if len(pressure.get('completed_requests',[]))!=n or not all(x>0 for x in pressure.get('completed_requests',[])):
        missing.append('no completed pressure request on every stream')
    p0,p1=pool(before),pool(after)
    evictions=sum(v.get('evictions',0)-p0.get(k,{}).get('evictions',0) for k,v in p1.items())
    state_ids=set()
    for line in after.splitlines():
        if 'FLASH_PP_LAYOUT ' in line:
            layout=json.loads(line.split('FLASH_PP_LAYOUT ',1)[1])
            state_ids.update(str(g['id']) for g in layout['groups'] if g.get('kind')=='MambaSpec')
    if not state_ids: missing.append('no state-group layout evidence')
    state_evictions=sum(v.get('evictions',0)-p0.get(k,{}).get('evictions',0) for k,v in p1.items() if k in state_ids)
    if state_evictions <= 0: missing.append('no state-group eviction observed during lifetime interval')
    if evictions < 0: fails.append('pool counters went backwards')
    if verify:
        lines=[l for l in after.splitlines() if 'QSA host-page CoW verify' in l]
        ok,why=cow_invariants.check(lines)
        if not ok: fails.append('CoW byte invariants: '+why)
        a,b=invariant_rows(before),invariant_rows(after)
        if set(b)!=cow_invariants.RANKS or any(b[w].get('src_reallocated',0)-a.get(w,{}).get('src_reallocated',0)<=0 for w in b):
            missing.append('copy-source reuse not observed during pressure on all four ranks')
    return result(fails,missing,state_evictions=state_evictions,evictions=evictions,
                  preemptions=pressure.get('preemptions'),
                  scope='QSA copy lifetime + grouped-state eviction; preemption is reported, not assumed reachable')


def canary(d,n,rounds,request_delta,traffic=None,run_id=None):
    fails,missing=[],[]
    sessions=d.get('sessions',[])
    if (d.get('expected')!=n*rounds or len(sessions)!=n*rounds or
        {(str(s.get('round')),str(s.get('session'))) for s in sessions} !=
        {(str(r),str(i)) for r in range(1,rounds+1) for i in range(1,n+1)}):
        fails.append('canary expected-session coverage mismatch')
    if d.get('verdict')=='FAIL': fails.append('canary grader failed')
    elif d.get('verdict')!='PASS': missing.append('canary grader did not pass')
    if any(s.get('verdict')!='OK' for s in sessions): missing.append('not every canary session OK')
    ids=[i for s in sessions for i in s.get('request_ids',[])]
    if not ids or len(set(ids))!=len(ids): missing.append('absent/duplicate canary request identity')
    traffic=traffic or {}
    calls=traffic.get('requests',[])
    markers={f'{run_id}-r{r}-s{i}' for r in range(1,rounds+1) for i in range(1,n+1)}
    if traffic.get('active')!=0: missing.append('relay requests still active or evidence missing')
    if not calls or any(c.get('session') not in markers for c in calls): missing.append('unattributed relay traffic')
    if len({c.get('id') for c in calls})!=len(calls): missing.append('duplicate relay request id')
    if any(c.get('outcome')!='completed' or c.get('status')!=200 for c in calls):
        missing.append('relay recorded disconnect, failure or incomplete stream')
    if any(c.get('kind') not in ('messages','count_tokens') for c in calls):
        missing.append('unclassified client request route')
    messages=[c for c in calls if c.get('kind')=='messages' and c.get('outcome')=='completed' and c.get('status')==200]
    if {c.get('session') for c in messages}!=markers: missing.append('not every canary session used the relay')
    if request_delta != len(messages): missing.append('engine success delta differs from relay completions: foreign traffic or gateway retry unresolved')
    if not d.get('partial_hits',0): missing.append('canary did not exercise prefix reuse')
    return result(fails,missing,transcript_requests=len(ids),relay_completed_messages=len(messages),
                  engine_request_delta=request_delta)


if __name__=='__main__':
    # Shell-facing commands use complete evidence files and never trust saved PASS alone.
    try:
        cmd=sys.argv[1]
        if cmd=='leak':
            d=json.loads(Path(sys.argv[2]).read_text())
            out=leak(d,int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]),sys.argv[6]=='1',sys.argv[7]=='1')
        elif cmd=='lifetime':
            out=lifetime(Path(sys.argv[2]).read_text(),Path(sys.argv[3]).read_text(),
                         json.loads(Path(sys.argv[4]).read_text()),json.loads(Path(sys.argv[5]).read_text()),sys.argv[6]=='1')
        else: raise ValueError('unknown grading command')
    except (ValueError,KeyError,TypeError,OSError,IndexError) as e:
        out=result(['missing/malformed evidence: '+type(e).__name__])
    print(json.dumps(out,indent=2))
    raise SystemExit({'PASS':0,'FAIL':1,'INCONCLUSIVE':2}[out['verdict']])
