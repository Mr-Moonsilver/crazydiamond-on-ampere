"""Native35 L reading: native lifetime (cancel/drain/reuse) and unchanged hygiene/leak gates.

Consolidates the retained per-phase records of the L arm. Every record keeps
its own verdict; this grade only requires them all to be present and PASS,
and it never relaxes a leak oracle. An engine crash marker is a FAIL.
"""
import argparse
import json
from pathlib import Path
import re

LEAK_SUMMARY=re.compile(r'^sessions 8 rounds 3 bad 0 in ',re.M)
LEAK_BAD=re.compile(r'^s\d+ r\d+ (ERR|LEAK|MISS)',re.M)


def _json(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return None


def grade(queue):
    q=Path(queue);failures,missing,records=[],[],{}
    def verdict_of(name,record,label):
        records[name]=None if record is None else {k:record.get(k) for k in ('verdict','failures','fails','missing','rc') if k in record}
        if record is None:missing.append(label+':record-missing');return
        v=record.get('verdict')
        if v=='FAIL':failures.append(label+':FAIL')
        elif v!='PASS':missing.append(label+':'+str(v))
    lifecycle=_json(q/'lifecycle.json');verdict_of('lifecycle',lifecycle,'native-lifetime')
    if type(lifecycle) is dict:
        close=(lifecycle.get('cancel_grade') or {})
        if close.get('verdict')!='PASS':missing.append('native-lifetime:typed-close:'+str(close.get('verdict')))
        if (lifecycle.get('reuse') or {}).get('complete') is not True:missing.append('native-lifetime:reuse-incomplete')
    hygiene=_json(q/'hygiene-full.json');records['hygiene_full']=hygiene is not None
    if hygiene is None or (hygiene.get('prepare') or {}).get('route')!='full':missing.append('hygiene-full:typed-full-route')
    try:text=(q/'cross-session-leak.log').read_text(errors='replace')
    except OSError:text=None
    if text is None:missing.append('cross-session-leak:log-missing')
    elif LEAK_BAD.search(text):failures.append('cross-session-leak:bad-session')
    elif not LEAK_SUMMARY.search(text):missing.append('cross-session-leak:summary-not-exact')
    records['cross_session_leak']=None if text is None else dict(exact_summary=bool(LEAK_SUMMARY.search(text)),bad=bool(LEAK_BAD.search(text)))
    verdict_of('partial_leak',_json(q/'partial-leak.grade.json'),'partial-leak')
    completion=_json(q/'pressure.completion.json');records['pressure_completion']=completion
    if completion is None:missing.append('pressure-completed:record-missing')
    elif completion.get('rc')!=0:(failures if completion.get('rc')==1 else missing).append('pressure-completed:rc='+str(completion.get('rc')))
    verdict_of('pressure_lifetime',_json(q/'pressure.coverage.json'),'pressure-lifetime')
    observer=_json(q/'host-observer-summary.json')
    if type(observer) is dict and (observer.get('shutdown') or {}).get('crash'):failures.append('engine-crash:host-observer')
    for name in ('engine-stream.log','engine-stream.err'):
        try:stream=(q/name).read_text(errors='replace')
        except OSError:continue
        if re.search(r'DistBackendError|illegal memory access|Worker proc .* died unexpectedly',stream):failures.append('engine-crash:'+name)
    return dict(schema='native35-lifetime-reading/1',verdict='FAIL' if failures else 'INCONCLUSIVE' if missing else 'PASS',
        failures=failures,missing=missing,records=records,release_qualified=False,
        scope='native cancel/drain/reuse on the private-metadata lane plus the unchanged FULL-route hygiene and leak oracles; no performance, placement or numerical claim')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--queue',required=True);p.add_argument('--out',required=True);a=p.parse_args(argv)
    result=grade(a.queue)
    with Path(a.out).open('x') as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
    print('NATIVE35_LIFETIME '+result['verdict'])
    return {'PASS':0,'FAIL':1}.get(result['verdict'],2)


if __name__=='__main__':raise SystemExit(main())
