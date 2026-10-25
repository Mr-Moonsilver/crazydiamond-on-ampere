#!/usr/bin/env python3
"""Realized-exposure comparison for the q16 matched point (Codex q16 v4 review, reviews/codex-20261025as).

Equal protocol (same leak tool, SEED, cache namespace, pressure_case code) is not identical realized history: pressure2
streams until STOP at temperature 0.7 with variable completed request counts, leak sessions may retry, cancellations
land at different points. The matched retry verdict (e2_ledger_grade.py --matched) is therefore attributable only when
the realized work agrees. This tool compares the control and shipping artifacts of the two matched steps:
  leak pairs      per (round, session): prompt/generation lengths (identical required), prefix hits cached/cached3
                  (identical required), retry-path counts (answer_retries/answer3_retries records) and cancellations
                  (within tolerance).
  pressure pairs  issued and completed request totals, wall duration, KV occupancy mean/max, preemptions, prefix-cache
                  hits delta, error counts (within tolerance; preemptions within max(2, tolerance)).
usage: exposure_compare.py OUT.json --leak CONTROL.json SHIP.json [--leak ...] --pressure CONTROL.json SHIP.json [...]
       [--tolerance 0.10]
Exit 0 MATCHED (attribution allowed), 2 DIFFERS (attribution INCONCLUSIVE), 1 on unreadable/incomplete artifacts
(INCONCLUSIVE as well; nothing is invented). Numbers only; prompts and answers are never printed.
"""
import argparse, json, sys


def rel_ok(a, b, tol):
    if a is None or b is None: return False
    m = max(abs(a), abs(b))
    return True if m == 0 else abs(a - b) / m <= tol


def leak_pair(a, b, tol):
    out, diffs = {}, []
    key = lambda d: {(r['round'], r['session']): (len(r['p1_ids']), len(r['gen1_ids']), r['p2_len'], len(r.get('p3', []))) for r in d['results']}
    hits = lambda d: {(r['round'], r['session']): (r.get('cached'), r.get('cached3')) for r in d['results']}
    retries = lambda d: sum(1 for r in d['results'] for k in ('answer_retries', 'answer3_retries') if r.get(k))  # sessions that needed a path-recreating retry
    ka, kb, ha, hb = key(a), key(b), hits(a), hits(b)
    out['sessions'] = (len(ka), len(kb))
    out['length_mismatches'] = [list(k) for k in sorted(set(ka) | set(kb)) if ka.get(k) != kb.get(k)][:8]
    out['hit_mismatches'] = [[list(k), ha.get(k), hb.get(k)] for k in sorted(set(ha) | set(hb)) if ha.get(k) != hb.get(k)][:8]
    out['retry_records'] = (retries(a), retries(b))
    out['cancelled'] = (len((a.get('run') or {}).get('cancelled') or []), len((b.get('run') or {}).get('cancelled') or []))
    out['errors'] = (len(a.get('errors') or []), len(b.get('errors') or []))
    if not ka or len(ka) != len(kb): diffs.append('session counts differ or empty')
    if out['length_mismatches']: diffs.append('prompt/generation lengths differ')
    if out['hit_mismatches']: diffs.append('prefix hits differ')
    if not (abs(out['retry_records'][0] - out['retry_records'][1]) <= 2 or rel_ok(*out['retry_records'], tol)): diffs.append('retry-path counts differ')
    if not (abs(out['cancelled'][0] - out['cancelled'][1]) <= 2 or rel_ok(*out['cancelled'], tol)): diffs.append('cancellation counts differ')
    return out, diffs


def pressure_pair(a, b, tol):
    out, diffs = {}, []
    tot = lambda d, k: sum(d.get(k) or []) if isinstance(d.get(k), list) else d.get(k)
    dur = lambda d: (d['ended_wall'] - d['started_wall']) if 'ended_wall' in d and 'started_wall' in d else None
    occ = lambda d, s: ((d.get('occupancy') or {}).get('kv_cache_usage_perc') or {}).get(s)
    out['issued'] = (tot(a, 'issued'), tot(b, 'issued'))
    out['completed'] = (tot(a, 'completed_requests'), tot(b, 'completed_requests'))
    out['duration_s'] = (None if dur(a) is None else round(dur(a), 1), None if dur(b) is None else round(dur(b), 1))
    out['kv_occupancy_mean'] = (occ(a, 'mean'), occ(b, 'mean')); out['kv_occupancy_max'] = (occ(a, 'max'), occ(b, 'max'))
    out['preemptions'] = (a.get('preemptions'), b.get('preemptions'))
    out['prefix_cache_hits_delta'] = (a.get('prefix_cache_hits_delta'), b.get('prefix_cache_hits_delta'))
    out['errors'] = (len(a.get('errors') or []), len(b.get('errors') or []))
    for k in ('issued', 'completed', 'duration_s'):
        if not rel_ok(out[k][0], out[k][1], tol): diffs.append(k + ' differ beyond tolerance')
    if out['kv_occupancy_mean'][0] is None or out['kv_occupancy_mean'][1] is None: diffs.append('kv occupancy missing')
    elif abs(out['kv_occupancy_mean'][0] - out['kv_occupancy_mean'][1]) > 100 * tol: diffs.append('kv occupancy mean differs beyond tolerance (percentage points)')
    p0, p1 = out['preemptions']
    if p0 is None or p1 is None: diffs.append('preemptions missing')
    elif not (abs(p0 - p1) <= 2 or rel_ok(p0, p1, tol)): diffs.append('preemptions differ')
    if out['errors'] != (0, 0) and out['errors'][0] != out['errors'][1]: diffs.append('error counts differ')
    return out, diffs


def compare(leaks, pressures, tol=0.10):
    report, diffs = dict(tolerance=tol, leak=[], pressure=[]), []
    for name, a, b in leaks:
        o, d = leak_pair(a, b, tol); o['name'] = name; report['leak'].append(o); diffs += [name + ': ' + x for x in d]
    for name, a, b in pressures:
        o, d = pressure_pair(a, b, tol); o['name'] = name; report['pressure'].append(o); diffs += [name + ': ' + x for x in d]
    if not leaks and not pressures: diffs.append('nothing compared')
    report['differences'] = diffs
    report['verdict'] = 'MATCHED' if not diffs else 'DIFFERS'
    report['scope'] = 'realized exposure of the matched steps; DIFFERS = retry attribution INCONCLUSIVE, not a regression verdict'
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('out'); p.add_argument('--leak', nargs=2, action='append', default=[], metavar=('CONTROL', 'SHIP'))
    p.add_argument('--pressure', nargs=2, action='append', default=[], metavar=('CONTROL', 'SHIP'))
    p.add_argument('--tolerance', type=float, default=0.10)
    a = p.parse_args()
    name = lambda f: f.rsplit('/', 1)[-1].removesuffix('.json')
    try:
        leaks = [(name(ca), json.load(open(ca)), json.load(open(sb))) for ca, sb in a.leak]
        pressures = [(name(ca), json.load(open(ca)), json.load(open(sb))) for ca, sb in a.pressure]
        rep = compare(leaks, pressures, a.tolerance)
    except (OSError, ValueError, KeyError, TypeError) as e:
        rep = dict(verdict='INCONCLUSIVE', error=type(e).__name__ + ': ' + str(e)[:200])
    json.dump(rep, open(a.out, 'x'), sort_keys=True)
    print(json.dumps({k: v for k, v in rep.items() if k not in ('leak', 'pressure')}, sort_keys=True))
    sys.exit({'MATCHED': 0, 'DIFFERS': 2}.get(rep['verdict'], 1))
