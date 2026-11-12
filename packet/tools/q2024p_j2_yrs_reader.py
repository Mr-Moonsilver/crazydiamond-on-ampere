#!/usr/bin/env python3
"""Exploratory reader for the q2024p-j2 yield re-entry settle pair (yrs off/on). Read-only; no rule, no thresholds, no
verdict.

Everything the hgb reader prints (tools/q2024p_j2_hgb_reader.py, reused unchanged: turns/h, x_sp, native share, the
small_piece_batch delta, <=176-row stage times, decode token gaps, numerics/grades/OOM/fatal), and per arm:
  * the policy's yield re-entry block delta (append_yield.declines.yield_reentry, after minus before): refusals by
    stage:field, refusals by retry index, retries-until-armed histogram, exceeded, barrier passes and first-try passes,
    settle waits/ticks/settled/bound/skips, errors, armed ms (count, mean; lifetime max);
  * the 'flash-s5 yield-reentry proof-miss' log lines inside the arm's replay window, by stage:field and by retry;
  * the FULL return handoff per path inside the arm's replay window, cut as in the lever-2 seam read: the run's last
    all-rank 'memory point=released' -> the next rank0 'memory point=open'; path 'yield' when the interval carries
    append_yield_full/append_yield_reenter, else 'run'. Per path: n, mean/p50 ms, lane-queued agent-s (admitted and not
    yet at its own last release), the re-entry's barrier-loop time (first drain after auto_reentry -> last drain before
    admission_armed) and the drains per episode;
and per record: FLASH_S5_YIELD_REENTRY_SETTLE in the engine's start/end identity env.

The diagnostic is always on, so both records must carry the block on every n3 arm status row; the off record must show
settle_on 0 and no settle waits, the on record settle_on 1. Each mismatch is printed as a NOTE, never a decision.
handbacks_served (append_yield counters) moves by design when the settle is on: a handback whose first token is already
sampled at the re-entry's start is counted; it is printed beside the block, not read as a fault.

Usage:
  python3 -B tools/q2024p_j2_yrs_reader.py off=<FETCH-OFF>/q18a-parity-n3s24q2024p-j2-yrs-off \
      on=<FETCH-ON>/q18a-parity-n3s24q2024p-j2-yrs-on [--json-out <fresh.json>]
"""
import argparse
import bisect
import collections
import importlib.util
import json
from pathlib import Path
import re

_spec = importlib.util.spec_from_file_location('q2024p_j2_hgb_reader', Path(__file__).resolve().parent / 'q2024p_j2_hgb_reader.py')
hgb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hgb)

KEY = 'yield_reentry'
YRS_ENV = 'FLASH_S5_YIELD_REENTRY_SETTLE'
SCALARS = ('exceeded', 'barrier_pass', 'barrier_first_try', 'settle_waits', 'settle_ticks', 'settled', 'settle_bound',
           'errors')
MAPS = ('refused', 'refused_retry', 'retries_until_armed', 'settle_skipped')
MISS = re.compile(r'^(\S+) .*?flash-s5 yield-reentry proof-miss stage=(\S+) field=(\S+) .*?\bretry=(\d+) ')
GAP_EVENTS = {'chunk_commit', 'admission_disarmed', 'append_yield_full', 'append_yield_reenter', 'auto_reentry',
              'drain_empty', 'admission_armed', 'open_sent', 'chunk_open', 'request_admitted', 'control_submit_begin',
              'control_reply'}
TXN = re.compile(r'txn_seq=(\d+)')


def declines_block(row):
    declines = ((row or {}).get('append_yield') or {}).get('declines')
    return declines.get(KEY) if type(declines) is dict else None


def served(row):
    counters = ((row or {}).get('append_yield') or {}).get('counters')
    return counters.get('handbacks_served') if type(counters) is dict else None


def reentry_delta(before, after):
    if type(before) is not dict or type(after) is not dict:
        return None
    delta = {name: hgb._sub(after.get(name), before.get(name)) for name in SCALARS}
    for name in MAPS:
        a, b = after.get(name) or {}, before.get(name) or {}
        delta[name] = {key: hgb._sub(a.get(key, 0), b.get(key, 0)) for key in sorted(set(a) | set(b))
                       if hgb._sub(a.get(key, 0), b.get(key, 0))}
    ma, mb = after.get('armed_ms') or (), before.get('armed_ms') or ()
    if len(ma) == 3 and len(mb) == 3:
        n = ma[0] - mb[0]
        delta['armed'] = dict(n=n, mean_ms=round((ma[1] - mb[1]) / n, 3) if n else None, max_ms_lifetime=ma[2])
    delta['settle_on'] = (before.get('settle_on'), after.get('settle_on'))
    delta['retry_bound'] = after.get('retry_bound')
    return delta


def identity_settle(root):
    values = {}
    for phase in ('start', 'end'):
        doc = hgb.load(root / ('engine.identity-%s.json' % phase))
        env = ((doc or {}).get('identity') or {}).get('env') if doc else None
        values[phase] = None if type(env) is not dict else env.get(YRS_ENV, 'unset')
    return values


def scan_log(path, windows):
    """One pass over the engine log: proof-miss lines and the handoff events, both kept per arm window."""
    misses = {label: dict(n=0, by_field=collections.Counter(), by_retry=collections.Counter()) for label in windows}
    events, released, txn_key, admitted = [], collections.defaultdict(float), {}, {}
    with path.open(errors='replace') as handle:
        for line in handle:
            if 'flash-s5 ' not in line:
                continue
            try:
                at = hgb.stamp(line.split(' ', 1)[0])
            except ValueError:
                continue
            if 'flash-s5 yield-reentry proof-miss ' in line:
                m = MISS.match(line)
                if m is None:
                    continue
                for label, (start, end) in windows.items():
                    if start <= at <= end:
                        slot = misses[label]
                        slot['n'] += 1
                        slot['by_field'][m[2] + ':' + m[3]] += 1
                        slot['by_retry'][m[4]] += 1
                        break
            elif 'flash-s5 memory point=open rank=0 ' in line:
                m = TXN.search(line)
                if m:
                    events.append((at, 'open0', dict(txn=int(m[1]))))
            elif 'flash-s5 memory point=released ' in line:
                m = TXN.search(line)
                if m:
                    released[int(m[1])] = max(released[int(m[1])], at)
            elif 'flash-s5 native-gap ' in line:
                try:
                    doc = json.loads(line.split('flash-s5 native-gap ', 1)[1])
                except ValueError:
                    continue
                kind = doc.get('event') if type(doc) is dict else None
                if kind not in GAP_EVENTS:
                    continue
                if kind == 'chunk_open':
                    txn_key[doc.get('txn')] = doc.get('request_key')
                if kind == 'request_admitted':
                    admitted.setdefault(doc.get('request_key'), at)
                events.append((at, kind, doc))
    events.sort(key=lambda e: e[0])
    last = collections.defaultdict(float)
    for txn, at in released.items():
        key = txn_key.get(txn)
        if key is not None:
            last[key] = max(last[key], at)
    queued = sorted((admitted[k], last[k]) for k in last if k in admitted and admitted[k] < last[k])
    return ({label: dict(n=s['n'], by_field=dict(sorted(s['by_field'].items())), by_retry=dict(sorted(s['by_retry'].items())))
             for label, s in misses.items()}, events, released, queued)


def queued_s(queued, a, b):
    return sum(max(0.0, min(b, y) - max(a, x)) for x, y in queued if x < b and y > a)


def _first(marks, kind, after=None):
    for at, k, _doc in marks:
        if k == kind and (after is None or at >= after):
            return at
    return None


def handoffs(events, released, queued, start, end):
    """FULL returns in [start, end]: per path, the release->rank0-open interval, its queued agent-s and the barrier loops."""
    times = [e[0] for e in events]
    out = collections.defaultdict(lambda: dict(ms=[], queued=0.0, loops_ms=[], drains=collections.Counter()))
    for i in range(bisect.bisect_left(times, start), bisect.bisect_right(times, end)):
        at, kind, doc = events[i]
        if kind != 'admission_disarmed' or doc.get('reason') != 'route-returned-full':
            continue
        j = i - 1
        while j >= 0 and events[j][1] != 'chunk_commit':
            j -= 1
        if j < 0 or events[j][2].get('txn') not in released:
            continue
        last_txn, t_rel = events[j][2].get('txn'), released[events[j][2].get('txn')]
        marks, k = [], i + 1
        while k < len(events) and not (events[k][1] == 'open0' and events[k][2]['txn'] > last_txn):
            marks.append(events[k])
            k += 1
            if events[k - 1][0] - at > 30:
                break
        if k >= len(events) or events[k][1] != 'open0':
            continue
        t_open = events[k][0]
        kinds = [m[1] for m in marks]
        path = 'yield' if 'append_yield_full' in kinds or 'append_yield_reenter' in kinds else 'run'
        slot = out[path]
        slot['ms'].append(1000 * (t_open - t_rel))
        slot['queued'] += queued_s(queued, t_rel, t_open)
        slot['drains'][kinds.count('drain_empty')] += 1
        if path == 'yield':
            reentry = _first(marks, 'auto_reentry')
            first = _first(marks, 'drain_empty', after=reentry)
            armed = _first(marks, 'admission_armed', after=first)
            if first is not None and armed is not None:
                lastdrain = max(t for t, k2, _d in marks if k2 == 'drain_empty' and first <= t <= armed)
                slot['loops_ms'].append(1000 * (lastdrain - first))
    result = {}
    for path, slot in sorted(out.items()):
        ms = slot['ms']
        result[path] = dict(n=len(ms), mean_ms=round(sum(ms) / len(ms), 1) if ms else None,
                            p50_ms=round(hgb.quantile(ms, .5), 1) if ms else None,
                            queued_agent_s=round(slot['queued'], 2),
                            loops=dict(n=len(slot['loops_ms']),
                                       mean_ms=round(sum(slot['loops_ms']) / len(slot['loops_ms']), 1) if slot['loops_ms'] else None,
                                       sum_s=round(sum(slot['loops_ms']) / 1000, 2)),
                            drains_per_episode={str(c): m for c, m in sorted(slot['drains'].items())})
    return result


def collect(root, variant):
    root = Path(root)
    result = hgb.collect(root, variant)
    settle = identity_settle(root)
    result['identity_settle'] = settle
    want = {'off': '0', 'on': '1'}[variant]
    if any(value != want for value in settle.values()):
        result['notes'].append('identity-settle-%s-start-%s-end-%s' % (variant, settle['start'], settle['end']))
    windows = {}
    for label in hgb.ARMS:
        arm = hgb.load(root / ('arm-' + label + '.json'))
        if arm is None or label not in result['arms']:
            continue
        rows = hgb.status_rows(arm)
        blocks = {tag: declines_block(row) for tag, row in rows.items()}
        if label.startswith('n3.') and not all(type(v) is dict for v in blocks.values()):
            result['notes'].append('n3-record-lacks-yield_reentry:' + label)
        delta = reentry_delta(blocks.get('before'), blocks.get('after'))
        if delta is not None:
            on = delta['settle_on']
            if on != ((0, 0) if variant == 'off' else (1, 1)):
                result['notes'].append('settle-on-%s-%s:%s' % (variant, on, label))
            if variant == 'off' and (delta['settle_waits'] or delta['settled']):
                result['notes'].append('off-record-settled:' + label)
        result['arms'][label]['yield_reentry'] = delta
        result['arms'][label]['handbacks_served'] = hgb._sub(served(rows.get('after')), served(rows.get('before')))
        start, end = arm.get('replay_start_s'), arm.get('replay_end_s')
        if type(start) in (int, float) and type(end) in (int, float):
            windows[label] = (start, end)
    log = root / 'engine.log'
    if not log.is_file():
        log = root / 'engine-stream.log'
    if log.is_file():
        misses, events, released, queued = scan_log(log, windows)
        for label, (start, end) in windows.items():
            result['arms'][label]['proof_miss_lines'] = misses[label]
            result['arms'][label]['full_handoff'] = handoffs(events, released, queued, start, end)
    return result


def render(results):
    lines = [hgb.render(results), '-- yield re-entry']
    for result in results:
        lines.append('== %s  settle identity=%s/%s' % (result['variant'], result['identity_settle']['start'],
                                                      result['identity_settle']['end']))
        for label, arm in result['arms'].items():
            d = arm.get('yield_reentry')
            if d is not None:
                lines.append('  %-13s settle_on=%s bound=%s pass=%s first_try=%s exceeded=%s armed=%s served=%s errors=%s' % (
                    label, d['settle_on'], d['retry_bound'], d['barrier_pass'], d['barrier_first_try'], d['exceeded'],
                    json.dumps(d.get('armed'), sort_keys=True), arm.get('handbacks_served'), d['errors']))
                lines.append('      refused=%s by_retry=%s until_armed=%s' % (
                    json.dumps(d['refused'], sort_keys=True), json.dumps(d['refused_retry'], sort_keys=True),
                    json.dumps(d['retries_until_armed'], sort_keys=True)))
                lines.append('      settle waits=%s ticks=%s settled=%s bound=%s skipped=%s' % (
                    d['settle_waits'], d['settle_ticks'], d['settled'], d['settle_bound'],
                    json.dumps(d['settle_skipped'], sort_keys=True)))
            miss = arm.get('proof_miss_lines')
            if miss is not None:
                lines.append('      proof-miss lines n=%s by_field=%s by_retry=%s' % (
                    miss['n'], json.dumps(miss['by_field'], sort_keys=True), json.dumps(miss['by_retry'], sort_keys=True)))
            for path, h in (arm.get('full_handoff') or {}).items():
                lines.append('      FULL %-5s n=%s mean=%sms p50=%sms queued_agent_s=%s loops=%s drains=%s' % (
                    path, h['n'], h['mean_ms'], h['p50_ms'], h['queued_agent_s'], json.dumps(h['loops'], sort_keys=True),
                    json.dumps(h['drains_per_episode'], sort_keys=True)))
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('records', nargs='+', help='off=<record dir> and/or on=<record dir>')
    parser.add_argument('--json-out', type=Path)
    args = parser.parse_args(argv)
    results = []
    for item in args.records:
        variant, sep, path = item.partition('=')
        if not sep or variant not in ('off', 'on'):
            parser.error('records are off=<dir> or on=<dir>')
        results.append(collect(Path(path), variant))
    print(render(results))
    print('EXPLORATORY no rule, no thresholds, no verdict')
    if args.json_out:
        with args.json_out.open('x') as handle:
            json.dump(dict(schema='q2024p-j2-yrs-read/1', exploratory=True, results=results), handle, indent=2, sort_keys=True)
            handle.write('\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
