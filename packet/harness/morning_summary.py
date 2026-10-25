#!/usr/bin/env python3
"""Offline q10/q11/q12b/q13/q14b evidence table; no serving-box access.

This is a report, not a release authorizer. PASS retains each test's scope.
Only fixed labels, finite numbers and allowlisted enums reach the output. Logs,
payloads, identity env/args/defaults and exception text are never reproduced.
"""
import argparse
import json
import math
from pathlib import Path
import re

import agent_replay
import s5_followup
import s6_grade
from vision_tower_probe import MODEL_COMMIT, TOWER_SHA, embedding_budget

QUEUES = ('q10', 'q11', 'q12b', 'q13', 'q14b')
VERDICTS = ('PASS', 'INCONCLUSIVE', 'FAIL')
ERRORS = (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError, AssertionError, ZeroDivisionError)


def load(p):
    # Reject NaN/Infinity, including nested fields, rather than emit invalid JSON.
    return s5_followup.load(p)


def num(v, minimum=0):
    if type(v) not in (int, float) or not math.isfinite(v) or v < minimum:
        raise ValueError('invalid numeric evidence')
    return v


def numbers(d, keys):
    return {k: num(d[k]) if d[k] is not None else None for k in keys}


def verdict(v):
    return v if v in VERDICTS else 'INCONCLUSIVE'


def worst(*vs):
    return 'FAIL' if 'FAIL' in vs else 'INCONCLUSIVE' if any(v != 'PASS' for v in vs) else 'PASS'


class Ledger:
    """Parse only structured queue markers. Free-form tails remain private."""
    def __init__(self, path):
        try:
            self.lines = path.read_text().splitlines()
        except (OSError, ValueError):
            self.lines = []
        self.steps = []
        for i, line in enumerate(self.lines):
            m = re.match(r'^STEP ([A-Za-z0-9_.-]+) (PASS|FAIL|INCONCLUSIVE|SKIP|SKIPPED)(?:\s|$)', line)
            if m:
                self.steps.append((i, m[1], verdict(m[2])))

    def step(self, name):
        rows = [v for _, n, v in self.steps if n == name]
        return worst(*rows) if rows else 'INCONCLUSIVE'

    def finished(self, name):
        matches = [re.fullmatch(re.escape(name) + r' DONE \d{2}:\d{2}:\d{2} fails=(\d+) inconclusive=(\d+)', l)
                   for l in self.lines]
        matches = [m for m in matches if m]
        if len(matches) != 1:
            return 'INCONCLUSIVE'
        f, i = map(int, matches[0].groups())
        return 'FAIL' if f else 'INCONCLUSIVE' if i else 'PASS'

    def grade(self, name, required):
        # A late cleanup failure overrides an earlier PASS and the final count.
        return worst(self.finished(name), *(v for _, _, v in self.steps),
                     *(self.step(s) for s in required))

    def engine(self, queue):
        if queue in ('q10', 'q11'):
            return 'TEST_ENGINE_RESTORED' if self.step('restart-final') == 'PASS' else 'UNVERIFIED'
        if queue in ('q12b', 'q13'):
            stop = 'engine-stop' if queue == 'q12b' else 'stop'
            ends = [(i,v) for i, n, v in self.steps if n == stop]
            idle = [(i,v) for i, n, v in self.steps if n == 'gpu-handoff']
            if ends and idle and ends[-1][1] == idle[-1][1] == 'PASS' and idle[-1][0] > ends[-1][0]:
                # A later engine action invalidates this historical proof.
                if not any(i > idle[-1][0] and ('restart' in n or n == stop) for i, n, _ in self.steps):
                    return 'STOPPED_GPU_IDLE_RECORDED'
            return 'UNVERIFIED'
        rows = [re.fullmatch(r'STEP vision-probe (PASS|FAIL|INCONCLUSIVE)(?: cleanup: GPU not idle after probe)? rc=(\d+) \d+ s gpu-idle-after=(\d+)', l)
                for l in self.lines]
        rows = [m for m in rows if m]
        if len(rows) == 1:
            return 'STOPPED_GPU_IDLE_RECORDED' if rows[0][3] == '0' else 'GPU_HANDOFF_FAILED'
        return 'UNVERIFIED'


def row(q, case, v='INCONCLUSIVE', aggregate=None, streams=None, metrics=None, engine='UNVERIFIED'):
    return dict(queue=q, case=case, verdict=verdict(v), aggregate=aggregate or {},
                streams=streams or [], metrics=metrics or {}, engine_end=engine)


def replay(root, ledger):
    rows = []
    for n in (4, 6):
        case = f'on-{n}'
        r = row('q10', case, engine=ledger.engine('q10'))
        try:
            d = load(root / (case + '.json')); c, turns = d['config'], d['turns']
            expected = dict(AGENTS=n, DURATION=1500, WARMUP=300, SEED=2, START_MIN=60000,
                            START_MAX=150000, APPEND_MIN=500, APPEND_MAX=5000,
                            REPLY_MIN=200, REPLY_MAX=1000, TOOL_MIN=1, TOOL_MAX=5, ROLL_AT=200000)
            assert all(c[k] == v for k, v in expected.items())
            assert turns and {t['agent'] for t in turns} == set(range(n))
            for t in turns:
                assert type(t['agent']) is int and t['kind'] in ('fresh', 'append')
                num(t['t'])
                if 'error' not in t and t.get('ttft') is not None:
                    for k in ('seconds', 'ttft', 'completion_tokens', 'prompt_tokens'): num(t[k])
                    if t.get('cached_tokens') is not None: num(t['cached_tokens'])
            s = agent_replay.summarize(turns, c['WARMUP'], c['DURATION'])
            r['aggregate'] = numbers(s, ('output_tokens_per_s', 'turns_per_hour'))
            r['metrics'] = numbers(s, ('append_ttft_p50', 'append_ttft_p95', 'turn_s_p95',
                                      'prefix_hit_rate_appends', 'append_full_reuse', 'errors', 'incomplete'))
            for i in range(n):
                a = agent_replay.summarize([t for t in turns if t['agent'] == i], c['WARMUP'], c['DURATION'])
                r['streams'].append(dict(stream=i, **numbers(a, ('output_tokens_per_s', 'decode_tps_p50', 'decode_tps_p5', 'warm_turns'))))
            v = 'FAIL' if s['errors'] or s['incomplete'] else 'PASS'
            if any(not a['warm_turns'] for a in r['streams']): v = worst(v, 'INCONCLUSIVE')
            r['verdict'] = worst(v, *(ledger.step(s) for s in ('overlay', 'launcher', case+'.restart', case, case+'.layout')))
        except ERRORS:
            r['verdict'] = worst(r['verdict'], ledger.step(case))
        rows.append(r)
    rows.append(row('q10', 'queue completion', worst(*(r['verdict'] for r in rows),
                    ledger.grade('Q10', ('overlay', 'launcher', 'launcher-restore', 'restart-final'))),
                    engine=ledger.engine('q10')))
    return rows


PHASES = {'initializing', 'distributed-and-helper-setup', 'clock-validation', 'source-validation',
          'reducers-and-kernel-setup', 'cleanup-and-grade', 'completed'} | {
          m+':'+p for m in ('prio', 'green24', 'serial', 'decode', 'prefill', 'green34', 'green44')
          for p in ('setup', 'stream-clock-and-graph-smoke', 'kernel-graph-capture', 'kernel-graph-equality', 'warmup', 'measurement')}


def liveness(root, ledger):
    d = s5_followup.q11(root)
    rows = []
    for mode in ('prio', 'green24'):
        s = d['stages'][mode]
        r = row('q11', mode+' liveness ONLY', worst(s['verdict'], ledger.step('s5-'+mode)),
                engine=ledger.engine('q11'))
        r['metrics']['measurement_allowed'] = (d['measurement_allowed'] is True and
            ledger.grade('Q11', ('s5-prio', 's5-green24', 'restart-final')) == 'PASS')
        for pair in ('0,2', '1,3'):
            for p in s.get('pairs', {}).get(pair, {}).get('rank_progress', []):
                r['streams'].append(dict(pair=pair, rank=p['rank'],
                    phase=p['phase'] if p['phase'] in PHASES else 'unavailable', watchdog=p.get('watchdog') is True))
        rows.append(r)
    rows.append(row('q11', 'queue completion', worst(d['verdict'], ledger.grade('Q11', ('s5-prio', 's5-green24', 'restart-final'))),
                    engine=ledger.engine('q11')))
    return rows


RELEASE_MEASURES = ('verify-7392', 'verify-7568', 'verify-9504', 'verify-8096',
                    'verify-lifetime', 'shipping-leak', 'shipping-lifetime', 'canary')


def io_metrics(root):
    """Read ONLY the numerical io section, never the enclosing engine identity."""
    paths = set(root.glob('*.before.json')) | set(root.glob('*.after.json')) | set(root.glob('*.identity.json'))
    count, seconds, size = 0, 0, 0
    for p in paths:
        try:
            d = load(p)['io']; wall, read = num(d['wall_seconds']), num(d['bytes_read'])
            seconds += wall; size += read; count += 1
        except ERRORS:
            continue
    return dict(captures_with_io=count, capture_wall_s=seconds, capture_bytes_read=size,
                capture_io_unavailable=len(paths)-count)


def release_steps():
    return ['pins', 'launcher-install', 'verify.restart', 'verify-layout', 'shipping.restart',
            'shipping-layout', 'engine-errors', 'engine-stop', 'gpu-handoff', 'launcher-restore', 'defaults-restore'] + [
            n+s for n in RELEASE_MEASURES for s in ('', '.identity-before', '.identity-after', '.identity-stable')]


def release(root, ledger):
    rows = []
    for name in RELEASE_MEASURES:
        r = row('q12b', name, engine=ledger.engine('q12b'))
        try:
            if name == 'canary':
                d = load(root / 'canary.result')['release_grade']
                fields = ('transcript_requests', 'relay_completed_messages', 'engine_request_delta')
            elif name.endswith('lifetime'):
                d = load(root / (name+'.coverage.json')); fields = ('state_evictions', 'evictions')
            else:
                d = load(root / (name+'.grade.json')); fields = ()
            r['metrics'] = numbers(d, fields)
            r['verdict'] = worst(verdict(d.get('verdict')), *(ledger.step(name+s) for s in ('', '.identity-before', '.identity-after', '.identity-stable')))
        except ERRORS:
            r['verdict'] = worst(r['verdict'], ledger.step(name))
        rows.append(r)
    rows.append(row('q12b', 'release gates (not production authorization)',
                    worst(*(r['verdict'] for r in rows), ledger.grade('QN-PP-RELEASE', release_steps())),
                    metrics=io_metrics(root),
                    engine=ledger.engine('q12b')))
    return rows


def s6_steps():
    return ['copy-verify', 'prepare', 'cow-invariants', 'byte-equality', 'performance', 'stop',
            'gpu-handoff', 'launcher-restore', 'defaults-restore'] + [
            n+s for n in ('baseline', 'verify', 'candidate', 'baseline-last')
            for s in ('.declare', '.restart', '.identity')] + [
            n+s for n in ('baseline-w2', 'verify-w2', 'candidate-w2', 'baseline-last-w2', 'verify-leak', 'candidate-leak')
            for s in ('', '.before', '.after', '.stable')] + [
            'verify-leak.grade', 'candidate-leak.grade', 'baseline-w2.recompute', 'candidate-w2.recompute', 'baseline-last-w2.recompute']


def sparse(root, ledger):
    rows, data = [], []
    for case in ('baseline-w2', 'candidate-w2', 'baseline-last-w2'):
        try:
            d = load(root / (case+'.json'))
            assert d['streams'] == 8 and d['stream_prompt'] == 153537 and d['long_prompt'] == 200000
            assert d['newcomer']['cached_tokens'] == 0
            data.append(d)
        except ERRORS:
            d = {}
        for window, name in (('before', 'W1'), ('during', 'W2')):
            r = row('q13', case+' '+name, engine=ledger.engine('q13'))
            try:
                w = d[window]; assert len(w['per_stream']) == 8
                r['aggregate'] = numbers(w, ('agg_tps',))
                assert w['agg_tps'] > 0
                for i, s in enumerate(w['per_stream']):
                    r['streams'].append(dict(stream=i, tps=num(s['tps']), gap_p99_ms=num(s['gap_ms']['p99'])))
                assert abs(sum(s['tps'] for s in r['streams'])-w['agg_tps']) < .2
                r['metrics'] = dict(fresh_ttft_s=num(d['newcomer']['client_ttft_s']),
                                    worst_p99_ms=num(w['worst_stream_gap_ms']['p99']))
                r['verdict'] = worst(verdict(d.get('verdict')), 'FAIL' if d['fails'] else 'PASS',
                                     *(ledger.step(case+s) for s in ('', '.before', '.after', '.stable', '.recompute')))
            except ERRORS:
                r['verdict'] = worst(r['verdict'], ledger.step(case))
            rows.append(r)
    g = row('q13', 'ABA measured gain', engine=ledger.engine('q13'))
    try:
        assert len(data) == 3
        d = s6_grade.performance(*data)
        g['verdict'] = worst(d['verdict'], ledger.step('performance'))
        for key, label in (('before', 'W1'), ('during', 'W2')):
            a = d['measurements'][key]
            for metric in ('candidate_pct_vs_mean', 'control_spread_pct'):
                g['metrics'][label+'_'+metric] = num(a['aggregate_tps'][metric], -100)
            g['streams'] += [dict(stream=i, window=label, candidate_pct_vs_mean=num(v['candidate_pct_vs_mean'], -100))
                             for i, v in enumerate(a['per_stream_tps'])]
        g['metrics']['fresh_ttft_candidate_pct_vs_mean'] = num(d['measurements']['fresh_ttft_s']['candidate_pct_vs_mean'], -100)
    except ERRORS:
        g['verdict'] = worst('INCONCLUSIVE', g['verdict'], ledger.step('performance'))
    rows.append(g)
    rows.append(row('q13', 'full S6b queue', worst(*(r['verdict'] for r in rows),
                    ledger.grade('EXPERIMENT-LEFT-STOPPED', s6_steps())), metrics=io_metrics(root), engine=ledger.engine('q13')))
    return rows


def vision(root, ledger):
    r = row('q14b', 'BF16 tower ONLY', engine=ledger.engine('q14b'))
    try:
        p = root / 'vision-probe-q14'
        d = load(p / 'probe.json')
        assert not (p / 'error.json').exists()
        assert d['phase'] == 'complete' and d['mode'] == 'tower_only' and d['lm_loaded'] is False
        assert d['source_commit'] == MODEL_COMMIT and d['tower_sha256'] == TOWER_SHA
        assert d['tp'] == d['pp'] == d['warmups'] == d['repetitions'] == 1
        assert [i['input_square'] for i in d['images']] == [512, 1024, 1536]
        r['metrics'] = {'weights_bytes': num(d['weights']['bytes']), 'weight_tensors': num(d['weights']['tensors'])}
        assert d['weights']['bytes'] > 0 and d['weights']['tensors'] == 333
        for im in d['images']:
            budget = embedding_budget(im['grid_thw'])
            assert im['bf16_bytes'] == budget['bf16_bytes'] and im['visual_tokens'] == budget['visual_tokens']
            assert im['base64_payload_bytes'] >= budget['raw_base64_bytes_lower_bound']
            for key in ('encode_cuda_event_ms', 'encode_wall_ms', 'peak_allocated_bytes'): assert num(im[key]) > 0
            r['streams'].append(numbers(im, ('input_square', 'visual_tokens', 'bf16_bytes', 'base64_payload_bytes',
                                             'encode_cuda_event_ms', 'encode_wall_ms', 'peak_allocated_bytes')))
        ends = [re.fullmatch(r'Q14B DONE \d{2}:\d{2}:\d{2} rc=(\d+) \(engine untouched\)', l) for l in ledger.lines]
        ends = [m for m in ends if m]
        end = 'PASS' if len(ends) == 1 and ends[0][1] == '0' else 'INCONCLUSIVE'
        r['verdict'] = worst(end, ledger.step('vision-probe'))
        if r['engine_end'] != 'STOPPED_GPU_IDLE_RECORDED': r['verdict'] = worst(r['verdict'], 'INCONCLUSIVE')
    except ERRORS:
        r['verdict'] = worst(r['verdict'], ledger.step('vision-probe'))
    if r['engine_end'] == 'GPU_HANDOFF_FAILED': r['verdict'] = 'FAIL'
    return [r]


def find_root(base, q, override=None):
    if override is not None:
        return override if override.is_dir() else None
    pattern = q+'-*' if q in ('q10', 'q11') else {'q12b': 'q12b-gate', 'q13': 'q13-s6b', 'q14b': 'q14b-vision'}[q]
    candidates = [p for p in base.glob(pattern) if p.is_dir() and (q not in ('q10', 'q11') or
                  re.fullmatch(q+r'-\d{8}T\d{6}Z', p.name))]
    return candidates[0] if len(candidates) == 1 else None


def build(base, overrides=None, logs=None):
    overrides, logs = overrides or {}, logs or {}
    rows = []
    for q, reader in zip(QUEUES, (replay, liveness, release, sparse, vision)):
        root = find_root(base, q, overrides.get(q))
        ledger = Ledger(logs.get(q, base / (q+'.log')))
        missing = worst('INCONCLUSIVE', *(v for _, _, v in ledger.steps))
        if root is None:
            rows.append(row(q, 'missing or ambiguous artifact directory', missing, engine=ledger.engine(q)))
            continue
        try:
            rows += reader(root, ledger)
        except ERRORS:
            rows.append(row(q, 'malformed evidence', missing, engine=ledger.engine(q)))
    return dict(schema=1, scope='Offline evidence summary; PASS is test-specific, never production authorization',
                live_engine_state='UNVERIFIED_OFFLINE', rows=rows)


def compact(d):
    def val(v):
        return 'n/a' if v is None else f'{v:.2f}' if type(v) is float else str(v)
    return ', '.join(k+'='+val(v) for k, v in d.items()) or '—'


def markdown(d):
    lines = ['| Queue / case | Verdict | Aggregate | Per stream / rank / image | Other measurements | Recorded engine end |',
             '|---|---|---|---|---|---|']
    for r in d['rows']:
        lines.append('| '+' | '.join((r['queue']+' / '+r['case'], r['verdict'], compact(r['aggregate']),
            '<br>'.join(compact(s) for s in r['streams']) or '—', compact(r['metrics']), r['engine_end']))+' |')
    lines += ['', 'Rates are tok/s unless labelled turns/hour. Replay output is completed-turn tokens per wall second; '
              'decode p50/p5 is per-turn active generation, separately for each agent. W1/W2 use each stream’s measured window.',
              'q11 PASS means liveness only; q14b is a tower probe, with encode times in ms and memory/embedding sizes in bytes. '
              'ABA deltas compare the candidate with the mean of both controls; the gain gate must beat both controls.',
              'Engine states describe recorded cleanup only. Current live engine state is UNVERIFIED_OFFLINE. '
              'Missing logs/artifacts, skipped runs and ambiguous directories cannot pass. This report does not authorize production.']
    return '\n'.join(lines)+'\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True, help='local artifact mirror containing q10.log … q14b.log')
    for q in QUEUES:
        p.add_argument('--'+q, type=Path, help='explicit run directory, required when discovery is ambiguous')
        p.add_argument('--'+q+'-log', type=Path, help='explicit queue log')
    p.add_argument('--json', action='store_true', help='JSON on stdout instead of Markdown')
    p.add_argument('--out', type=Path, help='also write JSON to a NEW file; never overwrite evidence')
    a = p.parse_args()
    d = build(a.root, {q: getattr(a,q) for q in QUEUES if getattr(a,q)},
              {q: getattr(a,q+'_log') for q in QUEUES if getattr(a,q+'_log')})
    if a.out:
        try:
            with a.out.open('x') as f: json.dump(d, f, indent=2, allow_nan=False); f.write('\n')
        except OSError:
            print('INCONCLUSIVE: output could not be created; existing files are never overwritten')
            return 2
    print(json.dumps(d, indent=2, allow_nan=False) if a.json else markdown(d), end='\n')
    return {'PASS':0, 'FAIL':1, 'INCONCLUSIVE':2}[worst(*(r['verdict'] for r in d['rows']))]


if __name__ == '__main__':
    raise SystemExit(main())
