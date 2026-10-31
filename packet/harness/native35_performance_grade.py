#!/usr/bin/env python3
"""Native35 P aggregate: both retained W2 rounds beside the W1 baseline, raw rates exposed.

Harness-only reading. Every round is regraded from its raw record with the
frozen ``e3_native_performance.grade`` (which rechecks each native source
packet) and must equal the grade the round wrote. The 80/80 marker stays a
marker: ``target_is_gate`` is False and a SHORTFALL never fails. No round is
dropped, retried, averaged or picked; a round whose coverage is unproved keeps
its raw numbers under ``raw`` but has no headline. Memory, lifetime and
correctness verdicts keep their own grades and are only cited here.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import e3_native_performance as perf
import e3_overlap_grade as legacy

SCHEMA = 'native35-p-performance-grade/1'
ROUNDS = 2
FAMILIES = ('fresh', 'append')


def number(value):
    return type(value) in (int, float) and not isinstance(value, bool) and math.isfinite(value)


def ratio(a, b):
    return a / b if number(a) and number(b) and b > 0 else None


def load(path):
    if path is None:
        return None, {'path': None, 'present': False}
    p = Path(path)
    if not p.is_file():
        return None, {'path': str(p), 'present': False}
    data = p.read_bytes()
    try:
        value = json.loads(data)
    except ValueError:
        return None, {'path': str(p), 'present': True, 'sha256': hashlib.sha256(data).hexdigest(),
                      'readable': False}
    return value, {'path': str(p), 'present': True, 'sha256': hashlib.sha256(data).hexdigest(),
                   'readable': True}


def w1_baseline(w1):
    """FULL repeats are the serving denominator; split solo repeats are reported beside it."""
    repeats = []
    for row in (w1 or {}).get('repeats') or []:
        if type(row) is not dict:
            continue
        metrics = row.get('metrics') or {}
        repeats.append({'repeat': row.get('repeat'), 'route': row.get('route'),
                        'valid': row.get('valid') is True, 'not_run': row.get('not_run') is True,
                        'reasons': row.get('reasons') or [],
                        'aggregate_tokens_per_s': metrics.get('aggregate_tokens_per_s'),
                        'total_completion_delta': metrics.get('total_completion_delta'),
                        'window_s': metrics.get('wall_s'),
                        'stream_rates_tokens_per_s': metrics.get('stream_rates_tokens_per_s')})
    routes = {}
    for route in ('full', 'split'):
        rows = [r for r in repeats if r['route'] == route]
        rates = [r['aggregate_tokens_per_s'] for r in rows]
        complete = bool(rows) and all(r['valid'] and number(r['aggregate_tokens_per_s']) for r in rows)
        routes[route] = {
            'declared_repeats': [r['repeat'] for r in rows],
            'valid_repeats': [r['repeat'] for r in rows if r['valid']],
            'aggregate_tokens_per_s_by_repeat': rates,
            'min': min(rates) if complete else None,
            'median': statistics.median(rates) if complete else None,
            'max': max(rates) if complete else None,
            'grade_denominator': legacy._w1_route(w1, route) if w1 else None,
        }
    summary = (w1 or {}).get('summary') or {}
    return {'verdict': summary.get('verdict') or (w1 or {}).get('verdict') or 'MISSING',
            'repeats': repeats, 'routes': routes,
            'full_is_serving_denominator': True, 'split_solo_never_substituted': True,
            'error': (w1 or {}).get('error')}


def lane_row(record, lane, family):
    return (((record or {}).get('lanes') or {}).get(lane) or {}).get(family) or {}


def decode_view(row):
    window = row.get('decode_window') or {}
    streams = window.get('streams') or []
    deltas = [s.get('completion_delta') for s in streams]
    eight = (len(streams) == 8 and all(type(d) is int and d >= 1 for d in deltas)
             and not any(s.get('error_before_window_end') for s in streams))
    return {'present': bool(window),
            'start_ns': window.get('start_ns'), 'end_ns': window.get('end_ns'),
            'window_s': window.get('wall_s'),
            'aggregate_tokens_per_s': window.get('aggregate_tokens_per_s'),
            'stream_rates_tokens_per_s': window.get('stream_rates_tokens_per_s'),
            'completion_delta_by_stream': deltas,
            'completion_tokens_in_window': sum(deltas) if deltas and all(type(d) is int for d in deltas) else None,
            'start_completion_by_stream': [s.get('start_completion') for s in streams],
            'end_completion_by_stream': [s.get('end_completion') for s in streams],
            'in_window_arrivals_by_stream': [s.get('in_window_arrivals') for s in streams],
            'gap_ms_p99': window.get('gap_ms_p99'), 'gap_ms_max': window.get('gap_ms_max'),
            'eight_live_streams_advanced': eight,
            'row_reasons': row.get('reasons') or [],
            'basis': 'client-visible SSE counts inside the source-joined native prefill window'}


def native_view(row, family, epoch):
    proof = perf.native_window(row, family, epoch)
    group = proof.get('group') or {}
    request = row.get('request') or {}
    clock = ((row.get('end') or {}).get('clock')) or {}
    return {'complete': row.get('complete') is True and proof['complete'],
            'source_reasons': proof['reasons'],
            'useful_prefill_tokens': group.get('useful_tokens'),
            'source_joined_wall_s': group.get('wall_s'),
            'prefill_tokens_per_s': proof.get('prefill_tokens_per_s'),
            'group_prompt_tokens': group.get('prompt_tokens'),
            'chunks': group.get('chunks'), 'initial_frontier': group.get('initial_frontier'),
            'start_ns': group.get('start_ns'), 'end_ns': group.get('end_ns'),
            'prompt_tokens_observed': request.get('prompt_tokens_observed'),
            'prompt_tokens_declared': request.get('prompt_tokens_declared'),
            'prompt_sha256': request.get('prompt_sha256'),
            'ttft_s': request.get('ttft_s'),
            'cache_prefix_evidence': {
                'observed_prompt_tokens': request.get('prompt_tokens_observed'),
                'native_useful_tokens': group.get('useful_tokens'),
                'reused_prefix_tokens_upper_bound': (request['prompt_tokens_observed'] - group['useful_tokens']
                    if type(request.get('prompt_tokens_observed')) is int and type(group.get('useful_tokens')) is int else None),
                'basis': 'descriptive: observed prompt tokens minus native computed tokens; the final token returns to stock'},
            'clock': {name: {k: (clock.get(name) or {}).get(k) for k in ('proved', 'boot_id', 'time_namespace')}
                      for name in ('begin', 'end')},
            'headline_basis': proof.get('headline_basis'),
            'error': row.get('error')}


def serving_view(row):
    request = row.get('request') or {}
    return {'complete': row.get('complete') is True,
            'prompt_tokens_observed': request.get('prompt_tokens_observed'),
            'prompt_tokens_declared': request.get('prompt_tokens_declared'),
            'prompt_sha256': request.get('prompt_sha256'),
            'ttft_s': request.get('ttft_s'),
            'serving_prefill_tokens_per_s': row.get('serving_prefill_tokens_per_s'),
            'serving_rate_basis': row.get('serving_rate_basis'),
            'error': row.get('error')}


def round_view(index, record, meta, memory, memory_meta, w1):
    missing = []
    view = {'round': index, 'artifact': meta, 'memory_grade': memory_meta}
    if record is None:
        missing.append('round-%d:artifact-%s' % (index, 'absent-not-run' if not meta['present'] else 'unreadable'))
        view.update(verdict='NOT_RUN' if not meta['present'] else 'INCONCLUSIVE', missing=missing,
                    failures=[], headline=None, raw=None, target=None, target_is_gate=False,
                    coverage_proved=False, memory=None)
        return view
    round_meta = record.get('native35_round') or {}
    if round_meta.get('round') != index:
        missing.append('round-%d:round-identity-differs' % index)
    written = record.get('grade') or {}
    lanes_present = type(record.get('lanes')) is dict
    # Always regrade from the raw record; a written verdict is never trusted.
    regraded = perf.grade(record)
    if lanes_present and regraded != written:
        missing.append('round-%d:recheck-differs-from-written-grade' % index)
    if not lanes_present:
        missing.append('round-%d:lanes-absent' % index)
        missing.extend('round-%d:written:%s' % (index, m) for m in written.get('missing') or [])
    if record.get('error'):
        missing.append('round-%d:run-refused' % index)
    if lanes_present and record.get('w1') != w1:
        missing.append('round-%d:embedded-w1-differs-from-queue-w1' % index)
    missing.extend('round-%d:%s' % (index, m) for m in regraded.get('missing') or [])
    epoch = record.get('engine_epoch')
    w1_full = legacy._w1_route(w1, 'full') if w1 else {}
    overlap_fresh = lane_row(record, 'overlap', 'fresh')
    decode = {family: decode_view(lane_row(record, 'overlap', family)) for family in FAMILIES}
    native = {lane: {family: native_view(lane_row(record, lane, family), family, epoch)
                     for family in FAMILIES} for lane in ('prefill_split', 'overlap')}
    serving = {family: serving_view(lane_row(record, 'prefill_full', family)) for family in FAMILIES}
    baseline = w1_full.get('decode_tokens_per_s')
    raw = {
        'decode_during_prefill': decode,
        'decode_share_of_w1_full_median': ratio(decode['fresh']['aggregate_tokens_per_s'], baseline),
        'decode_share_by_stream': regraded.get('decode_shares_by_stream'),
        'w1_full_median_tokens_per_s': baseline,
        'w1_full_stream_median_rates': w1_full.get('stream_rates_tokens_per_s'),
        'native_prefill': native,
        'prefill_share_overlap_over_split_solo': {
            family: ratio(native['overlap'][family]['prefill_tokens_per_s'],
                          native['prefill_split'][family]['prefill_tokens_per_s']) for family in FAMILIES},
        'full_serving_solo': serving,
        'overlap_native_rate_over_full_serving_rate_fresh': ratio(
            native['overlap']['fresh']['prefill_tokens_per_s'], serving['fresh']['serving_prefill_tokens_per_s']),
        'timing_bases': {
            'native': 'useful native prefill tokens / source-joined elapsed (first accepted work .. final queried STATE_WRITE)',
            'full_serving': 'observed fresh prompt tokens / client TTFT (request send .. first generated token)',
            'comparison': 'different timing bases; the cross ratio is descriptive, never the 80/80 prefill share'},
        'ttft_s': regraded.get('ttft_s'),
        'first_error': record.get('first_error'),
        'error': record.get('error'),
        'salts': round_meta.get('salts'), 'prompts': round_meta.get('prompts'),
        'session_run_id': round_meta.get('session_run_id'),
        'salt_namespace': round_meta.get('salt_namespace'),
    }
    if not decode['fresh']['eight_live_streams_advanced'] and overlap_fresh:
        missing.append('round-%d:eight-live-streams-unproved' % index)
    verdict = regraded.get('verdict', 'INCONCLUSIVE')
    if verdict == 'PASS' and missing:
        verdict = 'INCONCLUSIVE'
    headline = None
    if verdict == 'PASS':
        target = regraded.get('target') or {}
        headline = {'decode_share': target.get('decode_share'), 'prefill_share': target.get('prefill_share'),
                    'decode_tokens_per_s_during_prefill': decode['fresh']['aggregate_tokens_per_s'],
                    'w1_full_median_tokens_per_s': baseline,
                    'overlap_prefill_tokens_per_s': native['overlap']['fresh']['prefill_tokens_per_s'],
                    'split_solo_prefill_tokens_per_s': native['prefill_split']['fresh']['prefill_tokens_per_s'],
                    'target_status': target.get('status')}
    memory_view = None
    if memory is not None:
        memory_view = {k: memory.get(k) for k in ('verdict', 'failures', 'missing', 'reserve_mib',
                                                     'min_free_mib_under_pressure')}
    view.update(verdict=verdict, missing=sorted(set(missing)), failures=list(regraded.get('failures') or []),
                headline=headline,
                target=regraded.get('target'), target_met=regraded.get('target_met'),
                target_is_gate=False, coverage_proved=verdict == 'PASS', raw=raw, memory=memory_view)
    return view


def grade(w1, w1_meta, rounds, memories, gates=None):
    if len(rounds) != ROUNDS or len(memories) != ROUNDS:
        raise ValueError('native35-p-two-rounds-required')
    baseline = w1_baseline(w1)
    views = [round_view(i, rounds[i][0], rounds[i][1], memories[i][0], memories[i][1], w1)
             for i in range(ROUNDS)]
    missing = []
    if baseline['verdict'] != 'COMPLETE':
        missing.append('w1-fixed-repeats-incomplete')
    for view in views:
        missing.extend(view['missing'])
    failures = sorted({'round-%d:%s' % (v['round'], f) for v in views for f in v.get('failures') or []})
    if any(v['verdict'] == 'FAIL' for v in views):
        failures.append('round-grade-fail')
    verdict = 'FAIL' if failures else 'PASS' if not missing and all(v['verdict'] == 'PASS' for v in views) else 'INCONCLUSIVE'
    return {'schema': SCHEMA, 'verdict': verdict, 'failures': failures, 'missing': sorted(set(missing)),
            'target_is_gate': False, 'target_label': 's5-overlap-80-80',
            'target_status_by_round': [(v.get('target') or {}).get('status') if v['verdict'] == 'PASS' else None for v in views],
            'headline_by_round': [v['headline'] for v in views],
            'rounds': views, 'w1': dict(baseline, artifact=w1_meta),
            'gates': (gates or {}).get('outcomes'),
            'all_declared_rounds_retained': True, 'filtered_invalid': False, 'retry_selected': False,
            'picked': False, 'averaged': False,
            'sample_scope': 'two fixed matched W2 rounds and six W1 repeats; limited samples, not a confidence interval or drift-isolated experiment',
            'scope': 'native decode-during-prefill and prefill speed reading; memory, lifetime and correctness keep their own grades'}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--w1', required=True); p.add_argument('--gates')
    p.add_argument('--round', action='append', required=True)
    p.add_argument('--memory', action='append', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args(argv)
    if Path(a.out).exists():
        return 2
    try:
        w1, w1_meta = load(a.w1)
        gates, _ = load(a.gates)
        record = grade(w1, w1_meta, [load(path) for path in a.round], [load(path) for path in a.memory], gates)
    except Exception as exc:
        record = {'schema': SCHEMA, 'verdict': 'INCONCLUSIVE', 'failures': [], 'target_is_gate': False,
                  'missing': ['aggregate-grade-refused:' + type(exc).__name__]}
    with open(a.out, 'x') as handle:
        json.dump(record, handle, sort_keys=True)
    print('NATIVE35_PERFORMANCE ' + record['verdict'])
    return {'PASS': 0, 'FAIL': 1}.get(record['verdict'], 2)


if __name__ == '__main__':
    raise SystemExit(main())
