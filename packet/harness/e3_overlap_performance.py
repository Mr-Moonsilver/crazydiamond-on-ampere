#!/usr/bin/env python3
"""q18a real E3 overlap performance driver (offline build; no engine control).

This is the W2-shaped client for *actual* decode-during-prefill, not a renamed
decode-only diagnostic.  It runs three lanes against one owned engine and one
controller session:

  prefill_full   fresh-200k + append prefill solo on the full route
                 (the actual serving prefill baseline)
  prefill_split  the same prefill work on the split route
                 (the matched split solo denominator)
  overlap        eight live 153,537-token decode streams on split plus one
                 fresh 200k prompt and one append while the prefill runs

Order is fixed and calibration is frozen: every declared repeat is retained,
there is no retry, no picking and no survivor averaging.

The overlap lane only runs when the owner queue's gates are PASS and the
package passes ``e3_overlap_grade.candidate_readiness``.  The driver never
bypasses ``require_native_ready`` and never invents a ready receipt.

Required prefill evidence (validated again by e3_overlap_grade)
  * accepted prefill segments on route split
  * per rank copy proof with queried completion (an enqueued wait is refused)
  * attention ops and per-rank STATE_WRITE queried-event proof bound to the
    known final writers
  * PP receive/consume proof on every rank
  * decode replay submitted/completed counters increasing *during* each
    accepted segment, plus decode arrivals in that segment
  * useful completed prefill tokens (offered-only work is never a speed claim)
  * naturally occurring sparse/dense/serialized copy routes reported by count
  * explicit stock-forward fallback segments retained and measured, never
    accepted as overlap

Client measurement is client-visible SSE arrival timing only
(``time.monotonic_ns``), anchored once to wall time.  It is a different clock
from engine events; the boundary is stated and never silently combined.

usage: e3_overlap_performance.py URL MODEL OUT.json
env: WORKLOAD=<q18a-w2-overlap.json> CONTROLLER_CMD=<executable+args>
     RUN_ID=... [GATES=<json>] [READINESS=<json>] [GENERATION=<json>]
     [PACKAGE_MANIFEST=<SHA256SUMS>] [S5_OVERLAY=<package dir>]
     [S5_LAUNCHER_NAME=run-flash-s5-test.sh] [REQUIRE_GATES=1]
"""
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import threading
import time
from types import SimpleNamespace

import e3_decode_correctness as correctness
import e3_overlap_grade as grading
import e3_w1_decode_driver as protocol

SCHEMA = 1
ENDPOINT = '/v1/chat/completions'
MAX_STREAMS = 32
MAX_PROMPT_TOKENS = 200000
MAX_GENERATION_BUDGET = 8192
MAX_REPEATS = 4
LANES = ('prefill_full', 'prefill_split', 'overlap')
CLOCK_NOTE = (
    'client events use time.monotonic_ns with one wall anchor; engine segment '
    'timestamps and event queries are a separate clock. This driver never '
    'treats a client gap as a kernel completion time.')


class ConfigError(Exception):
    def __init__(self, reasons):
        super().__init__(','.join(reasons))
        self.reasons = list(reasons)


def _is_int(value):
    return type(value) is int


def _is_num(value):
    return type(value) in (int, float)


def _sha256_json(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _ms(ns):
    return None if ns is None else round(ns / 1e6, 3)


def _pct(values, q):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def _field_int(data, key, minimum, maximum, problems, label=None):
    label = label or key
    if key not in data or not _is_int(data.get(key)) \
            or not minimum <= data[key] <= maximum:
        problems.append('manifest_field_invalid:' + label)
        return None
    return data[key]


def _field_float(data, key, problems, label=None, maximum=3600.0):
    label = label or key
    if key not in data or not _is_num(data.get(key)) \
            or not 0 < float(data[key]) <= maximum:
        problems.append('manifest_field_invalid:' + label)
        return None
    return float(data[key])


def validate_manifest(data):
    """Pure manifest validation -> (problems, config|None)."""
    if type(data) is not dict:
        return ['manifest_not_object'], None
    problems = []
    if data.get('schema') != SCHEMA:
        problems.append('manifest_schema_must_be_1')
    if not isinstance(data.get('workload'), str) or not data['workload'].strip():
        problems.append('manifest_field_missing:workload')
    if not isinstance(data.get('target_label'), str) or not data['target_label']:
        problems.append('manifest_field_missing:target_label')
    decode = data.get('decode')
    decode_values = {}
    if type(decode) is not dict:
        problems.append('manifest_field_missing:decode')
    else:
        decode_values['streams'] = _field_int(decode, 'streams', 1, MAX_STREAMS,
                                              problems, 'decode.streams')
        decode_values['prompt_tokens'] = _field_int(
            decode, 'prompt_tokens', 1, MAX_PROMPT_TOKENS, problems,
            'decode.prompt_tokens')
        decode_values['generation_budget'] = _field_int(
            decode, 'generation_budget', 1, MAX_GENERATION_BUDGET, problems,
            'decode.generation_budget')
        decode_values['prompt_seed'] = _field_int(decode, 'seed', 0,
                                                  2 ** 63 - 1, problems,
                                                  'decode.seed')
        decode_values['open_timeout_s'] = _field_float(
            decode, 'open_timeout_s', problems, 'decode.open_timeout_s')
        sampling = decode.get('sampling')
        if type(sampling) is not dict:
            problems.append('manifest_field_missing:decode.sampling')
        else:
            decode_values['sampling_seed'] = _field_int(
                sampling, 'seed', 0, 2 ** 63 - 1, problems,
                'decode.sampling.seed')
            decode_values['temperature'] = _field_float(
                sampling, 'temperature', problems,
                'decode.sampling.temperature', maximum=2.0)
            decode_values['top_p'] = _field_float(
                sampling, 'top_p', problems, 'decode.sampling.top_p',
                maximum=1.0)
            if sampling.get('ignore_eos') is not True:
                problems.append('manifest_sampling_ignore_eos_must_be_true')
            if sampling.get('min_tokens') != 'equal_max':
                problems.append('manifest_sampling_min_tokens_must_equal_max')
    prefill = data.get('prefill')
    prefill_values = {}
    if type(prefill) is not dict:
        problems.append('manifest_field_missing:prefill')
    else:
        for name in ('fresh', 'append'):
            row = prefill.get(name)
            if type(row) is not dict:
                problems.append('manifest_field_missing:prefill.' + name)
                continue
            prefill_values[name] = {
                'prompt_tokens': _field_int(
                    row, 'prompt_tokens', 1, MAX_PROMPT_TOKENS, problems,
                    'prefill.%s.prompt_tokens' % name),
                'generation_budget': _field_int(
                    row, 'generation_budget', 1, MAX_GENERATION_BUDGET,
                    problems, 'prefill.%s.generation_budget' % name),
                'seed': _field_int(row, 'seed', 0, 2 ** 63 - 1, problems,
                                   'prefill.%s.seed' % name),
            }
    timeouts = data.get('timeouts')
    timeout_values = {}
    if type(timeouts) is not dict:
        problems.append('manifest_field_missing:timeouts')
    else:
        for key in ('controller_s', 'read_s', 'drain_s', 'join_s', 'finish_s'):
            timeout_values[key] = _field_float(timeouts, key, problems,
                                               'timeouts.' + key)
    window_s = _field_float(data, 'window_s', problems,
                            maximum=1200.0)
    repeats = _field_int(data, 'repeats', 1, MAX_REPEATS, problems,
                         'repeats')
    reserve = _field_int(data, 'reserve_mib', 1, 65536, problems,
                         'reserve_mib')
    if reserve is not None and reserve != grading.DEFAULT_RESERVE_MIB:
        problems.append('reserve_mib_must_stay_512')
    route_plan = data.get('route_plan')
    if type(route_plan) is not list or tuple(route_plan) != LANES:
        problems.append('route_plan_must_be_prefill_full_prefill_split_overlap')
    claims = data.get('claims', {})
    if type(claims) is not dict:
        problems.append('manifest_field_invalid:claims')
        claims = {}
    if problems:
        return problems, None
    return [], {
        'schema': SCHEMA, 'workload': data['workload'],
        'target_label': data['target_label'], 'decode': decode_values,
        'prefill': prefill_values, 'timeouts': timeout_values,
        'window_s': window_s, 'repeats': repeats, 'reserve_mib': reserve,
        'claims': claims, 'manifest_sha256': _sha256_json(data),
        'route_plan': tuple(route_plan),
    }


def _workload_view(cfg, **overrides):
    """The minimal attributes the reviewed correctness stream helpers read."""
    decode = cfg['decode']
    values = dict(
        streams=decode['streams'], prompt_seed=decode['prompt_seed'],
        prompt_tokens=decode['prompt_tokens'],
        generation_budget=decode['generation_budget'],
        sampling_seed=decode['sampling_seed'],
        temperature=decode['temperature'], top_p=decode['top_p'],
        open_timeout_s=decode['open_timeout_s'],
        join_timeout_s=cfg['timeouts']['join_s'],
        finish_timeout_s=cfg['timeouts']['finish_s'],
        drain_timeout_s=cfg['timeouts']['drain_s'],
        read_timeout_s=cfg['timeouts']['read_s'],
        pause_timeout_s=cfg['timeouts']['controller_s'],
        controller_timeout_s=cfg['timeouts']['controller_s'],
        window_s=cfg['window_s'], window_tokens=1,
        fresh_namespace_per_run=True, manifest_sha256=cfg['manifest_sha256'],
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _prompt(seed, index, tokens):
    rng = random.Random(seed * 1000 + index)
    return correctness.AR.text(rng, tokens)


def _request(model, prompt, workload, salt):
    return correctness.request_body(model, prompt, workload, salt)


def _cache_salt(run_id, lane, index):
    raw = 'q18a|%s|%s|%d' % (run_id, lane, index)
    return 'q18a-' + hashlib.sha256(raw.encode()).hexdigest()[:40]


class CountSampler(threading.Thread):
    """Sample cumulative client-visible completion counts at a fixed cadence."""

    def __init__(self, states, interval_s=0.01):
        super().__init__(daemon=True)
        self.states = states
        self.interval_s = interval_s
        self.stop = threading.Event()
        self.samples = {state.index: [] for state in states}
        self.lock = threading.Lock()

    def run(self):
        while not self.stop.wait(self.interval_s):
            now = time.monotonic_ns()
            with self.lock:
                for state in self.states:
                    self.samples[state.index].append((now, state.count()))

    def snapshot(self):
        now = time.monotonic_ns()
        with self.lock:
            for state in self.states:
                self.samples[state.index].append((now, state.count()))
        return now

    def count_at(self, index, ns):
        with self.lock:
            rows = self.samples.get(index) or []
            value = 0
            for stamp, count in rows:
                if stamp <= ns:
                    value = count
                else:
                    break
            return value

    def close(self):
        self.stop.set()
        self.join(timeout=2.0)


def _arrivals_between(state, start_ns, end_ns):
    return sorted(stamp for stamp in state.arrivals
                  if _is_int(stamp) and start_ns <= stamp <= end_ns)


def _gaps(times):
    return [round((b - a) / 1e6, 3) for a, b in zip(times, times[1:])]


def _stream_window(states, sampler, start_ns, end_ns):
    wall_s = (end_ns - start_ns) / 1e9
    rows = []
    total = 0
    all_gaps = []
    for state in states:
        begin = sampler.count_at(state.index, start_ns)
        end = sampler.count_at(state.index, end_ns)
        delta = max(0, end - begin)
        arrivals = _arrivals_between(state, start_ns, end_ns)
        gaps = _gaps(arrivals)
        open_gap = _ms(arrivals[0] - start_ns) if arrivals else None
        total += delta
        all_gaps.extend(gaps)
        rows.append({
            'stream': state.index,
            'start_completion': begin,
            'end_completion': end,
            'completion_delta': delta,
            'rate_tokens_per_s': round(delta / wall_s, 3) if wall_s > 0 else None,
            'in_window_arrivals': len(arrivals),
            'coalesced_note': correctness.COALESCING_NOTE,
            'gaps_ms': gaps,
            'gap_ms_p99': _pct(gaps, 0.99),
            'open_gap_ms': open_gap,
            'error_before_window_end': state.error is not None
            or state.malformed is not None,
        })
    return {
        'start_ns': start_ns, 'end_ns': end_ns,
        'wall_s': round(wall_s, 6),
        'aggregate_tokens_per_s': (round(total / wall_s, 3)
                                   if wall_s > 0 else None),
        'stream_rates_tokens_per_s': [row['rate_tokens_per_s']
                                      for row in rows],
        'gap_ms_p50': _pct(all_gaps, 0.5),
        'gap_ms_p95': _pct(all_gaps, 0.95),
        'gap_ms_p99': _pct(all_gaps, 0.99),
        'gap_ms_max': max(all_gaps) if all_gaps else None,
        'open_gap_ms_p50': _pct([r['open_gap_ms'] for r in rows
                                 if r['open_gap_ms'] is not None], 0.5),
        'streams': rows,
        'client_visible': True,
        'clock': {'basis': 'time.monotonic_ns', 'wall_anchor_ns': start_ns},
        'basis': correctness.WINDOW_BASIS,
    }


class ReceiptError(Exception):
    pass


def _receipt(receipt, action, run_id):
    if type(receipt) is not dict:
        raise ReceiptError('receipt_not_object')
    if receipt.get('schema') != 1:
        raise ReceiptError('receipt_schema')
    if receipt.get('action') != action:
        raise ReceiptError('receipt_action')
    if receipt.get('run_id') != run_id:
        raise ReceiptError('receipt_run_id')
    if not _is_int(receipt.get('engine_epoch')) or receipt['engine_epoch'] <= 0:
        raise ReceiptError('receipt_engine_epoch')
    return receipt


def _call(controller, cfg, command, action):
    receipt, transport = controller.execute(command, cfg['timeouts']['controller_s'])
    receipt = _receipt(receipt, action, command['run_id'])
    if command.get('engine_epoch') is not None and receipt['engine_epoch'] != command['engine_epoch']:
        raise ReceiptError('receipt_engine_epoch_mismatch')
    return receipt, transport


def _prefill_prompts(cfg):
    """Fresh prompt plus a real appended suffix sharing its cache namespace.

    The append request is the *same conversation*: the fresh prompt text
    followed by a new suffix, under the same ``cache_salt``.  It is therefore a
    genuine prefix-cache append probe, not a second unrelated prompt.  Only
    hashes and token lengths leave this function.
    """
    fresh = cfg['prefill']['fresh']
    append = cfg['prefill']['append']
    fresh_prompt = _prompt(fresh['seed'], 0, fresh['prompt_tokens'])
    suffix = _prompt(append['seed'], 1, append['prompt_tokens'])
    append_prompt = fresh_prompt + '\n' + suffix
    return {
        'fresh': fresh_prompt,
        'append': append_prompt,
        'fresh_tokens': fresh['prompt_tokens'],
        'append_tokens': fresh['prompt_tokens'] + append['prompt_tokens'],
        'shared_prefix_tokens': fresh['prompt_tokens'],
        'fresh_sha256': hashlib.sha256(fresh_prompt.encode()).hexdigest(),
        'append_sha256': hashlib.sha256(append_prompt.encode()).hexdigest(),
    }


def _send_prefill(url, model, cfg, lane, index, prompt, generation_budget,
                  prompt_seed, salt):
    workload = _workload_view(cfg, streams=1, generation_budget=generation_budget,
                              prompt_seed=prompt_seed,
                              finish_timeout_s=cfg['timeouts']['finish_s'])
    body = _request(model, prompt, workload, salt)
    states = correctness._start_streams(url, model, [body],
                                        [{'prompt_sha256': hashlib.sha256(
                                            prompt.encode()).hexdigest(),
                                          'prompt_chars': len(prompt)}],
                                        workload)
    return states[0], workload


def _wait_first_advance(state, timeout_s):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if state.first_advance_ns is not None:
            return state.first_advance_ns
        if state.error is not None or state.malformed is not None \
                or state.finish_at_ns is not None:
            return None
        state.progress.wait(0.01)
        state.progress.clear()
    return None


def _finish_or_cancel(states, workload):
    timed_out = correctness._wait_finish(states, workload)
    record = {'timed_out': timed_out, 'readers_alive_after_join': []}
    correctness._stop_streams(record, states, workload)
    return record


def _loads(path):
    return json.loads(Path(path).read_text())


def _package_gate(env):
    """Resolve the trusted launcher and reviewed readiness without env trust."""
    overlay = env.get('S5_OVERLAY')
    launcher_name = env.get('S5_LAUNCHER_NAME', 'run-flash-s5-test.sh')
    launcher_sha = None
    if overlay:
        launcher = Path(overlay) / launcher_name
        if not launcher.is_file() or launcher.is_symlink():
            return {'ok': False, 'reasons': ['candidate_launcher_missing'],
                    'launcher_sha256': None}
        launcher_sha = hashlib.sha256(launcher.read_bytes()).hexdigest()
    readiness = _loads(env['READINESS']) if env.get('READINESS') else None
    generation = _loads(env['GENERATION']) if env.get('GENERATION') else None
    if generation is None and overlay:
        candidate = Path(overlay) / 'S5-OVERLAP-GENERATION.json'
        generation = json.loads(candidate.read_text()) if candidate.is_file() \
            else None
    entries = {}
    if env.get('PACKAGE_MANIFEST'):
        for line in Path(env['PACKAGE_MANIFEST']).read_text().splitlines():
            fields = line.split(maxsplit=1)
            if len(fields) == 2:
                entries[fields[1].lstrip('*')] = fields[0]
    ready = grading.candidate_readiness(
        generation, readiness, launcher_sha=launcher_sha,
        package_sha256=(readiness or {}).get('package_sha256'),
        manifest_entries=entries,
        env_launcher_sha=env.get('S5_LAUNCHER_SHA'))
    return {'ok': ready['verdict'] == 'PASS', 'reasons': ready['failures'],
            'missing': ready['missing'], 'launcher_sha256': launcher_sha,
            'package_sha256': ready.get('package_sha256'),
            'readiness': ready, 'generation': generation}


def _gates(env):
    if env.get('REQUIRE_GATES', '1') != '1':
        return {'ok': False, 'reasons': ['gate_enforcement_bypassed']}
    path = env.get('GATES')
    if not path:
        return {'ok': False, 'reasons': ['gates_unavailable']}
    data = _loads(path)
    outcomes = data.get('outcomes') if type(data) is dict else None
    if type(outcomes) is not dict or not outcomes:
        return {'ok': False, 'reasons': ['gates_not_declared']}
    bad = sorted(name for name, value in outcomes.items() if value != 'PASS')
    if data.get('require_native_ready') is not True:
        bad.append('require_native_ready')
    return {'ok': not bad, 'reasons': bad, 'outcomes': outcomes}


def _plan_only(cfg, env, reasons):
    run_id = env.get('RUN_ID') or ('q18a-plan-%d' % time.time_ns())
    return {
        'schema': SCHEMA, 'run_id': run_id,
        'workload': (cfg or {}).get('workload'),
        'manifest_sha256': (cfg or {}).get('manifest_sha256'),
        'target_label': (cfg or {}).get('target_label'),
        'not_run': True, 'reasons': reasons,
        'lanes': {}, 'overlap': {},
        'scope': 'plan-only; either the manifest, controller, gates or package '
                 'readiness is incomplete. No request was issued.',
    }


def run_lane_prefill(cfg, controller, url, model, ctx, lane, route, run_id):
    _call(controller, cfg, {'schema': 1, 'run_id': run_id, 'action': 'prepare',
                            'target_route': route, 'engine_epoch': ctx.get('engine_epoch')},
          'prepare')
    fresh = cfg['prefill']['fresh']
    append = cfg['prefill']['append']
    prompts = _prefill_prompts(cfg)
    salt = _cache_salt(run_id, lane, 0)
    rows = []
    started = time.monotonic_ns()
    for name, spec in (('fresh', fresh), ('append', append)):
        state, workload = _send_prefill(
            url, model, cfg, lane, 0 if name == 'fresh' else 1,
            prompts[name], spec['generation_budget'], spec['seed'], salt)
        first = _wait_first_advance(state, cfg['timeouts']['read_s'])
        finish = _finish_or_cancel([state], workload)
        rows.append({'request': name, 'send_at_ns': state.send_at_ns,
                     'first_advance_at_ns': first,
                     'ttft_s': (round((first - state.send_at_ns) / 1e9, 3)
                                if first and state.send_at_ns else None),
                     'latency_s': (round((state.finish_at_ns - state.send_at_ns)
                                         / 1e9, 3)
                                   if state.finish_at_ns and state.send_at_ns
                                   else None),
                     'prompt_tokens': (prompts['fresh_tokens'] if name == 'fresh'
                                       else prompts['append_tokens']),
                     'shared_prefix_tokens': (0 if name == 'fresh'
                                              else prompts['shared_prefix_tokens']),
                     'prompt_sha256': prompts[name + '_sha256'],
                     'error': state.error is not None,
                     'malformed': state.malformed,
                     'finish': finish})
        if state.error is not None or state.malformed is not None:
            break
    end = time.monotonic_ns()
    receipt, _transport = _call(
        controller, cfg,
        {'schema': 1, 'run_id': run_id, 'action': 'overlap_end', 'lane': lane,
         'engine_epoch': ctx.get('engine_epoch'), 'route': route},
        'overlap_end')
    prefill = receipt.get('prefill') or {}
    useful = prefill.get('useful_completed_tokens')
    wall = prefill.get('prefill_wall_s')
    valid = (_is_int(useful) and useful > 0 and _is_num(wall) and wall > 0
             and all(not row['error'] and row['malformed'] is None
                     for row in rows))
    _call(controller, cfg, {'schema': 1, 'run_id': run_id,
                            'action': 'leave_full', 'target_route': 'full',
                            'engine_epoch': ctx.get('engine_epoch')},
          'leave_full')
    return {
        'route': route, 'valid': valid,
        'serving_baseline': route == 'full',
        'prefill_tokens_per_s': (round(useful / wall, 3)
                                 if valid else None),
        'useful_completed_tokens': useful if _is_int(useful) else None,
        'prompt_tokens': prefill.get('prompt_tokens'),
        'prefill_wall_s': wall if _is_num(wall) else None,
        'completed_prefills': prefill.get('completed_prefills'),
        'fresh_ttft_s': rows[0]['ttft_s'] if rows else None,
        'append_ttft_s': rows[1]['ttft_s'] if len(rows) > 1 else None,
        'requests': rows,
        'workload_sha256': cfg['manifest_sha256'],
        'driver_wall_s': round((end - started) / 1e9, 3),
        'scope': 'prefill solo lane; no decode cohort',
    }


def _overlap_evidence(receipt):
    overlap = receipt.get('overlap')
    if type(overlap) is not dict:
        raise ReceiptError('overlap_section_missing')
    return overlap


def _invalid_overlap(exc, cfg):
    trusted = {'receipt_not_object', 'receipt_schema', 'receipt_action',
               'receipt_run_id', 'receipt_engine_epoch', 'receipt_engine_epoch_mismatch',
               'measurement_start_unproved', 'measurement_end_unproved',
               'prefill_completion_unproved', 'overlap_section_missing'}
    reason = str(exc) if type(exc) is ReceiptError and str(exc) in trusted else type(exc).__name__
    return {'route': 'split', 'valid': False,
            'invalid_intervals': [{'reasons': [reason]}],
            'engine_crash': None,
            'controller_transport_failed': isinstance(exc, protocol.ControllerTransportError),
            'workload_sha256': cfg['manifest_sha256'],
            'scope': 'incomplete controller evidence; engine crash requires separate engine observation'}


def run_lane_overlap(cfg, controller, url, model, ctx, run_id):
    decode = cfg['decode']
    workload = _workload_view(cfg)
    cohort = correctness.build_cohort(workload)
    bodies = [_request(model, row['prompt'], workload,
                       _cache_salt(run_id, 'overlap', index))
              for index, row in enumerate(cohort)]
    try:
        prepared, _transport = _call(controller, cfg,
                         {'schema': 1, 'run_id': run_id, 'action': 'prepare',
                          'target_route': 'full',
                          'engine_epoch': ctx.get('engine_epoch')}, 'prepare')
    except (ReceiptError, protocol.ControllerTransportError) as exc:
        return _invalid_overlap(exc, cfg)
    engine_epoch = prepared['engine_epoch']
    ctx['engine_epoch'] = engine_epoch
    states = correctness._start_streams(url, model, bodies, cohort, workload)
    opened = correctness._wait_open(states, workload)
    if not opened.get('complete'):
        correctness._stop_streams({}, states, workload)
        return {'route': 'split', 'valid': False,
                'invalid_intervals': [{'reasons': ['open_incomplete']}],
                'open': opened, 'workload_sha256': cfg['manifest_sha256']}
    sampler = CountSampler(states)
    sampler.start()
    try:
        _call(controller, cfg,
              {'schema': 1, 'run_id': run_id, 'action': 'pause_enter_split',
               'target_route': 'split', 'engine_epoch': engine_epoch},
              'pause_enter_split')
        _call(controller, cfg,
              {'schema': 1, 'run_id': run_id, 'action': 'resume',
               'target_route': 'split', 'engine_epoch': engine_epoch},
              'resume')
        begin, _transport = _call(
            controller, cfg,
            {'schema': 1, 'run_id': run_id, 'action': 'overlap_begin',
             'route': 'split', 'decode_streams': len(states),
             'engine_epoch': engine_epoch}, 'overlap_begin')
        measure_start_ns = begin.get('measure_start_ns')
        if not _is_int(measure_start_ns) or measure_start_ns <= 0:
            raise ReceiptError('measurement_start_unproved')
        fresh_spec = cfg['prefill']['fresh']
        append_spec = cfg['prefill']['append']
        prompts = _prefill_prompts(cfg)
        salt = _cache_salt(run_id, 'overlap', 0)
        prefill_states = []
        ttfts = {}
        prefill_finishes = {}
        for name, spec in (('fresh', fresh_spec), ('append', append_spec)):
            state, prefill_workload = _send_prefill(
                url, model, cfg, 'overlap', 0 if name == 'fresh' else 1,
                prompts[name], spec['generation_budget'], spec['seed'], salt)
            prefill_states.append((name, state, prefill_workload))
            first = _wait_first_advance(state, cfg['timeouts']['read_s'])
            ttfts[name] = (round((first - state.send_at_ns) / 1e9, 3)
                           if first and state.send_at_ns else None)
            # Append is offered only after the fresh request really completed;
            # sharing a cache salt while both cold requests run is not append.
            prefill_finishes[name] = _finish_or_cancel([state], prefill_workload)
            if (state.error is not None or state.malformed is not None
                    or not state.finish_at_ns):
                raise ReceiptError('prefill_completion_unproved')
        measure_end_driver_ns = sampler.snapshot()
        end, _transport = _call(
            controller, cfg,
            {'schema': 1, 'run_id': run_id, 'action': 'overlap_end',
             'route': 'split', 'engine_epoch': engine_epoch}, 'overlap_end')
        measure_end_ns = end.get('measure_end_ns')
        if not _is_int(measure_end_ns) or measure_end_ns <= measure_start_ns:
            raise ReceiptError('measurement_end_unproved')
        window = _stream_window(states, sampler, measure_start_ns,
                                measure_end_ns)
        try:
            overlap = _overlap_evidence(end)
        except ReceiptError as exc:
            overlap = {'missing': [str(exc)]}
        for segment in overlap.get('segments') or []:
            if type(segment) is not dict:
                continue
            start = segment.get('start_monotonic_ns')
            stop = segment.get('end_monotonic_ns')
            if _is_int(start) and _is_int(stop) and stop > start:
                segment['client_decode_arrivals'] = sum(
                    len(_arrivals_between(state, start, stop))
                    for state in states)
        interval = end.get('interval') or {}
        overlap['interval_counters'] = {
            'reset_count': interval.get('reset_count'),
            'source': 'overlap_end.pause receipt',
        }
        cleanup_record = dict(prefill_finishes)
        _call(controller, cfg,
              {'schema': 1, 'run_id': run_id, 'action': 'leave_full',
               'target_route': 'full', 'engine_epoch': engine_epoch},
              'leave_full')
        _call(controller, cfg,
              {'schema': 1, 'run_id': run_id, 'action': 'resume',
               'target_route': 'full', 'engine_epoch': engine_epoch},
              'resume')
        sampler.close()
        driver_cleanup = correctness._stop_streams({}, states, workload)
    except (ReceiptError, protocol.ControllerTransportError) as exc:
        sampler.close()
        correctness._stop_streams({}, states, workload)
        return _invalid_overlap(exc, cfg)
    prefill = overlap.get('prefill') or end.get('prefill') or {}
    useful = prefill.get('useful_completed_tokens')
    wall = prefill.get('prefill_wall_s')
    enabled = overlap.get('enabled') is True and overlap.get('native_ready') is True
    failures = []
    if not enabled:
        failures.append('overlap_not_enabled_or_native_ready')
    client_arrivals = sum(row.get('in_window_arrivals') or 0
                          for row in window['streams'])
    if client_arrivals <= 0:
        failures.append('no_client_decode_arrivals_during_prefill')
    for segment in overlap.get('segments') or []:
        if type(segment) is not dict or segment.get('accepted') is not True:
            continue
        if segment.get('client_decode_arrivals') == 0:
            failures.append('serialized_segment_no_client_arrivals')
    valid = (enabled and not failures and _is_int(useful) and useful > 0
             and _is_num(wall) and wall > 0)
    overlap.update({
        'route': 'split', 'valid': valid, 'target_label': cfg['target_label'],
        'decode_tokens_per_s': window['aggregate_tokens_per_s'],
        'stream_rates_tokens_per_s': window['stream_rates_tokens_per_s'],
        'gap_p99_ms': window['gap_ms_p99'],
        'gap_p50_ms': window['gap_ms_p50'],
        'window': window,
        'fresh_ttft_s': ttfts.get('fresh'),
        'append_ttft_s': ttfts.get('append'),
        'prefill_prompt_sha256': {'fresh': prompts['fresh_sha256'],
                                  'append': prompts['append_sha256']},
        'prefill_prompt_tokens': {'fresh': prompts['fresh_tokens'],
                                  'append': prompts['append_tokens']},
        'append_shared_prefix_tokens': prompts['shared_prefix_tokens'],
        'prefill_tokens_per_s': (round(useful / wall, 3)
                                 if _is_num(useful) and _is_num(wall)
                                 and wall > 0 else None),
        'useful_completed_tokens': useful if _is_int(useful) else None,
        'prompt_tokens': prefill.get('prompt_tokens'),
        'prefill_wall_s': wall if _is_num(wall) else None,
        'completed_prefills': prefill.get('completed_prefills'),
        'valid_completed_turns': end.get('valid_completed_turns'),
        'cancellations': end.get('cancellations'),
        'errors': end.get('errors'),
        'valid_intervals': 1 if valid else 0,
        'invalid_intervals': [] if valid else [{'reasons': failures
                                                or ['overlap_lane_incomplete']}],
        'summary_basis': {'filtered_invalid': False,
                          'retry_selected': False, 'picked': False,
                          'all_declared_repeats_retained': True},
        'claims': dict(cfg['claims']),
        'engine_crash': end.get('engine_crash') is True,
        'worker_incarnation': end.get('worker_incarnation'),
        'engine_epoch': engine_epoch,
        'storage_epoch': end.get('storage_epoch'),
        'route_generation': end.get('route_generation'),
        'allocator_peaks': (end.get('allocator_peaks')
                            or (end.get('memory') or {}).get('allocator_peaks')),
        'workload_sha256': cfg['manifest_sha256'],
        'prefill_cleanup': cleanup_record,
        'driver_cleanup': driver_cleanup,
        'scope': 'real decode-during-prefill window on split; client-visible '
                 'decode counts plus engine segment receipts',
    })
    if failures:
        overlap['invalid_intervals'] = [{'reasons': sorted(set(failures))}]
    return overlap


def run(cfg, controller, url, model, env):
    run_id = env.get('RUN_ID') or ('q18a-%d' % time.time_ns())
    wall_anchor = time.time()
    wall_anchor_ns = time.monotonic_ns()
    ctx = {}
    lanes = {}
    lanes['prefill_full'] = run_lane_prefill(cfg, controller, url, model, ctx,
                                             'prefill_full', 'full', run_id)
    lanes['prefill_split'] = run_lane_prefill(cfg, controller, url, model, ctx,
                                              'prefill_split', 'split', run_id)
    overlap = run_lane_overlap(cfg, controller, url, model, ctx, run_id)
    readiness = {}
    if env.get('READINESS'):
        readiness = _loads(env['READINESS'])
    package = _package_gate(env)
    return {
        'schema': SCHEMA, 'run_id': run_id,
        'workload': cfg['workload'],
        'manifest_sha256': cfg['manifest_sha256'],
        'target_label': cfg['target_label'],
        'wall_anchor_s': wall_anchor,
        'wall_anchor_ns': wall_anchor_ns,
        'clock': {'client_basis': 'time.monotonic_ns',
                  'engine_basis': 'controller receipt / native event query',
                  'boundary': CLOCK_NOTE},
        'launcher_sha256': package.get('launcher_sha256'),
        'env_launcher_sha256': env.get('S5_LAUNCHER_SHA'),
        'package_sha256': package.get('package_sha256'),
        'lanes': lanes,
        'overlap': overlap,
        'memory': {'reserve_mib': cfg['reserve_mib'],
                   'allocator_peaks': overlap.get('allocator_peaks')
                   or (overlap.get('memory') or {}).get('allocator_peaks')},
    }


def main(argv, controller=None, env=None):
    env = dict(os.environ if env is None else env)
    if len(argv) != 3:
        print('usage: e3_overlap_performance.py URL MODEL OUT.json',
              file=sys.stderr)
        return 2
    url, model, out_path = argv
    if Path(out_path).exists() or Path(out_path).is_symlink():
        print('e3_overlap_performance refuses an existing output artifact',
              file=sys.stderr)
        return 2
    try:
        data = json.loads(Path(env.get('WORKLOAD', '')).read_text())
    except (OSError, ValueError):
        artifact = _plan_only(None, env, ['workload_manifest_unavailable'])
        Path(out_path).write_text(json.dumps(artifact, sort_keys=True) + '\n')
        print('e3_overlap_performance INCONCLUSIVE')
        return 2
    problems, cfg = validate_manifest(data)
    if problems:
        artifact = _plan_only(None, env, problems)
        Path(out_path).write_text(json.dumps(artifact, sort_keys=True) + '\n')
        print('e3_overlap_performance INCONCLUSIVE')
        return 2
    package = _package_gate(env)
    gates = _gates(env)
    if not package['ok'] or not gates['ok']:
        reasons = ['package_readiness:' + item
                   for item in package.get('reasons') or []] + \
                  ['gates:' + item for item in gates.get('reasons') or []]
        artifact = _plan_only(cfg, env, reasons)
        artifact.update({'package_readiness': package,
                         'gates': gates})
        Path(out_path).write_text(json.dumps(artifact, sort_keys=True) + '\n')
        print('e3_overlap_performance INCONCLUSIVE')
        return 2
    if controller is None:
        if not env.get('CONTROLLER_CMD'):
            artifact = _plan_only(cfg, env, ['controller_command_unavailable'])
            Path(out_path).write_text(json.dumps(artifact, sort_keys=True) + '\n')
            print('e3_overlap_performance INCONCLUSIVE')
            return 2
        controller = protocol.CommandController(env['CONTROLLER_CMD'])
    try:
        artifact = run(cfg, controller, url, model, env)
    except Exception as exc:  # noqa: BLE001 - retain an owned artifact
        artifact = _plan_only(cfg, env, ['run_failed:' + type(exc).__name__])
        Path(out_path).write_text(json.dumps(artifact, sort_keys=True) + '\n')
        print('e3_overlap_performance INCONCLUSIVE')
        return 2
    artifact['package_readiness'] = package
    artifact['gates'] = gates
    Path(out_path).write_text(json.dumps(artifact, sort_keys=True) + '\n')
    valid = artifact.get('overlap', {}).get('valid') is True
    print('e3_overlap_performance ' + ('COMPLETE' if valid else 'INCONCLUSIVE'))
    return 0 if valid else 2


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
