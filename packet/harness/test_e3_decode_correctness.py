"""Offline tests for the q17b short decode-correctness client.

Fake completion SSE server + injected schema-1 controller.  No torch, no vLLM,
no GPU, no network beyond loopback.  One composed test drives the client through
the actual ProbeSession/ProbeController validator chain.
"""
import asyncio
import copy
import hashlib
import json
import os
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import e3_decode_correctness as client
import e3_w1_decode_driver as driver

HERE = Path(__file__).resolve().parent
SECRET = '***'
MODEL = 'correctness-test-model'


def deep_update(base, extra):
    for key, value in extra.items():
        if type(value) is dict and type(base.get(key)) is dict:
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def base_manifest(**overrides):
    data = {
        'schema': 1,
        'workload': 'correctness-test',
        'streams': 3,
        'prompt': {'seed': 17, 'tokens': 24},
        'generation_budget': 16,
        'sampling': {'seed': 909, 'temperature': 0.0, 'top_p': 1.0,
                     'ignore_eos': True, 'min_tokens': 'equal_max'},
        'bounds': {'window_s': 1.0, 'window_tokens': 3},
        'timeouts': {'open_s': 2.0, 'pause_s': 2.0, 'read_s': 5.0,
                     'drain_s': 1.0, 'join_s': 0.5, 'finish_s': 3.0,
                     'controller_s': 2.0},
        'expected': {'output_tokens': 16, 'finish_reason': 'length'},
        'cache': {'fresh_namespace_per_run': True},
        'route_plan': ['full', 'split', 'full'],
    }
    return deep_update(data, overrides)


def workload_for(data):
    problems, workload, sha = client.validate_manifest(data)
    if problems:
        raise AssertionError(problems)
    return workload


class PhaseGate:
    """Persistent run-phase events: fast pause/resume pairs cannot be missed."""

    NAMES = ('entry', 'measure', 'end', 'cleanup')

    def __init__(self):
        self.stop = threading.Event()
        self.events = {name: threading.Event() for name in self.NAMES}
        self.lock = threading.Condition()
        self.phase = 'initial'

    def reset(self):
        with self.lock:
            self.phase = 'initial'
            for event in self.events.values():
                event.clear()

    def set_phase(self, name):
        if name not in self.events:
            raise AssertionError('unknown gate phase ' + name)
        with self.lock:
            self.phase = name
            self.events[name].set()
            self.lock.notify_all()

    def wait(self, name, timeout=5.0):
        deadline = time.monotonic() + timeout
        while not self.events[name].is_set():
            if self.stop.is_set() or time.monotonic() >= deadline:
                return False
            with self.lock:
                self.lock.wait(0.005)
        return True

    def sleep(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self.stop.is_set():
                return False
            time.sleep(0.002)
        return True


DEFAULT_PLAN = {
    'chunk_size': lambda run, stream: 1,
    'mutate': lambda run, stream, pos, token: token,
    'finish_at': lambda run, stream: None,       # None -> full budget
    'finish_reason': lambda run, stream: 'length',
    'usage_final': lambda run, stream: None,     # None -> token count
    'malformed': lambda run, stream: None,       # ('replace'|'negative'|'not_int', pos)
    'missing_id': lambda run, stream: None,      # position
    'not_list': lambda run, stream: False,
    'usage_first': lambda run, stream: False,
    'duplicate_usage_after': lambda run, stream: None,
    'content_first': lambda run, stream: False,
    'fragment': lambda run, stream: False,
    'start_delay': lambda run, stream: 0.0,
    'silent_after': lambda run, stream: None,
    'http_error': lambda run, stream: False,
    'prompt_ids': lambda run, stream: False,
}


class Flow:
    """Shared cohort state, request log and per-(run,stream) emission plan."""

    def __init__(self, workload, gate, *, open_tokens=2, delay_s=0.003):
        self.workload = workload
        self.gate = gate
        self.open_tokens = open_tokens
        self.delay_s = delay_s
        self.measure_token_cap = None
        self.budget = workload.generation_budget
        self.lock = threading.Lock()
        self.cohort = client.build_cohort(workload)
        self.by_hash = {row['prompt_sha256']: row['stream']
                        for row in self.cohort}
        self.request_count = 0
        self.requests = []
        self.plan = dict(DEFAULT_PLAN)

    def configure(self, **plans):
        for key, value in plans.items():
            if key not in DEFAULT_PLAN:
                raise AssertionError('unknown flow plan ' + key)
            self.plan[key] = value

    def allocate(self, body, prompt):
        prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
        with self.lock:
            stream = self.by_hash.get(prompt_sha)
            ordinal = self.request_count
            self.request_count += 1
            run = ordinal // self.workload.streams
            self.requests.append({
                'ordinal': ordinal,
                'run': run,
                'stream': stream,
                'at_ns': time.monotonic_ns(),
                'prompt_sha256': prompt_sha,
                'model': body.get('model'),
                'n': body.get('n'),
                'stream': body.get('stream'),
                'max_tokens': body.get('max_tokens'),
                'min_tokens': body.get('min_tokens'),
                'return_token_ids': body.get('return_token_ids'),
                'stream_options': copy.deepcopy(body.get('stream_options')),
                'cache_salt_sha256': client._sha256_text(str(body.get('cache_salt'))),
                'body_sha256': client._sha256_json(body),
            })
        return run, stream


class FakeServer:
    def __init__(self, flow):
        self.flow = flow

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                if self.path != client.ENDPOINT:
                    self.send_error(404)
                    return
                try:
                    size = int(self.headers.get('Content-Length', '0'))
                except ValueError:
                    size = 0
                raw = self.rfile.read(size)
                try:
                    body = json.loads(raw)
                except ValueError:
                    self.send_error(400)
                    return
                prompt = body.get('prompt')
                if not isinstance(prompt, str):
                    self.send_error(400)
                    return
                run, stream = self.flow.allocate(body, prompt)
                if stream is None:
                    self.send_error(400)
                    return
                self.serve_stream(run, stream, body, prompt)

            def serve_stream(self, run, stream, body, prompt):
                plan = self.flow.plan
                if plan['http_error'](run, stream):
                    self.send_response(503)
                    self.send_header('Content-Type', 'text/plain')
                    self.end_headers()
                    self.wfile.write(b'busy')
                    return
                try:
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.send_header('Cache-Control', 'no-cache')
                    self.send_header('Connection', 'close')
                    self.end_headers()
                except OSError:
                    return
                budget = self.flow.budget
                finish_at = plan['finish_at'](run, stream)
                if finish_at is None:
                    finish_at = budget
                reason = plan['finish_reason'](run, stream)
                usage_limit = plan['usage_final'](run, stream)
                if usage_limit is None:
                    usage_limit = finish_at
                tokens = [100000 + stream * 1000 + position
                          for position in range(budget)]
                tokens = [plan['mutate'](run, stream, position, token)
                          for position, token in enumerate(tokens)]
                prompt_tokens = 24

                def usage(count):
                    return {'prompt_tokens': prompt_tokens,
                            'completion_tokens': min(count, usage_limit)}

                def event(ids, count, finish=None, text='', prompt_ids=None):
                    choice = {'index': 0, 'text': text, 'logprobs': None,
                              'finish_reason': finish}
                    if ids is not None:
                        choice['token_ids'] = ids
                    if prompt_ids is not None:
                        choice['prompt_token_ids'] = prompt_ids
                    return {'id': 'cmpl-test', 'object': 'text_completion',
                            'created': 0, 'model': MODEL,
                            'choices': [choice], 'usage': usage(count)}

                def usage_only(count):
                    return {'id': 'cmpl-test', 'object': 'text_completion',
                            'created': 0, 'model': MODEL, 'choices': [],
                            'usage': usage(count)}

                def write(payload, fragmented=False):
                    text = 'data: ' + json.dumps(payload, separators=(',', ':')) + '\n\n'
                    data = text.encode()
                    try:
                        if fragmented and len(data) > 2:
                            middle = len(data) // 2
                            self.wfile.write(data[:middle])
                            self.wfile.flush()
                            time.sleep(0.003)
                            self.wfile.write(data[middle:])
                        else:
                            self.wfile.write(data)
                        self.wfile.flush()
                        return True
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        return False

                delay = plan['start_delay'](run, stream)
                silent_after = plan['silent_after'](run, stream)
                if plan['content_first'](run, stream):
                    if not write(event([], 0, text=SECRET)):
                        return
                if plan['usage_first'](run, stream):
                    if not write(usage_only(1)):
                        return
                if delay and not self.flow.gate.sleep(delay):
                    return
                first_chunk = True
                emitted = 0

                def emit_chunk():
                    nonlocal emitted, first_chunk
                    size = plan['chunk_size'](run, stream)
                    count = min(size, finish_at - emitted)
                    positions = list(range(emitted, emitted + count))
                    ids = [tokens[position] for position in positions]
                    missing = plan['missing_id'](run, stream)
                    if missing is not None and missing in positions:
                        ids = None
                    malformed = plan['malformed'](run, stream)
                    if malformed is not None and ids is not None:
                        kind, position = malformed
                        if position in positions:
                            index = position - emitted
                            if kind == 'replace':
                                ids[index] = 'not-an-int'
                            elif kind == 'negative':
                                ids[index] = -5
                            elif kind == 'not_int':
                                ids[index] = True
                    last = (emitted + count >= finish_at)
                    if plan['not_list'](run, stream) and first_chunk:
                        ids = 'not-a-list'
                    prompt_ids = list(range(prompt_tokens)) if (
                        first_chunk and plan['prompt_ids'](run, stream)) else None
                    payload = event(ids, emitted + count,
                                    reason if last else None,
                                    prompt_ids=prompt_ids)
                    if not write(payload, plan['fragment'](run, stream)
                                 and first_chunk):
                        return False
                    emitted += count
                    first_chunk = False
                    duplicate_at = plan['duplicate_usage_after'](run, stream)
                    if duplicate_at is not None and emitted > duplicate_at:
                        if not write(usage_only(emitted)):
                            return False
                    return True

                while emitted < min(self.flow.open_tokens, finish_at):
                    if self.flow.gate.stop.is_set() or not emit_chunk():
                        return
                    if not self.flow.gate.sleep(self.flow.delay_s):
                        return
                if silent_after is not None and emitted >= silent_after:
                    while not self.flow.gate.stop.is_set():
                        time.sleep(0.01)
                    return
                if emitted < finish_at:
                    if not self.flow.gate.wait('entry'):
                        return
                    if not self.flow.gate.wait('measure'):
                        return
                    if silent_after is not None and emitted >= silent_after:
                        while not self.flow.gate.stop.is_set():
                            time.sleep(0.01)
                        return
                    while (emitted < finish_at
                           and (self.flow.measure_token_cap is None
                                or emitted < self.flow.measure_token_cap)
                           and not self.flow.gate.events['end'].is_set()):
                        if self.flow.gate.stop.is_set() or not emit_chunk():
                            return
                        if not self.flow.gate.sleep(self.flow.delay_s):
                            return
                    if not self.flow.gate.wait('end'):
                        return
                    if not self.flow.gate.wait('cleanup'):
                        return
                    while emitted < finish_at:
                        if self.flow.gate.stop.is_set() or not emit_chunk():
                            return
                        if not self.flow.gate.sleep(self.flow.delay_s):
                            return
                if not write(usage_only(finish_at)):
                    return
                write_text = 'data: [DONE]\n\n'
                try:
                    self.wfile.write(write_text.encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return

        Handler.flow = flow
        self.httpd = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.url = 'http://127.0.0.1:%d' % self.httpd.server_port

    def close(self):
        self.flow.gate.stop.set()
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=3)


class FakeController:
    """Injected schema-1 controller with the W1 receipt shape and chain."""

    def __init__(self, gate, streams, *, epoch=71, faults=None,
                 free_text='fake controller monotonic'):
        self.gate = gate
        self.streams = streams
        self.epoch = epoch
        self.faults = faults or {}
        self.free_text = free_text
        self.seq = 0
        self.prior = None
        self.route = 'full'
        self.generation = 0
        self.storage = 1
        self.steps = 10
        self.paused = False
        self.measured = False
        self.calls = []

    def fault(self, action, repeat):
        spec = self.faults.get((action, repeat), self.faults.get(action))
        if callable(spec):
            spec = spec()
        return spec or {}

    def _barriers(self, route, generation, drained=8):
        return [{'rank': rank, 'ack': 'flash_s5_probe_barrier',
                 'mode': 'decode-only-probe', 'route': route,
                 'route_generation': generation, 'storage_epoch': self.storage,
                 'engine_epoch': self.epoch, 'drained_streams': drained}
                for rank in driver.EXPECTED_RANKS]

    def _eligibility(self, command, reason='paused live request counts'):
        return {
            'cohort_id': command['cohort_id'],
            'cohort_cardinality': self.streams,
            'reason': reason,
            'ranks': [{'rank': rank, 'batch_reqs': self.streams,
                       'scheduled_tokens': self.streams, 'prefill': False}
                      for rank in driver.EXPECTED_RANKS],
        }

    def _ranks(self, route_after, generation_after, transition, fault):
        rows = []
        for rank in driver.EXPECTED_RANKS:
            row = {'rank': rank, 'status': 'ok'}
            if transition:
                ack_generation = generation_after
                if fault.get('stale_ack') and rank == 2:
                    ack_generation = generation_after - 1
                row['transition_ack'] = {
                    'rank': rank, 'ack': 'flash_s5_transition',
                    'mode': 'decode-only-probe', 'route': route_after,
                    'route_generation': ack_generation,
                    'storage_epoch': self.storage, 'engine_epoch': self.epoch,
                    'drained_streams': 8,
                }
            rows.append(row)
        if fault.get('partial_acks') and transition and rows:
            rows[-1].pop('transition_ack', None)
        if fault.get('drop_receipt_rank') and rows:
            rows = rows[:-1]
        return rows

    def _observations(self, route, generation, fault):
        rows = []
        for rank in driver.EXPECTED_RANKS:
            counters = {
                'route': route, 'route_generation': generation,
                'storage_epoch': self.storage,
                'interval_invalidations': 0, 'in_flight': 0,
                'sampling_pending': False, 'transition_pending': False,
                'replay_streams_match_owner': True,
                'batch_observations': {'decode_batches': self.steps,
                                       'batches': self.steps,
                                       'scheduler_observations': self.steps},
            }
            replays = fault.get('replays', 3)
            if replays is not None:
                counters['replays'] = replays
            if fault.get('worker_route') is not None:
                counters['route'] = fault['worker_route']
            if fault.get('missing_decode'):
                counters['batch_observations'].pop('decode_batches', None)
            if fault.get('transition_pending'):
                counters['transition_pending'] = True
            rows.append({'rank': rank, 'counters': counters})
        if fault.get('drop_worker_rank') and rows:
            rows = rows[:-1]
        return rows

    def execute(self, command, timeout_s):
        action = command['action']
        repeat = command['repeat']
        self.calls.append({'action': action, 'repeat': repeat,
                           'at_ns': time.monotonic_ns(),
                           'command': copy.deepcopy(command)})
        if command['command_seq'] != self.seq + 1:
            raise AssertionError('client command sequence broke')
        if command.get('prior_receipt') != self.prior:
            raise AssertionError('client prior-receipt chain broke')
        fault = self.fault(action, repeat)
        if fault.get('timeout'):
            self.seq = command['command_seq']
            raise driver.ControllerTransportError(
                'controller_timeout', {'kind': 'injected', 'timeout': True,
                                       'seconds': 0.01, 'rc': None,
                                       'stdout_len': 0, 'stdout_sha256': None,
                                       'stderr_len': 0})
        if fault.get('exception'):
            self.seq = command['command_seq']
            raise RuntimeError(SECRET)
        if fault.get('omit_receipt'):
            self.seq = command['command_seq']
            return None, {'kind': 'injected', 'rc': 0, 'seconds': 0.0001,
                          'stdout_len': 0, 'stdout_sha256': None,
                          'stderr_len': 0, 'timeout': False}
        before_route, before_generation = self.route, self.generation
        before_steps = self.steps
        route_after, generation_after = before_route, before_generation
        ranks_transition = False
        receipt = {
            'schema': 1, 'run_id': command['run_id'], 'repeat': repeat,
            'command_seq': command['command_seq'], 'action': action,
            'cohort_id': command['cohort_id'], 'status': 'ok',
            'engine_epoch': self.epoch, 'storage_epoch': self.storage,
            'route_before': before_route, 'route_after': before_route,
            'route_generation_before': before_generation,
            'route_generation_after': before_generation,
            'expected_rank_set': list(driver.EXPECTED_RANKS),
            'observed_clock_ns': time.monotonic_ns(),
            'clock_basis': fault.get('clock_basis', self.free_text),
            'ranks': [],
        }
        if action == 'prepare':
            self.gate.reset()
            self.measured = False
            if fault.get('prepare_changes_route'):
                route_after = 'split'
            if fault.get('prepare_changes_generation'):
                generation_after = before_generation + 1
        elif action in ('pause_enter_split', 'pause_full'):
            self.paused = True
            self.measured = False
            self.gate.set_phase('entry')
            if action == 'pause_enter_split':
                route_after = 'split'
                generation_after = before_generation + 1
                ranks_transition = True
            receipt['pause'] = {'mode': 'keep', 'clear_cache': False,
                                'completed': True, 'drained': True}
            if fault.get('pause_not_completed'):
                receipt['pause']['completed'] = False
            if fault.get('omit_pause'):
                receipt.pop('pause')
            receipt['barrier_replies'] = self._barriers(
                before_route, before_generation)
            receipt['decode_eligibility'] = self._eligibility(
                command, reason=fault.get('eligibility_reason',
                                          'paused live request counts'))
        elif action == 'resume':
            self.paused = False
            self.gate.set_phase('cleanup' if self.measured else 'measure')
            self.measured = True
            receipt['resume'] = {'completed': True}
            if fault.get('omit_resume'):
                receipt.pop('resume')
            receipt['barrier_replies'] = self._barriers(
                before_route, before_generation)
            if route_after != before_route:
                receipt['route_after'] = route_after
        elif action == 'pause_end':
            self.paused = True
            self.gate.set_phase('end')
            if fault.get('engine_changed'):
                receipt['engine_epoch'] = self.epoch + 1
            self.steps += 5
            if fault.get('interval_no_progress'):
                after_steps = before_steps
            else:
                after_steps = self.steps
            receipt['pause'] = {'mode': 'keep', 'clear_cache': False,
                                'completed': True, 'drained': True}
            if fault.get('pause_not_completed'):
                receipt['pause']['completed'] = False
            if fault.get('omit_pause'):
                receipt.pop('pause')
            if fault.get('route_after_split') and before_route == 'split':
                receipt['route_after'] = 'full'
            receipt['barrier_replies'] = self._barriers(
                before_route, before_generation)
            invalidation = {'prefill': 0, 'mixed': 0, 'unsupported': 0,
                            'preempted': 0}
            if fault.get('interval_prefill'):
                invalidation['prefill'] = 1
            if isinstance(fault.get('invalidation'), dict):
                invalidation.update(fault['invalidation'])
            if not fault.get('omit_interval'):
                receipt['interval'] = {
                    'observer_basis': fault.get('observer_basis', self.free_text),
                    'counters_before': [{'rank': rank, 'decode_steps': before_steps,
                                         'counters': {'replays': fault.get('replays_before', 0)}}
                                        for rank in driver.EXPECTED_RANKS],
                    'counters_after': [{'rank': rank, 'decode_steps': after_steps,
                                        'counters': {'replays': fault.get('replays', 3)}}
                                       for rank in driver.EXPECTED_RANKS],
                    'invalidation': invalidation,
                    'invalidation_complete': {key: True for key in invalidation},
                    'invalidation_lower_bounds': [],
                    'invalidation_basis': 'all fixture invalidation counters complete',
                }
            if not fault.get('omit_worker_observations'):
                receipt['worker_observations'] = self._observations(
                    before_route, before_generation, fault)
            route_after = before_route
        elif action == 'leave_full':
            if before_route == 'split':
                route_after = 'full'
                generation_after = before_generation + 1
                ranks_transition = True
            receipt['barrier_replies'] = []
        else:
            raise AssertionError('unknown action ' + action)
        if fault.get('drop_receipt_rank') and action == 'pause_end':
            pass  # handled in ranks below
        receipt['route_after'] = route_after
        receipt['route_generation_after'] = generation_after
        receipt['ranks'] = self._ranks(route_after, generation_after,
                                       ranks_transition, fault)
        if fault.get('stale_seq'):
            receipt['command_seq'] = command['command_seq'] - 1
        self.seq = command['command_seq']
        self.prior = {'command_seq': self.seq,
                      'receipt_sha256': driver._sha256_json(driver._receipt_core(receipt))}
        self.route = route_after
        self.generation = generation_after
        return receipt, {'kind': 'injected', 'rc': 0, 'seconds': 0.0001,
                         'stdout_len': 0, 'stdout_sha256': None,
                         'stderr_len': 0, 'timeout': False}


class ClientHarness(unittest.TestCase):
    streams = 3
    manifest_overrides = {}
    flow_options = {}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.use_manifest(base_manifest(streams=self.streams,
                                        **self.manifest_overrides))
        self.source = {
            'image_sha256': client.API_IMAGE_SHA256,
            'provenance': 'reference/PROVENANCE-completion-api.txt',
            'facts': dict(client.PROTOCOL_FACTS),
            'files': {}, 'available': True, 'verified': True, 'unavailable': [],
        }

    def use_manifest(self, manifest):
        self.manifest = manifest
        self.workload = workload_for(manifest)
        self.gate = PhaseGate()
        self.flow = Flow(self.workload, self.gate, **self.flow_options)
        self.flow.configure(**self.extra_flow_plans())
        self.server = FakeServer(self.flow)
        self.addCleanup(self.server.close)
        self.controller = FakeController(self.gate, self.workload.streams)

    def extra_flow_plans(self):
        return {}

    def config(self, manifest=None, run_id='test-run'):
        workload = self.workload if manifest is None else workload_for(manifest)
        return client.Config(workload=workload, run_id=run_id,
                             out_path=str(Path(self.tmp.name) / 'out.json'))

    def run_client(self, controller=None, manifest=None, cfg=None):
        cfg = cfg or self.config(manifest)
        return client.run(cfg, self.server.url, MODEL,
                          controller or self.controller, source=self.source)

    def assert_no_contract(self, data):
        self.assertFalse([reason for reason in data['verdict_reasons']
                          if reason.startswith('contract:')],
                         data['verdict_reasons'])

    def actions(self, record):
        return [call['action'] for call in record['calls']]


class PreSplitControlTests(ClientHarness):
    manifest_overrides = {'route_plan': ['full', 'full', 'split', 'full']}

    def test_fixed_four_runs_and_both_controls_gate_split(self):
        data = self.run_client()
        self.assertEqual(data['verdict'], 'PASS', data['verdict_reasons'])
        self.assertEqual(data['route_plan'], ['full', 'full', 'split', 'full'])
        self.assertEqual([r['label'] for r in data['runs']],
                         ['full-A1', 'full-A2', 'split-B', 'full-C'])
        self.assertTrue(data['pre_split_controls_before_transition']['equal'])
        for key in ('pre_split_controls', 'post_split_controls', 'split'):
            self.assertTrue(data['comparison'][key]['equal'])
        self.assertEqual(self.flow.request_count, 4 * self.workload.streams)
        for record, pause in zip(data['runs'], ('pause_full', 'pause_full',
                                                'pause_enter_split', 'pause_full')):
            self.assertEqual(self.actions(record), [
                'prepare', pause, 'resume', 'pause_end', 'leave_full', 'resume'])

    def test_pre_split_instability_does_not_hide_placement_or_choose_other_control(self):
        self.flow.configure(mutate=lambda run, stream, pos, token:
                            token + 7 if (run == 1 and stream == 0 and pos == 2)
                            else token)
        data = self.run_client()
        self.assertEqual(data['verdict'], 'INCONCLUSIVE')
        self.assertIn('inconclusive:pre_split_controls_differ', data['verdict_reasons'])
        self.assertFalse(data['pre_split_controls_before_transition']['equal'])
        self.assertTrue(data['comparison']['post_split_controls']['equal'])
        self.assertFalse(data['comparison']['split']['used_for_verdict'])
        self.assertTrue(data['runs'][2]['complete'])
        self.assertEqual(self.flow.request_count, 4 * self.workload.streams)

    def test_post_control_must_still_agree(self):
        self.flow.configure(mutate=lambda run, stream, pos, token:
                            token + 7 if (run == 3 and stream == 0 and pos == 2)
                            else token)
        data = self.run_client()
        self.assertEqual(data['verdict'], 'INCONCLUSIVE')
        self.assertIn('inconclusive:post_split_controls_differ', data['verdict_reasons'])
        self.assertTrue(data['pre_split_controls_before_transition']['equal'])
        self.assertFalse(data['comparison']['split']['used_for_verdict'])

    def test_split_only_difference_with_two_stable_controls_is_failure(self):
        self.flow.configure(mutate=lambda run, stream, pos, token:
                            token + 7 if (run == 2 and stream == 0 and pos == 2)
                            else token)
        data = self.run_client()
        self.assertEqual(data['verdict'], 'FAIL')
        self.assertIn('contract:split_trajectory_differs', data['verdict_reasons'])


class NormalPassTests(ClientHarness):
    def test_exact_three_run_match_passes_and_action_order_is_explicit(self):
        data = self.run_client()
        self.assertEqual(data['verdict'], 'PASS', data['verdict_reasons'])
        self.assertEqual([run['route'] for run in data['runs']],
                         ['full', 'split', 'full'])
        self.assertEqual([run['label'] for run in data['runs']],
                         ['full-A', 'split-B', 'full-C'])
        for record, pause in zip(data['runs'],
                                 ('pause_full', 'pause_enter_split', 'pause_full')):
            self.assertEqual(self.actions(record), [
                'prepare', pause, 'resume', 'pause_end', 'leave_full', 'resume'])
            self.assertTrue(record['complete'])
            self.assertTrue(record['cleanup']['leave_ok'])
            self.assertTrue(record['cleanup']['resume_ok'])
            self.assertTrue(record['cleanup']['restore_before_drain'])
            self.assertIsNone(record['cleanup']['reason'])
        for run in data['runs']:
            for row in run['streams']:
                self.assertEqual(row['length'], self.workload.generation_budget)
                self.assertEqual(row['finish_reason'], 'length')
                self.assertEqual(row['usage_completion'], row['length'])
                self.assertTrue(row['reader_joined'])
                self.assertEqual(row['problems'], [])
        self.assertEqual(self.flow.request_count,
                         3 * self.workload.streams)
        self.assertEqual(data['comparison']['controls']['equal'], True)
        self.assertEqual(data['comparison']['split']['equal'], True)
        self.assertEqual(data['comparison']['split']['used_for_verdict'], True)

    def test_performance_is_separate_and_has_no_threshold(self):
        data = self.run_client()
        self.assertEqual(data['verdict'], 'PASS')
        for run in data['runs']:
            performance = run['performance']
            self.assertTrue(performance['reported_separately'])
            self.assertTrue(performance['no_speed_threshold'])
            self.assertTrue(performance['verdict_independent'])
            self.assertIn('window_wall_s', performance)
        self.assertFalse(any('performance' in reason
                             for reason in data['verdict_reasons']))

    def test_same_live_requests_no_new_request_during_pause_and_fresh_salts(self):
        data = self.run_client()
        self.assertEqual(data['verdict'], 'PASS')
        requests = self.flow.requests
        self.assertEqual(len(requests), 3 * self.workload.streams)
        for row in requests:
            self.assertEqual(row['n'], 1)
            self.assertIs(row['return_token_ids'], True)
            self.assertIs(row['stream'], True)
            self.assertEqual(row['stream_options'], {
                'include_usage': True, 'continuous_usage_stats': True})
            self.assertEqual(row['max_tokens'], self.workload.generation_budget)
            self.assertEqual(row['min_tokens'], self.workload.generation_budget)
        for run_index, record in enumerate(data['runs']):
            pause = record['calls'][1]
            resume = record['calls'][2]
            for row in requests:
                if row['run'] != run_index:
                    continue
                self.assertFalse(pause['issued_at_ns'] < row['at_ns']
                                 < resume['returned_at_ns'],
                                 'a request was opened inside the paused window')
        by_run = {}
        for row in requests:
            by_run.setdefault(row['run'], set()).add(row['cache_salt_sha256'])
        self.assertEqual(len(by_run), 3)
        salts = [next(iter(values)) for values in by_run.values()]
        self.assertEqual(len(set(salts)), 3)
        prompt_sets = []
        for run in data['runs']:
            prompt_sets.append({row['prompt_sha256'] for row in run['requests']})
        self.assertEqual(prompt_sets[0], prompt_sets[1])
        self.assertEqual(prompt_sets[1], prompt_sets[2])

    def test_out_of_order_readiness_waits_for_every_request(self):
        self.flow.configure(start_delay=lambda run, stream:
                            0.12 if (run == 0 and stream == 0) else 0.0)
        data = self.run_client()
        self.assertEqual(data['verdict'], 'PASS', data['verdict_reasons'])
        opened = data['runs'][0]['open']
        self.assertEqual(opened['opened_streams'], self.workload.streams)
        self.assertNotEqual(opened['advance_order'][0], 0)
        sealed = opened['sealed_at_ns']
        for row in opened['streams']:
            self.assertLessEqual(row['first_advance_at_ns'], sealed)
        self.assertEqual(opened['submissions_after_seal'], 0)

    def test_usage_only_and_duplicate_usage_do_not_invent_tokens(self):
        self.flow.configure(usage_first=lambda run, stream: stream == 0,
                            duplicate_usage_after=lambda run, stream:
                            1 if stream == 1 else None)
        data = self.run_client()
        self.assertEqual(data['verdict'], 'PASS', data['verdict_reasons'])
        first = data['runs'][0]['streams'][0]
        second = data['runs'][0]['streams'][1]
        self.assertGreaterEqual(first['usage_only_chunks'], 1)
        self.assertGreaterEqual(second['duplicate_usage_chunks'], 1)
        for row in data['runs'][0]['streams']:
            self.assertEqual(row['length'], self.workload.generation_budget)

    def test_fragmented_and_coalesced_events_are_one_arrival_each(self):
        self.flow.configure(fragment=lambda run, stream: stream == 0,
                            chunk_size=lambda run, stream: 2 if stream == 1 else 1)
        data = self.run_client()
        self.assertEqual(data['verdict'], 'PASS', data['verdict_reasons'])
        rows = {row['stream']: row for row in data['runs'][0]['streams']}
        self.assertGreaterEqual(rows[1]['coalesced_arrivals'], 1)
        self.assertLess(rows[1]['arrivals'], rows[1]['length'])
        self.assertEqual(rows[0]['length'], self.workload.generation_budget)

    def test_content_only_events_do_not_open_a_stream(self):
        manifest = base_manifest(streams=1, prompt={'seed': 17, 'tokens': 24},
                                 timeouts={'open_s': 0.25})
        self.use_manifest(manifest)
        self.flow.configure(content_first=lambda run, stream: True,
                            start_delay=lambda run, stream: 0.5)
        data = self.run_client()
        self.assertEqual(data['verdict'], 'INCONCLUSIVE')
        self.assertIn('inconclusive:open_incomplete:stream_not_advanced:0',
                      data['verdict_reasons'])
        first = data['runs'][0]['streams'][0]
        self.assertEqual(first['content_only_chunks'], 1)
        self.assertEqual(first['length'], 0)

    def test_artifact_receipt_hashes_are_recomputable_and_chained(self):
        data = self.run_client()
        self.assertEqual(data['verdict'], 'PASS')
        previous = None
        for record in data['runs']:
            for call in record['calls']:
                receipt = call.get('receipt')
                if receipt is not None:
                    self.assertEqual(client._sha256_json(receipt),
                                     call['retained_receipt_sha256'])
                    self.assertEqual(client._sha256_json(call['receipt_core']),
                                     call['receipt_sha256'])
                if previous is None:
                    self.assertIsNone(call['prior_receipt'])
                else:
                    self.assertEqual(call['prior_receipt'], previous)
                if call['ok']:
                    previous = {'command_seq': call['command_seq'],
                                'receipt_sha256': call['receipt_sha256']}


class StreamContractTests(ClientHarness):
    def _one(self, **plans):
        manifest = base_manifest(streams=1, prompt={'seed': 17, 'tokens': 24})
        self.use_manifest(manifest)
        self.flow.configure(**plans)
        return self.run_client()

    def test_malformed_token_ids_are_contract_failures(self):
        for plans in (
                {'malformed': lambda run, stream: ('replace', 0)},
                {'malformed': lambda run, stream: ('negative', 0)},
                {'malformed': lambda run, stream: ('not_int', 0)},
                {'missing_id': lambda run, stream: 0},
                {'not_list': lambda run, stream: True}):
            with self.subTest(plans=list(plans)):
                data = self._one(**plans)
                self.assertEqual(data['verdict'], 'FAIL', data['verdict_reasons'])
                self.assertTrue(any(reason.startswith('contract:')
                                    for reason in data['verdict_reasons']))

    def test_wrong_length_and_finish_reason_are_contract_failures(self):
        cases = (
            {'finish_at': lambda run, stream: 15},
            {'finish_reason': lambda run, stream: 'stop'},
            {'usage_final': lambda run, stream: 15},
        )
        for case in cases:
            with self.subTest(case=list(case)):
                data = self._one(**case)
                self.assertEqual(data['verdict'], 'FAIL', data['verdict_reasons'])
                reasons = ' '.join(data['verdict_reasons'])
                self.assertTrue(any(code in reasons for code in
                                    ('length_mismatch', 'finish_mismatch',
                                     'usage_token_mismatch', 'early_eos')),
                                reasons)

    def test_early_eos_is_rejected_before_the_declared_budget(self):
        data = self._one(finish_at=lambda run, stream: 4,
                         finish_reason=lambda run, stream: 'stop')
        self.assertEqual(data['verdict'], 'FAIL', data['verdict_reasons'])
        self.assertTrue(any('early_eos' in reason
                            for reason in data['verdict_reasons']),
                        data['verdict_reasons'])


class OracleTests(ClientHarness):
    def test_stable_controls_with_one_split_token_wrong_fail_with_metadata(self):
        self.flow.configure(mutate=lambda run, stream, pos, token:
                            token + 7 if (run == 1 and stream == 0 and pos == 2)
                            else token)
        data = self.run_client()
        self.assertEqual(data['verdict'], 'FAIL', data['verdict_reasons'])
        self.assertIn('contract:split_trajectory_differs', data['verdict_reasons'])
        self.assertEqual(data['comparison']['controls']['equal'], True)
        split = data['comparison']['split']
        self.assertEqual(split['equal'], False)
        self.assertEqual(len(split['mismatches']), 1)
        row = split['mismatches'][0]
        self.assertEqual(row['stream'], 0)
        self.assertEqual(row['differing_positions'], 1)
        self.assertEqual(row['first_position'], 2)
        self.assertIn('a_sha256', row)
        self.assertNotIn('tokens', row)
        self.assertNotIn('token_ids', row)

    def test_unstable_full_controls_are_inconclusive_not_model_noise(self):
        self.flow.configure(mutate=lambda run, stream, pos, token:
                            token + 7 if (run == 0 and stream == 0 and pos == 1)
                            else token)
        data = self.run_client()
        self.assertEqual(data['verdict'], 'INCONCLUSIVE', data['verdict_reasons'])
        self.assertIn('inconclusive:control_instability', data['verdict_reasons'])
        self.assertEqual(data['comparison']['controls']['equal'], False)
        split = data['comparison']['split']
        self.assertFalse(split.get('used_for_verdict'))
        self.assertTrue(split.get('equal') is None or 'equal' not in split)

    def test_zero_split_replay_fails_but_missing_replay_is_inconclusive(self):
        self.controller.faults[('pause_end', 1)] = {'replays': 0}
        data = self.run_client()
        self.assertEqual(data['verdict'], 'FAIL', data['verdict_reasons'])
        self.assertTrue(any(reason.startswith('contract:split_replay_zero')
                            for reason in data['verdict_reasons']),
                        data['verdict_reasons'])
        self.assertTrue(data['runs'][1]['cleanup']['restore_before_drain'])

        self.setUp()
        self.controller.faults[('pause_end', 1)] = {'replays': None}
        data = self.run_client()
        self.assertEqual(data['verdict'], 'INCONCLUSIVE', data['verdict_reasons'])
        self.assertIn('inconclusive:split_replay_unproven',
                      ' '.join(data['verdict_reasons']))
        self.assertTrue(data['runs'][1]['cleanup']['restore_before_drain'])

    def test_partial_worker_and_receipt_rank_coverage_refuse_cleanup(self):
        self.controller.faults[('pause_end', 1)] = {'drop_worker_rank': True}
        data = self.run_client()
        self.assertEqual(data['verdict'], 'FAIL', data['verdict_reasons'])
        self.assertIn('contract:worker_observation_rank_coverage',
                      data['verdict_reasons'])
        self.assertNotIn('leave_full', self.actions(data['runs'][1]))

        self.setUp()
        self.controller.faults[('pause_end', 1)] = {'drop_receipt_rank': True}
        data = self.run_client()
        self.assertEqual(data['verdict'], 'FAIL', data['verdict_reasons'])
        self.assertFalse(data['runs'][1]['cleanup']['attempted'])
        self.assertNotIn('leave_full', self.actions(data['runs'][1]))

    def test_worker_route_mismatch_and_off_route_interval_are_contract(self):
        self.controller.faults[('pause_end', 1)] = {'worker_route': 'full'}
        data = self.run_client()
        self.assertEqual(data['verdict'], 'FAIL', data['verdict_reasons'])
        self.assertTrue(any('worker_route_mismatch' in reason
                            for reason in data['verdict_reasons']))
        self.assertFalse(data['runs'][1]['cleanup']['restore_before_drain'])
        self.assertNotIn('leave_full', self.actions(data['runs'][1]))

    def test_old_split_replays_cannot_qualify_this_window(self):
        self.controller.faults[('pause_end', 1)] = {'replays_before': 7, 'replays': 7}
        data = self.run_client()
        self.assertEqual(data['verdict'], 'FAIL')
        self.assertTrue(any('split_replay_zero' in r for r in data['verdict_reasons']))
        self.assertTrue(data['runs'][1]['cleanup']['restore_before_drain'])


class TransitionAndCleanupTests(ClientHarness):
    def test_invalid_interval_does_not_override_pending_transition(self):
        self.controller.faults[('pause_end', 1)] = {
            'transition_pending': True, 'interval_prefill': True}
        data = self.run_client()
        self.assertEqual(data['verdict'], 'FAIL')
        split = next(row for row in data['runs'] if row['route'] == 'split')
        self.assertFalse(split['cleanup']['attempted'])
        self.assertNotIn('leave_full', self.actions(split))
        self.assertTrue(data['runs'][2]['not_run'])

    def test_unproven_reader_join_blocks_the_next_cohort(self):
        original = client.StreamRecord.cancel
        def unproven(state, timeout):
            original(state, timeout)
            return False
        with mock.patch.object(client.StreamRecord, 'cancel', unproven):
            data = self.run_client()
        self.assertEqual(data['verdict'], 'INCONCLUSIVE')
        self.assertTrue(data['runs'][0]['abort_run'])
        self.assertTrue(data['runs'][0]['reader_join_timeouts'])
        self.assertTrue(data['runs'][1]['not_run'])
        self.assertTrue(data['runs'][2]['not_run'])

    def test_stale_transition_ack_fails_and_never_resumes_or_leaves(self):
        self.controller.faults[('pause_enter_split', 1)] = {'stale_ack': True}
        data = self.run_client()
        self.assertEqual(data['verdict'], 'FAIL', data['verdict_reasons'])
        record = data['runs'][1]
        self.assertEqual(record['status'], 'pause_failed')
        self.assertFalse(record['cleanup']['attempted'])
        self.assertNotIn('resume', self.actions(record))
        self.assertNotIn('leave_full', self.actions(record))
        self.assertEqual(data['runs'][2]['status'], 'not_run')

    def test_partial_transition_proof_is_inconclusive_without_resume(self):
        self.controller.faults[('pause_enter_split', 1)] = {'partial_acks': True}
        data = self.run_client()
        self.assertEqual(data['verdict'], 'INCONCLUSIVE', data['verdict_reasons'])
        record = data['runs'][1]
        self.assertFalse(record['cleanup']['attempted'])
        self.assertNotIn('leave_full', self.actions(record))
        self.assertNotIn('resume', self.actions(record))

    def test_timed_out_pause_is_inconclusive_and_never_cleans_up(self):
        self.controller.faults['pause_full'] = {'timeout': True}
        data = self.run_client()
        self.assertEqual(data['verdict'], 'INCONCLUSIVE', data['verdict_reasons'])
        record = data['runs'][0]
        self.assertEqual(record['status'], 'pause_failed')
        self.assertFalse(record['cleanup']['attempted'])
        self.assertNotIn('leave_full', self.actions(record))
        self.assertNotIn('resume', self.actions(record))

    def test_incomplete_pause_blocks_leave_full(self):
        self.controller.faults['pause_end'] = {'pause_not_completed': True}
        data = self.run_client()
        self.assertEqual(data['verdict'], 'INCONCLUSIVE', data['verdict_reasons'])
        record = data['runs'][0]
        self.assertFalse(record['cleanup']['attempted'])
        self.assertNotIn('leave_full', self.actions(record))

    def test_interval_prefill_invalidates_measurement_but_still_restores_full(self):
        self.controller.faults[('pause_end', 1)] = {'interval_prefill': True}
        data = self.run_client()
        self.assertEqual(data['verdict'], 'FAIL', data['verdict_reasons'])
        self.assertTrue(any('interval_invalidated:prefill' in reason
                            for reason in data['verdict_reasons']))
        record = data['runs'][1]
        self.assertTrue(record['cleanup']['restore_before_drain'],
                        (record['cleanup'], record['reasons'],
                         record['calls'][3].get('reasons'),
                         record['calls'][4].get('reasons'),
                         record['calls'][4].get('receipt')))
        self.assertIn('leave_full', self.actions(record))
        self.assertEqual(record['calls'][-1]['action'], 'resume')

    def test_each_interval_invalidation_kind_is_contract_and_cleanup_safe(self):
        for key in ('mixed', 'preempted', 'unsupported'):
            with self.subTest(key=key):
                self.setUp()
                self.controller.faults[('pause_end', 1)] = {
                    'invalidation': {key: 1}}
                data = self.run_client()
                self.assertEqual(data['verdict'], 'FAIL', data['verdict_reasons'])
                self.assertTrue(any(('interval_invalidated:' + key) in reason
                                    for reason in data['verdict_reasons']),
                                data['verdict_reasons'])
                self.assertTrue(
                    data['runs'][1]['cleanup']['restore_before_drain'])

    def test_missing_pause_end_receipt_blocks_cleanup(self):
        self.controller.faults[('pause_end', 0)] = {'omit_receipt': True}
        data = self.run_client()
        # The reviewed W1 validator calls a non-object receipt a contract
        # contradiction; FAIL must not be diluted by cleanup uncertainty.
        self.assertEqual(data['verdict'], 'FAIL', data['verdict_reasons'])
        self.assertIn('contract:receipt_not_object',
                      ' '.join(data['verdict_reasons']))
        record = data['runs'][0]
        self.assertFalse(record['cleanup']['attempted'])
        self.assertNotIn('leave_full', self.actions(record))
        self.assertNotIn('resume', self.actions(record)[3:])

    def test_missing_interval_progress_is_inconclusive_but_cleanup_is_safe(self):
        self.controller.faults[('pause_end', 1)] = {'interval_no_progress': True}
        data = self.run_client()
        self.assertEqual(data['verdict'], 'INCONCLUSIVE', data['verdict_reasons'])
        self.assertTrue(any('interval_decode_progress_missing' in reason
                            for reason in data['verdict_reasons']))
        record = data['runs'][1]
        self.assertTrue(record['cleanup']['restore_before_drain'])
        self.assertIn('leave_full', self.actions(record))

    def test_changed_incarnation_fails_and_blocks_cleanup(self):
        self.controller.faults[('pause_end', 1)] = {'engine_changed': True}
        data = self.run_client()
        self.assertEqual(data['verdict'], 'FAIL', data['verdict_reasons'])
        self.assertTrue(any('engine_incarnation_changed' in reason
                            for reason in data['verdict_reasons']))
        record = data['runs'][1]
        self.assertFalse(record['cleanup']['attempted'])
        self.assertNotIn('leave_full', self.actions(record))

    def test_unexpected_exception_still_cancels_and_preserves_partial_evidence(self):
        def boom(states, starts, workload):
            raise RuntimeError('injected offline failure')

        with mock.patch.object(client, '_wait_window', boom):
            data = self.run_client()
        self.assertEqual(data['verdict'], 'INCONCLUSIVE', data['verdict_reasons'])
        record = data['runs'][0]
        self.assertEqual(record['status'], 'exception')
        self.assertIn('inconclusive:run_exception:RuntimeError', record['reasons'])
        self.assertTrue(record['calls'])
        self.assertEqual(record['readers_alive_after_join'], [])
        self.assertEqual(record['streams'][0]['reader_joined'], True)
        self.assertEqual(data['runs'][1]['status'], 'not_run')

    def test_silent_paused_sse_is_actively_cancelled_within_bound(self):
        manifest = base_manifest(streams=2, bounds={'window_s': 0.2},
                                 timeouts={'read_s': 5.0, 'join_s': 0.3,
                                           'finish_s': 0.5})
        self.use_manifest(manifest)
        self.flow.delay_s = 0.03
        self.flow.configure(silent_after=lambda run, stream:
                            2 if (run == 0 and stream == 0) else None)
        self.controller.faults[('pause_end', 0)] = {'timeout': True}
        started = time.monotonic()
        data = self.run_client()
        elapsed = time.monotonic() - started
        self.assertEqual(data['verdict'], 'INCONCLUSIVE', data['verdict_reasons'])
        self.assertLess(elapsed, 1.5)
        record = data['runs'][0]
        self.assertEqual(record['readers_alive_after_join'], [])
        self.assertFalse(record['cleanup']['attempted'])
        self.assertNotIn('leave_full', self.actions(record))
        # Only the measurement resume precedes pause_end; no cleanup resume.
        self.assertEqual(record['calls'][-1]['action'], 'pause_end')
        self.assertEqual(self.actions(record).count('resume'), 1)


class ConfigAndArtifactTests(ClientHarness):
    def test_manifest_strictness_rejects_missing_budget_window_and_route_plan(self):
        valid = base_manifest()
        self.assertEqual(client.validate_manifest(valid)[0], [])
        cases = []
        for change in (
                {'generation_budget': 0},
                {'bounds': {'window_tokens': 16}},
                {'route_plan': ['full', 'full', 'full']},
                {'sampling': {'min_tokens': 1}},
                {'expected': {'finish_reason': 'stop'}},
                {'cache': {'fresh_namespace_per_run': False}}):
            data = base_manifest()
            deep_update(data, change)
            cases.append(data)
        empty = base_manifest()
        empty.pop('sampling')
        cases.append(empty)
        for data in cases:
            problems, workload, _ = client.validate_manifest(data)
            self.assertNotEqual(problems, [])
            self.assertIsNone(workload)

    def test_plan_only_without_manifest_issues_no_requests(self):
        path = Path(self.tmp.name) / 'plan.json'
        code = client.main([self.server.url, MODEL, str(path)], env={})
        self.assertEqual(code, 2)
        data = json.loads(path.read_text())
        self.assertEqual(data['verdict'], 'INCONCLUSIVE')
        self.assertEqual(data['runs'], [])
        self.assertEqual(self.flow.request_count, 0)

    def test_source_hash_mismatch_is_plan_only_before_any_request(self):
        source_dir = Path(self.tmp.name) / 'src'
        target = source_dir / 'entrypoints' / 'openai' / 'completion'
        target.mkdir(parents=True)
        (target / 'protocol.py').write_text('tampered\n')
        path = Path(self.tmp.name) / 'source.json'
        env = {'API_SOURCE_DIR': str(source_dir)}
        code = client.main([self.server.url, MODEL, str(path)], env=env)
        self.assertEqual(code, 2)
        data = json.loads(path.read_text())
        self.assertEqual(data['verdict'], 'INCONCLUSIVE')
        self.assertTrue(any('api_source_hash_mismatch' in reason
                            for reason in data['verdict_reasons']))
        self.assertEqual(self.flow.request_count, 0)

    def test_result_collision_is_refused_before_any_request(self):
        path = Path(self.tmp.name) / 'existing.json'
        path.write_text('KEEP')
        before = path.read_text()
        code = client.main([self.server.url, MODEL, str(path)], env={})
        self.assertEqual(code, 2)
        self.assertEqual(path.read_text(), before)
        self.assertEqual(self.flow.request_count, 0)

    def test_exclusive_writer_refuses_second_write(self):
        path = Path(self.tmp.name) / 'exclusive.json'
        client.write_output(str(path), {'first': 1})
        with self.assertRaises(FileExistsError):
            client.write_output(str(path), {'second': 2})
        self.assertEqual(json.loads(path.read_text()), {'first': 1})

    def test_exit_codes_are_zero_pass_two_inconclusive_one_fail(self):
        manifest_path = Path(self.tmp.name) / 'manifest.json'
        manifest_path.write_text(json.dumps(base_manifest()))
        env = {'WORKLOAD': str(manifest_path), 'CONTROLLER_CMD': 'unused'}
        out = Path(self.tmp.name) / 'out.json'

        code = client.main([self.server.url, MODEL, str(out)],
                           controller=self.controller, env=env)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.read_text())['verdict'], 'PASS')

        self.setUp()
        self.flow.configure(mutate=lambda run, stream, pos, token:
                            token + 1 if (run == 1 and stream == 0 and pos == 2)
                            else token)
        out = Path(self.tmp.name) / 'out-fail.json'
        code = client.main([self.server.url, MODEL, str(out)],
                           controller=self.controller, env=env)
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(out.read_text())['verdict'], 'FAIL')

        self.setUp()
        self.controller.faults[('pause_end', 1)] = {'replays': None}
        out = Path(self.tmp.name) / 'out-inconclusive.json'
        code = client.main([self.server.url, MODEL, str(out)],
                           controller=self.controller, env=env)
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(out.read_text())['verdict'], 'INCONCLUSIVE')

    def test_pinned_completion_source_facts_and_hashes_are_verified(self):
        source, problems = client.verify_sources({})
        if not source['available']:
            self.skipTest('pinned reference tree is not present')
        self.assertEqual(problems, [])
        self.assertTrue(source['verified'])
        self.assertEqual(source['image_sha256'], client.API_IMAGE_SHA256)
        self.assertEqual(source['facts']['request_field'], 'return_token_ids')
        self.assertEqual(source['facts']['stream_field'], 'choices[].token_ids')
        self.assertIs(source['facts']['encoded_token_parser'], False)

    def test_unverified_pinned_source_cannot_produce_pass(self):
        self.source['verified'] = False
        self.source['available'] = False
        data = self.run_client()
        self.assertEqual(data['verdict'], 'INCONCLUSIVE', data['verdict_reasons'])
        self.assertIn('inconclusive:api_source_unverified',
                      data['verdict_reasons'])
        self.assertFalse(data['api_source']['verified'])

    def test_secret_payload_and_prompt_and_token_arrays_never_persisted(self):
        os.environ['VLLM_API_KEY'] = SECRET
        self.addCleanup(os.environ.pop, 'VLLM_API_KEY', None)
        self.controller.free_text = 'clock ' + SECRET
        self.controller.faults[('pause_end', 0)] = {
            'clock_basis': SECRET,
            'eligibility_reason': SECRET,
            'observer_basis': SECRET,
        }
        data = self.run_client()
        self.assertEqual(data['verdict'], 'PASS', data['verdict_reasons'])
        artifact = json.dumps(data, sort_keys=True)
        path = Path(self.tmp.name) / 'secret.json'
        client.write_output(str(path), data)
        persisted = path.read_text()
        for forbidden in (SECRET, 'Authorization', 'sk-',
                          '"token_ids"', 'Bearer'):
            self.assertNotIn(forbidden, persisted)
        prompt = self.flow.cohort[0]['prompt']
        self.assertNotIn(prompt[:40], persisted)
        tokens = [100000, 100001, 100002]
        self.assertNotIn(json.dumps(tokens), persisted)
        self.assertIn('clock_basis_sha256', persisted)
        self.assertIn('reason_sha256', persisted)
        # The raw payload still exists only in memory for the oracle.
        self.assertIsInstance(data, dict)
        self.assertNotIn(SECRET, artifact)


class ComposedActualSessionTests(ClientHarness):
    """The client's own transport adapter over the real ProbeSession."""

    streams = 8
    manifest_overrides = {'prompt': {'seed': 17, 'tokens': 16}}

    class SessionController:
        def __init__(self, engine, streams, gate):
            self.engine = engine
            self.streams = streams
            self.gate = gate
            self.loop = asyncio.new_event_loop()
            self.thread = threading.Thread(target=self.loop.run_forever,
                                           daemon=True)
            self.thread.start()
            self.session = None

        def start(self):
            from probe_session import ProbeSession
            from test_e3_probe_control import ProbeController
            self.session = ProbeSession(ProbeController(self.engine, epoch=71))

        def execute(self, command, timeout_s):
            action = command['action']
            phase_before = self.session.phase
            if action == 'prepare':
                self.engine.unfinished = 0
                self.gate.reset()
            elif action == 'pause_end':
                self.engine.steps += 5
                self.engine.unfinished = self.streams
            elif action == 'resume':
                self.engine.unfinished = (0 if phase_before == 'left'
                                          else self.streams)
            elif action == 'leave_full':
                self.engine.unfinished = 0
            elif action in ('pause_enter_split', 'pause_full'):
                self.engine.unfinished = self.streams
            future = asyncio.run_coroutine_threadsafe(
                self.session.execute(command), self.loop)
            receipt = future.result(timeout=timeout_s)
            if action in ('pause_enter_split', 'pause_full'):
                self.gate.set_phase('entry')
            elif action == 'resume':
                self.gate.set_phase('cleanup' if phase_before == 'left'
                                    else 'measure')
            elif action == 'pause_end':
                self.gate.set_phase('end')
            return receipt, {'kind': 'injected',
                             'rc': 0, 'seconds': 0.0, 'stdout_len': 0,
                             'stdout_sha256': None, 'stderr_len': 0,
                             'timeout': False}

        def close(self):
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=3)
            self.loop.close()

    def setUp(self):
        super().setUp()
        # Positive composition fixtures test protocol sequencing, not whether
        # a host thread wins a race against a 16-token synthetic response.
        # Supply the measured prefix, then retain a live tail until pause/end.
        # General fixtures keep the uncapped stream for early-finish negatives.
        self.flow.measure_token_cap = (self.flow.open_tokens
                                       + self.workload.window_tokens + 1)
        from test_e3_probe_session import CountedEngine

        class ReplayEngine(CountedEngine):
            async def collective_rpc(self, *args, **kwargs):
                rows = await super().collective_rpc(*args, **kwargs)
                if kwargs['args'][0]['action'] == 'status':
                    for row in rows:
                        row['counters']['replays'] = self.steps
                return rows

        self.engine = ReplayEngine()
        self.session_controller = self.SessionController(
            self.engine, self.workload.streams, self.gate)
        self.session_controller.start()
        self.addCleanup(self.session_controller.close)

    def extra_flow_plans(self):
        return {}

    def test_client_drives_actual_session_and_validator_to_pass(self):
        data = self.run_client(controller=self.session_controller)
        self.assertEqual(data['verdict'], 'PASS', data['verdict_reasons'])
        split = next(row for row in data['runs'] if row['route'] == 'split')
        self.assertEqual(split['route'], 'split')
        self.assertEqual(self.actions(split), [
            'prepare', 'pause_enter_split', 'resume', 'pause_end',
            'leave_full', 'resume'])
        self.assertTrue(split['cleanup']['restore_before_drain'])
        receipt = split['calls'][3]['receipt']
        self.assertIn('worker_observations', receipt)
        self.assertTrue(all(row['counters']['replays'] > 0
                            for row in receipt['worker_observations']))
        for row in split['streams']:
            self.assertEqual(row['length'], self.workload.generation_budget)
            self.assertEqual(row['finish_reason'], 'length')
        self.assertFalse(self.session_controller.session.failed)


class ComposedPreControlSessionTests(ComposedActualSessionTests):
    manifest_overrides = {'prompt': {'seed': 17, 'tokens': 16},
                          'route_plan': ['full', 'full', 'split', 'full']}


if __name__ == '__main__':
    unittest.main()
