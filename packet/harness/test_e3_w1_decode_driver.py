#!/usr/bin/env python3
"""Offline tests for e3_w1_decode_driver.

Everything runs on 127.0.0.1 with a fake SSE server and an injected controller;
no engine, GPU, queue or network peer is touched.  The fake controller records
every schema-1 command so the driver's action order and sealing can be asserted
from actual calls rather than from source strings.
"""
import copy
import hashlib
import importlib.util
import json
import re
import socket
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load_driver():
    spec = importlib.util.spec_from_file_location('e3_w1_decode_driver_for_test',
                                                  HERE / 'e3_w1_decode_driver.py')
    if spec is None or spec.loader is None:
        raise ImportError('e3_w1_decode_driver.py not found')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


D = _load_driver()

BASE_MANIFEST = {
    'schema': 1,
    'workload': 'test-small',
    'streams': 3,
    'prompt_tokens': 40,
    'seed': 1,
    'generation_budget': 160,
    'repeats': 2,
    'route_plan': ['full', 'split'],
    'request_shape': {
        'temperature': 0.0,
        'min_tokens': 'equal_max',
        'include_usage': True,
        'continuous_usage_stats': True,
        'enable_thinking': False,
    },
    'window_s': 0.08,
    'open_timeout_s': 0.6,
    'pause_timeout_s': 2.0,
    'cache_preparation': False,
    'completion_policy': {
        'per_request_delta_min': 1,
        'per_request_delta_max': 100000,
        'matched_across_routes': True,
    },
    'context_position_policy': 'same prompt, seed and context position per stream across routes',
    'block_size': 16,
}


def _deep_update(base, overrides):
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


class FakeConfig:
    def __init__(self, streams=3, prompt_tokens=100, cached_tokens=0,
                 interval=0.004, coalesce=1, start_delays=None,
                 never_start=(), http_error_streams=(), close_after=None,
                 early_eos_streams=(), early_eos_after=6,
                 finish_after=None, usage_only_every=0,
                 continuous=True, split_reads=False):
        self.streams = streams
        self.prompt_tokens = prompt_tokens
        self.cached_tokens = cached_tokens
        self.interval = interval
        self.coalesce = coalesce
        self.start_delays = dict(start_delays or {})
        self.never_start = set(never_start)
        self.http_error_streams = set(http_error_streams)
        self.close_after = dict(close_after or {})
        self.early_eos_streams = set(early_eos_streams)
        self.early_eos_after = early_eos_after
        self.finish_after = dict(finish_after or {})
        self.usage_only_every = usage_only_every
        self.continuous = continuous
        self.split_reads = split_reads


class FakeServer:
    def __init__(self, config):
        self.config = config
        self.requests = []
        self.lock = threading.Lock()
        self.httpd = ThreadingHTTPServer(('127.0.0.1', 0), _Handler)
        self.httpd.fake = self
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.url = 'http://127.0.0.1:%d' % self.httpd.server_address[1]

    def record(self, kind, stream, max_tokens, sha, t_ns):
        with self.lock:
            self.requests.append({'kind': kind, 'stream': stream,
                                  'max_tokens': max_tokens, 'sha256': sha,
                                  't_ns': t_ns})

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def measure_requests(self):
        with self.lock:
            return [row for row in self.requests if row['kind'] == 'measure']

    def prepare_requests(self):
        with self.lock:
            return [row for row in self.requests if row['kind'] == 'prepare']


class _Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *_args):
        pass

    def _gone(self):
        return getattr(self, '_client_gone', False)

    def _mark_gone(self):
        self._client_gone = True
        self.close_connection = True

    def _write(self, data):
        if self._gone():
            return False
        try:
            self.wfile.write(data)
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            self._mark_gone()
            return False

    def do_POST(self):
        fake = self.server.fake
        try:
            length = int(self.headers.get('Content-Length', '0'))
            raw = self.rfile.read(length)
            body = json.loads(raw)
            stream_match = re.search(r'coding agent (\d+)',
                                     body['messages'][0]['content'])
            stream = int(stream_match.group(1)) if stream_match else -1
            max_tokens = body.get('max_tokens')
            kind = 'prepare' if max_tokens == 1 else 'measure'
            sha = hashlib.sha256(raw).hexdigest()
            fake.record(kind, stream, max_tokens, sha, time.monotonic_ns())
        except Exception:  # noqa: BLE001 - malformed probe request
            self.send_response(400)
            self.send_header('Content-Length', '0')
            self.end_headers()
            return
        if kind == 'prepare':
            self._prepare(fake.config)
            return
        if stream in fake.config.http_error_streams:
            self.send_response(500)
            self.send_header('Content-Length', '0')
            self.send_header('Connection', 'close')
            self.end_headers()
            return
        self._measure(fake.config, stream, int(max_tokens or 0))

    def _headers(self):
        if self._gone():
            return False
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'close')
            self.end_headers()
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            self._mark_gone()
            return False

    def _raw(self, data):
        if self._gone():
            return False
        if self.server.fake.config.split_reads and len(data) > 8:
            middle = len(data) // 2
            if not self._write(data[:middle]):
                return False
            time.sleep(0.001)
            return self._write(data[middle:])
        return self._write(data)

    def _event(self, obj):
        return self._raw(('data: ' + json.dumps(obj) + '\n\n').encode())

    def _usage(self, cfg, completion):
        if not cfg.continuous:
            return None
        return {
            'prompt_tokens': cfg.prompt_tokens,
            'completion_tokens': completion,
            'total_tokens': cfg.prompt_tokens + completion,
            'prompt_tokens_details': {'cached_tokens': cfg.cached_tokens},
        }

    def _prepare(self, cfg):
        if not self._headers():
            return
        usage = {
            'prompt_tokens': cfg.prompt_tokens,
            'completion_tokens': 1,
            'total_tokens': cfg.prompt_tokens + 1,
            'prompt_tokens_details': {'cached_tokens': cfg.cached_tokens},
        }
        if not self._event({'id': 'prep',
                            'choices': [{'index': 0, 'delta': {'content': 'ok'},
                                         'finish_reason': None}],
                            'usage': usage}):
            return
        if not self._event({'choices': [{'index': 0, 'delta': {},
                                         'finish_reason': 'stop'}],
                            'usage': usage}):
            return
        self._raw(b'data: [DONE]\n\n')

    def _measure(self, cfg, stream, budget):
        if not self._headers():
            return
        delay = cfg.start_delays.get(stream, 0.0)
        if delay:
            time.sleep(delay)
        if stream in cfg.never_start:
            time.sleep(0.3)
            return
        cumulative = 0
        sent = 0
        while sent < budget:
            sent += 1
            cumulative += cfg.coalesce
            content = ' '.join('t%d' % (cumulative - offset)
                               for offset in range(cfg.coalesce))
            finish = None
            if stream in cfg.early_eos_streams and sent >= cfg.early_eos_after:
                finish = 'stop'
            if stream in cfg.finish_after and sent >= cfg.finish_after[stream]:
                finish = 'length'
            event = {'id': 'x',
                     'choices': [{'index': 0, 'delta': {'content': content},
                                  'finish_reason': finish}]}
            usage = self._usage(cfg, cumulative)
            if usage is not None:
                event['usage'] = usage
            if not self._event(event):
                return
            if cfg.usage_only_every and sent % cfg.usage_only_every == 0:
                if not self._event({'choices': [],
                                    'usage': usage or {'completion_tokens': cumulative}}):
                    return
            if finish:
                break
            if stream in cfg.close_after and sent >= cfg.close_after[stream]:
                return  # EOF without finish: a disconnect before the window ends
            time.sleep(cfg.interval)
        final_usage = self._usage(cfg, cumulative)
        if final_usage is not None:
            if not self._event({'choices': [], 'usage': final_usage}):
                return
        self._raw(b'data: [DONE]\n\n')


class FakeController(D.Controller):
    """Schema-1 controller with the same state machine as the reviewed protocol."""

    def __init__(self, streams=3, engine_epoch=11, storage_epoch=1,
                 generation=7, mode='valid', modes=None):
        self.streams = streams
        self.state = {'route': 'full', 'generation': generation,
                      'engine_epoch': engine_epoch,
                      'storage_epoch': storage_epoch}
        self.mode = mode
        self.modes = dict(modes or {})
        self.calls = []

    def _mode_for(self, action):
        spec = self.modes.get(action, self.mode)
        if isinstance(spec, list):
            return spec.pop(0) if spec else 'valid'
        return spec

    def execute(self, command, timeout_s):
        self.calls.append({'command': copy.deepcopy(command),
                           't_ns': time.monotonic_ns()})
        action = command['action']
        mode = self._mode_for(action)
        if mode == 'transport_timeout':
            raise D.ControllerTransportError(
                'controller_timeout',
                {'kind': 'injected', 'rc': None, 'timeout': True})
        if mode == 'transport_exception':
            raise RuntimeError('injected transport failure')
        before = dict(self.state)
        route_before = before['route']
        generation_before = before['generation']
        if action == 'prepare':
            route_after, generation_after = 'full', generation_before
        elif action == 'pause_enter_split':
            route_after, generation_after = 'split', generation_before + 1
        elif action == 'pause_full':
            route_after, generation_after = 'full', generation_before
        elif action in ('resume', 'pause_end'):
            route_after, generation_after = route_before, generation_before
        elif action == 'leave_full':
            if route_before == 'split':
                route_after, generation_after = 'full', generation_before + 1
            else:
                route_after, generation_after = 'full', generation_before
        else:
            raise AssertionError('unknown action ' + repr(action))

        transition = route_after != route_before
        ranks = []
        for rank in D.EXPECTED_RANKS:
            entry = {'rank': rank, 'status': 'ok'}
            if transition:
                entry['transition_ack'] = {
                    'rank': rank,
                    'ack': 'flash_s5_transition',
                    'mode': 'decode-only-probe',
                    'route': route_after,
                    'route_generation': generation_after,
                    'storage_epoch': before['storage_epoch'],
                    'engine_epoch': before['engine_epoch'],
                    'drained_streams': self.streams,
                }
            ranks.append(entry)
        receipt = {
            'schema': 1,
            'run_id': command['run_id'],
            'repeat': command['repeat'],
            'command_seq': command['command_seq'],
            'action': action,
            'cohort_id': command['cohort_id'],
            'status': 'ok',
            'engine_epoch': before['engine_epoch'],
            'storage_epoch': before['storage_epoch'],
            'route_before': route_before,
            'route_after': route_after,
            'route_generation_before': generation_before,
            'route_generation_after': generation_after,
            'expected_rank_set': list(D.EXPECTED_RANKS),
            'observed_clock_ns': time.monotonic_ns(),
            'clock_basis': 'fake-controller-monotonic-ns',
            'ranks': ranks,
        }
        if action in ('pause_enter_split', 'pause_full', 'pause_end'):
            receipt['pause'] = {'mode': 'keep', 'clear_cache': False,
                                'completed': True, 'drained': True}
        if action == 'resume':
            receipt['resume'] = {'completed': True}
        if action in ('pause_enter_split', 'pause_full'):
            receipt['decode_eligibility'] = {
                'cohort_id': command['cohort_id'],
                'cohort_cardinality': self.streams,
                'ranks': [{'rank': rank,
                           'batch_reqs': self.streams,
                           'scheduled_tokens': self.streams * 8,
                           'prefill': False}
                          for rank in D.EXPECTED_RANKS],
                'reason': 'fake worker observation: live cohort on every rank in a decode batch',
            }
        if action == 'pause_end':
            receipt['interval'] = {
                'observer_basis': 'fake controller per-rank batch counters',
                'counters_before': [{'rank': rank, 'decode_steps': 100}
                                    for rank in D.EXPECTED_RANKS],
                'counters_after': [{'rank': rank, 'decode_steps': 120}
                                   for rank in D.EXPECTED_RANKS],
                'invalidation': {'prefill': 0, 'mixed': 0,
                                 'unsupported': 0, 'preempted': 0},
            }
        self._mutate(receipt, mode, action, route_before, route_after)
        self.state['route'] = route_after
        self.state['generation'] = generation_after
        return receipt, {'kind': 'injected', 'rc': 0, 'seconds': 0.0}

    def _mutate(self, receipt, mode, action, route_before, route_after):
        if mode == 'valid':
            return
        if mode == 'missing_rank':
            receipt['ranks'] = receipt['ranks'][:-1]
        elif mode == 'partial':
            receipt.pop('ranks', None)
            receipt.pop('pause', None)
        elif mode == 'ack_false':
            for entry in receipt['ranks']:
                if 'transition_ack' in entry:
                    entry['transition_ack']['ack'] = True
        elif mode == 'pause_incomplete':
            receipt['pause']['completed'] = False
        elif mode == 'pause_timeout':
            receipt['status'] = 'timeout'
        elif mode == 'resume_missing':
            receipt.pop('resume', None)
        elif mode == 'wrong_run_id':
            receipt['run_id'] = 'other-run'
        elif mode == 'wrong_cohort':
            receipt['cohort_id'] = 'other-cohort'
        elif mode == 'stale_seq':
            receipt['command_seq'] = receipt['command_seq'] - 1
        elif mode == 'engine_changed':
            receipt['engine_epoch'] = receipt['engine_epoch'] + 1
        elif mode == 'storage_changed':
            receipt['storage_epoch'] = receipt['storage_epoch'] + 1
        elif mode == 'noop_generation_bump':
            if route_before == 'full' and route_after == 'full':
                receipt['route_generation_after'] += 1
        elif mode == 'route_skip':
            receipt['route_after'] = 'full'
        elif mode == 'counter_prefill':
            receipt['interval']['invalidation']['prefill'] = 1
        elif mode == 'counter_regress':
            for row in receipt['interval']['counters_after']:
                row['decode_steps'] = 1
        elif mode == 'malformed':
            receipt.clear()
        else:
            raise AssertionError('unknown fake mode ' + repr(mode))


class DriverTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.server = None

    def start_server(self, **overrides):
        config = FakeConfig(**overrides)
        self.server = FakeServer(config)
        self.addCleanup(self.server.stop)
        return self.server

    def write_manifest(self, **overrides):
        manifest = _deep_update(copy.deepcopy(BASE_MANIFEST), overrides)
        path = Path(self.temp.name) / ('manifest-%d.json' % len(
            list(Path(self.temp.name).glob('manifest-*.json'))))
        path.write_text(json.dumps(manifest))
        return path, manifest

    def run_driver(self, controller=None, server=None, manifest=None,
                   env_overrides=None, out_name='out.json', use_controller_env=False):
        server = server or self.server
        manifest_path, manifest_data = manifest or self.write_manifest()
        if controller is None:
            controller = FakeController(streams=manifest_data['streams'])
        out_path = Path(self.temp.name) / out_name
        env = {
            'WORKLOAD': str(manifest_path),
            'RUN_ID': 'test-run',
            'CONTROLLER_TIMEOUT_S': '3',
            'READ_TIMEOUT_S': '3',
            'DRAIN_TIMEOUT_S': '1',
        }
        env.update(env_overrides or {})
        if not use_controller_env:
            rc = D.main([server.url, 'model-x', str(out_path)],
                        controller=controller, env=env)
        else:
            rc = D.main([server.url, 'model-x', str(out_path)], env=env)
        if not out_path.exists():
            self.fail('driver did not write ' + str(out_path))
        data = json.loads(out_path.read_text())
        return rc, data, server, controller, out_path

    # --- manifest / plan-only ------------------------------------------------

    def test_missing_manifest_is_plan_only_and_issues_no_request(self):
        self.start_server()
        out_path = Path(self.temp.name) / 'plan.json'
        rc = D.main(['http://127.0.0.1:9', 'model-x', str(out_path)], env={})
        self.assertEqual(rc, 2)
        data = json.loads(out_path.read_text())
        self.assertEqual(data['summary']['verdict'], 'INCONCLUSIVE')
        self.assertEqual(data['repeats'], [])
        self.assertEqual(self.server.requests, [])

    def test_incomplete_manifest_and_zero_budget_are_plan_only(self):
        self.start_server()
        for override in ({'generation_budget': 0},
                         {'window_s': None},
                         {'seed': None}):
            manifest = _deep_update(copy.deepcopy(BASE_MANIFEST), override)
            path = Path(self.temp.name) / ('bad-%d.json' % len(
                list(Path(self.temp.name).glob('bad-*.json'))))
            path.write_text(json.dumps(manifest))
            out_path = Path(self.temp.name) / ('plan-%d.json' % len(
                list(Path(self.temp.name).glob('plan-*.json'))))
            rc = D.main(['http://127.0.0.1:9', 'model-x', str(out_path)],
                        controller=FakeController(), env={'WORKLOAD': str(path)})
            self.assertEqual(rc, 2)
            data = json.loads(out_path.read_text())
            self.assertEqual(data['summary']['verdict'], 'INCONCLUSIVE')
            self.assertEqual(data['repeats'], [])
        self.assertEqual(self.server.requests, [])

    def test_standing_w1_pins_streams_and_prompt_tokens(self):
        self.start_server()
        path, _manifest = self.write_manifest(
            workload='w1-standing', streams=7,
            prompt_tokens=D.STANDING_PROMPT_TOKENS)
        out_path = Path(self.temp.name) / 'standing.json'
        rc = D.main(['http://127.0.0.1:9', 'model-x', str(out_path)],
                    controller=FakeController(), env={'WORKLOAD': str(path)})
        self.assertEqual(rc, 2)
        data = json.loads(out_path.read_text())
        self.assertEqual(data['summary']['verdict'], 'INCONCLUSIVE')
        self.assertTrue(any('standing_w1' in str(item)
                            for item in data['workload']['reasons']))
        self.assertEqual(self.server.requests, [])

    def test_missing_controller_is_plan_only(self):
        self.start_server()
        path, _manifest = self.write_manifest(repeats=1, route_plan=['split'])
        out_path = Path(self.temp.name) / 'no-controller.json'
        rc = D.main(['http://127.0.0.1:9', 'model-x', str(out_path)],
                    env={'WORKLOAD': str(path)})
        self.assertEqual(rc, 2)
        data = json.loads(out_path.read_text())
        self.assertEqual(data['summary']['verdict'], 'INCONCLUSIVE')
        self.assertEqual(self.server.requests, [])

    # --- the valid protocol --------------------------------------------------

    def test_valid_full_and_split_control_exits_zero(self):
        self.start_server()
        rc, data, server, controller, _out = self.run_driver()
        self.assertEqual(rc, 0, data['summary'])
        self.assertEqual(data['summary']['verdict'], 'COMPLETE')
        self.assertEqual(data['summary']['totals']['valid'], 2)
        self.assertEqual(data['summary']['totals']['invalid'], 0)
        self.assertEqual(len(server.measure_requests()), 6)
        self.assertEqual(len(controller.calls), 12)
        expected = ['prepare', 'pause_full', 'resume', 'pause_end',
                    'leave_full', 'resume',
                    'prepare', 'pause_enter_split', 'resume', 'pause_end',
                    'leave_full', 'resume']
        self.assertEqual([call['command']['action'] for call in controller.calls],
                         expected)
        seqs = [call['command']['command_seq'] for call in controller.calls]
        self.assertEqual(seqs, sorted(set(seqs)))
        self.assertEqual(seqs, list(range(1, len(seqs) + 1)))

    def test_full_route_has_no_fictitious_transition(self):
        self.start_server()
        manifest = self.write_manifest(repeats=1, route_plan=['full'])
        rc, data, _server, controller, _out = self.run_driver(manifest=manifest)
        self.assertEqual(rc, 0, data['summary'])
        calls = [call['command'] for call in controller.calls]
        self.assertEqual([call['action'] for call in calls],
                         ['prepare', 'pause_full', 'resume', 'pause_end',
                          'leave_full', 'resume'])
        receipts = data['repeats'][0]['calls']
        pause = receipts['pause']['receipt']
        leave = receipts['leave']['receipt']
        self.assertEqual(pause['route_generation_after'],
                         pause['route_generation_before'])
        self.assertEqual(leave['route_generation_after'],
                         leave['route_generation_before'])
        self.assertFalse(any('transition_ack' in rank
                             for rank in pause['ranks'] + leave['ranks']))
        self.assertTrue(data['repeats'][0]['window']['full'])

    def test_request_bytes_identical_across_repeats_and_prepare_uncounted(self):
        self.start_server()
        manifest = self.write_manifest(cache_preparation=True)
        rc, data, server, _controller, _out = self.run_driver(manifest=manifest)
        self.assertEqual(rc, 0, data['summary'])
        measures = server.measure_requests()
        self.assertEqual(len(measures), 6)
        by_stream = {}
        for row in measures:
            by_stream.setdefault(row['stream'], set()).add(row['sha256'])
        self.assertTrue(all(len(shas) == 1 for shas in by_stream.values()))
        self.assertEqual(len(server.prepare_requests()), 3)
        self.assertTrue(all(row['max_tokens'] == 1
                            for row in server.prepare_requests()))
        prep = data['preparation_evidence']
        self.assertFalse(prep['counted'])
        self.assertEqual(prep['passes'], 1)
        self.assertEqual(len(prep['requests']), 3)
        self.assertTrue(prep['descriptive_only'])
        self.assertEqual(data['summary']['preparation_ttft']['samples'], 6)

    def test_no_requests_during_pause_or_window(self):
        self.start_server()
        manifest = self.write_manifest(repeats=1, route_plan=['split'])
        rc, data, server, controller, _out = self.run_driver(manifest=manifest)
        self.assertEqual(rc, 0, data['summary'])
        measures = server.measure_requests()
        self.assertEqual(len(measures), 3)
        pause = [call for call in controller.calls
                 if call['command']['action'] == 'pause_enter_split'][0]
        resume = [call for call in controller.calls
                  if call['command']['action'] == 'resume'][0]
        self.assertLess(max(row['t_ns'] for row in measures), pause['t_ns'])
        self.assertEqual(data['repeats'][0]['submissions_after_seal'], 0)
        self.assertEqual(len({row['stream'] for row in measures}), 3)
        self.assertGreater(resume['t_ns'], pause['t_ns'])

    def test_window_starts_after_resume_and_ends_before_pause_end(self):
        self.start_server()
        manifest = self.write_manifest(repeats=1, route_plan=['split'])
        rc, data, _server, controller, _out = self.run_driver(manifest=manifest)
        self.assertEqual(rc, 0, data['summary'])
        window = data['repeats'][0]['window']
        resume = [call for call in controller.calls
                  if call['command']['action'] == 'resume'][0]
        pause_end = [call for call in controller.calls
                     if call['command']['action'] == 'pause_end'][0]
        self.assertGreaterEqual(window['start_ns'], resume['t_ns'])
        self.assertLessEqual(window['end_ns'], pause_end['t_ns'])
        self.assertTrue(window['end_boundary_before_pause_end'])
        self.assertTrue(window['client_visible'])
        self.assertLess(window['wall_s'], 0.08 + 0.05)

    def test_streams_advance_out_of_order_but_all_must_open(self):
        self.start_server(start_delays={0: 0.04, 1: 0.0, 2: 0.02})
        manifest = self.write_manifest(repeats=1, route_plan=['split'])
        rc, data, _server, _controller, _out = self.run_driver(manifest=manifest)
        self.assertEqual(rc, 0, data['summary'])
        order = data['repeats'][0]['open']['advance_order']
        self.assertEqual(order, [1, 2, 0])
        self.assertEqual(data['repeats'][0]['open']['opened_streams'], 3)

    def test_sse_event_split_across_reads_is_parsed(self):
        self.start_server(split_reads=True)
        manifest = self.write_manifest(repeats=1, route_plan=['split'])
        rc, data, _server, _controller, _out = self.run_driver(manifest=manifest)
        self.assertEqual(rc, 0, data['summary'])
        for stream in data['repeats'][0]['streams']:
            self.assertGreaterEqual(stream['event_count'], 2)
            self.assertIsNotNone(stream['first_arrival_at_ns'])

    def test_usage_only_and_coalesced_arrivals_are_labeled(self):
        self.start_server(coalesce=3, usage_only_every=2)
        manifest = self.write_manifest(repeats=1, route_plan=['split'])
        rc, data, _server, _controller, _out = self.run_driver(manifest=manifest)
        self.assertEqual(rc, 0, data['summary'])
        stream = data['repeats'][0]['streams'][0]
        window = stream['window']
        self.assertEqual(window['token_count_basis'], 'continuous_usage_stats')
        self.assertGreater(window['coalesced_arrivals'], 0)
        self.assertGreater(window['completion_delta'], window['arrivals'])
        self.assertGreater(window['arrivals'], 0)
        kinds = [event['kind'] for event in stream['events']]
        self.assertIn('usage', kinds)
        self.assertNotIn('data', stream['events'][0])
        self.assertFalse(any('content' in event for event in stream['events']))
        for key in ('gap_ms_p50', 'gap_ms_p95', 'gap_ms_p99', 'gap_ms_max',
                    'open_gap_ms', 'rate_tokens_per_s'):
            self.assertIn(key, window)

    def test_explicit_end_boundary_before_client_completion(self):
        self.start_server()
        manifest = self.write_manifest(repeats=1, route_plan=['split'])
        rc, data, _server, _controller, _out = self.run_driver(manifest=manifest)
        self.assertEqual(rc, 0, data['summary'])
        record = data['repeats'][0]
        end_ns = record['window']['end_ns']
        for stream in record['streams']:
            if stream['disposition'] == 'cancelled':
                self.assertGreater(stream['done_at_ns'], end_ns)
            self.assertIsNone(stream['finish_at_ns'])
        self.assertTrue(record['cleanup']['restore_before_drain'])
        self.assertTrue(record['drain']['note'])
        self.assertTrue(record['exposure']['proven'])
        for row in record['exposure']['streams']:
            self.assertTrue(row['in_range'])
            self.assertIsNotNone(row['start_completion'])
            self.assertIsNotNone(row['end_completion'])

    # --- invalid evidence ----------------------------------------------------

    def test_early_eos_invalidates_repeat_and_is_retained(self):
        self.start_server(early_eos_streams={1}, early_eos_after=3)
        manifest = self.write_manifest(repeats=2, route_plan=['full', 'split'])
        rc, data, server, _controller, _out = self.run_driver(manifest=manifest)
        self.assertEqual(rc, 2, data['summary'])
        self.assertEqual(data['summary']['verdict'], 'INCONCLUSIVE')
        self.assertEqual(data['summary']['totals']['invalid'], 2)
        invalid = data['summary']['invalid']
        self.assertTrue(any('early_finish:1' in reason
                            for row in invalid for reason in row['reasons']))
        self.assertFalse(data['repeats'][1]['not_run'])
        self.assertEqual(data['summary']['totals']['attempted'], 2)
        self.assertEqual(data['summary']['totals']['not_run'], 0)
        self.assertEqual(len(server.measure_requests()), 6)

    def test_one_stream_never_starts_is_incomplete_cohort(self):
        self.start_server(never_start={2})
        manifest = self.write_manifest(repeats=1, route_plan=['split'],
                                       open_timeout_s=0.2)
        rc, data, _server, controller, _out = self.run_driver(manifest=manifest)
        self.assertEqual(rc, 2, data['summary'])
        record = data['repeats'][0]
        self.assertFalse(record['open']['complete'])
        self.assertTrue(any('stream_not_advanced:2' in reason
                            for reason in record['reasons']))
        self.assertEqual([call['command']['action'] for call in controller.calls],
                         ['prepare'])

    def test_early_finish_before_seal_is_incomplete_cohort(self):
        self.start_server(early_eos_streams={0}, early_eos_after=1,
                          start_delays={1: 0.08, 2: 0.08})
        manifest = self.write_manifest(repeats=1, route_plan=['split'])
        rc, data, _server, controller, _out = self.run_driver(manifest=manifest)
        self.assertEqual(rc, 2, data['summary'])
        record = data['repeats'][0]
        self.assertFalse(record['open']['complete'])
        self.assertTrue(any('early_finish_before_seal:0' in reason
                            for reason in record['reasons']))
        self.assertEqual([call['command']['action'] for call in controller.calls],
                         ['prepare'])

    def test_disconnect_before_window_end_invalidates(self):
        self.start_server(close_after={1: 3})
        manifest = self.write_manifest(repeats=1, route_plan=['split'])
        rc, data, _server, _controller, _out = self.run_driver(manifest=manifest)
        self.assertEqual(rc, 2, data['summary'])
        record = data['repeats'][0]
        self.assertFalse(record['window']['full'])
        self.assertIn(1, record['window']['errors_before_end'])
        self.assertEqual(record['streams'][1]['error'],
                         'stream_closed_without_finish')

    def test_http_error_before_open_is_incomplete(self):
        self.start_server(http_error_streams={0})
        manifest = self.write_manifest(repeats=1, route_plan=['split'])
        rc, data, _server, _controller, _out = self.run_driver(manifest=manifest)
        self.assertEqual(rc, 2, data['summary'])
        self.assertFalse(data['repeats'][0]['open']['complete'])
        self.assertTrue(any('stream_not_advanced_error:0' in reason
                            for reason in data['repeats'][0]['reasons']))

    def test_completion_exposure_out_of_range_invalidates(self):
        self.start_server()
        manifest = self.write_manifest(
            repeats=1, route_plan=['split'],
            completion_policy={'per_request_delta_min': 100000,
                               'per_request_delta_max': 100001})
        rc, data, _server, _controller, _out = self.run_driver(manifest=manifest)
        self.assertEqual(rc, 2, data['summary'])
        record = data['repeats'][0]
        self.assertFalse(record['exposure']['proven'])
        self.assertTrue(any('completion_delta_out_of_range' in reason
                            for reason in record['reasons']))
        self.assertFalse(any(row['in_range'] for row in
                             record['exposure']['streams']))

    def test_missing_completion_counts_is_inconclusive(self):
        self.start_server(continuous=False)
        manifest = self.write_manifest(
            repeats=1, route_plan=['split'],
            request_shape={'continuous_usage_stats': False})
        rc, data, _server, _controller, _out = self.run_driver(manifest=manifest)
        self.assertEqual(rc, 2, data['summary'])
        record = data['repeats'][0]
        self.assertFalse(record['exposure']['proven'])
        self.assertTrue(any('completion_counters_unavailable' in reason
                            for reason in record['reasons']))

    # --- controller receipt contract ----------------------------------------

    def _run_with_mode(self, mode, route='split', repeats=1):
        self.start_server()
        routes = [route] * repeats
        manifest = self.write_manifest(repeats=repeats, route_plan=routes)
        controller = FakeController(streams=3, modes=mode)
        return self.run_driver(controller=controller, manifest=manifest)

    def test_missing_rank_is_a_contract_failure(self):
        rc, data, _server, _controller, _out = self._run_with_mode(
            {'pause_enter_split': 'missing_rank'})
        self.assertEqual(rc, 1, data['summary'])
        self.assertEqual(data['summary']['totals']['contract'], 1)
        self.assertFalse(data['repeats'][0]['window'])
        self.assertTrue(any('rank_coverage' in reason
                            for reason in data['repeats'][0]['reasons']))

    def test_synthetic_ack_is_not_a_worker_ack(self):
        rc, data, _server, _controller, _out = self._run_with_mode(
            {'pause_enter_split': 'ack_false'})
        self.assertEqual(rc, 1, data['summary'])
        self.assertTrue(any('transition_ack_synthetic' in reason
                            for reason in data['repeats'][0]['reasons']))

    def test_stale_command_sequence_is_rejected(self):
        rc, data, _server, _controller, _out = self._run_with_mode(
            {'resume': 'stale_seq'})
        self.assertEqual(rc, 1, data['summary'])
        self.assertFalse(data['repeats'][0]['window'])
        self.assertTrue(any('command_seq_mismatch' in reason
                            for reason in data['repeats'][0]['reasons']))

    def test_wrong_run_id_and_cohort_are_rejected(self):
        for mode in ('wrong_run_id', 'wrong_cohort'):
            with self.subTest(mode=mode):
                rc, data, _server, _controller, _out = self._run_with_mode(
                    {'pause_enter_split': mode})
                self.assertEqual(rc, 1, data['summary'])
                self.assertTrue(any('mismatch' in reason
                                    for reason in data['repeats'][0]['reasons']))

    def test_engine_or_storage_incarnation_change_is_rejected(self):
        for mode in ('engine_changed', 'storage_changed'):
            with self.subTest(mode=mode):
                rc, data, _server, _controller, _out = self._run_with_mode(
                    {'pause_enter_split': mode})
                self.assertEqual(rc, 1, data['summary'])
                self.assertTrue(any('changed' in reason
                                    for reason in data['repeats'][0]['reasons']))

    def test_pause_incomplete_or_timed_out_is_inconclusive(self):
        for mode in ('pause_incomplete', 'pause_timeout'):
            with self.subTest(mode=mode):
                rc, data, _server, _controller, _out = self._run_with_mode(
                    {'pause_enter_split': mode})
                self.assertEqual(rc, 2, data['summary'])
                self.assertFalse(data['repeats'][0]['window'])
                self.assertTrue(any('pause' in reason
                                    for reason in data['repeats'][0]['reasons']))

    def test_missing_resume_evidence_does_not_measure(self):
        rc, data, _server, _controller, _out = self._run_with_mode(
            {'resume': 'resume_missing'})
        self.assertEqual(rc, 2, data['summary'])
        self.assertFalse(data['repeats'][0]['window'])
        self.assertTrue(any('resume_missing' in reason
                            for reason in data['repeats'][0]['reasons']))

    def test_full_noop_with_generation_bump_is_rejected(self):
        rc, data, _server, _controller, _out = self._run_with_mode(
            {'pause_full': 'noop_generation_bump'}, route='full')
        self.assertEqual(rc, 1, data['summary'])
        self.assertTrue(any('pause_full_changed_generation' in reason
                            for reason in data['repeats'][0]['reasons']))

    def test_final_interval_prefill_invalidates(self):
        rc, data, _server, _controller, _out = self._run_with_mode(
            {'pause_end': 'counter_prefill'})
        self.assertEqual(rc, 1, data['summary'])
        record = data['repeats'][0]
        self.assertTrue(record['window']['measured'])
        self.assertTrue(any('interval_invalidated:prefill' in reason
                            for reason in record['reasons']))
        self.assertFalse(record['cleanup']['attempted'])

    def test_final_interval_counter_regression_invalidates(self):
        rc, data, _server, _controller, _out = self._run_with_mode(
            {'pause_end': 'counter_regress'})
        self.assertEqual(rc, 1, data['summary'])
        self.assertTrue(any('interval_counter_regressed' in reason
                            for reason in data['repeats'][0]['reasons']))

    def test_restore_failure_is_a_contract_failure(self):
        rc, data, _server, _controller, _out = self._run_with_mode(
            {'leave_full': 'ack_false'})
        self.assertEqual(rc, 1, data['summary'])
        record = data['repeats'][0]
        self.assertTrue(record['window']['measured'])
        self.assertFalse(record['cleanup']['restore_before_drain'])
        self.assertTrue(any('restore_not_before_drain' in reason
                            for reason in record['reasons']))

    def test_malformed_receipt_is_inconclusive_without_requests(self):
        rc, data, _server, _controller, _out = self._run_with_mode(
            {'prepare': 'malformed'})
        self.assertEqual(rc, 2, data['summary'])
        self.assertEqual(data['repeats'][0]['phase'], 'prepare_failed')

    def test_transport_failure_after_rc_zero_is_not_proof(self):
        rc, data, _server, _controller, _out = self._run_with_mode(
            {'resume': 'transport_timeout'})
        self.assertEqual(rc, 2, data['summary'])
        self.assertFalse(data['repeats'][0]['window'])
        self.assertTrue(any('controller_transport' in reason
                            for reason in data['repeats'][0]['reasons']))

    def test_subprocess_controller_gets_one_json_command_on_stdin(self):
        self.start_server()
        script = Path(self.temp.name) / 'controller.py'
        trace = Path(self.temp.name) / 'controller-trace.txt'
        script.write_text(
            'import json, sys\n'
            'payload = sys.stdin.read()\n'
            'with open(sys.argv[1], "a") as handle:\n'
            '    handle.write(payload + "\\n")\n'
            'print(json.dumps({"nonsense": True}))\n')
        manifest = self.write_manifest(repeats=1, route_plan=['split'])
        command = '%s %s %s' % (sys.executable, script, trace)
        rc, data, server, _controller, _out = self.run_driver(
            manifest=manifest, use_controller_env=True,
            env_overrides={'CONTROLLER_CMD': command})
        self.assertEqual(rc, 2, data['summary'])
        self.assertEqual(server.requests, [])
        lines = trace.read_text().strip().splitlines()
        self.assertEqual(len(lines), 1)
        command_json = json.loads(lines[0])
        self.assertEqual(command_json['schema'], 1)
        self.assertEqual(command_json['action'], 'prepare')
        self.assertEqual(command_json['target_route'], 'full')
        self.assertEqual(command_json['command_seq'], 1)
        self.assertEqual(command_json['expected_rank_set'], [0, 1, 2, 3])

    # --- receipt/summary pure checks ----------------------------------------

    def test_validate_receipt_rejects_bool_as_int_and_accepts_valid(self):
        controller = FakeController(streams=3)
        ctx = {'run_id': 'r', 'repeat': 0, 'cohort_id': 'c',
               'command_seq': 0, 'action': 'pause_enter_split',
               'engine_epoch': None, 'storage_epoch': None,
               'route': 'full', 'route_generation': None, 'streams': 3}
        command = D._next_command(ctx, 'pause_enter_split', 'split')
        ctx['command_seq'] = command['command_seq']
        receipt, _transport = controller.execute(command, 1.0)
        ok, reasons, severity, state = D.validate_receipt(receipt, command, ctx)
        self.assertTrue(ok, reasons)
        self.assertEqual(severity, 'valid')
        self.assertEqual(state['route'], 'split')
        mutated = copy.deepcopy(receipt)
        mutated['route_generation_after'] = True
        ok, reasons, severity, _state = D.validate_receipt(mutated, command, ctx)
        self.assertFalse(ok)
        self.assertEqual(severity, 'contract')
        self.assertTrue(any('route_generation_after_invalid' in reason
                            for reason in reasons))

    def test_interval_valid_is_pure_and_retains_reasons(self):
        self.start_server()
        manifest = self.write_manifest(repeats=1, route_plan=['split'])
        rc, data, _server, _controller, _out = self.run_driver(manifest=manifest)
        self.assertEqual(rc, 0, data['summary'])
        record = copy.deepcopy(data['repeats'][0])
        config = copy.deepcopy(data['config'])
        valid, reasons, severity = D.interval_valid(record, config)
        self.assertTrue(valid, reasons)
        record['window']['full'] = False
        record['window']['early_finish'] = [2]
        record['window']['errors_before_end'] = [2]
        valid, reasons, severity = D.interval_valid(record, config)
        self.assertFalse(valid)
        self.assertEqual(severity, 'inconclusive')
        self.assertTrue(any('early_finish:2' in reason for reason in reasons))
        record['required_calls'].append('pause_end')
        record['calls'].pop('pause_end', None)
        valid, reasons, severity = D.interval_valid(record, config)
        self.assertFalse(valid)
        self.assertTrue(any('pause_end_not_called' in reason
                            for reason in reasons))

    def test_resummarize_reproduces_all_verdicts_including_invalid(self):
        self.start_server(early_eos_streams={1}, early_eos_after=3)
        manifest = self.write_manifest(repeats=2, route_plan=['full', 'split'])
        rc, data, _server, _controller, out_path = self.run_driver(
            manifest=manifest)
        self.assertEqual(rc, 2)
        before = {
            'verdict': data['summary']['verdict'],
            'invalid': [(row['repeat'], row['route'], row['severity'],
                         list(row['reasons']))
                        for row in data['summary']['invalid']],
            'valid': [record['valid'] for record in data['repeats']],
            'totals': dict(data['summary']['totals']),
        }
        rc2 = D.main(['--resummarize', str(out_path)], env={})
        self.assertEqual(rc2, 2)
        after_data = json.loads(out_path.read_text())
        after = {
            'verdict': after_data['summary']['verdict'],
            'invalid': [(row['repeat'], row['route'], row['severity'],
                         list(row['reasons']))
                        for row in after_data['summary']['invalid']],
            'valid': [record['valid'] for record in after_data['repeats']],
            'totals': dict(after_data['summary']['totals']),
        }
        self.assertEqual(before, after)

    def test_no_prompt_or_output_payload_in_artifact(self):
        self.start_server()
        manifest = self.write_manifest(cache_preparation=True)
        rc, data, _server, _controller, out_path = self.run_driver(
            manifest=manifest)
        self.assertEqual(rc, 0, data['summary'])
        text = out_path.read_text()
        self.assertNotIn('Repository context:', text)
        self.assertNotIn('You are coding agent', text)
        for forbidden in ('"messages"', '"content"', '"token_ids"',
                          '"authorization"', '"headers"'):
            self.assertNotIn(forbidden, text.lower())
        for key in self._walk_keys(data):
            self.assertNotIn(key, ('messages', 'content', 'token_ids',
                                   'authorization', 'headers', 'data', 'text'))

    def _walk_keys(self, value):
        if isinstance(value, dict):
            for key, child in value.items():
                yield key
                yield from self._walk_keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from self._walk_keys(child)

    def test_summary_is_valid_only_labeled_and_keeps_performance_separate(self):
        self.start_server()
        rc, data, _server, _controller, _out = self.run_driver()
        self.assertEqual(rc, 0)
        summary = data['summary']
        self.assertTrue(summary['descriptive_valid_only'])
        self.assertEqual(summary['totals']['required'], 2)
        self.assertEqual(summary['totals']['attempted'], 2)
        self.assertIn('not a PASS', summary['note'])
        self.assertFalse(summary['output_correctness']['checked'])
        self.assertIn('split_over_full', summary['matched'])
        self.assertIn('gap_ms_p99', summary['routes']['full'])
        self.assertIn('open_gap_ms_p95', summary['routes']['split'])
        self.assertIn('ttft_ms_p50', summary['preparation_ttft'])
        self.assertNotIn('ttft_ms_p50', summary['routes']['full'])
        self.assertEqual(summary['routes']['full']['valid_repeats'], [0])
        self.assertEqual(summary['routes']['split']['valid_repeats'], [1])

    def test_clock_anchor_and_monotonic_event_times_are_recorded(self):
        self.start_server()
        rc, data, _server, _controller, _out = self.run_driver()
        self.assertEqual(rc, 0)
        self.assertIsNotNone(data['clock']['monotonic_anchor_ns'])
        self.assertIsNotNone(data['clock']['wall_anchor_ns'])
        for record in data['repeats']:
            self.assertGreater(record['window']['start_ns'],
                               data['clock']['monotonic_anchor_ns'])
            for stream in record['streams']:
                times = [event['t_ns'] for event in stream['events']]
                self.assertEqual(times, sorted(times))
                self.assertTrue(all(type(value) is int for value in times))
                self.assertNotIn('data', stream['events'][0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
