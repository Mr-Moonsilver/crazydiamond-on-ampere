#!/usr/bin/env python3
"""Focused CPU tests for the native-qualification orchestration driver.

No network, no GPU, no engine: the loopback transport is replaced by explicit
fakes at the transport boundary, and every provider is a supplied seam.  The
tests exercise the real driver validation, retention, digest binding, phase
isolation, typed-ACK checks and failure cleanup.
"""
import asyncio
import copy
import hashlib
import http.client
import http.server
import json
import os
import socket
import stat
import tempfile
import threading
import time
import types
import unittest
from dataclasses import replace
from unittest import mock

import e3_decode_correctness as correctness
import e3_native_qualification_driver as driver_module
import e3_teacher_decode_client as teacher_client
from e3_native_qualification_driver import (
    ImmutableStore, NativePlacementRunner, NativeRow1Producer, PhaseConflict,
    PhaseUnavailable, QualificationDriver, QualificationError,
    QualificationRefused, RetainedBankReader, canonical_sha, freeze_row1_plan)

# The actual source-owned row1 dataclasses/digest envelope, imported from the
# pinned runtime tree (the driver put it on sys.path).  No peer_inputs copy.
import row1_oracle as row1  # noqa: E402
from native_overlap_control import NativeControlClient  # noqa: E402
from native_overlap_session import NativeOverlapSession  # noqa: E402
from wire_v1 import ROUTE_BANK_FULL, ROUTE_BANK_SPLIT  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def sha(value):
    return canonical_sha(value)


def digest_char(char='a'):
    return char * 64


def k_packet(k, rank, *, epoch=3, run_id='run-1', generation=5,
             disabled_generation=None, enabled=False, records=None,
             layer_count=1):
    identity = k.Row1Identity(
        rank=rank, worker_incarnation=100 + rank, engine_epoch=epoch,
        route_generation=generation, storage_epoch=2, source_epoch=1,
        state_epoch=1, mapping_version=1,
        layer='model.layers.%d.self_attn' % (rank // 2),
        operation='sparse_row_bucket')
    if records is None:
        records = [k.Row1Comparison(
            identity=identity, bucket=rank, route='sparse_dma', rows=16,
            pieces=2, compact_start=0, compared_bytes=8192,
            different_bytes=None, equal=True, finite=None, coverage_rows=16,
            coverage_bytes=8192, observed_ns=100 + rank,
            digest=digest_char('b'))]
    layer = k.Row1LayerRecord(
        identity=identity, passed=True, aborted=None, route='sparse_dma',
        bucket_checks=len(records), bucket_rows=16 * len(records),
        bucket_bytes=8192 * len(records), output_checks=0, output_rows=0,
        output_bytes=0, hold_released=True, hold_stats=(),
        digest=digest_char('d'))
    layers = [layer.as_dict()] if layer_count else []
    raw = {'enabled': enabled, 'comparisons': len(records),
           'layers': len(layers), 'failures': 0, 'capacity': 4096,
           'records': [record.as_dict() for record in records],
           'layer_records': layers, 'failure_records': []}
    disabled = generation if disabled_generation is None else disabled_generation
    return {'schema': driver_module.ROW1_SCHEMA, 'run_id': run_id,
            'rank': rank, 'worker_incarnation': 100 + rank,
            'engine_epoch': epoch, 'route_generation': generation,
            'disabled_route_generation': disabled,
            'local_layers': ['model.layers.%d.self_attn' % (rank // 2)],
            'enabled': enabled, 'raw': raw}


class Row1Control:
    """Session admission transport plus the exact two row1 worker actions."""

    def __init__(self, epoch=3, run_id='run-1', packets=None):
        self.epoch = epoch
        self.run_id = run_id
        self.base_url = 'http://127.0.0.1:9'
        self.packets = packets
        self.armed = False
        self.arm_calls = 0
        self.disarm_calls = 0
        self.worker_calls = []
        self.worker_actions = []
        self.worker_refusal = None

    def session(self, command, timeout_s=None):
        action = command['action']
        identity = {'schema': 1, 'action': action,
                    'run_id': command['run_id']}
        if action == 'arm_admission':
            assert command['phase'] == 'row1-native'
            self.armed = True
            self.arm_calls += 1
            return dict(identity, phase='row1-native', armed=True,
                        admission_intent={
                            'schema': 1, 'owner': 'engine-core',
                            'engine_epoch': self.epoch, 'phase': 'row1-native',
                            'route': 'split', 'route_generation': 5,
                            'armed': True, 'seq': 1, 'armed_ns': 10},
                        previous_admission_intent=None, route='split',
                        route_generation=5)
        if action == 'disarm_admission':
            assert self.armed
            self.armed = False
            self.disarm_calls += 1
            return dict(identity, armed=False, admission_intent=None,
                        previous_admission_intent={
                            'schema': 1, 'owner': 'engine-core',
                            'engine_epoch': self.epoch, 'phase': 'row1-native',
                            'route': 'split', 'route_generation': 5,
                            'armed': True, 'seq': 1, 'armed_ns': 10},
                        route='split', route_generation=5)
        raise AssertionError(action)

    def _reply(self, inner_by_rank):
        rows = []
        keyed = (type(inner_by_rank) is dict
                 and any(type(key) is int for key in inner_by_rank))
        for rank in (0, 1, 2, 3):
            if keyed and rank not in inner_by_rank:
                continue
            inner = inner_by_rank[rank] if keyed else inner_by_rank
            rows.append({'schema': 1, 'action': 'qualification',
                         'worker_rank': rank, 'engine_epoch': self.epoch,
                         'route_generation': 5,
                         'qualification': copy.deepcopy(inner)})
        return {'schema': 1, 'action': 'qualification',
                'engine_epoch': self.epoch, 'ranks': [0, 1, 2, 3],
                'workers': rows}

    def _dispatch(self, aggregate, command):
        return {'schema': 1, 'action': 'qualification_worker',
                'run_id': command['run_id'],
                'command_seq': command['command_seq'],
                'client_id': command['client_id'],
                'engine_epoch': self.epoch, 'worker_reply': aggregate}

    def worker(self, command, timeout_s=None):
        self.worker_calls.append(copy.deepcopy(command))
        action = command['worker_action']
        self.worker_actions.append(action)
        if self.worker_refusal is not None:
            raise PhaseUnavailable(self.worker_refusal)
        payload = command['payload']
        if action == 'row1_configure':
            return self._dispatch(self._reply({
                'run_id': self.run_id, 'engine_epoch': self.epoch,
                'route_generation': 5, 'enabled': payload['enabled']}), command)
        if action == 'row1_evidence':
            if self.packets is None:
                raise AssertionError('no packets were prepared')
            return self._dispatch(self._reply({
                packet['rank']: packet for packet in self.packets}), command)
        raise AssertionError(action)


def row1_acks(enabled):
    return [{'rank': rank,
             'ack': {'run_id': 'run-1', 'engine_epoch': 3,
                     'route_generation': 5, 'enabled': enabled}}
            for rank in (0, 1, 2, 3)]


def teacher_workload():
    with open(os.path.join(HERE, 'workloads', 'q17b-correctness-precontrols.json'),
              encoding='utf-8') as handle:
        manifest = json.load(handle)
    problems, workload, _ = correctness.validate_manifest(manifest)
    assert not problems, problems
    return replace(workload, generation_budget=teacher_client.TOTAL)


def placement_workload():
    with open(os.path.join(HERE, 'workloads', 'q17b-correctness-precontrols.json'),
              encoding='utf-8') as handle:
        manifest = json.load(handle)
    problems, workload, _ = correctness.validate_manifest(manifest)
    assert not problems, problems
    return workload


class ReplyControl:
    """Transport fake that echoes the exact identity every reply must carry."""

    def __init__(self, epoch=3):
        self.epoch = epoch
        self.base_url = 'http://127.0.0.1:9'
        self.controller = object()
        self.frontend_calls = []
        self.worker_calls = []
        self.session_calls = []
        self.enabled = False

    @staticmethod
    def _identity(command):
        return {'schema': 1, 'run_id': command['run_id'],
                'command_seq': command['command_seq'],
                'client_id': command['client_id'],
                'action': command['action']}

    def session(self, command, timeout_s=None):
        self.session_calls.append(command)
        reply = self._identity(command)
        raise AssertionError('session transport was not expected in this test')

    def frontend(self, command, timeout_s=None):
        self.frontend_calls.append(copy.deepcopy(command))
        self.enabled = command['enabled']
        reply = self._identity(command)
        reply.update(engine_epoch=self.epoch, enabled=self.enabled,
                     plan_sha256=None if command['plan'] is None else sha(command['plan']),
                     admitted=0, pending=0, steps=768)
        return reply

    def worker(self, command, timeout_s=None):
        self.worker_calls.append(copy.deepcopy(command))
        raise AssertionError('worker transport was not expected in this test')


class FakeInstrument:
    """Reviewed Instrument shape used to drive the controls retention path."""

    def __init__(self, evidence):
        self.evidence = evidence
        self.plan = None
        self.plan_calls = []

    def requests(self, bodies, cohort, repeat):
        self.plan = {'schema': 's5-teacher-decode/1',
                     'requests': [{'request_id': 'tf-%d-%d' % (repeat, index),
                                   'prompt_sha256': digest_char('b'),
                                   'continuation': [index] * teacher_client.TOTAL}
                                  for index in range(8)],
                     'snapshot_max_bytes': 2 * 1024 ** 3}

    def cohort(self, url, model):
        return []

    def finish(self, runs):
        return {'schema': 1, 'grade': {'verdict': 'PASS', 'cached': True},
                'evidence': copy.deepcopy(self.evidence)}


def teacher_evidence(workload):
    runs = {}
    for name in driver_module.TEACHER_SEQUENCE:
        runs[name] = {
            'route': driver_module.TEACHER_ROUTES[name],
            'row_mapping': {
                'overflow': False, 'defects': [],
                'records': [{'stream': index % 8,
                             'position': 256 + (index // 16),
                             'rank': 2 + ((index // 8) % 2)}
                            for index in range(8 * 256 * 2)],
            },
        }
    return {'schema': driver_module.TEACHER_SCHEMA,
            'fixed_positions': list(range(256, 512)),
            'runs': runs}


def retained_teacher(store, artifact):
    record = {'schema': 1, 'phase': 'rule4_controls', 'kind': 'teacher',
              'run_id': 'run-1', 'artifact': artifact}
    store.write_once('rule4_controls.json', record)
    return artifact


def short_window(run_id='run-1', epoch=3):
    return {'kind': 'short', 'run_id': run_id, 'window_id': 'window-1',
            'engine_epoch': epoch, 'start_ns': 1_000, 'end_ns': 2_000,
            'window_tokens': 40, 'replays': 40, 'observed_replays': {0: 40, 1: 40, 2: 40, 3: 40},
            'teacher_overlap': False, 'teacher_window_end_ns': None,
            'cohort_id': 'placement-x', 'decode_progress': 40,
            'native_phase': 'w2-overlap',
            'admission': {'phase': 'w2-overlap', 'armed': True,
                          'admission_intent': {
                              'schema': 1, 'owner': 'engine-core',
                              'engine_epoch': epoch, 'phase': 'w2-overlap',
                              'route': 'split', 'route_generation': 1,
                              'armed': True, 'seq': 1, 'armed_ns': 900},
                          'route': 'split', 'route_generation': 1},
            'route_generation': 1,
            'native_prefill_request': {
                'complete': True, 'plan_sha256': digest_char('a'),
                'request_sha256': [digest_char('b')], 'streams': 1}}


def rank_rows(epoch=3, schema='s5ro-replay-observer/1'):
    rows = {}
    accounting = {}
    for rank in (0, 1, 2, 3):
        rows[rank] = {'schema': schema, 'rank': rank,
                      'engine_epoch': epoch, 'route_generation': 1,
                      'storage_epoch': 1, 'worker_incarnation': 'inc-%d' % rank,
                      'native_start_ns': 1_100, 'native_end_ns': 1_900}
        accounting[rank] = {'schema': 's5-engine-accounting/1', 'rank': rank}
    return rows, accounting


class FakePlacement:
    def __init__(self, result):
        self.result = result
        self.calls = []
        self.error = None

    def run(self, driver, **kwargs):
        self.calls.append({'kwargs': dict(kwargs),
                           'prepare_keys': [row['keys'] for row in
                                            driver.command_keys('row4', 'prepare')]})
        if self.error is not None:
            raise self.error
        return copy.deepcopy(self.result)


def placement_result(epoch=3, run_id='run-1'):
    rows, accounting = rank_rows(epoch)
    return {
        'window': short_window(run_id, epoch),
        'window_route_generation': 1,
        'native_prefill_request': {
            'complete': True, 'plan_sha256': digest_char('a'),
            'request_sha256': [digest_char('b')], 'streams': 1},
        'raw_placement': rows, 'engine_accounting': accounting,
        'raw_sha256': {str(rank): sha(rows[rank]) for rank in rows},
        'raw_graph_receipts': rows, 'engine_accounting_receipts': accounting,
        'per_rank_bounds': {rank: {'drained_streams': 1} for rank in rows},
        'replay_census': {rank: {'replays': 40} for rank in rows},
        'teacher_rows_during_placement': 0,
        'teacher_trace_receipt': {},
        'shipping_path_instrument_disabled': True,
        'stock_prefix': {'native_admission_intent': False},
        'cleanup': {'attempted': True, 'restored': True},
        'calls': [],
    }


class StoreTest(unittest.TestCase):
    def test_write_once_is_private_and_never_replaced(self):
        with tempfile.TemporaryDirectory() as root:
            store = ImmutableStore(root)
            meta = store.write_once('phase.json', {'a': 1})
            path = store.path('phase.json')
            self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
            self.assertEqual(meta['sha256'], sha({'a': 1}))
            with self.assertRaises(QualificationRefused) as caught:
                store.write_once('phase.json', {'a': 2})
            self.assertEqual(caught.exception.code, 'artifact-already-exists:phase.json')
            # The first evidence is untouched.
            record, read_meta = store.read('phase.json')
            self.assertEqual(record, {'a': 1})
            self.assertEqual(read_meta['sha256'], meta['sha256'])
            # The retained-artifact inventory is listable without rewriting.
            self.assertIn('phase.json', store.names())

    def test_names_are_flat_and_symlink_root_refused(self):
        with tempfile.TemporaryDirectory() as root:
            store = ImmutableStore(root)
            with self.assertRaises(QualificationError):
                store.write_once('../escape.json', {})
            target = os.path.join(root, 'target')
            os.makedirs(target)
            link = os.path.join(root, 'link')
            os.symlink(target, link)
            with self.assertRaises(QualificationError):
                ImmutableStore(link)


def row1_manifest(streams=1, prompt_tokens=8, budget=2):
    return {
        'schema': 1, 'workload': 'row1-native-test', 'streams': streams,
        'prompt': {'seed': 17, 'tokens': prompt_tokens},
        'generation_budget': budget,
        'sampling': {'seed': 909, 'temperature': 0.0, 'top_p': 1.0,
                     'ignore_eos': True, 'min_tokens': 'equal_max'},
        'expected': {'output_tokens': budget, 'finish_reason': 'length'},
        'bounds': {'window_s': 2.0, 'window_tokens': 1},
        'timeouts': {'open_s': 5.0, 'pause_s': 5.0, 'read_s': 5.0,
                     'drain_s': 5.0, 'join_s': 5.0, 'finish_s': 5.0,
                     'controller_s': 5.0},
        'cache': {'fresh_namespace_per_run': True},
        'route_plan': ['full', 'full', 'split', 'full'],
    }


def row1_plan(streams=1):
    return {'schema': 's5-native-row1-request-plan/1',
            'workload': row1_manifest(streams=streams), 'run_index': 0}


class FakeCore:
    """Controlled EngineCore utility: the only channel NativeControlClient uses."""

    def __init__(self, engine):
        self.engine = engine
        self.client_count = 1
        self.client_index = 0

    async def call_utility_async(self, utility, payload):
        assert utility == 'flash_s5_native_control'
        return self.engine.handle_native_control(payload)


class FakeEngine:
    """Actual AsyncLLM/EngineCore surface the session drives over the utility."""

    def __init__(self, epoch=9, terminal_full=False):
        from native_overlap_metrics import clock_identity
        self.engine_core = FakeCore(self)
        self.epoch = epoch
        self.route = 'full'
        self.generation = 4
        self.intent = None
        self.paused = False
        self.oracle = False
        self.terminal_full = terminal_full
        self.calls = []
        self.identity = clock_identity()
        assert self.identity['proved'] is True

    async def is_paused(self):
        return self.paused

    async def pause_generation(self, mode='keep', clear_cache=False):
        self.paused = True
        return {'paused': True}

    async def resume_generation(self):
        self.paused = False
        return {'resumed': True}

    def acks(self, bank):
        return [{'typed': 'TransitionAck', 'worker_rank': rank,
                 'engine_epoch': self.epoch,
                 'expected_route_generation': self.generation,
                 'route_generation': self.generation,
                 'applied_route_generation': self.generation,
                 'applied_bank': bank, 'drained_streams': 1,
                 'decode_replay_watermark': 0} for rank in (0, 1, 2, 3)]

    def _clock(self):
        return {'basis': 'clock-monotonic',
                'boot_id': self.identity['boot_id'],
                'time_namespace': self.identity['time_namespace'],
                'sample_ns': time.monotonic_ns()}

    def handle_native_control(self, payload):
        action = payload.get('action')
        self.calls.append(copy.deepcopy(payload))
        reply = {'schema': 1, 'action': action, 'engine_epoch': self.epoch,
                 'bootstrapped': True, 'ranks': [0, 1, 2, 3],
                 'clock': self._clock()}
        if action in ('status', 'snapshot'):
            reply.update({'mode': 'native-overlap', 'route': self.route,
                          'route_generation': self.generation,
                          'admission_intent': copy.deepcopy(self.intent),
                          'owned_work': {'retained_request_row': False},
                          'control_pending': None, 'workers': [],
                          'paused': self.paused})
        elif action == 'census':
            reply.update({'complete': True, 'workers': [], 'census': []})
        elif action == 'enter_split':
            before = self.route
            self.route = 'split'
            reply.update({'route_before': before, 'route_after': self.route,
                          'route_generation_before': self.generation,
                          'route_generation_after': self.generation,
                          'changed': before != self.route,
                          'transition_acks': self.acks(ROUTE_BANK_SPLIT),
                          'bootstrap_acks': [], 'typed_acks': True,
                          'admission_intent': copy.deepcopy(self.intent)})
        elif action == 'leave_split':
            before = self.route
            self.route = 'full'
            self.generation += 1
            reply.update({'route_before': before, 'route_after': self.route,
                          'route_generation_before': self.generation - 1,
                          'route_generation_after': self.generation,
                          'changed': before != self.route,
                          'transition_acks': self.acks(ROUTE_BANK_FULL),
                          'bootstrap_acks': [], 'typed_acks': True,
                          'admission_intent': copy.deepcopy(self.intent)})
        elif action == 'arm_admission':
            phase = payload.get('phase')
            previous = copy.deepcopy(self.intent)
            self.intent = {'schema': 1, 'owner': 'engine-core',
                           'engine_epoch': self.epoch, 'phase': phase,
                           'route': self.route,
                           'route_generation': self.generation,
                           'armed': True, 'seq': 1, 'armed_ns': 10}
            reply.update({'admission_intent': copy.deepcopy(self.intent),
                          'previous_admission_intent': previous,
                          'route': self.route,
                          'route_generation': self.generation})
        elif action == 'disarm_admission':
            previous = copy.deepcopy(self.intent)
            self.intent = None
            reply.update({'admission_intent': None,
                          'previous_admission_intent': previous,
                          'route': self.route,
                          'route_generation': self.generation})
        else:
            raise AssertionError('unexpected native-control action: ' + str(action))
        return reply


class EngineSessionControl:
    """HTTP-side session/worker transport over the real NativeOverlapSession."""

    def __init__(self, engine, packets, *, terminal_full=False,
                 lie_configure=False):
        self.engine = engine
        self.session_impl = NativeOverlapSession(engine)
        self.packets = packets
        self.terminal_full = terminal_full
        self.lie_configure = lie_configure
        self.base_url = 'http://127.0.0.1:9'
        self.session_actions = []
        self.worker_actions = []
        self.events = []
        self.evidence_when_enabled = None
        self.enabled_at_request = None
        self.intent_at_request = None

    def session(self, command, timeout_s=None):
        self.session_actions.append(command['action'])
        self.events.append(command['action'])
        return asyncio.run(self.session_impl.execute(copy.deepcopy(command)))

    def _rows(self, inner_by_rank):
        rows = []
        for rank in (0, 1, 2, 3):
            inner = copy.deepcopy(inner_by_rank[rank])
            rows.append({'schema': 1, 'action': 'qualification',
                         'worker_rank': rank,
                         'engine_epoch': self.engine.epoch,
                         'route_generation': self.engine.generation,
                         'qualification': inner})
        return {'schema': 1, 'action': 'qualification',
                'engine_epoch': self.engine.epoch, 'ranks': [0, 1, 2, 3],
                'workers': rows}

    def worker(self, command, timeout_s=None):
        action = command['worker_action']
        payload = command['payload']
        self.worker_actions.append(action)
        suffix = (':on' if payload.get('enabled') is True
                  else ':off' if payload.get('enabled') is False else '')
        self.events.append(action + suffix)
        if action == 'row1_configure':
            wanted = payload['enabled']
            if wanted and (self.engine.route != 'split'
                           or self.engine.intent is None):
                raise PhaseUnavailable('configure-before-split-arm')
            actual = wanted and not self.lie_configure
            self.engine.oracle = actual
            if not actual and self.terminal_full:
                self.engine.route = 'full'
                self.engine.generation += 1
            inner = {rank: {'run_id': command['run_id'],
                            'engine_epoch': self.engine.epoch,
                            'route_generation': self.engine.generation,
                            'enabled': actual} for rank in (0, 1, 2, 3)}
        elif action == 'row1_evidence':
            self.evidence_when_enabled = self.engine.oracle
            inner = {}
            for packet in self.packets:
                row = copy.deepcopy(packet)
                row['disabled_route_generation'] = self.engine.generation
                inner[row['rank']] = row
        else:
            raise AssertionError('unexpected worker action: ' + action)
        return {'schema': 1, 'action': 'qualification_worker',
                'run_id': command['run_id'],
                'command_seq': command['command_seq'],
                'client_id': command['client_id'],
                'engine_epoch': self.engine.epoch,
                'worker_reply': self._rows(inner)}


class _CompositionHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.0'

    def log_message(self, *args):
        pass

    def _send(self, status, body, content_type='application/json'):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Connection', 'close')
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get('Content-Length', '0'))
        body = json.loads(self.rfile.read(length) or b'{}')
        control = self.server.control
        if self.path == '/v1/completions':
            control.events.append('completion')
            control.enabled_at_request = control.engine.oracle
            control.intent_at_request = copy.deepcopy(control.engine.intent)
            return self._completions(body)
        try:
            if self.path == '/v1/flash_s5_native_overlap':
                reply = control.session(body)
            elif self.path == '/v1/flash_s5_native_qualification_worker':
                reply = control.worker(body)
            else:
                return self._send(404, b'{}')
        except Exception as exc:  # noqa: BLE001 - transport reflects refusal
            return self._send(400, json.dumps(
                {'error': type(exc).__name__, 'detail': str(exc)[:120]}).encode())
        self._send(200, json.dumps(reply).encode())

    def _completions(self, body):
        tokens = [11] * int(body.get('max_tokens', 0))
        prompt_tokens = self.server.prompt_tokens
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Connection', 'close')
        self.end_headers()
        first = {'choices': [{'index': 0, 'token_ids': tokens,
                              'finish_reason': None,
                              'prompt_token_ids': list(range(prompt_tokens)),
                              'text': ''}]}
        self.wfile.write(('data: ' + json.dumps(first) + '\n\n').encode())
        self.wfile.flush()
        # The open boundary must seal before the terminal event: a stream that
        # finished before it was observed cannot prove it was ever open.
        time.sleep(0.25)
        final = {'choices': [{'index': 0, 'token_ids': [],
                              'finish_reason': 'length', 'text': ''}],
                 'usage': {'completion_tokens': len(tokens),
                           'prompt_tokens': prompt_tokens}}
        self.wfile.write(('data: ' + json.dumps(final) + '\n\n'
                          'data: [DONE]\n\n').encode())
        self.wfile.flush()
        self.close_connection = True


class ProducedRow1Test(unittest.TestCase):
    """Composition over real HTTP session/worker/completions client seams."""

    def _run(self, root, *, terminal_full=False, lie_configure=False,
             request_runner=None):
        k = row1
        engine = FakeEngine(terminal_full=terminal_full)
        packets = [k_packet(k, rank, epoch=engine.epoch)
                   for rank in (0, 1, 2, 3)]
        control = EngineSessionControl(
            engine, packets, terminal_full=terminal_full,
            lie_configure=lie_configure)
        httpd = http.server.ThreadingHTTPServer(
            ('127.0.0.1', 0), _CompositionHandler)
        httpd.control = control
        httpd.prompt_tokens = 8
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(httpd.shutdown)
        self.addCleanup(httpd.server_close)
        url = 'http://127.0.0.1:%d' % httpd.server_address[1]
        frozen = driver_module.freeze_row1_plan(
            row1_plan(), run_id='run-1', model='m')
        producer = NativeRow1Producer(
            plan=frozen, url=url, model='m', request_runner=request_runner)
        driver = QualificationDriver(
            root, run_id='run-1', model='m', engine_epoch=engine.epoch,
            url=url, key='test-key', row1_producer=producer, timeout_s=5.0)
        return driver, control, engine

    def test_real_control_session_and_http_compose_the_full_lifecycle(self):
        with tempfile.TemporaryDirectory() as root:
            driver, control, engine = self._run(root)
            result = driver.collect_phase('row1')
            # Explicit lifecycle: arm only after a paused split, one real
            # request, then query only after the engine ack disabled the oracle.
            self.assertEqual(control.session_actions, [
                'prepare', 'pause_enter_split', 'arm_admission', 'resume',
                'status', 'pause_enter_split', 'status', 'disarm_admission',
                'leave_full', 'status'])
            self.assertEqual(control.worker_actions,
                             ['row1_configure', 'row1_configure',
                              'row1_evidence'])
            # The real engine-side intent was armed and the oracle enabled
            # before the actual /v1/completions request was served.
            self.assertTrue(control.enabled_at_request)
            self.assertEqual(control.intent_at_request['phase'], 'row1-native')
            self.assertLess(control.events.index('arm_admission'),
                            control.events.index('completion'))
            self.assertLess(control.events.index('completion'),
                            control.events.index('row1_configure:off'))
            self.assertFalse(control.evidence_when_enabled)
            self.assertFalse(engine.oracle)
            self.assertEqual(engine.route, 'full')
            evidence = result['evidence']
            self.assertEqual(evidence['native_phase'], 'row1-native')
            self.assertEqual([row['rank'] for row in evidence['workers']],
                             [0, 1, 2, 3])
            self.assertTrue(all(row['enabled'] is False
                                for row in evidence['workers']))
            self.assertTrue(all(row['disabled_route_generation'] == 5
                                for row in evidence['workers']))
            self.assertEqual(evidence['lifecycle']['arm']['armed'], True)
            self.assertEqual(evidence['lifecycle']['prepare']['route'], 'full')
            self.assertEqual(evidence['lifecycle']['pause_enter_split']['route'],
                             'split')
            self.assertTrue(evidence['lifecycle']['requests']['complete'])
            self.assertEqual(evidence['lifecycle']['requests']['streams'], 1)
            self.assertTrue(evidence['lifecycle']['requests']
                            ['streams_detail'][0]['valid'])
            self.assertTrue(evidence['cleanup']['restored'])
            self.assertTrue(result['complete'])
            self.assertEqual(result['row1_sha256'], sha(evidence))

    def test_terminal_full_keeps_the_qualified_generation(self):
        with tempfile.TemporaryDirectory() as root:
            driver, control, engine = self._run(root, terminal_full=True)
            result = driver.collect_phase('row1')
            workers = result['evidence']['workers']
            self.assertTrue(all(row['route_generation'] == 5 for row in workers))
            self.assertTrue(all(row['disabled_route_generation'] == 6
                                for row in workers))
            self.assertTrue(result['evidence']['terminal']['terminal_full'])
            self.assertTrue(result['evidence']['cleanup']['terminal_full'])
            self.assertNotIn('leave_full', control.session_actions)
            self.assertIn('terminal_resume',
                          result['evidence']['lifecycle'])
            self.assertEqual(engine.route, 'full')

    def test_request_failure_still_disables_and_returns_full(self):
        with tempfile.TemporaryDirectory() as root:
            driver, control, engine = self._run(
                root, request_runner=lambda request: {
                    'complete': False, 'reason': 'bounded-test-refusal'})
            # This runner seam is still behind the real armed/configured state.
            with self.assertRaises(QualificationRefused) as caught:
                driver.collect_phase('row1')
            self.assertEqual(caught.exception.code,
                             'row1-request-runner-incomplete')
            self.assertFalse(engine.oracle)
            self.assertEqual(engine.route, 'full')
            cleanup = caught.exception.cleanup
            self.assertTrue(cleanup['restored'])
            self.assertIn('disable', cleanup['steps'])
            self.assertIn('disarm', cleanup['steps'])
            refusal, _ = driver.store.read('refusal.row1.json')
            self.assertEqual(refusal['refusal'],
                             'row1-request-runner-incomplete')
            self.assertTrue(
                refusal['evidence']['producer_evidence']['cleanup']
                ['restored'])

    def test_missing_request_plan_never_arms(self):
        control = Row1Control()
        with tempfile.TemporaryDirectory() as root:
            driver = QualificationDriver(
                root, run_id='run-1', model='m', engine_epoch=3,
                control=control)
            with self.assertRaises(PhaseUnavailable) as caught:
                driver.collect_phase('row1')
            self.assertEqual(caught.exception.code,
                             'missing-row1-request-plan')
            self.assertEqual(control.arm_calls, 0)
            self.assertEqual(control.worker_calls, [])

    def test_wrong_request_digest_is_refused_before_any_engine_action(self):
        plan = row1_plan()
        plan['request_sha256'] = ['f' * 64]
        control = Row1Control()
        with tempfile.TemporaryDirectory() as root:
            driver = QualificationDriver(
                root, run_id='run-1', model='m', engine_epoch=3,
                control=control)
            with self.assertRaises(QualificationRefused) as caught:
                driver.collect_phase('row1', request_plan=plan)
            self.assertEqual(caught.exception.code,
                             'row1-request-plan-requests-differ')
            self.assertEqual(control.arm_calls, 0)
            self.assertEqual(control.worker_calls, [])

    def test_lieing_configure_ack_is_refused_after_real_disable(self):
        with tempfile.TemporaryDirectory() as root:
            driver, control, engine = self._run(root, lie_configure=True)
            with self.assertRaises(QualificationRefused) as caught:
                driver.collect_phase('row1')
            self.assertEqual(caught.exception.code,
                             'row1-configure-ack-differs:rank-0')
            self.assertFalse(engine.oracle)
            self.assertEqual(engine.route, 'full')
            self.assertTrue(caught.exception.cleanup['restored'])


class Row1Test(unittest.TestCase):
    def test_missing_admission_transport_is_inconclusive_and_retained(self):
        with tempfile.TemporaryDirectory() as root:
            driver = QualificationDriver(
                root, run_id='run-1', model='m', engine_epoch=3,
                control=types.SimpleNamespace())
            with self.assertRaises(PhaseUnavailable) as caught:
                driver.collect_phase('row1')
            self.assertEqual(caught.exception.code,
                             'missing-native-admission-transport')
            refusal, _ = driver.store.read('refusal.row1.json')
            self.assertEqual(refusal['refusal'],
                             'missing-native-admission-transport')
            self.assertTrue(refusal['retained'])
            self.assertEqual(refusal['retries'], 0)
            self.assertFalse(refusal['picked'])

    def _provider(self, packets):
        return lambda request: {
            'workers': copy.deepcopy(packets),
            'configure_acks': copy.deepcopy(row1_acks(True)),
            'disable_acks': copy.deepcopy(row1_acks(False))}

    def test_provider_seam_retains_flat_packets_without_worker_actions(self):
        k = row1
        packets = [k_packet(k, rank) for rank in (0, 1, 2, 3)]
        control = Row1Control()
        calls = []

        def provider(request):
            calls.append(copy.deepcopy(request))
            return self._provider(packets)(request)

        with tempfile.TemporaryDirectory() as root:
            driver = QualificationDriver(
                root, run_id='run-1', model='m', engine_epoch=3,
                control=control, row1_provider=provider)
            result = driver.collect_phase('row1')
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]['schema'], driver_module.ROW1_SCHEMA)
            self.assertEqual(control.worker_calls, [])
            self.assertEqual(control.arm_calls, 1)
            self.assertEqual(control.disarm_calls, 1)
            evidence = result['evidence']
            self.assertEqual(evidence['workers'], packets)
            self.assertTrue(all(row['enabled'] is False
                                for row in evidence['workers']))
            self.assertTrue(evidence['disabled_on_every_rank'])
            self.assertEqual(result['row1_sha256'], sha(evidence))
            self.assertTrue(result['complete'])

    def test_packet_with_the_oracle_enabled_is_refused(self):
        k = row1
        packets = [k_packet(k, rank, enabled=True) for rank in (0, 1, 2, 3)]
        control = Row1Control()
        with tempfile.TemporaryDirectory() as root:
            driver = QualificationDriver(
                root, run_id='run-1', model='m', engine_epoch=3,
                control=control, row1_provider=self._provider(packets))
            with self.assertRaises(QualificationRefused) as caught:
                driver.collect_phase('row1')
            self.assertEqual(caught.exception.code,
                             'row1-packet-oracle-not-disabled:rank-0')
            self.assertEqual(control.disarm_calls, 1)

    def test_foreign_packet_identity_is_refused_after_cleanup(self):
        k = row1
        packets = [k_packet(k, rank) for rank in (0, 1, 2, 3)]
        packets[2]['engine_epoch'] = 4
        control = Row1Control()
        with tempfile.TemporaryDirectory() as root:
            driver = QualificationDriver(
                root, run_id='run-1', model='m', engine_epoch=3,
                control=control, row1_provider=self._provider(packets))
            with self.assertRaises(QualificationRefused) as caught:
                driver.collect_phase('row1')
            self.assertEqual(caught.exception.code,
                             'row1-packet-epoch-differs:rank-2')
            self.assertEqual(control.disarm_calls, 1)
            refusal, _ = driver.store.read('refusal.row1.json')
            self.assertEqual(refusal['refusal'],
                             'row1-packet-epoch-differs:rank-2')
            raw = refusal['evidence']['partial']['provider_result']
            self.assertEqual(raw['workers'][2]['engine_epoch'], 4)

    def test_missing_rank_is_refused(self):
        k = row1
        packets = [k_packet(k, rank) for rank in (0, 1, 2)]
        control = Row1Control()
        with tempfile.TemporaryDirectory() as root:
            driver = QualificationDriver(
                root, run_id='run-1', model='m', engine_epoch=3,
                control=control, row1_provider=self._provider(packets))
            with self.assertRaises(QualificationRefused) as caught:
                driver.collect_phase('row1')
            self.assertEqual(caught.exception.code,
                             'row1-provider-ranks-incomplete')
            self.assertEqual(control.disarm_calls, 1)

    def test_empty_records_are_retained_inconclusive_never_pass(self):
        k = row1
        packets = [k_packet(k, rank, records=[], layer_count=0)
                   for rank in (0, 1, 2, 3)]
        with tempfile.TemporaryDirectory() as root:
            driver = QualificationDriver(
                root, run_id='run-1', model='m', engine_epoch=3,
                control=Row1Control(), row1_provider=self._provider(packets))
            result = driver.collect_phase('row1')
            self.assertFalse(result['complete'])
            self.assertIn('row1-records-empty', result['missing'])
            for rank in (0, 1, 2, 3):
                self.assertIn('row1-rank-comparisons-empty:%d' % rank,
                              result['missing'])
            self.assertEqual(result['evidence']['comparisons'], 0)
            self.assertNotIn('verdict', result)
            retained, _ = driver.store.read('row1.json')
            self.assertEqual(retained['evidence'], result['evidence'])

    def test_provider_vm_receipt_cannot_be_engine_evidence(self):
        control = Row1Control()
        with tempfile.TemporaryDirectory() as root:
            driver = QualificationDriver(
                root, run_id='run-1', model='m', engine_epoch=3,
                control=control,
                row1_provider=lambda request: {
                    'workers': [], 'configure_acks': [],
                    'disable_acks': [],
                    'vm_receipt_used_as_engine_evidence': True})
            with self.assertRaises(QualificationRefused) as caught:
                driver.collect_phase('row1')
            self.assertEqual(caught.exception.code,
                             'row1-vm-receipt-as-engine-evidence')
            self.assertEqual(control.disarm_calls, 1)
            refusal, _ = driver.store.read('refusal.row1.json')
            self.assertEqual(refusal['refusal'],
                             'row1-vm-receipt-as-engine-evidence')

    def test_a_second_collection_never_queries_the_oracle_again(self):
        k = row1
        packets = [k_packet(k, rank) for rank in (0, 1, 2, 3)]
        control = Row1Control()
        calls = []

        def provider(request):
            calls.append(copy.deepcopy(request))
            return self._provider(packets)(request)

        with tempfile.TemporaryDirectory() as root:
            driver = QualificationDriver(
                root, run_id='run-1', model='m', engine_epoch=3,
                control=control, row1_provider=provider)
            driver.collect_phase('row1')
            with self.assertRaises(QualificationRefused) as caught:
                driver.collect_phase('row1')
            self.assertEqual(caught.exception.code, 'phase-already-recorded')
            self.assertEqual(len(calls), 1)
            self.assertEqual(control.arm_calls, 1)
            refusal, _ = driver.store.read('refusal.row1.json')
            self.assertEqual(refusal['refusal'], 'phase-already-recorded')


class CandidateTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = ImmutableStore(self.temp.name)
        self.control = ReplyControl()
        self.artifact = teacher_evidence(None)
        retained_teacher(self.store, self.artifact)
        self.driver = QualificationDriver(
            self.store, run_id='run-1', model='m', engine_epoch=3,
            control=self.control)

    def tearDown(self):
        self.temp.cleanup()

    def test_candidate_reads_the_retained_artifact_without_teacher_runs(self):
        result = self.driver.collect_phase(
            'rule4_candidate', artifact_sha256=sha(self.artifact))
        self.assertEqual(result['evidence']['teacher_runs'], 0)
        self.assertTrue(result['evidence']['selected_from_same_artifact'])
        self.assertEqual(result['artifact_sha256'], sha(self.artifact))
        self.assertEqual(self.control.session_calls, [])
        retained, _ = self.store.read('rule4_candidate.json')
        self.assertEqual(retained['evidence']['artifact'], self.artifact)

    def test_recollect_and_mismatch_are_refused(self):
        with self.assertRaises(QualificationRefused) as caught:
            self.driver.collect_phase('rule4_candidate',
                                      artifact_sha256=sha(self.artifact),
                                      recollect=True)
        self.assertEqual(caught.exception.code, 'candidate-recollect-refused')
        with self.assertRaises(QualificationRefused) as caught:
            self.driver.collect_phase('rule4_candidate',
                                      artifact_sha256=digest_char('f'))
        self.assertEqual(caught.exception.code, 'retained-teacher-artifact-not-found')
        self.assertEqual(self.control.session_calls, [])

    def test_ambiguous_retention_is_never_picked(self):
        self.store.write_once('copy.json', {
            'schema': 1, 'phase': 'rule4_controls', 'kind': 'teacher',
            'run_id': 'run-1', 'artifact': self.artifact})
        with self.assertRaises(QualificationRefused) as caught:
            self.driver.collect_phase('rule4_candidate',
                                      artifact_sha256=sha(self.artifact))
        self.assertEqual(caught.exception.code,
                         'retained-teacher-artifact-ambiguous')


class ControlsTest(unittest.TestCase):
    def run_controls(self, *, with_frontend=True, evidence=None):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        control = ReplyControl()
        if not with_frontend:
            control.frontend = None
        workload = teacher_workload()
        instrument = FakeInstrument(evidence or teacher_evidence(workload))
        driver = QualificationDriver(
            temp.name, run_id='run-1', model='m', engine_epoch=3,
            control=control, teacher_workload=workload,
            teacher_instrument=instrument)

        def fake_run(sequence):
            sequence.instrument.requests([], [], 0)
            receipt=sequence.driver._frontend('rule4_controls',True,sequence.instrument.plan)
            return {'teacher_forced': sequence.instrument.finish([]),
                    'native_frontend_calls':[dict(repeat=0,receipt=receipt)]}

        from e3_native_teacher_sequence import NativeTeacherSequence
        with mock.patch.object(NativeTeacherSequence, 'run', fake_run):
            result = driver.collect_phase('rule4_controls')
        return driver, control, instrument, result

    def test_controls_are_one_actual_sequence_retained_once(self):
        driver, control, instrument, result = self.run_controls()
        self.assertEqual(result['complete'], True)
        self.assertEqual(result['artifact_sha256'],
                         sha(result['artifact']))
        self.assertEqual(result['evidence']['schema'],
                         driver_module.TEACHER_SCHEMA)
        self.assertEqual(set(result['artifact']['runs']),
                         set(driver_module.TEACHER_SEQUENCE))
        self.assertEqual(len(result['artifact']['fixed_positions']), 256)
        self.assertEqual(result['flags']['teacher_runs'], 1)
        self.assertEqual(result['flags']['retries'], 0)
        # The plan was owned once per run and cleared afterwards.
        self.assertTrue(result['frontend_plan_calls'])
        self.assertTrue(control.frontend_calls)
        self.assertTrue(all(call['enabled'] for call in control.frontend_calls[:-1]))
        self.assertEqual(control.frontend_calls[-1]['enabled'], False)
        self.assertIsNone(control.frontend_calls[-1]['plan'])
        self.assertEqual(result['frontend_status']['enabled'], False)
        # One retained raw artifact, no cached verdict used as proof.
        retained, _ = driver.store.read('rule4_controls.json')
        self.assertEqual(retained['artifact'], result['artifact'])
        self.assertEqual(retained['kind'], 'teacher')
        self.assertNotIn('grade', retained['artifact'])

    def test_missing_frontend_owner_is_reported_not_faked(self):
        with self.assertRaisesRegex(PhaseUnavailable,'native-teacher-frontend-owner-unavailable'):
            self.run_controls(with_frontend=False)

    def test_wrong_sequence_route_or_coverage_is_refused(self):
        workload = teacher_workload()
        bad = teacher_evidence(workload)
        bad['runs']['split-B']['route'] = 'full'
        with self.assertRaises(QualificationRefused) as caught:
            self.run_controls(evidence=bad)
        self.assertEqual(caught.exception.code, 'teacher-route-differs:split-B')

        bad = teacher_evidence(workload)
        bad['runs']['full-A1']['row_mapping']['records'] = []
        with self.assertRaises(PhaseUnavailable) as caught:
            self.run_controls(evidence=bad)
        self.assertEqual(caught.exception.code,
                         'teacher-row-coverage-incomplete:full-A1')

        bad = teacher_evidence(workload)
        bad['runs']['full-C']['row_mapping']['overflow'] = True
        with self.assertRaises(QualificationRefused) as caught:
            self.run_controls(evidence=bad)
        self.assertEqual(caught.exception.code, 'teacher-collector-overflow:full-C')

    def test_no_teacher_workload_is_unavailable(self):
        with tempfile.TemporaryDirectory() as root:
            driver = QualificationDriver(
                root, run_id='run-1', model='m', engine_epoch=3,
                control=ReplyControl())
            with self.assertRaises(PhaseUnavailable) as caught:
                driver.collect_phase('rule4_controls')
            self.assertEqual(caught.exception.code, 'missing-teacher-workload')


class Row4Test(unittest.TestCase):
    def make_driver(self, root, placement, *, native_read=True,
                    prefill=None, control=None, placement_epoch=3):
        artifact = teacher_evidence(None)
        retained_teacher(ImmutableStore(root), artifact)
        read = ((lambda request: {'bytes': driver_module.NATIVE_READ_BYTES,
                                  'basis': 'actual-native-read',
                                  'completed': True,
                                  'buffer_sha256': digest_char('d'),
                                  'source': 'retained-split-bank',
                                  'window_id': request['window_id']})
                if native_read else None)
        return artifact, QualificationDriver(
            root, run_id='run-1', model='m', engine_epoch=placement_epoch,
            control=control or ReplyControl(epoch=placement_epoch),
            placement=placement, native_read_provider=read,
            native_prefill_provider=prefill, route_banks=(0, 1))

    def test_short_window_keeps_every_rank_and_raw_download_separately(self):
        with tempfile.TemporaryDirectory() as root:
            placement = FakePlacement(placement_result())
            artifact, driver = self.make_driver(root, placement)
            result = driver.collect_phase(
                'row4', artifact_sha256=sha(artifact),
                row1_sha256=digest_char('e'), window_id='window-1')
            self.assertEqual(result['complete'], False)
            self.assertEqual(result['missing'],
                             ['native-report-read-unavailable', 'native-prefill-ledger-unproved'])
            evidence = result['evidence']
            self.assertEqual(set(evidence['raw_placement']), {0, 1, 2, 3})
            self.assertEqual(set(evidence['engine_accounting']), {0, 1, 2, 3})
            self.assertEqual(evidence['raw_placement'][2]['rank'], 2)
            for rank in (0, 1, 2, 3):
                self.assertEqual(evidence['raw_sha256'][str(rank)],
                                 sha(evidence['raw_placement'][rank]))
                self.assertIsNot(evidence['raw_graph_receipts'][rank],
                                 evidence['engine_accounting_receipts'][rank])
            self.assertEqual(evidence['native_prefill_attribution']['state'],
                             'unavailable')
            self.assertTrue(evidence['shipping_path_instrument_disabled'])
            self.assertEqual(evidence['window']['replays'], 40)
            self.assertFalse(evidence['window']['teacher_overlap'])
            # The placement record is not a teacher/probe receipt.
            self.assertEqual(evidence['teacher_trace_receipt'], {})

    def test_foreign_run_epoch_window_and_long_window_are_refused(self):
        cases = [
            ('foreign-placement-run', {'run_id': 'other'}),
            ('foreign-placement-epoch', {'engine_epoch': 4}),
            ('placement-window-is-not-short', {'kind': 'teacher'}),
            ('placement-replay-census-differs', {'replays': 8}),
            ('placement-overlaps-teacher-window', {'teacher_overlap': True}),
            ('placement-native-phase-differs', {'native_phase': 'row1-native'}),
            ('placement-admission-not-proved', {'admission': None}),
            ('placement-admission-not-proved',
             {'admission': {'phase': 'w2-overlap', 'armed': False}}),
        ]
        for code, patch in cases:
            result = placement_result()
            result['window'].update(patch)
            with tempfile.TemporaryDirectory() as root:
                placement = FakePlacement(result)
                artifact, driver = self.make_driver(root, placement)
                with self.assertRaises(QualificationRefused) as caught:
                    driver.collect_phase(
                        'row4', artifact_sha256=sha(artifact),
                        row1_sha256=digest_char('e'), window_id='window-1')
                self.assertEqual(caught.exception.code, code)
                refusal, _ = driver.store.read('refusal.row4.json')
                self.assertEqual(refusal['refusal'], code)

    def test_missing_rank_raw_digest_or_native_read_is_refused(self):
        result = placement_result()
        del result['raw_placement'][3]
        with tempfile.TemporaryDirectory() as root:
            placement = FakePlacement(result)
            artifact, driver = self.make_driver(root, placement)
            with self.assertRaises(QualificationRefused) as caught:
                driver.collect_phase('row4', artifact_sha256=sha(artifact),
                                     row1_sha256=digest_char('e'),
                                     window_id='window-1')
            self.assertEqual(caught.exception.code, 'raw-placement-missing-rank')

        result = placement_result()
        result['raw_sha256']['2'] = digest_char('f')
        with tempfile.TemporaryDirectory() as root:
            placement = FakePlacement(result)
            artifact, driver = self.make_driver(root, placement)
            with self.assertRaises(QualificationRefused) as caught:
                driver.collect_phase('row4', artifact_sha256=sha(artifact),
                                     row1_sha256=digest_char('e'),
                                     window_id='window-1')
            self.assertEqual(caught.exception.code, 'raw-placement-digest-differs:2')

        with tempfile.TemporaryDirectory() as root:
            placement = FakePlacement(placement_result())
            artifact, driver = self.make_driver(root, placement,
                                               native_read=False)
            out = driver.collect_phase('row4', artifact_sha256=sha(artifact),
                                       row1_sha256=digest_char('e'),
                                       window_id='window-1')
            self.assertFalse(out['complete'])
            self.assertIn('native-report-read-unavailable', out['missing'])
            self.assertEqual(out['evidence']['native_read']['state'],
                             'unavailable')
            self.assertEqual(set(out['evidence']['raw_placement']),
                             {0, 1, 2, 3})

    def test_foreign_window_and_native_admission_intent_are_refused(self):
        with tempfile.TemporaryDirectory() as root:
            placement = FakePlacement(placement_result())
            artifact, driver = self.make_driver(root, placement)
            with self.assertRaises(QualificationRefused) as caught:
                driver.collect_phase('row4', artifact_sha256=sha(artifact),
                                     row1_sha256=digest_char('e'),
                                     window_id='other-window')
            self.assertEqual(caught.exception.code, 'foreign-placement-window')

        result = placement_result()
        result['stock_prefix'] = {'native_admission_intent': True,
                                  'prepare_keys': []}
        with tempfile.TemporaryDirectory() as root:
            placement = FakePlacement(result)
            artifact, driver = self.make_driver(root, placement)
            with self.assertRaises(QualificationRefused) as caught:
                driver.collect_phase('row4', artifact_sha256=sha(artifact),
                                     row1_sha256=digest_char('e'),
                                     window_id='window-1')
            self.assertEqual(caught.exception.code,
                             'stock-prefix-native-admission-intent')

    def test_file_provider_cannot_replace_missing_native_read_and_raw_is_retained(self):
        result = placement_result()
        with tempfile.TemporaryDirectory() as root:
            artifact = teacher_evidence(None)
            retained_teacher(ImmutableStore(root), artifact)
            driver = QualificationDriver(
                root, run_id='run-1', model='m', engine_epoch=3,
                control=ReplyControl(), placement=FakePlacement(result),
                native_read_provider=lambda request: self.fail('legacy file provider called'),
                native_prefill_provider=lambda request: self.fail('legacy prefill flag called'),
                route_banks=(0, 1))
            out = driver.collect_phase('row4', artifact_sha256=sha(artifact),
                row1_sha256=digest_char('e'), window_id='window-1')
            self.assertFalse(out['complete'])
            self.assertIn('native-report-read-unavailable', out['missing'])
            self.assertEqual(set(out['evidence']['raw_placement']), {0,1,2,3})

    def test_teacher_trace_during_placement_is_reported(self):
        result = placement_result()
        result['teacher_rows_during_placement'] = 2
        with tempfile.TemporaryDirectory() as root:
            placement = FakePlacement(result)
            artifact, driver = self.make_driver(root, placement)
            out = driver.collect_phase('row4', artifact_sha256=sha(artifact),
                                       row1_sha256=digest_char('e'),
                                       window_id='window-1')
            self.assertIn('placement-teacher-trace-overlapped', out['missing'])
            self.assertEqual(out['evidence']['teacher_trace_receipt'], {})

    def test_missing_retained_artifact_or_window_id_is_unavailable(self):
        with tempfile.TemporaryDirectory() as root:
            artifact, driver = self.make_driver(
                root, FakePlacement(placement_result()))
            with self.assertRaises(PhaseUnavailable) as caught:
                driver.collect_phase('row4', artifact_sha256=digest_char('f'),
                                     row1_sha256=digest_char('e'),
                                     window_id='window-1')
            self.assertEqual(caught.exception.code,
                             'retained-teacher-artifact-not-found')
            with self.assertRaises(PhaseUnavailable) as caught:
                driver.collect_phase('row4', artifact_sha256=sha(artifact),
                                     row1_sha256=digest_char('e'))
            self.assertEqual(caught.exception.code,
                             'missing-placement-window-id')

    def test_retained_bank_file_is_never_a_native_report_read(self):
        reader = RetainedBankReader('/unused-no-file-access')
        with self.assertRaises(QualificationRefused) as caught:
            reader({'bytes': driver_module.NATIVE_READ_BYTES})
        self.assertEqual(caught.exception.code,
                         'bank-file-is-not-native-correlation-report')


class TransportTest(unittest.TestCase):
    def test_transport_timeout_is_a_refusal_never_a_verdict(self):
        class Slow:
            def __init__(self, *args, **kwargs):
                pass

            def request(self, *args, **kwargs):
                raise socket.timeout('slow')

            def close(self):
                pass

        with mock.patch.object(http.client, 'HTTPConnection', Slow):
            control = driver_module.LoopbackControl(
                'http://127.0.0.1:9', key='not-a-real-key', timeout_s=0.1)
            with self.assertRaises(QualificationRefused) as caught:
                control.session({'schema': 1, 'action': 'status',
                                 'run_id': 'run-1'})
        self.assertEqual(caught.exception.code,
                         'transport-failed:TimeoutError')

    def test_foreign_identity_and_missing_adapter_echo_are_refused(self):
        class Response:
            def __init__(self, status, body):
                self.status = status
                self._body = body

            def read(self, size):
                return self._body

        class Fake:
            def __init__(self, status, body):
                self.status = status
                self.body = body

            def request(self, *args, **kwargs):
                pass

            def getresponse(self):
                return Response(self.status, self.body)

            def close(self):
                pass

        def exchange(status, body, action='qualification_worker'):
            with mock.patch.object(http.client, 'HTTPConnection',
                                   lambda *a, **k: Fake(status, body)):
                control = driver_module.LoopbackControl(
                    'http://127.0.0.1:9', key='not-a-real-key')
                return control.worker({'schema': 1, 'action': action,
                                       'run_id': 'run-1', 'client_id': 'c',
                                       'command_seq': 1, 'engine_epoch': 3})

        good = json.dumps({'schema': 1, 'action': 'qualification_worker',
                           'run_id': 'run-1', 'command_seq': 1,
                           'client_id': 'c'}).encode()
        self.assertEqual(exchange(200, good)['client_id'], 'c')
        for status, body, code in [
                (409, b'{}', 'controller-refusal'),
                (500, b'{}', 'http-status-500'),
                (200, good.replace(b'run-1', b'run-2'),
                 'response-identity-differs'),
                (200, json.dumps({'schema': 1,
                                  'action': 'qualification_worker',
                                  'run_id': 'run-1', 'command_seq': 1}).encode(),
                 'response-identity-differs')]:
            with self.assertRaises(QualificationRefused) as caught:
                exchange(status, body)
            self.assertEqual(caught.exception.code, code, (status, body))


class IsolationTest(unittest.TestCase):
    def test_overlapping_phase_is_refused_and_retained(self):
        started = threading.Event()
        release = threading.Event()
        k = row1

        def blocking(request):
            started.set()
            release.wait(5)
            return {'workers': [k_packet(k, rank) for rank in (0, 1, 2, 3)],
                    'configure_acks': row1_acks(True),
                    'disable_acks': row1_acks(False)}

        with tempfile.TemporaryDirectory() as root:
            driver = QualificationDriver(
                root, run_id='run-1', model='m', engine_epoch=3,
                control=Row1Control(), row1_provider=blocking)
            worker = threading.Thread(target=driver.collect_phase,
                                      args=('row1',))
            worker.start()
            self.assertTrue(started.wait(5))
            try:
                with self.assertRaises(PhaseConflict) as caught:
                    driver.collect_phase('row4')
                self.assertEqual(caught.exception.code, 'phase-overlap-refused')
                refusal, _ = driver.store.read('refusal.row4.json')
                self.assertEqual(refusal['refusal'], 'phase-overlap-refused')
            finally:
                release.set()
                worker.join(5)
            self.assertFalse(worker.is_alive())

    def test_unknown_phase_is_an_error(self):
        with tempfile.TemporaryDirectory() as root:
            driver = QualificationDriver(
                root, run_id='run-1', model='m', engine_epoch=3,
                url='http://127.0.0.1:9', key='not-a-real-key')
            with self.assertRaises(QualificationError) as caught:
                driver.collect_phase('not-a-phase')
            self.assertEqual(str(caught.exception), 'phase-invalid')


# -- placement runner --------------------------------------------------------

class FakeState:
    def __init__(self, index, prompt_tokens=1024):
        self.index = index
        self.value = 0
        self.lock = threading.Lock()
        self.progress = threading.Event()
        self.tokens = []
        self.prompt_sha256 = digest_char('1')
        self.prompt_chars = 32
        self.request_sha256 = digest_char('2')
        self.prompt_tokens = prompt_tokens
        self.prompt_token_ids_len = prompt_tokens
        self.send_at_ns = time.monotonic_ns()
        self.first_advance_ns = 1234
        self.finish_at_ns = None
        self.finish_reason = None
        self.error = None
        self.error_at_ns = None
        self.malformed = None

    def count(self):
        return self.value


def prefill_plan():
    return {'schema': 's5-native-prefill-request-plan/1',
            'workload': row1_manifest(), 'run_index': 0}


class PlacementDriver:
    """Minimal driver surface consumed by NativePlacementRunner."""

    def __init__(self, epoch=3, fail_at=None, leave_restored=True):
        self.engine_epoch = epoch
        self.control = object()
        self.url = 'http://127.0.0.1:9'
        self._teacher_window = {'kind': 'teacher', 'ended_ns': 500}
        self.calls = []
        self.session_fields = []
        self.worker_calls = []
        self.frontend_calls = []
        self.fail_at = fail_at
        self.leave_restored = leave_restored
        self.history = []

    def route_banks(self):
        return 0, 1

    def command_keys(self, phase=None, action=None):
        rows = [{'phase': 'row4', 'action': 'prepare',
                 'keys': ['action', 'client_id', 'command_seq', 'engine_epoch',
                          'run_id', 'schema', 'target_route']}]
        return [row for row in rows
                if (phase is None or row['phase'] == phase)
                and (action is None or row['action'] == action)]

    def _ack_rows(self, bank):
        return [{'typed': 'TransitionAck', 'worker_rank': rank,
                 'engine_epoch': self.engine_epoch, 'applied_bank': bank,
                 'applied_route_generation': 1, 'drained_streams': 8}
                for rank in (0, 1, 2, 3)]

    def _session(self, phase_name, action, *, require_epoch=True, **fields):
        self.calls.append(action)
        self.session_fields.append((action, dict(fields)))
        if self.fail_at == action:
            raise QualificationRefused('injected-' + action)
        split_admission = {
            'phase': 'w2-overlap', 'armed': True,
            'admission_intent': {
                'schema': 1, 'owner': 'engine-core',
                'engine_epoch': self.engine_epoch, 'phase': 'w2-overlap',
                'route': 'split', 'route_generation': 1, 'armed': True,
                'seq': 1, 'armed_ns': 900},
            'previous_admission_intent': None,
            'route': 'split', 'route_generation': 1}
        replies = {
            'prepare': {'route': 'full', 'engine_epoch': self.engine_epoch,
                        'typed_acks': True, 'transition_acks': self._ack_rows(0)},
            'pause_enter_split': {'route': 'split', 'engine_epoch': self.engine_epoch,
                                  'phase': 'w2-overlap',
                                  'admission': split_admission,
                                  'route_generation_after': 1,
                                  'typed_acks': True,
                                  'transition_acks': self._ack_rows(1)},
            'overlap_begin': {'route': 'split', 'engine_epoch': self.engine_epoch,
                              'measure_start_ns': 1000,
                              'clock_proof': {'proved': True}},
            'resume': {'resumed': True, 'already_running': False},
            'overlap_end': {'route': 'split', 'engine_epoch': self.engine_epoch,
                            'measure_end_ns': 2000, 'decode_progress': 40},
            'leave_full': ({'route': 'full', 'typed_acks': True,
                            'admission': {
                                'phase': 'full-prefill-solo', 'armed': False,
                                'admission_intent': None,
                                'previous_admission_intent': {
                                    'phase': 'w2-overlap', 'armed': True}},
                            'transition_acks': self._ack_rows(0),
                            'worker_drain': [{'worker_rank': rank,
                                              'drained_streams': 8}
                                             for rank in (0, 1, 2, 3)]}
                           if self.leave_restored else
                           {'route': 'split', 'typed_acks': False,
                            'transition_acks': [], 'worker_drain': []}),
        }
        return replies[action]

    def _qualification_rows(self, action):
        rows = []
        for rank in (0, 1, 2, 3):
            record = {'engine_epoch': self.engine_epoch, 'route_generation': 1,
                      'storage_epoch': 1, 'worker_incarnation': 'inc-%d' % rank,
                      'drained_streams': 8, 'route': 'full',
                      'replay_census': {'replays': 40}}
            if action == 'observer_begin':
                record['route'] = 'split'
            if action == 'observer_end':
                record['route'] = 'split'
                record['receipt'] = {'schema': 's5ro-replay-observer/1',
                                     'rank': rank, 'bounds': {'lo': 1, 'hi': 2}}
                record['engine_accounting'] = {'schema': 's5-engine-accounting/1',
                                               'rank': rank}
            if action == 'teacher_evidence':
                record['positions'] = []
            rows.append({'worker_rank': rank, 'qualification': record})
        return {'schema': 1, 'action': 'qualification',
                'engine_epoch': self.engine_epoch, 'workers': rows}

    def _worker(self, phase, action, payload):
        self.worker_calls.append(action)
        if self.fail_at == action:
            raise QualificationRefused('injected-' + action)
        return self._qualification_rows(action)

    def _frontend(self, phase, enabled, plan):
        self.frontend_calls.append(enabled)
        return {'enabled': enabled, 'admitted': 0, 'pending': 0}


class PlacementRunnerTest(unittest.TestCase):
    def test_real_control_worker_qualification_pauses_until_session_resume(self):
        class QualificationEngine(FakeEngine):
            def handle_native_control(self, payload):
                if payload.get('action') == 'qualification':
                    self.calls.append(copy.deepcopy(payload))
                    return {'schema': 1, 'action': 'qualification',
                            'engine_epoch': self.epoch, 'clock': self._clock(),
                            'ranks': [0, 1, 2, 3], 'workers': []}
                return super().handle_native_control(payload)

        engine = QualificationEngine()
        worker_client = NativeControlClient(engine)
        session = NativeOverlapSession(engine)
        asyncio.run(worker_client.qualification(
            'teacher_configure', {'enabled': True,
                                  'snapshot_max_bytes': 64 * 1024 ** 2}))
        self.assertTrue(engine.paused)
        resumed = asyncio.run(session.execute(
            {'schema': 1, 'action': 'resume', 'run_id': 'run-1'}))
        self.assertTrue(resumed['resumed'])
        self.assertFalse(engine.paused)

    def test_terminal_full_telemetry_keeps_the_retained_split_window_generation(self):
        class Returned(PlacementDriver):
            def _session(self, phase_name, action, **fields):
                reply=super()._session(phase_name,action,**fields)
                if action=='overlap_end':reply['route']='full'
                return reply
            def _qualification_rows(self, action):
                reply=super()._qualification_rows(action)
                if action=='observer_end':
                    for worker in reply['workers']:
                        worker['qualification'].update(route='full', current_route='full',
                            current_route_generation=2,state='closed',observer_stopped=True)
                return reply
        result=self.run_runner(Returned())
        self.assertEqual(result['overlap_end']['route'],'full')
        self.assertEqual(result['window']['route_generation'],1)
        for row in result['native_observer_packets'].values():
            self.assertEqual(row['route_generation'],1)
            self.assertEqual(row['current_route_generation'],2)

    def test_full_telemetry_cannot_override_workers_without_a_closed_window(self):
        class Contradiction(PlacementDriver):
            def _session(self, phase_name, action, **fields):
                reply=super()._session(phase_name,action,**fields)
                if action=='overlap_end':reply['route']='full'
                return reply
        with self.assertRaisesRegex(QualificationRefused,'full-return-worker-window-contradiction'):
            self.run_runner(Contradiction())

    def run_runner(self, driver, workload=None):
        workload = workload or placement_workload()
        runner = NativePlacementRunner(workload=workload,
                                       prefill_plan=prefill_plan(),
                                       url='http://127.0.0.1:9')
        states = [FakeState(index) for index in range(8)]
        prefill_states = [FakeState(0, prompt_tokens=8)]

        def start_streams(url, model, bodies, cohort, workload, **kwargs):
            return prefill_states if len(bodies) == 1 else states

        def wait_open(states, workload):
            return {'complete': True}

        def wait_window(states, starts, workload):
            return {'complete': True, 'start_ns': 1100, 'end_ns': 1900,
                    'streams': [], 'measured': True}

        def stop_streams(record, states, workload):
            record['readers_alive_after_join'] = []
            return []

        with mock.patch.object(correctness, '_start_streams', start_streams), \
             mock.patch.object(correctness, '_wait_open', wait_open), \
             mock.patch.object(correctness, '_wait_window', wait_window), \
             mock.patch.object(correctness, '_stop_streams', stop_streams):
            return runner.run(driver, window_id='window-1', run_id='run-1',
                              engine_epoch=3, model='m')

    def test_runner_drives_real_order_and_retains_separate_receipts(self):
        driver = PlacementDriver()
        result = self.run_runner(driver)
        self.assertEqual(driver.calls, ['prepare', 'resume',
                                        'pause_enter_split', 'overlap_begin',
                                        'resume', 'overlap_end', 'leave_full',
                                        'resume'])
        self.assertEqual(driver.worker_calls, ['teacher_configure',
                                               'observer_begin', 'observer_end',
                                               'teacher_evidence',
                                               'teacher_configure'])
        # Preparation precedes observer/cohort; the observer closes only after
        # the owned cohort is stopped; full is re-entered through typed acks.
        self.assertLess(driver.calls.index('prepare'),
                        driver.calls.index('resume'))
        self.assertLess(driver.calls.index('overlap_end'),
                        driver.calls.index('leave_full'))
        self.assertEqual(result['window']['kind'], 'short')
        self.assertEqual(result['window']['replays'], 40)
        self.assertEqual(set(result['raw_placement']), {0, 1, 2, 3})
        self.assertEqual(set(result['engine_accounting']), {0, 1, 2, 3})
        for rank in (0, 1, 2, 3):
            self.assertEqual(result['raw_sha256'][str(rank)],
                             sha(result['raw_placement'][rank]))
            self.assertIsNot(result['raw_graph_receipts'][rank],
                             result['engine_accounting_receipts'][rank])
        self.assertTrue(result['shipping_path_instrument_disabled'])
        self.assertEqual(result['cleanup']['restored'], True)
        self.assertEqual(result['teacher_rows_during_placement'], 0)
        self.assertFalse(result['stock_prefix']['native_admission_intent'])
        self.assertTrue(result['stock_prefix']['prepare_keys'])
        for keys in result['stock_prefix']['prepare_keys']:
            self.assertFalse([key for key in keys
                              if 'native' in key or 'admission' in key
                              or 'prefill' in key or 'teacher' in key])
        # The stock preparation is phase-free; only the split transition
        # carries the declared native placement phase.
        fields = dict(driver.session_fields)
        self.assertNotIn('phase', fields['prepare'])
        self.assertEqual(fields['pause_enter_split']['phase'], 'w2-overlap')
        self.assertEqual(result['window']['native_phase'], 'w2-overlap')
        self.assertTrue(result['window']['admission']['armed'])
        self.assertEqual(result['window']['admission']['phase'], 'w2-overlap')
        self.assertFalse(result['cleanup']['admission']['armed'])
        self.assertEqual(result['cleanup']['admission']['phase'],
                         'full-prefill-solo')

    def test_stock_cohort_opens_after_qualification_pause_is_resumed(self):
        class PausingWorker(PlacementDriver):
            paused = False
            pending_states = ()

            def _worker(self, phase, action, payload):
                if action == 'teacher_configure':
                    self.paused = True
                return super()._worker(phase, action, payload)

            def _session(self, phase_name, action, **fields):
                reply = super()._session(phase_name, action, **fields)
                if action == 'resume':
                    self.paused = False
                    for state in self.pending_states:
                        state.first_advance_ns = time.monotonic_ns()
                        state.progress.set()
                    self.pending_states = ()
                return reply

        driver = PausingWorker()
        runner = NativePlacementRunner(workload=placement_workload(),
                                       prefill_plan=prefill_plan(),
                                       url='http://127.0.0.1:9')
        states = [FakeState(index) for index in range(8)]
        prefill_states = [FakeState(0, prompt_tokens=8)]

        # Use the real open-wait for both the paused negative and the resumed
        # cohort. A paused fixture engine emits no first advance, matching the
        # eight live streams in native21's retained row4 artifact.
        paused_state = FakeState(99)
        paused_state.first_advance_ns = None
        paused_open = correctness._wait_open(
            [paused_state], types.SimpleNamespace(open_timeout_s=0.01))
        self.assertFalse(paused_open['complete'])
        self.assertEqual(paused_open['reasons'], ['stream_not_advanced:99'])
        for state in states + prefill_states:
            state.first_advance_ns = None

        def start_streams(url, model, bodies, cohort, workload, **kwargs):
            if len(bodies) == 8:
                self.assertTrue(driver.paused)
            selected = prefill_states if len(bodies) == 1 else states
            driver.pending_states = selected
            return selected

        def stop_streams(record, states, workload):
            record['readers_alive_after_join'] = []
            return []

        with mock.patch.object(correctness, '_start_streams', start_streams), \
             mock.patch.object(correctness, '_wait_window',
                               return_value={'complete': True, 'start_ns': 1100,
                                             'end_ns': 1900, 'streams': [],
                                             'measured': True}), \
             mock.patch.object(correctness, '_stop_streams', stop_streams):
            runner.run(driver, window_id='window-1', run_id='run-1',
                       engine_epoch=3, model='m')
        self.assertEqual(driver.calls[:2], ['prepare', 'resume'])
        self.assertFalse(driver.paused)
        self.assertEqual(driver.calls[-2:], ['leave_full', 'resume'])

    def test_placement_admission_must_be_the_declared_native_phase(self):
        driver = PlacementDriver()
        original = driver._session

        def foreign(phase_name, action, *, require_epoch=True, **fields):
            reply = original(phase_name, action, require_epoch=require_epoch, **fields)
            if action == 'pause_enter_split':
                reply['phase'] = 'row1-native'
            return reply

        driver._session = foreign
        with self.assertRaises(QualificationRefused) as caught:
            self.run_runner(driver)
        self.assertEqual(caught.exception.code,
                         'placement-admission-phase-differs')
        self.assertNotIn('overlap_begin', driver.calls)

    def test_failure_still_returns_to_full_and_preserves_the_first_error(self):
        driver = PlacementDriver(fail_at='observer_end')
        runner = NativePlacementRunner(workload=placement_workload(),
                                       prefill_plan=prefill_plan(),
                                       url='http://127.0.0.1:9')
        states = [FakeState(index) for index in range(8)]
        prefill_states = [FakeState(0, prompt_tokens=8)]
        with mock.patch.object(correctness, '_start_streams',
                               lambda url, model, bodies, *a, **k:
                                   prefill_states if len(bodies) == 1 else states), \
             mock.patch.object(correctness, '_wait_open',
                               lambda *a, **k: {'complete': True}), \
             mock.patch.object(correctness, '_wait_window',
                               lambda *a, **k: {'complete': True}), \
             mock.patch.object(correctness, '_stop_streams',
                               lambda record, *a: record.update(
                                   readers_alive_after_join=[])):
            with self.assertRaises(QualificationRefused) as caught:
                runner.run(driver, window_id='window-1', run_id='run-1',
                           engine_epoch=3, model='m')
        self.assertEqual(caught.exception.code, 'injected-observer_end')
        self.assertEqual(caught.exception.cleanup['restored'], True)
        self.assertIn('leave_full', driver.calls)

    def test_failed_cleanup_never_invents_a_drain(self):
        driver = PlacementDriver(fail_at='observer_end', leave_restored=False)
        runner = NativePlacementRunner(workload=placement_workload(),
                                       prefill_plan=prefill_plan(),
                                       url='http://127.0.0.1:9')
        states = [FakeState(index) for index in range(8)]
        prefill_states = [FakeState(0, prompt_tokens=8)]
        with mock.patch.object(correctness, '_start_streams',
                               lambda url, model, bodies, *a, **k:
                                   prefill_states if len(bodies) == 1 else states), \
             mock.patch.object(correctness, '_wait_open',
                               lambda *a, **k: {'complete': True}), \
             mock.patch.object(correctness, '_wait_window',
                               lambda *a, **k: {'complete': True}), \
             mock.patch.object(correctness, '_stop_streams',
                               lambda record, *a: record.update(
                                   readers_alive_after_join=[])):
            with self.assertRaises(QualificationRefused) as caught:
                runner.run(driver, window_id='window-1', run_id='run-1',
                           engine_epoch=3, model='m')
        self.assertEqual(caught.exception.code, 'injected-observer_end')
        self.assertEqual(caught.exception.cleanup['attempted'], True)
        self.assertEqual(caught.exception.cleanup['restored'], False)
        self.assertNotIn('worker_drain', caught.exception.cleanup)

    def test_failure_after_full_return_preserves_the_first_cleanup(self):
        driver = PlacementDriver()
        original = driver._frontend
        calls = []

        def frontend(phase, enabled, plan):
            calls.append(enabled)
            if len(calls) == 2:
                raise QualificationRefused('injected-frontend-after')
            return original(phase, enabled, plan)

        driver._frontend = frontend
        with self.assertRaises(QualificationRefused) as caught:
            self.run_runner(driver)
        self.assertEqual(caught.exception.code, 'injected-frontend-after')
        self.assertEqual(caught.exception.cleanup['restored'], True)
        self.assertEqual(caught.exception.cleanup['route'], 'full')
        self.assertEqual(driver.calls.count('leave_full'), 1)

    def test_partial_transition_is_refused_before_measurement(self):
        driver = PlacementDriver()
        original = driver._session

        def partial(phase_name, action, *, require_epoch=True, **fields):
            reply = original(phase_name, action, require_epoch=require_epoch, **fields)
            if action == 'pause_enter_split':
                reply['transition_acks'] = reply['transition_acks'][:3]
            return reply

        driver._session = partial
        with self.assertRaises(QualificationRefused) as caught:
            self.run_runner(driver)
        self.assertEqual(caught.exception.code, 'missing-worker:enter-split')
        self.assertNotIn('overlap_begin', driver.calls)


if __name__ == '__main__':
    unittest.main(verbosity=2)
