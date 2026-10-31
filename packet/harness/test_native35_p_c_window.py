"""Native35 P-c row4: the decode cohort stops at the window boundary, and the
observer_end reply is read through a bounded spool (lead decision 31 Oct 2026:
option A plus the spool).

Proves, on CPU:
(1) ``StoppedWindowPlacementRunner.run`` is the live L2 ``run`` byte for byte
    plus one inserted block after the window refusal; every module name it
    resolves is the driver's object; the window is ``PLACEMENT_WINDOW_TOKENS``
    = 20 tokens per stream and the declared replays stay 40.
(2) With the real ``_wait_window``/``_wait_open``/``_prefill_progress``, the
    frozen workload and prefill plan and a prefill whose first token arrives
    long after the window, P-c stops every decode stream at start+20 (the
    64-token budget is never reached) before the prefill opens, and the run
    completes.  L2 on the same timeline lets the cohort run to its budget
    inside the observer window (counterfactual).
(3) The spool end to end: a real loopback HTTP server, the real
    ``LoopbackControl``, the real ``QualificationDriver`` row4 collection and the
    real P-c runner.  A ~118 MiB observer_end reply (over e3_probe_rpc's
    112 MiB, under 512 MiB) is spooled, parsed and collected.  Body bytes and
    sha256 equal the server's, the retained 0600 spool file is the exact body,
    and the row4 evidence carries per-rank per-field sizes, ring fills against
    their caps, correlation-list fills against 131072 and the replay bound.
    Ring and list overflows are explicit missing reasons.  Counterfactual: the
    same body on another worker call is still refused at 112 MiB.
(4) Over the ceiling: a 513 MiB observer_end body (declared Content-Length) is
    refused ``observer-end-response-too-large`` through the same composition.
    The retained refusal record carries the full body's bytes and sha256 and
    the partial window evidence, and nothing is stored.  A close-delimited body
    is stored up to the ceiling and still fully counted and hashed.
(5) Census rules over the observer's own receipt and report key sets.
(6) Hazard (a), stopping the decoders under the split route: with a native
    chunk in flight, the generated EngineCore ``abort_requests`` for ordinary
    decode ids leaves the native driver untouched and hands the ids to the
    stock finish path, and the chunk completes.  The same call with the native
    request id stops the driver (counterfactual).
(7) The exact P-c queue row4 argv reaches its first network call through the
    real CLI with the P-c runner.
Fixture leaves: native35_p_c_fixture_leaves.json.
"""
import ast
import copy
import hashlib
import http.server
import inspect
import json
import os
from pathlib import Path
import shlex
import stat
from string import Template
import tempfile
import textwrap
import threading
import time
import types
import unittest
from unittest import mock

import e3_decode_correctness as correctness
import e3_native_qualification_driver as driver_module
import e3_observer_census as census_module
import e3_probe_rpc as rpc
from e3_native_qualification_driver import (
    ImmutableStore, NativePlacementRunner, QualificationDriver, QualificationRefused,
    PLACEMENT_GENERATION_BUDGET, PLACEMENT_REPLAYS, PLACEMENT_REPLAYS_HIGH,
    PLACEMENT_REPLAYS_LOW, PLACEMENT_WINDOW_TOKENS, OBSERVER_END_RESPONSE_MAX,
    WORKER_PATH, canonical_sha)
import e3_native35_l2_placement as l2
import e3_native35_p_c_placement as pc
import test_e3_native_qualification_driver as base
import test_native35_l2 as l2_suite
import replay_observer  # noqa: E402 - pinned engine tree (driver put it on sys.path)

HERE = Path(__file__).resolve().parent
RUNS = HERE / 'runs'
MIB = 1024 * 1024
TIMING_RECEIPTS = []
SPOOL_RECEIPTS = []
ABORT_RECEIPTS = []
CLI_RECEIPTS = []
KEY = 'native35-p-c-cpu-transport-fixture'

WINDOW_BLOCK = l2_suite.WINDOW_BLOCK
STOP_BLOCK = ("            # P-c: stop the owned decode cohort at the window boundary, before\n"
              "            # the native prefill's first token, so the observer window holds the\n"
              "            # declared decode replays (2 parity cohorts x PLACEMENT_WINDOW_TOKENS).\n"
              "            correctness._stop_streams(record, states, self.workload)\n"
              "            if record.get('readers_alive_after_join'):\n"
              "                raise QualificationRefused('placement-readers-alive')\n")


class SourceRelation(unittest.TestCase):
    def test_run_is_the_l2_run_plus_only_the_stop_block(self):
        parent = inspect.getsource(l2.ResumeWindowPlacementRunner.run)
        child = inspect.getsource(pc.StoppedWindowPlacementRunner.run)
        self.assertEqual(parent.count(WINDOW_BLOCK), 1)
        self.assertEqual(child, parent.replace(WINDOW_BLOCK, WINDOW_BLOCK + STOP_BLOCK))
        # The stop comes before the prefill open-wait and its refusal.
        self.assertEqual(child.count(STOP_BLOCK + l2_suite.OPEN_BLOCK), 1)
        own = set(vars(pc.StoppedWindowPlacementRunner)) - {
            '__module__', '__doc__', '__qualname__', '__firstlineno__', '__static_attributes__'}
        self.assertEqual(own, {'run'})
        self.assertTrue(issubclass(pc.StoppedWindowPlacementRunner, l2.ResumeWindowPlacementRunner))
        names = set(pc.StoppedWindowPlacementRunner.run.__code__.co_names)
        for name in sorted(n for n in names & set(vars(driver_module)) if not n.startswith('__')):
            self.assertIs(vars(pc)[name], vars(driver_module)[name], name)
            self.assertIs(vars(pc)[name], vars(l2)[name], name)
        self.assertIs(pc.correctness, correctness)
        # Declared window: 2 parity cohorts x 20 tokens = the observer's 40 replays.
        self.assertEqual((PLACEMENT_WINDOW_TOKENS, PLACEMENT_REPLAYS), (20, 40))
        self.assertEqual((PLACEMENT_REPLAYS_LOW, PLACEMENT_REPLAYS_HIGH), (40, 56))
        self.assertEqual(driver_module.PLACEMENT_STREAMS // 4 * PLACEMENT_WINDOW_TOKENS, PLACEMENT_REPLAYS)
        runner = pc.StoppedWindowPlacementRunner(workload=base.placement_workload(), url='http://127.0.0.1:9')
        self.assertEqual((runner.workload.window_tokens, runner.workload.generation_budget,
                          runner.replicates), (20, PLACEMENT_GENERATION_BUDGET, 40))
        self.assertEqual(rpc.MAX_RESPONSE, 112 * MIB)
        self.assertEqual(OBSERVER_END_RESPONSE_MAX, 512 * MIB)


# --- (2) timing ---------------------------------------------------------------

def run_timeline(runner_cls, *, initial, open_tick, control=None, plan=None):
    """test_native35_l2's Timeline and control fake; the stop fake ends a
    stopped decode stream (client close -> engine abort: no further tokens)."""
    states = [base.FakeState(index) for index in range(8)]
    for state, value in zip(states, initial):
        state.value = value
    prefill = [base.FakeState(0, prompt_tokens=4167)]
    prefill[0].first_advance_ns = None
    timeline = l2_suite.Timeline(states, prefill, open_tick=open_tick)
    driver = control(timeline) if control else l2_suite.TimelineDriver(timeline)
    runner = runner_cls(workload=base.placement_workload(), prefill_plan=plan or l2_suite.real_prefill_plan(),
                        url='http://127.0.0.1:9')
    stops, windows = [], []
    real_wait_window = correctness._wait_window

    def recording_wait_window(states_, starts, workload):
        window = real_wait_window(states_, starts, workload)
        windows.append(window)
        return window

    def start_streams(url, model, bodies, cohort, workload, **kwargs):
        return prefill if len(bodies) == 1 else states

    def stop_streams(record, states_, workload):
        now = time.monotonic_ns()
        kind = 'prefill' if states_ is prefill else 'decode'
        if kind == 'decode':
            for state in states_:
                with state.lock:
                    if state.finish_at_ns is None:
                        state.finish_at_ns, state.finish_reason = now, 'abort'
        stops.append(dict(kind=kind, at_ns=now, counts=[state.count() for state in states_],
                          prefill_open=timeline.prefill_open_ns is not None))
        record['readers_alive_after_join'] = []
        return []

    outcome = None
    try:
        with mock.patch.object(correctness, '_start_streams', start_streams), \
             mock.patch.object(correctness, '_stop_streams', stop_streams), \
             mock.patch.object(correctness, '_wait_window', recording_wait_window):
            try:
                outcome = (runner.run(driver, window_id='window-1', run_id='run-1', engine_epoch=3, model='m')
                           if control is None else driver.collect(runner))
            except QualificationRefused as exc:
                outcome = exc
    finally:
        timeline.stop.set()
        if timeline.thread.ident is not None:
            timeline.thread.join(5)
    return outcome, driver, windows, stops, timeline


class Timing(unittest.TestCase):
    BOX_INITIAL = (1, 2, 3, 4, 5, 6, 7, 8)
    OPEN_TICK = 120  # 0.6 s: the cohort's natural finish (~tick 56) comes first

    def test_p_c_stops_the_cohort_at_the_window_and_l2_lets_it_run_to_budget(self):
        child, driver, windows, stops, timeline = run_timeline(
            pc.StoppedWindowPlacementRunner, initial=self.BOX_INITIAL, open_tick=self.OPEN_TICK)
        self.assertIsInstance(child, dict, getattr(child, 'code', child))
        window = windows[0]
        self.assertEqual((window['complete'], window['reason']), (True, 'token_bound'))
        starts = [row['start_completion'] for row in window['streams']]
        decode = [row for row in stops if row['kind'] == 'decode']
        # First stop: the decode cohort, at the window boundary, before the prefill opened.
        self.assertEqual(stops[0]['kind'], 'decode')
        self.assertFalse(stops[0]['prefill_open'])
        self.assertLess(stops[0]['at_ns'], timeline.prefill_open_ns)
        for start, count in zip(starts, stops[0]['counts']):
            self.assertGreaterEqual(count, start + PLACEMENT_WINDOW_TOKENS)
            self.assertLessEqual(count, start + PLACEMENT_WINDOW_TOKENS + 10)
        self.assertLess(max(stops[0]['counts']), PLACEMENT_GENERATION_BUDGET)
        # The parent's later stop at the observer boundary finds the stopped
        # counts (at most one token the timeline had already begun to add).
        for first, last in zip(stops[0]['counts'], decode[-1]['counts']):
            self.assertIn(last - first, (0, 1))
        self.assertTrue(child['native_prefill_request']['complete'])
        self.assertEqual(child['window']['window_tokens'], 20)
        self.assertEqual(driver.calls, ['prepare', 'resume', 'pause_enter_split', 'overlap_begin',
                                        'resume', 'overlap_end', 'leave_full', 'resume'])
        self.assertEqual(driver.worker_calls, ['teacher_configure', 'observer_begin', 'observer_end',
                                               'teacher_evidence', 'teacher_configure'])

        parent, parent_driver, _, parent_stops, parent_timeline = run_timeline(
            l2.ResumeWindowPlacementRunner, initial=self.BOX_INITIAL, open_tick=self.OPEN_TICK)
        self.assertIsInstance(parent, dict, getattr(parent, 'code', parent))
        self.assertEqual(parent_stops[0]['kind'], 'decode')
        self.assertTrue(parent_stops[0]['prefill_open'])
        self.assertEqual(parent_stops[0]['counts'], [PLACEMENT_GENERATION_BUDGET] * 8)
        self.assertLess(parent_timeline.decode_done_ns, parent_timeline.prefill_open_ns)
        self.assertEqual((parent_driver.calls, parent_driver.worker_calls), (driver.calls, driver.worker_calls))
        TIMING_RECEIPTS.append(dict(case='stop-at-window', verdict='PASS', p_c_counts_at_stop=stops[0]['counts'],
                                    window_starts=starts, l2_counts_at_first_stop=parent_stops[0]['counts'],
                                    p_c_stop_before_prefill_open=True, real_wait_functions=True))

    def test_p_c_still_refuses_an_early_finished_stream_and_a_prefill_that_never_opens(self):
        early, _, _, stops, _ = run_timeline(pc.StoppedWindowPlacementRunner,
                                             initial=(50, 1, 2, 3, 4, 5, 6, 7), open_tick=1)
        self.assertIsInstance(early, QualificationRefused)
        self.assertEqual(early.code, 'placement-window-incomplete')
        self.assertEqual(early.evidence['client_window']['early_finish'], [0])
        self.assertTrue(early.cleanup['restored'])
        never, _, windows, stops, _ = run_timeline(pc.StoppedWindowPlacementRunner,
                                                   initial=self.BOX_INITIAL, open_tick=None,
                                                   plan=l2_suite.real_prefill_plan(open_s=0.5))
        self.assertIsInstance(never, QualificationRefused)
        self.assertEqual(never.code, 'native-prefill-not-opened')
        self.assertTrue(windows[0]['complete'])
        self.assertEqual(stops[0]['kind'], 'decode')
        TIMING_RECEIPTS.append(dict(case='refusals-kept', verdict='PASS',
                                    early_finish='placement-window-incomplete',
                                    never_opens='native-prefill-not-opened after the decode stop'))


# --- (3)/(4) spool over real loopback HTTP --------------------------------------

def receipt_census(rank, *, kernel_overflow=0):
    """Receipt replay/limits blocks with the observer's own key sets."""
    counts = {key: 0 for key in replay_observer._COUNT_KEYS}
    counts.update(kernel_records=61400 + 131 * rank, copy_records=1040, context_records=2,
                  resource_events=900, records_in_window=62440 + 131 * rank)
    rings = {'context': 0, 'kernel': kernel_overflow, 'copy': 0, 'resource': 0, 'aliases': 0}
    if kernel_overflow:
        counts['kernel_records'] = replay_observer.ObserverLimits().max_kernel_records
    flags = {key: False for key in replay_observer._FLAG_KEYS}
    flags['permanent_overflow'] = bool(kernel_overflow)
    return {'replay': {'counts': counts, 'ring_overflows': rings, 'flags': flags},
            'limits': replay_observer.ObserverLimits().as_record()}


def native_report(rank, *, apis=64, overflow=0):
    row = {'domain': 'runtime', 'name': 'cudaEventQuery', 'correlation_id': 0, 'thread': 4242,
           'return_value': 0, 'start': 1, 'end': 2}
    return {'schema': 's5ro-correlation/2', 'window': {'start_ns': 1, 'end_ns': 2, 'route_generation': 1},
            'thread_id_type': 'system-gettid', 'complete': overflow == 0, 'capacity': 131072,
            'reserved_bytes': 1, 'errors': 0, 'overflow': overflow, 'open_scopes': 0,
            'boundary_records': 0, 'activity_records': 8,
            'apis': [dict(row, correlation_id=index) for index in range(apis)],
            'external': [{'kind': 'custom0', 'correlation_id': 1, 'external_id': 9}],
            'scopes': [{'id': 1, 'parent': 0, 'thread': 1, 'system_thread': 1, 'action': 'push'}],
            'activities': [{'correlation_id': 1, 'kind': 'kernel', 'context_id': 1, 'stream_id': 7,
                            'graph_id': 3, 'copy_kind': 0, 'name': 'k', 'count': 8}]}


class Peer:
    """Loopback HTTP peer for the session, worker and frontend adapter paths.

    Replies are the established row4 control fake (test_e3_native_qualification_driver
    PlacementDriver) with the exact identity echo the transport requires;
    observer_end is built per case (declared leaf).
    """

    def __init__(self, timeline, *, observer_end):
        self.fake = base.PlacementDriver()
        self.timeline = timeline
        self.observer_end = observer_end
        self.resumes = 0
        self.calls, self.worker_calls = [], []
        self.sent = {}
        self.big_for = None

    def session(self, command):
        action = command['action']
        self.calls.append(action)
        reply = copy.deepcopy(self.fake._session('row4', action))
        if action == 'resume':
            self.resumes += 1
            if self.resumes == 2:
                self.timeline.start()
        reply.update(schema=1, action=action, run_id=command['run_id'])
        return reply

    def envelope(self, command, worker_reply):
        return {'schema': 1, 'action': 'qualification_worker', 'run_id': command['run_id'],
                'command_seq': command['command_seq'], 'client_id': command['client_id'],
                'engine_epoch': 3, 'worker_action': command['worker_action'],
                'worker_reply': worker_reply}

    def frontend(self, command):
        reply = self.fake._frontend('row4', command['enabled'], command.get('plan'))
        return dict(reply, schema=1, action=command['action'], run_id=command['run_id'],
                    command_seq=command['command_seq'], client_id=command['client_id'],
                    engine_epoch=3)


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        # A client that stops reading at its cap closes the socket mid-body.
        pass


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.0'

    def log_message(self, *args):
        pass

    def _headers(self, length=None):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        if length is not None:
            self.send_header('Content-Length', str(length))
        self.send_header('Connection', 'close')
        self.end_headers()

    def _json(self, value):
        body = json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()
        self._headers(len(body))
        self.wfile.write(body)
        return body

    def do_POST(self):
        peer = self.server.peer
        command = json.loads(self.rfile.read(int(self.headers.get('Content-Length', '0'))))
        assert self.headers.get('Authorization') == 'Bearer ' + KEY
        if self.path == '/v1/flash_s5_native_overlap':
            self._json(peer.session(command))
            return
        if self.path == driver_module.FRONTEND_PATH:
            self._json(peer.frontend(command))
            return
        assert self.path == WORKER_PATH, self.path
        action = command['worker_action']
        peer.worker_calls.append(action)
        if action == 'observer_end' or action == peer.big_for:
            mode, value = peer.observer_end(peer, command)
            if mode == 'json':
                peer.sent[action] = body = self._json(value)
                peer.sent[action + ':sha256'] = hashlib.sha256(body).hexdigest()
                return
            # Streamed body: ``value`` bytes of a JSON string, declared or close-delimited.
            digest, size = hashlib.sha256(), 0
            self._headers(value if mode == 'stream' else None)
            head, tail = b'{"pad":"', b'"}'
            chunk = b'x' * MIB
            remaining = value - len(head) - len(tail)
            for piece in [head] + [chunk] * (remaining // MIB) + [b'x' * (remaining % MIB), tail]:
                self.wfile.write(piece)
                digest.update(piece)
                size += len(piece)
            peer.sent[action + ':bytes'] = size
            peer.sent[action + ':sha256'] = digest.hexdigest()
            return
        self._json(peer.envelope(command, peer.fake._qualification_rows(action)))


class Composition:
    """Real QualificationDriver + LoopbackControl over the loopback peer."""

    def __init__(self, root, timeline, *, observer_end):
        self.peer = Peer(timeline, observer_end=observer_end)
        self.httpd = Server(('127.0.0.1', 0), Handler)
        self.httpd.peer = self.peer
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.root = root
        self.artifact = base.retained_teacher(ImmutableStore(root), base.teacher_evidence(None))
        self.url = 'http://127.0.0.1:%d' % self.httpd.server_address[1]
        self.runner = None
        self.calls, self.worker_calls = self.peer.calls, self.peer.worker_calls

    def collect(self, runner):
        self.runner = runner
        self.driver = QualificationDriver(self.root, run_id='run-1', model='m', engine_epoch=3,
                                          url=self.url, key=KEY, placement=runner,
                                          route_banks=(0, 1), timeout_s=60.0)
        return self.driver.collect_phase('row4', artifact_sha256=canonical_sha(self.artifact),
                                         row1_sha256='e' * 64, window_id='window-1')

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def observer_end_rows(peer, *, pad_per_rank):
    rows = peer.fake._qualification_rows('observer_end')
    for row in rows['workers']:
        rank = row['worker_rank']
        record = row['qualification']
        record['receipt'].update(receipt_census(rank, kernel_overflow=1200 if rank == 2 else 0))
        record['native_report'] = native_report(rank, apis=131072 if rank == 1 else 64,
                                                overflow=7 if rank == 1 else 0)
        record['decode_window'] = {'binding': {'router_census': {'total': 44 + rank}}}
        # Bulk standing in for the unbounded fields (declared leaf).
        record['launch_config'] = {'schema': 1, 'verdict': 'INCONCLUSIVE', 'bulk': 'x' * pad_per_rank}
    return rows


def with_composition(test, observer_end, body):
    with tempfile.TemporaryDirectory() as root:
        holder = {}

        def control(timeline):
            holder['c'] = Composition(root, timeline, observer_end=observer_end)
            return holder['c']

        try:
            result = run_timeline(pc.StoppedWindowPlacementRunner, initial=Timing.BOX_INITIAL,
                                  open_tick=1, control=control)
            return body(result, holder['c'], Path(root))
        finally:
            if 'c' in holder:
                holder['c'].close()


class Spool(unittest.TestCase):
    PAD = 29 * MIB

    def test_reply_over_112_mib_is_spooled_and_collected_through_the_p_c_runner(self):
        def reply(peer, command):
            return 'json', peer.envelope(command, observer_end_rows(peer, pad_per_rank=self.PAD))

        def check(result, composition, root):
            outcome, _, _, stops, _ = result
            self.assertIsInstance(outcome, dict, getattr(outcome, 'code', outcome))
            peer = composition.peer
            sent = len(peer.sent['observer_end'])
            self.assertGreater(sent, rpc.MAX_RESPONSE)
            self.assertLess(sent, OBSERVER_END_RESPONSE_MAX)
            evidence = outcome['evidence']
            transport = evidence['observer_end_transport']
            self.assertEqual(transport['state'], 'spooled')
            self.assertEqual(len(transport['receipts']), 1)
            receipt = transport['receipts'][0]
            self.assertEqual((receipt['state'], receipt['drained_to_eof'], receipt['http_status']), ('read', True, 200))
            self.assertEqual((receipt['body_bytes'], receipt['content_length'], receipt['spool_bytes']), (sent, sent, sent))
            self.assertEqual(receipt['body_sha256'], peer.sent['observer_end:sha256'])
            self.assertEqual((receipt['ceiling_bytes'], receipt['default_cap_bytes']), (512 * MIB, 112 * MIB))
            spool = root / receipt['spool']
            self.assertEqual(stat.S_IMODE(spool.stat().st_mode), 0o600)
            self.assertEqual(hashlib.sha256(spool.read_bytes()).hexdigest(), receipt['body_sha256'])
            sizes = receipt['field_sizes']
            self.assertEqual(sizes['state'], 'measured')
            self.assertEqual(sorted(sizes['per_rank']), ['0', '1', '2', '3'])
            for rank, row in sizes['per_rank'].items():
                self.assertEqual(row['largest_field'], 'launch_config')
                self.assertGreater(row['fields']['launch_config'], self.PAD)
                self.assertEqual(row['fields']['native_report'],
                                 census_module.encoded_size(evidence['native_observer_packets'][int(rank)]['native_report']))
            self.assertGreater(sizes['per_rank']['1']['fields']['native_report'], 10 * MIB)
            # Census in the evidence, explicit reasons in the missing list.
            observer = evidence['observer_census']
            self.assertEqual(observer['schema'], 'q18a-observer-census/1')
            ring = observer['per_rank']['2']['observer_rings']['rings']['kernel']
            self.assertEqual((ring['entries'], ring['cap'], ring['overflow']), (131072, 131072, 1200))
            self.assertEqual(observer['per_rank']['0']['observer_rings']['rings']['copy'],
                             {'entries': 1040, 'cap': 32768, 'overflow': 0, 'count_key': 'copy_records',
                              'cap_key': 'max_copy_records'})
            apis = observer['per_rank']['1']['native_report']['lists']['apis']
            self.assertEqual(apis, {'entries': 131072, 'cap': 131072})
            self.assertEqual(observer['per_rank']['3']['replays'],
                             {'observed': 47, 'declared': 40, 'low': 40, 'high': 56, 'within': True})
            explicit = [reason for reason in outcome['missing'] if reason.startswith('observer:')]
            self.assertEqual(explicit, ['observer:api-ring-overflow:1',
                                        'observer:correlation-report-incomplete:1',
                                        'observer:kernel-ring-overflow:2'])
            self.assertEqual(outcome['complete'], False)
            # Retained once in the store, with the census and transport.
            retained = json.loads((root / 'row4.json').read_text())
            self.assertEqual(retained['evidence']['observer_end_transport'], transport)
            self.assertEqual(retained['missing'], outcome['missing'])
            # Every other call: unchanged cap and never spooled.
            control = composition.driver.control
            self.assertEqual(peer.worker_calls, ['teacher_configure', 'observer_begin', 'observer_end',
                                                 'teacher_evidence', 'teacher_configure'])
            self.assertEqual(len(control.transport_receipts), 1)
            peer.big_for = 'teacher_evidence'
            with self.assertRaises(QualificationRefused) as caught:
                composition.driver._worker('row4', 'teacher_evidence', {'action': 'teacher_evidence'})
            self.assertEqual(caught.exception.code, 'response-too-large')
            self.assertEqual(len(control.transport_receipts), 1)
            self.assertEqual(stops[0]['kind'], 'decode')
            SPOOL_RECEIPTS.append(dict(case='over-112-under-512-end-to-end', verdict='PASS', body_bytes=sent,
                                       body_sha256=receipt['body_sha256'], spool_bytes=receipt['spool_bytes'],
                                       explicit_missing=explicit, other_call_refused='response-too-large',
                                       runner='StoppedWindowPlacementRunner', real_http=True))

        with_composition(self, reply, check)

    def test_reply_over_512_mib_refuses_with_bytes_and_sha_recorded(self):
        size = OBSERVER_END_RESPONSE_MAX + MIB

        def check(result, composition, root):
            outcome, _, _, stops, _ = result
            self.assertIsInstance(outcome, QualificationRefused)
            self.assertEqual(outcome.code, 'observer-end-response-too-large')
            peer = composition.peer
            self.assertEqual(peer.sent['observer_end:bytes'], size)
            refusal = json.loads((root / 'refusal.row4.json').read_text())
            self.assertEqual(refusal['missing'], ['observer-end-response-too-large'])
            transport = refusal['evidence']['observer_end_transport']
            receipt = transport['receipts'][0]
            self.assertEqual((receipt['state'], receipt['drained_to_eof']), ('refused:too-large', True))
            self.assertEqual((receipt['body_bytes'], receipt['content_length']), (size, size))
            self.assertEqual(receipt['body_sha256'], peer.sent['observer_end:sha256'])
            # Declared over the ceiling: drained and hashed, nothing stored.
            self.assertEqual((receipt['spool'], receipt['spool_bytes']), (None, 0))
            self.assertEqual(sorted(p.name for p in root.iterdir() if p.name.startswith('observer_end.')), [])
            # The partial window evidence and the typed cleanup stay with the refusal.
            self.assertTrue(refusal['evidence']['client_window']['complete'])
            self.assertTrue(refusal['cleanup']['restored'])
            self.assertIn('leave_full', peer.calls)
            self.assertEqual(stops[0]['kind'], 'decode')
            SPOOL_RECEIPTS.append(dict(case='over-512-refused', verdict='PASS', body_bytes=size,
                                       body_sha256=receipt['body_sha256'], stored_bytes=0,
                                       refusal='observer-end-response-too-large', real_http=True))

        with_composition(self, lambda peer, command: ('stream', size), check)

    def test_close_delimited_body_is_stored_to_the_ceiling_and_fully_hashed(self):
        with tempfile.TemporaryDirectory() as root:
            peer = Peer(None, observer_end=lambda peer, command: ('close', 5 * MIB))
            httpd = Server(('127.0.0.1', 0), Handler)
            httpd.peer = peer
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            try:
                control = driver_module.LoopbackControl('http://127.0.0.1:%d' % httpd.server_address[1],
                                                        key=KEY, spool_dir=root)
                command = {'schema': 1, 'action': 'qualification_worker', 'run_id': 'run-1',
                           'client_id': 'c', 'command_seq': 1, 'engine_epoch': 3,
                           'worker_action': 'observer_end', 'payload': {}}
                # The store-then-stop branch at a 4 MiB ceiling (declared leaf: the
                # constant, so the branch runs without writing 512 MiB).
                with mock.patch.object(driver_module, 'OBSERVER_END_RESPONSE_MAX', 4 * MIB):
                    with self.assertRaises(QualificationRefused) as caught:
                        control.worker(command)
            finally:
                httpd.shutdown()
                httpd.server_close()
            self.assertEqual(caught.exception.code, 'observer-end-response-too-large')
            receipt = control.transport_receipts[0]
            self.assertEqual((receipt['content_length'], receipt['body_bytes'], receipt['spool_bytes']),
                             (None, 5 * MIB, 4 * MIB))
            self.assertEqual(receipt['body_sha256'], peer.sent['observer_end:sha256'])
            stored = (Path(root) / receipt['spool']).read_bytes()
            self.assertEqual(len(stored), 4 * MIB)
            self.assertTrue(stored.startswith(b'{"pad":"x'))
            # Without a spool directory the observer_end read is refused, never unbounded.
            bare = driver_module.LoopbackControl('http://127.0.0.1:9', key=KEY)
            response = types.SimpleNamespace(status=200, getheader=lambda name: '10', read=lambda n: b'')
            with self.assertRaises(QualificationRefused) as caught:
                bare._spool_read(response, command)
            self.assertEqual(caught.exception.code, 'observer-end-spool-unavailable')
        SPOOL_RECEIPTS.append(dict(case='close-delimited-store-to-ceiling', verdict='PASS',
                                   body_bytes=5 * MIB, stored_bytes=4 * MIB, patched_ceiling_bytes=4 * MIB))


# --- (5) census rules -------------------------------------------------------------

class Census(unittest.TestCase):
    def packets(self, **changes):
        packets = {}
        for rank in (0, 1, 2, 3):
            packet = {'receipt': receipt_census(rank), 'native_report': native_report(rank)}
            packets[rank] = packet
        for rank, change in changes.items():
            change(packets[int(rank[1:])])
        return packets

    def run_census(self, packets, observed=None):
        observed = {rank: 44 for rank in (0, 1, 2, 3)} if observed is None else observed
        return census_module.census(packets, observed, ranks=(0, 1, 2, 3), declared=40, low=40, high=56)

    def test_clean_window_has_no_reason_and_records_every_fill(self):
        result = self.run_census(self.packets())
        self.assertEqual(result['missing'], [])
        rings = result['per_rank']['0']['observer_rings']
        self.assertEqual(sorted(rings['rings']), ['aliases', 'context', 'copy', 'kernel', 'resource'])
        self.assertEqual(rings['rings']['kernel']['cap'], 131072)
        self.assertEqual(result['per_rank']['0']['native_report']['lists']['apis'], {'entries': 64, 'cap': 131072})

    def test_each_ring_and_list_overflow_is_named(self):
        def ring(name):
            def change(packet):
                packet['receipt']['replay']['ring_overflows'][name] = 3
            return change

        def full(key, label):
            def change(packet):
                report = packet['native_report']
                report[key] = [report[key][0]] * 131072
                report.update(overflow=2, complete=False)
            return change

        for name in ('kernel', 'copy', 'context', 'resource', 'aliases'):
            self.assertEqual(self.run_census(self.packets(r3=ring(name)))['missing'],
                             ['observer:%s-ring-overflow:3' % name], name)
        for key, label in (('apis', 'api'), ('external', 'external'), ('scopes', 'scope')):
            self.assertEqual(self.run_census(self.packets(r0=full(key, label)))['missing'],
                             ['observer:%s-ring-overflow:0' % label, 'observer:correlation-report-incomplete:0'])

        def unattributed(packet):
            packet['native_report'].update(overflow=1, complete=False)
        self.assertEqual(self.run_census(self.packets(r1=unattributed))['missing'],
                         ['observer:correlation-overflow-unattributed:1', 'observer:correlation-report-incomplete:1'])

        def boundary(packet):
            packet['native_report'].update(boundary_records=5, complete=False)
        self.assertEqual(self.run_census(self.packets(r2=boundary))['missing'],
                         ['observer:correlation-report-incomplete:2:boundary_records'])

    def test_replay_bound_and_absent_census_are_named(self):
        result = self.run_census(self.packets(), observed={0: 39, 1: 40, 2: 56, 3: 57})
        self.assertEqual(result['missing'], ['observer:replays-below-declared:0:39<40',
                                             'observer:replays-above-tolerance:3:57>56'])
        absent = self.run_census({0: {}, 1: None}, observed={2: None})
        self.assertEqual(absent['missing'][:3], ['observer:ring-census-unavailable:0',
                                                 'observer:native-report-unavailable:0',
                                                 'observer:replays-unmeasured:0'])
        self.assertEqual(len(absent['missing']), 12)


# --- (6) hazard (a): decode abort with a native chunk in flight -----------------------

def generated_abort():
    """The pinned EngineCore.abort_requests after the E3 transform (first class)."""
    import executor_hooks as eh
    import test_e3_native_overlap as overlap
    text = eh.generate_core(overlap.CORE_PATH.read_text())
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.FunctionDef) and node.name == 'abort_requests':
            source = textwrap.dedent('    ' + ast.get_source_segment(text, node))
            break
    else:
        raise AssertionError('generated core has no abort_requests')
    assert '_flash_s5_hooks.before_abort(self, request_ids)' in source
    namespace = {'_flash_s5_hooks': eh,
                 'RequestStatus': types.SimpleNamespace(FINISHED_ABORTED='aborted')}
    exec(compile(source, '<generated-core-abort>', 'exec'), namespace)
    return namespace['abort_requests'], source


class AbortUnderSplit(unittest.TestCase):
    DECODE = ['placement-decode-%d' % index for index in range(8)]

    def stack(self):
        import test_e3_native_overlap as overlap
        stack = overlap.active_stack()
        core, _frontend = overlap.core_for(stack)
        finished = []
        stack.scheduler.finish_requests = lambda ids, status: finished.append((list(ids), status))
        driver = stack.driver
        stack.drive(lambda: driver.tx is not None and driver._split_ready, limit=400)
        for rank in driver.ranks:
            self.assertEqual(stack.worker(rank).controller.route, 'split')
        stops = []
        real_stop = driver.stop

        def recording_stop(kind, **kwargs):
            stops.append((kind, kwargs))
            return real_stop(kind, **kwargs)

        driver.stop = recording_stop
        return stack, core, driver, finished, overlap.UID, stops

    def test_decode_abort_leaves_the_in_flight_native_chunk_running(self):
        abort, source = generated_abort()
        stack, core, driver, finished, _uid, stops = self.stack()
        transaction = driver.tx
        abort(core, self.DECODE)
        self.assertEqual(finished, [(self.DECODE, 'aborted')])
        self.assertEqual(stops, [])
        self.assertIsNone(driver.stop_kind)
        self.assertIsNone(driver.error)
        self.assertIs(driver.tx, transaction)
        # The chunk runs to its end exactly as without the abort.
        stack.drive_to_finalize()
        stack.complete_everything()
        stack.drive(lambda: driver.tx is None, limit=400)
        self.assertIsNone(driver.error)
        self.assertIsNone(driver.stop_kind)
        self.assertEqual(stops, [])
        ABORT_RECEIPTS.append(dict(case='decode-abort-chunk-in-flight', verdict='PASS',
                                   generated_method_sha256=hashlib.sha256(source.encode()).hexdigest(),
                                   driver_stopped=False, chunk_completed=True, route_at_abort='split'))

    def test_native_request_abort_stops_the_driver(self):
        abort, _source = generated_abort()
        stack, core, driver, finished, uid, stops = self.stack()
        abort(core, [uid])
        self.assertEqual(finished, [([uid], 'aborted')])
        self.assertEqual(stops, [('abort', {'drain': False})])
        ABORT_RECEIPTS.append(dict(case='native-uid-abort-counterfactual', verdict='PASS', driver_stopped=True))


# --- (7) the exact P-c queue row4 argv --------------------------------------------

def queue_row4_argv(directory):
    text = (RUNS / 'q18a_native35_p_c.draft.sh').read_text()
    invocation = text.split('python3 "$H/$cli"', 1)[1].split('> "$Q/qual-$name.log"', 1)[0]
    values = dict(H=str(HERE), Q=str(directory), PROBE_URL='http://127.0.0.1:1', MODEL='qwen38-w8a8-tp4',
                  RUN_ID='q18a-native35-p-c-cpu', EPOCH='7', Q18A_STORE=str(directory / 'private/qualification-store'),
                  SNAPSHOT_MAX_BYTES=str(512 * MIB), name='row4')
    return [Template(token).substitute(values) for token in shlex.split(invocation.replace('\\\n', ' '))], values


class QueueCLI(unittest.TestCase):
    def test_exact_p_c_row4_argv_reaches_first_network_call_with_the_p_c_runner(self):
        seen = []

        def refuse_request(connection, method, url, body=None, headers=None, **kwargs):
            command = json.loads(body)
            runners = [type(frame.frame.f_locals['self']) for frame in inspect.stack()
                       if frame.function == 'run' and isinstance(frame.frame.f_locals.get('self'), NativePlacementRunner)]
            seen.append(dict(url=url, action=command.get('action'), target_route=command.get('target_route'),
                             runners=runners, authorized=headers.get('Authorization') == 'Bearer ' + KEY))
            raise ConnectionRefusedError('declared CPU boundary')

        text = (RUNS / 'q18a_native35_p_c.draft.sh').read_text()
        self.assertIn('if [[ $name == row4 ]]; then cli=e3_native35_p_c_placement.py; fi', text)
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            argv, values = queue_row4_argv(directory)
            self.assertEqual(argv[:2], ['--phase', 'row4'])
            (directory / 'private').mkdir()
            (directory / 'private/row1-plan.json').write_text(json.dumps({'schema': 'cpu-fixture-row1-plan'}))
            artifact = base.retained_teacher(ImmutableStore(values['Q18A_STORE']), {'schema': 'cpu-fixture', 'rows': []})
            env = dict(VLLM_API_KEY=KEY, Q18A_TEACHER_ARTIFACT_SHA=canonical_sha(artifact), Q18A_ROW1_SHA='e' * 64)
            with mock.patch.dict(os.environ, env), mock.patch('http.client.HTTPConnection.request', refuse_request):
                rc = pc.main(argv)
            self.assertIs(driver_module.NativePlacementRunner, NativePlacementRunner)
            record = json.loads((directory / 'qual-row4.json').read_text())
        self.assertEqual((rc, record['verdict']), (2, 'INCONCLUSIVE'))
        self.assertEqual(len(seen), 1, seen)
        self.assertEqual((seen[0]['url'], seen[0]['action'], seen[0]['target_route']),
                         ('/v1/flash_s5_native_overlap', 'prepare', 'full'))
        self.assertTrue(seen[0]['authorized'])
        self.assertEqual(seen[0]['runners'], [pc.StoppedWindowPlacementRunner])
        self.assertEqual(record['missing'], ['collect:transport-failed:ConnectionRefusedError'])
        # The transport receipt list is attached to the refusal: no observer_end call was made.
        self.assertEqual(record['refusal_evidence']['observer_end_transport'],
                         {'state': 'no-observer-end-call', 'receipts': []})
        self.assertNotIn(KEY, json.dumps(record))
        CLI_RECEIPTS.append(dict(case='exact-p-c-row4-argv', verdict='PASS', first_network_call='prepare-full',
                                 runner='StoppedWindowPlacementRunner', substitution_restored=True,
                                 runner_sha256=hashlib.sha256((HERE / 'e3_native35_p_c_placement.py').read_bytes()).hexdigest()))


if __name__ == '__main__':
    unittest.main()
