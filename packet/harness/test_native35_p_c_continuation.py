"""Native35 P-c row4 capacity continuation (lead decision 31 Oct 2026), CPU only.

(1) Composition.  Source-produced observer packets
    (native_placement_evidence_support.packet: producer-shaped ledgers, the
    real offer grading and the real ctypes report read) go through the real
    driver row4 collection (observer census, native read, prefill regrade) and
    the real e3_q18a_qualification.run grading into qual-row4.json.  The
    classifier reads that record.
    Eligible: a kernel ring overflow, a correlation-list overflow, both.
    Withheld: a clean PASS (nothing to continue), the replay bound exceeded
    beside an overflow, a non-capacity native reason beside ring_overflow, a
    raw FAIL with an overflow, report errors beside a list overflow.
(2) Refusal path over real loopback HTTP: an observer_end body over the 512 MiB
    spool ceiling through the real P-c runner, driver, LoopbackControl and
    e3_q18a_qualification.run is eligible.  The same record without its
    transport receipt, or with P-b's generic response-too-large, is withheld.
(3) Every precondition withholds on its own: client window, open, cleanup,
    crash, error marker, NRestarts, health.
(4) The queue's own bytes (P-c gates + measurement tail) under bash with the
    real classifier, the real P W2 CLI gate and the real aggregate grade.
    Capacity-only: W1 and both W2 rounds pass the gate, gates.json records the
    continuation, row4 stays INCONCLUSIVE.  Mixed reasons, or an engine error
    marker: W1/W2 withheld exactly as before.
(5) The W2 gate re-proves the receipt from the retained files; tampering
    refuses, and gates.json without the continuation is the unchanged gate.
Fixture leaves: native35_p_c_fixture_leaves.json (cpu-native35-p-c-continuation-*).
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / 'patches/s5_engine')]

import e3_decode_correctness as correctness
import e3_native35_p_c_continuation as cont
import e3_native35_p_c_placement as pc
import e3_native35_performance as p35
import e3_native_performance as perf
import e3_q18a_qualification as qualification
from e3_native_qualification_driver import ImmutableStore, canonical_sha
import native_placement_evidence_support as support
import test_e3_engine_offers as offers_fixture
import test_e3_native_qualification_driver as base
import test_native35_p_c_window as window_suite
import test_native35_performance_queue as queue_suite

RANKS = (0, 1, 2, 3)
KEY = window_suite.KEY  # the loopback peer checks it
MIB = 1024 * 1024
ENGINE = {'crashed': False, 'healthy': True, 'nrestarts_start': '4', 'nrestarts': '4',
          'error_markers': 0}
# The P-b box refusal cleanup shape (results/box/q18a-native35-p-b-run-1 qual-row4).
CLEANUP = {'attempted': True, 'owner': 'engine-leave-split', 'restored': True,
           'route': 'full', 'typed_acks': True}
OPEN = {'complete': True, 'opened_streams': 8, 'reasons': []}
COMPOSITION_RECEIPTS, REFUSAL_RECEIPTS, QUEUE_RECEIPTS = [], [], []
OriginalBridge = support.s.FakeBridge


class OverflowBridge(OriginalBridge):
    """The correlation lib at capacity: a list full, overflow counted, report incomplete."""

    def report(self):
        self.overflow = 7
        self.complete = False
        self.capacity = max(len(self.apis), len(self.external), len(self.scopes))
        return super().report()


class ErrorBridge(OverflowBridge):
    """The same, plus a callback error: incompleteness is no longer capacity alone."""

    def report(self):
        self.errors = 1
        return super().report()


def packets(*, kernel=(), correlation=(), native_extra=(), raw_verdict=None, bridge=OverflowBridge):
    """Four source-produced packets; ``kernel`` ranks carry the observer's own
    INC ring-overflow shape, ``correlation`` ranks a full correlation list."""
    out = {}
    for rank in RANKS:
        original = offers_fixture.raw_receipt

        def raw_receipt(*args, _rank=rank, **kwargs):
            if _rank in kernel:
                kwargs['native_reasons'] = ['ring_overflow', 'unattributed_expected_context'] + list(native_extra)
                if raw_verdict:
                    kwargs['verdict'] = raw_verdict
            receipt = original(*args, **kwargs)
            if _rank in kernel:
                receipt['replay']['ring_overflows']['kernel'] = 1200
                receipt['replay']['flags']['permanent_overflow'] = True
            return receipt

        with mock.patch.object(offers_fixture, 'raw_receipt', raw_receipt), \
             mock.patch.object(support.s, 'FakeBridge', bridge if rank in correlation else OriginalBridge):
            out[rank] = support.packet(rank)
    return out


def client_window(window_tokens=None):
    """The real _wait_window record over eight streams that each gained the window."""
    workload = pc.StoppedWindowPlacementRunner(workload=base.placement_workload(),
                                               url='http://127.0.0.1:9').workload
    states = [base.FakeState(index) for index in range(8)]
    starts = [index + 1 for index in range(8)]
    for state, start in zip(states, starts):
        state.value = start + (workload.window_tokens if window_tokens is None else window_tokens)
    return correctness._wait_window(states, starts, workload)


def raw_for(source, observed=44):
    """The runner's collected return shape, with the source packets in it."""
    raw = base.placement_result(epoch=3, run_id='run-1')
    raw['window'].update(window_id='win-1', route_generation=7,
                         observed_replays={rank: observed for rank in RANKS})
    raw['window_route_generation'] = 7
    raw['raw_placement'] = {rank: row['receipt'] for rank, row in source.items()}
    raw['engine_accounting'] = {rank: row['engine_accounting'] for rank, row in source.items()}
    raw['raw_sha256'] = {str(rank): canonical_sha(row['receipt']) for rank, row in source.items()}
    raw['raw_graph_receipts'] = copy.deepcopy(raw['raw_placement'])
    raw['engine_accounting_receipts'] = copy.deepcopy(raw['engine_accounting'])
    raw['native_observer_packets'] = source
    raw['cleanup'] = dict(CLEANUP)
    raw['client_window'] = client_window()
    raw['client_open'] = dict(OPEN)
    return raw


def collected_record(raw):
    """qual-row4.json written by the real e3_q18a_qualification.run."""
    with tempfile.TemporaryDirectory() as d:
        store = Path(d) / 'store'
        artifact = base.retained_teacher(ImmutableStore(str(store)), base.teacher_evidence(None))
        out = Path(d) / 'qual-row4.json'
        rc = qualification.run('row4', 'http://127.0.0.1:9', 'm', 'run-1', 3, str(out),
                               artifact_sha256=canonical_sha(artifact), row1_sha256='e' * 64,
                               store=str(store), placement=base.FakePlacement(raw),
                               window_id='win-1', key=KEY, env={})
        return rc, out.read_bytes()


_CACHE = {}


def record(label):
    if label not in _CACHE:
        cases = {
            'clean': lambda: raw_for(packets()),
            'kernel': lambda: raw_for(packets(kernel={2})),
            'correlation': lambda: raw_for(packets(correlation={1})),
            'both': lambda: raw_for(packets(kernel={2}, correlation={1})),
            'kernel-replays-60': lambda: raw_for(packets(kernel={2}), observed=60),
            'kernel-unknown-node': lambda: raw_for(packets(kernel={2}, native_extra=('unknown_node',))),
            'kernel-raw-fail': lambda: raw_for(packets(kernel={2}, native_extra=('known_wrong_context',),
                                                       raw_verdict='FAIL')),
            'correlation-errors': lambda: raw_for(packets(correlation={1}, bridge=ErrorBridge)),
        }
        _CACHE[label] = collected_record(cases[label]())
    rc, data = _CACHE[label]
    return rc, json.loads(data), data


class Composition(unittest.TestCase):
    def assert_eligible(self, label, primaries):
        rc, row4, _ = record(label)
        self.assertEqual((rc, row4['verdict'], row4['failures']), (2, 'INCONCLUSIVE', []))
        result = cont.classify(row4, engine=ENGINE)
        self.assertTrue(result['eligible'], (result['unmatched'], result['checks']))
        self.assertEqual(result['unmatched'], [])
        self.assertEqual(sorted(r['reason'] for r in result['matched']), sorted(row4['missing']))
        for reason in primaries:
            self.assertIn({'reason': reason, 'rule': 'primary:' + reason.split(':')[2]}, result['matched'])
        COMPOSITION_RECEIPTS.append(dict(case=label, verdict='PASS', eligible=True, row4='INCONCLUSIVE',
                                         missing=row4['missing'], real_grader=True))
        return row4, result

    def test_kernel_ring_overflow_is_capacity_only(self):
        row4, _ = self.assert_eligible('kernel', ['collector:observer:kernel-ring-overflow:2'])
        # The exact strings the unchanged graders produce for a ring overflow.
        for reason in ('row4:r2:composition:raw_not_dischargeable:ring_overflow,unattributed_expected_context',
                       'row4:r2:binding', 'row4:r2:composed-inc:', 'collector:phase-incomplete',
                       'collector:native-prefill-ledger-unproved'):
            self.assertIn(reason, row4['missing'])

    def test_correlation_list_overflow_is_capacity_only(self):
        row4, _ = self.assert_eligible('correlation', ['collector:observer:scope-ring-overflow:1'])
        self.assertIn('row4:r1:composition:baseline:correlation_overflow', row4['missing'])
        self.assertIn('collector:observer:correlation-report-incomplete:1', row4['missing'])

    def test_both_kinds_on_two_ranks_are_capacity_only(self):
        self.assert_eligible('both', ['collector:observer:kernel-ring-overflow:2',
                                      'collector:observer:scope-ring-overflow:1'])

    def test_clean_pass_and_mixed_or_other_outcomes_are_withheld(self):
        rc, row4, _ = record('clean')
        self.assertEqual((rc, row4['verdict']), (0, 'PASS'))
        self.assertFalse(cont.classify(row4, engine=ENGINE)['eligible'])  # PASS takes the unchanged gate
        COMPOSITION_RECEIPTS.append(dict(case='clean-pass', verdict='PASS', eligible=False, row4='PASS'))
        cases = {
            'kernel-replays-60': ['collector:observer:replays-above-tolerance:%d:60>56' % r for r in RANKS],
            'kernel-unknown-node': ['row4:r2:composition:raw_not_dischargeable:ring_overflow,unattributed_expected_context,unknown_node'],
            'correlation-errors': ['collector:observer:correlation-report-incomplete:1:errors',
                                   'row4:r1:composition:baseline:correlation_errors'],
        }
        for label, expected in cases.items():
            with self.subTest(label=label):
                rc, row4, _ = record(label)
                self.assertEqual((rc, row4['verdict'], row4['failures']), (2, 'INCONCLUSIVE', []))
                result = cont.classify(row4, engine=ENGINE)
                self.assertFalse(result['eligible'])
                for reason in expected:
                    self.assertIn(reason, result['unmatched'])
                # The run-level consequences are never matched beside an unmatched reason.
                self.assertIn('collector:phase-incomplete', result['unmatched'])
                COMPOSITION_RECEIPTS.append(dict(case=label, verdict='PASS', eligible=False,
                                                 unmatched=result['unmatched']))
        rc, row4, _ = record('kernel-raw-fail')
        self.assertEqual((rc, row4['verdict']), (1, 'FAIL'))
        result = cont.classify(row4, engine=ENGINE)
        self.assertFalse(result['eligible'])
        self.assertFalse(result['checks']['row4_failures_empty'])
        COMPOSITION_RECEIPTS.append(dict(case='kernel-raw-fail', verdict='PASS', eligible=False,
                                         failures=row4['failures']))


class Preconditions(unittest.TestCase):
    def test_each_precondition_withholds_on_its_own(self):
        _, row4, _ = record('kernel')
        self.assertTrue(cont.classify(row4, engine=ENGINE)['eligible'])
        evidence = 'evidence'
        mutations = {
            'client_window_complete': lambda r: r[evidence]['client_window'].update(complete=False),
            'client_window_token_bound': lambda r: r[evidence]['client_window'].update(reason='wall_bound'),
            'client_window_tokens': lambda r: r[evidence]['client_window']['streams'][3].update(completion_delta=19),
            'client_window_streams': lambda r: r[evidence]['client_window']['streams'].pop(),
            'client_window_no_early_finish': lambda r: r[evidence]['client_window'].update(early_finish=[5]),
            'client_window_no_stream_error': lambda r: r[evidence]['client_window'].update(errors_before_end=[2]),
            'client_open_all_streams': lambda r: r[evidence]['client_open'].update(opened_streams=7),
            'client_open_complete': lambda r: r[evidence]['client_open'].update(reasons=['stream_not_advanced:7']),
            'cleanup_restored_full': lambda r: r[evidence]['placement_cleanup'].update(route='split'),
            'cleanup_typed_acks': lambda r: r[evidence]['placement_cleanup'].update(typed_acks=False),
        }
        for check, mutate in mutations.items():
            with self.subTest(check=check):
                changed = copy.deepcopy(row4)
                mutate(changed)
                result = cont.classify(changed, engine=ENGINE)
                self.assertFalse(result['eligible'])
                self.assertFalse(result['checks'][check])
        engines = {
            'engine_not_crashed': dict(ENGINE, crashed=True),
            'engine_healthy': dict(ENGINE, healthy=False),
            'engine_nrestarts_unchanged': dict(ENGINE, nrestarts='5'),
            'engine_no_error_marker': dict(ENGINE, error_markers=2),
        }
        for check, engine in list(engines.items()) + [
                ('engine_nrestarts_unchanged', dict(ENGINE, nrestarts='unknown')),
                ('engine_no_error_marker', dict(ENGINE, error_markers='unavailable'))]:
            with self.subTest(check=check, engine=engine):
                result = cont.classify(row4, engine=engine)
                self.assertFalse(result['eligible'])
                self.assertFalse(result['checks'][check])
        # Absent records (a runner that does not return them) withhold too.
        bare = copy.deepcopy(row4)
        for key in ('client_window', 'client_open', 'placement_cleanup'):
            bare[evidence][key] = None
        self.assertFalse(cont.classify(bare, engine=ENGINE)['eligible'])


# --- (2) refusal path over real loopback HTTP --------------------------------

class QualComposition(window_suite.Composition):
    """The window suite's loopback peer, collected through e3_q18a_qualification.run."""

    def collect(self, runner):
        self.runner = runner
        out = Path(self.out_dir) / 'qual-row4.json'
        rc = qualification.run('row4', self.url, 'm', 'run-1', 3, str(out),
                               artifact_sha256=canonical_sha(self.artifact), row1_sha256='e' * 64,
                               store=self.root, placement=runner, window_id='window-1', key=KEY,
                               timeout_s=60.0, env={})
        return rc, json.loads(out.read_text())


class Refusal(unittest.TestCase):
    def test_observer_end_over_the_spool_ceiling_is_capacity_only(self):
        size = 512 * MIB + MIB
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as out_dir:
            holder = {}

            def control(timeline):
                holder['c'] = QualComposition(root, timeline,
                                              observer_end=lambda peer, command: ('stream', size))
                holder['c'].out_dir = out_dir
                return holder['c']

            try:
                outcome, _, _, stops, _ = window_suite.run_timeline(
                    pc.StoppedWindowPlacementRunner, initial=window_suite.Timing.BOX_INITIAL,
                    open_tick=1, control=control)
            finally:
                if 'c' in holder:
                    holder['c'].close()
        rc, row4 = outcome
        self.assertEqual((rc, row4['verdict'], row4['missing'], row4['failures']),
                         (2, 'INCONCLUSIVE', ['collect:observer-end-response-too-large'], []))
        transport = row4['refusal_evidence']['observer_end_transport']['receipts'][-1]
        self.assertEqual((transport['body_bytes'], transport['state']), (size, 'refused:too-large'))
        self.assertEqual(stops[0]['kind'], 'decode')
        result = cont.classify(row4, engine=ENGINE)
        self.assertTrue(result['eligible'], (result['unmatched'], result['checks']))
        self.assertEqual(result['path'], 'refusal')
        self.assertEqual(result['matched'], [{'reason': 'collect:observer-end-response-too-large',
                                              'rule': 'primary:observer-end-over-spool-ceiling'}])
        # Counterfactuals on the same record.
        bare = copy.deepcopy(row4)
        bare['refusal_evidence']['observer_end_transport'] = {'state': 'no-observer-end-call', 'receipts': []}
        generic = copy.deepcopy(row4)
        generic['missing'] = ['collect:response-too-large']  # P-b's transport cap, not the spool ceiling
        unrestored = copy.deepcopy(row4)
        unrestored['refusal_cleanup']['restored'] = False
        for label, changed in (('no-transport-receipt', bare), ('generic-cap', generic), ('cleanup', unrestored)):
            with self.subTest(label=label):
                self.assertFalse(cont.classify(changed, engine=ENGINE)['eligible'])
        REFUSAL_RECEIPTS.append(dict(case='observer-end-over-spool-ceiling', verdict='PASS', eligible=True,
                                     body_bytes=size, real_http=True, real_qualification_run=True,
                                     counterfactuals=['no-transport-receipt', 'generic-cap', 'cleanup']))


# --- (4) the queue's own bytes -------------------------------------------------

class QueueBytes(unittest.TestCase):
    """P-c gates + measurement under bash: real classifier, real W2 gate, real grade."""

    def run_queue(self, fake, row4_bytes, *, stream=b'INFO ordinary engine line\n'):
        p = queue_suite.PC_DRAFT.read_text()
        functions = ''.join(p[p.index(name + '() {\n'):p.index('\n}\n', p.index(name + '() {\n')) + 3]
                            for name in ('step', 'record'))
        _, _, gates, measurement = queue_suite.p_parts(queue_suite.PC_DRAFT)
        h = fake / 'h'
        h.mkdir()
        for name in ('e3_native_w1', 'e3_native_memory_grade', 'e3_native28_rule4_continuation'):
            (h / (name + '.py')).write_text(queue_suite.FAKE_PHASE)
        for name in ('e3_native35_performance.py', 'e3_native35_p_c_continuation.py', 'native35_performance_grade.py'):
            shutil.copyfile(HERE / name, h / name)  # the real CLI gate, classifier and aggregate grade
        q = fake / 'q'
        q.mkdir()
        (q / 'qual-row4.json').write_bytes(row4_bytes)
        (q / 'engine-stream.log').write_bytes(stream)
        (q / 'engine-stream.err').write_bytes(b'')
        verdicts = dict(queue_suite.MeasurementTail.QUALIFIED, row4=json.loads(row4_bytes)['verdict'])
        preset = ''.join('PHASE_VERDICT[%s]=%s\n' % item for item in verdicts.items())
        script = ('set -uo pipefail\nFAILS=0; INCS=0; ENGINE_CRASHED=0; N0=4; SERVICE=vllm-flash\ndeclare -A PHASE_VERDICT=()\n'
                  + functions + 'health_ok() { return 0; }\nhost_crash_detected() { return 1; }\nsystemctl() { echo 4; }\n'
                  + preset + gates + measurement
                  + 'for k in "${!PHASE_VERDICT[@]}"; do echo "PV $k=${PHASE_VERDICT[$k]}"; done\n'
                  + 'echo "TOTAL FAILS=$FAILS INCS=$INCS ENGINE_CRASHED=$ENGINE_CRASHED"\n')
        env = dict(os.environ, FAKE=str(fake), H=str(h), Q=str(q), PROBE_URL='http://127.0.0.1:1', MODEL='m',
                   RUN_ID='q18a-native35-p-c-cpu-1', EPOCH='7', Q18A_FIRST_ADVANCE_READ_TIMEOUT_S='1200',
                   Q18A_READINESS='/overlay/e3/native-overlap-readiness.json', S5_OVERLAY='/overlay',
                   S5_LAUNCHER_NAME='run-flash-s5-test.sh', PYTHONDONTWRITEBYTECODE='1',
                   PYTHONPATH=os.pathsep.join((str(HERE), str(ROOT / 'patches/s5_engine'))))
        done = subprocess.run(['bash', '-c', script], env=env, capture_output=True, text=True, timeout=300,
                              cwd=str(fake))
        phases = dict(line[3:].split('=', 1) for line in done.stdout.splitlines() if line.startswith('PV '))
        calls = ([json.loads(line) for line in (fake / 'calls.jsonl').read_text().splitlines()]
                 if (fake / 'calls.jsonl').exists() else [])
        return done, phases, calls, q

    def test_capacity_only_row4_runs_w1_and_both_rounds_and_row4_stays_inconclusive(self):
        _, _, data = record('kernel')
        with tempfile.TemporaryDirectory() as d:
            done, phases, calls, q = self.run_queue(Path(d), data)
            self.assertIn('STEP row4-capacity-continuation PASS', done.stdout, done.stderr)
            self.assertEqual(phases['row4'], 'INCONCLUSIVE')
            self.assertEqual([c['name'] for c in calls], ['e3_native_w1', 'e3_native_memory_grade', 'e3_native_memory_grade'])
            receipt_bytes = (q / 'row4-capacity-continuation.json').read_bytes()
            receipt = json.loads(receipt_bytes)
            gates = json.loads((q / 'gates.json').read_text())
            self.assertEqual(gates['outcomes']['row4'], 'INCONCLUSIVE')
            self.assertIs(gates['row4_capacity_continuation'], True)
            self.assertEqual(gates['row4_capacity_continuation_sha256'], hashlib.sha256(receipt_bytes).hexdigest())
            self.assertEqual(gates['row4_capacity_matched'], [row['reason'] for row in receipt['matched']])
            self.assertEqual(receipt['row4_record_sha256'], hashlib.sha256(data).hexdigest())
            self.assertEqual(receipt['engine'], dict(ENGINE, error_marker_basis=receipt['engine']['error_marker_basis']))
            # Both rounds ran the real W2 CLI past its same-queue gate; they stop at
            # the package gate because the CPU fixture has no overlay.
            for index in (0, 1):
                artifact = json.loads((q / ('overlap-round-%d.json' % index)).read_text())
                self.assertEqual(artifact['error']['error_message'], 'reviewed-package-readiness-not-pass')
            self.assertIn('STEP overlap-gates PASS', done.stdout)
            self.assertEqual(json.loads((q / 'native35-performance-grade.json').read_text())['gates']['row4'], 'INCONCLUSIVE')
        QUEUE_RECEIPTS.append(dict(case='capacity-only-continues', verdict='PASS', w1_run=True, w2_gate_passed=[0, 1],
                                   row4='INCONCLUSIVE', gates_continuation=True))

    def test_mixed_reasons_or_an_engine_error_marker_withhold_w1_and_w2_as_before(self):
        cases = (('mixed', record('kernel-replays-60')[2], b'INFO ordinary engine line\n'),
                 ('error-marker', record('kernel')[2], b'INFO line\nTraceback (most recent call last):\n'))
        for label, data, stream in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as d:
                done, phases, calls, q = self.run_queue(Path(d), data, stream=stream)
                self.assertIn('STEP row4-capacity-continuation INCONCLUSIVE', done.stdout, done.stderr)
                self.assertEqual(calls, [])
                self.assertIn('STEP overlap-gates INCONCLUSIVE', done.stdout)
                self.assertEqual((phases['row4'], phases['w1_controls'], phases['overlap']),
                                 ('INCONCLUSIVE', 'INCONCLUSIVE', 'INCONCLUSIVE'))
                gates = json.loads((q / 'gates.json').read_text())
                self.assertIs(gates['row4_capacity_continuation'], False)
                self.assertEqual((gates['row4_capacity_continuation_sha256'], gates['row4_capacity_matched']), (None, []))
                receipt = json.loads((q / 'row4-capacity-continuation.json').read_text())
                self.assertFalse(receipt['eligible'])
                if label == 'error-marker':
                    self.assertEqual(receipt['engine']['error_markers'], 1)
                QUEUE_RECEIPTS.append(dict(case=label + '-withheld', verdict='PASS', w1_run=False, w2_run=False))


# --- (5) the W2 gate -----------------------------------------------------------

class Gate(unittest.TestCase):
    def folder(self, d, data, *, engine=ENGINE):
        q = Path(d)
        (q / 'qual-row4.json').write_bytes(data)
        receipt = cont.classify(json.loads(data), engine=engine)
        receipt.update(row4_record='qual-row4.json', row4_record_sha256=hashlib.sha256(data).hexdigest())
        receipt_bytes = json.dumps(receipt, sort_keys=True).encode()
        (q / cont.RECEIPT_NAME).write_bytes(receipt_bytes)
        outcomes = {k: 'PASS' for k in ('startup', 'row1', 'rule4_controls', 'rule4_candidate', 'w1_controls')}
        outcomes['row4'] = 'INCONCLUSIVE'
        gates = {'schema': 1, 'outcomes': outcomes, 'no_retry': True, 'calibration_frozen': True,
                 'rule4_control_continuation': False, 'rule4_continuation_sha256': None,
                 'row4_capacity_continuation': True,
                 'row4_capacity_continuation_sha256': hashlib.sha256(receipt_bytes).hexdigest(),
                 'row4_capacity_matched': [row['reason'] for row in receipt['matched']]}
        return q, gates

    def test_receipt_is_reproved_and_tampering_refuses(self):
        _, _, data = record('kernel')
        with tempfile.TemporaryDirectory() as d:
            q, gates = self.folder(d, data)
            path = str(q / 'gates.json')
            self.assertTrue(p35.gates_allow_run(gates, path))
            # Without the continuation the same outcomes are the unchanged gate's refusal.
            plain = {k: v for k, v in gates.items() if not k.startswith('row4_capacity_')}
            self.assertFalse(p35.gates_allow_run(plain, path))
            self.assertFalse(perf.gates_allow_run(plain, path))
            refusals = {
                'matched-list': dict(gates, row4_capacity_matched=gates['row4_capacity_matched'][:-1]),
                'receipt-sha': dict(gates, row4_capacity_continuation_sha256='0' * 64),
                'w1-not-pass': dict(gates, outcomes=dict(gates['outcomes'], w1_controls='INCONCLUSIVE')),
                'row4-fail': dict(gates, outcomes=dict(gates['outcomes'], row4='FAIL')),
            }
            for label, changed in refusals.items():
                with self.subTest(label=label):
                    self.assertFalse(p35.gates_allow_run(changed, path))
            (q / 'qual-row4.json').write_bytes(data + b' ')
            self.assertFalse(p35.gates_allow_run(gates, path))
        with tempfile.TemporaryDirectory() as d:
            # A receipt whose cached eligible flag is not reproduced by its own inputs.
            q, gates = self.folder(d, data, engine=dict(ENGINE, error_markers=3))
            receipt = json.loads((q / cont.RECEIPT_NAME).read_text())
            receipt['eligible'] = True
            receipt_bytes = json.dumps(receipt, sort_keys=True).encode()
            (q / cont.RECEIPT_NAME).write_bytes(receipt_bytes)
            gates['row4_capacity_continuation_sha256'] = hashlib.sha256(receipt_bytes).hexdigest()
            self.assertFalse(p35.gates_allow_run(gates, str(q / 'gates.json')))

    def test_gates_without_the_continuation_are_the_unchanged_gate(self):
        outcomes = {k: 'PASS' for k in ('startup', 'row1', 'rule4_controls', 'rule4_candidate', 'row4', 'w1_controls')}
        for gates in ({'schema': 1, 'outcomes': outcomes},
                      {'schema': 1, 'outcomes': outcomes, 'row4_capacity_continuation': False},
                      {'schema': 1, 'outcomes': dict(outcomes, row4='INCONCLUSIVE')}):
            with self.subTest(gates=gates):
                self.assertEqual(p35.gates_allow_run(gates, '/nonexistent/gates.json'),
                                 perf.gates_allow_run(gates, '/nonexistent/gates.json'))


if __name__ == '__main__':
    unittest.main()
