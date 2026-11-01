"""Held-g L4: lean replies on the real driver, bridges, dispatchers and census.

The connected cases run test_e3_r4_worker_bridge.Stack (real admission
controller, NativeSegmentDriver, four NativeWorkerBridge/LaneDispatcher
workers, PP channel, measurement census) once with full and once with lean
replies, under the same seeded order of device-event completions, and compare
what every consumer reads.

Fixture leaves (besides r4's own, declared there):

* reply-format selection: ``NativeSegmentDriver._open_replies`` returns the
  format under test. It replaces the arm selection (``native_progress``
  ``replies``) and the auto-arm ownership check, which the held-g container
  probe runs for real; everything after the OPEN command is production code.
* completion order: a seeded subset of the fake device events completes per
  driver tick (r4's ControllableEvent), so completions land out of op order
  and pending sets are not empty; both formats see the same order.
"""
import collections
import copy
from dataclasses import replace
import pickle
import random
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))

import engine_driver as ed  # noqa: E402
import wire_v1 as w  # noqa: E402
from e3_types import E3ContractError, TransactionId  # noqa: E402
from worker_bridge import WorkerBridgeError  # noqa: E402

import test_e3_r4_worker_bridge as r4  # noqa: E402

ENTRY_KEYS = {'kind', 'observed_ns', 'clock_basis', 'engine_epoch', 'route_generation', 'route',
              'transaction', 'worker_rank', 'completed_sequence', 'completed_op_seqs',
              'pending_op_seqs', 'failed_code', 'state_written', 'proofs'}


def normal(report):
    """Everything a consumer reads from a cumulative report; proofs as a set."""
    return dict(type=type(report).__name__, transaction=report.transaction,
                rank=report.worker_rank, completed=report.completed_op_seqs,
                pending=report.pending_op_seqs, watermark=report.completed_sequence,
                proofs=sorted((p.ticket.op_seq, p.kind.value, p.completion_seq, p.ticket)
                              for p in report.proofs),
                hosts=report.host_submit_ns, cursor=report.cursor,
                state_epoch=report.state_epoch, failed=report.failed_code,
                state_written=report.state_written())


def validating(**fields):
    """The validating constructor in place of wire_v1._accumulated."""
    return w.WorkerCompletionReport(**fields)


class Run:
    """One connected r4 chunk in one reply format, with every reply captured."""

    def __init__(self, replies, *, seed=7, fraction=0.5):
        self.stack = stack = r4.Stack()
        stack.driver._open_replies = lambda: replies
        self.rng = random.Random(seed)
        self.fraction = fraction
        self.replies = []          # (kind, values, pickled bytes, transport sequence)
        self.commits = []
        self.release_requests = []
        self.worker_ends = []
        real_submit = stack.transport.submit

        def submit(kind, send, **kwargs):
            sequence = real_submit(kind, send, **kwargs)
            values = stack.transport.ready.get(sequence)
            self.replies.append((kind, values, len(pickle.dumps(values, protocol=5)), sequence))
            return sequence
        stack.transport.submit = submit
        real_commit = stack.controller.commit

        def commit(transaction, proof):
            self.commits.append({rank: copy.copy(report)
                                 for rank, report in stack.driver.reports.items()})
            return real_commit(transaction, proof)
        stack.controller.commit = commit
        for rank, worker in stack.workers.items():
            real_release = worker._flash_s5_bridge.release

            def release(request, worker=worker, real_release=real_release):
                self.release_requests.append(request)
                if worker._flash_s5_bridge._lean is not None:
                    self.worker_ends.append((worker._flash_s5_bridge.rank, worker._flash_s5_bridge._lean.end()))
                return real_release(request)
            worker._flash_s5_bridge.release = release

    def complete_some(self):
        for worker in self.stack.workers.values():
            for event in worker.bus.events:
                if not event.complete and self.rng.random() < self.fraction:
                    event.complete = True

    def drive(self, limit=600):
        stack = self.stack
        for _ in range(limit):
            stack.advance()
            if stack.driver.phase == 'finalize':
                stack.complete_everything()
            else:
                self.complete_some()
            if stack.cache_calls() and stack.driver.phase == 'idle':
                return self
        raise AssertionError('chunk did not commit: phase=%s error=%s'
                             % (stack.driver.phase, stack.driver.error))

    def reports(self, applied=False):
        """Every typed report the replies carried, per rank, in reply order.

        ``applied``: only replies the driver has already taken.
        """
        out = collections.defaultdict(list)
        for kind, values, _size, sequence in self.replies:
            if applied and sequence in self.stack.transport.ready:
                continue
            for value in values or ():
                report = getattr(value, 'report', value)
                if type(report) in (w.WorkerCompletionReport, w.LeanCompletionReport):
                    out[report.worker_rank].append(report)
        return out

    def tokens(self):
        return sum(1 for kind, values, _size, _seq in self.replies if kind == 'segment'
                   for value in values if value.report is None)

    def journal_reports(self):
        return [entry for entry in self.stack.driver._journal if entry['kind'] == 'report']

    def census(self):
        census = self.stack.driver.measurement_census
        out = {}
        for txn, chunk in census._chunks.items():
            ranks = {}
            for rank, record in chunk['ranks'].items():
                ranks[rank] = dict(
                    completed=sorted(record['completed_ops'] or ()),
                    proofs=sorted((key, value['completion_seq'])
                                  for key, value in record['proofs'].items()),
                    watermark=record['completed_sequence'],
                    state_write=(record['state_write'] or {}).get('op_seq'),
                    pp_consumer=(record['pp_consumer'] or {}).get('op_seq'),
                    failed=record['failed_code'])
            out[txn] = dict(ranks=ranks, conflicts=list(chunk['conflicts']))
        return out


class LeanConnected(unittest.TestCase):
    def runs(self, seed):
        return Run('full', seed=seed).drive(), Run('lean', seed=seed).drive()

    def test_lean_and_full_drive_the_same_chunk_to_the_same_reports(self):
        for seed in (1, 7, 23):
            with self.subTest(seed=seed):
                full, lean = self.runs(seed)
                # The same commands in the same order: lean changes no decision.
                self.assertEqual([e[0] for e in full.stack.sender.envelopes],
                                 [e[0] for e in lean.stack.sender.envelopes])
                self.assertEqual([r[0] for r in full.replies], [r[0] for r in lean.replies])
                # What the controller and every report consumer read at commit.
                self.assertEqual(len(full.commits), 1)
                self.assertEqual(len(lean.commits), 1)
                self.assertEqual({r: normal(x) for r, x in full.commits[0].items()},
                                 {r: normal(x) for r, x in lean.commits[0].items()})
                self.assertEqual(full.stack.cache_calls(), lean.stack.cache_calls())
                self.assertEqual(full.stack.driver.error, None)
                self.assertEqual(lean.stack.driver.error, None)
                # The census holds the same per-rank facts and no conflict.
                self.assertEqual(full.census(), lean.census())
                for chunk in lean.census().values():
                    self.assertEqual(chunk['conflicts'], [])
                # Lean really ran lean, full really ran full.
                self.assertTrue(all(type(x) is w.LeanCompletionReport
                                    for xs in lean.reports().values() for x in xs))
                self.assertTrue(all(type(x) is w.WorkerCompletionReport
                                    for xs in full.reports().values() for x in xs))
                self.assertGreater(lean.tokens(), 0)
                self.assertEqual(full.tokens(), 0)
                # The release carried the engine's applied stream, equal to what
                # each worker sent; a full release carries none.
                self.assertTrue(all(not r.lean_streams for r in full.release_requests))
                ends = dict(lean.worker_ends)
                for request in lean.release_requests:
                    self.assertEqual({e[0]: e[1:] for e in request.lean_streams}, ends)
                full_bytes = sum(r[2] for r in full.replies if r[0] in ('segment', 'poll'))
                lean_bytes = sum(r[2] for r in lean.replies if r[0] in ('segment', 'poll'))
                self.assertLess(lean_bytes, full_bytes)

    def test_out_of_order_completions_are_exercised(self):
        # The seeded order is a real test of the merge: some report completes
        # an op below an op it reported before, and some proof is owed later.
        lean = Run('lean', seed=7).drive()
        late_below = owed = False
        for reports in lean.reports().values():
            seen = set()
            for report in reports:
                if seen and report.completed_op_seqs and min(report.completed_op_seqs) < max(seen):
                    late_below = True
                if any(p.ticket.op_seq in seen for p in report.proofs):
                    owed = True
                seen.update(report.completed_op_seqs)
        self.assertTrue(late_below)
        self.assertTrue(owed)

    def test_validating_constructor_gives_the_same_run(self):
        trusted = Run('lean', seed=7).drive()
        with mock.patch.object(w, '_accumulated', validating):
            checked = Run('lean', seed=7).drive()
        self.assertEqual({r: normal(x) for r, x in trusted.commits[0].items()},
                         {r: normal(x) for r, x in checked.commits[0].items()})
        strip = lambda entries: [{k: v for k, v in e.items() if k != 'observed_ns'} for e in entries]
        self.assertEqual(strip(trusted.journal_reports()), strip(checked.journal_reports()))

    def test_journal_report_entries_keep_their_keys(self):
        full, lean = self.runs(7)
        for entry in full.journal_reports():
            self.assertEqual(set(entry), ENTRY_KEYS)
        lean_entries = lean.journal_reports()
        self.assertTrue(lean_entries)
        for entry in lean_entries:
            self.assertEqual(set(entry), ENTRY_KEYS | {'lean'})
            self.assertEqual(set(entry['lean']), {'base_completed', 'base_proofs', 'base_hosts',
                                                  'completed', 'proofs', 'hosts', 'digest'})
        # Expanding the deltas gives the full run's last cumulative vectors.
        last_full = {e['worker_rank']: e for e in full.journal_reports()}
        expanded = collections.defaultdict(set)
        for entry in lean_entries:
            expanded[entry['worker_rank']].update(entry['completed_op_seqs'])
        self.assertEqual({r: sorted(v) for r, v in expanded.items()},
                         {r: list(e['completed_op_seqs']) for r, e in last_full.items()})


class LeanDriverRefusals(unittest.TestCase):
    """Faulty lean reports reach the real driver mid-chunk: each is a driver error."""

    def setUp(self):
        self.run = Run('lean', seed=7)
        stack = self.run.stack
        # Stop with a lean stream open on every rank, mid-plan.
        for _ in range(400):
            stack.advance()
            self.run.complete_some()
            if stack.driver.phase == 'segments' and len(stack.driver._lean_streams) == 4 \
                    and any(len(xs) >= 3 for xs in self.run.reports(applied=True).values()):
                break
        self.driver = stack.driver
        self.rank, reports = max(self.run.reports(applied=True).items(),
                                 key=lambda item: len(item[1]))
        self.last = reports[-1]
        self.before = self.driver._lean_streams[self.rank].end()

    def refuse(self, report, text):
        with self.assertRaises(ed.DriverError) as caught:
            self.driver._ingest_report(report)
        self.assertIn(text, str(caught.exception))
        self.assertEqual(self.driver._lean_streams[self.rank].end(), self.before)

    def test_replay_of_the_last_report(self):
        self.refuse(self.last, 'does not continue the applied stream')

    def test_foreign_and_stale_transactions(self):
        tx = self.last.transaction
        foreign = TransactionId(tx.engine_epoch, 'other-request', tx.request_generation,
                                tx.chunk_start, tx.chunk_end, tx.txn_seq)
        stale = replace(tx, txn_seq=tx.txn_seq - 1) if tx.txn_seq else replace(tx, txn_seq=tx.txn_seq + 1)
        for transaction in (foreign, stale):
            forged = object.__new__(w.LeanCompletionReport)
            for name in w.LeanCompletionReport.__dataclass_fields__:
                object.__setattr__(forged, name, getattr(self.last, name))
            object.__setattr__(forged, 'transaction', transaction)
            self.refuse(forged, 'belongs to another transaction')

    def test_lean_report_in_a_full_transaction(self):
        self.driver._replies = 'full'
        with self.assertRaises(ed.DriverError) as caught:
            self.driver._ingest_report(self.last)
        self.assertIn('full-reply transaction', str(caught.exception))

    def test_unaccepted_operation(self):
        forged = replace(self.last, completed_op_seqs=(), pending_op_seqs=(10_000,),
                         proofs=(), host_submit_ns=())
        self.refuse(forged, 'unaccepted operation')


class LeanStreamOrder(unittest.TestCase):
    """The stream on real bridge-built lean reports: gap, replay, reorder, substitution."""

    @classmethod
    def setUpClass(cls):
        run = Run('lean', seed=7).drive()
        cls.streams = {rank: [x for x in xs if not x.empty()] for rank, xs in run.reports().items()}
        cls.rank, cls.reports = max(cls.streams.items(), key=lambda item: len(item[1]))
        assert len(cls.reports) >= 4

    def apply(self, reports):
        stream = w.LeanStream()
        for report in reports:
            stream.extend(report)
        return stream

    def test_the_sent_order_applies(self):
        self.assertIsNotNone(self.apply(self.reports).report)

    def test_gap_replay_reorder(self):
        r = self.reports
        for name, order in (('gap', r[:1] + r[2:]), ('replay', r[:2] + r[1:]),
                            ('reorder', [r[0], r[2], r[1]] + r[3:])):
            with self.subTest(name), self.assertRaises(E3ContractError) as caught:
                self.apply(order)
            self.assertIn('does not continue the applied stream', str(caught.exception))

    def substituted(self):
        """A report with one completed and one pending op_seq swapped: every count,
        base and the digest kept, every other check still satisfied."""
        r = self.reports
        index = next(i for i in range(1, len(r))
                     if r[i].completed_op_seqs and r[i].pending_op_seqs)
        victim = r[index]
        done, pending = victim.completed_op_seqs, victim.pending_op_seqs
        return r[:index], victim, replace(
            victim, completed_op_seqs=tuple(sorted(done[:-1] + pending[:1])),
            pending_op_seqs=tuple(sorted(pending[1:] + done[-1:])), proofs=(),
            host_submit_ns=())

    def test_substitution_breaks_the_chain(self):
        prefix, _victim, forged = self.substituted()
        stream = self.apply(prefix)
        with self.assertRaises(E3ContractError) as caught:
            stream.extend(forged)
        self.assertIn('chain differs', str(caught.exception))

    def test_mutant_without_the_chain_accepts_the_substitution(self):
        class Equal(bytes):
            def __ne__(self, other):
                return False
        prefix, victim, forged = self.substituted()
        with mock.patch.object(w, 'lean_digest', lambda *a, **k: Equal(32)):
            stream = self.apply(prefix)
            stream.extend(forged)
        # The mutant took an operation the worker reported pending as completed.
        self.assertIn(victim.pending_op_seqs[0], stream.completed)

    def test_a_full_report_closes_the_stream(self):
        stream = self.apply(self.reports[:1])
        stream.closed = True
        with self.assertRaises(E3ContractError):
            stream.extend(self.reports[1])


class LeanRelease(unittest.TestCase):
    """The worker's release check on the real bridge, against forged release requests."""

    def setUp(self):
        self.run = Run('lean', seed=7)
        stack = self.run.stack
        # The driver reaches its release; the request itself is held back here.
        stack.driver._send_release = lambda: None
        for _ in range(600):
            if stack.driver.phase == 'release':
                break
            stack.advance()
            if stack.driver.phase == 'finalize':
                stack.complete_everything()
            else:
                self.run.complete_some()
        self.driver = stack.driver
        self.worker = stack.worker(0)
        self.bridge = self.worker._flash_s5_bridge
        streams = tuple((r,) + self.driver._lean_streams[r].end() for r in self.driver.ranks)
        self.request = w.ReleaseRequest(
            self.driver.tx, self.driver.admission.state_epoch,
            tuple((r, self.driver.reports[r].completed_sequence) for r in self.driver.ranks),
            self.driver.admission.terminal, streams)

    def refused(self, request, text):
        with self.assertRaises(WorkerBridgeError) as caught:
            self.bridge.release(request)
        self.assertIn(text, str(caught.exception))
        self.assertIsNotNone(self.bridge.transaction)

    def forged(self, rank_entry):
        streams = tuple(rank_entry if e[0] == 0 else e for e in self.request.lean_streams)
        return replace(self.request, lean_streams=streams)

    def test_the_engine_stream_releases(self):
        self.assertEqual(self.request.lean_streams[0][1:], self.bridge._lean.end())
        ack = self.bridge.release(self.request)
        self.assertTrue(ack.released)

    def test_a_different_applied_stream_is_refused(self):
        c, p, h, d = self.request.lean_streams[0][1:]
        for entry in ((0, c, p, h, bytes(32)), (0, c - 1, p, h, d), (0, c, p, h - 1, d)):
            with self.subTest(entry=entry[1:4]):
                self.refused(self.forged(entry), 'differs from the sent report stream')
        self.refused(replace(self.request, lean_streams=()), 'differs from the sent report stream')

    def test_an_unsent_completion_is_refused(self):
        sent = self.bridge._lean
        op = max(sent.hosts)
        sent.hosts.discard(op)
        self.refused(self.forged((0,) + sent.end()), 'leaves completions unreported')

    def test_mutant_without_the_release_check_releases_a_diverged_stream(self):
        c, p, h, _d = self.request.lean_streams[0][1:]
        diverged = self.forged((0, c, p, h, bytes(32)))
        with mock.patch.object(type(self.bridge._lean), 'end', lambda s: diverged.lean_streams[0][1:]):
            ack = self.bridge.release(diverged)
        self.assertTrue(ack.released)

    def test_lean_streams_on_a_full_transaction(self):
        self.bridge._lean = None
        self.refused(self.request, 'full-reply transaction')


class LeanWire(unittest.TestCase):
    def setUp(self):
        run = Run('lean', seed=7).drive()
        self.report = next(x for xs in run.reports().values() for x in xs if x.proofs)

    def test_structure_is_validated(self):
        r = self.report
        bad = {
            'unordered': dict(completed_op_seqs=tuple(reversed(r.completed_op_seqs)) if len(r.completed_op_seqs) > 1 else (2, 1)),
            'overlap': dict(pending_op_seqs=r.completed_op_seqs[:1]),
            'digest-size': dict(digest=b'x'),
            'duplicate-proof': dict(proofs=r.proofs + r.proofs[:1]),
            'host-order': dict(host_submit_ns=((5, 1), (4, 1))),
            'base': dict(base_hosts=-1),
        }
        for name, change in bad.items():
            with self.subTest(name), self.assertRaises(E3ContractError):
                replace(r, **change)

    def test_open_and_release_fields(self):
        with self.assertRaises(E3ContractError):
            w.OpenRequest.__post_init__(replace_open(replies='brief'))
        tx = self.report.transaction
        good = w.ReleaseRequest(tx, 1, ((0, 1), (1, 1)), False,
                                ((0, 1, 1, 1, bytes(32)), (1, 1, 1, 1, bytes(32))))
        self.assertEqual(len(good.lean_streams), 2)
        for streams in (((0, 1, 1, 1, b'x'), (1, 1, 1, 1, bytes(32))),
                        ((1, 1, 1, 1, bytes(32)), (0, 1, 1, 1, bytes(32))),
                        ((0, 1, 1, 1, bytes(32)),),
                        ((0, 1, 1, bytes(32)), (1, 1, 1, bytes(32)))):
            with self.subTest(streams=len(streams)), self.assertRaises(E3ContractError):
                w.ReleaseRequest(tx, 1, ((0, 1), (1, 1)), False, streams)

    def test_acceptance_carries_a_lean_report(self):
        r = self.report
        acceptance = w.SegmentAcceptance(r.transaction, r.worker_rank, w.SegmentKind.LAYER, 0, 1,
                                         skipped=True, cursor=r.cursor, report=r)
        self.assertIs(acceptance.report, r)


def replace_open(**change):
    """An OpenRequest instance with one field changed and no validation run yet."""
    stack = r4.Stack()
    real = w.OpenRequest
    record = stack.record
    request = real(record, (0, 1, 2, 3), record.state_epoch, 1)
    forged = object.__new__(real)
    for name in real.__dataclass_fields__:
        object.__setattr__(forged, name, change.get(name, getattr(request, name)))
    return forged


class LeanCensus(unittest.TestCase):
    """Census refusals of a malformed lean stream; the connected cases prove the clean one."""

    def setUp(self):
        self.run = Run('lean', seed=7).drive()
        self.entries = [copy.deepcopy(e) for e in self.run.stack.driver._journal]

    def replay(self, entries):
        census = ed.NativeMeasurementCensus(engine_epoch=self.run.stack.driver.engine_epoch,
                                            ranks=self.run.stack.driver.ranks,
                                            stage_ranks=self.run.stack.driver.stage_ranks,
                                            pp_senders=self.run.stack.driver.stage_ranks[0],
                                            pool_chunk_capacity=True)
        for entry in entries:
            census.ingest(entry)
        return [c for chunk in census._chunks.values() for c in chunk['conflicts']]

    def test_clean_stream_has_no_conflict(self):
        self.assertEqual(self.replay(self.entries), [])

    def faulted(self, mutate):
        entries = copy.deepcopy(self.entries)
        index = next(i for i, e in enumerate(entries)
                     if e['kind'] == 'report' and e['completed_op_seqs'] and e['lean']['base_completed'])
        mutate(entries, index)
        return self.replay(entries)

    def test_gap_replay_and_header(self):
        def gap(entries, index):
            del entries[index]

        def replay(entries, index):
            entries.insert(index + 1, copy.deepcopy(entries[index]))

        def header(entries, index):
            entries[index]['lean'] = {'base_completed': 'x'}
        self.assertTrue(any(c.startswith('report-lean-gap') or c.startswith('report-lean-count')
                            for c in self.faulted(gap)))
        self.assertTrue(any(c.startswith('report-lean-replay') for c in self.faulted(replay)))
        self.assertTrue(any(c.startswith('report-lean-header-invalid') for c in self.faulted(header)))


if __name__ == '__main__':
    unittest.main()
