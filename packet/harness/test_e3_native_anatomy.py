"""Held-g anatomy receipts: R1 device spans per lane command, R5 engine step anatomy.

R1 runs on the real ``QsaLayerDriver``/``LaneForward``/``LaneDispatcher``/
``NativeWorkerBridge`` composition of ``test_e3_native_submission_credit``
(dense and sparse QSA routes, delayed attention consumers and selection
readbacks), with ``native_measurement.DeviceSpans`` installed on each
dispatcher. Fixture leaves (declared in native41_held_g_fixture_leaves.json):

* ``TimingEvent`` stands for ``torch.cuda.Event(enable_timing=True)``. Its
  ``record`` stamps a shared fake clock, so a device time is the fake clock at
  the moment the command's event was recorded; ``elapsed_time`` is the stamp
  difference in ms; ``synchronize`` returns at once and logs the lane state.
  The spans' host clock is the same fake clock (one CLOCK_MONOTONIC domain).
* The credit harness leaves (fake device, generated QSA arithmetic, fake model)
  are declared in native41_fixture_leaves.json and used unchanged.

Passing is a contract check of the receipts, never device timing evidence.
"""
import json
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))
sys.path.insert(0, str(ROOT / 'harness'))

import native_measurement as nm  # noqa: E402
import test_e3_native_submission_credit as credit  # noqa: E402


class FakeClock:
    """Shared host/device clock: every reading advances one microsecond."""

    def __init__(self, start=10 ** 12):
        self.t = start

    def __call__(self):
        self.t += 1000
        return self.t


class TimingEvent:
    """Leaf for torch.cuda.Event(enable_timing=True) (see module docstring)."""

    def __init__(self, clock, log):
        self.clock, self.log = clock, log
        self.t = None
        self.stream = None
        self.records = 0

    def record(self, stream=None):
        self.stream = stream
        self.t = self.clock()
        self.records += 1

    def query(self):
        return self.t is not None

    def elapsed_time(self, other):
        if self.t is None or other.t is None:
            raise RuntimeError('elapsed_time on an unrecorded event')
        return (other.t - self.t) / 1e6

    def synchronize(self):
        self.log.append(('synchronize', self.stream))


def parse(lines, prefix=nm.ANATOMY_LINE):
    """Whole anatomy flushes from emitted lines: {(side, rank, line): (identity, rows)}."""
    groups = {}
    for text in lines:
        assert text.startswith(prefix), text[:40]
        body = json.loads(text[len(prefix):])
        assert body['schema'] == nm.ANATOMY_SCHEMA
        groups.setdefault((body['side'], body['rank'], body['line'], body['base_ns']), []).append(body)
    out = {}
    for key, parts in groups.items():
        parts.sort(key=lambda b: b['part'])
        assert [b['part'] for b in parts] == list(range(parts[0]['parts']))
        head = parts[0]
        assert head['fields'] == {k: list(v) for k, v in nm.ANATOMY_ROWS.items()}
        rows = {}
        for b in parts:
            for tag, items in b['rows'].items():
                rows.setdefault(tag, []).extend(
                    {n: (head['base_ns'] + v * 1000 if n.startswith('t_') and type(v) is int else v)
                     for n, v in zip(nm.ANATOMY_ROWS[tag], item)} for item in items)
        identity = {k: v for k, v in head.items() if k not in (
            'rows', 'fields', 'part', 'parts', 'dropped', 'schema', 'side', 'rank', 'line', 'base_ns')}
        out[key[:3]] = (identity, rows, head['dropped'])
    return out


def command(kind, op_seq, layer=None):
    return types.SimpleNamespace(kind=kind, op_seq=op_seq, layer_start=layer)


TXN = credit.et.TransactionId(engine_epoch=1, request_uid='anatomy', request_generation=0,
                              chunk_start=0, chunk_end=8, txn_seq=3)


class Spans(unittest.TestCase):
    """DeviceSpans on scripted commands: exact sums, reasons, pool and errors."""

    def make(self, **kwargs):
        self.clock = FakeClock()
        self.log = []
        self.lines = []
        spans = nm.DeviceSpans(lambda: TimingEvent(self.clock, self.log), rank=1,
                               emit=self.lines.append, clock=self.clock, **kwargs)
        spans.begin(TXN, 0)
        return spans

    def run_command(self, spans, kind, op_seq, layer=None, driver=None, work=0, phase=None):
        mark = spans.start('prefill', driver)
        self.clock.t += work
        if phase is not None:
            driver.records.append((phase, layer))
        spans.end(mark, 'prefill', command(kind, op_seq, layer), driver)
        return mark

    def refuse(self, spans, kind, op_seq, layer, text, driver=None, hold=0):
        mark = spans.start('prefill', driver)
        spans.refused(mark, command(kind, op_seq, layer), credit.et.E3Pending(text))
        self.clock.t += hold

    def test_exact_sums_classes_and_reasons(self):
        spans = self.make()
        driver = types.SimpleNamespace(records=[])
        self.run_command(spans, 'prepare', 0, work=5000)
        self.clock.t += 7000                                           # idle 7 us + clock steps
        self.run_command(spans, 'layer', 1, layer=0, driver=driver, work=3000)   # GDN
        self.run_command(spans, 'layer', 2, layer=1, driver=driver, work=2000, phase='pre')
        self.refuse(spans, 'layer', 3, 1, 'lane submission credit pending: attention consumer not observed',
                    driver, hold=4000)
        self.refuse(spans, 'layer', 3, 1, 'lane submission credit pending: attention consumer not observed',
                    driver, hold=6000)
        self.run_command(spans, 'layer', 3, layer=1, driver=driver, work=1000, phase='bucket')
        self.refuse(spans, 'layer', 4, 1, 'QSA schedule awaits active-stage TP agreement', driver, hold=2000)
        self.run_command(spans, 'layer', 4, layer=1, driver=driver, work=1000, phase='tail')
        self.assertTrue(spans.release('prefill', None))
        self.assertEqual(self.log, [('synchronize', 'prefill')])
        (identity, rows, dropped), = parse(self.lines).values()
        self.assertEqual(identity, dict(txn=3, request_generation=0, chunk_start=0, chunk_end=8, stage=0))
        self.assertEqual(dropped, 0)
        x = rows['x']
        self.assertEqual([(r['op_seq'], r['kind'], r['refused']) for r in x],
                         [(0, 'prepare', 0), (1, 'layer', 0), (2, 'qsa-pre', 0), (3, 'qsa-bucket', 2),
                          (4, 'qsa-tail', 1)])
        # Device times are the event stamps shifted by the anchor's host offset.
        busy = [r['t_end'] - r['t_start'] for r in x]
        # work plus the one clock reading between the two event records
        self.assertEqual(busy, [6000, 4000, 3000, 2000, 2000])
        gaps = [b['t_start'] - a['t_end'] for a, b in zip(x, x[1:])]
        c, = rows['c']
        self.assertEqual((c['commands'], c['timed'], c['untimed'], c['refused'], c['cancelled'], c['detail']),
                         (5, 5, 0, 3, 0, 1))
        self.assertEqual(c['busy_us'], sum(busy) // 1000)
        self.assertEqual(c['idle_us'], sum(gaps) // 1000)
        self.assertEqual(c['idle_qsa_us'], (gaps[2] + gaps[3]) // 1000)
        self.assertEqual((c['t_first'], c['t_last']), (x[0]['t_start'], x[-1]['t_end']))
        k = {r['kind']: r for r in rows['k']}
        self.assertEqual(sorted(k), ['layer', 'prepare', 'qsa-bucket', 'qsa-pre', 'qsa-tail'])
        self.assertEqual((k['qsa-bucket']['refused'], k['qsa-tail']['refused']), (2, 1))
        q, = rows['q']
        self.assertEqual((q['layer'], q['commands'], q['credit'], q['grant']), (1, 3, 2, 1))
        f = {r['reason']: r for r in rows['f']}
        self.assertEqual({n: r['count'] for n, r in f.items()}, {'credit': 2, 'grant': 1})
        # The held time: first refusal to the accepted attempt, split by the
        # reason in force; the same totals in k, q and f.
        self.assertEqual(f['credit']['wait_us'], k['qsa-bucket']['wait_us'])
        self.assertEqual(f['credit']['wait_us'], q['credit_wait_us'])
        self.assertEqual(f['grant']['wait_us'], q['grant_wait_us'])
        self.assertGreaterEqual(f['credit']['wait_us'], 10)
        self.assertGreaterEqual(f['grant']['wait_us'], 2)
        self.assertEqual(q['busy_us'], (busy[2] + busy[3] + busy[4]) // 1000)
        self.assertEqual(q['idle_us'], (gaps[2] + gaps[3]) // 1000)
        self.assertEqual((q['t_start'], q['t_end']), (x[2]['t_start'], x[4]['t_end']))
        # Lag: host stamp of a continuation minus the previous command's device end.
        lags = [x[3]['t_host'] - x[2]['t_end'], x[4]['t_host'] - x[3]['t_end']]
        self.assertEqual(q['lag_us'], sum(max(0, v) for v in lags) // 1000)
        self.assertEqual(q['lag_max_us'], max(0, max(lags)) // 1000)
        # The next chunk starts clean.
        self.assertEqual((spans.commands, spans.reasons, spans.waiting, spans.used), ([], {}, {}, 0))

    def test_pool_spent_counts_untimed_and_refuses_nothing(self):
        spans = self.make(pool=2)
        for op_seq in range(5):
            self.run_command(spans, 'prefix' if op_seq else 'prepare', op_seq, work=1000)
        self.assertTrue(spans.release('prefill', None))
        (_, rows, _), = parse(self.lines).values()
        c, = rows['c']
        self.assertEqual((c['commands'], c['timed'], c['untimed']), (5, 2, 3))
        self.assertEqual([r['t_start'] is None for r in rows['x']], [False, False, True, True, True])

    def test_refused_and_abandoned_slots_are_reused(self):
        spans = self.make(pool=1)
        self.refuse(spans, 'prefix', 0, None, 'PP receive has not completed')
        self.assertEqual(spans.used, 0)
        mark = spans.start('prefill', None)
        spans.abandon(mark)
        self.assertEqual(spans.used, 0)
        self.run_command(spans, 'prefix', 0, work=1000)
        self.assertEqual(spans.used, 1)
        self.assertTrue(spans.release('prefill', 'cancelled'))
        (_, rows, _), = parse(self.lines).values()
        c, = rows['c']
        self.assertEqual((c['timed'], c['refused'], c['cancelled']), (1, 1, 1))
        self.assertEqual([(r['reason'], r['count']) for r in rows['f']], [('pp', 1)])

    def test_a_failed_read_is_counted_never_raised(self):
        spans = self.make()
        self.run_command(spans, 'prepare', 0, work=1000)

        def boom():
            raise RuntimeError('synthetic CUDA error')
        spans.anchor.synchronize = boom
        self.assertFalse(spans.release('prefill', None))
        self.assertEqual((spans.errors, self.lines, spans.trace.rows), (1, [], []))
        spans.anchor.synchronize = lambda: None
        spans.begin(TXN, 0)
        self.run_command(spans, 'prepare', 0, work=1000)
        self.assertTrue(spans.release('prefill', None))
        (_, rows, _), = parse(self.lines).values()
        self.assertEqual(rows['c'][0]['errors'], 1)

    def test_detail_rows_ride_one_chunk_in_the_cadence(self):
        spans = self.make(detail_every=2)
        details = []
        for chunk in range(4):
            if chunk:
                spans.begin(TXN, 0)
            self.run_command(spans, 'prepare', 0, work=1000)
            spans.release('prefill', None)
        for _, rows, _ in parse(self.lines).values():
            details.append((rows['c'][0]['detail'], 'x' in rows))
        self.assertEqual(details, [(1, True), (0, False), (1, True), (0, False)])

    def test_no_transaction_records_nothing(self):
        spans = nm.DeviceSpans(lambda: TimingEvent(FakeClock(), []), rank=0, emit=list().append)
        self.assertIsNone(spans.pool)
        self.assertFalse(spans.release('prefill', None))


class Steps(unittest.TestCase):
    """StepAnatomy on a scripted step loop: classes, exact sums, contiguous detail."""

    def test_classes_sums_and_detail(self):
        clock = FakeClock()
        counters = dict(commands=0, stock_rpcs=0)
        steps = nm.StepAnatomy(counters, clock_ns=clock, detail_every=2)
        steps.reset()                                   # the chunk begins inside an advance
        clock.t += 5000
        steps.advanced(clock.t - 3000, False)           # its advance began before the chunk
        counters['commands'] += 1
        clock.t += 2000
        steps.scheduled(types.SimpleNamespace(total_num_scheduled_tokens=0))
        counters['stock_rpcs'] += 1
        steps.begin()                                   # step 2: empty, with a wait
        started = clock()
        steps.advanced(started, False)
        steps.scheduled(types.SimpleNamespace(total_num_scheduled_tokens=0))
        counters['stock_rpcs'] += 1
        wait = clock()
        clock.t += 9000
        steps.waited(wait, clock())
        steps.begin()                                   # step 3: the idle tick bypassed it
        steps.bypass('tick')
        clock.t += 1000
        steps.begin()                                   # step 4: the driver bypassed it
        started = clock()
        counters['commands'] += 1
        steps.advanced(started, True)
        steps.begin()                                   # step 5: stock tokens
        steps.advanced(clock(), False)
        steps.scheduled(types.SimpleNamespace(total_num_scheduled_tokens=7))
        steps.begin()                                   # step 6: the chunk ends inside it
        lines = steps.lines(txn=4, mode='current')
        (identity, rows, dropped), = parse(lines).values()
        self.assertEqual((identity, dropped), (dict(txn=4, mode='current', detail=1), 0))
        e = {r['cls']: r for r in rows['e']}
        self.assertEqual({c: r['steps'] for c, r in e.items()},
                         dict(empty=2, tick=1, advance=1, stock=1, partial=1))
        self.assertEqual((e['empty']['sends'], e['empty']['rpcs'], e['empty']['waits']), (1, 2, 1))
        self.assertEqual(e['advance']['sends'], 1)
        p = rows['p']
        self.assertEqual([r['cls'] for r in p], ['empty', 'empty', 'tick', 'advance', 'stock', 'partial'])
        # The steps tile the chunk: each begins where the previous ended.
        self.assertTrue(all(abs(a['t_begin'] + a['total_us'] * 1000 - b['t_begin']) < 1000
                            for a, b in zip(p, p[1:])))
        for cls, r in e.items():
            mine = [s for s in p if s['cls'] == cls]
            self.assertLessEqual(abs(r['total_us'] - sum(s['total_us'] for s in mine)), len(mine))
            self.assertLessEqual(r['advance_us'] + r['schedule_us'] + r['wait_us'], r['total_us'] + 1)
        self.assertGreaterEqual(e['empty']['wait_us'], 9)
        # The advance of step 1 counts only from the chunk begin, not before it.
        self.assertLess(p[0]['advance_us'], 5 + 1)
        # Chunk 2 (detail_every 2) carries no per-step rows.
        steps.reset()
        steps.begin()
        (identity, rows, _), = parse(steps.lines(txn=5)).values()
        self.assertEqual((identity['detail'], 'p' in rows), (0, False))

    def test_stamps_outside_a_step_are_ignored(self):
        steps = nm.StepAnatomy(dict(commands=0, stock_rpcs=0), clock_ns=FakeClock())
        steps.advanced(0, True)
        steps.scheduled(None)
        steps.waited(0, 1)
        steps.bypass('tick')
        self.assertEqual(steps.trace.lines(), [])


class IdleWaitTransport(unittest.TestCase):
    """ReplyTransport.wait_landed over real receiver threads (CPU reader doubles)."""

    def setUp(self):
        import threading
        from enum import Enum
        from reply_transport import ReplyTransport
        import test_s5_reply_transport as rt
        self.rt, self.threading = rt, threading
        self.Status = rt.Status
        self.readers = {r: rt.Reader() for r in range(2)}
        self.t = ReplyTransport(self.readers, self.Status.SUCCESS, handoff_idle=True, max_calls=4)

    def tearDown(self):
        for reader in self.readers.values():
            reader.q.put((self.Status.SUCCESS, None))
        self.t.close(3)

    def test_returns_on_landing_and_never_reads_or_polls(self):
        import time
        sequence = self.t.submit('native', lambda _: None, ranks=(0, 1), timeout_s=5)
        self.assertTrue(self.readers[0].entered.wait(1) and self.readers[1].entered.wait(1))
        started = time.monotonic()
        self.assertFalse(self.t.wait_landed(0.02))        # nothing landed: the bound
        self.assertGreaterEqual(time.monotonic() - started, 0.015)
        landed = []
        waiter = self.threading.Thread(target=lambda: landed.append(self.t.wait_landed(5)))
        waiter.start()
        time.sleep(0.05)
        self.readers[1].q.put((self.Status.SUCCESS, 'one'))
        waiter.join(2)
        self.assertEqual(landed, [True])
        # The waiter consumed nothing: the reply is still for poll, the call not ready.
        self.assertEqual(self.t._done.qsize(), 1)
        self.assertFalse(self.t._calls[sequence].ready)
        self.assertTrue(self.t.wait_landed(0))            # still landed, no wait
        self.readers[0].q.put((self.Status.SUCCESS, 'zero'))
        self.rt.await_condition(self, lambda: self.t.poll(8) >= 0 and self.t._calls[sequence].ready)
        self.assertEqual(self.t.take(sequence)[0], ['zero', 'one'])
        self.assertFalse(self.t.wait_landed(0))

    def test_a_failed_transport_returns_at_once(self):
        import time
        self.t.submit('native', lambda _: None, ranks=(0,), timeout_s=5)
        self.assertTrue(self.readers[0].entered.wait(1))
        self.readers[0].q.put((self.Status.FAILURE, None))
        self.rt.await_condition(self, lambda: self.t._failure is not None)
        started = time.monotonic()
        self.assertFalse(self.t.wait_landed(5))
        self.assertLess(time.monotonic() - started, 1)

    def test_refuses_an_unbounded_wait(self):
        for value in (None, -1, float('inf'), '1'):
            with self.assertRaises(ValueError):
                self.t.wait_landed(value)


class IdleSourceNames(unittest.TestCase):
    """Every scheduler/engine name the L3 predicate reads exists in the pinned vLLM source."""

    def test_names_exist_in_the_pinned_source(self):
        import ast
        import executor_hooks as eh
        source = ast.unparse(ast.parse(Path(eh.__file__).read_text()).body[[
            i for i, n in enumerate(ast.parse(Path(eh.__file__).read_text()).body)
            if getattr(n, 'name', None) == '_stock_step_idle'][0]])
        ref = ROOT / 'reference/vllm-src/v1'
        if not ref.is_dir():
            self.skipTest('reference/vllm-src is not available on this host')
        scheduler = (ref / 'core/sched/scheduler.py').read_text()
        core = (ref / 'engine/core.py').read_text()
        managers = (ref / 'core/single_type_kv_cache_manager.py').read_text()
        encoder = (ref / 'core/encoder_cache_manager.py').read_text()
        coordinator = (ref / 'core/kv_cache_coordinator.py').read_text()
        expect = {
            'scheduler.running': ('self.running', scheduler), 'scheduler.waiting': ('self.waiting =', scheduler),
            'scheduler.skipped_waiting': ('self.skipped_waiting =', scheduler),
            'scheduler.finished_req_ids': ('self.finished_req_ids: set', scheduler),
            'scheduler.reset_preempted_req_ids': ('self.reset_preempted_req_ids', scheduler),
            'scheduler.connector': ('self.connector = None', scheduler),
            "'ec_connector'": ('self.ec_connector = None', scheduler),
            'scheduler.kv_cache_manager.coordinator.single_type_managers': ('self.single_type_managers', coordinator),
            "'new_block_ids'": ('self.new_block_ids', managers),
            "'_pending_cow_copies'": ('self._pending_cow_copies', managers),
            'scheduler.encoder_cache_manager': ('self.encoder_cache_manager', scheduler),
            "'freed'": ('self.freed', encoder), "'to_free'": ('self.to_free', encoder),
            "'allocated'": ('self.allocated', encoder),
            "'aborts_queue'": ('self.aborts_queue', core), 'engine.batch_queue': ('self.batch_queue', core),
            'total_num_scheduled_tokens': ('total_num_scheduled_tokens', scheduler),
        }
        for name, (needle, text) in expect.items():
            self.assertIn(name, source, name)
            self.assertIn(needle, text, needle)


class ConnectedSpans(credit.CreditHarness, unittest.TestCase):
    """R1 on the real driver/lane/dispatcher/bridge: classes, refusals and the only sync."""

    def install(self, world):
        self.clock = FakeClock()
        self.sync_log = []
        self.lines = []
        for stack in world.stacks:
            log = []
            spans = nm.DeviceSpans(lambda log=log: TimingEvent(self.clock, log), rank=stack.rank,
                                   emit=self.lines.append, clock=self.clock)
            spans.log = log
            stack.dispatcher.spans = spans
            # The bridge opened the lane before the spans were attached here.
            spans.begin(world.txn, stack.stage)

            def checked(stack=stack, spans=spans):
                # The anchor is synchronized only after every lane event completed.
                self.sync_log.append(all(record.event.query() for record in stack.dispatcher.events))
            spans.anchor_synchronize = checked
        return world

    def run_dense(self, world, polls):
        stack = self.stack(world)
        self.begin_layer0(world)
        command, acks = world.send('layer', layer=0)
        self.expect_pending(world, command, acks)
        self.poll(world, command, polls)
        stack.ready_attention()
        self.complete_tail_after_boundary(world, command, world.deliver(command))
        self.finish(world)
        return self.complete_and_release(world)[0]

    def test_dense_classes_refusals_and_one_anchor_sync(self):
        world = self.install(self.world(route='dense'))
        stack = self.stack(world)
        anchor = stack.dispatcher.spans.anchor
        real = anchor.synchronize

        def sync():
            stack.dispatcher.spans.anchor_synchronize()
            real()
        anchor.synchronize = sync
        report = self.run_dense(world, 40)
        self.assertEqual(self.sync_log, [True])
        self.assertEqual(stack.device.synchronized, 0)
        (identity, rows, dropped), = parse(self.lines).values()
        self.assertEqual((identity['txn'], identity['stage'], dropped), (1, 0, 0))
        x = rows['x']
        self.assertEqual([r['op_seq'] for r in x], list(report.completed_op_seqs))
        self.assertEqual([r['kind'] for r in x],
                         ['prepare', 'prefix', 'qsa-pre', 'qsa-bucket', 'qsa-attended', 'qsa-tail',
                          'layer', 'suffix', 'state_finalize'])
        # The first pending send and the 40 polls were credit refusals of op 4.
        self.assertEqual([r['refused'] for r in x], [0, 0, 0, 0, 41, 0, 0, 0, 0])
        self.assertEqual(rows['f'][0]['reason'], 'credit')
        self.assertEqual(rows['f'][0]['count'], 41)
        c, = rows['c']
        self.assertEqual((c['commands'], c['timed'], c['untimed'], c['refused']), (9, 9, 0, 41))
        q, = rows['q']
        self.assertEqual((q['layer'], q['commands'], q['credit'], q['grant']), (0, 4, 41, 0))
        self.assertEqual(q['credit_wait_us'], rows['f'][0]['wait_us'])
        self.assertTrue(all(r['t_start'] <= r['t_end'] for r in x))
        self.assertTrue(all(a['t_end'] <= b['t_start'] for a, b in zip(x, x[1:])))

    def test_sparse_selection_and_bucket_classes(self):
        world = self.install(self.world(route='sparse'))
        stack = self.stack(world)
        self.begin_layer0(world)
        command, acks = world.send('layer', layer=0)
        self.expect_pending(world, command, acks)
        self.poll(world, command, 5)
        stack.ready_selection()
        self.expect_accepted(world, command, world.deliver(command), paused=True)
        command, acks = world.send('layer', layer=0)
        self.expect_pending(world, command, acks)
        self.poll(world, command, 7)
        stack.ready_attention()
        self.complete_tail_after_boundary(world, command, world.deliver(command))
        self.finish(world)
        report = self.complete_and_release(world)[0]
        (_, rows, _), = parse(self.lines).values()
        x = rows['x']
        self.assertEqual([r['op_seq'] for r in x], list(report.completed_op_seqs))
        kinds = [r['kind'] for r in x]
        self.assertEqual(kinds[:3] + kinds[-3:], ['prepare', 'prefix', 'qsa-pre', 'layer', 'suffix',
                                                  'state_finalize'])
        self.assertEqual(kinds[3:-3], [nm.ANATOMY_QSA[r[0]] for r in stack.driver.records[1:]])
        self.assertIn('qsa-selection', kinds)
        self.assertIn('qsa-bucket', kinds)
        self.assertEqual(sum(r['refused'] for r in x), 6 + 8)
        q, = rows['q']
        self.assertEqual(q['credit'], 14)

    def test_spans_leave_the_lane_unchanged(self):
        def state(spans):
            world = self.world(route='dense')
            if spans:
                self.install(world)
            report = self.run_dense(world, 12)
            stack = self.stack(world)
            return (report.completed_op_seqs, tuple(p.kind for p in report.proofs),
                    len(stack.bus.events), len(stack.device.events), dict(stack.operation.stats),
                    len(stack.operation._records), len(self.kernel.calls))
        plain = state(False)
        self.kernel.calls.clear()
        timed = state(True)
        self.assertEqual(plain, timed)


if __name__ == '__main__':
    unittest.main()
