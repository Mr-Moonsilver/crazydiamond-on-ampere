"""Quiet-SPLIT control v2 (e3_quiet_split_box, brief 13 item 2): a fake engine and fake streams, CPU only.

The fake engine keeps the state the tool reads (route, generation, pause, the stage-local selection, merge_steps,
native admitted, the auto-arm row, the control route, scheduler requests = live fake streams, the attn-meta selection
and its batched counter behind a fake worker adapter) and a fake clock advanced by sleep(); each status poll advances
every live stream by one chunk on the fake clock. Every guard has a negative (one fault injected, the verdict
INCONCLUSIVE with that guard named, FULL restored, the split lease released), the declaration guard sends no control,
the disconnect check and its max_tokens fallback are both exercised, a real SIGTERM delivered mid-run still runs the
cleanup and writes the record, and mutants of the tool (a guard removed) turn a negative green, so the negatives are
live.
"""
import importlib.util
import json
import os
from pathlib import Path
import signal
import tempfile
import types
import unittest

import e3_quiet_split_box as q

HERE = Path(__file__).resolve().parent


def scratch():
    base = Path(os.environ.get('TMPDIR') or tempfile.gettempdir())
    return tempfile.mkdtemp(prefix='qsbox-', dir=str(base))


class Clock:
    def __init__(self):
        self.t = 1000.0
        self.hooks = []

    def __call__(self):
        return self.t

    def ns(self):
        return int(self.t * 1e9)

    def sleep(self, s):
        self.t += s
        for hook in list(self.hooks):
            hook(self.t)


class FakeStream(q.Stream):
    pass


class Control:
    """The worker adapter: attn_meta_select / attn_meta_read, qualification rows for ranks 0-3."""

    def __init__(self, engine):
        self.engine = engine

    def worker(self, command):
        e = self.engine
        action = command['worker_action']
        e.calls.append(action)
        assert e.paused and e.route == 'full', 'worker actions need paused idle FULL'
        before = dict(selected=e.attn, batched=e.batched)
        if action == q.SELECT:
            e.attn = command['payload']['on'] if e.fault != 'attn-not-selected' else False
        after = dict(selected=e.attn, batched=e.batched)
        rows = [dict(worker_rank=r, qualification=dict(worker_action=action, engine_supplied_verdict=False, before=before,
                                                       after=after, counters=dict(selected=e.attn, batched=e.batched),
                                                       order=dict(ok=True))) for r in q.RANKS]
        return dict(action=q.WORKER_ACTION, worker_action=action, engine_epoch=command['engine_epoch'],
                    worker_reply=dict(qualification=rows))


class Engine:
    """The control surface e3_quiet_split_box calls; ``fault`` names one misbehaviour."""

    def __init__(self, clock, fault=None):
        self.clock, self.fault = clock, fault
        self.route, self.generation, self.paused = 'full', 7, False
        self.selection = dict(local=True, overlap=True, defer_bind=True, lazy_recv=False, decode_merge='off')
        self.merge_steps, self.admitted = 100, 42
        self.streams, self.calls, self.polls_on_split = [], [], 0
        self.manual_owned, self.history = True, []
        self.epoch = 3
        self.attn, self.batched = False, 500
        self.control_route = None
        self.closed_at = {}             # id(stream) -> the first status after its close
        self.control = Control(self)
        self.timeout_s = 60

    # NativeRuntimeClient surface
    def call(self, action, **fields):
        self.calls.append(action)
        if action == 'status':
            return self.status()
        if action == q.STAGE_CONTROL:
            assert self.paused and self.route == 'full', 'stage-local needs paused idle FULL'
            previous = dict(self.selection)
            new = {k: fields[k] for k in q.SELECTION_FIELDS}
            if self.fault == 'merge-not-acked':
                new['decode_merge'] = 'chunk'
            self.selection = new
            reply = dict(new, previous=previous['local'])
            reply.update({'previous_' + k: previous[k] for k in q.SELECTION_FIELDS if k != 'local'})
            return reply
        raise AssertionError(action)

    def pause(self):
        self.calls.append('pause')
        self.paused = True
        return {}

    def resume(self):
        self.calls.append('resume')
        self.paused = False
        return {'resumed': True}

    def enter(self, phase):
        self.calls.append('enter')
        assert phase == q.SPLIT_PHASE
        if self.fault == 'enter-refused':
            raise ValueError('native-split-not-applied')
        self.route, self.generation, self.paused = 'split', self.generation + 1, True
        self.control_route = dict(owner='native-control', intent='enter_split', route_generation=self.generation)
        self.polls_on_split = 0
        return {}

    def leave(self):
        self.calls.append('leave')
        self.route, self.generation = 'full', self.generation + 1
        self.control_route = None
        return {}

    def live(self):
        return [s for s in self.streams if not s.stop.is_set() and s.finished_ns is None]

    def lingering(self):
        """Closed streams the engine still runs: none (a disconnect aborts), 30 s each (it does not: the max_tokens
        fallback ends them), or for ever."""
        now, out = self.clock(), 0
        for s in self.streams:
            if s.stop.is_set() and s.finished_ns is None:
                closed = self.closed_at.setdefault(id(s), now)
                if self.fault == 'requests-never-drain' or (self.fault == 'disconnect-not-aborting' and now - closed < 30):
                    out += 1
        return out

    def status(self):
        live = self.live()
        lingering = self.lingering()
        if not self.paused:
            for s in live:
                s.chunks.append(self.clock.ns())
            if self.attn and self.fault != 'attn-not-engaged':
                self.batched += 1
            if (self.route == 'split' and self.selection['decode_merge'] == 'always'
                    and self.fault != 'merge-not-active-in-hold'):
                self.merge_steps += 1
        out = dict(route=self.route, route_generation=self.generation, paused=self.paused,
                   transition_pending=False, return_pending=None, admission_intent=None,
                   scheduler_requests=len(live) + lingering, chunk=None, owned_work=dict(chunk=False),
                   native_admission=dict(admitted=self.admitted), control_route=self.control_route,
                   auto_arm=dict(active=False, manual_owned=self.manual_owned),
                   native_stage_local=dict(overlap=dict(decode_step=dict(
                       decode_merge=self.selection['decode_merge'], counters=dict(merge_steps=self.merge_steps)))))
        if self.route == 'split':
            self.polls_on_split += 1
            if self.polls_on_split == 5:
                self.inject(out, live)
        return out

    def inject(self, out, live):
        """Faults seen on the fifth split poll of each stage."""
        f = self.fault
        if f == 'route-left-split':
            self.route = out['route'] = 'full'
            self.control_route = out['control_route'] = None      # any FULL transition clears the split lease
        elif f == 'route-generation-changed':
            out['route_generation'] += 1
        elif f == 'transition-or-return-pending':
            out['transition_pending'] = True
        elif f == 'admission-armed':
            out['admission_intent'] = dict(kind='fresh')
        elif f == 'scheduler-requests-not-live':
            out['scheduler_requests'] += 1
        elif f == 'native-admitted':
            self.admitted += 1
            out['native_admission']['admitted'] = self.admitted
        elif f == 'native-chunk-open':
            out['chunk'] = 9
        elif f == 'auto-arm-active':
            out['auto_arm']['active'] = True
        elif f == 'manual-lease-lost':
            out['auto_arm']['manual_owned'] = False
        elif f == 'decode-merge-not-always':
            out['native_stage_local']['overlap']['decode_step']['decode_merge'] = 'chunk'
        elif f == 'decoder-ended-in-hold':
            live[0].finished_ns = self.clock.ns()
        elif f == 'stage-live-mismatch' and len(live) == 6:
            out['scheduler_requests'] = 8

    def starter(self, url, model, prompt_tokens, index, now_ns, timeout_s):
        s = FakeStream(index)
        s.prompt_tokens = prompt_tokens
        s.stall = self.fault == 'decoder-stalled-in-hold' and index == 0   # no chunk after its first token
        if self.fault != 'decoders-not-live-on-full' or index:
            s.opened_ns = now_ns() - 2_000_000
            s.first_ns = now_ns() - 1_000_000      # the first token came before the split window
            s.chunks.append(s.first_ns)
            if self.fault == 'history-off-band' and index == 0:
                s.chunks.extend([s.first_ns] * q.PRE_HOLD_TOKENS)
        s.opened.set()
        self.streams.append(s)
        return s

    def warmer(self, url, model, index, timeout_s):
        self.calls.append('warm')
        if self.fault == 'warm-failed' and index == 3:
            return dict(index=index, status=500)
        return dict(index=index, status=200, prompt_tokens=q.HIGH, cached_tokens=0, seconds=1.0)


def args(**over):
    a = types.SimpleNamespace(url='http://127.0.0.1:8000', model='m', run_id='r', epoch=3, plan=q.PLAN,
                              hold_s=25.0, decode_merge='always', max_tokens=q.MAX_TOKENS, control_timeout_s=60,
                              open_timeout_s=1, warm_timeout_s=5, drain_timeout_s=60, stream_timeout_s=5)
    for k, v in over.items():
        setattr(a, k, v)
    return a


def run(fault=None, module=q, clock=None, **over):
    clock = clock or Clock()
    engine = Engine(clock, fault)
    if fault == 'decoder-stalled-in-hold':
        status = engine.status

        def stalled():
            out = status()
            for s in engine.streams:
                if getattr(s, 'stall', False) and s.chunks:
                    s.chunks.pop()
            return out
        engine.status = stalled
    record = module.run(args(**over), client=engine, starter=engine.starter, warmer=engine.warmer, clock=clock,
                        sleep=clock.sleep, now_ns=clock.ns)
    return record, engine


def restore_stub(client):
    out = dict(attempted=True)
    try:
        client.pause()
        if client.route == 'split':
            client.leave()
        client.resume()
        out['full_restored'] = client.route == 'full'
    except Exception:  # noqa: BLE001
        out['full_restored'] = False
    return out


class Base(unittest.TestCase):
    def setUp(self):
        self._restore = q.restore_full
        q.restore_full = restore_stub

    def tearDown(self):
        q.restore_full = self._restore


class Clean(Base):
    def test_clean_run_passes_with_the_declared_plan(self):
        record, engine = run()
        self.assertEqual(record['verdict'], 'PASS', record.get('cause'))
        self.assertEqual(record['schema'], 'quiet-split-box/2')
        self.assertEqual([(s['prompt_tokens'], s['live']) for s in record['stages']], list(q.PLAN))
        self.assertEqual(q.PLAN, ((120000, 8), (120000, 6), (120000, 4), (120000, 2), (120000, 1), (72000, 4), (72000, 1)))
        for row in record['stages']:
            self.assertGreaterEqual(row['window']['wall_s'], 25.0)
            self.assertLessEqual(row['window']['wall_s'], 26.0)
            self.assertGreater(row['merge_steps'], 0)
            self.assertEqual(len(row['streams']), row['live'])
            self.assertTrue(all(v['window_chunks'] >= 1 for v in row['streams']))
            self.assertIsNotNone(row['aggregate_tokens_per_s'])
            self.assertEqual(row['abort'], 'disconnect')
            self.assertEqual(row['history_tokens'][0], row['prompt_tokens'] + 1)
        self.assertEqual(record['selection']['decode_merge'], 'always')
        self.assertIs(record['selection']['lazy_recv'], True)
        self.assertEqual(record['attn_meta']['selected'], True)
        self.assertTrue(all(v > 0 for v in record['attn_meta']['batched_delta'].values()))
        self.assertEqual(engine.calls.count('warm'), 8)
        self.assertEqual(engine.route, 'full')
        self.assertFalse(engine.paused)
        self.assertNotIn('cleanup', record)

    def test_previous_selection_and_attn_restored(self):
        record, engine = run()
        self.assertTrue(record['selection_restore']['restored'])
        self.assertEqual(engine.selection['decode_merge'], 'off')
        self.assertIs(engine.selection['lazy_recv'], False)
        self.assertIs(engine.attn, False)
        self.assertEqual(record['selection_restore']['attn_previous'], False)

    def test_no_native_work_and_counts(self):
        record, engine = run()
        self.assertEqual(record['native_admitted_before'], record['native_admitted_after'])
        self.assertEqual(engine.calls.count('enter'), 7)
        self.assertEqual(engine.calls.count('leave'), 7)
        self.assertEqual(engine.calls.count(q.STAGE_CONTROL), 2)
        self.assertEqual(engine.calls.count(q.SELECT), 2)
        self.assertEqual(engine.calls.count(q.READ), 1)
        self.assertLess(engine.calls.index(q.SELECT), engine.calls.index('warm'))

    def test_snapshots_bounded(self):
        record, _ = run(hold_s=30.0)
        self.assertTrue(all(len(r['snapshots']) <= q.SNAPSHOT_LIMIT for r in record['stages']))

    def test_prompts_seeded_distinct_and_low_is_the_high_prefix(self):
        a, b = q.prompt_ids(0, q.HIGH), q.prompt_ids(1, q.HIGH)
        self.assertEqual(len(a), q.HIGH)
        self.assertNotEqual(a[:64], b[:64])
        self.assertEqual(q.prompt_ids(0, q.LOW), a[:q.LOW])
        self.assertEqual(q.prompt_ids(0, q.HIGH), a)
        self.assertTrue(all(q.ID_LO <= x < q.ID_HI for x in a))


class Disconnect(Base):
    def test_fallback_when_a_disconnect_does_not_abort(self):
        record, _ = run('disconnect-not-aborting')
        self.assertEqual(record['verdict'], 'PASS', record.get('cause'))
        self.assertTrue(all(s['abort'] == 'max_tokens' for s in record['stages']))
        self.assertTrue(all(s['drain_s'] > q.ABORT_BOUND_S for s in record['stages']))
        self.assertEqual(len(record['stages']), len(q.PLAN))

    def test_never_drained_is_a_cause(self):
        record, engine = run('requests-never-drain')
        self.assertEqual(record['cause'], 'requests-not-drained')
        self.assertEqual(len(record['stages']), 1)
        self.assertEqual(engine.route, 'full')


class Declaration(Base):
    def test_other_declarations_send_no_control(self):
        for over, reason in ((dict(plan=q.PLAN[:5]), 'plan-not-declared'), (dict(plan=((8, 120000),)), 'plan-not-declared'),
                             (dict(plan=tuple(reversed(q.PLAN))), 'plan-not-declared'),
                             (dict(hold_s=19.5), 'hold-not-20-30-s'), (dict(hold_s=31.0), 'hold-not-20-30-s'),
                             (dict(decode_merge='chunk'), 'decode-merge-not-always'),
                             (dict(max_tokens=8192), 'max-tokens-not-4096')):
            record, engine = run(**over)
            self.assertEqual(record['verdict'], 'INCONCLUSIVE')
            self.assertIn(reason, record['cause'])
            self.assertFalse(record['control_sent'])
            self.assertEqual(engine.calls, [], over)

    def test_bounds_accepted_and_plan_parsed(self):
        for hold in (20.0, 30.0):
            self.assertEqual(q.declaration_reasons(q.PLAN, hold, 'always', q.MAX_TOKENS), [])
        self.assertEqual(q.parse_plan('120000x8,120000x6,120000x4,120000x2,120000x1,72000x4,72000x1'), q.PLAN)
        self.assertEqual(q.parse_plan('8,4,1'), ())

    def test_loopback_required(self):
        record, engine = run(url='http://<internal-host>:8000')
        self.assertEqual(record['cause'], 'loopback-required')
        self.assertEqual(engine.calls, [])

    def test_cli_refuses_existing_out(self):
        d = scratch()
        out = Path(d) / 'x.json'
        out.write_text('{}')
        self.assertEqual(q.main(['--url', 'http://127.0.0.1:1', '--model', 'm', '--run-id', 'r', '--epoch', '1',
                                 '--out', str(out)]), 2)
        self.assertEqual(out.read_text(), '{}')


NEGATIVES = q.POLL_GUARDS + ('decoders-not-live-on-full', 'decoder-ended-in-hold', 'decoder-stalled-in-hold',
                             'merge-not-active-in-hold', 'enter-refused', 'merge-not-acked', 'history-off-band',
                             'warm-failed', 'attn-not-selected', 'attn-not-engaged')
CAUSE = {'enter-refused': 'native-split-not-applied', 'merge-not-acked': 'selection-not-acknowledged',
         'attn-not-selected': 'attn-meta-selection-unproved', 'attn-not-engaged': 'attn-meta-not-engaged'}


class Negatives(Base):
    def test_each_guard_refuses_restores_full_and_releases_the_split_lease(self):
        for fault in NEGATIVES:
            with self.subTest(fault=fault):
                record, engine = run(fault)
                self.assertEqual(record['verdict'], 'INCONCLUSIVE')
                self.assertEqual(record['failure_class'], 'HARNESS')
                self.assertIn(CAUSE.get(fault, fault), record['cause'])
                self.assertEqual(engine.route, 'full')
                self.assertFalse(engine.paused)
                self.assertTrue(record['cleanup']['full_restored'])
                self.assertTrue(record['cleanup']['split_lease_released'])
                self.assertTrue(all(s.stop.is_set() for s in engine.streams))

    def test_fault_on_the_second_stage(self):
        record, _ = run('stage-live-mismatch')
        self.assertEqual(record['verdict'], 'INCONCLUSIVE')
        self.assertIn('scheduler-requests-not-live', record['cause'])
        self.assertEqual([s['live'] for s in record['stages']], [8, 6])

    def test_split_lease_left_reads_not_released(self):
        """A FULL route that kept the native-control split lease: the cleanup records it as not released."""
        clock = Clock()
        engine = Engine(clock, 'enter-refused')
        engine.control_route = dict(owner='native-control', intent='enter_split', route_generation=7)
        record = q.run(args(), client=engine, starter=engine.starter, warmer=engine.warmer, clock=clock,
                       sleep=clock.sleep, now_ns=clock.ns)
        self.assertTrue(record['cleanup']['full_restored'])
        self.assertIs(record['cleanup']['split_lease_released'], False)
        self.assertEqual(record['cleanup']['control_route']['owner'], 'native-control')

    def test_entry_must_be_idle_full(self):
        clock = Clock()
        engine = Engine(clock)
        engine.route = 'split'
        record = q.run(args(), client=engine, starter=engine.starter, warmer=engine.warmer, clock=clock,
                       sleep=clock.sleep, now_ns=clock.ns)
        self.assertEqual(record['cause'], 'entry-not-idle-full')
        self.assertNotIn(q.STAGE_CONTROL, engine.calls)

    def test_no_selection_restore_when_full_not_restored(self):
        clock = Clock()
        engine = Engine(clock, 'route-left-split')
        q.restore_full = lambda c: dict(full_restored=False)
        record = q.run(args(), client=engine, starter=engine.starter, warmer=engine.warmer, clock=clock,
                       sleep=clock.sleep, now_ns=clock.ns)
        self.assertFalse(record['selection_restore']['attempted'])
        self.assertEqual(engine.calls.count(q.STAGE_CONTROL), 1)


def term_run(case, at_stage=2, second=False, module=q):
    """A real SIGTERM to this process during stage ``at_stage``'s hold; the previous handler restored after the case."""
    saved = signal.getsignal(signal.SIGTERM)
    case.addCleanup(signal.signal, signal.SIGTERM, saved)
    state = module.install_term_handler()
    clock = Clock()
    engine = Engine(clock)
    sent = []

    def hook(t):
        if not sent and engine.route == 'split' and engine.calls.count('enter') == at_stage:
            sent.append(t)
            os.kill(os.getpid(), signal.SIGTERM)
    clock.hooks.append(hook)
    if second:
        restore = module.restore_full

        def restore_and_term(client):
            os.kill(os.getpid(), signal.SIGTERM)
            return restore(client)
        module.restore_full = restore_and_term
    record = module.run(args(), client=engine, starter=engine.starter, warmer=engine.warmer, clock=clock,
                        sleep=clock.sleep, now_ns=clock.ns)
    return record, engine, state


class Sigterm(Base):
    """A real SIGTERM to this process mid-hold (as `timeout` sends it): Terminated in the main thread, the cleanup
    runs (FULL, split lease released), the cleanup's control calls are bounded, no selection restore; a second TERM
    during the cleanup is ignored; main still writes the record."""

    def test_term_runs_the_cleanup(self):
        record, engine, state = term_run(self)
        self.assertEqual(record['cause'], 'terminated:SIGTERM')
        self.assertEqual(state['count'], 1)
        self.assertEqual(len(record['stages']), 2)
        self.assertTrue(record['cleanup']['full_restored'])
        self.assertTrue(record['cleanup']['split_lease_released'])
        self.assertFalse(record['selection_restore']['attempted'])
        self.assertEqual(engine.timeout_s, q.CLEANUP_TIMEOUT_S)
        self.assertEqual(engine.route, 'full')
        self.assertTrue(all(s.stop.is_set() for s in engine.streams))
        self.assertFalse(q.TERM['armed'])

    def test_second_term_in_the_cleanup_is_ignored(self):
        record, engine, state = term_run(self, second=True)
        self.assertEqual(state['count'], 2)
        self.assertEqual(record['cause'], 'terminated:SIGTERM')
        self.assertTrue(record['cleanup']['full_restored'])
        self.assertTrue(record['cleanup']['split_lease_released'])

    def test_term_after_the_run_is_counted_not_raised(self):
        """Disarmed (the record write after run()): a first TERM there is counted, nothing raised."""
        record, engine, state = term_run(self, at_stage=99)
        self.assertEqual(record['verdict'], 'PASS', record.get('cause'))
        os.kill(os.getpid(), signal.SIGTERM)
        for _ in range(1000):
            pass
        self.assertEqual(state['count'], 1)

    def test_main_writes_the_record_after_a_term(self):
        saved = signal.getsignal(signal.SIGTERM)
        self.addCleanup(signal.signal, signal.SIGTERM, saved)
        out = Path(scratch()) / 'quiet-split.json'
        clock = Clock()
        engine = Engine(clock)
        status = engine.status

        def status_then_term():
            reply = status()
            if engine.calls.count('status') == 1:
                os.kill(os.getpid(), signal.SIGTERM)       # the entry status: Terminated before the selection
            return reply
        engine.status = status_then_term
        real = q.NativeRuntimeClient
        q.NativeRuntimeClient = lambda *a, **k: engine
        self.addCleanup(setattr, q, 'NativeRuntimeClient', real)
        self.assertEqual(q.main(['--url', 'http://127.0.0.1:1', '--model', 'm', '--run-id', 'r', '--epoch', '3',
                                 '--out', str(out)]), 2)
        record = json.loads(out.read_text())
        self.assertEqual((record['cause'], record['sigterm_count']), ('terminated:SIGTERM', 1))
        self.assertTrue(record['cleanup']['split_lease_released'])
        self.assertNotIn(q.STAGE_CONTROL, engine.calls)


class RealReader(unittest.TestCase):
    """The real stream_reader, close_streams and warm_one over a loopback server: chunks counted, the bearer header
    sent, a reader blocked between chunks (a paused engine sends nothing) ends on close, [DONE] marks an early end,
    the warm request is non-streaming with max_tokens 1 and its usage is read."""

    def serve(self, chunks, pause_s, done):
        import http.server
        import threading
        seen = {}

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                seen['auth_len'] = len(self.headers.get('Authorization') or '')
                seen['body'] = self.rfile.read(int(self.headers['Content-Length']))
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                try:
                    for i in range(chunks):
                        self.wfile.write(b'data: {"choices":[{"text":"x"}]}\n\n')
                        self.wfile.flush()
                    if done:
                        self.wfile.write(b'data: [DONE]\n\n')
                        self.wfile.flush()
                    else:
                        import time
                        time.sleep(pause_s)
                except OSError:
                    pass

            def log_message(self, *a):
                pass
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return 'http://127.0.0.1:%d' % server.server_address[1], seen

    def test_blocked_reader_closes(self):
        import time
        url, seen = self.serve(5, 20, False)
        old = os.environ.get('VLLM_API_KEY')
        os.environ['VLLM_API_KEY'] = 'k' * 12
        try:
            stream = q.start_stream(url, 'm', q.LOW, 3, time.monotonic_ns, 30)
            self.assertTrue(stream.opened.wait(10))
            deadline = time.monotonic() + 10
            while len(stream.chunks) < 5 and time.monotonic() < deadline:
                time.sleep(.01)
            t0 = time.monotonic()
            self.assertEqual(q.close_streams([stream], join_s=5), [])
            self.assertLess(time.monotonic() - t0, 4)
        finally:
            if old is None:
                os.environ.pop('VLLM_API_KEY')
            else:
                os.environ['VLLM_API_KEY'] = old
        self.assertEqual(len(stream.chunks), 5)
        self.assertIsNone(stream.error)
        self.assertIsNone(stream.finished_ns)
        self.assertIsNotNone(stream.view(None)['first'])
        self.assertEqual(seen['auth_len'], len('Bearer ') + 12)
        body = json.loads(seen['body'])
        self.assertEqual((body['stream'], body['ignore_eos'], body['max_tokens'], len(body['prompt'])),
                         (True, True, q.MAX_TOKENS, q.LOW))
        self.assertEqual(body['prompt'], q.prompt_ids(3, q.LOW))

    def test_done_marks_early_end(self):
        import time
        url, _ = self.serve(3, 0, True)
        stream = q.start_stream(url, 'm', q.LOW, 0, time.monotonic_ns, 30)
        stream.thread.join(10)
        self.assertEqual(len(stream.chunks), 3)
        self.assertIsNotNone(stream.finished_ns)
        self.assertTrue(stream.view((0, time.monotonic_ns()))['finished_early'])

    def test_warm_one_is_non_streaming_and_reads_usage(self):
        import http.server
        import threading
        seen = {}

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                seen['body'] = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                data = json.dumps(dict(usage=dict(prompt_tokens=q.HIGH, prompt_tokens_details=dict(cached_tokens=0)))).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        out = q.warm_one('http://127.0.0.1:%d' % server.server_address[1], 'm', 5, 10)
        self.assertEqual((out['status'], out['prompt_tokens'], out['cached_tokens']), (200, q.HIGH, 0))
        self.assertEqual((seen['body']['stream'], seen['body']['max_tokens'], len(seen['body']['prompt'])), (False, 1, q.HIGH))


class VerbatimCopy(unittest.TestCase):
    def defs(self, text, names):
        import ast
        tree = ast.parse(text)
        return {n.name: ast.get_source_segment(text, n) for n in tree.body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names}

    def test_restoration_copied_verbatim(self):
        """status_view and restore_full are native41's, byte for byte, when that source is present."""
        src = HERE / 'e3_native41_unarmed_box.py'
        if not src.is_file():
            return                                   # a bundle without native41's client: the copy stands alone
        ours = (HERE / 'e3_quiet_split_box.py').read_text()
        self.assertEqual(self.defs(ours, ('status_view', 'restore_full')), self.defs(src.read_text(), ('status_view', 'restore_full')))
        self.assertEqual(len(self.defs(ours, ('status_view', 'restore_full'))), 2)

    def test_worker_copied_verbatim(self):
        """Worker and rank_rows are e3_attn_meta_box's, byte for byte, when that source is present."""
        src = HERE / 'e3_attn_meta_box.py'
        if not src.is_file():
            return
        ours = (HERE / 'e3_quiet_split_box.py').read_text()
        self.assertEqual(self.defs(ours, ('Worker', 'rank_rows')), self.defs(src.read_text(), ('Worker', 'rank_rows')))
        self.assertEqual(len(self.defs(ours, ('Worker', 'rank_rows'))), 2)
        import e3_attn_meta_box as am
        self.assertEqual((q.WORKER_ACTION, q.SELECT, q.READ, q.RANKS), (am.WORKER_ACTION, am.SELECT, am.READ, am.RANKS))


def mutant(old, new):
    path = HERE / 'e3_quiet_split_box.py'
    text = path.read_text()
    assert text.count(old) == 1, old
    spec = importlib.util.spec_from_loader('qs_mutant', loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(compile(text.replace(old, new), 'qs_mutant', 'exec'), module.__dict__)
    module.restore_full = restore_stub
    return module


class Mutants(Base):
    """Each guard removed: its negative passes, so the negative above is what catches it."""
    CASES = (
        ("    if status.get('route') != 'split':\n        out.append('route-left-split')\n", '', 'route-left-split'),
        ("    if status.get('admission_intent') is not None:\n        out.append('admission-armed')\n", '',
         'admission-armed'),
        # the poll guard removed, the run-end count still refuses (cause without the poll prefix)
        ("    if (status.get('native_admission') or {}).get('admitted') != admitted:\n"
         "        out.append('native-admitted')\n", '', 'poll:native-admitted'),
        ("    if type(arm) is not dict or arm.get('manual_owned') is not True:\n        out.append('manual-lease-lost')\n",
         '', 'manual-lease-lost'),
        ("    if type(step) is not dict or step.get('decode_merge') != DECLARED['decode_merge']:\n"
         "        out.append('decode-merge-not-always')\n", '', 'decode-merge-not-always'),
        ("        if any(v['window_chunks'] < 1 for v in views):\n            raise GuardError('decoder-stalled-in-hold')\n",
         '', 'decoder-stalled-in-hold'),
        ("        if type(row['merge_steps']) is not int or row['merge_steps'] < 1:\n"
         "            raise GuardError('merge-not-active-in-hold')\n", '', 'merge-not-active-in-hold'),
        ("        if row['pre_hold_tokens_max'] > PRE_HOLD_TOKENS:\n            raise GuardError('history-off-band')\n",
         '', 'history-off-band'),
        ("        if any(v is None or v <= 0 for v in record['attn_meta']['batched_delta'].values()):\n"
         "            raise GuardError('attn-meta-not-engaged')\n", '', 'attn-not-engaged'),
        ("        if record['attn_meta']['selected'] is not True:\n            raise GuardError('attn-meta-selection-unproved')\n",
         '', 'attn-not-selected'),
    )

    def test_removed_guard_lets_its_fault_pass(self):
        for old, new, fault in self.CASES:
            with self.subTest(fault=fault):
                record, _ = run(fault.replace('poll:', ''), module=mutant(old, new))
                self.assertNotIn(CAUSE.get(fault, fault), record.get('cause') or '', 'mutant still refused: the case is dead')

    def test_native_admitted_both_guards_removed_passes(self):
        text = (HERE / 'e3_quiet_split_box.py').read_text()
        assert text.count("        if record['native_admitted_after'] != admitted:\n") == 1
        m = mutant("    if (status.get('native_admission') or {}).get('admitted') != admitted:\n"
                   "        out.append('native-admitted')\n", '')
        record, _ = run('native-admitted', module=m)
        self.assertEqual(record['cause'], 'native-admitted')          # the run-end guard alone
        src = text.replace("    if (status.get('native_admission') or {}).get('admitted') != admitted:\n"
                           "        out.append('native-admitted')\n", '').replace(
            "        if record['native_admitted_after'] != admitted:\n            raise GuardError('native-admitted')\n", '')
        module = types.ModuleType('qs_mutant2')
        exec(compile(src, 'qs_mutant2', 'exec'), module.__dict__)
        module.restore_full = restore_stub
        record, _ = run('native-admitted', module=module)
        self.assertEqual(record['verdict'], 'PASS')

    def test_declaration_mutant(self):
        m = mutant("    if tuple(plan) != DECLARED['plan']:\n        out.append('plan-not-declared')\n", '')
        record, engine = run(module=m, plan=q.PLAN[:2])
        self.assertNotIn('declaration', record.get('cause') or '')
        self.assertIn('enter', engine.calls)

    def test_fallback_mutant(self):
        """The drain labelled 'disconnect' whatever it took: the fallback test catches it."""
        m = mutant("            return ('disconnect' if spent <= ABORT_BOUND_S else 'max_tokens'), round(spent, 3)\n",
                   "            return 'disconnect', round(spent, 3)\n")
        record, _ = run('disconnect-not-aborting', module=m)
        self.assertTrue(all(s['abort'] == 'disconnect' for s in record['stages']))      # what the fallback test refuses

    def test_term_handler_mutant(self):
        """A handler that does not raise: the run goes on to PASS, so the SIGTERM test's cause check catches it."""
        m = mutant("        if TERM['armed'] and TERM['count'] == 1:\n            raise Terminated('terminated:SIGTERM')\n",
                   "        pass\n")
        record, _, _ = term_run(self, module=m)
        self.assertEqual(record['verdict'], 'PASS')

    def test_lease_check_mutant(self):
        """The lease check reading only the route: a split lease left in place reads released."""
        m = mutant("    return dict(split_lease_released=status.get('route') == 'full' and status.get('control_route') is None,",
                   "    return dict(split_lease_released=status.get('route') == 'full',")
        clock = Clock()
        engine = Engine(clock)
        engine.control_route = dict(owner='native-control')
        self.assertTrue(m.lease_check(engine)['split_lease_released'])
        self.assertFalse(q.lease_check(engine)['split_lease_released'])


if __name__ == '__main__':
    unittest.main()
