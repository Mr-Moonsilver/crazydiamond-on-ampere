"""Quiet-SPLIT control (e3_quiet_split_box, brief 11 C): a fake engine and fake streams, CPU only.

The fake engine keeps the state the tool reads (route, generation, pause, the stage-local selection, merge_steps,
native admitted, the auto-arm row, scheduler requests = live fake streams) and a fake clock advanced by sleep();
each status poll advances every live stream by one chunk on the fake clock. Every guard has a negative (one fault
injected, the verdict INCONCLUSIVE with that guard named, FULL restored), the declaration guard sends no control,
and mutants of the tool (a guard removed) turn a negative green, so the negatives are live.
"""
import importlib.util
from pathlib import Path
import types
import unittest

import e3_quiet_split_box as q

HERE = Path(__file__).resolve().parent


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def ns(self):
        return int(self.t * 1e9)

    def sleep(self, s):
        self.t += s


class FakeStream(q.Stream):
    pass


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
        self.polls_on_split = 0
        return {}

    def leave(self):
        self.calls.append('leave')
        self.route, self.generation = 'full', self.generation + 1
        return {}

    def live(self):
        return [s for s in self.streams if not s.stop.is_set() and s.finished_ns is None]

    def status(self):
        live = self.live()
        if not self.paused:
            for s in live:
                s.chunks.append(self.clock.ns())
            if (self.route == 'split' and self.selection['decode_merge'] == 'always'
                    and self.fault != 'merge-not-active-in-hold'):
                self.merge_steps += 1
        out = dict(route=self.route, route_generation=self.generation, paused=self.paused,
                   transition_pending=False, return_pending=None, admission_intent=None,
                   scheduler_requests=len(live), chunk=None, owned_work=dict(chunk=False),
                   native_admission=dict(admitted=self.admitted),
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
        elif f == 'stage-live-mismatch' and len(live) == 4:
            out['scheduler_requests'] = 8

    def starter(self, url, model, stage, index, now_ns, timeout_s):
        s = FakeStream(index)
        s.stall = self.fault == 'decoder-stalled-in-hold' and index == 0   # no chunk after its first token
        if self.fault != 'decoders-not-live-on-full' or index:
            s.first_ns = now_ns() - 1_000_000      # the first token came before the split window
            s.chunks.append(s.first_ns)
        s.opened.set()
        self.streams.append(s)
        return s


def args(**over):
    a = types.SimpleNamespace(url='http://127.0.0.1:8000', model='m', run_id='r', epoch=3, stages=(8, 4, 1),
                              hold_s=25.0, decode_merge='always', control_timeout_s=60, open_timeout_s=1,
                              drain_timeout_s=5, stream_timeout_s=5)
    for k, v in over.items():
        setattr(a, k, v)
    return a


def run(fault=None, module=q, **over):
    clock = Clock()
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
    record = module.run(args(**over), client=engine, starter=engine.starter, clock=clock, sleep=clock.sleep,
                        now_ns=clock.ns)
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
    def test_clean_run_passes_with_three_stages(self):
        record, engine = run()
        self.assertEqual(record['verdict'], 'PASS', record.get('cause'))
        self.assertEqual([s['live'] for s in record['stages']], [8, 4, 1])
        for row in record['stages']:
            self.assertGreaterEqual(row['window']['wall_s'], 25.0)
            self.assertLessEqual(row['window']['wall_s'], 26.0)
            self.assertGreater(row['merge_steps'], 0)
            self.assertEqual(len(row['streams']), row['live'])
            self.assertTrue(all(v['window_chunks'] >= 1 for v in row['streams']))
            self.assertIsNotNone(row['aggregate_tokens_per_s'])
        self.assertEqual(record['selection']['decode_merge'], 'always')
        self.assertIs(record['selection']['lazy_recv'], True)
        self.assertEqual(engine.route, 'full')
        self.assertFalse(engine.paused)

    def test_previous_selection_restored(self):
        record, engine = run()
        self.assertTrue(record['selection_restore']['restored'])
        self.assertEqual(engine.selection['decode_merge'], 'off')
        self.assertIs(engine.selection['lazy_recv'], False)

    def test_no_native_work_and_counts(self):
        record, engine = run()
        self.assertEqual(record['native_admitted_before'], record['native_admitted_after'])
        self.assertEqual(engine.calls.count('enter'), 3)
        self.assertEqual(engine.calls.count('leave'), 3)
        self.assertEqual(engine.calls.count(q.STAGE_CONTROL), 2)

    def test_snapshots_bounded(self):
        record, _ = run(hold_s=30.0)
        self.assertTrue(all(len(r['snapshots']) <= q.SNAPSHOT_LIMIT for r in record['stages']))


class Declaration(Base):
    def test_other_declarations_send_no_control(self):
        for over, reason in ((dict(stages=(8, 4)), 'stages-not-8-4-1'), (dict(stages=(4, 8, 1)), 'stages-not-8-4-1'),
                             (dict(hold_s=19.5), 'hold-not-20-30-s'), (dict(hold_s=31.0), 'hold-not-20-30-s'),
                             (dict(decode_merge='chunk'), 'decode-merge-not-always')):
            record, engine = run(**over)
            self.assertEqual(record['verdict'], 'INCONCLUSIVE')
            self.assertIn(reason, record['cause'])
            self.assertFalse(record['control_sent'])
            self.assertEqual(engine.calls, [], over)

    def test_bounds_accepted(self):
        for hold in (20.0, 30.0):
            self.assertEqual(q.declaration_reasons((8, 4, 1), hold, 'always'), [])

    def test_loopback_required(self):
        record, engine = run(url='http://<internal-host>:8000')
        self.assertEqual(record['cause'], 'loopback-required')
        self.assertEqual(engine.calls, [])

    def test_cli_refuses_existing_out(self):
        import tempfile
        with tempfile.TemporaryDirectory(dir=str(HERE.parent.parent / 'cx-20261114c' / 'scratch' / 'tmp')
                                         if (HERE.parent.parent / 'cx-20261114c').is_dir() else None) as d:
            out = Path(d) / 'x.json'
            out.write_text('{}')
            self.assertEqual(q.main(['--url', 'http://127.0.0.1:1', '--model', 'm', '--run-id', 'r', '--epoch', '1',
                                     '--out', str(out)]), 2)
            self.assertEqual(out.read_text(), '{}')


NEGATIVES = q.POLL_GUARDS + ('decoders-not-live-on-full', 'decoder-ended-in-hold', 'decoder-stalled-in-hold',
                             'merge-not-active-in-hold', 'enter-refused', 'merge-not-acked')
CAUSE = {'enter-refused': 'native-split-not-applied', 'merge-not-acked': 'selection-not-acknowledged'}


class Negatives(Base):
    def test_each_guard_refuses_and_restores_full(self):
        for fault in NEGATIVES:
            with self.subTest(fault=fault):
                record, engine = run(fault)
                self.assertEqual(record['verdict'], 'INCONCLUSIVE')
                self.assertEqual(record['failure_class'], 'HARNESS')
                self.assertIn(CAUSE.get(fault, fault), record['cause'])
                self.assertEqual(engine.route, 'full')
                self.assertFalse(engine.paused)
                self.assertTrue(record['cleanup']['full_restored'])
                self.assertTrue(all(s.stop.is_set() for s in engine.streams))

    def test_fault_on_the_second_stage(self):
        record, _ = run('stage-live-mismatch')
        self.assertEqual(record['verdict'], 'INCONCLUSIVE')
        self.assertIn('scheduler-requests-not-live', record['cause'])
        self.assertEqual([s['live'] for s in record['stages']], [8, 4])

    def test_entry_must_be_idle_full(self):
        clock = Clock()
        engine = Engine(clock)
        engine.route = 'split'
        record = q.run(args(), client=engine, starter=engine.starter, clock=clock, sleep=clock.sleep, now_ns=clock.ns)
        self.assertEqual(record['cause'], 'entry-not-idle-full')
        self.assertNotIn(q.STAGE_CONTROL, engine.calls)

    def test_no_selection_restore_when_full_not_restored(self):
        clock = Clock()
        engine = Engine(clock, 'route-left-split')
        q.restore_full = lambda c: dict(full_restored=False)
        record = q.run(args(), client=engine, starter=engine.starter, clock=clock, sleep=clock.sleep, now_ns=clock.ns)
        self.assertFalse(record['selection_restore']['attempted'])
        self.assertEqual(engine.calls.count(q.STAGE_CONTROL), 1)


class RealReader(unittest.TestCase):
    """The real stream_reader and close_streams over a loopback SSE server: chunks counted, the bearer header sent,
    a reader blocked between chunks (a paused engine sends nothing) ends on close, [DONE] marks an early end."""

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
        import os
        import time
        url, seen = self.serve(5, 20, False)
        old = os.environ.get('VLLM_API_KEY')
        os.environ['VLLM_API_KEY'] = 'k' * 12
        try:
            stream = q.start_stream(url, 'm', 8, 3, time.monotonic_ns, 30)
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
        self.assertEqual(seen['auth_len'], len('Bearer ') + 12)
        import json
        body = json.loads(seen['body'])
        self.assertEqual((body['stream'], body['ignore_eos'], body['max_tokens'], len(body['prompt'])),
                         (True, True, q.MAX_TOKENS, q.PROMPT_TOKENS))

    def test_done_marks_early_end(self):
        import time
        url, _ = self.serve(3, 0, True)
        stream = q.start_stream(url, 'm', 1, 0, time.monotonic_ns, 30)
        stream.thread.join(10)
        self.assertEqual(len(stream.chunks), 3)
        self.assertIsNotNone(stream.finished_ns)
        self.assertTrue(stream.view((0, time.monotonic_ns()))['finished_early'])


class VerbatimCopy(unittest.TestCase):
    def test_restoration_copied_verbatim(self):
        """status_view and restore_full are native41's, byte for byte, when that source is present."""
        src = HERE / 'e3_native41_unarmed_box.py'
        if not src.is_file():
            return                                   # a bundle without native41's client: the copy stands alone
        import ast
        def defs(text):
            tree = ast.parse(text)
            return {n.name: ast.get_source_segment(text, n) for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name in ('status_view', 'restore_full')}
        ours = (HERE / 'e3_quiet_split_box.py').read_text()
        self.assertEqual(defs(ours), defs(src.read_text()))
        self.assertEqual(len(defs(ours)), 2)


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
    )

    def test_removed_guard_lets_its_fault_pass(self):
        for old, new, fault in self.CASES:
            with self.subTest(fault=fault):
                record, _ = run(fault.replace('poll:', ''), module=mutant(old, new))
                self.assertNotIn(fault, record.get('cause') or '', 'mutant still refused: the case is dead')

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
        m = mutant("    if tuple(stages) != DECLARED['stages']:\n        out.append('stages-not-8-4-1')\n", '')
        record, engine = run(module=m, stages=(8, 4))
        self.assertNotIn('cause', {k: v for k, v in record.items() if k == 'cause' and 'declaration' in v})
        self.assertIn('enter', engine.calls)


if __name__ == '__main__':
    unittest.main()
