"""Native41 unarmed-hold clock proof across the CT/container identity split. CPU only: no docker, network or GPU.

Parent behaviour this fails on: e3_native40_unarmed_box.run (lines 113-118) compares the engine
status clock (boot_id, time namespace) with the client's own e3_native39_w1.clock_identity(). On the
box (q18a-native40-unarmed-hold run 1) the client is plain python3 in the LXC CT. It sees an lxcfs boot_id
(d7ebe876...) and the init time namespace. The engine container reports the kernel boot_id
(3bbacd29-...) and its own time namespace. So native40 refused with client-engine-clock-unproved
(HARNESS) after the engine had done everything right. The hub self-run missed it because client and
engine shared one namespace view. On the parent this module fails at import
(no e3_native41_unarmed_box). Connected.test_ct_view_split runs native40's run on the same composition
and asserts that box refusal.

Classes:
* Connected: a native38 startup-bound stack (real driver, endpoint, session) behind the loopback
  server, reached by the real NativeRuntimeClient and LoopbackControl over HTTP. Every status is the
  real driver status. Its clock record comes from engine_driver._clock_record in this process. Then:
  - The CT view is forced on the client side (the recorded CT values). Native40 refuses exactly as on
    the box. Native41 proves both phases with the real prove_clock_domain against the container
    identity. That identity is read from this process's real /proc files (the docker CLI is the leaf).
  - The exact queue CLI (with --container-id) runs through main.
  - Refusals: nonzero monotonic/boottime offsets on either side, missing or unparsable offsets,
    container time namespace or boot_id unlike the engine's, an engine sample outside the bracket, no
    container id, docker failure. All refuse before the split entry. The end proof refuses a stale
    sample or a changed container identity.
* Offsets: the parser accepts only the kernel's exact two-clock shape.

Real objects: e3_native41_unarmed_box (run, main, clock_proof, container_identity, client_identity,
the offsets parser), e3_native40_unarmed_box.run (parent), native_overlap_metrics.prove_clock_domain,
engine_driver._clock_record, the native38 stack, attach_native_router endpoint, NativeOverlapSession,
NativeRuntimeClient, LoopbackControl, e3_native39_w1.restore_full, /proc reads (os.readlink,
timens_offsets) in Connected.
Fixture leaves (native41_clock_fixture_leaves.json): the CT client view, the docker CLI, the unarmed
hold/return sequence (test_native40_box_queue.Client), full_idle, procfs-shaped trees for refusals,
a stale engine clock sample, plus the inherited native41 queue leaves (Server, EngineLoop, attach,
pinned_reads).
"""
import argparse
import contextlib
import io
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / 'tools'), str(ROOT / 'patches/s5_engine')]
import e3_native39_w1  # noqa: E402
import e3_native40_unarmed_box as n40box  # noqa: E402
import e3_native41_unarmed_box as box  # noqa: E402
from e3_native_runtime_client import NativeRuntimeClient  # noqa: E402
import test_native40_box_queue as n40q  # noqa: E402
import test_native41_box_queue as q41  # noqa: E402

RUN_ID = 'native41-unarmed-fixture'
CID = '41' * 32  # synthetic 64-hex container id (docker inspect --format {{.Id}} shape)
# Recorded CT view (native40 README): lxcfs boot_id d7ebe876... (only 8 hex chars recorded; the rest
# is synthetic) and the init time namespace 4026531834.
CT_VIEW = {'basis': 'CLOCK_MONOTONIC', 'proc': 'harness',
           'boot_id': 'd7ebe876-0000-4000-8000-000000000000', 'time_namespace': 'time:[4026531834]'}
# Recorded engine container identity (unarmed-stock-prefill.json cleanup.status.clock).
BOX_ENGINE_NS = 'time:[4026533959]'
ZERO = 'monotonic           0         0\nboottime            0         0\n'
PASS = {'verdict': 'PASS'}
SYNTHETIC_PID = 4242


def real_engine_identity():
    """What engine_driver._clock_record reads in this process (the engine side of the stack)."""
    return (Path('/proc/sys/kernel/random/boot_id').read_text().strip(), os.readlink('/proc/self/ns/time'))


class Docker:
    """Leaf for box.docker_cli: the two CT-side docker reads, answered from the files the container exposes.

    inspect -> this process's pid (its /proc/<pid> is the engine's own process here, the engine
    container's init on the box); exec cat boot_id -> the real boot_id file, read directly.
    """

    def __init__(self, pids=None, boot_id=None, fail=None):
        self.pids, self.boot_id, self.fail, self.calls = list(pids or []), boot_id, fail, []

    def __call__(self, argv, timeout_s):
        self.calls.append((list(argv), timeout_s))
        if self.fail == argv[0]:
            raise OSError('docker-rc-1')
        if argv == ['inspect', '--format', '{{.State.Pid}}', CID]:
            return '%d\n' % (self.pids.pop(0) if self.pids else os.getpid())
        if argv == ['exec', CID, 'cat', '/proc/sys/kernel/random/boot_id']:
            return self.boot_id + '\n' if self.boot_id else Path('/proc/sys/kernel/random/boot_id').read_text()
        raise OSError('docker-rc-125')


def proc_tree(root, *, namespace, container=ZERO, client=ZERO, pids=(SYNTHETIC_PID,)):
    """Leaf for /proc: the same relative paths (<pid>/ns/time, <pid>/timens_offsets, self/timens_offsets)."""
    for pid in pids:
        (root / str(pid) / 'ns').mkdir(parents=True)
        os.symlink(namespace, root / str(pid) / 'ns/time')
        if container is not None:
            (root / str(pid) / 'timens_offsets').write_text(container)
    (root / 'self').mkdir()
    if client is not None:
        (root / 'self/timens_offsets').write_text(client)
    return str(root)


class Held(n40q.Client):
    """Leaf: test_native40_box_queue.Client's unarmed hold -> FULL return sequence after the split entry.

    Every status is also a real status over the real control plane. Before enter the real reply is
    returned unchanged. After it, the fake hold rows carry the real engine clock record,
    observed_clock_ns and engine epoch. stale='begin'/'end' swaps in an earlier real engine clock
    record for that phase's call (a cached or old reply).
    """

    def __init__(self, real, stale=None, earlier=None):
        super().__init__()
        self.real, self.stale, self.earlier, self.entered = real, stale, earlier, False

    def enter(self, phase):
        super().enter(phase)
        self.entered = True

    def call(self, action):
        real = self.real.call(action)
        if not self.entered:
            self.actions.append(('status', 'engine'))
            row, phase = dict(real), 'begin'
        else:
            row = super().call(action)
            row.update(clock=real.get('clock'), observed_clock_ns=real.get('observed_clock_ns'),
                       engine_epoch=real.get('engine_epoch'))
            receipt = row['native_admission'].get('last_split_hold_timeout')
            if receipt:
                receipt['engine_epoch'] = real.get('engine_epoch')
            phase = 'end' if self.poll > 4 else 'hold'
        if self.stale == phase:
            row['clock'] = dict(self.earlier)
        return row


class Connected(unittest.TestCase):
    def setUp(self):
        for path in ('/proc/self/timens_offsets', '/proc/self/ns/time', '/proc/sys/kernel/random/boot_id'):
            # Zero skips: a kernel or container without these files fails here, loudly.
            self.assertTrue(os.path.lexists(path), path + ' absent: the clock proof cannot be exercised')
        # Leaf (as test_native41_box_queue.Arm): vllm.logger for the startup receipt line.
        self._logger = sys.modules.get('vllm.logger')
        sys.modules['vllm.logger'] = SimpleNamespace(init_logger=logging.getLogger)
        self.stack, self.core = q41.native38.OwnerStartup().stack()
        q41.pinned_reads(self.stack.scheduler)
        q41.native38.bootstrap_full_startup(self.core, self.stack.driver, timeout_s=2)
        self.driver = self.stack.driver
        q41.auto.install(self.core, self.driver, {q41.auto.ENV: '1'})
        self.env = mock.patch.dict(os.environ, {'FLASH_S5_AUTO_ARM': '1', 'VLLM_API_KEY': q41.KEY})
        self.env.start()
        self.loop = q41.EngineLoop(self.core)
        app, endpoint, _ = q41.attach(q41.native.Frontend(self.core), '1')
        self.server = q41.Server(self.loop, app, endpoint).__enter__()
        # The queue's typed full-idle entry binds the session run (as full_idle's prepare does).
        self.real = NativeRuntimeClient(self.server.url, RUN_ID, epoch=self.driver.engine_epoch, timeout_s=10)
        self.real.prepare('full', 'full-prefill-solo')
        self.real.resume()
        self.earlier = self.real.call('status')['clock']
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.server.__exit__()
        self.loop.close()
        self.env.stop()
        self.tmp.cleanup()
        if self._logger is None:
            sys.modules.pop('vllm.logger', None)
        else:
            sys.modules['vllm.logger'] = self._logger

    def args(self, container_id=CID):
        return argparse.Namespace(url=self.server.url, model='fixture', run_id=RUN_ID,
                                  epoch=self.driver.engine_epoch, control_timeout_s=30, timeout_s=100,
                                  container_id=container_id)

    def run_box(self, module, client, **kw):
        return module.run(self.args(kw.pop('container_id', CID)), client=client, send=client.send,
                          sleep=lambda _: None, idle=lambda _: dict(PASS), **kw)

    def assert_proved(self, proof, engine):
        self.assertIs(proof['proved'], True, proof['reasons'])
        self.assertEqual(proof['reasons'], [])
        self.assertEqual(proof['container']['init_pid'], os.getpid())
        self.assertEqual(proof['container']['timens_offsets'], box.ZERO_OFFSETS)
        self.assertEqual(proof['client']['timens_offsets'], box.ZERO_OFFSETS)
        self.assertEqual(proof['client']['view'], {k: CT_VIEW[k] for k in ('boot_id', 'time_namespace')})
        self.assertEqual(proof['expected'], {'proved': True, 'boot_id': engine[0], 'time_namespace': engine[1]})
        domain = proof['domain']
        self.assertEqual((domain['boot_id'], domain['time_namespace'], domain['engine_basis']),
                         (engine[0], engine[1], 'clock-monotonic'))
        self.assertLessEqual(domain['client_send_ns'], domain['engine_sample_ns'])
        self.assertLessEqual(domain['engine_sample_ns'], domain['client_recv_ns'])
        self.assertEqual(proof['bracket_ns'], domain['client_recv_ns'] - domain['client_send_ns'])

    def test_ct_view_split_native40_refuses_native41_proves(self):
        engine = real_engine_identity()
        self.assertNotEqual(engine[0], CT_VIEW['boot_id'])  # the CT/container split
        parent, child, docker = Held(self.real), Held(self.real), Docker()
        with mock.patch.object(e3_native39_w1, 'clock_identity', return_value=dict(CT_VIEW)):
            old = self.run_box(n40box, parent)
            new = self.run_box(box, child, docker=docker)
        # Parent: the engine side passed and the identity comparison refused, exactly as on the box.
        self.assertEqual((old['verdict'], old['failure_class'], old['cause']),
                         ('INCONCLUSIVE', 'HARNESS', 'client-engine-clock-unproved'))
        self.assertEqual((old['automatic_return']['route'], old['request']['http_status']), ('full', 200))
        # Native41 on the same composition, the same CT view.
        self.assertEqual((new['verdict'], new['failure_class']), ('PASS', None), new.get('cause'))
        self.assertEqual(new['schema'], 'native41-unarmed-box/1')
        self.assertEqual(new['container_id'], CID)
        for phase in ('begin', 'end'):
            with self.subTest(phase=phase):
                self.assert_proved(new['clock_proofs'][phase], engine)
        begin, end = new['clock_proofs']['begin']['domain'], new['clock_proofs']['end']['domain']
        # The begin proof precedes the split entry and the plain request; the end proof follows the reply.
        self.assertEqual(child.actions[:2], [('status', 'engine'), ('enter', 'split-w1-decode')])
        self.assertLess(begin['client_recv_ns'], new['request']['completed_ns'])
        self.assertLess(new['request']['completed_ns'], end['client_send_ns'])
        self.assertFalse(any('prefill_begin' in action for action in child.actions))
        self.assertEqual(new['automatic_return']['route'], 'full')
        self.assertEqual([call[0][0] for call in docker.calls], ['inspect', 'exec'] * 2)
        self.assertTrue(all(call[1] == box.DOCKER_TIMEOUT_S for call in docker.calls))
        self.assertNotIn('cleanup', new)

    def test_exact_queue_cli_through_main(self):
        made, docker = [], Docker()

        def factory(url, run_id, *, epoch, timeout_s):
            made.append(Held(NativeRuntimeClient(url, run_id, epoch=epoch, timeout_s=timeout_s)))
            return made[-1]

        def send(*args):
            return made[-1].send(*args)
        out = Path(self.tmp.name) / 'unarmed-stock-prefill.json'
        # The native40 queue line plus the engine container id: --container-id "${_cid:-none}".
        argv = ['--url', self.server.url, '--model', 'fixture', '--run-id', RUN_ID,
                '--epoch', str(self.driver.engine_epoch), '--timeout-s', '100', '--control-timeout-s', '30',
                '--out', str(out), '--container-id', CID]
        stdout = io.StringIO()
        with mock.patch.object(box, 'NativeRuntimeClient', factory), \
                mock.patch.object(box, 'plain_request', send), \
                mock.patch.object(box, 'full_idle', return_value=dict(PASS)), \
                mock.patch.object(box, 'docker_cli', docker), \
                mock.patch.object(e3_native39_w1, 'clock_identity', return_value=dict(CT_VIEW)), \
                contextlib.redirect_stdout(stdout):
            rc = box.main(argv)
            again = box.main(argv)  # an existing receipt is never overwritten
        record = json.loads(out.read_text())
        self.assertEqual((rc, again, stdout.getvalue()), (0, 2, 'NATIVE41_UNARMED PASS\n'), record.get('cause'))
        self.assertEqual(len(made), 1)
        self.assertEqual((record['schema'], record['verdict'], record['container_id']),
                         ('native41-unarmed-box/1', 'PASS', CID))
        for phase in ('begin', 'end'):
            self.assert_proved(record['clock_proofs'][phase], real_engine_identity())
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as missing:
            box.main(argv[:-2])
        self.assertEqual(missing.exception.code, 2)
        # The queue's fallback when docker inspect fails: refused before any engine action.
        out.unlink()
        stdout = io.StringIO()
        with mock.patch.object(box, 'NativeRuntimeClient', factory), \
                mock.patch.object(box, 'full_idle', return_value=dict(PASS)), \
                mock.patch.object(box, 'docker_cli', docker), contextlib.redirect_stdout(stdout):
            rc = box.main(argv[:-1] + ['none'])
        record = json.loads(out.read_text())
        self.assertEqual((rc, stdout.getvalue()), (2, 'NATIVE41_UNARMED INCONCLUSIVE\n'))
        self.assertTrue(record['cause'].startswith('client-engine-clock-unproved:begin:container-id-missing'))
        self.assertEqual(len(docker.calls), 4)

    def refuse(self, phase, reason, client, **kw):
        record = self.run_box(box, client, **kw)
        self.assertEqual((record['verdict'], record['failure_class']), ('INCONCLUSIVE', 'HARNESS'))
        self.assertTrue(record['cause'].startswith('client-engine-clock-unproved:%s:' % phase), record['cause'])
        proof = record['clock_proofs'][phase]
        self.assertIs(proof['proved'], False)
        self.assertIn(reason, proof['reasons'])
        self.assertIn(reason, record['cause'].split(':', 2)[2].split(','))
        self.assertIn('cleanup', record)
        return record

    def test_refusals_before_the_split_entry(self):
        boot_id, namespace = real_engine_identity()
        other_ns = next(ns for ns in (BOX_ENGINE_NS, 'time:[4026531834]', 'time:[1]') if ns != namespace)
        bad = {'container-monotonic': dict(container='monotonic 3 0\nboottime 0 0\n'),
               'container-boottime': dict(container='monotonic 0 0\nboottime 0 1\n'),
               'client-monotonic': dict(client='monotonic 0 5\nboottime 0 0\n'),
               'client-boottime': dict(client='monotonic 0 0\nboottime -2 0\n')}
        cases = [(name, 'container-timens-offsets-nonzero' if name.startswith('container') else
                  'client-timens-offsets-nonzero', dict(tree=tree)) for name, tree in bad.items()]
        cases += [
            ('client-offsets-missing', 'client-timens-offsets-unreadable:FileNotFoundError', dict(tree=dict(client=None))),
            ('container-offsets-missing', 'container-timens-offsets-unreadable:FileNotFoundError',
             dict(tree=dict(container=None))),
            ('client-offsets-unparsable', 'client-timens-offsets-unparsable', dict(tree=dict(client='monotonic 0 0\n'))),
            ('container-namespace-differs', 'time-namespace-differs', dict(tree=dict(namespace=other_ns))),
            ('container-boot-id-differs', 'boot-id-differs', dict(boot_id=CT_VIEW['boot_id'])),
            ('sample-outside-bracket', 'engine-sample-outside-request-bracket', dict(stale='begin')),
            ('no-container-id', 'container-id-missing', dict(container_id='none')),
            ('docker-inspect-fails', 'container-init-pid-unproved:docker-rc-1', dict(fail='inspect')),
            ('docker-exec-fails', 'container-boot-id-unproved:docker-rc-1', dict(fail='exec')),
        ]
        for i, (name, reason, case) in enumerate(cases):
            with self.subTest(case=name):
                kw = {}
                if 'tree' in case:
                    spec = dict(dict(namespace=namespace), **case['tree'])
                    kw['proc'] = proc_tree(Path(self.tmp.name) / str(i), **spec)
                    docker = Docker(pids=[SYNTHETIC_PID])
                else:
                    docker = Docker(boot_id=case.get('boot_id'), fail=case.get('fail'))
                if 'container_id' in case:
                    kw['container_id'] = case['container_id']
                client = Held(self.real, stale=case.get('stale'), earlier=self.earlier)
                record = self.refuse('begin', reason, client, docker=docker, **kw)
                self.assertNotIn(('enter', 'split-w1-decode'), client.actions)
                self.assertEqual(record['request'], {})  # the plain request was never sent
                self.assertNotIn('end', record['clock_proofs'])
                self.assertIs(record['cleanup']['full_restored'], True)
                self.assertEqual(self.real.call('status')['route'], 'full')
                if name == 'no-container-id':
                    self.assertEqual(docker.calls, [])
                if name == 'container-boot-id-differs':
                    self.assertEqual(record['clock_proofs']['begin']['domain']['boot_id'], boot_id)

    def test_end_proof_refuses_a_stale_sample_or_a_changed_container(self):
        _, namespace = real_engine_identity()
        record = self.refuse('end', 'engine-sample-outside-request-bracket',
                             Held(self.real, stale='end', earlier=self.earlier), docker=Docker())
        self.assertIs(record['clock_proofs']['begin']['proved'], True)
        self.assertEqual((record['automatic_return']['route'], record['request']['http_status']), ('full', 200))
        proc = proc_tree(Path(self.tmp.name) / 'restart', namespace=namespace, pids=(SYNTHETIC_PID, SYNTHETIC_PID + 1))
        record = self.refuse('end', 'container-identity-changed', Held(self.real),
                             docker=Docker(pids=[SYNTHETIC_PID, SYNTHETIC_PID + 1]), proc=proc)
        self.assertEqual(record['clock_proofs']['end']['reasons'][0], 'container-identity-changed')
        self.assertEqual(record['clock_proofs']['end']['container']['init_pid'], SYNTHETIC_PID + 1)


class Offsets(unittest.TestCase):
    def test_only_the_exact_two_clock_kernel_shape_parses(self):
        self.assertEqual(box.parse_timens_offsets(Path('/proc/self/timens_offsets').read_text()), box.ZERO_OFFSETS)
        self.assertEqual(box.parse_timens_offsets(ZERO), box.ZERO_OFFSETS)
        self.assertEqual(box.parse_timens_offsets('boottime 1 2\nmonotonic -3 4\n'),
                         {'monotonic': [-3, 4], 'boottime': [1, 2]})
        for text in ('', 'monotonic 0 0\n', 'monotonic 0 0\nboottime 0 0\nrealtime 0 0\n',
                     'monotonic 0 0\nmonotonic 0 0\nboottime 0 0\n', 'monotonic 0\nboottime 0 0\n',
                     'monotonic x 0\nboottime 0 0\n', 'unknown 0 0\nboottime 0 0\n', '1 0 0\n7 0 0\n', None):
            with self.subTest(text=text):
                self.assertIsNone(box.parse_timens_offsets(text))


if __name__ == '__main__':
    unittest.main()
