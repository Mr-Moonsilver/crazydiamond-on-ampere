"""Native43 lifecycle queue (short scope): the exact derivation from the box-run native39 lifecycle texts, the
short window at real wall clock, and the queue's own lifecycle bytes executed under bash against a loopback engine
leaf.  CPU only, no box action.

Connected compositions (the queue's bytes with the real checks, clients and grades):
  Tail      gates (startup, row1) -> instruments (per-rank maps) -> native lifetime -> typed FULL hygiene ->
            instruments end -> the on_exit native43 lifecycle grade: the queue segment with the real instruments
            check, the real NativeRuntimeClient (PY_HYGIENE), the real native43_lifecycle_grade over the real
            native35_lifetime_grade; a fake native lifetime client and a fake docker CLI.  No leak or pressure tool
            is invoked; the grade reports them NOT_RUN.
  Workload  the real e3_q18a_lifecycle CLI with the native43 queue's argv on the trunk W2 manifest: it validates the
            manifest and reaches its first control call.
  Settings  the native43 lifecycle settings sourced through the real setcfg algorithm, the S5 launcher render and
            the declaration projection (test_native39_trunk_queue.Launcher.declared), then the declare script's own
            PY_DECL, the preflight's env assertion lines and the native43 engine's own policy/trace parser
            (qsa_operation.native_qsa_settings) on that identity.
Fixture leaves: native43_lifecycle_fixture_leaves.json.
"""
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import native39_lifecycle_gate
import native43_lifecycle_gate as gate
import native43_lifecycle_grade
import test_native39_lifecycle_queue as lq  # leaves and helpers only; its TestCases are collected in their own module
import test_native39_trunk_queue as trunkq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = HERE / 'runs'
LIFECYCLE = RUNS / 'q18a_native43_lifecycle.draft.sh'
SETTINGS = RUNS / 'q18a_native43_lifecycle_settings.sh'
DECLARE = RUNS / 'q18a_native43_lifecycle_declare.sh'
BASE_SETTINGS = RUNS / 'q18a_native39_lifecycle_settings.sh'
DENSE_SETTINGS = RUNS / 'q18a_native43_dense_settings.sh'
# The parent texts: the native39 lifecycle texts as held in bundle q18a-native39-lifecycle-bundle-b90f1e292956 and run on
# the box (native39 lifecycle held-a run 1, 1 Nov 2026), and Codex's native43 dense arm settings (109313a).
PARENT_SHA = {'q18a_native39_lifecycle.draft.sh': '4222ae014296d193cbdc718615a3cefb51de196dc55c00321ec1dadfb58ee136',
              'q18a_native39_lifecycle_settings.sh': 'a5a6ac2acdfed7275d60fd5de27ae9dad155f0e0e3ecf38e5008c47c4a9d4e68',
              'q18a_native39_lifecycle_declare.sh': '59fc9ca64715f663ea66a69336122d259705306ae777be779f59112893f725f7',
              'q18a_native43_dense_settings.sh': 'dcb3f78efe86c7996115cfed1d2cb072a615c59d0dd41dd4eb0cc6bfc5b1667d'}
MANIFEST = lq.MANIFEST
KEY = lq.KEY
QUALIFIED = {'startup': 'PASS', 'row1': 'PASS'}
SLOT_RECEIPTS = []
QUEUE_RECEIPTS = []  # one row per connected case, appended only after its assertions (the bundle probe asserts the set)


def record_case(case, **fields):
    QUEUE_RECEIPTS.append(dict(case=case, verdict='PASS', **fields))


def _tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def drafts_tool():
    return _tool('native43_lifecycle_queue_drafts')


def base_drafts_tool():
    return _tool('native39_lifecycle_queue_drafts')


# Fixture leaf native43-lifecycle-client: the native lifetime CLI (e3_q18a_lifecycle) as a record writer with the
# record shape of e3_q18a_lifecycle.run: PASS (cancel PASS, reuse complete) or, with FAKE/reuse-incomplete present,
# the INCONCLUSIVE record native39 lifecycle held-a run 1 retained (missing native-reuse-incomplete, rc 2).
FAKE_LIFECYCLE = r'''
import json, os, sys
from pathlib import Path
fake = Path(os.environ['FAKE'])
with open(fake / 'calls.jsonl', 'a') as f:
    f.write(json.dumps({'name': 'e3_q18a_lifecycle', 'argv': sys.argv[1:]}) + '\n')
def arg(key): return sys.argv[sys.argv.index(key) + 1]
incomplete = (fake / 'reuse-incomplete').exists()
record = {'schema': 's5-native-lifetime/1', 'run_id': arg('--run-id'), 'engine_epoch': int(arg('--epoch')),
          'verdict': 'INCONCLUSIVE' if incomplete else 'PASS', 'failures': [],
          'missing': ['native-reuse-incomplete'] if incomplete else [], 'cancel_grade': {'verdict': 'PASS'},
          'reuse': {'complete': not incomplete}}
with open(arg('--out'), 'x') as f: json.dump(record, f)
print('NATIVE_LIFETIME ' + record['verdict']); sys.exit(2 if incomplete else 0)
'''
# Fixture leaves reused unchanged from test_native39_lifecycle_queue and declared again for this suite:
#   fixture leaf queue-native-session
#   fixture leaf queue-docker-exec-maps
#   fixture leaf lifecycle-engine-leaf (only /health and the native session endpoint are reached)
#   fixture leaf lifecycle-docker-cli (inspect and exec only)
#   fixture leaf lifecycle-identity-capture
#   fixture leaf lifecycle-refused-port (Workload)
#   fixture leaf queue-declare-fixture-tree (Settings, Tail)
ARTS = ('gates.json', 'instruments-lifecycle.json', 'instruments-end.json', 'lifecycle.json', 'hygiene-full.json',
        'native43-lifecycle-grade.json')
NOT_RUN_PHASES = ('pool-telemetry', 'cross-session-leak', 'pressure-ready', 'partial-leak-driver', 'partial-leak',
                  'pressure-completed', 'pressure-driver', 'pressure-lifetime', 'native35-lifetime')


class Tail(unittest.TestCase):
    """From the lifecycle preamble to the end of the lifecycle region, then the ending's stop_pressure and the
    on_exit native43 lifecycle grade lines: the queue's own bytes under bash with the real step/record/stop_pressure
    and health_ok, against the loopback leaf."""

    def run_tail(self, verdicts=QUALIFIED, crashed=0, mapped_rank=None, stuck=False, crash_stream=False,
                 reuse_incomplete=False):
        declared, _ = trunkq.Launcher.declared(self, SETTINGS)
        draft = LIFECYCLE.read_text()
        base = base_drafts_tool()
        start, end = draft.index(base.LIFECYCLE_PREAMBLE), draft.index(base.ENDING)
        region = draft[start:end]
        grade_lines = drafts_tool().GRADE
        self.assertEqual(draft.count(grade_lines), 1)
        health = [line for line in draft.splitlines(True) if line.startswith('health_ok() {')]
        self.assertEqual(len(health), 1)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            fake, h, q, bindir = d / 'fake', d / 'h', d / 'q', d / 'bin'
            for p in (fake, h, q, bindir, fake / 'maps', h / 'workloads'):
                p.mkdir()
            session = trunkq.NativeSession()
            session.stuck = stuck
            with lq.LifecycleLeaf(session) as leaf:
                for name in ('native43_lifecycle_grade.py', 'native35_lifetime_grade.py', 'e3_native39_trunk_checks.py'):
                    shutil.copyfile(HERE / name, h / name)  # the real tools, run from the harness dir as on the box
                (h / 'e3_q18a_lifecycle.py').write_text(FAKE_LIFECYCLE)
                shutil.copyfile(MANIFEST, h / 'workloads' / MANIFEST.name)
                (fake / 'current-id').write_text(trunkq.NEW_ID)
                if reuse_incomplete:
                    (fake / 'reuse-incomplete').write_text('')
                (q / 'engine-stream.log').write_text(trunkq.BIND_LINE + (lq.CRASH_LINE if crash_stream else ''))
                (q / 'engine.identity-start.json').write_text(json.dumps({'identity': declared['identity']}))
                for rank in range(4):
                    text = trunkq.OBSERVER_MAPS if rank == mapped_rank else trunkq.CLEAN_MAPS
                    (fake / 'maps' / str(1001 + rank)).write_text(text)
                (bindir / 'docker').write_text('#!' + sys.executable + '\n' + lq.FAKE_DOCKER)
                (bindir / 'docker').chmod(0o755)
                script = ('set -uo pipefail\nFAILS=0; INCS=0; ENGINE_CRASHED=%d; PRESSURE_PID=\'\'; PRESSURE_STOP=\'\'\n'
                          'declare -A PHASE_VERDICT=()\n' % crashed
                          + trunkq.functions(draft, ('step', 'record', 'stop_pressure')) + health[0]
                          + ''.join('PHASE_VERDICT[%s]=%s\n' % item for item in verdicts.items())
                          + region + 'stop_pressure || true\n' + grade_lines
                          + 'for k in "${!PHASE_VERDICT[@]}"; do echo "PV $k=${PHASE_VERDICT[$k]}"; done\n'
                            'echo "TOTAL FAILS=$FAILS INCS=$INCS"\n')
                done = subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=400, cwd=str(d),
                                      env=trunkq.env(PATH=str(bindir) + os.pathsep + os.environ['PATH'], FAKE=str(fake),
                                                     H=str(h), Q=str(q), PROBE_URL=leaf.url, MODEL='m',
                                                     CONTAINER='qwen-flash-prod', VLLM_API_KEY=KEY,
                                                     RUN_ID='q18a-native43-lifecycle-cpu-1', EPOCH='7',
                                                     PYTHONPATH=os.pathsep.join((str(h), str(HERE),
                                                                                 str(ROOT / 'patches/s5_engine')))))
            rows = [json.loads(line) for line in (fake / 'calls.jsonl').read_text().splitlines()] \
                if (fake / 'calls.jsonl').exists() else []
            arts = {name: json.loads((q / name).read_text()) for name in ARTS if (q / name).exists()}
            left = sorted(p.name for p in q.iterdir())
            return done, trunkq.phases(done.stdout), rows, arts, leaf, session, dict(h=str(h), q=str(q), files=left)

    def assert_short_scope(self, done, pv, rows, leaf, paths):
        for name in NOT_RUN_PHASES:
            self.assertNotIn(name, pv)
            self.assertNotIn('STEP ' + name + ' ', done.stdout)
        self.assertEqual([r['name'] for r in rows if r['name'] != 'e3_q18a_lifecycle'], [])
        self.assertEqual((leaf.chats, leaf.tokenized), (0, 0))
        for name in ('cross-session-leak.log', 'pressure.json', 'partial-leak.json', 'pool-telemetry.log'):
            self.assertNotIn(name, paths['files'])

    def test_connected_short_lifecycle_block_passes_without_leak_or_pressure(self):
        done, pv, rows, arts, leaf, session, paths = self.run_tail()
        self.assertEqual(done.returncode, 0, done.stderr)
        passed = ('instruments-lifecycle', 'lifecycle', 'hygiene-full', 'instruments-end', 'native43-lifecycle')
        self.assertEqual({k: pv.get(k) for k in passed}, dict.fromkeys(passed, 'PASS'), done.stdout)
        self.assertIn('STEP lifetime-gates PASS', done.stdout)
        self.assert_short_scope(done, pv, rows, leaf, paths)
        gates = arts['gates.json']
        self.assertEqual((gates['outcomes'], gates['rule4_control_continuation']), (QUALIFIED, False))
        for name in ('instruments-lifecycle', 'instruments-end'):
            ranks = arts[name + '.json']['ranks']
            self.assertEqual({r['observer_mapped'] for r in ranks.values()}, {False}, name)
            self.assertEqual(sorted(r['pid'] for r in ranks.values()), [1001, 1002, 1003, 1004])
        self.assertTrue(arts['instruments-end.json']['same_processes'])
        self.assertEqual(session.calls, ['pause', 'resume', 'prepare', 'resume', 'pause', 'resume'])
        hygiene = arts['hygiene-full.json']
        self.assertEqual((hygiene['prepare']['route'], hygiene['prepare']['typed_acks'], hygiene['resume']['resumed']),
                         ('full', True, True))
        h, q = paths['h'], paths['q']
        self.assertEqual(rows, [{'name': 'e3_q18a_lifecycle', 'argv': [
            '--url', leaf.url, '--model', 'm', '--run-id', 'q18a-native43-lifecycle-cpu-1', '--epoch', '7',
            '--workload', h + '/workloads/q18a-w2-overlap-native39.json', '--out', q + '/lifecycle.json']}])
        grade = arts['native43-lifecycle-grade.json']
        self.assertEqual((grade['verdict'], grade['failures'], grade['missing'], grade['not_run']),
                         ('PASS', [], [], ['cross-session-leak', 'partial-leak', 'pressure-completed', 'pressure-lifetime']))
        self.assertIs(grade['release_qualified'], False)
        self.assertIn('TOTAL FAILS=0 INCS=0', done.stdout)
        record_case('tail-short-lifecycle-connected-pass', calls=[r['name'] for r in rows], session_calls=session.calls,
                    lifecycle_grade='PASS', not_run=grade['not_run'])

    def test_reuse_incomplete_stays_inconclusive_in_the_queue_and_the_grade(self):
        # The native39 lifecycle held-a run 1 outcome (the 97,152-row split-prefill cliff): the lifecycle record is
        # INCONCLUSIVE with native-reuse-incomplete; hygiene and the end receipt still run; the grade names it.
        done, pv, rows, arts, leaf, session, paths = self.run_tail(reuse_incomplete=True)
        self.assertEqual((pv['lifecycle'], pv['hygiene-full'], pv['instruments-end']), ('INCONCLUSIVE', 'PASS', 'PASS'))
        self.assert_short_scope(done, pv, rows, leaf, paths)
        grade = arts['native43-lifecycle-grade.json']
        self.assertEqual((grade['verdict'], grade['failures']), ('INCONCLUSIVE', []))
        self.assertEqual(grade['missing'], ['native-lifetime:INCONCLUSIVE', 'native-lifetime:reuse-incomplete'])
        self.assertEqual(pv['native43-lifecycle'], 'INCONCLUSIVE')
        record_case('tail-reuse-incomplete-inconclusive', lifecycle='INCONCLUSIVE', lifecycle_grade='INCONCLUSIVE',
                    missing=grade['missing'])

    def test_row1_not_pass_withholds_instruments_and_the_block(self):
        done, pv, rows, arts, leaf, session, paths = self.run_tail(dict(QUALIFIED, row1='INCONCLUSIVE'))
        self.assertIn('STEP lifetime-gates INCONCLUSIVE', done.stdout)
        self.assertEqual(pv['lifecycle'], 'INCONCLUSIVE')
        self.assertNotIn('instruments-lifecycle', pv)
        self.assertNotIn('instruments-end', pv)
        self.assertEqual((rows, session.calls, leaf.chats, leaf.tokenized), ([], [], 0, 0))
        grade = arts['native43-lifecycle-grade.json']
        self.assertEqual((grade['verdict'], grade['failures']), ('INCONCLUSIVE', []))
        self.assertIn('native-lifetime:record-missing', grade['missing'])
        record_case('tail-row1-not-pass-withholds', calls=[], session_calls=[], lifecycle_grade='INCONCLUSIVE')

    def test_engine_crash_withholds_the_block_and_the_crash_stream_fails_the_grade(self):
        done, pv, rows, arts, leaf, session, paths = self.run_tail(crashed=1, crash_stream=True)
        self.assertIn('STEP lifetime-gates PASS', done.stdout)
        self.assertEqual(pv['lifecycle'], 'INCONCLUSIVE')
        self.assertNotIn('instruments-lifecycle', pv)
        self.assertEqual((rows, session.calls, leaf.chats), ([], [], 0))
        grade = arts['native43-lifecycle-grade.json']
        self.assertEqual((grade['verdict'], grade['failures']), ('FAIL', ['engine-crash:engine-stream.log']))
        self.assertEqual(pv['native43-lifecycle'], 'FAIL')
        record_case('tail-engine-crash-withholds', calls=[], lifecycle_grade='FAIL')

    def test_observer_mapped_on_one_rank_withholds_the_block(self):
        done, pv, rows, arts, leaf, session, paths = self.run_tail(mapped_rank=1)
        self.assertEqual(pv['instruments-lifecycle'], 'FAIL')
        self.assertTrue(arts['instruments-lifecycle.json']['ranks']['1']['observer_mapped'])
        self.assertEqual(pv['lifecycle'], 'INCONCLUSIVE')
        self.assertIn('observer and offers off not proven (instruments-lifecycle)', done.stdout)
        self.assertNotIn('instruments-end', pv)
        self.assertEqual((rows, session.calls, leaf.chats), ([], ['pause', 'resume'], 0))
        record_case('tail-observer-mapped-withholds', calls=[], instruments='FAIL', session_calls=session.calls)

    def test_refused_full_hygiene_is_inconclusive_and_nothing_follows_it(self):
        done, pv, rows, arts, leaf, session, paths = self.run_tail(stuck=True)
        self.assertEqual((pv['instruments-lifecycle'], pv['lifecycle'], pv['hygiene-full']), ('PASS', 'PASS', 'FAIL'))
        self.assert_short_scope(done, pv, rows, leaf, paths)
        self.assertEqual(pv['instruments-end'], 'PASS')
        grade = arts['native43-lifecycle-grade.json']
        self.assertEqual((grade['verdict'], grade['missing']), ('INCONCLUSIVE', ['hygiene-full:typed-full-route']))
        record_case('tail-hygiene-refused-inconclusive', calls=[r['name'] for r in rows], hygiene='FAIL',
                    lifecycle_grade='INCONCLUSIVE')


class Grade(unittest.TestCase):
    """native43_lifecycle_grade over retained-record shapes: the native35 checks unchanged, leak/pressure NOT_RUN."""

    def write(self, q, lifecycle=True, hygiene=True, extra=()):
        if lifecycle:
            (q / 'lifecycle.json').write_text(json.dumps({'verdict': 'PASS', 'cancel_grade': {'verdict': 'PASS'},
                                                          'reuse': {'complete': True}}))
        if hygiene:
            (q / 'hygiene-full.json').write_text(json.dumps({'prepare': {'route': 'full'}, 'resume': {'resumed': True}}))
        for name in extra:
            (q / name).write_text('{}' if name.endswith('.json') else 'sessions 8 rounds 3 bad 0 in 1s\n')

    def test_the_native35_reading_minus_the_not_run_oracles_and_a_foreign_leak_record_fails(self):
        with tempfile.TemporaryDirectory() as d:
            q = Path(d)
            self.write(q)
            full = native43_lifecycle_grade.l.grade(q)
            self.assertEqual(full['verdict'], 'INCONCLUSIVE')
            self.assertEqual(sorted(full['missing']), sorted(['cross-session-leak:log-missing', 'partial-leak:record-missing',
                                                              'pressure-completed:record-missing',
                                                              'pressure-lifetime:record-missing']))
            short = native43_lifecycle_grade.grade(q)
            self.assertEqual((short['verdict'], short['failures'], short['missing']), ('PASS', [], []))
            self.assertEqual(short['schema'], 'native43-lifecycle-reading/1')
            (q / 'cross-session-leak.log').write_text('sessions 8 rounds 3 bad 0 in 1s\n')
            foreign = native43_lifecycle_grade.grade(q)
            self.assertEqual((foreign['verdict'], foreign['failures']),
                             ('FAIL', ['short-scope-foreign-record:cross-session-leak.log']))
        with tempfile.TemporaryDirectory() as d:
            q = Path(d)
            self.write(q, hygiene=False)
            self.assertEqual(native43_lifecycle_grade.grade(q)['missing'], ['hygiene-full:typed-full-route'])
            out = q / 'grade.json'
            self.assertEqual(native43_lifecycle_grade.main(['--queue', str(q), '--out', str(out)]), 2)
            self.assertEqual(json.loads(out.read_text())['verdict'], 'INCONCLUSIVE')
        record_case('grade-short-scope-reading', full_missing_not_run=4, foreign_record='FAIL')


class Workload(unittest.TestCase):
    """The lifecycle's W2 manifest through the real lifecycle CLI with the native43 queue's own argv lines."""

    def test_the_lifecycle_cli_accepts_the_trunk_manifest_and_reaches_its_first_control_call(self):
        draft = LIFECYCLE.read_text()
        argv = trunkq.segment(draft, '  python3 "$H/e3_q18a_lifecycle.py"', '--out "$Q/lifecycle.json" > "$Q/lifecycle.log" 2>&1\n')
        self.assertEqual(argv, trunkq.segment((RUNS / 'q18a_native39_lifecycle.draft.sh').read_text(),
                                              '  python3 "$H/e3_q18a_lifecycle.py"',
                                              '--out "$Q/lifecycle.json" > "$Q/lifecycle.log" 2>&1\n'))
        with tempfile.TemporaryDirectory() as d:
            import socket
            s = socket.socket()
            s.bind(('127.0.0.1', 0))
            port = s.getsockname()[1]
            s.close()
            q = Path(d)
            done = subprocess.run(['bash', '-c', argv + 'echo "RC=$?"'], capture_output=True, text=True, timeout=120,
                                  env=trunkq.env(H=str(HERE), Q=str(q), PROBE_URL='http://127.0.0.1:%d' % port, MODEL='m',
                                                 RUN_ID='q18a-native43-lifecycle-cpu-2', EPOCH='7', VLLM_API_KEY=KEY,
                                                 PYTHONPATH=os.pathsep.join((str(HERE), str(ROOT / 'patches/s5_engine')))))
            self.assertIn('RC=2', done.stdout, done.stderr)
            record = json.loads((q / 'lifecycle.json').read_text())
        self.assertEqual(record['verdict'], 'INCONCLUSIVE')
        self.assertNotIn('lifetime-inputs', record['missing'])
        self.assertTrue(any(m.startswith('native-lifetime-refused:') for m in record['missing']), record['missing'])
        record_case('workload-lifecycle-cli-native39-manifest', first_call_reached=True)


class Settings(unittest.TestCase):
    """Sourced native43 lifecycle settings -> real setcfg -> real S5 launcher render -> declared identity -> PY_DECL,
    the preflight env lines and the native43 engine's own policy parser, over the trunk's fixture leaf
    queue-declare-fixture-tree."""

    def py_decl(self, path):
        code = trunkq.segment(DECLARE.read_text(), "python3 - \"$R/startup.json\" \"$Q18A_TARGET_LABEL\" <<'PY_DECL'",
                              '\nPY_DECL\n')
        body = code.split('\n', 1)[1].rsplit('\nPY_DECL\n', 1)[0]
        return subprocess.run([sys.executable, '-I', '-c', body, str(path), 's5-overlap-80-80'], capture_output=True, text=True)

    def test_settings_declare_the_dense_policy_without_trace_stop_or_pool_telemetry(self):
        declared, path = trunkq.Launcher.declared(self, SETTINGS)
        env = declared['identity']['env']
        self.assertEqual({k: env.get(k) for k in ('FLASH_S5_REPLAY_OBSERVER', 'FLASH_S5_ENGINE_OFFERS', 'FLASH_NATIVE30_LOCALIZE',
                                                  'FLASH_S5_NATIVE_QSA_POLICY')},
                         {'FLASH_S5_REPLAY_OBSERVER': '0', 'FLASH_S5_ENGINE_OFFERS': '0', 'FLASH_NATIVE30_LOCALIZE': '0',
                          'FLASH_S5_NATIVE_QSA_POLICY': 'dense'})
        for absent in ('FLASH_S5_NATIVE_QSA_TRACE', 'FLASH_S5_NATIVE_QSA_STOP_AFTER', 'FLASH_POOL_STATS'):
            self.assertNotIn(absent, env)
        args = declared['identity']['command']['args']
        hosts = [args[j + 1] for j, x in enumerate(args[:-1]) if x == '--host']
        self.assertEqual(hosts, ['0.0.0.0', '127.0.0.1'])
        self.assertEqual(self.py_decl(path).returncode, 0)
        # Against Codex's dense arm: the two diagnostic entries only; against the box-run native39 lifecycle: the pool
        # telemetry delta out, the dense policy in.  The args are identical to both.
        def delta(other):
            o = other['identity']['env']
            return {k: (o.get(k), env.get(k)) for k in set(o) | set(env) if o.get(k) != env.get(k)}
        dense, dense_path = trunkq.Launcher.declared(self, DENSE_SETTINGS)
        base, base_path = trunkq.Launcher.declared(self, BASE_SETTINGS)
        self.assertEqual(delta(dense), {'FLASH_S5_NATIVE_QSA_TRACE': ('1', None),
                                        'FLASH_S5_NATIVE_QSA_STOP_AFTER': ('101200', None)})
        self.assertEqual(delta(base), {'FLASH_POOL_STATS': ('200', None), 'FLASH_S5_NATIVE_QSA_POLICY': (None, 'dense')})
        self.assertEqual(dense['identity']['command']['args'], args)
        self.assertEqual(base['identity']['command']['args'], args)
        # The declaration refuses the dense diagnostic arm (trace and stop) and the native39 lifecycle engine.
        refused = self.py_decl(dense_path)
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn('no trace and no stop frontier', refused.stderr)
        refused = self.py_decl(base_path)
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn('native43 lifecycle runs the candidate dense policy', refused.stderr)
        # The preflight's own env assertion lines (PY_PREFLIGHT) over the three declared envs.
        draft, drafts = LIFECYCLE.read_text(), drafts_tool()
        body = trunkq.segment(draft, "<<'PY_PREFLIGHT' > \"$Q/preflight.log\" 2>&1\n", '\nPY_PREFLIGHT\n')
        first = "assert env.get('FLASH_S5_REPLAY_OBSERVER') == '0'\n"
        self.assertEqual((body.count(first), body.count(drafts.PREFLIGHT)), (1, 1))
        lines = body[body.index(first):body.index(drafts.PREFLIGHT) + len(drafts.PREFLIGHT)]
        exec(lines, {'env': env})
        for other in (dense, base):
            with self.assertRaises(AssertionError):
                exec(lines, {'env': other['identity']['env']})
        # The native43 engine's own startup parser on the declared env: dense, trace off; the stop frontier default.
        sys.path.insert(0, str(ROOT / 'patches/s5_engine'))
        import qsa_operation
        self.assertEqual(qsa_operation.native_qsa_settings(env), ('dense', False))
        self.assertEqual(qsa_operation.native_qsa_settings(dense['identity']['env']), ('dense', True))
        driver = (ROOT / 'patches/s5_engine/engine_driver.py').read_text()
        self.assertEqual(driver.count("stop_at = os.environ.get('FLASH_S5_NATIVE_QSA_STOP_AFTER', '0')"), 1)
        record_case('settings-dense-policy-no-trace-no-stop', hosts=hosts,
                    env_delta_vs_dense_arm={'FLASH_S5_NATIVE_QSA_TRACE': None, 'FLASH_S5_NATIVE_QSA_STOP_AFTER': None},
                    env_delta_vs_native39_lifecycle={'FLASH_POOL_STATS': None, 'FLASH_S5_NATIVE_QSA_POLICY': 'dense'},
                    dense_arm_refused=True, native39_lifecycle_refused=True, engine_policy=['dense', False])


class Derivation(unittest.TestCase):
    def test_lifecycle_texts_are_the_declared_derivation_of_the_parent_texts(self):
        for name, digest in PARENT_SHA.items():
            self.assertEqual(hashlib.sha256((RUNS / name).read_bytes()).hexdigest(), digest, name)
        for path, text in drafts_tool().generate().items():
            self.assertEqual(path.read_text(), text, path.name)

    def test_undoing_each_declared_change_gives_the_box_run_texts(self):
        drafts = drafts_tool()
        for suffix in ('.draft.sh', '_settings.sh', '_declare.sh'):
            undo = (RUNS / ('q18a_native43_lifecycle' + suffix)).read_text()
            undo = undo.replace('q18a-native43-lifecycle', 'q18a-native39-lifecycle').replace('q18a_native43_lifecycle',
                                                                                            'q18a_native39_lifecycle')
            base = (RUNS / ('q18a_native39_lifecycle' + suffix)).read_text()
            if suffix == '.draft.sh':
                undo = undo.replace(drafts.HEADER, [l for l in base.splitlines(True)
                                                    if l.startswith(drafts.BASE_HEADER_PREFIX)][0])
                for old, new in ((drafts.BASE_TITLE, drafts.TITLE), (drafts.BASE_TIMEOUT, drafts.TIMEOUT),
                                 (drafts.BASE_SLOT_NOTE, drafts.SLOT_NOTE), (drafts.BASE_SLOT, drafts.SLOT),
                                 (drafts.BASE_INTERFACES, drafts.INTERFACES), (drafts.BASE_GRADE, drafts.GRADE),
                                 (drafts.BASE_PREFLIGHT, drafts.PREFLIGHT)):
                    self.assertEqual(undo.count(new), 1, new[:60])
                    undo = undo.replace(new, old)
                for begin, end, note in ((drafts.POOL_BEGIN, drafts.POOL_END, drafts.POOL_NOTE),
                                         (drafts.LEAK_BEGIN, drafts.LEAK_END, drafts.LEAK_NOTE)):
                    self.assertEqual(undo.count(note), 1)
                    cut = base[base.index(begin):base.index(end) + len(end)]
                    undo = undo.replace(note, cut)
            elif suffix == '_settings.sh':
                for old, new in ((drafts.BASE_SETTINGS_NOTE, drafts.SETTINGS_NOTE), (drafts.BASE_EXTRA_ENV, drafts.EXTRA_ENV)):
                    self.assertEqual(undo.count(new), 1)
                    undo = undo.replace(new, old)
            else:
                for old, new in ((drafts.BASE_DECL, drafts.DECL), (drafts.BASE_DECL_PINS, drafts.DECL_PINS)):
                    self.assertEqual(undo.count(new), 1)
                    undo = undo.replace(new, old)
            self.assertEqual(undo, base, suffix)

    def test_phase_names_are_the_box_run_lifecycle_minus_the_short_scope(self):
        def names(text):
            return set(re.findall(r'(?:^|[;\s])(?:record|step) "?([A-Za-z0-9_.$\-{}]+)"? ', text, re.M))
        base = names((RUNS / 'q18a_native39_lifecycle.draft.sh').read_text())
        life = names(LIFECYCLE.read_text())
        self.assertEqual(life - base, {'native43-lifecycle'})
        self.assertEqual(base - life, {'pool-telemetry', 'cross-session-leak', 'pressure-ready', 'partial-leak-driver',
                                       'partial-leak', 'pressure-completed', 'pressure-lifetime', 'native35-lifetime'})

    def test_every_leaf_this_suite_names_is_declared_with_its_test(self):
        declared = {l['name']: l for l in json.loads((HERE / 'native43_lifecycle_fixture_leaves.json').read_text())['leaves']}
        named = set(re.findall(r'[Ff]ixture leaf ((?:native43-lifecycle|lifecycle|queue)-[a-z0-9-]+)', Path(__file__).read_text()))
        self.assertEqual(named, set(declared))
        for name, leaf in declared.items():
            self.assertTrue(leaf['replaces'] and leaf['equivalence'] and leaf['cannot_prove'], name)
            self.assertIs(leaf['box_evidence'], False)
            tests = re.findall(r'test_native43_lifecycle_queue\.(\w+)', leaf['test'])
            self.assertTrue(tests, name)
            for test in tests:
                self.assertIn(test, globals(), name)

    def test_scripts_parse_and_name_only_the_native43_lifecycle(self):
        for path in (LIFECYCLE, SETTINGS, DECLARE):
            done = subprocess.run(['bash', '-n', str(path)], capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr)
            text = path.read_text()
            for gone in ('q18a-native39-lifecycle', 'q18a_native39_lifecycle', 'q18a-native39-trunk', 'q18a_native39_trunk',
                         'workloads/q18a-w2-overlap.json'):
                self.assertNotIn(gone, text)
        text = LIFECYCLE.read_text()
        self.assertIn('--kill-after=900s 5000s env', text)
        for gone in ('qualification_phase rule4', 'ROW4_DECLARED_NOT_RUN', 'e3_native39_round.py" --url', 'overlap-round-',
                     '"$H/leaktest.py"', '"$H/pressure2.py"', '"$H/leak_partial_tail.py"', 'PY_POOL_TELEMETRY'):
            self.assertNotIn(gone, text)


class Window(unittest.TestCase):
    def test_bounds_unchanged_and_the_derived_short_budget_fits(self):
        self.assertEqual((gate.NOT_BEFORE, gate.HARD_END), (native39_lifecycle_gate.NOT_BEFORE, native39_lifecycle_gate.HARD_END))
        self.assertEqual(gate.HARD_END, datetime.datetime(2026, 11, 2, 6, tzinfo=datetime.timezone.utc).timestamp())
        self.assertEqual((gate.LIFECYCLE_ACTIVE_S, gate.LIFECYCLE_CLEANUP_MARGIN_S, gate.LIFECYCLE_ACTIVE_AND_CLEANUP),
                         (5000, 900, 5900))
        parent_rows = {n: (s, why) for n, s, why in native39_lifecycle_gate.ACTIVE_DERIVATION}
        for name, seconds, why in gate.ACTIVE_DERIVATION:
            self.assertEqual((seconds, why), parent_rows[name], name)
        self.assertEqual([n for n, _, _ in gate.LIFECYCLE_DERIVATION],
                         ['lifetime-gates', 'instruments-x2', 'native-lifetime', 'hygiene-full'])
        self.assertEqual(set(parent_rows) - {n for n, _, _ in gate.ACTIVE_DERIVATION}, set(gate.DROPPED))
        self.assertEqual(sum(s for _, s, _ in gate.ACTIVE_DERIVATION), 4957)
        self.assertEqual(gate.CLEANUP_DERIVATION, native39_lifecycle_gate.CLEANUP_DERIVATION)
        self.assertFalse(gate.lifecycle_window_ok(gate.NOT_BEFORE - 1))
        self.assertTrue(gate.lifecycle_window_ok(gate.HARD_END - 5900))
        self.assertFalse(gate.lifecycle_window_ok(gate.HARD_END - 5900 + 1))
        self.assertEqual(gate.LATEST_ENTRY_UTC, '2026-11-02T04:21:40Z')

    def test_lifecycle_draft_window_slot_executes_at_real_wall_clock(self):
        source = LIFECYCLE.read_text()
        actual = trunkq.segment(source, "python3 - <<'PY_SLOT'", '\nPY_SLOT\n')
        self.assertIn('from native43_lifecycle_gate import lifecycle_window_ok as window_ok', actual)
        done = subprocess.run(['bash', '-c', 'fail_early() { return 2; }\n' + actual], env=trunkq.env(HARNESS_DIR=str(HERE)),
                              capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        SLOT_RECEIPTS.append(dict(draft=LIFECYCLE.name, verdict='PASS', rc=0,
                                  slot_sha256=hashlib.sha256(actual.encode()).hexdigest(), real_wall_clock=True,
                                  utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), engine_contact=False,
                                  active_s=gate.LIFECYCLE_ACTIVE_S, cleanup_margin_s=gate.LIFECYCLE_CLEANUP_MARGIN_S,
                                  latest_entry_utc=gate.LATEST_ENTRY_UTC))


if __name__ == '__main__':
    unittest.main()
