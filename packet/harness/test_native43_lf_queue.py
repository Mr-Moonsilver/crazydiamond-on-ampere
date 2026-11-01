"""Native43 l-f queue: exact derivation from the box-run l-e texts and the declared deltas under bash.

* Texts: tools/native43_lf_queue_drafts.py reproduces the committed q18a_native43_lifecycle_f texts; all three
  parse; normalised back (names, D9, D8, the le-fix deltas) they are the l-e texts byte for byte; the engine
  settings (EXTRA_ENV) are l-e's.
* D8 under bash: the l-e and l-f qualification_phase functions run with recording clients.  For row1 the l-f
  client is native43_row1_qualification.py and its argv is --qsa-policy dense followed by the l-e argv exactly;
  row4 and the other phases keep their l-e client and argv.  The l-f row1 client still runs under engine_watched:
  on the l-e teardown line a hung client is stopped within seconds and row1 records FAIL.
* Tail (D6/D7 connected): the l-f lifecycle region, the ending's stop_pressure and the on_exit native43 lifecycle
  grade under bash, as test_native43_lifecycle_queue.Tail runs the l-e region (same loopback engine leaf, session,
  docker CLI and record-writer lifecycle client), plus the real engine_watched and native43_engine_death watch
  around the lifecycle client and the typed FULL hygiene return.  Healthy: both pass through (CLIENT_EXIT) and the
  block PASSes as in l-e.  A hung lifecycle client on the l-e teardown line: stopped within seconds and the lifecycle
  step FAILs; the hygiene client that follows in the same block is stopped by its own watch before the typed FULL
  prepare reaches the engine, so that step FAILs too (queue FAILS=2, ENGINE_CRASHED=1); the unchanged lifecycle
  grade reads the missing records as INCONCLUSIVE (native-lifetime record, typed FULL route), never PASS.
* Window: the l-f draft's own slot check at real wall clock (native43_lifecycle_gate, l-e's window).
Leaves: harness/native43_le_fix_queue_fixture_leaves.json (bash function cut, recording clients, tool fakes) and
the leaves test_native43_lifecycle_queue declares for its Tail (native43_lifecycle_fixture_leaves.json).
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'tools'))
import native43_le_fix_queue_deltas as le_fix  # noqa: E402
import native43_lf_queue_drafts as lf  # noqa: E402
import test_native43_le_fix_queue_deltas as le_tests  # noqa: E402
import native43_lifecycle_gate as gate  # noqa: E402
import test_native39_lifecycle_queue as lq  # noqa: E402  helpers only
import test_native39_trunk_queue as trunkq  # noqa: E402  helpers only
import test_native43_lifecycle_queue as life  # noqa: E402  helpers only

RUNS = ROOT / 'harness/runs'
LF_DRAFT = RUNS / 'q18a_native43_lifecycle_f.draft.sh'
LF_SETTINGS = RUNS / 'q18a_native43_lifecycle_f_settings.sh'
# The l-e texts as held in bundle q18a-native43-lifecycle-bundle-296adc0886b6 and run on the box (1 Nov 2026).
PARENT_SHA = {'q18a_native43_lifecycle.draft.sh': 'b5a20401f677cae0cc48e5ab026e52c7aab9f456c4ebddd63d597cae7bda072c',
              'q18a_native43_lifecycle_declare.sh': 'b1daded8d3d61812f86d79515878cf439bf7112f3a39a5ee0b75715967187f9d',
              'q18a_native43_lifecycle_settings.sh': 'b01060b41d425e34f3543fc9bf6bbc34db8191af2d68da262da09e28761389f4'}
SLOT_RECEIPTS = []
QUEUE_RECEIPTS = []  # one row per connected case, appended only after its assertions (the bundle probe asserts the set)
HUNG_LIFECYCLE = 'import time\ntime.sleep(600)\n'
RECORDER = ('import json,sys,pathlib\n'
            'q=pathlib.Path(sys.argv[sys.argv.index("--out")+1]).parent\n'
            'name=pathlib.Path(sys.argv[0]).name\n'
            '(q/("argv-"+name+".json")).write_text(json.dumps(sys.argv[1:]))\n'
            'json.dump({"verdict":"PASS"},open(sys.argv[sys.argv.index("--out")+1],"w"))\n')


def extra_env(settings):
    return [line for line in settings.splitlines() if '"EXTRA_ENV=' in line]


class Texts(unittest.TestCase):
    def setUp(self):
        self.base = lf.base_texts()
        self.generated = lf.generate()

    def test_generator_reproduces_the_committed_texts(self):
        for path, text in self.generated.items():
            with self.subTest(path=path.name):
                self.assertEqual(path.read_text(), text)

    def test_all_three_parse(self):
        root = Path(tempfile.mkdtemp(prefix='n43-lf-'))
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        for path, text in self.generated.items():
            (root / path.name).write_text(text)
            self.assertEqual(subprocess.run(['bash', '-n', str(root / path.name)]).returncode, 0, path.name)

    def test_normalised_back_they_are_the_l_e_texts(self):
        texts = {kind: self.generated[RUNS / (lf.LF + suffix)]
                 for kind, suffix in (('draft', '.draft.sh'), ('declare', '_declare.sh'), ('settings', '_settings.sh'))}
        draft = lf.unnames(texts['draft'])
        for new, old in ((lf.HEADER_ANCHOR + lf.HEADER_NOTE, lf.HEADER_ANCHOR), (lf.D9_DRAFT_NEW, lf.D9_DRAFT_OLD),
                         (lf.D8_NEW, lf.D8_OLD)):
            self.assertEqual(draft.count(new), 1)
            draft = draft.replace(new, old)
        for label, old, new in reversed(le_fix.DRAFT_DELTAS):
            self.assertEqual(draft.count(new), 1, label)
            draft = draft.replace(new, old)
        self.assertEqual(draft, self.base['draft'])
        declare = lf.unnames(texts['declare']).replace(lf.D9_DECLARE_NEW, lf.D9_DECLARE_OLD)
        for label, old, new in le_fix.DECLARE_DELTAS:
            declare = declare.replace(new, old)
        self.assertEqual(declare, self.base['declare'])
        self.assertEqual(lf.unnames(texts['settings']), self.base['settings'])

    def test_names_settings_and_policy(self):
        draft, declare, settings = (self.generated[RUNS / (lf.LF + s)] for s in ('.draft.sh', '_declare.sh', '_settings.sh'))
        self.assertIn('QNAME=${QNAME:-q18a-native43-lifecycle-f}', settings)
        self.assertIn('QNAME=${QNAME:-q18a-native43-lifecycle-f}', declare)
        self.assertIn('RUN_ID="q18a-native43-lifecycle-f-$(date', draft)
        self.assertIn('source "$H/runs/q18a_native43_lifecycle_f_settings.sh"', draft)
        self.assertIn('--queue "$H/runs/q18a_native43_lifecycle_f.draft.sh"', declare)
        self.assertNotIn('q18a_native43_lifecycle.draft.sh', draft + declare + settings)
        self.assertNotIn('q18a_native43_lifecycle_settings.sh', draft + declare + settings)
        self.assertEqual(extra_env(settings), extra_env(self.base['settings']))
        self.assertEqual(len(extra_env(settings)), 1)
        self.assertIn('FLASH_S5_NATIVE_QSA_POLICY=dense', extra_env(settings)[0])
        self.assertIn("assert env.get('FLASH_S5_NATIVE_QSA_POLICY') == 'dense'\n", draft)
        self.assertIn('from native43_lifecycle_gate import lifecycle_window_ok as window_ok\n', draft)
        self.assertIn('native43_row1_qualification.py native43_row1_grade.py native43_engine_death.py', draft)
        self.assertIn('native43_engine_death.py native43_row1_qualification.py native43_row1_grade.py \\\n', declare)


class RowOneBash(unittest.TestCase):
    for _name in ('setUp', 'tool', 'stream', 'bash', 'le_lines', 'phase_env'):
        locals()[_name] = getattr(le_tests.Bash, _name)

    def texts(self):
        base = lf.base_texts()['draft']
        return base, lf.derive_draft(base)

    def phase(self, text, name, clients=('e3_q18a_qualification.py', 'e3_native35_p_c_placement.py',
                                         'native43_row1_qualification.py')):
        for client in clients:
            (self.h / client).write_text(RECORDER)
        for path in self.q.glob('argv-*.json'):
            path.unlink()
        for path in self.q.glob('qual-*.json'):
            path.unlink()
        for path in self.q.glob('engine-watch-*.json'):
            path.unlink()
        names = ('step', 'record') + (('engine_watched',) if 'engine_watched() {' in text else ()) + (
            'qualification_phase',)
        result, _ = self.bash(text, names, self.phase_env() + '\nqualification_phase ' + name)
        argv = {p.name[len('argv-'):-len('.json')]: json.loads(p.read_text()) for p in self.q.glob('argv-*.json')}
        return result, argv

    def test_row1_runs_the_policy_wrapper_with_the_l_e_argv(self):
        self.stream(self.le_lines(until=b'[shutdown] Executor'))
        base, fixed = self.texts()
        before, old = self.phase(base, 'row1')
        after, new = self.phase(fixed, 'row1')
        self.assertEqual(list(old), ['e3_q18a_qualification.py'])
        self.assertEqual(list(new), ['native43_row1_qualification.py'])
        self.assertEqual(new['native43_row1_qualification.py'][:2], ['--qsa-policy', 'dense'])
        self.assertEqual(new['native43_row1_qualification.py'][2:], old['e3_q18a_qualification.py'])
        for result in (before, after):
            self.assertIn('STEP row1 PASS raw qualification artifact retained; no repeat or picking', result.stdout)
        self.assertEqual(json.loads((self.q / 'engine-watch-qual-row1.json').read_text())['verdict'], 'CLIENT_EXIT')

    def test_other_phases_keep_their_client_and_argv(self):
        self.stream(self.le_lines(until=b'[shutdown] Executor'))
        base, fixed = self.texts()
        for name, client in (('row4', 'e3_native35_p_c_placement.py'), ('rule4_controls', 'e3_q18a_qualification.py')):
            with self.subTest(phase=name):
                _, old = self.phase(base, name)
                _, new = self.phase(fixed, name)
                self.assertEqual(list(old), [client])
                self.assertEqual(new, old)

    def test_hung_policy_row1_client_on_the_l_e_teardown_fails_within_seconds(self):
        lines = self.le_lines()
        cut = next(i for i, line in enumerate(lines) if b'[shutdown] Executor' in line)
        self.stream(lines[:cut] + [lines[cut]])
        _, fixed = self.texts()
        (self.h / 'native43_row1_qualification.py').write_text('import time\ntime.sleep(600)\n')
        names = ('step', 'record', 'engine_watched', 'qualification_phase')
        result, wall = self.bash(fixed, names, self.phase_env() + '\nqualification_phase row1')
        self.assertLess(wall, 40)
        self.assertIn('STEP row1 FAIL rc=1 raw qualification artifact retained; no repeat or picking', result.stdout)
        self.assertIn('CRASHED=1 SOURCE=engine-death-watch', result.stdout)
        watch = json.loads((self.q / 'engine-watch-qual-row1.json').read_text())
        self.assertEqual((watch['verdict'], watch['dead']['first_death']['wall']),
                         ('DEAD', '2026-11-01T11:03:08.286892993Z'))


class Tail(unittest.TestCase):
    def run_tail(self, *, hung_lifecycle=False):
        declared, _ = trunkq.Launcher.declared(self, LF_SETTINGS)
        draft = LF_DRAFT.read_text()
        base = life.base_drafts_tool()
        start, end = draft.index(base.LIFECYCLE_PREAMBLE), draft.index(base.ENDING)
        region = draft[start:end]
        self.assertEqual(region.count('engine_watched lifecycle python3 "$H/e3_q18a_lifecycle.py"'), 1)
        self.assertEqual(region.count('engine_watched hygiene-full python3 - '), 1)
        grade_lines = life.drafts_tool().GRADE
        self.assertEqual(draft.count(grade_lines), 1)
        health = [line for line in draft.splitlines(True) if line.startswith('health_ok() {')]
        self.assertEqual(len(health), 1)
        teardown = next(line for line in (le_tests.FIX / 'le-engine-stream.log').read_bytes().split(b'\n')
                        if b'[shutdown] Executor' in line)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            fake, h, q, bindir = d / 'fake', d / 'h', d / 'q', d / 'bin'
            for p in (fake, h, q, bindir, fake / 'maps', h / 'workloads'):
                p.mkdir()
            session = trunkq.NativeSession()
            with lq.LifecycleLeaf(session) as leaf:
                for name in ('native43_lifecycle_grade.py', 'native35_lifetime_grade.py', 'e3_native39_trunk_checks.py',
                             'native43_engine_death.py'):
                    shutil.copyfile(HERE / name, h / name)  # the real tools, run from the harness dir as on the box
                (h / 'e3_q18a_lifecycle.py').write_text(HUNG_LIFECYCLE if hung_lifecycle else life.FAKE_LIFECYCLE)
                shutil.copyfile(life.MANIFEST, h / 'workloads' / life.MANIFEST.name)
                (fake / 'current-id').write_text(trunkq.NEW_ID)
                (q / 'engine-stream.log').write_text(trunkq.BIND_LINE)
                (q / 'engine-stream.err').write_text('')
                (q / 'engine.identity-start.json').write_text(json.dumps({'identity': declared['identity']}))
                for rank in range(4):
                    (fake / 'maps' / str(1001 + rank)).write_text(trunkq.CLEAN_MAPS)
                (bindir / 'docker').write_text('#!' + sys.executable + '\n' + lq.FAKE_DOCKER)
                (bindir / 'docker').chmod(0o755)
                feeder = None
                if hung_lifecycle:  # the l-e teardown line arrives while the lifecycle client hangs
                    feeder = ('( until [[ -e $Q/lifecycle.started ]]; do sleep 0.2; done; sleep 1; '
                              'cat %s >> "$Q/engine-stream.log" ) &\n' % (d / 'teardown.line'))
                    (d / 'teardown.line').write_bytes(teardown + b'\n')
                    region = region.replace('  engine_watched lifecycle python3 "$H/e3_q18a_lifecycle.py"',
                                            '  touch "$Q/lifecycle.started"\n'
                                            '  engine_watched lifecycle python3 "$H/e3_q18a_lifecycle.py"', 1)
                script = ('set -uo pipefail\nFAILS=0; INCS=0; ENGINE_CRASHED=0; ENGINE_CRASH_SOURCE=\'\'; '
                          'PRESSURE_PID=\'\'; PRESSURE_STOP=\'\'\ndeclare -A PHASE_VERDICT=()\n'
                          + trunkq.functions(draft, ('step', 'record', 'stop_pressure', 'engine_watched')) + health[0]
                          + ''.join('PHASE_VERDICT[%s]=%s\n' % item for item in life.QUALIFIED.items())
                          + (feeder or '') + region + 'stop_pressure || true\n' + grade_lines
                          + 'for k in "${!PHASE_VERDICT[@]}"; do echo "PV $k=${PHASE_VERDICT[$k]}"; done\n'
                            'echo "TOTAL FAILS=$FAILS INCS=$INCS CRASHED=$ENGINE_CRASHED SOURCE=$ENGINE_CRASH_SOURCE"\n')
                started = time.monotonic()
                done = subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=400, cwd=str(d),
                                      env=trunkq.env(PATH=str(bindir) + os.pathsep + os.environ['PATH'], FAKE=str(fake),
                                                     H=str(h), Q=str(q), PROBE_URL=leaf.url, MODEL='m',
                                                     CONTAINER='qwen-flash-prod', VLLM_API_KEY=life.KEY,
                                                     RUN_ID='q18a-native43-lifecycle-f-cpu-1', EPOCH='7',
                                                     PYTHONPATH=os.pathsep.join((str(h), str(HERE),
                                                                                 str(ROOT / 'patches/s5_engine')))))
                wall = time.monotonic() - started
            arts = {name: json.loads((q / name).read_text()) for name in life.ARTS + (
                'engine-watch-lifecycle.json', 'engine-watch-hygiene-full.json') if (q / name).exists()}
            return done, trunkq.phases(done.stdout), arts, session, wall

    def test_connected_l_f_lifecycle_block_passes_through_the_engine_watch(self):
        done, pv, arts, session, _ = self.run_tail()
        self.assertEqual(done.returncode, 0, done.stderr)
        passed = ('instruments-lifecycle', 'lifecycle', 'hygiene-full', 'instruments-end', 'native43-lifecycle')
        self.assertEqual({k: pv.get(k) for k in passed}, dict.fromkeys(passed, 'PASS'), done.stdout)
        for name in ('lifecycle', 'hygiene-full'):
            watch = arts['engine-watch-%s.json' % name]
            self.assertEqual((watch['verdict'], watch['client_rc']), ('CLIENT_EXIT', 0), name)
        self.assertEqual(session.calls, ['pause', 'resume', 'prepare', 'resume', 'pause', 'resume'])
        hygiene = arts['hygiene-full.json']
        self.assertEqual((hygiene['prepare']['route'], hygiene['prepare']['typed_acks'], hygiene['resume']['resumed']),
                         ('full', True, True))
        self.assertEqual(arts['native43-lifecycle-grade.json']['verdict'], 'PASS')
        self.assertIn('TOTAL FAILS=0 INCS=0 CRASHED=0 SOURCE=', done.stdout)
        QUEUE_RECEIPTS.append(dict(case='lf-tail-watched-pass', verdict='PASS', lifecycle_grade='PASS',
                                   watches=['CLIENT_EXIT', 'CLIENT_EXIT'], session_calls=session.calls))

    def test_hung_lifecycle_client_on_the_l_e_teardown_fails_and_the_full_return_never_reaches_the_engine(self):
        done, pv, arts, session, wall = self.run_tail(hung_lifecycle=True)
        self.assertLess(wall, 60)
        self.assertEqual((pv.get('lifecycle'), pv.get('hygiene-full')), ('FAIL', 'FAIL'), done.stdout)
        for name in ('lifecycle', 'hygiene-full'):
            watch = arts['engine-watch-%s.json' % name]
            self.assertEqual((watch['verdict'], watch['dead']['first_death']['wall']),
                             ('DEAD', '2026-11-01T11:03:08.286892993Z'), name)
        self.assertNotIn('prepare', session.calls)  # the typed FULL prepare was never sent to the dead engine
        self.assertNotIn('hygiene-full.json', arts)
        self.assertIn('TOTAL FAILS=2 INCS=1 CRASHED=1 SOURCE=engine-death-watch', done.stdout)
        grade = arts['native43-lifecycle-grade.json']
        self.assertEqual((grade['verdict'], grade['failures'], grade['missing']),
                         ('INCONCLUSIVE', [], ['native-lifetime:record-missing', 'hygiene-full:typed-full-route']))
        QUEUE_RECEIPTS.append(dict(case='lf-tail-hung-lifecycle-dead', verdict='PASS', lifecycle_grade='INCONCLUSIVE',
                                   queue_fails=2, watches=['DEAD', 'DEAD'], session_calls=session.calls,
                                   wall_s=round(wall, 3)))


class Window(unittest.TestCase):
    def test_parent_texts_are_the_box_run_l_e_bytes(self):
        for name, pin in PARENT_SHA.items():
            self.assertEqual(hashlib.sha256((RUNS / name).read_bytes()).hexdigest(), pin, name)

    def test_l_f_draft_window_slot_executes_at_real_wall_clock(self):
        source = LF_DRAFT.read_text()
        actual = trunkq.segment(source, "python3 - <<'PY_SLOT'", '\nPY_SLOT\n')
        self.assertIn('from native43_lifecycle_gate import lifecycle_window_ok as window_ok', actual)
        self.assertIn('--kill-after=900s 5000s env', source)
        done = subprocess.run(['bash', '-c', 'fail_early() { return 2; }\n' + actual], env=trunkq.env(HARNESS_DIR=str(HERE)),
                              capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        SLOT_RECEIPTS.append(dict(draft=LF_DRAFT.name, verdict='PASS', rc=0,
                                  slot_sha256=hashlib.sha256(actual.encode()).hexdigest(), real_wall_clock=True,
                                  utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), engine_contact=False,
                                  active_s=gate.LIFECYCLE_ACTIVE_S, cleanup_margin_s=gate.LIFECYCLE_CLEANUP_MARGIN_S,
                                  latest_entry_utc=gate.LATEST_ENTRY_UTC))


if __name__ == '__main__':
    unittest.main()
