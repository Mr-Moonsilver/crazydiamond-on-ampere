"""tools/box_admission_check.py and tools/box-watch-haiku-b.sh: the no-native-admission rule. CPU only, no box contact.

Committed records, read only:
  q18a-alpha-next-c1b       every arm served stock (its numerics fresh carried logprobs): NO_NATIVE_ADMISSION, first arm
                            c0off.g2-solo-0.
  q18a-alpha-next-c1        native engaged on all 20 arms: ADMITTED; its stock control (unarmed-stock-prefill.json) is no
                            arm record.
  q18a-native41-autoarm-run-2  kind 'stock' control arms beside kind 'auto' arms; records in .json.xz.
  q18a-returning-append     the bulk family: bulk true arms, nobulk controls.
  q18a-append-yield         no arm ran (no arm record): UNKNOWN.
Variants of real records (missing fields, partial write, admission seen only in the counter) live in a temp dir.

The watcher runs with BW_TEST_PVE pointing at a fake PVE command (snapshots from files, the check run locally on the
record dir), a stub 'claude' and a trap 'ssh' first in PATH (any ssh call fails the test), and HOME in a temp dir.

Mutation runs: ADMISSION_CHECK_UNDER_TEST / ADMISSION_WATCHER_UNDER_TEST point the suite at a mutant copy.
"""
import copy
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOL = Path(os.environ.get('ADMISSION_CHECK_UNDER_TEST') or ROOT / 'tools' / 'box_admission_check.py')
WATCHER = Path(os.environ.get('ADMISSION_WATCHER_UNDER_TEST') or ROOT / 'tools' / 'box-watch-haiku-b.sh')
BOX = ROOT / 'results' / 'box'
C1B = BOX / 'q18a-alpha-next-c1b' / 'box' / 'run' / 'q18a-alpha-next-c1b'
C1 = BOX / 'q18a-alpha-next-c1' / 'box' / 'run' / 'q18a-alpha-next-c1'
AUTO = BOX / 'q18a-native41-autoarm-run-2' / 'box' / 'n41e' / 'q18a-native41-autoarm'
RA = BOX / 'q18a-returning-append' / 'box' / 'run' / 'q18a-returning-append'
AY = BOX / 'q18a-append-yield' / 'box' / 'run' / 'q18a-append-yield'
C1B_ARMS = ('c0off.g2-solo-0', 'c0off.g2-solo-1', 'c1off.g2-solo-0', 'c1off.g2-solo-1', 'c1defer.g2-solo-0',
            'c0off.g2-solo-3')
C1B_ALERT = 'ALERT: no-native-admission c0off.g2-solo-0 (admitted 0, split 0/3 samples)'


def _load_tool():
    spec = importlib.util.spec_from_file_location('box_admission_check_under_test', str(TOOL))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bac = _load_tool()


def _tree_state(path):
    return sorted((str(p.relative_to(path)), p.stat().st_size, p.stat().st_mtime_ns) for p in Path(path).rglob('*'))


def _arm(path):
    return json.loads(Path(path).read_text())


class RecordVerdicts(unittest.TestCase):
    """The committed records."""

    def test_c1b_no_native_admission_on_its_first_finished_native_arm(self):
        before = _tree_state(C1B)
        out = bac.check(str(C1B))
        self.assertEqual(_tree_state(C1B), before)  # read only
        self.assertEqual(out['verdict'], bac.NO_NATIVE_ADMISSION)
        self.assertEqual(out['first_no_admission']['label'], 'c0off.g2-solo-0')
        self.assertEqual([a['label'] for a in out['arms']], list(C1B_ARMS))  # finish order (wall_end_s)
        for arm in out['arms']:
            self.assertEqual((arm['role'], arm['signal'], arm['verdict']), ('native', 'kind=auto', bac.NO_NATIVE_ADMISSION))
            self.assertEqual((arm['admitted'], arm['split_or_native_samples'], arm['progress_chunks']), (0, 0, 0))
            self.assertEqual((arm['admitted_receipt'], arm['admitted_polled']), (0, 0))
            self.assertEqual(arm['samples'], 3)
            self.assertGreater(arm['polls'], 250)
            self.assertIs(arm['native_request_served'], True)
        self.assertEqual(out['counts']['no_native_admission'], 6)
        self.assertEqual(bac.alert_lines(out), [C1B_ALERT, 'relaunch-skip: ' + ','.join(sorted(C1B_ARMS))])

    def test_c1b_alert_already_after_the_first_arm(self):
        # The queue writes one arm record when that arm ends: with only the first record present the alert is the same.
        with tempfile.TemporaryDirectory() as tmp:
            q = Path(tmp) / 'q18a-alpha-next-c1b'
            q.mkdir()
            for name in ('arm-c0off.g2-solo-0.json', 'arm-c0off.g2-solo-0.json.base', 'arm-c0off.g2-solo-0.log',
                         'entry-c0off.g2-solo-1.json'):
                shutil.copy(C1B / name, q / name)
            out = bac.check(str(q))
            self.assertEqual(out['verdict'], bac.NO_NATIVE_ADMISSION)
            self.assertEqual(out['counts']['arm_records'], 1)  # .json.base is not an arm record
            self.assertEqual(bac.alert_lines(out)[0], C1B_ALERT)

    def test_c1_admitted_on_every_native_arm(self):
        out = bac.check(str(C1))
        self.assertEqual(out['verdict'], bac.ADMITTED)
        self.assertIsNone(bac.alert_lines(out))
        self.assertEqual(out['counts'], dict(arm_records=20, native=20, control=0, undeclared_or_unreadable=0,
                                             admitted=20, no_native_admission=0, unknown=0, skipped=0))
        by = {a['label']: a for a in out['arms']}
        d = by['c1defer.g2-solo-1']  # the lead's contrast arm
        self.assertEqual((d['verdict'], d['admitted'], d['split_or_native_samples'], d['samples'], d['progress_chunks']),
                         (bac.ADMITTED, 66, 64, 67, 66))
        for arm in out['arms']:
            self.assertGreater(arm['admitted'], 0)
            self.assertGreater(arm['split_or_native_samples'], 0)
        # The queue's stock control is its unarmed-stock-prefill step, not an arm record: never judged.
        self.assertTrue((C1 / 'unarmed-stock-prefill.json').is_file())
        self.assertNotIn('unarmed-stock-prefill', by)

    def test_stock_control_arms_never_alert(self):
        out = bac.check(str(AUTO))  # .json.xz auto arms, plain .json stock arms
        roles = {a['label']: (a['role'], a['signal'], a['verdict']) for a in out['arms']}
        self.assertEqual(roles, {'auto-0': ('native', 'kind=auto', bac.ADMITTED), 'auto-1': ('native', 'kind=auto', bac.ADMITTED),
                                 'stock-0': ('control', 'kind=stock', None), 'stock-1': ('control', 'kind=stock', None)})
        stock = [a for a in out['arms'] if a['role'] == 'control']
        self.assertTrue(all(a['admitted'] == 0 and a['split_or_native_samples'] == 0 for a in stock))
        with tempfile.TemporaryDirectory() as tmp:  # a queue with only its stock controls finished so far
            q = Path(tmp) / 'q'
            q.mkdir()
            for name in ('arm-stock-0.json', 'arm-stock-1.json'):
                shutil.copy(AUTO / name, q / name)
            only = bac.check(str(q))
            self.assertEqual((only['verdict'], only['counts']['control'], only['counts']['native']), (bac.UNKNOWN, 2, 0))
            self.assertIsNone(bac.alert_lines(only))

    def test_bulk_family_native_and_nobulk_controls(self):
        out = bac.check(str(RA))
        self.assertEqual(out['verdict'], bac.ADMITTED)
        for arm in out['arms']:
            if arm['label'].startswith('nobulk-'):
                self.assertEqual((arm['role'], arm['signal'], arm['verdict']), ('control', 'bulk=false', None))
                self.assertEqual(arm['split_or_native_samples'], 0)
            else:
                self.assertEqual((arm['role'], arm['signal'], arm['verdict']), ('native', 'bulk=true', bac.ADMITTED))
                self.assertEqual(arm['admitted'], 41)
                self.assertIsNone(arm['progress_chunks'])  # not recorded in this family

    def test_append_yield_no_arm_ran_is_unknown(self):
        out = bac.check(str(AY))
        self.assertEqual((out['verdict'], out['reasons'], out['arms']), (bac.UNKNOWN, ['no-arm-record'], []))
        self.assertIsNone(bac.alert_lines(out))

    def test_sweep_subdirs_are_scanned(self):
        out = bac.check(str(BOX / 'q18a-stage1-sweep' / 'box' / 'sweep' / 'q18a-stage1-sweep'))
        self.assertEqual(out['verdict'], bac.ADMITTED)
        self.assertEqual(out['counts']['native'], 12)
        self.assertTrue(all('/' in a['label'] for a in out['arms']))


class Variants(unittest.TestCase):
    """Real c1b / c1 arm records, changed in a temp dir."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.q = Path(self.tmp.name) / 'q18a-variant'
        self.q.mkdir()
        self.base = _arm(C1B / 'arm-c0off.g2-solo-0.json')

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, record, label='c0off.g2-solo-0', text=None):
        (self.q / ('arm-%s.json' % label)).write_text(text if text is not None else json.dumps(record))

    def one(self):
        out = bac.check(str(self.q))
        self.assertEqual(len(out['arms']), 1)
        return out, out['arms'][0]

    def assert_unknown(self, reason):
        out, arm = self.one()
        self.assertEqual((out['verdict'], arm['verdict']), (bac.UNKNOWN, bac.UNKNOWN))
        self.assertIn(reason, arm['reasons'])
        self.assertIsNone(bac.alert_lines(out))

    def test_unchanged_copy_is_no_native_admission(self):
        self.write(self.base)
        out, arm = self.one()
        self.assertEqual(arm['verdict'], bac.NO_NATIVE_ADMISSION)

    # -- missing data: UNKNOWN, never a guess -------------------------------------------------------------------------
    def test_missing_queue_dir(self):
        out = bac.check(str(self.q / 'not-there'))
        self.assertEqual((out['verdict'], out['reasons']), (bac.UNKNOWN, ['queue-dir-missing']))
        self.assertIsNone(bac.alert_lines(out))

    def test_admitted_not_recorded(self):
        r = copy.deepcopy(self.base)
        del r['receipts']['admitted_delta']
        for s in r['polled']['samples'] + r['statuses']:
            s.pop('admitted', None)
        self.write(r)
        self.assert_unknown('admitted-not-recorded')

    def test_no_polled_samples(self):
        r = copy.deepcopy(self.base)
        del r['polled']
        self.write(r)
        self.assert_unknown('no-polled-samples')

    def test_partial_write(self):
        self.write(None, text=(C1B / 'arm-c0off.g2-solo-0.json').read_text()[:5000])
        self.assert_unknown('arm-record-unreadable-or-partial')

    def test_not_declared(self):
        r = copy.deepcopy(self.base)
        del r['kind']
        self.write(r)
        self.assert_unknown('native-or-control-not-declared')

    def test_fresh_not_served(self):
        r = copy.deepcopy(self.base)
        r['fresh'] = dict(r['fresh'], error='ReadTimeout', first_at_ns=None, output_completion=0)
        self.write(r)
        self.assert_unknown('native-request-not-served')

    def test_arm_error(self):
        r = copy.deepcopy(self.base)
        r['reasons'] = list(r['reasons']) + ['arm-error']
        self.write(r)
        self.assert_unknown('arm-error')

    def test_not_finished(self):
        r = copy.deepcopy(self.base)
        del r['wall_end_s']
        self.write(r)
        self.assert_unknown('arm-not-finished')

    # -- each engagement signal alone is enough ------------------------------------------------------------------------
    def test_admission_in_the_receipt_only(self):
        r = copy.deepcopy(self.base)
        r['receipts']['admitted_delta'] = 66
        self.write(r)
        out, arm = self.one()
        self.assertEqual((out['verdict'], arm['verdict'], arm['admitted'], arm['split_or_native_samples']),
                         (bac.ADMITTED, bac.ADMITTED, 66, 0))

    def test_admission_in_the_polled_counter_only(self):
        r = copy.deepcopy(self.base)
        del r['receipts']['admitted_delta']
        r['polled']['samples'][-1]['admitted'] = 66  # the cumulative counter rose; no sample caught route split
        self.write(r)
        out, arm = self.one()
        self.assertEqual((arm['verdict'], arm['admitted'], arm['admitted_polled'], arm['split_or_native_samples']),
                         (bac.ADMITTED, 66, 66, 0))

    def test_split_sample_only(self):
        r = copy.deepcopy(self.base)
        r['polled']['samples'][1]['route'] = 'split'
        self.write(r)
        self.assertEqual(self.one()[1]['verdict'], bac.ADMITTED)

    def test_native_phase_sample_only(self):
        r = copy.deepcopy(self.base)
        r['polled']['samples'][1]['auto_arm']['phase'] = 'native'
        self.write(r)
        self.assertEqual(self.one()[1]['verdict'], bac.ADMITTED)

    def test_progress_chunks_only(self):
        r = copy.deepcopy(self.base)
        r['progress_delta']['chunks'] = 66
        self.write(r)
        self.assertEqual(self.one()[1]['verdict'], bac.ADMITTED)

    def test_cumulative_counter_baseline_is_not_an_admission(self):
        # c1's counter stands at 594 when its c1defer.g2-solo-1 arm starts; a flat 594 is 0 admitted in that arm.
        r = copy.deepcopy(self.base)
        for s in r['polled']['samples'] + r['statuses']:
            s['admitted'] = 594
        self.write(r)
        out, arm = self.one()
        self.assertEqual((arm['verdict'], arm['admitted'], arm['admitted_polled']), (bac.NO_NATIVE_ADMISSION, 0, 0))

    def test_skip_moves_the_first_alert(self):
        for label in C1B_ARMS[:2]:
            shutil.copy(C1B / ('arm-%s.json' % label), self.q / ('arm-%s.json' % label))
        out = bac.check(str(self.q), skip=['c0off.g2-solo-0'])
        self.assertEqual(out['first_no_admission']['label'], 'c0off.g2-solo-1')
        self.assertEqual(bac.alert_lines(out)[1], 'relaunch-skip: c0off.g2-solo-0,c0off.g2-solo-1')
        out = bac.check(str(self.q), skip=list(C1B_ARMS[:2]))
        self.assertEqual(out['verdict'], bac.UNKNOWN)
        self.assertIsNone(bac.alert_lines(out))


class CommandLine(unittest.TestCase):

    def run_tool(self, args, stdin=None):
        return subprocess.run([sys.executable, '-I'] + args, input=stdin, capture_output=True, text=True, timeout=120)

    def test_check_from_stdin_as_on_the_box(self):
        p = self.run_tool(['-', 'check', str(C1B)], stdin=TOOL.read_text())
        self.assertEqual(p.returncode, 0, p.stderr)
        lines = p.stdout.splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0])['verdict'], bac.NO_NATIVE_ADMISSION)
        self.assertTrue(lines[1].startswith('box-admission NO_NATIVE_ADMISSION q18a-alpha-next-c1b: native arms 6'), lines[1])
        a = self.run_tool([str(TOOL), 'alert', '-'], stdin=lines[0] + '\n')
        self.assertEqual(a.returncode, 2)
        self.assertEqual(a.stdout.splitlines()[0], C1B_ALERT)

    def test_alert_quiet_on_admitted_and_refuses_garbage(self):
        p = self.run_tool([str(TOOL), 'check', str(C1)])
        a = self.run_tool([str(TOOL), 'alert', '-'], stdin=p.stdout)
        self.assertEqual((a.returncode, a.stdout), (0, ''))
        g = self.run_tool([str(TOOL), 'alert', '-'], stdin='not json\n')
        self.assertEqual((g.returncode, g.stdout), (3, ''))

    def test_out_is_created_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'check.json'
            p = self.run_tool([str(TOOL), 'check', str(C1B), '--out', str(out)])
            self.assertEqual(p.returncode, 0)
            self.assertEqual(json.loads(out.read_text())['verdict'], bac.NO_NATIVE_ADMISSION)
            again = self.run_tool([str(TOOL), 'check', str(C1B), '--out', str(out)])
            self.assertNotEqual(again.returncode, 0)

    def test_bad_skip_is_refused(self):
        p = self.run_tool([str(TOOL), 'check', str(C1B), '--skip', 'a;rm'])
        self.assertEqual(p.returncode, 2)


FAKE_PVE = r'''#!/usr/bin/env bash
# Fake PVE command for box-watch-haiku-b tests: no box. Snapshots from $FAKE/snap-<n>.txt (else snap-last.txt); the
# admission check runs locally on $FAKE_S0/<QNAME> with the watcher's stdin (the check source).
cmd="$*"
case $cmd in
  'pct exec <ct-id> -- bash -s')
    cat > /dev/null
    n=$(cat "$FAKE/polls" 2>/dev/null || echo 0); n=$((n+1)); echo "$n" > "$FAKE/polls"
    f="$FAKE/snap-$n.txt"; [ -f "$f" ] || f="$FAKE/snap-last.txt"; cat "$f" ;;
  'pct exec <ct-id> -- python3 -I - check /home/<service-user>/s0/'*)
    echo "$cmd" >> "$FAKE/check-calls"
    if [ "${FAKE_CHECK:-}" = fail ]; then cat > /dev/null; echo "Traceback: fake failure"; exit 1; fi
    rest=${cmd#'pct exec <ct-id> -- python3 -I - check /home/<service-user>/s0/'}
    exec python3 -I - check "$FAKE_S0/"$rest ;;
  *) echo "fake-pve: unexpected: $cmd" >&2; exit 97 ;;
esac
'''

SNAP_RUNNING = '''time 2026-11-03T17:20:00Z
engine_unit active
gpu_mib_util 22000,40;22000,40;22000,40;22000,40;
owner_proc running
QFILE /home/<service-user>/s0/{q}.log
queue_log {q}.log lines 40 age_s 5
queue_last_steps:
STEP entry-c0off.g2-solo-1 PASS typed all-rank FULL, resumed, idle, workers empty; lease taken
QLINES 40
EFILE /home/<service-user>/s0/{q}/engine-stream.log
engine_log {q} lines 900 age_s 2
engine_new_lines 10 warnings 0 errors 0
engine_hard:
engine_tail:
INFO flash-sched stats step 1
ELINES 900
'''

SNAP_DONE = '''time 2026-11-03T17:31:10Z
engine_unit inactive
gpu_mib_util 0,0;0,0;0,0;0,0;
owner_proc gone
QFILE /home/<service-user>/s0/{q}.log
queue_log {q}.log lines 90 age_s 3
queue_last_steps:
Q18A-OVERLAP DONE rc=1 scope=real-e3-overlap engine-state=inactive
QLINES 90
ELINES 900
'''

SNAP_STAGING = '''time 2026-11-03T17:01:30Z
engine_unit inactive
gpu_mib_util 0,0;0,0;0,0;0,0;
owner_proc gone
queue_log missing
QLINES 0
engine_log not yet present
ELINES 0
'''

SNAP_FAULT = SNAP_RUNNING.replace('engine_hard:\n', 'engine_hard:\n      1 (Worker pid=9) ERROR CUDA error: an illegal memory access was encountered\n')


class Watcher(unittest.TestCase):
    """box-watch-haiku-b.sh end to end against local records; any real ssh fails the test."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.fake, self.bin, self.home = t / 'fake', t / 'bin', t / 'home'
        for d in (self.fake, self.bin, self.home):
            d.mkdir()
        pve = t / 'fake-pve'
        pve.write_text(FAKE_PVE)
        self.pve = pve
        (self.bin / 'claude').write_text('#!/usr/bin/env bash\necho call >> "%s/claude-calls"\necho "OK: stub"\n' % self.fake)
        (self.bin / 'ssh').write_text('#!/usr/bin/env bash\necho "$*" >> "%s/SSH-CALLED"\nexit 255\n' % self.fake)
        for f in (pve, self.bin / 'claude', self.bin / 'ssh'):
            f.chmod(0o755)

    def tearDown(self):
        self.tmp.cleanup()

    def run_watch(self, q, s0, snaps, env_extra=None, watcher=None):
        for i, snap in enumerate(snaps, 1):
            (self.fake / ('snap-%d.txt' % i)).write_text(snap.format(q=q))
        (self.fake / 'snap-last.txt').write_text(snaps[-1].format(q=q))
        env = dict(os.environ, HOME=str(self.home), PATH='%s:%s' % (self.bin, os.environ.get('PATH', '/usr/bin:/bin')),
                   BW_TEST_PVE=str(self.pve), FAKE=str(self.fake), FAKE_S0=str(s0))
        env.pop('BW_ADMISSION_SKIP', None)
        env.update(env_extra or {})
        p = subprocess.run(['bash', str(watcher or WATCHER), q, '1', '0'], env=env, capture_output=True, text=True,
                           timeout=180)
        self.assertFalse((self.fake / 'SSH-CALLED').exists(), 'the watcher called ssh')
        logs = list((self.home / '.local' / 'state' / 'box-watch').glob(q + '-*/watch.log'))
        return p, (logs[0].read_text() if logs else '')

    def calls(self, name):
        f = self.fake / name
        return f.read_text().splitlines() if f.exists() else []

    def test_c1b_alerts_on_the_first_poll_after_its_first_arm(self):
        p, log = self.run_watch('q18a-alpha-next-c1b', C1B.parent, [SNAP_RUNNING, SNAP_DONE])
        self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
        out = p.stdout.splitlines()
        self.assertEqual(out[0], C1B_ALERT)
        self.assertTrue(out[1].startswith('relaunch-skip: ' + ','.join(sorted(C1B_ARMS))), out[1])
        self.assertEqual(self.calls('check-calls'), ['pct exec <ct-id> -- python3 -I - check /home/<service-user>/s0/q18a-alpha-next-c1b'])
        self.assertEqual(self.calls('polls'), ['1'])
        self.assertEqual(self.calls('claude-calls'), [])  # the fixed rule fires before Haiku
        self.assertIn('ADMISSION box-admission NO_NATIVE_ADMISSION', log)

    def test_c1_no_alert_then_done(self):
        p, log = self.run_watch('q18a-alpha-next-c1', C1.parent, [SNAP_RUNNING, SNAP_DONE])
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertTrue(p.stdout.startswith('DONE q18a-alpha-next-c1: Q18A-OVERLAP DONE'), p.stdout)
        self.assertNotIn('no-native-admission', p.stdout)
        self.assertEqual(len(self.calls('check-calls')), 1)
        self.assertEqual(len(self.calls('claude-calls')), 1)
        self.assertIn('ADMISSION box-admission ADMITTED q18a-alpha-next-c1: native arms 20', log)

    def test_relaunch_skip_clears_the_alert(self):
        skip = ','.join(sorted(C1B_ARMS))
        p, log = self.run_watch('q18a-alpha-next-c1b', C1B.parent, [SNAP_RUNNING, SNAP_DONE], {'BW_ADMISSION_SKIP': skip})
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertEqual(self.calls('check-calls'),
                         ['pct exec <ct-id> -- python3 -I - check /home/<service-user>/s0/q18a-alpha-next-c1b --skip ' + skip])
        self.assertIn('ADMISSION box-admission UNKNOWN', log)

    def test_no_check_before_the_owner_is_seen(self):
        p, log = self.run_watch('q18a-alpha-next-c1b', C1B.parent, [SNAP_STAGING, SNAP_DONE])
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertEqual(self.calls('check-calls'), [])

    def test_failed_check_is_logged_never_alerts(self):
        p, log = self.run_watch('q18a-alpha-next-c1b', C1B.parent, [SNAP_RUNNING, SNAP_DONE], {'FAKE_CHECK': 'fail'})
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn('ADMISSION_CHECK_FAIL rc=1', log)

    def test_existing_fault_rule_kept(self):
        p, _ = self.run_watch('q18a-alpha-next-c1', C1.parent, [SNAP_FAULT, SNAP_DONE])
        self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
        self.assertTrue(p.stdout.startswith('ALERT q18a-alpha-next-c1 engine fault marker:'), p.stdout)

    def test_bad_skip_is_usage(self):
        p, _ = self.run_watch('q18a-alpha-next-c1b', C1B.parent, [SNAP_RUNNING], {'BW_ADMISSION_SKIP': 'a b'})
        self.assertEqual((p.returncode, p.stdout.strip()), (4, 'USAGE bad BW_ADMISSION_SKIP'))

    def test_old_watcher_unchanged_and_superseded_by_name(self):
        old = (ROOT / 'tools' / 'box-watch-haiku.sh').read_text()
        self.assertNotIn('admission', old)
        new = WATCHER.read_text()
        self.assertIn('supersedes box-watch-haiku.sh by name', new)


if __name__ == '__main__':
    unittest.main()
