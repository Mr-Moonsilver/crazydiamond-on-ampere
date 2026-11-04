#!/usr/bin/env python3
"""q18a-inwait queue: the derivation, the preflight name checks executed with the runner's real QNAME, the budget, and the derived arm loop under real bash (stub arm and
check clients, the real watch functions, the real grade and read). CPU only; no pass here is box evidence.

Classes:
* Derivation: tools/inwait_queue.py --check (the files are the derivation); c1's texts unchanged (its --check); every declared delta present; the draft's body is the derived
  body; the switch in EXTRA_ENV and asserted in preflight and declaration; the header states numerics are not covered; no logprobs in the texts' arm call.
* NameChecks: every QNAME test in the three runs/ scripts, extracted and executed in bash with QNAME set to the settings' real default (the runner's name, q18a-inwait), passes;
  the declaration's request dir, the draft's RUN_ID and every ``$H/runs/...`` and ``python3 "$H/X.py"`` name resolve to files in this tree; nothing executable names c1's queue.
* Budget: the gate's limits are relative (no clock import), every arm enters at the walls and at 1.5x the walls (a stated property of the 7,500 s budget).
* ArmLoop: the derived body: the exact-path smoke (its failure stops the table), the decode-only baselines, then the core arms in table order with the selection and the band's workload, the core grade, no tail, the read. An off-arm leak in a record makes the core grade FAIL and the queue count it.
* QueueH: the health-scoped death scan (the lead's rule, byte for byte as returning-append's forwarded files): the draft calls _h, never the old.
* QueueArgparse: every python call of the texts parses with its real parser; unknown or undeclared options are refused (controls).
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / 'tools')]
import test_alpha_next_c1_queue as tq  # noqa: E402  (first: paths)
import inwait_arms as T  # noqa: E402
import inwait_gate as gate  # noqa: E402
import inwait_queue as queue  # noqa: E402
import alpha_next_c1_queue as c1q  # noqa: E402
import test_inwait_box as tbox  # noqa: E402
import stage1_step_queue as sq  # noqa: E402

tap = tq.tap
tan = tq.tan
# The pack's probe derive reads both (c1's shape): no slot (no absolute-time window), one row per python call of the derived texts parsed by its real parser (QueueArgparse).
SLOT_RECEIPTS = []
ARGPARSE_RECEIPTS = []
DRAFT = 'harness/runs/q18a_inwait.draft.sh'
SETTINGS = 'harness/runs/q18a_inwait_settings.sh'
DECLARE = 'harness/runs/q18a_inwait_declare.sh'
SCRIPTS = (DRAFT, SETTINGS, DECLARE)
CORE = T.labels('core')
TAIL = T.labels('tail')
DECODE = [d['label'] for d in T.DECODE_ARMS]


class Derivation(unittest.TestCase):
    def test_files_are_the_derivation_and_the_parents_are_unchanged(self):
        derived = queue.derive()
        self.assertEqual(sorted(derived), sorted(SCRIPTS))
        for name, text in derived.items():
            self.assertEqual((ROOT / name).read_text(), text, name)
        for name, text in sq.derive().items():
            self.assertEqual((ROOT / name).read_text(), text, name)
        for name, text in c1q.derive().items():
            self.assertEqual((ROOT / name).read_text(), text, name)
        self.assertIn(queue.body(), derived[DRAFT])

    def test_every_declared_delta_is_there(self):
        body = queue.body()
        for mark in ('@CORE@', '@TAIL@', '@BAND1@', '@DECODE', '@ATTN_ON@'):
            self.assertNotIn(mark, body)
        self.assertIn('python3 "$H/e3_inwait_box.py" arm', body)
        self.assertIn('python3 "$H/e3_inwait_box.py" decode-arm', body)
        self.assertIn('--c1 "${_c1#c1}" --size "${_group#g}" --cohort "$_cohort" --index "$_pair"', body)
        self.assertIn('if [[ $_pair == [01] ]]; then _workload=$H/workloads/q18a-native41-progress.json', body)
        self.assertNotIn('e3_stage1_step_box.py', body)
        self.assertIn('  for _label in decode-am1; do\n    inwait_decode_arm "$_label"\n  done\n', body)
        self.assertLess(body.index('inwait_decode_arm "$_label"'), body.index('for _arm in core:'))
        self.assertLess(body.index('  inwait_smoke ' + T.SMOKE_ARM + '\n'), body.index('inwait_decode_arm "$_label"'))
        self.assertIn('  for _label in ' + ' '.join(TAIL) + '; do\n    alpha_next_arm tail', body)
        self.assertLess(body.index('" grade '), body.index('alpha_next_arm tail "$_label"'), 'the cap arm after the core grade')
        self.assertIn('for _arm in ' + ' '.join('core:' + x for x in CORE[1:]) + '; do', body)

    def test_the_switch_in_extra_env_and_asserted(self):
        settings = (ROOT / SETTINGS).read_text()
        env = re.search(r'"EXTRA_ENV=([^"]*)"', settings).group(1).split()
        self.assertEqual(env.index('FLASH_S5_LAZY_RECV=1'), env.index('FLASH_S5_STAGE_OVERLAP_OBSERVE=1') + 1)
        self.assertEqual(env.index('FLASH_S5_DECODE_MERGE=1'), env.index('FLASH_S5_LAZY_RECV=1') + 1)
        self.assertEqual(env.index('FLASH_ATTN_META_GDN_BATCH=1'), env.index('FLASH_S5_DECODE_MERGE=1') + 1)
        self.assertEqual(env.index('FLASH_S5_INWAIT=1'), env.index('FLASH_ATTN_META_GDN_BATCH=1') + 1)
        for name in ('FLASH_S5_HANDLER_ANATOMY=1', 'FLASH_S5_STAGE_OVERLAP=1', 'FLASH_S5_LAYER_GROUPS=1', 'FLASH_S5_OVERLAP=1'):
            self.assertIn(name, env)
        for name in sq.SWITCH_NAMES + ('FLASH_ATTN_META_GDN_BATCH', queue.SWITCH_NAME):
            self.assertIn("assert env.get('%s') == '1'" % name, (ROOT / DRAFT).read_text())
            self.assertIn("assert env.get('%s')=='1'" % name, (ROOT / DECLARE).read_text())

    def test_the_header_states_numerics_are_not_covered(self):
        draft = (ROOT / DRAFT).read_text()
        self.assertIn('NUMERICS ARE NOT COVERED', draft)
        self.assertIn('numerics not covered', queue.body())

    def test_the_arm_call_sends_no_logprobs(self):
        body = queue.body()
        arm = body[body.index('python3 "$H/e3_inwait_box.py" arm'):body.index('> "$Q/arm-$_label.log"')]
        self.assertNotIn('logprob', arm)
        self.assertNotIn('numerics', arm)

    def test_no_absolute_time_window_in_the_texts(self):
        window = re.compile(r'NOT_BEFORE|KILL_TIME|HARD_END|LATEST_START|date -d|date \+%s')
        for name, text in queue.derive().items():
            self.assertIsNone(window.search(text), name)


def executable(text):
    """The script's lines that are not comments (heredoc bodies included: they run too)."""
    return [l for l in text.splitlines() if not l.lstrip().startswith('#')]


class NameChecks(unittest.TestCase):
    QNAME = re.search(r'QNAME=\$\{QNAME:-([^}]+)\}', (ROOT / SETTINGS).read_text()).group(1)

    def test_the_real_qname(self):
        self.assertEqual(self.QNAME, 'q18a-inwait')
        self.assertEqual(re.search(r'QNAME=\$\{QNAME:-([^}]+)\}', (ROOT / DECLARE).read_text()).group(1), self.QNAME)

    def test_every_qname_test_passes_with_it(self):
        found = 0
        for name in SCRIPTS:
            for line in executable((ROOT / name).read_text()):
                if '$QNAME' in line and '[[' in line:
                    test = line[line.index('[['):line.index(']]') + 2]
                    found += 1
                    run = subprocess.run(['bash', '-c', 'QNAME=%s; %s' % (self.QNAME, test)], capture_output=True, text=True)
                    self.assertEqual(run.returncode, 0, (name, test))
                    bad = subprocess.run(['bash', '-c', 'QNAME=%s; %s' % (shlex.quote('q18a stage1 step'), test)], capture_output=True)
                    if '=~' in test:
                        self.assertNotEqual(bad.returncode, 0, (name, test))
        self.assertEqual(found, 1)                  # the declaration's: [[ $QNAME =~ ^q18a[-_a-zA-Z0-9]*$ ]]

    def test_run_id_and_request_dir_carry_the_name(self):
        draft = (ROOT / DRAFT).read_text()
        self.assertIn('RUN_ID="%s-$(date -u +%%Y%%m%%dT%%H%%M%%SZ)-$$"' % self.QNAME, draft)
        self.assertIn('R=$S0/$QNAME-requests', (ROOT / DECLARE).read_text())

    def test_names_resolve_and_none_is_c1s_queue(self):
        for name in SCRIPTS:
            lines = executable((ROOT / name).read_text())
            for line in lines:
                for m in re.finditer(r'\$H/runs/([A-Za-z0-9_.]+)', line):
                    self.assertTrue((HERE / 'runs' / m.group(1)).is_file(), (name, m.group(1)))
                for m in re.finditer(r'python3 (?:-B )?"\$H/([A-Za-z0-9_]+\.py)"', line):
                    self.assertTrue((HERE / m.group(1)).is_file(), (name, m.group(1)))
                self.assertNotIn('q18a_alpha_next_c1', line, name)
                self.assertNotIn('q18a-alpha-next-c1', line, name)
                self.assertNotIn('alpha_next_c1_gate import', line, name)
        for f in queue.FILES.split():
            self.assertTrue((HERE / f).is_file(), f)
            self.assertIn(f, (ROOT / DECLARE).read_text())


class Budget(unittest.TestCase):
    def test_relative_and_every_arm_enters_at_the_walls(self):
        tree = ast.parse((HERE / 'inwait_gate.py').read_text())
        imports = {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                   for a in n.names} | {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        self.assertFalse(imports & {'time', 'datetime', 'calendar'})
        exp = gate.EXPECTED_RUNTIME_S
        self.assertEqual((exp['arms'], exp['decode_arms'], exp['core_arms'], exp['tail_arms']), (12, 1, 10, 1))
        self.assertEqual(exp['nominal']['not_run'], [])
        self.assertLess(gate.LAST_ARM_ENTRY_S, gate.LAST_TAIL_ENTRY_S)
        self.assertTrue(gate.arm_fits(gate.LAST_ARM_ENTRY_S, 'core'))
        self.assertFalse(gate.arm_fits(gate.LAST_ARM_ENTRY_S + 1, 'core'))
        self.assertIn('inwait_arms', gate.PHASES)
        self.assertEqual(gate.ACTIVE_S + gate.CLEANUP_S, gate.BOX_BUDGET_S)

    def test_what_the_budget_buys_at_1_5x_the_walls(self):
        self.assertEqual(gate.BOX_BUDGET_S, 8700)
        self.assertLessEqual(gate.EXPECTED_RUNTIME_S['nominal']['total_min'], 40)
        self.assertEqual(gate.EXPECTED_RUNTIME_S['slower']['not_run'], [])

    def test_the_tool_prints_the_table(self):
        got = subprocess.run([sys.executable, '-B', queue.__file__, '--table'], capture_output=True, text=True, timeout=60)
        self.assertEqual(got.returncode, 0, got.stderr[-400:])
        for name in CORE:
            self.assertIn(name, got.stdout)
        self.assertIn('box budget 8700 s', got.stdout)


STUB_BOX = r'''#!/usr/bin/env python3
import json, os, shutil, sys
if sys.argv[1] in ('grade', 'read'):
    with open(os.environ['CALLS'], 'a') as f:
        f.write(json.dumps(dict(script='box', action=sys.argv[1])) + '\n')
    sys.path.insert(0, os.environ['REAL_HARNESS'])
    import e3_inwait_box as real
    raise SystemExit(real.main(sys.argv[1:]))
a = sys.argv
out = a[a.index('--out') + 1]
if a[1] == 'smoke':
    with open(os.environ['CALLS'], 'a') as f:
        f.write(json.dumps(dict(script='box', action='smoke')) + '\n')
    raise SystemExit(int(os.environ['SMOKE_RC']))
if a[1] == 'decode-arm':
    label = a[a.index('--label') + 1]
    with open(os.environ['CALLS'], 'a') as f:
        f.write(json.dumps(dict(script='box', action=a[1], label=label, attn=a[a.index('--attn') + 1])) + '\n')
    shutil.copy(os.environ['ARMS'] + '/arm-%s.json' % label, out)
    raise SystemExit(0)
label = '%s.g%s-%s-%s' % (a[a.index('--c1') + 1], a[a.index('--size') + 1], a[a.index('--cohort') + 1], a[a.index('--index') + 1])
with open(os.environ['CALLS'], 'a') as f:
    f.write(json.dumps(dict(script='box', action=a[1], label=label, c1=a[a.index('--c1') + 1], workload=os.path.basename(a[a.index('--workload') + 1]))) + '\n')
shutil.copy(os.environ['ARMS'] + '/arm-%s.json' % label, out)
raise SystemExit(0)
'''


class ArmLoop(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='inwait-loop-'))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.q, self.h, self.bin, self.arms = (self.root / x for x in ('q', 'h', 'bin', 'arms'))
        for d in (self.q, self.h, self.bin, self.arms):
            d.mkdir()
        for name in ('native43_engine_death_h.py', 'inwait_gate.py', 'inwait_arms.py', 'attn_meta_arms.py', 'alpha_next_arms.py',
                     'alpha_next_stop_race_h.py', 'native39_trunk_gate.py', 'native40_box_gate.py', 'native41_box_gate.py'):
            (self.h / name).symlink_to(HERE / name)
        (self.h / 'e3_native39_trunk_checks.py').write_text(tap.STUB_CHECKS)
        (self.h / 'e3_inwait_box.py').write_text(STUB_BOX)
        self.tool('nvidia-smi', 'printf "0, 22925\\n1, 22835\\n2, 23037\\n3, 22947\\n"')
        self.tool('docker', 'case "$*" in *State.Running*) echo true;; *) echo 2026-11-01T10:48:09.000000000Z;; esac')
        self.tool('systemctl', 'echo 0')
        (self.q / 'unarmed-stock-prefill.json').write_text(json.dumps(tap.PASS))

    tool = tap.ArmLoop.tool

    def run_loop(self, edit=None, timeout=420, smoke_rc=0):
        emit, arms = tbox.full_run()
        arms += [a for a in tbox.run_of(None)[1] if a['label'] in TAIL]      # the stub arm's record; no lines here, so the read cannot be whole
        for a in arms:
            a.update((edit or {}).get(a['label'], {}))
            (self.arms / ('arm-%s.json' % a['label'])).write_text(json.dumps(a))
        tap.write_streams(self.q, list(emit.lines) + [(emit.lines[0][0], 'INFO x %s\n' % line) for line in tbox.tam.STARTUP])
        draft = queue.derive()[DRAFT]
        script = ['set -uo pipefail', 'FAILS=0; INCS=0; declare -A PHASE_VERDICT=()',
                  "ENGINE_CRASHED=0; ENGINE_CRASH_SOURCE=''; STOP_RACE=0; CONTAINER=qwen-flash-prod; SERVICE=vllm-flash; N0=0",
                  'Q=%s; H=%s; PROBE_URL=http://127.0.0.1:1; MODEL=m; RUN_ID=r; EPOCH=1; _cid=c; IDLE_TIMEOUT_S=1' % (self.q, self.h),
                  'Q18A_READINESS=/r/ready; S5_OVERLAY=/o; S5_LAUNCHER_NAME=l.sh', 'health_ok() { return 0; }']
        script += [tap.le_fix_function(draft, name) for name in tap.LOOP_FUNCTIONS]
        script += [queue.body(), 'echo "END FAILS=$FAILS INCS=$INCS CRASHED=$ENGINE_CRASHED SOURCE=$ENGINE_CRASH_SOURCE"']
        path = self.root / 'loop.sh'
        path.write_text('\n'.join(script) + '\n')
        env = {k: v for k, v in os.environ.items() if k != 'HARNESS_DIR'}
        env.update(SMOKE_RC=str(smoke_rc), PATH='%s:%s' % (self.bin, os.environ['PATH']), Q=str(self.q), CALLS=str(self.root / 'calls.jsonl'), ARMS=str(self.arms),
                   REAL_HARNESS=str(HERE), DEATH_LINE=tap.DEATH_LINE, PYTHONDONTWRITEBYTECODE='1')
        proc = subprocess.Popen(['bash', str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, start_new_session=True)
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            raise
        calls = [json.loads(x) for x in (self.root / 'calls.jsonl').read_text().splitlines()]
        steps = {m.group(1): m.group(2) for m in re.finditer(r'^STEP (\S+) (\S+)', out, flags=re.M)}
        return out, err, calls, steps

    def test_every_arm_in_order_then_grade_and_read(self):
        out, err, calls, steps = self.run_loop()
        self.assertIn('END FAILS=0 INCS=0 CRASHED=0', out, (out[-2000:], err[-1500:]))
        box = [c for c in calls if c['script'] == 'box']
        self.assertEqual([c.get('label') or c['action'] for c in box], CORE[:1] + ['smoke'] + DECODE + CORE[1:] + ['grade'] + TAIL + ['read'])
        self.assertEqual(steps['smoke'], 'PASS')
        rows = T.by_label()
        self.assertEqual([c['attn'] for c in box if c['action'] == 'decode-arm'], ['on'])
        for c in box:
            if c['action'] == 'arm':
                row = rows[c['label']]
                self.assertEqual(c['c1'], row['c1'], c['label'])
                self.assertEqual(c['workload'], tq.ap.BAND_WORKLOAD[row['band']] + '.json', c['label'])
        ingress = [c['out'] for c in calls if c['script'] == 'checks' and c['action'] == 'ingress']
        self.assertEqual(ingress, ['ingress-%s.json' % x for x in CORE[:1] + DECODE + CORE[1:] + TAIL])
        for name in DECODE + CORE:
            self.assertEqual(steps['arm-' + name], 'PASS', name)
        # The stub's tail arm wrote a record and no engine lines: the core grade passes, the read says it cannot be whole. Whole read: test_inwait_box.
        self.assertEqual((steps['inwait'], steps['inwait-read']), ('PASS', 'READ-INCONCLUSIVE'))
        grade = json.loads((self.q / 'inwait-grade.json').read_text())
        self.assertEqual((grade['schema'], grade['verdict'], grade['missing']), ('inwait-grade/1', 'PASS', []))
        self.assertEqual(sorted(grade['step_gate']['arms']), sorted(CORE))
        self.assertIs(grade['numerics_covered'], False)
        read = json.loads((self.q / 'inwait-read.json').read_text())
        self.assertEqual(read['schema'], 'inwait-read/1')
        self.assertTrue((self.q / 'inwait-report.md').read_text().startswith('# q18a-inwait'))
        self.assertEqual(sorted(grade['attn_meta']), sorted(DECODE + CORE))

    def test_a_leak_fails_the_core_grade_and_counts(self):
        emit, arms = tbox.run_of()
        arm = next(a for a in arms if a['label'] == 'cf.g2-live-1')
        arm['cow_host_filter']['counters']['skipped']['0'] = 2
        out, err, calls, steps = self.run_loop(edit={'cf.g2-live-1': dict(cow_host_filter=arm['cow_host_filter'])})
        self.assertEqual(steps['inwait'], 'FAIL', out[-1500:])
        self.assertIn('END FAILS=1', out)
        grade = json.loads((self.q / 'inwait-grade.json').read_text())
        self.assertIn('cf.g2-live-1:cow-host-filter-off-moved-its-counter', grade['failures'])

    def test_a_smoke_that_does_not_engage_stops_the_table(self):
        out, err, calls, steps = self.run_loop(smoke_rc=2)
        self.assertEqual(steps['smoke'], 'INCONCLUSIVE', out[-1500:])
        box = [c for c in calls if c['script'] == 'box']
        self.assertEqual([c.get('label') or c['action'] for c in box if c['action'] in ('arm', 'decode-arm')], CORE[:1], 'the smoke arm alone ran')
        for name in DECODE + CORE[1:]:
            self.assertEqual(steps['arm-' + name], 'INCONCLUSIVE', name)
        for name in TAIL:
            self.assertEqual(steps['arm-' + name], 'READ-INCONCLUSIVE', name)
        self.assertIn('NOT_RUN: the exact-path smoke did not engage', out)


ARGPARSE_VALUES = dict(tq.ARGPARSE_VALUES, QNAME='q18a-inwait', RUN_ID='q18a-inwait-fixture', _label='cf.g2-live-1',
                       _sel='cf.g2', _c1='cf', _group='g2', _arm='core:cf.g2-live-1', _pair='1', _cohort='live')


def bash_words(args, draft, values=None):
    """c1's helper over this queue's settings and labels: the words bash gives one python call."""
    budget = tap.le_fix_function(draft, 'native41_budget')
    values = dict(ARGPARSE_VALUES, **(values or {}))
    lines = ['set -u', 'fail_early() { :; }', 'fail() { :; }', 'H=%s' % shlex.quote(str(HERE))]
    lines += ['%s=%s' % (k, shlex.quote(v)) for k, v in values.items()]
    arrays = ('--arm "$Q/arm-cf.g2-live-1.json" --ingress "cf.g2-live-1=$Q/ingress-cf.g2-live-1.json" '
              '--watch "cf.g2-live-1=$Q/engine-watch-arm-cf.g2-live-1.json"')
    lines += ['GRADE_ARGS=(%s)' % arrays, 'READ_ARGS=(%s)' % arrays,
              'source "$H/runs/q18a_inwait_settings.sh" || exit 97', budget, "printf '%s\\0' " + args]
    done = subprocess.run(['bash', '-c', '\n'.join(lines)], capture_output=True, timeout=60)
    return done.returncode, done.stdout.decode().split('\0')[:-1], done.stderr.decode()[-400:]


class QueueH(unittest.TestCase):
    """The health-scoped death scan (lead 3 Nov 16:49; returning-append 14904f18, byte for byte): the draft calls _h, never the old."""
    OLD = ('native43_engine_death.py', 'alpha_next_stop_race.py')

    def test_the_bytes_are_the_forwarded_ones(self):
        for name, want in (('native43_engine_death_h.py', '8217760e769a500245c0bfe995cb3547e4a46775145cb81d42824a965107ac04'),
                           ('alpha_next_stop_race_h.py', '44f2954f2536bd6efb85dc90cfdf4e623e93dfe81b8cac95f45028ed84784dcd')):
            self.assertEqual(hashlib.sha256((HERE / name).read_bytes()).hexdigest(), want, name)

    def test_the_draft_names_the_new_modules_never_the_old(self):
        draft = queue.derive()[DRAFT]
        for old in self.OLD:
            self.assertNotIn(old, draft)
        self.assertEqual(draft.count('python3 "${H:-}/native43_engine_death_h.py" scan '), 1)
        self.assertEqual(draft.count('python3 "$H/native43_engine_death_h.py" watch '), 1)
        self.assertEqual(draft.count('python3 "$H/alpha_next_stop_race_h.py" '), 1)
        crash = draft[draft.index('host_crash_detected() {'):draft.index('engine_watched() {')]
        self.assertIn('native43_engine_death_h.py" scan', crash)
        self.assertIn('alpha_next_stop_race_h.py"', crash)

    def test_the_new_modules_import_nothing_old(self):
        for name in ('native43_engine_death_h.py', 'alpha_next_stop_race_h.py'):
            tree = ast.parse((HERE / name).read_text())
            names = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names} \
                | {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
            self.assertFalse(names & {'native43_engine_death', 'alpha_next_stop_race'}, name)

    def test_the_declaration_pins_them(self):
        h = queue.derive()
        for name in ('native43_engine_death_h.py', 'alpha_next_stop_race_h.py', 'e3_inwait_box.py', 'e3_alpha_next_c1b_box.py', 'e3_c1_open_rows.py', 'stage1_step_read.py', 'e3_stage1_step_box.py', 'e3_attn_meta_box.py', 'attn_meta_arms.py', 'inwait_arms.py'):
            self.assertEqual(h[DECLARE].count(' ' + name + ' '), 1, name)

    def test_a_startup_stall_before_the_first_health_200_is_a_read_and_one_after_it_is_a_death(self):
        import contextlib
        import io
        import native43_engine_death_h as death
        stall = '2026-11-03T15:06:22.000000000Z (EngineCore pid=1) shm_broadcast.py:801] No available shared memory broadcast block found in 60 seconds\n'
        health = '2026-11-03T15:06:54.000000000Z (APIServer pid=2) INFO:     127.0.0.1:1 - "GET /health HTTP/1.1" 200 OK\n'
        late = stall.replace('15:06:22', '15:07:30')
        with tempfile.TemporaryDirectory() as tmp:
            for lines, verdict, rc in (([stall, health], 'ALIVE', 0), ([health, late], 'DEAD', 1), ([stall], 'DEAD', 1)):
                log, out = Path(tmp) / 'engine.log', Path(tmp) / ('scan-%d.json' % len(list(Path(tmp).iterdir())))
                log.write_text(''.join(lines))
                with contextlib.redirect_stdout(io.StringIO()):
                    got = death.main(['scan', '--log', str(log), '--out', str(out)])
                self.assertEqual((got, json.loads(out.read_text())['verdict']), (rc, verdict), lines)


class QueueArgparse(unittest.TestCase):
    def test_every_queue_python_call_parses_with_its_real_parser(self):
        held = tan.held
        texts = queue.derive()
        draft = texts[DRAFT]
        calls = held.queue_python_calls(texts, (HERE / 'qlib.sh').read_text())
        parent = held.queue_python_calls(c1q.derive(), (HERE / 'qlib.sh').read_text())
        # c1's calls, one for one, with its box replaced by this queue's and the _h names.
        swap = lambda s: {'e3_alpha_next_c1_box.py': 'e3_inwait_box.py', 'native43_engine_death.py': 'native43_engine_death_h.py',
                          'alpha_next_stop_race.py': 'alpha_next_stop_race_h.py'}.get(s, s)
        decode = [c for c in calls if c[1] == 'e3_inwait_box.py' and c[2].split()[0] == 'decode-arm']
        self.assertEqual(len(decode), 1)
        smoke = [c for c in calls if c[1] == 'e3_inwait_box.py' and c[2].split()[0] == 'smoke']
        self.assertEqual(len(smoke), 1)
        added = sorted(c[1] for c in calls if c not in decode and c not in smoke and c[2].split()[0] not in ('ingress', 'full-idle'))
        self.assertEqual(added, sorted(swap(c[1]) for c in parent if c[2].split()[0] not in ('ingress', 'full-idle')))
        self.assertNotIn('e3_alpha_next_c1_box.py', {c[1] for c in calls})
        setcfg = next(p for p in (ROOT / 'deploy/setcfg.py', HERE / 'setcfg.reference.py') if p.is_file())
        rows = []
        for name, script, args, form in calls:
            with self.subTest(file=name.split('/')[-1], script=script, args=args[:60]):
                rc, words, err = bash_words(args, draft, dict(_label='decode-am1', _attn='on') if args.startswith('decode-arm') else None)
                self.assertEqual(rc, 0, err)
                if form != 'python3':
                    words = held.QLIB_CALLS[form][1](words)
                if script == 'SETCFG':
                    verdict, detail = held.setcfg_parse(setcfg, words)
                else:
                    verdict, detail = held.real_parse(HERE / script, words)
                self.assertEqual(verdict, 'parsed', (script, words[:4], detail))
                rows.append(dict(file=name.split('/')[-1], script=script, form=form,
                                 action=words[0] if words and not words[0].startswith('-') else None,
                                 words=len(words), verdict=verdict))
        self.assertEqual(len(rows), len(calls))
        ARGPARSE_RECEIPTS[:] = rows
        self.assertEqual(SLOT_RECEIPTS, [])

        def first(script, action):
            return next(a for n, s, a, f in calls if s == script and a.split()[0] == action)
        arm, grade = first('e3_inwait_box.py', 'arm'), first('e3_inwait_box.py', 'grade')
        for label, original, (old, new) in (
                ('arm-unknown-option', arm, ('--control-timeout-s', '--control-timeout-sec')),
                ('arm-without-selection', arm, ('--c1 "${_c1#c1}" ', '')),
                ('arm-undeclared-selection', arm, ('--c1 "${_c1#c1}"', '--c1 on')),
                ('arm-undeclared-round', arm, ('--index "$_pair"', '--index 6')),
                ('grade-without-out', grade, ('--out "$Q/inwait-grade.json"', ''))):
            with self.subTest(control=label):
                self.assertEqual(original.count(old), 1, (label, original[:200]))
                rc, words, err = bash_words(original.replace(old, new), draft)
                self.assertEqual(rc, 0, err)
                self.assertEqual(held.real_parse(HERE / 'e3_inwait_box.py', words)[0], 'refused', label)


if __name__ == '__main__':
    unittest.main()
