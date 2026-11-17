#!/usr/bin/env python3
"""q18a-attn-meta queue: the derivation, the preflight name checks executed with the runner's real QNAME, the budget, and the
derived arm loop under real bash (stub arm and check clients, the real watch functions, the real attn-meta grade and read).
CPU only; no pass here is box evidence.

Classes:
* Derivation: tools/attn_meta_queue_b.py --check (the files are the derivation); c1's texts unchanged (its --check); every
  declared delta present; the draft's body is the derived body; the switch in EXTRA_ENV and asserted in preflight and declaration.
* NameChecks: every QNAME test in the three runs/ scripts, extracted and executed in bash with QNAME set to the settings'
  real default (the runner's name, q18a-attn-meta), passes; the declaration's request dir, the draft's RUN_ID and every
  ``$H/runs/...`` and ``python3 "$H/X.py"`` name resolve to files in this tree; nothing executable names c1's queue.
* Budget: the gate's limits are relative (no clock import), every arm enters at the walls and at 1.5x.
* ArmLoop: the derived body: the decode-only baselines first, the core arms in table order with --attn from the index and the
  band 1 workload for indexes 0, 1, 3, 4, the core grade, the G2 tail, the read; END FAILS=0 INCS=0; an off-arm leak in a
  record makes the core grade FAIL and the queue count it.
"""
import ast
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
import attn_meta_arms as T  # noqa: E402
import attn_meta_gate as gate  # noqa: E402
import attn_meta_queue_b as queue  # noqa: E402
import alpha_next_c1_queue as c1q  # noqa: E402
import test_attn_meta_box as tbox  # noqa: E402

tap = tq.tap
tan = tq.tan
# The pack's probe derive reads both (c1's shape): no slot (no absolute-time window), one row per python call of the derived
# texts parsed by its real parser (QueueArgparse).
SLOT_RECEIPTS = []
ARGPARSE_RECEIPTS = []
DRAFT = 'harness/runs/q18a_attn_meta.draft.sh'
SETTINGS = 'harness/runs/q18a_attn_meta_settings.sh'
DECLARE = 'harness/runs/q18a_attn_meta_declare.sh'
SCRIPTS = (DRAFT, SETTINGS, DECLARE)
CORE = T.labels('core')
TAIL = T.labels('tail')
DECODE = [d['label'] for d in T.DECODE_ARMS]


def body_function(text, name):
    start = text.index(name + '() {\n')
    return text[start:text.index('\n}\n', start) + 3]


class Derivation(unittest.TestCase):
    def test_files_are_the_derivation_and_c1_is_unchanged(self):
        derived = queue.derive()
        self.assertEqual(sorted(derived), sorted(SCRIPTS))
        for name, text in derived.items():
            self.assertEqual((ROOT / name).read_text(), text, name)
        for name, text in c1q.derive().items():
            self.assertEqual((ROOT / name).read_text(), text, name)
        self.assertIn(queue.body(), derived[DRAFT])

    def test_every_declared_delta_is_there(self):
        text = queue.c1_template()
        for old, new in queue.BODY_EDITS:          # in order, as body() applies them
            self.assertIn(old, text, old[:80])
            text = text.replace(old, new)
        body = queue.body()
        for mark in ('@CORE@', '@TAIL@', '@BAND1@', '@DECODE@', '@ATTN_ON@', '@DECODE_FUNCTION@', '@DECODE_CASE@'):
            self.assertNotIn(mark, body)
        self.assertIn('if [[ $_pair == [234] ]]; then _attn=on; else _attn=off; fi', body)
        self.assertIn('if [[ $_pair == [0134] ]]; then _workload=$H/workloads/q18a-native41-progress.json', body)
        self.assertIn('python3 "$H/e3_attn_meta_box_b.py" arm --attn "$_attn"', body)
        self.assertNotIn('e3_alpha_next_c1_box.py', body)

    def test_switch_in_extra_env_and_asserted(self):
        settings = (ROOT / SETTINGS).read_text()
        env = re.search(r'"EXTRA_ENV=([^"]*)"', settings).group(1).split()
        self.assertIn('FLASH_ATTN_META_GDN_BATCH=1', env)
        self.assertEqual(env.index('FLASH_ATTN_META_GDN_BATCH=1'), env.index('FLASH_S5_STAGE_OVERLAP_OBSERVE=1') + 1)
        for name in ('FLASH_S5_HANDLER_ANATOMY=1', 'FLASH_S5_STAGE_OVERLAP=1', 'FLASH_S5_LAYER_GROUPS=1', 'FLASH_S5_OVERLAP=1'):
            self.assertIn(name, env)
        self.assertIn("assert env.get('FLASH_ATTN_META_GDN_BATCH') == '1'", (ROOT / DRAFT).read_text())
        self.assertIn("assert env.get('FLASH_ATTN_META_GDN_BATCH')=='1'", (ROOT / DECLARE).read_text())


def executable(text):
    """The script's lines that are not comments (heredoc bodies included: they run too)."""
    return [l for l in text.splitlines() if not l.lstrip().startswith('#')]


class NameChecks(unittest.TestCase):
    QNAME = re.search(r'QNAME=\$\{QNAME:-([^}]+)\}', (ROOT / SETTINGS).read_text()).group(1)

    def test_the_real_qname(self):
        self.assertEqual(self.QNAME, 'q18a-attn-meta')
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
                    bad = subprocess.run(['bash', '-c', 'QNAME=%s; %s' % ('q18a attn meta', test)], capture_output=True)
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
    def test_relative_and_every_arm_enters(self):
        tree = ast.parse((HERE / 'attn_meta_gate.py').read_text())
        imports = {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                   for a in n.names} | {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        self.assertFalse(imports & {'time', 'datetime', 'calendar'})
        exp = gate.EXPECTED_RUNTIME_S
        self.assertEqual(exp['arms'], 17)
        self.assertEqual(exp['nominal']['not_run'], [])
        self.assertEqual(exp['slower']['not_run'], [])
        self.assertLess(gate.LAST_ARM_ENTRY_S, gate.LAST_TAIL_ENTRY_S)
        self.assertTrue(gate.arm_fits(gate.LAST_ARM_ENTRY_S, 'decode'))
        self.assertFalse(gate.arm_fits(gate.LAST_ARM_ENTRY_S + 1, 'core'))
        self.assertIn('attn_meta_arms', gate.PHASES)


STUB_BOX = r'''#!/usr/bin/env python3
import json, os, shutil, sys
if sys.argv[1] in ('grade', 'read'):
    with open(os.environ['CALLS'], 'a') as f:
        f.write(json.dumps(dict(script='box', action=sys.argv[1])) + '\n')
    sys.path.insert(0, os.environ['REAL_HARNESS'])
    import e3_attn_meta_box_b as real
    raise SystemExit(real.main(sys.argv[1:]))
a = sys.argv
out = a[a.index('--out') + 1]
if a[1] == 'decode-arm':
    label = a[a.index('--label') + 1]
else:
    label = 'c1%s.g%s-%s-%s' % (a[a.index('--c1') + 1], a[a.index('--size') + 1], a[a.index('--cohort') + 1], a[a.index('--index') + 1])
with open(os.environ['CALLS'], 'a') as f:
    f.write(json.dumps(dict(script='box', action=a[1], label=label, attn=a[a.index('--attn') + 1],
                            workload=os.path.basename(a[a.index('--workload') + 1]))) + '\n')
if label in os.environ.get('TAIL_LABELS', '').split() and not os.path.exists(os.environ['ARMS'] + '/tail-appended'):
    for stream in ('log', 'err'):
        with open(os.environ['Q'] + '/engine-stream.' + stream, 'a') as f, open(os.environ['ARMS'] + '/tail.' + stream) as g:
            f.write(g.read())
    open(os.environ['ARMS'] + '/tail-appended', 'w').close()
shutil.copy(os.environ['ARMS'] + '/arm-%s.json' % label, out)
raise SystemExit(0)
'''


class ArmLoop(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='attn-meta-loop-'))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.q, self.h, self.bin, self.arms = (self.root / x for x in ('q', 'h', 'bin', 'arms'))
        for d in (self.q, self.h, self.bin, self.arms):
            d.mkdir()
        for name in ('native43_engine_death_h.py', 'attn_meta_gate.py', 'attn_meta_arms.py', 'alpha_next_arms.py',
                     'alpha_next_stop_race_h.py', 'native39_trunk_gate.py', 'native40_box_gate.py', 'native41_box_gate.py'):
            (self.h / name).symlink_to(HERE / name)
        (self.h / 'e3_native39_trunk_checks.py').write_text(tap.STUB_CHECKS)
        (self.h / 'e3_attn_meta_box_b.py').write_text(STUB_BOX)
        self.tool('nvidia-smi', 'printf "0, 22925\\n1, 22835\\n2, 23037\\n3, 22947\\n"')
        self.tool('docker', 'case "$*" in *State.Running*) echo true;; *) echo 2026-11-01T10:48:09.000000000Z;; esac')
        self.tool('systemctl', 'echo 0')
        (self.q / 'unarmed-stock-prefill.json').write_text(json.dumps(tap.PASS))

    tool = tap.ArmLoop.tool

    def records(self, edit=None):
        emit, arms = tq.synthetic_run(rows=T.arms())
        rows = T.by_label()
        for a in arms:
            r = rows[a['label']]
            if r['size'] == 1:
                a['group_delta'] = dict.fromkeys(a['group_delta'], 0)
            a['attn_meta'] = tbox.block(r['attn'], [9] * 4 if r['attn'] else [0] * 4)
        decode = [tbox.decode_record(d, rate) for d, rate in zip(T.DECODE_ARMS, (400.0, 410.0))]
        for d in decode:
            d.update(wall_start_s=arms[0]['wall_start_s'] - 100, wall_end_s=arms[0]['wall_start_s'] - 60)
        for a in arms + decode:
            a.update((edit or {}).get(a['label'], {}))
        return emit, arms, decode

    def run_loop(self, edit=None, timeout=420):
        emit, arms, decode = self.records(edit)
        head = [['log', '2026-11-01T11:50:00.000000000Z (Worker_TP%d pid=%d) INFO 11-01 11:50:00 [attn_meta_select.py:155] '
                 '%s\n' % (i % 2, 201 + i, t)] for i, t in enumerate(tbox.STARTUP)]
        first_tail = min(a['wall_start_s'] for a in arms if a['part'] == 'tail') * 1e9
        cut = next(i for i, (_, text) in enumerate(emit.lines) if (tq.tan.sr.death.wall_ns(text.encode()) or 0) >= first_tail)
        tap.write_streams(self.q, head + emit.lines[:cut])
        tap.write_streams(self.arms, emit.lines[cut:])
        for stream in ('log', 'err'):
            os.replace(self.arms / ('engine-stream.' + stream), self.arms / ('tail.' + stream))
        for a in arms + decode:
            (self.arms / ('arm-%s.json' % a['label'])).write_text(json.dumps(a))
        draft = queue.derive()[DRAFT]
        body = queue.body()
        script = ['set -uo pipefail', 'FAILS=0; INCS=0; declare -A PHASE_VERDICT=()',
                  "ENGINE_CRASHED=0; ENGINE_CRASH_SOURCE=''; STOP_RACE=0; CONTAINER=qwen-flash-prod; SERVICE=vllm-flash; N0=0",
                  'Q=%s; H=%s; PROBE_URL=http://127.0.0.1:1; MODEL=m; RUN_ID=r; EPOCH=1; _cid=c; IDLE_TIMEOUT_S=1' % (self.q, self.h),
                  'Q18A_READINESS=/r/ready; S5_OVERLAY=/o; S5_LAUNCHER_NAME=l.sh', 'health_ok() { return 0; }']
        script += [tap.le_fix_function(draft, name) for name in tap.LOOP_FUNCTIONS]
        script += [body, 'echo "END FAILS=$FAILS INCS=$INCS CRASHED=$ENGINE_CRASHED SOURCE=$ENGINE_CRASH_SOURCE"']
        path = self.root / 'loop.sh'
        path.write_text('\n'.join(script) + '\n')
        env = {k: v for k, v in os.environ.items() if k != 'HARNESS_DIR'}
        env.update(PATH='%s:%s' % (self.bin, os.environ['PATH']), Q=str(self.q), CALLS=str(self.root / 'calls.jsonl'),
                   ARMS=str(self.arms), REAL_HARNESS=str(HERE), TAIL_LABELS=' '.join(TAIL), DEATH_LINE=tap.DEATH_LINE,
                   PYTHONDONTWRITEBYTECODE='1')
        proc = subprocess.Popen(['bash', str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
                                start_new_session=True)
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            raise
        calls = [json.loads(x) for x in (self.root / 'calls.jsonl').read_text().splitlines()]
        steps = {m.group(1): m.group(2) for m in re.finditer(r'^STEP (\S+) (\S+)', out, flags=re.M)}
        return out, err, calls, steps

    def test_every_arm_in_order_then_grade_tail_read(self):
        out, err, calls, steps = self.run_loop()
        self.assertIn('END FAILS=0 INCS=0 CRASHED=0', out, (out[-2000:], err[-1500:]))
        box = [c for c in calls if c['script'] == 'box']
        self.assertEqual([c.get('label') or c['action'] for c in box], DECODE + CORE + ['grade'] + TAIL + ['read'])
        rows = T.by_label()
        for c in box:
            if c['action'] == 'arm':
                row = rows[c['label']]
                self.assertEqual(c['attn'], 'on' if row['attn'] else 'off', c['label'])
                self.assertEqual(c['workload'], tq.ap.BAND_WORKLOAD[row['band']] + '.json', c['label'])
            elif c['action'] == 'decode-arm':
                self.assertEqual(c['attn'], {'decode-am0': 'off', 'decode-am1': 'on'}[c['label']])
                self.assertEqual(c['workload'], 'q18a-native41-progress.json')
        ingress = [c['out'] for c in calls if c['script'] == 'checks' and c['action'] == 'ingress']
        self.assertEqual(ingress, ['ingress-%s.json' % x for x in DECODE + CORE + TAIL])
        fn = body_function(queue.body(), 'attn_meta_decode_arm')
        self.assertIn('--generation 8 --out "$Q/ingress-$_label.json"', fn)
        for name in DECODE + CORE:
            self.assertEqual(steps['arm-' + name], 'PASS', name)
        for name in TAIL:
            self.assertEqual(steps['arm-' + name], 'READ-PASS', name)
        self.assertEqual((steps['attn-meta'], steps['attn-meta-read']), ('PASS', 'READ-PASS'))
        grade = json.loads((self.q / 'attn-meta-grade.json').read_text())
        self.assertEqual((grade['schema'], grade['verdict']), ('attn-meta-grade/1', 'PASS'))
        self.assertEqual(sorted(grade['attn_meta']), sorted(DECODE + CORE))
        read = json.loads((self.q / 'attn-meta-read.json').read_text())
        self.assertEqual(sorted(read['attn_meta']), sorted(DECODE + CORE + TAIL))
        self.assertTrue((self.q / 'attn-meta-report.md').read_text().startswith('# q18a-attn-meta grade: PASS'))

    def test_a_leak_fails_the_core_grade_and_counts(self):
        out, err, calls, steps = self.run_loop(edit={'c1off.g1-live-1': dict(attn_meta=tbox.block(False, [0, 1, 0, 0]))})
        self.assertEqual(steps['attn-meta'], 'FAIL', out[-1500:])
        self.assertIn('END FAILS=1', out)
        grade = json.loads((self.q / 'attn-meta-grade.json').read_text())
        self.assertIn('c1off.g1-live-1:attn-meta-selection-leak', grade['failures'])


ARGPARSE_VALUES = dict(tq.ARGPARSE_VALUES, QNAME='q18a-attn-meta', RUN_ID='q18a-attn-meta-fixture', _label='c1defer.g1-live-3',
                       _sel='c1defer.g1', _c1='c1defer', _group='g1', _arm='core:c1defer.g1-live-3', _pair='3', _attn='on',
                       _cohort='live')


def bash_words(args, draft, values=None):
    """c1's helper over this queue's settings and labels: the words bash gives one python call."""
    budget = tap.le_fix_function(draft, 'native41_budget')
    values = dict(ARGPARSE_VALUES, **(values or {}))
    lines = ['set -u', 'fail_early() { :; }', 'fail() { :; }', 'H=%s' % shlex.quote(str(HERE))]
    lines += ['%s=%s' % (k, shlex.quote(v)) for k, v in values.items()]
    arrays = ('--arm "$Q/arm-c1defer.g1-live-3.json" --ingress "c1defer.g1-live-3=$Q/ingress-c1defer.g1-live-3.json" '
              '--watch "c1defer.g1-live-3=$Q/engine-watch-arm-c1defer.g1-live-3.json"')
    lines += ['GRADE_ARGS=(%s)' % arrays, 'READ_ARGS=(%s)' % arrays,
              'source "$H/runs/q18a_attn_meta_settings.sh" || exit 97', budget, "printf '%s\\0' " + args]
    done = subprocess.run(['bash', '-c', '\n'.join(lines)], capture_output=True, timeout=60)
    return done.returncode, done.stdout.decode().split('\0')[:-1], done.stderr.decode()[-400:]


class QueueH(unittest.TestCase):
    """The health-scoped death scan (lead 3 Nov 16:49; returning-append 14904f18, byte for byte): the draft calls _h, never the old."""
    OLD = ('native43_engine_death.py', 'alpha_next_stop_race.py')

    def test_the_bytes_are_the_forwarded_ones(self):
        import hashlib
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

    def test_the_declaration_pins_them_and_undoing_the_swaps_gives_the_base_texts(self):
        h, b = queue.derive(), queue.base.derive()
        for name in ('native43_engine_death_h.py', 'alpha_next_stop_race_h.py', 'e3_attn_meta_box_b.py', 'e3_c1_open_rows.py'):
            self.assertEqual(h[DECLARE].count(' ' + name + ' '), 1, name)
        self.assertEqual(h[DECLARE].replace(queue.PINS_NEW, queue.PINS_OLD), b[DECLARE])
        draft = h[DRAFT].replace(queue.INTERFACE_NEW, queue.INTERFACE_OLD)
        for old, new, _ in queue.DRAFT_SWAPS:
            draft = draft.replace(new, old)
        self.assertEqual(draft, b[DRAFT])
        self.assertEqual(h[SETTINGS], b[SETTINGS])

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
        # c1's calls, one for one, with its box replaced by this queue's, plus the decode arm's three (entry, arm, ingress).
        swap = lambda s: {'e3_alpha_next_c1_box.py': 'e3_attn_meta_box_b.py', 'native43_engine_death.py': 'native43_engine_death_h.py',
                          'alpha_next_stop_race.py': 'alpha_next_stop_race_h.py'}.get(s, s)
        extra = sorted(['e3_native39_trunk_checks.py', 'e3_attn_meta_box_b.py', 'e3_native39_trunk_checks.py'])
        self.assertEqual(sorted(c[1] for c in calls), sorted(sorted(swap(c[1]) for c in parent) + extra))
        self.assertNotIn('e3_alpha_next_c1_box.py', {c[1] for c in calls})
        setcfg = next(p for p in (ROOT / 'deploy/setcfg.py', HERE / 'setcfg.reference.py') if p.is_file())
        rows = []
        for name, script, args, form in calls:
            with self.subTest(file=name.split('/')[-1], script=script, args=args[:60]):
                values = dict(_label='decode-am1', _attn='on') if args.split()[0] == 'decode-arm' else None
                rc, words, err = bash_words(args, draft, values)
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
        arm, grade = first('e3_attn_meta_box_b.py', 'arm'), first('e3_attn_meta_box_b.py', 'grade')
        dec = first('e3_attn_meta_box_b.py', 'decode-arm')
        for label, original, (old, new), values in (
                ('arm-unknown-option', arm, ('--control-timeout-s', '--control-timeout-sec'), None),
                ('arm-without-attn', arm, ('--attn "$_attn" ', ''), None),
                ('arm-undeclared-attn', arm, ('--attn "$_attn"', '--attn maybe'), None),
                ('arm-undeclared-c1', arm, ('--c1 "${_c1#c1}"', '--c1 on'), None),
                ('arm-undeclared-round', arm, ('--index "$_pair"', '--index 6'), None),
                ('decode-undeclared-label', dec, ('--label "$_label"', '--label decode-am2'), dict(_attn='on')),
                ('decode-without-attn', dec, ('--attn "$_attn" ', ''), dict(_label='decode-am0')),
                ('grade-without-out', grade, ('--out "$Q/attn-meta-grade.json"', ''), None)):
            with self.subTest(control=label):
                self.assertEqual(original.count(old), 1, (label, original[:200]))
                rc, words, err = bash_words(original.replace(old, new), draft, values)
                self.assertEqual(rc, 0, err)
                self.assertEqual(held.real_parse(HERE / 'e3_attn_meta_box_b.py', words)[0], 'refused', label)


if __name__ == '__main__':
    unittest.main()
