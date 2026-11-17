"""C1 queue: the derived texts, the arm loop in bash over the real C1 grade, every python3 call's parser, the budget. CPU only.

Classes:

* ArmLoop: the derived body under real bash with the draft's own functions, the real death watch and the real C1 grade, over
  stub arm and check clients; the arm records and engine lines are test_alpha_c1_box's connected run (the real overlap
  composition, sixteen arms). The band follows from the round: rounds 2 and 3 read alpha-perf's band 2 manifest.
* Commands / QueueArgparse: every python3 call of the derived texts parses with its own module's real parser.
* Derivation: derived texts on disk and parse; everything else is C0's queue line for line, with the declared deltas only.
* Budget: the 9,000 s box budget, the arm entry rule, and the sixteen arms against alpha-perf's measured box walls (the
  expected last arm entry must fit with margin, not only the arithmetic identity); no absolute-time window.

The grade itself (engagement, early open, one command in flight, memory, OOM, tears) is test_alpha_c1_box.py; the typed selection
over the real control plane is test_alpha_c1_box.Selection and test_alpha_c0_queue.Arm.
Fixture leaves: alpha_c1_fixture_leaves.json, beside the inherited files it names.
"""
import argparse
import contextlib
import difflib
import io
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
import time
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / 'tools'), str(ROOT / 'patches/s5_engine')]
import alpha_c0_queue as c0queue  # noqa: E402
import alpha_c1_gate as gate  # noqa: E402
import alpha_c1_queue as queue  # noqa: E402
import e3_alpha_c1_box as box  # noqa: E402
import test_alpha_c1_box as c1box  # noqa: E402
import test_alpha_perf_queue as tap  # noqa: E402
import test_native41_box_queue as held  # noqa: E402

SLOT_RECEIPTS = []
ARGPARSE_RECEIPTS = []
DRAFT = 'harness/runs/q18a_alpha_c1.draft.sh'
SETTINGS = 'harness/runs/q18a_alpha_c1_settings.sh'
DECLARE = 'harness/runs/q18a_alpha_c1_declare.sh'
C0_DRAFT = 'harness/runs/q18a_alpha_c0.draft.sh'
C0_SETTINGS = 'harness/runs/q18a_alpha_c0_settings.sh'
C0_DECLARE = 'harness/runs/q18a_alpha_c0_declare.sh'
LABELS = box.required_labels()
BAND1, BAND2 = 'q18a-native41-progress.json', 'q18a-alpha-perf-band2.json'


def body_commands(text):
    body = text[text.index(queue.BODY.splitlines()[0]):]
    body = body[:body.index('record alpha-c1 ')]
    joined = re.sub(r'\\\n\s*', ' ', body)
    rows = []
    for line in joined.splitlines():
        match = re.search(r'python3 "\$H/([a-z0-9_]+)\.py"(.*)$', line)
        if match:
            rows.append((match.group(1), re.split(r'\s+>\s*"', match.group(2))[0].strip()))
    return rows


STUB_BOX = tap.STUB_BOX.replace('e3_alpha_perf_box', 'e3_alpha_c1_box')


class ArmLoop(unittest.TestCase):
    """queue.BODY runs as written; arm and check clients are stubs (leaf arm-loop-clients), the watch and grade real."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='alpha-c1-loop-'))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.q, self.h, self.bin, self.arms = (self.root / x for x in ('q', 'h', 'bin', 'arms'))
        for d in (self.q, self.h, self.bin, self.arms):
            d.mkdir()
        for name in ('native43_engine_death.py', 'alpha_c1_gate.py', 'native39_trunk_gate.py', 'native40_box_gate.py',
                     'native41_box_gate.py'):
            (self.h / name).symlink_to(HERE / name)
        (self.h / 'e3_native39_trunk_checks.py').write_text(tap.STUB_CHECKS)
        (self.h / 'e3_alpha_c1_box.py').write_text(STUB_BOX)
        self.tool('nvidia-smi', 'printf "0, 22925\\n1, 22835\\n2, 23037\\n3, 22947\\n"')
        self.tool('docker', 'case "$*" in *State.Running*) echo true;; *) echo 2026-11-01T10:48:09.000000000Z;; esac')
        self.tool('systemctl', 'echo 0')
        (self.q / 'unarmed-stock-prefill.json').write_text(json.dumps(tap.PASS))

    tool = tap.ArmLoop.tool

    def run_loop(self, count=None, *, jump_after=None, die_in=None, timeout=420):
        """The connected run's first ``count`` arms (all when None): their records and the engine lines up to their end."""
        lines, arms, spans = c1box.real_run()
        if count is not None:
            arms = arms[:count]
            lines = lines[:spans[arms[-1]['label']][1]]
        tap.write_streams(self.q, lines)
        for a in arms:
            (self.arms / ('arm-%s.json' % a['label'])).write_text(json.dumps(a))
        draft = queue.derive()[DRAFT]
        self.assertIn(queue.BODY, draft)
        script = ['set -uo pipefail', 'FAILS=0; INCS=0; declare -A PHASE_VERDICT=()',
                  "ENGINE_CRASHED=0; ENGINE_CRASH_SOURCE=''; CONTAINER=qwen-flash-prod; SERVICE=vllm-flash; N0=0",
                  'Q=%s; H=%s; PROBE_URL=http://127.0.0.1:1; MODEL=m; RUN_ID=r; EPOCH=1; _cid=c; IDLE_TIMEOUT_S=1' % (self.q, self.h),
                  'Q18A_READINESS=/r/ready; S5_OVERLAY=/o; S5_LAUNCHER_NAME=l.sh', 'health_ok() { return 0; }']
        script += [tap.le_fix_function(draft, name) for name in tap.LOOP_FUNCTIONS]
        if jump_after:
            # The box clock past the budget right after the named step (bash SECONDS is writable): leaf arm-loop-clock.
            script += ['eval "$(declare -f record | sed \'1s/record/real_record/\')"',
                       'record() { real_record "$@"; local rc=$?; [[ $1 == %s ]] && SECONDS=$((LAST_ARM_ENTRY_S + 10)); '
                       'return $rc; }' % jump_after]
        script += [queue.BODY, 'echo "END FAILS=$FAILS INCS=$INCS CRASHED=$ENGINE_CRASHED SOURCE=$ENGINE_CRASH_SOURCE"']
        path = self.root / 'loop.sh'
        path.write_text('\n'.join(script) + '\n')
        env = {k: v for k, v in os.environ.items() if k != 'HARNESS_DIR'}
        env.update(PATH='%s:%s' % (self.bin, os.environ['PATH']), Q=str(self.q), CALLS=str(self.root / 'calls.jsonl'),
                   ARMS=str(self.arms), REAL_HARNESS=str(HERE), DEATH_LINE=tap.DEATH_LINE, PYTHONDONTWRITEBYTECODE='1')
        if die_in:
            env['DIE_IN'] = die_in
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

    def grade_file(self):
        return json.loads((self.q / 'alpha-c1-grade.json').read_text())

    def test_sixteen_arms_in_order_with_each_bands_manifest_and_env_then_the_real_grade(self):
        out, err, calls, steps = self.run_loop()
        self.assertIn('END FAILS=0 INCS=0 CRASHED=0', out, (out[-1500:], err[-800:]))
        arm_calls = [c for c in calls if c['script'] == 'box']
        self.assertEqual([c['label'] for c in arm_calls], LABELS)
        for c, (_selection, _cohort, index) in zip(arm_calls, box.ARM_ORDER):
            self.assertEqual(c['workload'], BAND1 if index in (0, 1) else BAND2, c['label'])
            self.assertEqual(c['env'], dict(READINESS='/r/ready', GEN='/o/S5-OVERLAP-GENERATION.json', S5_OVERLAY='/o',
                                            S5_LAUNCHER_NAME='l.sh', PACKAGE_MANIFEST='/o/SHA256SUMS',
                                            GENERATION='/o/S5-OVERLAP-GENERATION.json'))
        self.assertEqual([c['out'] for c in calls if c['script'] == 'checks' and c['action'] == 'full-idle'],
                         ['entry-%s.json' % x for x in LABELS] + ['end-full-idle.json'])
        for name in LABELS:
            self.assertEqual(steps['arm-' + name], 'PASS')
            watch = json.loads((self.q / ('engine-watch-arm-%s.json' % name)).read_text())
            self.assertEqual((watch['verdict'], watch['client_rc'], watch['container']), ('CLIENT_EXIT', 0, 'qwen-flash-prod'))
        self.assertEqual(steps['alpha-c1'], 'PASS')
        result = self.grade_file()
        self.assertEqual((result['schema'], result['verdict'], result['missing']), ('alpha-c1-grade/1', 'PASS', []))
        self.assertEqual(result['required_arms'], LABELS)
        self.assertEqual({v['engine_watch'] for v in result['validity'].values()}, {'CLIENT_EXIT'})
        self.assertTrue((self.q / 'alpha-c1-report.md').read_text().startswith('# q18a-alpha-c1 report'))

    def test_budget_before_an_arm_is_not_run_and_inconclusive(self):
        out, err, calls, steps = self.run_loop(7, jump_after='ingress-serial-live-1')
        self.assertEqual([c['label'] for c in calls if c['script'] == 'box'], LABELS[:7])
        for name in LABELS[7:]:
            self.assertEqual(steps['arm-' + name], 'INCONCLUSIVE', name)
            self.assertFalse((self.q / ('entry-%s.json' % name)).exists())
        self.assertIn('END FAILS=0 INCS=10 CRASHED=0', out, out[-800:])      # the nine arms and the grade
        result = self.grade_file()
        self.assertEqual(result['verdict'], 'INC')                          # a budget stop is never an engine result
        self.assertEqual((result['failures'], result['classification']), ([], None))
        self.assertIn('declared-arm-missing:serial-live-3', result['missing'])

    def test_a_dead_engine_stops_the_run_and_keeps_the_earlier_arms(self):
        started = time.monotonic()
        out, err, calls, steps = self.run_loop(7, die_in='local-live-1')
        self.assertLess(time.monotonic() - started, 180)
        self.assertIn('CRASHED=1 SOURCE=engine-death', out, out[-800:])
        self.assertEqual(steps['arm-local-live-1'], 'FAIL')
        self.assertEqual([c['label'] for c in calls if c['script'] == 'box'], LABELS[:8])
        for name in LABELS[8:]:
            self.assertEqual(steps['arm-' + name], 'INCONCLUSIVE', name)
        watch = json.loads((self.q / 'engine-watch-arm-local-live-1.json').read_text())
        self.assertEqual(watch['verdict'], 'DEAD')
        result = self.grade_file()
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('local-live-1:engine-dead', result['failures'])
        self.assertEqual((result['validity']['local-live-1']['verdict'], result['validity']['local-live-1']['classification']),
                         ('FAIL', 'ENGINE'))
        self.assertEqual({result['validity'][x]['verdict'] for x in LABELS[:7]}, {'PASS'})


# --- commands, derivation, budget ----------------------------------------------------------------------

ARGPARSE_VALUES = dict(tap.ARGPARSE_VALUES, QNAME='q18a-alpha-c1', RUN_ID='q18a-alpha-c1-fixture', _mode='overlap',
                       _cohort='live', _pair='3', _label='overlap-live-3')


def bash_words(args, draft, values=None):
    budget = tap.le_fix_function(draft, 'native41_budget')
    values = dict(ARGPARSE_VALUES, **(values or {}))
    lines = ['set -u', 'fail_early() { :; }', 'fail() { :; }', 'H=%s' % shlex.quote(str(HERE))]
    lines += ['%s=%s' % (k, shlex.quote(v)) for k, v in values.items()]
    lines += ['GRADE_ARGS=(--arm "$Q/arm-overlap-live-3.json" --ingress "overlap-live-3=$Q/ingress-overlap-live-3.json" '
              '--watch "overlap-live-3=$Q/engine-watch-arm-overlap-live-3.json")',
              'source "$H/runs/q18a_alpha_c1_settings.sh" || exit 97', budget, "printf '%s\\0' " + args]
    done = subprocess.run(['bash', '-c', '\n'.join(lines)], capture_output=True, timeout=60)
    return done.returncode, done.stdout.decode().split('\0')[:-1], done.stderr.decode()[-400:]


class Commands(unittest.TestCase):
    def test_every_body_command_parses_with_its_module(self):
        import e3_native39_trunk_checks as checks
        modules = {'e3_alpha_c1_box': box, 'e3_native39_trunk_checks': checks}
        rows = body_commands(queue.derive()[DRAFT])
        self.assertEqual(sorted({m for m, _ in rows}), sorted(modules))
        actions = []
        original = argparse.ArgumentParser.parse_args

        def capture(parser, args=None, namespace=None):
            raise held.Parsed(original(parser, args, namespace))
        for module, args in rows:
            argv = held.expand(args, dict(tap.QUEUE_VALUES, _mode='overlap', _cohort='live', _pair='3', _label='overlap-live-3',
                                          _workload='/h/workloads/q18a-alpha-perf-band2.json'))
            with mock.patch.object(argparse.ArgumentParser, 'parse_args', capture):
                with self.assertRaises(held.Parsed) as parsed, contextlib.redirect_stderr(io.StringIO()):
                    modules[module].main(argv)
            actions.append((module, argv[0], vars(parsed.exception.args[0])))
        self.assertEqual([a[1] for a in actions], ['instruments', 'full-idle', 'arm', 'ingress', 'instruments', 'full-idle', 'grade'])
        arm = next(ns for m, a, ns in actions if a == 'arm')
        self.assertEqual((arm['mode'], arm['cohort'], arm['index'], arm['control_timeout_s'], arm['workload']),
                         ('overlap', 'live', 3, 60.0, '/h/workloads/q18a-alpha-perf-band2.json'))
        grade = actions[-1][2]
        self.assertEqual((grade['out'], grade['report']), ('/q/alpha-c1-grade.json', '/q/alpha-c1-report.md'))
        self.assertEqual(grade['log'], ['/q/engine-stream.log', '/q/engine-stream.err'])
        ingress = next(ns for m, a, ns in actions if a == 'ingress')
        self.assertEqual(ingress['generation'], box.progress.base.ARM_GENERATIONS)

    def test_the_loop_list_is_the_declared_order_and_the_band_split_runs_in_bash(self):
        draft = queue.derive()[DRAFT]
        (loop,) = re.findall(r'^\s*for _label in ([a-z0-9 -]+); do$', draft, flags=re.M)
        self.assertEqual(loop.split(), LABELS)
        self.assertEqual(list(queue.ARMS), LABELS)
        self.assertEqual(len(LABELS), gate.ARMS)
        self.assertEqual(sum(1 for _s, _c, i in box.ARM_ORDER if i in (0, 1)), gate.BAND1_ARMS)
        self.assertEqual(sum(1 for _s, _c, i in box.ARM_ORDER if i in (2, 3)), gate.BAND2_ARMS)
        head = ''.join(queue.LOOP_NEW.splitlines(True)[1:])
        self.assertIn(head, draft)
        script = ('H=/h\nfor _label in %s; do\n%sprintf "%%s %%s %%s %%s\\n" "$_mode" "$_cohort" "$_pair" "${_workload##*/}"\ndone'
                  % (loop, head))
        done = subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=30)
        self.assertEqual([tuple(x.split()) for x in done.stdout.splitlines()],
                         [(m, c, str(i), BAND1 if i in (0, 1) else BAND2) for m, c, i in box.ARM_ORDER])


class QueueArgparse(unittest.TestCase):
    def test_every_queue_python_call_parses_with_its_real_parser(self):
        texts = queue.derive()
        draft = texts[DRAFT]
        calls = held.queue_python_calls(texts, (HERE / 'qlib.sh').read_text())
        scripts = sorted({c[1] for c in calls})
        self.assertEqual(scripts, sorted([
            'SETCFG', 'e2_memory_attribution.py', 'e3_native28_host_observer.py', 'e3_native30_e_grade.py',
            'e3_native39_trunk_checks.py', 'e3_native39_w2_length.py', 'e3_alpha_c1_box.py',
            'e3_native41_unarmed_box.py', 'e3_native_speculation_gate.py', 'e3_native_startup_grade.py',
            'e3_private_auth.py', 'engine_identity.py', 'native35_ledger_grade.py', 'native43_engine_death.py',
            'native43_allocator_phase_deltas.py']))
        setcfg = next(p for p in (ROOT / 'deploy/setcfg.py', HERE / 'setcfg.reference.py') if p.is_file())
        rows = []
        for name, script, args, form in calls:
            with self.subTest(file=name.split('/')[-1], script=script, args=args[:60]):
                rc, words, err = bash_words(args, draft)
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
        # C0's calls one for one: only the body's client and grade changed module.
        parent = held.queue_python_calls(c0queue.derive(), (HERE / 'qlib.sh').read_text())
        self.assertEqual(len(rows), len(parent))
        ARGPARSE_RECEIPTS[:] = rows

        def first(script, action):
            return next(a for n, s, a, f in calls if s == script and a.split()[0] == action)
        arm, grade = first('e3_alpha_c1_box.py', 'arm'), first('e3_alpha_c1_box.py', 'grade')
        for label, original, (old, new) in (('arm-unknown-option', arm, ('--control-timeout-s', '--control-timeout-sec')),
                                            ('arm-undeclared-mode', arm, ('--mode "$_mode"', '--mode burst')),
                                            ('arm-undeclared-round', arm, ('--index "$_pair"', '--index 6')),
                                            ('grade-without-out', grade, ('--out "$Q/alpha-c1-grade.json"', ''))):
            with self.subTest(control=label):
                self.assertEqual(original.count(old), 1)
                rc, words, err = bash_words(original.replace(old, new), draft)
                self.assertEqual(rc, 0, err)
                self.assertEqual(held.real_parse(HERE / 'e3_alpha_c1_box.py', words)[0], 'refused', label)


class Derivation(unittest.TestCase):
    @staticmethod
    def diff(old, new):
        return [line for line in difflib.ndiff(old.splitlines(), new.splitlines()) if line[:1] in '+-']

    def test_derived_texts_are_on_disk_and_parse(self):
        derived = queue.derive()
        self.assertEqual(sorted(derived), sorted([DRAFT, SETTINGS, DECLARE]))
        for name, text in derived.items():
            self.assertEqual((ROOT / name).read_text(), text, name)
            self.assertEqual(subprocess.run(['bash', '-n', str(ROOT / name)], capture_output=True).returncode, 0, name)
        extra = [line for line in derived[SETTINGS].splitlines() if line.strip().startswith('"EXTRA_ENV=')]
        self.assertEqual(len(extra), 1)
        words = extra[0].split()
        for flag in ('FLASH_S5_AUTO_ARM=1', 'FLASH_S5_NATIVE_PROGRESS=burst', 'FLASH_S5_NATIVE_QSA_POLICY=dense',
                     'FLASH_S5_STAGE_LOCAL=1', 'FLASH_S5_STAGE_OVERLAP=1', 'FLASH_S5_REPLAY_OBSERVER=0', 'FLASH_S5_ENGINE_OFFERS=0'):
            self.assertEqual(words.count(flag), 1, flag)
        for absent in ('FLASH_S5_UNION_RETIRE', 'FLASH_S5_QSA_FUSED', 'FLASH_S5_NATIVE_QSA_TRACE', 'FLASH_S5_NATIVE_QSA_STOP_AFTER'):
            self.assertNotIn(absent, extra[0])

    def test_settings_add_that_token_and_its_comment_nothing_else(self):
        old = queue.rename((ROOT / C0_SETTINGS).read_text())
        new = queue.derive()[SETTINGS]
        diff = self.diff(old, new)
        self.assertEqual([line[2:].replace(' ' + queue.SWITCH, '') for line in diff if line[0] == '+' and 'EXTRA_ENV' in line],
                         [line[2:] for line in diff if line[0] == '-'])
        self.assertEqual([line[2:] for line in diff if line[0] == '+' and 'EXTRA_ENV' not in line], [queue.SETTINGS_NOTE.strip()])

    def test_the_declaration_adds_one_assert_and_the_pins(self):
        old = queue.rename((ROOT / C0_DECLARE).read_text())
        new = queue.derive()[DECLARE]
        diff = self.diff(old, new)
        removed = [x[2:] for x in diff if x[0] == '-']
        self.assertEqual(len(removed), 1)
        self.assertIn(queue.PINS_OLD, removed[0])
        self.assertIn('+ ' + queue.DECLARE_SWITCH.strip(), diff)
        self.assertIn('e3_alpha_c0_box.py alpha_c0_gate.py e3_alpha_c1_box.py alpha_c1_gate.py', new)
        self.assertIn(c0queue.DECLARE_SWITCH.strip(), new)               # C0's switch stays asserted

    def test_everything_kept_is_c0_line_for_line(self):
        old = queue.rename((ROOT / C0_DRAFT).read_text())
        body_old = queue.rename(c0queue.BODY)
        self.assertEqual(old.count(body_old), 1)
        new = queue.derive()[DRAFT]
        self.assertEqual(new.count(queue.BODY), 1)
        diff = self.diff(old.replace(body_old, ''), new.replace(queue.BODY, ''))
        removed = [line[2:] for line in diff if line[0] == '-']
        added = [line[2:] for line in diff if line[0] == '+']
        # Every removed line names C0's gate or grade, or is the interface list's last line; each has its alpha-c1 line.
        self.assertEqual(len(removed), 6, removed)
        for line in removed:
            self.assertTrue('alpha_c0_gate' in line or 'alpha-c0-grade' in line, line)
        declared = queue.HEADER_NOTE.splitlines() + [queue.DRAFT_SWITCH.strip()] + queue.INTERFACE_NEW.splitlines()
        self.assertEqual([x for x in added if not ('alpha_c1_gate' in x or 'alpha-c1-grade' in x) and x not in declared], [], diff)
        # The body: C0's, with the declared swaps only.
        body_diff = self.diff(c0queue.BODY, queue.BODY)
        self.assertEqual(len([x for x in body_diff if x[0] == '-']), 10, body_diff)

    def test_the_draft_reads_alpha_c1_and_keeps_c0s_switch(self):
        draft = queue.derive()[DRAFT]
        for text in ('from alpha_c1_gate import ACTIVE_S,CLEANUP_S,LAST_ARM_ENTRY_S,PHASES',
                     "perf=p.parent/'alpha-c1-grade.json'", 'python3 "$H/e3_alpha_c1_box.py" arm', 'python3 "$H/e3_alpha_c1_box.py" grade',
                     "assert env.get('FLASH_S5_STAGE_LOCAL') == '1'", "assert env.get('FLASH_S5_STAGE_OVERLAP') == '1'",
                     '_workload=$H/workloads/q18a-alpha-perf-band2.json'):
            self.assertIn(text, draft)
        for text in ('from alpha_c0_gate import', 'python3 "$H/e3_alpha_c0_box.py"', 'q18a_alpha_c0', 'q18a-alpha-c0',
                     'alpha-c0-grade', 'window_ok', 'PY_SLOT', 'HARD_END', 'NOT_BEFORE'):
            self.assertNotIn(text, draft)
        self.assertEqual(queue.BODY.count('engine_watched '), 3)
        self.assertEqual(len(re.findall(r'^\s*python3 "\$H/', queue.BODY, flags=re.M)), 0)


class Budget(unittest.TestCase):
    def test_the_box_budget_and_the_arm_entry_rule(self):
        import native40_box_gate as n40
        import native41_box_gate as n41
        self.assertEqual(gate.CLEANUP_S, n40.CLEANUP_S)
        for name in ('NOT_BEFORE', 'HARD_END', 'latest_arm', 'window_ok'):
            self.assertFalse(hasattr(gate, name), name)
        for name in n41.KEPT:
            self.assertEqual(gate.PHASES[name], n40.PHASES[name], name)
        self.assertEqual(gate.ACTIVE_S + gate.CLEANUP_S, gate.BOX_BUDGET_S)
        self.assertEqual(gate.BOX_BUDGET_S, 9000)                # C0 l's; DESIGN.md's 7,200 s left about 150 s (see below)
        self.assertEqual((gate.ARMS, len(queue.ARMS), gate.BAND1_ARMS, gate.BAND2_ARMS), (16, 16, 12, 4))
        self.assertEqual(gate.PHASES['alpha_c1_arms'], 16 * (n41.ARM_TIMEOUT_S + n41.ARM_KILL_AFTER_S))
        self.assertEqual(gate.ARM_RESERVE_S, 450 + 1500 + 30 + 60)
        self.assertEqual(gate.LAST_ARM_ENTRY_S + gate.ARM_RESERVE_S + gate.TAIL_RESERVE_S, gate.ACTIVE_S)
        self.assertEqual(gate.LAST_ARM_ENTRY_S, 4470)
        self.assertTrue(gate.arm_fits(gate.LAST_ARM_ENTRY_S))
        self.assertFalse(gate.arm_fits(gate.LAST_ARM_ENTRY_S + 1))
        self.assertLess(gate.EXPECTED_RUNTIME_S['nominal'], gate.BOX_BUDGET_S)
        body = queue.BODY
        for text, count in (('timeout 120s python3 "$H/e3_native39_trunk_checks.py" instruments', 2),
                            ('timeout 450s python3 "$H/e3_native39_trunk_checks.py" full-idle', 2),
                            ('timeout --signal=TERM --kill-after=30s 1500s python3 "$H/e3_alpha_c1_box.py" arm', 1),
                            ('timeout 60s python3 "$H/e3_native39_trunk_checks.py" ingress', 1),
                            ('timeout 600s python3 "$H/e3_alpha_c1_box.py" grade', 1)):
            self.assertEqual(body.count(text), count, text)
        func = tap.le_fix_function(queue.derive()[DRAFT], 'native41_budget')
        for key, value in (('active', gate.ACTIVE_S), ('cleanup', gate.CLEANUP_S), ('health', gate.PHASES['health']),
                           ('end_identity', gate.PHASES['end_identity']), ('last_arm_entry', gate.LAST_ARM_ENTRY_S)):
            done = subprocess.run(['bash', '-c', func + 'native41_budget ' + key],
                                  env=dict(os.environ, HARNESS_DIR=str(HERE)), capture_output=True, text=True)
            self.assertEqual((done.returncode, done.stdout.strip()), (0, str(value)), done.stderr)

    def test_the_sixteen_arms_fit_on_the_measured_box_walls_with_margin(self):
        # Alpha-perf's box run (1 Nov, file times under /home/<service-user>/s0/q18a-alpha-perf, read with tools/boxro.sh).
        self.assertEqual(gate.BOX_WALLS_S, dict(first_entry=480, cold_solo=246, solo=76, live=147, band2_solo=128, band2_live=273))
        offsets = gate.arm_start_offsets(box.ARM_ORDER)
        self.assertEqual(len(offsets), 16)
        self.assertEqual((offsets[0], offsets[1], offsets[12], offsets[-1]), (480, 726, 1988, 2517))
        self.assertTrue(all(gate.arm_fits(x) for x in offsets))
        # Margin, not only the identity: every arm 1.5 times its serial wall still lets the sixteenth arm enter.
        slow = gate.arm_start_offsets(box.ARM_ORDER, slow=1.5)
        self.assertTrue(all(gate.arm_fits(x) for x in slow), slow[-1])
        self.assertGreater(gate.LAST_ARM_ENTRY_S - offsets[-1], 1900)
        # DESIGN.md's 7,200 s: the same rule leaves about 150 s, and a 1.1 times slower path already cuts the last arm.
        last_7200 = 7200 - gate.CLEANUP_S - gate.ARM_RESERVE_S - gate.TAIL_RESERVE_S
        self.assertEqual(last_7200, 2670)
        self.assertLess(last_7200 - offsets[-1], 160)
        self.assertGreater(gate.arm_start_offsets(box.ARM_ORDER, slow=1.1)[-1], last_7200)
        self.assertLess(offsets[-1] + gate.BOX_WALLS_S['band2_live'], gate.EXPECTED_RUNTIME_S['nominal'])

    def test_no_absolute_time_window_only_the_relative_budget(self):
        draft = queue.derive()[DRAFT]
        self.assertEqual(draft.count(c0queue.SLOT_NEW), 1)                   # C0's no-window note, kept
        for text in ('PY_SLOT', 'window_ok', 'night-window-worst-case-exceeds-restore-margin', 'time.time()'):
            self.assertNotIn(text, draft)
        self.assertEqual(draft.count('exec timeout --signal=TERM --kill-after="$(native41_budget cleanup)s" '
                                     '"$(native41_budget active)s"'), 1)
        self.assertEqual(queue.BODY.count('if (( SECONDS > LAST_ARM_ENTRY_S )); then'), 1)
        self.assertEqual(queue.BODY.count('LAST_ARM_ENTRY_S=$(native41_budget last_arm_entry)'), 1)
        self.assertEqual(SLOT_RECEIPTS, [])


if __name__ == '__main__':
    unittest.main()
