#!/usr/bin/env python3
"""q18a-a1-trunk queue: the derivation, the name rules with the runner's real QNAME, the budget, the derived arm loop under real bash with the real grade, and every python call against its real parser. CPU only; no pass here is box evidence."""
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
import test_inwait_queue as tiq  # noqa: E402  (first: paths)
import a1_trunk_arms as T  # noqa: E402
import a1_trunk_gate as gate  # noqa: E402
import a1_trunk_queue as queue  # noqa: E402
import inwait_queue as parent  # noqa: E402
import inwait_queue_names as names  # noqa: E402
import test_a1_trunk_box as tbox  # noqa: E402

tap, tq, tan = tiq.tap, tiq.tq, tiq.tan
SLOT_RECEIPTS = []
ARGPARSE_RECEIPTS = []
RUNS = HERE / 'runs'
DRAFT = 'harness/runs/q18a_a1_trunk.draft.sh'
SETTINGS = 'harness/runs/q18a_a1_trunk_settings.sh'
DECLARE = 'harness/runs/q18a_a1_trunk_declare.sh'
CORE = T.labels('core')
DECODE = [d['label'] for d in T.DECODE_ARMS]
QNAME = 'q18a-a1-trunk'
STUB_BOX = tiq.STUB_BOX.replace('import e3_inwait_box as real', 'import e3_a1_trunk_box as real')
assert STUB_BOX != tiq.STUB_BOX


class Derivation(unittest.TestCase):
    def test_the_files_are_the_derivation_and_the_parent_is_unchanged(self):
        derived = queue.derive()
        self.assertEqual(sorted(derived), sorted((DRAFT, SETTINGS, DECLARE)))
        for name, text in derived.items():
            self.assertEqual((ROOT / name).read_text(), text, name)
        for name, text in parent.derive().items():
            self.assertEqual((ROOT / name).read_text(), text, name)
        self.assertIn(queue.final_body(), derived[DRAFT])

    def test_the_declared_deltas_are_there(self):
        derived = queue.derive()
        draft, settings, declare = derived[DRAFT], derived[SETTINGS], derived[DECLARE]
        self.assertIn('python3 "$H/e3_a1_trunk_box.py" arm', draft)
        self.assertNotIn('e3_inwait_box.py" arm', draft)
        self.assertIn('--generation "$(python3 "$H/a1_trunk_arms.py" generations "$_label")"', draft)
        self.assertIn('--generation 8 ', draft, 'the decode arm keeps its 8')
        self.assertIn('for _arm in ' + queue.CORE + '; do', draft)
        self.assertNotIn('cap.g2', draft)
        env = re.findall(r'^\s*"EXTRA_ENV=([^"]*)"', settings, re.M)
        self.assertEqual(len(env), 1)
        pairs = dict(tok.split('=', 1) for tok in env[0].split() if '=' in tok)
        for name, value in T.LAUNCH_ENV.items():
            self.assertEqual(pairs[name], str(value), name)
            self.assertIn("assert env.get('%s') == '%d'" % (name, value), draft)
            self.assertIn("assert env.get('%s')=='%d'" % (name, value), declare)
        self.assertEqual((pairs['FLASH_S5_INWAIT'], pairs['FLASH_S5_AUTO_ARM']), ('1', '1'))
        for module in ('e3_inwait_box.py', 'inwait_arms.py', 'e3_native_refusal_client.py', 'inwait_gate.py'):
            self.assertIn(module, draft, 'the box imports it: the preflight lists it')
            self.assertIn(module, declare, 'the box imports it: the declaration lists it')
        self.assertNotRegex(draft, r"inwait-(grade|read|report)", 'the records carry this queue\'s name')

    def test_the_header_states_numerics_are_not_covered(self):
        self.assertIn('NUMERICS ARE NOT COVERED', queue.derive()[DRAFT])


class Names(unittest.TestCase):
    def test_the_real_qname_passes_every_name_rule(self):
        phases = ['q18a_a1_trunk.draft.sh', 'q18a_a1_trunk_declare.sh']
        self.assertEqual(names.check_names(QNAME, phases, RUNS, HERE / 'qlib.sh'), [])
        self.assertEqual(re.findall(r"^CANDIDATE = '([^']+)'$", (ROOT / 'tools/build_a1_trunk.py').read_text(), re.M), [QNAME])
        for text in queue.derive().values():
            for name in re.findall(r'python3 "\$H/([a-z0-9_]+\.py)"', text) + re.findall(r'\$H/runs/([A-Za-z0-9_.]+\.sh)', text):
                self.assertTrue((HERE / name).is_file() or (RUNS / name).is_file(), name)


class Budget(unittest.TestCase):
    def test_relative_and_every_arm_enters_at_the_walls_and_at_one_and_a_half(self):
        source = (HERE / 'a1_trunk_gate.py').read_text()
        self.assertNotRegex(source, r'import (time|datetime)')
        for key in ('nominal', 'slower'):
            est = gate.EXPECTED_RUNTIME_S[key]
            self.assertEqual((est['not_run'], est['arms_entering']), ([], 9), key)
        self.assertLess(gate.EXPECTED_RUNTIME_S['slower']['total_min'], 45, 'about 35 min')
        self.assertIn('a1_trunk_arms', gate.PHASES)


class ArmLoop(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='a1-trunk-loop-'))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.q, self.h, self.bin, self.arms = (self.root / x for x in ('q', 'h', 'bin', 'arms'))
        for d in (self.q, self.h, self.bin, self.arms):
            d.mkdir()
        for name in ('native43_engine_death_h.py', 'a1_trunk_gate.py', 'a1_trunk_arms.py', 'inwait_arms.py', 'attn_meta_arms.py', 'alpha_next_arms.py',
                     'alpha_next_stop_race_h.py', 'native39_trunk_gate.py', 'native40_box_gate.py', 'native41_box_gate.py'):
            (self.h / name).symlink_to(HERE / name)
        (self.h / 'e3_native39_trunk_checks.py').write_text(tap.STUB_CHECKS)
        (self.h / 'e3_a1_trunk_box.py').write_text(STUB_BOX)
        self.tool('nvidia-smi', 'printf "0, 22925\\n1, 22835\\n2, 23037\\n3, 22947\\n"')
        self.tool('docker', 'case "$*" in *State.Running*) echo true;; *) echo 2026-11-01T10:48:09.000000000Z;; esac')
        self.tool('systemctl', 'echo 0')
        (self.q / 'unarmed-stock-prefill.json').write_text(json.dumps(tap.PASS))

    tool = tap.ArmLoop.tool

    def run_loop(self, timeout=420, smoke_rc=0):
        for a in tbox.run_records():
            (self.arms / ('arm-%s.json' % a['label'])).write_text(json.dumps(a))
        emit = tiq.tbox.run_of()[0]    # engine lines of the shape the log readers expect; the startup lines are the ones the grade reads
        tap.write_streams(self.q, [(emit.lines[0][0], 'INFO x %s\n' % line) for line in tbox.tam.STARTUP])
        draft = queue.derive()[DRAFT]
        script = ['set -uo pipefail', 'FAILS=0; INCS=0; declare -A PHASE_VERDICT=()',
                  "ENGINE_CRASHED=0; ENGINE_CRASH_SOURCE=''; STOP_RACE=0; CONTAINER=qwen-flash-prod; SERVICE=vllm-flash; N0=0",
                  'Q=%s; H=%s; PROBE_URL=http://127.0.0.1:1; MODEL=m; RUN_ID=r; EPOCH=1; _cid=c; IDLE_TIMEOUT_S=1' % (self.q, self.h),
                  'Q18A_READINESS=/r/ready; S5_OVERLAY=/o; S5_LAUNCHER_NAME=l.sh', 'health_ok() { return 0; }']
        script += [tap.le_fix_function(draft, name) for name in tap.LOOP_FUNCTIONS]
        script += [queue.final_body(), 'echo "END FAILS=$FAILS INCS=$INCS CRASHED=$ENGINE_CRASHED SOURCE=$ENGINE_CRASH_SOURCE"']
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
        self.assertIn('END FAILS=0 INCS=0 CRASHED=0', out, (out[-2500:], err[-1500:]))
        box = [c for c in calls if c['script'] == 'box']
        self.assertEqual([c.get('label') or c['action'] for c in box], CORE[:1] + ['smoke'] + DECODE + CORE[1:] + ['grade', 'read'])
        self.assertEqual(steps['smoke'], 'PASS')
        rows = T.by_label()
        for c in box:
            if c['action'] == 'arm':
                self.assertEqual(c['c1'], rows[c['label']]['c1'], c['label'])
                self.assertEqual(c['workload'], 'q18a-native41-progress.json', c['label'])
        for name in DECODE + CORE:
            self.assertEqual(steps['arm-' + name], 'PASS', name)
        ingress = {c['out'][len('ingress-'):-len('.json')]: c for c in calls if c['script'] == 'checks' and c['action'] == 'ingress'}
        self.assertEqual(sorted(ingress), sorted(CORE + DECODE))
        self.assertEqual(steps['a1_trunk'], 'PASS', out[-1500:])
        grade = json.loads((self.q / 'a1-trunk-grade.json').read_text())
        self.assertEqual((grade['schema'], grade['verdict'], grade['missing']), ('a1-trunk-grade/1', 'PASS', []))
        self.assertEqual(sorted(grade['arms']), sorted(CORE))
        self.assertTrue((self.q / 'a1-trunk-report.md').read_text().startswith('# q18a-a1-trunk'))
        self.assertEqual(json.loads((self.q / 'a1-trunk-read.json').read_text())['verdict'], 'PASS')

    def test_a_smoke_that_does_not_engage_stops_the_table(self):
        out, err, calls, steps = self.run_loop(smoke_rc=2)
        self.assertEqual(steps['smoke'], 'INCONCLUSIVE', out[-1500:])
        box = [c for c in calls if c['script'] == 'box']
        self.assertEqual([c.get('label') or c['action'] for c in box if c['action'] in ('arm', 'decode-arm')], CORE[:1], 'the smoke arm alone ran')
        for name in DECODE + CORE[1:]:
            self.assertEqual(steps['arm-' + name], 'INCONCLUSIVE', name)
        self.assertIn('NOT_RUN: the exact-path smoke did not engage', out)


class Generations(unittest.TestCase):
    def test_the_table_tells_the_ingress_check_its_count(self):
        for row in T.arms():
            done = subprocess.run([sys.executable, '-B', str(HERE / 'a1_trunk_arms.py'), 'generations', row['label']], capture_output=True, text=True, timeout=30)
            self.assertEqual((done.returncode, done.stdout.strip()), (0, str(row['generations'])), row['label'])


ARGPARSE_VALUES = dict(tiq.ARGPARSE_VALUES, QNAME=QNAME, RUN_ID=QNAME + '-fixture', _label='on.g2-live-1', _sel='on.g2', _c1='on', _group='g2', _arm='core:on.g2-live-1', _pair='1', _cohort='live')


def bash_words(args, draft, values=None):
    """The words bash gives one python call, over this queue's settings and labels (test_inwait_queue.bash_words')."""
    budget = tap.le_fix_function(draft, 'native41_budget')
    values = dict(ARGPARSE_VALUES, **(values or {}))
    lines = ['set -u', 'fail_early() { :; }', 'fail() { :; }', 'H=%s' % shlex.quote(str(HERE))]
    lines += ['%s=%s' % (k, shlex.quote(v)) for k, v in values.items()]
    arrays = '--arm "$Q/arm-on.g2-live-1.json" --ingress "on.g2-live-1=$Q/ingress-on.g2-live-1.json" --watch "on.g2-live-1=$Q/engine-watch-arm-on.g2-live-1.json"'
    lines += ['GRADE_ARGS=(%s)' % arrays, 'READ_ARGS=(%s)' % arrays, 'source "$H/runs/q18a_a1_trunk_settings.sh" || exit 97', budget, "printf '%s\\0' " + args]
    done = subprocess.run(['bash', '-c', '\n'.join(lines)], capture_output=True, timeout=60)
    return done.returncode, done.stdout.decode().split('\0')[:-1], done.stderr.decode()[-400:]


class QueueArgparse(unittest.TestCase):
    def test_every_queue_python_call_parses_with_its_real_parser(self):
        held = tan.held
        texts = queue.derive()
        draft = texts[DRAFT]
        calls = held.queue_python_calls(texts, (HERE / 'qlib.sh').read_text())
        setcfg = next(p for p in (ROOT / 'deploy/setcfg.py', HERE / 'setcfg.reference.py') if p.is_file())
        rows = []
        for name, script, args, form in calls:
            if script == 'a1_trunk_arms.py':
                continue        # no parser: argv read by hand, test Generations runs it
            with self.subTest(file=name.split('/')[-1], script=script, args=args[:60]):
                rc, words, err = bash_words(args, draft, dict(_label='decode-am1', _attn='on') if args.startswith('decode-arm') else None)
                self.assertEqual(rc, 0, err)
                if form != 'python3':
                    words = held.QLIB_CALLS[form][1](words)
                verdict, detail = held.setcfg_parse(setcfg, words) if script == 'SETCFG' else held.real_parse(HERE / script, words)
                self.assertEqual(verdict, 'parsed', (script, words[:4], detail))
                rows.append(dict(file=name.split('/')[-1], script=script, form=form, action=words[0] if words and not words[0].startswith('-') else None, words=len(words), verdict=verdict))
        self.assertTrue(rows)
        ARGPARSE_RECEIPTS[:] = rows
        self.assertEqual(SLOT_RECEIPTS, [])
        arm = next(a for _n, s, a, _f in calls if s == 'e3_a1_trunk_box.py' and a.split()[0] == 'arm')
        for label, (old, new) in (('arm-unknown-option', ('--control-timeout-s', '--control-timeout-sec')), ('arm-undeclared-selection', ('--c1 "${_c1#c1}"', '--c1 ps1')), ('arm-undeclared-round', ('--index "$_pair"', '--index 6'))):
            with self.subTest(control=label):
                self.assertEqual(arm.count(old), 1, label)
                rc, words, err = bash_words(arm.replace(old, new), draft)
                self.assertEqual(rc, 0, err)
                self.assertEqual(held.real_parse(HERE / 'e3_a1_trunk_box.py', words)[0], 'refused', label)


if __name__ == '__main__':
    unittest.main()
