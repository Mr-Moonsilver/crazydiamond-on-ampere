#!/usr/bin/env python3
"""q18a-compose-4 queue: the derivation from a1's committed texts, the name rules with the runner's real QNAME, the budget, the derived arm loop under real bash with the real grade, and every python call against its real parser. CPU only; no pass here is box evidence."""
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
import compose4_arms as T  # noqa: E402
import compose4_gate as gate  # noqa: E402
import compose4_queue as queue  # noqa: E402
import inwait_queue_names as names  # noqa: E402
import test_compose4_box as tbox  # noqa: E402

tap, tq, tan = tiq.tap, tiq.tq, tiq.tan
SLOT_RECEIPTS = []
ARGPARSE_RECEIPTS = []
RUNS = HERE / 'runs'
DRAFT = 'harness/runs/q18a_compose_4.draft.sh'
SETTINGS = 'harness/runs/q18a_compose_4_settings.sh'
DECLARE = 'harness/runs/q18a_compose_4_declare.sh'
CORE = T.labels('core')
DECODE = [d['label'] for d in T.DECODE_ARMS]
QNAME = 'q18a-compose-4'
STUB_BOX = tiq.STUB_BOX.replace('import e3_inwait_box as real', 'import e3_compose4_box as real')
assert STUB_BOX != tiq.STUB_BOX


class Derivation(unittest.TestCase):
    def test_the_files_are_the_derivation(self):
        derived = queue.derive()
        self.assertEqual(sorted(derived), sorted((DRAFT, SETTINGS, DECLARE)))
        for name, text in derived.items():
            self.assertEqual((ROOT / name).read_text(), text, name)
        self.assertIn(queue.final_body(), derived[DRAFT])

    def test_the_declared_deltas_are_there(self):
        derived = queue.derive()
        draft, settings, declare = derived[DRAFT], derived[SETTINGS], derived[DECLARE]
        self.assertIn('python3 "$H/e3_compose4_box.py" arm', draft)
        self.assertNotIn('e3_a1_trunk_box.py" arm', draft)
        self.assertIn('--generation "$(python3 "$H/compose4_arms.py" generations "$_label" "$Q/arm-$_label.json")"', draft)
        self.assertIn('--generation 8 ', draft, 'the decode arm keeps its 8')
        self.assertIn('for _arm in ' + queue.CORE + '; do', draft)
        self.assertNotRegex(draft, r'(on|off|mn|mo)\.g2-')
        env = re.findall(r'^\s*"EXTRA_ENV=([^"]*)"', settings, re.M)
        self.assertEqual(len(env), 1)
        pairs = dict(tok.split('=', 1) for tok in env[0].split() if '=' in tok)
        for name, value in T.LAUNCH_ENV.items():
            self.assertEqual(pairs[name], str(value), name)
        for name, value in (('CUDA_DEVICE_MAX_CONNECTIONS', '32'), ('FLASH_S5_PLE_PRIVATE', '1'), ('FLASH_S5_PLE_WITNESS', '1')):
            self.assertEqual(pairs[name], value, name)
            self.assertIn("assert env.get('%s') == '%s'" % (name, value), draft)
            self.assertIn("assert env.get('%s')=='%s'" % (name, value), declare)
        self.assertEqual((pairs['FLASH_S5_INWAIT'], pairs['FLASH_S5_AUTO_ARM']), ('1', '1'))
        self.assertNotIn('FLASH_S5_CHUNK_TOKENS', pairs, 'the chunk stays the default')
        for module in ('e3_a1_trunk_box.py', 'a1_trunk_arms.py', 'compose4_replay.py', 'agent_replay.py', 'e3_inwait_box.py', 'inwait_arms.py', 'e3_native_refusal_client.py', 'inwait_gate.py'):
            self.assertIn(module, draft, 'the box imports it: the preflight lists it')
            self.assertIn(module, declare, 'the box imports it: the declaration lists it')
        self.assertNotRegex(draft, r'a1-trunk-(grade|read|report)', 'the records carry this queue\'s name')

    def test_the_default_chunk_is_1024(self):
        import async_handoff
        self.assertEqual(async_handoff.DEFAULT_CHUNK_TOKENS, 1024)

    def test_the_header_states_numerics_are_not_covered(self):
        self.assertIn('NUMERICS ARE NOT COVERED', queue.derive()[DRAFT])


class Names(unittest.TestCase):
    def test_the_real_qname_passes_every_name_rule(self):
        phases = ['q18a_compose_4.draft.sh', 'q18a_compose_4_declare.sh']
        self.assertEqual(names.check_names(QNAME, phases, RUNS, HERE / 'qlib.sh'), [])
        self.assertEqual(re.findall(r"^CANDIDATE = '([^']+)'$", (ROOT / 'tools/build_compose4.py').read_text(), re.M), [QNAME])
        for text in queue.derive().values():
            for name in re.findall(r'python3 "\$H/([a-z0-9_]+\.py)"', text) + re.findall(r'\$H/runs/([A-Za-z0-9_.]+\.sh)', text):
                self.assertTrue((HERE / name).is_file() or (RUNS / name).is_file(), name)


class Budget(unittest.TestCase):
    def test_relative_and_every_arm_enters_at_the_walls_and_at_one_and_a_half(self):
        source = (HERE / 'compose4_gate.py').read_text()
        self.assertNotRegex(source, r'import (time|datetime)')
        for key in ('nominal', 'slower'):
            est = gate.EXPECTED_RUNTIME_S[key]
            self.assertEqual((est['not_run'], est['arms_entering']), ([], 5), key)
        self.assertLess(gate.EXPECTED_RUNTIME_S['nominal']['total_min'], 60, 'the slot: at most 60 min, startup included')
        self.assertIn('compose4_arms', gate.PHASES)

    def test_the_arm_timeout_holds_a_replay_and_its_finish_bound(self):
        import compose4_replay as replay
        self.assertLess(T.REPLAY['DURATION'] + replay.FINISH_BOUND_S + 60, 1500, 'the queue kills an arm at 1500 s')
        self.assertIn('--kill-after=30s 1500s python3 "$H/e3_compose4_box.py" arm', queue.derive()[DRAFT])


class ArmLoop(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='compose4-loop-'))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.q, self.h, self.bin, self.arms = (self.root / x for x in ('q', 'h', 'bin', 'arms'))
        for d in (self.q, self.h, self.bin, self.arms):
            d.mkdir()
        for name in ('native43_engine_death_h.py', 'compose4_gate.py', 'compose4_arms.py', 'a1_trunk_arms.py', 'inwait_arms.py', 'attn_meta_arms.py', 'alpha_next_arms.py',
                     'alpha_next_stop_race_h.py', 'native39_trunk_gate.py', 'native40_box_gate.py', 'native41_box_gate.py'):
            (self.h / name).symlink_to(HERE / name)
        (self.h / 'e3_native39_trunk_checks.py').write_text(tap.STUB_CHECKS)
        (self.h / 'e3_compose4_box.py').write_text(STUB_BOX)
        self.tool('nvidia-smi', 'printf "0, 22925\\n1, 22835\\n2, 23037\\n3, 22947\\n"')
        self.tool('docker', 'case "$*" in *State.Running*) echo true;; *) echo 2026-11-01T10:48:09.000000000Z;; esac')
        self.tool('systemctl', 'echo 0')
        (self.q / 'unarmed-stock-prefill.json').write_text(json.dumps(tap.PASS))

    tool = tap.ArmLoop.tool

    def run_loop(self, timeout=420, smoke_rc=0):
        grade = tbox.Grade()
        for a in grade.records():
            (self.arms / ('arm-%s.json' % a['label'])).write_text(json.dumps(a))
        emit = tiq.tbox.run_of()[0]
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
        for name in DECODE + CORE:
            self.assertEqual(steps['arm-' + name], 'PASS', name)
        ingress = {c['out'][len('ingress-'):-len('.json')]: c for c in calls if c['script'] == 'checks' and c['action'] == 'ingress'}
        self.assertEqual(sorted(ingress), sorted(CORE + DECODE))
        self.assertEqual(steps['compose4'], 'PASS', out[-1500:])
        grade = json.loads((self.q / 'compose4-grade.json').read_text())
        self.assertEqual((grade['schema'], grade['verdict'], grade['missing']), ('compose4-grade/1', 'PASS', []))
        self.assertEqual(sorted(grade['arms']), sorted(CORE))
        self.assertTrue((self.q / 'compose4-report.md').read_text().startswith('# q18a-compose-4'))
        self.assertEqual(json.loads((self.q / 'compose4-read.json').read_text())['verdict'], 'PASS')

    def test_a_smoke_that_does_not_engage_stops_the_table(self):
        out, err, calls, steps = self.run_loop(smoke_rc=2)
        self.assertEqual(steps['smoke'], 'INCONCLUSIVE', out[-1500:])
        box = [c for c in calls if c['script'] == 'box']
        self.assertEqual([c.get('label') or c['action'] for c in box if c['action'] in ('arm', 'decode-arm')], CORE[:1], 'the smoke arm alone ran')
        for name in DECODE + CORE[1:]:
            self.assertEqual(steps['arm-' + name], 'INCONCLUSIVE', name)
        self.assertIn('NOT_RUN: the exact-path smoke did not engage', out)


class Generations(unittest.TestCase):
    def call(self, label, record):
        return subprocess.run([sys.executable, '-B', str(HERE / 'compose4_arms.py'), 'generations', label, str(record)], capture_output=True, text=True, timeout=30)

    def test_the_table_tells_the_ingress_check_its_count_and_a_replay_its_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = Path(tmp) / 'arm.json'
            done = self.call(T.SMOKE_ARM, record)
            self.assertEqual((done.returncode, done.stdout.strip()), (0, str(T.by_label()[T.SMOKE_ARM]['generations'])))
            self.assertEqual(self.call('cmp.g2-live-0', record).stdout.strip(), '0', 'no record: the queue runs no check over it, and 0 would not pass one')
            record.write_text(json.dumps(dict(replay_requests_sent=217)))
            self.assertEqual(self.call('cmp.g2-live-0', record).stdout.strip(), '217')


ARGPARSE_VALUES = dict(tiq.ARGPARSE_VALUES, QNAME=QNAME, RUN_ID=QNAME + '-fixture', _label='cmp.g2-live-0', _sel='cmp.g2', _c1='cmp', _group='g2', _arm='core:cmp.g2-live-0', _pair='0', _cohort='live')


def bash_words(args, draft, values=None):
    budget = tap.le_fix_function(draft, 'native41_budget')
    values = dict(ARGPARSE_VALUES, **(values or {}))
    lines = ['set -u', 'fail_early() { :; }', 'fail() { :; }', 'H=%s' % shlex.quote(str(HERE))]
    lines += ['%s=%s' % (k, shlex.quote(v)) for k, v in values.items()]
    arrays = '--arm "$Q/arm-cmp.g2-live-0.json" --ingress "cmp.g2-live-0=$Q/ingress-cmp.g2-live-0.json" --watch "cmp.g2-live-0=$Q/engine-watch-arm-cmp.g2-live-0.json"'
    lines += ['GRADE_ARGS=(%s)' % arrays, 'READ_ARGS=(%s)' % arrays, 'source "$H/runs/q18a_compose_4_settings.sh" || exit 97', budget, "printf '%s\\0' " + args]
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
            if script == 'compose4_arms.py':
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
        arm = next(a for _n, s, a, _f in calls if s == 'e3_compose4_box.py' and a.split()[0] == 'arm')
        for label, (old, new) in (('arm-unknown-option', ('--control-timeout-s', '--control-timeout-sec')), ('arm-undeclared-selection', ('--c1 "${_c1#c1}"', '--c1 ps1')), ('arm-undeclared-round', ('--index "$_pair"', '--index 6'))):
            with self.subTest(control=label):
                self.assertEqual(arm.count(old), 1, label)
                rc, words, err = bash_words(arm.replace(old, new), draft)
                self.assertEqual(rc, 0, err)
                self.assertEqual(held.real_parse(HERE / 'e3_compose4_box.py', words)[0], 'refused', label)


if __name__ == '__main__':
    unittest.main()
