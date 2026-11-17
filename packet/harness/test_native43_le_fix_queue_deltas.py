"""Native43 le-fix queue deltas over the l-e reference texts, with l-e's own logs as the counterfactual.

The queue's real bash functions (step, record, host_crash_detected, capture_logs, engine_watched,
qualification_phase) are cut from the l-e draft before and after the deltas and run in bash.
Leaves: native43_le_fix_queue_fixture_leaves.json.
"""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'tools'))
import native43_le_fix_queue_deltas as deltas  # noqa: E402

FIX = HERE / 'fixtures/native43_engine_death'
# The l-e queue log's recorded line (fn-main results/box/q18a-native43-lifecycle-run-1, queue log).
LE_RECORDED = 'STEP engine-errors PASS no current-run engine error marker'


def function(text, name):
    lines = text.splitlines(keepends=True)
    start = [i for i, line in enumerate(lines) if line == name + '() {\n']
    assert len(start) == 1, (name, start)
    end = next(i for i in range(start[0] + 1, len(lines)) if lines[i] == '}\n')
    return ''.join(lines[start[0]:end + 1])


class Texts(unittest.TestCase):
    def test_deltas_apply_once_and_the_result_parses(self):
        draft, declare = deltas.reference_texts()
        new_draft, new_declare = deltas.apply_draft(draft), deltas.apply_declare(declare)
        for text in (new_draft, new_declare):
            path = Path(tempfile.mkdtemp(prefix='n43-le-fix-')) / 'q.sh'
            self.addCleanup(shutil.rmtree, path.parent, ignore_errors=True)
            path.write_text(text)
            self.assertEqual(subprocess.run(['bash', '-n', str(path)]).returncode, 0)
        self.assertEqual(new_draft.count('engine_watched "qual-$name" python3 "$H/$cli"'), 1)
        self.assertEqual(new_draft.count('engine_watched lifecycle python3 "$H/e3_q18a_lifecycle.py"'), 1)
        self.assertEqual(new_draft.count('engine_watched hygiene-full python3 - '), 1)
        self.assertIn('native43_engine_death.py native43_lifecycle_gate.py', new_draft)
        self.assertIn('native43_lifecycle_grade.py native43_engine_death.py', new_declare)
        # Only the declared lines change: every original line survives in order.
        old_lines, new_lines = draft.splitlines(), new_draft.splitlines()
        changed = {deltas.D1_OLD.rstrip('\n'), deltas.D5_OLD.rstrip('\n'), deltas.D6_OLD.rstrip('\n'),
                   deltas.D7_OLD.rstrip('\n')}
        it = iter(new_lines)
        for line in old_lines:
            if any(line.startswith(c) for c in changed):
                continue
            self.assertIn(line, it)
        with self.assertRaises(AssertionError):
            deltas.apply_draft(new_draft)


class Bash(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='n43-le-fix-'))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.draft, _ = deltas.reference_texts()
        self.fixed = deltas.apply_draft(self.draft)
        self.q = self.root / 'q'
        self.q.mkdir()
        self.h = self.root / 'h'
        self.h.mkdir()
        (self.h / 'native43_engine_death.py').symlink_to(HERE / 'native43_engine_death.py')
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        self.tool('nvidia-smi', 'printf "0, 22925\\n1, 22835\\n2, 23037\\n3, 22947\\n"')
        self.tool('docker', 'case "$1" in\n'
                  '  inspect) case "$*" in *State.Running*) echo true;; *) echo 2026-11-01T10:48:09.000000000Z;; esac;;\n'
                  '  logs) cat "$Q/docker-out.log"; cat "$Q/docker-err.log" >&2;;\n'
                  'esac')

    def tool(self, name, body):
        path = self.bin / name
        path.write_text('#!/bin/bash\n' + body + '\n')
        path.chmod(0o755)

    def stream(self, log_lines, err_lines=(), intent=None):
        data = b''.join(line + b'\n' for line in log_lines)
        err = b''.join(line + b'\n' for line in err_lines)
        for name, blob in (('engine-stream.log', data), ('docker-out.log', data),
                           ('engine-stream.err', err), ('docker-err.log', err)):
            (self.q / name).write_bytes(blob)
        if intent is not None:
            shutil.copy(intent, self.q / 'host-stop-intent.json')

    def bash(self, text, names, body, timeout=120):
        script = ['set -uo pipefail', 'FAILS=0; INCS=0; declare -A PHASE_VERDICT=()',
                  "ENGINE_CRASHED=0; ENGINE_CRASH_SOURCE=''; STARTED=1; LOGS_DONE=0; CONTAINER=qwen-flash-prod",
                  'Q=%s; H=%s' % (self.q, self.h)]
        script += [function(text, name) for name in names] + [body,
                   'echo "CRASHED=$ENGINE_CRASHED SOURCE=$ENGINE_CRASH_SOURCE"']
        path = self.root / 'run.sh'
        path.write_text('\n'.join(script) + '\n')
        env = dict(os.environ, PATH='%s:%s' % (self.bin, os.environ['PATH']), Q=str(self.q))
        started = time.monotonic()
        proc = subprocess.Popen(['bash', str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                env=env, start_new_session=True)
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)  # the hung client too, not only bash
            proc.communicate()
            raise
        return subprocess.CompletedProcess(proc.args, proc.returncode, out, err), time.monotonic() - started

    def le_lines(self, until=None):
        lines = (FIX / 'le-engine-stream.log').read_bytes().split(b'\n')[:-1]
        if until is not None:
            lines = lines[:next(i for i, line in enumerate(lines) if until in line)]
        return lines

    def capture(self, text):
        return self.bash(text, ('step', 'record', 'host_crash_detected', 'capture_logs'), 'capture_logs')[0]

    def test_le_capture_counterfactual(self):
        # capture_logs runs before stop_engine: the stream up to the queue's stop, no intent yet.
        self.stream(self.le_lines(until=b'API server: shutdown triggered'),
                    (FIX / 'le-engine-stream.err').read_bytes().split(b'\n')[:-2])
        before = self.capture(self.draft)
        self.assertIn(LE_RECORDED, before.stdout)
        self.assertIn('CRASHED=0 SOURCE=', before.stdout)
        after = self.capture(self.fixed)
        self.assertIn('STEP engine-errors FAIL rc=1 retained host stream proves engine crash', after.stdout)
        self.assertIn('CRASHED=1 SOURCE=engine-death-scan', after.stdout)
        scan = json.loads((self.q / 'engine-death-scan.json').read_text())
        self.assertEqual((scan['first_death']['kind'], scan['first_death']['wall']),
                         ('executor_teardown', '2026-11-01T11:03:08.286892993Z'))

    def test_orderly_trunk_tail_is_not_a_crash_before_or_after_the_stop(self):
        lines = (FIX / 'trunk-engine-stream.log').read_bytes().split(b'\n')[:-1]
        cut = next(i for i, line in enumerate(lines) if b'API server: shutdown triggered' in line)
        self.stream(lines[:cut])
        for text in (self.draft, self.fixed):
            out = self.capture(text).stdout
            self.assertIn('STEP engine-errors PASS no current-run engine error marker', out)
            self.assertIn('CRASHED=0', out)
        # finish_overlap/on_exit time: the full orderly stop and its intent are in the stream.
        self.stream(lines, intent=FIX / 'trunk-host-stop-intent.json')
        result, _ = self.bash(self.fixed, ('host_crash_detected',), 'host_crash_detected; echo "HCD=$?"')
        self.assertIn('HCD=1', result.stdout)
        self.assertIn('CRASHED=0', result.stdout)

    def phase_env(self):
        return ('PROBE_URL=http://127.0.0.1:1; MODEL=m; RUN_ID=r; EPOCH=1; Q18A_STORE=%s/store; '
                'SNAPSHOT_MAX_BYTES=1' % self.q)

    def qualification(self, text, client_body, feed_after=None, timeout=120):
        (self.h / 'e3_q18a_qualification.py').write_text(client_body)
        if feed_after is not None:
            feeder = subprocess.Popen([sys.executable, '-c', (
                'import pathlib,sys,time\n'
                'q=pathlib.Path(sys.argv[1]); time.sleep(float(sys.argv[2]))\n'
                'with (q/"engine-stream.log").open("ab") as h: h.write(pathlib.Path(sys.argv[3]).read_bytes())\n'),
                str(self.q), str(feed_after), str(self.root / 'teardown.line')])
            self.addCleanup(feeder.wait, 30)
        names = ('step', 'record') + (('engine_watched',) if 'engine_watched() {' in text else ()) + (
            'qualification_phase',)
        return self.bash(text, names, self.phase_env() + '\nqualification_phase row1', timeout=timeout)

    def test_hung_row1_client_on_the_le_teardown(self):
        lines = self.le_lines()
        cut = next(i for i, line in enumerate(lines) if b'[shutdown] Executor' in line)
        (self.root / 'teardown.line').write_bytes(lines[cut] + b'\n')
        self.stream(lines[:cut])
        hang = 'import time\ntime.sleep(600)\n'
        # Before: the l-e client never returns on a dead engine (bounded here at 8 s by the test).
        with self.assertRaises(subprocess.TimeoutExpired):
            self.qualification(self.draft, hang, feed_after=1.0, timeout=8)
        self.stream(lines[:cut])
        result, wall = self.qualification(self.fixed, hang, feed_after=1.0)
        self.assertLess(wall, 40)
        self.assertIn('STEP row1 FAIL rc=1 raw qualification artifact retained; no repeat or picking', result.stdout)
        self.assertIn('CRASHED=1 SOURCE=engine-death-watch', result.stdout)
        watch = json.loads((self.q / 'engine-watch-qual-row1.json').read_text())
        self.assertEqual((watch['verdict'], watch['dead']['reason'], watch['dead']['first_death']['wall']),
                         ('DEAD', 'engine-log', '2026-11-01T11:03:08.286892993Z'))
        self.assertEqual(watch['container'], 'qwen-flash-prod')
        log = (self.q / 'qual-row1.log').read_text()
        self.assertIn('ENGINE_DEATH_WATCH DEAD reason=engine-log', log)

    def test_healthy_row1_client_is_unchanged(self):
        self.stream(self.le_lines(until=b'[shutdown] Executor'))
        ok = ('import json,sys\n'
              'out=sys.argv[sys.argv.index("--out")+1]\n'
              'json.dump({"verdict":"PASS"},open(out,"w"))\n')
        for text in (self.draft, self.fixed):
            (self.q / 'qual-row1.json').unlink(missing_ok=True)
            (self.q / 'engine-watch-qual-row1.json').unlink(missing_ok=True)
            result, _ = self.qualification(text, ok)
            self.assertIn('STEP row1 PASS raw qualification artifact retained; no repeat or picking', result.stdout)
            self.assertIn('CRASHED=0', result.stdout)
        watch = json.loads((self.q / 'engine-watch-qual-row1.json').read_text())
        self.assertEqual((watch['verdict'], watch['client_rc'], watch['last_gpu'], watch['last_container']),
                         ('CLIENT_EXIT', 0, [22925, 22835, 23037, 22947], 'running'))


if __name__ == '__main__':
    unittest.main()
