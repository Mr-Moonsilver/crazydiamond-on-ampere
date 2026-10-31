"""Native35 P-c: P-b with FLASH_NATIVE30_LOCALIZE=0, the second declared diagnostic switch vs O.

Proves, on CPU:
(1) The P-c queue texts are the P-b texts under fresh names with exactly the
    declared deltas:
    - settings LOCALIZE 1 -> 0;
    - native35-diagnostic as an explicit NOT_RUN record;
    - the post-row4 step note;
    - the header comment.
    The row4 phase, the gates and the whole W1/W2 measurement tail are P-b's,
    byte for byte. native30-localize still runs, unchanged.
(2) Every P-c script parses.
(3) The P-c cleanup lines run under bash with the queue's own record/step and
    the real graders, over an engine log shaped like a LOCALIZE=0 run.
    native30-localize is INCONCLUSIVE with zero begins on every rank;
    native35-diagnostic is INCONCLUSIVE NOT_RUN and writes no grade.
    Counterfactuals on the same inputs:
    - P-b's lines print a vacuous native35 PASS, which is why P-c withholds it;
    - rows from the real LOCALIZE=1 emitter give non-zero begins, so zero
      begins discriminates.
The in-image check that the plugins stay inert under the P-c settings env is
the self-run's native35_p_c_inert_check, with a positive control under the
P-b settings.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import native30_localize as localize

HERE = Path(__file__).resolve().parent
RUNS = HERE / 'runs'
CLEANUP_RECEIPTS = []
# Declared deltas, retyped here independently of tools/native35_p_c_queue_drafts.py.
PB_HEADER = ('# Native35-p-b: native28 startup -> row1 -> rule4_controls -> rule4_candidate -> row4 '
             '(L2 client window at the accepted resume receipt) -> W1 -> two retained W2 rounds -> owned restore.\n')
PC_HEADER = ('# Native35-p-c: P-b with FLASH_NATIVE30_LOCALIZE=0 (plugins present, inert): native28 startup -> row1 -> '
             'rule4_controls -> rule4_candidate -> row4 (L2 client window at the accepted resume receipt) -> W1 -> '
             'two retained W2 rounds -> owned restore.\n')
PB_DIAGNOSTIC = ('  python3 "$H/native35_grade.py" --log "$Q/engine-stream.err" --log "$Q/engine-stream.log" '
                 '--arm-record "$Q/qual-row4.json" --out "$Q/native35-grade.json" > "$Q/native35-grade.log" 2>&1\n'
                 "  record native35-diagnostic \"$?\" 'unfenced O-style arm; no qualification claim'\n")
PC_DIAGNOSTIC = "  record native35-diagnostic 2 'NOT_RUN: LOCALIZE=0 declared; no receipts to read'\n"
PB_TAIL_NOTE = "  step native30-receipt-tail 0 '30 seconds after row4 client return; original clients already joined'\n"
PC_TAIL_NOTE = "  step native30-receipt-tail 0 '30 s post-row4 gap kept from P-b; LOCALIZE=0: no receipts, no receipt tail'\n"
LOCALIZE_GRADE = ('  python3 "$H/e3_native30_e_grade.py" --log "$Q/engine-stream.log" --log "$Q/engine-stream.err" --placement "$Q/qual-row4.json" \\\n'
                  '    --out "$Q/localize-grade.json" --engine-crashed "$ENGINE_CRASHED" > "$Q/localize-grade.log" 2>&1\n'
                  "  record native30-localize \"$?\" 'all-rank localization; no qualification or performance claim'\n")
PLUGIN_PATH = 'PYTHONPATH=/usr/local/lib/python3.12/dist-packages/vllm/v1/flash_e3/native30_plugins'
# Ordinary engine lines: what a LOCALIZE=0 engine log holds instead of NATIVE30 rows.
ENGINE_LINES = ('(EngineCore_DP0 pid=212) INFO 10-31 18:01:02 [core.py:94] Initializing a V1 LLM engine with config: model=qwen38-w8a8-tp4\n'
                '(APIServer pid=1) INFO:     127.0.0.1:41822 - "POST /v1/completions HTTP/1.1" 200 OK\n'
                '(VllmWorker TP0 pid=318) INFO 10-31 18:01:40 [gpu_worker.py:298] flash-sched stats step 53600 cohorts even=4 odd=4\n')


def text(name):
    return (RUNS / name).read_text()


def once(source, old, new):
    assert source.count(old) == 1, (old[:70], source.count(old))
    return source.replace(old, new)


def fresh_names(source):
    assert 'q18a-native35-p-c' not in source and 'q18a_native35_p_c' not in source
    return source.replace('q18a-native35-p-b', 'q18a-native35-p-c').replace('q18a_native35_p_b', 'q18a_native35_p_c')


def phase(source):
    start = source.index('qualification_phase() {')
    return source[start:source.index('\n}\n', start) + 3]


class Derivation(unittest.TestCase):
    def test_pc_is_pb_under_fresh_names_with_exactly_the_declared_deltas(self):
        pb, pc = text('q18a_native35_p_b.draft.sh'), text('q18a_native35_p_c.draft.sh')
        expected = pb
        for old, new in ((PB_HEADER, PC_HEADER), (PB_DIAGNOSTIC, PC_DIAGNOSTIC), (PB_TAIL_NOTE, PC_TAIL_NOTE)):
            expected = once(expected, old, new)
        self.assertEqual(pc, fresh_names(expected))
        settings = text('q18a_native35_p_c_settings.sh')
        self.assertEqual(settings, fresh_names(once(text('q18a_native35_p_b_settings.sh'),
                                                    ' FLASH_NATIVE30_LOCALIZE=1 ', ' FLASH_NATIVE30_LOCALIZE=0 ')))
        self.assertEqual(text('q18a_native35_p_c_declare.sh'), fresh_names(text('q18a_native35_p_b_declare.sh')))
        # Row4 is still the L2 collector; native30-localize still runs, unchanged.
        self.assertEqual(phase(pc), phase(pb))
        self.assertEqual(phase(pc), phase(text('q18a_native35_l2.draft.sh')))
        self.assertEqual((pb.count(LOCALIZE_GRADE), pc.count(LOCALIZE_GRADE)), (1, 1))
        self.assertNotIn('native35_grade.py', pc)
        self.assertEqual(pc.count('  sleep 30\n'), 1)
        # The plugins stay on PYTHONPATH; every switch they read is off.
        self.assertEqual(settings.count('FLASH_NATIVE30_LOCALIZE='), 1)
        self.assertIn(' FLASH_NATIVE30_LOCALIZE=0 FLASH_NATIVE33_DIAG=0 FLASH_NATIVE34_DIAG=0 FLASH_NATIVE30_SYNC=${FLASH_NATIVE30_SYNC:-none} ', settings)
        self.assertIn(' ' + PLUGIN_PATH + '"', settings)
        for absent in ('FLASH_NATIVE30_LOCALIZE=1', 'FLASH_NATIVE34_ARM', 'FLASH_NATIVE34_READBACK', 'CUDA_LAUNCH_BLOCKING'):
            self.assertNotIn(absent, settings)
        self.assertIn('\nQ18A_FIRST_ADVANCE_READ_TIMEOUT_S=1200\n', settings)

    def test_every_pc_script_parses_and_no_global_blocking(self):
        for suffix in ('.draft.sh', '_settings.sh', '_declare.sh'):
            path = RUNS / ('q18a_native35_p_c' + suffix)
            self.assertEqual(subprocess.run(['bash', '-n', str(path)], capture_output=True).returncode, 0, path.name)
            self.assertNotIn('set -e\n', path.read_text())
            self.assertNotIn('CUDA_LAUNCH_BLOCKING=1', path.read_text())


def emitted_rows():
    """LOCALIZE=1 begin/end rows from the real native30_localize.emit, all four ranks."""
    sink = io.StringIO()
    with contextlib.redirect_stderr(sink):
        call = 0
        for rank in range(4):
            for label in ('FIRST_ADMISSION', 'DECODE_STEP'):
                call += 1
                fields = dict(rank=rank, label=label, call_id=call)
                localize.emit('begin', **fields)
                localize.emit('end', last_memory=None, **fields)
    return sink.getvalue()


class Cleanup(unittest.TestCase):
    """The queue's own cleanup lines for native30-localize and native35-diagnostic, with the real graders."""

    def run_cleanup(self, draft, directory, err_extra=''):
        source = text(draft)
        functions = ''.join(source[source.index(n + '() {\n'):source.index('\n}\n', source.index(n + '() {\n')) + 3] for n in ('step', 'record'))
        begin = source.index('  python3 "$H/e3_native30_e_grade.py"')
        end = source.index('\n', source.index('  record native35-diagnostic', begin)) + 1
        q = directory / 'q'
        q.mkdir()
        (q / 'engine-stream.log').write_text(ENGINE_LINES)
        (q / 'engine-stream.err').write_text(ENGINE_LINES + err_extra)
        (q / 'qual-row4.json').write_text(json.dumps({'verdict': 'PASS', 'phase': 'row4'}))
        script = ('set -uo pipefail\nFAILS=0; INCS=0; ENGINE_CRASHED=0\ndeclare -A PHASE_VERDICT=()\n' + functions + source[begin:end] +
                  'echo "TOTAL FAILS=$FAILS INCS=$INCS"\n')
        env = dict(os.environ, H=str(HERE), Q=str(q), PYTHONDONTWRITEBYTECODE='1')
        done = subprocess.run(['bash', '-c', script], env=env, capture_output=True, text=True, timeout=120, cwd=str(directory))
        self.assertEqual(done.returncode, 0, done.stderr)
        steps = [line for line in done.stdout.splitlines() if line.startswith(('STEP ', 'TOTAL '))]
        return steps, q

    def test_pc_cleanup_withholds_the_diagnostic_and_localize_reads_zero_begins(self):
        with tempfile.TemporaryDirectory() as d:
            steps, q = self.run_cleanup('q18a_native35_p_c.draft.sh', Path(d))
            self.assertEqual(steps, ['STEP native30-localize INCONCLUSIVE all-rank localization; no qualification or performance claim',
                                     'STEP native35-diagnostic INCONCLUSIVE NOT_RUN: LOCALIZE=0 declared; no receipts to read',
                                     'TOTAL FAILS=0 INCS=2'])
            graded = json.loads((q / 'localize-grade.json').read_text())
            self.assertFalse((q / 'native35-grade.json').exists())
        self.assertEqual((graded['verdict'], graded['failure_class']), ('INCONCLUSIVE', 'HARNESS'))
        self.assertEqual({r: row['begins'] for r, row in graded['per_rank'].items()}, {'0': 0, '1': 0, '2': 0, '3': 0})
        self.assertFalse(graded['all_rank_admission_and_decode'])
        self.assertEqual((graded['malformed_receipts'], graded['missing_logs']), (0, []))
        self.assertTrue(all(v[k] is None for v in graded['host_coverage'].values() for k in ('admission', 'join', 'end')))
        CLEANUP_RECEIPTS.append(dict(case='p-c-localize-off', verdict='PASS', native30_localize='INCONCLUSIVE', begins_per_rank=0,
                                     native35_diagnostic='NOT_RUN', native35_grade_written=False))

    def test_pb_cleanup_on_the_same_log_prints_a_vacuous_native35_pass(self):
        with tempfile.TemporaryDirectory() as d:
            steps, q = self.run_cleanup('q18a_native35_p_b.draft.sh', Path(d))
            graded = json.loads((q / 'native35-grade.json').read_text())
        self.assertIn('STEP native35-diagnostic PASS unfenced O-style arm; no qualification claim', steps)
        self.assertEqual((graded['verdict'], graded['attribution'], graded['first_exception']), ('PASS', 'NO_DEVICE_FAULT_RECORDED', None))
        self.assertEqual((graded['malformed'], graded['missing_fragments'], graded['receipt_limits']), (0, 0, 0))
        CLEANUP_RECEIPTS.append(dict(case='p-b-lines-on-localize-off-log', verdict='PASS', native35_diagnostic='PASS-from-zero-rows'))

    def test_real_emitter_rows_give_nonzero_begins_so_zero_discriminates(self):
        with tempfile.TemporaryDirectory() as d:
            steps, q = self.run_cleanup('q18a_native35_p_c.draft.sh', Path(d), err_extra=emitted_rows())
            graded = json.loads((q / 'localize-grade.json').read_text())
        self.assertEqual({r: row['begins'] for r, row in graded['per_rank'].items()}, {'0': 2, '1': 2, '2': 2, '3': 2})
        self.assertTrue(graded['all_rank_admission_and_decode'] and graded['all_begins_paired'])
        self.assertEqual(graded['verdict'], 'INCONCLUSIVE')  # no host-policy coverage rows in this log
        self.assertIn('STEP native35-diagnostic INCONCLUSIVE NOT_RUN: LOCALIZE=0 declared; no receipts to read', steps)
        CLEANUP_RECEIPTS.append(dict(case='localize-on-rows-counterfactual', verdict='PASS', begins_per_rank=2))


if __name__ == '__main__':
    unittest.main()
