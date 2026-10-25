"""Reserve shortfall can continue diagnostically; all other failures stop.

The extracted real queue fragment plus qlib must retain FAIL even when a later
diagnostic is reached. No engine commands or GPU calls in these tests.
"""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from e2_diagnostic_continue import decide
from e2_ledger_grade import RANKS, grade

HERE = Path(__file__).resolve().parent
DRAFT = HERE / 'runs/q16_e2_serial_v3f.draft.sh'


def report():
    end = {'per_rank': {r: {'alloc_retries': 4, 'ooms': 0, 'peak_gib': 20.6, 'reserved_gib': 22.5} for r in RANKS}}
    peaks = {'samples': 262, 'gpus': {str(i): {'max_used_mib': 24115, 'total_mib': 24576} for i in range(4)}}
    return grade({}, end, peaks, reserve_only=True)


class ContinuationTests(unittest.TestCase):
    def test_only_reserve_failure_can_continue_without_rewriting_report(self):
        r = report(); saved = copy.deepcopy(r)
        d = decide(r, 1)
        self.assertEqual(d['action'], 'CONTINUE_DIAGNOSTIC')
        self.assertEqual(d['qualification_verdict'], 'FAIL')
        self.assertEqual(r, saved)

    def test_oom_counter_error_retry_failure_and_unknown_failure_stop(self):
        for failure in ('Worker_PP0_TP0: ooms', 'Worker_PP0_TP0: retry counter went backwards',
                        'Worker_PP0_TP0: more allocator retries since start than the matched control', 'unknown'):
            with self.subTest(failure=failure):
                r = report(); r['fails'].append(failure)
                self.assertEqual(decide(r, 1)['action'], 'STOP')
        r = report(); r['ranks'][RANKS[0]]['ooms'] = 1
        self.assertEqual(decide(r, 1)['action'], 'STOP')

    def test_incomplete_or_abnormal_evidence_does_not_get_waiver(self):
        for rc in (0, 2, 126, 137, -11):
            self.assertEqual(decide(report(), rc)['action'], 'STOP')
        for key in ('ranks', 'gpus'):
            r = report(); r[key].pop(next(iter(r[key])))
            self.assertEqual(decide(r, 1)['action'], 'STOP')
        r = report(); r['missing'] = ['a ledger row is absent']
        self.assertEqual(decide(r, 1)['action'], 'STOP')

    def test_altered_rule_nonfinite_and_no_positive_headroom_stop(self):
        r = report(); r['reserve_gib'] = 0.4
        self.assertEqual(decide(r, 1)['action'], 'STOP')
        for value in (float('nan'), float('inf'), -1):
            r = report(); r['ranks'][RANKS[0]]['peak_gib'] = value
            self.assertEqual(decide(r, 1)['action'], 'STOP')
        r = report(); r['gpus']['0']['max_used_gib'] = 24
        self.assertEqual(decide(r, 1)['action'], 'STOP')

    def test_cli_malformed_report_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'bad.json'
            for raw in ('{', '[]', '{"ranks":null}', 'null'):
                p.write_text(raw)
                x = subprocess.run(['python3', str(HERE/'e2_diagnostic_continue.py'), str(p), '1'], capture_output=True, text=True)
                self.assertEqual(x.returncode, 1, x.stderr)
                self.assertEqual(json.loads(x.stdout)['action'], 'STOP')

    def test_real_queue_continues_but_final_verdict_stays_fail(self):
        text = DRAFT.read_text()
        a = text.index('  # DIAGNOSTIC_RESERVE_BEGIN')
        b = text.index('  # DIAGNOSTIC_RESERVE_END', a)
        fragment = text[a:b]
        condition = next(line.strip() for line in text.splitlines() if line.strip().startswith('[[ $rc == 0 && $er == 0'))
        with tempfile.TemporaryDirectory() as tmp:
            q = Path(tmp)
            for bad, sampler in ((False, 0), (True, 0), (False, 1)):
                r = report()
                if bad: r['fails'].append('Worker_PP0_TP0: ooms')
                (q/'dense.memory.json').write_text(json.dumps(r))
                script = '''source "$1/qlib.sh"
H=$1; Q=$2; name=dense; mr=1; rc=0; er=0; ir=0; sr=0; pr=$3; memory_continue=0
step dense.memory-contract "$mr" 'raw FAIL kept'
phase() {
'''+fragment+condition+'''
}
if phase; then echo NEXT_DIAGNOSTIC_REACHED; fi
finish Q16-TEST
'''
                x = subprocess.run(['bash', '-c', script, 'test', str(HERE), str(q), str(sampler)], capture_output=True, text=True)
                self.assertEqual(x.returncode, 1, x.stdout+x.stderr)
                self.assertIn('dense.memory-contract FAIL', x.stdout)
                self.assertIn('fails=1', x.stdout)
                self.assertEqual('NEXT_DIAGNOSTIC_REACHED' in x.stdout, not bad and sampler == 0)

    def test_queue_does_not_change_grader_or_skip_safety_dependencies(self):
        text = DRAFT.read_text()
        self.assertIn('step "$name.memory-contract" "$mr"', text)
        self.assertNotIn('mr=0', text)
        self.assertIn('$rc == 0 && $er == 0 && $ir == 0 && $sr == 0 && $pr == 0', text)
        for f in (DRAFT, HERE/'runs/q16j.sh', HERE/'runs/q16j_declare.sh'):
            subprocess.run(['bash', '-n', str(f)], check=True)


if __name__ == '__main__':
    unittest.main()
