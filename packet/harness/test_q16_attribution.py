"""q16 v6 attribution fragment (Codex v5 review truth table, reviews/q16_v5_review_repro.py style): the real step()/finish()
from qlib.sh, every measured operation stubbed, the extracted queue fragment between the matched-retry grade and the A/B
step. Hard memory failures exit before the exposure comparison; a retry-only FAIL reaches it; unreadable exposure is
INCONCLUSIVE; attribution PASS only with PASS + MATCHED, FAIL only with retry-only + MATCHED. CPU only."""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
DRAFT = HERE / 'runs' / 'q16_e2_serial_v6.draft.sh'


def fragment():
    q = DRAFT.read_text()
    start = q.index('memory_matched > "$Q/shipping.memory-matched.log"')
    stop = q.index('run ab-inproc ab_inproc', start)
    return q[start:stop]


def run(memory_rc, retry_only, exposure_rc):
    with tempfile.TemporaryDirectory() as d:
        json.dump(dict(verdict='FAIL' if memory_rc == 1 else 'PASS', retry_only=retry_only, fails=['x: more allocator retries' if retry_only else 'x: ooms'] if memory_rc == 1 else []),
                  open(os.path.join(d, 'shipping.memory-matched.json'), 'w'))
        prelude = f'''
source "$1"
memory_matched() {{ echo grade; return {memory_rc}; }}
real_python3=$(command -v python3)
python3() {{ if [[ $1 == "$H/exposure_compare.py" ]]; then echo EXPOSURE_REACHED; return {exposure_rc}; fi; "$real_python3" "$@"; }}
Q={d}
H=/unused
DONE=0
'''
        r = subprocess.run(['bash', '-c', prelude + fragment() + '\necho END', 'test', str(HERE / 'qlib.sh')], text=True, capture_output=True)
        return r.returncode, r.stdout


class AttributionTests(unittest.TestCase):
    def test_hard_memory_failure_exits_before_exposure(self):
        rc, out = run(1, False, 0)
        self.assertEqual(rc, 1); self.assertNotIn('EXPOSURE_REACHED', out); self.assertIn('STEP shipping.memory-matched FAIL', out)

    def test_retry_only_failure_reaches_exposure_and_attributes_only_when_matched(self):
        rc, out = run(1, True, 0)   # retry-only + MATCHED exposure: attributable, the attribution step fails the gate
        self.assertEqual(rc, 1, out); self.assertIn('EXPOSURE_REACHED', out)
        self.assertIn('STEP shipping.memory-matched INCONCLUSIVE', out); self.assertIn('STEP shipping.attribution FAIL', out)
        rc, out = run(1, True, 2)
        self.assertEqual(rc, 0); self.assertIn('STEP shipping.attribution INCONCLUSIVE', out); self.assertIn('STEP shipping.exposure INCONCLUSIVE', out)

    def test_pass_attributes_only_with_matched_exposure_and_unreadable_is_inconclusive(self):
        rc, out = run(0, False, 0)
        self.assertEqual(rc, 0); self.assertIn('STEP shipping.attribution PASS', out)
        rc, out = run(0, False, 2)
        self.assertIn('STEP shipping.attribution INCONCLUSIVE', out)
        rc, out = run(0, False, 1)
        self.assertIn('STEP shipping.exposure INCONCLUSIVE', out); self.assertIn('STEP shipping.attribution INCONCLUSIVE', out)
        self.assertNotIn('FAIL', out)

    def test_inconclusive_memory_grade_continues(self):
        rc, out = run(2, False, 0)
        self.assertEqual(rc, 0); self.assertIn('STEP shipping.memory-matched INCONCLUSIVE', out); self.assertIn('STEP shipping.attribution INCONCLUSIVE', out)


if __name__ == '__main__':
    unittest.main()
