"""The raw rule-4 result stays inconclusive while only a control cap can continue."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import e3_native28_rule4_continuation as continuation
import e3_native_performance as performance
import e3_teacher_decode_rule4 as rule4
import test_e3_teacher_decode_rule4 as fixture


def records():
    artifact = fixture.valid_evidence()
    fixture.set_all(artifact, 'full-A1')
    fixture.set_all(artifact, 'full-A2')
    fixture.set_all(artifact, 'full-C', fixture.shifted(-1.3))
    fixture.set_all(artifact, 'split-B')
    fixture.bind(artifact)
    grade = rule4.grade(artifact)
    assert continuation._hard_cap_only(grade), grade
    digest = hashlib.sha256(json.dumps(
        artifact, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    result = []
    for name, kind, runs in (('rule4_controls', 'controls', 1),
                             ('rule4_candidate', 'candidate', 0)):
        evidence = {'artifact': artifact, 'artifact_sha256': digest,
                    'kind': kind, 'teacher_runs': runs, 'retries': 0,
                    'picked': False, 'model_settings_unchanged': True}
        if kind == 'candidate':
            evidence.update(selected_from_same_artifact=True, post_c_before_c=False)
        result.append({'phase': name, 'verdict': 'INCONCLUSIVE', 'failures': [],
                       'missing': ['teacher:rule4-inc:' + reason
                                   for reason in grade['missing']],
                       'checks': {'teacher:rule4-verdict': 'INCONCLUSIVE',
                                  'teacher:rule4-calibration': grade['calibration_sha256']},
                       'teacher_artifact_sha256': digest, 'evidence': evidence})
    return result


class Rule4Continuation(unittest.TestCase):
    def test_only_hard_cap_inconclusive_continues_and_raw_gates_remain(self):
        controls, candidate = records()
        receipt = continuation.decision(controls, candidate)
        self.assertTrue(receipt['eligible'], receipt)
        self.assertEqual(receipt['raw_verdicts'], {
            'rule4_controls': 'INCONCLUSIVE', 'rule4_candidate': 'INCONCLUSIVE'})
        with tempfile.TemporaryDirectory() as temp:
            gate_path = Path(temp) / 'gates.json'
            encoded = json.dumps(receipt, sort_keys=True).encode()
            (Path(temp) / 'rule4-continuation.json').write_bytes(encoded)
            outcomes = {name: 'PASS' for name in
                        ('startup','row1','row4','w1_controls')}
            outcomes.update(rule4_controls='INCONCLUSIVE',
                            rule4_candidate='INCONCLUSIVE')
            gates = {'outcomes': outcomes, 'rule4_control_continuation': True,
                     'rule4_continuation_sha256': hashlib.sha256(encoded).hexdigest()}
            self.assertTrue(performance.gates_allow_run(gates, gate_path))
            outcomes['row4'] = 'FAIL'
            self.assertFalse(performance.gates_allow_run(gates, gate_path))
            outcomes['row4'] = 'PASS'
            gates['rule4_continuation_sha256'] = '0' * 64
            self.assertFalse(performance.gates_allow_run(gates, gate_path))

    def test_split_failure_or_other_missing_never_continues(self):
        controls, candidate = records()
        candidate['verdict'] = 'FAIL'
        self.assertFalse(continuation.decision(controls, candidate)['eligible'])
        candidate['verdict'] = 'INCONCLUSIVE'
        candidate['missing'].append('collector:phase-incomplete')
        self.assertFalse(continuation.decision(controls, candidate)['eligible'])
        candidate['missing'].pop()
        candidate['teacher_artifact_sha256'] = '0' * 64
        self.assertFalse(continuation.decision(controls, candidate)['eligible'])


if __name__ == '__main__':
    unittest.main()
