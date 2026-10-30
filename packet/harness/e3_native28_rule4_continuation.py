#!/usr/bin/env python3
"""Bounded continuation for an inconclusive rule-4 *control* hard cap.

The raw qualification verdict is never changed. A split failure, structural
missing evidence, or any other inconclusive reason keeps the old hold.
"""
import argparse
import hashlib
import json
from pathlib import Path

import e3_teacher_decode_rule4 as rule4


def _hard_cap_only(grade):
    reasons = grade.get('missing')
    return (grade.get('verdict') == 'INCONCLUSIVE' and type(reasons) is list
            and bool(reasons) and all(type(reason) is str
                and reason.startswith('control ') and ' exceeds hard cap ' in reason
                for reason in reasons) and not grade.get('failures'))


def decision(controls, candidate):
    phases = (('rule4_controls', controls, 'controls', 1),
              ('rule4_candidate', candidate, 'candidate', 0))
    reasons = []
    grades = []
    for name, record, kind, runs in phases:
        evidence = record.get('evidence') if type(record) is dict else None
        artifact = evidence.get('artifact') if type(evidence) is dict else None
        if (type(record) is not dict or record.get('phase') != name
                or record.get('verdict') != 'INCONCLUSIVE'
                or record.get('failures') != []
                or type(evidence) is not dict or type(artifact) is not dict
                or evidence.get('kind') != kind or evidence.get('teacher_runs') != runs
                or evidence.get('model_settings_unchanged') is not True
                or evidence.get('retries') not in (0, None)
                or evidence.get('picked') is True):
            reasons.append(name + ':qualification-shape')
            continue
        if kind == 'candidate' and (evidence.get('selected_from_same_artifact') is not True
                                    or evidence.get('post_c_before_c') is True):
            reasons.append(name + ':candidate-reuse')
            continue
        grade = rule4.grade(artifact)
        grades.append(grade)
        actual_sha = hashlib.sha256(json.dumps(
            artifact, sort_keys=True, separators=(',', ':'), allow_nan=False
        ).encode()).hexdigest()
        expected = ['teacher:rule4-inc:' + reason for reason in grade.get('missing', [])]
        if (not _hard_cap_only(grade) or record.get('missing') != expected
                or record.get('teacher_artifact_sha256') != actual_sha
                or evidence.get('artifact_sha256') != actual_sha
                or record.get('checks', {}).get('teacher:rule4-verdict') != 'INCONCLUSIVE'
                or record.get('checks', {}).get('teacher:rule4-calibration')
                != grade.get('calibration_sha256')):
            reasons.append(name + ':not-control-hard-cap-only')
    if (controls.get('teacher_artifact_sha256') is None
            or controls.get('teacher_artifact_sha256')
            != candidate.get('teacher_artifact_sha256')):
        reasons.append('teacher-artifact-identity')
    if len(grades) == 2 and grades[0].get('missing') != grades[1].get('missing'):
        reasons.append('control-grade-differs')
    return {'schema': 1, 'eligible': not reasons, 'reasons': reasons,
            'raw_verdicts': {name: record.get('verdict') for name, record, _, _ in phases},
            'control_hard_cap_reasons': grades[0].get('missing') if grades else [],
            'teacher_artifact_sha256': controls.get('teacher_artifact_sha256')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--controls', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    controls = json.loads(Path(args.controls).read_text())
    candidate = json.loads(Path(args.candidate).read_text())
    result = decision(controls, candidate)
    with Path(args.out).open('x') as handle:
        json.dump(result, handle, sort_keys=True)
    return 0 if result['eligible'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
