#!/usr/bin/env python3
"""Owner-authorized diagnostic continuation after bounded rule-4 outcomes.

Raw INCONCLUSIVE/FAIL verdicts are retained. Only a reproduced control hard cap
or a finite split-comparison failure may continue; structural failures block.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import e3_teacher_decode_rule4 as rule4


def _hard_cap_only(grade):
    reasons = grade.get('missing')
    return (grade.get('verdict') == 'INCONCLUSIVE' and type(reasons) is list
            and bool(reasons) and all(type(reason) is str
                and reason.startswith('control ') and ' exceeds hard cap ' in reason
                for reason in reasons) and not grade.get('failures'))


def _split_failure_only(grade):
    number = r'[0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?'
    pattern = re.compile(r'^split-B\|full-(?:A1|A2|C): (?:'
                         r'(?:winners|top5|shared_info|flips) ' + number +
                         r' > allowance ' + number + r'|[0-9]+ flip\(s\) exceed '
                         r'flip_margin allowance ' + number + r')$')
    failures = grade.get('failures')
    return (grade.get('verdict') == 'FAIL' and not grade.get('missing')
            and type(failures) is list and bool(failures)
            and bool(grade.get('allowances')) and bool(grade.get('comparisons'))
            and all(type(reason) is str and pattern.fullmatch(reason)
                    for reason in failures))


def numerical_context(grade):
    """Expose the frozen allowance arithmetic; never choose a new allowance."""
    calibration, _, _ = rule4.load_calibration()
    derivation = {}
    for metric, allowance in grade.get('allowances', {}).items():
        control = grade['control_max'][metric]
        offset = calibration['flip_offset'] if metric == 'flips' else calibration['offset']
        floor = None if metric == 'flips' else calibration['floors'][metric]
        derivation[metric] = dict(control_max=control, factor=calibration['factor'],
            offset=offset, floor=floor, cap=calibration['caps'][metric], allowance=allowance,
            formula=('int(min(cap, factor*control_max+offset))' if metric == 'flips'
                     else 'min(cap, max(floor, factor*control_max+offset))'))
    exceedances = []
    for pair, metrics in grade.get('comparisons', {}).items():
        for metric, allowance in grade.get('allowances', {}).items():
            value = metrics.get(metric)
            if (type(value) in (int, float) and math.isfinite(value)
                    and value > allowance):
                exceedances.append(dict(pair=pair, metric=metric, observed=value,
                    allowance=allowance, ratio=value / allowance if allowance else None,
                    derivation=derivation[metric]))
    return dict(allowance_derivation=derivation, exceedances=exceedances)


def decision(controls, candidate):
    phases = (('rule4_controls', controls, 'controls', 1),
              ('rule4_candidate', candidate, 'candidate', 0))
    reasons = []
    grades = []
    for name, record, kind, runs in phases:
        evidence = record.get('evidence') if type(record) is dict else None
        artifact = evidence.get('artifact') if type(evidence) is dict else None
        if (type(record) is not dict or record.get('phase') != name
                or record.get('verdict') not in ('INCONCLUSIVE', 'FAIL')
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
        expected_missing = ['teacher:rule4-inc:' + reason for reason in grade.get('missing', [])]
        expected_failures = ['teacher:rule4-fail:' + reason for reason in grade.get('failures', [])]
        if (not (_hard_cap_only(grade) or _split_failure_only(grade))
                or record.get('verdict') != grade.get('verdict')
                or record.get('missing') != expected_missing
                or record.get('failures') != expected_failures
                or record.get('teacher_artifact_sha256') != actual_sha
                or evidence.get('artifact_sha256') != actual_sha
                or record.get('checks', {}).get('teacher:rule4-verdict') != grade.get('verdict')
                or record.get('checks', {}).get('teacher:rule4-calibration')
                != grade.get('calibration_sha256')):
            reasons.append(name + ':not-bounded-numerical-outcome')
    if (controls.get('teacher_artifact_sha256') is None
            or controls.get('teacher_artifact_sha256')
            != candidate.get('teacher_artifact_sha256')):
        reasons.append('teacher-artifact-identity')
    if len(grades) == 2 and grades[0] != grades[1]:
        reasons.append('control-grade-differs')
    return {'schema': 1, 'eligible': not reasons, 'reasons': reasons,
            'mode': ('split-comparison-fail' if grades and _split_failure_only(grades[0])
                     else 'control-hard-cap-inconclusive'),
            'raw_verdicts': {name: record.get('verdict') for name, record, _, _ in phases},
            'control_hard_cap_reasons': grades[0].get('missing') if grades else [],
            'split_comparison_failures': grades[0].get('failures') if grades else [],
            'numerical_context': numerical_context(grades[0]) if grades else {},
            'release_qualified': False,
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
