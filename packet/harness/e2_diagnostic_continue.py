#!/usr/bin/env python3
"""Operator-authorized diagnostic continuation; never changes a memory verdict.

Only a complete reserve-only failure of the unchanged 512 MiB rule may be
continued. OOM, counter errors, missing evidence, other failures, malformed
reports and unexpected grader exits stop. The queue records the raw FAIL first,
so its final qualification verdict remains FAIL even if later diagnostics finish.
This is not permission to start another engine or change an allocator policy.
"""
import argparse
import json
import math

from e2_ledger_grade import RANKS


def decide(report, grader_rc):
    reasons = []
    if grader_rc != 1 or report.get('verdict') != 'FAIL':
        reasons.append('expected an ordinary FAIL verdict and exit 1')
    if report.get('reserve_gib') != 0.5:
        reasons.append('reserve rule is not 512 MiB')
    if report.get('mode') not in ('reserve-only', 'full'):
        reasons.append('unsupported memory grade mode')
    if report.get('missing') != []:
        reasons.append('memory evidence incomplete')
    fails = report.get('fails')
    allowed = {f'gpu {i}: less than the reserve free at the sampled peak' for i in range(4)}
    if not isinstance(fails, list) or not fails or any(f not in allowed for f in fails):
        reasons.append('failure is not exclusively the sampled reserve shortfall')
    ranks = report.get('ranks', {})
    if set(ranks) != set(RANKS):
        reasons.append('four allocator ranks required')
    for row in ranks.values():
        for key in ('ooms', 'retries_since_start', 'peak_gib', 'reserved_gib'):
            value = row.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                reasons.append('invalid allocator counter')
        if row.get('ooms') != 0:
            reasons.append('OOM is a hard stop')
        if row.get('retries_phase_delta', 0) < 0:
            reasons.append('backwards retry counter')
    gpus = report.get('gpus', {})
    if set(gpus) != {'0', '1', '2', '3'}:
        reasons.append('four physical GPU rows required')
    for row in gpus.values():
        for key in ('max_used_gib', 'total_gib', 'free_at_peak_gib'):
            value = row.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                reasons.append('invalid physical memory row')
        if row.get('max_used_gib', 0) >= row.get('total_gib', 0):
            reasons.append('physical headroom not positive')
    return dict(action='STOP' if reasons else 'CONTINUE_DIAGNOSTIC',
                qualification_verdict='FAIL', raw_verdict=report.get('verdict'),
                reasons=sorted(set(reasons)),
                scope='512 MiB rule unchanged; reserve FAIL retained; later work is diagnostic, not qualification')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('report'); p.add_argument('grader_rc', type=int)
    a = p.parse_args()
    try:
        with open(a.report) as f:
            result = decide(json.load(f), a.grader_rc)
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        result = dict(action='STOP', qualification_verdict='FAIL', reasons=['unreadable or malformed memory evidence'])
    print(json.dumps(result, sort_keys=True))
    return 0 if result['action'] == 'CONTINUE_DIAGNOSTIC' else 1


if __name__ == '__main__':
    raise SystemExit(main())
