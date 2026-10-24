#!/usr/bin/env python3
"""Check startup grouping/pool telemetry evidence, not inference correctness.

Usage: pp_stats_grade.py ENGINE.log 0|1 OUT.json
Accepts Docker timestamp/worker prefixes before structured FLASH records.
"""
import json
import os
import re
from pathlib import Path
import sys


def grade(text, enabled):
    layouts, samples, errors = [], [], []
    for line in text.splitlines():
        line = re.sub(r'\x1b\[[0-9;]*m', '', line)
        for marker, rows in (('FLASH_PP_LAYOUT ', layouts), ('FLASH_POOL_STATS ', samples)):
            if marker in line:
                try:
                    rows.append(json.loads(line.split(marker, 1)[1]))
                except (ValueError, TypeError):
                    errors.append('malformed ' + marker.strip())
    # PP_PARTITION (default 26,22): production runs 24,24 (flash-next, 24 Oct); counts from round 9's planner tests
    part = os.environ.get("PP_PARTITION", "26,22")
    table = {"26,22": {True: (23, [0, 0, 5, 5]), False: (39, [16, 16, 21, 21])},
             "24,24": {True: (21, [0, 0, 1, 1]), False: (39, [18, 18, 19, 19])}}
    if part not in table:
        return {'verdict': 'FAIL', 'errors': [f'unknown PP_PARTITION {part}']}
    total_groups, empties = table[part][enabled]
    if len(layouts) != 4 or {r.get('worker_index') for r in layouts} != set(range(4)):
        errors.append('expected exactly four worker layouts')
    for r in layouts:
        groups = r.get('groups', [])
        if (r.get('enabled') is not enabled or r.get('num_blocks') != 1340 or r.get('page_bytes') != 1603584
                or len(groups) != total_groups or [g.get('id') for g in groups] != list(range(total_groups))):
            errors.append('wrong worker group/page/pool layout')
        if sum(g.get('kind') == 'MambaSpec' for g in groups) != total_groups - 2:
            errors.append('wrong state group count')
    if sorted(sum(g.get('empty') is True for g in r.get('groups', [])) for r in layouts) != (
            empties):
        errors.append(f'wrong {part} empty-stage distribution')
    if not samples:
        errors.append('no periodic pool evidence')
    previous = 0
    counts = {}
    for r in samples:
        integer_fields = ('step', 'usable', 'free', 'free_cached', 'free_unhashed', 'referenced', 'referenced_cached')
        if any(type(r.get(k)) is not int or r[k] < 0 for k in integer_fields):
            errors.append('invalid pool counters')
            continue
        if (r['step'] <= previous or r['usable'] != 1339 or r['free'] + r['referenced'] != 1339
                or r['free_cached'] + r['free_unhashed'] != r['free']
                or r['referenced_cached'] > r['referenced'] or r.get('free_count_consistent') is not True):
            errors.append('pool step/count invariant failed')
        previous = r['step']
        for gid, row in r.get('groups', {}).items():
            if not str(gid).isdigit() or not 0 <= int(gid) < total_groups:
                errors.append('unknown group id')
                continue
            for k in ('hits', 'misses', 'not_consulted', 'evictions', 'evicted_hashes'):
                n = row.get(k)
                if type(n) is not int or n < counts.get((gid, k), 0):
                    errors.append('invalid/decreasing group counter')
                else:
                    counts[gid, k] = n
    if not sum(v for (_, k), v in counts.items() if k in ('hits', 'misses')):
        errors.append('no per-group lookup coverage')
    return {'verdict': 'FAIL' if errors else 'PASS', 'errors': errors,
            'layout_records': len(layouts), 'pool_samples': len(samples),
            'last_sample': samples[-1] if samples else None,
            'claim': 'layout/telemetry structure only; no safety or hit-rate verdict'}


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[2] not in ('0', '1'):
        raise SystemExit('usage: pp_stats_grade.py ENGINE.log 0|1 OUT.json')
    try:
        result = grade(Path(sys.argv[1]).read_text(), sys.argv[2] == '1')
    except (KeyError, TypeError, ValueError, AttributeError):
        result = {'verdict': 'FAIL', 'errors': ['malformed evidence']}
    Path(sys.argv[3]).write_text(json.dumps(result, indent=2) + '\n')
    print(result['verdict'], result.get('errors', []))
    raise SystemExit(0 if result['verdict'] == 'PASS' else 1)
