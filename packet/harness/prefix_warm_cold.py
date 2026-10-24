#!/usr/bin/env python3
"""120k-prefix warm/cold teacher-forced gate (explicit cache proof, CPU client).

Usage: prefix_warm_cold.py URL MODEL OUT.json [--baseline BASE.json]
       prefix_warm_cold.py --regrade OUT.json [--baseline BASE.json]
Fixed token IDs and top-20 at identical positions; uses teacher_forced rule v3.
Two cold controls per probe, one separately seeded warm prefix. No resets.
This is numerical equivalence within the frozen rule, not bitwise equality.
"""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

from teacher_forced import RULE, compare, probe, token_ids

CONFIG = {'context': 120000, 'tail': 2048, 'probes': 8, 'seed': 29, 'block': 4048}


def grade(d, baseline=None):
    errors = []
    if d.get('config') != CONFIG or len(d.get('rows', [])) != CONFIG['probes']:
        return {'verdict': 'INCONCLUSIVE', 'reason': 'wrong config or probe count'}
    rows = d['rows']
    if [r.get('index') for r in rows] != list(range(CONFIG['probes'])):
        return {'verdict': 'INCONCLUSIVE', 'reason': 'wrong probe identities'}
    for r in rows:
        if (not isinstance(r.get('prompt_sha256'), str) or len(r['prompt_sha256']) != 64
                or any(c not in '0123456789abcdef' for c in r['prompt_sha256'])):
            errors.append((r['index'], 'missing prompt token digest'))
        for name in ('cold_a', 'cold_b', 'seed', 'warm'):
            p = r.get(name, {})
            expected_len = CONFIG['context'] + (0 if name == 'seed' else CONFIG['tail'])
            top = p.get('top', {})
            if (len(top) < 20 or not all(type(v) in (float, int) and math.isfinite(v) for v in top.values())
                    or p.get('first') not in top or top.get(p.get('first'), -math.inf) < max(top.values(), default=0) - 1e-6
                    or p.get('prompt_tokens') != expected_len):
                errors.append((r['index'], name, 'invalid token/logprob evidence'))
            cached = p.get('cached')
            if name == 'warm':
                if type(cached) is not int or not CONFIG['context'] - CONFIG['block'] <= cached <= CONFIG['context']:
                    errors.append((r['index'], name, 'prefix reuse not proven'))
            elif type(cached) is not int or cached != 0:
                errors.append((r['index'], name, 'cold path not proven'))
        if not (r.get('seed', {}).get('t1', math.inf) <= r.get('warm', {}).get('t0', -math.inf)):
            errors.append((r['index'], 'warm before seed completion'))
    if errors:
        return {'verdict': 'INCONCLUSIVE', 'errors': errors}
    by = lambda name: {r['index']: r[name] for r in rows}
    noise = compare(by('cold_a'), by('cold_b'))
    candidate = compare(by('cold_a'), by('warm'))
    verdict = 'INCONCLUSIVE' if noise['fail_probes'] else ('FAIL' if candidate['fail_probes'] else 'PASS')
    result = {'verdict': verdict, 'noise': noise, 'warm_cold': candidate, 'rule': RULE}
    if baseline is not None:
        b = grade(baseline)
        if b['verdict'] != 'PASS':
            result.update(verdict='FAIL' if verdict == 'FAIL' else 'INCONCLUSIVE', baseline='baseline not qualified')
        elif [r['prompt_sha256'] for r in baseline['rows']] != [r['prompt_sha256'] for r in rows]:
            result.update(verdict='FAIL' if verdict == 'FAIL' else 'INCONCLUSIVE', baseline='different prompt token digests')
        else:
            pair = compare({r['index']: r['cold_a'] for r in baseline['rows']}, by('cold_a'))
            result['baseline_candidate_cold'] = pair
            if verdict == 'PASS' and pair['fail_probes']:
                result['verdict'] = 'FAIL'
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('args', nargs='*')
    p.add_argument('--regrade', type=Path)
    p.add_argument('--baseline', type=Path)
    a = p.parse_args()
    baseline = json.loads(a.baseline.read_text()) if a.baseline else None
    if a.regrade:
        data = json.loads(a.regrade.read_text())
        result = grade(data, baseline)
    else:
        if len(a.args) != 3:
            p.error('requires URL MODEL OUT.json')
        url, model, filename = a.args
        out = Path(filename)
        if out.exists():
            p.error('output exists; use a new run directory')
        rng = random.Random(CONFIG['seed'])
        context = token_ids(url, model, rng, CONFIG['context'])
        rows = []
        data = {'config': CONFIG, 'rows': rows, 'model': model}
        for i in range(CONFIG['probes']):
            ids = context + token_ids(url, model, rng, CONFIG['tail'])
            salt = f'pp-wc-{time.time_ns()}-{i}'
            row = {'index': i, 'prompt_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
            rows.append(row)
            for name in ('cold_a', 'cold_b', 'seed', 'warm'):
                row[name] = probe(url, model, context if name == 'seed' else ids,
                                  salt if name in ('seed', 'warm') else salt + '-' + name)
                out.write_text(json.dumps(data, indent=2) + '\n')
        result = grade(data, baseline)
        data['result'] = result
        out.write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps(result, sort_keys=True))
    raise SystemExit({'PASS': 0, 'FAIL': 1, 'INCONCLUSIVE': 2}[result['verdict']])


if __name__ == '__main__':
    main()
