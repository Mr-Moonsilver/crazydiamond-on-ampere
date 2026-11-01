#!/usr/bin/env python3
"""One fresh W2 overlap, unchanged held-b requests; regression only, no speed claim."""
import argparse
import json
import os
from pathlib import Path
import time
import e3_native_performance as perf
import e3_overlap_performance as shared
from e3_native_runtime_client import NativeRuntimeClient
from e3_native39_w1 import restore_full
from e3_native35_performance import salt_namespace


def run(cfg, client, url, model):
    record = dict(schema='native40-targeted-w2/1', engine_epoch=client.epoch, round=0,
                  family='fresh', speed_claim=False, wall_anchor_s=time.time(),
                  parent_manifest='756f826c221c026440d7955a9f834fe72db83581439fed9bb69d07d314ea4264')
    standing = []
    workload = shared._workload_view(cfg)
    try:
        client.prepare('full', 'full-prefill-solo')
        cohort = perf.streams.build_cohort(workload)
        namespace = salt_namespace(client.run_id, 0)
        record['salt_namespace'] = namespace
        bodies = [shared._request(model, row['prompt'], workload, shared._cache_salt(namespace, 'overlap', i))
                  for i, row in enumerate(cohort)]
        standing = perf.streams._start_streams(url, model, bodies, cohort, workload)
        if perf.streams._wait_open(standing, workload).get('complete') is not True:
            raise ValueError('standing-cohort-unopened')
        record['row'] = perf.prefill(cfg, client, url, model, 'overlap', 'fresh', shared._prefill_prompts(cfg),
                                    shared._cache_salt(namespace, 'overlap', 0), decode_states=standing)
        if record['row'].get('complete') is not True:
            raise ValueError('native-fresh-overlap-incomplete')
    except Exception as exc:
        record['error_type'] = type(exc).__name__
        record['cause'] = str(exc) if type(exc) is ValueError else type(exc).__name__
    finally:
        if standing:
            perf.streams._stop_streams({}, standing, workload)
        record['cleanup'] = restore_full(client)
        record['wall_end_s'] = time.time()
    return record


def grade(record, expected_tokens):
    row = record.get('row') or {}
    proof = perf.native_window(row, 'fresh', record.get('engine_epoch'))
    reasons = list(proof['reasons'])
    if row.get('complete') is not True or record.get('error_type'):
        reasons.append('fresh-overlap-incomplete')
    if (row.get('request') or {}).get('prompt_tokens_observed') != expected_tokens:
        reasons.append('fresh-token-count-differs')
    if (proof.get('group') or {}).get('useful_tokens') != expected_tokens - 1:
        reasons.append('native-frontier-not-n-minus-one')
    streams = (row.get('decode_window') or {}).get('streams') or []
    if len(streams) != 8 or any(s.get('completion_delta', 0) < 1 or s.get('error_before_window_end') for s in streams):
        reasons.append('eight-live-decoders-unproved')
    if (record.get('cleanup') or {}).get('full_restored') is not True:
        reasons.append('full-restoration-unproved')
    return dict(verdict='INCONCLUSIVE' if reasons else 'PASS', missing=sorted(set(reasons)),
                speed_claim=False, descriptive=True, parent_manifest=record.get('parent_manifest'),
                native_prefill_tokens_per_s=proof.get('prefill_tokens_per_s'),
                decode_tokens_per_s=(row.get('decode_window') or {}).get('aggregate_tokens_per_s'),
                ttft_s=(row.get('request') or {}).get('ttft_s'),
                basis='one fresh overlap; no W1, solo denominator, repeats or 80/80 claim')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grade-dir', type=Path)
    for name in ('url', 'model', 'run-id', 'workload', 'row1', 'out'):
        p.add_argument('--' + name, required=name == 'out')
    p.add_argument('--epoch', type=int)
    a = p.parse_args(argv)
    if Path(a.out).exists():
        return 2
    if a.grade_dir:
        result = final_grade(a.grade_dir)
        with Path(a.out).open('x') as f:
            json.dump(result, f, sort_keys=True, indent=2)
        return 0 if result['verdict'] == 'PASS' else 2
    record = dict(speed_claim=False)
    try:
        if json.loads(Path(a.row1).read_text()).get('verdict') != 'PASS':
            raise ValueError('row1-not-pass')
        if shared._package_gate(os.environ).get('ok') is not True:
            raise ValueError('package-not-ready')
        data = json.loads(Path(a.workload).read_text())
        problems, cfg = shared.validate_manifest(data)
        if problems or cfg['repeats'] != 1:
            raise ValueError('frozen-workload-invalid')
        cfg = dict(cfg, timeouts=dict(cfg['timeouts'], read_s=perf.first_advance_read_timeout(cfg)))
        record = run(cfg, NativeRuntimeClient(a.url, a.run_id, epoch=a.epoch), a.url, a.model)
        record['grade'] = grade(record, data['length_declaration']['fresh']['tokens'])
    except Exception as exc:
        record.update(error_type=type(exc).__name__, grade=dict(verdict='INCONCLUSIVE', speed_claim=False,
                      missing=['run-refused:' + type(exc).__name__]))
    with Path(a.out).open('x') as f:
        json.dump(record, f, sort_keys=True)
    return 0 if record['grade']['verdict'] == 'PASS' else 2


def final_grade(root):
    def load(name):
        try:
            return json.loads((root / name).read_text())
        except (OSError, ValueError):
            return {}
    run = load('w2-fresh.json')
    grade = run.get('grade') or {}
    required = ['unarmed-stock-prefill.json', 'qual-row1.json', 'instruments-begin.json',
                'instruments-end.json', 'ingress-fresh.json', 'engine-alive.json']
    missing = [name for name in required if load(name).get('verdict') != 'PASS']
    if grade.get('verdict') != 'PASS' or grade.get('speed_claim') is not False:
        missing.append('fresh-source-joined-regression')
    return dict(schema='native40-regression-grade/1', verdict='INCONCLUSIVE' if missing else 'PASS',
                missing=missing, speed_claim=False, descriptive=True, measured=grade,
                comparison='raw single fresh overlap against held-b; no baseline/repeat speed conclusion')


if __name__ == '__main__':
    raise SystemExit(main())
