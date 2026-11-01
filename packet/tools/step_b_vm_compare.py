#!/usr/bin/env python3
"""Compare B5 VM records of different layer-group sizes: bytes, tokens, memory, commands per chunk, host launch time.

Usage: step_b_vm_compare.py --out FILE name=RECORD_DIR [name=RECORD_DIR ...] (the first record is the reference).

What a record can show: on one real 3090 and through the real worker path, the lane's hidden states after each chunk and
the decode cohort's tokens come out the same or not, what the host spent launching each command, and the memory peak.

What it cannot show, and this tool will not claim: real weights (the model has seeded dummy weights), TP or PP (one rank,
so no collective and no ledger), or the engine driver (the tool drives the bridge). Decode under load is the box's read.

Output digest rows are compared in chunk order. The reference is meaningful only against a repeat of the same size: a
size-1 repeat that differs says the run is not deterministic, and then a size-2 difference proves nothing. Such a row is INC.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys


def percentile(values, share):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(share * len(ordered)))] if ordered else None


def load(path):
    record = Path(path)
    probe = json.loads((record / 'probe.json').read_text())
    result = json.loads((record / 'result.json').read_text())
    section = probe['sections']['local_overlap']
    rows = [json.loads(line) for line in (record / 'progress.jsonl').read_text().splitlines()]
    chunks = {}
    begun = {}
    for row in rows:
        if row['step'] == 'segment-submit-begin':
            begun[row['chunk'], row['op_seq']] = row
        if row['step'] != 'segment-submit-end':
            continue
        chunks.setdefault(row['chunk'], []).append(dict(kind=row['kind'], layer=row.get('layer'), ns=row['host_submit_ns'],
                                                        profile=begun[row['chunk'], row['op_seq']].get('profile_granted')))
    grants = {row['chunk']: row for row in rows if row['step'] == 'fused-grants'}
    return dict(probe=probe, result=result, section=section, chunks=chunks, grants=grants, rows=rows)


def launch(run):
    """Host launch time of layer commands, per QSA-bearing or not, and of the chunk's first layer command."""
    plain, fused, first = [], [], []
    for commands in run['chunks'].values():
        layers = [row for row in commands if row['kind'] == 'layer']
        if not layers:
            continue
        first.append(layers[0]['ns'] / 1e6)
        for row in layers:
            (fused if row['profile'] else plain).append(row['ns'] / 1e6)

    def summary(values):
        if not values:
            return None
        return dict(n=len(values), median_ms=round(statistics.median(values), 3), p99_ms=round(percentile(values, 0.99), 3),
                    max_ms=round(max(values), 3))
    return dict(plain=summary(plain), qsa=summary(fused), first_of_chunk=summary(first))


def summarize(name, run):
    section = run['section']
    commands = run['chunks']
    per_chunk = sorted({len(rows) for rows in commands.values()})
    layer_cmds = sorted({sum(row['kind'] == 'layer' for row in rows) for rows in commands.values()})
    digests = [row['output_digest'] for row in section['chunks']]
    return dict(name=name, size=section['step_b_size'], verdict=run['probe']['verdict'], result=run['result']['verdict'],
                failures=run['probe']['failures'], chunks=len(digests), ops_per_chunk=per_chunk, layer_commands_per_chunk=layer_cmds,
                own_peak_mib=run['result']['own_peak_mib'], elapsed_s=round(run['result']['elapsed_s'], 1),
                prefill_output_ids=section['prefill_output_ids'], cohort_tokens=section['cohort_tokens'],
                distinct_digests=len({row['sha256'] for row in digests}), nonfinite=sum(row['nonfinite'] for row in digests),
                abs_max=max(row['abs_max'] for row in digests), launch=launch(run))


NOISE_FACTOR = 2  # declared before the fingerprint runs: a size is within noise up to twice the size 1 repeat's largest difference


def numeric(reference, other):
    """Largest per-row difference of the hidden state's row sums and row maxima over all chunks, or None without fingerprints."""
    sums, maxes = [], []
    for a, b in zip(reference['section']['chunks'], other['section']['chunks']):
        x, y = a['output_digest'], b['output_digest']
        if 'row_sums' not in x or 'row_sums' not in y:
            return None
        sums.append(max(abs(p - q) for p, q in zip(x['row_sums'], y['row_sums'])))
        maxes.append(max(abs(p - q) for p, q in zip(x['row_abs_max'], y['row_abs_max'])))
    return dict(row_sum_max_diff=max(sums), row_abs_max_max_diff=max(maxes),
                first_chunk_with_a_difference=next((i for i, v in enumerate(sums) if v), None))


def compare(reference, other):
    left = [row['output_digest']['sha256'] for row in reference['section']['chunks']]
    right = [row['output_digest']['sha256'] for row in other['section']['chunks']]
    differing = [i for i, (a, b) in enumerate(zip(left, right)) if a != b]
    return dict(chunks=(len(left), len(right)), hidden_differing_chunks=len(differing), first_differing=differing[:5],
                numeric=numeric(reference, other),
                prefill_tokens_equal=reference['section']['prefill_output_ids'] == other['section']['prefill_output_ids'],
                cohort_tokens_equal=reference['section']['cohort_output_ids'] == other['section']['cohort_output_ids'])


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('records', nargs='+', help='name=record_dir, the first is the reference')
    args = parser.parse_args(argv)
    runs = {}
    for row in args.records:
        name, path = row.split('=', 1)
        runs[name] = load(path)
    names = list(runs)
    reference = runs[names[0]]
    pairs = {name: compare(reference, runs[name]) for name in names[1:]}
    repeat = next((name for name in names[1:] if runs[name]['section']['step_b_size'] == reference['section']['step_b_size']), None)
    deterministic = None if repeat is None else pairs[repeat]['hidden_differing_chunks'] == 0
    other_sizes = [name for name in names[1:] if runs[name]['section']['step_b_size'] != reference['section']['step_b_size']]
    same = all(pairs[name]['hidden_differing_chunks'] == 0 and pairs[name]['chunks'][0] == pairs[name]['chunks'][1]
               and pairs[name]['cohort_tokens_equal'] and pairs[name]['prefill_tokens_equal'] for name in other_sizes)
    if not deterministic:
        bytes_verdict = 'INC'
    else:
        bytes_verdict = 'PASS' if same else 'FAIL'
    noise = pairs[repeat]['numeric'] if repeat else None
    if noise is None or not other_sizes or any(pairs[name]['numeric'] is None for name in other_sizes):
        numeric_verdict = 'INC'
    else:
        bound = NOISE_FACTOR * noise['row_sum_max_diff']
        numeric_verdict = 'WITHIN_NOISE' if all(pairs[name]['numeric']['row_sum_max_diff'] <= bound and pairs[name]['cohort_tokens_equal']
                                                and pairs[name]['prefill_tokens_equal'] for name in other_sizes) else 'OUTSIDE_NOISE'
    report = dict(schema='step-b-vm-compare/1', reference=names[0], repeat=repeat, deterministic=deterministic, bytes_verdict=bytes_verdict,
                  numeric_verdict=numeric_verdict, noise_factor=NOISE_FACTOR,
                  numeric_meaning='bytes differ between two identical size 1 runs from chunk 2 on, so equality of bytes cannot be '
                                  'required; a size is WITHIN_NOISE when its largest row-sum difference to the reference is at most '
                                  'NOISE_FACTOR times the size 1 repeat\'s, with equal tokens. That is numerical, not bytewise',
                  runs={name: summarize(name, run) for name, run in runs.items()}, pairs=pairs,
                  cannot_show=['real weights: seeded dummy weights', 'TP and PP: one rank, no collective, no ledger',
                               'the engine driver: the tool drives the bridge', 'decode under load on the real model'])
    Path(args.out).write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(dict(bytes_verdict=bytes_verdict, numeric_verdict=numeric_verdict, deterministic=deterministic, pairs=pairs,
                          runs={name: {k: row[k] for k in ('size', 'verdict', 'ops_per_chunk', 'own_peak_mib', 'distinct_digests', 'nonfinite', 'launch')}
                                for name, row in report['runs'].items()}), indent=1))
    return 0 if bytes_verdict == 'PASS' or numeric_verdict == 'WITHIN_NOISE' else 1


if __name__ == '__main__':
    sys.exit(main())
