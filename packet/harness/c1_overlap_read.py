#!/usr/bin/env python3
"""Per-successor overlap read over ``flash-s5 overlap-observe`` lines (FLASH_S5_STAGE_OVERLAP_OBSERVE=1).

A read, never a gate.  For each chunk n+1 that was a successor of chunk n:

* ``fin_lo``: a lower bound on when stage 1's STATE_FINALIZE of n completed on device, the
  latest over stage-1 ranks of the last submit time of a reply whose report still listed that
  rank's finalize op pending (the op completed after that report was built, which was after
  that submit);
* ``layer_hi``: an upper bound on when stage 0's first fully completed LAYER of n+1 completed,
  over every stage-0 TP rank: per layer the latest rank's first apply time of a reply listing
  it complete, and the earliest such layer.

``OVERLAP`` when ``layer_hi < fin_lo``: some LAYER of n+1 had finished on every stage-0 rank
while stage 1's finalize of n was provably still unfinished.  Otherwise ``NONE`` (with the
reason).  Both bounds are host CLOCK_MONOTONIC ns of the engine.  A report that arrives late
widens the bounds, so a late report can only turn OVERLAP into NONE, never the reverse.  An
in-flight-only host counter (early opens, both-stages seconds) is not consulted.

usage: c1_overlap_read.py LOG [--out JSON]
"""
import argparse
import json
import sys

LINE = 'flash-s5 overlap-observe '
SCHEMA = 's5-overlap-observe/1'


def expand_ranges(text):
    """The engine's 'a-b,c' sequence runs (stage_overlap._ranges) as a set of ints."""
    out = set()
    if text in ('', '-', None):
        return out
    for run in text.split(','):
        a, _, b = run.partition('-')
        out.update(range(int(a), int(b or a) + 1))
    return out


def parse(lines):
    """{txn: {'info': part-0 fields, 'ops': {(rank, op_seq): row dict}}}; refuses a foreign schema."""
    chunks = {}
    for line in lines:
        index = line.find(LINE)
        if index < 0:
            continue
        body = json.loads(line[index + len(LINE):])
        if body.get('schema') != SCHEMA:
            raise ValueError('overlap-observe line with schema %r' % (body.get('schema'),))
        chunk = chunks.setdefault(body['txn'], dict(info={}, ops={}, parts=set(), nparts=body['parts']))
        if body['part'] in chunk['parts']:
            raise ValueError('duplicate overlap-observe part for txn %s' % body['txn'])
        chunk['parts'].add(body['part'])
        if body['part'] == 0:
            chunk['info'] = {k: v for k, v in body.items() if k not in ('ops', 'part', 'parts', 'fields')}
        fields = body['fields']
        for values in body['ops']:
            row = dict(zip(fields, values))
            chunk['ops'][(row['rank'], row['op_seq'])] = row
    return chunks


def read_pair(older, newer):
    stage0, stage1 = newer['info']['stage_ranks']
    fin = {rank: row for (rank, _op), row in older['ops'].items()
           if rank in stage1 and row['kind'] == 'state_finalize'}
    if set(fin) != set(stage1):
        return dict(verdict='NONE', reason='finalize-not-observed-on-every-stage-1-rank')
    los = [row['lo_ns'] for row in fin.values()]
    fin_lo = None if any(lo is None for lo in los) else max(los)
    layers = {}
    for (rank, _op), row in newer['ops'].items():
        if rank in stage0 and row['kind'] == 'layer':
            layers.setdefault(row['layer'], {})[rank] = row['hi_ns']
    complete = [max(by_rank.values()) for by_rank in layers.values()
                if set(by_rank) == set(stage0) and all(hi is not None for hi in by_rank.values())]
    layer_hi = min(complete) if complete else None
    result = dict(fin_lo_ns=fin_lo, layer_hi_ns=layer_hi,
                  fin_hi_ns=max((row['hi_ns'] for row in fin.values() if row['hi_ns'] is not None), default=None))
    if fin_lo is None:
        result.update(verdict='NONE', reason='finalize-never-reported-pending')
    elif layer_hi is None:
        result.update(verdict='NONE', reason='no-stage-0-layer-of-the-successor-observed-complete')
    elif layer_hi < fin_lo:
        result.update(verdict='OVERLAP', reason='successor-layer-complete-before-older-finalize-complete',
                      margin_ns=fin_lo - layer_hi)
    else:
        result.update(verdict='NONE', reason='successor-layer-not-provably-before-older-finalize')
    return result


def overlap_read(lines):
    chunks = parse(lines)
    rows = []
    for txn, chunk in sorted(chunks.items()):
        if len(chunk['parts']) != chunk['nparts']:
            rows.append(dict(txn=txn, verdict='NONE', reason='torn-observation'))
            continue
        older_txn = chunk['info'].get('successor_of')
        if older_txn is None:
            continue
        older = chunks.get(older_txn)
        row = dict(txn=txn, older=older_txn, early=chunk['info'].get('early'),
                   deferred_bind=chunk['info'].get('deferred_bind'))
        if older is None or len(older['parts']) != older['nparts']:
            row.update(verdict='NONE', reason='older-chunk-not-observed')
        else:
            row.update(read_pair(older, chunk))
        rows.append(row)
    return dict(schema='c1-overlap-read/1', successors=len(rows),
                overlap=sum(1 for r in rows if r['verdict'] == 'OVERLAP'),
                none=sum(1 for r in rows if r['verdict'] == 'NONE'), rows=rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('log')
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    with open(args.log, errors='replace') as handle:
        result = overlap_read(handle)
    text = json.dumps(result, indent=1, sort_keys=True)
    if args.out:
        with open(args.out, 'x') as out:
            out.write(text + '\n')
    print(json.dumps({k: result[k] for k in ('successors', 'overlap', 'none')}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
