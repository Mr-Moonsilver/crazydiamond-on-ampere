#!/usr/bin/env python3
"""Step A post-run report (A2, M2): stepped vs fused from the box record. Reads the record, changes nothing.

Per chunk, from the progress rows the engine and workers already flush:
  owner time        replies all in -> applied, merged across commands, per stage and in total;
  stage own time    command submitted -> last reply of the stage's own ranks, merged;
  inactive tax      own ranks replied -> last reply of any rank, merged (M1);
  main thread       per worker rank, merged time in native handlers and in stock handlers inside the chunk;
  handler           per command on its target ranks, native handler enter -> exit, QSA layers apart from the rest;
  decode delay      per stock RPC, engine submit -> worker handler enter (A2's ready-to-submit delay), the slowest
                    rank per RPC, and the part of it spent behind a native handler on that same rank;
  service model     chunk interval against max(stage 0, stage 1, owner), scaled to 1,024 rows, beside 256 and 183 ms.

Not here: GPU-busy and exposed-wait splits. They need held-g's R1 receipts, which Step A's trunk does not carry.
Spans are merged before they are measured, so overlapping work never counts twice.
"""
import argparse
import bisect
import glob
import json
import lzma
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'harness'))
import e3_native41_progress_box as box  # noqa: E402
import step_a_torn_lines as torn_lines  # noqa: E402

TARGETS = {0: (0, 1), 1: (2, 3)}
ARENA_ROWS = 97152
TARGET_MS = {'4k': 256.0, '5.6k': 183.0}
NATIVE_PREFIX = 'flash_s5_'


def is_qsa_layer(layer):
    """Qwen3.8-Flash-Next repeats three GDN layers and one QSA layer."""
    return type(layer) is int and layer % 4 == 3


def merged(spans):
    out = []
    for lo, hi in sorted(s for s in spans if s[1] > s[0]):
        if out and lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return out


def seconds(spans):
    return sum(hi - lo for lo, hi in merged(spans)) / 1e9


def overlap_ns(a, b):
    a, b = merged(a), merged(b)
    total, j = 0, 0
    for lo, hi in a:
        while j < len(b) and b[j][1] <= lo:
            j += 1
        k = j
        while k < len(b) and b[k][0] < hi:
            total += min(hi, b[k][1]) - max(lo, b[k][0])
            k += 1
    return total


def dist(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    pick = lambda q: values[min(len(values) - 1, int(q * len(values)))]
    return dict(n=len(values), mean=round(statistics.fmean(values), 4), p10=round(pick(.1), 4), p50=round(pick(.5), 4),
                p90=round(pick(.9), 4), p99=round(pick(.99), 4), max=round(values[-1], 4))


def clip(row, lo, hi):
    if type(row.get('t_enter')) is not int or type(row.get('t_exit')) is not int:
        return None
    start, end = max(lo, row['t_enter']), min(hi, row['t_exit'])
    return (start, end) if end > start else None


class RankRows:
    """One worker rank's handler rows, ordered by entry."""

    def __init__(self, rows):
        self.rows = sorted((r for r in rows if type(r.get('t_enter')) is int), key=lambda r: r['t_enter'])
        self.starts = [r['t_enter'] for r in self.rows]

    def within(self, lo, hi):
        for row in self.rows[max(0, bisect.bisect_left(self.starts, lo) - 2):]:
            if row['t_enter'] >= hi:
                break
            span = clip(row, lo, hi)
            if span:
                yield row, span


def chunk_report(chunk, workers, by_rank):
    rows = sorted(chunk['rows'].get('n', []), key=lambda r: (r['t_submit'] or 0, r['seq']))
    identity = chunk['identity']
    out = dict(chunk_start=identity.get('chunk_start'), chunk_end=identity.get('chunk_end'),
               decode_live=identity.get('decode_live'), commands=len(rows))
    timed = [r for r in rows if type(r['t_submit']) is int and type(r['t_apply']) is int]
    if not timed:
        return out
    lo, hi = min(r['t_submit'] for r in timed), max(r['t_apply'] for r in timed)
    out['wall_s'] = (hi - lo) / 1e9
    width = (identity.get('chunk_end') or 0) - (identity.get('chunk_start') or 0)
    out['rows'] = width
    out['stage_local_commands'] = sum(1 for r in rows if r['stage'] in TARGETS)
    out['commands_by_kind'] = {}
    for r in rows:
        out['commands_by_kind'][r['kind']] = out['commands_by_kind'].get(r['kind'], 0) + 1

    chain = {stage: dict(span=[], own=[], tax=[], owner=[]) for stage in TARGETS}
    owner = []
    for r in timed:
        recv = {k: r['t_recv%d' % k] for k in range(4) if type(r['t_recv%d' % k]) is int}
        if not recv:
            continue
        last = max(recv.values())
        owner.append((last, r['t_apply']))
        stage = r['stage']
        if stage in TARGETS:
            active = [recv[k] for k in TARGETS[stage] if k in recv]
            if active:
                row = chain[stage]
                row['span'].append((r['t_submit'], r['t_apply']))
                row['own'].append((r['t_submit'], max(active)))
                row['tax'].append((max(active), last))
                row['owner'].append((last, r['t_apply']))
    out['owner_s'] = seconds(owner)
    out['stage'] = {str(stage): {k: seconds(v) for k, v in row.items()} for stage, row in chain.items()}
    out['stages_both_s'] = overlap_ns(chain[0]['span'], chain[1]['span']) / 1e9
    out['stages_neither_s'] = out['wall_s'] - seconds(chain[0]['span'] + chain[1]['span'])

    native = {}
    occupancy = {}
    for rank in range(4):
        spans = {'native': [], 'stock': []}
        for row, span in by_rank[rank].within(lo, hi):
            spans['native' if row['kind'].startswith(NATIVE_PREFIX) else 'stock'].append(span)
        native[rank] = spans['native']
        occupancy[str(rank)] = dict(native_s=seconds(spans['native']), stock_s=seconds(spans['stock']),
                                    idle_s=max(0.0, out['wall_s'] - seconds(spans['native'] + spans['stock'])))
    out['main_thread'] = occupancy

    # Target ranks only for a stage-local command: the other stage answers with a no-op.
    handlers = dict(qsa=[], other=[])
    by_kind = {}
    for r in rows:
        qsa = is_qsa_layer(r['layer'])
        for rank in TARGETS.get(r['stage'], range(4)):
            w = workers.get((r['seq'], rank))
            if w and type(w['t_enter']) is int and type(w['t_exit']) is int:
                took = (w['t_exit'] - w['t_enter']) / 1e9
                handlers['qsa' if qsa else 'other'].append(took)
                by_kind.setdefault(r['kind'] + ('-qsa' if qsa else ''), []).append(took)
    out['handler_s'] = handlers
    out['handler_by_kind_s'] = by_kind

    delays, behind = [], {str(rank): 0.0 for rank in range(4)}
    for s in chunk['rows'].get('s', []):
        if type(s['t_submit']) is not int:
            continue
        slowest = None
        for rank in range(4):
            w = workers.get((s['seq'], rank))
            if not w or type(w['t_enter']) is not int:
                continue
            slowest = max(slowest or 0, w['t_enter'] - s['t_submit'])
            if w['t_enter'] > s['t_submit']:
                behind[str(rank)] += overlap_ns([(s['t_submit'], w['t_enter'])], native[rank]) / 1e9
        if slowest is not None:
            delays.append(slowest / 1e6)
    out['decode_delay_ms'] = delays
    out['decode_delay_behind_native_s'] = behind

    scale = 1024.0 / width if width > 0 else None
    if scale:
        service = dict(stage0=out['stage']['0']['span'], stage1=out['stage']['1']['span'], owner=out['owner_s'])
        out['service_ms_per_1024'] = {k: 1e3 * v * scale for k, v in service.items()}
        out['service_ms_per_1024']['limit'] = max(out['service_ms_per_1024'].values())
        out['interval_ms_per_1024'] = 1e3 * out['wall_s'] * scale
        out['stage_own_ms_per_1024'] = {k: 1e3 * out['stage'][k]['own'] * scale for k in ('0', '1')}
    return out


def load(path):
    with (lzma.open(path, 'rt') if path.endswith('.xz') else open(path)) as handle:
        return json.load(handle)


def band(start):
    return 'beyond-arena' if (start or 0) >= ARENA_ROWS else 'below-arena'


def analyse(root):
    logs = [p for p in (os.path.join(root, 'engine-current.log'), os.path.join(root, 'engine-current.log.xz'))
            if os.path.exists(p)][:1]
    records, errors = box.read_lines(logs)
    torn = torn_lines.read_torn(logs)
    records, errors, _ = torn_lines.whole_records(records, errors, torn['progress'])
    flushes, problems = box.assemble(records)
    ledger = torn_lines.flush_ledger(records, torn['progress'])
    workers, per_rank = {}, {rank: [] for rank in range(4)}
    for flush in flushes:
        if flush['side'] == 'worker':
            for row in flush['rows'].get('r', []):
                workers[(row['seq'], flush['rank'])] = row
                per_rank[flush['rank']].append(row)
    by_rank = {rank: RankRows(rows) for rank, rows in per_rank.items()}
    engine = [f for f in flushes if f['side'] == 'engine']
    arms = {}
    for path in sorted(glob.glob(os.path.join(root, 'arm-*.json*'))):
        arm = load(path)
        name = arm.get('label') or os.path.basename(path).split('.')[0][4:]
        want = 'live' if arm.get('cohort') == 'live' else 'solo'
        start, end = arm['wall_start_s'], arm['wall_end_s']
        mine = sorted((f for f in engine if f['stamp'] is not None and start <= f['stamp'] <= end + box.LOG_LAG_S),
                      key=lambda f: f['identity'].get('chunk_start', -1))
        counted = [f for f in mine if box.chunk_label(f['identity'].get('decode_live'), 8) == want]
        # The torn-line rule: a chunk with rows in a lost flush drops out of the breakdown.
        whole = [f for f in counted if not any(torn_lines.touches(loss, torn_lines.chunk_seqs(f), ledger['rank_seqs'])
                                               for loss in ledger['lost'])]
        chunks = [chunk_report(f, workers, by_rank) for f in whole]
        arms[name] = dict(mode=arm.get('mode'), cohort=arm.get('cohort'), band=arm.get('band'),
                          chunks_dropped_torn=len(counted) - len(whole),
                          chunks=[c for c in chunks if 'wall_s' in c])
    return dict(errors=errors, problems=problems, arms=arms, torn_lines=len(torn['progress']) + len(torn['memory']),
                lost_flushes=len(ledger['lost']))


def group_summary(chunks):
    n = max(len(chunks), 1)
    pick = lambda fn: dist([fn(c) for c in chunks])
    ranks = ('0', '1', '2', '3')
    delays = [d for c in chunks for d in c['decode_delay_ms']]
    qsa = [h for c in chunks for h in c['handler_s']['qsa']]
    other = [h for c in chunks for h in c['handler_s']['other']]
    wall = sum(c['wall_s'] for c in chunks) or 1.0
    per_chunk_max = [max(c['handler_s']['qsa'] + c['handler_s']['other'] or [0]) for c in chunks]
    row = dict(
        chunks=len(chunks),
        chunk_interval_s=pick(lambda c: c['wall_s']),
        commands_per_chunk=round(sum(c['commands'] for c in chunks) / n, 1),
        stage_local_commands_per_chunk=round(sum(c['stage_local_commands'] for c in chunks) / n, 1),
        owner_s=pick(lambda c: c['owner_s']),
        stage={st: {k: pick(lambda c: c['stage'][st][k]) for k in ('span', 'own', 'tax', 'owner')} for st in ('0', '1')},
        stages_both_s=pick(lambda c: c['stages_both_s']),
        stages_neither_s=pick(lambda c: c['stages_neither_s']),
        main_thread_s={r: {k: round(sum(c['main_thread'][r][k] for c in chunks) / n, 4)
                           for k in ('native_s', 'stock_s', 'idle_s')} for r in ranks},
        main_thread_share={r: {k: round(sum(c['main_thread'][r][k] for c in chunks) / wall, 4)
                               for k in ('native_s', 'stock_s', 'idle_s')} for r in ranks},
        handler_ms=dict(qsa=dist([1e3 * h for h in qsa]), other=dist([1e3 * h for h in other]),
                        longest_per_chunk=dist([1e3 * h for h in per_chunk_max]),
                        by_kind={k: dist([1e3 * h for c in chunks for h in c['handler_by_kind_s'].get(k, [])])
                                 for k in sorted({k for c in chunks for k in c['handler_by_kind_s']})}),
        decode_delay_ms=dist(delays),
        decode_delay_behind_native_s_per_chunk={r: round(sum(c['decode_delay_behind_native_s'][r] for c in chunks) / n, 4)
                                                for r in ranks},
    )
    timed = [c for c in chunks if 'service_ms_per_1024' in c]
    if timed:
        row['service_ms_per_1024'] = {k: dist([c['service_ms_per_1024'][k] for c in timed])
                                      for k in ('stage0', 'stage1', 'owner', 'limit')}
        row['interval_ms_per_1024'] = dist([c['interval_ms_per_1024'] for c in timed])
        row['stage_own_ms_per_1024'] = {k: dist([c['stage_own_ms_per_1024'][k] for c in timed]) for k in ('0', '1')}
        row['share_interval_under'] = {k: round(sum(c['interval_ms_per_1024'] <= v for c in timed) / len(timed), 4)
                                       for k, v in TARGET_MS.items()}
    return row


def summarise(result):
    groups = {}
    for arm in result['arms'].values():
        for chunk in arm['chunks']:
            key = '%s-%s-%s' % (arm['mode'], arm['cohort'], band(chunk.get('chunk_start')))
            groups.setdefault(key, []).append(chunk)
    return {key: group_summary(chunks) for key, chunks in sorted(groups.items())}


def compare(summary, old, new):
    """New against old in each cohort and band, on the means the lead asked for."""
    out = {}
    for key, row in summary.items():
        mode, _, rest = key.partition('-')
        if mode != new or '%s-%s' % (old, rest) not in summary:
            continue
        base = summary['%s-%s' % (old, rest)]
        side = {}
        for label, read in (('chunk_interval_s', lambda r: r['chunk_interval_s']['mean']),
                            ('owner_s', lambda r: r['owner_s']['mean']),
                            ('stage1_own_s', lambda r: r['stage']['1']['own']['mean']),
                            ('stage1_owner_s', lambda r: r['stage']['1']['owner']['mean']),
                            ('stage0_tax_s', lambda r: r['stage']['0']['tax']['mean']),
                            ('commands_per_chunk', lambda r: r['commands_per_chunk']),
                            ('decode_delay_p50_ms', lambda r: (r['decode_delay_ms'] or {}).get('p50')),
                            ('decode_delay_p99_ms', lambda r: (r['decode_delay_ms'] or {}).get('p99')),
                            ('longest_handler_p50_ms', lambda r: (r['handler_ms']['longest_per_chunk'] or {}).get('p50'))):
            side[label] = dict(old=read(base), new=read(row))
        side['native_main_thread_s_by_rank'] = {r: dict(old=base['main_thread_s'][r]['native_s'], new=row['main_thread_s'][r]['native_s'])
                                                for r in ('0', '1', '2', '3')}
        side['stages_both_s'] = dict(old=base['stages_both_s']['mean'], new=row['stages_both_s']['mean'])
        out[rest] = side
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', help='directory holding arm-*.json and engine-current.log')
    parser.add_argument('out', help='fresh output JSON')
    parser.add_argument('--old', default='stepped')
    parser.add_argument('--new', default='fused')
    args = parser.parse_args(argv)
    if os.path.exists(args.out):
        parser.error('output exists: ' + args.out)
    result = analyse(args.root)
    summary = summarise(result)
    report = dict(schema='step-a-report/1', errors=result['errors'], problems=result['problems'],
                  torn_lines=result['torn_lines'], lost_flushes=result['lost_flushes'],
                  chunks_dropped_torn={k: v['chunks_dropped_torn'] for k, v in result['arms'].items()
                                       if v['chunks_dropped_torn']},
                  summary=summary, compare=compare(summary, args.old, args.new))
    with open(args.out, 'w') as handle:
        json.dump(report, handle, indent=1, sort_keys=True)
    print(json.dumps(report['compare'], indent=1, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
