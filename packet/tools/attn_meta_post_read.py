#!/usr/bin/env python3
"""q18a-attn-meta post-run anatomy read: metadata host time per decode step and stage-1 handler CPU per execute_model, per arm.
Reads, changes nothing; never a verdict input (amendment A: reads only, no new timers, no own numerics).

Source: the existing stage-1 handler anatomy lines (FLASH_S5_HANDLER_ANATOMY=1 in the queue's settings, schema
s5-handler-anatomy/1), parsed by tools/stage1_anatomy_report.py's own functions, unchanged (read_handler_lines, assemble,
seq_context). Its seam ``prep_attn_meta`` is the GDN/attention metadata build inside ``execute`` (the time attn-meta's batch
changes); ``cpu_ns`` is the main thread's CPU per stock RPC.

Assignment to arms. Handler rows carry the worker's monotonic clock and the arm records the client's wall clock, so a handler
flush is assigned to the arm whose [wall_start_s, wall_end_s] holds the flush's own log-line stamp (the engine stream's wall
time); a flush outside every arm is counted unassigned. A flush holds rows from shortly before its stamp, so the first flush in
an arm can carry the last rows of the gap before it: counted per arm (``boundary_flushes``: the first one in each arm), not
dropped. Within an arm, an execute_model row whose sequence lies in a prefill chunk's span (the anatomy report's join through the
engine chunk flushes' ``s`` rows; under burst progress that is every step of a prefill arm, decode streams included) is an
``in_chunk_span`` step; every other execute_model row is ``outside_chunk_span``. The decode-only arms (decode-am0, decode-am1) have
no chunk, so their steps are all outside_chunk_span: they carry the clean metadata host time per decode step
(``stage1_decode_step``, null where an arm has no such step); ``stage1_all_steps`` is every execute_model of the arm on stage 1.

Per arm, per rank: execute_model rows (decode and chunk); prep_attn_meta inclusive ns per step (median, p90, mean; ms) and the
seam's call count; cpu_ns per execute_model (median, p90; ms); stage 1 is ranks 2 and 3. Torn handler lines, incomplete
flushes and line gaps are counted as the anatomy report counts them; nothing is repaired. Peak memory and OOMs per arm are the
c1 grade's reads, already in the attn-meta grade and read; not recomputed here.
usage: attn_meta_post_read.py ROOT OUT
"""
import argparse
import glob
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'harness')]
import stage1_anatomy_report as anat  # noqa: E402

SCHEMA = 'attn-meta-post-read/1'
SEAM = 'prep_attn_meta'
STAGE1 = anat.STAGES['1']


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))] if values else None


def dist_ms(values):
    if not values:
        return dict(n=0, median_ms=None, p90_ms=None, mean_ms=None)
    return dict(n=len(values), median_ms=statistics.median(values) / 1e6, p90_ms=pct(values, 0.9) / 1e6,
                mean_ms=statistics.fmean(values) / 1e6)


def windows(root):
    out = {}
    for path in sorted(glob.glob(os.path.join(root, 'arm-*.json*'))):
        arm = anat.sar.load(path)
        name = arm.get('label') or os.path.basename(path).split('.')[0][4:]
        start, end = arm.get('wall_start_s'), arm.get('wall_end_s')
        if type(start) in (int, float) and type(end) in (int, float):
            out[name] = (start, end)
    return out


def read(root):
    logs = anat.sar.engine_logs(root)
    if not logs:
        raise SystemExit('no engine log in ' + root)
    records, torn, errors = anat.read_handler_lines(logs)
    flushes, incomplete = anat.assemble(records)
    gaps = anat.line_gaps(flushes)
    where = anat.seq_context(root, logs)[1]
    spans = windows(root)
    per, first, unassigned = {}, {}, 0
    for f in sorted(flushes, key=lambda f: (f['stamp'] is None, f['stamp'] or 0)):
        arm = next((n for n, (s, e) in spans.items() if f['stamp'] is not None and s <= f['stamp'] <= e), None)
        if arm is None:
            unassigned += 1
            continue
        first.setdefault((arm, f['rank']), f['line'])
        for r in f['main']:
            if r['flags'] & anat.FLAG_INSTALL or anat.KIND_NAMES.get(r['kind']) != 'execute_model':
                continue
            step = 'in_chunk_span' if r['seq'] in where else 'outside_chunk_span'
            cell = per.setdefault(arm, {}).setdefault(str(r['rank']), {}).setdefault(step, dict(meta=[], calls=0, cpu=[], n=0))
            cell['n'] += 1
            seam = r['seams'].get(SEAM)
            if seam:
                cell['meta'].append(seam['ns'])
                cell['calls'] += seam['n'] or 0
            if type(r['cpu_ns']) is int:
                cell['cpu'].append(r['cpu_ns'])
    arms = {}
    for arm, ranks in sorted(per.items()):
        arms[arm] = dict(boundary_flushes=sorted(str(rank) for (a, rank) in first if a == arm), ranks={
            rank: {step: dict(execute_model_rows=c['n'], prep_attn_meta=dist_ms(c['meta']), prep_attn_meta_calls=c['calls'],
                              cpu_per_execute_model=dist_ms(c['cpu']))
                   for step, c in sorted(steps.items())}
            for rank, steps in sorted(ranks.items())})
        def stage1(steps, key):
            return [v for rk in STAGE1 for s in steps for v in ranks.get(str(rk), {}).get(s, {}).get(key, [])]
        arms[arm]['stage1_decode_step'] = dict(prep_attn_meta=dist_ms(stage1(['outside_chunk_span'], 'meta')),
                                               cpu_per_execute_model=dist_ms(stage1(['outside_chunk_span'], 'cpu')))
        both = ['in_chunk_span', 'outside_chunk_span']
        arms[arm]['stage1_all_steps'] = dict(prep_attn_meta=dist_ms(stage1(both, 'meta')),
                                             cpu_per_execute_model=dist_ms(stage1(both, 'cpu')))
    return dict(schema=SCHEMA, root=os.path.abspath(root), logs=logs, read_only=True, is_verdict_input=False,
                handler_lines=len(records), torn_handler_lines=len(torn), incomplete_flushes=len(incomplete),
                line_gaps=len(gaps), schema_errors=len(errors), flushes=len(flushes), unassigned_flushes=unassigned,
                arms_with_windows=sorted(spans), arms=arms,
                rule='flush assigned by its log-line wall stamp to the arm window holding it; in_chunk_span = execute_model whose '
                     'seq lies in a prefill chunk span (engine s rows), outside_chunk_span = every other execute_model')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('root')
    p.add_argument('out')
    a = p.parse_args(argv)
    if os.path.exists(a.out):
        p.error('output exists: ' + a.out)
    report = read(a.root)
    with open(a.out, 'x') as handle:
        json.dump(report, handle, indent=1, sort_keys=True)
        handle.write('\n')
    for arm, row in report['arms'].items():
        for name in ('stage1_decode_step', 'stage1_all_steps'):
            d = row[name]
            print('%s %s: prep_attn_meta median %s ms (n %d), cpu/execute_model median %s ms'
                  % (arm, name, d['prep_attn_meta']['median_ms'], d['prep_attn_meta']['n'], d['cpu_per_execute_model']['median_ms']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
