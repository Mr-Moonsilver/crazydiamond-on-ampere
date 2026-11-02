#!/usr/bin/env python3
"""alpha-next-c1 post-run report: C1 on against C1 off (C0 local in both), per group size, from the box record. Reads, changes nothing.

alpha-next's report (tools/alpha_next_report.py, which wraps Step A's analyser) does the per-chunk work unchanged, with two
bindings for its one call: Step A's analyser assembles the engine flushes with each arm's flushes regrouped by that arm's
``flash-s5 stage-overlap`` lines (the grade's rule, harness/e3_alpha_next_c1_box.regroup and overlap_owners, not a second one), so
a C1-on chunk holds its own rows and not its successor's early ones; and the grouping key holds the C1 selection instead of C0's:
``fused.g2.c1on``, ``fused.g2.c1off``, ``fused.g4.c1on`` (no '-', so ``compare`` can split keys). C1-off arms log no overlap line,
so their chunks are alpha-next's, unchanged. Also read, never a verdict: the per-chunk ``flash-s5 memory-peak`` receipt per key.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'harness')]
import alpha_next_report as anr  # noqa: E402
import alpha_next_c1_arms as table  # noqa: E402
import e3_alpha_next_c1_box as box  # noqa: E402

SCHEMA = 'alpha-next-c1-report/1'
sar, ap = anr.sar, box.ap


def group_key(size, overlap):
    """The grouping token: QSA mode (fused in every arm), group size and C1 selection; no '-'."""
    return 'fused.g%d.c1%s' % (size, 'on' if overlap else 'off')


def arm_key(name, arm):
    """(token, qsa_mode, group_size, overlap) from the declared table by label (the record's selection names it too)."""
    row = table.by_label().get(name) or {}
    parsed = table.parse_selection(arm.get('selection')) if arm.get('selection') else None
    size, overlap = (parsed[1], parsed[0] == 'on') if parsed else (row.get('size'), row.get('overlap'))
    if type(size) is not int or type(overlap) is not bool:
        return None
    return group_key(size, overlap), 'fused', size, overlap


def regrouping(root, logs):
    """Step A's analyser assemble, with each arm's engine flushes regrouped by its stage-overlap lines."""
    original = sar.box.assemble
    arms = [dict(a, label=name) for name, a in anr.arm_records(root).items()]
    overlap, _ = box.overlap_rows(logs)

    def assemble(records):
        flushes, problems = original(records)
        replaced = {}
        for arm in arms:
            mine = [f for f in flushes if f['side'] == 'engine' and ap.in_span(f['stamp'], arm)]
            texts = [t for s, t in overlap if ap.in_span(s, arm)]
            if not mine or not texts:
                continue
            owner, faults = box.overlap_owners(texts)
            txns = [(f['identity'] or {}).get('txn') for f in mine]
            if faults or any(type(t) is not int for t in txns) or len(set(txns)) != len(txns):
                continue                                  # the grade reports these; the analyser keeps the flushes as they are
            here = {seq: txn for seq, txn in owner.items() if txn in txns}
            chunks = {c['identity']['txn']: c for c in box.regroup(mine, here, {})}
            for f in mine:
                replaced[id(f)] = dict(f, rows=dict(f['rows'], n=chunks[f['identity']['txn']]['rows'].get('n', [])))
        return [replaced.get(id(f), f) for f in flushes], problems
    return assemble


def analyse(root, logs=None):
    logs = logs if logs is not None else sar.engine_logs(root)
    with box._bound(sar.box, dict(assemble=regrouping(root, logs))), box._bound(anr, dict(arm_key=arm_key)):
        result = anr.analyse(root, logs)
    for arm in result['arms'].values():
        overlap = arm.pop('stage_local', None)
        if 'group' in arm:
            arm['overlap'], arm['stage_local'] = overlap, True
    return result


def memory_peaks(root, logs=None):
    with box._bound(anr, dict(arm_key=arm_key)):
        return anr.memory_peaks(root, logs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('root', help='directory holding arm-*.json and engine-current.log')
    parser.add_argument('out', help='fresh output JSON')
    parser.add_argument('--old', default=group_key(2, False), help='group token, e.g. fused.g2.c1off')
    parser.add_argument('--new', default=group_key(2, True), help='group token, e.g. fused.g2.c1on or fused.g4.c1on')
    args = parser.parse_args(argv)
    if os.path.exists(args.out):
        parser.error('output exists: ' + args.out)
    result = analyse(args.root)
    summary = sar.summarise(result)
    for key, row in summary.items():
        row['overlap'] = key.partition('-')[0].endswith('.c1on')
    report = dict(schema=SCHEMA, errors=result['errors'], problems=result['problems'], torn_lines=result['torn_lines'],
                  lost_flushes=result['lost_flushes'], unkeyed_arms=result['unkeyed_arms'],
                  chunks_dropped_torn={k: v['chunks_dropped_torn'] for k, v in result['arms'].items() if v['chunks_dropped_torn']},
                  summary=summary, compare=sar.compare(summary, args.old, args.new), old=args.old, new=args.new,
                  memory_peak=memory_peaks(args.root), memory_peak_is_gate=False)
    with open(args.out, 'x') as handle:
        json.dump(report, handle, indent=1, sort_keys=True)
    print(json.dumps(report['compare'], indent=1, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
