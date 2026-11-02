#!/usr/bin/env python3
"""alpha-next post-run report: C0 on against C0 off, per group size, from the box record. Reads the record, changes nothing.

Step A's analyser (tools/step_a_report.py) does the per-chunk work unchanged: owner time, stage own time, inactive tax, main
thread, handler and decode delay, the service model. It is frozen in the step-a and step-b spec bundles, so it is wrapped here,
not edited: ``analyse`` runs it, then sets each arm's grouping key from the arm record, because a Step A/B key (QSA mode and
group size) cannot tell C0 on from C0 off. The key holds the QSA mode, the group size and the C0 selection, with no '-' so
``compare`` can split keys: ``fused.g2.c0on``, ``fused.g2.c0off``, ``fused.g4.c0on``. step_a_report.summarise uses
``arm['group']`` when present, so its grouping is this key.

Also read, never a verdict: the engine's per-chunk ``flash-s5 memory-peak`` receipt (lead, 2 Nov 13:30), the highest peak
allocated and reserved per rank in each group, in MiB.
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'harness')]
import step_a_report as sar  # noqa: E402
import alpha_next_arms as table  # noqa: E402
import e3_alpha_perf_box as ap  # noqa: E402
import e3_alpha_next_box as nb  # noqa: E402

SCHEMA = 'alpha-next-report/1'


def group_key(size, stage_local):
    """The grouping token: QSA mode (fused in every alpha-next arm), group size and C0 selection; no '-'."""
    return 'fused.g%d.c0%s' % (size, 'on' if stage_local else 'off')


def arm_records(root):
    out = {}
    for path in sorted(glob.glob(os.path.join(root, 'arm-*.json*'))):
        arm = sar.load(path)
        out[arm.get('label') or os.path.basename(path).split('.')[0][4:]] = arm
    return out


def arm_key(name, arm):
    """(token, qsa_mode, group_size, stage_local) from the arm record, else from the declared table by label."""
    size, local = arm.get('size'), arm.get('stage_local')
    if type(size) is not int or type(local) is not bool:
        row = table.by_label().get(name) or {}
        size, local = row.get('size'), row.get('stage_local')
    if type(size) is not int or type(local) is not bool:
        return None
    return group_key(size, local), 'fused', size, local


def analyse(root, logs=None):
    result = sar.analyse(root, logs)
    records = arm_records(root)
    unkeyed = []
    for name, arm in result['arms'].items():
        key = arm_key(name, records.get(name) or {})
        if key is None:
            unkeyed.append(name)
            continue
        arm.update(group=key[0], qsa_mode=key[1], group_size=key[2], stage_local=key[3])
    result['unkeyed_arms'] = unkeyed
    return result


def memory_peaks(root, logs=None):
    """Per group key, the highest per-chunk peak allocated and reserved per rank (MiB), from the memory-peak receipts."""
    logs = logs if logs is not None else sar.engine_logs(root)
    rows = ap.read_stamped(logs, lambda line: nb.PEAK_LINE in line)
    out = {}
    for name, arm in arm_records(root).items():
        key = arm_key(name, arm)
        if key is None:
            continue
        cell = out.setdefault(key[0], dict(receipts=0, ranks={}))
        for stamp, text in rows:
            m = nb.PEAK_RE.search(text)
            if not m or not ap.in_span(stamp, arm):
                continue
            cell['receipts'] += 1
            rank = cell['ranks'].setdefault(m.group(1), dict(peak_allocated_mib=None, peak_reserved_mib=None))
            for field, group in (('peak_allocated_mib', 5), ('peak_reserved_mib', 6)):
                value = round(int(m.group(group)) / 2 ** 20, 1)
                rank[field] = value if rank[field] is None else max(rank[field], value)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('root', help='directory holding arm-*.json and engine-current.log')
    parser.add_argument('out', help='fresh output JSON')
    parser.add_argument('--old', default=group_key(2, False), help='group token, e.g. fused.g2.c0off')
    parser.add_argument('--new', default=group_key(2, True), help='group token, e.g. fused.g2.c0on or fused.g4.c0on')
    args = parser.parse_args(argv)
    if os.path.exists(args.out):
        parser.error('output exists: ' + args.out)
    result = analyse(args.root)
    summary = sar.summarise(result)
    for key, row in summary.items():
        token = key.partition('-')[0]
        row['stage_local'] = token.endswith('.c0on')
    report = dict(schema=SCHEMA, errors=result['errors'], problems=result['problems'], torn_lines=result['torn_lines'],
                  lost_flushes=result['lost_flushes'], unkeyed_arms=result['unkeyed_arms'],
                  chunks_dropped_torn={k: v['chunks_dropped_torn'] for k, v in result['arms'].items() if v['chunks_dropped_torn']},
                  summary=summary, compare=sar.compare(summary, args.old, args.new), old=args.old, new=args.new,
                  memory_peak=memory_peaks(args.root), memory_peak_is_gate=False)
    with open(args.out, 'w') as handle:
        json.dump(report, handle, indent=1, sort_keys=True)
    print(json.dumps(report['compare'], indent=1, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
