#!/usr/bin/env python3
"""C1 VM comparison: one plain and one stage run of tools/alpha_c1_vm_input.py on the same overlay, token for token. Reads records only.

C0's comparison (tools/alpha_c0_vm_compare.py) with C1's record keys and one more requirement per side: the overlay's own selection
functions, run on the image over the composed environment, report both switches off on the plain run and both on, with
OverlapSegmentDriver importable, on the stage run. PASS needs both runs PASS on the real device, the first with the plain path and
the second through the stage wire with a skip proved on every scoped call, and the prefill and cohort output tokens equal. Memory is
reported.

What it establishes: on one rank of a real 3090 with the reduced model, a stage-scoped command runs the same work as a plain one, a record
naming another rank runs nothing, and the local release frees this rank's lane. What it does not: stage 1, PP handoffs, retained sends,
a TP peer, collectives, the staged driver, C1's overlap (provisional admission, two chunks in flight: engine-side), the worker
hook, or any speed figure. The box is the first run of those.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def section(probe):
    """The GPU case's returned record: the only one carrying the C1 fields."""
    rows = [row for row in (probe.get('sections') or {}).values() if type(row) is dict and 'alpha_c1_stage' in row]
    assert len(rows) == 1, 'the probe record carries no single C1 section'
    return rows[0]


def side(record):
    record = Path(record)
    result, probe = json.loads((record / 'result.json').read_text()), json.loads((record / 'probe.json').read_text())
    return result, probe, section(probe)


def compare(plain, stage):
    reasons = []
    (pr, pp, ps), (sr, sp, ss) = side(plain), side(stage)
    for name, result, row, expected in (('plain', pr, ps, False), ('stage', sr, ss, True)):
        if result.get('verdict') != 'PASS' or result.get('rc') != 0 or result.get('gpu_execution') is not True:
            reasons.append(name + '-run-not-pass')
        if row['alpha_c1_stage'] is not expected:
            reasons.append(name + '-run-on-the-wrong-path')
    calls = ss['alpha_c1_stage_calls']
    if not calls['scoped'] or calls['skips'] != calls['scoped'] or not calls['local_releases']:
        reasons.append('stage-calls-not-proved')
    if any(ps['alpha_c1_stage_calls'].values()):
        reasons.append('plain-run-used-the-stage-wire')
    for name, row, on in (('plain', ps, False), ('stage', ss, True)):
        chosen = row.get('alpha_c1_selection') or {}
        switches = {k: '1' if on else '0' for k in ('FLASH_S5_STAGE_LOCAL', 'FLASH_S5_STAGE_OVERLAP')}
        if (chosen.get('local') is not on or chosen.get('overlap') is not on or chosen.get('switches') != switches
                or chosen.get('driver_class') != 'OverlapSegmentDriver'):
            reasons.append(name + '-run-selection-not-' + ('overlap' if on else 'off'))
    prefill_same = ps['prefill_output_ids'] == ss['prefill_output_ids'] and bool(ps['prefill_output_ids'])
    cohort_same = ps['cohort_output_ids'] == ss['cohort_output_ids'] and bool(ps['cohort_output_ids'])
    if not prefill_same:
        reasons.append('prefill-tokens-differ')
    if not cohort_same:
        reasons.append('cohort-tokens-differ')
    if pr.get('manifest_sha256') == sr.get('manifest_sha256'):
        reasons.append('the-two-inputs-are-the-same-bytes')
    return dict(schema='alpha-c1-vm-compare/1', verdict='FAIL' if reasons else 'PASS', rc=0 if not reasons else 1, reasons=reasons,
                prefill_tokens_equal=prefill_same, cohort_tokens_equal=cohort_same,
                prefill_tokens=len(ss['prefill_output_ids']), cohort_streams=len(ss['cohort_output_ids']),
                cohort_tokens=sum(map(len, ss['cohort_output_ids'])), stage_calls=calls,
                selection=dict(plain=ps.get('alpha_c1_selection'), stage=ss.get('alpha_c1_selection')),
                plain=dict(record=str(plain), result_sha256=sha(Path(plain) / 'result.json'), probe_sha256=sha(Path(plain) / 'probe.json'),
                           manifest_sha256=pr.get('manifest_sha256'), own_peak_mib=pr.get('own_peak_mib'), elapsed_s=pr.get('elapsed_s'),
                           memory=ps.get('memory'), memory_at_exit=pp.get('memory_at_exit')),
                stage=dict(record=str(stage), result_sha256=sha(Path(stage) / 'result.json'), probe_sha256=sha(Path(stage) / 'probe.json'),
                           manifest_sha256=sr.get('manifest_sha256'), own_peak_mib=sr.get('own_peak_mib'), elapsed_s=sr.get('elapsed_s'),
                           memory=ss.get('memory'), memory_at_exit=sp.get('memory_at_exit')),
                scope=dict(establishes='one rank, real device, reduced model: a scoped command does the plain work, another rank\'s '
                                       'record runs nothing, the local release frees the lane; the C1 overlay selects '
                                       'overlap with both switches on and nothing with both off',
                           does_not_establish=['C1 overlap execution (engine-side)', 'stage 1', 'PP handoff', 'retained sends', 'TP peer', 'collectives', 'StagedSegmentDriver',
                                               'the generated worker hook', 'speed']),
                gpu_execution=True, box_action=False)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--plain', type=Path, required=True)
    p.add_argument('--stage', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = compare(a.plain, a.stage)
    with a.out.open('x') as handle:
        json.dump(result, handle, indent=1, sort_keys=True)
        handle.write('\n')
    print(result['verdict'], result['reasons'])
    return result['rc']


if __name__ == '__main__':
    raise SystemExit(main())
