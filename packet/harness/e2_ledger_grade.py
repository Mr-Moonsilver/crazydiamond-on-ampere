#!/usr/bin/env python3
"""Grade the E2 memory contract from memory_ledger.py records plus sampled GPU peaks (Codex q16 review item 2).

The candidate declares incremental KV storage 0 B/rank and a 536,870,912 B reserve for its runtime resources
(copy stream, events, metadata). The contract on a loaded phase (end record taken after the phase's load):
  FAIL          any rank reports ooms > 0; an allocator retry counter went backwards; with a baseline: a rank's
                allocator peak exceeds the baseline rank's peak by more than the reserve, or the phase's retry delta
                exceeds the baseline's retry count, or a GPU's sampled peak used memory exceeds the baseline's used
                memory on that GPU by more than the reserve
  INCONCLUSIVE  fewer than four ranks with allocator rows (QSA_STAGE_TIMING lines absent), no baseline record, or
                no sampled peaks: the numbers are reported, the contract is not proven
  PASS          otherwise
usage: e2_ledger_grade.py START.json END.json [--peaks PEAKS.json] [--baseline BASE.json] [--reserve-bytes N]
PEAKS.json: {"gpus": {"0": {"max_used_mib": N, "total_mib": N}, ...}, "samples": N} from the queue's nvidia-smi sampler.
BASE.json: a memory_ledger.py record of the reference engine after comparable load (q15b: after the leak under pressure).
Exit 0 PASS, 1 FAIL, 2 INCONCLUSIVE. Only numbers are printed.
"""
import argparse, json, sys

RANKS = ('Worker_PP0_TP0', 'Worker_PP0_TP1', 'Worker_PP1_TP0', 'Worker_PP1_TP1')
GIB = 1024 ** 3


def grade(start, end, peaks=None, baseline=None, reserve_bytes=536870912):
    fails, missing, ranks = [], [], {}
    reserve = reserve_bytes / GIB
    for r in RANKS:
        e = (end.get('per_rank') or {}).get(r); s = (start.get('per_rank') or {}).get(r) or {}
        if not e or any(k not in e for k in ('alloc_retries', 'ooms', 'peak_gib', 'reserved_gib')):
            missing.append(r + ': no allocator row at the phase end'); continue
        row = dict(peak_gib=e['peak_gib'], reserved_gib=e['reserved_gib'], ooms=e['ooms'],
                   retries_start=s.get('alloc_retries'), retries_end=e['alloc_retries'])
        if e['ooms'] > 0: fails.append(r + ': ooms')
        if s.get('alloc_retries') is not None:
            row['retries_delta'] = e['alloc_retries'] - s['alloc_retries']
            if row['retries_delta'] < 0: fails.append(r + ': retry counter went backwards')
        else:
            missing.append(r + ': no allocator row at the phase start')
        if baseline is not None:
            b = (baseline.get('per_rank') or {}).get(r)
            if not b or 'peak_gib' not in b or 'alloc_retries' not in b:
                missing.append(r + ': no baseline allocator row')
            else:
                row['baseline_peak_gib'] = b['peak_gib']; row['baseline_retries'] = b['alloc_retries']
                row['peak_delta_gib'] = round(e['peak_gib'] - b['peak_gib'], 3)
                if e['peak_gib'] > b['peak_gib'] + reserve: fails.append(r + ': allocator peak above baseline + reserve')
                if row.get('retries_delta') is not None and row['retries_delta'] > b['alloc_retries']:
                    fails.append(r + ': more allocator retries than the baseline')
        ranks[r] = row
    gpus = {}
    if peaks is None:
        missing.append('no sampled GPU peaks')
    else:
        base_used = {str(g['index']): g['used_gib'] for g in (baseline or {}).get('gpus', []) if 'index' in g and 'used_gib' in g}
        for idx, p in sorted((peaks.get('gpus') or {}).items()):
            row = dict(max_used_gib=round(p['max_used_mib'] / 1024, 3), total_gib=round(p['total_mib'] / 1024, 3))
            if idx in base_used:
                row['baseline_used_gib'] = base_used[idx]; row['delta_gib'] = round(row['max_used_gib'] - base_used[idx], 3)
                if row['max_used_gib'] > base_used[idx] + reserve: fails.append('gpu ' + idx + ': sampled peak above baseline + reserve')
            elif baseline is not None:
                missing.append('gpu ' + idx + ': no baseline used-memory row')
            gpus[idx] = row
        if len(gpus) != 4: missing.append('sampled peaks do not cover four GPUs')
        if not peaks.get('samples'): missing.append('no GPU samples taken')
    if baseline is None: missing.append('no baseline ledger record')
    v = 'FAIL' if fails else 'INCONCLUSIVE' if missing else 'PASS'
    return dict(verdict=v, fails=sorted(set(fails)), missing=sorted(set(missing)), reserve_gib=round(reserve, 3), ranks=ranks, gpus=gpus,
                scope='declared 0 B incremental KV + reserve; peaks from allocator rows and nvidia-smi samples; no efficiency claim')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('start'); p.add_argument('end'); p.add_argument('--peaks'); p.add_argument('--baseline')
    p.add_argument('--reserve-bytes', type=int, default=536870912)
    a = p.parse_args()
    load = lambda f: json.load(open(f)) if f else None
    out = grade(load(a.start), load(a.end), load(a.peaks), load(a.baseline), a.reserve_bytes)
    print(json.dumps(out, sort_keys=True))
    sys.exit({'PASS': 0, 'FAIL': 1, 'INCONCLUSIVE': 2}[out['verdict']])
