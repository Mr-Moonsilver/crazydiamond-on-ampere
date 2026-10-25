#!/usr/bin/env python3
"""Grade the E2 memory contract from memory_ledger.py records plus sampled GPU peaks (Codex q16 review, v2 semantics).

The candidate declares incremental KV storage 0 B/rank and a 536,870,912 B reserve for its runtime resources
(copy stream, events, metadata). Two separate questions, never mixed (Codex: baseline deltas are overhead evidence,
not the reserve):
  reserve   at the sampled simultaneous peak every GPU still has at least the reserve FREE: total - max_used >= reserve
            (nvidia-smi samples through the whole phase). Less free memory than the reserve is FAIL.
  overhead  against a MATCHED baseline record (the reference engine's ledger after the same load shape, q15b
            mem-under-pressure): per rank the allocator peak and the allocator retry count, per GPU the used memory.
            Both records carry PyTorch's cumulative-since-process-start counters (num_alloc_retries, max reserved),
            so the END record's absolute values are the phase-inclusive values since the engine started; a START
            record's rows (absent before the first staged call) only add the within-phase delta when present. A rank
            with more retries than the baseline's, or any ooms, is FAIL; peak/used deltas are reported.
  INCONCLUSIVE  fewer than four ranks with allocator rows at END, no sampled peaks for four GPUs, no baseline record,
            or a baseline without the compared rows: numbers reported, contract not proven. Missing START rows are
            noted, not invented as zero.
  matched   (--matched, Codex q16 v3 review P2-2) the END record was taken at a MATCHED POINT: the same step list with
            the same seeds since the engine start as the control's record (control: restart, leak, lifetime; shipping:
            restart, leak, lifetime, then this snapshot, before the A/B, teacher-forced, warm/cold, perf and canary
            steps that the control does not run). Cumulative-since-start retries are then a matched regression
            comparison (more than the control's = FAIL); when both records' START rows exist (--baseline-start), the
            within-window deltas are compared as well. Peaks are optional at a matched point (the phase sampler is
            still running): the reserve is graded at the phase end. A baseline is required (else INCONCLUSIVE).
usage: e2_ledger_grade.py START.json END.json [--peaks PEAKS.json] [--baseline BASE.json] [--baseline-start BASE_START.json]
       [--reserve-bytes N] [--reserve-only | --matched]
PEAKS.json: {"gpus": {"0": {"max_used_mib": N, "total_mib": N}, ...}, "samples": N} from the queue's nvidia-smi sampler.
Exit 0 PASS, 1 FAIL, 2 INCONCLUSIVE. Only numbers are printed.
"""
import argparse, json, sys

RANKS = ('Worker_PP0_TP0', 'Worker_PP0_TP1', 'Worker_PP1_TP0', 'Worker_PP1_TP1')
GIB = 1024 ** 3


def grade(start, end, peaks=None, baseline=None, reserve_bytes=536870912, reserve_only=False, matched=False, baseline_start=None):
    """reserve_only: the phase has no matched control (instrumented phases, the control itself): the reserve, ooms and
    row coverage decide; a baseline, when given, is reported as overhead evidence only (no retry attribution).
    matched: END and the baseline are matched-point records (same step list and seeds since the engine start): the
    cumulative retries are compared (and the window deltas when both START records have rows); peaks optional."""
    if reserve_only and matched: raise ValueError('reserve_only and matched are exclusive')
    fails, missing, notes, ranks = [], [], [], {}
    reserve = reserve_bytes / GIB
    for r in RANKS:
        e = (end.get('per_rank') or {}).get(r); s = ((start or {}).get('per_rank') or {}).get(r) or {}
        if not e or any(k not in e for k in ('alloc_retries', 'ooms', 'peak_gib', 'reserved_gib')):
            missing.append(r + ': no allocator row at the phase end'); continue
        row = dict(peak_gib=e['peak_gib'], reserved_gib=e['reserved_gib'], ooms=e['ooms'], retries_since_start=e['alloc_retries'])
        if e['ooms'] > 0: fails.append(r + ': ooms')
        if s.get('alloc_retries') is not None:
            row['retries_at_phase_start'] = s['alloc_retries']; row['retries_phase_delta'] = e['alloc_retries'] - s['alloc_retries']
            if row['retries_phase_delta'] < 0: fails.append(r + ': retry counter went backwards')
        else:
            notes.append(r + ': no allocator row at the phase start (none before the first staged call); within-phase delta not available')
        if baseline is not None:
            b = (baseline.get('per_rank') or {}).get(r)
            if not b or 'peak_gib' not in b or 'alloc_retries' not in b:
                (notes if reserve_only else missing).append(r + ': no baseline allocator row')
            else:
                row['baseline_peak_gib'] = b['peak_gib']; row['baseline_retries'] = b['alloc_retries']
                row['peak_delta_gib'] = round(e['peak_gib'] - b['peak_gib'], 3)
                if e['alloc_retries'] > b['alloc_retries']:
                    (notes if reserve_only else fails).append(r + ': more allocator retries since start than the ' + ('reference (not attributed: no matched control)' if reserve_only else 'matched control'))
                bs = ((baseline_start or {}).get('per_rank') or {}).get(r) or {}
                if matched and bs.get('alloc_retries') is not None and 'retries_phase_delta' in row:
                    row['baseline_retries_delta'] = b['alloc_retries'] - bs['alloc_retries']
                    if row['retries_phase_delta'] > row['baseline_retries_delta']: fails.append(r + ': more allocator retries in the matched window than the control')
                elif matched and baseline_start is not None:
                    notes.append(r + ': matched-window delta not available (a START record without allocator rows)')
        ranks[r] = row
    gpus = {}
    if peaks is None:
        (notes if matched else missing).append('no sampled GPU peaks' + (' (matched point: the reserve is graded at the phase end)' if matched else ''))
    else:
        base_used = {str(g['index']): g['used_gib'] for g in (baseline or {}).get('gpus', []) if 'index' in g and 'used_gib' in g}
        for idx, p in sorted((peaks.get('gpus') or {}).items()):
            row = dict(max_used_gib=round(p['max_used_mib'] / 1024, 3), total_gib=round(p['total_mib'] / 1024, 3))
            row['free_at_peak_gib'] = round(row['total_gib'] - row['max_used_gib'], 3)
            if (p['total_mib'] - p['max_used_mib']) * 2 ** 20 < reserve_bytes: fails.append('gpu ' + idx + ': less than the reserve free at the sampled peak')
            if idx in base_used:
                row['baseline_used_gib'] = base_used[idx]; row['used_delta_gib'] = round(row['max_used_gib'] - base_used[idx], 3)
            elif baseline is not None:
                (notes if reserve_only else missing).append('gpu ' + idx + ': no baseline used-memory row')
            gpus[idx] = row
        if len(gpus) != 4: missing.append('sampled peaks do not cover four GPUs')
        if not peaks.get('samples'): missing.append('no GPU samples taken')
    if baseline is None: (notes if reserve_only else missing).append('no matched control record' + (' (reserve-only phase)' if reserve_only else ''))
    v = 'FAIL' if fails else 'INCONCLUSIVE' if missing else 'PASS'
    # retry_only: every failure is a retry-count excess against the control (attributable only with matched exposure);
    # anything else (ooms, counter backwards, reserve) is a hard failure the queue must not defer to the attribution step
    retry_only = bool(fails) and all('more allocator retries' in f for f in fails)
    return dict(verdict=v, retry_only=retry_only, mode='reserve-only' if reserve_only else 'matched' if matched else 'full', fails=sorted(set(fails)), missing=sorted(set(missing)), notes=sorted(set(notes)), reserve_gib=round(reserve, 3),
                ranks=ranks, gpus=gpus,
                scope='reserve = free memory at the sampled peak; baseline deltas = overhead evidence; counters cumulative since process start'
                      + ('; matched point: same step list and seeds since the engine start as the control' if matched else ''))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('start'); p.add_argument('end'); p.add_argument('--peaks'); p.add_argument('--baseline'); p.add_argument('--baseline-start')
    p.add_argument('--reserve-bytes', type=int, default=536870912)
    g = p.add_mutually_exclusive_group()
    g.add_argument('--reserve-only', action='store_true', help='no matched control for this phase: reserve/ooms/rows decide, baseline reported only')
    g.add_argument('--matched', action='store_true', help='END and --baseline are matched-point records (same steps and seeds since start); peaks optional')
    a = p.parse_args()
    load = lambda f: json.load(open(f)) if f else None
    out = grade(load(a.start), load(a.end), load(a.peaks), load(a.baseline), a.reserve_bytes, a.reserve_only, a.matched, load(a.baseline_start))
    print(json.dumps(out, sort_keys=True))
    sys.exit({'PASS': 0, 'FAIL': 1, 'INCONCLUSIVE': 2}[out['verdict']])
