#!/usr/bin/env python3
"""Alpha-perf box arms and grade: prefill speed and decode-during-prefill on the composed alpha build.

The arm is held-f's (e3_native41_progress_box.arm), called unchanged. Two bands:

* band 1: held-f's eight arms and manifest (workloads/q18a-native41-progress.json), labels and order unchanged, so
  each number is directly comparable with held-f run 2. The history stays below the 97,152-row arena (sparse/dense
  plan unchanged below it).
* band 2: burst only, live and solo, a fresh prompt of 131,094 tokens (workloads/q18a-alpha-perf-band2.json), so
  about 33 native chunks run past 97,152 rows on the dense route. Labels continue held-f's scheme with round indexes
  2..5: burst-solo-2, burst-live-2, burst-live-3, burst-solo-3, then the optional ABBA repeat with 4 and 5. A round
  index is part of the cache salt (held-f's arm), so no band 2 arm can hit a band 1 prefix.

Live cohort budget per band: band 1 is held-f's 16,384; band 2 declares 32,768 so the eight decoders stay open
through the first token up to about 5 s per chunk (47.5 tok/s per stream; held-f's standing_budget would refuse it).

The grade takes held-f's progress join unchanged (e3_native41_progress_box.progress_receipts) and adds, per arm: TTFT,
prefill tok/s, s/chunk, decode during/only and ratio, min stream ratio, gap p99/max, commands, the chunk breakdown
per history band, device memory from the `flash-s5 memory` receipts, allocator-retry warnings, and a validity row
with an explicit PASS/FAIL on crash-free completion and TP agreement (0 `disagree` lines). Then the same per band.
Speed is reported, never gated. The verdict is about the run being valid. Receipts hold counts, hashes, identities
and host clock stamps, never prompts or auth.

Route proof, as declared in the delivery: with FLASH_S5_NATIVE_QSA_TRACE unset (as l-f3 configures the dense policy)
no per-layer route line exists. The policy is proved by the queue's preflight on EXTRA_ENV, and the band 2 rows report
s/chunk per history band beside the known 4.3-4.4 s/chunk of the current policy past the boundary.
"""
import argparse
import json
from pathlib import Path
import re

import e3_native41_autoarm_box as base
import e3_native41_progress_box as progress
from e3_native_qualification_driver import exception_details
from e3_native_runtime_client import NativeRuntimeClient

GRADE_SCHEMA = 'alpha-perf-grade/1'
ARENA_ROWS = 97152
# One history band for everything past the arena boundary up to band 2's end, beside held-f's bands.
progress.BANDS = progress.BANDS[:-1] + ((65536, ARENA_ROWS), (ARENA_ROWS, 114688), (114688, 131072), (131072, 197632))
BAND1 = tuple(progress.ARM_ORDER)
BAND2_REQUIRED = (('burst', 'solo', 2), ('burst', 'live', 2), ('burst', 'live', 3), ('burst', 'solo', 3))
BAND2_OPTIONAL = (('burst', 'solo', 4), ('burst', 'live', 4), ('burst', 'live', 5), ('burst', 'solo', 5))
ARM_ORDER = BAND1 + BAND2_REQUIRED + BAND2_OPTIONAL
INDEXES = tuple(sorted({a[2] for a in ARM_ORDER}))
BAND_BUDGET = {1: 16384, 2: 32768}
BAND_WORKLOAD = {1: 'q18a-native41-progress', 2: 'q18a-alpha-perf-band2'}
LOG_LAG_S = progress.LOG_LAG_S
MEMORY_LINE = re.compile(r'flash-s5 memory point=(\w+) rank=(\d+) txn_seq=(\d+) chunk=(\d+)-(\d+) allocated=(-?\d+) '
                         r'reserved=(-?\d+) inactive_split=(-?\d+) alloc_retries=(-?\d+) ooms=(-?\d+) free=(-?\d+)')
RETRY_LINE = re.compile(r'memory allocation failed with OOM on device (\d+) while trying to allocate (\d+) bytes')
# Held-f run 2 (results/box/q18a-native41-progress-run-2/README.md), the comparison the band 1 arms are repeated for.
REFERENCE_HELD_F_RUN2 = {
    'current-solo-0': dict(ttft_s=48.9, prefill_tok_s=1339), 'burst-solo-0': dict(ttft_s=43.7, prefill_tok_s=1497),
    'burst-live-0': dict(ttft_s=116.6, prefill_tok_s=561, decode_ratio=0.860, min_stream_ratio=0.859),
    'current-live-0': dict(ttft_s=188.4, prefill_tok_s=347, decode_ratio=0.901, min_stream_ratio=0.901),
    'current-live-1': dict(ttft_s=190.2, prefill_tok_s=344, decode_ratio=0.903, min_stream_ratio=0.903),
    'burst-live-1': dict(ttft_s=114.4, prefill_tok_s=572, decode_ratio=0.854, min_stream_ratio=0.854),
    'burst-solo-1': dict(ttft_s=44.8, prefill_tok_s=1459), 'current-solo-1': dict(ttft_s=49.9, prefill_tok_s=1312)}


def band_of_index(index):
    return 1 if index in (0, 1) else 2


def required_labels():
    return [progress.label(*x) for x in BAND1 + BAND2_REQUIRED]


def optional_labels():
    return [progress.label(*x) for x in BAND2_OPTIONAL]


def standing_budget(data, band):
    """The manifest's declared cohort budget, exactly the band's."""
    block = data.get('progress') if type(data) is dict else None
    budget = (block or {}).get('standing_generation_budget')
    if (type(block) is not dict or block.get('schema') != 'native41-progress-workload/1'
            or budget != BAND_BUDGET[band] or data.get('workload') != BAND_WORKLOAD[band]):
        raise ValueError('alpha-perf-workload-declaration-invalid')
    return budget


# --- engine log: memory, retries, disagreements, per arm ------------------------

def read_stamped(paths, keep):
    """(stamp, text) of the followed log lines that match ``keep``; plain text lines, never parsed as JSON."""
    rows = []
    for path in paths:
        path = Path(path)
        if not path.is_file():
            continue
        with path.open('r', errors='replace') as handle:
            for line in handle:
                if keep(line):
                    rows.append((progress.docker_stamp(line), line))
    return rows


def is_disagreement(line):
    return 'disagree' in line and progress.PROGRESS_LINE not in line


def is_tracked(line):
    return is_disagreement(line) or 'flash-s5 memory point=' in line or RETRY_LINE.search(line) is not None


def in_span(stamp, arm):
    start, end = arm.get('wall_start_s'), arm.get('wall_end_s')
    if type(start) not in (int, float) or type(end) not in (int, float) or stamp is None:
        return False
    return start <= stamp <= end + LOG_LAG_S


def device_memory(rows):
    """Per-rank receipts at chunk open: min free, max reserved, max allocated, max inactive_split, counter values."""
    ranks = {}
    for _, text in rows:
        m = MEMORY_LINE.search(text)
        if not m or m.group(1) != 'open':
            continue
        rank = ranks.setdefault(int(m.group(2)), dict(open_receipts=0, min_free_mib=None, max_reserved_mib=None,
                                                      max_allocated_mib=None, max_inactive_split_mib=None,
                                                      max_alloc_retries=None, max_ooms=None))
        rank['open_receipts'] += 1
        for field, group, pick, scale in (('min_free_mib', 11, min, 2 ** 20), ('max_reserved_mib', 7, max, 2 ** 20),
                                          ('max_allocated_mib', 6, max, 2 ** 20),
                                          ('max_inactive_split_mib', 8, max, 2 ** 20),
                                          ('max_alloc_retries', 9, max, 1), ('max_ooms', 10, max, 1)):
            value = int(m.group(group)) / scale
            value = int(value) if scale == 1 else round(value, 1)
            rank[field] = value if rank[field] is None else pick(rank[field], value)
    return {str(k): v for k, v in sorted(ranks.items())}


def log_view(arm, tracked):
    mine = [(s, t) for s, t in tracked if in_span(s, arm)]
    retries = [m for m in (RETRY_LINE.search(t) for _, t in mine) if m]
    return dict(disagree_lines=sum(1 for _, t in mine if is_disagreement(t)), device_memory=device_memory(mine),
                allocator_retry_warnings=len(retries),
                max_retry_request_mib=round(max(int(m.group(2)) for m in retries) / 2 ** 20, 1) if retries else None)


# --- per-arm table --------------------------------------------------------------

def stream_ratios(during, only):
    a, b = (during or {}).get('stream_rates_tokens_per_s'), (only or {}).get('stream_rates_tokens_per_s')
    if not a or not b or len(a) != len(b):
        return None
    ratios = [progress.ratio(x, y) for x, y in zip(a, b)]
    return None if any(r is None for r in ratios) else ratios


def arm_table(arm, row, logs):
    """The held-f table row of one arm, from its own receipts."""
    fresh = arm.get('fresh') or {}
    client = arm.get('client_basis') or {}
    during, only = client.get('decode_window'), arm.get('decode_only')
    chunks = (row or {}).get('chunks_logged')
    ttft = fresh.get('ttft_s')
    ratios = stream_ratios(during, only)
    delta = arm.get('progress_delta') or {}
    d_rate, o_rate = (during or {}).get('aggregate_tokens_per_s'), (only or {}).get('aggregate_tokens_per_s')
    return dict(label=arm.get('label'), band=arm.get('band'), mode=arm.get('mode'), cohort=arm.get('cohort'),
                prompt_tokens=fresh.get('prompt_tokens_observed'), ttft_s=ttft,
                prefill_tok_s=client.get('prefill_tokens_per_s'),
                s_per_chunk=(ttft / chunks if type(ttft) in (int, float) and type(chunks) is int and chunks else None),
                chunks=chunks, decode_during_tok_s=d_rate, decode_only_tok_s=o_rate,
                decode_ratio=progress.ratio(d_rate, o_rate), min_stream_ratio=min(ratios) if ratios else None,
                gap_p99_ms=(during or {}).get('gap_ms_p99'), gap_max_ms=(during or {}).get('gap_ms_max'),
                commands=delta.get('commands'), direct_commands=delta.get('direct_commands'),
                wait_sends=delta.get('wait_sends'), stock_rpcs=delta.get('stock_rpcs'),
                all_open_share_of_span=client.get('all_open_share_of_span'),
                chunk_breakdown=(row or {}).get('bands'), chunk_breakdown_all=(row or {}).get('all_counted'),
                device_memory=logs['device_memory'], allocator_retry_warnings=logs['allocator_retry_warnings'],
                max_retry_request_mib=logs['max_retry_request_mib'],
                append_ttft_s=(arm.get('append') or {}).get('ttft_s'), metrics=arm.get('metrics'))


def validity(arm, ingress, watch, logs, reasons):
    """Explicit per-arm PASS/FAIL: crash-free completion and TP agreement. Speed is not in it."""
    verdict_of_watch = (watch or {}).get('verdict')
    crash_free = arm.get('complete') is True and verdict_of_watch != 'DEAD' and watch is not None
    tp = logs['disagree_lines'] == 0
    return dict(verdict='PASS' if crash_free and tp and (ingress or {}).get('verdict') == 'PASS' and not reasons
                else 'FAIL',
                crash_free_completion='PASS' if crash_free else 'FAIL', tp_agreement='PASS' if tp else 'FAIL',
                disagree_lines=logs['disagree_lines'], engine_watch=verdict_of_watch,
                ingress=(ingress or {}).get('verdict'), arm_reasons=list(arm.get('reasons') or []),
                progress_reasons=list(reasons))


def median_of(rows, key):
    return progress.median([r.get(key) for r in rows])


def group_summary(rows):
    """Medians of the band's table rows for one mode and cohort, and the history-band chunk walls."""
    out = {k: median_of(rows, k) for k in ('ttft_s', 'prefill_tok_s', 's_per_chunk', 'decode_during_tok_s',
                                           'decode_only_tok_s', 'decode_ratio', 'min_stream_ratio', 'gap_p99_ms',
                                           'gap_max_ms', 'commands', 'direct_commands', 'wait_sends', 'stock_rpcs')}
    out['arms'] = [r['label'] for r in rows]
    bands = sorted({b for r in rows for b in (r.get('chunk_breakdown') or {})}, key=lambda b: int(b.split('-')[0]))
    out['history_bands'] = {}
    for band in bands:
        cells = [(r.get('chunk_breakdown') or {}).get(band) or {} for r in rows]
        out['history_bands'][band] = dict(
            wall_s=progress.median([c.get('wall_s') for c in cells]),
            open_interval_s=progress.median([c.get('open_interval_s') for c in cells]),
            commands=progress.median([c.get('commands') for c in cells]),
            components_s={k: progress.median([(c.get('components_s') or {}).get(k) for c in cells])
                          for k in progress.COMPONENTS})
    return out


def memory_summary(rows):
    ranks = {}
    for r in rows:
        for rank, row in (r.get('device_memory') or {}).items():
            cell = ranks.setdefault(rank, dict(min_free_mib=None, max_reserved_mib=None, max_alloc_retries=None,
                                               max_ooms=None, max_inactive_split_mib=None, open_receipts=0))
            cell['open_receipts'] += row['open_receipts']
            for field, pick in (('min_free_mib', min), ('max_reserved_mib', max), ('max_alloc_retries', max),
                                ('max_ooms', max), ('max_inactive_split_mib', max)):
                cell[field] = row[field] if cell[field] is None else pick(cell[field], row[field])
    return ranks


def grade(arms, ingress, checks, logs, not_run=(), watches=None):
    """Valid-run verdict over the declared arms; every speed number is descriptive."""
    missing, failures = [], []
    watches = watches or {}
    required, optional = required_labels(), optional_labels()
    labels = [a.get('label') for a in arms]
    not_run = sorted(set(not_run))
    if not set(not_run) <= set(optional):
        missing.append('undeclared-budget-skip:' + ','.join(sorted(set(not_run) - set(optional))))
    if len(labels) != len(set(labels)) or not set(x for x in labels if x) <= set(required + optional):
        missing.append('undeclared-or-repeated-arm')
    for name in required:
        if name not in labels:
            missing.append('declared-arm-missing:' + name)
    for name in optional:
        if name not in labels and name not in not_run:
            missing.append('optional-arm-neither-run-nor-skipped-by-budget:' + name)
    for r in arms:
        name = r.get('label') or 'unlabelled'
        if r.get('complete') is not True:
            missing.extend(name + ':' + x for x in (r.get('reasons') or ['incomplete']))
        if name != 'unlabelled':
            band = r.get('band')
            if band != band_of_index(r.get('index')):
                missing.append(name + ':band-not-the-round-index-band')
    for name, data in sorted(ingress.items()):
        if (data or {}).get('verdict') == 'FAIL':
            failures.append('ingress-' + name + ':INVALID')
        elif (data or {}).get('verdict') != 'PASS':
            missing.append('ingress-' + name + ':' + str((data or {}).get('verdict')))
    for name, data in sorted(checks.items()):
        if (data or {}).get('verdict') == 'FAIL':
            failures.append(name + ':FAIL')
        elif (data or {}).get('verdict') != 'PASS':
            missing.append(name + ':' + str((data or {}).get('verdict')))
    records, errors = progress.read_lines(logs)
    known = [a for a in arms if a.get('label')]
    receipts = progress.progress_receipts(known, records, errors)
    if not records:
        missing.append('no-progress-lines-in-the-engine-log')
    for problem in receipts['problems']:
        missing.append('progress:' + problem)
    if receipts['dropped_rows'] or receipts['duplicate_worker_rows']:
        missing.append('progress:rows-dropped-or-duplicated')
    if receipts['unstamped_engine_flushes'] or receipts['engine_flushes_outside_every_arm']:
        missing.append('progress:engine-lines-outside-every-arm-span')
    if receipts['engine_flushes_in_two_arms']:
        missing.append('progress:engine-line-in-two-arm-spans')
    for name, row in sorted(receipts['arms'].items()):
        missing.extend('progress-%s:%s' % (name, x) for x in row['reasons'])
    tracked = read_stamped(logs, is_tracked)
    table, valid = {}, {}
    for arm in known:
        name = arm['label']
        row = receipts['arms'].get(name)
        view = log_view(arm, tracked)
        table[name] = arm_table(arm, row, view)
        valid[name] = validity(arm, ingress.get(name), watches.get(name), view, (row or {}).get('reasons') or [])
        if valid[name]['tp_agreement'] != 'PASS':
            failures.append(name + ':tp-disagree')
        if valid[name]['engine_watch'] == 'DEAD':
            failures.append(name + ':engine-dead')
        elif valid[name]['engine_watch'] is None:
            missing.append(name + ':engine-watch-record-missing')
    # A declared arm without a record still gets its explicit row: stopped by the engine watch, or never entered.
    for name in required + optional:
        if name in valid:
            continue
        verdict_of_watch = (watches.get(name) or {}).get('verdict')
        valid[name] = dict(verdict='NOT_RUN' if name in not_run or name not in watches else 'FAIL',
                           crash_free_completion='FAIL' if name in watches else 'NOT_RUN',
                           tp_agreement='NOT_RUN', disagree_lines=None, engine_watch=verdict_of_watch,
                           ingress=(ingress.get(name) or {}).get('verdict'), arm_reasons=['no-arm-record'],
                           progress_reasons=[])
        if verdict_of_watch == 'DEAD':
            failures.append(name + ':engine-dead')
    # TP disagreement anywhere in the followed engine log, also between the arm spans, makes the run invalid.
    between = [t for s, t in tracked if is_disagreement(t) and not any(in_span(s, a) for a in known)]
    if between:
        failures.append('tp-disagree-outside-every-arm-span')
    summary = {}
    for band in (1, 2):
        rows = [table[a.get('label')] for a in known if a.get('band') == band]
        groups = {}
        for mode in progress.MODES:
            for cohort in progress.COHORTS:
                cell = [r for r in rows if (r['mode'], r['cohort']) == (mode, cohort)]
                if cell:
                    groups[mode + '-' + cohort] = group_summary(cell)
        summary['band%d' % band] = dict(groups=groups, device_memory=memory_summary(rows),
                                        allocator_retry_warnings=sum(r['allocator_retry_warnings'] for r in rows),
                                        max_retry_request_mib=max((r['max_retry_request_mib'] or 0) for r in rows)
                                        if rows else None,
                                        arm_verdicts={r['label']: valid[r['label']]['verdict'] for r in rows})
    verdict = 'FAIL' if failures else 'INCONCLUSIVE' if missing else 'PASS'
    return {'schema': GRADE_SCHEMA, 'verdict': verdict, 'failures': sorted(set(failures)),
            'missing': sorted(set(missing)), 'required_arms': required, 'optional_arms': optional,
            'not_run_by_budget': not_run, 'disagree_lines_outside_every_arm': len(between), 'table': table, 'validity': valid, 'progress': receipts, 'summary': summary,
            'reference_held_f_run2': REFERENCE_HELD_F_RUN2, 'speed_is_gate': False, 'target_is_gate': False,
            'all_declared_arms_retained': True, 'retry_selected': False, 'picked': False,
            'components': dict(cadence='previous native apply -> this submit (owner progress cadence)',
                               send='submit -> broadcast sent', fifo='sent -> critical worker handler enter',
                               host='worker handler enter -> exit', hol='handler exit -> response MQ write',
                               mq='MQ write -> engine receive', apply='engine receive -> owner apply'),
            'basis': 'band 1 repeats held-f run 2 on the composed engine; band 2 runs past the 97,152-row arena on the '
                     'dense route with the same arm; per chunk, the critical rank of each native command; live arms '
                     'count chunks begun with all 8 decoders, solo arms chunks begun with none',
            'scope': 'prefill speed and decode-during-prefill on the composed alpha engine; validity gates, speed '
                     'descriptive; gains from separate experiments are never multiplied',
            'route_receipts': 'FLASH_S5_NATIVE_QSA_TRACE unset, so no per-layer route line exists; the dense policy is '
                              'proved by the queue preflight on EXTRA_ENV; s/chunk per history band is reported'}


def load_pairs(rows):
    return {x.split('=', 1)[0]: base.load(x.split('=', 1)[1]) for x in rows}


def run_arm(a):
    band = band_of_index(a.index)
    try:
        data = json.loads(Path(a.workload).read_text())
        budget = standing_budget(data, band)
        problems, cfg = progress.shared.validate_manifest(data)
        if problems or cfg['repeats'] != 1:
            raise ValueError('frozen-native-workload-invalid')
        cfg = dict(cfg, timeouts=dict(cfg['timeouts'], read_s=base.first_advance_read_timeout(cfg)))
        client = NativeRuntimeClient(a.url, a.run_id, epoch=a.epoch, timeout_s=a.control_timeout_s)
        # The receipt poller's own loopback client: status only, same run and epoch.
        poll = NativeRuntimeClient(a.url, a.run_id, epoch=a.epoch, timeout_s=a.control_timeout_s)
        result = progress.arm(a, cfg, client, poll_client=poll, budget=budget)
    except Exception as exc:  # noqa: BLE001 - retained, never retried
        result = {'schema': progress.ARM_SCHEMA, 'kind': 'auto', 'mode': a.mode, 'cohort': a.cohort, 'index': a.index,
                  'label': progress.label(a.mode, a.cohort, a.index), 'complete': False,
                  'reasons': ['arm-refused'], 'error': exception_details(exc)}
    result['band'] = band
    result['workload'] = BAND_WORKLOAD[band]
    with open(a.out, 'x') as handle:
        json.dump(result, handle, sort_keys=True)
    print('ALPHA_PERF_ARM %s %s' % (result['label'], 'COMPLETE' if result.get('complete') else 'INCOMPLETE'))
    return 0 if result.get('complete') is True else 2


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='action', required=True)
    r = sub.add_parser('arm')
    r.add_argument('--url', required=True)
    r.add_argument('--model', required=True)
    r.add_argument('--run-id', required=True)
    r.add_argument('--epoch', type=int, required=True)
    r.add_argument('--mode', choices=progress.MODES, required=True)
    r.add_argument('--cohort', choices=progress.COHORTS, required=True)
    r.add_argument('--index', type=int, choices=INDEXES, required=True)
    r.add_argument('--workload', required=True)
    r.add_argument('--control-timeout-s', type=float, default=120.0)
    r.add_argument('--out', required=True)
    g = sub.add_parser('grade')
    g.add_argument('--arm', action='append', required=True)
    g.add_argument('--ingress', action='append', default=[], help='label=path')
    g.add_argument('--watch', action='append', default=[], help='label=engine-watch record path')
    g.add_argument('--check', action='append', default=[], help='label=path')
    g.add_argument('--log', action='append', default=[], help='followed engine log (stdout/stderr)')
    g.add_argument('--not-run', action='append', default=[], help='optional arm skipped by the box budget')
    g.add_argument('--out', required=True)
    a = p.parse_args(argv)
    if Path(a.out).exists():
        return 2
    if a.action == 'arm':
        return run_arm(a)
    arms = [base.load(x) or {'label': None, 'complete': False, 'reasons': ['arm-artifact-unreadable:' + Path(x).name]}
            for x in a.arm]
    result = grade(arms, load_pairs(a.ingress), load_pairs(a.check), a.log, a.not_run, load_pairs(a.watch))
    with open(a.out, 'x') as handle:
        json.dump(result, handle, sort_keys=True)
    print('ALPHA_PERF_GRADE ' + result['verdict'])
    return {'PASS': 0, 'FAIL': 1}.get(result['verdict'], 2)


if __name__ == '__main__':
    raise SystemExit(main())
