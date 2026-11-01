#!/usr/bin/env python3
"""Step A box arms and grade: the fused dense QSA path against the stepped one, matched per arm on one engine.

The arm is held-f's (e3_native41_progress_box.arm), called unchanged with mode burst; the seams below add one typed control
and change no other arm step. Control order: pause, native_progress {burst}, native_qsa_fused {fused}, release_auto_arm,
resume, then status only. The record keeps ``mode`` burst (the progress lines carry it) and adds ``selection``.

Two bands, as alpha-perf's manifests: band 1 (rounds 0, 1; 65,445 fresh then the append) in ABBA, band 2 (round 2; 131,094
fresh, append 135,085, chunks past 97,152 rows, so multi-bucket on every QSA layer there). The round index is part of the
cache salt through the label, so no arm can hit another arm's prefix.

The grade is alpha-perf's progress join and table (e3_alpha_perf_box) plus:
  mechanism   per arm and overall PASS/FAIL/INC, HARNESS or ENGINE on a failure: crash-free, 0 ``disagree``, 0
              ``tp-ledger-mismatch``, 0 ``tp-collective-stall``, 0 ``fused-profile-*``, and engagement. A fused arm needs a
              ``fused_commands`` delta of at least 12 per auto-owned chunk, a ``chunks`` delta equal to the auto-owned chunk
              count and no profile query. A stepped arm needs every fused counter delta 0 and at least 12 profile queries on
              every chunk (held-f run 2's engine stream: 12 on each of 528 chunks). A fused arm that did not engage is INC.
  performance reported, never gated: per-arm TTFT, prefill tok/s, s/chunk, decode during and only and their ratio, min and
              p10 of the 8 streams, gap p99 and max, commands per chunk, chunk_breakdown, device memory, per band.
  memory      per arm from the `flash-s5 memory` receipts (chunk open and release): peak allocated, peak reserved, min free,
              allocator retries and OOMs in the arm, fused against the matched stepped arm. Reported, never a verdict input,
              except a real OOM warning counted as a line.
  report      three separate headed sections: mechanism, performance, release readiness.

Receipts hold counts, hashes, identities and host clock stamps, never prompts or auth.
"""
import argparse
import contextlib
import json
from pathlib import Path
import re
import statistics

import e3_alpha_perf_box as ap
from e3_native_qualification_driver import exception_details
from e3_native_runtime_client import NativeRuntimeClient

progress = ap.progress
base = ap.base
GRADE_SCHEMA = 'step-a-grade/1'
SELECTIONS = ('stepped', 'fused')
BAND1 = (('stepped', 'solo', 0), ('fused', 'solo', 0), ('fused', 'live', 0), ('stepped', 'live', 0),
         ('stepped', 'live', 1), ('fused', 'live', 1), ('fused', 'solo', 1), ('stepped', 'solo', 1))
BAND2 = (('fused', 'live', 2), ('stepped', 'live', 2), ('stepped', 'solo', 2), ('fused', 'solo', 2))
ARM_ORDER = BAND1 + BAND2
INDEXES = (0, 1, 2)
QSA_LAYERS_PER_CHUNK = 12
FUSED_COUNTERS = ('selections', 'chunks', 'grants', 'fused_commands', 'run_ahead_polls', 'ledger_checks')
CONTROL_SEQUENCE = ['status', 'pause', 'native_progress', 'native_qsa_fused', 'release_auto_arm', 'resume']
# Engine signatures in the followed log. Each stops the arm row at FAIL, classification ENGINE.
SIGNATURES = (('tp-ledger-mismatch', 'tp-ledger-mismatch'), ('tp-collective-stall', 'tp-collective-stall'),
              ('tp-collective-stall', 'tp_collective_stall'), ('fused-profile', 'fused-profile-'))
# The whole `flash-s5 memory` receipt, both points. ap.MEMORY_LINE stops at free=; the same words.
MEMORY_LINE = re.compile(r'flash-s5 memory point=(\w+) rank=(\d+) txn_seq=(\d+) chunk=(\d+)-(\d+) allocated=(-?\d+) '
                         r'reserved=(-?\d+) inactive_split=(-?\d+) alloc_retries=(-?\d+) ooms=(-?\d+) free=(-?\d+)')
RELEASE_READINESS = ('Not assessed by this queue. It holds no lifecycle, numerics or leak evidence; nothing in this report '
                     'supports a release decision.')


def label(selection, cohort, index):
    return '%s-%s-%d' % (selection, cohort, index)


def required_labels():
    return [label(*x) for x in ARM_ORDER]


# --- the arm --------------------------------------------------------------------------------------------

def fused_view(status):
    block = status.get('native_qsa_fused') if type(status) is dict else None
    if type(block) is not dict:
        return None
    return {k: block.get(k) for k in ('selected', 'run_ahead', 'stall_s') + FUSED_COUNTERS}


def fused_delta(after, before):
    return {k: progress.delta(after, before, k) for k in FUSED_COUNTERS}


def release_fused(fused):
    """held-f's release with the selection between native_progress and release_auto_arm; its checks, then the reply's."""
    def release(client, record, mode):
        reply = client.pause()
        record['release_pause'] = {'route': reply.get('route'), 'route_generation': reply.get('route_generation')}
        try:
            got = client.call('native_progress', mode=mode)
            record['mode_switch'] = {k: got.get(k) for k in ('mode', 'previous', 'installed', 'route',
                                                             'route_generation', 'engine_epoch')}
            got = client.call('native_qsa_fused', fused=fused)
            record['selection_switch'] = {k: got.get(k) for k in ('fused', 'previous', 'route', 'route_generation',
                                                                  'engine_epoch')}
            got = client.call('release_auto_arm')
            record['release'] = {k: got.get(k) for k in ('released', 'engine_epoch')}
        finally:
            record['release_resume'] = client.resume().get('resumed')
        switch = record['mode_switch']
        if switch.get('mode') != mode or switch.get('installed') != 'burst' or switch.get('route') != 'full':
            raise ValueError('native-progress-mode-unproved')
        selection = record['selection_switch']
        if selection.get('fused') is not fused or selection.get('route') != 'full':
            raise ValueError('native-qsa-fused-selection-unproved')
        if record['release'].get('released') is not True:
            raise ValueError('auto-arm-release-unproved')
    return release


def selection_reasons(record, fused):
    """The selection's receipts beside held-f's. Engagement is the grade's, on the same numbers."""
    reasons = []
    history = record.get('control_history') or []
    if [x for i, x in enumerate(history) if x != 'status' or i == 0][:6] != CONTROL_SEQUENCE:
        reasons.append('control-sequence-not-pause-progress-selection-release-resume')
    if history.count('native_qsa_fused') != 1:
        reasons.append('arm-sent-the-selection-more-than-once')
    views = {s.get('label'): s for s in record.get('progress') or []}
    before = (views.get('before-fresh') or {}).get('native_qsa_fused')
    after = (views.get('after-fresh') or {}).get('native_qsa_fused')
    if not before or not after:
        reasons.append('fused-status-missing')
        return reasons
    if before.get('selected') is not fused or after.get('selected') is not fused:
        reasons.append('fused-selection-not-this-arm')
    record['fused_delta'] = fused_delta(after, before)
    if record['fused_delta']['selections'] != 0:
        reasons.append('fused-selection-moved-during-the-fresh-prompt')
    return reasons


@contextlib.contextmanager
def arm_seams(selection):
    """The four names held-f's arm reads, for this one process's one arm; restored on exit."""
    fused = selection == 'fused'
    saved = {k: getattr(progress, k) for k in ('label', 'release', 'progress_reasons', 'progress_view', 'ACTIONS')}
    actions = base.ARM_ACTIONS
    original_reasons, original_view = saved['progress_reasons'], saved['progress_view']

    def reasons(record, mode, cohort):
        history = record.get('control_history') or []
        view = dict(record, control_history=[x for x in history if x != 'native_qsa_fused'])
        out = original_reasons(view, mode, cohort)
        if 'progress_delta' in view:
            record['progress_delta'] = view['progress_delta']
        return out + selection_reasons(record, fused)

    def view(status):
        row = original_view(status)
        if row is None:
            return None
        block = fused_view(status)
        if block is None:
            raise ValueError('native-qsa-fused-not-installed')
        return dict(row, native_qsa_fused=block)
    progress.label = lambda _mode, cohort, index: label(selection, cohort, index)
    progress.release = release_fused(fused)
    progress.progress_reasons = reasons
    progress.progress_view = view
    progress.ACTIONS = saved['ACTIONS'] | {'native_qsa_fused'}
    base.ARM_ACTIONS = dict(actions, auto=actions['auto'] | {'native_qsa_fused'})
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(progress, name, value)
        base.ARM_ACTIONS = actions


def run_arm(a):
    band = ap.band_of_index(a.index)
    selection = a.mode
    try:
        data = json.loads(Path(a.workload).read_text())
        budget = ap.standing_budget(data, band)
        problems, cfg = progress.shared.validate_manifest(data)
        if problems or cfg['repeats'] != 1:
            raise ValueError('frozen-native-workload-invalid')
        cfg = dict(cfg, timeouts=dict(cfg['timeouts'], read_s=base.first_advance_read_timeout(cfg)))
        client = NativeRuntimeClient(a.url, a.run_id, epoch=a.epoch, timeout_s=a.control_timeout_s)
        poll = NativeRuntimeClient(a.url, a.run_id, epoch=a.epoch, timeout_s=a.control_timeout_s)
        args = argparse.Namespace(**dict(vars(a), mode='burst'))
        with arm_seams(selection):
            result = progress.arm(args, cfg, client, poll_client=poll, budget=budget)
    except Exception as exc:  # noqa: BLE001 - retained, never retried
        result = {'schema': progress.ARM_SCHEMA, 'kind': 'auto', 'mode': 'burst', 'cohort': a.cohort, 'index': a.index,
                  'label': label(selection, a.cohort, a.index), 'complete': False,
                  'reasons': ['arm-refused'], 'error': exception_details(exc)}
    result['selection'] = selection
    result['band'] = band
    result['workload'] = ap.BAND_WORKLOAD[band]
    views = {s.get('label'): s for s in result.get('progress') or []}
    if 'after-append' in views:
        # The append is stock by design; the arm's own proof ran before it.
        result['fused_delta_append'] = fused_delta(views['after-append'].get('native_qsa_fused'),
                                                   views['after-fresh'].get('native_qsa_fused'))
        if any(v != 0 for v in result['fused_delta_append'].values()):
            result['reasons'] = list(result.get('reasons') or []) + ['append-moved-the-fused-counters']
            result['complete'] = False
    with open(a.out, 'x') as handle:
        json.dump(result, handle, sort_keys=True)
    print('STEP_A_ARM %s %s' % (result['label'], 'COMPLETE' if result.get('complete') else 'INCOMPLETE'))
    return 0 if result.get('complete') is True else 2


# --- engine log ----------------------------------------------------------------------------------------

def signature_of(line):
    if progress.PROGRESS_LINE in line:
        return None
    for name, text in SIGNATURES:
        if text in line:
            return name
    return None


def is_tracked(line):
    return ap.is_tracked(line) or signature_of(line) is not None


def memory_receipts(rows):
    """Per rank, both receipt points: peak allocated and reserved, min free, allocator counters' delta within the rows."""
    ranks = {}
    for _, text in rows:
        m = MEMORY_LINE.search(text)
        if not m:
            continue
        rank = ranks.setdefault(int(m.group(2)), dict(receipts=0, peak_allocated_mib=None, peak_reserved_mib=None,
                                                      min_free_mib=None, first_retries=None, last_retries=None,
                                                      first_ooms=None, last_ooms=None))
        rank['receipts'] += 1
        for field, group, pick in (('peak_allocated_mib', 6, max), ('peak_reserved_mib', 7, max), ('min_free_mib', 11, min)):
            value = round(int(m.group(group)) / 2 ** 20, 1)
            rank[field] = value if rank[field] is None else pick(rank[field], value)
        for field, group in (('retries', 9), ('ooms', 10)):
            value = int(m.group(group))
            if rank['first_' + field] is None:
                rank['first_' + field] = value
            rank['last_' + field] = value
    out = {}
    for number, rank in sorted(ranks.items()):
        out[str(number)] = {k: rank[k] for k in ('receipts', 'peak_allocated_mib', 'peak_reserved_mib', 'min_free_mib')}
        out[str(number)].update(alloc_retries=rank['last_retries'] - rank['first_retries'],
                                ooms=rank['last_ooms'] - rank['first_ooms'])
    return out


def memory_worst(ranks):
    if not ranks:
        return None
    rows = list(ranks.values())
    return dict(peak_allocated_mib=max(r['peak_allocated_mib'] for r in rows),
                peak_reserved_mib=max(r['peak_reserved_mib'] for r in rows),
                min_free_mib=min(r['min_free_mib'] for r in rows),
                alloc_retries=sum(r['alloc_retries'] for r in rows), ooms=sum(r['ooms'] for r in rows),
                receipts=sum(r['receipts'] for r in rows))


# --- per-arm rows --------------------------------------------------------------------------------------

def p10(values):
    """Linear-interpolated 10th percentile, the statistics module's inclusive rule: with 8 streams, between the two lowest."""
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=10, method='inclusive')[0]


def chunk_kinds(row):
    """Per chunk of the arm: profile queries and command rows, from the real progress join (kinds added below)."""
    chunks = (row or {}).get('chunks') or []
    return dict(chunks=len(chunks), profile=[c.get('kinds', {}).get('profile', 0) for c in chunks],
                layer=[c.get('segments', {}).get('layer', 0) for c in chunks])


def engagement(arm, row):
    selection = arm.get('selection')
    delta = arm.get('fused_delta') or {}
    owned = (arm.get('progress_delta') or {}).get('chunks')
    kinds = chunk_kinds(row)
    if type(owned) is not int or owned < 1 or not delta or not kinds['chunks']:
        return dict(selection=selection, engaged=False, reasons=['engagement-receipts-missing'], owned_chunks=owned,
                    delta=delta, profile_min=None, profile_max=None)
    reasons = []
    if selection == 'fused':
        if delta.get('chunks') != owned:
            reasons.append('fused-chunks-delta-differs-from-the-auto-owned-chunks')
        if type(delta.get('fused_commands')) is not int or delta['fused_commands'] < QSA_LAYERS_PER_CHUNK * owned:
            reasons.append('fused-commands-below-twelve-per-chunk')
        if max(kinds['profile']) != 0:
            reasons.append('fused-arm-sent-profile-queries')
    else:
        if any(delta.get(k) != 0 for k in FUSED_COUNTERS):
            reasons.append('stepped-arm-moved-a-fused-counter')
        if min(kinds['profile']) < QSA_LAYERS_PER_CHUNK:
            reasons.append('stepped-arm-without-the-stepped-protocol')
    return dict(selection=selection, engaged=not reasons, reasons=reasons, owned_chunks=owned, delta=delta,
                profile_min=min(kinds['profile']), profile_max=max(kinds['profile']))


def table_row(arm, row, view, mine):
    out = ap.arm_table(arm, row, view)
    out['selection'] = arm.get('selection')
    client = arm.get('client_basis') or {}
    during, only = client.get('decode_window'), arm.get('decode_only')
    ratios = ap.stream_ratios(during, only)
    rates = (during or {}).get('stream_rates_tokens_per_s') or []
    out.update(p10_stream_ratio=p10(sorted(ratios)) if ratios else None,
               min_stream_rate=min(rates) if rates else None, p10_stream_rate=p10(sorted(rates)) if rates else None)
    chunks, commands = out.get('chunks'), out.get('commands')
    out['commands_per_chunk'] = commands / chunks if type(commands) is int and type(chunks) is int and chunks else None
    ranks = memory_receipts(mine)
    out['memory'] = dict(ranks=ranks, worst=memory_worst(ranks))
    return out


def arm_verdict(arm, ingress, watch, view, signatures, progress_reasons, engaged):
    """(verdict, classification, reasons): FAIL on an engine signature or a dead engine, INC when unproven, else PASS."""
    verdict_of_watch = (watch or {}).get('verdict')
    engine, harness, unproven = [], [], []
    if verdict_of_watch == 'DEAD':
        engine.append('engine-dead')
    if view['disagree_lines']:
        engine.append('tp-disagree')
    engine += sorted(signatures)
    if (ingress or {}).get('verdict') == 'FAIL':
        harness.append('ingress-invalid')
    elif (ingress or {}).get('verdict') != 'PASS':
        unproven.append('ingress-not-pass')
    if watch is None:
        unproven.append('engine-watch-record-missing')
    if arm.get('complete') is not True:
        unproven += list(arm.get('reasons') or ['incomplete'])
    unproven += ['progress:' + x for x in progress_reasons]
    if not engaged['engaged']:
        unproven += engaged['reasons']
    if engine:
        return 'FAIL', 'ENGINE', engine + harness + unproven
    if harness:
        return 'FAIL', 'HARNESS', harness + unproven
    if unproven:
        return 'INC', None, unproven
    return 'PASS', None, []


def group_summary(rows):
    out = ap.group_summary(rows)
    for key in ('p10_stream_ratio', 'commands_per_chunk', 'min_stream_rate', 'p10_stream_rate'):
        out[key] = ap.median_of(rows, key)
    return out


def memory_compare(table):
    """Each fused arm against its matched stepped arm (same band, cohort and round)."""
    rows = {}
    for selection, cohort, index in ARM_ORDER:
        if selection != 'fused':
            continue
        fused = (table.get(label('fused', cohort, index)) or {}).get('memory', {}).get('worst')
        stepped = (table.get(label('stepped', cohort, index)) or {}).get('memory', {}).get('worst')
        name = '%s-%d' % (cohort, index)
        if not fused or not stepped:
            rows[name] = None
            continue
        rows[name] = dict(fused=fused, stepped=stepped,
                          fused_minus_stepped={k: round(fused[k] - stepped[k], 1) for k in fused if k != 'receipts'})
    return rows


def speed_compare(summary_band):
    groups = summary_band['groups']
    out = {}
    for cohort in progress.COHORTS:
        f, s = groups.get('fused-' + cohort), groups.get('stepped-' + cohort)
        if f and s:
            out[cohort] = {k: progress.ratio(f.get(k), s.get(k)) for k in (
                'ttft_s', 'prefill_tok_s', 's_per_chunk', 'decode_during_tok_s', 'decode_ratio', 'commands_per_chunk')}
    return out


# --- the grade -----------------------------------------------------------------------------------------

def kinds_in_breakdown(original):
    """chunk_breakdown with the chunk's command kinds kept, so the grade can read the protocol it ran."""
    def breakdown(chunk, workers):
        info = original(chunk, workers)
        kinds, segments = {}, {}
        for row in chunk['rows'].get('n', []):
            kinds[row['kind']] = kinds.get(row['kind'], 0) + 1
            if row['kind'] == 'segment':
                segments[row['segment']] = segments.get(row['segment'], 0) + 1
        info.update(kinds=kinds, segments=segments)
        return info
    return breakdown


progress.chunk_breakdown = kinds_in_breakdown(progress.chunk_breakdown)


def grade(arms, ingress, checks, logs, watches=None):
    """One verdict per arm and overall. Speed and memory are reported, never gated."""
    missing, failures = [], []
    watches = watches or {}
    required = required_labels()
    labels = [a.get('label') for a in arms]
    if len(labels) != len(set(labels)) or not set(x for x in labels if x) <= set(required):
        missing.append('undeclared-or-repeated-arm')
    for name in required:
        if name not in labels:
            missing.append('declared-arm-missing:' + name)
    for r in arms:
        name = r.get('label') or 'unlabelled'
        if r.get('complete') is not True:
            missing.extend(name + ':' + x for x in (r.get('reasons') or ['incomplete']))
        if name != 'unlabelled' and r.get('band') != ap.band_of_index(r.get('index')):
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
    tracked = ap.read_stamped(logs, is_tracked)
    table, valid = {}, {}
    for arm in known:
        name = arm['label']
        row = receipts['arms'].get(name)
        view = ap.log_view(arm, tracked)
        mine = [(s, t) for s, t in tracked if ap.in_span(s, arm)]
        signatures = {x for x in (signature_of(t) for _, t in mine) if x}
        error_text = json.dumps(arm.get('error') or {})
        signatures |= {name_ for name_, text in SIGNATURES if text in error_text}
        engaged = engagement(arm, row)
        table[name] = table_row(arm, row, view, mine)
        verdict, classification, reasons = arm_verdict(arm, ingress.get(name), watches.get(name), view, signatures,
                                                       (row or {}).get('reasons') or [], engaged)
        valid[name] = dict(verdict=verdict, classification=classification, reasons=reasons,
                           crash_free=arm.get('complete') is True and (watches.get(name) or {}).get('verdict') != 'DEAD',
                           disagree_lines=view['disagree_lines'], signatures=sorted(signatures),
                           engine_watch=(watches.get(name) or {}).get('verdict'), ingress=(ingress.get(name) or {}).get('verdict'),
                           engagement=engaged, selection_switch=arm.get('selection_switch'))
        if verdict == 'FAIL':
            failures.append('%s:%s' % (name, ','.join(reasons[:3])))
        elif verdict == 'INC':
            missing.extend('%s:%s' % (name, x) for x in reasons)
        if not engaged['engaged']:
            missing.append(name + ':not-engaged')
    # A declared arm without a record still gets its explicit row: stopped by the engine watch, or never entered.
    for name in required:
        if name in valid:
            continue
        watch = (watches.get(name) or {}).get('verdict')
        dead = watch == 'DEAD'
        valid[name] = dict(verdict='FAIL' if dead else 'INC', classification='ENGINE' if dead else None,
                           reasons=['engine-dead' if dead else 'no-arm-record'], crash_free=not dead and None,
                           disagree_lines=None, signatures=[], engine_watch=watch,
                           ingress=(ingress.get(name) or {}).get('verdict'), engagement=None, selection_switch=None)
        if dead:
            failures.append(name + ':engine-dead')
    # Any of the signatures anywhere in the followed log, also between the arm spans, makes the run invalid.
    between = [t for s, t in tracked if (ap.is_disagreement(t) or signature_of(t))
               and not any(ap.in_span(s, a) for a in known)]
    if between:
        failures.append('engine-signature-outside-every-arm-span')
    summary = {}
    for band in (1, 2):
        rows = [table[a['label']] for a in known if a.get('band') == band]
        groups = {}
        for selection in SELECTIONS:
            for cohort in progress.COHORTS:
                cell = [r for r in rows if (r['selection'], r['cohort']) == (selection, cohort)]
                if cell:
                    groups[selection + '-' + cohort] = group_summary(cell)
        summary['band%d' % band] = dict(groups=groups, device_memory=ap.memory_summary(rows),
                                        allocator_retry_warnings=sum(r['allocator_retry_warnings'] for r in rows),
                                        max_retry_request_mib=max((r['max_retry_request_mib'] or 0) for r in rows)
                                        if rows else None,
                                        arm_verdicts={r['label']: valid[r['label']]['verdict'] for r in rows})
        summary['band%d' % band]['fused_over_stepped'] = speed_compare(summary['band%d' % band])
    verdict = 'FAIL' if failures else 'INC' if missing else 'PASS'
    classes = sorted({v['classification'] for v in valid.values() if v['classification']})
    if verdict == 'FAIL' and not classes:
        classes = ['HARNESS']
    return {'schema': GRADE_SCHEMA, 'verdict': verdict, 'classification': classes or None, 'failures': sorted(set(failures)),
            'missing': sorted(set(missing)), 'required_arms': required, 'engine_lines_outside_every_arm': len(between),
            'table': table, 'validity': valid, 'progress': receipts, 'summary': summary,
            'memory_fused_vs_stepped': memory_compare(table), 'speed_is_gate': False, 'memory_is_gate': False,
            'target_is_gate': False, 'all_declared_arms_retained': True, 'retry_selected': False, 'picked': False,
            'release_readiness': RELEASE_READINESS,
            'basis': 'one engine, one fused selection per arm by typed control; band 1 matched ABBA at 65,445 tokens, band 2 '
                     'matched at 131,094 past the 97,152-row arena; per chunk, the critical rank of each native command; '
                     'live arms count chunks begun with all 8 decoders, solo arms chunks begun with none',
            'scope': 'Step A mechanism on the composed alpha engine, with speed and memory reported; gains from separate '
                     'experiments are never multiplied'}


# --- the report ----------------------------------------------------------------------------------------

def _cell(value, digits=1):
    if value is None:
        return '-'
    return ('%.' + str(digits) + 'f') % value if type(value) is float else str(value)


def report(result):
    """Three separate headed sections. Numbers come only from the graded record."""
    lines = ['# q18a-step-a report', '', 'Overall: **%s**%s' % (
        result['verdict'], ' (%s)' % ', '.join(result['classification']) if result['classification'] else ''), '']
    lines += ['## Mechanism', '', '| arm | verdict | class | crash-free | disagree | signatures | engaged | fused cmds | '
              'fused chunks | owned chunks | profile min-max | reasons |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for name in result['required_arms']:
        v = result['validity'].get(name) or {}
        e = v.get('engagement') or {}
        d = e.get('delta') or {}
        lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s-%s | %s |' % (
            name, v.get('verdict'), v.get('classification') or '-', v.get('crash_free'), _cell(v.get('disagree_lines')),
            ','.join(v.get('signatures') or []) or '0', e.get('engaged'), _cell(d.get('fused_commands')), _cell(d.get('chunks')),
            _cell(e.get('owned_chunks')), _cell(e.get('profile_min')), _cell(e.get('profile_max')),
            '; '.join(v.get('reasons') or []) or '-'))
    lines += ['']
    if result['failures']:
        lines += ['Failures: ' + '; '.join(result['failures']), '']
    if result['missing']:
        lines += ['Inconclusive items: ' + '; '.join(result['missing']), '']
    lines += ['## Performance', '', 'Reported, never gated.', '']
    for band in (1, 2):
        lines += ['### Band %d' % band, '', '| arm | TTFT s | prefill tok/s | s/chunk | decode during | decode only | ratio | '
                  'min stream ratio | p10 stream ratio | gap p99 ms | gap max ms | cmds/chunk |',
                  '|---|---|---|---|---|---|---|---|---|---|---|---|']
        for name in result['required_arms']:
            row = result['table'].get(name)
            if not row or row.get('band') != band:
                continue
            lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
                name, _cell(row.get('ttft_s')), _cell(row.get('prefill_tok_s'), 0), _cell(row.get('s_per_chunk'), 3),
                _cell(row.get('decode_during_tok_s')), _cell(row.get('decode_only_tok_s')), _cell(row.get('decode_ratio'), 3),
                _cell(row.get('min_stream_ratio'), 3), _cell(row.get('p10_stream_ratio'), 3), _cell(row.get('gap_p99_ms')),
                _cell(row.get('gap_max_ms')), _cell(row.get('commands_per_chunk'))))
        lines += ['', 'Fused over stepped, medians: ' + json.dumps(result['summary']['band%d' % band]['fused_over_stepped'],
                                                                   sort_keys=True), '']
    lines += ['### Device memory, fused against the matched stepped arm', '',
              '| pair | fused peak alloc MiB | stepped | fused peak reserved MiB | stepped | fused min free MiB | stepped | '
              'fused retries | stepped | fused OOMs | stepped |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for name, row in result['memory_fused_vs_stepped'].items():
        if not row:
            lines.append('| %s | not available | | | | | | | | | |' % name)
            continue
        f, s = row['fused'], row['stepped']
        lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            name, f['peak_allocated_mib'], s['peak_allocated_mib'], f['peak_reserved_mib'], s['peak_reserved_mib'],
            f['min_free_mib'], s['min_free_mib'], f['alloc_retries'], s['alloc_retries'], f['ooms'], s['ooms']))
    lines += ['', '## Release readiness', '', RELEASE_READINESS, '']
    return '\n'.join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='action', required=True)
    r = sub.add_parser('arm')
    r.add_argument('--url', required=True)
    r.add_argument('--model', required=True)
    r.add_argument('--run-id', required=True)
    r.add_argument('--epoch', type=int, required=True)
    r.add_argument('--mode', choices=SELECTIONS, required=True, help='this arm\'s QSA path; native progress is burst in every arm')
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
    g.add_argument('--report', help='write the three-section report here')
    g.add_argument('--out', required=True)
    a = p.parse_args(argv)
    if Path(a.out).exists():
        return 2
    if a.action == 'arm':
        return run_arm(a)
    arms = [base.load(x) or {'label': None, 'complete': False, 'reasons': ['arm-artifact-unreadable:' + Path(x).name]}
            for x in a.arm]
    result = grade(arms, ap.load_pairs(a.ingress), ap.load_pairs(a.check), a.log, ap.load_pairs(a.watch))
    with open(a.out, 'x') as handle:
        json.dump(result, handle, sort_keys=True)
    if a.report:
        with open(a.report, 'x') as handle:
            handle.write(report(result))
    print('STEP_A_GRADE ' + result['verdict'])
    return {'PASS': 0, 'FAIL': 1}.get(result['verdict'], 2)


if __name__ == '__main__':
    raise SystemExit(main())
