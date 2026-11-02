#!/usr/bin/env python3
"""Step B box arms and grade: layer group sizes matched per arm on one engine, fused selected in all of them.

The arm is held-f's (e3_native41_progress_box.arm), called unchanged with mode burst; the seams below add two typed controls and
change no other arm step. Control order: pause, native_progress {burst}, native_qsa_fused {true}, native_layer_group {size},
release_auto_arm, resume, then status only. The record keeps ``mode`` burst and adds ``selection`` (g1, g2, g4) and ``size``.

Two bands, as Step A's: band 1 (rounds 0, 1; 65,445 fresh then the append) sizes 1, 2, 4 in ABBA-style order, band 2 (round 2;
131,094 fresh, past the 97,152-row arena) sizes 1, 2. The label is part of the cache salt, so no arm can hit another arm's prefix.

The grade is e3_step_a_box.grade, called unchanged inside grade_seams, with its engagement rule extended:
  all arms    Step A's fused rule: fused chunks equal the auto-owned chunks, at least 12 fused commands per chunk, no profile query.
  size 1      every group counter delta 0, and 48 layer commands on every chunk: Step A's stream.
  size 2, 4   group chunks equal the auto-owned chunks, group commands equal the plan's commands (24 or 12 per chunk), 48 group
              layers per chunk, the largest group is the size, and the progress join shows that command count on every chunk.
A group arm that did not engage is INC. Speed and memory are reported, never gated. R3 (live decode below the matched size 1 arm by
more than 0.03 at size 2) is printed as a read for the lead, not applied.

Report: mechanism, performance and release readiness in separate headed sections. Receipts hold counts, hashes, identities and
host clock stamps, never prompts or auth.
"""
import argparse
import contextlib
import json
from pathlib import Path

import e3_step_a_box as sa
import step_a_torn_lines as torn_lines
from e3_native_qualification_driver import exception_details
from e3_native_runtime_client import NativeRuntimeClient

ap = sa.ap
progress = sa.progress
base = sa.base
GRADE_SCHEMA = 'step-b-grade/1'
SIZES = (1, 2, 4)
SELECTIONS = tuple('g%d' % size for size in SIZES)
BAND1 = (('g1', 'solo', 0), ('g2', 'solo', 0), ('g4', 'solo', 0), ('g4', 'live', 0), ('g2', 'live', 0), ('g1', 'live', 0),
         ('g1', 'live', 1), ('g2', 'live', 1), ('g4', 'live', 1), ('g4', 'solo', 1), ('g2', 'solo', 1), ('g1', 'solo', 1))
BAND2 = (('g2', 'live', 2), ('g1', 'live', 2), ('g1', 'solo', 2), ('g2', 'solo', 2))
ARM_ORDER = BAND1 + BAND2
INDEXES = (0, 1, 2)
LAYERS_PER_CHUNK = 48
QSA_LAYERS_PER_CHUNK = sa.QSA_LAYERS_PER_CHUNK
LAYER_COMMANDS_PER_CHUNK = {1: 48, 2: 24, 4: 12}          # 48 layers split 24/24, a QSA layer at every fourth, per stage
GROUP_COUNTERS = ('selections', 'chunks', 'plan_commands', 'group_commands', 'group_layers')
GROUP_MAX = 'group_layers_max'
CONTROL_SEQUENCE = ['status', 'pause', 'native_progress', 'native_qsa_fused', 'native_layer_group', 'release_auto_arm', 'resume']
R3_MARGIN = 0.03                                          # the lead's kill criterion at size 2, as a read
RELEASE_READINESS = sa.RELEASE_READINESS
# Memory read, the lead's rule (1 Nov, about 23:00 CET), a change from Step A's grade: per arm 0 OOMs, and for a group arm a min free
# no lower than its matched size 1 arm's by more than MEMORY_FREE_MARGIN_MIB. A torn or lost memory receipt in the arm (or, for a group
# arm, in its matched size 1 arm) makes the read INC: the lost line may have held the minimum. Torn memory lines are still scanned for
# OOM evidence (an OOM text, or an ooms= count above the rank's count when the arm began) and, through signature_of, contract errors.
# Reported next to Step A's memory rows; like them the read does not change the arm's verdict (memory_is_gate stays False). An OOM,
# from an intact or a torn receipt, does: it fails the arm as ENGINE (oom_fail).
MEMORY_FREE_MARGIN_MIB = 2.0
MEMORY_POINTS = ('open', 'released')
RANKS = 4
OOM_TEXTS = ('out of memory', 'OutOfMemory')
OOM_FIELD = __import__('re').compile(r'ooms=(\d+)')
RANK_FIELD = __import__('re').compile(r'rank=(\d+)')
# Slow rank per stage (descriptive, never graded), a change from Step A's grade. The stage's own ranks are Step A's report's TARGETS
# (vLLM's TP2 x PP2 layout, rank = stage * 2 + tp), declared here, not read from the log.
STAGE_RANKS = {0: (0, 1), 1: (2, 3)}


def label(selection, cohort, index):
    return '%s-%s-%d' % (selection, cohort, index)


def required_labels():
    return [label(*x) for x in ARM_ORDER]


def size_of(selection):
    return int(selection[1:])


# --- the arm --------------------------------------------------------------------------------------------

def group_view(status):
    block = status.get('native_layer_groups') if type(status) is dict else None
    if type(block) is not dict:
        return None
    return {k: block.get(k) for k in ('size', 'run_ahead') + GROUP_COUNTERS + (GROUP_MAX,)}


def group_delta(after, before):
    return {k: progress.delta(after, before, k) for k in GROUP_COUNTERS}


def release_groups(size):
    """held-f's release with both selections between native_progress and release_auto_arm; its checks, then the replies'."""
    def release(client, record, mode):
        reply = client.pause()
        record['release_pause'] = {'route': reply.get('route'), 'route_generation': reply.get('route_generation')}
        try:
            got = client.call('native_progress', mode=mode)
            record['mode_switch'] = {k: got.get(k) for k in ('mode', 'previous', 'installed', 'route',
                                                             'route_generation', 'engine_epoch')}
            got = client.call('native_qsa_fused', fused=True)
            record['selection_switch'] = {k: got.get(k) for k in ('fused', 'previous', 'route', 'route_generation',
                                                                  'engine_epoch')}
            got = client.call('native_layer_group', size=size)
            record['group_switch'] = {k: got.get(k) for k in ('size', 'previous', 'route', 'route_generation', 'engine_epoch')}
            got = client.call('release_auto_arm')
            record['release'] = {k: got.get(k) for k in ('released', 'engine_epoch')}
        finally:
            record['release_resume'] = client.resume().get('resumed')
        switch = record['mode_switch']
        if switch.get('mode') != mode or switch.get('installed') != 'burst' or switch.get('route') != 'full':
            raise ValueError('native-progress-mode-unproved')
        selection = record['selection_switch']
        if selection.get('fused') is not True or selection.get('route') != 'full':
            raise ValueError('native-qsa-fused-selection-unproved')
        group = record['group_switch']
        if group.get('size') != size or type(group.get('size')) is not int or group.get('route') != 'full':
            raise ValueError('native-layer-group-selection-unproved')
        if record['release'].get('released') is not True:
            raise ValueError('auto-arm-release-unproved')
    return release


def selection_reasons(record, size):
    """The selections' receipts beside held-f's. Engagement is the grade's, on the same numbers."""
    reasons = []
    history = record.get('control_history') or []
    if [x for i, x in enumerate(history) if x != 'status' or i == 0][:7] != CONTROL_SEQUENCE:
        reasons.append('control-sequence-not-pause-progress-fused-group-release-resume')
    for name in ('native_qsa_fused', 'native_layer_group'):
        if history.count(name) != 1:
            reasons.append(name + '-sent-other-than-once')
    views = {s.get('label'): s for s in record.get('progress') or []}
    fused_before = (views.get('before-fresh') or {}).get('native_qsa_fused')
    fused_after = (views.get('after-fresh') or {}).get('native_qsa_fused')
    before = (views.get('before-fresh') or {}).get('native_layer_groups')
    after = (views.get('after-fresh') or {}).get('native_layer_groups')
    if not fused_before or not fused_after:
        reasons.append('fused-status-missing')
    if not before or not after:
        reasons.append('group-status-missing')
    if reasons and ('fused-status-missing' in reasons or 'group-status-missing' in reasons):
        return reasons
    if fused_before.get('selected') is not True or fused_after.get('selected') is not True:
        reasons.append('fused-not-selected-in-this-arm')
    if before.get('size') != size or after.get('size') != size:
        reasons.append('group-size-not-this-arm')
    record['fused_delta'] = sa.fused_delta(fused_after, fused_before)
    record['group_delta'] = group_delta(after, before)
    record['group_max_after'] = after.get(GROUP_MAX)
    if record['fused_delta']['selections'] != 0 or record['group_delta']['selections'] != 0:
        reasons.append('selection-moved-during-the-fresh-prompt')
    return reasons


@contextlib.contextmanager
def arm_seams(size):
    """The names held-f's arm reads, for this one process's one arm; restored on exit."""
    selection = 'g%d' % size
    saved = {k: getattr(progress, k) for k in ('label', 'release', 'progress_reasons', 'progress_view', 'ACTIONS')}
    actions = base.ARM_ACTIONS
    original_reasons, original_view = saved['progress_reasons'], saved['progress_view']
    added = {'native_qsa_fused', 'native_layer_group'}

    def reasons(record, mode, cohort):
        history = record.get('control_history') or []
        view = dict(record, control_history=[x for x in history if x not in added])
        out = original_reasons(view, mode, cohort)
        if 'progress_delta' in view:
            record['progress_delta'] = view['progress_delta']
        return out + selection_reasons(record, size)

    def view(status):
        row = original_view(status)
        if row is None:
            return None
        fused, group = sa.fused_view(status), group_view(status)
        if fused is None:
            raise ValueError('native-qsa-fused-not-installed')
        if group is None:
            raise ValueError('native-layer-groups-not-installed')
        return dict(row, native_qsa_fused=fused, native_layer_groups=group)
    progress.label = lambda _mode, cohort, index: label(selection, cohort, index)
    progress.release = release_groups(size)
    progress.progress_reasons = reasons
    progress.progress_view = view
    progress.ACTIONS = saved['ACTIONS'] | added
    base.ARM_ACTIONS = dict(actions, auto=actions['auto'] | added)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(progress, name, value)
        base.ARM_ACTIONS = actions


def run_arm(a):
    band = ap.band_of_index(a.index)
    selection = 'g%d' % a.size
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
        with arm_seams(a.size):
            result = progress.arm(args, cfg, client, poll_client=poll, budget=budget)
    except Exception as exc:  # noqa: BLE001 - retained, never retried
        result = {'schema': progress.ARM_SCHEMA, 'kind': 'auto', 'mode': 'burst', 'cohort': a.cohort, 'index': a.index,
                  'label': label(selection, a.cohort, a.index), 'complete': False,
                  'reasons': ['arm-refused'], 'error': exception_details(exc)}
    result['selection'] = selection
    result['size'] = a.size
    result['band'] = band
    result['workload'] = ap.BAND_WORKLOAD[band]
    views = {s.get('label'): s for s in result.get('progress') or []}
    if 'after-append' in views:
        # The append is stock by design; the arm's own proof ran before it.
        result['fused_delta_append'] = sa.fused_delta(views['after-append'].get('native_qsa_fused'),
                                                      views['after-fresh'].get('native_qsa_fused'))
        result['group_delta_append'] = group_delta(views['after-append'].get('native_layer_groups'),
                                                   views['after-fresh'].get('native_layer_groups'))
        if any(v != 0 for v in result['fused_delta_append'].values()) or any(v != 0 for v in result['group_delta_append'].values()):
            result['reasons'] = list(result.get('reasons') or []) + ['append-moved-the-fused-or-group-counters']
            result['complete'] = False
    with open(a.out, 'x') as handle:
        json.dump(result, handle, sort_keys=True)
    print('STEP_B_ARM %s %s' % (result['label'], 'COMPLETE' if result.get('complete') else 'INCOMPLETE'))
    return 0 if result.get('complete') is True else 2


# --- engagement and comparisons -------------------------------------------------------------------------

def engagement(arm, row):
    selection = arm.get('selection')
    size = arm.get('size')
    delta = arm.get('fused_delta') or {}
    groups = arm.get('group_delta') or {}
    owned = (arm.get('progress_delta') or {}).get('chunks')
    kinds = sa.chunk_kinds(row)
    if (type(owned) is not int or owned < 1 or not delta or not groups or not kinds['chunks'] or size not in SIZES
            or selection != 'g%d' % size):
        return dict(selection=selection, size=size, engaged=False, reasons=['engagement-receipts-missing'], owned_chunks=owned,
                    delta=delta, groups=groups, layer_commands_min=None, layer_commands_max=None, profile_min=None, profile_max=None)
    reasons = []
    if delta.get('chunks') != owned:
        reasons.append('fused-chunks-delta-differs-from-the-auto-owned-chunks')
    if type(delta.get('fused_commands')) is not int or delta['fused_commands'] < QSA_LAYERS_PER_CHUNK * owned:
        reasons.append('fused-commands-below-twelve-per-chunk')
    if max(kinds['profile']) != 0:
        reasons.append('arm-sent-profile-queries')
    want = LAYER_COMMANDS_PER_CHUNK[size]
    if size == 1:
        if any(groups.get(k) != 0 for k in GROUP_COUNTERS):
            reasons.append('size-one-arm-moved-a-group-counter')
    else:
        if groups.get('chunks') != owned:
            reasons.append('group-chunks-delta-differs-from-the-auto-owned-chunks')
        if groups.get('plan_commands') != want * owned:
            reasons.append('plan-commands-differ-from-the-size')
        if groups.get('group_commands') != groups.get('plan_commands'):
            reasons.append('group-commands-differ-from-the-plan')
        if groups.get('group_layers') != LAYERS_PER_CHUNK * owned:
            reasons.append('group-layers-differ-from-the-model')
        if arm.get('group_max_after') != size:
            reasons.append('largest-group-is-not-the-size')
    if min(kinds['layer']) != want or max(kinds['layer']) != want:
        reasons.append('layer-commands-per-chunk-differ-from-the-size')
    return dict(selection=selection, size=size, engaged=not reasons, reasons=reasons, owned_chunks=owned, delta=delta, groups=groups,
                layer_commands_min=min(kinds['layer']), layer_commands_max=max(kinds['layer']),
                profile_min=min(kinds['profile']), profile_max=max(kinds['profile']))


def memory_compare(table):
    """Each group arm against its matched size 1 arm (same band, cohort and round)."""
    rows = {}
    for selection, cohort, index in ARM_ORDER:
        if selection == 'g1':
            continue
        mine = (table.get(label(selection, cohort, index)) or {}).get('memory', {}).get('worst')
        one = (table.get(label('g1', cohort, index)) or {}).get('memory', {}).get('worst')
        name = '%s-%s-%d' % (selection, cohort, index)
        if not mine or not one:
            rows[name] = None
            continue
        rows[name] = dict(group=mine, size_one=one, group_minus_size_one={k: round(mine[k] - one[k], 1) for k in mine if k != 'receipts'})
    return rows


def speed_compare(summary_band):
    groups = summary_band['groups']
    out = {}
    for selection in SELECTIONS[1:]:
        for cohort in progress.COHORTS:
            mine, one = groups.get(selection + '-' + cohort), groups.get('g1-' + cohort)
            if mine and one:
                out[selection + '-' + cohort] = {k: progress.ratio(mine.get(k), one.get(k)) for k in (
                    'ttft_s', 'prefill_tok_s', 's_per_chunk', 'decode_during_tok_s', 'decode_ratio', 'commands_per_chunk')}
    return out


def r3_read(table):
    """Live decode ratio at size 2 minus its matched size 1 arm, per round. A read for the lead, never a verdict input."""
    rows = {}
    for index in INDEXES:
        two = (table.get(label('g2', 'live', index)) or {}).get('decode_ratio')
        one = (table.get(label('g1', 'live', index)) or {}).get('decode_ratio')
        rows[str(index)] = None if two is None or one is None else round(two - one, 4)
    known = [value for value in rows.values() if value is not None]
    return dict(margin=R3_MARGIN, g2_minus_g1=rows, below_margin=[k for k, v in rows.items() if v is not None and v < -R3_MARGIN],
                is_gate=False, rounds=len(known))


@contextlib.contextmanager
def grade_seams():
    """The names e3_step_a_box.grade reads, for this one call; restored on exit, so Step A's own grade in this process is unchanged."""
    saved = {k: getattr(sa, k) for k in ('required_labels', 'engagement', 'SELECTIONS', 'ARM_ORDER', 'memory_compare', 'speed_compare',
                                         'memory_receipts', 'memory_worst')}
    breakdown = progress.chunk_breakdown
    progress.chunk_breakdown = slow_rank_breakdown(breakdown)
    sa.memory_receipts = memory_receipts
    sa.memory_worst = memory_worst
    sa.required_labels = required_labels
    sa.engagement = engagement
    sa.SELECTIONS = SELECTIONS
    sa.ARM_ORDER = ARM_ORDER
    sa.memory_compare = memory_compare
    sa.speed_compare = speed_compare
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(sa, name, value)
        progress.chunk_breakdown = breakdown


def grade(arms, ingress, checks, logs, watches=None):
    with grade_seams():
        result = sa.grade(arms, ingress, checks, logs, watches)
    for band in result['summary'].values():
        band['over_size_one'] = band.pop('fused_over_stepped')
    result['memory_over_size_one'] = result.pop('memory_fused_vs_stepped')
    for row in result['table'].values():
        row['size'] = size_of(row['selection'])
    torn = torn_lines.read_torn(logs)['memory']
    reads = memory_read(result['table'], torn, arms)
    for name, cell in reads.items():
        if name in result['validity']:
            result['validity'][name]['memory_read'] = cell['verdict']
    oom_fail(result, reads)
    result.update(memory_read=reads, memory_read_is_gate=False, memory_free_margin_mib=MEMORY_FREE_MARGIN_MIB,
                  slow_rank=slow_rank(result['progress']))
    result.update(schema=GRADE_SCHEMA, r3_read=r3_read(result['table']),
                  basis='one engine, fused selected in every arm, one group size per arm by typed control; band 1 ABBA-style at '
                        '65,445 tokens, band 2 matched at 131,094 past the 97,152-row arena; per chunk, the critical rank of each '
                        'native command; live arms count chunks begun with all 8 decoders, solo arms chunks begun with none',
                  scope='Step B mechanism on the composed Step A engine, with speed and memory reported; gains from separate '
                        'experiments are never multiplied')
    return result


# --- memory read ---------------------------------------------------------------------------------------

def memory_receipts(rows):
    """Step A's receipts per rank, plus the receipts a rank is missing against the transactions any rank reported and the rank's
    OOM count when the arm began (for the torn-line scan)."""
    out = _step_a_memory_receipts(rows)
    seen, first_ooms = set(), {}
    for _, text in rows:
        m = sa.MEMORY_LINE.search(text)
        if m:
            seen.add((int(m.group(2)), int(m.group(3)), m.group(1)))
            first_ooms.setdefault(int(m.group(2)), int(m.group(10)))
    txns = {txn for _, txn, _ in seen}
    for rank in range(RANKS):
        cell = out.setdefault(str(rank), dict(receipts=0))
        cell['missing_receipts'] = sum(1 for txn in txns for point in MEMORY_POINTS if (rank, txn, point) not in seen)
        cell['first_ooms'] = first_ooms.get(rank)
    return out


_step_a_memory_receipts = sa.memory_receipts


def memory_worst(ranks):
    """Step A's worst row over the ranks that reported; ranks that reported nothing are counted by memory_read instead."""
    return sa_memory_worst({k: v for k, v in (ranks or {}).items() if v.get('receipts')})


sa_memory_worst = sa.memory_worst


def torn_memory_evidence(text, ranks):
    """OOM evidence a torn memory line still shows: an OOM text, or an ooms= above its rank's count when the arm began."""
    if any(word in text for word in OOM_TEXTS):
        return 'oom-text-in-a-torn-memory-line'
    whole = torn_lines.INNER_HEADER.sub('', text)
    count, rank = OOM_FIELD.search(whole), RANK_FIELD.search(whole)
    if count is None:
        return None
    firsts = [r.get('first_ooms') for k, r in ranks.items() if r.get('first_ooms') is not None
              and (rank is None or k == rank.group(1))]
    if not firsts or int(count.group(1)) > min(firsts):
        return 'oom-count-in-a-torn-memory-line'
    return None


def arm_memory(name, table, torn, arms):
    """(established, reasons, oom evidence, worst) of one arm's memory receipts."""
    row = (table.get(name) or {}).get('memory') or {}
    ranks, worst = row.get('ranks') or {}, row.get('worst')
    arm = next((a for a in arms if a.get('label') == name), {})
    mine = [text for stamp, text in torn if torn_lines.in_span(stamp, arm)]
    reasons, evidence = [], []
    if mine:
        reasons.append('torn-memory-lines:%d' % len(mine))
    missing = sum(r.get('missing_receipts') or 0 for r in ranks.values())
    silent = [k for k, r in sorted(ranks.items()) if not r.get('receipts')]
    if missing:
        reasons.append('memory-receipts-missing:%d' % missing)
    if silent:
        reasons.append('ranks-without-memory-receipts:' + ','.join(silent))
    if not worst:
        reasons.append('no-memory-receipts')
    for text in mine:
        found = torn_memory_evidence(text, ranks)
        if found:
            evidence.append(found)
    if worst and worst.get('ooms'):
        evidence.append('ooms:%d' % worst['ooms'])
    return not reasons, reasons, evidence, worst


def memory_read(table, torn, arms):
    """The lead's memory read per arm: MET, NOT_MET or INC. Never a verdict input."""
    rows = {}
    for selection, cohort, index in ARM_ORDER:
        name = label(selection, cohort, index)
        established, reasons, evidence, worst = arm_memory(name, table, torn, arms)
        cell = dict(established=established, reasons=list(reasons), oom_evidence=evidence, min_free_breach=False,
                    min_free_mib=(worst or {}).get('min_free_mib'), against=None, min_free_below_size_one_mib=None)
        if selection != 'g1':
            one = label('g1', cohort, index)
            one_ok, one_reasons, _, one_worst = arm_memory(one, table, torn, arms)
            cell['against'] = one
            if not one_ok:
                cell['reasons'] += ['size-one-arm-' + x for x in one_reasons]
            if established and one_ok and worst and one_worst:
                below = round(one_worst['min_free_mib'] - worst['min_free_mib'], 1)
                cell['min_free_below_size_one_mib'] = below
                cell['min_free_breach'] = below > MEMORY_FREE_MARGIN_MIB
        if evidence or cell['min_free_breach']:
            cell['verdict'] = 'NOT_MET'
        elif cell['reasons']:
            cell['verdict'] = 'INC'
        else:
            cell['verdict'] = 'MET'
        rows[name] = cell
    return rows


def oom_fail(result, reads):
    """A real OOM in an arm, read from an intact receipt (ooms rose within the arm) or a torn one, fails that arm as ENGINE, as an
    engine signature does. Step A's grade only reports the counter; the lead asked for this (1 Nov about 23:05 CET)."""
    for name, cell in reads.items():
        row = result['validity'].get(name)
        if not cell['oom_evidence'] or row is None:
            continue
        row['verdict'], row['classification'] = 'FAIL', 'ENGINE'
        row['reasons'] = ['oom:' + ','.join(cell['oom_evidence'])] + [x for x in row['reasons'] if not x.startswith('oom:')]
        row['signatures'] = sorted(set(row['signatures']) | {'oom'})
        result['failures'] = sorted(set(result['failures']) | {'%s:oom' % name})
    if result['failures']:
        result['verdict'] = 'FAIL'
        classes = {v['classification'] for v in result['validity'].values() if v['classification']}
        result['classification'] = sorted(classes) or ['HARNESS']


# --- slow rank per stage --------------------------------------------------------------------------------

def _merged_s(spans):
    total, end = 0, None
    for lo, hi in sorted(spans):
        if end is None or lo > end:
            total += hi - lo
            end = hi
        elif hi > end:
            total += hi - end
            end = hi
    return total / 1e9


def slow_rank_breakdown(original):
    """chunk_breakdown with each stage's own ranks timed separately: per rank, the merged span from each of the stage's commands'
    submit to that rank's reply, and the rank's handler time (enter to exit). Both ranks sit on the same engine row (t_recv0..3) and
    their worker rows are matched by (seq, rank), so chunk identity across ranks is exact."""
    def breakdown(chunk, workers):
        info = original(chunk, workers)
        stages = {}
        for stage, ranks in STAGE_RANKS.items():
            own, handler = {r: [] for r in ranks}, {r: 0.0 for r in ranks}
            for r in chunk['rows'].get('n', []):
                if r['stage'] != stage or type(r['t_submit']) is not int:
                    continue
                for rank in ranks:
                    recv = r['t_recv%d' % rank]
                    if type(recv) is int:
                        own[rank].append((r['t_submit'], recv))
                    w = workers.get((r['seq'], rank))
                    if w and type(w['t_enter']) is int and type(w['t_exit']) is int:
                        handler[rank] += (w['t_exit'] - w['t_enter']) / 1e9
            if all(own[r] for r in ranks):
                stages[str(stage)] = dict(own_s={str(r): _merged_s(own[r]) for r in ranks},
                                          handler_s={str(r): handler[r] for r in ranks})
        info['stage_ranks'] = stages
        return info
    return breakdown


def slow_rank(receipts):
    """Per arm and per context band: the median over counted chunks of the slower own rank's stage time beside the pooled median
    over both ranks, and how often each rank was the slower one. Descriptive, never graded."""
    median = progress.median

    def summary(chunks):
        out = {}
        for stage in map(str, STAGE_RANKS):
            cells = [c['stage_ranks'][stage] for c in chunks if stage in (c.get('stage_ranks') or {})]
            if not cells:
                continue
            row = dict(chunks=len(cells))
            for metric in ('own_s', 'handler_s'):
                row[metric] = dict(slow_rank_median=median([max(x[metric].values()) for x in cells]),
                                   pooled_median=median([v for x in cells for v in x[metric].values()]))
            slower = {}
            for x in cells:
                rank = max(x['own_s'], key=x['own_s'].get)
                slower[rank] = slower.get(rank, 0) + 1
            row['slower_rank_chunks'] = dict(sorted(slower.items()))
            out[stage] = row
        return out
    arms = {}
    for name, row in receipts['arms'].items():
        counted = [c for c in row.get('chunks') or [] if c['label'] == row.get('counted_label') and not c.get('dropped_out')]
        arms[name] = dict(all=summary(counted),
                          bands={band: summary([c for c in counted if c['band'] == band]) for band in sorted({c['band'] for c in counted})})
    return dict(arms=arms, is_gate=False, chunk_identity='exact: both ranks of a stage on one engine row (t_recv0..3), workers by (seq, rank)',
                stage_ranks={str(k): list(v) for k, v in STAGE_RANKS.items()},
                stage_ranks_basis='declared: vLLM TP2 x PP2 layout, as tools/step_a_report.py TARGETS; not read from the log',
                own_s='per rank, merged span from submit of each of the stage\'s commands to that rank\'s reply',
                handler_s='per rank, summed native handler time (enter to exit) over the stage\'s commands')


# --- the report ----------------------------------------------------------------------------------------

def report(result):
    """Three separate headed sections. Numbers come only from the graded record."""
    cell = sa._cell
    lines = ['# q18a-step-b report', '', 'Overall: **%s**%s' % (
        result['verdict'], ' (%s)' % ', '.join(result['classification']) if result['classification'] else ''), '']
    lines += ['## Mechanism', '', '| arm | verdict | class | crash-free | disagree | signatures | engaged | owned chunks | fused cmds | '
              'group chunks | plan cmds | group cmds | group layers | layer cmds per chunk | reasons |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for name in result['required_arms']:
        v = result['validity'].get(name) or {}
        e = v.get('engagement') or {}
        d, g = e.get('delta') or {}, e.get('groups') or {}
        lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s-%s | %s |' % (
            name, v.get('verdict'), v.get('classification') or '-', v.get('crash_free'), cell(v.get('disagree_lines')),
            ','.join(v.get('signatures') or []) or '0', e.get('engaged'), cell(e.get('owned_chunks')), cell(d.get('fused_commands')),
            cell(g.get('chunks')), cell(g.get('plan_commands')), cell(g.get('group_commands')), cell(g.get('group_layers')),
            cell(e.get('layer_commands_min')), cell(e.get('layer_commands_max')), '; '.join(v.get('reasons') or []) or '-'))
    lines += ['']
    if result['failures']:
        lines += ['Failures: ' + '; '.join(result['failures']), '']
    if result['missing']:
        lines += ['Inconclusive items: ' + '; '.join(result['missing']), '']
    tears = result['progress']['tears']
    lines += ['### Torn lines and lost records', '',
              'Limits declared before the run, per arm: at most %d lost record and %d dropped chunks. Over the run: %d torn progress '
              'and %d torn memory lines, %d lost flushes, %d outside every arm.' % (
                  tears['lost_record_limit'], tears['dropped_chunk_limit'], tears['torn_progress_lines'],
                  tears['torn_memory_lines'], tears['lost_flushes'], len(tears['lost_flushes_outside_every_arm'])), '',
              '| arm | torn progress | torn memory | lost flushes | lost records | dropped chunks | within limits |',
              '|---|---|---|---|---|---|---|']
    for name in result['required_arms']:
        t = (result['validity'].get(name) or {}).get('tears')
        if t:
            lines.append('| %s | %s | %s | %s | %s | %s | %s |' % (
                name, t['torn_progress_lines'], t['torn_memory_lines'], len(t['lost_flushes']), t['lost_records'],
                ','.join(str(x) for x in t['dropped_chunk_starts']) or '0', t['within_limits']))
    lines += ['', '## Performance', '', 'Reported, never gated.', '']
    read = result['r3_read']
    lines += ['R3 read, live decode ratio at size 2 minus its matched size 1 arm, per round: %s. Margin %s; rounds below it: %s. '
              'This is a read for the lead, not a verdict input.' % (json.dumps(read['g2_minus_g1'], sort_keys=True), read['margin'],
                                                                     read['below_margin'] or 'none'), '']
    for band in (1, 2):
        lines += ['### Band %d' % band, '', '| arm | TTFT s | prefill tok/s | s/chunk | cmds/chunk | decode during | decode only | ratio | '
                  'min stream ratio | p10 stream ratio | min stream tok/s | p10 stream tok/s | gap p99 ms | gap max ms |',
                  '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for name in result['required_arms']:
            row = result['table'].get(name)
            if not row or row.get('band') != band:
                continue
            lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
                name, cell(row.get('ttft_s')), cell(row.get('prefill_tok_s'), 0), cell(row.get('s_per_chunk'), 3),
                cell(row.get('commands_per_chunk')), cell(row.get('decode_during_tok_s')), cell(row.get('decode_only_tok_s')),
                cell(row.get('decode_ratio'), 3), cell(row.get('min_stream_ratio'), 3), cell(row.get('p10_stream_ratio'), 3),
                cell(row.get('min_stream_rate')), cell(row.get('p10_stream_rate')), cell(row.get('gap_p99_ms')), cell(row.get('gap_max_ms'))))
        lines += ['', 'Over the size 1 arm, medians: ' + json.dumps(result['summary']['band%d' % band]['over_size_one'], sort_keys=True), '']
    lines += ['### Device memory, each group arm against its matched size 1 arm', '',
              '| arm | peak alloc MiB | size 1 | peak reserved MiB | size 1 | min free MiB | size 1 | retries | size 1 | OOMs | size 1 |',
              '|---|---|---|---|---|---|---|---|---|---|---|']
    for name, row in result['memory_over_size_one'].items():
        if not row:
            lines.append('| %s | not available | | | | | | | | | |' % name)
            continue
        g, s = row['group'], row['size_one']
        lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            name, g['peak_allocated_mib'], s['peak_allocated_mib'], g['peak_reserved_mib'], s['peak_reserved_mib'],
            g['min_free_mib'], s['min_free_mib'], g['alloc_retries'], s['alloc_retries'], g['ooms'], s['ooms']))
    lines += ['', '### Memory read, per arm', '',
              'Declared before the run (a change from Step A\'s grade): 0 OOMs, and a group arm\'s min free no lower than its matched '
              'size 1 arm\'s by more than %s MiB. A torn or lost memory receipt makes the read INC: the lost line may have held the '
              'minimum. The read is not a verdict input; a NOT_MET or INC read blocks adopting that group size. An OOM, from an '
              'intact or a torn receipt, fails the arm as ENGINE.' % result['memory_free_margin_mib'], '',
              '| arm | read | min free MiB | against | below size 1 MiB | OOM evidence | reasons |', '|---|---|---|---|---|---|---|']
    for name, read in result['memory_read'].items():
        lines.append('| %s | %s | %s | %s | %s | %s | %s |' % (
            name, read['verdict'], read['min_free_mib'], read['against'] or '-', read['min_free_below_size_one_mib'],
            ','.join(read['oom_evidence']) or '0', '; '.join(read['reasons']) or '-'))
    slow = result['slow_rank']
    lines += ['', '### Slow rank per stage', '',
              'Descriptive, never graded (a change from Step A\'s report). Per counted chunk, the slower of the stage\'s own ranks (%s) '
              'beside the median pooled over both. Chunk identity across ranks: %s.' % (slow['stage_ranks_basis'], slow['chunk_identity']), '',
              '| arm | band | stage | chunks | own s slow | own s pooled | handler s slow | handler s pooled | slower rank chunks |',
              '|---|---|---|---|---|---|---|---|---|']
    for name in result['required_arms']:
        row = slow['arms'].get(name)
        if not row:
            continue
        for band, cells in [('all', row['all'])] + sorted(row['bands'].items()):
            for stage, s in sorted(cells.items()):
                lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
                    name, band, stage, s['chunks'], cell(s['own_s']['slow_rank_median'], 4), cell(s['own_s']['pooled_median'], 4),
                    cell(s['handler_s']['slow_rank_median'], 4), cell(s['handler_s']['pooled_median'], 4),
                    json.dumps(s['slower_rank_chunks'], sort_keys=True)))
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
    r.add_argument('--size', type=int, choices=SIZES, required=True, help='this arm\'s layer group size; fused and burst in every arm')
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
    print('STEP_B_GRADE ' + result['verdict'])
    return {'PASS': 0, 'FAIL': 1}.get(result['verdict'], 2)


if __name__ == '__main__':
    raise SystemExit(main())
