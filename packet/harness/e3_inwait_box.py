#!/usr/bin/env python3
"""q18a-inwait box arms and grade: native handoffs in the stock wait, on and off, over attn-meta and L1 + L2, every prefill arm C1 defer at group size 2. Discovery mode.

e3_compose2_box by name over inwait_arms, using its parents the same way (c1, c1b, am, sb); nothing of theirs is edited. The arm is compose-2's with one more field on native_stage_local, ``inwait``.
Control order: pause, native_progress {burst}, native_qsa_fused {true}, native_layer_group {size}, native_stage_local {local, overlap, defer_bind, lazy_recv, decode_merge, inwait},
qualification_worker attn_meta_select {on}, release_auto_arm, resume, then status; after the arm's own record one end read. ACTIONS below is the list the router smoke holds.

The grade: compose-2's, plus inwait_check: the arm acknowledged its selection; a live on arm applied handoffs in the wait; an off arm applied none and sent no new direct reply. Speed is never a gate.
Numerics are not covered: this queue runs no fixed-schedule check.

Reads, never gates (compose_read): per arm and per cell live and solo prefill tok/s, decode own, against this run's decode-am1 and against the fixed reference (inwait_arms.FIXED_REFERENCE_TOK_S); wait_sends and
wait_capped; the in-wait counters; refused sends per worker method. The per-chunk split (stage 0 non-layer apply, stage 1 open + release hold, PP_RECEIVE gate, FIFO) is read on the hub from the fetched logs.
"""
import argparse
import json
from pathlib import Path

import inwait_arms as T
import e3_alpha_next_c1_box as c1
import e3_alpha_next_c1b_box as c1b
import e3_attn_meta_box as am
import e3_stage1_step_box as sb
from types import SimpleNamespace

an, progress, ap, base = c1.an, c1.progress, c1.ap, c1.base
fread = sb.fread
rule = c1b.rule
_bound = c1._bound
STAGE_CONTROL = c1.STAGE_CONTROL
GRADE_SCHEMA = 'inwait-grade/1'
READ_SCHEMA = 'inwait-read/1'
ARM_SCHEMA = 'inwait-arm/1'
APPEND_REASON = 'append-moved-the-fused-group-or-stage-counters'
ACTIONS = (('pause', ()), ('native_progress', ('mode',)), ('native_qsa_fused', ('fused',)), ('native_layer_group', ('size',)),
           ('native_stage_local', ('local', 'overlap', 'defer_bind', 'lazy_recv', 'decode_merge', 'inwait', 'early_successor')), ('qualification_worker', ('worker_action', 'payload')),
           ('release_auto_arm', ()), ('resume', ()), ('status', ()))
AXIS = ("alpha-next's C0 on/off names and c1's C1-on/off names in this record (over_c0_off, c0_decode_delta, vs_c1_off, memory_c0_on_vs_off, min_free_below_c0_off_mib, C1-on/C1-off table headings) name "
        'cf (the fast CoW copy) against its matched ps1, and cap (the reserved cap) against ps1 too; every arm is C1 defer, C0 local, attn-meta, L1 + L2, in-wait, early successor, fast slot mappings and PREPARE in slices')
_ROLE = 'measurement (C1 defer, attn-meta and L1 + L2, in-wait on, early successor, fast slot mappings'
_ROLE = _ROLE + ', PREPARE in slices'
OBSERVE_ROLE = {'ps1': _ROLE + ')', 'cf': _ROLE + ', fast CoW copy)', 'sm': _ROLE + ', fast CoW copy; the exact-path smoke)', 'cap': _ROLE + ', reserved cap)'}
NOT_COVERED = ('numerics: this queue runs no fixed-schedule check; these arms are speed reads, batch 8 against 2 x 4 changes batch composition so OUTSIDE on merged arms is expected and is not a defect, '
               'and the fixed-schedule check owed for C1 defer stays owed')
BASIS = ('one engine, FLASH_ATTN_META_GDN_BATCH=1 FLASH_S5_LAZY_RECV=1 FLASH_S5_DECODE_MERGE=1 FLASH_S5_INWAIT=1 for the whole run, every prefill arm C1 defer with fused, group size 2, attn-meta and L1 + L2 '
         'selected and in-wait on or off by typed control; the exact-path smoke first, then one decode-only baseline; band 1 at 65,445 tokens live over two counterbalanced rounds and solo once; band 2 at 131,094 tokens, one pair on cf; one cap arm last')
SCOPE = ("in-wait handoffs on compose-2's engine; selection, counters, crash, OOM and one in flight are gated; speed, decode, per-step thread time and memory are reads; "
         'not a release assessment and not a numerics result')
IW_COUNTERS = ('direct_opens', 'direct_releases', 'polls_paced', 'refused_releases', 'refused_in_wait', 'early_begins')


def table_view(sel=None):
    """This table as alpha-next's code reads alpha_next_arms: the candidate axis is cf against ps1 (c0 'off' = ps1; c0 'on' = cf). For one arm the label and selection
    carry ``sel`` whatever value alpha-next passes; the record keeps c0 'on'."""
    names = {k: getattr(T, k) for k in c1._TABLE_NAMES}
    sel_of = dict({s: s for s in T.SELECTIONS}, off=T.OFF, on='cf')

    def axis(value):
        return 'off' if value == T.OFF else 'on'

    def arms(part=None):
        return [dict(r, c0=axis(r['c1'])) for r in T.arms(part)]

    def parse_selection(token):
        parsed = T.parse_selection(token)
        return None if parsed is None else (axis(parsed[0]), parsed[1])
    names.update(arms=arms, by_label=lambda: {r['label']: r for r in arms()}, C0={'on': True, 'off': True}, parse_selection=parse_selection,
                 selection=lambda value, size: T.selection(sel_of[value], size),
                 label=lambda value, size, cohort, index: T.label(sel_of[value], size, cohort, index))
    if sel is not None:
        names.update(selection=lambda _c0, size: T.selection(sel, size),
                     label=lambda _c0, size, cohort, index: T.label(sel, size, cohort, index), C0={'on': True})
    return SimpleNamespace(**names)


def sb_bindings():
    """sb's names over this table, for one call."""
    return dict(T=T, table_view=table_view, OBSERVE_ROLE=OBSERVE_ROLE, AXIS=AXIS, NOT_COVERED=NOT_COVERED, GRADE_SCHEMA=GRADE_SCHEMA, READ_SCHEMA=READ_SCHEMA)


# --- the arm --------------------------------------------------------------------------------------------

RANK_LEVERS = ('slots_fast', 'prepare_slices', 'score_warm', 'cow_fast', 'reserved_cap')


def levers_into(client, attn, wanted, record):
    """Inside the arm's pause: attn_meta_select with the rank levers, every rank proved as asked; attn-meta's own select follows and keeps them."""
    rows = am.Worker(client.control, client.run_id, client.epoch).call(am.SELECT, dict(wanted, on=attn))
    selected = am._keyed(rows)
    for name in RANK_LEVERS:
        record[name] = dict(requested=wanted[name], select=selected)
        if any(((r.get('after') or {}).get(name) or {}).get('selected') is not wanted[name] for r in rows.values()):
            raise ValueError(name.replace('_', '-') + '-selection-unproved')


class _InwaitSelecting:
    """The arm's client with ``inwait`` on the stage command; the reply's value goes to the record, and the grade reads it."""

    def __init__(self, client, inwait, record, next_levers=False, attn=True, raw=None, slices=False, warm=False, cow_fast=False, reserved_cap=False):
        self._client, self._inwait, self._record, self._next, self._attn = client, inwait, record, next_levers, attn
        self._wanted = dict(slots_fast=next_levers, prepare_slices=slices, score_warm=warm, cow_fast=cow_fast, reserved_cap=reserved_cap)
        self._raw, self._slots_sent = client if raw is None else raw, False

    def __getattr__(self, name):
        return getattr(self._client, name)

    def call(self, action, **fields):
        if action == 'release_auto_arm' and not self._slots_sent:
            # The rank levers, inside the arm's pause, before attn-meta's own select.
            self._slots_sent = True
            levers_into(self._raw, self._attn, self._wanted, self._record)
        if action != STAGE_CONTROL:
            return self._client.call(action, **fields)
        reply = self._client.call(action, **dict(fields, inwait=self._inwait, early_successor=self._next))
        self._record['inwait'] = dict(requested=self._inwait, acknowledged=reply.get('inwait') if type(reply) is dict else None,
                                      early_requested=self._next, early_acknowledged=reply.get('early_successor') if type(reply) is dict else None)
        return reply


def release_compose(row):
    """sb's release (c1's C1 defer with both step fields on the stage command) with ``inwait`` on that command and attn_meta_select just before release_auto_arm; the record must show both ran."""
    inner = sb.release_step(row['lazy_recv'], row['decode_merge'])

    def factory(size, local):
        run_inner = inner(size, local)

        def run(client, record, mode):
            run_inner(_InwaitSelecting(am._Selecting(client, row['attn'], record), row['inwait'], record, row['next_levers'], row['attn'], client, row['prepare_slices'], row['score_warm'], row['cow_fast'], row['reserved_cap']), record, mode)
            if (record.get('attn_meta') or {}).get('select') is None:
                raise ValueError('attn-meta-select-not-sent')
            if record.get('inwait') is None:
                raise ValueError('inwait-field-not-sent')
            if record.get('slots_fast') is None:
                raise ValueError('slots-fast-not-sent')
        return run
    return factory


def stage_view(status):
    """sb's view plus the inwait block, flat under step_iw_; the per-kind rows stay dicts, read from the views."""
    row = sb.stage_view(status)
    if row is None:
        return None
    block = status[STAGE_CONTROL]['overlap'].get('inwait')
    block = block if type(block) is dict else {}
    counters = block.get('counters') if type(block.get('counters')) is dict else {}
    refused = block.get('refused') if type(block.get('refused')) is dict else {}
    applied, sent = counters.get('wait_applied') or {}, counters.get('wait_sent') or {}
    return dict(row, step_iw_installed=block.get('installed'), step_iw_selected=block.get('selected'), step_iw_refused=dict(refused), step_iw_wait_applied=dict(applied), step_iw_wait_sent=dict(sent),
                step_iw_refused_receipts=refused.get('flash_s5_receipt', 0), step_iw_refused_total=sum(refused.values()), step_iw_applied=sum(applied.values()), step_iw_sent=sum(sent.values()),
                **{'step_iw_' + k: counters.get(k) for k in IW_COUNTERS})


IW_DELTA = ('refused_receipts', 'refused_total', 'applied', 'sent') + IW_COUNTERS


def stage_delta(after, before):
    out = sb.stage_delta(after, before)
    out.update({'step_iw_' + k: progress.delta(after, before, 'step_iw_' + k) for k in IW_DELTA})
    return out


def stage_reasons(row):
    inner = sb.stage_reasons(row['lazy_recv'], row['decode_merge'])

    def reasons(record, local):
        out = inner(record, local)
        views = {s.get('label'): s for s in record.get('progress') or []}
        before = (views.get('before-fresh') or {}).get(STAGE_CONTROL)
        after = (views.get('after-fresh') or {}).get(STAGE_CONTROL)
        if before and after and any(v.get('step_iw_selected') is not row['inwait'] for v in (before, after)):
            out.append('stage-inwait-selection-not-this-arm')
        return out
    return reasons


def _arm_print(*args, **kwargs):
    text = ' '.join(str(x) for x in args)
    print(text.replace('ALPHA_NEXT_ARM ', 'INWAIT_ARM ', 1), **kwargs)


def arm_names(sel, row):
    return dict(T=table_view(sel), release_alpha_next=release_compose(row), stage_reasons=stage_reasons(row),
                stage_view=stage_view, stage_delta=stage_delta, print=_arm_print)


def append_rule(record):
    """Exempt the step_* keys from alpha-next's append rule, in place (module text). True when the reason went away; the delta stays in the record either way."""
    if APPEND_REASON not in (record.get('reasons') or ()):
        return False
    stage = record.get('stage_delta_append') or {}
    other = [record.get('fused_delta_append') or {}, record.get('group_delta_append') or {}, {k: v for k, v in stage.items() if not k.startswith('step_')}]
    if any(v != 0 for block in other for v in block.values()):
        return False
    record['reasons'] = [r for r in record['reasons'] if r != APPEND_REASON]
    record['append_step_exempt'] = {k: v for k, v in stage.items() if k.startswith('step_') and v != 0}
    record['complete'] = not record['reasons'] and not record.get('error')
    return True


def lever_delta(select, end, name, field):
    """Per rank: one rank lever's counter in the arm, end read minus select; None when either is missing."""
    if not select or not end:
        return None
    out = {}
    for rank in sorted(set(select) | set(end)):
        a = (((select.get(rank) or {}).get('after') or {}).get(name) or {}).get(field)
        b = (((end.get(rank) or {}).get('counters') or {}).get(name) or {}).get(field)
        out[rank] = b - a if type(a) is int and type(b) is int else None
    return out


def slots_calls(select, end):
    """Per rank: fast slot mapping calls in the arm, end read minus select; None when either is missing."""
    return lever_delta(select, end, 'slots_fast', 'calls')


def rank_lever_counters(record, row, end):
    """The build-3 levers' counters on the arm record, per rank; the grade and the read use them."""
    for name, wanted, fields in (('prepare_slices', row['prepare_slices'], ('sliced', 'yields', 'peek_missing')),
                                 ('score_warm', row['score_warm'], ('allocs', 'flushes', 'rewarms', 'errors')),
                                 ('cow_fast', row['cow_fast'], ('calls', 'pairs', 'fallbacks', 'rows', 'row_errors', 'verify_pairs', 'verify_unequal', 'pages_errors')),
                                 ('reserved_cap', row['reserved_cap'], ('flushes',))):
        block = record.setdefault(name, dict(requested=wanted, select=None))
        block['counters'] = {field: lever_delta(block.get('select'), end, name, field) for field in fields}


def run_arm(a):
    """alpha-next's arm, C0 local ('on'), C1 defer, with this arm's selection; the append rule; then attn-meta's end read. ``a.out`` = the arm's record + attn_meta."""
    name = T.label(a.c1, a.size, a.cohort, a.index)
    row = T.by_label().get(name)
    if row is None:
        record, rc = dict(schema=ARM_SCHEMA, label=name, complete=False, reasons=['arm-not-in-the-declared-table']), 2
    else:
        inner = a.out + '.inner.json'
        args = argparse.Namespace(**dict(vars(a), stage_local='on', out=inner))
        with _bound(c1, dict(T=T)), _bound(an, arm_names(a.c1, row)):
            rc = an.run_arm(args)
        record = base.load(inner) or dict(label=name, complete=False, reasons=['arm-record-unreadable'])
        if append_rule(record) and rc == 2 and record['complete']:
            rc = 0
        block = record.setdefault('attn_meta', dict(requested=row['attn'], select=None))
        block['end'] = am.end_read(a)
        block['deltas'] = am.deltas(block.get('select'), (block['end'] or {}).get('read'))
        record.setdefault('slots_fast', dict(requested=row['next_levers'], select=None))['calls'] = slots_calls((record.get('slots_fast') or {}).get('select'), (block['end'] or {}).get('read'))
        rank_lever_counters(record, row, (block['end'] or {}).get('read'))
        block['table'] = row['attn']
        record.update(inwait_schema=ARM_SCHEMA, inner_record=Path(inner).name, levers=dict(attn=row['attn'], lazy_recv=row['lazy_recv'], decode_merge=row['decode_merge'], inwait=row['inwait'], next_levers=row['next_levers'], prepare_slices=row['prepare_slices'], score_warm=row['score_warm']))
    with open(a.out, 'x') as handle:
        json.dump(record, handle, sort_keys=True)
    print('INWAIT_ARM_RECORD %s %s' % (name, 'COMPLETE' if record.get('complete') is True else 'INCOMPLETE'))
    return 0 if record.get('complete') is True and rc == 0 else 2


# --- the gate ---------------------------------------------------------------------------------------------

def attn_gate(arms, decode_records, ingress, watches, logs, want):
    """am's per-arm check over the prefill arms, the decode arms' verdict and am's startup check; (failures, missing, classes, per-arm facts, startup)."""
    records, attn = {a.get('label'): a for a in arms if a.get('label')}, {}
    failures, missing, classes = set(), set(), set()
    for row in want:
        rec = records.get(row['label'])
        if rec is None:
            continue                       # c1's grade already counts the missing record
        f, m, facts = am.attn_check(rec, row['attn'])
        attn[row['label']] = dict(facts, failures=f, missing=m)
        failures.update(f)
        missing.update(m)
        if f:
            classes.add('HARNESS' if any(x.endswith('requested-differs-from-table') for x in f) else 'ENGINE')
    for d in T.DECODE_ARMS:
        rec = decode_records.get(d['label'])
        if rec is None:
            missing.add(d['label'] + ':no-arm-record')
            continue
        f, m = am.decode_verdict(rec, (ingress or {}).get(d['label']), (watches or {}).get(d['label']))
        af, amiss, facts = am.attn_check(rec, d['attn'])
        attn[d['label']] = dict(facts, failures=af, missing=amiss)
        failures.update(f + af)
        missing.update(m + amiss)
        if f or af:
            classes.add('ENGINE')
    start = am.startup_check(logs)
    failures.update(start['failures'])
    missing.update(start['missing'])
    if start['failures']:
        classes.add('ENGINE')
    return failures, missing, classes, attn, start


def inwait_check(arms, want):
    """(failures, missing, per-arm facts): the selection took and the lever engaged only where selected. Live on arms must apply in the wait; a solo arm's waits are too short to require it."""
    records = {a.get('label'): a for a in arms if a.get('label')}
    failures, missing, facts = set(), set(), {}
    for row in want:
        rec = records.get(row['label'])
        if rec is None:
            continue
        ack, delta = (rec.get('inwait') or {}).get('acknowledged'), rec.get('stage_delta') or {}
        applied, direct = delta.get('step_iw_applied'), [delta.get('step_iw_direct_opens'), delta.get('step_iw_direct_releases')]
        early, calls = delta.get('step_iw_early_begins'), (rec.get('slots_fast') or {}).get('calls')
        facts[row['label']] = dict(requested=row['inwait'], acknowledged=ack, applied=applied, direct=direct, next_levers=row['next_levers'], early_begins=early, slots_fast_calls=calls)
        if ack is not row['inwait']:
            failures.add(row['label'] + ':inwait-selection-not-acknowledged')
        if (rec.get('inwait') or {}).get('early_acknowledged') is not row['next_levers']:
            failures.add(row['label'] + ':early-successor-selection-not-acknowledged')
        if not row['next_levers'] and early:
            failures.add(row['label'] + ':early-successor-off-moved-its-counter')
        if type(calls) is not dict or any(type(v) is not int for v in calls.values()):
            missing.add(row['label'] + ':slots-fast-calls-absent')
        elif row['next_levers'] and not all(calls.values()):
            missing.add(row['label'] + ':slots-fast-selected-not-engaged')
        elif not row['next_levers'] and any(calls.values()):
            failures.add(row['label'] + ':slots-fast-off-moved-its-counter')
        slices, warm = (rec.get('prepare_slices') or {}).get('counters') or {}, (rec.get('score_warm') or {}).get('counters') or {}
        sliced, yields = slices.get('sliced'), slices.get('yields')
        facts[row['label']].update(prepare_slices=row['prepare_slices'], sliced=sliced, yields=yields, peek_missing=slices.get('peek_missing'),
                                   score_warm=row['score_warm'], warm=warm)
        if type(sliced) is not dict or any(type(v) is not int for v in sliced.values()):
            missing.add(row['label'] + ':prepare-slices-counters-absent')
        elif row['prepare_slices'] and not all(sliced.values()):
            missing.add(row['label'] + ':prepare-slices-selected-not-engaged')
        elif not row['prepare_slices'] and (any(sliced.values()) or any((yields or {}).values())):
            failures.add(row['label'] + ':prepare-slices-off-moved-its-counter')
        cow = (rec.get('cow_fast') or {}).get('counters') or {}
        calls, cow_rows = cow.get('calls'), cow.get('rows')
        facts[row['label']].update(cow_fast=row['cow_fast'], cow=cow, reserved_cap=row['reserved_cap'], cap_flushes=((rec.get('reserved_cap') or {}).get('counters') or {}).get('flushes'))
        if type(calls) is not dict or any(type(v) is not int for v in calls.values()):
            missing.add(row['label'] + ':cow-fast-counters-absent')
        elif row['cow_fast'] and not all(calls.values()):
            missing.add(row['label'] + ':cow-fast-selected-not-engaged')
        elif not row['cow_fast'] and any(calls.values()):
            failures.add(row['label'] + ':cow-fast-off-moved-its-counter')
        if type(cow_rows) is not dict or not all(type(v) is int and v for v in cow_rows.values()):
            missing.add(row['label'] + ':cow-receipt-rows-absent')
        allocs, errors = warm.get('allocs'), warm.get('errors')
        if type(allocs) is not dict or any(type(v) is not int for v in allocs.values()):
            missing.add(row['label'] + ':score-warm-counters-absent')
        elif not row['score_warm'] and any(allocs.values()):
            failures.add(row['label'] + ':score-warm-off-moved-its-counter')
        elif any((errors or {}).values()):
            missing.add(row['label'] + ':score-warm-errors')
        if row['next_levers'] and row['cohort'] == 'live' and type(early) is int and not early:
            missing.add(row['label'] + ':early-successor-selected-not-engaged')
        if type(applied) is not int or any(type(x) is not int for x in direct):
            missing.add(row['label'] + ':inwait-counters-absent')
        elif row['inwait'] and row['cohort'] == 'live' and not applied:
            failures.add(row['label'] + ':inwait-selected-applied-nothing-in-the-wait')
        elif not row['inwait'] and (applied or any(direct)):
            failures.add(row['label'] + ':inwait-off-moved-the-in-wait-counters')
    return failures, missing, facts


# --- the open-rows rule on in-wait arms ----------------------------------------------------------------------

_RULE_CHECK_ARM = rule.check_arm


def check_arm_inwait(flushes, overlap):
    """e3_c1_open_rows.check_arm's boundary checks over the commit boundary the engine recorded (review section 7), not n's last apply:
    a flush with open rows names commit_ns and open_seqs. Every named row must be a successor's, in n+1's line, sent at or before the commit
    and not applied before it; and n+1's line holds no other row sent before the commit. The successor and last-line reasons stand."""
    out = _RULE_CHECK_ARM(flushes, overlap)
    ordered = sorted(flushes, key=lambda f: f.get('line') or 0)
    index = {f.get('line'): i for i, f in enumerate(ordered)}
    for cell in out['flagged']:
        # Both are check 3's inference; the record below decides them.
        cell['reasons'] = [r for r in cell['reasons'] if r not in ('c1-open-rows-not-carried', 'c1-open-row-not-successor')]
        if cell['reasons'] or index[cell['line']] + 1 >= len(ordered):
            continue
        flush, nxt = ordered[index[cell['line']]], ordered[index[cell['line']] + 1]
        cell.update(_reconcile(flush, nxt, cell, overlap))
    kept = {'c1-open-rows-unreadable'} & set(out['reasons'])
    return dict(out, reasons=sorted(kept.union(*[c['reasons'] for c in out['flagged']])))


def _reconcile(flush, nxt, cell, overlap):
    identity = flush.get('identity') or {}
    commit_ns, named, k = identity.get('commit_ns'), identity.get('open_seqs'), cell['open_rows']
    if type(commit_ns) is not int or type(named) is not list:
        return dict(reasons=['c1-commit-boundary-absent'])
    reasons = []
    # Times on the line are in whole microseconds below its base: a row at the boundary may read up to 1 us off it.
    rows = {r.get('seq'): r for r in (nxt.get('rows') or {}).get('n', [])}
    if len(named) != k or any(seq not in rows for seq in named):
        reasons.append('c1-open-rows-not-carried')
    own, successor = (overlap.get(cell['txn']) or (frozenset(), None))[0], (overlap.get(cell['next_txn']) or (frozenset(), None))[0]
    if any(seq not in successor or seq in own for seq in named):
        reasons.append('c1-open-row-not-successor')
    for seq in named:
        row = rows.get(seq) or {}
        if type(row.get('t_submit')) is int and row['t_submit'] > commit_ns + 1000:
            reasons.append('c1-open-row-sent-after-the-commit')
        if type(row.get('t_apply')) is int and row['t_apply'] < commit_ns - 1000:
            reasons.append('c1-open-row-applied-before-the-commit')
    extra = [seq for seq, row in rows.items() if seq not in named and type(row.get('t_submit')) is int and row['t_submit'] < commit_ns - 1000]
    if extra:
        reasons.append('c1-row-out-at-the-commit-not-recorded')
    return dict(reasons=sorted(set(reasons)), carried_seqs=list(named), commit_ns=commit_ns)


def commit_boundary_read(flushes_by_arm):
    """Per arm: flushes, how many name the boundary, and rows out at it; a read."""
    out = {}
    for label, flushes in flushes_by_arm.items():
        idents = [f.get('identity') or {} for f in flushes]
        out[label] = dict(flushes=len(idents), recorded=sum(type(i.get('commit_ns')) is int for i in idents),
                          open_rows=sum(i.get('open_rows') or 0 for i in idents), max_open=max((i.get('open_rows') or 0 for i in idents), default=0))
    return out


# --- the exact-path smoke (review section 6) ------------------------------------------------------------------

def smoke_check(rec, row=None):
    """Arm 1's facts: real request shape, real control dispatch, positive native engagement, one complete observation. (ok, reasons); the queue aborts the table on not ok."""
    row = row or T.by_label()[T.SMOKE_ARM]
    if not isinstance(rec, dict):
        return False, ['smoke-record-unreadable']
    reasons = []
    if rec.get('label') != row['label'] or rec.get('complete') is not True or rec.get('reasons'):
        reasons.append('smoke-arm-not-complete')
    outputs = rec.get('outputs') or {}
    if not (outputs.get('fresh') and outputs.get('append') and outputs.get('standing')):
        reasons.append('smoke-request-shape-not-observed')
    if rec.get('generation_requests_expected') != 10:
        reasons.append('smoke-request-shape-differs')
    if (rec.get('inwait') or {}).get('acknowledged') is not row['inwait'] or (rec.get('inwait') or {}).get('early_acknowledged') is not row['next_levers']:
        reasons.append('smoke-control-not-acknowledged')
    if (rec.get('attn_meta') or {}).get('select') is None:
        reasons.append('smoke-select-not-sent')
    delta, chunks = rec.get('stage_delta') or {}, (rec.get('progress_delta') or {}).get('chunks')
    if type(chunks) is not int or chunks < 1 or (rec.get('progress_delta') or {}).get('line_errors'):
        reasons.append('smoke-no-native-chunks')
    if not delta.get('step_iw_applied'):
        reasons.append('smoke-in-wait-not-engaged')
    if not delta.get('step_iw_early_begins'):
        reasons.append('smoke-early-successor-not-engaged')
    for name, field in (('slots_fast', None), ('prepare_slices', 'sliced'), ('cow_fast', 'calls'), ('cow_fast', 'rows')):
        block = rec.get(name) or {}
        values = block.get('calls') if field is None else (block.get('counters') or {}).get(field)
        if type(values) is not dict or len(values) != 4 or not all(type(v) is int and v > 0 for v in values.values()):
            reasons.append('smoke-%s-not-engaged-on-every-rank' % (field or name).replace('_', '-'))
    return not reasons, sorted(set(reasons))


# --- the reads --------------------------------------------------------------------------------------------

def _ratio(x, y):
    return round(x / y, 4) if type(x) in (int, float) and type(y) in (int, float) and y else None


def compose_read(result, decode_records, attn):
    """The lead's table: per arm, per (selection, band, cohort), the decode baselines and the stack. A read."""
    table, gate_rows, step_arms = result.get('table') or {}, (result.get('step_gate') or {}).get('arms') or {}, (result.get('step_read') or {}).get('arms') or {}
    rate = {d['label']: ((decode_records.get(d['label']) or {}).get('decode_only') or {}).get('aggregate_tokens_per_s') for d in T.DECODE_ARMS}
    reference = T.FIXED_REFERENCE_TOK_S
    arm_records = {a.get('label'): a for a in (result.get('prefill_records') or ()) if a.get('label')}
    arms = {}
    for row in T.arms():
        cell, step, facts = table.get(row['label']) or {}, (step_arms.get(row['label']) or {}).get('step') or {}, attn.get(row['label']) or {}
        during = cell.get('decode_during_tok_s')
        rec = arm_records.get(row['label']) or {}
        delta, wait = rec.get('stage_delta') or {}, rec.get('progress_delta') or {}
        arms[row['label']] = dict(
            c1=row['c1'], band=row['band'], cohort=row['cohort'], index=row['index'], cold=row['label'] == T.COLD_ARM, prefill_tok_s=cell.get('prefill_tok_s'), decode_during_tok_s=during,
            decode_ratio_own_window=cell.get('decode_ratio'), decode_vs_fixed_reference=_ratio(during, reference), decode_vs_this_run_am1=_ratio(during, rate[T.REFERENCE_ARM]),
            wait_sends=wait.get('wait_sends'), wait_capped=wait.get('wait_capped'), waits=wait.get('waits'), inwait={k: delta.get('step_iw_' + k) for k in IW_DELTA},
            stage1_thread_share=fread.slower(step, 'thread_share'), stage1_device_forward_p50_ms=fread.slower(step, 'dev_forward_p50_ms'), stage1_pair_ms=fread.slower(step, 'pair_ms'),
            stage1_prep_attn_meta_ms=fread.slower(step, 'prep_attn_meta_ms'), stage1_full_steps=fread.slower(step, 'full_n'), merge=(gate_rows.get(row['label']) or {}).get('counters'),
            attn_batched=facts.get('batched'), attn_fallback=facts.get('fallback'))
    cells = {}
    for name, r in arms.items():
        cells.setdefault((r['c1'], r['band'], r['cohort']), []).append(r)
    medians = {'%s band %d %s' % key: {k: progress.median([r[k] for r in rows if type(r[k]) in (int, float)] or [None]) if any(type(r[k]) in (int, float) for r in rows) else None
                                       for k in ('prefill_tok_s', 'decode_ratio_own_window', 'decode_vs_fixed_reference', 'decode_vs_this_run_am1', 'stage1_thread_share', 'stage1_device_forward_p50_ms', 'stage1_pair_ms')}
               for key, rows in sorted(cells.items())}

    def cell_of(sel, cohort):
        return medians.get('%s band 1 %s' % (sel, cohort)) or {}
    pair = dict(live_on_over_off=_ratio(cell_of('cf', 'live').get('prefill_tok_s'), cell_of(T.OFF, 'live').get('prefill_tok_s')),
                solo_on_over_off=_ratio(cell_of('cf', 'solo').get('prefill_tok_s'), cell_of(T.OFF, 'solo').get('prefill_tok_s')),
                decode_vs_fixed_on=cell_of('cf', 'live').get('decode_vs_fixed_reference'), decode_vs_fixed_off=cell_of(T.OFF, 'live').get('decode_vs_fixed_reference'),
                live_cap_over_off=_ratio(cell_of('cap', 'live').get('prefill_tok_s'), cell_of(T.OFF, 'live').get('prefill_tok_s')),
                decode_vs_fixed_cap=cell_of('cap', 'live').get('decode_vs_fixed_reference'))
    return dict(is_gate=False, reference_arm=T.REFERENCE_ARM, fixed_reference_tok_s=reference, decode_only={k: v for k, v in rate.items()}, arms=arms, cells=medians, pair=pair)


# --- the grade --------------------------------------------------------------------------------------------

def grade(arms, ingress, checks, logs, watches=None, part='core'):
    """sb's grade over this table, then am's check and the decode arms; the reads and the record fields."""
    known = [a for a in arms if a.get('label')]
    decode_records = {a['label']: a for a in known if a.get('kind') == 'decode-only'}
    prefill = [a for a in arms if a.get('kind') != 'decode-only']
    with _bound(sb, sb_bindings()), _bound(rule, dict(check_arm=check_arm_inwait)):
        result = sb.grade(prefill, ingress, checks, logs, watches, part=part)
    want = T.arms(None if part == 'all' else T.GRADED)
    afail, amiss, acls, attn, start = attn_gate(prefill, decode_records, ingress, watches, logs, want)
    ifail, imiss, ifacts = inwait_check(prefill, want)
    failures, missing = set(result['failures']) | afail | ifail, set(result['missing']) | amiss | imiss
    classes = set(result['classification'] or ()) | acls | ({'ENGINE'} if ifail else set())
    for name, cell in result['table'].items():
        row = T.by_label().get(name)
        if row:
            cell.update(attn=row['attn'])
    verdict = 'FAIL' if failures else 'INC' if missing else 'PASS'
    if verdict == 'FAIL':
        classes = classes or {'HARNESS'}
    elif verdict == 'INC' and result['open_rows_rule']['engine_classified_arms']:
        classes = {'ENGINE'}
    else:
        classes = set()
    result['prefill_records'] = prefill
    read = compose_read(result, decode_records, attn)
    del result['prefill_records']
    result.update(verdict=verdict, classification=sorted(classes) or None, failures=sorted(failures), missing=sorted(missing), attn_meta=attn, attn_meta_is_gate=True, startup=start,
                  inwait_check=ifacts, compose_read=read, basis=BASIS, scope=SCOPE,
                  append_step_exempt={a['label']: a['append_step_exempt'] for a in known if a.get('append_step_exempt')})
    if part != 'core':
        result['read_only'] = 'a read over every arm; the queue verdict is the core grade (inwait-grade.json)'
    return result


# --- the report ---------------------------------------------------------------------------------------------

_cell = c1._cell


def report(result):
    with _bound(sb, sb_bindings()):
        text = sb.report(result)
    text = text.replace('# q18a-stage1-step report', '# q18a-inwait report', 1).replace('# q18a-stage1-step read', '# q18a-inwait read', 1)
    read = result['compose_read']
    lines = ['## inwait reads', '', 'Pair, band 1 medians: %s.' % json.dumps(read['pair'], sort_keys=True), '',
             'Decode-only tok/s: %s; fixed reference %s tok/s.' % (json.dumps(read['decode_only'], sort_keys=True), read['fixed_reference_tok_s']), '',
             'Append rule, step_* exempted, delta kept: %s.' % (json.dumps(result['append_step_exempt'], sort_keys=True) or '{}'), '',
             '| arm | prefill tok/s | decode own | decode vs fixed | decode vs this run | wait sends | wait capped | applied in wait | refused (receipts) | S1 thread share | S1 forward p50 ms | merge steps | cold |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for name, r in read['arms'].items():
        lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s (%s) | %s | %s | %s | %s |' % (
            name, _cell(r['prefill_tok_s']), _cell(r['decode_ratio_own_window'], 4), _cell(r['decode_vs_fixed_reference'], 4), _cell(r['decode_vs_this_run_am1'], 4), _cell(r['wait_sends']), _cell(r['wait_capped']),
            _cell(r['inwait']['applied']), _cell(r['inwait']['refused_total']), _cell(r['inwait']['refused_receipts']), _cell(r['stage1_thread_share'], 3), _cell(r['stage1_device_forward_p50_ms'], 3),
            _cell((r['merge'] or {}).get('merge_steps')), 'cold arm' if r['cold'] else '-'))
    lines += ['', '| cell | prefill tok/s | decode own | decode vs fixed | decode vs this run | S1 thread share | S1 forward p50 ms |', '|---|---|---|---|---|---|---|']
    for name, m in read['cells'].items():
        lines.append('| %s | %s | %s | %s | %s | %s | %s |' % (name, _cell(m['prefill_tok_s']), _cell(m['decode_ratio_own_window'], 4), _cell(m['decode_vs_fixed_reference'], 4), _cell(m['decode_vs_this_run_am1'], 4),
                                                              _cell(m['stage1_thread_share'], 3), _cell(m['stage1_device_forward_p50_ms'], 3)))
    head, sep, tail = text.partition('\n## Stage-1 per-step levers\n')
    return head + '\n' + '\n'.join(lines) + '\n' + sep + tail if sep else text + '\n' + '\n'.join(lines) + '\n'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='action', required=True)
    r = sub.add_parser('arm')
    r.add_argument('--url', required=True)
    r.add_argument('--model', required=True)
    r.add_argument('--run-id', required=True)
    r.add_argument('--epoch', type=int, required=True)
    r.add_argument('--c1', choices=T.SELECTIONS, required=True, help="this arm's selection; C1 defer, attn-meta and L1 + L2 in every arm")
    r.add_argument('--size', type=int, choices=T.SIZES, required=True)
    r.add_argument('--cohort', choices=progress.COHORTS, required=True)
    r.add_argument('--index', type=int, choices=T.INDEXES, required=True)
    r.add_argument('--workload', required=True)
    r.add_argument('--control-timeout-s', type=float, default=120.0)
    r.add_argument('--out', required=True)
    k = sub.add_parser('smoke', help='the exact-path smoke: 0 engaged, 2 not (the queue aborts the table)')
    k.add_argument('--arm', required=True)
    k.add_argument('--out', required=True)
    d = sub.add_parser('decode-arm')
    for flag in ('--url', '--model', '--run-id', '--workload', '--out'):
        d.add_argument(flag, required=True)
    d.add_argument('--epoch', type=int, required=True)
    d.add_argument('--label', choices=tuple(x['label'] for x in T.DECODE_ARMS), required=True)
    d.add_argument('--attn', choices=('on', 'off'), required=True)
    d.add_argument('--control-timeout-s', type=float, default=120.0)
    for action, text in (('grade', 'the core grade: the queue verdict'), ('read', 'every arm: a read only')):
        g = sub.add_parser(action, help=text)
        g.add_argument('--arm', action='append', required=True)
        g.add_argument('--ingress', action='append', default=[], help='label=path')
        g.add_argument('--watch', action='append', default=[], help='label=engine-watch record path')
        g.add_argument('--check', action='append', default=[], help='label=path')
        g.add_argument('--log', action='append', default=[], help='followed engine log (stdout/stderr)')
        g.add_argument('--report', help='write the report here')
        g.add_argument('--out', required=True)
    a = p.parse_args(argv)
    if Path(a.out).exists():
        return 2
    if a.action == 'arm':
        return run_arm(a)
    if a.action == 'decode-arm':
        return am.run_decode_arm(a)
    if a.action == 'smoke':
        ok, reasons = smoke_check(base.load(a.arm))
        with open(a.out, 'x') as handle:
            json.dump(dict(schema='inwait-smoke/1', engaged=ok, reasons=reasons), handle, sort_keys=True)
        print('INWAIT_SMOKE %s %s' % ('ENGAGED' if ok else 'NOT_ENGAGED', ','.join(reasons)))
        return 0 if ok else 2
    arms = [base.load(x) or {'label': None, 'complete': False, 'reasons': ['arm-artifact-unreadable:' + Path(x).name]} for x in a.arm]
    result = grade(arms, ap.load_pairs(a.ingress), ap.load_pairs(a.check), a.log, ap.load_pairs(a.watch), part='core' if a.action == 'grade' else 'all')
    with open(a.out, 'x') as handle:
        json.dump(result, handle, sort_keys=True)
    if a.report:
        with open(a.report, 'x') as handle:
            handle.write(report(result))
    print('INWAIT_%s %s' % ('GRADE' if a.action == 'grade' else 'READ', result['verdict']))
    if a.action == 'read':
        return 0 if result['verdict'] == 'PASS' else 2      # a read: never 1, the queue does not count it
    return {'PASS': 0, 'FAIL': 1}.get(result['verdict'], 2)


if __name__ == '__main__':
    raise SystemExit(main())
