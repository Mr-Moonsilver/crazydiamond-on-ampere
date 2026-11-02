#!/usr/bin/env python3
"""alpha-next-c1 box arms and grade: C1 (two chunks in flight) against C0's local path on the fused, grouped alpha-next engine.

Derived from e3_alpha_next_box (imported as ``an``, unchanged): every arm and grade step not defined here is alpha-next's, run with
the names below bound in its place for the duration of one call (``_bound``); alpha-next's own arm and grade in the same process
are untouched. The arm table is alpha_next_c1_arms (one declared table): an arm not in it is refused before any traffic.

The arm: alpha-next's (Step B's arm with alpha-next's seams), with the stage control sending ``overlap`` and ``defer_bind``
(explicitly in every arm: true on a C1-defer arm, false otherwise). Control order: pause, native_progress {burst},
native_qsa_fused {true}, native_layer_group {size}, native_stage_local {local: true, overlap, defer_bind}, release_auto_arm,
resume, then status only. The reply must carry ``local`` true and the arm's ``overlap``; the status before and after the fresh
prompt must show stage-local and the arm's overlap selected (the overlap block, its counters flat as ``overlap_<counter>`` in the
stage view and the stage delta; its defer_bind block {installed, selected, counters} flat as ``defer_bind_installed``,
``defer_bind_selected`` and ``defer_<counter>``). The record keeps ``c0`` 'on' and ``stage_local`` true (C0 local in every arm),
carries ``selection`` (``c1on.g2``, ``c1defer.g2``) and ``defer_bind`` {requested, acknowledged, previous, replied}: the value
the engine's reply acknowledged beside the one the arm sent. The arm does not refuse a different acknowledgement (the record must
reach the grade, which refuses the arm).

C1's grade pieces are restated from e3_alpha_c1_box (claude/c1), not imported: importing it imports e3_alpha_c0_box, which wraps
progress.chunk_breakdown for the whole process (alpha-next's reason for restating C0). test_alpha_next_c1_queue.C1Parity compares
them with C1's source. alpha-next's record shapes differ in one place: the alpha-next engine writes ``flash-s5 memory`` receipts
through the PLAN 2 writer, which appends `` pid=N seq=N sha=H``; MEMORY_WHOLE accepts that identity after ``device_frees``.

The grade is alpha-next's grade (e3_alpha_next_box.grade: Step B's grade inside alpha-next's seams; numerics a read with the
control envelope, NUMERICS_IS_GATE False; RECEIPT_INTEGRITY_IS_GATE False; STOP_RACE in the draft; the memory-peak receipt and
the C0 ring as reads) over this table through a view (``table_view``) in which alpha-next's C0 on/off axis is the C1 selection:
the controls of the numerics envelope, the matched pairs of the memory read and of the speed and decode reads are the graded
C1-off arms (C0 local); alpha-next's result names (``over_c0_off``, ``c0_decode_delta``, ``memory_c0_on_vs_off``,
``min_free_below_c0_off_mib``) keep their keys and there name C1 on against C1 off (``axis`` in the record says so). C0's own
engagement (stage-local chunks, two local releases per chunk, one stage-local line per chunk, no reply from a non-target rank) is
checked on every arm, C1 on included. Plus:
  regroup     on a C1-on arm one engine progress line is flushed per commit and may hold the successor's early rows. Within each
              arm span the engine flushes' native rows are regrouped by the ``flash-s5 stage-overlap txn=T seqs=RANGES
              successor=S`` lines before the per-chunk join (C1's rule). A sequence two lines name, or engine flushes without one
              transaction each, are reasons (INC). Rows whose chunk has no flush in the arm are a reason (INC).
  engagement  a C1-on arm needs, from the status delta of the fresh prompt, a successor frame and an overlapped commit (else INC
              'overlap-ran-no-second-chunk') and at least one early open (else INC 'overlap-stage-zero-never-opened-a-successor-
              early': coverage not reached), one stage-overlap line per auto-owned chunk, and every regrouped chunk with exactly
              two opens and two releases. A C1-off arm needs every overlap counter delta 0 and no stage-overlap line.
  in flight   one command in flight per stage (FAIL ENGINE on a breach), from the engine's own segment rows: per stage, in
              submit order, each segment command submitted no earlier than the previous one's host apply (its last target reply
              when it has no apply). Host submit to host apply is one RPC; the fused and grouped run-ahead is device-side (accepted
              ops not yet host-queried complete, the worker's pending_op_seqs) and does not show in the rows: the stage driver
              submits nothing on a stage while an RPC of that stage is in flight (stage_progress._drive_stage and progress_in_wait),
              so C0-local arms with fused and grouped run-ahead pass. Declared limit (LIMITS['in_flight']).
  signatures  the overlap path's contract errors (OVERLAP_SIGNATURES) join alpha-next's (FAIL ENGINE).
  OOM         any rise of a rank's ``ooms`` counter, read from a whole receipt or from each ``flash-s5 memory`` fragment of a torn
              line, fails the arm whose span holds it as ENGINE (tail arms: in the read), outside every arm the run (C1's rule).
              Step B's OOM rule (an OOM in the arm's receipts) still applies.
  memory      alpha-next's memory read (MET / NOT_MET / INC, never a verdict input), each graded C1-on arm against its matched
              C1-off arm (same band, cohort and round), plus per chunk the ``flash-s5 memory-peak`` receipt: worst rank peak
              allocated and reserved, C1 on minus its matched C1-off arm per chunk start, the chunks past 97,152 rows apart. The
              g4 tail is reported only (no matched arm). Per stage (stage 1 = ranks 2 and 3, the least free memory), the lowest
              free MiB over the stage's receipts and the highest memory-peak allocated, against the matched C1-off arm (a read).
  tears       alpha-next's rule (Step A's torn-line rule) plus torn stage-overlap lines: each counts as a lost record, and the
              chunk without its line drops out with its predecessor (whose flush kept the successor's early rows). A lost engine
              flush on a C1-on arm is INC ('overlap-arm-engine-flush-lost'): its successor's early rows cannot be placed.
  budget      an arm the queue did not start (budget) has no record and is INC ('no-arm-record'), never an engine result.
  defer_bind  every arm sends defer_bind explicitly (DEFER_BIND: true on C1-defer arms, false on C1 on and off). The arm's
              record must carry the requested and the acknowledged value (else FAIL HARNESS); the engine's reply must have
              acknowledged the requested value and its status before and after the fresh prompt must show it selected with the
              deferred bind installed (else FAIL ENGINE: 'defer-bind-acknowledged-differs-from-requested', ...). Engagement: a
              C1-defer arm needs a deferred bind in its status delta (else INC, coverage not reached), any other arm none
              moved (else INC); missing counters are INC.
  observe     c1_overlap_read over the ``flash-s5 overlap-observe`` lines (FLASH_S5_STAGE_OVERLAP_OBSERVE=1), joined to arms by
              sequence (each chunk's ``seqs`` against the arm's 'n' rows; the row schema is unchanged): per arm OVERLAP or NONE
              per successor with the reason, the per-op device bounds summary and the observe log bytes. A read, never a
              verdict input: plain C1-on arms are the control (NONE expected), reported as a control.
  numbers     beside alpha-next's C0/C1 numbers, per arm (``arm_numbers``): live and solo prefill tok/s, the decode ratio, the
              episode loss TTFT_live x (1 - live/solo), stage 0 and stage 1 s/chunk on the slower TP rank, per rank min free,
              max reserved, alloc retries and OOMs (stage 0 rank 0 apart) and alpha-next's memory MET / NOT_MET against the
              matched C1-off arm, unchanged; a C1-defer arm also against its matched C1-on arm. A read.

Two grades as alpha-next: ``grade`` over the core arms (the queue verdict, written before the g4 tail), ``read`` over every arm
(a read only). Receipts hold counts, hashes, identities, token ids and host clock stamps, never prompts or auth.
"""
import argparse
import contextlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

import alpha_next_c1_arms as T
import c1_overlap_read as observe_read
import e3_alpha_next_box as an

b, sa, ap, progress, base = an.b, an.sa, an.ap, an.progress, an.base
torn_lines = sa.torn_lines
GRADE_SCHEMA = 'alpha-next-c1-grade/1'
READ_SCHEMA = 'alpha-next-c1-read/1'
STAGE_CONTROL = an.STAGE_CONTROL
TARGETS = an.TARGETS
ARENA_ROWS = ap.ARENA_ROWS
# C1's declarations (e3_alpha_c1_box), restated; test_alpha_next_c1_queue.C1Parity compares them with its source text.
OVERLAP_COUNTERS = ('successor_frames', 'overlapped_commits', 'early_opens', 'lane_checks', 'closed_by_successor',
                    'successor_retires', 'unbegun_successor_cancels', 'partial_replays')
# The overlap path's contract errors (patches/s5_engine/stage_overlap.py, DriverError texts).
OVERLAP_SIGNATURES = (('overlap-lane', 'stage 1 opened another lane than the successor was bound to'),
                      ('overlap-row', 'open does not continue the retained request row'),
                      ('overlap-continue', 'a successor does not continue the committed row'),
                      ('overlap-foreign-partial', 'stop found a foreign partially accepted command'),
                      ('overlap-unopened', 'an unopened rank has accepted operations'),
                      ('overlap-retire', 'successor cancellation is not a clean terminal row'),
                      ('overlap-unbound', 'accepted operations exist without a bound producer'),
                      ('overlap-order', 'a stage opened the successor before releasing the older chunk'),
                      ('overlap-closed', 'a row closed by the successor lacks the local release'))
# The deferred PP bind's counters (patches/s5_engine/stage_overlap.py OverlapSegmentDriver.defer_counters), in the status at
# native_stage_local.overlap.defer_bind.counters.
DEFER_COUNTERS = ('deferred_opens', 'deferred_binds')
OVERLAP_MARK = 'flash-s5 stage-overlap '
OVERLAP_LINE = re.compile(r'flash-s5 stage-overlap txn=(\d+) seqs=(-|\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*) successor=(\d+|none)\s*$')
MEMORY_MARK = torn_lines.MEMORY_MARK
# The whole receipt, to its last field (worker_bridge._memory_receipt), and on alpha-next the PLAN 2 writer's identity after it
# (native_measurement.ReceiptWriter._emit_kv).
RECEIPT_IDENTITY = r'(?: pid=\d+ seq=\d+ sha=[0-9a-f]+)?'
MEMORY_WHOLE = re.compile(torn_lines.MEMORY_LINE.pattern + r' total=(-?\d+) sync_all_streams=(-?\d+) device_allocs=(-?\d+) '
                                                           r'device_frees=(-?\d+)' + RECEIPT_IDENTITY + r'\s*$')
FRAGMENT_FIELD = re.compile(r'\b(rank|ooms)=(\d+)\b')
LIMITS = dict(
    in_flight='host submit to host apply of the engine\'s own segment rows (CLOCK_MONOTONIC on the owner); device execution and '
              'the fused and grouped run-ahead (accepted, not yet complete) are not visible; only segment rows carry a stage '
              '(opens, binds, releases, receipts, polls and profile queries are not judged); a lost engine flush hides its rows',
    oom='a torn receipt is read per fragment; a misread number errs toward FAIL',
    memory_peak='the engine\'s per-chunk memory-peak receipt (worst rank), C1 on minus its matched C1-off arm by chunk start; a read',
    budget='an arm the queue did not start (budget) has no record: INC, never an engine result')
AXIS = ('alpha-next\'s C0 on/off names in this record (over_c0_off, c0_decode_delta, memory_c0_on_vs_off, '
        'min_free_below_c0_off_mib, the numerics controls and candidates) are the C1 selection: on = C1 overlap, off = C0 local; '
        'C0 stage-local is selected in every arm')

# alpha-next's own functions, taken before any binding.
AN_RELEASE, AN_STAGE_REASONS = an.release_alpha_next, an.stage_reasons
AN_STAGE_VIEW, AN_STAGE_DELTA = an.stage_view, an.stage_delta
AN_ENGAGEMENT, AN_REPORT = an.engagement, an.report
TORN_SETTLE = torn_lines.settle_arm
SA_ARM_VERDICT = sa.arm_verdict


def label(c1, size, cohort, index):
    return T.label(c1, size, cohort, index)


def required_labels(part='core'):
    return T.labels(None if part == 'all' else T.GRADED)


_MISSING = object()


@contextlib.contextmanager
def _bound(module, names):
    """``names`` in place of the module's for one call; every name restored (or removed) on exit."""
    saved = {k: getattr(module, k, _MISSING) for k in names}
    for k, v in names.items():
        setattr(module, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is _MISSING:
                delattr(module, k)
            else:
                setattr(module, k, v)


_TABLE_NAMES = ('STANDING_DECODERS', 'STANDING_GREEDY', 'GREEDY_TEMPERATURE', 'STANDING_SAMPLED', 'BAND1_ROUNDS',
                'BAND1_REQUIRED_ROUNDS', 'BAND1_INDEXES', 'BAND2_INDEX', 'BAND_SIZE', 'ARM_TABLE', 'PARTS', 'ENTRY_LIMIT',
                'GRADED', 'COHORTS', 'INDEXES', 'SIZES', 'band_of_index', 'selection', 'label', 'parse_selection', 'labels',
                'matched_off', 'band1_glob', 'check')


def table_view(c1=None):
    """This table as alpha-next's code reads alpha_next_arms.

    For the grade (``c1`` None): every row's ``c0`` is the C1 selection, the candidate axis of alpha-next's comparisons (a row
    keeps ``c1``, ``overlap`` and ``defer_bind``): 'off' for C1 off (the controls), 'on' for C1 on and for C1 defer (both
    candidates against their matched C1-off arm; the defer arm's own comparison with its matched C1-on arm is this module's,
    ``arm_numbers``); ``parse_selection`` reads ``c1defer`` as a candidate the same way, so alpha-next's speed read keys it
    against C1 off under its own group key. ``C0`` maps every selection to stage-local True. For one arm (``c1`` 'on', 'off' or
    'defer'): the arm's label and selection carry ``c1`` whatever C0 value alpha-next passes ('on', C0 local), so the record
    keeps c0 'on'."""
    names = {k: getattr(T, k) for k in _TABLE_NAMES}

    def axis(c1):
        return 'off' if c1 == 'off' else 'on'

    def arms(part=None):
        return [dict(r, c0=axis(r['c1'])) for r in T.arms(part)]

    def parse_selection(token):
        parsed = T.parse_selection(token)
        return None if parsed is None else (axis(parsed[0]), parsed[1])
    names.update(arms=arms, by_label=lambda: {r['label']: r for r in arms()}, C0={'on': True, 'off': True},
                 parse_selection=parse_selection)
    if c1 is not None:
        names.update(selection=lambda _c0, size: T.selection(c1, size),
                     label=lambda _c0, size, cohort, index: T.label(c1, size, cohort, index), C0={'on': True})
    return SimpleNamespace(**names)


# --- the arm --------------------------------------------------------------------------------------------

def stage_view(status):
    """alpha-next's view plus the overlap block, flat (``overlap_selected``, ``overlap_<counter>``)."""
    row = AN_STAGE_VIEW(status)
    if row is None:
        return None
    block = status[STAGE_CONTROL].get('overlap')
    if type(block) is not dict or type(block.get('counters')) is not dict:
        raise ValueError('native-stage-overlap-not-installed')
    defer = block.get('defer_bind')
    defer = defer if type(defer) is dict else {}
    counters = defer.get('counters') if type(defer.get('counters')) is dict else {}
    return dict(row, overlap_selected=block.get('selected'),
                **{'overlap_' + k: block['counters'].get(k) for k in OVERLAP_COUNTERS},
                defer_bind_installed=defer.get('installed'), defer_bind_selected=defer.get('selected'),
                **{'defer_' + k: counters.get(k) for k in DEFER_COUNTERS})


def stage_delta(after, before):
    out = AN_STAGE_DELTA(after, before)
    out.update({'overlap_' + k: progress.delta(after, before, 'overlap_' + k) for k in OVERLAP_COUNTERS})
    out.update({'defer_' + k: progress.delta(after, before, 'defer_' + k) for k in DEFER_COUNTERS})
    return out


def release_c1(overlap, defer_bind=False):
    """alpha-next's release with ``overlap`` and ``defer_bind`` added to the stage command; the reply must carry ``overlap``. The
    reply's ``defer_bind`` (the selected value after the command) is recorded beside the requested one, never refused here."""
    def factory(size, local):
        release = AN_RELEASE(size, local)

        def run(client, record, mode):
            replies = {}

            class Selecting:
                def __getattr__(self, name):
                    return getattr(client, name)

                def call(self, action, **fields):
                    if action == STAGE_CONTROL:
                        fields = dict(fields, overlap=overlap, defer_bind=defer_bind)
                    reply = client.call(action, **fields)
                    if action == STAGE_CONTROL:
                        replies['stage'] = reply
                    return reply
            try:
                release(Selecting(), record, mode)
            finally:
                reply = replies.get('stage')
                if type(reply) is dict and type(record.get('stage_switch')) is dict:
                    record['stage_switch'].update(overlap=reply.get('overlap'), previous_overlap=reply.get('previous_overlap'),
                                                  defer_bind=reply.get('defer_bind'),
                                                  previous_defer_bind=reply.get('previous_defer_bind'))
                record['defer_bind'] = dict(requested=defer_bind, replied=type(reply) is dict,
                                            acknowledged=reply.get('defer_bind') if type(reply) is dict else None,
                                            previous=reply.get('previous_defer_bind') if type(reply) is dict else None)
            if record['stage_switch'].get('overlap') is not overlap:
                raise ValueError('native-stage-overlap-selection-unproved')
        return run
    return factory


def stage_reasons_c1(overlap):
    def reasons(record, local):
        out = AN_STAGE_REASONS(record, local)
        views = {s.get('label'): s for s in record.get('progress') or []}
        before = (views.get('before-fresh') or {}).get(STAGE_CONTROL)
        after = (views.get('after-fresh') or {}).get(STAGE_CONTROL)
        if before and after and (before.get('overlap_selected') is not overlap or after.get('overlap_selected') is not overlap):
            out.append('stage-overlap-selection-not-this-arm')
        return out
    return reasons


def _arm_print(*args, **kwargs):
    text = ' '.join(str(x) for x in args)
    print(text.replace('ALPHA_NEXT_ARM ', 'ALPHA_NEXT_C1_ARM ', 1), **kwargs)


def arm_names(c1):
    overlap = T.C1[c1]
    return dict(T=table_view(c1), release_alpha_next=release_c1(overlap, T.DEFER_BIND[c1]), stage_reasons=stage_reasons_c1(overlap),
                stage_view=stage_view, stage_delta=stage_delta, print=_arm_print)


def run_arm(a):
    """alpha-next's arm, C0 local ('on') with this arm's C1 selection and defer_bind."""
    args = argparse.Namespace(**dict(vars(a), stage_local='on'))
    with _bound(an, arm_names(a.c1)):
        return an.run_arm(args)


# --- regroup (the post-run report imports these) -----------------------------------------------------------

def seqs_of(text):
    out = []
    for part in text.split(','):
        if part == '-':
            continue
        a, _, b = part.partition('-')
        out += list(range(int(a), int(b or a) + 1))
    return out


def overlap_owners(texts):
    """seq -> txn from the whole stage-overlap lines; a sequence named by two chunks is a fault, never resolved here."""
    owner, faults = {}, []
    for text in texts:
        m = OVERLAP_LINE.search(text)
        if not m:
            continue
        txn = int(m.group(1))
        for seq in seqs_of(m.group(2)):
            if seq in owner and owner[seq] != txn:
                faults.append('seq-%d-named-by-two-chunks' % seq)
            owner[seq] = txn
    return owner, faults


def regroup(flushes, owner, stage_ranges):
    """Engine flushes -> one chunk dict per transaction (rows, identity, flushes).

    ``stage_ranges`` is txn -> (chunk_start, chunk_end) from the stage-local lines. Stock rows ('s', 'w') stay with the flush
    that holds them: they belong to time, not to a chunk. A flush's rows no overlap line names stay with that flush's own chunk.
    """
    chunks = {}
    for flush in flushes:
        # The engine flush names its transaction in its identity (the box record's shape); a flush without one is a fault,
        # never a silent drop of its rows.
        own_txn = (flush.get('identity') or {}).get('txn', flush.get('txn'))
        if type(own_txn) is not int:
            raise ValueError('engine flush at line %r names no transaction' % flush.get('line'))
        for tag, rows in (flush.get('rows') or {}).items():
            for row in rows:
                txn = owner.get(row.get('seq'), own_txn) if tag == 'n' else own_txn
                chunk = chunks.setdefault(txn, dict(rows={}, identity={}, flushes=[]))
                chunk['rows'].setdefault(tag, []).append(row)
        mine = chunks.setdefault(own_txn, dict(rows={}, identity={}, flushes=[]))
        mine['flushes'].append(flush.get('line'))
        mine['identity'] = dict(flush.get('identity') or {})
    for txn, chunk in chunks.items():
        if txn in stage_ranges:
            chunk['identity'].update(chunk_start=stage_ranges[txn][0], chunk_end=stage_ranges[txn][1])
        chunk['identity']['txn'] = txn
    return [chunks[t] for t in sorted(chunks)]


# --- engine log ----------------------------------------------------------------------------------------

def overlap_rows(logs):
    rows = ap.read_stamped(logs, lambda line: OVERLAP_MARK in line)
    return [r for r in rows if OVERLAP_LINE.search(r[1])], [r for r in rows if not OVERLAP_LINE.search(r[1])]


def _fragments(text):
    """(rank or None, ooms or None) per ``flash-s5 memory`` fragment of a line."""
    out = []
    for piece in text.split(MEMORY_MARK)[1:]:
        fields = {}
        for name, value in FRAGMENT_FIELD.findall(piece):
            fields.setdefault(name, int(value))
        out.append((fields.get('rank'), fields.get('ooms')))
    return out


def oom_rises(logs, arms):
    """Every rise of a rank's ooms counter, intact or torn, attributed to the arm span holding it (else run level)."""
    previous = {rank: 0 for rank in range(4)}
    by_arm, outside = {}, []
    for stamp, text in sorted(ap.read_stamped(logs, lambda line: MEMORY_MARK in line), key=lambda r: (r[0] is None, r[0] or 0)):
        whole = MEMORY_WHOLE.search(text)
        pieces = [(int(whole.group(2)), int(whole.group(10)))] if whole else _fragments(text)
        for rank, ooms in pieces:
            if ooms is None:
                continue
            if rank is None:
                rose = ooms > min(previous.values())
            else:
                rose = ooms > previous.get(rank, 0)
                previous[rank] = max(previous.get(rank, 0), ooms)
            if not rose:
                continue
            row = dict(rank=rank, ooms=ooms, torn=whole is None, stamp=stamp)
            owner = next((a['label'] for a in arms if ap.in_span(stamp, a)), None)
            (by_arm.setdefault(owner, []) if owner else outside).append(row)
    return by_arm, outside


def in_flight(segments):
    """Per stage, segment commands in submit order; each must be submitted no earlier than the previous one ended."""
    out = {}
    for stage in TARGETS:
        rows = sorted((s for s in segments if s[0] == stage), key=lambda s: (s[1], s[3]))
        breaches = [dict(seq=b[3], previous_seq=a[3], early_s=(a[2] - b[1]) / 1e9)
                    for a, b in zip(rows, rows[1:]) if a[2] is not None and b[1] < a[2]]
        out[str(stage)] = dict(commands=len(rows), unbounded=sum(1 for s in rows if s[2] is None), breaches=breaches)
    return out


def _segment(row, stage):
    if type(row.get('t_apply')) is int:
        end = row['t_apply']
    else:
        replies = [row.get('t_recv%d' % k) for k in TARGETS[stage] if type(row.get('t_recv%d' % k)) is int]
        end = max(replies) if replies else None
    return (stage, row['t_submit'], end, row['seq'])


# --- the bindings during alpha-next's grade --------------------------------------------------------------

_RUN = {}


def _overlap_arm(arm):
    return (T.by_label().get(arm.get('label')) or {}).get('overlap') is True


def regrouping(original):
    """progress.assemble with each arm's engine flushes regrouped by that arm's stage-overlap lines."""
    def assemble(records):
        flushes, problems = original(records)
        engine = [f for f in flushes if f['side'] == 'engine']
        replaced = {}
        for arm in _RUN['arms']:
            mine = [f for f in engine if id(f) not in replaced and ap.in_span(f['stamp'], arm)]
            # The arm's 'n' row sequences (the row schema unchanged): the overlap observation joins its chunks to arms by these.
            _RUN['seqs'][arm['label']] = {r['seq'] for f in mine for r in f['rows'].get('n', []) if type(r.get('seq')) is int}
            texts = [t for s, t in _RUN['overlap'] if ap.in_span(s, arm)]
            if not mine or not texts:
                continue
            facts = _RUN['regroup'].setdefault(arm['label'], dict(faults=[], orphan_rows=0, moved_rows=0))
            owner, faults = overlap_owners(texts)
            txns = [(f['identity'] or {}).get('txn') for f in mine]
            if any(type(t) is not int for t in txns) or len(set(txns)) != len(txns):
                faults = faults + ['engine-flushes-without-one-transaction-each']
            facts['faults'] = sorted(set(faults))
            if faults:
                continue
            here = {seq: txn for seq, txn in owner.items() if txn in txns}
            rows = [(f['identity']['txn'], r) for f in mine for r in f['rows'].get('n', [])]
            facts['orphan_rows'] = sum(1 for txn, r in rows if r['seq'] in owner and r['seq'] not in here)
            facts['moved_rows'] = sum(1 for txn, r in rows if here.get(r['seq'], txn) != txn)
            chunks = {c['identity']['txn']: c for c in regroup(mine, here, {})}
            for f in mine:
                replaced[id(f)] = dict(f, rows=dict(f['rows'], n=chunks[f['identity']['txn']]['rows'].get('n', [])))
        return [replaced.get(id(f), f) for f in flushes], problems
    return assemble


def breakdown(original):
    """The chunk breakdown with the chunk's staged segment rows (for the in-flight check; popped by settle_arm)."""
    def run(chunk, workers):
        info = original(chunk, workers)
        info['_segments'] = [_segment(r, r['stage']) for r in chunk['rows'].get('n', [])
                             if r.get('kind') == 'segment' and r.get('stage') in TARGETS and type(r.get('t_submit')) is int]
        return info
    return run


def settle_arm(arm, row, ledger, torn, last_line, seqs_by_line):
    """Step A's torn-line settle (alpha-next's), then the overlap facts: regroup faults, torn stage-overlap lines, a lost engine
    flush on a C1-on arm, and the in-flight check."""
    tears = TORN_SETTLE(arm, row, ledger, torn, last_line, seqs_by_line)
    name = arm['label']
    chunks = row.get('chunks') or []
    facts = _RUN['regroup'].get(name) or {}
    row['reasons'] += ['overlap-regroup:' + x for x in facts.get('faults', ())]
    if facts.get('orphan_rows'):
        row['reasons'].append('overlap-rows-of-a-chunk-without-its-flush')
    overlap = _overlap_arm(arm)
    if overlap and any(x['side'] == 'engine' for x in tears['lost_flushes']):
        row['reasons'].append('overlap-arm-engine-flush-lost')
    torn_overlap = [t for s, t in _RUN['torn_overlap'] if ap.in_span(s, arm)]
    if torn_overlap and overlap and chunks:
        logged = {int(OVERLAP_LINE.search(t).group(1)) for s, t in _RUN['overlap'] if ap.in_span(s, arm)}
        ordered = sorted(chunks, key=lambda c: c.get('chunk_start') or 0)
        without = [i for i, c in enumerate(ordered) if c.get('txn') not in logged]
        if len(without) <= len(torn_overlap):
            for i in without:
                for chunk in ordered[max(0, i - 1):i + 1]:
                    chunk.setdefault('dropped_out', []).append('overlap-line-lost')
            torn_lines.recount(row)
    _RUN['in_flight'][name] = in_flight([s for c in chunks for s in c.pop('_segments', ())])
    records = tears['lost_records'] + len(torn_overlap)
    starts = sorted(set(tears['dropped_chunk_starts']) | {c['chunk_start'] for c in chunks if c.get('dropped_out')})
    row['reasons'] = [x for x in row['reasons'] if not x.startswith(('lost-records-above', 'dropped-chunks-above'))]
    if records > torn_lines.LOST_RECORD_LIMIT:
        row['reasons'].append('lost-records-above-the-declared-limit:%d>%d' % (records, torn_lines.LOST_RECORD_LIMIT))
    if len(starts) > torn_lines.DROPPED_CHUNK_LIMIT:
        row['reasons'].append('dropped-chunks-above-the-declared-limit:%d>%d' % (len(starts), torn_lines.DROPPED_CHUNK_LIMIT))
    tears.update(torn_overlap_lines=len(torn_overlap), lost_records=records, dropped_chunk_starts=starts,
                 within_limits=records <= torn_lines.LOST_RECORD_LIMIT and len(starts) <= torn_lines.DROPPED_CHUNK_LIMIT)
    return tears


def engagement(arm, row, lines, declared):
    """alpha-next's engagement (Step B's group rule and C0's local rule, on the regrouped chunks), then C1's both ways."""
    # A chunk dropped for a lost stage-overlap line holds rows its successor owns (or lacks its early rows): it is a lost record,
    # so alpha-next's per-chunk rules (Step B's layer commands per chunk) read the chunks still kept, as the torn-line rule does.
    if row and row.get('chunks'):
        row = dict(row, chunks=[c for c in row['chunks'] if 'overlap-line-lost' not in (c.get('dropped_out') or ())])
    out = AN_ENGAGEMENT(arm, row, lines, declared)
    overlap = _overlap_arm(arm)
    delta = arm.get('stage_delta') or {}
    counters = {k: delta.get('overlap_' + k) for k in OVERLAP_COUNTERS}
    logged = sum(1 for s, _ in _RUN['overlap'] if ap.in_span(s, arm))
    lost = sum(1 for s, _ in _RUN['torn_overlap'] if ap.in_span(s, arm))
    kept = [c for c in ((row or {}).get('chunks') or []) if not c.get('dropped_out')]
    uneven = [c['chunk_start'] for c in kept if (c.get('kinds') or {}).get('open') != 2 or (c.get('kinds') or {}).get('release') != 2]
    owned = (arm.get('progress_delta') or {}).get('chunks')
    reasons = []
    if any(type(v) is not int for v in counters.values()):
        reasons.append('overlap-counters-missing')
    elif overlap:
        if counters['successor_frames'] < 1 or counters['overlapped_commits'] < 1:
            reasons.append('overlap-ran-no-second-chunk')
        if counters['early_opens'] < 1:
            reasons.append('overlap-stage-zero-never-opened-a-successor-early')
    elif any(counters.values()):
        reasons.append('c1-off-arm-moved-an-overlap-counter')
    if overlap:
        if logged + lost != owned:
            reasons.append('overlap-lines-differ-from-the-auto-owned-chunks')
        if uneven:
            reasons.append('regrouped-chunk-without-two-opens-and-two-releases')
    elif logged or lost:
        reasons.append('c1-off-arm-logged-an-overlap-line')
    defer = (T.by_label().get(arm.get('label')) or {}).get('defer_bind') is True
    defer_counters = {k: delta.get('defer_' + k) for k in DEFER_COUNTERS}
    if any(type(v) is not int for v in defer_counters.values()):
        reasons.append('defer-bind-counters-missing')
    elif defer and defer_counters['deferred_binds'] < 1:
        reasons.append('defer-bind-arm-never-deferred-a-bind')          # coverage not reached
    elif not defer and any(defer_counters.values()):
        reasons.append('defer-bind-off-arm-moved-a-defer-counter')
    c1 = dict(overlap=overlap, defer_bind=defer, defer_counters=defer_counters, engaged=not reasons, reasons=reasons,
              counters=counters, overlap_lines=logged,
              lost_overlap_lines=lost, regrouped_chunks_checked=len(kept) if overlap else None,
              regrouped_chunks_uneven=uneven if overlap else None)
    every = list(out['reasons']) + reasons
    return dict(out, engaged=not every, reasons=every, c1=c1)


def arm_verdict(arm, ingress, watch, view, signatures, progress_reasons, engaged):
    name = arm.get('label')
    extra = {'oom'} if _RUN['oom'].get(name) else set()
    for stage, cell in (_RUN['in_flight'].get(name) or {}).items():
        if cell['breaches']:
            extra.add('two-commands-in-flight-on-stage-' + stage)
    return SA_ARM_VERDICT(arm, ingress, watch, view, set(signatures) | extra, progress_reasons, engaged)


# --- memory-peak per chunk against the matched C1-off arm ------------------------------------------------

def chunk_peaks(lines):
    """chunk_start -> worst rank's peak allocated and reserved (MiB) from one arm's memory-peak receipts."""
    out = {}
    for x in lines['peaks']:
        cell = out.setdefault(x['chunk_start'], dict(chunk_start=x['chunk_start'], ranks=0, peak_allocated_mib=None,
                                                     peak_reserved_mib=None, worst_rank=None))
        cell['ranks'] += 1
        allocated = round(x['peak_allocated'] / 2 ** 20, 1)
        if cell['peak_allocated_mib'] is None or allocated > cell['peak_allocated_mib']:
            cell['peak_allocated_mib'], cell['worst_rank'] = allocated, x['rank']
        reserved = round(x['peak_reserved'] / 2 ** 20, 1)
        cell['peak_reserved_mib'] = reserved if cell['peak_reserved_mib'] is None else max(cell['peak_reserved_mib'], reserved)
    return out


def peak_compare(lines, rows):
    """Per C1-on arm: per chunk the worst rank's memory-peak, and for a graded arm its difference from the matched C1-off arm by
    chunk start; extremes over all chunks and over the chunks past the 97,152-row arena. A read, never a verdict input."""
    out = {}
    for row in rows:
        if not row['overlap']:
            continue
        mine = chunk_peaks(lines.get(row['label']) or an.EMPTY_LINES)
        partner = T.matched_off(row)
        cell = dict(against=partner, chunks=len(mine), torn=(lines.get(row['label']) or an.EMPTY_LINES)['torn_peaks'],
                    max_peak_allocated_mib=max((c['peak_allocated_mib'] for c in mine.values()), default=None),
                    reported_only=partner is None)
        if partner is not None:
            theirs = chunk_peaks(lines.get(partner) or an.EMPTY_LINES)
            per_chunk = [dict(chunk_start=s, allocated_delta_mib=round(c['peak_allocated_mib'] - theirs[s]['peak_allocated_mib'], 1),
                              reserved_delta_mib=round(c['peak_reserved_mib'] - theirs[s]['peak_reserved_mib'], 1),
                              worst_rank=c['worst_rank'], off_worst_rank=theirs[s]['worst_rank'])
                         for s, c in sorted(mine.items()) if s in theirs]

            def extremes(cells):
                return dict(chunks=len(cells), max_allocated_delta_mib=max((x['allocated_delta_mib'] for x in cells), default=None),
                            max_reserved_delta_mib=max((x['reserved_delta_mib'] for x in cells), default=None))
            cell.update(per_chunk=per_chunk, unmatched_chunks=sorted(set(mine) ^ set(theirs)),
                        off_max_peak_allocated_mib=max((c['peak_allocated_mib'] for c in theirs.values()), default=None),
                        below_arena=extremes([x for x in per_chunk if x['chunk_start'] < ARENA_ROWS]),
                        past_arena=extremes([x for x in per_chunk if x['chunk_start'] >= ARENA_ROWS]))
        out[row['label']] = cell
    return dict(arms=out, arena_rows=ARENA_ROWS, is_gate=False)


def stage_margin(table, lines, rows):
    """Per C1-on arm and stage (stage 0: ranks 0 and 1, stage 1: ranks 2 and 3): the lowest free MiB over the stage's memory
    receipts and the highest memory-peak allocated, against the matched C1-off arm (the g4 tail: reported only). Stage 1 holds
    the least free memory on the box (C1's risk), so its margin is read on its own. A read, never a verdict input."""
    def per_stage(name):
        ranks = (((table.get(name) or {}).get('memory') or {}).get('ranks')) or {}
        peaks = chunk_peaks_by_rank(lines.get(name) or an.EMPTY_LINES)
        out = {}
        for stage, own in TARGETS.items():
            free = [ranks[str(r)]['min_free_mib'] for r in own if (ranks.get(str(r)) or {}).get('min_free_mib') is not None]
            peak = [peaks[r] for r in own if r in peaks]
            out[str(stage)] = dict(min_free_mib=min(free) if len(free) == len(own) else None,
                                   max_peak_allocated_mib=max(peak) if peak else None, ranks_reporting=len(free))
        return out
    out = {}
    for row in rows:
        if not row['overlap']:
            continue
        partner = T.matched_off(row)
        mine = per_stage(row['label'])
        cell = dict(against=partner, stages=mine, reported_only=partner is None)
        if partner is not None:
            theirs = per_stage(partner)
            cell['off_stages'] = theirs
            for stage in mine:
                a, b_ = mine[stage], theirs[stage]
                mine[stage]['min_free_below_off_mib'] = (None if a['min_free_mib'] is None or b_['min_free_mib'] is None
                                                         else round(b_['min_free_mib'] - a['min_free_mib'], 1))
                mine[stage]['peak_allocated_over_off_mib'] = (
                    None if a['max_peak_allocated_mib'] is None or b_['max_peak_allocated_mib'] is None
                    else round(a['max_peak_allocated_mib'] - b_['max_peak_allocated_mib'], 1))
        out[row['label']] = cell
    return dict(arms=out, stage_ranks={str(k): list(v) for k, v in TARGETS.items()}, is_gate=False)


def chunk_peaks_by_rank(lines):
    """rank -> the highest memory-peak allocated (MiB) over one arm's receipts."""
    out = {}
    for x in lines['peaks']:
        value = round(x['peak_allocated'] / 2 ** 20, 1)
        out[x['rank']] = max(out.get(x['rank'], value), value)
    return out


# --- defer_bind: the value the engine acknowledged ------------------------------------------------------

def defer_bind_check(arm, row):
    """(reasons, classification) of an arm whose defer_bind is not proved to be the table's: the arm's record must carry what it
    sent and what the engine acknowledged (HARNESS when it does not, or when it sent another value than the table's); the engine's
    reply and its status before and after the fresh prompt must show the requested value, with the deferred bind installed (ENGINE
    when they do not). An arm whose stage command never got a reply is left to alpha-next's rules (it is incomplete already)."""
    want = row['defer_bind']
    rec = arm.get('defer_bind')
    if type(rec) is not dict or not {'requested', 'acknowledged', 'replied'} <= set(rec):
        return ['arm-record-without-the-acknowledged-defer-bind'], 'HARNESS'
    if rec['requested'] is not want:
        return ['arm-requested-another-defer-bind-than-the-table'], 'HARNESS'
    if rec['replied'] is not True:
        return [], None
    reasons = []
    if rec['acknowledged'] is not want:
        reasons.append('defer-bind-acknowledged-differs-from-requested')
    views = {v.get('label'): (v.get(STAGE_CONTROL) or {}) for v in arm.get('progress') or [] if type(v) is dict}
    for point in ('before-fresh', 'after-fresh'):
        view = views.get(point)
        if not view:
            continue                      # alpha-next's own reasons cover a missing status view
        if view.get('defer_bind_installed') is not True:
            reasons.append('defer-bind-not-installed-at-' + point)
        elif view.get('defer_bind_selected') is not want:
            reasons.append('defer-bind-status-differs-from-requested-at-' + point)
    return reasons, 'ENGINE' if reasons else None


# --- the overlap observation (c1_overlap_read), a read ---------------------------------------------------

def observe_rows(logs):
    """(stamp, text, body) per whole ``flash-s5 overlap-observe`` line, and (stamp, text) per line whose JSON body does not parse
    or names another schema (torn or foreign: kept out of the read, counted)."""
    whole, torn = [], []
    for stamp, text in ap.read_stamped(logs, lambda line: observe_read.LINE in line):
        at = text.find(observe_read.LINE)
        try:
            body = json.loads(text[at + len(observe_read.LINE):])
        except ValueError:
            torn.append((stamp, text))
            continue
        if type(body) is not dict or body.get('schema') != observe_read.SCHEMA or type(body.get('txn')) is not int:
            torn.append((stamp, text))
            continue
        whole.append((stamp, text, body))
    return whole, torn


def observe_join(whole, seqs):
    """txn -> arm label by sequence: the chunk's part 0 names its sequences (``seqs``, the 'n' rows' ``seq``), joined to the arm
    whose engine rows hold them (c1_overlap_read's join; the row schema is unchanged). A chunk no arm or two arms hold is not
    joined (counted)."""
    owner, unjoined, ambiguous = {}, [], []
    for _stamp, _text, body in whole:
        if body.get('part') != 0:
            continue
        mine = observe_read.expand_ranges(body.get('seqs'))
        arms = sorted(name for name, held in seqs.items() if mine & held)
        if len(arms) == 1:
            owner[body['txn']] = arms[0]
        else:
            (ambiguous if arms else unjoined).append(body['txn'])
    return owner, sorted(set(unjoined)), sorted(set(ambiguous))


def bounds_summary(chunks):
    """Per op kind over one arm's observed ops: how many, how many with a lower bound (``lo_ns``: a submit seen while the op was
    pending), an upper bound (``hi_ns``: the first apply listing it complete) and both, and the width hi - lo (ns) where both."""
    kinds = {}
    for chunk in chunks.values():
        for row in chunk['ops'].values():
            cell = kinds.setdefault(str(row.get('kind')), dict(ops=0, lo=0, hi=0, both=0, widths=[]))
            cell['ops'] += 1
            lo, hi = row.get('lo_ns'), row.get('hi_ns')
            cell['lo'] += lo is not None
            cell['hi'] += hi is not None
            if lo is not None and hi is not None:
                cell['both'] += 1
                cell['widths'].append(hi - lo)
    out = {}
    for kind, cell in sorted(kinds.items()):
        widths = sorted(cell.pop('widths'))
        cell.update(width_ns_median=widths[len(widths) // 2] if widths else None, width_ns_max=widths[-1] if widths else None,
                    width_negative=sum(1 for w in widths if w < 0))
        out[kind] = cell
    return out


OBSERVE_ROLE = {'off': 'no successor expected (C1 off)', 'on': 'control: NONE expected (C1 without the deferred bind)',
                'defer': 'measurement (C1 with the deferred bind)'}


def observe_read_arms(logs, seqs, rows, arms):
    """Per declared arm present: c1_overlap_read over the observe lines joined to it by sequence (OVERLAP or NONE per successor,
    with the reason), the per-op device bounds summary and the observe log bytes. A read, never a verdict input; plain C1-on arms
    are the control (NONE expected), reported as a control, never gated."""
    whole, torn = observe_rows(logs)
    owner, unjoined, ambiguous = observe_join(whole, seqs)
    known = {a.get('label'): a for a in arms if a.get('label')}
    out = {}
    for row in rows:
        name = row['label']
        if name not in known:
            continue
        mine = [(text, body) for _stamp, text, body in whole if owner.get(body['txn']) == name]
        lines = [text for text, _body in mine]
        cell = dict(c1=row['c1'], defer_bind=row['defer_bind'], role=OBSERVE_ROLE[row['c1']], observe_lines=len(lines),
                    observe_bytes=sum(len(t.encode()) for t in lines), chunks=len({b['txn'] for _t, b in mine}),
                    torn_or_foreign_lines=sum(1 for stamp, _t in torn if ap.in_span(stamp, known[name])))
        try:
            read = observe_read.overlap_read(lines)
            cell.update(successors=read['successors'], overlap=read['overlap'], none=read['none'],
                        rows=[{k: r.get(k) for k in ('txn', 'older', 'verdict', 'reason', 'margin_ns', 'early', 'deferred_bind')}
                              for r in read['rows']],
                        bounds=bounds_summary(observe_read.parse(lines)), error=None)
        except (ValueError, KeyError, TypeError) as exc:
            cell.update(successors=None, overlap=None, none=None, rows=[], bounds={}, error='%s: %s' % (type(exc).__name__, exc))
        if row['c1'] == 'on':
            cell['control_as_expected'] = None if cell['successors'] is None else cell['overlap'] == 0
        out[name] = cell
    return dict(arms=out, schema=observe_read.SCHEMA, unjoined_chunks=unjoined, ambiguous_chunks=ambiguous,
                torn_or_foreign_lines=len(torn), observe_bytes=sum(len(t.encode()) for _s, t, _b in whole), is_gate=False,
                rule='c1_overlap_read: OVERLAP when a stage-0 layer of the successor completed on every stage-0 rank before '
                     'stage 1\'s finalize of the older chunk provably completed (host monotonic bounds); else NONE with the '
                     'reason. Chunks join arms by sequence.')


# --- the numbers beside C0/C1 (a read) -----------------------------------------------------------------

def arm_numbers(result, rows):
    """Per declared arm present: live and solo prefill tok/s and TTFT, the live decode ratio, the episode loss
    TTFT_live x (1 - prefill_live / prefill_solo) (the live arm against the solo arm of the same selection and round: the seconds
    the live episode lost to the standing decoders, = TTFT_live - TTFT_solo at equal tokens), stage 0 and stage 1 s/chunk on the
    slower TP rank (Step B's slow_rank: the median over counted chunks of the slower own rank's stage time), per rank min free,
    max reserved, alloc retries and OOMs (stage 0 rank 0 apart), and alpha-next's memory read (MET / NOT_MET / INC against the
    matched C1-off arm, unchanged). Ratios against the matched C1-off arm and, for a C1-defer arm, its matched C1-on arm. Never a
    verdict input."""
    table, slow = result['table'], ((result.get('slow_rank') or {}).get('arms') or {})
    memory = result.get('memory_read') or {}

    def ratio(a, b_):
        return None if a is None or not b_ else round(a / b_, 4)

    out = {}
    for row in rows:
        name = row['label']
        cell = table.get(name)
        if cell is None:
            continue
        ranks = (cell.get('memory') or {}).get('ranks') or {}
        per_rank = {r: dict(min_free_mib=v.get('min_free_mib'), max_reserved_mib=v.get('peak_reserved_mib'),
                            alloc_retries=v.get('alloc_retries'), ooms=v.get('ooms')) for r, v in sorted(ranks.items())}
        stages = (slow.get(name) or {}).get('all') or {}
        solo = T.label(row['c1'], row['size'], 'solo', row['index']) if row['cohort'] == 'live' else None
        solo_cell = table.get(solo) if solo else None
        loss = None
        if solo_cell and cell.get('ttft_s') is not None and cell.get('prefill_tok_s') and solo_cell.get('prefill_tok_s'):
            loss = round(cell['ttft_s'] * (1 - cell['prefill_tok_s'] / solo_cell['prefill_tok_s']), 3)
        mem = memory.get(name) or {}
        item = dict(c1=row['c1'], defer_bind=row['defer_bind'], cohort=row['cohort'], band=row['band'], part=row['part'],
                    prefill_tok_s=cell.get('prefill_tok_s'), ttft_s=cell.get('ttft_s'),
                    decode_ratio=cell.get('decode_ratio') if row['cohort'] == 'live' else None,
                    episode_loss_s=loss, solo_partner=solo if solo_cell else None,
                    stage_s_per_chunk_slow_rank={st: ((stages.get(st) or {}).get('own_s') or {}).get('slow_rank_median')
                                                 for st in ('0', '1')},
                    ranks=per_rank, stage0_rank0=per_rank.get('0'),
                    memory_read=dict(verdict=mem.get('verdict'), against=mem.get('against'),
                                     min_free_below_off_mib=mem.get('min_free_below_c0_off_mib')))
        for key, partner in (('vs_c1_off', T.matched_off(row) if row['c1'] != 'off' else None),
                             ('vs_c1_on', T.matched_on(row))):
            other = table.get(partner) if partner else None
            if other is not None:
                item[key] = dict(against=partner, prefill_ratio=ratio(cell.get('prefill_tok_s'), other.get('prefill_tok_s')),
                                 ttft_ratio=ratio(cell.get('ttft_s'), other.get('ttft_s')),
                                 decode_ratio_delta=(None if item['decode_ratio'] is None or other.get('decode_ratio') is None
                                                     else round(item['decode_ratio'] - other['decode_ratio'], 4)))
        out[name] = item
    for name, item in out.items():
        for key in ('vs_c1_off', 'vs_c1_on'):
            partner = (item.get(key) or {}).get('against')
            if partner in out and item['episode_loss_s'] is not None and out[partner]['episode_loss_s'] is not None:
                item[key]['episode_loss_delta_s'] = round(item['episode_loss_s'] - out[partner]['episode_loss_s'], 3)
    return dict(arms=out, is_gate=False, episode_loss_rule='TTFT_live x (1 - prefill_tok_s_live / prefill_tok_s_solo), the live '
                'and solo arms of the same C1 selection, size and round', memory_against='the matched C1-off arm (C0 local; '
                'this queue has no serial arm), alpha-next\'s memory read unchanged')


# --- the grade ----------------------------------------------------------------------------------------

def grade(arms, ingress, checks, logs, watches=None, part='core'):
    """alpha-next's grade over this table with the C1 names bound; then the run-level OOM and the C1 record fields."""
    known = [a for a in arms if a.get('label')]
    intact, torn_overlap = overlap_rows(logs)
    by_arm, outside = oom_rises(logs, known)
    _RUN.clear()
    _RUN.update(arms=known, overlap=intact, torn_overlap=torn_overlap, oom=by_arm, regroup={}, in_flight={}, seqs={})
    view = table_view()
    try:
        with _bound(an, dict(T=view, engagement=engagement)), \
                _bound(sa, dict(SIGNATURES=sa.SIGNATURES + OVERLAP_SIGNATURES, arm_verdict=arm_verdict)), \
                _bound(torn_lines, dict(settle_arm=settle_arm)), \
                _bound(progress, dict(assemble=regrouping(progress.assemble), chunk_breakdown=breakdown(progress.chunk_breakdown))):
            result = an.grade(arms, ingress, checks, logs, watches, part=part)
            lines, _ = an.arm_lines(arms, logs)
        in_flight_rows, regroup_facts, seqs = dict(_RUN['in_flight']), dict(_RUN['regroup']), dict(_RUN['seqs'])
    finally:
        _RUN.clear()
    present = {a.get('label') for a in arms} | set(watches or {})
    rows = [r for r in T.arms(None if part == 'all' else T.GRADED) if r['label'] in present or r['part'] in T.GRADED]
    failures, missing = set(result['failures']), set(result['missing'])
    classes = set(result['classification'] or ())
    if outside:
        failures.add('oom-outside-every-arm-span')
        classes.add('ENGINE')
    for name, v in result['validity'].items():
        v['oom_rises'] = by_arm.get(name, [])
        v['in_flight'] = in_flight_rows.get(name)
        v['regroup'] = regroup_facts.get(name)
        extra = {'oom'} if by_arm.get(name) else set()
        extra |= {'two-commands-in-flight-on-stage-' + s for s, c in (v['in_flight'] or {}).items() if c['breaches']}
        v['signatures'] = sorted(set(v.get('signatures') or ()) | extra)
    declared = T.by_label()
    records = {a.get('label'): a for a in known}
    for name, v in result['validity'].items():
        row, arm = declared.get(name), records.get(name)
        if row is None or arm is None:
            continue
        v['defer_bind'] = dict(arm.get('defer_bind') or {}, table=row['defer_bind'])
        reasons, cls = defer_bind_check(arm, row)
        if reasons:
            v['reasons'] = list(v.get('reasons') or []) + reasons
            v['verdict'], v['classification'] = 'FAIL', cls
            failures.update(name + ':' + x for x in reasons)
            classes.add(cls)
    for name, cell in result['table'].items():
        row = declared.get(name)
        if row:
            cell.update(c0='on', c1=row['c1'], overlap=row['overlap'], defer_bind=row['defer_bind'], stage_local=True)
    for row in result['arm_table']['arms']:
        row.update(c0='on')
    verdict = 'FAIL' if failures else 'INC' if missing else 'PASS'
    if verdict != 'FAIL':
        classes = set()
    elif not classes:
        classes = {'HARNESS'}
    result.update(schema=GRADE_SCHEMA if part == 'core' else READ_SCHEMA, verdict=verdict, classification=sorted(classes) or None,
                  failures=sorted(failures), missing=sorted(missing), oom_outside_every_arm=outside, oom_is_gate=True,
                  memory_peak_vs_c1_off=peak_compare(lines, [r for r in rows if r['label'] in result['table']]),
                  memory_stage_vs_c1_off=stage_margin(result['table'], lines, [r for r in rows if r['label'] in result['table']]),
                  in_flight_is_gate=True, declared_limits=LIMITS, axis=AXIS, defer_bind_is_gate=True,
                  overlap_observe=observe_read_arms(logs, seqs, [r for r in rows if r['label'] in result['table']], known),
                  overlap_observe_is_gate=False,
                  arm_numbers=arm_numbers(result, [r for r in rows if r['label'] in result['table']]),
                  basis='one engine, fused and C0 stage-local selected in every arm, one group size, one C1 selection and one '
                        'defer_bind per arm by typed control; band 1 ABBA at 65,445 tokens over two rounds, band 2 matched at 131,094 past the '
                        '97,152-row arena; per chunk, the critical rank of each native command, rows regrouped by chunk on C1-on '
                        'arms; live arms count chunks begun with all 8 decoders, solo arms chunks begun with none',
                  scope='C1 (two chunks in flight, one command in flight per stage), with and without the deferred PP bind, '
                        'against C0 local on the composed alpha-next engine, with the overlap observation, speed, decode and memory reported and never gated; gains from separate experiments are '
                        'never multiplied')
    if part != 'core':
        result['read_only'] = ('a read over every arm, g4 tail included; the queue verdict is the core grade '
                               '(alpha-next-c1-grade.json)')
    arm_table = result['arm_table']
    arm_table.update(extra_not_entered=[], axis='c1')
    return result


# --- the report ----------------------------------------------------------------------------------------

_cell = an._cell


def report(result):
    """alpha-next's sections (title and C0/C1 names) plus C1 engagement, one command in flight per stage, OOM and memory-peak."""
    with _bound(an, dict(T=table_view())):
        text = AN_REPORT(result)
    text = text.replace('# q18a-alpha-next report', '# q18a-alpha-next-c1 report', 1)
    text = text.replace('# q18a-alpha-next read', '# q18a-alpha-next-c1 read', 1)
    text = text.replace('C0-on', 'C1-on').replace('C0-off', 'C1-off')
    head, sep, tail = text.partition('\n## Performance\n')
    lines = ['', 'Axis: ' + result['axis'] + '.', '',
             '### C1 engagement and one command in flight per stage', '',
             '| arm | C1 | engaged | successor frames | early opens | overlapped commits | overlap lines | lost overlap lines | '
             'uneven chunks | moved rows | stage 0 commands / breaches | stage 1 commands / breaches | OOM rises | reasons |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for name in result['required_arms']:
        v = result['validity'].get(name) or {}
        c1 = (v.get('engagement') or {}).get('c1') or {}
        o = c1.get('counters') or {}
        f = v.get('in_flight') or {}
        lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            name, (T.by_label().get(name) or {}).get('c1', '-'), c1.get('engaged'), _cell(o.get('successor_frames')),
            _cell(o.get('early_opens')), _cell(o.get('overlapped_commits')), _cell(c1.get('overlap_lines')),
            _cell(c1.get('lost_overlap_lines')), ','.join(str(x) for x in c1.get('regrouped_chunks_uneven') or []) or '0',
            _cell((v.get('regroup') or {}).get('moved_rows')),
            *('%s / %d' % (f[s]['commands'], len(f[s]['breaches'])) if s in f else '-' for s in ('0', '1')),
            len(v.get('oom_rises') or []), '; '.join(c1.get('reasons') or []) or '-'))
    if result['oom_outside_every_arm']:
        lines += ['', 'OOM outside every arm span: %d rises (FAIL ENGINE).' % len(result['oom_outside_every_arm'])]
    lines += ['', 'Declared limits: ' + '; '.join('%s: %s' % kv for kv in sorted(result['declared_limits'].items())) + '.', '']
    peaks = result['memory_peak_vs_c1_off']
    lines += ['### Memory-peak per chunk, C1 on against the matched C1-off arm (a read)', '',
              'Worst rank per chunk, by chunk start; past the %s-row arena apart. The g4 tail is reported only.' % format(
                  peaks['arena_rows'], ','), '',
              '| arm | against | chunks | max peak alloc MiB | C1-off max | below arena: max alloc delta | max reserved delta | '
              'past arena: max alloc delta | max reserved delta | unmatched chunks |', '|---|---|---|---|---|---|---|---|---|---|']
    for name, cell in peaks['arms'].items():
        lo, hi = cell.get('below_arena') or {}, cell.get('past_arena') or {}
        lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            name, cell['against'] or 'reported only', cell['chunks'], _cell(cell['max_peak_allocated_mib']),
            _cell(cell.get('off_max_peak_allocated_mib')), _cell(lo.get('max_allocated_delta_mib')),
            _cell(lo.get('max_reserved_delta_mib')), _cell(hi.get('max_allocated_delta_mib')), _cell(hi.get('max_reserved_delta_mib')),
            ','.join(str(x) for x in cell.get('unmatched_chunks') or []) or '0'))
    margin = result['memory_stage_vs_c1_off']
    lines += ['', '### Memory margin per stage, C1 on against the matched C1-off arm (a read)', '',
              'Stage 0: ranks 0 and 1; stage 1: ranks 2 and 3 (the least free memory on the box). Lowest free MiB over the stage\'s '
              'memory receipts and highest memory-peak allocated MiB.', '',
              '| arm | against | stage 0 min free | below off | stage 0 peak | over off | stage 1 min free | below off | stage 1 peak | '
              'over off |', '|---|---|---|---|---|---|---|---|---|---|']
    for name, cell in margin['arms'].items():
        s0, s1 = cell['stages']['0'], cell['stages']['1']
        lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            name, cell['against'] or 'reported only', _cell(s0['min_free_mib']), _cell(s0.get('min_free_below_off_mib')),
            _cell(s0['max_peak_allocated_mib']), _cell(s0.get('peak_allocated_over_off_mib')), _cell(s1['min_free_mib']),
            _cell(s1.get('min_free_below_off_mib')), _cell(s1['max_peak_allocated_mib']), _cell(s1.get('peak_allocated_over_off_mib'))))
    lines += observe_report(result) + numbers_report(result)
    return head + '\n'.join(lines) + sep + tail


def observe_report(result):
    obs = result['overlap_observe']
    lines = ['', '### Overlap observation per arm (c1_overlap_read, a read)', '',
             'OVERLAP when a stage-0 layer of the successor completed on every stage-0 rank before stage 1\'s finalize of the older '
             'chunk provably completed; else NONE. C1-on arms are the control (NONE expected), C1-defer arms the measurement. '
             'Chunks joined to arms by sequence: unjoined %d, joined to two arms %d; torn or foreign lines %d; observe bytes %d.' % (
                 len(obs['unjoined_chunks']), len(obs['ambiguous_chunks']), obs['torn_or_foreign_lines'], obs['observe_bytes']), '',
             '| arm | role | successors | OVERLAP | NONE | control as expected | NONE reasons | layer ops lo/hi/both | finalize ops '
             'lo/hi/both | layer width median ns | observe lines | observe bytes | error |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for name, cell in obs['arms'].items():
        reasons = {}
        for r in cell['rows']:
            if r['verdict'] == 'NONE':
                reasons[r['reason']] = reasons.get(r['reason'], 0) + 1
        layer, fin = cell['bounds'].get('layer') or {}, cell['bounds'].get('state_finalize') or {}
        lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            name, cell['role'], _cell(cell['successors']), _cell(cell['overlap']), _cell(cell['none']),
            cell.get('control_as_expected', '-'), '; '.join('%s x%d' % kv for kv in sorted(reasons.items())) or '-',
            '/'.join(str(layer.get(k, 0)) for k in ('lo', 'hi', 'both')), '/'.join(str(fin.get(k, 0)) for k in ('lo', 'hi', 'both')),
            _cell(layer.get('width_ns_median')), cell['observe_lines'], cell['observe_bytes'], cell['error'] or '-'))
    return lines


def numbers_report(result):
    nums = result['arm_numbers']
    lines = ['', '### Beside the C0/C1 numbers, per arm (a read)', '',
             'Episode loss: %s. Memory read against %s.' % (nums['episode_loss_rule'], nums['memory_against']), '',
             '| arm | c1 | defer | prefill tok/s | TTFT s | decode ratio | episode loss s | stage 0 s/chunk slow rank | '
             'stage 1 s/chunk slow rank | rank 0 (stage 0) min free / max reserved / retries / OOMs | min free per rank | '
             'memory read | vs C1 off: prefill ratio | vs C1 on: prefill ratio / episode loss delta s |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for name, n in nums['arms'].items():
        r0 = n['stage0_rank0'] or {}
        off, on = n.get('vs_c1_off') or {}, n.get('vs_c1_on') or {}
        lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s / %s / %s / %s | %s | %s | %s | %s |' % (
            name, n['c1'], n['defer_bind'], _cell(n['prefill_tok_s']), _cell(n['ttft_s'], 3), _cell(n['decode_ratio'], 4),
            _cell(n['episode_loss_s'], 3), _cell(n['stage_s_per_chunk_slow_rank']['0'], 4),
            _cell(n['stage_s_per_chunk_slow_rank']['1'], 4), _cell(r0.get('min_free_mib')), _cell(r0.get('max_reserved_mib')),
            _cell(r0.get('alloc_retries')), _cell(r0.get('ooms')),
            ','.join('%s:%s' % (r, _cell(v['min_free_mib'])) for r, v in n['ranks'].items()) or '-',
            '%s (%s)' % (n['memory_read']['verdict'], n['memory_read']['against'] or '-'), _cell(off.get('prefill_ratio'), 4),
            '%s / %s' % (_cell(on.get('prefill_ratio'), 4), _cell(on.get('episode_loss_delta_s'), 3)) if on else '-'))
    return lines


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='action', required=True)
    r = sub.add_parser('arm')
    r.add_argument('--url', required=True)
    r.add_argument('--model', required=True)
    r.add_argument('--run-id', required=True)
    r.add_argument('--epoch', type=int, required=True)
    r.add_argument('--c1', choices=tuple(T.C1), required=True,
                   help='this arm\'s C1 selection: on (overlap), defer (overlap with the deferred PP bind) or off (C0 local)')
    r.add_argument('--size', type=int, choices=T.SIZES, required=True, help='this arm\'s layer group size; fused and burst in every arm')
    r.add_argument('--cohort', choices=progress.COHORTS, required=True)
    r.add_argument('--index', type=int, choices=T.INDEXES, required=True)
    r.add_argument('--workload', required=True)
    r.add_argument('--control-timeout-s', type=float, default=120.0)
    r.add_argument('--out', required=True)
    for action, text in (('grade', 'the core grade: the queue verdict'), ('read', 'every arm, g4 tail included: a read only')):
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
    arms = [base.load(x) or {'label': None, 'complete': False, 'reasons': ['arm-artifact-unreadable:' + Path(x).name]}
            for x in a.arm]
    result = grade(arms, ap.load_pairs(a.ingress), ap.load_pairs(a.check), a.log, ap.load_pairs(a.watch),
                   part='core' if a.action == 'grade' else 'all')
    with open(a.out, 'x') as handle:
        json.dump(result, handle, sort_keys=True)
    if a.report:
        with open(a.report, 'x') as handle:
            handle.write(report(result))
    print('ALPHA_NEXT_C1_%s %s' % ('GRADE' if a.action == 'grade' else 'READ', result['verdict']))
    if a.action == 'read':
        return 0 if result['verdict'] == 'PASS' else 2      # a read: never 1, the queue does not count it
    return {'PASS': 0, 'FAIL': 1}.get(result['verdict'], 2)


if __name__ == '__main__':
    raise SystemExit(main())
