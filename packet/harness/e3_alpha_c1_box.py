#!/usr/bin/env python3
"""C1 box arms and grade: two chunks in flight (stage overlap) against the frozen serial path and C0's local path, per arm on one engine.

Derived from e3_alpha_c0_box (imported as ``c0``, unchanged): every name not defined here is C0's. The arm and the grade run C0's
code with the names below bound in its place for the duration of one call (``_bound``), and nothing else of C0 is changed.

Arms (one incarnation with FLASH_S5_STAGE_LOCAL=1 and FLASH_S5_STAGE_OVERLAP=1; each arm selects by typed control):
  band 1  65,445 fresh then the append, 8 standing decoders, rounds 0 and 1 mirrored over serial, local (C0) and overlap (C1),
          solo and live (lead, 2 Nov: C1 against C0 and serial, ABBA, live and solo): the selections read
          S-solo L-solo O-solo O-live L-live S-live | S-live L-live O-live O-solo L-solo S-solo, so each local and overlap arm
          has its matched serial arm at the same cohort and round, and the two rounds mirror each other against linear drift.
  band 2  alpha-perf's band 2 (131,094 tokens, workloads/q18a-alpha-perf-band2.json), past the 97,152-row arena: serial-solo-2,
          overlap-solo-2, overlap-live-3, serial-live-3. Each candidate arm has its matched serial arm at the same cohort and index.
  control pause, native_progress {burst}, native_stage_local {local, overlap}, release_auto_arm, resume; the reply must carry both
          values, and the status before and after the fresh prompt must show both selected.

The grade is C0's grade plus:
  regroup     on an overlap arm one engine progress line is flushed per commit and may hold the successor's early rows. The
              ``flash-s5 stage-overlap txn=T seqs=RANGES successor=S`` line names each chunk's command sequences; within each arm
              span the engine flushes' native rows are regrouped by it before C0's join reads them (``regroup``; the post-run
              report imports this rule, there is no second one). Stock rows stay with their flush. A sequence two lines name, or
              rows whose chunk has no flush in the arm, are reasons (INC), never resolved here: C0's tolerance of one lost record
              does not reach a lost engine flush on an overlap arm, whose successor's early rows cannot be placed.
  engagement  an overlap arm passes C0's local checks and needs, from the status delta of the fresh prompt, a successor frame, an
              overlapped commit and an early open (stage 0 opened n+1 before stage 1 released n; without one the pipelining this
              queue measures did not happen), one stage-overlap line per auto-owned chunk, and every regrouped chunk with exactly
              two opens and two releases. Serial and local arms need every overlap counter delta 0 and no stage-overlap line.
  in flight   one command in flight per stage, read from the engine's own rows: per stage, the segment commands in submit order,
              each submitted no earlier than the previous one's apply (its last target reply when it has no apply). A breach is
              FAIL ENGINE. Declared limit: owner-side CLOCK_MONOTONIC stamps of the engine's command chain; device execution is
              not visible, only segment rows carry a stage (opens, binds, releases, receipts and polls are not judged), and a lost
              engine flush hides its rows.
  signatures  the overlap path's contract errors join C0's (FAIL ENGINE).
  OOM         any rise of a rank's ``ooms`` counter, read from an intact receipt or from a torn one (each ``flash-s5 memory``
              fragment of a torn line is read for its own rank and ooms; a fragment without a readable rank rises when its value
              exceeds any rank's previous one), fails the arm whose span holds it as ENGINE; outside every arm it fails the run.
              A misread torn number errs toward FAIL.
  memory      reported as MET / NOT_MET / INC beside the verdict, never a verdict gate (the lead, 2 Nov: as Step A, Step B and C0;
              min free varies by tens of MiB between identical runs); a NOT_MET or INC read blocks enabling C1 (hub reading).
              Per arm, max allocated and max reserved over its receipts against the matched serial arm. Per candidate arm against its
              matched serial arm, 0 OOMs and min free no lower than the serial arm's by more than 2 MiB. Only whole receipts are
              read (the full line to ``device_frees``; C0's rule reads up to ``free=``). A torn receipt, or a receipt missing for a
              (rank, chunk, point) of the arm's chunks, makes that arm's memory read INC; a torn or lost receipt in the matched
              serial arm makes the pair INC; NOT_MET stands when the candidate's whole receipts already show it. Per chunk, the
              transient peak: reserved and free at the open and released receipts, worst rank, candidate minus serial, with the
              chunks past 97,152 rows apart. Declared limit: there is no peak field; reserved is the caching allocator's held
              bytes, which keeps a transient's blocks until empty_cache or an allocator retry frees them, so reserved at the
              released receipt bounds the chunk's peak from below only when the chunk moved no alloc_retries (such chunks are
              flagged), and free is the driver's value at the receipt instant.
  slow rank   per regrouped chunk, per stage, each own TP rank's merged send-to-reply seconds and the slower rank (the lead's
              addition 2); per arm and per history band the median of the slower rank's seconds next to the pooled per-rank median.
  numerics    C0's comparison per band, for each candidate selection against the serial arms of that band.
  budget      the queue does not start an arm past its budget; such an arm has no record and is INC (``no-arm-record``), never an
              engine result. Only an engine watch DEAD makes a missing arm FAIL.
  tears       C0's rule (f1ed8a6) for progress, stage and memory lines, plus torn stage-overlap lines: each counts as a lost
              record, and the chunk without its line drops out with its predecessor (whose flush kept the successor's early rows).

Receipts hold counts, hashes, identities and host clock stamps, never prompts or auth.
"""
import argparse
import contextlib
import json
from pathlib import Path
import re
import statistics

import e3_alpha_c0_box as c0

ap, progress, base = c0.ap, c0.progress, c0.base
GRADE_SCHEMA = 'alpha-c1-grade/1'
SELECTIONS = ('serial', 'local', 'overlap')
BAND1_ORDER = (('serial', 'solo', 0), ('local', 'solo', 0), ('overlap', 'solo', 0), ('overlap', 'live', 0), ('local', 'live', 0),
               ('serial', 'live', 0), ('serial', 'live', 1), ('local', 'live', 1), ('overlap', 'live', 1), ('overlap', 'solo', 1),
               ('local', 'solo', 1), ('serial', 'solo', 1))
BAND2_ORDER = (('serial', 'solo', 2), ('overlap', 'solo', 2), ('overlap', 'live', 3), ('serial', 'live', 3))
ARM_ORDER = BAND1_ORDER + BAND2_ORDER
INDEXES = (0, 1, 2, 3)
TARGETS = c0.TARGETS
ARENA_ROWS = ap.ARENA_ROWS
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
SIGNATURES = c0.SIGNATURES + OVERLAP_SIGNATURES
OVERLAP_MARK = 'flash-s5 stage-overlap '
OVERLAP_LINE = re.compile(r'flash-s5 stage-overlap txn=(\d+) seqs=(-|\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*) successor=(\d+|none)\s*$')
MEMORY_MARK = c0.MEMORY_MARK
# The whole receipt, to its last field (worker_bridge._memory_receipt).
MEMORY_WHOLE = re.compile(c0.MEMORY_LINE.pattern + r' total=(-?\d+) sync_all_streams=(-?\d+) device_allocs=(-?\d+) '
                                                   r'device_frees=(-?\d+)\s*$')
FRAGMENT_FIELD = re.compile(r'\b(rank|ooms)=(\d+)\b')
POINTS = ('open', 'released')
SPEED_KEYS = ('ttft_s', 'prefill_tok_s', 's_per_chunk', 'decode_during_tok_s', 'decode_ratio', 'commands_per_chunk')
RELEASE_READINESS = c0.RELEASE_READINESS
LIMITS = dict(
    in_flight='owner-side stamps of segment rows only; device execution, unstaged kinds and rows of a lost engine flush not judged',
    transient_peak='no peak field: reserved and free at the open and released receipts; a chunk that moved alloc_retries is flagged',
    oom='a torn receipt is read per fragment; a misread number errs toward FAIL',
    budget='an arm the queue did not start (budget) has no record: INC, never an engine result')

# C0's own functions, taken before any binding.
C0_RELEASE, C0_SELECTION_REASONS = c0.release_selected, c0.selection_reasons
C0_STAGE_VIEW, C0_STAGE_DELTA = c0.stage_view, c0.stage_delta
C0_ENGAGEMENT, C0_SETTLE, C0_TABLE_ROW, C0_ARM_VERDICT = c0.engagement, c0.settle_arm, c0.table_row, c0.arm_verdict
C0_NUMERICS = c0.numerics
label = c0.label


def required_labels():
    return [label(*x) for x in ARM_ORDER]


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


# --- the arm --------------------------------------------------------------------------------------------

def stage_view(status):
    """C0's view plus the overlap block, flat (``overlap_selected``, ``overlap_<counter>``)."""
    row = C0_STAGE_VIEW(status)
    if row is None:
        return None
    block = status['native_stage_local'].get('overlap')
    if type(block) is not dict or type(block.get('counters')) is not dict:
        raise ValueError('native-stage-overlap-not-installed')
    return dict(row, overlap_selected=block.get('selected'),
                **{'overlap_' + k: block['counters'].get(k) for k in OVERLAP_COUNTERS})


def stage_delta(after, before):
    out = C0_STAGE_DELTA(after, before)
    out.update({'overlap_' + k: progress.delta(after, before, 'overlap_' + k) for k in OVERLAP_COUNTERS})
    return out


def release_selected(selection):
    """C0's release with ``overlap`` added to the selection command; the reply must carry both values."""
    local, overlap = selection != 'serial', selection == 'overlap'
    release = C0_RELEASE(local)

    def run(client, record, mode):
        replies = {}

        class Selecting:
            def __getattr__(self, name):
                return getattr(client, name)

            def call(self, action, **fields):
                if action == 'native_stage_local':
                    fields = dict(fields, overlap=overlap)
                reply = client.call(action, **fields)
                if action == 'native_stage_local':
                    replies['selection'] = reply
                return reply
        try:
            release(Selecting(), record, mode)
        finally:
            reply = replies.get('selection')
            if type(reply) is dict and type(record.get('selection_switch')) is dict:
                record['selection_switch'].update(overlap=reply.get('overlap'), previous_overlap=reply.get('previous_overlap'))
        if record['selection_switch'].get('overlap') is not overlap:
            raise ValueError('native-stage-overlap-selection-unproved')
    return run


def selection_reasons(record, selection):
    reasons = C0_SELECTION_REASONS(record, selection != 'serial')
    views = {s.get('label'): s for s in record.get('progress') or []}
    before = (views.get('before-fresh') or {}).get('native_stage_local')
    after = (views.get('after-fresh') or {}).get('native_stage_local')
    overlap = selection == 'overlap'
    if before and after and (before.get('overlap_selected') is not overlap or after.get('overlap_selected') is not overlap):
        reasons.append('stage-overlap-selection-not-this-arm')
    return reasons


def _arm_print(*args, **kwargs):
    text = ' '.join(str(x) for x in args)
    print(text.replace('ALPHA_C0_ARM ', 'ALPHA_C1_ARM ', 1), **kwargs)


def arm_names(selection):
    return dict(release_selected=lambda _local: release_selected(selection),
                selection_reasons=lambda record, _local: selection_reasons(record, selection),
                stage_view=stage_view, stage_delta=stage_delta, print=_arm_print)


def run_arm(a):
    """C0's arm with this selection's control and receipts."""
    with _bound(c0, arm_names(a.mode)):
        return c0.run_arm(a)


# --- regroup and slow rank (the post-run report imports these) ------------------------------------------

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


def merged_seconds(spans):
    total, current = 0, None
    for lo, hi in sorted((a, b) for a, b in spans if b >= a):
        if current is None or lo > current[1]:
            if current is not None:
                total += current[1] - current[0]
            current = [lo, hi]
        else:
            current[1] = max(current[1], hi)
    if current is not None:
        total += current[1] - current[0]
    return total / 1e9


def slow_rank(chunk):
    """Per stage: each own rank's merged send -> reply seconds over the stage's commands; the slower one."""
    out = {}
    for stage, ranks in TARGETS.items():
        spans = {rank: [] for rank in ranks}
        for row in chunk['rows'].get('n', []):
            if row.get('stage') != stage or type(row.get('t_sent')) is not int:
                continue
            for rank in ranks:
                recv = row.get('t_recv%d' % rank)
                if type(recv) is int:
                    spans[rank].append((row['t_sent'], recv))
        per_rank = {str(rank): merged_seconds(v) for rank, v in spans.items()}
        slow = max(per_rank, key=lambda r: per_rank[r]) if any(per_rank.values()) else None
        out[str(stage)] = dict(per_rank_s=per_rank, slow_rank=slow, slow_s=per_rank[slow] if slow else None)
    return out


def slow_summary(reports):
    """Per stage: median of the slower rank's seconds, the pooled per-rank median, and how often each rank was the slower one."""
    out = {}
    for stage in ('0', '1'):
        cells = [r['slow_rank'][stage] for r in reports if (r.get('slow_rank') or {}).get(stage, {}).get('slow_s') is not None]
        counts = {}
        for cell in cells:
            counts[cell['slow_rank']] = counts.get(cell['slow_rank'], 0) + 1
        pooled = [v for cell in cells for v in cell['per_rank_s'].values()]
        out[stage] = dict(median_slow_s=statistics.median(c['slow_s'] for c in cells) if cells else None,
                          median_pooled_s=statistics.median(pooled) if pooled else None,
                          chunks=len(cells), slower_counts=counts)
    return out


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


# --- the bindings during C0's grade ------------------------------------------------------------------------

_RUN = {}


def regrouping(original):
    """progress.assemble with each arm's engine flushes regrouped by that arm's stage-overlap lines."""
    def assemble(records):
        flushes, problems = original(records)
        engine = [f for f in flushes if f['side'] == 'engine']
        replaced = {}
        for arm in _RUN['arms']:
            mine = [f for f in engine if id(f) not in replaced and ap.in_span(f['stamp'], arm)]
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
    """The chunk breakdown with the slower rank per stage and the chunk's staged segment rows (for the in-flight check)."""
    def run(chunk, workers):
        info = original(chunk, workers)
        info['slow_rank'] = slow_rank(chunk)
        info['_segments'] = [_segment(r, r['stage']) for r in chunk['rows'].get('n', [])
                             if r.get('kind') == 'segment' and r.get('stage') in TARGETS and type(r.get('t_submit')) is int]
        return info
    return run


def settle_arm(arm, row, ledger, torn, lines, last_line):
    """C0's settle, then the overlap facts: regroup faults, torn stage-overlap lines and the in-flight check."""
    tears = C0_SETTLE(arm, row, ledger, torn, lines, last_line)
    name = arm['label']
    chunks = row.get('chunks') or []
    facts = _RUN['regroup'].get(name) or {}
    row['reasons'] += ['overlap-regroup:' + x for x in facts.get('faults', ())]
    if facts.get('orphan_rows'):
        row['reasons'].append('overlap-rows-of-a-chunk-without-its-flush')
    torn_overlap = [t for s, t in _RUN['torn_overlap'] if ap.in_span(s, arm)]
    if torn_overlap and arm.get('selection') == 'overlap' and chunks:
        logged = {int(OVERLAP_LINE.search(t).group(1)) for s, t in _RUN['overlap'] if ap.in_span(s, arm)}
        ordered = sorted(chunks, key=lambda c: c.get('chunk_start') or 0)
        without = [i for i, c in enumerate(ordered) if c.get('txn') not in logged]
        if len(without) <= len(torn_overlap):
            for i in without:
                for chunk in ordered[max(0, i - 1):i + 1]:
                    chunk.setdefault('dropped_out', []).append('overlap-line-lost')
            c0.recount(row)
    _RUN['in_flight'][name] = in_flight([s for c in chunks for s in c.pop('_segments', ())])
    records = tears['lost_records'] + len(torn_overlap)
    starts = sorted(set(tears['dropped_chunk_starts']) | {c['chunk_start'] for c in chunks if c.get('dropped_out')})
    row['reasons'] = [x for x in row['reasons'] if not x.startswith(('lost-records-above', 'dropped-chunks-above'))]
    if records > c0.LOST_RECORD_LIMIT:
        row['reasons'].append('lost-records-above-the-declared-limit:%d>%d' % (records, c0.LOST_RECORD_LIMIT))
    if len(starts) > c0.DROPPED_CHUNK_LIMIT:
        row['reasons'].append('dropped-chunks-above-the-declared-limit:%d>%d' % (len(starts), c0.DROPPED_CHUNK_LIMIT))
    tears.update(torn_overlap_lines=len(torn_overlap), lost_records=records, dropped_chunk_starts=starts,
                 within_limits=records <= c0.LOST_RECORD_LIMIT and len(starts) <= c0.DROPPED_CHUNK_LIMIT)
    return tears


def engagement(arm, row, lines, lost_stage=0):
    selection = arm.get('selection')
    out = C0_ENGAGEMENT(dict(arm, selection='local' if selection == 'overlap' else selection), row, lines, lost_stage)
    out['selection'] = selection
    delta = arm.get('stage_delta') or {}
    counters = {k: delta.get('overlap_' + k) for k in OVERLAP_COUNTERS}
    logged = sum(1 for s, _ in _RUN['overlap'] if ap.in_span(s, arm))
    lost = sum(1 for s, _ in _RUN['torn_overlap'] if ap.in_span(s, arm))
    kept = [c for c in ((row or {}).get('chunks') or []) if not c.get('dropped_out')]
    uneven = [c['chunk_start'] for c in kept if (c.get('kinds') or {}).get('open') != 2 or (c.get('kinds') or {}).get('release') != 2]
    out.update(overlap=counters, overlap_lines=logged, lost_overlap_lines=lost,
               regrouped_chunks_checked=len(kept) if selection == 'overlap' else None,
               regrouped_chunks_uneven=uneven if selection == 'overlap' else None)
    if 'engagement-receipts-missing' in out['reasons']:
        return out
    reasons = list(out['reasons'])
    if any(type(v) is not int for v in counters.values()):
        reasons.append('overlap-counters-missing')
    elif selection == 'overlap':
        if counters['successor_frames'] < 1 or counters['overlapped_commits'] < 1:
            reasons.append('overlap-ran-no-second-chunk')
        if counters['early_opens'] < 1:
            reasons.append('overlap-stage-zero-never-opened-a-successor-early')
    elif any(counters.values()):
        reasons.append('%s-arm-moved-an-overlap-counter' % selection)
    if selection == 'overlap':
        if logged + lost != out['owned_chunks']:
            reasons.append('overlap-lines-differ-from-the-auto-owned-chunks')
        if uneven:
            reasons.append('regrouped-chunk-without-two-opens-and-two-releases')
    elif logged or lost:
        reasons.append('%s-arm-logged-an-overlap-line' % selection)
    out.update(engaged=not reasons, reasons=reasons)
    return out


def _receipt(text):
    m = MEMORY_WHOLE.search(text)
    if not m:
        return None
    return dict(point=m.group(1), rank=int(m.group(2)), txn=int(m.group(3)), chunk_start=int(m.group(4)), chunk_end=int(m.group(5)),
                reserved_mib=round(int(m.group(7)) / 2 ** 20, 1), retries=int(m.group(9)), free_mib=round(int(m.group(11)) / 2 ** 20, 1))


def memory_detail(row, mine):
    """Whole receipts only: per-rank worst, per-chunk transient (worst rank), torn and lost receipts."""
    lines = [(s, t) for s, t in mine if MEMORY_MARK in t]
    whole = [(s, t) for s, t in lines if MEMORY_WHOLE.search(t)]
    receipts = [_receipt(t) for _, t in whole]
    txns = {c.get('txn'): c for c in ((row or {}).get('chunks') or []) if type(c.get('txn')) is int}
    present = {(r['rank'], r['txn'], r['point']) for r in receipts}
    lost = sorted('%d:%d:%s' % key for key in ((rank, txn, point) for txn in txns for rank in range(4) for point in POINTS)
                  if key not in present)
    chunks = {}
    for r in receipts:
        if r['txn'] not in txns:
            continue
        cell = chunks.setdefault(r['txn'], dict(txn=r['txn'], chunk_start=r['chunk_start'], chunk_end=r['chunk_end'], receipts=0,
                                                peak_reserved_mib=None, min_free_mib=None, retries={}))
        cell['receipts'] += 1
        cell['peak_reserved_mib'] = r['reserved_mib'] if cell['peak_reserved_mib'] is None else max(cell['peak_reserved_mib'], r['reserved_mib'])
        cell['min_free_mib'] = r['free_mib'] if cell['min_free_mib'] is None else min(cell['min_free_mib'], r['free_mib'])
        cell['retries'].setdefault(r['rank'], {})[r['point']] = r['retries']
    rows = []
    for cell in sorted(chunks.values(), key=lambda c: c['chunk_start']):
        moved = any(p.get('released') is None or p.get('open') is None or p['released'] != p['open'] for p in cell.pop('retries').values())
        rows.append(dict(cell, retries_moved=moved))
    ranks = c0.memory_receipts(whole)
    torn = len(lines) - len(whole)
    return dict(ranks=ranks, worst=c0.memory_worst(ranks), torn_receipts=torn, lost_receipts=lost, chunks=rows,
                established=not torn and not lost)


def table_row(arm, row, view, mine):
    out = C0_TABLE_ROW(arm, row, view, mine)
    out['memory'] = memory_detail(row, mine)
    kept = [c for c in ((row or {}).get('chunks') or []) if c.get('label') == (row or {}).get('counted_label')
            and not c.get('dropped_out')]
    out['slow_rank'] = dict(all=slow_summary(kept),
                            bands={band: slow_summary([c for c in kept if c['band'] == band]) for band in sorted({c['band'] for c in kept})})
    return out


def arm_verdict(arm, ingress, watch, view, signatures, progress_reasons, engaged):
    name = arm.get('label')
    extra = {'oom'} if _RUN['oom'].get(name) else set()
    for stage, cell in (_RUN['in_flight'].get(name) or {}).items():
        if cell['breaches']:
            extra.add('two-commands-in-flight-on-stage-' + stage)
    return C0_ARM_VERDICT(arm, ingress, watch, view, set(signatures) | extra, progress_reasons, engaged)


def numerics(arms):
    """C0's comparison per band and candidate selection, each against that band's serial arms."""
    parts = {}
    for band in (1, 2):
        for candidate in ('overlap', 'local'):
            subset = [dict(a, selection='local' if a.get('selection') == candidate else a.get('selection')) for a in arms
                      if a.get('band') == band and a.get('selection') in ('serial', candidate)]
            if any(a['selection'] == 'local' for a in subset):
                parts['%s-band%d' % (candidate, band)] = C0_NUMERICS(subset)
    differs = ['%s:%s' % (k, x) for k, v in parts.items() for x in v['differs']]
    unproven = ['%s:%s' % (k, x) for k, v in parts.items() for x in v['unproven']]
    verdict = 'DIFFERS' if differs else 'UNPROVEN' if unproven or not parts else 'IDENTICAL'
    return dict(verdict=verdict, differs=differs, unproven=unproven or ([] if parts else ['no-comparison']), parts=parts,
                prefix_tokens=c0.STREAM_PREFIX_TOKENS, limit='per band; ' + 'one sampled token per prefill and stream token digests; '
                                                                     'not a tensor comparison')


def memory_compare(table):
    """Each candidate arm against its matched serial arm (same cohort and index): the reading and the per-chunk transient."""
    rows = {}
    for selection, cohort, index in ARM_ORDER:
        if selection == 'serial':
            continue
        name = label(selection, cohort, index)
        cand = (table.get(name) or {}).get('memory')
        serial = (table.get(label('serial', cohort, index)) or {}).get('memory')
        if not cand or not serial or not cand['worst'] or not serial['worst']:
            rows[name] = dict(read='INC', reason='receipts-missing', serial=label('serial', cohort, index))
            continue
        f, s = cand['worst'], serial['worst']
        met = f['ooms'] == 0 and f['min_free_mib'] >= s['min_free_mib'] - c0.MEMORY_FREE_MARGIN_MIB
        if not serial['established']:
            read, reason = 'INC', 'serial-arm-receipt-torn-or-lost'
        elif not met:
            read, reason = 'NOT_MET', None
        elif not cand['established']:
            read, reason = 'INC', 'receipt-torn-or-lost'
        else:
            read, reason = 'MET', None
        base_chunks = {(c['chunk_start'], c['chunk_end']): c for c in serial['chunks']}
        per_chunk = []
        for c in cand['chunks']:
            b = base_chunks.get((c['chunk_start'], c['chunk_end']))
            if b is None:
                continue
            per_chunk.append(dict(chunk_start=c['chunk_start'], chunk_end=c['chunk_end'],
                                  reserved_delta_mib=round(c['peak_reserved_mib'] - b['peak_reserved_mib'], 1),
                                  free_delta_mib=round(c['min_free_mib'] - b['min_free_mib'], 1),
                                  retries_moved=c['retries_moved'] or b['retries_moved']))

        def extremes(cells):
            return dict(chunks=len(cells), max_reserved_delta_mib=max((x['reserved_delta_mib'] for x in cells), default=None),
                        min_free_delta_mib=min((x['free_delta_mib'] for x in cells), default=None),
                        chunks_with_retries=sum(1 for x in cells if x['retries_moved']))
        rows[name] = dict(read=read, reason=reason, serial=label('serial', cohort, index), candidate=f, serial_worst=s,
                          minus_serial={k: round(f[k] - s[k], 1) for k in f if k != 'receipts'}, zero_ooms=f['ooms'] == 0,
                          min_free_within_margin=f['min_free_mib'] >= s['min_free_mib'] - c0.MEMORY_FREE_MARGIN_MIB,
                          per_chunk=per_chunk, transient=extremes(per_chunk),
                          past_arena=extremes([x for x in per_chunk if x['chunk_start'] >= ARENA_ROWS]))
    return rows


def speed_compare(groups):
    out = {}
    for key, group in groups.items():
        selection, cohort, band = key.split('-')
        serial = groups.get('serial-%s-%s' % (cohort, band))
        if selection != 'serial' and serial:
            out[key] = {k: progress.ratio(group.get(k), serial.get(k)) for k in SPEED_KEYS}
    return out


def grade(arms, ingress, checks, logs, watches=None):
    """C0's grade with the names above bound; then the run-level OOM, the band groups and the C1 record fields."""
    known = [a for a in arms if a.get('label')]
    intact, torn_overlap = overlap_rows(logs)
    by_arm, outside = oom_rises(logs, known)
    _RUN.clear()
    _RUN.update(arms=known, overlap=intact, torn_overlap=torn_overlap, oom=by_arm, regroup={}, in_flight={})
    names = dict(ARM_ORDER=ARM_ORDER, SIGNATURES=SIGNATURES, settle_arm=settle_arm, engagement=engagement, table_row=table_row,
                 arm_verdict=arm_verdict, numerics=numerics, memory_compare=memory_compare)
    try:
        with _bound(c0, names), _bound(progress, dict(assemble=regrouping(progress.assemble),
                                                      chunk_breakdown=breakdown(progress.chunk_breakdown))):
            result = c0.grade(arms, ingress, checks, logs, watches)
        in_flight_rows, regroup_facts = dict(_RUN['in_flight']), dict(_RUN['regroup'])
    finally:
        _RUN.clear()
    failures, missing = list(result['failures']), list(result['missing'])
    classes = set(result['classification'] or ())
    if outside:
        failures.append('oom-outside-every-arm-span')
        classes.add('ENGINE')
    for name, v in result['validity'].items():
        v['oom_rises'] = by_arm.get(name, [])
        v['in_flight'] = in_flight_rows.get(name)
        v['regroup'] = regroup_facts.get(name)
    rows = [result['table'][a['label']] for a in known if a['label'] in result['table']]
    groups = {}
    for selection in SELECTIONS:
        for cohort in progress.COHORTS:
            for band in (1, 2):
                cell = [r for r in rows if (r['selection'], r['cohort'], r.get('band')) == (selection, cohort, band)]
                if cell:
                    groups['%s-%s-%d' % (selection, cohort, band)] = c0.group_summary(cell)
    summary = dict(result['summary'], groups=groups, over_serial=speed_compare(groups))
    summary.pop('local_over_serial', None)
    verdict = 'FAIL' if failures else 'INC' if missing else 'PASS'
    if verdict == 'FAIL' and not classes:
        classes = {'HARNESS'}
    result.update(schema=GRADE_SCHEMA, verdict=verdict, classification=sorted(classes) or None, failures=sorted(set(failures)),
                  missing=sorted(set(missing)), summary=summary, memory_vs_serial=result.pop('memory_local_vs_serial'),
                  oom_outside_every_arm=outside, oom_is_gate=True, memory_is_gate=False, declared_limits=LIMITS,
                  basis='one engine, one selection per arm by typed control (local and overlap); band 1 ABBA at 65,445 tokens '
                        'with local solo and live in both rounds; band 2 at 131,094 tokens past the 97,152-row arena; per chunk, the '
                        'critical rank of each native command, rows regrouped by chunk on overlap arms',
                  scope='C1 mechanism (two chunks in flight, one command in flight per stage) on the composed alpha engine, with '
                        'speed and memory reported; gains from separate experiments are never multiplied')
    return result


# --- the report ----------------------------------------------------------------------------------------

_cell = c0._cell


def report(result):
    """C0's sections (title and memory section replaced), plus overlap engagement, memory against serial and the slower rank."""
    text = c0.report(dict(result, memory_local_vs_serial={}, summary=dict(result['summary'], local_over_serial=None)))
    text = text.replace('# q18a-alpha-c0 report', '# q18a-alpha-c1 report', 1)
    head, _, rest = text.partition('### Device memory, local against the matched serial arm')
    _, _, tail = rest.partition('## Release readiness')
    head = head.replace('Local over serial, medians: null', 'Over the matched serial arms, medians: '
                        + json.dumps(result['summary']['over_serial'], sort_keys=True))
    lines = ['### Overlap engagement and one command in flight per stage', '',
             '| arm | successor frames | early opens | overlapped commits | overlap lines | lost overlap lines | uneven chunks | '
             'stage 0 commands / breaches | stage 1 commands / breaches | OOM rises |', '|---|---|---|---|---|---|---|---|---|---|']
    for name in result['required_arms']:
        v = result['validity'].get(name) or {}
        e = v.get('engagement') or {}
        o = e.get('overlap') or {}
        f = v.get('in_flight') or {}
        lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            name, _cell(o.get('successor_frames')), _cell(o.get('early_opens')), _cell(o.get('overlapped_commits')),
            _cell(e.get('overlap_lines')), _cell(e.get('lost_overlap_lines')),
            ','.join(str(x) for x in e.get('regrouped_chunks_uneven') or []) or '0',
            *('%s / %d' % (f[s]['commands'], len(f[s]['breaches'])) if s in f else '-' for s in ('0', '1')),
            len(v.get('oom_rises') or [])))
    lines += ['', '### Device memory against the matched serial arm', '',
              'A reading, never a verdict gate; a NOT_MET or INC read blocks enabling C1 (hub reading). An OOM fails the arm '
              '(ENGINE). Whole receipts only; max allocated and max reserved are over the arm\'s receipts (worst rank), per chunk '
              'the worst rank at the open and released receipts (no peak field; see declared_limits).', '',
              '| arm | serial | read | min free MiB | serial | max allocated MiB | serial | max reserved MiB | serial | OOMs | max chunk '
              'reserved delta | min chunk free delta | past 97,152: max reserved delta | min free delta | chunks with retries |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for name, row in result['memory_vs_serial'].items():
        if 'candidate' not in row:
            lines.append('| %s | %s | %s | %s | | | | | | | | | | | |' % (name, row['serial'], row['read'], row['reason']))
            continue
        f, s, t, p = row['candidate'], row['serial_worst'], row['transient'], row['past_arena']
        lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            name, row['serial'], row['read'] + (' (%s)' % row['reason'] if row['reason'] else ''), f['min_free_mib'], s['min_free_mib'],
            f['peak_allocated_mib'], s['peak_allocated_mib'], f['peak_reserved_mib'], s['peak_reserved_mib'], f['ooms'], _cell(t['max_reserved_delta_mib']),
            _cell(t['min_free_delta_mib']), _cell(p['max_reserved_delta_mib']), _cell(p['min_free_delta_mib']),
            t['chunks_with_retries']))
    lines += ['', '### Slower TP rank per stage', '',
              'Per chunk, each own rank\'s merged send-to-reply seconds per stage; median of the slower rank next to the pooled '
              'per-rank median.', '',
              '| arm | band | stage 0 slower s | pooled | stage 1 slower s | pooled | slower rank counts (s0; s1) |',
              '|---|---|---|---|---|---|---|']
    for name in result['required_arms']:
        row = result['table'].get(name)
        if not row:
            continue
        for band, cell in [('all', row['slow_rank']['all'])] + sorted(row['slow_rank']['bands'].items()):
            lines.append('| %s | %s | %s | %s | %s | %s | %s; %s |' % (
                name, band, _cell(cell['0']['median_slow_s'], 4), _cell(cell['0']['median_pooled_s'], 4),
                _cell(cell['1']['median_slow_s'], 4), _cell(cell['1']['median_pooled_s'], 4),
                json.dumps(cell['0']['slower_counts'], sort_keys=True), json.dumps(cell['1']['slower_counts'], sort_keys=True)))
    if result['oom_outside_every_arm']:
        lines += ['', 'OOM outside every arm span: %d rises.' % len(result['oom_outside_every_arm'])]
    return head + '\n'.join(lines) + '\n\n## Release readiness' + tail


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='action', required=True)
    r = sub.add_parser('arm')
    r.add_argument('--url', required=True)
    r.add_argument('--model', required=True)
    r.add_argument('--run-id', required=True)
    r.add_argument('--epoch', type=int, required=True)
    r.add_argument('--mode', choices=SELECTIONS, required=True, help='this arm\'s path; native progress is burst in every arm')
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
    g.add_argument('--report', help='write the report here')
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
    print('ALPHA_C1_GRADE ' + result['verdict'])
    return {'PASS': 0, 'FAIL': 1}.get(result['verdict'], 2)


if __name__ == '__main__':
    raise SystemExit(main())
