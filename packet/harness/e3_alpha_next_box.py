#!/usr/bin/env python3
"""alpha-next box arms and grade: C0 stage-local against serial on the fused, grouped engine, matched per arm on one engine.

The arm is Step B's (e3_step_b_box.arm_seams over held-f's e3_native41_progress_box.arm), called unchanged; the seams below add
one typed control, native_stage_local {local}, after native_layer_group, and two receipts, and change no other arm step. Control
order: pause, native_progress {burst}, native_qsa_fused {true}, native_layer_group {size}, native_stage_local {local},
release_auto_arm, resume, then status only. The record keeps ``mode`` burst and carries ``selection`` (``c0on.g2``), ``size``,
``stage_local``, ``c0``, ``part`` and ``band``. The arm table is alpha_next_arms (one declared table): an arm not in it is refused
before any traffic.

The grade is e3_step_b_box.grade, called unchanged inside grade_seams (which set Step B's module names for this one call, as Step
B sets Step A's), so Step B's own grade in the same process is untouched. Gated as Step B: crash-free, 0 ``disagree``, 0
``tp-ledger-mismatch``, 0 ``tp-collective-stall``, 0 ``fused-profile-*``, an OOM fails the arm (Step B's rule), torn lines under
the pre-registered rule, and engagement:
  group      Step B's rule at the arm's own size (size 1: every group counter delta 0; size 2 and 4: group chunks equal the
             auto-owned chunks, plan and group commands at the size, 48 group layers per chunk, the largest group is the size).
  C0 on      stage-local chunks equal the auto-owned chunks, two local releases per chunk, one ``flash-s5 stage-local`` line per
             chunk (a torn one counts for its chunk) and no reply from a non-target rank on a stage command (C0's rule).
  C0 off     every stage counter delta 0 and no stage-local line.
  An arm that did not engage is INC. Added to Step B's signatures: the C0 stage path's contract errors (C0's SIGNATURES), ENGINE.
Numerics, declared before the run (lead, 2 Nov 13:28 and 13:35): the greedy standing streams (alpha_next_arms.STANDING_GREEDY,
temperature 0 on those request bodies only) and the fresh and append tokens of each C0-on arm against the spread of the graded
C0-off arms of its band, rule v4 row 2 style (NUMERICS_* below). Two readings are declared, one constant apart:
  in force (NUMERICS_IS_GATE = False; lead 13:35, the later note: "still classed as a read under the v4-style envelope"): the
             envelope verdict WITHIN, OUTSIDE or UNPROVEN is reported and is never a verdict input;
  the other  (NUMERICS_IS_GATE = True; lead 13:28 read as a gate): OUTSIDE fails the queue as ENGINE and UNPROVEN makes it INC.
The coordinator asked the lead to confirm (2 Nov); flipping the one constant is the whole change.
Reads, never verdict inputs (operator, 2 Nov): decode per arm as the per-stream minimum and mean ratio against the 0.8 and 0.9
lines (4C.7), split by the sampled standing streams (STANDING_SAMPLED), the greedy ones and all 8, with each stream's ratio;
speed; memory (the min-free read and the per-chunk ``flash-s5 memory-peak`` receipt); the C0 ring reading (headroom
refusals, ``headroom_blocked_ns`` on the stage line, status ``ring``: blocked_ns, episodes, blocked_max_ns, limiting and
limiting_sole per rank); the item-1 stop receipts (``flash-s5 stop`` lines, schema s5-stop-receipt/1, status ``stop_receipts``);
the stage-1 handler anatomy line count (tools/stage1_anatomy_report.py reads the lines).
Receipt integrity (RECEIPT_INTEGRITY_IS_GATE = False; lead, 2 Nov: a read this run, a gate next run, because the PLAN 2 writer's
behaviour on the box FIFO is unproven and this run proves it): receipt_lines' per-process counts, headline under Overall beside
the numerics line, and the reads its rule for new queues would have made INC.

Two grades: ``grade`` over the core arms (parts core and extra; an extra arm that was not entered is not required) is the
queue's verdict, written before the g4 tail starts. ``read`` over every arm (core, extra and tail) is a read only: same rules,
never the queue's verdict. Band 1's third round uses round index 3 (alpha_next_arms.band_of_index); Step A's band_of_index is
replaced by it for the arm and the grade only, as the other seams. Report: mechanism, performance and release readiness in separate
headed sections. Receipts hold counts, hashes, identities, token ids and host clock stamps, never prompts or auth.
"""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import re
import statistics

import alpha_next_arms as T
import e3_step_b_box as b
import receipt_lines
from e3_native_qualification_driver import exception_details
from e3_native_runtime_client import NativeRuntimeClient

sa = b.sa
ap = b.ap
progress = b.progress
base = b.base
GRADE_SCHEMA = 'alpha-next-grade/1'
READ_SCHEMA = 'alpha-next-read/1'
STAGE_CONTROL = 'native_stage_local'
CONTROL_SEQUENCE = ['status', 'pause', 'native_progress', 'native_qsa_fused', 'native_layer_group', STAGE_CONTROL,
                    'release_auto_arm', 'resume']
# C0's declarations (e3_alpha_c0_box), restated: importing that module wraps progress.chunk_breakdown for the whole process.
# test_alpha_next_queue.C0Parity compares these with its source text.
STAGE_COUNTERS = ('chunks', 'headroom_refused', 'local_releases', 'both_stages_ns', 'selections')
TARGETS = {0: (0, 1), 1: (2, 3)}
STAGE_SIGNATURES = (('stage-one-chunk', 'stage-local progress runs one chunk'), ('stage-commit', 'stage-local commit without'),
                    ('stage-release', 'local release lacks'), ('stage-scope', 'a chunk command needs a stage scope'),
                    ('stage-target', 'a stage command targets another stage'), ('stage-nest', 'a stage scope does not nest'),
                    ('stage-open', 'open reply does not cover every stage rank'),
                    ('stage-retained', 'a retained PP send already holds'), ('stage-receipt', 'a receipt delivery spans retained'),
                    ('stage-channel', 'a retained send has no PP channel'), ('stage-skip', 'a stage skip never leaves'))
STAGE_LINE = 'flash-s5 stage-local '
# C0's line plus alpha-next's headroom_blocked_ns (stage_progress._stage_line); C0's own regex still reads the head.
STAGE_RE = re.compile(r'flash-s5 stage-local txn=(\d+) chunk=(\d+)-(\d+) commands=(\d+),(\d+) both_stages_ns=(\d+) '
                      r'first_ns=(\S+),(\S+) last_ns=(\S+),(\S+) local_release_ns=(\S+),(\S+) headroom_refused=(\d+)'
                      r'(?: headroom_blocked_ns=(\d+))?')
RING_COUNTERS = ('blocked_ns', 'episodes')
STOP_LINE = 'flash-s5 stop '                              # engine_driver.STOP_RECEIPT_LINE
STOP_SCHEMA = 's5-stop-receipt/1'                         # engine_driver.STOP_RECEIPT_SCHEMA
STOP_KINDS = ('stop_partial_replay', 'stop_bound_exhausted')
STOP_COUNTERS = ('partial_replays', 'bound_exhausted')
ANATOMY_LINE = 'flash-s5 handler '                        # handler_anatomy.LINE_PREFIX
PEAK_LINE = 'flash-s5 memory-peak '                       # worker_bridge.PEAK_LINE (lead, 2 Nov 13:30: recorded, never gating)
PEAK_RE = re.compile(r'flash-s5 memory-peak rank=(\d+) txn_seq=(\d+) chunk=(\d+)-(\d+) peak_allocated=(-?\d+) '
                     r'peak_reserved=(-?\d+)(?: allocated_at_open=(-?\d+))?(?: stage_timer=(\d+))?')
DECODE_LINES = (0.8, 0.9)
# Numerics control envelope, declared before the run (lead, 2 Nov 13:28 CET, rule v4 row 2 style: a per-metric envelope from the
# same-instrument controls times a generous factor; 13:35: greedy streams only, a read). Per band and greedy standing stream: the
# controls are the graded C0-off arms with at least NUMERICS_PREFIX tokens; their floor is the shortest common prefix over every control pair; the envelope is the floor
# divided by NUMERICS_FACTOR, rounded down. A C0-on arm is inside when its common prefix with its nearest control is at least the
# envelope. For the single generated token of the fresh prompt and of the append: inside when it is one of the controls' tokens.
# Verdict WITHIN, OUTSIDE (anywhere), or UNPROVEN (fewer than two controls, or a missing or short receipt). Controls that already
# differ from each other widen the envelope instead of leaving the read UNPROVEN by construction. Sampled streams are not compared
# (their digests are recorded). NUMERICS_IS_GATE True would make OUTSIDE a FAIL ENGINE and UNPROVEN an INC.
NUMERICS_IS_GATE = False
# PLAN 2 receipt record integrity (harness/receipt_lines.py: missing seq, bad sha, incomplete record, unsafe path, per process).
# A read this run, a gate next run (lead, 2 Nov 14:4x): the PLAN 2 writer's behaviour on the box FIFO is unproven, and this run
# proves it. Shown under Overall as prominently as the numerics line, with per-process counts and the reads receipt_lines' rule
# for new queues would have made INC (RECEIPT_READS); True would make the queue INC on any dirty process.
RECEIPT_INTEGRITY_IS_GATE = False
RECEIPT_PROBLEMS = ('missing_seq', 'bad_sha', 'incomplete', 'unsafe_path')
# Which grade reads each receipt kind feeds (the receipt_lines kind is the prefix's second word).
RECEIPT_READS = {'progress': ('engagement', 'speed'), 'stage-local': ('c0_engagement', 'c0_ring'),
                 'stop': ('stop_receipts',), 'memory': ('memory',), 'native-alloc': ('memory',),
                 'memory-peak': ('memory_peak',), 'anatomy': ('anatomy',), 'handler': ('anatomy_lines',), 'receipts': ()}
NUMERICS_PREFIX = 256
NUMERICS_FACTOR = 2
OUTPUT_TOKENS = 8
RELEASE_READINESS = sa.RELEASE_READINESS
STEP_B_ENGAGEMENT = b.engagement


def label(c0, size, cohort, index):
    return T.label(c0, size, cohort, index)


def required_labels(part='core'):
    return T.labels(None if part == 'all' else T.GRADED)


def size_of(selection):
    parsed = T.parse_selection(selection)
    return parsed[1] if parsed else None


# --- the arm --------------------------------------------------------------------------------------------

def stage_view(status):
    block = status.get(STAGE_CONTROL) if type(status) is dict else None
    if type(block) is not dict or type(block.get('counters')) is not dict:
        return None
    ring = block.get('ring')
    ring = None if type(ring) is not dict else {k: ring.get(k) for k in ('blocked_ns', 'episodes', 'blocked_max_ns', 'limiting',
                                                                         'limiting_sole', 'blocked_open')}
    return dict({k: block['counters'].get(k) for k in STAGE_COUNTERS}, selected=block.get('selected'),
                ring_reserve=block.get('ring_reserve'), poll_reserve=block.get('poll_reserve'), ring=ring)


def stop_view(status):
    block = status.get('stop_receipts') if type(status) is dict else None
    return {k: block.get(k) for k in STOP_COUNTERS} if type(block) is dict else None


def stage_delta(after, before):
    return {k: progress.delta(after, before, k) for k in STAGE_COUNTERS}


def _per_rank_delta(after, before):
    if type(after) is not dict or type(before) is not dict:
        return None
    keys = sorted({str(k) for k in after} | {str(k) for k in before})
    get = lambda d, k: d.get(k, d.get(int(k) if k.isdigit() else k, 0)) or 0
    return {k: get(after, k) - get(before, k) for k in keys}


def ring_delta(after, before):
    """The ring reading over the fresh prompt (status counters are engine-lifetime). Descriptive."""
    if type(after) is not dict or type(before) is not dict:
        return None
    out = {k: progress.delta(after, before, k) for k in RING_COUNTERS}
    out.update(blocked_max_ns_after=after.get('blocked_max_ns'), blocked_open_after=after.get('blocked_open'),
               limiting=_per_rank_delta(after.get('limiting'), before.get('limiting')),
               limiting_sole=_per_rank_delta(after.get('limiting_sole'), before.get('limiting_sole')))
    return out


def stop_delta(after, before):
    if type(after) is not dict or type(before) is not dict:
        return None
    return {k: progress.delta(after, before, k) for k in STOP_COUNTERS}


def release_alpha_next(size, local):
    """Step B's release with native_stage_local after native_layer_group; its checks, then the stage reply's."""
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
            got = client.call(STAGE_CONTROL, local=local)
            record['stage_switch'] = {k: got.get(k) for k in ('local', 'previous', 'route', 'route_generation', 'engine_epoch')}
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
        stage = record['stage_switch']
        if stage.get('local') is not local or stage.get('route') != 'full':
            raise ValueError('native-stage-local-selection-unproved')
        if record['release'].get('released') is not True:
            raise ValueError('auto-arm-release-unproved')
    return release


def stage_reasons(record, local):
    """The stage selection's receipts beside Step B's. Engagement is the grade's, on the same numbers."""
    reasons = []
    history = record.get('control_history') or []
    if [x for i, x in enumerate(history) if x != 'status' or i == 0][:8] != CONTROL_SEQUENCE:
        reasons.append('control-sequence-not-pause-progress-fused-group-stage-release-resume')
    if history.count(STAGE_CONTROL) != 1:
        reasons.append('native_stage_local-sent-other-than-once')
    views = {s.get('label'): s for s in record.get('progress') or []}
    before = (views.get('before-fresh') or {}).get(STAGE_CONTROL)
    after = (views.get('after-fresh') or {}).get(STAGE_CONTROL)
    if not before or not after:
        reasons.append('stage-status-missing')
        return reasons
    if before.get('selected') is not local or after.get('selected') is not local:
        reasons.append('stage-selection-not-this-arm')
    record['stage_delta'] = stage_delta(after, before)
    record['ring_delta'] = ring_delta(after.get('ring'), before.get('ring'))
    if record['stage_delta']['selections'] != 0:
        reasons.append('stage-selection-moved-during-the-fresh-prompt')
    return reasons


def stream_outputs(states):
    """Per standing stream: tokens seen, the digest of the first NUMERICS_PREFIX token ids, and for a greedy stream the ids."""
    rows = {}
    for state in states:
        tokens = list(state.tokens)
        head = tokens[:NUMERICS_PREFIX]
        greedy = state.index in T.STANDING_GREEDY
        rows[str(state.index)] = dict(tokens=len(tokens), prefix_tokens=len(head), greedy=greedy,
                                      prefix_sha256=hashlib.sha256(json.dumps(head).encode()).hexdigest())
        if greedy:
            rows[str(state.index)]['prefix'] = head
    return rows


def greedy_salts(run_id):
    """The cache salts of the greedy standing streams, exactly as held-f's arm builds them."""
    return {progress.shared._cache_salt(run_id, progress.STANDING, i): i for i in T.STANDING_GREEDY}


@contextlib.contextmanager
def arm_seams(c0, size, outputs, run_id=None):
    """Inside Step B's arm_seams: the names held-f's arm reads, for this one process's one arm; restored on exit."""
    local = T.C0[c0]
    saved = {k: getattr(progress, k) for k in ('label', 'release', 'progress_reasons', 'progress_view', 'ACTIONS')}
    actions = base.ARM_ACTIONS
    request_row = base.request_row
    stop = progress.streams._stop_streams
    request = progress.shared._request
    band_of_index = ap.band_of_index
    greedy = greedy_salts(run_id) if run_id is not None else {}
    temperatures = outputs.setdefault('standing_temperature', {})
    step_b_reasons, step_b_view = saved['progress_reasons'], saved['progress_view']

    def reasons(record, mode, cohort):
        history = record.get('control_history') or []
        view = dict(record, control_history=[x for x in history if x != STAGE_CONTROL])
        out = step_b_reasons(view, mode, cohort)
        for key in ('progress_delta', 'fused_delta', 'group_delta', 'group_max_after'):
            if key in view:
                record[key] = view[key]
        return out + stage_reasons(record, local)

    def view(status):
        row = step_b_view(status)
        if row is None:
            return None
        block = stage_view(status)
        if block is None:
            raise ValueError('native-stage-local-not-installed')
        return dict(row, native_stage_local=block, stop_receipts=stop_view(status))

    def row_with_output(state, first, prompts, family):
        outputs[family] = list(state.tokens)[:OUTPUT_TOKENS]
        return request_row(state, first, prompts, family)

    def request_greedy(model, prompt, workload, salt):
        body = request(model, prompt, workload, salt)
        if salt in greedy:
            body = dict(body, temperature=T.GREEDY_TEMPERATURE)
            temperatures[str(greedy[salt])] = body['temperature']
        return body

    def stop_with_outputs(requests, states, workload):
        if 'standing' not in outputs and len(states) > 1:
            outputs['standing'] = stream_outputs(states)
        return stop(requests, states, workload)
    progress.label = lambda _mode, cohort, index: label(c0, size, cohort, index)
    progress.release = release_alpha_next(size, local)
    progress.progress_reasons = reasons
    progress.progress_view = view
    progress.ACTIONS = saved['ACTIONS'] | {STAGE_CONTROL}
    base.ARM_ACTIONS = dict(actions, auto=actions['auto'] | {STAGE_CONTROL})
    base.request_row = row_with_output
    progress.streams._stop_streams = stop_with_outputs
    progress.shared._request = request_greedy
    ap.band_of_index = T.band_of_index
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(progress, name, value)
        base.ARM_ACTIONS = actions
        base.request_row = request_row
        progress.streams._stop_streams = stop
        progress.shared._request = request
        ap.band_of_index = band_of_index


def run_arm(a):
    band = T.band_of_index(a.index)
    c0, local = a.stage_local, T.C0[a.stage_local]
    selection, name = T.selection(c0, a.size), label(c0, a.size, a.cohort, a.index)
    declared = T.by_label().get(name)
    outputs = {}
    try:
        if declared is None:
            raise ValueError('arm-not-in-the-declared-table')
        data = json.loads(Path(a.workload).read_text())
        budget = ap.standing_budget(data, band) if band else None
        if budget is None:
            raise ValueError('round-index-without-a-band')
        problems, cfg = progress.shared.validate_manifest(data)
        if problems or cfg['repeats'] != 1:
            raise ValueError('frozen-native-workload-invalid')
        cfg = dict(cfg, timeouts=dict(cfg['timeouts'], read_s=base.first_advance_read_timeout(cfg)))
        client = NativeRuntimeClient(a.url, a.run_id, epoch=a.epoch, timeout_s=a.control_timeout_s)
        poll = NativeRuntimeClient(a.url, a.run_id, epoch=a.epoch, timeout_s=a.control_timeout_s)
        args = argparse.Namespace(**dict(vars(a), mode='burst'))
        with b.arm_seams(a.size), arm_seams(c0, a.size, outputs, a.run_id):
            result = progress.arm(args, cfg, client, poll_client=poll, budget=budget)
    except Exception as exc:  # noqa: BLE001 - retained, never retried
        result = {'schema': progress.ARM_SCHEMA, 'kind': 'auto', 'mode': 'burst', 'cohort': a.cohort, 'index': a.index,
                  'label': name, 'complete': False, 'reasons': ['arm-refused'], 'error': exception_details(exc)}
    result.update(selection=selection, size=a.size, stage_local=local, c0=c0, part=(declared or {}).get('part'), band=band,
                  workload=ap.BAND_WORKLOAD.get(band), outputs=outputs,
                  standing=dict(greedy=list(T.STANDING_GREEDY), greedy_temperature=T.GREEDY_TEMPERATURE,
                                sampled=list(T.STANDING_SAMPLED), sampled_temperature='manifest'))
    views = {s.get('label'): s for s in result.get('progress') or []}
    if 'after-append' in views:
        # The append is stock by design; the arm's own proof ran before it.
        result['fused_delta_append'] = sa.fused_delta(views['after-append'].get('native_qsa_fused'),
                                                      views['after-fresh'].get('native_qsa_fused'))
        result['group_delta_append'] = b.group_delta(views['after-append'].get('native_layer_groups'),
                                                     views['after-fresh'].get('native_layer_groups'))
        result['stage_delta_append'] = stage_delta(views['after-append'].get(STAGE_CONTROL),
                                                   views['after-fresh'].get(STAGE_CONTROL))
        moved = [k for k in ('fused_delta_append', 'group_delta_append', 'stage_delta_append')
                 if any(v != 0 for v in result[k].values())]
        if moved:
            result['reasons'] = list(result.get('reasons') or []) + ['append-moved-the-fused-group-or-stage-counters']
            result['complete'] = False
    ordered = [s for s in result.get('progress') or [] if s.get('stop_receipts')]
    if len(ordered) >= 2:
        result['stop_delta'] = stop_delta(ordered[-1]['stop_receipts'], ordered[0]['stop_receipts'])
    with open(a.out, 'x') as handle:
        json.dump(result, handle, sort_keys=True)
    print('ALPHA_NEXT_ARM %s %s' % (result['label'], 'COMPLETE' if result.get('complete') else 'INCOMPLETE'))
    return 0 if result.get('complete') is True else 2


# --- engine log lines: stage, stop, anatomy, peak --------------------------------------------------------

def _int(text):
    return int(text) if text is not None and text.lstrip('-').isdigit() else None


def parse_stage(text):
    m = STAGE_RE.search(text)
    if not m:
        return None
    return dict(txn=int(m.group(1)), chunk_start=int(m.group(2)), chunk_end=int(m.group(3)),
                commands=[int(m.group(4)), int(m.group(5))], both_stages_ns=int(m.group(6)),
                headroom_refused=int(m.group(13)), headroom_blocked_ns=_int(m.group(14)))


def parse_stop(text):
    """A stop receipt's JSON body, or None when it does not read or is off schema."""
    try:
        body = json.loads(text[text.index(STOP_LINE) + len(STOP_LINE):])
    except ValueError:
        return None
    return body if type(body) is dict and body.get('schema') == STOP_SCHEMA else None


def is_line(text):
    return STAGE_LINE in text or STOP_LINE in text or ANATOMY_LINE in text or PEAK_LINE in text


def arm_lines(arms, logs):
    """Per arm, the stage, stop, anatomy and peak lines in its span; and the stop lines outside every arm."""
    rows = ap.read_stamped(logs, is_line)
    known = [a for a in arms if a.get('label')]
    out, inside = {}, set()
    for arm in known:
        mine = [(i, t) for i, (s, t) in enumerate(rows) if ap.in_span(s, arm)]
        inside |= {i for i, _ in mine}
        stage = [parse_stage(t) for _, t in mine if STAGE_LINE in t]
        stops = [(parse_stop(t), t) for _, t in mine if STOP_LINE in t]
        peaks = [PEAK_RE.search(t) for _, t in mine if PEAK_LINE in t]
        out[arm['label']] = dict(stage=[x for x in stage if x], torn_stage=sum(1 for x in stage if x is None),
                                 stops=[x for x, _ in stops if x], torn_stops=sum(1 for x, _ in stops if x is None),
                                 anatomy=sum(1 for _, t in mine if ANATOMY_LINE in t),
                                 peaks=[dict(rank=int(m.group(1)), txn=int(m.group(2)), chunk_start=int(m.group(3)),
                                             peak_allocated=int(m.group(5)), peak_reserved=int(m.group(6)),
                                             allocated_at_open=_int(m.group(7))) for m in peaks if m],
                                 torn_peaks=sum(1 for m in peaks if m is None))
    outside = [t for i, (s, t) in enumerate(rows) if STOP_LINE in t and i not in inside]
    return out, dict(stop_lines_outside_every_arm=len(outside))


EMPTY_LINES = dict(stage=[], torn_stage=0, stops=[], torn_stops=0, anatomy=0, peaks=[], torn_peaks=0)


# --- engagement --------------------------------------------------------------------------------------

def stage_reply_breakdown(original):
    """chunk_breakdown with each stage command's replying ranks counted, and C0's join rule for a receipt row that lands after
    the receiver's local release (reported as a release row for the run's-last-flush rule only), as e3_alpha_c0_box does."""
    def breakdown(chunk, workers):
        info = original(chunk, workers)
        foreign = short = 0
        for row in chunk['rows'].get('n', []):
            if row['kind'] == 'segment' and row.get('stage') in TARGETS:
                replied = {rank for rank in range(4) if type(row.get('t_recv%d' % rank)) is int}
                foreign += int(bool(replied - set(TARGETS[row['stage']])))
                short += int(len(replied) != 4)
        late = 0
        for missing in info.get('missing_worker_rows') or ():
            if missing['kind'] == 'receipt':
                missing['kind'] = 'release'
                late += 1
        info.update(foreign_replies=foreign, short_replies=short, late_receipt_rows=info.get('late_receipt_rows', 0) + late)
        return info
    return breakdown


def c0_engagement(arm, row, lines):
    local = arm.get('stage_local')
    delta = arm.get('stage_delta') or {}
    owned = (arm.get('progress_delta') or {}).get('chunks')
    chunks = (row or {}).get('chunks') or []
    stage, torn = lines['stage'], lines['torn_stage']
    if type(owned) is not int or owned < 1 or not delta or not chunks or type(local) is not bool:
        return dict(stage_local=local, engaged=False, reasons=['c0-engagement-receipts-missing'], owned_chunks=owned, delta=delta,
                    stage_lines=len(stage), torn_stage_lines=torn, chunks_without_a_line=None, foreign_replies=None,
                    short_replies=None)
    foreign = sum(c.get('foreign_replies', 0) for c in chunks)
    short = sum(c.get('short_replies', 0) for c in chunks)
    logged = {x['chunk_start'] for x in stage}
    without = sum(1 for c in chunks if c.get('chunk_start') not in logged)
    reasons = []
    if local:
        if delta.get('chunks') != owned:
            reasons.append('stage-chunks-delta-differs-from-the-auto-owned-chunks')
        if delta.get('local_releases') != 2 * owned:
            reasons.append('stage-releases-not-two-per-chunk')
        if len(stage) + torn != owned:
            reasons.append('stage-lines-differ-from-the-auto-owned-chunks')
        if without > torn:
            reasons.append('chunk-without-a-stage-line')
        if foreign:
            reasons.append('non-target-rank-replied-to-a-stage-command')
    else:
        if any(delta.get(k) != 0 for k in STAGE_COUNTERS):
            reasons.append('c0-off-arm-moved-a-stage-counter')
        if stage or torn:
            reasons.append('c0-off-arm-logged-a-stage-line')
        if short:
            reasons.append('c0-off-arm-stage-command-without-four-replies')
    return dict(stage_local=local, engaged=not reasons, reasons=reasons, owned_chunks=owned, delta=delta, stage_lines=len(stage),
                torn_stage_lines=torn, chunks_without_a_line=without, foreign_replies=foreign, short_replies=short)


def engagement(arm, row, lines, declared):
    """Step B's rule at the arm's own size (unchanged, on a view whose selection is g<size>), then C0's both ways."""
    size = arm.get('size')
    groups = STEP_B_ENGAGEMENT(dict(arm, selection='g%d' % size if type(size) is int else None), row)
    c0 = c0_engagement(arm, row, lines)
    reasons = list(groups['reasons'])
    want = declared.get(arm.get('label'))
    if want is None or (arm.get('selection'), size, arm.get('stage_local')) != (want['selection'], want['size'], want['stage_local']):
        reasons.append('arm-not-the-declared-selection')
    reasons += c0['reasons']
    return dict(groups, selection=arm.get('selection'), size=size, stage_local=arm.get('stage_local'), engaged=not reasons,
                reasons=reasons, group_engaged=groups['engaged'], c0=c0)


# --- comparisons (Step B's names, C0 matched pairs) ---------------------------------------------------

def memory_compare(table, rows):
    """Each graded C0-on arm against its matched C0-off arm (same size, cohort and round)."""
    out = {}
    for row in rows:
        partner = T.matched_off(row) if row['c0'] == 'on' and row['part'] in T.GRADED else None
        if not partner:
            continue
        on = (table.get(row['label']) or {}).get('memory', {}).get('worst')
        off = (table.get(partner) or {}).get('memory', {}).get('worst')
        out[row['label']] = None if not on or not off else dict(
            c0_on=on, c0_off=off, on_minus_off={k: round(on[k] - off[k], 1) for k in on if k != 'receipts'})
    return out


def speed_compare(summary_band):
    groups = summary_band['groups']
    out = {}
    for key in sorted(groups):
        selection, _, cohort = key.partition('-')
        parsed = T.parse_selection(selection)
        if not parsed or parsed[0] != 'on':
            continue
        off = groups.get(T.selection('off', parsed[1]) + '-' + cohort)
        if off:
            out[key] = {k: progress.ratio(groups[key].get(k), off.get(k)) for k in (
                'ttft_s', 'prefill_tok_s', 's_per_chunk', 'decode_during_tok_s', 'decode_ratio', 'commands_per_chunk')}
    return out


def c0_decode_delta(table, rows):
    """Live decode ratio of each graded C0-on arm minus its matched C0-off arm. A read, never a verdict input."""
    out = {}
    for row in rows:
        partner = T.matched_off(row) if row['c0'] == 'on' and row['part'] in T.GRADED and row['cohort'] == 'live' else None
        if partner:
            on = (table.get(row['label']) or {}).get('decode_ratio')
            off = (table.get(partner) or {}).get('decode_ratio')
            out[row['label']] = None if on is None or off is None else round(on - off, 4)
    return dict(on_minus_off=out, is_gate=False)


def memory_read(table, torn, arms, rows):
    """Step B's memory read, C0-on against its matched C0-off arm instead of size 1. MET, NOT_MET or INC; never a verdict input."""
    out = {}
    names = {r['label'] for r in rows}
    for row in rows:
        name = row['label']
        established, reasons, evidence, worst = b.arm_memory(name, table, torn, arms)
        cell = dict(established=established, reasons=list(reasons), oom_evidence=evidence, min_free_breach=False,
                    min_free_mib=(worst or {}).get('min_free_mib'), against=None, min_free_below_c0_off_mib=None)
        partner = T.matched_off(row) if row['c0'] == 'on' and row['part'] in T.GRADED else None
        if partner and partner in names:
            ok, other, _, other_worst = b.arm_memory(partner, table, torn, arms)
            cell['against'] = partner
            if not ok:
                cell['reasons'] += ['c0-off-arm-' + x for x in other]
            if established and ok and worst and other_worst:
                below = round(other_worst['min_free_mib'] - worst['min_free_mib'], 1)
                cell['min_free_below_c0_off_mib'] = below
                cell['min_free_breach'] = below > b.MEMORY_FREE_MARGIN_MIB
        cell['verdict'] = 'NOT_MET' if evidence or cell['min_free_breach'] else 'INC' if cell['reasons'] else 'MET'
        out[name] = cell
    return out


# --- numerics envelope --------------------------------------------------------------------------------

def common_prefix(a, b_):
    n = 0
    for x, y in zip(a, b_):
        if x != y:
            break
        n += 1
    return n


def numerics(arms, rows):
    """Each C0-on arm against the spread of its band's graded C0-off arms, on the greedy streams. See NUMERICS_* above."""
    declared = {r['label']: r for r in rows}
    known = [a for a in arms if a.get('label') in declared]
    outside, unproven, bands = [], [], {}
    for band in sorted({declared[a['label']]['band'] for a in known}):
        mine = [a for a in known if declared[a['label']]['band'] == band]
        controls = [a for a in mine if declared[a['label']]['c0'] == 'off' and declared[a['label']]['part'] in T.GRADED]
        candidates = [a for a in mine if declared[a['label']]['c0'] == 'on']
        if not candidates:
            continue
        cell = dict(controls=[a['label'] for a in controls], candidates=[a['label'] for a in candidates], families={}, streams={})
        bands[str(band)] = cell
        if len(controls) < 2:
            unproven.append('band%d-fewer-than-two-controls' % band)
            continue
        for family in ('fresh', 'append'):
            values = [(a.get('outputs') or {}).get(family) for a in controls]
            row = dict(control_values=sorted({json.dumps(v) for v in values if v}), candidates={})
            if not all(values):
                unproven.append('band%d-%s-control-tokens-missing' % (band, family))
            for a in candidates:
                got = (a.get('outputs') or {}).get(family)
                if not got:
                    unproven.append('%s:%s-tokens-missing' % (a['label'], family))
                    row['candidates'][a['label']] = None
                    continue
                inside = json.dumps(got) in row['control_values']
                row['candidates'][a['label']] = inside
                if not inside and all(values):
                    outside.append('%s:%s' % (a['label'], family))
            cell['families'][family] = row
        streams = [str(i) for i in T.STANDING_GREEDY]
        if not any((a.get('outputs') or {}).get('standing') for a in controls + candidates):
            unproven.append('band%d-standing-outputs-missing' % band)
        for stream in streams:
            full = lambda a: [x for x in [((a.get('outputs') or {}).get('standing') or {}).get(stream)]
                              if x and x.get('prefix_tokens') == NUMERICS_PREFIX and type(x.get('prefix')) is list
                              and len(x['prefix']) == NUMERICS_PREFIX]
            heads = [full(a)[0]['prefix'] for a in controls if full(a)]
            row = dict(controls_full=len(heads), control_floor=None, envelope=None, candidates={})
            cell['streams'][stream] = row
            if len(heads) < 2:
                unproven.append('band%d-stream-%s-fewer-than-two-full-controls' % (band, stream))
                continue
            floor = min(common_prefix(x, y) for i, x in enumerate(heads) for y in heads[i + 1:])
            row.update(control_floor=floor, envelope=floor // NUMERICS_FACTOR)
            for a in candidates:
                got = full(a)
                if not got:
                    unproven.append('%s:stream-%s-prefix-short' % (a['label'], stream))
                    row['candidates'][a['label']] = None
                    continue
                nearest = max(common_prefix(got[0]['prefix'], h) for h in heads)
                row['candidates'][a['label']] = dict(nearest_common_prefix=nearest, inside=nearest >= row['envelope'])
                if nearest < row['envelope']:
                    outside.append('%s:stream-%s' % (a['label'], stream))
    verdict = 'OUTSIDE' if outside else 'UNPROVEN' if unproven else 'WITHIN'
    return dict(verdict=verdict, outside=outside, unproven=sorted(set(unproven)), bands=bands, prefix_tokens=NUMERICS_PREFIX,
                factor=NUMERICS_FACTOR, greedy_streams=list(T.STANDING_GREEDY), is_gate=NUMERICS_IS_GATE,
                rule='per band and greedy standing stream (temperature 0): envelope = floor(shortest common prefix over every '
                'pair of graded C0-off controls / %d); a C0-on arm is inside when its common prefix with its nearest control is '
                'at least the envelope; the fresh and append tokens are inside when they are among the controls\' tokens. '
                'Declared before the run (lead, 2 Nov 13:28 and 13:35); %s; not a tensor comparison' % (
                    NUMERICS_FACTOR, 'a gate' if NUMERICS_IS_GATE else 'a read, never a verdict input'))


# --- reads --------------------------------------------------------------------------------------------

def _ratio_read(ratios):
    if not ratios:
        return dict(streams=0, min_stream_ratio=None, mean_stream_ratio=None, lines={str(x): None for x in DECODE_LINES})
    low, mean = min(ratios), statistics.mean(ratios)
    return dict(streams=len(ratios), min_stream_ratio=low, mean_stream_ratio=mean,
                lines={str(x): dict(min_below=low < x, mean_below=mean < x, streams_below=sum(1 for r in ratios if r < x))
                       for x in DECODE_LINES})


def decode_read(arm):
    """4C.7: each standing stream's decode ratio (its rate in the live arm's decode window during the prefill over its own
    decode-only rate, held-f's stream_ratios), and the minimum and mean of those ratios against the 0.8 and 0.9 lines, split by
    stream kind: the sampled streams (the declared decode reading, lead 13:35; the top-level fields), the greedy streams (they
    carry the numerics check; a read here too) and all 8. Stream i is the i-th rate (held-f's order). A read, never a gate."""
    client = arm.get('client_basis') or {}
    ratios = ap.stream_ratios(client.get('decode_window'), arm.get('decode_only')) or []
    whole = len(ratios) == T.STANDING_DECODERS
    sampled = [ratios[i] for i in T.STANDING_SAMPLED] if whole else []
    greedy = [ratios[i] for i in T.STANDING_GREEDY] if whole else []
    out = _ratio_read(sampled)
    out.update(sampled_streams=list(T.STANDING_SAMPLED), greedy_streams=list(T.STANDING_GREEDY), greedy=_ratio_read(greedy),
               all_streams=_ratio_read(ratios), per_stream={str(i): r for i, r in enumerate(ratios)} if whole else {},
               is_gate=False)
    return out


def ring_read(arm, lines):
    stage = lines['stage']
    blocked = [x['headroom_blocked_ns'] for x in stage if x['headroom_blocked_ns'] is not None]
    return dict(headroom_refused_counter=(arm.get('stage_delta') or {}).get('headroom_refused'),
                headroom_refused_lines=sum(x['headroom_refused'] for x in stage), headroom_blocked_ns_lines=sum(blocked),
                headroom_blocked_ns_max_chunk=max(blocked) if blocked else None,
                lines_without_blocked_field=sum(1 for x in stage if x['headroom_blocked_ns'] is None),
                status=arm.get('ring_delta'), is_gate=False)


def stop_read(arm, lines):
    kinds = {k: 0 for k in STOP_KINDS}
    for body in lines['stops']:
        kinds[body.get('kind')] = kinds.get(body.get('kind'), 0) + 1
    return dict(lines=len(lines['stops']), torn_or_off_schema=lines['torn_stops'], by_kind=kinds, status=arm.get('stop_delta'),
                is_gate=False)


def peak_read(lines):
    ranks = {}
    for x in lines['peaks']:
        cell = ranks.setdefault(str(x['rank']), dict(receipts=0, peak_allocated_mib=None, peak_reserved_mib=None))
        cell['receipts'] += 1
        for field, key in (('peak_allocated_mib', 'peak_allocated'), ('peak_reserved_mib', 'peak_reserved')):
            value = round(x[key] / 2 ** 20, 1)
            cell[field] = value if cell[field] is None else max(cell[field], value)
    return dict(ranks=ranks, receipts=len(lines['peaks']), torn=lines['torn_peaks'],
                peak_allocated_mib=max((c['peak_allocated_mib'] for c in ranks.values()), default=None), is_gate=False)


def _reads_of(kinds):
    return sorted({r for k in kinds for r in RECEIPT_READS.get(k, ())})


def receipt_read(logs):
    """PLAN 2 receipt record integrity over the followed logs (receipt_lines, merged from claude/alpha-next): per process the
    counts of missing seq, bad sha, incomplete record and unsafe path; the log-wide rows (legacy lines without identity, the
    startup path); the reads receipt_lines' rule for new queues would have made INC, by process (would_be_inc); a headline.
    RECEIPT_INTEGRITY_IS_GATE False: a read, never a verdict input this run."""
    is_gate = RECEIPT_INTEGRITY_IS_GATE
    try:
        report = receipt_lines.check(receipt_lines.read(logs), require_start=False)
    except (OSError, ValueError) as exc:
        return dict(verdict='UNAVAILABLE', error=type(exc).__name__, is_gate=is_gate, clean=False,
                    headline='RECEIPT INTEGRITY UNAVAILABLE (%s): %s' % (
                        'a gate' if is_gate else 'a read this run, the verdict does not change', type(exc).__name__))
    out = receipt_lines.verdict(report)
    per_process, log_wide = {}, {k: 0 for k in RECEIPT_PROBLEMS}
    legacy_kinds, startup_wide = set(), False
    for name in RECEIPT_PROBLEMS:
        for item in report[name]:
            pid = item.get('pid')
            if pid is None:
                log_wide[name] += 1
                reason = str(item.get('reason', ''))
                if reason.startswith('legacy:'):
                    legacy_kinds.add(reason.split(':', 1)[1])
                else:
                    startup_wide = True
                continue
            per_process.setdefault(str(pid), {k: 0 for k in RECEIPT_PROBLEMS})[name] += 1
    procs = report.get('per_process') or {}
    would = []
    for pid, counts in sorted(per_process.items(), key=lambda kv: int(kv[0])):
        info = procs.get(pid) or {}
        kinds = sorted(info.get('kinds') or {})
        would.append(dict(pid=int(pid), problems=sorted(k for k, v in counts.items() if v), kinds=kinds,
                          sides=info.get('sides') or [], ranks=info.get('ranks') or [], reads=_reads_of(kinds)))
    if legacy_kinds:
        would.append(dict(pid=None, problems=['unsafe_path'], kinds=sorted(legacy_kinds), sides=[], ranks=[],
                          reads=_reads_of(legacy_kinds), reason='receipt lines without identity (the old logger path)'))
    if startup_wide:
        every = sorted(RECEIPT_READS)
        would.append(dict(pid=None, problems=['unsafe_path'], kinds=every, sides=[], ranks=[], reads=_reads_of(every),
                          reason='the startup receipt path is not atomic-fifo'))
    clean = out['verdict'] == 'OK'
    if clean:
        detail = 'all clean, %d records from %d processes' % (report['records'], report['processes'])
    else:
        rows = ['pid %s: %s' % (pid, ', '.join('%s %d' % (k, c[k]) for k in RECEIPT_PROBLEMS))
                for pid, c in sorted(per_process.items(), key=lambda kv: int(kv[0]))]
        if any(log_wide.values()):
            rows.append('log-wide: %s' % ', '.join('%s %d' % (k, log_wide[k]) for k in RECEIPT_PROBLEMS))
        detail = '; '.join(rows)
    out.update(records=report['records'], processes=report['processes'], is_gate=is_gate, clean=clean,
               counts={k: len(report[k]) for k in RECEIPT_PROBLEMS}, per_process=per_process, log_wide_counts=log_wide,
               would_be_inc=would, would_be_inc_reads=sorted({r for w in would for r in w['reads']}),
               headline='RECEIPT INTEGRITY %s (%s): %s' % (
                   'OK' if clean else 'INC', 'a gate' if is_gate else 'a read this run, the verdict does not change', detail))
    return out


# --- the grade ----------------------------------------------------------------------------------------

@contextlib.contextmanager
def grade_seams(rows, lines):
    """The names e3_step_b_box.grade reads (and Step A's signature list), for this one call; restored on exit."""
    names = ('required_labels', 'engagement', 'SELECTIONS', 'ARM_ORDER', 'memory_compare', 'speed_compare', 'size_of',
             'memory_read', 'r3_read')
    saved = {k: getattr(b, k) for k in names}
    signatures = sa.SIGNATURES
    breakdown = progress.chunk_breakdown
    declared = {r['label']: r for r in rows}
    b.required_labels = lambda: [r['label'] for r in rows]
    b.engagement = lambda arm, row: engagement(arm, row, lines.get(arm.get('label')) or EMPTY_LINES, declared)
    b.SELECTIONS = tuple(sorted({r['selection'] for r in rows}))
    b.ARM_ORDER = tuple((r['selection'], r['cohort'], r['index']) for r in rows)
    b.memory_compare = lambda table: memory_compare(table, rows)
    b.speed_compare = speed_compare
    b.size_of = size_of
    b.memory_read = lambda table, torn, arms: memory_read(table, torn, arms, rows)
    b.r3_read = lambda table: c0_decode_delta(table, rows)
    sa.SIGNATURES = signatures + STAGE_SIGNATURES
    progress.chunk_breakdown = stage_reply_breakdown(breakdown)
    band_of_index = ap.band_of_index
    ap.band_of_index = T.band_of_index
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(b, name, value)
        sa.SIGNATURES = signatures
        progress.chunk_breakdown = breakdown
        ap.band_of_index = band_of_index


def grade(arms, ingress, checks, logs, watches=None, part='core'):
    """The core grade (part 'core', the queue's verdict) or the read over every arm (part 'all', never the verdict)."""
    present = {a.get('label') for a in arms} | set(watches or {})
    rows = [r for r in T.arms(None if part == 'all' else T.GRADED) if not r['extra'] or r['label'] in present]
    lines, run_lines = arm_lines(arms, logs)
    with grade_seams(rows, lines):
        result = b.grade(arms, ingress, checks, logs, watches)
    for band in result['summary'].values():
        band['over_c0_off'] = band.pop('over_size_one')
    result['memory_c0_on_vs_off'] = result.pop('memory_over_size_one')
    result['c0_decode_delta'] = result.pop('r3_read')
    known = {a['label']: a for a in arms if a.get('label')}
    decode, ring, stops, peaks, anatomy = {}, {}, {}, {}, {}
    for row in rows:
        name = row['label']
        arm = known.get(name)
        cell = result['table'].get(name)
        if cell is not None:
            cell.update(stage_local=row['stage_local'], c0=row['c0'], part=row['part'])
        if arm is None:
            continue
        mine = lines.get(name) or EMPTY_LINES
        decode[name] = decode_read(arm)
        if cell is not None:
            cell.update(mean_stream_ratio=decode[name]['mean_stream_ratio'], decode_streams=decode[name]['streams'])
        ring[name] = ring_read(arm, mine)
        stops[name] = stop_read(arm, mine)
        peaks[name] = peak_read(mine)
        anatomy[name] = mine['anatomy']
    compared = numerics(list(known.values()), rows)
    failures, missing = set(result['failures']), set(result['missing'])
    classes = set(result['classification'] or [])
    if NUMERICS_IS_GATE and compared['verdict'] == 'OUTSIDE':
        failures.add('numerics-outside-the-control-envelope:' + ','.join(compared['outside']))
        classes.add('ENGINE')
    elif NUMERICS_IS_GATE and compared['verdict'] == 'UNPROVEN':
        missing.update('numerics:' + x for x in compared['unproven'])
    receipts = receipt_read(logs)
    if RECEIPT_INTEGRITY_IS_GATE and not receipts['clean']:
        missing.add('receipt-integrity:' + receipts['verdict'])
    verdict = 'FAIL' if failures else 'INC' if missing else 'PASS'
    # Lead, 14:29: OUTSIDE is shown prominently in the grade and the report, and changes no verdict while NUMERICS_IS_GATE is
    # False; this run establishes the envelope (the graded C0-off greedy arms against each other, per band and stream).
    headline = None
    if compared['verdict'] != 'WITHIN':
        headline = 'NUMERICS %s (%s): %s' % (
            compared['verdict'], 'a gate' if NUMERICS_IS_GATE else 'a read this run, the verdict does not change',
            ', '.join(compared['outside'] + compared['unproven']))
    if verdict != 'FAIL':
        classes = set()
    result.update(failures=sorted(failures), missing=sorted(missing), verdict=verdict,
                  classification=sorted(classes) or (['HARNESS'] if verdict == 'FAIL' else None))
    result.update(schema=GRADE_SCHEMA if part == 'core' else READ_SCHEMA, part=part, is_queue_verdict=part == 'core',
                  numerics=compared, numerics_verdict=compared['verdict'], numerics_outside=list(compared['outside']),
                  numerics_headline=headline, receipt_integrity=receipts, receipt_integrity_is_gate=RECEIPT_INTEGRITY_IS_GATE,
                  receipt_integrity_headline=receipts['headline'], receipt_integrity_clean=receipts['clean'], decode_read=decode, decode_lines=list(DECODE_LINES), decode_is_gate=False, c0_ring=ring,
                  stop_receipts=dict(arms=stops, schema=STOP_SCHEMA, **run_lines), memory_peak=peaks, anatomy_lines=anatomy,
                  arm_table=dict(band_size=T.BAND_SIZE, arms=rows, standing_decoders=T.STANDING_DECODERS,
                                 band1_rounds=T.BAND1_ROUNDS, band1_required_rounds=T.BAND1_REQUIRED_ROUNDS,
                                 extra_not_entered=[r['label'] for r in T.arms('extra') if r['label'] not in present],
                                 standing_greedy=list(T.STANDING_GREEDY), greedy_temperature=T.GREEDY_TEMPERATURE,
                                 standing_sampled=list(T.STANDING_SAMPLED)),
                  basis='one engine, fused selected in every arm, one group size and one C0 selection per arm by typed control; '
                        'band 1 ABBA at 65,445 tokens, band 2 matched at 131,094 past the 97,152-row arena; per chunk, the '
                        'critical rank of each native command; live arms count chunks begun with all 8 decoders, solo arms '
                        'chunks begun with none',
                  scope='C0 stage-local against serial on the composed Step A + Step B engine (alpha-next), with speed, decode '
                        'and memory reported and never gated; gains from separate experiments are never multiplied')
    if part != 'core':
        result['read_only'] = 'a read over every arm, g4 tail included; the queue verdict is the core grade (alpha-next-grade.json)'
    return result


# --- the report ----------------------------------------------------------------------------------------

def _cell(value, digits=1):
    return sa._cell(value, digits)


def report(result):
    """Three separate headed sections. Numbers come only from the graded record."""
    cell = _cell
    core = result['part'] == 'core'
    title = '# q18a-alpha-next report' if core else '# q18a-alpha-next read (every arm, g4 tail included; not the queue verdict)'
    lines = [title, '', 'Overall: **%s**%s' % (
        result['verdict'], ' (%s)' % ', '.join(result['classification']) if result['classification'] else ''), '']
    if not core:
        lines += [result['read_only'], '']
    headline = result.get('numerics_headline')
    if headline:
        lines += ['**%s**' % headline, '']
    receipts = result.get('receipt_integrity') or {}
    if receipts.get('headline'):
        lines += ['**%s**' % receipts['headline'], '']
    for row in receipts.get('would_be_inc') or []:
        lines.append('- receipt_lines rule would make INC: process %s (%s; kinds %s) feeds %s' % (
            'log-wide' if row['pid'] is None else row['pid'], ', '.join(row['problems']), ','.join(row['kinds']) or '-',
            ', '.join(row['reads']) or 'no grade read'))
    if receipts.get('would_be_inc'):
        lines.append('')
    lines += ['## Mechanism', '']
    lines += ['| arm | part | verdict | class | crash-free | disagree | signatures | engaged | owned chunks | '
              'fused cmds | group chunks | group cmds | layer cmds per chunk | stage chunks | releases | stage lines | foreign '
              'replies | reasons |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for name in result['required_arms']:
        v = result['validity'].get(name) or {}
        e = v.get('engagement') or {}
        d, g, c = e.get('delta') or {}, e.get('groups') or {}, e.get('c0') or {}
        lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s-%s | %s | %s | %s | %s | %s |' % (
            name, (T.by_label().get(name) or {}).get('part'), v.get('verdict'), v.get('classification') or '-', v.get('crash_free'),
            cell(v.get('disagree_lines')), ','.join(v.get('signatures') or []) or '0', e.get('engaged'), cell(e.get('owned_chunks')),
            cell(d.get('fused_commands')), cell(g.get('chunks')), cell(g.get('group_commands')), cell(e.get('layer_commands_min')),
            cell(e.get('layer_commands_max')), cell((c.get('delta') or {}).get('chunks')), cell((c.get('delta') or {}).get('local_releases')),
            cell(c.get('stage_lines')), cell(c.get('foreign_replies')), '; '.join(v.get('reasons') or []) or '-'))
    compared = result['numerics']
    lines += ['', 'Numerics (%s): **%s** on greedy streams %s. %s.' % (
        'a gate' if compared['is_gate'] else 'a read, not a verdict input', compared['verdict'],
        ','.join(str(x) for x in compared['greedy_streams']), compared['rule'])]
    if compared['outside']:
        lines.append('Outside: ' + ', '.join(compared['outside']))
    if compared['unproven']:
        lines.append('Unproven: ' + ', '.join(compared['unproven']))
    for band, row in sorted(compared['bands'].items()):
        floors = {s: (r['control_floor'], r['envelope']) for s, r in sorted(row['streams'].items())}
        lines.append('Band %s: controls %s; per stream (control floor, envelope) %s.' % (
            band, ', '.join(row['controls']) or 'none', json.dumps(floors, sort_keys=True)))
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
              '| arm | torn progress | torn memory | torn stage | lost flushes | lost records | dropped chunks | within limits |',
              '|---|---|---|---|---|---|---|---|']
    for name in result['required_arms']:
        t = (result['validity'].get(name) or {}).get('tears')
        c = ((result['validity'].get(name) or {}).get('engagement') or {}).get('c0') or {}
        if t:
            lines.append('| %s | %s | %s | %s | %s | %s | %s | %s |' % (
                name, t['torn_progress_lines'], t['torn_memory_lines'], cell(c.get('torn_stage_lines')), len(t['lost_flushes']),
                t['lost_records'], ','.join(str(x) for x in t['dropped_chunk_starts']) or '0', t['within_limits']))
    lines += ['', '### C0 ring reading (descriptive, never a verdict input)', '',
              '| arm | refused (counter) | refused (lines) | blocked ms (lines) | max chunk blocked ms | blocked ms (status) | '
              'episodes | max episode ms | limiting | limiting sole |', '|---|---|---|---|---|---|---|---|---|---|']
    ms = lambda ns: None if ns is None else ns / 1e6
    for name in result['required_arms']:
        r = result['c0_ring'].get(name)
        if not r:
            continue
        s = r['status'] or {}
        lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            name, cell(r['headroom_refused_counter']), r['headroom_refused_lines'], cell(ms(r['headroom_blocked_ns_lines']), 3),
            cell(ms(r['headroom_blocked_ns_max_chunk']), 3), cell(ms(s.get('blocked_ns')), 3), cell(s.get('episodes')),
            cell(ms(s.get('blocked_max_ns_after')), 3), json.dumps(s.get('limiting'), sort_keys=True),
            json.dumps(s.get('limiting_sole'), sort_keys=True)))
    stops = result['stop_receipts']
    lines += ['', '### Stop receipts (item 1; descriptive, never a verdict input)', '',
              'Schema %s. Stop lines outside every arm: %d.' % (stops['schema'], stops['stop_lines_outside_every_arm']), '',
              '| arm | lines | partial replays | bound exhausted | torn or off schema | status partial replays | status bound exhausted | '
              'anatomy lines |', '|---|---|---|---|---|---|---|---|']
    for name in result['required_arms']:
        s = stops['arms'].get(name)
        if not s:
            continue
        status = s['status'] or {}
        lines.append('| %s | %s | %s | %s | %s | %s | %s | %s |' % (
            name, s['lines'], s['by_kind'].get('stop_partial_replay', 0), s['by_kind'].get('stop_bound_exhausted', 0),
            s['torn_or_off_schema'], cell(status.get('partial_replays')), cell(status.get('bound_exhausted')),
            result['anatomy_lines'].get(name, 0)))
    lines += ['', '## Performance', '', 'Reported, never gated: speed, decode and memory are reads (operator, 2 Nov).', '',
              '### Decode per arm, per-stream ratio against the %s lines' % ' and '.join(str(x) for x in result['decode_lines']), '',
              'Each standing stream\'s ratio is its decode rate during the prefill over its own decode-only rate. Sampled: the '
              'declared decode reading; greedy: the streams that carry the numerics check; then the minimum and mean over all 8. '
              'Reads, never gates.', '',
              '| arm | sampled streams | sampled min | sampled mean | min < 0.8 | mean < 0.8 | streams < 0.8 | min < 0.9 | '
              'mean < 0.9 | streams < 0.9 | greedy min | greedy mean | greedy < 0.8 | greedy < 0.9 | all-8 min | all-8 mean |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for name in result['required_arms']:
        d = result['decode_read'].get(name)
        if not d:
            continue
        a, c = d['lines'].get('0.8') or {}, d['lines'].get('0.9') or {}
        g = d.get('greedy') or _ratio_read([])
        lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            name, d['streams'], cell(d['min_stream_ratio'], 3), cell(d['mean_stream_ratio'], 3), a.get('min_below', '-'),
            a.get('mean_below', '-'), a.get('streams_below', '-'), c.get('min_below', '-'), c.get('mean_below', '-'),
            c.get('streams_below', '-'), cell(g['min_stream_ratio'], 3), cell(g['mean_stream_ratio'], 3),
            (g['lines'].get('0.8') or {}).get('streams_below', '-'), (g['lines'].get('0.9') or {}).get('streams_below', '-'),
            cell(d['all_streams']['min_stream_ratio'], 3),
            cell(d['all_streams']['mean_stream_ratio'], 3)))
    lines += ['', 'Live decode ratio, C0-on minus its matched C0-off arm: %s (a read).' % json.dumps(
        result['c0_decode_delta']['on_minus_off'], sort_keys=True), '']
    for band in (1, 2):
        lines += ['### Band %d' % band, '', '| arm | TTFT s | prefill tok/s | s/chunk | cmds/chunk | decode during | decode only | '
                  'ratio | p10 stream ratio | gap p99 ms | gap max ms |', '|---|---|---|---|---|---|---|---|---|---|---|']
        for name in result['required_arms']:
            row = result['table'].get(name)
            if not row or row.get('band') != band:
                continue
            lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
                name, cell(row.get('ttft_s')), cell(row.get('prefill_tok_s'), 0), cell(row.get('s_per_chunk'), 3),
                cell(row.get('commands_per_chunk')), cell(row.get('decode_during_tok_s')), cell(row.get('decode_only_tok_s')),
                cell(row.get('decode_ratio'), 3), cell(row.get('p10_stream_ratio'), 3), cell(row.get('gap_p99_ms')),
                cell(row.get('gap_max_ms'))))
        lines += ['', 'C0-on over C0-off, medians: ' + json.dumps(result['summary']['band%d' % band]['over_c0_off'], sort_keys=True), '']
    lines += ['### Memory read, per arm', '',
              'Step B\'s rule with the matched C0-off arm in place of size 1: 0 OOMs, and a C0-on arm\'s min free no lower than its '
              'matched C0-off arm\'s by more than %s MiB; a torn or lost memory receipt makes the read INC. Not a verdict input; an '
              'OOM fails the arm as ENGINE (Step B\'s rule). Peak allocated per chunk is the engine\'s memory-peak receipt '
              '(recorded, never gating).' % result['memory_free_margin_mib'], '',
              '| arm | read | min free MiB | against | below C0-off MiB | OOM evidence | peak alloc MiB (memory-peak) | reasons |',
              '|---|---|---|---|---|---|---|---|']
    for name, read in result['memory_read'].items():
        peak = (result['memory_peak'].get(name) or {}).get('peak_allocated_mib')
        lines.append('| %s | %s | %s | %s | %s | %s | %s | %s |' % (
            name, read['verdict'], read['min_free_mib'], read['against'] or '-', read['min_free_below_c0_off_mib'],
            ','.join(read['oom_evidence']) or '0', cell(peak), '; '.join(read['reasons']) or '-'))
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
    r.add_argument('--stage-local', choices=tuple(T.C0), required=True, help='this arm\'s C0 selection: on (stage-local) or off')
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
        g.add_argument('--report', help='write the three-section report here')
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
    print('ALPHA_NEXT_%s %s' % ('GRADE' if a.action == 'grade' else 'READ', result['verdict']))
    if a.action == 'read':
        return 0 if result['verdict'] == 'PASS' else 2      # a read: never 1, the queue does not count it
    return {'PASS': 0, 'FAIL': 1}.get(result['verdict'], 2)


if __name__ == '__main__':
    raise SystemExit(main())
