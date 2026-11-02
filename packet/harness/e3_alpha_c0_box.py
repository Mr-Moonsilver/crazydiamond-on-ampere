#!/usr/bin/env python3
"""C0 box arms and grade: stage-local progress against the frozen serial path, matched per arm on one engine.

The arm is held-f's (e3_native41_progress_box.arm), called unchanged with mode burst; the seams below add one typed control and
two output receipts and change no other arm step. Control order: pause, native_progress {burst}, native_stage_local {local},
release_auto_arm, resume, then status only. The record keeps ``mode`` burst (the progress lines carry it) and adds ``selection``.

One band, alpha-perf's band 1 (65,445 fresh then the append, 8 standing decoders), in ABBA over two rounds. The round index is
part of the cache salt through the label, so no arm can hit another arm's prefix.

The grade is alpha-perf's progress join and table (e3_alpha_perf_box) plus:
  mechanism   per arm and overall PASS/FAIL/INC, HARNESS or ENGINE on a failure: crash-free, 0 ``disagree``, 0 stage-path
              contract errors, and engagement. A local arm needs a ``chunks`` delta equal to the auto-owned chunk count, two local
              releases per chunk, one ``flash-s5 stage-local`` line per chunk and no reply from a non-target rank on any stage
              command. A serial arm needs every counter delta 0, no such line, and all four replies on every stage command. An
              arm that did not engage is INC.
  numerics    the generated token of the fresh prefill and of the append (one sampled token each, same prompt and seed in every
              arm), and a digest of each standing stream's first tokens, compared across arms. FAIL ENGINE when a local arm
              differs while the serial arms agree; INC when the serial arms already differ, which is noise this queue cannot
              separate. Declared limit: this is not a tensor comparison; the per-rank work comparison is tools/c0_bit_identity.py.
  performance reported, never gated: alpha-perf's table, per-stream decode ratios, commands per chunk per stage, both-stages-in-
              flight per chunk, ring headroom refusals. Owner time, worker main-thread occupancy and the inactive-stage tax are
              tools/alpha_c0_report.py over the same record.
  memory      per arm from the `flash-s5 memory` receipts: peak allocated, peak reserved, min free, allocator retries and OOMs,
              local against the matched serial arm. The proposed gates are reported as met or not: 0 OOMs and min free no lower
              than the matched serial arm's by more than 2 MiB. A real OOM warning is a counted line; retries are descriptive.
  tears       the docker log can splice one writer's bytes into another's line (q18a-alpha-perf: 2 of 22,063 progress lines). A
              torn line or a lost flush is counted per arm, the chunks it touches drop out of the breakdown and stay listed, and
              the arm goes INC only above LOST_RECORD_LIMIT or DROPPED_CHUNK_LIMIT, declared and argued where they are defined. A
              signature inside a torn line still counts. No line is repaired.
  report      three separate headed sections: mechanism and numerics, performance, release readiness.

Receipts hold counts, hashes, identities and host clock stamps, never prompts or auth.
"""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import re
import statistics

import e3_alpha_perf_box as ap
from e3_native_qualification_driver import exception_details
from e3_native_runtime_client import NativeRuntimeClient

progress = ap.progress
base = ap.base
GRADE_SCHEMA = 'alpha-c0-grade/1'
SELECTIONS = ('serial', 'local')
ARM_ORDER = (('serial', 'solo', 0), ('local', 'solo', 0), ('local', 'live', 0), ('serial', 'live', 0),
             ('serial', 'live', 1), ('local', 'live', 1), ('local', 'solo', 1), ('serial', 'solo', 1))
INDEXES = (0, 1)
STAGE_COUNTERS = ('chunks', 'headroom_refused', 'local_releases', 'both_stages_ns', 'selections')
CONTROL_SEQUENCE = ['status', 'pause', 'native_progress', 'native_stage_local', 'release_auto_arm', 'resume']
TARGETS = {0: (0, 1), 1: (2, 3)}
STREAM_PREFIX_TOKENS = 256
MEMORY_FREE_MARGIN_MIB = 2.0
# The stage path's contract errors in the followed log. Each stops the arm row at FAIL, classification ENGINE.
SIGNATURES = (('stage-one-chunk', 'stage-local progress runs one chunk'), ('stage-commit', 'stage-local commit without'),
              ('stage-release', 'local release lacks'), ('stage-scope', 'a chunk command needs a stage scope'),
              ('stage-target', 'a stage command targets another stage'), ('stage-nest', 'a stage scope does not nest'),
              ('stage-open', 'open reply does not cover every stage rank'),
              ('stage-retained', 'a retained PP send already holds'), ('stage-receipt', 'a receipt delivery spans retained'),
              ('stage-channel', 'a retained send has no PP channel'), ('stage-skip', 'a stage skip never leaves'))
STAGE_LINE = 'flash-s5 stage-local '
STAGE_RE = re.compile(r'flash-s5 stage-local txn=(\d+) chunk=(\d+)-(\d+) commands=(\d+),(\d+) both_stages_ns=(\d+) '
                      r'first_ns=(\S+),(\S+) last_ns=(\S+),(\S+) local_release_ns=(\S+),(\S+) headroom_refused=(\d+)')
# The whole `flash-s5 memory` receipt, both points. ap.MEMORY_LINE stops at free=; the same words.
MEMORY_LINE = re.compile(r'flash-s5 memory point=(\w+) rank=(\d+) txn_seq=(\d+) chunk=(\d+)-(\d+) allocated=(-?\d+) '
                         r'reserved=(-?\d+) inactive_split=(-?\d+) alloc_retries=(-?\d+) ooms=(-?\d+) free=(-?\d+)')
RELEASE_READINESS = ('Not assessed by this queue. It holds no lifecycle, tensor numerics or leak evidence; nothing in this report '
                     'supports a release decision.')


def label(selection, cohort, index):
    return '%s-%s-%d' % (selection, cohort, index)


def required_labels():
    return [label(*x) for x in ARM_ORDER]


# --- the arm --------------------------------------------------------------------------------------------

def stage_view(status):
    block = status.get('native_stage_local') if type(status) is dict else None
    if type(block) is not dict or type(block.get('counters')) is not dict:
        return None
    return dict({k: block['counters'].get(k) for k in STAGE_COUNTERS}, selected=block.get('selected'),
                ring_reserve=block.get('ring_reserve'), poll_reserve=block.get('poll_reserve'))


def stage_delta(after, before):
    return {k: progress.delta(after, before, k) for k in STAGE_COUNTERS}


def release_selected(local):
    """held-f's release with the selection between native_progress and release_auto_arm; its checks, then the reply's."""
    def release(client, record, mode):
        reply = client.pause()
        record['release_pause'] = {'route': reply.get('route'), 'route_generation': reply.get('route_generation')}
        try:
            got = client.call('native_progress', mode=mode)
            record['mode_switch'] = {k: got.get(k) for k in ('mode', 'previous', 'installed', 'route',
                                                             'route_generation', 'engine_epoch')}
            got = client.call('native_stage_local', local=local)
            record['selection_switch'] = {k: got.get(k) for k in ('local', 'previous', 'route', 'route_generation',
                                                                  'engine_epoch')}
            got = client.call('release_auto_arm')
            record['release'] = {k: got.get(k) for k in ('released', 'engine_epoch')}
        finally:
            record['release_resume'] = client.resume().get('resumed')
        switch = record['mode_switch']
        if switch.get('mode') != mode or switch.get('installed') != 'burst' or switch.get('route') != 'full':
            raise ValueError('native-progress-mode-unproved')
        selection = record['selection_switch']
        if selection.get('local') is not local or selection.get('route') != 'full':
            raise ValueError('native-stage-local-selection-unproved')
        if record['release'].get('released') is not True:
            raise ValueError('auto-arm-release-unproved')
    return release


def selection_reasons(record, local):
    """The selection's receipts beside held-f's. Engagement is the grade's, on the same numbers."""
    reasons = []
    history = record.get('control_history') or []
    if [x for i, x in enumerate(history) if x != 'status' or i == 0][:6] != CONTROL_SEQUENCE:
        reasons.append('control-sequence-not-pause-progress-selection-release-resume')
    if history.count('native_stage_local') != 1:
        reasons.append('arm-sent-the-selection-more-than-once')
    views = {s.get('label'): s for s in record.get('progress') or []}
    before = (views.get('before-fresh') or {}).get('native_stage_local')
    after = (views.get('after-fresh') or {}).get('native_stage_local')
    if not before or not after:
        reasons.append('stage-status-missing')
        return reasons
    if before.get('selected') is not local or after.get('selected') is not local:
        reasons.append('stage-selection-not-this-arm')
    record['stage_delta'] = stage_delta(after, before)
    if record['stage_delta']['selections'] != 0:
        reasons.append('stage-selection-moved-during-the-fresh-prompt')
    return reasons


def stream_digests(states):
    """Per standing stream: tokens seen and the digest of the first STREAM_PREFIX_TOKENS, when it got that far."""
    rows = {}
    for state in states:
        tokens = list(state.tokens)
        head = tokens[:STREAM_PREFIX_TOKENS]
        rows[str(state.index)] = dict(tokens=len(tokens), prefix_tokens=len(head),
                                      prefix_sha256=hashlib.sha256(json.dumps(head).encode()).hexdigest())
    return rows


@contextlib.contextmanager
def arm_seams(selection, outputs):
    """The names held-f's arm reads, for this one process's one arm; restored on exit."""
    local = selection == 'local'
    saved = {k: getattr(progress, k) for k in ('label', 'release', 'progress_reasons', 'progress_view', 'ACTIONS')}
    actions = base.ARM_ACTIONS
    request_row = base.request_row
    stop = progress.streams._stop_streams
    original_reasons, original_view = saved['progress_reasons'], saved['progress_view']

    def reasons(record, mode, cohort):
        history = record.get('control_history') or []
        view = dict(record, control_history=[x for x in history if x != 'native_stage_local'])
        out = original_reasons(view, mode, cohort)
        if 'progress_delta' in view:
            record['progress_delta'] = view['progress_delta']
        return out + selection_reasons(record, local)

    def view(status):
        row = original_view(status)
        if row is None:
            return None
        block = stage_view(status)
        if block is None:
            raise ValueError('native-stage-local-not-installed')
        return dict(row, native_stage_local=block)

    def row_with_output(state, first, prompts, family):
        outputs[family] = list(state.tokens)[:8]
        return request_row(state, first, prompts, family)

    def stop_with_digests(requests, states, workload):
        if 'standing' not in outputs and len(states) > 1:
            outputs['standing'] = stream_digests(states)
        return stop(requests, states, workload)
    progress.label = lambda _mode, cohort, index: label(selection, cohort, index)
    progress.release = release_selected(local)
    progress.progress_reasons = reasons
    progress.progress_view = view
    progress.ACTIONS = saved['ACTIONS'] | {'native_stage_local'}
    base.ARM_ACTIONS = dict(actions, auto=actions['auto'] | {'native_stage_local'})
    base.request_row = row_with_output
    progress.streams._stop_streams = stop_with_digests
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(progress, name, value)
        base.ARM_ACTIONS = actions
        base.request_row = request_row
        progress.streams._stop_streams = stop


def run_arm(a):
    band = ap.band_of_index(a.index)
    selection = a.mode
    outputs = {}
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
        with arm_seams(selection, outputs):
            result = progress.arm(args, cfg, client, poll_client=poll, budget=budget)
    except Exception as exc:  # noqa: BLE001 - retained, never retried
        result = {'schema': progress.ARM_SCHEMA, 'kind': 'auto', 'mode': 'burst', 'cohort': a.cohort, 'index': a.index,
                  'label': label(selection, a.cohort, a.index), 'complete': False,
                  'reasons': ['arm-refused'], 'error': exception_details(exc)}
    result['selection'] = selection
    result['band'] = band
    result['workload'] = ap.BAND_WORKLOAD[band]
    result['outputs'] = outputs
    views = {s.get('label'): s for s in result.get('progress') or []}
    if 'after-append' in views:
        # The append is stock by design; the arm's own proof ran before it.
        result['stage_delta_append'] = stage_delta(views['after-append'].get('native_stage_local'),
                                                   views['after-fresh'].get('native_stage_local'))
        if any(v != 0 for v in result['stage_delta_append'].values()):
            result['reasons'] = list(result.get('reasons') or []) + ['append-moved-the-stage-counters']
            result['complete'] = False
    with open(a.out, 'x') as handle:
        json.dump(result, handle, sort_keys=True)
    print('ALPHA_C0_ARM %s %s' % (result['label'], 'COMPLETE' if result.get('complete') else 'INCOMPLETE'))
    return 0 if result.get('complete') is True else 2


# --- engine log ----------------------------------------------------------------------------------------

def signature_of(line):
    """A contract error in a followed line. A whole progress line holds none; a torn one can hold another writer's bytes."""
    marked = progress.PROGRESS_LINE in line
    if marked and not is_torn_progress(line):
        return None
    whole = INNER_HEADER.sub('', line)
    for name, text in SIGNATURES:
        if text in whole:
            return name
    # ap.is_disagreement sees neither a marked line nor a word a header split.
    if 'disagree' in whole and (marked or 'disagree' not in line):
        return 'tp-disagree-in-a-torn-line'
    return None


def is_tracked(line):
    return ap.is_tracked(line) or STAGE_LINE in line or signature_of(line) is not None


def _maybe_int(text):
    return int(text) if text.lstrip('-').isdigit() else None


def stage_lines(rows):
    """The per-chunk ``flash-s5 stage-local`` lines in one arm's span, by chunk."""
    out = []
    for _, text in rows:
        m = STAGE_RE.search(text)
        if not m:
            continue
        release = _maybe_int(m.group(11)), _maybe_int(m.group(12))
        out.append(dict(txn=int(m.group(1)), chunk_start=int(m.group(2)), chunk_end=int(m.group(3)),
                        commands=[int(m.group(4)), int(m.group(5))], both_stages_s=int(m.group(6)) / 1e9,
                        headroom_refused=int(m.group(13)),
                        release_skew_s=abs(release[1] - release[0]) / 1e9 if None not in release else None))
    return out


def stage_summary(lines):
    if not lines:
        return dict(lines=0, both_stages_s_per_chunk=None, commands_per_chunk=None, headroom_refused=0, release_skew_s=None)
    return dict(lines=len(lines), both_stages_s_per_chunk=statistics.median(x['both_stages_s'] for x in lines),
                commands_per_chunk=[statistics.median(x['commands'][stage] for x in lines) for stage in (0, 1)],
                headroom_refused=sum(x['headroom_refused'] for x in lines),
                release_skew_s=statistics.median(x['release_skew_s'] for x in lines if x['release_skew_s'] is not None)
                if any(x['release_skew_s'] is not None for x in lines) else None)


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


# --- torn lines and lost flushes -----------------------------------------------------------------------

# Declared before any run. q18a-alpha-perf's 22,063 progress lines (7,878 flushes, 2.8 lines per flush) held 2 tears, each losing
# one worker flush. A C0 arm is about 64 chunks and about 576 flushes on the local path (one engine, two per rank per chunk). At
# that rate an arm expects about 0.08 lost flushes, 0.16 if the local path's extra flush events double it, so limit 0 would turn
# about half of the eight-arm runs INC for a log artefact, while limit 1 leaves about 1 % of local arms INC. A share of flushes is
# not used: 0.1 % of 576 is below one flush, so it would mean limit 0. Damage is bounded separately. The two real losses touched
# one and two chunks (with 29 of the 30 lost rows in one chunk), so 2 of 64 chunks is what one lost flush may cost.
# Above either limit the arm is INC: the log no longer shows what it claims to, whatever the cause.
LOST_RECORD_LIMIT = 1
DROPPED_CHUNK_LIMIT = 2
# Each process writes its header and its body separately, so a header alone landing in another line is common and harmless (525
# lines in that run). Only a body that no longer reads is a tear.
INNER_HEADER = re.compile(r'\((?:Worker_\w+|EngineCore\w*) pid=\d+\) ?')
INCOMPLETE = re.compile(r'line-parts-incomplete:(\w+):(\w+):(\d+)')
MEMORY_MARK = 'flash-s5 memory '


def is_torn_progress(line):
    """The body no longer reads, or reads with another writer's header in it: inside a JSON string a splice still parses."""
    at = line.find(progress.PROGRESS_LINE)
    if at < 0:
        return False
    body = line[at + len(progress.PROGRESS_LINE):]
    if INNER_HEADER.search(body):
        return True
    try:
        json.loads(body)
    except ValueError:
        return True
    return False


def whole_records(records, errors, torn):
    """The join's input without the torn lines: those that still parse leave its records, those that did not leave its errors.

    Returns (records, other errors, how many torn lines failed to parse), so the caller can check the two readers agree.
    """
    spliced, off_schema, unparsable = set(), 0, 0
    for _, text in torn:
        try:
            body = json.loads(text[text.find(progress.PROGRESS_LINE) + len(progress.PROGRESS_LINE):])
        except ValueError:
            unparsable += 1
            continue
        if type(body) is dict and body.get('schema') == progress.PROGRESS_SCHEMA:
            spliced.add(tuple(str(body.get(k)) for k in ('side', 'rank', 'line', 'part', 'base_ns')))
        else:
            off_schema += 1
    kept = [r for r in records if tuple(str(r[1].get(k)) for k in ('side', 'rank', 'line', 'part', 'base_ns')) not in spliced]
    others = [e for e in errors if e != 'unparsable-progress-line']
    for _ in range(off_schema):
        others.remove('progress-line-schema')
    return kept, others, unparsable


def is_torn(line):
    return (is_torn_progress(line) or (STAGE_LINE in line and not STAGE_RE.search(line))
            or (MEMORY_MARK in line and not MEMORY_LINE.search(line)))


def torn_identity(line):
    """(side, rank, line) from the text before the rows, which a tear may have left readable; else None."""
    head, found, _ = line[line.find(progress.PROGRESS_LINE):].partition('"rows":')
    number = re.search(r'"line":(\d+)', head) if found else None
    if not number:
        return None
    rank = re.search(r'"rank":(\d+)', head)
    return ('worker', int(rank.group(1)), int(number.group(1))) if rank else ('engine', None, int(number.group(1)))


def torn_lines(paths):
    """The followed lines that no longer read, by what they were: (stamp, text) lists."""
    rows = ap.read_stamped(paths, is_torn)
    return dict(progress=[r for r in rows if is_torn_progress(r[1])],
                stage=[r for r in rows if STAGE_LINE in r[1] and not STAGE_RE.search(r[1])],
                memory=[r for r in rows if MEMORY_MARK in r[1] and not MEMORY_LINE.search(r[1])])


def flush_ledger(records, torn):
    """Every flush the log lost, from three independent signs: a torn line, a flush short of parts, a gap in a writer's lines.

    A writer numbers its flushes without a gap (all 7,878 in q18a-alpha-perf but the two lost), so a tear that destroyed even
    the marker still shows. A torn line whose identity does not read, or reads as a flush that is present, counts as its own
    lost flush, which can count one loss twice and only ever errs toward INC.
    """
    flushes, problems = progress.assemble(records)
    present = {}
    for f in flushes:
        present.setdefault((f['side'], f['rank']), {})[f['line']] = f
    survivors = {}
    for stamp, body in records:
        if stamp is not None:
            survivors.setdefault((body.get('side'), body.get('rank'), body.get('line')), stamp)
    lost, unidentified = {}, []

    def note(key, stamps, evidence, exact):
        row = lost.setdefault(key, dict(side=key[0], rank=key[1], line=key[2], stamps=[], evidence=[], exact=False))
        row['stamps'] = sorted(set(row['stamps']) | {x for x in stamps if x is not None})
        row['evidence'] = sorted(set(row['evidence']) | {evidence})
        row['exact'] = row['exact'] or exact
    for stamp, text in torn:
        key = torn_identity(text)
        if key is None or key[2] in present.get(key[:2], {}):
            unidentified.append(stamp)
        else:
            note(key, [stamp], 'torn-line', True)
    for problem in problems:
        m = INCOMPLETE.fullmatch(problem)
        if m:
            key = (m.group(1), None if m.group(2) == 'None' else int(m.group(2)), int(m.group(3)))
            note(key, [survivors.get(key)], 'parts-incomplete', True)
    for who, lines in present.items():
        numbers = sorted(lines)
        for a, b in zip(numbers, numbers[1:]):
            for number in range(a + 1, b):
                note(who + (number,), [lines[a]['stamp'], lines[b]['stamp']], 'line-gap', False)
    seqs = {}
    for (side, rank), lines in present.items():
        if side == 'worker':
            seqs[rank] = {r['seq'] for f in lines.values() for r in f['rows'].get('r', [])}
    for row in lost.values():
        if row['side'] != 'worker':
            continue
        # The neighbours' whole seq ranges: rows finish out of seq order across a flush boundary. Only a seq the rank does not
        # hold can be blamed on this loss.
        lines = present.get(('worker', row['rank']), {})
        below = [n for n in lines if n < row['line'] and lines[n]['rows'].get('r')]
        above = [n for n in lines if n > row['line'] and lines[n]['rows'].get('r')]
        row['seq_low'] = min(r['seq'] for r in lines[max(below)]['rows']['r']) if below else None
        row['seq_high'] = max(r['seq'] for r in lines[min(above)]['rows']['r']) if above else None
    return dict(lost=[lost[k] for k in sorted(lost, key=str)], unidentified_torn_stamps=unidentified, rank_seqs=seqs)


def _touches(loss, seqs, rank_seqs):
    if loss['side'] != 'worker':
        return False
    low, high = loss.get('seq_low'), loss.get('seq_high')
    held = rank_seqs.get(loss['rank'], ())
    return any(s not in held and (low is None or s >= low) and (high is None or s <= high) for s in seqs)


def _medians(rows, *, band):
    out = dict(chunks=len(rows), wall_s=progress.median([c['wall_s'] for c in rows]),
               open_interval_s=progress.median([c.get('open_interval_s') for c in rows]),
               gap_before_s=progress.median([c.get('gap_before_s') for c in rows]),
               residual_s=progress.median([c['residual_s'] for c in rows]),
               components_s={k: progress.median([c['components_s'][k] for c in rows]) for k in progress.COMPONENTS})
    if band:
        out.update(stock_rpcs=progress.median([c['stock_rpcs'] for c in rows]), wait_s=progress.median([c['wait_s'] for c in rows]),
                   wait_sends=progress.median([c['wait_sends'] for c in rows]), commands=progress.median([c['commands'] for c in rows]))
    return out


def recount(row):
    """The join's per-band and whole-arm medians over the chunks still counted (same expressions, a test pins the equality)."""
    counted = [c for c in row['chunks'] if c['label'] == row['counted_label'] and not c.get('dropped_out')]
    row['counted_chunks'] = len(counted)
    row['bands'] = {band: _medians([c for c in counted if c['band'] == band], band=True)
                    for band in sorted({c['band'] for c in counted})}
    row['all_counted'] = _medians(counted, band=False)


def _forgive(row, reason):
    row['reasons'] = [x for x in row['reasons'] if x != reason]


def settle_arm(arm, row, ledger, torn, lines, last_line):
    """Count this arm's torn lines and lost records, drop the chunks they touch, and clear only the reasons they explain."""
    chunks = row.get('chunks') or []
    seqs = [c.pop('_seqs', ()) for c in chunks]
    in_arm = lambda rows: [t for s, t in rows if ap.in_span(s, arm)]
    torn_progress, torn_stage, torn_memory = (in_arm(torn[k]) for k in ('progress', 'stage', 'memory'))
    expected = (arm.get('progress_delta') or {}).get('chunks')
    shortfall = expected - len(chunks) if type(expected) is int and chunks else 0
    mine, engine_lost = [], 0
    for loss in ledger['lost']:
        if loss['side'] == 'engine':
            if engine_lost < shortfall and any(ap.in_span(s, arm) for s in loss['stamps']):
                engine_lost += 1
                mine.append(loss)
            continue
        touched = False
        for chunk, held in zip(chunks, seqs):
            if _touches(loss, held, ledger['rank_seqs']):
                touched = True
                chunk.setdefault('dropped_out', []).append('worker-flush-lost:%s:%s' % (loss['rank'], loss['line']))
        if touched or (loss['exact'] and any(ap.in_span(s, arm) for s in loss['stamps'])):
            mine.append(loss)
    unidentified = sum(1 for s in ledger['unidentified_torn_stamps'] if ap.in_span(s, arm))
    if torn_stage and arm.get('selection') == 'local':
        logged = {x['chunk_start'] for x in lines}
        without = [c for c in chunks if c.get('chunk_start') not in logged]
        if len(without) <= len(torn_stage):
            for chunk in without:
                chunk.setdefault('dropped_out', []).append('stage-line-lost')
    dropped = [c for c in chunks if c.get('dropped_out')]
    starts = sorted(c['chunk_start'] for c in dropped)
    # What a lost engine flush leaves: a hole in the chain, which must be exactly as many chunks as were lost.
    ordered = sorted(chunks, key=lambda c: c.get('chunk_start') or 0)
    prompt = (arm.get('fresh') or {}).get('prompt_tokens_observed')
    holes = [a['chunk_end'] for a, b in zip(ordered, ordered[1:]) if a['chunk_end'] != b['chunk_start']]
    if ordered and ordered[0].get('chunk_start') != 0:
        holes.insert(0, 0)
    if ordered and type(prompt) is int and ordered[-1].get('chunk_end') != prompt - 1:
        holes.append(ordered[-1].get('chunk_end'))
    if engine_lost:
        if shortfall == engine_lost:
            _forgive(row, 'progress-lines-differ-from-the-chunk-counter')
        if len(holes) == engine_lost:
            _forgive(row, 'native-chunks-not-row-0-through-n-minus-1')
            starts = sorted(starts + holes)
        if arm.get('cohort') == 'solo' and 0 in holes and all(c['label'] == 'solo' for c in ordered):
            _forgive(row, 'solo-arm-not-a-live-prefix-then-decoder-free-chunks')
        for a, b in zip(ordered, ordered[1:]):
            if a['chunk_end'] != b['chunk_start']:       # The join's neighbours span the hole.
                a.pop('open_interval_s', None)
                b.pop('gap_before_s', None)
    explained = sum(1 for c in dropped for m in c['missing_worker_rows']
                    if not (c['line'] == last_line and m['kind'] == 'release'))
    if explained and 'missing_worker_rows' in row:
        row['missing_worker_rows'] -= explained
        row['missing_worker_rows_in_dropped_chunks'] = explained
        if not row['missing_worker_rows']:
            _forgive(row, 'worker-rows-missing')
    if chunks and (dropped or engine_lost):
        recount(row)
    records = len(mine) + unidentified + len(torn_stage) + len(torn_memory)
    chunk_cost = len(starts)
    if records > LOST_RECORD_LIMIT:
        row['reasons'].append('lost-records-above-the-declared-limit:%d>%d' % (records, LOST_RECORD_LIMIT))
    if chunk_cost > DROPPED_CHUNK_LIMIT:
        row['reasons'].append('dropped-chunks-above-the-declared-limit:%d>%d' % (chunk_cost, DROPPED_CHUNK_LIMIT))
    row['tears'] = dict(torn_progress_lines=len(torn_progress), torn_stage_lines=len(torn_stage), torn_memory_lines=len(torn_memory),
                        lost_flushes=[{k: x[k] for k in ('side', 'rank', 'line', 'evidence')} for x in mine],
                        unidentified_torn_lines=unidentified, lost_records=records, dropped_chunk_starts=starts,
                        within_limits=records <= LOST_RECORD_LIMIT and chunk_cost <= DROPPED_CHUNK_LIMIT)
    return row['tears']


# --- per-arm rows --------------------------------------------------------------------------------------

def p10(values):
    """Linear-interpolated 10th percentile, the statistics module's inclusive rule: with 8 streams, between the two lowest."""
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=10, method='inclusive')[0]


def engagement(arm, row, lines, lost_stage=0):
    selection = arm.get('selection')
    delta = arm.get('stage_delta') or {}
    owned = (arm.get('progress_delta') or {}).get('chunks')
    chunks = (row or {}).get('chunks') or []
    if type(owned) is not int or owned < 1 or not delta or not chunks:
        return dict(selection=selection, engaged=False, reasons=['engagement-receipts-missing'], owned_chunks=owned,
                    delta=delta, stage_lines=len(lines), foreign_replies=None, missing_replies=None)
    foreign = sum(c.get('foreign_replies', 0) for c in chunks)
    short = sum(c.get('short_replies', 0) for c in chunks)
    reasons = []
    if selection == 'local':
        if delta.get('chunks') != owned:
            reasons.append('stage-chunks-delta-differs-from-the-auto-owned-chunks')
        if delta.get('local_releases') != 2 * owned:
            reasons.append('stage-releases-not-two-per-chunk')
        if len(lines) + lost_stage != owned:
            reasons.append('stage-lines-differ-from-the-auto-owned-chunks')
        if foreign:
            reasons.append('non-target-rank-replied-to-a-stage-command')
    else:
        if any(delta.get(k) != 0 for k in STAGE_COUNTERS):
            reasons.append('serial-arm-moved-a-stage-counter')
        if lines or lost_stage:
            reasons.append('serial-arm-logged-a-stage-line')
        if short:
            reasons.append('serial-arm-stage-command-without-four-replies')
    return dict(selection=selection, engaged=not reasons, reasons=reasons, owned_chunks=owned, delta=delta,
                stage_lines=len(lines), lost_stage_lines=lost_stage, foreign_replies=foreign, missing_replies=short)


def table_row(arm, row, view, mine):
    out = ap.arm_table(arm, row, view)
    out['selection'] = arm.get('selection')
    client = arm.get('client_basis') or {}
    during, only = client.get('decode_window'), arm.get('decode_only')
    ratios = ap.stream_ratios(during, only)
    rates = (during or {}).get('stream_rates_tokens_per_s') or []
    out.update(stream_ratios=ratios, p10_stream_ratio=p10(sorted(ratios)) if ratios else None,
               min_stream_rate=min(rates) if rates else None, p10_stream_rate=p10(sorted(rates)) if rates else None)
    chunks, commands = out.get('chunks'), out.get('commands')
    out['commands_per_chunk'] = commands / chunks if type(commands) is int and type(chunks) is int and chunks else None
    ranks = memory_receipts(mine)
    out['memory'] = dict(ranks=ranks, worst=memory_worst(ranks))
    tears = (row or {}).get('tears') or {}
    out['tears'] = tears
    out['stage'] = stage_summary([x for x in stage_lines(mine) if x['chunk_start'] not in (tears.get('dropped_chunk_starts') or ())])
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
    both = [r['stage']['both_stages_s_per_chunk'] for r in rows if r['stage']['both_stages_s_per_chunk'] is not None]
    out['both_stages_s_per_chunk'] = statistics.median(both) if both else None
    out['headroom_refused'] = sum(r['stage']['headroom_refused'] for r in rows)
    return out


def memory_compare(table):
    """Each local arm against its matched serial arm (same cohort and round), with the proposed gates reported."""
    rows = {}
    for selection, cohort, index in ARM_ORDER:
        if selection != 'local':
            continue
        local = (table.get(label('local', cohort, index)) or {}).get('memory', {}).get('worst')
        serial = (table.get(label('serial', cohort, index)) or {}).get('memory', {}).get('worst')
        name = '%s-%d' % (cohort, index)
        if not local or not serial:
            rows[name] = None
            continue
        rows[name] = dict(local=local, serial=serial,
                          local_minus_serial={k: round(local[k] - serial[k], 1) for k in local if k != 'receipts'},
                          zero_ooms=local['ooms'] == 0,
                          min_free_within_margin=local['min_free_mib'] >= serial['min_free_mib'] - MEMORY_FREE_MARGIN_MIB)
    return rows


def speed_compare(groups):
    out = {}
    for cohort in progress.COHORTS:
        f, s = groups.get('local-' + cohort), groups.get('serial-' + cohort)
        if f and s:
            out[cohort] = {k: progress.ratio(f.get(k), s.get(k)) for k in (
                'ttft_s', 'prefill_tok_s', 's_per_chunk', 'decode_during_tok_s', 'decode_ratio', 'commands_per_chunk')}
    return out


def numerics(arms):
    """The generated tokens and the standing streams' first tokens, compared across arms. See the module docstring."""
    table = {}
    for arm in arms:
        name = arm.get('label')
        outputs = arm.get('outputs') or {}
        table[name] = dict(selection=arm.get('selection'), fresh=outputs.get('fresh'), append=outputs.get('append'),
                           standing=outputs.get('standing'))
    verdict, differs, unproven = 'IDENTICAL', [], []
    for family in ('fresh', 'append'):
        serial = {json.dumps(r[family]) for r in table.values() if r['selection'] == 'serial' and r[family]}
        local = {json.dumps(r[family]) for r in table.values() if r['selection'] == 'local' and r[family]}
        if not serial or not local:
            unproven.append(family + '-tokens-missing')
        elif len(serial) > 1:
            unproven.append(family + '-serial-arms-already-differ')
        elif local != serial:
            differs.append(family)
    streams = sorted({stream for r in table.values() for stream in (r['standing'] or {})})
    stream_rows = {}
    for stream in streams:
        full = {}
        for row in table.values():
            cell = (row['standing'] or {}).get(stream)
            if cell and cell['prefix_tokens'] == STREAM_PREFIX_TOKENS:
                full.setdefault(row['selection'], set()).add(cell['prefix_sha256'])
        stream_rows[stream] = {k: sorted(v) for k, v in full.items()}
    if not streams:
        unproven.append('standing-digests-missing')
    for stream, row in stream_rows.items():
        serial, local = set(row.get('serial', ())), set(row.get('local', ()))
        if not serial or not local:
            unproven.append('standing-%s-prefix-short' % stream)
        elif len(serial) > 1:
            unproven.append('standing-%s-serial-arms-already-differ' % stream)
        elif local != serial:
            differs.append('standing-' + stream)
    if differs:
        verdict = 'DIFFERS'
    elif unproven:
        verdict = 'UNPROVEN'
    return dict(verdict=verdict, differs=differs, unproven=unproven, arms={k: {f: v[f] for f in ('selection', 'fresh', 'append')}
                                                                           for k, v in table.items()},
                standing=stream_rows, prefix_tokens=STREAM_PREFIX_TOKENS,
                limit='one sampled token per prefill and stream token digests; not a tensor comparison')


# --- the grade -----------------------------------------------------------------------------------------

def stage_breakdown(original):
    """chunk_breakdown with the chunk's kinds and each stage command's replying ranks kept.

    The join's rule allows one row kind to ride past the run's last flush: the release reply. On the stage path the sender's
    consumer receipt can also land after the receiver's local release, which is that chunk's last flush on every rank. Such a
    row is reported as a release row for that rule only and counted in ``late_receipt_rows``; a receipt row missing from any
    other chunk still fails the join, because the rule applies only to the run's last line.
    """
    def breakdown(chunk, workers):
        info = original(chunk, workers)
        kinds, segments = {}, {}
        foreign = short = 0
        for row in chunk['rows'].get('n', []):
            kinds[row['kind']] = kinds.get(row['kind'], 0) + 1
            if row['kind'] == 'segment':
                segments[row['segment']] = segments.get(row['segment'], 0) + 1
                stage = row.get('stage')
                if stage in TARGETS:
                    replied = {rank for rank in range(4) if type(row.get('t_recv%d' % rank)) is int}
                    foreign += int(bool(replied - set(TARGETS[stage])))
                    short += int(len(replied) != 4)
        late = 0
        for missing in info.get('missing_worker_rows') or ():
            if missing['kind'] == 'receipt':
                missing['kind'] = 'release'
                late += 1
        info.update(kinds=kinds, segments=segments, foreign_replies=foreign, short_replies=short, late_receipt_rows=late,
                    _seqs=[row['seq'] for tag in ('n', 's') for row in chunk['rows'].get(tag, [])])
        return info
    return breakdown


progress.chunk_breakdown = stage_breakdown(progress.chunk_breakdown)


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
    torn = torn_lines(logs)
    unparsable = errors.count('unparsable-progress-line')
    records, errors, failed = whole_records(records, errors, torn['progress'])
    if unparsable != failed:
        missing.append('progress:torn-lines-differ-from-the-unparsable-lines')
    ledger = flush_ledger(records, torn['progress'])
    receipts = progress.progress_receipts(known, records, errors)
    receipts['problems'] = [x for x in receipts['problems'] if not INCOMPLETE.fullmatch(x)]
    if not records:
        missing.append('no-progress-lines-in-the-engine-log')
    for problem in receipts['problems']:
        missing.append('progress:' + problem)
    tracked = ap.read_stamped(logs, is_tracked)
    last_line = max((f['line'] for f in progress.assemble(records)[0] if f['side'] == 'engine'), default=None)
    claimed = {}
    for arm in known:
        row = receipts['arms'][arm['label']]
        mine = [(s, t) for s, t in tracked if ap.in_span(s, arm)]
        tears = settle_arm(arm, row, ledger, torn, stage_lines(mine), last_line)
        for x in tears['lost_flushes']:
            claimed[(x['side'], x['rank'], x['line'])] = arm['label']
    receipts['tears'] = dict(lost_record_limit=LOST_RECORD_LIMIT, dropped_chunk_limit=DROPPED_CHUNK_LIMIT,
                             torn_progress_lines=len(torn['progress']), torn_stage_lines=len(torn['stage']),
                             torn_memory_lines=len(torn['memory']), lost_flushes=len(ledger['lost']),
                             unidentified_torn_lines=len(ledger['unidentified_torn_stamps']),
                             lost_flushes_outside_every_arm=[{k: x[k] for k in ('side', 'rank', 'line', 'evidence')} for x in ledger['lost']
                                                             if (x['side'], x['rank'], x['line']) not in claimed])
    if receipts['dropped_rows'] or receipts['duplicate_worker_rows']:
        missing.append('progress:rows-dropped-or-duplicated')
    if receipts['unstamped_engine_flushes'] or receipts['engine_flushes_outside_every_arm']:
        missing.append('progress:engine-lines-outside-every-arm-span')
    if receipts['engine_flushes_in_two_arms']:
        missing.append('progress:engine-line-in-two-arm-spans')
    for name, row in sorted(receipts['arms'].items()):
        missing.extend('progress-%s:%s' % (name, x) for x in row['reasons'])
    table, valid = {}, {}
    for arm in known:
        name = arm['label']
        row = receipts['arms'].get(name)
        view = ap.log_view(arm, tracked)
        mine = [(s, t) for s, t in tracked if ap.in_span(s, arm)]
        signatures = {x for x in (signature_of(t) for _, t in mine) if x}
        error_text = json.dumps(arm.get('error') or {})
        signatures |= {name_ for name_, text in SIGNATURES if text in error_text}
        engaged = engagement(arm, row, stage_lines(mine), ((row or {}).get('tears') or {}).get('torn_stage_lines', 0))
        table[name] = table_row(arm, row, view, mine)
        verdict, classification, reasons = arm_verdict(arm, ingress.get(name), watches.get(name), view, signatures,
                                                       (row or {}).get('reasons') or [], engaged)
        valid[name] = dict(verdict=verdict, classification=classification, reasons=reasons,
                           crash_free=arm.get('complete') is True and (watches.get(name) or {}).get('verdict') != 'DEAD',
                           disagree_lines=view['disagree_lines'], signatures=sorted(signatures),
                           engine_watch=(watches.get(name) or {}).get('verdict'), ingress=(ingress.get(name) or {}).get('verdict'),
                           engagement=engaged, selection_switch=arm.get('selection_switch'), tears=(row or {}).get('tears'))
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
    compared = numerics(known)
    forced = []
    if compared['verdict'] == 'DIFFERS':
        failures.append('numerics-differ:' + ','.join(compared['differs']))
        forced.append('ENGINE')
    elif compared['verdict'] == 'UNPROVEN':
        missing.extend('numerics:' + x for x in compared['unproven'])
    rows = [table[a['label']] for a in known]
    groups = {}
    for selection in SELECTIONS:
        for cohort in progress.COHORTS:
            cell = [r for r in rows if (r['selection'], r['cohort']) == (selection, cohort)]
            if cell:
                groups[selection + '-' + cohort] = group_summary(cell)
    summary = dict(groups=groups, device_memory=ap.memory_summary(rows),
                   allocator_retry_warnings=sum(r['allocator_retry_warnings'] for r in rows),
                   max_retry_request_mib=max((r['max_retry_request_mib'] or 0) for r in rows) if rows else None,
                   arm_verdicts={r['label']: valid[r['label']]['verdict'] for r in rows},
                   local_over_serial=speed_compare(groups))
    verdict = 'FAIL' if failures else 'INC' if missing else 'PASS'
    classes = sorted({v['classification'] for v in valid.values() if v['classification']} | set(forced))
    if verdict == 'FAIL' and not classes:
        classes = ['HARNESS']
    return {'schema': GRADE_SCHEMA, 'verdict': verdict, 'classification': classes or None, 'failures': sorted(set(failures)),
            'missing': sorted(set(missing)), 'required_arms': required, 'engine_lines_outside_every_arm': len(between),
            'table': table, 'validity': valid, 'progress': receipts, 'summary': summary, 'numerics': compared,
            'memory_local_vs_serial': memory_compare(table), 'speed_is_gate': False, 'memory_is_gate': False,
            'target_is_gate': False, 'all_declared_arms_retained': True, 'retry_selected': False, 'picked': False,
            'release_readiness': RELEASE_READINESS,
            'basis': 'one engine, one selection per arm by typed control; band 1 matched ABBA at 65,445 tokens; per chunk, the '
                     'critical rank of each native command; live arms count chunks begun with all 8 decoders, solo arms chunks '
                     'begun with none',
            'scope': 'C0 mechanism on the composed alpha engine with overlap off, with speed and memory reported; gains from '
                     'separate experiments are never multiplied'}


# --- the report ----------------------------------------------------------------------------------------

def _cell(value, digits=1):
    if value is None:
        return '-'
    return ('%.' + str(digits) + 'f') % value if type(value) is float else str(value)


def report(result):
    """Three separate headed sections. Numbers come only from the graded record."""
    lines = ['# q18a-alpha-c0 report', '', 'Overall: **%s**%s' % (
        result['verdict'], ' (%s)' % ', '.join(result['classification']) if result['classification'] else ''), '']
    lines += ['## Mechanism and numerics', '', '| arm | verdict | class | crash-free | disagree | signatures | engaged | stage chunks | '
              'owned chunks | releases | foreign replies | reasons |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for name in result['required_arms']:
        v = result['validity'].get(name) or {}
        e = v.get('engagement') or {}
        d = e.get('delta') or {}
        lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            name, v.get('verdict'), v.get('classification') or '-', v.get('crash_free'), _cell(v.get('disagree_lines')),
            ','.join(v.get('signatures') or []) or '0', e.get('engaged'), _cell(d.get('chunks')), _cell(e.get('owned_chunks')),
            _cell(d.get('local_releases')), _cell(e.get('foreign_replies')), '; '.join(v.get('reasons') or []) or '-'))
    compared = result['numerics']
    lines += ['', 'Numerics: **%s**. %s.' % (compared['verdict'], compared['limit'])]
    if compared['differs']:
        lines.append('Differs: ' + ', '.join(compared['differs']))
    if compared['unproven']:
        lines.append('Unproven: ' + ', '.join(compared['unproven']))
    lines += ['']
    if result['failures']:
        lines += ['Failures: ' + '; '.join(result['failures']), '']
    if result['missing']:
        lines += ['Inconclusive items: ' + '; '.join(result['missing']), '']
    tears = result['progress']['tears']
    lines += ['### Torn lines and lost records', '',
              'Limits declared before the run, per arm: at most %d lost record and %d dropped chunks. Over the run: %d torn progress, '
              '%d torn stage and %d torn memory lines, %d lost flushes, %d outside every arm.' % (
                  tears['lost_record_limit'], tears['dropped_chunk_limit'], tears['torn_progress_lines'], tears['torn_stage_lines'],
                  tears['torn_memory_lines'], tears['lost_flushes'], len(tears['lost_flushes_outside_every_arm'])), '',
              '| arm | torn progress | torn stage | torn memory | lost flushes | lost records | dropped chunks | within limits |',
              '|---|---|---|---|---|---|---|---|']
    for name in result['required_arms']:
        t = (result['validity'].get(name) or {}).get('tears')
        if t:
            lines.append('| %s | %s | %s | %s | %s | %s | %s | %s |' % (
                name, t['torn_progress_lines'], t['torn_stage_lines'], t['torn_memory_lines'], len(t['lost_flushes']), t['lost_records'],
                ','.join(str(x) for x in t['dropped_chunk_starts']) or '0', t['within_limits']))
    lines += ['', '## Performance', '', 'Reported, never gated.', '',
              '| arm | TTFT s | prefill tok/s | s/chunk | decode during | decode only | ratio | min stream ratio | p10 stream '
              'ratio | gap p99 ms | gap max ms | cmds/chunk | both stages s/chunk | ring refusals |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for name in result['required_arms']:
        row = result['table'].get(name)
        if not row:
            continue
        lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            name, _cell(row.get('ttft_s')), _cell(row.get('prefill_tok_s'), 0), _cell(row.get('s_per_chunk'), 3),
            _cell(row.get('decode_during_tok_s')), _cell(row.get('decode_only_tok_s')), _cell(row.get('decode_ratio'), 3),
            _cell(row.get('min_stream_ratio'), 3), _cell(row.get('p10_stream_ratio'), 3), _cell(row.get('gap_p99_ms')),
            _cell(row.get('gap_max_ms')), _cell(row.get('commands_per_chunk')), _cell(row['stage']['both_stages_s_per_chunk'], 4),
            _cell(row['stage']['headroom_refused'])))
    lines += ['', 'Local over serial, medians: ' + json.dumps(result['summary']['local_over_serial'], sort_keys=True), '',
              '### Device memory, local against the matched serial arm', '',
              '| pair | local peak alloc MiB | serial | local peak reserved MiB | serial | local min free MiB | serial | '
              'local retries | serial | local OOMs | serial | 0 OOMs | min free within 2 MiB |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for name, row in result['memory_local_vs_serial'].items():
        if not row:
            lines.append('| %s | not available | | | | | | | | | | | |' % name)
            continue
        f, s = row['local'], row['serial']
        lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            name, f['peak_allocated_mib'], s['peak_allocated_mib'], f['peak_reserved_mib'], s['peak_reserved_mib'],
            f['min_free_mib'], s['min_free_mib'], f['alloc_retries'], s['alloc_retries'], f['ooms'], s['ooms'],
            row['zero_ooms'], row['min_free_within_margin']))
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
    print('ALPHA_C0_GRADE ' + result['verdict'])
    return {'PASS': 0, 'FAIL': 1}.get(result['verdict'], 2)


if __name__ == '__main__':
    raise SystemExit(main())
