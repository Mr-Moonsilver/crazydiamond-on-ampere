#!/usr/bin/env python3
"""Busy-versus-far decode forwards within an n3 arm, from one fetched record root (lead brief 13 Nov 2026 16:34).

Codex's recipe (READING-split-decode-sizing-20261113.md, "13 Nov 2026 15:36 CET -- exact night-reader recipe"),
implemented as written. Reads only; never recompresses, moves or deletes anything.

    python3 -B tools/q2024p_j2_busyfar_reader.py --root RECORD_ROOT --out FRESH_DIR [--engine-log PATH]

Engine log: RECORD_ROOT/engine.log, .xz or .zst (a .zst is read through ``zstd -dc``), else engine-current.log /
engine-stream.log the same way, or ``--engine-log``. Writes FRESH_DIR/busyfar.json and FRESH_DIR/busyfar.txt.

Method.
  1. Atomic receipts: ``flash-s5 handler`` / ``progress`` / ``anatomy`` flushes reassembled by writer pid, side, rank,
     line and base_ns with every part present and one record sha; identical repeated parts deduplicated, a second sha
     or differing content rejects the flush; torn lines (unparsable, or another log header inside) dropped. Every
     receipt's per-process ``seq`` (+1 across all kinds) shows lost writes of any kind. Torn / missing-part /
     conflicting / line-gap / receipt-seq / recorder dropped and unresolved counts are printed beside every table; a
     loss window touching a forward's [L - 200 ms, U + 200 ms] makes it unsupported. Null timings stay null (never 0).
     Rows flagged failed / aborted / install / dev-unresolved / nested, or without dev_forward, are excluded (counted).
  2. Joins by transport seq, rank and writer, never by neighbouring lines. Execute <-> sample pairing walks the stock
     RPCs in transport-seq (send) order: a sample pairs with the single nonempty execute (forward seam recorded on a
     rank) sent since the previous sample. Several nonempty, an execute whose emptiness is unknown (a rank's row
     missing), or any transport seq in between absent from the record leave it unpaired.
  3. Bracket: L = the rank's execute t_enter; U = the matched sample's AsyncOutput t_emit (+1 us quantisation) on the
     last-stage rank that records get_output (rank 2: stock returns the output from one rank). It bounds that rank's
     forward and stage 0's (whose pp_send it received). Rank 3 has no proof of its own: unlocated. No U: unlocated.
  4. Native envelopes per rank: worker anatomy ``c`` [t_first, t_last], expanded by read_us + 1 us (blocking) and
     shrunk by the same (core); a cancelled / errored chunk blocks but has no core; an engine chunk with no anatomy on
     that rank blocks with its engine window; every native-gap event instant blocks.
       certified far_compute:          [L - 200 ms, U + 200 ms] disjoint from every blocking envelope of the rank;
       certified inside_native_window: max(0, D - |[L,U] outside the rank's core union|) >= 0.8 D;
     everything else AMBIGUOUS, excluded from the contrast. A route generation / engine epoch change inside [L, U]
     (native-gap timeline) excludes the forward as a transition.
  5. Arm: an engine progress chunk belongs to an arm when its log stamp minus its monotonic span and the 2 s log lag
     starts, and the stamp ends, inside the arm's wall window; a chunk across a boundary is excluded. Strata: arm (n3
     only), route generation, engine epoch, merge mode, and decode_live / history band / width of the engine chunk
     overlapping the bracket most (else the nearest): proxies, not the padded descriptor. Each busy forward takes the
     nearest-in-time far control of its stratum without replacement; busy forwards left without one:
     INSUFFICIENT_SAME_ARM_SUPPORT.
  6. Per rank / arm / live bin: counts, D p50 / mean, paired D difference, H (forward seam) and its CPU, observer_ns,
     idle-before / after shares, worker_enter - engine_submit and the native-handler occupancy inside it,
     handle_output, the proof row's get_output, MQ write; the host-only lower bound D_busy - H_busy - observer_busy -
     D_far. PP stages: TP peers side by side, the stage value their max (never the sum). Bootstrap of the mean paired
     difference by native chunk, not by forward. Also the clearance of every bracketed forward from the nearest
     blocking envelope, and the clear stretches inside each n3 arm, so an empty far class can be read.
  7. ESTIMATED: the old endpoint reconstruction ([U - D, U]) is printed in its own block, never in the contrast.
  8. Decode-timeline stamp (FLASH_S5_DECODE_TIMELINE, Codex 1469bd85; lead brief 13 Nov 18:48). When the record has
     ``flash-s5 decode-timeline`` d rows the reader is in timeline mode: U is each rank's OWN queried forward
     completion (t_complete_query + 1 us), on all four ranks, never rank 2's get_output; a forward without its own d
     row is unlocated (no_own_completion); a d row whose device_ns or t_enter differs from its handler row, or a
     second different d row for one (rank, seq), is inconsistent. Sessions are keyed by sample / txn / anchor_sample
     beside line and base_ns (line is 0 per session, base_ns 0 without a forward); sample-number gaps, incomplete
     sessions and every ``s`` counter are printed. Outside-native-progress samples supply far controls: engine send
     from the stamp header (engine_send_ns, transport send) and the arm from the certified engine chunks around
     [send, completion] (both neighbours and any overlapping chunk in one arm), never the line's log stamp. Strata come
     from the actual metadata (arm, bank, route generation, engine epoch, bank SMs, cg mode, padded num_tokens,
     num_reqs, actual rows, max-history band); forward_calls != 1, handler_flags != 0, null history or no descriptor:
     metadata_unsupported. Pairs report how many share the uid digest. Exposed wait: per native-chunk execute the
     worker queue_native on the critical PP path (max of TP peers per stage, then max over stages; any rank null:
     none), binned at 0 / 0.5 / 1 / 2 / 4 ms, with the engine interval to the next nonempty execute of the chunk
     (cadence), the idle-before share and the exposed candidate (queue_native where the stream was idle at launch);
     descriptive and conditional only (EXPOSED_NOTE). Without the stamp, items 1-7 run unchanged.
  Not computed: client adjacent token gaps; fused-phase / producer-witness copy durations (recipe item 6).
"""
import argparse
import bisect
import json
import lzma
import os
import random
import re
import statistics
import subprocess
import sys

SCHEMA = 'q2024p-j2-busyfar/1'
GUARD_NS = 200_000_000            # far: 200 ms clear on either side of [L, U]
INSIDE_SHARE = 0.8                # inside: at least 0.8 D provably inside the core union
QUANT_NS = 1000                   # offsets are whole microseconds: +/- 1 us per endpoint
BOOTSTRAP = 2000
STAGES = {0: (0, 1), 1: (2, 3)}
BANDS = ((0, 16384), (16384, 32768), (32768, 49152), (49152, 65536), (65536, 97152), (97152, 1 << 40))
KINDS = {'handler': 'flash-s5 handler ', 'progress': 'flash-s5 progress ', 'anatomy': 'flash-s5 anatomy ',
         'gap': 'flash-s5 native-gap ', 'timeline': 'flash-s5 decode-timeline '}
SCHEMAS = {'handler': 's5-handler-anatomy/1', 'progress': 's5-native-progress/1', 'anatomy': 's5-native-anatomy/1',
           'gap': 's5-native-gap/1', 'timeline': 's5-decode-timeline/1'}
# decode-timeline (Codex 1469bd85, docs/e3-briefs/n3-decode-timeline.md): every session is a new trace, so ``line`` is
# 0 per session and an incomplete no-forward session has base_ns 0. Parts are keyed by these too.
TIMELINE_KEY = ('sample', 'txn', 'anchor_sample')
D_FIELDS = ('seq', 'metadata', 'start_ms', 'end_ms', 'device_ns', 'forward_host_ns', 'observer_ns', 't_enter', 't_exit',
            't_complete_query')
OUTSIDE_SCOPE = 'outside-native-progress'
QUEUE_BINS_MS = (0.0, 0.5, 1.0, 2.0, 4.0)
M_FIELDS = ('seq', 'kind', 'out', 'flags', 't_enter', 't_exit', 'wall_ns', 'cpu_ns', 'observer_ns',
            'dev_forward_ns', 'dev_sample_ns', 'idle_before_launch', 'idle_after_launch', 'handle_output_ns', 'seams')
A_FIELDS = ('seq', 'thread', 't_take', 'get_output_ns', 'get_output_cpu_ns', 'mq_write_ns', 't_emit', 'observer_ns')
STAMP = re.compile(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d+Z')
LOG_LAG_S = 2.0
CANNOT_SAY = (
    "What this contrast cannot say (Codex, READING-split-decode-sizing-20261113.md, 15:36 CET):",
    "* \"Tonight can separate recorded forward-span growth from host queuing/reply cost, and compare TP peers. It cannot "
    "uniquely divide that growth into SM allocation, HBM/cache contention and x8 effects.\"",
    "* \"busy and far comparisons on the same SPLIT route both have 24 decode SMs: a static 24-versus-44 allocation change "
    "is not a component one can measure from that contrast.\" Never call the remainder the SM penalty.",
    "* Copy overlap: \"This does not certify absence of copy-stream activity, which lacks absolute endpoints.\" \"A "
    "far_compute forward can still overlap a copy.\"",
    "* x8: do not \"assign the peer-duration difference to PCIe; TP collectives can propagate the slower card's delay to "
    "both peers. The x8 link is fixed in all states, not a counterfactual switch.\"",
    "* \"The handler rows do not record the actual padded graph descriptor and per-execute UID/row set, so these are "
    "matched strata, not proven identical workloads.\"",
    "* \"Even the inside label means inside a native command envelope, not proof of simultaneous non-idle kernel "
    "execution.\" A forward CUDA span includes any stream wait/launch gap between its events.",
)


class Refused(SystemExit):
    pass


def refuse(why):
    raise Refused('REFUSED: ' + why)


# --- reading -------------------------------------------------------------------------------------------------------
def engine_log(root, override=None):
    if override:
        if not os.path.isfile(override):
            refuse('engine log %s not found' % override)
        return override
    for stem in ('engine.log', 'engine-current.log', 'engine-stream.log'):
        for suffix in ('', '.xz', '.zst'):
            path = os.path.join(root, stem + suffix)
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                return path
    refuse('no engine log (engine.log / engine-current.log / engine-stream.log, plain, .xz or .zst) in %s' % root)


def lines_of(path):
    if path.endswith('.zst'):
        proc = subprocess.Popen(['zstd', '-dc', path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with proc.stdout, proc.stderr:
            for raw in proc.stdout:
                yield raw.decode('utf-8', 'replace')
            err = proc.stderr.read()
        if proc.wait() != 0:
            refuse('zstd -dc %s failed (rc %d): %s' % (path, proc.returncode, err.decode()[:200]))
        return
    opener = lzma.open if path.endswith('.xz') else open
    with opener(path, 'rt', errors='replace') as handle:
        yield from handle


def stamp_s(line):
    match = STAMP.match(line)
    if not match:
        return None
    import datetime
    text = match.group(0)
    head, frac = text[:-1].split('.')
    when = datetime.datetime.strptime(head, '%Y-%m-%dT%H:%M:%S').replace(tzinfo=datetime.timezone.utc)
    return when.timestamp() + float('0.' + frac)


IDENTITY = re.compile(r'flash-s5 [a-z0-9-]+ (?:\{"pid":(\d+),"seq":(\d+),|.* pid=(\d+) seq=(\d+) sha=)')


def read_records(lines):
    """{kind: [(stamp, body)]}, the torn counts per kind and every receipt identity (pid, seq) of any kind.

    Receipts carry ``pid`` and a per-process ``seq`` that rises by exactly 1 across every record of every kind
    (native_measurement.py, receipt writer), so a skipped seq is a lost write even in a kind this reader ignores."""
    out = {k: [] for k in KINDS}
    torn = {k: 0 for k in KINDS}
    identities = {}
    for line in lines:
        if 'flash-s5 ' not in line:
            continue
        ident = IDENTITY.search(line)
        if ident:
            pid, seq = (ident.group(1), ident.group(2)) if ident.group(1) else (ident.group(3), ident.group(4))
            identities.setdefault(int(pid), set()).add(int(seq))
        for kind, tag in KINDS.items():
            at_ = line.find(tag)
            if at_ < 0:
                continue
            text = line[at_ + len(tag):].strip()
            if STAMP.search(text) or 'flash-s5 ' in text:
                torn[kind] += 1                 # another writer's line inside: torn
                break
            try:
                body = json.loads(text)
            except ValueError:
                torn[kind] += 1
                break
            if type(body) is not dict or body.get('schema', SCHEMAS[kind]) != SCHEMAS[kind]:
                torn[kind] += 1
                break
            out[kind].append((stamp_s(line), body))
            break
    return out, torn, identities


def receipt_gaps(identities):
    """Per pid: the skipped receipt seqs [(after, before)] between the first and last seen."""
    gaps = {}
    for pid, seqs in identities.items():
        seqs = sorted(seqs)
        gaps[pid] = [(a, b) for a, b in zip(seqs, seqs[1:]) if b != a + 1]
    return gaps


def assemble(records, extra=(), line_gaps=True):
    """Whole flushes per writer pid and (side, rank, line, base_ns[, extra header fields]); counts of what was
    rejected and why. ``extra``: decode-timeline sessions add TIMELINE_KEY (line is per session there, so line gaps
    are not a loss measure for that kind: ``line_gaps`` None).

    A flush needs every part 0..parts-1 and one ``sha`` (the record hash written on every part); identical repeated
    parts are deduplicated, a second sha or differing content for one part rejects the flush."""
    groups = {}
    for stamp, body in records:
        key = (body.get('pid'), body.get('side'), body.get('rank'), body.get('line'), body.get('base_ns')) + tuple(
            json.dumps(body.get(k), sort_keys=True) for k in extra)
        groups.setdefault(key, []).append((stamp, body))
    flushes = []
    counts = dict(flushes=0, missing_parts=0, conflicting_duplicates=0, duplicate_parts=0, line_gaps=0, dropped=0)
    for key, parts in groups.items():
        by_part = {}
        conflict = len({b.get('sha') for _, b in parts}) > 1
        for stamp, body in parts:
            p = body.get('part')
            if p in by_part:
                if json.dumps(by_part[p][1], sort_keys=True) == json.dumps(body, sort_keys=True):
                    counts['duplicate_parts'] += 1
                    continue
                conflict = True
            by_part[p] = (stamp, body)
        if conflict:
            counts['conflicting_duplicates'] += 1
            continue
        order = sorted(by_part, key=lambda x: (type(x) is not int, x if type(x) is int else 0))
        count = by_part[order[0]][1].get('parts')
        if type(count) is not int or order != list(range(count)) or any(
                b.get('parts') != count for _, b in by_part.values()):
            counts['missing_parts'] += 1
            continue
        bodies = [by_part[i][1] for i in range(count)]
        head = bodies[0]
        rows = {}
        for b in bodies:
            for tag, items in (b.get('rows') or {}).items():
                rows.setdefault(tag, []).extend(items)
        counts['dropped'] += head.get('dropped') or 0
        rseq = [b.get('seq') for b in bodies if type(b.get('seq')) is int]
        flushes.append(dict(writer=(key[0],), side=key[1], rank=key[2], line=key[3], base=key[4] or 0, head=head,
                            rows=rows, rseq=(min(rseq), max(rseq)) if rseq else None,
                            stamp=min((s for s, _ in by_part.values() if s is not None), default=None)))
    counts['flushes'] = len(flushes)
    if not line_gaps:
        counts['line_gaps'] = None
        return flushes, counts
    lines = {}
    for f in flushes:
        lines.setdefault((f['writer'], f['side'], f['rank']), set()).add(f['line'])
    for numbers in lines.values():
        numbers = sorted(n for n in numbers if type(n) is int)
        counts['line_gaps'] += sum(b - a - 1 for a, b in zip(numbers, numbers[1:]))
    return flushes, counts


def num(value):
    """A recorded number, or None: a null timing is never 0."""
    return value if type(value) in (int, float) and not isinstance(value, bool) else None


def at(base, offset_us):
    offset_us = num(offset_us)
    return None if offset_us is None else base + offset_us * 1000


def stage_value(values):
    """A PP stage's value from its TP peers: the max (the peers run the same step in parallel), never the sum."""
    return max(values)


# --- tables ----------------------------------------------------------------------------------------------------------
def handler_rows(flushes):
    """Main and output rows per (writer, rank, seq), restored to absolute ns; recorder counters per writer (cumulative)."""
    main, outs, seams_of, counters = {}, {}, {}, {}
    for f in flushes:
        head = f['head'].get('head') or {}
        if head.get('seams'):
            seams_of[f['writer']] = head['seams']
        mine = counters.setdefault(f['writer'], {})
        for name, value in (head.get('counters') or {}).items():
            if num(value) is not None:
                mine[name] = max(mine.get(name, 0), value)
    for f in flushes:
        names = seams_of.get(f['writer'])
        for item in f['rows'].get('m', []):
            r = dict(zip(M_FIELDS, item))
            r['t_enter'], r['t_exit'] = at(f['base'], r['t_enter']), at(f['base'], r['t_exit'])
            r['seams'] = ({names[c[0]]: dict(ns=c[1], excl_ns=c[2], cpu_ns=c[3], n=c[4]) for c in r['seams'] or ()}
                          if names else None)
            r['rank'], r['writer'], r['line'] = f['rank'], f['writer'], f['line']
            main[(f['rank'], r['seq'])] = r
        for item in f['rows'].get('a', []):
            r = dict(zip(A_FIELDS, item))
            r['t_take'], r['t_emit'] = at(f['base'], r['t_take']), at(f['base'], r['t_emit'])
            r['rank'], r['writer'] = f['rank'], f['writer']
            outs.setdefault((f['rank'], r['seq']), []).append(r)
    total = {}
    for mine in counters.values():
        for name, value in mine.items():
            total[name] = total.get(name, 0) + value
    return main, outs, total


def flush_span(f):
    """Absolute [min, max] of a flush's row times (offset fields named t_*), or None."""
    times = []
    for tag, items in f['rows'].items():
        names = ((f['head'].get('fields') or {}).get(tag) or
                 {'m': M_FIELDS, 'a': A_FIELDS}.get(tag) or ANATOMY_FIELDS.get(tag) or ())
        for item in items:
            for name, value in zip(names, item):
                t = at(f['base'], value) if name.startswith('t_') else None
                if t is not None:
                    times.append(t)
    return (min(times), max(times)) if times else None


ANATOMY_FIELDS = {'c': ('commands', 'timed', 'untimed', 'refused', 't_first', 't_last', 'busy_us', 'idle_us',
                        'idle_qsa_us', 'read_us', 'cancelled', 'detail', 'errors'),
                  'q': ('layer', 'commands', 'credit', 'credit_wait_us', 'grant', 'grant_wait_us', 'lag_us', 'lag_max_us',
                        'busy_us', 'idle_us', 't_start', 't_end'),
                  'x': ('op_seq', 'kind', 'layer', 'refused', 't_host', 't_start', 't_end')}


def loss_windows(flushes, kind):
    """Monotonic windows where a flush of this stream is missing (line gap) or reports recorder loss.

    A gap's window runs from the last row of the complete flush before it to the first row of the one after it;
    a flush reporting dropped rows (anatomy / progress ``dropped``, handler ``rows_dropped`` rising) covers the
    span since the previous flush ended. A forward whose comparison window touches one is unsupported."""
    streams = {}
    for f in flushes:
        if type(f['line']) is int:
            streams.setdefault((f['writer'], f['side'], f['rank']), []).append(f)
    out = []
    for seq in streams.values():
        seq.sort(key=lambda f: f['line'])
        prev_end, prev_line, prev_dropped = None, None, 0
        for f in seq:
            span = flush_span(f) or (f['base'], f['base'])
            if kind == 'handler':
                dropped = ((f['head'].get('head') or {}).get('counters') or {}).get('rows_dropped') or 0
                lossy = dropped > prev_dropped
                prev_dropped = max(prev_dropped, dropped)
            else:
                lossy = bool(f['head'].get('dropped'))
            if prev_line is not None and f['line'] != prev_line + 1:
                out.append((prev_end, span[0]))
            if lossy:
                out.append((prev_end if prev_end is not None else span[0], span[1]))
            prev_end, prev_line = span[1], f['line']
    return out


def receipt_loss_windows(receipt_gaps, flush_lists, gap_records):
    """A skipped receipt seq of pid P: from the last monotonic time P wrote before it to the first after it
    (row spans of P's flushes, native-gap observed_ns). Without a neighbour on one side, that side is unbounded."""
    points = {}
    for flushes in flush_lists:
        for f in flushes:
            span = flush_span(f)
            if f['rseq'] and span:
                points.setdefault(f['writer'][0], []).append((f['rseq'][0], f['rseq'][1], span[0], span[1]))
    for _, g in gap_records:
        if type(g.get('seq')) is int and num(g.get('observed_ns')) is not None:
            points.setdefault(g.get('pid'), []).append((g['seq'], g['seq'], g['observed_ns'], g['observed_ns']))
    out = []
    for pid, gaps in receipt_gaps.items():
        mine = sorted(points.get(pid, []))
        for a, b in gaps:
            before = [p[3] for p in mine if p[1] <= a]
            after = [p[2] for p in mine if p[0] >= b]
            out.append((max(before) if before else float('-inf'), min(after) if after else float('inf')))
    return out


def gap_loss_windows(gap_records):
    """native-gap lines are single receipts with a per-writer gap_seq: a skipped gap_seq loses an event."""
    by = {}
    for _, g in gap_records:
        if type(g.get('gap_seq')) is int and num(g.get('observed_ns')) is not None:
            by.setdefault((g.get('pid'), g.get('sha'), g.get('engine_epoch')), []).append((g['gap_seq'], g['observed_ns']))
    out, missing = [], 0
    for rows in by.values():
        rows = sorted(set(rows))
        for (a, ta), (b, tb) in zip(rows, rows[1:]):
            if b > a + 1:
                missing += b - a - 1
                out.append((ta, tb))
    return out, missing


def engine_chunks(flushes):
    """Engine progress flushes with their s rows, chunk identity, merge mode and monotonic window."""
    chunks = []
    for f in flushes:
        if f['side'] != 'engine':
            continue
        h = f['head']
        rows = f['rows']
        s = sorted((dict(seq=x[0], kind=x[1], t_submit=at(f['base'], x[2]), t_recv=at(f['base'], x[3]),
                         t_take=at(f['base'], x[4])) for x in rows.get('s', [])), key=lambda r: r['seq'])
        span = flush_span(f)
        times = [t for t in (span or ()) if t is not None] + [x for x in [num(h.get('commit_ns'))] if x is not None]
        chunks.append(dict(writer=f['writer'], line=f['line'], stamp=f['stamp'], txn=h.get('txn'),
                           generation=h.get('request_generation'), chunk_start=h.get('chunk_start'),
                           chunk_end=h.get('chunk_end'), decode_live=h.get('decode_live'), mode=h.get('mode'), s=s,
                           n=[x[0] for x in rows.get('n', [])],
                           window=(min(times), max(times)) if times else None))
    chunks.sort(key=lambda c: (str(c['writer']), c['line'] if type(c['line']) is int else -1))
    return chunks


def observed_seqs(progress_flushes, main):
    """Every transport seq the record proves was sent (engine s / n rows, worker r / w rows, handler rows)."""
    seen = set()
    for f in progress_flushes:
        for tag in ('s', 'n', 'r', 'w'):
            for x in f['rows'].get(tag, []):
                if type(x[0]) is int:
                    seen.add(x[0])
    seen.update(seq for _, seq in main)
    return seen


def emptiness(main, seq):
    """True: the execute ran a forward on some rank; False: all four ranks recorded it without one; None: unknown."""
    rows = [main.get((rank, seq)) for rank in (0, 1, 2, 3)]
    if any(r is not None and r['seams'] is not None and 'forward' in r['seams'] for r in rows):
        return True
    if all(r is not None and r['seams'] is not None for r in rows):
        return False
    return None


def pair_samples(chunks, main, seen):
    """execute seq -> sample seq by the ordered stock-RPC sequence (transport seq is the send order).

    Walking the engine's stock RPCs in seq order, a sample pairs with the single nonempty execute sent since the
    previous sample, provided every transport seq between them is in the record (else a stock RPC could be missing)
    and every execute in between has a known emptiness. Zero or several nonempty executes, an unknown emptiness or a
    seq gap leave the sample unpaired."""
    pairs, why = {}, dict(paired=0, none_pending=0, several_pending=0, unknown_emptiness=0, seq_gap=0)
    stock = {}
    for c in chunks:
        for r in c['s']:
            stock[r['seq']] = r['kind']
    pending, unknown, gap, prev = [], False, False, None
    for seq in sorted(stock):
        if prev is not None and any(x not in seen for x in range(prev + 1, seq)):
            gap = True
        prev = seq
        if stock[seq] == 'execute_model':
            state = emptiness(main, seq)
            if state is None:
                unknown = True
            elif state:
                pending.append(seq)
        elif stock[seq] == 'sample_tokens':
            if gap:
                why['seq_gap'] += 1
            elif unknown:
                why['unknown_emptiness'] += 1
            elif len(pending) == 1:
                pairs[pending[0]] = seq
                why['paired'] += 1
            elif not pending:
                why['none_pending'] += 1
            else:
                why['several_pending'] += 1
            pending, unknown, gap = [], False, False
    return pairs, why


def envelopes(anat_flushes, chunks, gap_records):
    """Per rank: blocking intervals (expanded / uncertain / engine fallback / gap instants) and core intervals."""
    block, core, have = {r: [] for r in range(4)}, {r: [] for r in range(4)}, {r: set() for r in range(4)}
    counts = dict(anatomy_chunks=0, uncertain=0, engine_fallback=0, gap_instants=0)
    for f in anat_flushes:
        if f['side'] != 'worker' or f['rank'] not in block:
            continue
        h = f['head']
        have[f['rank']].add((f['writer'][0], h.get('txn'), h.get('chunk_start'), h.get('chunk_end')))
        have[f['rank']].add((None, h.get('txn'), h.get('chunk_start'), h.get('chunk_end')))
        for c in f['rows'].get('c', []):
            first, last, read = at(f['base'], c[4]), at(f['base'], c[5]), num(c[9])
            counts['anatomy_chunks'] += 1
            if first is None or last is None:
                counts['uncertain'] += 1
                continue                          # unlocated: the engine window below blocks for it
            pad = (read or 0) * 1000 + QUANT_NS
            block[f['rank']].append((first - pad, last + pad))
            if (num(c[10]) or 0) == 0 and (num(c[12]) or 0) == 0 and read is not None and last - pad > first + pad:
                core[f['rank']].append((first + pad, last - pad))
            else:
                counts['uncertain'] += 1
    for c in chunks:
        ident = (None, c['txn'], c['chunk_start'], c['chunk_end'])
        for rank in block:
            if ident not in have[rank] and c['window'] is not None:
                block[rank].append(c['window'])
                counts['engine_fallback'] += 1
    for _, g in gap_records:
        t = num(g.get('observed_ns'))
        if t is not None:
            counts['gap_instants'] += 1
            for rank in block:
                block[rank].append((t, t))
    return ({r: merge(v) for r, v in block.items()}, {r: merge(v) for r, v in core.items()}, counts)


class Spans(list):
    """Sorted, merged, disjoint [lo, hi] intervals with bisect lookups."""

    def __init__(self, items=()):
        super().__init__(items)
        self.starts = [a for a, _ in self]
        self.ends = [b for _, b in self]


def merge(intervals):
    out = []
    for lo, hi in sorted(intervals):
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return Spans(out)


def spans(intervals):
    return intervals if isinstance(intervals, Spans) else merge(intervals)


def overlap(lo, hi, intervals):
    """Length of [lo, hi] covered by the union of the intervals."""
    s = spans(intervals)
    total, i = 0, bisect.bisect_left(s.ends, lo)
    while i < len(s) and s.starts[i] <= hi:
        total += max(0, min(hi, s.ends[i]) - max(lo, s.starts[i]))
        i += 1
    return total


def touches(lo, hi, intervals):
    """Whether [lo, hi] meets any interval (closed ends)."""
    s = spans(intervals)
    i = bisect.bisect_left(s.ends, lo)
    return i < len(s) and s.starts[i] <= hi


def clearance(lo, hi, intervals):
    """Distance from [lo, hi] to the nearest interval (0 when touching; None when there is none)."""
    s = spans(intervals)
    if not s:
        return None
    if touches(lo, hi, s):
        return 0
    i = bisect.bisect_left(s.ends, lo)
    gaps = ([lo - s.ends[i - 1]] if i > 0 else []) + ([s.starts[i] - hi] if i < len(s) else [])
    return min(gaps)


def classify(L, U, D, blocking, core_union):
    """certified far / inside, or why not."""
    if L is None or U is None:
        return 'unlocated'
    if U - L < D:
        return 'inconsistent'
    lo, hi = L - GUARD_NS, U + GUARD_NS
    if not touches(lo, hi, blocking):
        return 'far'
    outside = (U - L) - overlap(L, U, core_union)
    if max(0, D - outside) >= INSIDE_SHARE * D:
        return 'inside'
    return 'ambiguous'


def band_of(start):
    for lo, hi in BANDS:
        if type(start) is int and lo <= start < hi:
            return '%d-%d' % (lo, hi) if hi < 1 << 40 else '%d+' % lo
    return None


def width_of(chunk):
    if type(chunk['chunk_end']) is not int or type(chunk['chunk_start']) is not int:
        return None
    w = chunk['chunk_end'] - chunk['chunk_start']
    for lo, hi in WIDTHS:
        if lo <= w < hi:
            return '%d-%d' % (lo, hi)
    return '%d+' % WIDTHS[-1][1]


WIDTHS = ((0, 1025), (1025, 2049), (2049, 4097))


class Timeline:
    """native-gap route generation / engine epoch at a monotonic instant (last event at or before it)."""

    def __init__(self, gap_records):
        self._bisect = bisect
        rows = sorted((g['observed_ns'], g.get('route_generation'), g.get('engine_epoch'))
                      for _, g in gap_records if num(g.get('observed_ns')) is not None)
        self.t = [r[0] for r in rows]
        self.state = [(r[1], r[2]) for r in rows]

    def at(self, t):
        i = self._bisect.bisect_right(self.t, t) - 1
        return self.state[i] if i >= 0 else (None, None)

    def changes(self, lo, hi):
        i = self._bisect.bisect_right(self.t, lo) - 1
        j = self._bisect.bisect_right(self.t, hi) - 1
        return len(set(self.state[max(i, 0):j + 1])) > 1 if j >= 0 else False


# --- the contrast ----------------------------------------------------------------------------------------------------
def dist(values):
    values = sorted(x for x in (num(v) for v in values) if x is not None)
    if not values:
        return dict(n=0)
    return dict(n=len(values), p50=round(statistics.median(values), 4), mean=round(statistics.fmean(values), 4))


def ms(ns):
    value = num(ns)
    return None if value is None else value / 1e6


def share(flags):
    flags = [x for x in (num(f) for f in flags) if x is not None]
    return dict(n=len(flags), share=round(sum(1 for f in flags if f) / len(flags), 4) if flags else None)


def bootstrap(diffs_by_cluster, seed=1311):
    clusters = [v for v in diffs_by_cluster.values() if v]
    if len(clusters) < 2:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(BOOTSTRAP):
        pick = [x for _ in clusters for x in rng.choice(clusters)]
        means.append(statistics.fmean(pick))
    means.sort()
    return dict(clusters=len(clusters), ci95=[round(means[int(.025 * BOOTSTRAP)], 4),
                                              round(means[int(.975 * BOOTSTRAP)], 4)])


def contrast(items):
    """Pairs each certified busy (inside) forward with the nearest-in-time certified far forward of its stratum,
    without replacement; a stratum with busy forwards and no control left is INSUFFICIENT_SAME_ARM_SUPPORT."""
    strata = {}
    for it in items:
        if it['cls'] in ('inside', 'far'):
            strata.setdefault(it['stratum'], {'inside': [], 'far': []})[it['cls']].append(it)
    pairs, support = [], []
    for key, groups in sorted(strata.items(), key=lambda kv: str(kv[0])):
        busy, free = sorted(groups['inside'], key=lambda x: x['t']), sorted(groups['far'], key=lambda x: x['t'])
        row = dict(stratum=list(key), busy=len(busy), far=len(free), paired=0)
        for b in busy:
            if not free:
                break
            f = min(free, key=lambda x: (abs(x['t'] - b['t']), x['t']))
            free.remove(f)
            pairs.append((b, f))
            row['paired'] += 1
        if row['paired'] < row['busy']:
            row['verdict'] = 'INSUFFICIENT_SAME_ARM_SUPPORT'
        support.append(row)
    diff = [ms(b['D'] - f['D']) for b, f in pairs]
    host_lb = [ms(b['D'] - b['H'] - b['observer'] - f['D']) if None not in (b['H'], b['observer']) else None
               for b, f in pairs]
    clusters = {}
    for (b, f), d in zip(pairs, diff):
        clusters.setdefault(b['cluster'], []).append(d)
    uid_known = [(b['uid'], f['uid']) for b, f in pairs if b.get('uid') and f.get('uid')]
    return dict(pairs=len(pairs), paired_D_diff_ms=dist(diff), host_only_lower_bound_ms=dist(host_lb),
                pairs_same_uid_set=dict(known=len(uid_known), same=sum(1 for a, b in uid_known if a == b)),
                pairs_control_outside=sum(1 for _, f in pairs if f.get('send_source') == 'transport-send'),
                host_only_lower_bound_positive=share([x > 0 for x in host_lb if x is not None]),
                bootstrap_mean_diff_by_native_chunk=bootstrap(clusters), support=support,
                insufficient=sum(1 for s in support if s.get('verdict')))


def describe(items):
    out = {}
    for cls in ('inside', 'far'):
        xs = [i for i in items if i['cls'] == cls]
        out[cls] = dict(n=len(xs), D_ms=dist([ms(i['D']) for i in xs]), H_ms=dist([ms(i['H']) for i in xs]),
                        H_cpu_ms=dist([ms(i['H_cpu']) for i in xs]), observer_ms=dist([ms(i['observer']) for i in xs]),
                        idle_before=share([i['idle_before'] for i in xs]), idle_after=share([i['idle_after'] for i in xs]),
                        enter_minus_submit_ms=dist([ms(i['queue']) for i in xs]),
                        native_handler_in_queue_ms=dist([ms(i['queue_native']) for i in xs]),
                        handle_output_ms=dist([ms(i['handle_output']) for i in xs]),
                        sample_get_output_ms=dist([ms(i['get_output']) for i in xs]),
                        mq_write_ms=dist([ms(i['mq']) for i in xs]),
                        bracket_U_minus_L_ms=dist([ms(i.get('bracket')) for i in xs]),
                        send_source={s: sum(1 for i in xs if i.get('send_source') == s)
                                     for s in ('engine-progress', 'transport-send')})
    return out


def read(root, log_override=None):
    path = engine_log(root, log_override)
    records, torn, identities = read_records(lines_of(path))
    if not records['handler']:
        refuse('no flash-s5 handler rows in %s (FLASH_S5_HANDLER_ANATOMY off?)' % path)
    if not records['anatomy']:
        refuse('no flash-s5 anatomy rows in %s: no native envelope, nothing can be certified' % path)
    loss = {'torn_lines': torn}
    hf, loss['handler'] = assemble(records['handler'])
    pf, loss['progress'] = assemble(records['progress'])
    af, loss['anatomy'] = assemble(records['anatomy'])
    tf, loss['timeline'] = assemble(records['timeline'], extra=TIMELINE_KEY, line_gaps=False)
    arms = {}
    for name in sorted(os.listdir(root)):
        if name.startswith('arm-') and name.endswith('.json'):
            with open(os.path.join(root, name)) as handle:
                doc = json.load(handle)
            arms[name[4:-5]] = (num(doc.get('wall_start_s')), num(doc.get('wall_end_s')))
    if not any(a.startswith('n3.') for a in arms):
        refuse('no n3 arm file (arm-n3.*.json) in %s' % root)
    gaps = receipt_gaps(identities)
    loss['receipt_writers'] = len(identities)
    loss['receipt_seq_missing'] = sum(b - a - 1 for v in gaps.values() for a, b in v)
    return dict(path=path, handler=hf, progress=pf, anatomy=af, timeline=tf, gaps=records['gap'], loss=loss, arms=arms,
                receipt_gaps=gaps)


def arm_owner(chunk, arms):
    """The arm whose wall window holds the whole chunk: its log stamp (emission, at or after the last event) minus
    the chunk's monotonic span and the log lag must start inside the window, the stamp itself must end inside it.
    None: no arm; 'boundary': touches an arm boundary (ambiguous, excluded)."""
    if chunk['stamp'] is None or chunk['window'] is None:
        return None
    hi = chunk['stamp']
    lo = hi - (chunk['window'][1] - chunk['window'][0]) / 1e9 - LOG_LAG_S
    for name, (start, end) in arms.items():
        if start is None or end is None:
            continue
        if start <= lo and hi <= end:
            return name
        if lo <= end and hi >= start:
            return 'boundary'
    return None


def upper_bound(outs, sample, rank):
    """U for this rank's execute and the sample output row it rests on.

    Only an AsyncOutput row whose get_output completion is recorded is a proof (the copy behind the main stream was
    synchronised). Stock returns the sampled output from one last-stage rank (rank 2 here), so only that rank carries
    the proof. Its emit bounds the forwards it depends on: its own, and stage 0's (whose pp_send its forward
    received). A rank without its own proof on the last stage (rank 3) stays unlocated: its trailing kernels after
    the last collective are not ordered before rank 2's copy."""
    proofs = {}
    for r in (2, 3):
        proven = [o for o in outs.get((r, sample), []) if num(o['get_output_ns']) is not None and o['t_emit'] is not None]
        if proven:
            proofs[r] = max(proven, key=lambda o: o['t_emit'])
    if rank in (2, 3):
        row = proofs.get(rank)
    else:
        row = max(proofs.values(), key=lambda o: o['t_emit']) if proofs else None
    return (row['t_emit'] + QUANT_NS, row) if row else (None, None)


# --- decode-timeline stamp (FLASH_S5_DECODE_TIMELINE) -----------------------------------------------------------------
def timeline_rows(flushes):
    """d rows per (rank, seq) with their session identity, and the stamp's own loss facts.

    Host ``t_*`` fields are restored like every ProgressTrace (base_ns + offset us x 1000). A second, different d row
    for one (rank, seq) is a conflict: that forward is inconsistent, never resolved by picking one. Sessions are
    numbered 1..N per worker process (native ints, outside 'outside-N'); a missing number below the highest seen is a
    lost session line. Every ``s`` counter is summed as written; nothing is filled."""
    rows, conflicts = {}, set()
    stats = dict(sessions_native=0, sessions_outside=0, sessions_incomplete=0, d_rows=0, p_rows=0, dropped=0,
                 sample_gaps=0, log_budget_exhausted_sessions=0, sessions_without_s_row=0, counters={})
    numbers = {}
    for f in flushes:
        h = f['head']
        outside = h.get('scope') == OUTSIDE_SCOPE
        stats['sessions_outside' if outside else 'sessions_native'] += 1
        stats['dropped'] += num(h.get('dropped')) or 0
        sample = h.get('sample')
        if outside and type(sample) is str and sample.startswith('outside-') and sample[8:].isdigit():
            numbers.setdefault((f['writer'], f['rank'], 'outside'), set()).add(int(sample[8:]))
        elif not outside and type(sample) is int:
            numbers.setdefault((f['writer'], f['rank'], 'native'), set()).add(sample)
        s_rows = f['rows'].get('s', [])
        if not s_rows:
            stats['sessions_without_s_row'] += 1
        for s in s_rows:
            complete, counters = (s + [None, None])[:2]
            if complete is not True:
                stats['sessions_incomplete'] += 1
            for name, value in (counters or {}).items():
                if num(value) is not None:
                    stats['counters'][name] = stats['counters'].get(name, 0) + value
            if (counters or {}).get('log_budget_exhausted'):
                stats['log_budget_exhausted_sessions'] += 1
        stats['p_rows'] += len(f['rows'].get('p', []))
        for item in f['rows'].get('d', []):
            r = dict(zip(D_FIELDS, item))
            for name in ('t_enter', 't_exit', 't_complete_query'):
                r[name] = at(f['base'], r[name])
            r.update(rank=f['rank'], writer=f['writer'], stamp=f['stamp'], scope='outside' if outside else 'native',
                     sample=sample, engine_epoch=h.get('engine_epoch'), stage=h.get('stage'),
                     engine_send_ns=num(h.get('engine_send_ns')) if outside else None,
                     execute_seq=h.get('execute_seq') if outside else None)
            stats['d_rows'] += 1
            key = (f['rank'], r['seq'])
            if key in rows and json.dumps(rows[key], sort_keys=True, default=str) != json.dumps(
                    r, sort_keys=True, default=str):
                conflicts.add(key)
            rows.setdefault(key, r)
    for seen in numbers.values():
        stats['sample_gaps'] += max(seen) - len(seen) if seen else 0
    stats['conflicting_d_rows'] = len(conflicts)
    return rows, conflicts, stats


def own_upper_bound(timeline, rank, seq, outs, sample):
    """Timeline mode: U from this rank's own queried forward completion (t_complete_query + 1 us), never borrowed
    from another rank's get_output. No own d row: (None, None, 'no_own_completion')."""
    row = timeline.get((rank, seq))
    if row is None or row['t_complete_query'] is None:
        return None, None, 'no_own_completion'
    return row['t_complete_query'] + QUANT_NS, row, None


def outside_arm(lo, hi, windows, owners):
    """The n3 arm of an outside-native-progress forward [lo, hi] (engine send .. own completion): the engine chunks
    nearest before lo and after hi, and any overlapping it, must all belong to one arm (each chunk certified by
    arm_owner). A missing neighbour on either side: None; mixed owners: 'boundary'. Never the log stamp of the
    timeline line (emission lags the forward by the flush)."""
    before = [c for c in windows if c['window'][1] <= lo]
    after = [c for c in windows if c['window'][0] >= hi]
    if not before or not after:
        return None
    near = [max(before, key=lambda c: c['window'][1]), min(after, key=lambda c: c['window'][0])]
    near += [c for c in windows if c['window'][0] < hi and c['window'][1] > lo]
    names = {owners[id(c)] for c in near}
    if len(names) == 1:
        return names.pop()
    return 'boundary'


def meta_stratum(arm, row):
    """Exact-cell stratum from the forward's own metadata, or None when unsupported (Codex: forward_calls 1, valid
    handler flags, a descriptor and host history; multi-forward / sub-cohort membership is unsupported)."""
    meta = row.get('metadata') if row else None
    if type(meta) is not dict:
        return None
    history = meta.get('history_tokens')
    if (meta.get('forward_calls') != 1 or meta.get('handler_flags') != 0 or type(history) is not list or not history
            or any(type(x) is not int for x in history) or meta.get('bank') not in ('split', 'full')
            or type(meta.get('num_tokens')) is not int or meta.get('cg_mode') is None):
        return None
    return (arm, meta['bank'], meta.get('route_generation'), row.get('engine_epoch'), meta.get('bank_sms'),
            meta['cg_mode'], meta['num_tokens'], meta.get('num_reqs'), meta.get('actual_rows'), band_of(max(history)))


def over_stages(values):
    """The critical PP path over stages: the max (stage 1 waits on stage 0's send), never the sum."""
    return max(values)


def critical(per_rank):
    """One execute's value on the critical PP path from per-rank values: max of TP peers per stage, then over stages.
    Any rank's value missing (None): None, never a partial max."""
    if any(per_rank.get(r) is None for r in range(4)):
        return None
    return over_stages([stage_value([per_rank[a], per_rank[b]]) for a, b in STAGES.values()])


def cadence(chunks, main, seen):
    """execute seq -> engine submit interval to the next nonempty execute of the same chunk (READING item: engine
    successive nonempty execute submission intervals). Every execute between them must be proven empty and every
    transport seq between them in the record; otherwise no interval."""
    out = {}
    for c in chunks:
        execs = [r for r in c['s'] if r['kind'] == 'execute_model']
        last = None
        for r in execs:
            state = emptiness(main, r['seq'])
            if state is None:
                last = None
                continue
            if not state:
                continue
            if (last is not None and last['t_submit'] is not None and r['t_submit'] is not None
                    and all(x in seen for x in range(last['seq'] + 1, r['seq']))):
                out[last['seq']] = r['t_submit'] - last['t_submit']
            last = r
    return out


def bin_of(ms_value):
    if ms_value is None:
        return None
    if ms_value <= 0:
        return '0'
    edges = QUEUE_BINS_MS
    for lo, hi in zip(edges, edges[1:]):
        if lo < ms_value <= hi:
            return '%g-%g' % (lo, hi)
    return '>%g' % edges[-1]


def exposed_wait(items, chunks, main, seen, lost):
    """How much of the worker queue delay behind native handlers shows up in decode cadence (descriptive).

    Per execute recorded in a native chunk (engine progress send): queue_native per rank, deduplicated to the critical
    PP path (critical); the exposed candidate is a rank's queue_native when its stream was idle at launch (0 when the
    host launched behind queued GPU work, None when idle_before is null); binned by the critical queue_native, the
    median engine interval to the next nonempty execute. Outside samples give the same-arm queue baseline without
    native handlers. A queue interval touching a loss window is left out."""
    gaps = cadence(chunks, main, seen)
    by_seq = {}
    for i in items:
        if i['send_source'] != 'engine-progress' or i['q_lo'] is None or i['t'] is None:
            continue
        if touches(i['q_lo'], i['t'], lost):
            continue
        by_seq.setdefault(i['seq'], {})[i['rank']] = i
    rows, groups = [], {}
    skipped = dict(ranks_missing=0)
    for seq, ranks in sorted(by_seq.items()):
        qn = critical({r: ranks[r]['queue_native'] if r in ranks else None for r in range(4)})
        if qn is None:
            skipped['ranks_missing'] += 1
            continue
        exp = critical({r: (None if r not in ranks or num(ranks[r]['idle_before']) is None else
                            (ranks[r]['queue_native'] if ranks[r]['idle_before'] else 0)) for r in range(4)})
        any_rank = ranks[0]
        rows.append(dict(seq=seq, arm=any_rank['arm'], mode=any_rank['ctx_mode'], live=any_rank['ctx_live'],
                         queue_native_crit=qn, queue_crit=critical({r: ranks[r]['queue'] for r in ranks}),
                         exposed_candidate=exp, cadence=gaps.get(seq),
                         idle=[num(ranks[r]['idle_before']) for r in range(4)]))
    for r in rows:
        groups.setdefault((r['arm'], r['mode'], r['live'], bin_of(ms(r['queue_native_crit']))), []).append(r)
    table = []
    for (arm, mode, live, b), xs in sorted(groups.items(), key=lambda kv: tuple(str(k) for k in kv[0])):
        table.append(dict(arm=arm, mode=mode, decode_live=live, bin_ms=b, n=len(xs),
                          cadence_ms=dist([ms(x['cadence']) for x in xs]),
                          queue_crit_ms=dist([ms(x['queue_crit']) for x in xs]),
                          queue_native_crit_ms=dist([ms(x['queue_native_crit']) for x in xs]),
                          exposed_candidate_ms=dist([ms(x['exposed_candidate']) for x in xs]),
                          exposed_candidate_known=sum(1 for x in xs if x['exposed_candidate'] is not None),
                          idle_before=share([v for x in xs for v in x['idle']])))
    base = {}
    for i in items:
        if i['send_source'] == 'transport-send' and i['queue'] is not None:
            base.setdefault('%s rank %d' % (i['arm'], i['rank']), []).append(ms(i['queue']))
    return dict(rows=len(rows), skipped=skipped, table=table,
                outside_queue_baseline_ms={k: dist(v) for k, v in sorted(base.items())},
                note=EXPOSED_NOTE)


EXPOSED_NOTE = (
    'Descriptive and conditional (READING 13 Nov: 1.1-3.2 ms is host occupancy overlapping an execute\'s queue '
    'interval, not a matched reduction in output cadence; the 515-1499 agent-s ceiling is conditional gross pricing). '
    'Critical path = max of TP peers per stage, then max over stages, never a sum of rank delays. The exposed '
    'candidate counts occupancy only where the stream was idle at launch; it is an upper reading, not a removable '
    'saving: engine t_submit precedes bounded_enqueue, so part of the occupancy can precede availability of the decode '
    'message in the worker queue, and native commands already executing cannot be overtaken. Cadence differences '
    'between bins are confounded by whatever native work also slows the busy forward itself. Cadence exists only '
    'inside recorded native chunks; outside samples give the queue baseline only.')


def analyse(data):
    main, outs, hcounters = handler_rows(data['handler'])
    data['loss']['handler_counters'] = hcounters
    tl, tl_conflicts, tl_stats = timeline_rows(data.get('timeline', []))
    data['loss']['timeline_stamp'] = tl_stats
    timeline_mode = bool(tl)                       # own completion bounds; the legacy borrowed U is never used then
    chunks = engine_chunks(data['progress'])
    seen = observed_seqs(data['progress'], main)
    pairs, pairing = pair_samples(chunks, main, seen)
    blocking, core, env_counts = envelopes(data['anatomy'], chunks, data['gaps'])
    gap_loss, gap_missing = gap_loss_windows(data['gaps'])
    data['loss']['native_gap_missing'] = gap_missing
    lost = merge(loss_windows(data['handler'], 'handler') + loss_windows(data['progress'], 'progress') +
                 loss_windows(data['anatomy'], 'anatomy') + gap_loss +
                 receipt_loss_windows(data.get('receipt_gaps', {}),
                                      (data['handler'], data['progress'], data['anatomy'], data.get('timeline', [])),
                                      data['gaps']))
    data['loss']['loss_windows'] = len(lost)
    timeline = Timeline(data['gaps'])
    native_busy = {r: [] for r in range(4)}         # worker RPC-thread time in native (flash_s5_*) handlers
    for f in data['progress']:
        if f['side'] == 'worker' and f['rank'] in native_busy:
            for x in f['rows'].get('r', []):
                lo, hi = at(f['base'], x[2]), at(f['base'], x[3])
                if str(x[1]).startswith('flash_s5_') and lo is not None and hi is not None:
                    native_busy[f['rank']].append((lo, hi))
    native_busy = {r: merge(v) for r, v in native_busy.items()}
    submit, chunk_of, owner = {}, {}, {}
    arm_counts, owners = {}, {}
    for c in chunks:
        arm = arm_owner(c, data['arms'])
        owners[id(c)] = arm
        arm_counts[str(arm)] = arm_counts.get(str(arm), 0) + 1
        for r in c['s']:
            submit[r['seq']] = r['t_submit']
            chunk_of[r['seq']] = c
            owner[r['seq']] = arm
    windows = sorted((c for c in chunks if c['window'] is not None), key=lambda c: c['window'])
    starts = [c['window'][0] for c in windows]
    excluded = dict(flagged_failed_aborted_install_unresolved_nested=0, no_dev_forward=0, no_engine_send=0,
                    not_n3_arm=0, arm_boundary=0, unpaired=0, record_loss_in_window=0, route_or_epoch_transition=0)
    if timeline_mode:
        excluded.update(no_own_completion=0, timeline_mismatch_or_conflict=0, metadata_unsupported=0,
                        outside_no_neighbour_chunk=0)
    outside_used = 0
    items, estimated = [], []
    for (rank, seq), row in sorted(main.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if row['kind'] != 'e' or rank not in blocking:
            continue
        if num(row['flags']) is None or row['flags'] != 0:
            excluded['flagged_failed_aborted_install_unresolved_nested'] += 1
            continue
        D = num(row['dev_forward_ns'])
        if D is None:
            excluded['no_dev_forward'] += 1
            continue
        d_row = tl.get((rank, seq)) if timeline_mode else None
        outside = (d_row is not None and d_row['scope'] == 'outside' and d_row['execute_seq'] == seq
                   and d_row['engine_send_ns'] is not None)
        if seq in owner:
            arm, q_lo, send_source = owner[seq], submit.get(seq), 'engine-progress'
        elif outside:
            hi = d_row['t_complete_query'] if d_row['t_complete_query'] is not None else row['t_enter']
            arm = outside_arm(d_row['engine_send_ns'], hi, windows, owners) if hi is not None else None
            q_lo, send_source = d_row['engine_send_ns'], 'transport-send'
            if arm is None:
                excluded['outside_no_neighbour_chunk'] += 1
                continue
        else:
            excluded['no_engine_send'] += 1
            continue
        if arm == 'boundary':
            excluded['arm_boundary'] += 1
            continue
        if arm is None or not arm.startswith('n3.'):
            excluded['not_n3_arm'] += 1
            continue
        L = row['t_enter']
        sample = pairs.get(seq)
        U, sample_row, why = None, None, None
        if timeline_mode:
            U, own_row, why = own_upper_bound(tl, rank, seq, outs, sample)
            if why:
                excluded[why] += 1
            elif ((rank, seq) in tl_conflicts or own_row['device_ns'] != D or own_row['t_enter'] is None
                  or L is None or abs(own_row['t_enter'] - L) >= QUANT_NS):
                excluded['timeline_mismatch_or_conflict'] += 1
                U, why = None, 'timeline_mismatch_or_conflict'
            if sample is not None:
                sample_row = upper_bound(outs, sample, rank)[1]    # get_output column only, never U
        elif sample is None:
            excluded['unpaired'] += 1
        else:
            U, sample_row = upper_bound(outs, sample, rank)
        cls = classify(L, U, D, blocking[rank], core[rank])
        if why == 'timeline_mismatch_or_conflict':
            cls = 'inconsistent'
        if cls in ('far', 'inside', 'ambiguous'):
            if touches(L - GUARD_NS, U + GUARD_NS, lost):
                excluded['record_loss_in_window'] += 1
                cls = 'unsupported_loss'
            elif timeline.changes(L, U):
                excluded['route_or_epoch_transition'] += 1
                cls = 'transition'
        ctx = context_chunk(windows, starts, L, U) or chunk_of.get(seq)
        route, epoch = timeline.at(L) if L is not None else (None, None)
        if timeline_mode:
            stratum = meta_stratum(arm, d_row)
            if stratum is None and cls in ('far', 'inside', 'ambiguous'):
                excluded['metadata_unsupported'] += 1
                cls = 'metadata_unsupported'
            group = 'bank=%s' % ((d_row or {}).get('metadata') or {}).get('bank')
        else:
            stratum = (arm, route, epoch, ctx['mode'], ctx['decode_live'], band_of(ctx['chunk_start']), width_of(ctx))
            group = ctx['decode_live']
        if outside and cls in ('far', 'inside', 'ambiguous'):
            outside_used += 1
        fwd = (row['seams'] or {}).get('forward') or {}
        own = outs.get((rank, seq), [])
        items.append(dict(rank=rank, seq=seq, arm=arm, cls=cls, t=L, D=D, stratum=stratum, group=group,
                          send_source=send_source, q_lo=q_lo, why=why,
                          ctx_mode=ctx['mode'] if ctx else None, ctx_live=ctx['decode_live'] if ctx else None,
                          uid=((d_row or {}).get('metadata') or {}).get('uid_digest'),
                          bracket=(U - L) if None not in (U, L) else None,
                          clear=clearance(L, U, blocking[rank]) if L is not None and U is not None else None,
                          cluster=(ctx['writer'], ctx['txn'], ctx['chunk_start']) if ctx and not outside
                          else ('outside', seq),
                          H=num(fwd.get('ns')), H_cpu=num(fwd.get('cpu_ns')), observer=num(row['observer_ns']),
                          idle_before=num(row['idle_before_launch']), idle_after=num(row['idle_after_launch']),
                          queue=(L - q_lo) if L is not None and q_lo is not None else None,
                          queue_native=overlap(q_lo, L, native_busy[rank]) if L is not None and q_lo is not None
                          else None,
                          handle_output=num(row['handle_output_ns']),
                          get_output=num(sample_row['get_output_ns']) if sample_row else None,
                          mq=num(own[0]['mq_write_ns']) if own else None))
        if U is not None:                          # ESTIMATED only: the old endpoint reconstruction
            lo = U - D
            est = 'far' if not touches(lo - GUARD_NS, U + GUARD_NS, blocking[rank]) \
                else ('busy' if touches(lo, U, blocking[rank]) else 'between')
            estimated.append(dict(rank=rank, arm=arm, cls=est, D=D))
    reach = {}
    for i in items:
        if i['clear'] is not None:
            reach.setdefault('%s rank %d' % (i['arm'], i['rank']), []).append(i['clear'] / 1e6)
    reach = {k: dict(n=len(v), p50_ms=round(statistics.median(v), 3), max_ms=round(max(v), 3),
                     at_least_guard=sum(1 for x in v if x * 1e6 > GUARD_NS)) for k, v in sorted(reach.items())}
    return items, estimated, dict(clearance_of_bracketed=reach, pairing=pairing, envelopes=env_counts, excluded=excluded, chunk_arms=arm_counts,
                                  engine_chunks=len(chunks), clear_gaps=clear_gaps(chunks, data['arms'], blocking),
                                  timeline_mode=timeline_mode, outside_certified_or_ambiguous=outside_used,
                                  exposed_wait=exposed_wait(items, chunks, main, seen, lost))


def clear_gaps(chunks, arms, blocking):
    """Per n3 arm and rank: the stretches between blocking envelopes inside the arm's chunk extent, and how many are
    long enough (> 2 x guard) to hold any certified far forward at all."""
    out = {}
    for name in sorted(arms):
        if not name.startswith('n3.'):
            continue
        mine = [c['window'] for c in chunks if c['window'] is not None and arm_owner(c, arms) == name]
        if not mine:
            continue
        lo, hi = min(w[0] for w in mine), max(w[1] for w in mine)
        for rank, env in blocking.items():
            inside = [(max(a, lo), min(b, hi)) for a, b in env if b >= lo and a <= hi]
            gaps = [b[0] - a[1] for a, b in zip(inside, inside[1:])]
            long = [g for g in gaps if g > 2 * GUARD_NS]
            out['%s rank %d' % (name, rank)] = dict(extent_s=round((hi - lo) / 1e9, 1),
                                                     max_gap_ms=round(max(gaps) / 1e6, 1) if gaps else None,
                                                     gaps_over_2guard=len(long),
                                                     time_in_those_s=round(sum(long) / 1e9, 2))
    return out


def context_chunk(windows, starts, L, U):
    """The engine chunk overlapping [L, U] most, else the nearest one to L (windows sorted by start)."""
    if L is None or U is None or not windows:
        return None
    i = bisect.bisect_right(starts, U)
    near = windows[max(0, i - 8):i + 1]
    best = max(near, key=lambda c: (max(0, min(U, c['window'][1]) - max(L, c['window'][0])),
                                    -min(abs(c['window'][0] - L), abs(c['window'][1] - L))))
    return best


def join_peer(by_rank, peer, item):
    """The TP peer's row for the same transport seq (never the row at the same position in the peer's list)."""
    return by_rank.get(peer, {}).get(item['seq'])


def stage_items(items):
    """Per PP stage and seq: both TP peers present with the same certified class; the stage D is their max."""
    by_rank = {}
    for i in items:
        by_rank.setdefault(i['rank'], {})[i['seq']] = i
    out = []
    for stage, (a, b) in STAGES.items():
        for seq, x in sorted(by_rank.get(a, {}).items()):
            y = join_peer(by_rank, b, x)
            if y is None or y['cls'] != x['cls']:
                continue
            out.append(dict(x, rank='stage%d' % stage, D=stage_value([x['D'], y['D']]),
                            peers={a: ms(x['D']), b: ms(y['D'])},
                            H=None if None in (x['H'], y['H']) else stage_value([x['H'], y['H']]),
                            observer=None if None in (x['observer'], y['observer'])
                            else stage_value([x['observer'], y['observer']])))
    return out


def report(data, items, estimated, meta):
    groups = {}
    for i in items + stage_items(items):
        groups.setdefault((str(i['rank']), i['arm'], i['group']), []).append(i)
    tables = []
    for key, xs in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]), str(kv[0][2]))):
        counts = {c: sum(1 for x in xs if x['cls'] == c) for c in CLASSES}
        row = dict(rank=key[0], arm=key[1], decode_live=key[2], group=key[2], counts=counts, certified=describe(xs),
                   contrast=contrast(xs))
        if key[0].startswith('stage'):
            peers = [x['peers'] for x in xs if x['cls'] in ('inside', 'far')]
            ranks = sorted(peers[0]) if peers else []
            row['peers_side_by_side_ms'] = {
                cls: {str(r): dist([x['peers'][r] for x in xs if x['cls'] == cls]) for r in ranks}
                for cls in ('inside', 'far')}
        tables.append(row)
    est = {}
    for e in estimated:
        est.setdefault((e['rank'], e['arm'], e['cls']), []).append(ms(e['D']))
    return dict(schema=SCHEMA, engine_log=data['path'], cannot_say=list(CANNOT_SAY), loss=data['loss'], meta=meta,
                rules=dict(guard_ms=GUARD_NS / 1e6, inside_share=INSIDE_SHARE, quantisation_ns=QUANT_NS,
                           bootstrap=BOOTSTRAP, stage_value='max of TP peers',
                           strata=list(STRATUM_TIMELINE if meta.get('timeline_mode') else STRATUM),
                           upper_bound=('own: t_complete_query + 1 us on every rank (decode-timeline stamp); never '
                                        'borrowed' if meta.get('timeline_mode') else
                                        'borrowed: rank 2 AsyncOutput t_emit + 1 us for ranks 0/1/2; rank 3 unlocated'),
                           live_and_band='decode_live / chunk band of the overlapping (else nearest) engine chunk header:'
                                         ' a proxy for the cohort, not the forward\'s padded descriptor',
                           not_computed=['client adjacent token gaps (turn journals not joined by this reader)',
                                         'fused-phase / producer-witness copy durations (recipe item 6)']),
                tables=tables,
                ESTIMATED=dict(note='old endpoint reconstruction [U - D, U]; descriptive only, never in the contrast',
                               rows=[dict(rank=k[0], arm=k[1], cls=k[2], D_ms=dist(v)) for k, v in sorted(
                                   est.items(), key=lambda kv: (kv[0][0], str(kv[0][1]), kv[0][2]))]))


CLASSES = ('far', 'inside', 'ambiguous', 'unlocated', 'inconsistent', 'unsupported_loss', 'transition',
           'metadata_unsupported')
STRATUM = ('arm', 'route_generation', 'engine_epoch', 'merge_mode', 'decode_live', 'history_band', 'chunk_width')
STRATUM_TIMELINE = ('arm', 'bank', 'route_generation', 'engine_epoch', 'bank_sms', 'cg_mode', 'num_tokens', 'num_reqs',
                    'actual_rows', 'max_history_band')


def loss_line(doc):
    l = doc['loss']
    hc = l['handler_counters']
    return ('loss: torn %s | handler missing_parts %d conflicts %d line_gaps %d rows_dropped %d unresolved %d late %d '
            'failed %d | progress missing_parts %d conflicts %d line_gaps %d dropped %d | anatomy missing_parts %d '
            'conflicts %d line_gaps %d dropped %d | native-gap missing %d | receipt seq missing %d (%d writers) | loss windows %d' % (
                json.dumps(l['torn_lines'], sort_keys=True), l['handler']['missing_parts'],
                l['handler']['conflicting_duplicates'], l['handler']['line_gaps'], hc.get('rows_dropped', 0),
                hc.get('events_unresolved_at_flush', 0), hc.get('events_late', 0), hc.get('failed_rows', 0),
                l['progress']['missing_parts'], l['progress']['conflicting_duplicates'], l['progress']['line_gaps'],
                l['progress']['dropped'], l['anatomy']['missing_parts'], l['anatomy']['conflicting_duplicates'],
                l['anatomy']['line_gaps'], l['anatomy']['dropped'], l['native_gap_missing'], l.get('receipt_seq_missing', 0),
                l.get('receipt_writers', 0), l['loss_windows'])) + timeline_loss(l)


def timeline_loss(l):
    if 'timeline' not in l or not l['timeline']['flushes'] and not l['timeline']['missing_parts']:
        return ''
    s, a = l['timeline_stamp'], l['timeline']
    c = s['counters']
    return (' | timeline missing_parts %d conflicts %d sample_gaps %d incomplete %d of %d sessions dropped %d d-row '
            'conflicts %d budget_exhausted %d pending %d timing_errors %d unresolved_forward %d metadata_errors %d' % (
                a['missing_parts'], a['conflicting_duplicates'], s['sample_gaps'], s['sessions_incomplete'],
                s['sessions_native'] + s['sessions_outside'], s['dropped'], s['conflicting_d_rows'],
                s['log_budget_exhausted_sessions'], c.get('pending', 0), c.get('timing_errors', 0),
                c.get('unresolved_forward', 0), c.get('metadata_errors', 0)))


def text(doc):
    m = doc['meta']
    out = list(doc['cannot_say']) + [
        '', 'engine log: %s' % doc['engine_log'],
        'rules: far = [L-%g ms, U+%g ms] clear of every expanded same-rank native envelope; inside = max(0, D - '
        '|[L,U] outside core|) >= %.1f D; stage = max of TP peers; strata %s' % (
            doc['rules']['guard_ms'], doc['rules']['guard_ms'], doc['rules']['inside_share'], ','.join(doc['rules']['strata'])),
        'engine chunks %d, by arm %s' % (m['engine_chunks'], json.dumps(m['chunk_arms'], sort_keys=True)),
        'pairing: %s' % json.dumps(m['pairing'], sort_keys=True),
        'envelopes: %s' % json.dumps(m['envelopes'], sort_keys=True),
        'excluded executes: %s' % json.dumps(m['excluded'], sort_keys=True),
        'clear stretches (between blocking envelopes, n3 arm extent): %s' % json.dumps(m['clear_gaps'], sort_keys=True),
        'clearance of each bracketed forward [L,U] from the nearest blocking envelope (far needs > %g ms): %s' % (
            doc['rules']['guard_ms'], json.dumps(m['clearance_of_bracketed'], sort_keys=True)),
        'not computed: %s' % '; '.join(doc['rules']['not_computed']),
        'upper bound: %s' % doc['rules']['upper_bound']]
    if m.get('timeline_mode'):
        s = doc['loss']['timeline_stamp']
        out.append('timeline stamp: sessions native %d outside %d | d rows %d p rows %d | outside forwards certified or '
                   'ambiguous %d | counters %s' % (s['sessions_native'], s['sessions_outside'], s['d_rows'], s['p_rows'],
                                                   m['outside_certified_or_ambiguous'],
                                                   json.dumps(s['counters'], sort_keys=True)))
    out.append('')
    fmt = lambda d: '-' if not d or not d.get('n') else '%.3f/%.3f (n %d)' % (d['p50'], d['mean'], d['n'])
    out.append('CERTIFIED  rank arm live | far inside ambig unloc loss trans | D far p50/mean | D inside p50/mean | '
               'pairs | paired dD p50/mean | host-only LB p50/mean | boot ci95 (chunks)')
    for t in doc['tables']:
        c, cert, k = t['counts'], t['certified'], t['contrast']
        boot = k['bootstrap_mean_diff_by_native_chunk']
        out.append('  %s %s live %s | %d %d %d %d %d %d | %s | %s | %d | %s | %s | %s%s' % (
            t['rank'], t['arm'], t['decode_live'], c['far'], c['inside'], c['ambiguous'], c['unlocated'],
            c['unsupported_loss'], c['transition'], fmt(cert['far']['D_ms']), fmt(cert['inside']['D_ms']), k['pairs'],
            fmt(k['paired_D_diff_ms']), fmt(k['host_only_lower_bound_ms']),
            '-' if not boot else '%s (%d)' % (boot['ci95'], boot['clusters']),
            ' | INSUFFICIENT_SAME_ARM_SUPPORT in %d of %d strata' % (k['insufficient'], len(k['support']))
            if k['insufficient'] else ''))
        for cls in ('inside', 'far'):
            d = cert[cls]
            if d['n']:
                out.append('      %-6s H %s cpu %s obs %s idle-before %s idle-after %s enter-submit %s native-in-queue %s '
                           'handle_out %s get_output %s mq %s' % (
                               cls, fmt(d['H_ms']), fmt(d['H_cpu_ms']), fmt(d['observer_ms']), d['idle_before']['share'],
                               d['idle_after']['share'], fmt(d['enter_minus_submit_ms']),
                               fmt(d['native_handler_in_queue_ms']), fmt(d['handle_output_ms']),
                               fmt(d['sample_get_output_ms']), fmt(d['mq_write_ms'])))
        if 'peers_side_by_side_ms' in t:
            for cls, peers in t['peers_side_by_side_ms'].items():
                if peers:
                    out.append('      peers %-6s %s' % (cls, '  '.join('rank %s %s' % (r, fmt(d)) for r, d in peers.items())))
        out.append('      ' + loss_line(doc))
        if k.get('pairs'):
            out.append('      pairs with an outside-sample control %d | same uid set %d of %d known' % (
                k['pairs_control_outside'], k['pairs_same_uid_set']['same'], k['pairs_same_uid_set']['known']))
    e = m['exposed_wait']
    out += ['', 'EXPOSED-WAIT (native-chunk executes, critical PP path; descriptive, conditional) rows %d skipped %s' % (
        e['rows'], json.dumps(e['skipped'], sort_keys=True)), '  ' + e['note'],
        '  arm mode live | queue_native bin ms | n | cadence p50/mean | queue p50/mean | queue_native p50/mean | '
        'exposed candidate p50/mean (known) | idle-before share']
    for r in e['table']:
        out.append('  %s %s %s | %s | %d | %s | %s | %s | %s (%d) | %s' % (
            r['arm'], r['mode'], r['decode_live'], r['bin_ms'], r['n'], fmt(r['cadence_ms']), fmt(r['queue_crit_ms']),
            fmt(r['queue_native_crit_ms']), fmt(r['exposed_candidate_ms']), r['exposed_candidate_known'],
            r['idle_before']['share']))
    for k, d in e['outside_queue_baseline_ms'].items():
        out.append('  outside-sample queue baseline (send -> enter) %s %s' % (k, fmt(d)))
    out.append('      ' + loss_line(doc))
    out += ['', 'ESTIMATED (old endpoint reconstruction [U-D, U]; never in the certified contrast)']
    for r in doc['ESTIMATED']['rows']:
        out.append('  %s %s %s D %s' % (r['rank'], r['arm'], r['cls'], fmt(r['D_ms'])))
    out.append('      ' + loss_line(doc))
    return '\n'.join(out) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=(__doc__ or '').split('\n')[0])
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', required=True, help='a fresh directory (must not exist)')
    parser.add_argument('--engine-log', default=None)
    args = parser.parse_args(argv)
    if os.path.lexists(args.out):
        refuse('output %s exists: give a fresh path' % args.out)
    if not os.path.isdir(args.root):
        refuse('record root %s is not a directory' % args.root)
    data = read(args.root, args.engine_log)
    items, estimated, meta = analyse(data)
    doc = report(data, items, estimated, meta)
    doc['root'] = os.path.abspath(args.root)
    os.makedirs(args.out)
    with open(os.path.join(args.out, 'busyfar.json'), 'x') as handle:
        json.dump(doc, handle, indent=1, sort_keys=True, default=str)
    body = text(doc)
    with open(os.path.join(args.out, 'busyfar.txt'), 'x') as handle:
        handle.write(body)
    sys.stdout.write(body)
    return 0


if __name__ == '__main__':
    sys.exit(main())
