#!/usr/bin/env python3
"""Stage-1 stock-handler anatomy report: what the worker main thread does inside each stock RPC. Reads, changes nothing.

    python3 tools/stage1_anatomy_report.py ROOT OUT [--no-native]

ROOT holds ``arm-*.json[.xz]`` and one engine log (``engine-current.log[.xz]`` or ``engine-stream.log[.xz]``) with the
``flash-s5 progress`` and ``flash-s5 handler`` lines (schema ``s5-handler-anatomy/1``, ``patches/s5_engine/handler_anatomy.py``).

Rows. One main row per stock RPC per rank (enter/exit, thread CPU, per seam inclusive/exclusive/CPU ns and count, device
forward/sample ns, stream idle before/after the forward launch, handle_output ns) and one output row per ordered reply
(async thread or main thread: get_output ns and CPU, response-MQ write ns). Seam nesting (exclusive time is the seam minus
the seams that ran inside it):

    execute_model RPC: pp_send_wait (worker top: previous step's send handles), pp_recv (metadata receive), execute
        (runner.execute_model) > prep_states/prep_inputs/prep_attn > prep_attn_meta, pp_recv_wait/pp_recv_post (V2: in
        execute; V1: inside forward), forward (launch; V1 _model_forward, V2 run_fullgraph/run_pw_graph, S5 split
        manager), logits (V1); pp_send (non-last stage).
    sample_tokens RPC: sample_tokens > sample > logits (V2), bookkeeping (V1), post_sampled (V2), pp_tok_send (last
        stage), pp_tok_recv (non-last stage).
    other = wall - sum(exclusive) - observer.

Joins. Each row's sequence is the engine's transport sequence, also on the engine's ``s`` row in the chunk flush that
submitted it: that gives the arm, the chunk, its history rows (chunk_start) and so the context band (below/beyond the
97,152-row arena, and the alpha-perf bands). Ranks 2 and 3 (stage 1) and 0 and 1 (stage 0) are joined by sequence: the
slower TP rank per step, and the seam that carries the difference.

Torn lines (harness/step_a_torn_lines.py's rule, descriptive here): a handler line whose body does not read, or reads
with another writer's header inside, is counted and its rows dropped; a flush short of parts or a gap in a rank's line
numbers is counted as lost. Nothing is repaired.

Decision readout (pre-registered, docs/e3-briefs/stage1-handler-anatomy-design.md): on the slower stage-1 rank, live
arms, median over chunks below the arena (and repeated beyond it), shares of the stock-handler wall.
"""
import argparse
import glob
import json
import lzma
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'harness')]
import e3_native41_progress_box as box  # noqa: E402
import step_a_report as sar  # noqa: E402
import step_a_torn_lines as torn_lines  # noqa: E402

HANDLER_LINE = 'flash-s5 handler '
SCHEMA = 's5-handler-anatomy/1'
SEAMS = ('pp_send_wait', 'pp_recv', 'pp_recv_wait', 'pp_recv_post', 'pp_send', 'execute', 'prep_states',
         'prep_inputs', 'prep_attn', 'prep_attn_meta', 'forward', 'logits', 'sample_tokens', 'sample',
         'bookkeeping', 'post_sampled', 'pp_tok_send', 'pp_tok_recv')
M_FIELDS = ('seq', 'kind', 'out', 'flags', 't_enter', 't_exit', 'wall_ns', 'cpu_ns', 'observer_ns',
            'dev_forward_ns', 'dev_sample_ns', 'idle_before_launch', 'idle_after_launch', 'handle_output_ns', 'seams')
A_FIELDS = ('seq', 'thread', 't_take', 'get_output_ns', 'get_output_cpu_ns', 'mq_write_ns', 't_emit', 'observer_ns')
KIND_NAMES = {'e': 'execute_model', 's': 'sample_tokens', 'o': 'other'}
FLAG_FAILED, FLAG_ABORTED, FLAG_INSTALL = 1, 2, 4
STAGES = {'1': (2, 3), '0': (0, 1)}
DOMINATES, SIGNIFICANT = 0.5, 0.25
CPU_BOUND = 0.8
CLOSE = 0.8   # host launch/sample wall within 80 % of the device time: "close to"

# Categories of the handler wall for the decision rule (exclusive seam ns; a seam is in one category only).
CATEGORIES = {
    'pp_receive': ('pp_recv', 'pp_recv_wait', 'pp_recv_post', 'pp_send_wait', 'pp_tok_recv'),
    'gpu_cpu_sync': ('bookkeeping',),          # + main-thread get_output (output rows, thread 1)
    'forward_launch': ('forward',),
    'logits_sampling': ('sample_tokens', 'sample', 'logits', 'post_sampled', 'pp_tok_send'),
    'prepare': ('execute', 'prep_states', 'prep_inputs', 'prep_attn', 'prep_attn_meta'),
    'pp_send': ('pp_send',),
}
CAVEAT = ("CUDA's default synchronisation can spin: a spin-wait counts as CPU time. The seam it sits in still names it "
          '(bookkeeping, get_output, a handle wait), so read the CPU share together with the seam, not alone.')


def dist(values):
    values = sorted(v for v in values if type(v) in (int, float))
    if not values:
        return None
    pick = lambda q: values[min(len(values) - 1, int(q * len(values)))]
    return dict(n=len(values), p50=round(statistics.median(values), 4), p90=round(pick(.9), 4),
                mean=round(statistics.fmean(values), 4))


def ms(ns):
    return None if ns is None else ns / 1e6


# --- reading -------------------------------------------------------------------

def is_torn(line):
    at = line.find(HANDLER_LINE)
    if at < 0:
        return False
    body = line[at + len(HANDLER_LINE):]
    if torn_lines.INNER_HEADER.search(body):
        return True
    try:
        json.loads(body)
    except ValueError:
        return True
    return False


def read_handler_lines(paths):
    """(stamp, body) per whole handler line, torn lines apart (counted, rows dropped), schema errors."""
    records, torn, errors = [], [], []
    for _, line in torn_lines._lines(paths):
        at = line.find(HANDLER_LINE)
        if at < 0:
            continue
        if is_torn(line):
            torn.append((box.docker_stamp(line), line))
            continue
        body = json.loads(line[at + len(HANDLER_LINE):])
        if type(body) is not dict or body.get('schema') != SCHEMA:
            errors.append('handler-line-schema')
            continue
        records.append((box.docker_stamp(line), body))
    return records, torn, errors


def assemble(records):
    """Flushes by (rank, line, base_ns); a flush short of parts keeps the parts that read and is counted."""
    groups = {}
    for stamp, body in records:
        groups.setdefault((body.get('rank'), body.get('line'), body.get('base_ns')), []).append((stamp, body))
    flushes, incomplete = [], []
    for key, parts in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), kv[0][1] or 0)):
        parts.sort(key=lambda p: p[1].get('part', -1))
        bodies = [b for _, b in parts]
        count = bodies[0].get('parts')
        whole = [b.get('part') for b in bodies] == list(range(count or 0))
        if not whole:
            incomplete.append(dict(rank=key[0], line=key[1], parts=[b.get('part') for b in bodies], of=count))
        head = next((b['head'] for b in bodies if b.get('part') == 0 and 'head' in b), None)
        base = key[2] or 0
        main, out = [], []
        for b in bodies:
            for item in (b.get('rows') or {}).get('m', []):
                r = dict(zip(M_FIELDS, item))
                for name in ('t_enter', 't_exit'):
                    r[name] = base + r[name] * 1000 if type(r[name]) is int else None
                r['seams'] = {SEAMS[c[0]]: dict(ns=c[1], excl_ns=c[2], cpu_ns=c[3], n=c[4]) for c in r['seams']}
                r['rank'] = key[0]
                main.append(r)
            for item in (b.get('rows') or {}).get('a', []):
                r = dict(zip(A_FIELDS, item))
                for name in ('t_take', 't_emit'):
                    r[name] = base + r[name] * 1000 if type(r[name]) is int else None
                r['rank'] = key[0]
                out.append(r)
        flushes.append(dict(rank=key[0], line=key[1], whole=whole, head=head, main=main, out=out,
                            stamp=min((s for s, _ in parts if s is not None), default=None)))
    return flushes, incomplete


def line_gaps(flushes):
    lines = {}
    for f in flushes:
        lines.setdefault(f['rank'], set()).add(f['line'])
    gaps = []
    for rank, numbers in lines.items():
        numbers = sorted(numbers)
        for a, b in zip(numbers, numbers[1:]):
            gaps += [dict(rank=rank, line=n) for n in range(a + 1, b)]
    return gaps


def seq_context(root, logs):
    """seq -> (arm label, chunk identity) from the engine chunk flushes' ``s`` rows, by the arms' wall windows."""
    records, errors = box.read_lines(logs)
    torn = torn_lines.read_torn(logs)
    records, errors, _ = torn_lines.whole_records(records, errors, torn['progress'])
    flushes, _ = box.assemble(records)
    engine = [f for f in flushes if f['side'] == 'engine']
    arms, where = {}, {}
    for path in sorted(glob.glob(os.path.join(root, 'arm-*.json*'))):
        arm = sar.load(path)
        name = arm.get('label') or os.path.basename(path).split('.')[0][4:]
        token, qsa, size = sar.arm_group(name, arm)
        arms[name] = dict(group=token, qsa_mode=qsa, group_size=size, cohort=arm.get('cohort'))
        start, end = arm.get('wall_start_s'), arm.get('wall_end_s')
        if type(start) not in (int, float) or type(end) not in (int, float):
            continue
        for f in engine:
            if f['stamp'] is None or not start <= f['stamp'] <= end + box.LOG_LAG_S:
                continue
            identity = f['identity']
            label = box.chunk_label(identity.get('decode_live'), 8)
            for s in f['rows'].get('s', []):
                where[s['seq']] = dict(arm=name, chunk_start=identity.get('chunk_start'),
                                       chunk_end=identity.get('chunk_end'), chunk_label=label, flush_line=f['line'])
    return arms, where, dict(progress_errors=errors, torn_progress_lines=len(torn['progress']))


# --- arithmetic ----------------------------------------------------------------

def excl(row, seam):
    cell = row['seams'].get(seam)
    return cell['excl_ns'] if cell else 0


def total(row):
    """The main thread's stock time for one RPC: the RPC (enter to return) plus its handle_output."""
    return (row['wall_ns'] or 0) + (row['handle_output_ns'] or 0)


def main_outputs(row, outs):
    return [o for o in outs.get((row['rank'], row['seq']), ()) if o['thread'] == 1]


def categories(row, outs):
    """ns per decision category for one main row; the values add up to total(row).

    Exclusive seam ns inside the RPC; handle_output (outside the RPC's wall) splits into main-thread get_output (the
    GPU->CPU copy wait, gpu_cpu_sync) and the rest (the reply write, 'output'). other = RPC wall - seams - observer.
    """
    got = {name: sum(excl(row, s) for s in seams) for name, seams in CATEGORIES.items()}
    main_get = sum(o['get_output_ns'] for o in main_outputs(row, outs) if type(o.get('get_output_ns')) is int)
    handle = row['handle_output_ns'] or 0
    got['gpu_cpu_sync'] += min(main_get, handle) if handle else main_get
    got['output'] = max(0, handle - main_get)
    accounted = sum(sum(excl(row, s) for s in seams) for seams in CATEGORIES.values())
    got['other'] = max(0, (row['wall_ns'] or 0) - accounted - (row['observer_ns'] or 0))
    got['observer'] = row['observer_ns'] or 0
    return got


def blocked_cpu(row, seams, outs=None):
    """(wall ns, thread CPU ns) over the given seams, plus main-thread get_output when outs are given."""
    ns = sum(row['seams'][s]['ns'] for s in seams if s in row['seams'])
    cpu = sum(row['seams'][s]['cpu_ns'] for s in seams if s in row['seams'])
    for o in main_outputs(row, outs or {}):
        if type(o.get('get_output_ns')) is int:
            ns += o['get_output_ns']
            cpu += o.get('get_output_cpu_ns') or 0
    return ns, cpu


def per_rank_kind(rows, outs):
    """Medians and p90 per rank and method kind (ms)."""
    table = {}
    for row in rows:
        key = '%s:%s' % (row['rank'], KIND_NAMES.get(row['kind'], row['kind']))
        table.setdefault(key, []).append(row)
    out = {}
    for key, group in sorted(table.items()):
        seams = sorted({s for r in group for s in r['seams']})
        rank = int(key.split(':')[0])
        replies = [o for r in group for o in outs.get((rank, r['seq']), ())]
        out[key] = dict(
            rpcs=len(group),
            # Rows carrying each seam / RPCs: a seam the path always takes (forward on execute_model) short of ~1.0
            # means a missed seam (VM run stage1-anatomy-c: forward on 10 of 1,106 execute rows).
            seam_coverage={s: round(sum(1 for r in group if s in r['seams']) / len(group), 4) for s in seams},
            wall_ms=dist([ms(r['wall_ns']) for r in group]), total_ms=dist([ms(total(r)) for r in group]), cpu_ms=dist([ms(r['cpu_ns']) for r in group]),
            cpu_share=dist([r['cpu_ns'] / r['wall_ns'] for r in group if r['wall_ns'] and r['cpu_ns'] is not None]),
            observer_ms=dist([ms(r['observer_ns']) for r in group]),
            handle_output_ms=dist([ms(r['handle_output_ns']) for r in group]),
            dev_forward_ms=dist([ms(r['dev_forward_ns']) for r in group]),
            dev_sample_ms=dist([ms(r['dev_sample_ns']) for r in group]),
            idle_before_launch=dist([r['idle_before_launch'] for r in group]),
            idle_after_launch=dist([r['idle_after_launch'] for r in group]),
            seams_ms={s: dict(ns=dist([ms(r['seams'][s]['ns']) for r in group if s in r['seams']]),
                              excl=dist([ms(r['seams'][s]['excl_ns']) for r in group if s in r['seams']]),
                              cpu=dist([ms(r['seams'][s]['cpu_ns']) for r in group if s in r['seams']]),
                              calls_per_rpc=dist([r['seams'][s]['n'] for r in group if s in r['seams']]))
                      for s in seams},
            other_ms=dist([ms(categories(r, outs)['other']) for r in group]),
            category_ms={c: dist([ms(categories(r, outs)[c]) for r in group]) for c in tuple(CATEGORIES) + (
                'output', 'other', 'observer')},
            reply=dict(get_output_ms=dist([ms(o['get_output_ns']) for o in replies]),
                       get_output_cpu_ms=dist([ms(o['get_output_cpu_ns']) for o in replies]),
                       mq_write_ms=dist([ms(o['mq_write_ns']) for o in replies]),
                       async_thread=sum(1 for o in replies if o['thread'] == 0),
                       main_thread=sum(1 for o in replies if o['thread'] == 1)))
    return out


def slower_rank(rows, stage):
    """Per sequence, the slower of the stage's two TP ranks (by total()) and the seam carrying the difference."""
    a, b = STAGES[stage]
    by = {}
    for row in rows:
        if row['rank'] in (a, b) and row['wall_ns'] is not None:
            by.setdefault(row['seq'], {})[row['rank']] = row
    pairs, carriers, counts, diffs = [], {}, {a: 0, b: 0}, {}
    for seq, both in by.items():
        if len(both) != 2:
            continue
        slow, fast = sorted(both.values(), key=lambda r: -total(r))
        counts[slow['rank']] += 1
        seams = set(slow['seams']) | set(fast['seams'])
        delta = {s: excl(slow, s) - excl(fast, s) for s in seams}
        delta['other'] = categories(slow, {})['other'] - categories(fast, {})['other']
        delta['handle_output'] = (slow['handle_output_ns'] or 0) - (fast['handle_output_ns'] or 0)
        for s, d in delta.items():
            diffs.setdefault(s, []).append(d / 1e6)
        carrier = max(delta, key=delta.get) if delta else None
        carriers[carrier] = carriers.get(carrier, 0) + 1
        pairs.append(dict(seq=seq, slow=slow, fast=fast, diff_ms=(total(slow) - total(fast)) / 1e6))
    return pairs, dict(pairs=len(pairs), slower_rank_counts={str(k): v for k, v in counts.items()},
                       wall_difference_ms=dist([p['diff_ms'] for p in pairs]),
                       carrier_counts=dict(sorted(carriers.items(), key=lambda kv: -kv[1])),
                       seam_difference_ms={s: dist(v) for s, v in sorted(diffs.items())})


def decision(pairs, outs, where):
    """The pre-registered readout over the slower stage-1 rank, live arms, per chunk then median across chunks."""
    per_chunk = {}
    for p in pairs:
        ctx = where.get(p['seq'])
        if ctx is None or ctx['chunk_label'] != 'live':
            continue
        key = (ctx['arm'], ctx['chunk_start'])
        per_chunk.setdefault(key, []).append(p['slow'])
    result = {}
    for band_name in ('below-arena', 'beyond-arena'):
        chunks = [rows for (arm, start), rows in per_chunk.items() if sar.band(start) == band_name]
        if not chunks:
            result[band_name] = dict(chunks=0, verdict='no live chunks with both stage-1 ranks in this band')
            continue
        shares, extra = {}, dict(cpu_share=[], blocked_share_pp_sync=[], idle_before=[], forward_vs_device=[],
                                 sample_vs_device=[], queued_after=[], handler_ms=[], forward_coverage=[])
        for rows in chunks:
            wall = sum(total(r) for r in rows) or 1
            cats = {}
            for r in rows:
                for k, v in categories(r, outs).items():
                    cats[k] = cats.get(k, 0) + v
            for k, v in cats.items():
                shares.setdefault(k, []).append(v / wall)
            extra['handler_ms'].append(wall / 1e6)
            executes = [r for r in rows if r['kind'] == 'e']
            extra['forward_coverage'].append(sum(1 for r in executes if 'forward' in r['seams']) / len(executes)
                                             if executes else None)
            extra['cpu_share'].append(sum(r['cpu_ns'] or 0 for r in rows) / (sum(r['wall_ns'] or 0 for r in rows) or 1))
            pp_seams = CATEGORIES['pp_receive'] + CATEGORIES['gpu_cpu_sync']
            ns, cpu = 0, 0
            for r in rows:
                a, b = blocked_cpu(r, pp_seams, outs)
                ns, cpu = ns + a, cpu + b
            extra['blocked_share_pp_sync'].append(1 - cpu / ns if ns else None)
            launches = [r for r in rows if r['idle_before_launch'] is not None]
            extra['idle_before'].append(sum(r['idle_before_launch'] for r in launches) / len(launches) if launches else None)
            queued = [r for r in rows if r['idle_after_launch'] is not None]
            extra['queued_after'].append(1 - sum(r['idle_after_launch'] for r in queued) / len(queued) if queued else None)
            fwd = [(r['seams']['forward']['ns'], r['dev_forward_ns']) for r in rows
                   if 'forward' in r['seams'] and r['dev_forward_ns']]
            extra['forward_vs_device'].append(sum(h for h, _ in fwd) / sum(d for _, d in fwd) if fwd else None)
            smp = [(r['seams']['sample_tokens']['ns'], r['dev_sample_ns']) for r in rows
                   if 'sample_tokens' in r['seams'] and r['dev_sample_ns']]
            extra['sample_vs_device'].append(sum(h for h, _ in smp) / sum(d for _, d in smp) if smp else None)
        med = {k: round(statistics.median(v), 4) for k, v in shares.items()}
        side = {k: (round(statistics.median([x for x in v if x is not None]), 4)
                    if any(x is not None for x in v) else None) for k, v in extra.items()}
        result[band_name] = readout(len(chunks), med, side)
    return result


def readout(chunks, shares, side):
    """Apply the pre-registered rule to the per-chunk medians.

    pp/sync     share = pp_receive + gpu_cpu_sync of the handler total; its blocked share (1 - CPU/wall over those
                seams) is reported beside it: a high CPU share there means a spin wait or host work (the caveat).
    CPU-bound   share = thread CPU / RPC wall, counted only when the decode stream was idle at the forward launch in
                at least half the launches (the GPU waits on the thread); 'holds' at >= 0.8.
    GPU-wait    share = forward_launch, when the host launch wall is >= 0.8 x the device forward time and work was still
                queued after the launch in at least half the launches; + logits_sampling, when the host sampling wall
                is >= 0.8 x the device sampling time.
    Each share: 'dominates' >= 0.5, 'significant' >= 0.25, else 'minor'. No outcome dominating: say so, name the
    largest category.
    """
    pp_sync = shares.get('pp_receive', 0) + shares.get('gpu_cpu_sync', 0)
    gpu_wait = 0.0
    if (side['forward_vs_device'] or 0) >= CLOSE and (side['queued_after'] or 0) >= 0.5:
        gpu_wait += shares.get('forward_launch', 0)
    if (side['sample_vs_device'] or 0) >= CLOSE:
        gpu_wait += shares.get('logits_sampling', 0)
    idle_at_launch = (side['idle_before'] or 0) >= 0.5
    cpu_value = (side['cpu_share'] or 0) if idle_at_launch else 0.0

    def cls(value):
        return 'dominates' if value >= DOMINATES else 'significant' if value >= SIGNIFICANT else 'minor'
    outcomes = {
        'pp_receive_or_gpu_cpu_sync': dict(
            share=round(pp_sync, 4), blocked_share=side['blocked_share_pp_sync'], class_=cls(pp_sync),
            lever='non-blocking last-stage decode output (the thread returns while the GPU works)'),
        'cpu_bound': dict(
            share=round(cpu_value, 4), cpu_share=side['cpu_share'], stream_idle_at_launch=side['idle_before'],
            holds=idle_at_launch and (side['cpu_share'] or 0) >= CPU_BOUND, class_=cls(cpu_value),
            lever='qualified yield boundaries / bounded launch batches for native commands'),
        'waiting_on_decode_gpu': dict(
            share=round(gpu_wait, 4), forward_host_vs_device=side['forward_vs_device'],
            queued_after_launch=side['queued_after'], sample_host_vs_device=side['sample_vs_device'],
            class_=cls(gpu_wait), lever='FLASH_S5_DECODE_SMS sweep (today 44 of 82 SMs)'),
    }
    for row in outcomes.values():
        row['class'] = row.pop('class_')
    if (side['blocked_share_pp_sync'] or 0) < 0.5 and pp_sync >= SIGNIFICANT:
        outcomes['pp_receive_or_gpu_cpu_sync']['note'] = ('these seams are mostly on-CPU, not blocked: a spin wait or '
                                                          'host work, see the caveat')
    winners = [k for k, v in outcomes.items() if v['class'] == 'dominates']
    largest = max(shares, key=lambda k: shares[k]) if shares else None
    if winners:
        verdict = 'dominates: ' + ', '.join(winners)
    elif largest:
        verdict = 'no outcome reaches 50 %% of the handler total; the largest category is %s (%.0f %%)' % (
            largest, 100 * shares[largest])
    else:
        verdict = 'no rows'
    notes = []
    if side['forward_coverage'] is not None and side['forward_coverage'] < 0.95:
        notes.append('forward seam on %.0f %% of execute_model rows: a missed seam, the launch is in prepare; '
                     'do not read the rule' % (100 * side['forward_coverage']))
    if shares.get('other', 0) >= SIGNIFICANT:
        notes.append("'other' is %.0f %% of the handler total: find its seam before reading the rule"
                     % (100 * shares['other']))
    return dict(chunks=chunks, handler_ms_per_chunk=side['handler_ms'], forward_coverage=side['forward_coverage'],
                notes=notes, category_shares=shares, outcomes=outcomes,
                significant=[k for k, v in outcomes.items() if v['class'] == 'significant'], verdict=verdict,
                caveat=CAVEAT)


# --- the report ----------------------------------------------------------------

def analyse(root, native=True):
    logs = sar.engine_logs(root)
    if not logs:
        raise SystemExit('no engine log in ' + root)
    records, torn, errors = read_handler_lines(logs)
    flushes, incomplete = assemble(records)
    gaps = line_gaps(flushes)
    arms, where, progress = seq_context(root, logs)
    rows = [r for f in flushes for r in f['main'] if not r['flags'] & FLAG_INSTALL]
    outs = {}
    for f in flushes:
        for o in f['out']:
            outs.setdefault((o['rank'], o['seq']), []).append(o)
    heads = {}
    for f in flushes:
        if f['head']:
            heads.setdefault(str(f['rank']), []).append(f['head'])
    header = {}
    for rank, items in sorted(heads.items()):
        last = items[-1]
        header[rank] = dict(identity=last['identity'], calibration=last['calibration'], counters=last['counters'],
                            installed=last['installed'], missing=last['missing'], flushes=len(items))
    groups = {}
    for r in rows:
        ctx = where.get(r['seq'])
        if ctx is None:
            key = 'unassigned'
        else:
            arm = arms[ctx['arm']]
            key = '%s-%s-%s' % (arm['group'], arm['cohort'], sar.band(ctx['chunk_start']))
        groups.setdefault(key, []).append(r)
    report = dict(schema='stage1-anatomy-report/1', root=os.path.abspath(root), logs=logs,
                  handler_lines=len(records), torn_handler_lines=len(torn), incomplete_flushes=incomplete,
                  line_gaps=gaps, schema_errors=errors, progress=progress, header=header,
                  rows=dict(main=len(rows), output=sum(len(v) for v in outs.values()),
                            failed=sum(1 for r in rows if r['flags'] & FLAG_FAILED),
                            aborted=sum(1 for r in rows if r['flags'] & FLAG_ABORTED),
                            unassigned=len(groups.get('unassigned', ()))),
                  groups={})
    for key, group in sorted(groups.items()):
        entry = dict(per_rank_kind=per_rank_kind(group, outs), bands=bands(group, where))
        entry['slower_tp_rank'] = {name: slower_rank(group, name)[1] for name in ('1', '0')}
        report['groups'][key] = entry
    # The decision: per QSA group, live arms, both bands (the rule reads each band on its own).
    by_token = {}
    for r in rows:
        ctx = where.get(r['seq'])
        if ctx is not None and arms[ctx['arm']]['cohort'] == 'live':
            by_token.setdefault(arms[ctx['arm']]['group'], []).append(r)
    report['decision'] = {token: decision(slower_rank(group, '1')[0], outs, where)
                          for token, group in sorted(by_token.items())}
    if native:
        summary = sar.summarise(sar.analyse(root, logs))
        for key, entry in report['groups'].items():
            if key in summary:
                entry['native_spans'] = {k: summary[key].get(k) for k in ('chunks', 'chunk_interval_s', 'main_thread_share',
                                                                            'handler_ms', 'decode_delay_ms')}
    return report


def bands(rows, where):
    """Per alpha-perf band of the chunk's history rows: the stage-1 stock handler wall per RPC kind."""
    out = {}
    for r in rows:
        ctx = where.get(r['seq'])
        if ctx is None or ctx['chunk_start'] is None:
            continue
        start = ctx['chunk_start']
        name = box.band_of(start) if start < box.BANDS[-1][1] else 'beyond-%d' % box.BANDS[-1][1]
        out.setdefault(name, {}).setdefault('%s:%s' % (r['rank'], KIND_NAMES.get(r['kind'], r['kind'])), []).append(
            ms(total(r)))
    return {b: {k: dist(v) for k, v in sorted(rows_.items())} for b, rows_ in sorted(out.items())}


def text(report):
    lines = ['stage-1 anatomy: %d handler lines (%d torn, %d incomplete flushes, %d line gaps), %d main rows, %d output rows'
             % (report['handler_lines'], report['torn_handler_lines'], len(report['incomplete_flushes']),
                len(report['line_gaps']), report['rows']['main'], report['rows']['output'])]
    for key, entry in report['decision'].items():
        for band_name, d in entry.items():
            lines.append('%s %s: %s' % (key, band_name, d.get('verdict')))
            for name, share in sorted((d.get('category_shares') or {}).items(), key=lambda kv: -kv[1]):
                lines.append('    %-18s %5.1f %%' % (name, 100 * share))
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('root')
    parser.add_argument('out', help='fresh output JSON')
    parser.add_argument('--no-native', action='store_true', help='skip the step_a_report native spans beside each group')
    args = parser.parse_args(argv)
    if os.path.exists(args.out):
        parser.error('output exists: ' + args.out)
    report = analyse(args.root, native=not args.no_native)
    with open(args.out, 'x') as handle:
        json.dump(report, handle, indent=1, sort_keys=True, default=str)
    print(text(report))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
