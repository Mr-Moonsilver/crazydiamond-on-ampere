"""q18a-stage1-step grader-side reads: never verdict inputs.

Over the grader's own inputs: the flushes from e3_native41_progress_box.read_lines and assemble, the sequence -> transaction owners from the whole stage-overlap lines
(e3_alpha_next_c1_box.overlap_owners), the counted chunks from the progress receipts of the grade. Pure functions, no clock, no box. No forecast here: the ranges live in DESIGN only.

fifo_arm      per counted chunk, per stage: on each rank of the stage the sum over the stage's commands of [t_sent, that rank's t_enter] (the grader's rows, chunk_breakdown's
              FIFO term per rank), and the slower rank's sum. Median, p10 and p90 in seconds, per chunk; the median per command kind.
pending_arm   per counted chunk: the replies that came back 'pending' per stage and the polls. Native17/18's read: a LAYER arriving before its selection copy landed.
handler_read  the ``flash-s5 handler`` lines inside an arm's window, per rank: execute_model and sample_tokens counts per chunk, wall and CPU per RPC, thread seconds per chunk
              and the thread share of the window.
step_rows     the same lines, per rank: full against empty execute_model (a row with the ``forward`` seam is full; empty steps keep the PP cadence), sample_tokens, pair ms, the pp_recv
              block (seam wall minus its CPU: the wait moved by L1), attention metadata, device forward (T(8) under L2) and the thread share.
sched_read    the ``flash-sched stats`` lines of one arm: decoders scheduled on even and on odd steps across the arm (L2 moves every decoder to even) and the last cohort counts.
"""
import ast
import json
import re
import statistics

SCHED_MARK = 'flash-sched stats '
HANDLER_MARK = 'flash-s5 handler '
STAGE_RANKS = {0: (0, 1), 1: (2, 3)}
HANDLER_KINDS = {'e': 'execute_model', 's': 'sample_tokens'}
GATE_FALSE = dict(is_gate=False)


def _int(value):
    return type(value) is int


def _median(values):
    return statistics.median(values) if values else None


def _pct(values, p):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(p * len(ordered)))] if ordered else None


def _round(value, digits=4):
    return None if value is None else round(value, digits)


def index_rows(flushes, owner):
    """(txn -> engine 'n' rows, (seq, rank) -> worker 'r' row). A row keeps the flush's transaction when no overlap line names it."""
    bytxn, workers = {}, {}
    for flush in flushes:
        rows = flush.get('rows') or {}
        if flush.get('side') == 'worker':
            for row in rows.get('r') or ():
                workers[(row['seq'], flush.get('rank'))] = row
        elif flush.get('side') == 'engine':
            own = (flush.get('identity') or {}).get('txn', flush.get('txn'))
            for row in rows.get('n') or ():
                bytxn.setdefault(owner.get(row['seq'], own), []).append(row)
    return bytxn, workers


def rank_rows(flushes):
    """rank -> that rank's 'r' rows (one per seq, the last seen), and the rows the writers said they dropped, per rank."""
    rows, dropped = {}, {}
    for flush in flushes:
        if flush.get('side') != 'worker':
            continue
        rank = flush.get('rank')
        cell = rows.setdefault(rank, {})
        for row in (flush.get('rows') or {}).get('r') or ():
            cell[row['seq']] = row
        if _int(flush.get('dropped')):
            dropped[rank] = dropped.get(rank, 0) + flush['dropped']
    return {rank: list(cell.values()) for rank, cell in rows.items()}, dropped


def engine_rows(flushes):
    """seq -> the engine's 'n' row, over every engine flush."""
    return {row['seq']: row for flush in flushes if flush.get('side') == 'engine' for row in (flush.get('rows') or {}).get('n') or ()}


def counted_txns(progress_row):
    """The arm's counted chunks in transaction order."""
    row = progress_row or {}
    return sorted(c['txn'] for c in row.get('chunks') or () if c.get('label') == row.get('counted_label') and not c.get('dropped_out') and _int(c.get('txn')))


def fifo_arm(txns, bytxn, workers, stage):
    """The stage's FIFO wait per counted chunk (module docstring)."""
    per_chunk, kinds, commands = [], {}, []
    for txn in txns:
        best = None
        for rank in STAGE_RANKS[stage]:
            total, count, by_kind = 0, 0, {}
            for row in bytxn.get(txn) or ():
                if row.get('stage') != stage or not _int(row.get('t_sent')):
                    continue
                worker = workers.get((row['seq'], rank))
                if not worker or not _int(worker.get('t_enter')):
                    continue
                wait = max(0, worker['t_enter'] - row['t_sent'])
                total += wait
                count += 1
                key = row.get('segment') or row.get('kind') or '?'
                by_kind[key] = by_kind.get(key, 0) + wait
            if count and (best is None or total > best[0]):
                best = (total, by_kind, count, rank)
        if best is not None:
            per_chunk.append(best[0] / 1e9)
            commands.append(best[2])
            for key, value in best[1].items():
                kinds.setdefault(key, []).append(value / 1e9)
    return dict(stage=stage, chunks=len(per_chunk), median_s=_round(_median(per_chunk)), p10_s=_round(_pct(per_chunk, .1)), p90_s=_round(_pct(per_chunk, .9)),
                commands_per_chunk=_median(commands), by_kind_median_s={k: _round(_median(v)) for k, v in sorted(kinds.items())}, **GATE_FALSE)


def pending_arm(txns, bytxn):
    """Pending replies and polls per counted chunk (medians and means)."""
    pending = {0: [], 1: []}
    polls = []
    for txn in txns:
        rows = bytxn.get(txn) or ()
        for stage in pending:
            pending[stage].append(sum(1 for r in rows if r.get('stage') == stage and r.get('outcome') == 'pending'))
        polls.append(sum(1 for r in rows if r.get('kind') == 'poll'))

    def cell(values):
        return dict(median=_median(values), mean=_round(sum(values) / len(values)) if values else None, total=sum(values))
    return dict(chunks=len(txns), pending_stage0=cell(pending[0]), pending_stage1=cell(pending[1]), polls=cell(polls), **GATE_FALSE)


def parse_handler(text, fields):
    """One handler line: (rank, base_ns, list of 'm' rows as dicts), and the fields it declares; None when it is not a whole line. ``fields``: the last declared."""
    at = text.find(HANDLER_MARK)
    if at < 0:
        return None
    try:
        body, _ = json.JSONDecoder().raw_decode(text[at + len(HANDLER_MARK):])
    except ValueError:
        return None
    if type(body) is not dict:
        return None
    head = body.get('head')
    declared = (head.get('fields') or {}).get('m') if type(head) is dict else None
    names = declared or fields
    rows = (body.get('rows') or {}).get('m') or ()
    if rows and not names:
        return None
    if not _int(body.get('base_ns')):
        return None
    return body.get('rank'), body['base_ns'], [dict(zip(names, row)) for row in rows], names


def handler_read(lines, windows, chunks):
    """``lines``: the raw texts in log order. ``windows``: arm -> (start_ns, end_ns); ``chunks``: arm -> the chunk count of that window.
    arm -> rank -> kind -> n, per chunk, wall and CPU per RPC in ms, thread seconds per chunk, share of the window; and the malformed line count."""
    fields, malformed = None, 0
    rows = {}
    for text in lines:
        parsed = parse_handler(text, fields)
        if parsed is None:
            malformed += 1
            continue
        rank, base, items, fields = parsed
        for row in items:
            kind = HANDLER_KINDS.get(row.get('kind'))
            if kind and _int(row.get('t_enter')) and _int(row.get('wall_ns')):
                rows.setdefault((rank, kind), []).append((base + row['t_enter'] * 1000, row['wall_ns'], row.get('cpu_ns') if _int(row.get('cpu_ns')) else 0))
    arms = {}
    for name, (start, end) in windows.items():
        count = chunks.get(name) or 0
        span = (end - start) / 1e9
        cell = {}
        for (rank, kind), items in rows.items():
            inside = [i for i in items if start <= i[0] <= end]
            if not inside:
                continue
            wall = [i[1] for i in inside]
            cell.setdefault(str(rank), {})[kind] = dict(
                n=len(inside), per_chunk=_round(len(inside) / count, 2) if count else None, wall_ms_median=_round(_median(wall) / 1e6, 3),
                wall_ms_mean=_round(sum(wall) / len(wall) / 1e6, 3), cpu_ms_mean=_round(sum(i[2] for i in inside) / len(inside) / 1e6, 3),
                thread_s_per_chunk=_round(sum(wall) / 1e9 / count) if count else None, thread_share=_round(sum(wall) / 1e9 / span) if span > 0 else None)
        arms[name] = cell
    return dict(arms=arms, malformed=malformed, **GATE_FALSE)


def _seam(row, names, name, field):
    """A seam's field (SEAM_FIELDS: ns, excl_ns, cpu_ns, n) of one handler row, 0 when it did not run."""
    for cell in row.get('seams') or ():
        if names[cell[0]] == name:
            return cell[1 + ('ns', 'excl_ns', 'cpu_ns', 'n').index(field)]
    return 0


def step_rows(lines, windows):
    """``lines``: the raw handler texts in log order. arm -> rank -> the per-step cells (module docstring), milliseconds per RPC."""
    fields, seams, rows = None, None, {}
    for text in lines:
        at = text.find(HANDLER_MARK)
        try:
            body, _ = json.JSONDecoder().raw_decode(text[at + len(HANDLER_MARK):]) if at >= 0 else (None, 0)
        except ValueError:
            continue
        if type(body) is not dict or not _int(body.get('base_ns')):
            continue
        head = body.get('head')
        if type(head) is dict:
            fields = (head.get('fields') or {}).get('m') or fields
            seams = head.get('seams') or seams
        if not fields or not seams:
            continue
        for items in (body.get('rows') or {}).get('m') or ():
            row = dict(zip(fields, items))
            if _int(row.get('t_enter')) and _int(row.get('wall_ns')):
                rows.setdefault(body.get('rank'), []).append((body['base_ns'] + row['t_enter'] * 1000, row))

    def mean_ms(values):
        return _round(sum(values) / len(values) / 1e6, 3) if values else None
    out = {}
    for name, (start, end) in windows.items():
        span = (end - start) / 1e9
        cell = {}
        for rank, items in rows.items():
            inside = [r for at, r in items if start <= at <= end]
            kinds = {k: [r for r in inside if r.get('kind') == k] for k in ('e', 's')}
            full = [r for r in kinds['e'] if _seam(r, seams, 'forward', 'n')]
            empty = [r for r in kinds['e'] if not _seam(r, seams, 'forward', 'n')]
            if not inside:
                continue
            block = [_seam(r, seams, 'pp_recv', 'ns') - _seam(r, seams, 'pp_recv', 'cpu_ns') for r in kinds['e']]
            device = [r['dev_forward_ns'] for r in full if _int(r.get('dev_forward_ns'))]
            full_wall, sample = mean_ms([r['wall_ns'] for r in full]), mean_ms([r['wall_ns'] for r in kinds['s']])
            cell[str(rank)] = dict(
                full_n=len(full), empty_n=len(empty), sample_n=len(kinds['s']), full_wall_ms=full_wall, full_cpu_ms=mean_ms([r.get('cpu_ns') or 0 for r in full]),
                empty_wall_ms=mean_ms([r['wall_ns'] for r in empty]), sample_wall_ms=sample, pair_ms=_round(full_wall + sample, 3) if full_wall is not None and sample is not None else None,
                pp_recv_block_ms=mean_ms(block), pp_recv_block_p50_ms=_round(_pct(block, .5) / 1e6, 3) if block else None, pp_recv_block_p90_ms=_round(_pct(block, .9) / 1e6, 3) if block else None,
                prep_attn_meta_ms=mean_ms([_seam(r, seams, 'prep_attn_meta', 'ns') for r in full]), dev_forward_p50_ms=_round(_median(device) / 1e6, 3) if device else None,
                thread_share=_round(sum(r['wall_ns'] for r in inside) / 1e9 / span) if span > 0 else None)
        out[name] = cell
    return out


def slower(cell, field):
    """The larger of ranks 2 and 3, or None."""
    values = [(cell.get(rank) or {}).get(field) for rank in ('2', '3')]
    values = [v for v in values if v is not None]
    return max(values) if values else None


_SCHED_STATS = re.compile(r'flash-sched stats step (\d+): (\{[^}]*\}) cohorts even=(\d+) odd=(\d+) none=(\d+)')


def sched_read(lines):
    """One arm's lines: first and last step, the decoders scheduled on even and odd steps between them, the last cohort counts; a line that does not parse is counted and kept out."""
    rows, malformed = [], 0
    for text in lines:
        found = _SCHED_STATS.search(text)
        try:
            stats = ast.literal_eval(found.group(2)) if found else None
        except (ValueError, SyntaxError):
            stats = None
        if type(stats) is not dict or not _int(stats.get('dec_even')) or not _int(stats.get('dec_odd')):
            malformed += 1
            continue
        rows.append((int(found.group(1)), stats, tuple(int(found.group(k)) for k in (3, 4, 5))))
    if len(rows) < 2:
        return dict(lines=len(rows), malformed=malformed, dec_even=None, dec_odd=None, cohorts_last=None, **GATE_FALSE)
    rows.sort(key=lambda r: r[0])
    return dict(lines=len(rows), malformed=malformed, first_step=rows[0][0], last_step=rows[-1][0], dec_even=rows[-1][1]['dec_even'] - rows[0][1]['dec_even'],
                dec_odd=rows[-1][1]['dec_odd'] - rows[0][1]['dec_odd'], cohorts_last='even=%d odd=%d none=%d' % rows[-1][2], **GATE_FALSE)
