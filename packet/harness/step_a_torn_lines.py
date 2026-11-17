#!/usr/bin/env python3
"""Torn log lines, the rule registered before any q18a-step-a or q18a-step-b run.

This is the rule of the C0 grader (`harness/e3_alpha_c0_box.py`, claude/c0-c1), so that the queues agree.

The docker log can splice one writer's bytes into another's line. q18a-alpha-perf tore 2 of its 22,063 progress
lines, and each one failed a whole arm through `line-parts-incomplete` and `worker-rows-missing`.

The rule:
- A progress line is torn when its body no longer reads, or when it reads with another writer's header inside: a
  splice inside a JSON string still parses and may have changed a value. A stray header with an intact body after
  it is not a tear (525 such lines in q18a-alpha-perf). A memory line that no longer reads is torn too.
- A lost flush is found from a torn line, a flush short of parts, or a gap in its writer's line numbers.
- Each is counted per arm. The chunks a lost flush touches drop out of the breakdown and stay listed.
- An arm is INCONCLUSIVE above LOST_RECORD_LIMIT lost records (lost flushes, unreadable torn lines and torn memory
  lines) or DROPPED_CHUNK_LIMIT dropped chunks.
- A missing row no lost flush explains still fails the arm, as does every other progress problem.
- A contract error or `disagree` inside a torn line still counts, as ENGINE. Lines are never repaired.

Client-side speed and decode numbers do not read these rows, so the rule never touches them.

Why these limits (q18a-alpha-perf: 2 lost of 7,878 flushes): an arm of about 330 to 640 flushes expects 0.08 to 0.16
lost flushes. Limit 0 would turn many runs INC for a log artefact, and a share of flushes means limit 0 at this size.
The two real losses touched 1 and 2 chunks, so 2 chunks is what one lost flush may cost.
"""
import json
import re

import e3_native41_progress_box as progress

LOST_RECORD_LIMIT = 1
DROPPED_CHUNK_LIMIT = 2
INNER_HEADER = re.compile(r'\((?:Worker_\w+|EngineCore\w*) pid=\d+\) ?')
INCOMPLETE = re.compile(r'line-parts-incomplete:(\w+):(\w+):(\d+)')
MEMORY_MARK = 'flash-s5 memory '
MEMORY_LINE = re.compile(r'flash-s5 memory point=(\w+) rank=(\d+) txn_seq=(\d+) chunk=(\d+)-(\d+) allocated=(-?\d+) '
                         r'reserved=(-?\d+) inactive_split=(-?\d+) alloc_retries=(-?\d+) ooms=(-?\d+) free=(-?\d+)')


def is_torn_progress(line):
    """The body no longer reads, or reads with another writer's header in it."""
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


def is_torn_memory(line):
    return MEMORY_MARK in line and not MEMORY_LINE.search(line)


def signature_of(line, signatures):
    """A contract error in a followed line. A whole progress line holds none; a torn one can hold another writer's bytes."""
    marked = progress.PROGRESS_LINE in line
    if marked and not is_torn_progress(line):
        return None
    whole = INNER_HEADER.sub('', line)
    for name, text in signatures:
        if text in whole:
            return name
    # A plain disagreement scan sees neither a marked line nor a word a header split.
    if 'disagree' in whole and (marked or 'disagree' not in line):
        return 'tp-disagree-in-a-torn-line'
    return None


def read_torn(paths):
    """The followed lines that no longer read, by what they were: (stamp, text) lists."""
    rows = dict(progress=[], memory=[])
    for path, line in _lines(paths):
        if is_torn_progress(line):
            rows['progress'].append((progress.docker_stamp(line), line))
        elif is_torn_memory(line):
            rows['memory'].append((progress.docker_stamp(line), line))
    return rows


def _lines(paths):
    import lzma
    from pathlib import Path
    for path in paths:
        path = Path(path)
        if path.is_file():
            with (lzma.open if path.suffix == '.xz' else open)(path, 'rt', errors='replace') as handle:
                for line in handle:
                    yield path, line


def whole_records(records, errors, torn):
    """The join's input without the torn lines. Returns (records, other errors, torn lines that failed to parse)."""
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


def torn_identity(line):
    """(side, rank, line) from the text before the rows, which a tear may have left readable; else None."""
    head, found, _ = line[line.find(progress.PROGRESS_LINE):].partition('"rows":')
    number = re.search(r'"line":(\d+)', head) if found else None
    if not number:
        return None
    rank = re.search(r'"rank":(\d+)', head)
    return ('worker', int(rank.group(1)), int(number.group(1))) if rank else ('engine', None, int(number.group(1)))


def flush_ledger(records, torn):
    """Every flush the log lost, from a torn line, a flush short of parts, or a gap in a writer's lines.

    A torn line whose identity does not read, or reads as a flush that is present, counts as its own lost record:
    that can count one loss twice and only ever errs toward INC.
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
        # The neighbours' whole seq ranges: rows finish out of seq order across a flush boundary. Only a seq the
        # rank does not hold can be blamed on this loss.
        lines = present.get(('worker', row['rank']), {})
        below = [n for n in lines if n < row['line'] and lines[n]['rows'].get('r')]
        above = [n for n in lines if n > row['line'] and lines[n]['rows'].get('r')]
        row['seq_low'] = min(r['seq'] for r in lines[max(below)]['rows']['r']) if below else None
        row['seq_high'] = max(r['seq'] for r in lines[min(above)]['rows']['r']) if above else None
    return dict(lost=[lost[k] for k in sorted(lost, key=str)], unidentified_torn_stamps=unidentified, rank_seqs=seqs)


def touches(loss, seqs, rank_seqs):
    """Whether this lost flush holds one of these seqs: only a seq its rank holds nowhere else can be blamed on it."""
    if loss['side'] != 'worker':
        return False
    low, high = loss.get('seq_low'), loss.get('seq_high')
    held = rank_seqs.get(loss['rank'], ())
    return any(s not in held and (low is None or s >= low) and (high is None or s <= high) for s in seqs)


def chunk_seqs(flush):
    return [row['seq'] for tag in ('n', 's') for row in flush['rows'].get(tag, [])]


def _medians(rows, *, band):
    median = progress.median
    out = dict(chunks=len(rows), wall_s=median([c['wall_s'] for c in rows]),
               open_interval_s=median([c.get('open_interval_s') for c in rows]),
               gap_before_s=median([c.get('gap_before_s') for c in rows]),
               residual_s=median([c['residual_s'] for c in rows]),
               components_s={k: median([c['components_s'][k] for c in rows]) for k in progress.COMPONENTS})
    if band:
        out.update(stock_rpcs=median([c['stock_rpcs'] for c in rows]), wait_s=median([c['wait_s'] for c in rows]),
                   wait_sends=median([c['wait_sends'] for c in rows]), commands=median([c['commands'] for c in rows]))
    return out


def recount(row):
    """The join's per-band and whole-arm medians over the chunks still counted (a test pins the equality)."""
    counted = [c for c in row['chunks'] if c['label'] == row['counted_label'] and not c.get('dropped_out')]
    row['counted_chunks'] = len(counted)
    row['bands'] = {band: _medians([c for c in counted if c['band'] == band], band=True)
                    for band in sorted({c['band'] for c in counted})}
    row['all_counted'] = _medians(counted, band=False)


def _forgive(row, reason):
    row['reasons'] = [x for x in row['reasons'] if x != reason]


def in_span(stamp, arm):
    start, end = arm.get('wall_start_s'), arm.get('wall_end_s')
    if type(start) not in (int, float) or type(end) not in (int, float) or stamp is None:
        return False
    return start <= stamp <= end + progress.LOG_LAG_S


def settle_arm(arm, row, ledger, torn, last_line, seqs_by_line):
    """Count this arm's torn lines and lost records, drop the chunks they touch, and clear only the reasons they explain."""
    chunks = row.get('chunks') or []
    in_arm = lambda rows: [t for s, t in rows if in_span(s, arm)]
    torn_progress, torn_memory = in_arm(torn['progress']), in_arm(torn['memory'])
    expected = (arm.get('progress_delta') or {}).get('chunks')
    shortfall = expected - len(chunks) if type(expected) is int and chunks else 0
    mine, engine_lost = [], 0
    for loss in ledger['lost']:
        if loss['side'] == 'engine':
            if engine_lost < shortfall and any(in_span(s, arm) for s in loss['stamps']):
                engine_lost += 1
                mine.append(loss)
            continue
        touched = False
        for chunk in chunks:
            if touches(loss, seqs_by_line.get(chunk['line'], ()), ledger['rank_seqs']):
                touched = True
                chunk.setdefault('dropped_out', []).append('worker-flush-lost:%s:%s' % (loss['rank'], loss['line']))
        if touched or (loss['exact'] and any(in_span(s, arm) for s in loss['stamps'])):
            mine.append(loss)
    unidentified = sum(1 for s in ledger['unidentified_torn_stamps'] if in_span(s, arm))
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
    records = len(mine) + unidentified + len(torn_memory)
    chunk_cost = len(starts)
    if records > LOST_RECORD_LIMIT:
        row['reasons'].append('lost-records-above-the-declared-limit:%d>%d' % (records, LOST_RECORD_LIMIT))
    if chunk_cost > DROPPED_CHUNK_LIMIT:
        row['reasons'].append('dropped-chunks-above-the-declared-limit:%d>%d' % (chunk_cost, DROPPED_CHUNK_LIMIT))
    row['tears'] = dict(torn_progress_lines=len(torn_progress), torn_memory_lines=len(torn_memory),
                        lost_flushes=[{k: x[k] for k in ('side', 'rank', 'line', 'evidence')} for x in mine],
                        unidentified_torn_lines=unidentified, lost_records=records, dropped_chunk_starts=starts,
                        within_limits=records <= LOST_RECORD_LIMIT and chunk_cost <= DROPPED_CHUNK_LIMIT)
    return row['tears']


def receipts_for(arms, logs, streams_count=8):
    """The grader's one call: (records, progress receipts with the rule applied, problems for the missing list)."""
    records, errors = progress.read_lines(logs)
    torn = read_torn(logs)
    unparsable = errors.count('unparsable-progress-line')
    records, errors, failed = whole_records(records, errors, torn['progress'])
    extra = [] if unparsable == failed else ['progress:torn-lines-differ-from-the-unparsable-lines']
    ledger = flush_ledger(records, torn['progress'])
    receipts = progress.progress_receipts(arms, records, errors, streams_count)
    receipts['problems'] = [x for x in receipts['problems'] if not INCOMPLETE.fullmatch(x)]
    flushes, _ = progress.assemble(records)
    engine = [f for f in flushes if f['side'] == 'engine']
    last_line = max((f['line'] for f in engine), default=None)
    seqs_by_line = {f['line']: chunk_seqs(f) for f in engine}
    claimed = set()
    for arm in arms:
        tears = settle_arm(arm, receipts['arms'][arm['label']], ledger, torn, last_line, seqs_by_line)
        claimed.update((x['side'], x['rank'], x['line']) for x in tears['lost_flushes'])
    receipts['tears'] = dict(lost_record_limit=LOST_RECORD_LIMIT, dropped_chunk_limit=DROPPED_CHUNK_LIMIT,
                             torn_progress_lines=len(torn['progress']), torn_memory_lines=len(torn['memory']),
                             lost_flushes=len(ledger['lost']), unidentified_torn_lines=len(ledger['unidentified_torn_stamps']),
                             lost_flushes_outside_every_arm=[{k: x[k] for k in ('side', 'rank', 'line', 'evidence')}
                                                             for x in ledger['lost']
                                                             if (x['side'], x['rank'], x['line']) not in claimed])
    return records, receipts, extra
