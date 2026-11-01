#!/usr/bin/env python3
"""Torn progress lines, the rule pre-registered before any q18a-step-a run.

Why it exists: other writers' log lines can land inside a long `flash-s5 progress` line. q18a-alpha-perf
tore 2 lines of 22,063 (2 of its 7,878 flushes), and each one failed its whole arm through
`line-parts-incomplete` and `worker-rows-missing`.

The rule:
- A torn progress line or an incomplete flush is a lost flush. Each is counted and reported per arm.
- A chunk whose missing rows all lie in a lost flush drops out of the arm's breakdown. A missing row no lost
  flush explains still fails the arm, as does every other progress problem.
- An arm is INCONCLUSIVE only when it loses more than LOST_FLUSH_LIMIT flushes.

Why 2: at the rate seen (1 flush in 3,939), an arm of 330 to 650 flushes loses 3 or more with probability below
0.1 %, while 2 in one arm happens about 1 % of the time, too often to fail a run on. More than 2 therefore points
at something other than chance. A lost worker flush takes at most 2 chunks out (its own and the one whose release
reply rode on it), so 2 lost flushes leave at least 62 of 66 chunks and move a median by at most 2 ranks. A
percentage limit does not work here: 0.1 % of an arm's flushes is less than one flush.

Client-side speed and decode numbers do not read these rows, so this rule never touches them.
"""
import json
import lzma
import re
from pathlib import Path

import e3_native41_progress_box as progress

LOST_FLUSH_LIMIT = 2
TORN_PROBLEMS = ('unparsable-progress-line',)
INCOMPLETE = 'line-parts-incomplete:'
FIELD = {name: re.compile(r'"%s":(\d+)' % name) for name in ('base_ns', 'line', 'part', 'parts', 'rank')}


def torn_head(line, at):
    """What a torn line still says about its flush: rank and line come before the rows."""
    text = line[at:]
    cut = text.find('"rows":')
    head = text if cut < 0 else text[:cut]
    found = {name: pattern.search(head) for name, pattern in FIELD.items()}
    row = {name: int(match.group(1)) if match else None for name, match in found.items()}
    row['side'] = 'worker' if '(Worker_' in line[:at] else 'engine'
    row['stamp'] = progress.docker_stamp(line)
    row['bytes'] = len(line)
    return row


def read_lines(paths):
    """progress.read_lines, also keeping what each torn line still says."""
    out, errors, torn = [], [], []
    for path in paths:
        path = Path(path)
        if not path.is_file():
            continue
        opener = lzma.open if path.suffix == '.xz' else open
        with opener(path, 'rt', errors='replace') as handle:
            for line in handle:
                at = line.find(progress.PROGRESS_LINE)
                if at < 0:
                    continue
                try:
                    body = json.loads(line[at + len(progress.PROGRESS_LINE):])
                except ValueError:
                    errors.append('unparsable-progress-line')
                    torn.append(torn_head(line, at))
                    continue
                if type(body) is not dict or body.get('schema') != progress.PROGRESS_SCHEMA:
                    errors.append('progress-line-schema')
                    continue
                out.append((progress.docker_stamp(line), body))
    return out, errors, torn


def lost_flushes(records, torn):
    """Every lost flush: parts missing from a group, or a torn line with no group left."""
    groups = {}
    for stamp, body in records:
        key = (body.get('side'), body.get('rank'), body.get('line'))
        groups.setdefault(key, []).append((stamp, body))
    lost = {}
    for key, parts in groups.items():
        bodies = sorted((b for _, b in parts), key=lambda b: b.get('part', -1))
        count = bodies[0].get('parts')
        if [b.get('part') for b in bodies] != list(range(count or 0)) or any(b.get('parts') != count for b in bodies):
            lost[key] = dict(side=key[0], rank=key[1], line=key[2], stamp=parts[0][0], torn_lines=0)
    unattributed = 0
    for row in torn:
        key = (row['side'], row['rank'], row['line'])
        if row['line'] is None or (row['side'] == 'worker' and row['rank'] is None):
            unattributed += 1
            key = ('unattributed', unattributed, row['stamp'])
        entry = lost.setdefault(key, dict(side=key[0], rank=row['rank'], line=row['line'], stamp=row['stamp'], torn_lines=0))
        entry['torn_lines'] += 1
    return lost


def worker_gaps(flushes, lost):
    """The seq interval each lost worker flush covered: between its rank's neighbouring whole flushes."""
    ranges = {}
    for flush in flushes:
        seqs = [row['seq'] for row in flush['rows'].get('r', [])]
        if flush['side'] == 'worker' and seqs:
            ranges.setdefault(flush['rank'], []).append((flush['line'], min(seqs), max(seqs)))
    gaps = {}
    for key, entry in lost.items():
        if entry['side'] != 'worker' or entry['rank'] is None or entry['line'] is None:
            continue
        rows = ranges.get(entry['rank'], [])
        below = [hi for line, _, hi in rows if line < entry['line']]
        above = [lo for line, lo, _ in rows if line > entry['line']]
        gaps[key] = (entry['rank'], max(below) if below else float('-inf'), min(above) if above else float('inf'))
    return gaps


def explaining(row, gaps):
    """The lost flush holding this missing row, or None."""
    for key, (rank, low, high) in gaps.items():
        if row['rank'] == rank and low < row['seq'] < high:
            return key
    return None


def seq_is_lost(seq, rank, gaps):
    return any(rank == r and low < seq < high for r, low, high in gaps.values())


def aggregate(row, counted):
    """The arm's breakdown, as progress_receipts builds it, over the chunks that remain."""
    median = progress.median
    row['counted_chunks'] = len(counted)
    row['bands'] = {}
    for band in sorted({c['band'] for c in counted}):
        rows = [c for c in counted if c['band'] == band]
        row['bands'][band] = dict(chunks=len(rows), wall_s=median([c['wall_s'] for c in rows]),
                                  open_interval_s=median([c.get('open_interval_s') for c in rows]),
                                  gap_before_s=median([c.get('gap_before_s') for c in rows]),
                                  residual_s=median([c['residual_s'] for c in rows]),
                                  components_s={k: median([c['components_s'][k] for c in rows])
                                                for k in progress.COMPONENTS},
                                  stock_rpcs=median([c['stock_rpcs'] for c in rows]),
                                  wait_s=median([c['wait_s'] for c in rows]),
                                  wait_sends=median([c['wait_sends'] for c in rows]),
                                  commands=median([c['commands'] for c in rows]))
    row['all_counted'] = dict(chunks=len(counted), wall_s=median([c['wall_s'] for c in counted]),
                              open_interval_s=median([c.get('open_interval_s') for c in counted]),
                              gap_before_s=median([c.get('gap_before_s') for c in counted]),
                              residual_s=median([c['residual_s'] for c in counted]),
                              components_s={k: median([c['components_s'][k] for c in counted])
                                            for k in progress.COMPONENTS})


def _in_span(entry, arm):
    start, end = arm.get('wall_start_s'), arm.get('wall_end_s')
    return (type(start) in (int, float) and type(end) in (int, float) and entry['stamp'] is not None
            and start <= entry['stamp'] <= end + progress.LOG_LAG_S)


def _discontinuities(starts, ends, prompt):
    breaks = sum(1 for end, start in zip(ends, starts[1:]) if end != start)
    return breaks + int(starts[:1] != [0]) + int(ends[-1:] != [prompt - 1])


def apply_rule(arms, receipts, records, torn, *, limit=LOST_FLUSH_LIMIT):
    """Apply the rule to progress_receipts' result, in place. Other problems stay as they were."""
    flushes, _ = progress.assemble(records)
    lost = lost_flushes(records, torn)
    gaps = worker_gaps(flushes, lost)
    engine_lines = [f['line'] for f in flushes if f['side'] == 'engine']
    last_line = max(engine_lines) if engine_lines else None
    receipts['problems'] = [p for p in receipts['problems']
                            if p not in TORN_PROBLEMS and not p.startswith(INCOMPLETE)]
    claimed = set()
    for arm in arms:
        row = receipts['arms'].get(arm.get('label'))
        if row is None or 'chunks' not in row:
            continue
        mine = {key for key, entry in lost.items() if _in_span(entry, arm)}
        kept, dropped, unexplained = [], 0, 0
        for chunk in row['chunks']:
            rows = [m for m in chunk['missing_worker_rows']
                    if not (chunk['line'] == last_line and m['kind'] == 'release')]
            keys = [explaining(m, gaps) for m in rows]
            unexplained += sum(1 for key in keys if key is None)
            if rows and all(key is not None for key in keys):
                dropped += 1
                mine.update(keys)
            else:
                kept.append(chunk)
        engine_lost = sum(1 for key in mine if lost[key]['side'] == 'engine')
        expected = (arm.get('progress_delta') or {}).get('chunks')
        absent = expected - row['chunks_logged'] if type(expected) is int else None
        if engine_lost and absent is not None and 0 < absent <= engine_lost:
            dropped += absent
            if 'progress-lines-differ-from-the-chunk-counter' in row['reasons']:
                row['reasons'].remove('progress-lines-differ-from-the-chunk-counter')
            ordered = sorted(row['chunks'], key=lambda c: c['chunk_start'])
            prompt = (arm.get('fresh') or {}).get('prompt_tokens_observed')
            if ('native-chunks-not-row-0-through-n-minus-1' in row['reasons'] and type(prompt) is int
                    and _discontinuities([c['chunk_start'] for c in ordered],
                                         [c['chunk_end'] for c in ordered], prompt) <= absent):
                row['reasons'].remove('native-chunks-not-row-0-through-n-minus-1')
            # An interval that spans a lost chunk is not one chunk's interval.
            for chunk, following in zip(ordered, ordered[1:]):
                if chunk['chunk_end'] != following['chunk_start']:
                    chunk.pop('open_interval_s', None)
                    following.pop('gap_before_s', None)
        row['chunks'] = kept
        row['missing_worker_rows'] = unexplained
        if not unexplained and 'worker-rows-missing' in row['reasons']:
            row['reasons'].remove('worker-rows-missing')
        aggregate(row, [c for c in kept if c['label'] == row['counted_label']])
        claimed |= mine
        row['torn'] = dict(lost_flushes=len(mine), torn_lines=sum(lost[key]['torn_lines'] for key in mine),
                           chunks_dropped=dropped, limit=limit, over_limit=len(mine) > limit)
        if len(mine) > limit:
            row['reasons'].append('torn-progress-flushes-over-limit')
    receipts['torn'] = dict(limit=limit, torn_lines=len(torn), lost_flushes=len(lost),
                            lost_outside_every_arm=len(set(lost) - claimed),
                            detail=[{k: entry[k] for k in ('side', 'rank', 'line', 'stamp', 'torn_lines')}
                                    for _, entry in sorted(lost.items(), key=lambda kv: str(kv[0]))])
    return receipts


def receipts_for(arms, logs, streams_count=8):
    """The grader's one call: progress receipts with the rule applied."""
    records, errors, torn = read_lines(logs)
    receipts = progress.progress_receipts(arms, records, errors, streams_count)
    return records, apply_rule(arms, receipts, records, torn)
