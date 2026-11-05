"""Read only, hub. Usage: compose5_rule.py RECORD_DIR [--json OUT]. The run's rule (lead, 5 Nov; tmp/compose-4/DESIGN-compose-5.md). Stdlib only, beside compose4_rule.py.

compose-4's rule, unchanged in its numbers and its clauses, over this table: cmp is the arm it reads, ny its control. On top:
  - INC if the score op's stock rescue ran on cmp (5f's stock_rescue counters; absent until 5f is merged, and then read as none);
  - SMOKE R1: the first read of r1 under yields at 32 connections (lead, 5 Nov 07:49). A read, never in the verdict. Same line for cmp and ny;
  - CHUNK ROWS: this run's b13 (2048 below 52624, 1024 from there): rows per chunk below and past the bound, on the replay pair and the frozen pair. A read; a chunk past the bound over 1024 rows is INC, the plan isn't the one that was confirmed;
  - HOST FILTER: hf's counters per arm, a read;
  - CRASH: on an engine death, a measurement instead of a loss: the arm, each rank's last chunk, torch's 'Tried to allocate' per GPU and the position that size implies, the prompts the journal holds. A read, never in the verdict;
  - PRICE: the yield's price on the frozen pair, each index on and off, with the noise of the two. Numbers, never a word (DESIGN-compose-5.md item 6).
The replay's price is compose-4's cmp and ctl lines, here cmp and ny, one pass each.
"""
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compose4_rule as base  # noqa: E402

SM, ON, OFF = 'sm.g2-live-0', 'cmp.g2-live-0', 'ny.g2-live-0'
PRICE_PAIRS = (('fon.g2-live-0', 'foff.g2-live-0'), ('fon.g2-live-1', 'foff.g2-live-1'))
ARMS = (SM, 'dec1.g2-live-0', 'fon.g2-live-0', 'foff.g2-live-0', 'foff.g2-live-1', 'fon.g2-live-1', ON, OFF)
READ_RANK = '1'
BOUND_END, BOUND_ROWS = 52624, 1024


def __getattr__(name):
    """compose-4's helpers by their names: this rule is that rule over a different table."""
    return getattr(base, name)


def rate(top, under, digits=4):
    return round(top / under, digits) if type(top) in (int, float) and type(under) in (int, float) and under else None


def frozen_arm(record):
    """What one frozen arm says about the yield; None if it left no record."""
    if not isinstance(record, dict):
        return None
    basis = record.get('client_basis') or {}
    window = basis.get('decode_window') or {}
    rates = [v for v in window.get('stream_rates_tokens_per_s') or () if type(v) in (int, float)]
    ends = [s for s in record.get('statuses') or () if s.get('label') in ('after', 'after-append')]
    counters = ((ends[-1].get('append_yield') or {}).get('counters') or {}) if ends else {}
    return dict(bulk_s=basis.get('bulk_complete_s'), prefill_tok_s=basis.get('prefill_tokens_per_s'), decode_aggregate=window.get('aggregate_tokens_per_s'), decode_per_stream=statistics.median(rates) if rates else None,
                gap_ms_p50=window.get('gap_ms_p50'), gap_ms_p99=window.get('gap_ms_p99'), gap_ms_max=window.get('gap_ms_max'), yields_selected=counters.get('selected'),
                returners=[(r.get('suffix_tokens'), r.get('ttft_s')) for r in record.get('returners') or ()], complete=record.get('complete') is True)


def price_read(arms):
    """Each index on over off, then how far the two indexes sit apart: the only noise a pass of the pair has."""
    pairs, ratios = [], []
    for on_label, off_label in PRICE_PAIRS:
        on, off = frozen_arm(arms[on_label].get('record')), frozen_arm(arms[off_label].get('record'))
        row = dict(on=on_label, off=off_label, on_read=on, off_read=off)
        if on and off:
            row.update(bulk_on_over_off=rate(on['bulk_s'], off['bulk_s']), bulk_extra_s=round(on['bulk_s'] - off['bulk_s'], 3) if None not in (on['bulk_s'], off['bulk_s']) else None,
                       prefill_on_over_off=rate(on['prefill_tok_s'], off['prefill_tok_s']), decode_per_stream_on_over_off=rate(on['decode_per_stream'], off['decode_per_stream']))
            ratios.append(row['bulk_on_over_off'])
        pairs.append(row)
    seen = [r for r in ratios if r is not None]
    arm_bulk = lambda which: [frozen_arm(arms[p[which]].get('record')) for p in PRICE_PAIRS]
    spread = lambda rows: (round(abs(rows[0]['bulk_s'] - rows[1]['bulk_s']), 3) if all(r and r.get('bulk_s') is not None for r in rows) else None)
    return dict(pairs=pairs, bulk_ratio_mean=round(statistics.mean(seen), 4) if seen else None, bulk_ratio_spread=round(max(seen) - min(seen), 4) if len(seen) == 2 else None, on_on_bulk_spread_s=spread(arm_bulk(0)), off_off_bulk_spread_s=spread(arm_bulk(1)))


def rank_read(arm, rank=READ_RANK):
    """One rank's memory on one arm: the read DESIGN-compose-5.md's item 2 asks of the smoke and of the replay pair."""
    record = arm.get('record') or {}
    warm = (record.get('score_warm') or {}).get('counters') or {}
    cap = (record.get('reserved_cap') or {}).get('counters') or {}
    memory = arm.get('memory') or {}
    of = lambda block, name: (block.get(name) or {}).get(rank)
    return dict(retries=(memory.get('alloc_retries') or {}).get(rank), rewarms=of(warm, 'rewarms'), allocs=of(warm, 'allocs'), overdrafts=of(cap, 'overdrafts'), overdraft_refused=of(cap, 'overdraft_refused'),
                oom_rise=(memory.get('oom_rise') or {}).get(rank), min_free_mib=(memory.get('min_free_mib') or {}).get(rank))


TRIED = re.compile(r"Tried to allocate ([\d.]+) (KiB|MiB|GiB)\. GPU (\d+) has a total capacity of [\d.]+ GiB of which ([\d.]+) (KiB|MiB|GiB) is free")
UNIT_MIB = dict(KiB=1 / 1024, MiB=1, GiB=1024)


def death_lines(root):
    """(first death stamp or None, torch's message per GPU, first one each), from both engine streams."""
    first, tried = None, {}
    for name in ('engine-stream.log', 'engine-stream.err'):
        try:
            with open('%s/%s' % (root, name), errors='replace') as handle:
                for line in handle:
                    if 'OutOfMemoryError' not in line and 'FLASH_S5_WORKER_FAILURE' not in line:
                        continue
                    try:
                        at = base.epoch(line.split()[0])
                    except (ValueError, IndexError):
                        at = None
                    if at is not None and (first is None or at < first):
                        first = at
                    found = TRIED.search(line)
                    if found and found.group(3) not in tried:
                        size = float(found.group(1)) * UNIT_MIB[found.group(2)]
                        # A score buffer is rows x (position / 4) x 4 bytes, so the size names the position, given the rows. Only a score buffer: torch doesn't say what the allocation was.
                        tried[found.group(3)] = dict(tried_mib=size, free_mib=float(found.group(4)) * UNIT_MIB[found.group(5)], at=at,
                                                     position_if_score_at_2048=round(size * 1048576 / 2048), position_if_score_at_1024=round(size * 1048576 / 1024))
        except OSError:
            continue
    return first, tried


def arm_span(root, label):
    """(start, end) by the arm's record, or by its turns' header with end None: a stopped arm leaves no record. (None, None) when neither says."""
    record = base.load_json('%s/arm-%s.json' % (root, label)) or {}
    header = base.load_turns('%s/arm-%s.json.turns.jsonl' % (root, label))[0] or {}
    return record.get('wall_start_s') or header.get('replay_start_s'), record.get('wall_end_s')


def arm_at(root, labels, at):
    """The arm running at the stamp, by its record or its turns' header. (label, seconds into it) or ('after <label>', None)."""
    last = None
    for label in labels:
        start, end = arm_span(root, label)
        if start is None or (at is not None and start > at):
            continue
        last = (label, start, end)
    if last is None:
        return None, None
    label, start, end = last
    if at is None or (end is not None and end < at):
        return 'after ' + label, None
    return label, round(at - start, 1)


def crash_read(root, labels=ARMS):
    """None without a death line. Each rank's last memory row, torch's sizes, and the largest prompts the arm's journal holds. An in-flight turn is not in the journal, and the engine logs no prompt length."""
    first, tried = death_lines(root)
    if first is None and not tried:
        return None
    label, into = arm_at(root, labels, first)
    rows, _fatal = base.read_log(root + '/engine-stream.log')
    ranks = {}
    for row in rows:
        if first is None or row['t'] <= first:
            ranks[str(row['rank'])] = dict(point=row['point'], chunk='%d-%d' % (row['a'], row['b']), free_mib=row['free'], retries=row['retries'], t=row['t'])
    turns = None
    named = label[len('after '):] if label and label.startswith('after ') else label
    if named:
        _header, rows_, _malformed = base.load_turns('%s/arm-%s.json.turns.jsonl' % (root, named))
        if rows_:
            last = {}
            for row in rows_:
                if type(row.get('prompt_tokens')) is int:
                    last[row.get('agent')] = dict(prompt_tokens=row['prompt_tokens'], kind=row.get('kind'), t=row.get('t'))
            turns = dict(finished=len(rows_), largest_prompt=max((row['prompt_tokens'] for row in rows_ if type(row.get('prompt_tokens')) is int), default=None), last_per_agent={str(k): v for k, v in sorted(last.items(), key=lambda x: str(x[0]))})
    return dict(first_death_s=first, arm=label, into_arm_s=into, ranks=ranks, tried=tried, turns=turns)


def chunk_rows(rows, start, end):
    """Chunks opened in the arm's span, on rank 0, by rows: how many over 1024 and the largest, below and past the bound."""
    below, past = [], []
    for row in rows:
        if row['point'] == 'open' and row['rank'] == 0 and start <= row['t'] <= end:
            (below if row['a'] < BOUND_END else past).append(row['b'] - row['a'])
    return dict(below_bound=len(below), below_over_1024=sum(1 for n in below if n > BOUND_ROWS), below_max=max(below, default=None), past_bound=len(past), past_over_1024=sum(1 for n in past if n > BOUND_ROWS), past_max=max(past, default=None))


def arm_chunk_rows(root, label, rows):
    """chunk_rows over the arm's span. No final record: from its journal's header to the first death line or the next arm's start, marked partial. No start known: unavailable, never a count of zero."""
    start, end = arm_span(root, label)
    if start is None:
        return dict(unavailable='no final record and no journal header')
    if end is not None:
        return chunk_rows(rows, start, end)
    limits = [at for at in (death_lines(root)[0],) if at is not None and at >= start]
    for later in ARMS[ARMS.index(label) + 1:]:
        following = arm_span(root, later)[0]
        if following is not None:
            limits.append(following)
            break
    return dict(chunk_rows(rows, start, min(limits) if limits else float('inf')), partial=True)


def stock_rescues(arm):
    counters = ((arm.get('record') or {}).get('stock_rescue') or {}).get('counters') or {}
    return sum(v for name in ('attempts', 'refused') for v in (counters.get(name) or {}).values() if type(v) is int)


def report(root):
    bound = dict(ARMS=base.ARMS, ON=base.ON, OFF=base.OFF)
    base.ARMS, base.ON, base.OFF = ARMS, ON, OFF
    try:
        word, lines, facts = base.report(root)
    finally:
        base.ARMS, base.ON, base.OFF = bound['ARMS'], bound['ON'], bound['OFF']
    lines = [re.sub(r'\bctl\b', 'ny', line) for line in lines]
    arms = facts['arms']
    for label in ARMS:     # facts carry no record; the reads below want it
        arms[label]['record'] = base.load_json('%s/arm-%s.json' % (root, label))
    rule_line = lines.pop()
    rescues = stock_rescues(arms[ON])
    reasons = list(facts.get('reasons') or [])      # what this rule adds goes in the JSON too, beside the base verdict's
    if rescues:
        reasons.append('stock-rescue: cmp %d' % rescues)
        lines.append('REASON ' + reasons[-1])
        word = 'FAIL' if word == 'FAIL' else 'INC'
    price = price_read(arms)
    for pair in price['pairs']:
        lines.append('PRICE %s over %s: bulk on/off %s, %s s extra; prefill %s; decode per stream %s; on %s; off %s' % (pair['on'], pair['off'], pair.get('bulk_on_over_off'), pair.get('bulk_extra_s'), pair.get('prefill_on_over_off'), pair.get('decode_per_stream_on_over_off'), json.dumps(pair['on_read'], sort_keys=True), json.dumps(pair['off_read'], sort_keys=True)))
    lines.append('PRICE NOISE: bulk on/off mean %s, spread between the indexes %s; on-on bulk apart %s s, off-off %s s (the two indexes are different prompts of one band)' % (price['bulk_ratio_mean'], price['bulk_ratio_spread'], price['on_on_bulk_spread_s'], price['off_off_bulk_spread_s']))
    read = {label: rank_read(arms[label]) for label in (SM, ON, OFF)}
    lines.append('SMOKE R1, the first read of r1 under yields at 32: %s' % json.dumps(read[SM], sort_keys=True))
    lines.append('R1 replay: cmp %s; ny %s' % (json.dumps(read[ON], sort_keys=True), json.dumps(read[OFF], sort_keys=True)))
    lines.append('STOCK RESCUE cmp: %d' % rescues)
    rows, _fatal = base.read_log(root + '/engine-stream.log')
    chunks = {}
    for label in ARMS:
        record = arms[label].get('record') or {}
        chunks[label] = arm_chunk_rows(root, label, rows)
        lines.append('CHUNK ROWS %s: %s; HOST FILTER %s' % (label, json.dumps(chunks[label], sort_keys=True), json.dumps((record.get('cow_host_filter') or {}).get('counters'), sort_keys=True)))
    over = sum(chunks[label].get('past_over_1024') or 0 for label in ARMS)     # a partial span's chunks are real rows; an arm with no span has no count
    if over:
        reasons.append('chunk-rows: %d chunks past the bound over 1024 rows' % over)
        lines.append('REASON ' + reasons[-1])
        word = 'FAIL' if word == 'FAIL' else 'INC'
    crash = crash_read(root)
    if crash:
        lines.append('CRASH arm %s, %s s into it, first death line at %s' % (crash['arm'], crash['into_arm_s'], crash['first_death_s']))
        for rank, row in sorted(crash['ranks'].items()):
            lines.append('CRASH rank %s last chunk: %s' % (rank, json.dumps(row, sort_keys=True)))
        for gpu, row in sorted(crash['tried'].items()):
            lines.append('CRASH GPU %s, torch: %s' % (gpu, json.dumps(row, sort_keys=True)))
        lines.append("CRASH turns, finished only, the one in flight isn't there: %s" % json.dumps(crash['turns'], sort_keys=True))
    assert rule_line.startswith('RULE '), rule_line
    lines.append('RULE %s' % word)
    for label in ARMS:
        arms[label].pop('record', None)
    facts.update(rule=word, reasons=reasons, price=price, r1=read, stock_rescues=rescues, chunk_rows=chunks, crash=crash)
    return word, lines, facts


def main(argv):
    if not argv or argv[0].startswith('-'):
        print(__doc__)
        return 2
    word, lines, facts = report(argv[0])
    print('\n'.join(lines))
    if '--json' in argv:
        with open(argv[argv.index('--json') + 1], 'x') as handle:
            json.dump(facts, handle, sort_keys=True, default=str)
    return {'PASS': 0, 'FAIL': 1}.get(word, 2)


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
