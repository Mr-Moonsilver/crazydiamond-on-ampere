"""Read only, hub. Usage: compose5_rule.py RECORD_DIR [--json OUT]. The run's rule (lead, 5 Nov; tmp/compose-4/DESIGN-compose-5.md). Stdlib only, beside compose4_rule.py.

compose-4's rule, unchanged in its numbers and its clauses, over this table: cmp is the arm it reads, ny its control. On top:
  - INC if the score op's stock rescue ran on cmp (5f's stock_rescue counters; absent until 5f is merged, and then read as none);
  - SMOKE R1: the first read of r1 under yields at 32 connections (lead, 5 Nov 07:49). A read, never in the verdict. Same line for cmp and ny;
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
    if rescues:
        lines.append('REASON stock-rescue: cmp %d' % rescues)
        word = 'FAIL' if word == 'FAIL' else 'INC'
    price = price_read(arms)
    for pair in price['pairs']:
        lines.append('PRICE %s over %s: bulk on/off %s, %s s extra; prefill %s; decode per stream %s; on %s; off %s' % (pair['on'], pair['off'], pair.get('bulk_on_over_off'), pair.get('bulk_extra_s'), pair.get('prefill_on_over_off'), pair.get('decode_per_stream_on_over_off'), json.dumps(pair['on_read'], sort_keys=True), json.dumps(pair['off_read'], sort_keys=True)))
    lines.append('PRICE NOISE: bulk on/off mean %s, spread between the indexes %s; on-on bulk apart %s s, off-off %s s (the two indexes are different prompts of one band)' % (price['bulk_ratio_mean'], price['bulk_ratio_spread'], price['on_on_bulk_spread_s'], price['off_off_bulk_spread_s']))
    read = {label: rank_read(arms[label]) for label in (SM, ON, OFF)}
    lines.append('SMOKE R1, the first read of r1 under yields at 32: %s' % json.dumps(read[SM], sort_keys=True))
    lines.append('R1 replay: cmp %s; ny %s' % (json.dumps(read[ON], sort_keys=True), json.dumps(read[OFF], sort_keys=True)))
    lines.append('STOCK RESCUE cmp: %d' % rescues)
    assert rule_line.startswith('RULE '), rule_line
    lines.append('RULE %s' % word)
    for label in ARMS:
        arms[label].pop('record', None)
    facts.update(rule=word, price=price, r1=read, stock_rescues=rescues)
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
