"""Read only, hub, python3 -I. Usage: compose4_rule.py RECORD_DIR [--json OUT]. The run's rule (lead, 5 Nov, fixed before the run; READY-compose-4.md). Stdlib only: it runs on a fetched record, never on the box.

The report leads with the native share of prefill tokens per arm: how much of a closed-loop replay's prefill the native path carries at all.

FAIL, any of:
  - a crash on any arm: the death scan not ALIVE, a worker failure, an OutOfMemoryError in the engine log (fatal OOM);
  - a hang on any arm: an arm that started and left no record, or replay turns still open past the finish bound;
  - on cmp, a stock prefill the levers claim to cover:
    (a) a fresh of at least 32768 new rows, sent with a decoder running and no other prefill in flight or queued ahead, that did not go native;
    (b) an append of at most the yield's max_new new tokens, sent during a live bulk, with the bulk's four yields not spent and time left in it, that went stock instead of being served.
INC, any of: an arm not run or incomplete; a rescued overdraft on cmp, counted apart (sm and ctl print beside it); fewer than 3 native bulks in cmp's window; cmp's numbers under the floor below.
PASS: cmp's live prefill median >= 3300 tok/s and decode vs fixed >= 0.60 over the window, nothing above.
Every other stock prefill is counted and reported: an append with no bulk live, an append over max_new, a fresh with nothing decoding (no_decoder), a fresh in contention (stock_fresh_contended). The literal reading, any stock prefill at all, prints beside the rule.

Prefill: prompt tokens over first output, the client basis of every lever track's number, on native fresh turns whose first chunk opened within QUEUED_S of the send (a queued wait is not prefill speed). Beside it, rows over the chunk span.
Decode: tokens per second per stream while decoding, over the native bulks' spans, times 8 over the fixed 452.8. A closed loop has fewer than 8 decoding, and fewer streams each run faster, so it reads high against an arm's: the mean count, the aggregate and the cmp/ctl ratio print beside it.
Times: memory lines are engine clock (UTC stamps), turns are the client's; the same host.
"""
import collections
import datetime
import glob
import json
import os
import re
import statistics
import sys

RULE_PREFILL_TOK_S, RULE_DECODE_VS_FIXED, RULE_MIN_NATIVE_BULKS = 3300.0, 0.60, 3
FIXED_REFERENCE_TOK_S, STREAMS = 452.8, 8
MIN_NATIVE_ROWS, MAX_NEW, PER_BULK = 32768, 2048, 4
QUEUED_S = 3.0
NEAR_SEND_S = 0.5           # another request sent this close can take the budget before the auto-arm's tick
AFTER_BULK_S = 1.0          # the tail and the cooldown let stock run (native_auto_arm.py:463, :490)
BULK_LEFT_S, PER_HELD_S = 2.0, 2.5    # an append needs a chunk boundary and its turn among the held
ARMS = ('sm.g2-live-0', 'dec1.g2-live-0', 'cmp.g2-live-0', 'ctl.g2-live-0')
ON, OFF = 'cmp.g2-live-0', 'ctl.g2-live-0'
KV = re.compile(r'(\w+)=(\S+)')
STAMP = re.compile(r'^(\S+?Z?)\s')


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def pct(values, q):
    values = sorted(v for v in values if v is not None)
    return values[min(len(values) - 1, int(q * len(values)))] if values else None


def ratio(x, y):
    return round(x / y, 4) if x is not None and y else None


def epoch(stamp):
    head, _, frac = stamp.rstrip('Z').partition('.')
    return datetime.datetime.strptime(head, '%Y-%m-%dT%H:%M:%S').replace(tzinfo=datetime.timezone.utc).timestamp() + float('0.' + (frac or '0'))


def read_log(path):
    """Memory rows (stamp, rank, point, txn, a, b, free MiB, retries, ooms), and the lines that name a fatal OOM or a worker failure."""
    rows, fatal = [], []
    if not os.path.exists(path):
        return rows, fatal
    for line in open(path, errors='replace'):
        if 'OutOfMemoryError' in line or 'FLASH_S5_WORKER_FAILURE' in line:
            fatal.append(line[:300].rstrip())
        at = line.find('flash-s5 memory ')
        if at < 0:
            continue
        kv = dict(KV.findall(line[at + 16:]))
        try:
            a, _, b = kv['chunk'].partition('-')
            rows.append(dict(t=epoch(line.split()[0]), rank=int(kv['rank']), point=kv['point'], txn=int(kv['txn_seq']), a=int(a), b=int(b), free=int(kv['free']) // 1048576, retries=int(kv['alloc_retries']), ooms=int(kv['ooms'])))
        except (KeyError, ValueError, IndexError):
            continue
    return rows, fatal


def chunk_runs(rows):
    """Rank 0's native chunks as runs, one per request: a chunk continues the run that ended where it starts, so a bulk carries on across its yields."""
    chunks = {}
    for row in rows:
        if row['rank'] == 0:
            chunks.setdefault(row['txn'], dict(a=row['a'], b=row['b']))[row['point']] = row['t']
    runs = []
    for chunk in sorted((c for c in chunks.values() if 'open' in c), key=lambda c: c['open']):
        for run in reversed(runs):
            if run['end'] == chunk['a']:
                run.update(end=chunk['b'], last=chunk.get('released', chunk['open']))
                run['chunks'] += 1
                break
        else:
            runs.append(dict(start=chunk['a'], end=chunk['b'], first=chunk['open'], last=chunk.get('released', chunk['open']), chunks=1))
    return runs


def arm_memory(rows, start, end):
    """Min free per rank, retries and the rise in torch's OOM count, in the arm's span."""
    free, retries, ooms = {}, {}, {}
    for row in rows:
        if start <= row['t'] <= end:
            rank = str(row['rank'])
            free[rank] = min(free.get(rank, row['free']), row['free'])
            retries.setdefault(rank, []).append(row['retries'])
            ooms.setdefault(rank, []).append(row['ooms'])
    span = lambda d: {k: max(v) - min(v) for k, v in sorted(d.items())}
    return dict(min_free_mib=dict(sorted(free.items())), alloc_retries=span(retries), oom_rise=span(ooms))


def stamp_turns(turns, origin):
    out = []
    for turn in turns:
        row = dict(turn, send=origin + turn['t'])
        if turn.get('ttft') is not None:
            row['first'] = row['send'] + turn['ttft']
            row['end'] = row['send'] + turn['seconds']
        out.append(row)
    return out


def classify(turns, runs, warmup_at, yield_on):
    """Each turn native or stock, the stock ones by the rule's classes. Only turns sent inside the window count."""
    facts = collections.Counter()
    fresh_rates, span_rates, served, held, violations, bulks = [], [], [], [], [], []
    native_rows = stock_rows = 0
    used = set()
    done = [t for t in turns if t.get('first') is not None and type(t.get('prompt_tokens')) is int]

    def run_of(turn):
        want = turn['prompt_tokens'] - 1
        exact = [(i, r) for i, r in enumerate(runs) if i not in used and r['end'] == want and turn['send'] - 1 <= r['first'] <= turn['first']]
        if exact:
            return exact[0]
        partial = [(i, r) for i, r in enumerate(runs) if i not in used and r['start'] == 0 and turn['kind'] == 'fresh' and r['end'] < want and turn['send'] - 1 <= r['first'] <= turn['first']]
        return partial[0] if partial else (None, None)

    for turn in done:
        index, run = run_of(turn)
        turn['run'] = run
        if run is not None:
            used.add(index)
    live_bulks = [t for t in done if t['kind'] == 'fresh' and t.get('run') is not None]
    for turn in done:
        if turn['send'] < warmup_at:
            continue
        others = [o for o in turns if o is not turn]
        at = turn['send']
        decoding = sum(1 for o in others if o.get('first') is not None and o['first'] <= at < o['end'])
        held_ahead = sum(1 for o in others if o['send'] < at and (o.get('first') is None or at < o['first']))
        near = sum(1 for o in others if abs(o['send'] - at) <= NEAR_SEND_S)
        just_ended = any(b['run']['last'] <= at <= b['run']['last'] + AFTER_BULK_S for b in live_bulks if b is not turn)
        run = turn.get('run')
        cached = turn.get('cached_tokens') or 0
        kind = turn['kind']
        facts[kind] += 1
        if run is not None:
            rows = run['end'] - run['start']
            native_rows += rows
            remainder = turn['prompt_tokens'] - 1 - run['end']
            if remainder > 0:
                stock_rows += remainder
                facts['partial_native'] += 1
            if kind == 'fresh':
                facts['fresh_native'] += 1
                bulks.append(turn)
                if run['first'] - turn['send'] <= QUEUED_S:
                    fresh_rates.append(turn['prompt_tokens'] / turn['ttft'])
                if run['last'] > run['first']:
                    span_rates.append(rows / (run['last'] - run['first']))
            else:
                facts['append_served'] += 1
                served.append(dict(new=rows, ttft=turn['ttft']))
            continue
        new = turn['prompt_tokens'] - cached
        stock_rows += new
        if kind == 'fresh':
            if not decoding:
                facts['stock_fresh_no_decoder'] += 1
            elif held_ahead or near > 0 or just_ended:
                facts['stock_fresh_contended'] += 1
            elif new >= MIN_NATIVE_ROWS:
                facts['stock_fresh_clean'] += 1
                violations.append(dict(clause='a', t=turn['t'], agent=turn['agent'], new=new, decoding=decoding))
            continue
        live = [b for b in live_bulks if b['run']['first'] <= at <= b['run']['last']]
        if not live:
            facts['stock_append_no_bulk'] += 1
            continue
        bulk = live[0]
        held.append(dict(new=new, ttft=turn['ttft']))
        if new > MAX_NEW:
            facts['stock_append_over_max_new'] += 1
            continue
        spent = sum(1 for o in done if o['kind'] == 'append' and o.get('run') is not None and bulk['run']['first'] <= o['run']['first'] <= at)
        left = bulk['run']['last'] - at
        if spent >= PER_BULK:
            facts['stock_append_per_bulk_spent'] += 1
        elif left < BULK_LEFT_S + PER_HELD_S * held_ahead:
            facts['stock_append_bulk_ending'] += 1
        elif yield_on:
            facts['stock_append_clean'] += 1
            violations.append(dict(clause='b', t=turn['t'], agent=turn['agent'], new=new, bulk_left_s=round(left, 2), held_ahead=held_ahead))
        else:
            facts['stock_append_yield_off'] += 1
    facts['stock_total'] = sum(v for k, v in facts.items() if k.startswith('stock_'))
    return dict(counts=dict(facts), bulks=bulks, prefill_tok_s=median(fresh_rates), prefill_n=len(fresh_rates), span_tok_s=median(span_rates), native_bulks=len(bulks), native_rows=native_rows, stock_rows=stock_rows,
                native_share=round(native_rows / (native_rows + stock_rows), 4) if native_rows + stock_rows else None, served=served, held=held, violations=violations)


def decode_over(turns, bulks, origin):
    """Client-visible decode beside the native bulks: per decoding stream, the mean count, the aggregate."""
    tokens, active, total = 0, 0.0, 0.0
    for bulk in bulks:
        first, last = bulk['run']['first'], bulk['run']['last']
        total += last - first
        for turn in turns:
            if turn is bulk or not turn.get('arrivals_ms'):
                continue
            inside = [origin + ms / 1000.0 for ms in turn['arrivals_ms']]
            inside = [t for t in inside if first <= t <= last]
            if len(inside) >= 2:
                tokens += len(inside) - 1
                active += inside[-1] - inside[0]
    per_stream = tokens / active if active else None
    return dict(per_stream_tok_s=round(per_stream, 2) if per_stream else None, decode_vs_fixed=round(per_stream * STREAMS / FIXED_REFERENCE_TOK_S, 4) if per_stream else None, mean_decoding=round(active / total, 2) if total else None,
                aggregate_vs_fixed=round(tokens / total / FIXED_REFERENCE_TOK_S, 4) if total else None)


def ttft_read(rows):
    values = [r['ttft'] for r in rows]
    return dict(n=len(rows), p50=pct(values, 0.5), p95=pct(values, 0.95), max=max(values) if values else None)


def decode_only_read(record):
    window = (record or {}).get('decode_only') or {}
    rates = window.get('stream_rates_tokens_per_s') or ()
    return dict(streams=len(rates), per_stream_tok_s=round(sum(rates) / len(rates), 2) if rates else None, aggregate=window.get('aggregate_tokens_per_s'), gap_ms_p50=window.get('gap_ms_p50'), gap_ms_p99=window.get('gap_ms_p99'))


def load_json(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None


def read_arm(root, label, rows, runs, yield_on):
    record = load_json('%s/arm-%s.json' % (root, label))
    out = dict(record=record, started=os.path.exists('%s/arm-%s.log' % (root, label)))
    if record is None:
        return out
    memory = arm_memory(rows, record.get('wall_start_s') or 0, record.get('wall_end_s') or 0)
    cap = (record.get('reserved_cap') or {}).get('counters') or {}
    rescued = sum(v for name in ('overdrafts', 'overdraft_refused') for v in (cap.get(name) or {}).values() if type(v) is int)
    out.update(memory=memory, overdrafts=cap.get('overdrafts'), overdraft_refused=cap.get('overdraft_refused'), rescued=rescued, oom_unmatched=max(0, sum(memory['oom_rise'].values()) - rescued))
    turns = load_json('%s/arm-%s.json.turns.json' % (root, label))
    if turns is not None:
        origin = record['replay_start_s']
        stamped = stamp_turns(turns, origin)
        warmup = (record.get('replay_config') or {}).get('WARMUP', 0)
        read = classify(stamped, runs, origin + warmup, yield_on)
        read['decode'] = decode_over(stamped, read['bulks'], origin)
        read['returner_first_output_s'] = dict(all=ttft_read(read['served']), up_to_1000=ttft_read([s for s in read['served'] if s['new'] <= 1000]), held_stock=ttft_read(read['held']))
        read['summary'] = record.get('replay_summary')
        read['bulks'] = [dict(t=b['t'], prompt=b['prompt_tokens'], ttft=round(b['ttft'], 2), chunks=b['run']['chunks']) for b in read['bulks']]
        out['read'] = read
    return out


def verdict(arms, death, fatal):
    fail, inc = [], []
    if death not in ('ALIVE', None):
        fail.append('crash: the death scan reads %s' % death)
    if fatal:
        fail.append('crash: %d log lines name OutOfMemoryError or a worker failure' % len(fatal))
    for label in ARMS:
        arm = arms[label]
        record = arm.get('record')
        if record is None:
            (fail if arm.get('started') else inc).append(('hang-or-crash: %s started and left no record' if arm.get('started') else 'not-run: %s') % label)
            continue
        if record.get('replay_still_open'):
            fail.append('hang: %s has %d turns still open' % (label, len(record['replay_still_open'])))
        elif record.get('complete') is not True:
            inc.append('incomplete: %s: %s' % (label, ','.join(record.get('reasons') or ['no-reason'])))
    on = arms[ON]
    read = on.get('read')
    if read is None:
        inc.append('no-read: cmp has no turns')
        return ('FAIL' if fail else 'INC'), fail + inc
    for row in read['violations']:
        fail.append('stock-prefill (%s): cmp %s' % (row['clause'], json.dumps(row, sort_keys=True)))
    if on.get('rescued'):
        inc.append('rescued: cmp %d overdrafts or refused' % on['rescued'])
    if on.get('oom_unmatched'):
        inc.append('oom-unmatched: cmp %d OOMs the overdraft counters do not explain' % on['oom_unmatched'])
    if read['native_bulks'] < RULE_MIN_NATIVE_BULKS:
        inc.append('too-few: cmp %d native bulks in the window, floor %d' % (read['native_bulks'], RULE_MIN_NATIVE_BULKS))
    if read['prefill_n'] < RULE_MIN_NATIVE_BULKS or read['prefill_tok_s'] is None:
        inc.append('too-few: cmp %d unqueued native bulks for the prefill median' % read['prefill_n'])
    elif read['prefill_tok_s'] < RULE_PREFILL_TOK_S:
        inc.append('prefill: cmp %.1f under %.0f' % (read['prefill_tok_s'], RULE_PREFILL_TOK_S))
    decode = read['decode']['decode_vs_fixed']
    if decode is None:
        inc.append('no-decode: cmp has no decode beside a native bulk')
    elif decode < RULE_DECODE_VS_FIXED:
        inc.append('decode: cmp %.4f under %.2f' % (decode, RULE_DECODE_VS_FIXED))
    return ('FAIL' if fail else 'INC' if inc else 'PASS'), fail + inc


def report(root):
    rows, fatal = read_log(root + '/engine-stream.log')
    err = root + '/engine-stream.err'
    if os.path.exists(err):
        fatal += [line[:300].rstrip() for line in open(err, errors='replace') if 'OutOfMemoryError' in line or 'FLASH_S5_WORKER_FAILURE' in line]
    runs = chunk_runs(rows)
    arms = {label: read_arm(root, label, rows, runs, label != OFF) for label in ARMS}
    death = (load_json(root + '/engine-death-scan.json') or {}).get('verdict')
    word, reasons = verdict(arms, death, fatal)
    lines = []
    on, off = arms[ON].get('read') or {}, arms[OFF].get('read') or {}
    lines.append('NATIVE SHARE of prefill tokens, window: cmp %s (%s rows native, %s stock), ctl %s (%s, %s)' % (on.get('native_share'), on.get('native_rows'), on.get('stock_rows'), off.get('native_share'), off.get('native_rows'), off.get('stock_rows')))
    lines.append('PREFILL median, unqueued native bulks: cmp %s (n %s of %s native), ctl %s (n %s of %s); cmp/ctl %s; by chunk span cmp %s ctl %s' % (on.get('prefill_tok_s'), on.get('prefill_n'), on.get('native_bulks'), off.get('prefill_tok_s'), off.get('prefill_n'), off.get('native_bulks'), ratio(on.get('prefill_tok_s'), off.get('prefill_tok_s')), on.get('span_tok_s'), off.get('span_tok_s')))
    d_on, d_off = on.get('decode') or {}, off.get('decode') or {}
    lines.append('DECODE vs fixed beside native bulks: cmp %s, ctl %s; cmp/ctl %s; per stream %s, %s; mean decoding %s, %s; aggregate %s, %s' % (d_on.get('decode_vs_fixed'), d_off.get('decode_vs_fixed'), ratio(d_on.get('decode_vs_fixed'), d_off.get('decode_vs_fixed')), d_on.get('per_stream_tok_s'), d_off.get('per_stream_tok_s'), d_on.get('mean_decoding'), d_off.get('mean_decoding'), d_on.get('aggregate_vs_fixed'), d_off.get('aggregate_vs_fixed')))
    lines.append('STOCK cmp: %s' % json.dumps(on.get('counts'), sort_keys=True))
    lines.append('STOCK ctl: %s' % json.dumps(off.get('counts'), sort_keys=True))
    literal = (on.get('counts') or {}).get('stock_total')
    lines.append('LITERAL reading, any stock prefill on cmp: %s (%s)' % (literal, 'would FAIL' if literal else 'none'))
    lines.append('RETURNER first output s, cmp served: %s; held then stock: %s; ctl held then stock: %s' % ((on.get('returner_first_output_s') or {}).get('all'), (on.get('returner_first_output_s') or {}).get('held_stock'), (off.get('returner_first_output_s') or {}).get('held_stock')))
    sm = (arms['sm.g2-live-0'].get('record') or {}).get('returners') or ()
    lines.append('RETURNER first output s, sm (frozen shape, 153k): %s' % [(r.get('suffix_tokens'), r.get('ttft_s')) for r in sm])
    for label in ARMS:
        arm = arms[label]
        lines.append('MEMORY %s: min free MiB %s, retries %s, OOM rise %s; overdrafts %s refused %s, unmatched %s' % (label, (arm.get('memory') or {}).get('min_free_mib'), (arm.get('memory') or {}).get('alloc_retries'), (arm.get('memory') or {}).get('oom_rise'), arm.get('overdrafts'), arm.get('overdraft_refused'), arm.get('oom_unmatched')))
    summaries = {k: (arms[k].get('read') or {}).get('summary') or {} for k in (ON, OFF)}
    for name in ('turns_per_hour', 'turn_s_p50', 'turn_s_p95', 'append_ttft_p50', 'append_ttft_p95', 'fresh_ttft_p50', 'decode_tps_p50', 'decode_tps_p5', 'output_tokens_per_s', 'prefix_hit_rate_appends', 'warm_turns', 'errors', 'incomplete'):
        lines.append('REPLAY %s: cmp %s ctl %s, cmp/ctl %s' % (name, summaries[ON].get(name), summaries[OFF].get(name), ratio(summaries[ON].get(name), summaries[OFF].get(name))))
    one = decode_only_read(arms['dec1.g2-live-0'].get('record'))
    eight = decode_only_read(load_json(root + '/arm-decode-am1.json'))
    lines.append('ONE STREAM decode only: %s; decode-am1, eight: %s' % (json.dumps(one, sort_keys=True), json.dumps(eight, sort_keys=True)))
    for reason in reasons:
        lines.append('REASON ' + reason)
    for line in fatal[:3]:
        lines.append('FATAL ' + line)
    lines.append('RULE %s' % word)
    facts = {label: {k: v for k, v in arm.items() if k != 'record'} for label, arm in arms.items()}
    return word, lines, dict(rule=word, reasons=reasons, arms=facts, one_stream=one, decode_am1=eight)


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
