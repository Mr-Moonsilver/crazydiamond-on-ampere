"""Read only, hub, python3 -I. Usage: compose4_rule.py RECORD_DIR [--json OUT]. The run's rule (lead, 5 Nov, fixed before the run; READY-compose-4.md). Stdlib only: it runs on a fetched record, never on the box.

The report leads with the native share of prefill tokens per arm: how much of a closed-loop replay's prefill the native path carries at all.

FAIL, any of:
  - a crash on any arm: the death scan not ALIVE, a worker failure, an OutOfMemoryError in the engine log (fatal OOM);
  - a hang on any arm: an arm that started and left no record, replay turns still open past the finish bound, or a turn's read timed out;
  - on cmp, a stock prefill the levers claim to cover:
    (a) a fresh of at least 32768 new rows, sent with a decoder running and no other prefill in flight or queued ahead, that did not go native;
    (b) an append of at most the yield's max_new new tokens, sent during a live bulk, with the bulk's four yields not spent and time left in it, that went stock instead of being served.
INC, any of: an arm not run or incomplete; a rescued overdraft on cmp, counted apart (sm and ctl print beside it); fewer than 3 native bulks in cmp's window; cmp's numbers under the floor below.
PASS: cmp's live prefill median >= 3300 tok/s and decode vs fixed >= 0.60 over the window, nothing above.
Every other stock prefill is counted and reported: an append with no bulk live, an append over max_new, a fresh with nothing decoding (no_decoder), a fresh in contention (stock_fresh_contended). The literal reading, any stock prefill at all, prints beside the rule.

Prefill: prompt tokens over first output, the client basis of every lever track's number, on native fresh turns whose first chunk opened within QUEUED_S of the send (a queued wait is not prefill speed). Beside it, rows over the chunk span.
Decode: tokens per second per stream while decoding, over the native bulks' spans, times 8 over the fixed 452.8. A closed loop has fewer than 8 decoding, and fewer streams each run faster, so it reads high against an arm's: the mean count, the aggregate and the cmp/ctl ratio print beside it.
Per arm, read only, none of them in the verdict (review item 10, lead, 5 Nov):
  rate retention: tokens per second per decoding stream over the fixed 56.6, beside native bulks and with none live;
  tail gaps: the gap between arrivals in a turn's decode, p50, p99 and max, over the window and beside native bulks;
  return-turn first output: every append's first output, and split as served by the yield or stock;
  completed valid turns: turns sent in the window that finished as agent_replay.turn_complete counts them, against the others;
  decode debt: output delivered over the window against the reference. The reference is explicit: 56.6 per stream-second actually decoding (the debt in tokens is that minus delivered), and beside it 452.8 over the whole window, an upper bound a closed loop does not reach because its agents pause;
  extra prefill duration: seconds to first output on fresh turns, native and stock. It is apart from the debt: the first is how long a prefill took, the second what the streams beside it delivered.
An arm stopped mid-run (the death watch's SIGTERM, a timeout) leaves no record, only its turns as they finished (<arm>.json.turns.jsonl, compose4_replay.TurnSink). The verdict stays FAIL for the crash; its numbers print over the turns it has, every line marked PARTIAL window=<first..last turn>, so nobody reads them as a full arm. A turn in flight at the stop is not in them, nor are the arm's end-read counters. If it stopped inside the warmup, the partial window starts at 0.
A record does not prove its journal whole (Codex review, 5 Nov). A full-arm reading needs a header naming this arm and this replay's start, no malformed line except a torn last one, and the journal's turn count equal to the record's replay_requests_sent less replay_still_open. Otherwise the arm reads INC and its numbers print marked PARTIAL. A journal write failure on the box kills an agent and looks like a finished one (build-3; compose-5 fixes it): its turn is short in the journal, and the arm log's thread traceback reads INC here too.
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
AFTER_BULK_S = 1.0          # the tail and the cooldown let stock run (native_auto_arm.py:514, :490)
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
    with open(path, errors='replace') as handle:
        text = handle.readlines()
    for line in text:
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
    return dict(counts=dict(facts), bulks=bulks, live_bulks=live_bulks, prefill_tok_s=median(fresh_rates), prefill_n=len(fresh_rates), span_tok_s=median(span_rates), native_bulks=len(bulks), native_rows=native_rows, stock_rows=stock_rows,
                native_share=round(native_rows / (native_rows + stock_rows), 4) if native_rows + stock_rows else None, served=served, held=held, violations=violations)


def tokens_per_arrival(turn):
    """A server can coalesce tokens into one piece: a turn's arrivals count pieces, its usage counts tokens."""
    arrivals = turn.get('arrivals_ms') or ()
    tokens = turn.get('completion_tokens')
    return tokens / len(arrivals) if arrivals and type(tokens) is int and tokens > 0 else 1.0


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
                tokens += (len(inside) - 1) * tokens_per_arrival(turn)
                active += inside[-1] - inside[0]
    per_stream = tokens / active if active else None
    return dict(per_stream_tok_s=round(per_stream, 2) if per_stream else None, decode_vs_fixed=round(per_stream * STREAMS / FIXED_REFERENCE_TOK_S, 4) if per_stream else None, mean_decoding=round(active / total, 2) if total else None,
                aggregate_vs_fixed=round(tokens / total / FIXED_REFERENCE_TOK_S, 4) if total else None)


def gap_fields(window):
    """A record's decode window: its rows hold each stream's gaps."""
    gaps = [g for row in window.get('rows') or window.get('streams') or () for g in row.get('gaps_ms') or ()]
    return gap_read(gaps)


def gap_read(gaps):
    return dict(n=len(gaps), p50=pct(gaps, 0.5), p99=pct(gaps, 0.99), max=max(gaps) if gaps else None)


def serving_reads(turns, bulks, origin, warmup, duration):
    """The per-arm reads of item 10 over [warmup, duration] on the replay's own clock. Offsets in ms from the replay's origin."""
    start_ms, end_ms = warmup * 1000.0, duration * 1000.0
    spans = sorted(((b['run']['first'] - origin) * 1000.0, (b['run']['last'] - origin) * 1000.0) for b in bulks)
    beside = lambda a, b: any(lo <= a and b <= hi for lo, hi in spans)
    inside = lambda a, b: start_ms <= a and b <= end_ms
    gaps_all, gaps_beside, quiet_tokens, quiet_ms, beside_tokens, beside_ms = [], [], 0, 0.0, 0, 0.0
    bulk_turns = {id(b) for b in bulks}
    delivered = 0
    for turn in turns:      # a fresh turn's own arrivals come after its prefill, so they count as any others
        weight = tokens_per_arrival(turn)
        for a, b in zip(turn.get('arrivals_ms') or (), (turn.get('arrivals_ms') or ())[1:]):
            if not inside(a, b):
                continue
            gaps_all.append(b - a)
            if beside(a, b):
                gaps_beside.append(b - a)
                beside_tokens, beside_ms = beside_tokens + weight, beside_ms + b - a
            else:
                quiet_tokens, quiet_ms = quiet_tokens + weight, quiet_ms + b - a
    for turn in turns:
        delivered += sum(1 for at in turn.get('arrivals_ms') or () if start_ms <= at <= end_ms) * tokens_per_arrival(turn)
    rate = lambda tokens, ms: tokens * 1000.0 / ms if ms else None
    per = lambda value: round(value / (FIXED_REFERENCE_TOK_S / STREAMS), 4) if value else None
    decoding_s = (beside_ms + quiet_ms) / 1000.0
    reference = decoding_s * FIXED_REFERENCE_TOK_S / STREAMS
    window_s = duration - warmup
    sent = [t for t in turns if warmup <= t['t'] <= duration]
    fresh = [t for t in sent if t['kind'] == 'fresh' and t.get('ttft') is not None]
    appends = [t for t in sent if t['kind'] == 'append' and t.get('ttft') is not None]
    return dict(rate_retention=dict(beside_native_bulks=per(rate(beside_tokens, beside_ms)), no_bulk_live=per(rate(quiet_tokens, quiet_ms))),
                tail_gaps_ms=dict(window=gap_read(gaps_all), beside_native_bulks=gap_read(gaps_beside)),
                return_turn_first_output_s=ttft_read([dict(ttft=t['ttft']) for t in appends]),
                completed_valid_turns=dict(valid=sum(1 for t in sent if t.get('complete') is True), not_valid=sum(1 for t in sent if t.get('complete') is not True)),
                decode_debt=dict(delivered_tokens=round(delivered), decoding_stream_s=round(decoding_s, 1), reference_tokens=round(reference), debt_tokens=round(reference - beside_tokens - quiet_tokens), debt_share=round(1 - (beside_tokens + quiet_tokens) / reference, 4) if reference else None,
                                 window_s=window_s, window_reference_tokens=round(window_s * FIXED_REFERENCE_TOK_S), delivered_vs_window_reference=round(delivered / (window_s * FIXED_REFERENCE_TOK_S), 4)),
                prefill_duration_s=dict(fresh_native=round(sum(t['ttft'] for t in fresh if id(t) in bulk_turns), 1), fresh_stock=round(sum(t['ttft'] for t in fresh if id(t) not in bulk_turns), 1), fresh_turns=len(fresh)))


def ttft_read(rows):
    values = [r['ttft'] for r in rows]
    return dict(n=len(rows), p50=pct(values, 0.5), p95=pct(values, 0.95), max=max(values) if values else None)


def decode_only_read(record):
    window = (record or {}).get('decode_only') or {}
    rates = window.get('stream_rates_tokens_per_s') or ()
    return dict(streams=len(rates), per_stream_tok_s=round(sum(rates) / len(rates), 2) if rates else None, aggregate=window.get('aggregate_tokens_per_s'), gaps=gap_fields(window))


def load_json(path):
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def valid_turn(row):
    return type(row) is dict and type(row.get('t')) in (int, float) and row.get('kind') in ('fresh', 'append')


def load_turns(path):
    """(header, turns, malformed) from the arm's sink. Only a last line a kill cut short is skipped silently; any other bad line is counted. (None, None, 0) without a header."""
    try:
        with open(path, errors='replace') as handle:
            text = handle.read()
    except OSError:
        return None, None, 0
    lines = text.split('\n')
    torn = lines.pop()        # after the last newline: empty, or a write a kill cut
    rows, malformed = [], 0
    for line in lines:
        try:
            rows.append(json.loads(line))
        except ValueError:
            malformed += 1
            rows.append(None)
    if torn:
        try:
            rows.append(json.loads(torn))     # whole but for its newline
        except ValueError:
            pass
    if not rows or type(rows[0]) is not dict or rows[0].get('header') is not True:
        return None, None, 0
    turns = [row for row in rows[1:] if valid_turn(row)]
    return rows[0], turns, sum(1 for row in rows[1:] if row is not None and not valid_turn(row)) + malformed


def journal_problem(label, record, header, turns, malformed):
    """Why this journal can't carry a full-arm reading; None when it can. ``record`` None: the arm stopped, so only identity and damage count."""
    if header is None:
        return 'journal-missing-or-no-header'
    if header.get('label') != label:
        return 'journal-names-another-arm:%s' % header.get('label')
    if type(header.get('replay_start_s')) not in (int, float):
        return 'journal-header-has-no-start'
    if malformed:
        return 'journal-malformed-lines:%d' % malformed
    if record is None:
        return None
    if header['replay_start_s'] != record.get('replay_start_s'):
        return 'journal-from-another-replay'
    sent, still = record.get('replay_requests_sent'), record.get('replay_still_open')
    if type(sent) is not int or type(still) is not list:
        return 'record-has-no-request-count'
    if len(turns) != sent - len(still):
        return 'journal-count:%d-turns-against-%d-sent-%d-open' % (len(turns), sent, len(still))
    return None


def arm_log_errors(root, label):
    """An agent thread that raised, as the arm's own log shows it: build-3 doesn't carry it to the record."""
    try:
        with open('%s/arm-%s.log' % (root, label), errors='replace') as handle:
            return [line[:200].rstrip() for line in handle if 'Exception in thread compose4-agent' in line]
    except OSError:
        return []


def partial_of(turns, why):
    first, last = min(t['t'] for t in turns), max(t['t'] + (t.get('seconds') or 0.0) for t in turns)
    return dict(first_turn_s=first, last_turn_s=round(last, 1), turns=len(turns), why=why, word='PARTIAL window=%.1f..%.1f' % (first, last))


def replay_read(stamped, runs, origin, config, yield_on, partial_end=None):
    warmup, duration = config.get('WARMUP', 0), config.get('DURATION', 0)
    if partial_end is not None:
        duration = min(duration, partial_end)
        warmup = warmup if warmup < duration else 0
    read = classify(stamped, runs, origin + warmup, yield_on)
    read['decode'] = decode_over(stamped, read['bulks'], origin)
    read['serving'] = serving_reads(stamped, read.pop('live_bulks'), origin, warmup, duration)
    read['serving']['rate_retention']['beside_native_bulks'] = read['decode']['decode_vs_fixed']    # one number for one thing: the gate's
    read['returner_first_output_s'] = dict(all=ttft_read(read['served']), up_to_1000=ttft_read([s for s in read['served'] if s['new'] <= 1000]), held_stock=ttft_read(read['held']))
    read['bulks'] = [dict(t=b['t'], prompt=b['prompt_tokens'], ttft=round(b['ttft'], 2), chunks=b['run']['chunks']) for b in read['bulks']]
    return read


def read_arm(root, label, rows, runs, yield_on):
    record = load_json('%s/arm-%s.json' % (root, label))
    out = dict(record=record, started=os.path.exists('%s/arm-%s.log' % (root, label)))
    header, turns, malformed = load_turns('%s/arm-%s.json.turns.jsonl' % (root, label))
    if label in (ON, OFF):
        out['agent_errors'] = arm_log_errors(root, label)
        problem = journal_problem(label, record, header, turns, malformed)
        if problem and (record is not None or header is not None):
            out['journal_problem'] = problem
        if problem and header is not None and header.get('label') != label:
            header, turns = None, None        # another arm's turns are no numbers of this one
    if turns is not None:
        out['read_timeouts'] = sum(1 for t in turns if 'timeout' in (t.get('error') or '').lower() or 'timed out' in (t.get('error') or '').lower())
    if record is None:
        if turns:
            origin = header['replay_start_s']
            out['partial'] = partial_of(turns, 'the arm stopped: the rule\'s FAIL stands for the stop')
            last = out['partial']['last_turn_s']
            out['memory'] = arm_memory(rows, origin, origin + last)
            out['read'] = replay_read(stamp_turns(turns, origin), runs, origin, header.get('replay_config') or {}, yield_on, partial_end=last)
        return out
    memory = arm_memory(rows, record.get('wall_start_s') or 0, record.get('wall_end_s') or 0)
    cap = (record.get('reserved_cap') or {}).get('counters') or {}
    rescued = sum(v for name in ('overdrafts', 'overdraft_refused') for v in (cap.get(name) or {}).values() if type(v) is int)
    out.update(memory=memory, overdrafts=cap.get('overdrafts'), overdraft_refused=cap.get('overdraft_refused'), rescued=rescued, oom_unmatched=max(0, sum(memory['oom_rise'].values()) - rescued))
    if turns:
        origin = record.get('replay_start_s') or header['replay_start_s']
        partial_end = None
        if out.get('journal_problem'):
            out['partial'] = partial_of(turns, 'the journal doesn\'t carry the whole arm: ' + out['journal_problem'])
            partial_end = out['partial']['last_turn_s']
        read = replay_read(stamp_turns(turns, origin), runs, origin, record.get('replay_config') or {}, yield_on, partial_end=partial_end)
        read['summary'] = record.get('replay_summary')
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
        if arm.get('read_timeouts'):
            fail.append('hang: %s has %d turns whose read timed out' % (label, arm['read_timeouts']))
        if arm.get('journal_problem'):
            inc.append('journal: %s %s' % (label, arm['journal_problem']))
        if arm.get('agent_errors'):
            inc.append('agent-died: %s %d agent threads raised, see its arm log' % (label, len(arm['agent_errors'])))
        if record is None:
            (fail if arm.get('started') else inc).append(('hang-or-crash: %s started and left no record' if arm.get('started') else 'not-run: %s') % label)
            continue
        if record.get('replay_still_open'):
            fail.append('hang: %s has %d turns still open' % (label, len(record['replay_still_open'])))
        elif record.get('complete') is not True and not arm.get('read_timeouts'):
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
        with open(err, errors='replace') as handle:
            fatal += [line[:300].rstrip() for line in handle if 'OutOfMemoryError' in line or 'FLASH_S5_WORKER_FAILURE' in line]
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
    served_on, served_off = on.get('serving') or {}, off.get('serving') or {}
    for name, text in (('rate_retention', 'RATE RETENTION'), ('tail_gaps_ms', 'TAIL GAPS ms'), ('return_turn_first_output_s', 'RETURN-TURN FIRST OUTPUT s'), ('completed_valid_turns', 'COMPLETED VALID TURNS'), ('decode_debt', 'DECODE DEBT, delivered output against the reference'), ('prefill_duration_s', 'PREFILL DURATION s, apart from the debt')):
        lines.append('%s: cmp %s; ctl %s' % (text, json.dumps(served_on.get(name), sort_keys=True), json.dumps(served_off.get(name), sort_keys=True)))
    one = decode_only_read(arms['dec1.g2-live-0'].get('record'))
    eight = decode_only_read(load_json(root + '/arm-decode-am1.json'))
    lines.append('ONE STREAM decode only: %s; decode-am1, eight: %s' % (json.dumps(one, sort_keys=True), json.dumps(eight, sort_keys=True)))
    sm_window = ((arms['sm.g2-live-0'].get('record') or {}).get('client_basis') or {}).get('decode_window') or {}
    lines.append('SM, frozen shape: tail gaps ms beside the bulk %s; first output of its returners above; completed %s' % (json.dumps(gap_fields(sm_window), sort_keys=True), (arms['sm.g2-live-0'].get('record') or {}).get('complete')))
    for short, label in (('cmp', ON), ('ctl', OFF)):
        partial = arms[label].get('partial')
        if partial:      # every line that names the arm says so
            lines = [re.sub(r'\b%s\b' % short, '%s %s' % (short, partial['word']), line) for line in lines]
    for reason in reasons:
        lines.append('REASON ' + reason)
    for line in fatal[:3]:
        lines.append('FATAL ' + line)
    lines.append('RULE %s' % word)
    for label in (ON, OFF):
        if arms[label].get('partial'):
            lines.insert(-1, '%s %s: %d turns; %s, and no number here is a full arm' % (label, arms[label]['partial']['word'], arms[label]['partial']['turns'], arms[label]['partial']['why']))
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
