#!/usr/bin/env python3
"""Exploratory reader for the q2024p-j2 host-gap batch pair (off/on). Read-only; no rule, no thresholds, no verdict.

Per record and arm (sp.g2-live-0, n3.g2-live-0, n3.g2-live-1, sp.g2-live-1) it prints:
  turns/h and the n3 x sp ratio per seed; native share (n3 arms, from compose5-grade coverage);
  the arm's small_piece_batch status-block delta (after minus before): batched submissions, commands per submission,
    both-ran-without-yield, worker stops by reason, refusals by reason, batch failures, outstanding, send and host times;
  stage time per <=176-row piece from the engine's stage-local receipt lines inside the arm's replay window
    (off and on both log them, so off vs on compares the same receipt);
  decode token-gap p50/p95/p99 from the arm's turn journal (consecutive SSE arrivals within one turn);
and per record: numerics_o3 verdict, queue/compose5 grade verdicts, OOM max and fatal lines per rank.

The off record must not carry the block (the flag is 0, the engine class is unchanged); the on record must. Either
mismatch is printed as a NOTE, never turned into a decision.

Usage:
  python3 -B tools/q2024p_j2_hgb_reader.py off=<FETCH-OFF>/q18a-parity-n3s24q2024p-j2-hgb-off \
      on=<FETCH-ON>/q18a-parity-n3s24q2024p-j2-hgb-on [--json-out <fresh.json>]
"""
import argparse
import datetime
import json
import math
from pathlib import Path
import re

ARMS = ('sp.g2-live-0', 'n3.g2-live-0', 'n3.g2-live-1', 'sp.g2-live-1')
SMALL_PIECE_ROWS = 176
KEY = 'small_piece_batch'
STAGE = re.compile(r'^(\S+) flash-s5 stage-local txn=(\d+) chunk=(\d+)-(\d+) commands=(\d+),(\d+) '
                   r'both_stages_ns=(\d+) first_ns=(\S+),(\S+) last_ns=(\S+),(\S+) ')
MEMORY = re.compile(r'flash-s5 memory .*?\brank=(\d+)\b.*?\booms=(\d+)\b')
FATAL = re.compile(r'FLASH_S5_WORKER_FAILURE|CUDA (?:error:.*(?:illegal|out of memory)|out of memory)|OutOfMemoryError|'
                   r'WorkerRPCError.*worker-fatal|Segmentation fault', re.I)
COUNTERS = ('sent', 'commands_sent', 'single_layer_commands', 'applied_commands', 'second_applied', 'second_not_run',
            'partial', 'failures')
MAPS = ('worker_stop', 'refused', 'pieces', 'unbatched_reason')


def load(path):
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return value if type(value) is dict else None


def stamp(text):
    """Docker's RFC 3339 nanosecond stamp to Unix seconds (microsecond precision)."""
    text = text.rstrip('Z')
    head, _, frac = text.partition('.')
    when = datetime.datetime.fromisoformat(head + ('.' + (frac + '000000')[:6] if frac else ''))
    return when.replace(tzinfo=datetime.timezone.utc).timestamp()


def parse_stage_line(line):
    """One stage-local receipt as a dict, or None. A stage that logged no command has first/last 'None'."""
    m = STAGE.match(line)
    if m is None:
        return None
    start, end = int(m[3]), int(m[4])
    stages = []
    for first, last in ((m[8], m[10]), (m[9], m[11])):
        stages.append(int(last) - int(first) if first.isdigit() and last.isdigit() else None)
    return dict(at_s=stamp(m[1]), txn=int(m[2]), rows=end - start, commands=[int(m[5]), int(m[6])],
                stage_ns=stages)


def quantile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]


def summary_ms(values):
    if not values:
        return dict(n=0)
    return dict(n=len(values), mean_ms=round(sum(values) / len(values) / 1e6, 3),
                p50_ms=round(quantile(values, .5) / 1e6, 3), p90_ms=round(quantile(values, .9) / 1e6, 3))


def stage_pieces(lines, windows):
    """Per arm window: <=176-row piece stage times (both stages pooled and per stage) and the piece counts."""
    out = {label: dict(small=[], per_stage=([], []), pieces=0, small_pieces=0) for label in windows}
    for line in lines:
        if 'flash-s5 stage-local ' not in line:
            continue
        row = parse_stage_line(line)
        if row is None:
            continue
        for label, (start, end) in windows.items():
            if start <= row['at_s'] <= end:
                slot = out[label]
                slot['pieces'] += 1
                if row['rows'] <= SMALL_PIECE_ROWS:
                    slot['small_pieces'] += 1
                    for stage, value in enumerate(row['stage_ns']):
                        if value is not None:
                            slot['small'].append(value)
                            slot['per_stage'][stage].append(value)
                break
    return {label: dict(pieces=slot['pieces'], small_pieces=slot['small_pieces'],
                        small_stage=summary_ms(slot['small']),
                        small_stage0=summary_ms(slot['per_stage'][0]),
                        small_stage1=summary_ms(slot['per_stage'][1]))
            for label, slot in out.items()}


def token_gaps(path):
    """Consecutive SSE arrival gaps (ms) inside each turn of one arm's journal."""
    gaps = []
    try:
        with Path(path).open() as handle:
            for line in handle:
                try:
                    turn = json.loads(line)
                except ValueError:
                    continue
                arrivals = turn.get('arrivals_ms') if type(turn) is dict else None
                if type(arrivals) is list and len(arrivals) > 1:
                    gaps += [b - a for a, b in zip(arrivals, arrivals[1:]) if type(a) in (int, float) and type(b) in (int, float)]
    except OSError:
        return None
    return dict(n=len(gaps), p50_ms=quantile(gaps, .5), p95_ms=quantile(gaps, .95), p99_ms=quantile(gaps, .99))


def _sub(after, before):
    if type(after) in (int, float) and type(before) in (int, float):
        return after - before
    return None


def batch_delta(before, after):
    """The arm's small_piece_batch delta (after minus before). None when either status row lacks the block."""
    if type(before) is not dict or type(after) is not dict:
        return None
    delta = {name: _sub(after.get(name), before.get(name)) for name in COUNTERS}
    for name in MAPS:
        a, b = after.get(name) or {}, before.get(name) or {}
        delta[name] = {key: _sub(a.get(key, 0), b.get(key, 0)) for key in sorted(set(a) | set(b))
                       if _sub(a.get(key, 0), b.get(key, 0))}
    submit = [_sub(x, y) for x, y in zip(after.get('submit_ns') or (), before.get('submit_ns') or ())]
    host = [_sub(x, y) for x, y in zip(after.get('worker_host_ns') or (), before.get('worker_host_ns') or ())]
    apply_a, apply_b = after.get('send_to_apply_ns') or (), before.get('send_to_apply_ns') or ()
    delta['outstanding_after'] = after.get('outstanding')
    sent = delta['sent']
    delta['commands_per_submission'] = (round(delta['commands_sent'] / sent, 3)
                                        if sent and delta['commands_sent'] is not None else None)
    delta['both_ran_without_yield'] = delta['second_applied']
    delta['send_ms_per_submission'] = (round(submit[0] / submit[1] / 1e6, 6)
                                       if len(submit) == 2 and submit[1] else None)
    delta['worker_host_ms_per_command'] = (round(host[0] / host[1] / 1e6, 6) if len(host) == 2 and host[1] else None)
    if len(apply_a) == 3 and len(apply_b) == 3 and apply_a[2] - apply_b[2]:
        delta['send_to_apply_ms_mean'] = round((apply_a[0] - apply_b[0]) / (apply_a[2] - apply_b[2]) / 1e6, 4)
        delta['send_to_apply_ms_max_lifetime'] = round(apply_a[1] / 1e6, 4)
    rows = {}
    for key in ('batched', 'unbatched'):
        a = (after.get('piece_stage_ns') or {}).get(key) or [0, 0]
        b = (before.get('piece_stage_ns') or {}).get(key) or [0, 0]
        n = a[1] - b[1]
        rows[key] = dict(n=n, mean_ms=round((a[0] - b[0]) / n / 1e6, 3) if n else None)
    delta['piece_stage'] = rows
    return delta


def status_rows(arm):
    rows = {}
    for row in arm.get('statuses') or ():
        if type(row) is dict and row.get('label') in ('entry', 'before', 'after'):
            rows[row['label']] = row
    return rows


def turns_per_hour(arm):
    cfg, summary = arm.get('replay_config') or {}, arm.get('replay_summary') or {}
    try:
        done = summary['warm_turns'] - summary['turns_finished_after_window']
        return done, done * 3600 / (cfg['DURATION'] - cfg['WARMUP'])
    except (KeyError, TypeError, ZeroDivisionError):
        return None, None


def scan_engine(root):
    path = root / 'engine-stream.log'
    if not path.is_file():
        return dict(missing='engine-stream.log')
    ooms, fatal = {}, 0
    with path.open(errors='replace') as handle:
        for line in handle:
            m = MEMORY.search(line)
            if m:
                ooms[int(m[1])] = max(ooms.get(int(m[1]), 0), int(m[2]))
            fatal += bool(FATAL.search(line))
    return dict(oom_max=max(ooms.values()) if ooms else None, oom_by_rank={str(k): v for k, v in sorted(ooms.items())},
                fatal_lines=fatal)


def identity_flag(root):
    """FLASH_S5_SMALL_PIECE_BATCH in the engine's start and end identity env ('unset' when absent, None when unreadable)."""
    values = {}
    for phase in ('start', 'end'):
        doc = load(root / ('engine.identity-%s.json' % phase))
        env = ((doc or {}).get('identity') or {}).get('env') if doc else None
        values[phase] = None if type(env) is not dict else env.get('FLASH_S5_SMALL_PIECE_BATCH', 'unset')
    return values


def collect(root, variant):
    root = Path(root)
    notes = []
    flag = identity_flag(root)
    if any(value != {'off': '0', 'on': '1'}[variant] for value in flag.values()):
        notes.append('identity-flag-%s-start-%s-end-%s' % (variant, flag['start'], flag['end']))
    grade, queue = load(root / 'compose5-grade.json') or {}, load(root / 'queue-grade.json') or {}
    arms, windows = {}, {}
    for label in ARMS:
        arm = load(root / ('arm-' + label + '.json'))
        if arm is None:
            notes.append('missing-arm:' + label)
            continue
        done, tph = turns_per_hour(arm)
        rows = status_rows(arm)
        block = {tag: row.get(KEY) for tag, row in rows.items() if KEY in row}
        if variant == 'off' and block:
            notes.append('off-record-carries-small_piece_batch:' + label)
        if variant == 'on' and set(block) != set(rows):
            notes.append('on-record-lacks-small_piece_batch:' + label)
        coverage = (grade.get('coverage') or {}).get(label) or {}
        arms[label] = dict(completed_turns=done, turns_per_hour=tph, complete=arm.get('complete'),
                           reasons=arm.get('reasons'),
                           native_share=None if label.startswith('sp.') else
                           {k: coverage.get(k) for k in ('live_native_share', 'all_native_share',
                                                         'live_native_share_strict', 'all_native_share_strict')},
                           small_piece_batch=batch_delta(block.get('before'), block.get('after')),
                           decode_gaps=token_gaps(root / ('arm-' + label + '.json.turns.jsonl')))
        start, end = arm.get('replay_start_s'), arm.get('replay_end_s')
        if type(start) in (int, float) and type(end) in (int, float):
            windows[label] = (start, end)
    log = root / 'engine.log'
    if not log.is_file():
        log = root / 'engine-stream.log'
    if log.is_file():
        with log.open(errors='replace') as handle:
            pieces = stage_pieces(handle, windows)
        for label, row in pieces.items():
            arms[label]['stage_pieces'] = row
    else:
        notes.append('missing-engine-log')
    for seed in (0, 1):
        n3, sp = arms.get('n3.g2-live-%d' % seed), arms.get('sp.g2-live-%d' % seed)
        if n3 and sp and n3['completed_turns'] and sp['completed_turns']:
            n3['x_sp'] = round(n3['completed_turns'] / sp['completed_turns'], 4)
    totals = [sum(arms[k]['completed_turns'] or 0 for k in arms if k.startswith(p)) for p in ('n3.', 'sp.')]
    return dict(record=str(root), variant=variant, arms=arms, notes=notes, identity_flag=flag,
                x_sp=round(totals[0] / totals[1], 4) if totals[1] else None,
                numerics=(grade.get('numerics_o3') or {}).get('verdict'),
                compose5_grade=grade.get('verdict'), queue_grade=queue.get('verdict'), queue_rc=queue.get('rc'),
                engine_crashed=queue.get('engine_crashed'), engine=scan_engine(root))


def _fmt(value, digits=1):
    return '-' if value is None else (('%.' + str(digits) + 'f') % value if type(value) is float else str(value))


def render(results):
    lines = []
    for result in results:
        lines.append('== %s  %s  flag=%s/%s x_sp=%s numerics=%s grade=%s/%s rc=%s oom_max=%s fatal=%s' % (
            result['variant'], result['record'], result['identity_flag']['start'], result['identity_flag']['end'],
            _fmt(result['x_sp'], 4), result['numerics'], result['queue_grade'],
            result['compose5_grade'], result['queue_rc'], result['engine'].get('oom_max'), result['engine'].get('fatal_lines')))
        for label, arm in result['arms'].items():
            share = arm['native_share'] or {}
            gaps = arm['decode_gaps'] or {}
            pieces = arm.get('stage_pieces') or {}
            small = pieces.get('small_stage') or {}
            lines.append('  %-13s turns/h=%-7s x_sp=%-6s native=%s/%s gap_p95=%sms small_pieces=%s/%s stage_mean=%sms p50=%sms' % (
                label, _fmt(arm['turns_per_hour']), _fmt(arm.get('x_sp'), 4), _fmt(share.get('live_native_share_strict'), 6),
                _fmt(share.get('all_native_share_strict'), 6), _fmt(gaps.get('p95_ms')), _fmt(pieces.get('small_pieces')),
                _fmt(pieces.get('pieces')), _fmt(small.get('mean_ms'), 3), _fmt(small.get('p50_ms'), 3)))
            batch = arm['small_piece_batch']
            if batch is not None:
                lines.append('      batch sent=%s cmds/sub=%s both_ran=%s not_run=%s partial=%s failures=%s outstanding=%s '
                             'stop=%s refused=%s piece_stage=%s send_ms=%s host_ms=%s' % (
                                 batch['sent'], batch['commands_per_submission'], batch['both_ran_without_yield'],
                                 batch['second_not_run'], batch['partial'], batch['failures'], batch['outstanding_after'],
                                 json.dumps(batch['worker_stop'], sort_keys=True), json.dumps(batch['refused'], sort_keys=True),
                                 json.dumps(batch['piece_stage'], sort_keys=True), batch['send_ms_per_submission'],
                                 batch['worker_host_ms_per_command']))
        for note in result['notes']:
            lines.append('  NOTE ' + note)
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('records', nargs='+', help='off=<record dir> and/or on=<record dir>')
    parser.add_argument('--json-out', type=Path)
    args = parser.parse_args(argv)
    results = []
    for item in args.records:
        variant, sep, path = item.partition('=')
        if not sep or variant not in ('off', 'on'):
            parser.error('records are off=<dir> or on=<dir>')
        results.append(collect(Path(path), variant))
    print(render(results))
    print('EXPLORATORY no rule, no thresholds, no verdict')
    if args.json_out:
        with args.json_out.open('x') as handle:
            json.dump(dict(schema='q2024p-j2-hgb-read/1', exploratory=True, results=results), handle, indent=2, sort_keys=True)
            handle.write('\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
