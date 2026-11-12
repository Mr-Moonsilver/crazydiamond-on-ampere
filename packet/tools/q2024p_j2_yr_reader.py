#!/usr/bin/env python3
"""Exploratory reader for the q2024p-j2 yield-rule packets (yr off / y1 / y12). Read-only; no rule, no thresholds,
no verdict.

Everything the yrs reader prints (tools/q2024p_j2_yrs_reader.py, reused unchanged; the yr packets keep the batch and
the settle at 0, so it reads each record as 'off'), and per n3 arm, inside the arm's replay window:
  * yields (native-gap append_yield_selected) by owner class, two ways:
      owner-new   the owner's own native rows (native_end - its first native chunk start) <= max_new: the
                  simulator's Y1 class ('an append owner');
      rows-left   native_end - the junction chunk's end <= max_new: the rule FLASH_S5_YIELD_NO_APPEND_OWNER applies;
  * follow-on yields (an owner's second and later yields) and the owner chunks opened between consecutive yields of one
    owner, each yield placed at its first append's admission, i.e. in lane order (the simulator's '260 of 441 after
    exactly one owner chunk');
  * appends per window (append_yield_admitted per yield window) and the in-window append->append gap (none before Y2);
  * Y1: 'flash-s5 n3-yield-owner-refused' receipts and the queue_policy.no_append_owner delta;
  * per turn (the arm's turn journal joined to the engine's request_admitted by prompt tokens and time): append pre-open
    agent-s (sent -> first native chunk open, clipped to the window), append TTFT mean, longest fresh TTFT, and the
    turns the join could not place.
Per record: FLASH_S5_YIELD_NO_APPEND_OWNER and FLASH_S5_YIELD_WINDOW in the identity env. NOTEs, never verdicts.

Usage:
  python3 -I -B tools/q2024p_j2_yr_reader.py off=<FETCH>/q18a-parity-n3s24q2024p-j2-yrs-off \
      y1=<FETCH>/q18a-parity-n3s24q2024p-j2-yr-y1 [y12=<FETCH>/...-yr-y12] [--json-out <fresh.json>]
"""
import argparse
import bisect
import collections
import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location('q2024p_j2_yrs_reader', Path(__file__).resolve().parent / 'q2024p_j2_yrs_reader.py')
yrs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(yrs)
hgb = yrs.hgb

N3 = ('n3.g2-live-0', 'n3.g2-live-1')
Y1_ENV, Y2_ENV = 'FLASH_S5_YIELD_NO_APPEND_OWNER', 'FLASH_S5_YIELD_WINDOW'
VARIANTS = {'off': ('0', '0'), 'y1': ('1', '0'), 'y12': ('1', '1')}
RECEIPT = 'flash-s5 n3-yield-owner-refused '
DEFAULT_MAX_NEW = 8192
JOIN_SLACK_S = 2.0


def identity_env(root, name):
    values = {}
    for phase in ('start', 'end'):
        doc = hgb.load(root / ('engine.identity-%s.json' % phase))
        env = ((doc or {}).get('identity') or {}).get('env') if doc else None
        values[phase] = None if type(env) is not dict else env.get(name, 'unset')
    return values


def read_log(path):
    """Native-gap events (time, event, doc) sorted, and the Y1 receipts (time, doc)."""
    events, receipts = [], []
    with path.open(errors='replace') as handle:
        for line in handle:
            if 'flash-s5 ' not in line:
                continue
            if 'flash-s5 native-gap ' in line:
                try:
                    at = hgb.stamp(line.split(' ', 1)[0])
                    doc = json.loads(line.split('flash-s5 native-gap ', 1)[1])
                except ValueError:
                    continue
                if type(doc) is dict and doc.get('event') in ('chunk_open', 'append_yield_selected', 'append_yield_admitted',
                                                              'append_yield_end', 'request_admitted'):
                    events.append((at, doc['event'], doc))
            elif RECEIPT in line:
                try:
                    receipts.append((hgb.stamp(line.split(' ', 1)[0]), json.loads(line.split(RECEIPT, 1)[1])))
                except ValueError:
                    continue
    events.sort(key=lambda e: e[0])
    return events, receipts


def owners(events):
    """(key, generation) -> dict(native_end, first_start, opens=[(time, txn)])."""
    out = {}
    for at, kind, doc in events:
        if kind != 'chunk_open':
            continue
        owner = out.setdefault((doc.get('request_key'), doc.get('request_generation')),
                               dict(native_end=doc.get('native_end'), first_start=doc.get('chunk_start'), opens=[]))
        owner['opens'].append((at, doc.get('txn')))
    return out


def yields(events, table, start, end, max_new):
    sel = [(at, doc) for at, kind, doc in events if kind == 'append_yield_selected' and start <= at <= end]
    admitted = [at for at, kind, _ in events if kind == 'append_yield_admitted']
    ends = [at for at, kind, _ in events if kind == 'append_yield_end']
    cls = collections.Counter()
    by_owner = collections.defaultdict(list)
    windows = []
    for at, doc in sel:
        owner = table.get((doc.get('request_key'), doc.get('request_generation')))
        frontier = doc.get('chunk_end')
        if owner is None or type(owner['native_end']) is not int or type(frontier) is not int:
            cls['unknown-owner'] += 1
            continue
        own_new = owner['native_end'] - owner['first_start']
        left = owner['native_end'] - frontier
        cls['owner-new<=max_new' if own_new <= max_new else 'owner-new>max_new'] += 1
        cls['rows-left<=max_new' if left <= max_new else 'rows-left>max_new'] += 1
        nxt = ends[bisect.bisect_left(ends, at)] if bisect.bisect_left(ends, at) < len(ends) else float('inf')
        inside = [t for t in admitted if at <= t <= nxt]
        windows.append(len(inside))
        # lane order (the simulator's cut): the yield sits at its first append's admission (its first chunk open)
        by_owner[(doc.get('request_key'), doc.get('request_generation'))].append(inside[0] if inside else at)
    between = collections.Counter()
    follow = 0
    for key, times in by_owner.items():
        opens = [t for t, _ in table[key]['opens']]
        for a, b in zip(times, times[1:]):
            follow += 1
            between[str(sum(1 for t in opens if a < t < b))] += 1
    return dict(yields=len(sel), classes=dict(sorted(cls.items())), follow_on=follow,
                follow_on_after_one_owner_chunk=between.get('1', 0),
                owner_chunks_between=dict(sorted(between.items(), key=lambda kv: int(kv[0]))),
                appends_per_window=dict(sorted(collections.Counter(str(n) for n in windows).items())),
                in_window_gap_ms=None)


def turn_join(arm_path, events, origin, window):
    """Per n3 turn: TTFT (client), pre-open (sent -> first native chunk_open of the admitted request)."""
    turns = []
    try:
        lines = arm_path.read_text().splitlines()
    except OSError:
        return None
    for line in lines[1:]:
        try:
            t = json.loads(line)
        except ValueError:
            continue
        if type(t) is dict and 't' in t and 'kind' in t:
            turns.append(t)
    admitted = collections.defaultdict(list)
    for at, kind, doc in events:
        if kind == 'request_admitted':
            admitted[doc.get('prompt_tokens')].append((at, doc.get('request_key')))
    first_open = {}
    for at, kind, doc in events:
        if kind == 'chunk_open':
            first_open.setdefault(doc.get('request_key'), []).append(at)
    used = set()
    out = dict(append_preopen_agent_s=0.0, fresh_preopen_agent_s=0.0, unplaced=0, stock=0)
    ttft = collections.defaultdict(list)
    for t in turns:
        sent = origin + t['t']
        if type(t.get('ttft')) in (int, float):
            ttft[t['kind']].append(t['ttft'])
        cands = [(at, key) for at, key in admitted.get(t.get('prompt_tokens'), ())
                 if sent - JOIN_SLACK_S <= at and (at, key) not in used]
        if not cands:
            out['unplaced'] += 1
            continue
        at, key = min(cands)
        used.add((at, key))
        opens = [o for o in first_open.get(key, ()) if o >= at]
        if not opens:
            out['stock'] += 1
            continue
        clip = max(0.0, min(opens[0], origin + window) - min(sent, origin + window))
        out[('append' if t['kind'] == 'append' else 'fresh') + '_preopen_agent_s'] += clip
    out['append_ttft_mean_s'] = round(sum(ttft['append']) / len(ttft['append']), 3) if ttft['append'] else None
    out['fresh_ttft_max_s'] = round(max(ttft['fresh']), 3) if ttft['fresh'] else None
    out['turns'] = len(turns)
    for k in ('append_preopen_agent_s', 'fresh_preopen_agent_s'):
        out[k] = round(out[k], 2)
    return out


def owner_block(row):
    queue = (((row or {}).get('native_admission') or {}).get('queue_policy') or {})
    return queue.get('no_append_owner')


def collect(root, variant):
    root = Path(root)
    result = yrs.collect(root, 'off')
    result['variant'] = variant
    want = VARIANTS[variant]
    result['identity_yield_rules'] = {Y1_ENV: identity_env(root, Y1_ENV), Y2_ENV: identity_env(root, Y2_ENV)}
    for name, value in zip((Y1_ENV, Y2_ENV), want):
        got = result['identity_yield_rules'][name]
        if any((v if v != 'unset' else '0') != value for v in got.values()):
            result['notes'].append('identity-%s-%s-start-%s-end-%s' % (name, variant, got['start'], got['end']))
    log = root / 'engine.log'
    if not log.is_file():
        log = root / 'engine-stream.log'
    events, receipts = read_log(log) if log.is_file() else ([], [])
    table = owners(events)
    for label in N3:
        arm = hgb.load(root / ('arm-' + label + '.json'))
        if arm is None or label not in result['arms']:
            continue
        start, end = arm.get('replay_start_s'), arm.get('replay_end_s')
        rows = hgb.status_rows(arm)
        config = (((rows.get('after') or {}).get('append_yield') or {}).get('config') or {})
        max_new = config.get('max_new') if type(config.get('max_new')) is int else DEFAULT_MAX_NEW
        slot = result['arms'][label]
        if type(start) in (int, float) and type(end) in (int, float):
            slot['yield_rules'] = yields(events, table, start, end, max_new)
            slot['yield_rules']['max_new'] = max_new
            slot['y1_receipts'] = sum(1 for at, _ in receipts if start <= at <= end)
            cfg = arm.get('replay_config') or {}
            slot['turn_join'] = turn_join(root / ('arm-' + label + '.json.turns.jsonl'), events, start,
                                          cfg.get('DURATION', 480.0))
        before, after = owner_block(rows.get('before')), owner_block(rows.get('after'))
        slot['no_append_owner'] = (None if type(before) is not dict or type(after) is not dict else
                                   {k: hgb._sub(after.get(k), before.get(k)) for k in after})
        if want[0] == '1' and slot['no_append_owner'] is None:
            result['notes'].append('y1-record-lacks-no_append_owner:' + label)
        if want[0] == '0' and (after is not None or slot.get('y1_receipts')):
            result['notes'].append('off-record-carries-y1:' + label)
    return result


def render(results):
    lines = [yrs.render(results), '-- yield rules']
    for result in results:
        lines.append('== %s  %s' % (result['variant'], json.dumps(result['identity_yield_rules'], sort_keys=True)))
        for label in N3:
            arm = result['arms'].get(label) or {}
            y = arm.get('yield_rules')
            if y is None:
                continue
            lines.append('  %-13s yields=%s classes=%s follow_on=%s after_one_owner_chunk=%s appends/window=%s y1_receipts=%s '
                         'y1_block=%s' % (label, y['yields'], json.dumps(y['classes'], sort_keys=True), y['follow_on'],
                                          y['follow_on_after_one_owner_chunk'], json.dumps(y['appends_per_window']),
                                          arm.get('y1_receipts'), json.dumps(arm.get('no_append_owner'), sort_keys=True)))
            j = arm.get('turn_join') or {}
            lines.append('      turns=%s append_preopen_agent_s=%s fresh_preopen_agent_s=%s append_ttft_mean=%ss '
                         'fresh_ttft_max=%ss unplaced=%s stock=%s' % (
                             j.get('turns'), j.get('append_preopen_agent_s'), j.get('fresh_preopen_agent_s'),
                             j.get('append_ttft_mean_s'), j.get('fresh_ttft_max_s'), j.get('unplaced'), j.get('stock')))
        pooled = collections.Counter()
        for label in N3:
            y = (result['arms'].get(label) or {}).get('yield_rules') or {}
            pooled['yields'] += y.get('yields', 0)
            pooled['follow_on'] += y.get('follow_on', 0)
            pooled['after_one'] += y.get('follow_on_after_one_owner_chunk', 0)
            for k, v in (y.get('classes') or {}).items():
                pooled[k] += v
        lines.append('  pooled %s' % json.dumps(dict(sorted(pooled.items()))))
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('records', nargs='+', help='off=<dir>, y1=<dir> and/or y12=<dir>')
    parser.add_argument('--json-out', type=Path)
    args = parser.parse_args(argv)
    results = []
    for item in args.records:
        variant, sep, path = item.partition('=')
        if not sep or variant not in VARIANTS:
            parser.error('records are off=<dir>, y1=<dir> or y12=<dir>')
        results.append(collect(Path(path), variant))
    print(render(results))
    print('EXPLORATORY no rule, no thresholds, no verdict')
    if args.json_out:
        with args.json_out.open('x') as handle:
            json.dump(dict(schema='q2024p-j2-yr-read/1', exploratory=True, results=results), handle, indent=2, sort_keys=True)
            handle.write('\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
