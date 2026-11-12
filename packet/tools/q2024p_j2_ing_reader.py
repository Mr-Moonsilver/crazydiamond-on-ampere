#!/usr/bin/env python3
"""Exploratory reader for the q2024p-j2 ingress packets (ing off = the yrs-off record / ing-on). Read-only; no rule, no
thresholds, no verdict.

Everything the yr reader prints (tools/q2024p_j2_yr_reader.py, reused unchanged and read as 'off': the ing packet keeps
the batch, the settle and both yield rules at 0), and per arm inside its replay window:
  * sent -> admit per turn class (the arm's turn journal joined to the engine's request_admitted by prompt tokens and
    time, the yr reader's join): count, mean, p50, p90 and by prompt size (<80k, 80-100k, 100-120k, >=120k), for the n3
    arms and the stock sp arms (sp is the fixed reference: the splice never runs there);
  * the engine's append common-prefix counter (native_admission.ingress_prefix, log-only, recorded on engine arms only):
    the arm's delta of compared / first / through_reply / through_reply_but_last / block_resolved, the histogram of the
    match end against the previous prompt end, and the mean lcp / previous prompt / previous output;
  * the API splice counters (native_admission.ingress_api, flag 1 only): spliced, full, verified, verify mismatches,
    fallbacks by reason, splice and full-call ms histograms and means, characters re-tokenised, binding flips.
Per record: FLASH_S5_INGRESS_INCREMENTAL in the identity env. NOTEs (never verdicts): a record whose identity differs from
its arm, an on record lacking the API block on an n3 arm, an off record carrying it, an sp arm with any splice.

Usage:
  python3 -I -B tools/q2024p_j2_ing_reader.py off=<FETCH>/q18a-parity-n3s24q2024p-j2-yrs-off \
      on=<FETCH>/q18a-parity-n3s24q2024p-j2-ing-on [--json-out <fresh.json>]
"""
import argparse
import collections
import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location('q2024p_j2_yr_reader', Path(__file__).resolve().parent / 'q2024p_j2_yr_reader.py')
yr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(yr)
hgb = yr.hgb

ING_ENV = 'FLASH_S5_INGRESS_INCREMENTAL'
VARIANTS = {'off': '0', 'on': '1'}
ARMS = hgb.ARMS
SIZE_BINS = ((80000, '<80k'), (100000, '80-100k'), (120000, '100-120k'), (None, '>=120k'))


def size_bin(n):
    for bound, name in SIZE_BINS:
        if bound is None or n < bound:
            return name


def stats(xs):
    if not xs:
        return dict(n=0)
    xs = sorted(xs)
    return dict(n=len(xs), mean=round(sum(xs) / len(xs), 3), p50=round(xs[len(xs) // 2], 3),
                p90=round(xs[min(len(xs) - 1, (9 * len(xs)) // 10)], 3))


def sent_admit(arm_path, events, origin):
    """Per turn class and prompt size: admitted - sent (s). The yr reader's join (prompt tokens, first admit after send)."""
    try:
        lines = arm_path.read_text().splitlines()
    except OSError:
        return None
    turns = []
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
    used, by_kind, by_size, unplaced = set(), collections.defaultdict(list), collections.defaultdict(list), 0
    for t in turns:
        sent = origin + t['t']
        cands = [(at, key) for at, key in admitted.get(t.get('prompt_tokens'), ())
                 if sent - yr.JOIN_SLACK_S <= at and (at, key) not in used]
        if not cands:
            unplaced += 1
            continue
        at, key = min(cands)
        used.add((at, key))
        kind = 'append' if t['kind'] == 'append' else 'fresh'
        by_kind[kind].append(at - sent)
        if type(t.get('prompt_tokens')) is int:
            by_size[kind + ' ' + size_bin(t['prompt_tokens'])].append(at - sent)
    return dict(turns=len(turns), unplaced=unplaced, by_kind={k: stats(v) for k, v in sorted(by_kind.items())},
                by_size={k: stats(v) for k, v in sorted(by_size.items())})


def block(row, name):
    return ((row or {}).get('native_admission') or {}).get(name)


def counter_delta(before, after, keys):
    if type(after) is not dict:
        return None
    before = before if type(before) is dict else {}
    out = {}
    for k in keys:
        a, b = after.get(k) or {}, before.get(k) or {}
        out[k] = {n: a.get(n, 0) - b.get(n, 0) for n in sorted(set(a) | set(b)) if a.get(n, 0) - b.get(n, 0)}
    return out


def prefix_delta(before, after):
    d = counter_delta(before, after, ('counts', 'delta_vs_prev_prompt_end', 'sums'))
    if d is None:
        return None
    n = d['counts'].get('compared', 0)
    d['means'] = {k: round(d['sums'].get(k, 0) / n, 1) for k in ('lcp', 'prev_prompt', 'prev_output', 'new_prompt')} if n else {}
    d['unit'] = after.get('unit')
    return d


def api_delta(before, after):
    d = counter_delta(before, after, ('counts', 'fallbacks', 'sums', 'splice_ms', 'full_ms'))
    if d is None:
        return None
    s, c = d['sums'], d['counts']
    d['splice_ms_mean'] = round(s.get('splice_us', 0) / c['spliced'] / 1000, 2) if c.get('spliced') else None
    d['full_ms_mean'] = round(s.get('full_us', 0) / c['full'] / 1000, 2) if c.get('full') else None
    d['retokenised_share'] = (round(s.get('chars_retokenised', 0) / s['chars_total'], 4) if s.get('chars_total') else None)
    d['binding'] = after.get('binding')
    d['disabled'] = after.get('disabled')
    d['entries'], d['bytes'] = after.get('entries'), after.get('bytes')
    return d


def collect(root, variant):
    root = Path(root)
    result = yr.collect(root, 'off')
    result['variant'] = variant
    result['identity_ingress'] = got = yr.identity_env(root, ING_ENV)
    if any((v if v != 'unset' else '0') != VARIANTS[variant] for v in got.values()):
        result['notes'].append('identity-%s-%s-start-%s-end-%s' % (ING_ENV, variant, got['start'], got['end']))
    log = root / 'engine.log'
    if not log.is_file():
        log = root / 'engine-stream.log'
    events, _ = yr.read_log(log) if log.is_file() else ([], [])
    for label in ARMS:
        arm = hgb.load(root / ('arm-' + label + '.json'))
        if arm is None:
            continue
        slot = result['arms'].setdefault(label, {})
        rows = hgb.status_rows(arm)
        before, after = rows.get('before'), rows.get('after')
        start = arm.get('replay_start_s')
        if type(start) in (int, float):
            slot['sent_admit'] = sent_admit(root / ('arm-' + label + '.json.turns.jsonl'), events, start)
        slot['ingress_prefix'] = prefix_delta(block(before, 'ingress_prefix'), block(after, 'ingress_prefix'))
        slot['ingress_api'] = api_delta(block(before, 'ingress_api'), block(after, 'ingress_api'))
        api = slot['ingress_api']
        if variant == 'on' and label.startswith('n3') and api is None:
            result['notes'].append('on-record-lacks-ingress_api:' + label)
        if variant == 'off' and block(after, 'ingress_api') is not None:
            result['notes'].append('off-record-carries-ingress_api:' + label)
        if label.startswith('sp') and api is not None and api['counts'].get('spliced'):
            result['notes'].append('sp-arm-spliced:%s:%d' % (label, api['counts']['spliced']))
        if label.startswith('sp') and slot['ingress_prefix'] and slot['ingress_prefix']['counts']:
            result['notes'].append('sp-arm-prefix-recorded:' + label)
    return result


def fmt(d):
    return json.dumps(d, sort_keys=True)


def render(results):
    lines = [yr.render(results), '-- ingress']
    for result in results:
        lines.append('== %s  %s=%s' % (result['variant'], ING_ENV, fmt(result['identity_ingress'])))
        for label in ARMS:
            arm = result['arms'].get(label) or {}
            sa = arm.get('sent_admit')
            if sa is None and arm.get('ingress_prefix') is None and arm.get('ingress_api') is None:
                continue
            if sa is not None:
                lines.append('  %-13s sent->admit s: %s unplaced=%s/%s' % (label, fmt(sa['by_kind']), sa['unplaced'], sa['turns']))
                lines.append('      by size: %s' % fmt(sa['by_size']))
            p = arm.get('ingress_prefix')
            if p is not None:
                lines.append('      prefix: counts=%s match_end_vs_prev_prompt_end=%s means=%s unit=%s' % (
                    fmt(p['counts']), fmt(p['delta_vs_prev_prompt_end']), fmt(p['means']), p['unit']))
            a = arm.get('ingress_api')
            if a is not None:
                lines.append('      api: counts=%s fallbacks=%s splice_ms_mean=%s full_ms_mean=%s retokenised_share=%s '
                             'splice_ms=%s full_ms=%s binding=%s disabled=%s' % (
                                 fmt(a['counts']), fmt(a['fallbacks']), a['splice_ms_mean'], a['full_ms_mean'],
                                 a['retokenised_share'], fmt(a['splice_ms']), fmt(a['full_ms']), fmt(a['binding']),
                                 a['disabled']))
        if result['notes']:
            lines.append('  NOTES %s' % fmt(sorted(result['notes'])))
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('records', nargs='+', help='off=<dir> and/or on=<dir>')
    parser.add_argument('--json-out', type=Path)
    args = parser.parse_args(argv)
    results = []
    for item in args.records:
        variant, sep, path = item.partition('=')
        if not sep or variant not in VARIANTS:
            parser.error('records are off=<dir> or on=<dir>')
        results.append(collect(Path(path), variant))
    print(render(results))
    print('EXPLORATORY no rule, no thresholds, no verdict')
    if args.json_out:
        with args.json_out.open('x') as handle:
            json.dump(dict(schema='q2024p-j2-ing-read/1', exploratory=True, results=results), handle, indent=2, sort_keys=True)
            handle.write('\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
