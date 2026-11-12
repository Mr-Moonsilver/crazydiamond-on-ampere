#!/usr/bin/env python3
"""Exploratory reader for the q2024p-j2 night-pack family (np-off, np-y12, np-ing, np-all, np-set, np-hgb and their -n2
repeats; lead 12 Nov 2026). Read-only; no rule, no thresholds, no verdict.

Every packet of the family is built from one commit and differs only in launch flags, so np-off is the exact baseline of
each lever packet. A record is given as <variant>[-<tag>]=<dir>, the variant naming its flags (FAMILY below; e.g.
off-n2=..., y12=..., off-v1b1=<a V1 j2 record>). The chained readers (tools/q2024p_j2_ing_reader.py, through it the yr,
yrs and hgb readers) are told each record's flags, so their identity checks and NOTEs read against that variant.

Per arm, in this order: n3 absolute turns/h, then x sp; native share; agent-seconds over the arm's completed turns
(TTFT-s, decode-s = seconds - ttft, output tokens). Then each lever's counters (status-block deltas inside the arm,
log lines inside its replay window):
  * re-entry: refusals by stage:field, the retry histogram, retries until armed, exceeded, barrier pass / first try,
    the settle counters, armed ms; the proof-miss lines' append_dec / append_out / running (night-pack fix 6);
  * yields: by owner class, follow-ons, appends per window, window ends, not-served split (fix 5), next-left, gap ms by
    path (o3 | full) from the status block and the append receipts, tail FULL handbacks, re-entries avoided, Y1;
  * ingress: sent -> admit by turn class and its split at the API arrival stamp (sent -> arrival, arrival -> admit;
    fix 3), splice / full / verify counts and mismatches, disabled, the async submit -> done against the executor
    run by class (fix 3), the prefix histogram, the reply-aligned match and offsets, the divergence windows (fix 4);
  * batch: the small_piece_batch block (sent, second applied, partial, failures, worker stops).
NOTEs (never verdicts): a record whose identity env differs from its variant's flags (all five flags and, with ingress
on, the verify settings), the chained readers' NOTEs.

Usage:
  python3 -I -B tools/q2024p_j2_np_reader.py off=<FETCH>/q18a-parity-n3s24q2024p-j2-np-off \\
      y12=<FETCH>/...-np-y12 ing=<FETCH>/...-np-ing all=<FETCH>/...-np-all [--full] [--json-out <fresh.json>]
"""
import argparse
import collections
import importlib.util
import json
from pathlib import Path
import re

_spec = importlib.util.spec_from_file_location('q2024p_j2_ing_reader', Path(__file__).resolve().parent / 'q2024p_j2_ing_reader.py')
ing = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ing)
yr = ing.yr
yrs = yr.yrs
hgb = ing.hgb

BATCH_ENV, SETTLE_ENV = 'FLASH_S5_SMALL_PIECE_BATCH', 'FLASH_S5_YIELD_REENTRY_SETTLE'
VERIFY_EVERY_ENV, VERIFY_ASYNC_ENV = 'FLASH_S5_INGRESS_VERIFY_EVERY', 'FLASH_S5_INGRESS_VERIFY_ASYNC'
ENVS = (('batch', BATCH_ENV), ('settle', SETTLE_ENV), ('y1', yr.Y1_ENV), ('y2', yr.Y2_ENV), ('ingress', ing.ING_ENV))
_OFF = dict(batch='0', settle='0', y1='0', y2='0', ingress='0')
FAMILY = {
    'off': dict(_OFF),
    'y12': dict(_OFF, y1='1', y2='1'),
    'ing': dict(_OFF, ingress='1'),
    'all': dict(_OFF, y1='1', y2='1', ingress='1'),
    'set': dict(_OFF, settle='1'),
    'hgb': dict(_OFF, batch='1'),
}
VERIFY = {VERIFY_EVERY_ENV: '1', VERIFY_ASYNC_ENV: '1'}          # with ingress on: every splice, off-path
ARMS = hgb.ARMS
MISS_FIELDS = re.compile(r'flash-s5 yield-reentry proof-miss .*?\bappend_dec=(\S+) append_out=(\S+) running=(\S+)')
ADMIT = 'flash-s5 native-gap '


def variant_of(label):
    head = label.split('-', 1)[0]
    if head not in FAMILY:
        raise ValueError('unknown variant %r (one of %s)' % (head, ', '.join(sorted(FAMILY))))
    return head


def chained(root, flags):
    """ing.collect with every layer's expected flags set to this record's (each layer looks VARIANTS up at call time)."""
    saved = (dict(yrs.VARIANTS), dict(yr.VARIANTS), dict(ing.VARIANTS))
    try:
        yrs.VARIANTS['off'] = (flags['batch'], flags['settle'])
        yr.VARIANTS['off'] = (flags['y1'], flags['y2'])
        ing.VARIANTS['off'] = '0'
        ing.VARIANTS['on'] = '1'
        return ing.collect(root, 'on' if flags['ingress'] == '1' else 'off')
    finally:
        for table, old in zip((yrs.VARIANTS, yr.VARIANTS, ing.VARIANTS), saved):
            table.clear()
            table.update(old)


def agent_seconds(path):
    """Sums over the arm's completed turns: TTFT-s, decode-s (seconds - ttft), output tokens, by turn class."""
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return None
    out = collections.defaultdict(lambda: dict(n=0, ttft_s=0.0, decode_s=0.0, output_tokens=0))
    for line in lines[1:]:
        try:
            t = json.loads(line)
        except ValueError:
            continue
        if type(t) is not dict or type(t.get('ttft')) not in (int, float) or type(t.get('seconds')) not in (int, float):
            continue
        for key in ('all', 'append' if t.get('kind') == 'append' else 'fresh'):
            row = out[key]
            row['n'] += 1
            row['ttft_s'] += t['ttft']
            row['decode_s'] += max(0.0, t['seconds'] - t['ttft'])
            row['output_tokens'] += int(t.get('completion_tokens') or 0)
    return {k: dict(v, ttft_s=round(v['ttft_s'], 2), decode_s=round(v['decode_s'], 2)) for k, v in sorted(out.items())}


def scan(log, windows):
    """One pass: request_admitted rows with the arrival stamps, proof-miss fields, Y2 append receipts by path."""
    admitted, misses, receipts = [], {k: [] for k in windows}, {k: [] for k in windows}

    def labels_of(at):          # arms' replay windows do not overlap on a box record; every containing window counts
        return [label for label, (start, end) in windows.items() if start <= at <= end]
    try:
        handle = Path(log).open(errors='replace')
    except OSError:
        return admitted, misses, receipts
    with handle:
        for line in handle:
            if 'flash-s5 ' not in line:
                continue
            try:
                at = hgb.stamp(line.split(' ', 1)[0])
            except ValueError:
                continue
            if ADMIT in line and '"request_admitted"' in line:
                try:
                    doc = json.loads(line.split(ADMIT, 1)[1])
                except ValueError:
                    continue
                admitted.append((at, doc))
                continue
            labels = labels_of(at)
            if not labels:
                continue
            m = MISS_FIELDS.search(line)
            if m:
                for label in labels:
                    misses[label].append(m.groups())
            elif yr.WINDOW_RECEIPT in line:
                try:
                    doc = json.loads(line.split(yr.WINDOW_RECEIPT, 1)[1])
                except ValueError:
                    continue
                if doc.get('event') == 'append':
                    for label in labels:
                        receipts[label].append(doc)
    return admitted, misses, receipts


def admit_split(turns_path, admitted, origin):
    """sent -> admit split at the API arrival stamp (request_admitted arrival_s / admit_wall_s, wall clock), by class;
    the ing reader's join (prompt tokens, first admission after send). None-valued when the record has no stamps."""
    try:
        lines = Path(turns_path).read_text().splitlines()
    except OSError:
        return None
    by_tokens = collections.defaultdict(list)
    for at, doc in admitted:
        by_tokens[doc.get('prompt_tokens')].append((at, doc))
    used = set()
    parts = collections.defaultdict(lambda: collections.defaultdict(list))
    stamped = 0
    for line in lines[1:]:
        try:
            t = json.loads(line)
        except ValueError:
            continue
        if type(t) is not dict or 't' not in t:
            continue
        sent = origin + t['t']
        cands = [(at, i) for i, (at, doc) in enumerate(by_tokens.get(t.get('prompt_tokens'), ()))
                 if sent - yr.JOIN_SLACK_S <= at and (t.get('prompt_tokens'), i) not in used]
        if not cands:
            continue
        at, i = min(cands)
        used.add((t.get('prompt_tokens'), i))
        doc = by_tokens[t.get('prompt_tokens')][i][1]
        kind = 'append' if t.get('kind') == 'append' else 'fresh'
        arrival, wall = doc.get('arrival_s'), doc.get('admit_wall_s')
        if type(arrival) in (int, float) and type(wall) in (int, float):
            stamped += 1
            parts[kind]['sent_to_arrival'].append(arrival - sent)
            parts[kind]['arrival_to_admit'].append(wall - arrival)
    return dict(stamped=stamped, by_kind={k: {p: ing.stats(v) for p, v in sorted(d.items())} for k, d in sorted(parts.items())})


def _num_delta(after, before):
    if type(after) is not dict:
        return None
    before = before if type(before) is dict else {}
    out = {}
    for k, v in after.items():
        if type(v) in (int, float) and type(v) is not bool:
            d = v - (before.get(k) or 0)
            if d:
                out[k] = round(d, 3)
    return out


def ingress_extra(rows):
    """The fix 3 / fix 4 parts of the API and prefix blocks: verify and timing deltas; reply-aligned deltas; the
    lifetime divergence windows (the engine's first appends) and settings."""
    b, a = (ing.block(rows.get('before'), 'ingress_api'), ing.block(rows.get('after'), 'ingress_api'))
    out = {}
    if type(a) is dict:
        v_a, v_b = a.get('verify') or {}, (b or {}).get('verify') or {}
        out['api'] = dict(verify=_num_delta(v_a, v_b), verify_ms=_num_delta(v_a.get('ms'), v_b.get('ms')),
                          mismatches=v_a.get('mismatches'), pending=v_a.get('pending'),
                          verify_every=a.get('verify_every'), verify_async=a.get('verify_async'),
                          timing=_num_delta(a.get('timing'), (b or {}).get('timing')), disabled=a.get('disabled'))
        t = out['api']['timing'] or {}
        out['api']['queue_pickup_ms_mean'] = {
            cls: round((t.get('async_%s_us' % cls, 0) - t.get('run_%s_us' % cls, 0)) / t['async_%s_n' % cls] / 1000, 2)
            for cls in ('n3', 'stock') if t.get('async_%s_n' % cls)}
        out['api']['run_ms_mean'] = {cls: round(t.get('run_%s_us' % cls, 0) / t['run_%s_n' % cls] / 1000, 2)
                                     for cls in ('n3', 'stock') if t.get('run_%s_n' % cls)}
    pb, pa = (ing.block(rows.get('before'), 'ingress_prefix'), ing.block(rows.get('after'), 'ingress_prefix'))
    if type(pa) is dict:
        d = ing.counter_delta(pb, pa, ('reply_aligned', 'reply_offset', 'counts'))
        out['prefix'] = dict(reply_aligned=d['reply_aligned'], reply_offset=d['reply_offset'],
                             after_eviction=d['counts'].get('after_eviction', 0), max_entries=pa.get('max_entries'),
                             divergence_windows=pa.get('divergence_windows'))
    return out


def window_extra(rows, receipts):
    b, a = yr.owner_block(rows.get('before'), 'yield_window'), yr.owner_block(rows.get('after'), 'yield_window')
    out = dict(receipts_by_path={})
    by_path = collections.defaultdict(list)
    for doc in receipts:
        if type(doc.get('gap_ms')) in (int, float):
            by_path[str(doc.get('path'))].append(doc['gap_ms'])
    out['receipts_by_path'] = {p: dict(n=len(v), mean=round(sum(v) / len(v), 1), max=round(max(v), 1))
                               for p, v in sorted(by_path.items())}
    if type(a) is dict:
        out['not_served'] = {k: v - ((b or {}).get('not_served') or {}).get(k, 0)
                             for k, v in (a.get('not_served') or {}).items() if v - ((b or {}).get('not_served') or {}).get(k, 0)}
        out['next_left'] = hgb._sub(a.get('next_left'), (b or {}).get('next_left'))
        gp = {}
        for p, v in (a.get('gap_by_path') or {}).items():
            w = ((b or {}).get('gap_by_path') or {}).get(p) or {}
            n, total = v.get('n', 0) - w.get('n', 0), v.get('total', 0.0) - w.get('total', 0.0)
            if n:
                gp[p] = dict(n=n, mean=round(total / n, 1), max_lifetime=v.get('max'))
        out['gap_by_path'] = gp
    return out


def miss_extra(rows):
    if not rows:
        return dict(n=0)
    out = dict(n=len(rows))
    for i, name in enumerate(('append_dec', 'append_out', 'running')):
        out[name] = dict(sorted(collections.Counter(r[i] for r in rows).items()))
    return out


def identity_notes(root, label, flags):
    notes, got = [], {}
    names = [env for _, env in ENVS] + (list(VERIFY) if flags['ingress'] == '1' else [])
    want = dict({env: flags[key] for key, env in ENVS}, **(VERIFY if flags['ingress'] == '1' else {}))
    for env in names:
        got[env] = yr.identity_env(root, env)
        for phase, value in got[env].items():
            if (value if value != 'unset' or env in VERIFY else '0') != want[env]:     # the flags: unset reads as 0
                notes.append('identity-%s-%s-%s-%s' % (label, env, phase, value))
    return got, notes


def collect(root, label):
    root = Path(root)
    flags = FAMILY[variant_of(label)]
    result = chained(root, flags)
    result['label'], result['variant'], result['flags'] = label, variant_of(label), dict(flags)
    result['identity_family'], notes = identity_notes(root, label, flags)
    result['family_notes'] = notes
    log = root / 'engine.log'
    if not log.is_file():
        log = root / 'engine-stream.log'
    windows, arms = {}, {}
    for name in ARMS:
        arm = hgb.load(root / ('arm-' + name + '.json'))
        if arm is None or name not in result['arms']:
            continue
        arms[name] = arm
        start, end = arm.get('replay_start_s'), arm.get('replay_end_s')
        if type(start) in (int, float) and type(end) in (int, float):
            windows[name] = (start, end)
    admitted, misses, receipts = scan(log, windows) if log.is_file() else ([], {}, {})
    for name, arm in arms.items():
        slot = result['arms'][name]
        rows = hgb.status_rows(arm)
        turns = root / ('arm-' + name + '.json.turns.jsonl')
        slot['agent_seconds'] = agent_seconds(turns)
        if type(arm.get('replay_start_s')) in (int, float):
            slot['admit_split'] = admit_split(turns, admitted, arm['replay_start_s'])
        slot['ingress_extra'] = ingress_extra(rows)
        slot['window_extra'] = window_extra(rows, receipts.get(name, ()))
        slot['proof_miss_fields'] = miss_extra(misses.get(name, ()))
    return result


def fmt(d):
    return json.dumps(d, sort_keys=True, separators=(',', ':'))


def _agent(a):
    if not a or 'all' not in a:
        return 'agent_s=-'
    x = a['all']
    return 'agent_s: ttft=%.1f decode=%.1f out_tok=%d (n=%d)' % (x['ttft_s'], x['decode_s'], x['output_tokens'], x['n'])


def render(results, full=False):
    lines = []
    for r in results:
        lines.append('== %s  %s  flags=%s grade=%s/%s rc=%s x_sp(record)=%s numerics=%s oom_max=%s fatal=%s' % (
            r['label'], r['record'], fmt(r['flags']), r.get('queue_grade'), r.get('compose5_grade'), r.get('queue_rc'),
            r.get('x_sp'), r.get('numerics'), (r.get('engine') or {}).get('oom_max'), (r.get('engine') or {}).get('fatal_lines')))
        for name in ARMS:
            a = r['arms'].get(name)
            if a is None:
                continue
            native = a.get('native_share') or {}
            lines.append('  %-13s turns/h=%-7s x_sp=%-7s native=%s/%s %s' % (
                name, a.get('turns_per_hour'), a.get('x_sp') or '-', native.get('live_native_share_strict', '-'),
                native.get('all_native_share_strict', '-'), _agent(a.get('agent_seconds'))))
            sa, split = a.get('sent_admit'), a.get('admit_split')
            if sa:
                lines.append('      sent->admit s %s' % fmt({k: (v['n'], v['mean']) for k, v in sa['by_kind'].items()}))
            if split and split['stamped']:
                lines.append('      split stamped=%d %s' % (split['stamped'], fmt(
                    {k: {p: s['mean'] for p, s in d.items()} for k, d in split['by_kind'].items()})))
            if not name.startswith('n3'):
                continue
            ye = a.get('yield_reentry') or {}
            lines.append('      reentry refused=%s by_retry=%s until_armed=%s exceeded=%s pass/first=%s/%s settle=%s '
                         'armed_ms=%s miss_fields=%s' % (
                             fmt(ye.get('refused')), fmt(ye.get('refused_retry')), fmt(ye.get('retries_until_armed')),
                             ye.get('exceeded'), ye.get('barrier_pass'), ye.get('barrier_first_try'),
                             fmt({k: ye.get(k) for k in ('settle_waits', 'settle_ticks', 'settled', 'settle_bound')}),
                             fmt(ye.get('armed_ms')), fmt(a.get('proof_miss_fields'))))
            fh = a.get('full_handoff') or {}
            if fh:
                lines.append('      full_handoff %s' % fmt({k: {f: v.get(f) for f in ('n', 'mean_ms', 'queued_agent_s')}
                                                            for k, v in fh.items() if type(v) is dict}))
            rules, win, we = a.get('yield_rules') or {}, a.get('yield_window'), a.get('window_extra') or {}
            lines.append('      yields=%s classes=%s follow_on=%s appends/window=%s ends=%s y1=%s served=%s' % (
                rules.get('yields'), fmt(rules.get('classes')), rules.get('follow_on'),
                fmt(rules.get('appends_per_window')), fmt(rules.get('window_ends')), fmt(a.get('no_append_owner')),
                a.get('handbacks_served')))
            if win is not None or we.get('receipts_by_path'):
                lines.append('      window per_window=%s ends=%s not_served=%s next_left=%s gap_by_path=%s '
                             'receipt_gap_by_path=%s tail_full=%s avoided=%s' % (
                                 fmt((win or {}).get('per_window')), fmt((win or {}).get('ends')), fmt(we.get('not_served')),
                                 we.get('next_left'), fmt(we.get('gap_by_path')), fmt(we.get('receipts_by_path')),
                                 (win or {}).get('tail_full_handbacks'), (win or {}).get('reentries_avoided')))
            api, ix = a.get('ingress_api'), a.get('ingress_extra') or {}
            if api is not None:
                ex = ix.get('api') or {}
                lines.append('      api counts=%s fallbacks=%s disabled=%s splice_ms=%s full_ms=%s verify=%s mismatches=%s '
                             'run_ms=%s queue+pickup_ms=%s' % (
                                 fmt(api['counts']), fmt(api['fallbacks']), api['disabled'], api['splice_ms_mean'],
                                 api['full_ms_mean'], fmt(ex.get('verify')), fmt(ex.get('mismatches')),
                                 fmt(ex.get('run_ms_mean')), fmt(ex.get('queue_pickup_ms_mean'))))
            p = a.get('ingress_prefix')
            if p is not None:
                px = ix.get('prefix') or {}
                lines.append('      prefix counts=%s vs_prev_end=%s reply_aligned=%s offsets=%s after_eviction=%s windows=%d' % (
                    fmt(p['counts']), fmt(p['delta_vs_prev_prompt_end']), fmt(px.get('reply_aligned')),
                    fmt(px.get('reply_offset')), px.get('after_eviction'), len(px.get('divergence_windows') or ())))
                for w in (px.get('divergence_windows') or ())[:2]:
                    lines.append('        window k=%s d=%s exact=%s new=%s old=%s' % (
                        w.get('k'), w.get('d'), w.get('exact'), fmt(w.get('new')), fmt(w.get('old'))))
            b = a.get('small_piece_batch')
            if b is not None:
                lines.append('      batch %s' % fmt({k: b.get(k) for k in ('sent', 'second_applied', 'second_not_run',
                                                                          'partial', 'failures', 'worker_stop')}))
        notes = sorted(set(r.get('notes') or ()) | set(r.get('family_notes') or ()))
        if notes:
            lines.append('  NOTES %s' % fmt(notes))
    if full:
        lines += ['-- chained readers', ing.render(results)]
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('records', nargs='+', help='<variant>[-<tag>]=<dir>, variant one of ' + ', '.join(sorted(FAMILY)))
    parser.add_argument('--full', action='store_true', help='also print the chained readers (ing, yr, yrs, hgb)')
    parser.add_argument('--json-out', type=Path)
    args = parser.parse_args(argv)
    results = []
    for item in args.records:
        label, sep, path = item.partition('=')
        try:
            variant_of(label)
        except ValueError as exc:
            parser.error(str(exc))
        if not sep:
            parser.error('records are <variant>[-<tag>]=<dir>')
        results.append(collect(Path(path), label))
    print(render(results, full=args.full))
    print('EXPLORATORY no rule, no thresholds, no verdict')
    if args.json_out:
        with args.json_out.open('x') as handle:
            json.dump(dict(schema='q2024p-j2-np-read/1', exploratory=True, results=results), handle, indent=2,
                      sort_keys=True, default=str)
            handle.write('\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
