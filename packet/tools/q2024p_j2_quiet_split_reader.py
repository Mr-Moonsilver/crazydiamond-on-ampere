#!/usr/bin/env python3
"""Cross-arm decode-forward table from the decode-timeline stamp's d rows (brief 13 item 1, 14 Nov 2026).

The busy/far reader (tools/q2024p_j2_busyfar_reader.py) keeps a forward only inside an n3 arm, with an engine chunk on
both sides and a stratum that holds arm, route generation and history band; the quiet-SPLIT control's forwards fail
all three (REVIEW-qsplit2-20261114.md finding 1). This reader takes the d rows directly, as the review's scratch
scripts do (investigate-scratch-20261113/review-qsplit2/tl_budget.py, tl_rows.py), and keeps every refusal counted.

    python3 -I -B tools/q2024p_j2_quiet_split_reader.py --root RECORD_ROOT --out FRESH_DIR
        [--quiet-split PATH] [--reference ROOT ...] [--engine-log PATH]

Rows. The busy/far reader's own receipt reading: torn lines dropped, sessions whole (every part, one sha), a second
different d row for one (rank, seq) a conflict (never resolved). A row is used only with forward_calls 1,
handler_flags 0, a non-empty integer history, bank split|full, an integer num_tokens and actual_rows, a cg_mode and a
positive device_ns; otherwise it is refused by its first failing field.

Classes (per rank; cell = bank, bank_sms, cg_mode, num_tokens, actual_rows):
  Q_full   outside-native-progress rows, bank full, inside an n3 arm: D quiet on the full bank.
  Q_split  outside rows of the control: route_generation equal to a stage's split_generation in quiet-split.json, bank
           split, logged after the last arm's wall end. One cell per stage (band and live count).
  C_split  outside rows, bank split, inside an n3 arm: near native work, NOT certified far (the consistency line).
  B_split  native-scope rows (a native session), bank split, inside an n3 arm: D busy on the split bank.
An n3 arm owns a row when its log stamp minus LOG_LAG_S and the stamp itself are inside the arm's wall window (the
session line is written after its forward); a row across a boundary is refused ('arm_boundary'). A native-scope row
with a control generation is a hard stop ('native_session_in_control'), never a Q_split row.
Per cell: n, D p10/p50/p90 (ms), max-history p50 and its band (the busy/far reader's BANDS), attn-meta state (the arm's
attn_meta.requested and every rank's select.after.selected; the control's quiet-split.json attn_meta block).
TP stage value = max of the two peers' p50 (stage 0 ranks 0,1; stage 1 ranks 2,3), never a sum.
Three numbers per live count (1, 2, 4, 6, 8; at 8 B_split pools actual rows 7 and 8 at num_tokens 8): Q_full, Q_split
per stage, B_split, with S = Q_split - Q_full and N = B_split - Q_split per rank; the consistency check (QUIET_SPLIT_READ
draft): |p50 Q_split - p50 C_split| <= CONSIST_MS on at least 3 of 4 ranks with C_split n >= MIN_REF, C_split pooled
over this record and every --reference record (same rank, num_tokens, actual_rows). Descriptive only: no verdict.
Budget per rank: native sessions and outside samples (highest number seen), log_budget_exhausted sessions, and bytes
as sum(len(payload) + 97) per line (the stamp adds 97 per line for the receipt identity and newline; the logged payload
already holds that identity, so this is an upper estimate), split before and after the control's first row.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys

SCHEMA = 'q2024p-j2-quiet-split-read/1'
HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('q2024p_j2_busyfar_reader_qs', HERE / 'q2024p_j2_busyfar_reader.py')
bf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bf)

LOG_LAG_S = 2.0                   # the session line is written after its forward (busy/far reader's LOG_LAG_S)
LIVE = (1, 2, 4, 6, 8)            # the control's live counts (qsplit3); 1, 4 and 8 are qsplit2's
PADDED = {1: 1, 2: 2, 3: 3, 4: 4, 5: 6, 6: 6, 7: 8, 8: 8}     # actual rows -> the stock graph's num_tokens (ayc2t2)
CONSIST_MS = 0.5
MIN_REF = 5
MIN_CTL = 12
MIN_BUSY, MIN_BUSY8 = 50, 30
STAGES = {0: (0, 1), 1: (2, 3)}
LINE_BYTES = 97                   # decode_timeline._emit: len(line) + 97 per line
MAX_SESSIONS, MAX_OUTSIDE, MAX_BYTES = 256, 1024, 4 * 1024 * 1024
TAG = 'flash-s5 decode-timeline '


def refuse(why):
    raise bf.Refused('REFUSED: ' + why)


def p(values, q):
    v = sorted(values)
    if not v:
        return None
    return v[min(len(v) - 1, int(q * len(v)))] if q != 0.5 else statistics.median(v)


def band_name(value):
    for lo, hi in bf.BANDS:
        if lo <= value < hi:
            return '%d-%s' % (lo, hi if hi < 1 << 40 else 'up')
    return None


def scan(path):
    """Timeline lines only: the busy/far reader's records, plus per-rank bytes and line stamps."""
    kept, sizes = [], []
    for line in bf.lines_of(path):
        at = line.find(TAG)
        if at < 0:
            continue
        kept.append(line)
        payload = line[at + len(TAG):].rstrip('\n')
        try:
            body = json.loads(payload)
            rank = body.get('rank') if type(body) is dict else None
        except ValueError:
            rank = None
        sizes.append((bf.stamp_s(line), rank, len(payload.encode()) + LINE_BYTES))
    records, torn, _ = bf.read_records(kept)
    return records['timeline'], torn['timeline'], sizes


def arms_of(root):
    arms = {}
    for name in sorted(os.listdir(root)):
        if name.startswith('arm-') and name.endswith('.json'):
            with open(os.path.join(root, name)) as handle:
                doc = json.load(handle)
            am = doc.get('attn_meta') if type(doc.get('attn_meta')) is dict else {}
            sel = sorted({str(((r or {}).get('after') or {}).get('selected')) for r in (am.get('select') or {}).values()})
            arms[name[4:-5]] = dict(window=(bf.num(doc.get('wall_start_s')), bf.num(doc.get('wall_end_s'))),
                                    attn_meta=dict(requested=am.get('requested'), selected=sel))
    return arms


def arm_of(stamp, arms):
    """(label | None | 'boundary'): the arm whose window holds [stamp - LOG_LAG_S, stamp]."""
    if stamp is None:
        return None
    lo, hi = stamp - LOG_LAG_S, stamp
    for label, a in arms.items():
        start, end = a['window']
        if start is None or end is None:
            continue
        if start <= lo and hi <= end:
            return label
        if lo <= end and hi >= start:
            return 'boundary'
    return None


def usable(row):
    """None when the row may be used, else the refusal (first failing field)."""
    meta = row.get('metadata')
    if type(meta) is not dict:
        return 'no_metadata'
    if meta.get('forward_calls') != 1:
        return 'forward_calls_not_1'
    if meta.get('handler_flags') != 0:
        return 'handler_flags_not_0'
    history = meta.get('history_tokens')
    if type(history) is not list or not history or any(type(x) is not int for x in history):
        return 'no_history'
    if meta.get('bank') not in ('split', 'full'):
        return 'bank_unknown'
    if type(meta.get('num_tokens')) is not int or type(meta.get('actual_rows')) is not int:
        return 'no_descriptor'
    if meta.get('cg_mode') is None:
        return 'no_cg_mode'
    if bf.num(row.get('device_ns')) is None or row['device_ns'] <= 0:
        return 'no_device_ns'
    return None


def control_of(path):
    if path is None or not os.path.isfile(path):
        return None
    with open(path) as handle:
        doc = json.load(handle)
    stages = []
    for i, s in enumerate(doc.get('stages') or []):
        stages.append(dict(index=i, live=s.get('live'), band=s.get('prompt_tokens'), generation=s.get('split_generation'),
                           counted=bool(s.get('window')) and type(s.get('merge_steps')) is int and s['merge_steps'] >= 1
                           and not s.get('reasons'), reasons=s.get('reasons') or []))
    am = doc.get('attn_meta') if type(doc.get('attn_meta')) is dict else None
    attn = ('on' if am and am.get('selected') is True else 'off' if am and am.get('selected') is False else
            'off (not selected: the control sends no attn_meta_select and inherits the last arm\'s)')
    return dict(path=path, schema=doc.get('schema'), verdict=doc.get('verdict'), cause=doc.get('cause'),
                stages=stages, attn_meta=attn, admitted=(doc.get('native_admitted_before'), doc.get('native_admitted_after')),
                disconnect=[s.get('abort') for s in doc.get('stages') or []])


def classify(root, quiet_split=None, engine_log=None, n3_only=True):
    path = bf.engine_log(root, engine_log)
    records, torn, sizes = scan(path)
    flushes, loss = bf.assemble(records, extra=bf.TIMELINE_KEY, line_gaps=False)
    rows, conflicts, stats = bf.timeline_rows(flushes)
    if not rows:
        refuse('no decode-timeline d rows in %s (FLASH_S5_DECODE_TIMELINE off?)' % path)
    arms = arms_of(root)
    if not any(a.startswith('n3.') for a in arms):
        refuse('no n3 arm file (arm-n3.*.json) in %s' % root)
    last_end = max(a['window'][1] for a in arms.values() if a['window'][1] is not None)
    control = control_of(quiet_split if quiet_split else os.path.join(root, 'quiet-split.json'))
    gens = {s['generation']: s for s in (control or {}).get('stages', []) if type(s['generation']) is int}
    used, refusals, hard = [], {}, []

    def refused(why):
        refusals[why] = refusals.get(why, 0) + 1
    for key in sorted(rows, key=lambda k: (k[0], str(k[1]))):
        r = rows[key]
        if key in conflicts:
            refused('conflicting_d_row')
            continue
        meta = r.get('metadata') if type(r.get('metadata')) is dict else {}
        gen = meta.get('route_generation')
        if gen in gens and r['scope'] == 'native':
            hard.append(dict(rank=r['rank'], seq=r['seq'], generation=gen, why='native_session_in_control'))
            refused('native_session_in_control')
            continue
        why = usable(r)
        if why:
            refused(why)
            continue
        cell = dict(rank=r['rank'], bank=meta['bank'], bank_sms=meta.get('bank_sms'), cg_mode=meta['cg_mode'],
                    num_tokens=meta['num_tokens'], actual_rows=meta['actual_rows'],
                    D=r['device_ns'] / 1e6, history=max(meta['history_tokens']), generation=gen, stamp=r['stamp'])
        if gen in gens:
            if meta['bank'] != 'split':
                refused('control_row_not_split')
            elif r['stamp'] is None or r['stamp'] <= last_end:
                refused('control_generation_inside_arms')
            elif r['scope'] != 'outside':
                refused('control_row_not_outside')
            else:
                used.append(dict(cell, cls='Q_split', stage=gens[gen]['index'], arm='control'))
            continue
        arm = arm_of(r['stamp'], arms)
        if arm is None:
            refused('after_arms_no_control_stage' if r['stamp'] is not None and r['stamp'] > last_end else 'no_arm')
            continue
        if arm == 'boundary':
            refused('arm_boundary')
            continue
        if n3_only and not arm.startswith('n3.'):
            refused('not_n3_arm:' + arm)
            continue
        if r['scope'] == 'outside':
            used.append(dict(cell, cls='Q_full' if meta['bank'] == 'full' else 'C_split', arm=arm))
        elif meta['bank'] == 'split':
            used.append(dict(cell, cls='B_split', arm=arm))
        else:
            refused('native_on_full_bank')
    return dict(root=str(root), engine_log=path, rows=used, refusals=refusals, hard_stops=hard, arms=arms,
                control=control, last_arm_end=last_end, loss=dict(torn_lines=torn, sessions=loss,
                conflicting_d_rows=stats['conflicting_d_rows'], sample_gaps=stats['sample_gaps'],
                sessions_incomplete=stats['sessions_incomplete'], counters=stats['counters']),
                budget=budget(flushes, sizes, used))


def budget(flushes, sizes, used):
    """Per rank: highest native session and outside sample numbers, log_budget_exhausted; the line sizes are split
    before / from the control's first Q_split row by split_bytes."""
    out = {}
    for f in flushes:
        rank = f['rank']
        b = out.setdefault(str(rank), dict(native_max=0, outside_max=0, log_budget_exhausted=0, bytes_before_control=0,
                                           bytes_control=0, lines=0))
        sample = f['head'].get('sample')
        if f['head'].get('scope') == bf.OUTSIDE_SCOPE and type(sample) is str and sample[8:].isdigit():
            b['outside_max'] = max(b['outside_max'], int(sample[8:]))
        elif type(sample) is int:
            b['native_max'] = max(b['native_max'], sample)
        for s in f['rows'].get('s', []):
            if ((s + [None, None])[1] or {}).get('log_budget_exhausted'):
                b['log_budget_exhausted'] += 1
    return dict(per_rank=out, caps=dict(sessions=MAX_SESSIONS, outside=MAX_OUTSIDE, bytes=MAX_BYTES - 4096),
                sizes=sizes)


def split_bytes(doc, control_start):
    for stamp, rank, size in doc['budget'].pop('sizes'):
        b = doc['budget']['per_rank'].get(str(rank))
        if b is None:
            continue
        b['lines'] += 1
        b['bytes_control' if control_start is not None and stamp is not None and stamp >= control_start
          else 'bytes_before_control'] += size


def cells(rows):
    out = {}
    for r in rows:
        key = (r['cls'], r['rank'], r.get('stage'), r['bank'], r['bank_sms'], r['cg_mode'], r['num_tokens'],
               r['actual_rows'])
        out.setdefault(key, []).append(r)
    table = []
    for key, items in sorted(out.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        Ds = [i['D'] for i in items]
        hs = [i['history'] for i in items]
        table.append(dict(cls=key[0], rank=key[1], stage=key[2], bank=key[3], bank_sms=key[4], cg_mode=key[5],
                          num_tokens=key[6], actual_rows=key[7], n=len(items), D_p10=round(p(Ds, 0.1), 3),
                          D_p50=round(p(Ds, 0.5), 3), D_p90=round(p(Ds, 0.9), 3), history_p50=int(p(hs, 0.5)),
                          history_min=min(hs), history_max=max(hs), band=band_name(int(p(hs, 0.5))),
                          arms=sorted({i['arm'] for i in items})))
    return table


def pooled(raw, cls, rank, live, stage=None, pool_8=False):
    """One class at a live count: num_tokens PADDED[live], actual_rows live (8 live with pool_8: rows 7 and 8)."""
    out = [r for r in raw if r['cls'] == cls and r['rank'] == rank and r['num_tokens'] == PADDED[live]
           and (r['actual_rows'] == live or (pool_8 and live == 8 and r['actual_rows'] in (7, 8)))
           and (stage is None or r.get('stage') == stage)]
    return dict(n=len(out), p50=round(p([r['D'] for r in out], 0.5), 3) if out else None,
                history_p50=int(p([r['history'] for r in out], 0.5)) if out else None)


def stage_value(per_rank):
    """{stage: max of the two TP peers' p50} (both peers needed), never a sum."""
    out = {}
    for stage, (a, b) in STAGES.items():
        va, vb = (per_rank.get(a) or {}).get('p50'), (per_rank.get(b) or {}).get('p50')
        out[stage] = None if va is None or vb is None else max(va, vb)
    return out


def three(doc, reference_rows):
    """Per live count: Q_full, Q_split per control stage, C_split (consistency), B_split; S, N; the check."""
    raw = doc['rows']
    stages = [s for s in ((doc['control'] or {}).get('stages') or [])]
    out = []
    for live in LIVE:
        row = dict(live=live, num_tokens=PADDED[live], ranks={})
        for rank in range(4):
            q_full = pooled(raw, 'Q_full', rank, live)
            busy = pooled(raw, 'B_split', rank, live, pool_8=True)
            ref = pooled(raw + reference_rows, 'C_split', rank, live)
            ctl = {s['index']: pooled(raw, 'Q_split', rank, live, stage=s['index'])
                   for s in stages if s['live'] == live}
            row['ranks'][rank] = dict(Q_full=q_full, B_split=busy, C_split=ref, Q_split=ctl)
        row['stage_values'] = dict(
            Q_full=stage_value({k: v['Q_full'] for k, v in row['ranks'].items()}),
            B_split=stage_value({k: v['B_split'] for k, v in row['ranks'].items()}),
            C_split=stage_value({k: v['C_split'] for k, v in row['ranks'].items()}),
            Q_split={i: stage_value({k: v['Q_split'][i] for k, v in row['ranks'].items()})
                     for i in sorted({i for v in row['ranks'].values() for i in v['Q_split']})})
        checks = {}
        for s in stages:
            if s['live'] != live:
                continue
            ok, why = 0, []
            for rank, v in row['ranks'].items():
                q, c = v['Q_split'][s['index']], v['C_split']
                if q['n'] < MIN_CTL:
                    why.append('rank%d:control-n%d<%d' % (rank, q['n'], MIN_CTL))
                elif c['n'] < MIN_REF:
                    why.append('rank%d:reference-n%d<%d' % (rank, c['n'], MIN_REF))
                elif abs(q['p50'] - c['p50']) <= CONSIST_MS:
                    ok += 1
                else:
                    why.append('rank%d:|%.2f-%.2f|>%.1f' % (rank, q['p50'], c['p50'], CONSIST_MS))
            diffs = {}
            for rank, v in row['ranks'].items():
                q = v['Q_split'][s['index']]['p50']
                diffs[rank] = dict(S=None if q is None or v['Q_full']['p50'] is None else round(q - v['Q_full']['p50'], 3),
                                   N=None if q is None or v['B_split']['p50'] is None else round(v['B_split']['p50'] - q, 3))
            checks[s['index']] = dict(band=s['band'], counted=s['counted'], consistency='PASS' if ok >= 3 else 'FAIL',
                                      ranks_within=ok, why=why, differences=diffs)
        row['control_stages'] = checks
        out.append(row)
    return out


def read(root, quiet_split=None, references=(), engine_log=None):
    doc = classify(root, quiet_split, engine_log)
    control_start = min((r['stamp'] for r in doc['rows'] if r['cls'] == 'Q_split' and r['stamp'] is not None),
                        default=None)
    doc['control_first_row_stamp'] = control_start
    split_bytes(doc, control_start)
    ref_rows = []
    for ref in references:
        other = classify(ref, None, None)
        other['budget'].pop('sizes', None)
        ref_rows += [r for r in other['rows'] if r['cls'] == 'C_split']
    doc['references'] = [str(r) for r in references]
    doc['table'] = cells(doc['rows'])
    doc['three'] = three(doc, ref_rows)
    doc['schema'] = SCHEMA
    doc['row_counts'] = {c: sum(1 for r in doc['rows'] if r['cls'] == c) for c in ('Q_full', 'Q_split', 'C_split', 'B_split')}
    n3_attn = {k: v['attn_meta'] for k, v in doc['arms'].items() if k.startswith('n3.')}
    doc['attn_meta'] = dict(n3_arms=n3_attn, control=(doc['control'] or {}).get('attn_meta'))
    del doc['rows']
    return doc


def fmt(v):
    return '-' if v is None or v.get('p50') is None else '%.2f (n %d, h %dk)' % (v['p50'], v['n'], v['history_p50'] // 1000)


def text(doc):
    out = ['QUIET-SPLIT CROSS-ARM TABLE %s (decode-timeline d rows; descriptive, no verdict)' % doc['root'],
           'engine log %s; references (C_split pooled) %s' % (doc['engine_log'], doc['references'] or 'none'),
           'rows used %s; refusals %s' % (json.dumps(doc['row_counts']), json.dumps(doc['refusals'], sort_keys=True)),
           'hard stops %s' % (json.dumps(doc['hard_stops']) if doc['hard_stops'] else 'none'),
           'loss %s' % json.dumps(doc['loss'], sort_keys=True),
           'attn-meta: n3 arms %s; control %s' % (json.dumps(doc['attn_meta']['n3_arms'], sort_keys=True),
                                                  doc['attn_meta']['control'] or 'no quiet-split.json')]
    c = doc['control']
    if c:
        out.append('control %s verdict %s%s; native admitted before/after %s; stages %s' % (
            c['schema'], c['verdict'], '' if c['verdict'] == 'PASS' else ' (' + str(c['cause']) + ')', c['admitted'],
            json.dumps([dict(i=s['index'], live=s['live'], band=s['band'], gen=s['generation'], counted=s['counted'])
                        for s in c['stages']])))
    else:
        out.append('control: no quiet-split.json (no Q_split class)')
    out.append('budget per rank (caps %s): %s' % (json.dumps(doc['budget']['caps']), json.dumps(doc['budget']['per_rank'], sort_keys=True)))
    out.append('== cells (cls rank stage bank bank_sms cg_mode num_tokens actual_rows: n D p10/p50/p90 ms, max-history p50 '
               '[min..max] band, arms)')
    for t in doc['table']:
        out.append('  %-7s r%d %-4s %-5s %-3s %-4s nt%-2d rows%-2d n %4d D %.2f/%.2f/%.2f h %d [%d..%d] %s %s' % (
            t['cls'], t['rank'], '-' if t['stage'] is None else 's%d' % t['stage'], t['bank'], t['bank_sms'], t['cg_mode'],
            t['num_tokens'], t['actual_rows'], t['n'], t['D_p10'], t['D_p50'], t['D_p90'], t['history_p50'],
            t['history_min'], t['history_max'], t['band'], ','.join(t['arms'])))
    out.append('== the three numbers per live count, p50 ms (n, max-history p50): Q_full | Q_split per control stage | '
               'C_split (uncertified, consistency) | B_split (8 live: rows 7+8); TP stage = max of peers')
    for row in doc['three']:
        out.append('  live %d (num_tokens %d)' % (row['live'], row['num_tokens']))
        for rank in range(4):
            v = row['ranks'][rank]
            ctl = ' '.join('s%d %s' % (i, fmt(q)) for i, q in sorted(v['Q_split'].items())) or '-'
            out.append('    r%d Q_full %s | Q_split %s | C_split %s | B_split %s' % (
                rank, fmt(v['Q_full']), ctl, fmt(v['C_split']), fmt(v['B_split'])))
        out.append('    TP stage values %s' % json.dumps(row['stage_values'], sort_keys=True))
        for i, chk in sorted(row['control_stages'].items()):
            out.append('    control stage %d (band %s, counted %s): consistency %s (%d of 4 ranks within %.1f ms) %s; '
                       'S = Q_split - Q_full, N = B_split - Q_split per rank %s' % (
                           i, chk['band'], chk['counted'], chk['consistency'], chk['ranks_within'], CONSIST_MS,
                           chk['why'] or '', json.dumps(chk['differences'], sort_keys=True)))
    return '\n'.join(out) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--quiet-split')
    ap.add_argument('--reference', action='append', default=[])
    ap.add_argument('--engine-log')
    a = ap.parse_args(argv)
    out = Path(a.out)
    for name in ('quiet-split-read.json', 'quiet-split-read.txt'):
        if (out / name).exists():
            refuse('%s exists: a fresh --out is required' % (out / name))
    try:
        doc = read(a.root, a.quiet_split, a.reference, a.engine_log)
    except bf.Refused as exc:
        print(str(exc), file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'quiet-split-read.json').open('x') as handle:
        json.dump(doc, handle, sort_keys=True, indent=1, default=str)
    with (out / 'quiet-split-read.txt').open('x') as handle:
        handle.write(text(doc))
    print(text(doc), end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
