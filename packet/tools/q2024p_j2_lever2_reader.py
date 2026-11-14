#!/usr/bin/env python3
"""Lever 2 pair reader (soft split, FLASH_S5_DECODE_OVERLAP_SMS; claude/n3-lever2-split, 14 Nov 2026). Hub only.

Two fetched single packets from ONE source commit, the cxay12 flags plus the decode-timeline stamp, differing only in
FLASH_S5_DECODE_OVERLAP_SMS (off = 0, on = 42):

    python3 -I -B tools/q2024p_j2_lever2_reader.py --off RECORD_ROOT --on RECORD_ROOT --out FRESH_DIR
        [--engine-log-off PATH] [--engine-log-on PATH]

Per packet, from the engine log (tools/q2024p_j2_quiet_split_reader.classify: the decode-timeline d rows of the n3
arms, every refusal counted):
  per-forward decode time, ms (D = the rank's own device span), per rank and live count 1..8:
    B_split  native-session rows on the split bank (decode beside a native piece), the number lever 2 moves;
    C_split  outside rows on the split bank (near native work, not certified far);
    Q_full   outside rows on the full bank;
  each cell one (cg_mode, bank_sms) key (a second key: 'mixed', no p50); off reads bank_sms 24, on 42. TP stage value
  = max of the two peers' p50, never a sum. Off less on per rank and stage at every live count, with both n.
  native chunk rate per rank: worker anatomy chunks (s5-native-anatomy/1 'c' rows) inside an n3 arm, rows =
  chunk_end - chunk_start, by width band: p50 busy_us per chunk and p50 rows / (t_last - t_first) per chunk; off less
  on in percent of off.
  the overlap startup record per rank (s5-decode-overlap/1: own / borrowed SMs, superset, pair memory delta) from any
  JSON under the root holding a native construction inventory, else from 'flash-s5 decode-overlap' log lines.
The sizing's decision inputs (SIZING-lever2.md) at 4 live: B_split stage-0 and stage-1 off less on, and the native
full-chunk (1025-2048 rows, BULK) rate change per rank. Descriptive only: no verdict, no turns/h (the cx reader reads each packet's R).
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import statistics

SCHEMA = 'q2024p-j2-lever2-read/1'
HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('q2024p_j2_quiet_split_reader_l2', HERE / 'q2024p_j2_quiet_split_reader.py')
qs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qs)
bf = qs.bf

LIVE = (1, 2, 3, 4, 5, 6, 7, 8)
CLASSES = ('B_split', 'C_split', 'Q_full')
WIDTHS = ((1, 1025), (1025, 2049), (2049, 4097))
OVERLAP_SCHEMA = 's5-decode-overlap/1'
OVERLAP_LINE = 'flash-s5 decode-overlap '
MAX_JSON_BYTES = 64 * 1024 * 1024
BULK = '1025-2048'


def p50(values):
    return round(statistics.median(values), 3) if values else None


def width_of(rows):
    for lo, hi in WIDTHS:
        if lo <= rows < hi:
            return '%d-%d' % (lo, hi - 1)
    return None


def forwards(rows):
    """{cls: {live: {rank: pooled cell}}} and stage values."""
    out = {}
    for cls in CLASSES:
        per_live = {}
        for live in LIVE:
            ranks = {r: qs.pooled(rows, cls, r, live) for r in range(4)}
            per_live[live] = dict(ranks=ranks, stages=qs.stage_value(ranks))
        out[cls] = per_live
    return out


def native_rates(root, engine_log=None):
    """Per rank and width band: chunks, p50 busy_us, p50 rows/s over the chunk's device window (t_first..t_last)."""
    path = bf.engine_log(str(root), engine_log)
    records, torn, _ = bf.read_records(bf.lines_of(path))
    flushes, loss = bf.assemble(records['anatomy'])
    arms = qs.arms_of(str(root))
    cells, refusals = {}, {}

    def refused(why):
        refusals[why] = refusals.get(why, 0) + 1
    for f in flushes:
        if f['side'] != 'worker' or f['rank'] not in range(4):
            continue
        h = f['head']
        start, end = h.get('chunk_start'), h.get('chunk_end')
        if type(start) is not int or type(end) is not int or end <= start:
            refused('no_chunk_bounds')
            continue
        arm = qs.arm_of(f['stamp'], arms)
        if arm in (None, 'boundary') or not arm.startswith('n3.'):
            refused('not_inside_n3_arm' if arm != 'boundary' else 'arm_boundary')
            continue
        for c in f['rows'].get('c', []):
            first, last, busy = bf.at(f['base'], c[4]), bf.at(f['base'], c[5]), bf.num(c[6])
            if (bf.num(c[10]) or 0) or (bf.num(c[12]) or 0):
                refused('cancelled_or_errors')
                continue
            if first is None or last is None or last <= first or busy is None:
                refused('no_device_window')
                continue
            cell = cells.setdefault((f['rank'], width_of(end - start)), dict(busy=[], rate=[]))
            cell['busy'].append(busy)
            cell['rate'].append((end - start) / ((last - first) / 1e9))
    table = {}
    for (rank, width), cell in sorted(cells.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        table.setdefault(str(rank), {})[width] = dict(chunks=len(cell['busy']), busy_us_p50=p50(cell['busy']),
                                                      rows_per_s_p50=p50(cell['rate']))
    return dict(engine_log=path, per_rank=table, refusals=refusals, loss=dict(torn_lines=torn['anatomy'], **loss))


def _walk(value, found):
    if type(value) is dict:
        split = value.get('split') if value.get('schema') == 's5-native-construction/1' else None
        if type(split) is dict and type(split.get('decode_overlap')) is dict:
            found.setdefault(str(value.get('worker_rank')), split['decode_overlap'])
        for v in value.values():
            _walk(v, found)
    elif type(value) is list:
        for v in value:
            _walk(v, found)


def overlap_records(root, engine_log=None):
    """{rank: s5-decode-overlap/1 record} from JSON under the root (two levels), else the startup log lines."""
    found = {}
    for base, dirs, files in os.walk(str(root)):
        if base[len(str(root)):].count(os.sep) >= 2:
            dirs[:] = []
        for name in sorted(files):
            path = os.path.join(base, name)
            if not name.endswith('.json') or os.path.getsize(path) > MAX_JSON_BYTES:
                continue
            with open(path, errors='replace') as handle:
                text = handle.read()
            if OVERLAP_SCHEMA not in text:
                continue
            try:
                _walk(json.loads(text), found)
            except ValueError:
                continue
    lines = []
    if not found:
        for line in bf.lines_of(bf.engine_log(str(root), engine_log)):
            at = line.find(OVERLAP_LINE)
            if at >= 0:
                try:
                    lines.append(json.loads(line[at + len(OVERLAP_LINE):]))
                except ValueError:
                    continue
    return dict(per_rank=found, log_lines=lines, source='inventory' if found else 'log' if lines else 'absent')


def side(root, engine_log=None):
    doc = qs.classify(str(root), engine_log=engine_log)
    return dict(root=str(root), engine_log=doc['engine_log'], refusals=doc['refusals'], loss=doc['loss'],
                hard_stops=doc['hard_stops'], row_counts={c: sum(1 for r in doc['rows'] if r['cls'] == c)
                                                         for c in CLASSES},
                forwards=forwards(doc['rows']), native=native_rates(root, engine_log),
                overlap=overlap_records(root, engine_log))


def delta(a, b):
    return None if a is None or b is None else round(a - b, 3)


def compare(off, on):
    out = {}
    for cls in CLASSES:
        rows = []
        for live in LIVE:
            a, b = off['forwards'][cls][live], on['forwards'][cls][live]
            rows.append(dict(live=live, ranks={r: dict(off=a['ranks'][r], on=b['ranks'][r],
                                                       off_less_on=delta(a['ranks'][r]['p50'], b['ranks'][r]['p50']))
                                               for r in range(4)},
                             stages={s: dict(off=a['stages'][s], on=b['stages'][s],
                                             off_less_on=delta(a['stages'][s], b['stages'][s])) for s in (0, 1)}))
        out[cls] = rows
    native = {}
    for rank in map(str, range(4)):
        for width in sorted(set(off['native']['per_rank'].get(rank, {})) | set(on['native']['per_rank'].get(rank, {})),
                            key=str):
            a = off['native']['per_rank'].get(rank, {}).get(width) or {}
            b = on['native']['per_rank'].get(rank, {}).get(width) or {}
            ra, rb = a.get('rows_per_s_p50'), b.get('rows_per_s_p50')
            native.setdefault(rank, {})[width] = dict(
                off=a, on=b, rate_change_pct=None if not ra or rb is None else round(100.0 * (rb - ra) / ra, 2))
    four = next(r for r in out['B_split'] if r['live'] == 4)
    # full native chunks are 2024-2048 rows on this line (chunk bound [97152, 2024], CHUNK_TOKENS 2048; ayc2t2: none wider)
    bulk = {rank: (native.get(rank, {}).get(BULK) or {}).get('rate_change_pct') for rank in map(str, range(4))}
    return dict(forwards=out, native=native,
                decision_inputs=dict(b_split_4_live_off_less_on_ms={s: four['stages'][s]['off_less_on'] for s in (0, 1)},
                                     native_bulk_rate_change_pct=bulk,
                                     sizing=('build the per-step choice if decode >= 1.0 ms faster and native <= 3 % slower; '
                                             'drop SM borrowing if decode < 0.6 ms faster; native > 6 % slower rules out '
                                             'lever (a) too (SIZING-lever2.md; descriptive here, no verdict)')))


def fmt(v):
    return '-' if v is None else ('%.3f' % v if type(v) is float else str(v))


def cell(c):
    if c.get('mixed'):
        return 'mixed %s' % c['mixed']
    return '%s (n %d, %s)' % (fmt(c['p50']), c['n'], c.get('key') or '-')


def text(doc):
    out = ['lever 2 pair read (%s): off %s, on %s' % (SCHEMA, doc['off']['root'], doc['on']['root'])]
    for name in ('off', 'on'):
        s = doc[name]
        out.append('%s: rows %s; refusals %s; hard stops %d; overlap record: %s %s' % (
            name, s['row_counts'], s['refusals'], len(s['hard_stops']), s['overlap']['source'],
            {r: (o.get('own_decode_sms'), o.get('borrowed_prefill_sms'), o.get('superset'), o.get('pair_delta_bytes'))
             for r, o in sorted(s['overlap']['per_rank'].items())}))
    for cls in CLASSES:
        out.append('')
        out.append('%s per-forward D p50 ms: off | on | off-on, per rank; stage = max of peers' % cls)
        for row in doc['compare']['forwards'][cls]:
            parts = ['live %d' % row['live']]
            for r in range(4):
                x = row['ranks'][r]
                parts.append('r%d %s | %s | %s' % (r, cell(x['off']), cell(x['on']), fmt(x['off_less_on'])))
            parts.append('stage0 %s' % fmt(row['stages'][0]['off_less_on']))
            parts.append('stage1 %s' % fmt(row['stages'][1]['off_less_on']))
            out.append('  ' + '; '.join(parts))
    out.append('')
    out.append('native chunk rate per rank and width (p50 rows/s over t_first..t_last; chunks): off | on | change %')
    for rank, widths in sorted(doc['compare']['native'].items()):
        for width, x in sorted(widths.items(), key=lambda kv: str(kv[0])):
            out.append('  r%s %s: %s (%s) | %s (%s) | %s' % (rank, width, fmt(x['off'].get('rows_per_s_p50')),
                                                           x['off'].get('chunks'), fmt(x['on'].get('rows_per_s_p50')),
                                                           x['on'].get('chunks'), fmt(x['rate_change_pct'])))
    d = doc['compare']['decision_inputs']
    out.append('')
    out.append('decision inputs: B_split 4 live off-on stage0 %s stage1 %s ms; native bulk rate change %% %s' % (
        fmt(d['b_split_4_live_off_less_on_ms'][0]), fmt(d['b_split_4_live_off_less_on_ms'][1]),
        d['native_bulk_rate_change_pct']))
    out.append('  ' + d['sizing'])
    return '\n'.join(out) + '\n'


def read(off, on, engine_log_off=None, engine_log_on=None):
    a, b = side(off, engine_log_off), side(on, engine_log_on)
    return dict(schema=SCHEMA, off=a, on=b, compare=compare(a, b))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--off', required=True)
    parser.add_argument('--on', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--engine-log-off')
    parser.add_argument('--engine-log-on')
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise SystemExit('fresh --out required: %s exists' % out)
    doc = read(args.off, args.on, args.engine_log_off, args.engine_log_on)
    out.mkdir(parents=True)
    (out / 'lever2.json').write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + '\n')
    (out / 'lever2.txt').write_text(text(doc))
    print(text(doc), end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
