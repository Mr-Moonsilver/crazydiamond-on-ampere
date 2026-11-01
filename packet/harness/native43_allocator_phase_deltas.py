#!/usr/bin/env python3
"""Per-phase allocator deltas and indexer score sizes from a retained engine log.  Receipt only (review item M3).

Reads the workers' per-chunk lines (`flash-s5 memory`, `flash-s5 native-alloc`) from a Docker-timestamped log.
The phases are the queue's watched clients (engine-watch-<name>.json: started, finished).  For each phase and
rank, a counter's delta is its last value in the window minus its last value before the window, or minus its
first value in the window when no earlier line exists.  `whole-run` spans every line.  The counters are the
caching allocator's own cumulative ones, so a delta covers everything the rank allocated in that time,
decode included.  The receipts come only at chunk open and release: activity after the last line in a window
lands in the next window that has a line.

Nothing here is a verdict.  Exit 0 with a record, 2 when there is nothing to read.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

SCHEMA = 'native43-allocator-phase-deltas/1'
COUNTERS = ('alloc_retries', 'ooms', 'sync_all_streams', 'device_allocs', 'device_frees')
LINE = re.compile(r'^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?Z) .*?flash-s5 (memory|native-alloc) (.*)$')
FIELD = re.compile(r'(\w+)=(\S+)')


def wall_ns(text):
    """RFC 3339 UTC (any fraction, trailing Z) to integer nanoseconds."""
    head, _, rest = text.rstrip('Z').partition('.')
    seconds = datetime.strptime(head, '%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc).timestamp()
    return int(seconds) * 10**9 + int((rest + '0' * 9)[:9])


def parse_row(text):
    row = {}
    for key, value in FIELD.findall(text):
        row[key] = int(value) if re.fullmatch(r'-?\d+', value) else value
    return row


def read_rows(paths):
    memory, alloc, lines = {}, {}, 0
    for path in paths:
        with open(path, errors='replace') as stream:
            for raw in stream:
                if 'flash-s5 ' not in raw:
                    continue
                match = LINE.match(raw.rstrip('\n'))
                if match is None:
                    continue
                row = parse_row(match.group(3))
                if type(row.get('rank')) is not int:
                    continue
                row['wall_ns'] = wall_ns(match.group(1))
                lines += 1
                (memory if match.group(2) == 'memory' else alloc).setdefault(row['rank'], []).append(row)
    for table in (memory, alloc):
        for rows in table.values():
            rows.sort(key=lambda row: row['wall_ns'])
    return memory, alloc, lines


def read_phases(directory):
    phases, invalid = [], []
    for path in sorted(Path(directory).glob('engine-watch-*.json')):
        name = path.name[len('engine-watch-'):-len('.json')]
        try:
            record = json.loads(path.read_text())
            phases.append(dict(name=name, started=record['started'], finished=record['finished'],
                               start_ns=wall_ns(record['started']), end_ns=wall_ns(record['finished'])))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            invalid.append(dict(name=name, error=type(exc).__name__))
    return phases, invalid


def counter_deltas(rows, start_ns, end_ns):
    window = [row for row in rows if start_ns <= row['wall_ns'] <= end_ns]
    if not window:
        return dict(lines=0)
    earlier = [row for row in rows if row['wall_ns'] < start_ns]
    base, last = (earlier[-1] if earlier else window[0]), window[-1]
    out = dict(lines=len(window), baseline='before-window' if earlier else 'first-in-window')
    for name in COUNTERS:
        known = name in base and name in last and base[name] >= 0 and last[name] >= 0
        out[name] = last[name] - base[name] if known else None
    return out


def alloc_summary(rows, start_ns, end_ns):
    window = [row for row in rows if start_ns <= row['wall_ns'] <= end_ns]
    if not window:
        return dict(chunks=0)

    def top(name):
        return max(row.get(name, 0) for row in window)

    def total(name):
        return sum(row.get(name, 0) for row in window)
    return dict(chunks=len(window), union_sets=total('union_sets'), union_retired=total('union_retired'),
                union_single_max=top('union_single'), union_live_max=top('union_live_max'),
                indexer_layers=total('indexer_layers'), indexer_rows=total('indexer_rows'),
                indexer_width_max=top('indexer_width'), indexer_width_last=window[-1].get('indexer_width', 0),
                indexer_chunks=total('indexer_chunks'), indexer_peak_max=top('indexer_peak'),
                indexer_total=total('indexer_total'), indexer_errors=total('indexer_errors'))


def build(streams, watch_dir):
    memory, alloc, lines = read_rows(streams)
    phases, invalid = read_phases(watch_dir) if watch_dir else ([], [])
    walls = [row['wall_ns'] for table in (memory, alloc) for rows in table.values() for row in rows]
    spans = [dict(name='whole-run', started=None, finished=None, start_ns=min(walls), end_ns=max(walls))] if walls else []
    out = []
    for phase in spans + phases:
        ranks = {}
        for rank in sorted(set(memory) | set(alloc)):
            ranks[str(rank)] = dict(
                counters=counter_deltas(memory.get(rank, []), phase['start_ns'], phase['end_ns']),
                allocations=alloc_summary(alloc.get(rank, []), phase['start_ns'], phase['end_ns']))
        out.append(dict(name=phase['name'], started=phase['started'], finished=phase['finished'], ranks=ranks))
    return dict(schema=SCHEMA, verdict='RECEIPT', lines=lines, phases=out, invalid_watch_records=invalid,
                basis='cumulative caching-allocator counters at chunk open and release; indexer sizes are host-known')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--stream', action='append', required=True)
    parser.add_argument('--watch-dir')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    try:
        record = build(args.stream, args.watch_dir)
    except OSError as exc:
        print('ALLOCATOR_PHASE_DELTAS unreadable: ' + type(exc).__name__, file=sys.stderr)
        return 2
    with open(args.out, 'x') as stream:
        json.dump(record, stream, sort_keys=True, indent=1)
        stream.write('\n')
    print('ALLOCATOR_PHASE_DELTAS lines=%d phases=%d' % (record['lines'], len(record['phases'])))
    return 0 if record['lines'] else 2


if __name__ == '__main__':
    sys.exit(main())
