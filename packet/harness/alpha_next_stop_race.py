#!/usr/bin/env python3
"""STOP_RACE: a declared class for the vLLM abort-stop race at the queue's own stop (lead, 2 Nov 2026 13:30 CET). Stdlib only.

Why: q18a-step-b-e (2 Nov) and q18a-native26 (30 Oct) both logged, at the queue-owned ``systemctl stop`` with the engine idle and
every worker empty, the API server's ``shutdown triggered`` and ``stopping engine client mode=abort timeout=0s``, a force kill of
EngineCore, ``MPClient: complete`` and then ``AsyncLLM output_handler failed`` with an ``EngineDeadError`` 9.3 ms (Step B) and
6.4 ms (native26) after the trigger. native43_engine_death.py counts every EngineDeadError as a hard death marker, so the queue
recorded an engine crash. The diagnosis (fn-main results/box/q18a-step-b-e/README.md, tmp/step-b-oom/REPORT.md) is a race in
vLLM's abort stop, not an engine fault. native43_engine_death.py is frozen (past queues ran it); this module reads its record
and its events and decides only whether that one DEAD is this race.

STOP_RACE when every rule holds, else the death stays a crash (exit 1):
  1. the death scan (native43_engine_death.Scanner over the same logs and stop intent) says DEAD;
  2. its only death event is one ``engine_dead`` hard marker: no other hard marker in either stream and no unplanned teardown
     before the orderly bound (a worker failure record, FLASH_S5_WORKER_FAILURE, is a hard marker and is named on its own);
  3. that EngineDeadError line belongs to an ``AsyncLLM output_handler failed`` traceback: the handler line precedes it, from the
     same process, at most HANDLER_SPAN_NS earlier;
  4. the queue's stop intent was read and is at or before the API server's first ``shutdown triggered`` after it, which is at or
     before the EngineDeadError, and the EngineDeadError is at most WINDOW_NS (500 ms) after that trigger;
  5. the end-full-idle record is PASS and every worker row in it is empty (0 running, prefill and live decode requests).
Exit 0 STOP_RACE (record ``stop_race`` true, ``engine_crashed`` false), 1 crash (``engine_crashed`` true, ``reasons`` name the
rules that failed), 2 not dead (nothing to classify). The record is written with --out (replaced, as the scan's).
"""
import argparse
import json
from pathlib import Path
import re
import sys

import native43_engine_death as death

SCHEMA = 'alpha-next-stop-race/1'
WINDOW_NS = 500 * 10 ** 6
HANDLER_SPAN_NS = 50 * 10 ** 6
TRIGGER = re.compile(rb'\[shutdown\] API server: shutdown triggered')
HANDLER = re.compile(rb'AsyncLLM output_handler failed')
WORKER_FAILURE = re.compile(rb'FLASH_S5_WORKER_FAILURE')
PROCESS = re.compile(rb'\((\w+) pid=(\d+)\)')
WORKER_FIELDS = ('running_requests', 'prefill_requests', 'live_decode_requests')


def _lines(paths):
    for index, path in enumerate(paths):
        try:
            with open(path, 'rb') as handle:
                for number, line in enumerate(handle, 1):
                    yield index, number, line.rstrip(b'\n')
        except OSError:
            continue


def _process(line):
    m = PROCESS.search(line)
    return m.group(0).decode() if m else None


def idle_empty(path):
    """(ok, reason) of the end-full-idle record: PASS and every worker row empty."""
    if path is None:
        return False, 'end-full-idle-not-given'
    try:
        row = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return False, 'end-full-idle-unreadable'
    if row.get('verdict') != 'PASS':
        return False, 'end-full-idle-not-pass'
    workers = row.get('workers')
    if type(workers) is not dict or not workers:
        return False, 'end-full-idle-without-worker-rows'
    for rank, cell in sorted(workers.items()):
        if type(cell) is not dict or any(cell.get(k) != 0 for k in WORKER_FIELDS):
            return False, 'end-full-idle-worker-not-empty:%s' % rank
    return True, None


def classify(logs, stop_intent, end_full_idle):
    scanner = death.Scanner(logs, stop_intent)
    scanner.poll(final=True)
    scan = scanner.decide()
    record = dict(schema=SCHEMA, scan_verdict=scan['verdict'], stop_race=False, engine_crashed=scan['verdict'] == 'DEAD',
                  window_ms=WINDOW_NS / 1e6, reasons=[], stop_intent=scan['stop_intent'], first_death=scan['first_death'],
                  death_counts=scan['death_counts'], trigger=None, handler=None, gap_ms=None, end_full_idle=None)
    if scan['verdict'] != 'DEAD':
        record['reasons'].append('not-dead')
        return record, 2
    reasons = record['reasons']
    hard = [e for e in scanner.events if e['family'] == 'hard']
    if scan['death_counts'] != {'engine_dead': 1} or len(hard) != 1:
        reasons.append('deaths-other-than-one-engine-dead:%s' % json.dumps(scan['death_counts'], sort_keys=True))
    dead = next((e for e in hard if e['kind'] == 'engine_dead'), None)
    intent = scan['stop_intent'].get('wall_ns')
    if intent is None:
        reasons.append('stop-intent-not-read:%s' % scan['stop_intent'].get('state'))
    triggers, handlers, failures = [], [], 0
    for index, number, line in _lines(logs):
        if TRIGGER.search(line):
            triggers.append(dict(file=index, line=number, wall_ns=death.wall_ns(line)))
        if HANDLER.search(line):
            handlers.append(dict(file=index, line=number, wall_ns=death.wall_ns(line), process=_process(line)))
        if WORKER_FAILURE.search(line):
            failures += 1
    if failures:
        reasons.append('worker-failure-record:%d' % failures)
    trigger = None
    if intent is not None:
        trigger = min((t for t in triggers if t['wall_ns'] is not None and t['wall_ns'] >= intent),
                      key=lambda t: t['wall_ns'], default=None)
    if trigger is None:
        reasons.append('no-shutdown-trigger-after-the-stop-intent')
    if dead is None or dead['wall_ns'] is None:
        reasons.append('engine-dead-line-unstamped' if dead else 'no-engine-dead-line')
    else:
        process = _process(dead['text'].encode())
        handler = max((h for h in handlers if h['wall_ns'] is not None and h['file'] == dead['file'] and h['line'] < dead['line']
                       and h['process'] == process and 0 <= dead['wall_ns'] - h['wall_ns'] <= HANDLER_SPAN_NS),
                      key=lambda h: h['line'], default=None)
        record['handler'] = handler and dict(handler, wall=death.iso(handler['wall_ns']))
        if handler is None:
            reasons.append('engine-dead-not-in-an-output-handler-traceback')
        if trigger is not None:
            gap = dead['wall_ns'] - trigger['wall_ns']
            record['gap_ms'] = round(gap / 1e6, 3)
            if gap < 0:
                reasons.append('engine-dead-before-the-shutdown-trigger')
            elif gap > WINDOW_NS:
                reasons.append('engine-dead-more-than-500-ms-after-the-shutdown-trigger')
    record['trigger'] = trigger and dict(trigger, wall=death.iso(trigger['wall_ns']))
    ok, why = idle_empty(end_full_idle)
    record['end_full_idle'] = dict(path=end_full_idle, all_empty=ok)
    if not ok:
        reasons.append(why)
    if reasons:
        return record, 1
    record.update(stop_race=True, engine_crashed=False)
    return record, 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--log', action='append', required=True, help='engine log file; repeat for each (the scan\'s logs)')
    p.add_argument('--stop-intent', required=True, help="the queue's host-stop-intent.json")
    p.add_argument('--end-full-idle', required=True, help='the end-full-idle check record')
    p.add_argument('--out', required=True)
    a = p.parse_args(argv)
    record, rc = classify(a.log, a.stop_intent, a.end_full_idle)
    death.write_json(a.out, record, exclusive=False)
    print('ALPHA_NEXT_STOP_RACE %s%s' % ({0: 'STOP_RACE', 1: 'CRASH', 2: 'NOT_DEAD'}[rc],
                                         (' ' + ','.join(record['reasons'])) if rc == 1 else ''))
    return rc


if __name__ == '__main__':
    sys.exit(main())
