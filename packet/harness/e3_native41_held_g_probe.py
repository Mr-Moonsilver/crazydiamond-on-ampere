#!/usr/bin/env python3
"""Held-g connected CPU probe: the engine step anatomy (R5) on held-f's connected composition.

Runs e3_native41_progress_probe.run unchanged (production install_engine, real
EngineCore.step_with_batch_queue, scheduler, controller, driver, worker
bridges and dispatchers, the auto-arm policy, every held-f check) and also
captures the engine's anatomy lines. Per mode it checks:

* unset: no anatomy line;
* otherwise one engine anatomy flush per native chunk, with the chunk's
  identity and arm mode, no dropped row, known step classes only;
* the chunk's step sends equal its native commands (the 'n' rows of its
  progress line), so every send happened inside a recorded step;
* detail chunks (one in ANATOMY_DETAIL_EVERY) tile the chunk with
  contiguous steps; outside an owner wait a step sends at most one command
  (mode current: every step), and never more than one plus the wait cap;
* chunks begun with no live decoder (the solo modes) run empty stock steps,
  the pacing L3 removes; live chunks run stock steps with tokens.

Fixture leaves are e3_native41_progress_probe's. Its execute and sample
fixtures do not go through collective_rpc, so the 'rpcs' column is zero here
by construction; the stock RPC counter it reads is the one collective_rpc
increments in test_e3_native_reply_hol.
"""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import traceback
from unittest import mock

import e3_native41_progress_probe as pp

MODES = pp.MODES
ANATOMY_LINE = 'flash-s5 anatomy '
PROGRESS_LINE = 'flash-s5 progress '


class Capture(pp.Capture):
    """Held-f's capture (progress lines in ``lines``) plus the anatomy lines."""
    anatomy = []
    progress = []

    def emit(self, record):
        super().emit(record)
        message = record.getMessage()
        if message.startswith(ANATOMY_LINE):
            Capture.anatomy.append(message)
        elif message.startswith(PROGRESS_LINE):
            Capture.progress.append(message)


def flushes(lines, prefix):
    out = {}
    for text in lines:
        body = json.loads(text[len(prefix):])
        out.setdefault((body['side'], body['line']), []).append(body)
    joined = []
    for key in sorted(out):
        parts = sorted(out[key], key=lambda b: b['part'])
        assert [b['part'] for b in parts] == list(range(parts[0]['parts'])), key
        rows = {}
        for b in parts:
            for tag, items in b['rows'].items():
                rows.setdefault(tag, []).extend(items)
        joined.append((parts[0], rows))
    return joined


def run(mode):
    Capture.anatomy, Capture.progress = [], []
    with mock.patch.object(pp, 'Capture', Capture):
        result = pp.run(mode)
    import native_measurement as nm
    import engine_driver as ed
    checks = result['checks']

    def check(name, ok, **details):
        checks.append(dict(name=name, ok=bool(ok), **details))
        assert ok, name + ' ' + json.dumps(details, sort_keys=True, default=str)[:2000]
    lines = Capture.anatomy
    if mode == 'unset':
        check('unset-no-anatomy', not lines)
        result.update(anatomy_lines=0)
        return result
    check('anatomy-lines-bounded', lines and all(len(t) < nm.PROGRESS_LINE_BYTES for t in lines))
    anatomy = flushes(lines, ANATOMY_LINE)
    check('only-engine-anatomy-in-process', all(h['side'] == 'engine' for h, _ in anatomy))
    # The progress lines, re-read from the same records the held-f checks used.
    progress = flushes(Capture.progress, PROGRESS_LINE)
    native = result['native']
    check('one-anatomy-flush-per-native-chunk', len(anatomy) == len(native)
          and [(h['chunk_start'], h['chunk_end']) for h, _ in anatomy]
          == [(r['start'], r['end']) for r in native], flushes=len(anatomy), chunks=len(native))
    check('anatomy-fields-and-no-drop', all(
        h['fields'] == {k: list(v) for k, v in nm.ANATOMY_ROWS.items()} and h['dropped'] == 0
        and h['schema'] == nm.ANATOMY_SCHEMA for h, _ in anatomy))
    fields = nm.ANATOMY_ROWS
    limit = ed.NativeProgress.WAIT_SEND_LIMIT
    burst = mode in ('burst', 'burst-skew', 'burst-solo')
    summary = []
    for index, (head, rows) in enumerate(anatomy):
        e = [dict(zip(fields['e'], r)) for r in rows.get('e', [])]
        p = [dict(zip(fields['p'], r)) for r in rows.get('p', [])]
        classes = {r['cls']: r['steps'] for r in e}
        sends = sum(r['sends'] for r in e)
        txn = head['txn']
        check('chunk-%d-anatomy-identity' % txn, head['mode'] == ('burst' if burst else 'current')
              and type(head['decode_live']) is int and head['detail'] == int(index % nm.ANATOMY_DETAIL_EVERY == 0),
              head={k: head.get(k) for k in ('txn', 'mode', 'decode_live', 'detail')})
        check('chunk-%d-known-classes' % txn, classes and set(classes) <= set(nm.StepAnatomy.CLASSES),
              classes=classes)
        commands = None
        if index < len(progress):
            ph, prow = progress[index]
            commands = len(prow.get('n', []))
            check('chunk-%d-progress-and-anatomy-are-the-same-chunk' % txn,
                  (ph['txn'], ph['chunk_start'], ph['chunk_end']) == (txn, head['chunk_start'], head['chunk_end']))
        check('chunk-%d-sends-are-the-native-commands' % txn, sends == commands, sends=sends, commands=commands)
        if p:
            check('chunk-%d-detail-steps-tile-the-chunk' % txn, len(p) == sum(classes.values()) and all(
                abs(a[1] + a[2] - b[1]) <= 1 for a, b in
                zip([(r['cls'], r['t_begin'], r['total_us']) for r in p][:-1],
                    [(r['cls'], r['t_begin'], r['total_us']) for r in p][1:])),
                steps=len(p))
            cap = limit + 1 if burst else 1
            check('chunk-%d-sends-per-step-bounded' % txn, all(r['sends'] <= cap for r in p),
                  most=max(r['sends'] for r in p), cap=cap)
        if head['decode_live'] == 0:
            check('chunk-%d-solo-paces-on-empty-stock-steps' % txn, classes.get('empty', 0) > 0
                  and classes.get('stock', 0) == 0, classes=classes)
        else:
            check('chunk-%d-live-runs-stock-steps' % txn, classes.get('stock', 0) > 0, classes=classes)
        summary.append(dict(txn=txn, decode_live=head['decode_live'], classes=classes, sends=sends,
                            total_us={r['cls']: r['total_us'] for r in e}))
    result.update(anatomy_lines=len(lines), anatomy_flushes=len(anatomy), anatomy=summary,
                  anatomy_line_sha256=hashlib.sha256('\n'.join(lines).encode()).hexdigest())
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mode', choices=MODES, default='trace')
    a = p.parse_args()
    try:
        r = run(a.mode)
        rc = 0
    except Exception as exc:
        r = dict(verdict='FAIL', mode=a.mode, error_type=type(exc).__name__, error=str(exc),
                 traceback=traceback.format_exc())
        rc = 1
    r.update(mode=a.mode, schema='native41-held-g-cpu/1', gpu_execution=False, box_evidence=False,
             probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with a.out.open('x') as f:
        json.dump(r, f, sort_keys=True, indent=2)
        f.write('\n')
    print(json.dumps({k: r.get(k) for k in ('verdict', 'mode', 'error_type', 'error')}))
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
