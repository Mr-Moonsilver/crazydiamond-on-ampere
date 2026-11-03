#!/usr/bin/env python3
"""Box admission check: did each finished native arm of a box queue reach the native prefill path?

Why: q18a-alpha-next-c1b ran 25 min and measured nothing. Its fresh requests carried logprobs, the engine's auto-arm
planner refuses those (patches/s5_engine/native_auto_arm.py ``_plan``), and every arm was served stock. Every arm record
showed it as soon as the arm ended: route ``full``, auto_arm phase ``idle``, ``admitted`` flat, progress chunks 0. This
tool reads those records so a watcher can raise it after the first native arm instead of after the queue.

Reads only. Python stdlib only (runs on the box as ``python3 -I - check DIR`` with this file on stdin). Writes nothing,
except the JSON to ``--out`` when given (created, never overwritten).

Input: a queue output dir (on the box /home/<service-user>/s0/<QNAME>; offline a fetched record's box/run/<QNAME> dir). The dir
and its ``<QNAME>-*`` sibling dirs are scanned for ``arm-<label>.json`` (or ``.json.xz`` offline). The queue writes each
record once, at the end of its arm (``--out "$Q/arm-$_label.json"``, opened with 'x'), so a record that exists is a
finished arm; one that does not parse yet is reported UNKNOWN (possibly still being written) and never alerts.

Which arms are native: the arm table's own declared field, which every arm record carries (copied from its table row):
  kind  ``auto`` = native auto-arm cycle, ``stock`` = stock control. Native41 progress/auto-arm records and every queue
        built on them (alpha-perf, step-b, stage1-sweep, alpha-next, alpha-next-c1, alpha-next-c1b, ...).
  bulk  ``true`` = the arm sends the ~40k bulk prefill that must run native, ``false`` = the nobulk control.
        returning-append (returning_append_arms) and append-yield (q18b_append_yield_arms).
A record with neither field is ``undeclared`` and UNKNOWN; no arm name is interpreted.

Per native arm: admitted (the engine's cumulative ``admitted`` counter, as the arm's own ``receipts.admitted_delta`` and as
the rise over the arm's polled samples and statuses; the larger known value), polled samples with route ``split`` or
auto_arm phase ``native``, and progress chunks (``progress_delta.chunks``; absent in the bulk family).
  ADMITTED             any of the three is above 0.
  NO_NATIVE_ADMISSION  the arm finished, served its native-eligible request (``fresh``, or ``requests.bulk``) without
                       error, has polled samples, admitted is known and 0, no split/native sample, progress chunks 0 or
                       not recorded.
  UNKNOWN              anything else (unreadable, unfinished, arm error, request not served, no samples, admitted not
                       recorded, not declared). Never a guess.

Output: line 1 one JSON object (schema box-admission-check/1), line 2 one summary line starting ``box-admission``.
The overall verdict is NO_NATIVE_ADMISSION when any finished native arm (not skipped) is, else ADMITTED when one is,
else UNKNOWN.

  box_admission_check.py check QUEUE_DIR [--skip LABEL[,LABEL...]] [--out FILE]
  box_admission_check.py alert FILE|-     reads a check's JSON line; on NO_NATIVE_ADMISSION prints
                                          'ALERT: no-native-admission <arm> (admitted 0, split 0/<n> samples)' and the
                                          labels to skip on a relaunch, exit 2; else prints nothing, exit 0; bad input 3.
"""
import argparse
import json
import lzma
import os
import re
import sys

SCHEMA = 'box-admission-check/1'
ADMITTED = 'ADMITTED'
NO_NATIVE_ADMISSION = 'NO_NATIVE_ADMISSION'
UNKNOWN = 'UNKNOWN'
MAX_BYTES = 64 << 20
ARM_FILE = re.compile(r'^arm-(?P<label>[A-Za-z0-9._-]+?)\.json(?P<xz>\.xz)?$')
LABEL = re.compile(r'^[A-Za-z0-9._/-]+$')


def _int(value):
    """An int that is not a bool, else None."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _subdirs(path):
    try:
        names = sorted(os.listdir(path))
    except OSError:
        return []
    return [n for n in names if os.path.isdir(os.path.join(path, n)) and not os.path.islink(os.path.join(path, n))]


def queue_dirs(queue_dir):
    """[(rel, path)]: the queue dir ('' ), its direct subdirs (a sweep's per-engine dirs, 'p0-d44') and its
    ``<name>-*`` sibling dirs (per-arm engine dirs of older queues). No symlinks."""
    queue_dir = os.path.abspath(queue_dir)
    parent, name = os.path.split(queue_dir)
    if not os.path.isdir(queue_dir) or os.path.islink(queue_dir):
        return []
    dirs = [('', queue_dir)] + [(n, os.path.join(queue_dir, n)) for n in _subdirs(queue_dir)]
    dirs += [(n, os.path.join(parent, n)) for n in _subdirs(parent) if n.startswith(name + '-')]
    return dirs


def arm_files(dirs):
    """[(arm id, path)]; the id is the label, prefixed '<rel>/' outside the queue dir itself. In one dir a plain .json
    wins over a .json.xz of the same label."""
    found = {}
    for rel, d in dirs:
        try:
            names = sorted(os.listdir(d))
        except OSError:
            continue
        for entry in names:
            m = ARM_FILE.match(entry)
            path = os.path.join(d, entry)
            if not m or not os.path.isfile(path) or os.path.islink(path):
                continue
            key, xz = (rel + '/' if rel else '') + m.group('label'), bool(m.group('xz'))
            if key in found and (found[key][1] is False or xz):
                continue
            found[key] = (path, xz)
    return sorted((k, v[0]) for k, v in found.items())


def load(path):
    """(record, error); error is a short reason, the record a dict or None."""
    try:
        if path.endswith('.xz'):
            with lzma.open(path, 'rb') as handle:
                data = handle.read(MAX_BYTES + 1)
        else:
            if os.path.getsize(path) > MAX_BYTES:
                return None, 'arm-record-too-large'
            with open(path, 'rb') as handle:
                data = handle.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            return None, 'arm-record-too-large'
        record = json.loads(data.decode('utf-8'))
    except (OSError, ValueError, lzma.LZMAError, EOFError):
        return None, 'arm-record-unreadable-or-partial'
    if not isinstance(record, dict):
        return None, 'arm-record-not-an-object'
    return record, None


def declared(record):
    """(True native / False control / None undeclared, the signal used)."""
    kind = record.get('kind')
    if kind == 'auto':
        return True, 'kind=auto'
    if kind == 'stock':
        return False, 'kind=stock'
    bulk = record.get('bulk')
    if bulk is True:
        return True, 'bulk=true'
    if bulk is False:
        return False, 'bulk=false'
    return None, 'undeclared'


def native_request(record):
    """(served True/False/None, which request): the request the native path must take."""
    fresh = record.get('fresh')
    if isinstance(fresh, dict):
        first = fresh.get('first_at_ns')
        out = _int(fresh.get('output_completion'))
        served = not fresh.get('error') and not fresh.get('malformed') and (first is not None or (out or 0) > 0)
        return served, 'fresh'
    requests = record.get('requests')
    bulk = requests.get('bulk') if isinstance(requests, dict) else None
    if isinstance(bulk, dict):
        served = (bulk.get('sent') is True and not bulk.get('error') and not bulk.get('malformed')
                  and (bulk.get('first_ns') is not None or (_int(bulk.get('output_tokens')) or 0) > 0))
        return served, 'requests.bulk'
    return None, None


def measure(record):
    """The counts of one arm record."""
    polled = record.get('polled')
    samples = polled.get('samples') if isinstance(polled, dict) else None
    samples = [s for s in samples if isinstance(s, dict)] if isinstance(samples, list) else []
    statuses = record.get('statuses')
    statuses = [s for s in statuses if isinstance(s, dict)] if isinstance(statuses, list) else []

    def phase(s):
        auto = s.get('auto_arm')
        return auto.get('phase') if isinstance(auto, dict) else None

    split = sum(1 for s in samples if s.get('route') == 'split')
    native = sum(1 for s in samples if phase(s) == 'native')
    either = sum(1 for s in samples if s.get('route') == 'split' or phase(s) == 'native')
    polls = sum(1 + max(0, _int(s.get('repeats')) or 0) for s in samples)
    seen = [_int(s.get('admitted')) for s in samples + statuses]
    seen = [v for v in seen if v is not None]
    polled_delta = max(seen) - min(seen) if seen else None
    receipts = record.get('receipts')
    receipt_delta = _int(receipts.get('admitted_delta')) if isinstance(receipts, dict) else None
    known = [v for v in (receipt_delta, polled_delta) if v is not None]
    progress = record.get('progress_delta')
    chunks = _int(progress.get('chunks')) if isinstance(progress, dict) else None
    return dict(samples=len(samples), polls=polls, split_samples=split, native_phase_samples=native,
                split_or_native_samples=either, admitted=max(known) if known else None,
                admitted_receipt=receipt_delta, admitted_polled=polled_delta, progress_chunks=chunks)


def arm_errored(record):
    reasons = record.get('reasons')
    reasons = reasons if isinstance(reasons, list) else []
    return bool(record.get('error')) or 'arm-error' in reasons or 'arm-refused' in reasons


def judge(label, path, record, error):
    row = dict(label=label, file=os.path.basename(path), role=None, signal=None, verdict=UNKNOWN, reasons=[],
               wall_end_s=None)
    if record is None:
        row['reasons'].append(error)
        return row
    if isinstance(record.get('label'), str) and record['label'] != label.rsplit('/', 1)[-1]:
        row['record_label'] = record['label']
    native, signal = declared(record)
    row['signal'] = signal
    row['role'] = {True: 'native', False: 'control', None: 'undeclared'}[native]
    end = record.get('wall_end_s')
    row['wall_end_s'] = end if isinstance(end, (int, float)) and not isinstance(end, bool) else None
    row.update(measure(record))
    served, which = native_request(record)
    row['native_request'], row['native_request_served'] = which, served
    if native is False:
        row['verdict'] = None
        return row
    if native is None:
        row['reasons'].append('native-or-control-not-declared')
        return row
    if (row['admitted'] or 0) > 0 or row['split_or_native_samples'] > 0 or (row['progress_chunks'] or 0) > 0:
        row['verdict'] = ADMITTED
        return row
    if row['wall_end_s'] is None:
        row['reasons'].append('arm-not-finished')
    if arm_errored(record):
        row['reasons'].append('arm-error')
    if served is None:
        row['reasons'].append('native-request-not-recorded')
    elif served is not True:
        row['reasons'].append('native-request-not-served')
    if row['samples'] == 0:
        row['reasons'].append('no-polled-samples')
    if row['admitted'] is None:
        row['reasons'].append('admitted-not-recorded')
    if not row['reasons']:
        row['verdict'] = NO_NATIVE_ADMISSION
    return row


def check(queue_dir, skip=()):
    dirs = queue_dirs(queue_dir)
    out = dict(schema=SCHEMA, queue_dir=os.path.abspath(queue_dir), dirs=[rel or '.' for rel, _ in dirs],
               verdict=UNKNOWN, reasons=[], skipped=sorted(set(skip)), first_no_admission=None, arms=[])
    if not dirs:
        out['reasons'].append('queue-dir-missing')
    for label, path in arm_files(dirs):
        record, error = load(path)
        row = judge(label, path, record, error)
        row['skipped'] = label in skip
        out['arms'].append(row)
    out['arms'].sort(key=lambda r: (r['wall_end_s'] is None, r['wall_end_s'] or 0.0, r['label']))
    counted = [r for r in out['arms'] if not r['skipped']]
    native = [r for r in counted if r['role'] == 'native']
    out['counts'] = dict(arm_records=len(out['arms']), native=len(native),
                         control=sum(1 for r in counted if r['role'] == 'control'),
                         undeclared_or_unreadable=sum(1 for r in counted if r['role'] in (None, 'undeclared')),
                         admitted=sum(1 for r in native if r['verdict'] == ADMITTED),
                         no_native_admission=sum(1 for r in native if r['verdict'] == NO_NATIVE_ADMISSION),
                         unknown=sum(1 for r in native if r['verdict'] == UNKNOWN),
                         skipped=len(out['arms']) - len(counted))
    missing = [r for r in native if r['verdict'] == NO_NATIVE_ADMISSION]
    if missing:
        out['verdict'] = NO_NATIVE_ADMISSION
        first = missing[0]
        out['first_no_admission'] = {k: first[k] for k in ('label', 'admitted', 'split_or_native_samples', 'samples',
                                                           'polls', 'progress_chunks', 'signal')}
        out['no_admission_labels'] = [r['label'] for r in missing]
    elif any(r['verdict'] == ADMITTED for r in native):
        out['verdict'] = ADMITTED
    elif dirs:
        out['reasons'].append('no-arm-record' if not out['arms'] else 'no-finished-native-arm-with-evidence')
    return out


def summary(out):
    c = out.get('counts', {})
    text = 'box-admission %s %s: native arms %d (admitted %d, no admission %d, unknown %d), control %d, undeclared %d' % (
        out['verdict'], os.path.basename(out['queue_dir']), c.get('native', 0), c.get('admitted', 0),
        c.get('no_native_admission', 0), c.get('unknown', 0), c.get('control', 0), c.get('undeclared_or_unreadable', 0))
    if c.get('skipped'):
        text += ', skipped %d' % c['skipped']
    first = out.get('first_no_admission')
    if first:
        text += '; first %s admitted %s split %d/%d samples (%d polls)' % (
            first['label'], first['admitted'], first['split_or_native_samples'], first['samples'], first['polls'])
    if out.get('reasons'):
        text += '; ' + ','.join(out['reasons'])
    return text


def alert_lines(out):
    """The watcher's alert for a check result, or None."""
    if not isinstance(out, dict) or out.get('schema') != SCHEMA:
        raise ValueError('not a %s object' % SCHEMA)
    if out.get('verdict') != NO_NATIVE_ADMISSION:
        return None
    first = out.get('first_no_admission')
    if not isinstance(first, dict) or not LABEL.match(str(first.get('label'))):
        raise ValueError('first_no_admission missing')
    labels = [x for x in out.get('no_admission_labels') or [] if LABEL.match(str(x))]
    return ['ALERT: no-native-admission %s (admitted %s, split %s/%s samples)' % (
                first['label'], first.get('admitted'), first.get('split_or_native_samples'), first.get('samples')),
            'relaunch-skip: %s' % ','.join(sorted(set(labels + list(out.get('skipped') or [])))) ]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = p.add_subparsers(dest='cmd')
    c = sub.add_parser('check', help='check a queue dir and its <name>-* siblings')
    c.add_argument('queue_dir')
    c.add_argument('--skip', default='', help='comma-separated arm labels left out of the verdict (a cleared alert)')
    c.add_argument('--out', help='also write the JSON here (created; never overwritten)')
    a_ = sub.add_parser('alert', help="render the watcher's alert from a check's JSON line")
    a_.add_argument('source', help='file with the check output, or - for stdin')
    a = p.parse_args(argv)
    if a.cmd == 'check':
        skip = [x for x in a.skip.split(',') if x]
        if any(not LABEL.match(x) for x in skip):
            p.error('bad --skip label')
        out = check(a.queue_dir, skip)
        text = json.dumps(out, sort_keys=True, separators=(',', ':'))
        if a.out:
            with open(a.out, 'x') as handle:
                handle.write(text + '\n')
        print(text)
        print(summary(out))
        return 0
    if a.cmd == 'alert':
        try:
            raw = sys.stdin.read(MAX_BYTES) if a.source == '-' else open(a.source).read(MAX_BYTES)
            line = next((x for x in raw.splitlines() if x.strip()), '')
            lines = alert_lines(json.loads(line))
        except (OSError, ValueError) as exc:
            print('box-admission alert: unreadable check output: %s' % exc, file=sys.stderr)
            return 3
        if lines is None:
            return 0
        print('\n'.join(lines))
        return 2
    p.print_help(sys.stderr)
    return 3


if __name__ == '__main__':
    sys.exit(main())
