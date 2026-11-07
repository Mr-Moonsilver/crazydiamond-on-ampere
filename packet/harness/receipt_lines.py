#!/usr/bin/env python3
"""Receipt record integrity for queues on the alpha-next writer (PLAN section 2). New name; old graders unchanged.

The engine writes every receipt record (one progress/anatomy/handler part, one native-alloc, memory, stage-local or
receipts line) with one atomic write of at most 3,968 bytes, carrying pid, seq (per process, +1 across every record)
and sha (sha256[:16] of the whole record before identity, on every part). This module reads a followed engine log and
reports, per process:

- missing_seq: any step in a process's seq that is not exactly +1 in log order (a gap, a repeat or a reorder), a first
  seq that is not 0 (when the log starts with the process), and a record that carries a receipt prefix and an identity
  but no longer reads;
- bad_sha: a whole record whose parts disagree on sha, or whose sha differs from the one recomputed over its parts;
- incomplete: a record whose parts are not all present;
- unsafe_path: a process whose `flash-s5 receipts` record (seq 0) is not `receipt_path=atomic-fifo`, a startup line
  that says unsafe or is missing (when required), and a receipt line without identity (the old logger path).

Records are found by prefix anywhere in a physical line: another process's non-atomic vLLM line may precede one of ours
on the same physical line; that is their tear, not ours. The last receipt prefix in the line starts the record.

Rule for new queues: any missing seq, bad sha or incomplete record makes every read fed by that process's receipts
INC, and so does an unsafe path. Nothing is tolerated. `verdict()` returns that per process and for the whole log.
"""
import hashlib
import json
import lzma
import re
import sys
from pathlib import Path

# `flash-s5 stop ` is one JSON object without parts; `flash-s5 memory-peak ` a k=v line (both added on claude/alpha-next
# after 5990af80).
JSON_PREFIXES = ('flash-s5 progress ', 'flash-s5 anatomy ', 'flash-s5 handler ', 'flash-s5 stop ')
KV_PREFIXES = ('flash-s5 receipts ', 'flash-s5 native-alloc ', 'flash-s5 memory ', 'flash-s5 stage-local ',
               'flash-s5 memory-peak ', 'flash-s5 cow ', 'flash-s5 ahead-split ')    # parity-na2: the writer's list, kept equal
PREFIXES = JSON_PREFIXES + KV_PREFIXES
PATH_PREFIX = 'flash-s5 receipts '
STARTUP_PATH = re.compile(r'flash-s5 startup receipt_path=(\S+)(?: pipe_buf=(\d+))?')
JSON_IDENTITY = re.compile(r'^\{"pid":(\d+),"seq":(\d+),"sha":"([0-9a-f]{16})",')
KV_IDENTITY = re.compile(r' pid=(\d+) seq=(\d+) sha=([0-9a-f]{16})$')
KV_RANK = re.compile(r' rank=(\d+) ')
SAFE = 'atomic-fifo'
_FIND = re.compile('|'.join(re.escape(p) for p in PREFIXES))


def sha16(texts):
    return hashlib.sha256('\n'.join(texts).encode()).hexdigest()[:16]


def _record_of(line):
    """(prefix, text) of the last receipt record in a physical line, else None."""
    last = None
    for m in _FIND.finditer(line):
        last = m
    if last is None:
        return None
    return last.group(0), line[last.start():].rstrip('\n')


def parse_line(line):
    """One physical line -> a record dict, a legacy marker, an unreadable marker, or None."""
    found = _record_of(line)
    if found is None:
        return None
    prefix, text = found
    kind = prefix.split()[1]
    if prefix in JSON_PREFIXES:
        body = text[len(prefix):]
        m = JSON_IDENTITY.match(body)
        if m is None:
            return dict(kind=kind, legacy=True)
        pid, seq, sha = int(m.group(1)), int(m.group(2)), m.group(3)
        bare = prefix + '{' + body[m.end():]
        try:
            head = json.loads(body)
        except ValueError:
            return dict(kind=kind, pid=pid, seq=seq, sha=sha, unreadable=True)
        if type(head) is dict and 'part' not in head and 'parts' not in head:
            # A single-object JSON record (stop rows): judged alone, like a k=v record.
            return dict(kind=kind, pid=pid, seq=seq, sha=sha, text=bare, side=head.get('side'), rank=head.get('rank'),
                        line=None, part=0, parts=1)
        if type(head) is not dict or type(head.get('part')) is not int or type(head.get('parts')) is not int:
            return dict(kind=kind, pid=pid, seq=seq, sha=sha, unreadable=True)
        return dict(kind=kind, pid=pid, seq=seq, sha=sha, text=bare, side=head.get('side'), rank=head.get('rank'),
                    line=head.get('line'), part=head['part'], parts=head['parts'])
    m = KV_IDENTITY.search(text)
    if m is None:
        return dict(kind=kind, legacy=True)
    bare = text[:m.start()]
    rank = KV_RANK.search(bare)
    record = dict(kind=kind, pid=int(m.group(1)), seq=int(m.group(2)), sha=m.group(3), text=bare,
                  side='engine' if kind == 'stage-local' else None, rank=int(rank.group(1)) if rank else None,
                  line=None, part=0, parts=1)
    if kind == 'receipts':
        path = re.search(r'receipt_path=(\S+)', bare)
        record['path'] = path.group(1) if path else None
    return record


def read(paths):
    """Physical lines of the followed logs (.xz too), in order."""
    for path in paths:
        path = Path(path)
        if not path.is_file():
            continue
        opener = lzma.open if path.suffix == '.xz' else open
        with opener(path, 'rt', errors='replace') as handle:
            yield from handle


def _judge(key, parts, bad_sha, incomplete):
    pid, kind, side, rank, line = key
    parts.sort(key=lambda r: r['part'])
    count = parts[0]['parts']
    if [r['part'] for r in parts] != list(range(count)) or any(r['parts'] != count for r in parts):
        incomplete.append(dict(pid=pid, kind=kind, side=side, rank=rank, line=line,
                               parts=[r['part'] for r in parts], of=count))
    elif len({r['sha'] for r in parts}) != 1 or sha16([r['text'] for r in parts]) != parts[0]['sha']:
        bad_sha.append(dict(pid=pid, kind=kind, side=side, rank=rank, line=line))


def check(lines, *, require_start=True, require_startup=False):
    """The four reports over physical lines; see the module docstring."""
    missing_seq, bad_sha, incomplete, unsafe_path = [], [], [], []
    last_seq, procs, groups, startup = {}, {}, {}, []
    legacy = {}
    records = 0
    for line in lines:
        m = STARTUP_PATH.search(line)
        if m:
            startup.append(m.group(1))
        rec = parse_line(line)
        if rec is None:
            continue
        if rec.get('legacy'):
            legacy[rec['kind']] = legacy.get(rec['kind'], 0) + 1
            continue
        records += 1
        pid, seq = rec['pid'], rec['seq']
        proc = procs.setdefault(pid, dict(pid=pid, records=0, kinds={}, sides=set(), ranks=set(), path=None,
                                          first_seq=seq))
        proc['records'] += 1
        proc['kinds'][rec['kind']] = proc['kinds'].get(rec['kind'], 0) + 1
        if pid in last_seq:
            if seq != last_seq[pid] + 1:
                missing_seq.append(dict(pid=pid, after=last_seq[pid], got=seq))
        elif require_start and seq != 0:
            missing_seq.append(dict(pid=pid, after=None, got=seq))
        last_seq[pid] = seq
        if rec.get('unreadable'):
            missing_seq.append(dict(pid=pid, unreadable=seq, kind=rec['kind']))
            continue
        if rec.get('side') is not None:
            proc['sides'].add(rec['side'])
        if rec.get('rank') is not None:
            proc['ranks'].add(rec['rank'])
        if rec['kind'] == 'receipts':
            proc['path'] = rec.get('path')
        if rec['parts'] == 1 and rec['line'] is None:
            if sha16([rec['text']]) != rec['sha']:
                bad_sha.append(dict(pid=pid, seq=seq, kind=rec['kind']))
            continue
        key = (pid, rec['kind'], rec['side'], rec['rank'], rec['line'])
        group = groups.setdefault(key, [])
        group.append(rec)
        if len(group) >= rec['parts']:
            # Judged as soon as it can be whole; a later part with the same key is a new, incomplete group.
            _judge(key, groups.pop(key), bad_sha, incomplete)
    for key, parts in groups.items():
        _judge(key, parts, bad_sha, incomplete)
    for pid, proc in procs.items():
        if proc['path'] is None:
            if require_start:
                unsafe_path.append(dict(pid=pid, reason='no-path-record'))
        elif not proc['path'].startswith(SAFE):
            unsafe_path.append(dict(pid=pid, reason=proc['path']))
    for kind, count in sorted(legacy.items()):
        unsafe_path.append(dict(pid=None, reason='legacy:%s' % kind, count=count))
    if require_startup and not startup:
        unsafe_path.append(dict(pid=None, reason='no-startup-line'))
    for value in startup:
        if value != SAFE:
            unsafe_path.append(dict(pid=None, reason='startup:%s' % value))
    return dict(schema='receipt-lines/1', records=records, processes=len(procs),
                missing_seq=missing_seq, bad_sha=bad_sha, incomplete=incomplete, unsafe_path=unsafe_path,
                startup=startup,
                per_process={str(pid): dict(records=p['records'], kinds=p['kinds'], sides=sorted(p['sides']),
                                            ranks=sorted(p['ranks']), path=p['path'], first_seq=p['first_seq'],
                                            last_seq=last_seq[pid])
                             for pid, p in sorted(procs.items())})


def verdict(report):
    """Per process INC/OK; the log is OK only when nothing is reported. Nothing is tolerated."""
    bad = {}
    for name in ('missing_seq', 'bad_sha', 'incomplete', 'unsafe_path'):
        for item in report[name]:
            bad.setdefault(item.get('pid'), set()).add(name)
    whole = 'OK' if not bad else 'INC'
    return dict(verdict=whole,
                inc_processes={str(pid): sorted(names) for pid, names in bad.items() if pid is not None},
                log_wide=sorted(bad.get(None, ())))


def main(argv):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('logs', nargs='+')
    parser.add_argument('--window', action='store_true', help='the log does not start with the processes')
    parser.add_argument('--require-startup', action='store_true')
    args = parser.parse_args(argv)
    report = check(read(args.logs), require_start=not args.window, require_startup=args.require_startup)
    report.update(verdict(report))
    print(json.dumps(report, sort_keys=True))
    return 0 if report['verdict'] == 'OK' else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
