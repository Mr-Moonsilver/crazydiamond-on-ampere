#!/usr/bin/env python3
"""Native39 trunk queue checks.  Each subcommand writes one receipt and exits 0 PASS, 1 FAIL, 2 INCONCLUSIVE.

bind         The first 'flash-s5 startup owner bound' line of the current engine
             container must come before its first /health 200.
             Log: the queue's host observer follows `docker logs --follow
             --timestamps` of the one container that started after the restart
             (the old container id is excluded) into Q/engine-stream.log and
             .err, so the followed log is persisted although the container is
             --rm, and it is run-scoped.  The receipt also requires that every
             host-disk sample row so far names the same container as
             `docker inspect` now (--container-id), that the line is
             timestamped after restart-begin, and that exactly one such line
             exists.
             Health: Q/health.tsv holds one row per /health request,
             "<sent> <returned> <code>" ($EPOCHREALTIME before and after curl).
             PASS iff the bind line's docker timestamp is before the *sent* time
             of the first 200 request.  If it falls inside that request
             (sent <= bind <= returned) the order is unproven: INCONCLUSIVE.
             After the returned time: FAIL.  No 200, or no line within
             --wait-s: INCONCLUSIVE.  Docker stamps a line when it reads it,
             after the logger call, so the stamp can only make the bind look
             later, never earlier.
bind-epoch   The bind line's engine_epoch equals the startup grade's.
host         The last --host in a declaration/capture identity command is
             127.0.0.1 (EXTRA_ARGS last-wins, appended after the launcher's
             --host 0.0.0.0), and with --q the API server's own
             'Starting vLLM server on http://HOST:PORT' line in the followed
             container log names 127.0.0.1 exactly once (q18b-w1ab2 B0 logged
             http://127.0.0.1:8000 with this image and setting).
found-extra-args  /etc/default/vllm-flash EXTRA_ARGS (engine_identity's
             EnvironmentFile parser) has the declared production words,
             compared silently (no defaults value is written or printed).
idle         /metrics running + waiting requests reach 0 within --timeout-s
             (after the plain W1 client, whose requests end on disconnect).
summary      Trunk reading beside native35_performance_grade: per W2 round the
             decode tok/s during the fresh prefill, its share of the native W1
             FULL median (the grade's denominator) and of the plain W1 median
             (descriptive), the prefill share, plus the checks above.  Never a
             gate; 80/80 stays a marker; always rc 0 once written.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
import time
import urllib.request

BIND_RE = re.compile(r'flash-s5 startup owner bound engine_epoch=(\d+)')
SERVED_RE = re.compile(r'Starting vLLM server on http://([^\s:/]+):(\d+)')
TS = re.compile(r'^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(\.\d+)?Z ')
LOOPBACK = '127.0.0.1'


def write(out, record):
    with Path(out).open('x') as f:
        json.dump(record, f, sort_keys=True, indent=2)
        f.write('\n')
    print(json.dumps({k: record.get(k) for k in ('check', 'verdict', 'rc', 'detail')}, sort_keys=True))
    return record['rc']


def verdict(record, rc, detail):
    record.update(rc=rc, verdict={0: 'PASS', 1: 'FAIL', 2: 'INCONCLUSIVE'}[rc], detail=detail)
    return record


def stamp(line):
    m = TS.match(line)
    if not m:
        return None
    frac = (m.group(2) or '.0')[:7]
    return datetime.datetime.fromisoformat(m.group(1) + frac + '+00:00').timestamp()


def health_rows(path):
    begin, rows = None, []
    for line in Path(path).read_text().splitlines():
        p = line.split()
        if len(p) == 2 and p[1] == 'restart-begin':
            begin = float(p[0])
        elif len(p) == 3 and re.fullmatch(r'\d{3}', p[2]):
            rows.append((float(p[0]), float(p[1]), p[2]))
    return begin, rows


def log_matches(q, pattern):
    found = []
    for name in ('engine-stream.log', 'engine-stream.err'):
        path = q / name
        if not path.is_file():
            continue
        with path.open(errors='replace') as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    found.append((name, line, m))
    return found


def bind_lines(q):
    return [dict(file=name, t=stamp(line), engine_epoch=int(m.group(1)), line=line.rstrip('\n')[:300])
            for name, line, m in log_matches(q, BIND_RE)]


def observed_containers(q):
    ids = set()
    path = q / 'host-disk.jsonl'
    if path.is_file():
        for line in path.read_text().splitlines():
            try:
                ids.add(json.loads(line).get('container_id'))
            except ValueError:
                ids.add('unreadable-row')
    return ids


def bind(a):
    q = Path(a.q)
    record = dict(check='bind-before-health', schema='native39-trunk-checks/1', pattern=BIND_RE.pattern,
                  log='Q/engine-stream.log + .err (host observer, docker --follow --timestamps, one container)')
    begin, rows = health_rows(q / 'health.tsv')
    first = next((r for r in rows if r[2] == '200'), None)
    record.update(restart_begin=begin, health_samples=len(rows),
                  first_200=None if first is None else dict(sent=first[0], returned=first[1]))
    deadline = time.monotonic() + a.wait_s
    lines = bind_lines(q)
    while not lines and time.monotonic() < deadline:
        time.sleep(1)
        lines = bind_lines(q)
    containers = observed_containers(q)
    record.update(bind_lines=lines, observed_containers=sorted(c for c in containers if c),
                  current_container=a.container_id[:12])
    if first is None:
        return verdict(record, 2, 'no /health 200 in %d samples; order not testable' % len(rows))
    if containers != {a.container_id[:12]}:
        return verdict(record, 2, 'followed log not proven to be the current container')
    if not lines:
        return verdict(record, 2, 'no bind line within %s s of the check' % a.wait_s)
    if len(lines) != 1 or lines[0]['t'] is None or begin is None or lines[0]['t'] < begin:
        return verdict(record, 1, 'bind line not exactly once, unstamped, or before restart-begin')
    t = lines[0]['t']
    record['bind_t'] = t
    if t < first[0]:
        return verdict(record, 0, 'bind line %.3f s before the first /health 200 request was sent '
                                  '(ready %.1f s after restart-begin, %d samples)' % (first[0] - t, first[1] - begin, len(rows)))
    if t <= first[1]:
        return verdict(record, 2, 'bind line inside the first 200 request (sent %.3f, bind %.3f, returned %.3f)'
                       % (first[0], t, first[1]))
    return verdict(record, 1, '/health 200 returned %.3f s BEFORE the bind line' % (t - first[1]))


def bind_epoch(a):
    receipt = json.loads(Path(a.bind).read_text())
    startup = json.loads(Path(a.startup).read_text())
    lines = receipt.get('bind_lines') or []
    record = dict(check='bind-epoch', schema='native39-trunk-checks/1',
                  bind_engine_epoch=lines[0]['engine_epoch'] if len(lines) == 1 else None,
                  startup_engine_epoch=startup.get('engine_epoch'), bind_verdict=receipt.get('verdict'))
    if record['bind_engine_epoch'] is None or type(record['startup_engine_epoch']) is not int:
        return verdict(record, 2, 'epoch unavailable')
    same = record['bind_engine_epoch'] == record['startup_engine_epoch']
    return verdict(record, 0 if same else 1, 'bind epoch %s, startup epoch %s' % (
        record['bind_engine_epoch'], record['startup_engine_epoch']))


def host(a):
    record = dict(check='effective-host', schema='native39-trunk-checks/1', identities=[])
    ok = True
    for path in a.identity:
        data = json.loads(Path(path).read_text())
        args = (data.get('identity') or {}).get('command', {}).get('args') or []
        hosts = [args[j + 1] for j, x in enumerate(args[:-1]) if x == '--host']
        extra = ((data.get('identity') or {}).get('defaults') or {}).get('EXTRA_ARGS')
        row = dict(path=str(path), host_args=hosts, effective_host=hosts[-1] if hosts else None,
                   defaults_extra_args=extra)
        row['ok'] = row['effective_host'] == LOOPBACK and extra == a.expected_extra_args.split()
        ok = ok and row['ok']
        record['identities'].append(row)
    detail = 'effective --host ' + ','.join(str(r['effective_host']) for r in record['identities'])
    if a.q:
        served = [dict(file=name, t=stamp(line), host=m.group(1), port=int(m.group(2)))
                  for name, line, m in log_matches(Path(a.q), SERVED_RE)]
        record['served'] = served
        ok = ok and len(served) == 1 and served[0]['host'] == LOOPBACK
        detail += '; served ' + ','.join('%s:%s' % (s['host'], s['port']) for s in served) if served else '; no served line'
    return verdict(record, 0 if ok else 1, detail)


def found_extra_args(a):
    # engine_identity's EnvironmentFile parser (the declaration's own); the launcher word-splits the value.
    import engine_identity
    value = engine_identity.defaults_snapshot(Path(a.defaults).read_bytes(), redact=False)['values'].get('EXTRA_ARGS')
    record = dict(check='found-extra-args', schema='native39-trunk-checks/1', expected=a.expected,
                  matched=value is not None and value.split() == a.expected.split())  # the found value is never written
    return verdict(record, 0 if record['matched'] else 1, 'found EXTRA_ARGS equals the declared production value'
                   if record['matched'] else 'found EXTRA_ARGS differs')


def metrics(url):
    text = urllib.request.urlopen(url.rstrip('/') + '/metrics', timeout=10).read().decode()
    values = {}
    for name in ('num_requests_running', 'num_requests_waiting'):
        nums = [float(line.split()[-1]) for line in text.splitlines() if re.match(r'^vllm:' + name + r'(\{| )', line)]
        if not nums:
            raise ValueError('metric missing: ' + name)
        values[name] = sum(nums)
    return values


def idle(a):
    record = dict(check='engine-idle', schema='native39-trunk-checks/1', samples=[])
    deadline = time.monotonic() + a.timeout_s
    while True:
        try:
            m = metrics(a.url)
            record['samples'].append(dict(t=time.time(), **m))
            if m['num_requests_running'] == 0 and m['num_requests_waiting'] == 0:
                return verdict(record, 0, 'running 0, waiting 0 after %d samples' % len(record['samples']))
        except (OSError, ValueError) as exc:
            record['samples'].append(dict(t=time.time(), error=type(exc).__name__))
        if time.monotonic() >= deadline:
            return verdict(record, 2, 'engine not idle within %s s' % a.timeout_s)
        time.sleep(1)


def load(path):
    p = Path(path)
    if not p.is_file():
        return None, dict(path=str(p), present=False)
    data = p.read_bytes()
    try:
        return json.loads(data), dict(path=str(p), present=True, sha256=hashlib.sha256(data).hexdigest())
    except ValueError:
        return None, dict(path=str(p), present=True, readable=False)


def ratio(a, b):
    ok = all(type(v) in (int, float) and not isinstance(v, bool) for v in (a, b))
    return a / b if ok and b > 0 else None


def summary(a):
    q = Path(a.q)
    grade, grade_ref = load(q / 'native35-performance-grade.json')
    plain, plain_ref = load(q / 'w1-plain.json')
    record = dict(check='trunk-summary', schema='native39-trunk-summary/1', target_is_gate=False,
                  inputs=dict(grade=grade_ref, plain_w1=plain_ref))
    for name in ('bind-receipt', 'bind-epoch', 'effective-host', 'w2-length', 'w1-plain-idle'):
        data, _ = load(q / (name + '.json'))
        record[name.replace('-', '_')] = None if data is None else {k: data.get(k) for k in ('verdict', 'detail', 'failures')}
    windows = [w.get('agg_tps') for w in (plain or {}).get('windows') or []]
    plain_median = statistics.median(windows) if windows and all(type(v) in (int, float) for v in windows) else None
    record['plain_w1'] = dict(verdict=(plain or {}).get('verdict'), window_agg_tps=windows, median=plain_median,
                              client='w1_trace.py (q18b-w1ab2), 8 x 153537, 3 x 20 s')
    native_full = (((grade or {}).get('w1') or {}).get('routes') or {}).get('full') or {}
    record['native_w1_full'] = dict(by_repeat=native_full.get('aggregate_tokens_per_s_by_repeat'),
                                    median=native_full.get('median'),
                                    denominator=(native_full.get('grade_denominator') or {}).get('decode_tokens_per_s'))
    record['native_w1_full_over_plain_w1'] = ratio(record['native_w1_full']['denominator'], plain_median)
    rounds = []
    for index, view in enumerate((grade or {}).get('rounds') or []):
        raw = (view or {}).get('raw') or {}
        decode = (((raw.get('decode_during_prefill') or {}).get('fresh')) or {}).get('aggregate_tokens_per_s')
        rounds.append(dict(round=index, verdict=(view or {}).get('verdict'), headline=(view or {}).get('headline'),
                           decode_tokens_per_s_during_fresh_prefill=decode,
                           decode_share_of_native_w1_full=raw.get('decode_share_of_w1_full_median'),
                           decode_share_of_plain_w1_median=ratio(decode, plain_median),
                           prefill_share_overlap_over_split_solo=raw.get('prefill_share_overlap_over_split_solo'),
                           missing=(view or {}).get('missing')))
    record['rounds'] = rounds
    record['grade_verdict'] = (grade or {}).get('verdict')
    # A report, never a verdict of its own: absent inputs are visible above and in their own phases.
    return verdict(record, 0, 'descriptive trunk reading; grade %s' % record['grade_verdict'])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='action', required=True)
    b = sub.add_parser('bind')
    b.add_argument('--q', required=True)
    b.add_argument('--container-id', required=True)
    b.add_argument('--wait-s', type=float, default=30.0)
    b.add_argument('--out', required=True)
    e = sub.add_parser('bind-epoch')
    e.add_argument('--bind', required=True)
    e.add_argument('--startup', required=True)
    e.add_argument('--out', required=True)
    h = sub.add_parser('host')
    h.add_argument('--identity', action='append', required=True)
    h.add_argument('--expected-extra-args', required=True)
    h.add_argument('--q', help='run directory whose followed container log must name the served host')
    h.add_argument('--out', required=True)
    f = sub.add_parser('found-extra-args')
    f.add_argument('--defaults', required=True)
    f.add_argument('--expected', required=True)
    f.add_argument('--out', required=True)
    i = sub.add_parser('idle')
    i.add_argument('--url', required=True)
    i.add_argument('--timeout-s', type=float, default=120.0)
    i.add_argument('--out', required=True)
    s = sub.add_parser('summary')
    s.add_argument('--q', required=True)
    s.add_argument('--out', required=True)
    a = p.parse_args(argv)
    action = dict(bind=bind, host=host, idle=idle, summary=summary)
    action['bind-epoch'] = bind_epoch
    action['found-extra-args'] = found_extra_args
    try:
        record = action[a.action](a)
    except Exception as exc:  # noqa: BLE001 - classified HARNESS, never retried
        record = verdict(dict(check=a.action, schema='native39-trunk-checks/1', error=type(exc).__name__,
                              classification='HARNESS'), 2, 'check unavailable: ' + type(exc).__name__)
    return write(a.out, record)


if __name__ == '__main__':
    sys.exit(main())
