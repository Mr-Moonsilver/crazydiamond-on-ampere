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
instruments  Actual observer/offers state per rank (the speed engine runs with
             FLASH_S5_REPLAY_OBSERVER=0 and FLASH_S5_ENGINE_OFFERS=0): rank
             pids from a paused all-rank barrier (then resumed), and
             `docker exec <container> cat /proc/<pid>/maps` must not map
             flash_s5_replay_observer on any rank (FAIL if it does), must map
             libcuda (else INCONCLUSIVE: not the worker); unreadable maps are
             INCONCLUSIVE.  --same-as: the same pids and incarnations as an
             earlier receipt (FAIL if not).
full-idle    Before the plain W1: a typed all-rank FULL (session prepare full,
             full-prefill-solo, typed TransitionAcks), resumed; /metrics
             running + waiting 0; status route full, unpaused, no transition
             or control pending, scheduler_requests 0; a paused barrier with
             every rank on full with no live or prefill request, then resumed.
             Unproven is INCONCLUSIVE, a contradiction FAIL.  The plain W1 runs
             only on PASS.
route-check  After the plain W1: status route full at the full-idle receipt's
             route generation (no transition inside the plain W1), unpaused.
ingress-w1   The followed access log (uvicorn lines, docker-stamped) over each
             native W1 repeat's wall span: every line from 127.0.0.1, only
             generation or accounted quiet/control paths, no 401, and the
             generation lines' client ports exactly the repeat's 8 cohort
             ports; no generation line between repeats.  Any breach: FAIL
             (measurement INVALID).
ingress      The same for one span: the plain W1 (--start/--end from the queue,
             --generation 8) or a W2 round (--round: its wall span, and the
             generation requests the record shows it sent; unknown count:
             INCONCLUSIVE).
returns      Every split->FULL return (W1 split repeats, W2 native lanes, and
             cleanups): per-rank typed acks (applied generation/bank, drained
             streams, watermark), the all-rank ack time, the census-to-ack and
             (W1) client-send-to-ack intervals, graded descriptively against
             row1 1.5 ms and row4 42.76 s.  Per-rank send/arrival/
             before_route_change/applied stamps are not exposed by this engine
             and are listed as missing.  No observation window at a return:
             every instruments receipt PASS with the same rank pids, and the
             return's census names those pids.
summary      Trunk reading beside native35_performance_grade: per W2 round the
             decode tok/s during the fresh prefill, its share of the native W1
             FULL median (the grade's denominator) and of the plain W1 median,
             the prefill share, plus the checks above.  The plain median and
             its ratios exist only when the plain W1 is valid: verdict PASS, no
             errors, 3 windows each with 8 live streams, full-idle, route-check,
             ingress-plain and instruments-w1 PASS; otherwise the raw windows
             are kept under raw_invalid_samples and every plain ratio is None.
             Never a gate; 80/80 stays a marker; always rc 0 once written.
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


# -- native39 blockers (Codex queue review of 1c334bc) -------------------------

OBSERVER_SO = 'flash_s5_replay_observer'
RANKS = (0, 1, 2, 3)
ROW1_RETURN_MS = 1.5          # row1 FULL return, descriptive band edge (lead)
ROW4_RETURN_MS = 42760.0      # row4 FULL return 42.76 s, descriptive band edge (lead)
ACCESS_RE = re.compile(r'INFO:\s+(\[?[0-9A-Fa-f.:]+\]?):(\d+) - "(\w+) (\S+) HTTP/[\d.]+" (\d{3})')
GENERATION_PATHS = ('/v1/chat/completions', '/v1/completions', '/v1/responses', '/v1/embeddings',
                    '/pooling', '/classify', '/score', '/v1/score', '/rerank', '/v1/rerank', '/v2/rerank',
                    '/invocations', '/v1/audio/transcriptions', '/v1/audio/translations')
QUIET_PATHS = ('/health', '/metrics', '/tokenize', '/detokenize', '/v1/models', '/version', '/ping')
INGRESS_LAG_S = 0.25          # docker stamps a line when it reads it: late, never early


def session_client(a):
    from e3_native_runtime_client import NativeRuntimeClient
    return NativeRuntimeClient(a.url, a.run_id, epoch=a.epoch, timeout_s=a.control_timeout_s)


def exc_text(exc):
    return '%s: %s' % (type(exc).__name__, str(exc)[:200])


def docker_maps(container, pid, timeout_s):
    import subprocess
    done = subprocess.run(['docker', 'exec', container, 'cat', '/proc/%d/maps' % pid],
                          capture_output=True, text=True, timeout=timeout_s)
    if done.returncode != 0:
        raise OSError('docker exec rc=%d' % done.returncode)
    return done.stdout


def instruments(a):
    """Actual per-rank observer/offers state: the replay observer library is not mapped in any rank.

    start_worker_observer returns None when FLASH_S5_REPLAY_OBSERVER=0 (observer_runtime.py) and
    ReplayBinding is the only loader of the library; an engine offers ledger is built only by
    an observer (WorkerObserver._begin_ledger) or with an observer session.  So a rank process
    without the library mapped has no observer, no ledger and no observation window.
    """
    record = dict(check='instruments-off', schema='native39-trunk-checks/1', ranks={},
                  basis='docker exec <container> cat /proc/<rank pid>/maps; rank pids from a paused all-rank barrier')
    client = session_client(a)
    try:
        rows = client.workers(client.pause().get('barrier'))
    finally:
        try:
            record['resume'] = client.resume().get('resumed')
        except Exception as exc:  # noqa: BLE001 - recorded; the verdict below stays unproven
            record['resume_error'] = exc_text(exc)
    if 'resume_error' in record:
        return verdict(record, 2, 'engine not resumed after the instrument barrier')
    if set(rows) != set(RANKS):
        return verdict(record, 2, 'rank coverage')
    mapped, unread = [], []
    for rank in RANKS:
        row = rows[rank]
        pid = row.get('pid')
        out = dict(pid=pid, worker_incarnation=row.get('worker_incarnation'), route=row.get('route'))
        try:
            if type(pid) is not int or pid < 1:
                raise ValueError('pid unavailable')
            text = docker_maps(a.container_id, pid, a.exec_timeout_s)
            out.update(maps_sha256=hashlib.sha256(text.encode()).hexdigest(), maps_lines=text.count('\n'),
                       observer_mapped=OBSERVER_SO in text, cuda_mapped='libcuda.so' in text,
                       cupti_mapped='libcupti' in text)
            if not out['cuda_mapped']:
                unread.append(rank)  # not proven to be the CUDA worker
            if out['observer_mapped']:
                mapped.append(rank)
        except Exception as exc:  # noqa: BLE001
            out['error'] = exc_text(exc)
            unread.append(rank)
        record['ranks'][str(rank)] = out
    if a.same_as:
        prior = json.loads(Path(a.same_as).read_text())
        same = all((prior.get('ranks') or {}).get(str(r), {}).get(k) == record['ranks'][str(r)].get(k)
                   for r in RANKS for k in ('pid', 'worker_incarnation'))
        record.update(same_as=str(a.same_as), same_processes=same, same_as_verdict=prior.get('verdict'))
        if not same:
            return verdict(record, 1, 'rank pids or incarnations differ from ' + Path(a.same_as).name)
        if prior.get('verdict') != 'PASS':
            return verdict(record, 2, 'the reference instruments receipt is not PASS')
    if mapped:
        return verdict(record, 1, 'replay observer mapped on rank(s) %s' % ','.join(map(str, mapped)))
    if unread:
        return verdict(record, 2, 'maps unreadable or not a CUDA worker on rank(s) %s' % ','.join(map(str, unread)))
    return verdict(record, 0, 'replay observer not mapped on ranks 0-3 (libcupti mapped: %s; descriptive)'
                   % ','.join(str(r) for r in RANKS if record['ranks'][str(r)]['cupti_mapped']))


def status_view(status):
    keys = ('route', 'route_generation', 'paused', 'transition_pending', 'control_pending',
            'return_pending', 'scheduler_requests', 'observed_clock_ns')
    return {k: status.get(k) for k in keys} if type(status) is dict else None


def full_idle(a):
    """Typed all-rank FULL, resumed, idle and empty before the plain W1 (Codex blocker 1)."""
    record = dict(check='full-idle', schema='native39-trunk-checks/1', samples=[])
    client = session_client(a)
    try:
        prep = client.prepare('full', 'full-prefill-solo')
    except Exception as exc:  # noqa: BLE001 - the typed FULL is not proven
        record['prepare_error'] = exc_text(exc)
        return verdict(record, 2, 'typed all-rank FULL not proven: ' + record['prepare_error'])
    record['prepare'] = dict(route=prep.get('route'), typed_acks=prep.get('typed_acks'),
                             transition_acks=prep.get('transition_acks'), resumed=prep.get('resumed'))
    deadline = time.monotonic() + a.timeout_s
    while True:
        try:
            m = metrics(a.url)
            record['samples'].append(dict(t=time.time(), **m))
            if m['num_requests_running'] == 0 and m['num_requests_waiting'] == 0:
                break
        except (OSError, ValueError) as exc:
            record['samples'].append(dict(t=time.time(), error=type(exc).__name__))
        if time.monotonic() >= deadline:
            return verdict(record, 2, 'engine not idle within %s s after the typed FULL' % a.timeout_s)
        time.sleep(1)
    status = status_view(client.call('status'))
    record['status'] = status
    try:
        rows = client.workers(client.pause().get('barrier'))
    finally:
        record['resume'] = client.resume().get('resumed')
    record['workers'] = {str(r): {k: rows[r].get(k) for k in ('route', 'route_generation', 'live_decode_requests',
                                                              'prefill_requests', 'running_requests', 'pid',
                                                              'worker_incarnation')} for r in sorted(rows)}
    after = status_view(client.call('status'))
    record['status_after'] = after
    contradictions = []
    if status.get('route') != 'full' or after.get('route') != 'full':
        contradictions.append('route-not-full')
    if status.get('transition_pending') is not False or status.get('control_pending') is not None:
        contradictions.append('transition-or-control-pending')
    if status.get('route_generation') != after.get('route_generation'):
        contradictions.append('route-generation-moved')
    if status.get('scheduler_requests') != 0 or status.get('paused') is not False:
        contradictions.append('scheduler-not-empty-or-paused')
    if any(w['route'] != 'full' or w['live_decode_requests'] != 0 or w['prefill_requests'] != 0
           for w in record['workers'].values()) or set(record['workers']) != {str(r) for r in RANKS}:
        contradictions.append('workers-not-empty-full')
    if after.get('paused') is not False:
        contradictions.append('not-resumed')
    record['route_generation'] = after.get('route_generation')
    if contradictions:
        return verdict(record, 1, 'typed FULL acked but ' + ','.join(contradictions))
    return verdict(record, 0, 'typed all-rank FULL (generation %s), resumed, running 0 waiting 0, workers empty'
                   % record['route_generation'])


def route_check(a):
    """The route stayed FULL at the same generation across the plain W1 (no transition inside it)."""
    record = dict(check='route-check', schema='native39-trunk-checks/1')
    ref = json.loads(Path(a.full_idle).read_text())
    record.update(full_idle_verdict=ref.get('verdict'), expected_generation=ref.get('route_generation'))
    if ref.get('verdict') != 'PASS' or type(ref.get('route_generation')) is not int:
        return verdict(record, 2, 'no PASS full-idle reference')
    status = status_view(session_client(a).call('status'))
    record['status'] = status
    ok = (status.get('route') == 'full' and status.get('route_generation') == ref['route_generation']
          and status.get('transition_pending') is False and status.get('paused') is False)
    return verdict(record, 0 if ok else 1, 'route %s generation %s (expected full %s)' % (
        status.get('route'), status.get('route_generation'), ref['route_generation']))


def access_lines(q):
    rows = []
    for name, line, m in log_matches(q, ACCESS_RE):
        path = m.group(4).split('?', 1)[0]
        kind = ('generation' if path in GENERATION_PATHS else
                'quiet' if path in QUIET_PATHS or path.startswith('/v1/flash_s5') else 'other')
        rows.append(dict(t=stamp(line), addr=m.group(1), port=int(m.group(2)), method=m.group(3),
                         path=path, code=m.group(5), kind=kind))
    return rows


def span(lines, start, end, expect, ports=None):
    """One harness span: every line loopback and accounted for; generation lines exactly the harness's."""
    sel = [r for r in lines if r['t'] is not None and start <= r['t'] <= end]
    gen = [r for r in sel if r['kind'] == 'generation']
    out = dict(start=start, end=end, access_lines=len(sel), generation_lines=len(gen), expected_generation=expect,
               by_path={}, invalid=[], unknown=[])
    for r in sel:
        out['by_path'][r['path']] = out['by_path'].get(r['path'], 0) + 1
    if any(r['addr'] != LOOPBACK for r in sel):
        out['invalid'].append('non-loopback-client')
    if any(r['kind'] == 'other' for r in sel):
        out['invalid'].append('unaccounted-path')
    if any(r['code'] == '401' for r in sel):
        out['invalid'].append('unauthenticated-request')
    if any(r['code'] != '200' for r in gen):
        out['invalid'].append('generation-not-200')
    if expect is None:
        out['unknown'].append('expected-generation-count-unknown')
    elif len(gen) != expect:
        out['invalid'].append('generation-count %d != %d' % (len(gen), expect))
    if ports is not None:
        from collections import Counter
        if any(p is None for p in ports):
            out['unknown'].append('harness-client-ports-unknown')
        elif Counter(r['port'] for r in gen) != Counter(ports):
            out['invalid'].append('generation-client-ports-differ')
    if not sel:
        out['unknown'].append('no-access-lines')  # the harness's own control lines are missing: not observable
    return out


def ingress_verdict(record, spans, lines_present):
    record['spans'] = spans
    if not lines_present:
        return verdict(record, 2, 'no followed engine log')
    invalid = sorted({'%s:%s' % (s['label'], r) for s in spans for r in s['invalid']})
    unknown = sorted({'%s:%s' % (s['label'], r) for s in spans for r in s['unknown']})
    record.update(measurement='INVALID' if invalid else 'UNPROVEN' if unknown else 'VALID',
                  invalid=invalid, unknown=unknown)
    if invalid:
        return verdict(record, 1, 'INVALID: ' + '; '.join(invalid))
    if unknown:
        return verdict(record, 2, 'unproven: ' + '; '.join(unknown))
    return verdict(record, 0, 'every request in %d span(s) is the harness\'s own (loopback, accounted paths, '
                   'exact generation lines)' % len(spans))


def followed(q):
    return any((q / n).is_file() for n in ('engine-stream.log', 'engine-stream.err'))


def ingress_w1(a):
    """Native W1: per repeat the 8 cohort client ports exactly, and no generation line between repeats."""
    q = Path(a.q)
    record = dict(check='ingress-w1', schema='native39-trunk-checks/1', log='Q/engine-stream.log + .err',
                  lag_s=INGRESS_LAG_S)
    data = json.loads(Path(a.w1).read_text())
    lines = access_lines(q)
    spans, repeats = [], [r for r in data.get('repeats') or [] if not r.get('not_run')]
    for i, row in enumerate(repeats):
        wall = row.get('wall') or {}
        start, end = wall.get('open_s'), wall.get('end_s')
        if type(start) not in (int, float) or type(end) not in (int, float):
            spans.append(dict(label='repeat%s' % row.get('repeat'), invalid=[], unknown=['wall-span-missing']))
            continue
        nxt = ((repeats[i + 1].get('wall') or {}).get('open_s') if i + 1 < len(repeats) else None)
        stop = end + INGRESS_LAG_S if type(nxt) not in (int, float) else min(end + INGRESS_LAG_S, nxt)
        ports = (row.get('cohort') or {}).get('client_ports')
        s = span(lines, start, stop, len(ports) if ports else None, ports)
        s['label'] = 'repeat%s' % row.get('repeat')
        spans.append(s)
    if repeats and all(type((r.get('wall') or {}).get(k)) in (int, float) for r in repeats for k in ('open_s', 'end_s')):
        whole = span(lines, repeats[0]['wall']['open_s'], repeats[-1]['wall']['end_s'] + INGRESS_LAG_S,
                     sum(s.get('generation_lines', 0) for s in spans))
        whole['label'] = 'w1-whole'
        whole['unknown'] = [u for u in whole['unknown'] if u != 'expected-generation-count-unknown']
        spans.append(whole)
    elif not repeats:
        spans.append(dict(label='w1', invalid=[], unknown=['no-attempted-repeat']))
    return ingress_verdict(record, spans, followed(q))


def round_expected(data):
    """Generation requests a W2 round sent: 2 per lane plus its standing decoders, when the record shows all."""
    lanes = data.get('lanes') or {}
    if data.get('first_error') or set(lanes) != {'prefill_full', 'prefill_split', 'overlap'}:
        return None
    sent = 0
    for lane, rows in lanes.items():
        for family in ('fresh', 'append'):
            row = rows.get(family) or {}
            if row.get('error') or type((row.get('request') or {}).get('send_at_ns')) is not int:
                return None
            sent += 1
    standing = {len(((lanes['overlap'].get(f) or {}).get('decode_window') or {}).get('streams') or [])
                for f in ('fresh', 'append')}
    if len(standing) != 1 or 0 in standing:
        return None
    return sent + standing.pop()


def ingress(a):
    """A plain W1 or W2 round span: loopback, accounted paths, exactly the harness's generation count."""
    q = Path(a.q)
    record = dict(check='ingress', label=a.label, schema='native39-trunk-checks/1', lag_s=INGRESS_LAG_S)
    start, end, expect = a.start, a.end, a.generation
    if a.round:
        data = json.loads(Path(a.round).read_text())
        start, end, expect = data.get('wall_anchor_s'), data.get('wall_end_s'), round_expected(data)
    if type(start) not in (int, float) or type(end) not in (int, float):
        return ingress_verdict(record, [dict(label=a.label, invalid=[], unknown=['span-missing'])], followed(q))
    s = span(access_lines(q), start, end + INGRESS_LAG_S, expect)
    s['label'] = a.label
    return ingress_verdict(record, [s], followed(q))


def band(ms):
    if type(ms) not in (int, float):
        return None
    if ms <= ROW1_RETURN_MS:
        return 'at-or-below-row1-1.5ms'
    return 'between-row1-and-row4' if ms <= ROW4_RETURN_MS else 'above-row4-42.76s'


MISSING_STAMPS = ('full_send', 'arrival', 'before_route_change_done', 'applied')
MISSING_WHY = ('per-rank FULL send/arrival/before_route_change/applied stamps are not exposed by this engine '
               '(they exist only in the native36-w witness branch, 50a4fed); adding them is an engine change')


def return_row(source, reply, clock=None, bracket=None):
    acks = (reply or {}).get('transition_acks') or []
    census = (((reply or {}).get('census_before_return') or {}).get('workers')) or []
    ack_ns = {a.get('observed_ns') for a in acks}
    ack_ns = ack_ns.pop() if len(ack_ns) == 1 else None
    census_ns = [w.get('observed_clock_ns') for w in census if type(w) is dict]
    last_census = max(census_ns) if census_ns and all(type(n) is int for n in census_ns) else None
    row = dict(source=source, route_after=(reply or {}).get('route'), typed_acks=(reply or {}).get('typed_acks'),
               ranks={str(a.get('worker_rank')): dict(applied_route_generation=a.get('applied_route_generation'),
                                                       applied_bank=a.get('applied_bank'),
                                                       drained_streams=a.get('drained_streams'),
                                                       decode_replay_watermark=a.get('decode_replay_watermark'),
                                                       **{k: None for k in MISSING_STAMPS}) for a in acks},
               all_rank_ack_observed_ns=ack_ns, census_before_return_last_ns=last_census,
               census_pids={str(w.get('worker_rank')): w.get('pid') for w in census if type(w) is dict},
               missing_per_rank=list(MISSING_STAMPS), missing_why=MISSING_WHY)
    row['census_to_all_rank_ack_ms'] = ((ack_ns - last_census) / 1e6 if type(ack_ns) is int
                                        and type(last_census) is int else None)
    if bracket:
        row['client_round_trip_ms'] = (bracket['reply_ns'] - bracket['send_ns']) / 1e6
        same = (type(clock) is dict and type(bracket.get('clock')) is dict
                and clock.get('boot_id') and clock.get('boot_id') == bracket['clock'].get('boot_id')
                and clock.get('time_namespace') == bracket['clock'].get('time_namespace'))
        row['clock_shared'] = bool(same)
        row['client_send_to_all_rank_ack_ms'] = ((ack_ns - bracket['send_ns']) / 1e6
                                                 if same and type(ack_ns) is int else None)
    graded = row.get('client_send_to_all_rank_ack_ms')
    if graded is None:
        graded = row['census_to_all_rank_ack_ms']
    row.update(graded_ms=graded, band=band(graded))
    return row


def returns(a):
    """Every split->FULL return: per-rank typed acks and the all-rank ack, graded descriptively."""
    q = Path(a.q)
    record = dict(check='returns', schema='native39-trunk-returns/1', descriptive=True,
                  bands_ms=dict(row1=ROW1_RETURN_MS, row4=ROW4_RETURN_MS), rows=[])
    w1, _ = load(q / 'w1.json')
    for row in (w1 or {}).get('repeats') or []:
        rec = row.get('native_receipts') or {}
        if row.get('route') == 'split' and rec.get('leave_full'):
            clock = ((rec.get('end') or {}).get('snapshot') or {}).get('clock')
            record['rows'].append(return_row('w1:repeat%s' % row.get('repeat'), rec['leave_full'], clock,
                                             row.get('return_bracket')))
        cleanup = (row.get('cleanup') or {}).get('leave_full')
        if cleanup:
            record['rows'].append(return_row('w1:repeat%s:cleanup' % row.get('repeat'), cleanup))
    for index in (0, 1):
        data, _ = load(q / ('overlap-round-%d.json' % index))
        for lane, rows in sorted(((data or {}).get('lanes') or {}).items()):
            for family, rec in sorted((rows or {}).items()):
                if lane == 'prefill_full' or type(rec) is not dict:
                    continue
                for key, reply in (('return', rec.get('return_full')),
                                   ('cleanup', (rec.get('cleanup') or {}).get('leave_full'))):
                    if reply:
                        record['rows'].append(return_row('round%d:%s:%s:%s' % (index, lane, family, key), reply))
    # No observation window at any return: the rank processes had no observer (instrument receipts),
    # and each return's census names those same pids.
    receipts = {}
    for path in sorted(q.glob('instruments-*.json')):
        data, _ = load(path)
        receipts[path.stem] = data or {}
    record['instruments'] = {k: v.get('verdict') for k, v in receipts.items()}
    pids = {json.dumps({r: (v.get('ranks') or {}).get(r, {}).get('pid') for r in map(str, RANKS)}, sort_keys=True)
            for v in receipts.values()}
    ok = bool(receipts) and all(v.get('verdict') == 'PASS' for v in receipts.values()) and len(pids) == 1
    ref = json.loads(pids.pop()) if ok else {}
    for row in record['rows']:
        same = not row['census_pids'] or all(ref.get(r) == p for r, p in row['census_pids'].items())
        row['no_observation_window'] = bool(ok and same)
    record['no_observation_window_at_every_return'] = all(r['no_observation_window'] for r in record['rows'])
    if not record['rows']:
        return verdict(record, 2, 'no split->FULL return recorded')
    if any(v.get('verdict') == 'FAIL' for v in receipts.values()):
        return verdict(record, 1, 'an instruments receipt FAILed: an observer was mapped or the ranks changed')
    if not record['no_observation_window_at_every_return']:
        return verdict(record, 2, 'no-observation-window not proven at every return')
    graded = [r['graded_ms'] for r in record['rows'] if type(r['graded_ms']) in (int, float)]
    return verdict(record, 0, '%d returns, no observation window at any; graded ms min %s max %s (descriptive)' % (
        len(record['rows']), min(graded) if graded else None, max(graded) if graded else None))


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


def plain_validity(q, plain):
    """The plain W1 is an accepted denominator only when every condition holds (Codex item 4)."""
    reasons = []
    if (plain or {}).get('verdict') != 'PASS':
        reasons.append('plain-verdict-%s' % (plain or {}).get('verdict'))
    if (plain or {}).get('errors') or (plain or {}).get('fails'):
        reasons.append('plain-errors')
    windows = (plain or {}).get('windows') or []
    if len(windows) != 3:
        reasons.append('windows-%d-of-3' % len(windows))
    for w in windows:
        tps, per = w.get('agg_tps'), w.get('per_stream') or []
        if type(tps) not in (int, float) or isinstance(tps, bool) or tps <= 0:
            reasons.append('window%s-agg-tps' % w.get('repeat'))
        if len(per) != 8 or any(type(r.get('tps')) not in (int, float) or r['tps'] <= 0 for r in per):
            reasons.append('window%s-eight-live-streams' % w.get('repeat'))
    if (plain or {}).get('streams') != 8:
        reasons.append('streams')
    for name in ('w1-plain-full-idle', 'w1-plain-route', 'ingress-plain', 'instruments-w1'):
        data, _ = load(q / (name + '.json'))
        if (data or {}).get('verdict') != 'PASS':
            reasons.append('%s-%s' % (name, (data or {}).get('verdict', 'missing')))
    return reasons


def summary(a):
    q = Path(a.q)
    grade, grade_ref = load(q / 'native35-performance-grade.json')
    plain, plain_ref = load(q / 'w1-plain.json')
    record = dict(check='trunk-summary', schema='native39-trunk-summary/2', target_is_gate=False,
                  inputs=dict(grade=grade_ref, plain_w1=plain_ref))
    receipts = ('bind-receipt', 'bind-epoch', 'effective-host', 'w2-length', 'w1-plain-idle', 'w1-plain-full-idle',
                'w1-plain-route', 'instruments-w1', 'instruments-round-0', 'instruments-round-1', 'instruments-end',
                'ingress-w1', 'ingress-plain', 'ingress-round-0', 'ingress-round-1', 'returns')
    for name in receipts:
        data, _ = load(q / (name + '.json'))
        record[name.replace('-', '_')] = None if data is None else {k: data.get(k) for k in ('verdict', 'detail', 'failures')}
    def ok(name):
        return (record.get(name.replace('-', '_')) or {}).get('verdict') == 'PASS'
    windows = [w.get('agg_tps') for w in (plain or {}).get('windows') or []]
    invalid = plain_validity(q, plain)
    accepted = not invalid
    plain_median = statistics.median(windows) if accepted else None
    record['plain_w1'] = dict(accepted=accepted, invalid_reasons=invalid, median=plain_median,
                              window_agg_tps=windows if accepted else None,
                              raw_invalid_samples=None if accepted else dict(
                                  verdict=(plain or {}).get('verdict'), window_agg_tps=windows,
                                  note='raw samples of an invalid plain W1; never a denominator'),
                              client='w1_trace.py (q18b-w1ab2), 8 x 153537, 3 x 20 s')
    native_full = (((grade or {}).get('w1') or {}).get('routes') or {}).get('full') or {}
    w1_clean = ok('instruments-w1') and ok('ingress-w1')
    record['native_w1_full'] = dict(by_repeat=native_full.get('aggregate_tokens_per_s_by_repeat'),
                                    median=native_full.get('median'), instruments_off_and_sealed=w1_clean,
                                    denominator=(native_full.get('grade_denominator') or {}).get('decode_tokens_per_s'))
    record['native_w1_full_over_plain_w1'] = ratio(record['native_w1_full']['denominator'], plain_median)
    rounds = []
    for index, view in enumerate((grade or {}).get('rounds') or []):
        raw = (view or {}).get('raw') or {}
        decode = (((raw.get('decode_during_prefill') or {}).get('fresh')) or {}).get('aggregate_tokens_per_s')
        clean = w1_clean and ok('instruments-round-%d' % index) and ok('ingress-round-%d' % index)
        rounds.append(dict(round=index, verdict=(view or {}).get('verdict'), headline=(view or {}).get('headline'),
                           instruments_off_and_sealed=clean,
                           decode_tokens_per_s_during_fresh_prefill=decode,
                           decode_share_of_native_w1_full=raw.get('decode_share_of_w1_full_median'),
                           decode_share_of_plain_w1_median=ratio(decode, plain_median),
                           prefill_share_overlap_over_split_solo=raw.get('prefill_share_overlap_over_split_solo'),
                           missing=(view or {}).get('missing')))
    record['rounds'] = rounds
    record['clean_speed_reading'] = bool(rounds) and all(r['instruments_off_and_sealed'] for r in rounds) \
        and ok('instruments-end')
    record['grade_verdict'] = (grade or {}).get('verdict')
    # A report, never a verdict of its own: absent inputs are visible above and in their own phases.
    return verdict(record, 0, 'descriptive trunk reading; grade %s; plain W1 %s; clean %s' % (
        record['grade_verdict'], 'accepted' if accepted else 'not accepted', record['clean_speed_reading']))

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
    for name in ('instruments', 'full-idle', 'route-check'):
        c = sub.add_parser(name)
        c.add_argument('--url', required=True)
        c.add_argument('--run-id', required=True)
        c.add_argument('--epoch', type=int, required=True)
        c.add_argument('--control-timeout-s', type=float, default=120.0)
        c.add_argument('--out', required=True)
        if name == 'instruments':
            c.add_argument('--container-id', required=True)
            c.add_argument('--same-as')
            c.add_argument('--exec-timeout-s', type=float, default=30.0)
        elif name == 'full-idle':
            c.add_argument('--timeout-s', type=float, default=120.0)
        else:
            c.add_argument('--full-idle', required=True)
    g = sub.add_parser('ingress-w1')
    g.add_argument('--q', required=True)
    g.add_argument('--w1', required=True)
    g.add_argument('--out', required=True)
    g = sub.add_parser('ingress')
    g.add_argument('--q', required=True)
    g.add_argument('--label', required=True)
    g.add_argument('--start', type=float)
    g.add_argument('--end', type=float)
    g.add_argument('--generation', type=int)
    g.add_argument('--round', help='a W2 round record: its wall span and the generation requests it sent')
    g.add_argument('--out', required=True)
    r = sub.add_parser('returns')
    r.add_argument('--q', required=True)
    r.add_argument('--out', required=True)
    a = p.parse_args(argv)
    action = dict(bind=bind, host=host, idle=idle, summary=summary, instruments=instruments, ingress=ingress,
                  returns=returns)
    action['bind-epoch'] = bind_epoch
    action['found-extra-args'] = found_extra_args
    action['full-idle'] = full_idle
    action['route-check'] = route_check
    action['ingress-w1'] = ingress_w1
    try:
        record = action[a.action](a)
    except Exception as exc:  # noqa: BLE001 - classified HARNESS, never retried
        record = verdict(dict(check=a.action, schema='native39-trunk-checks/1', error=type(exc).__name__,
                              classification='HARNESS'), 2, 'check unavailable: ' + type(exc).__name__)
    return write(a.out, record)


if __name__ == '__main__':
    sys.exit(main())
