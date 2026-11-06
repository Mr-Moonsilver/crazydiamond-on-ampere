#!/usr/bin/env python3
"""q18a-compose-throughput box arms and grade: compose-4's box over this table (harness/compose5_arms.py). Discovery mode. dnr added a per-row replay seed and the cap off; hold added the per-row native_hold, acknowledged;
coverage-residue adds the per-row native_residue, acknowledged, keeps the engine's decline record (with its prefill coverage counter) in each status row, and grades coverage per replay arm.
coverage-soak adds, per replay arm: the requests still open when the replay gave up on them, by agent, session and send time (a hang, named), and
the engine's memory at the traffic's start and end (per-rank free VRAM by nvidia-smi, the engine container's process RSS by docker top), read on
the box host, never raising; the grade flags growth on nr, it does not fail on it.

Every arm kind compose-5 runs, compose-4's box already runs: the live arm for sm, fon and foff, the replay arm for cmp and ny, the one-stream arm. Only the table and the names differ, so the table is bound in here
and nothing that ran is edited. The rule is compose5_rule.py on the fetched record; this grade only says the record is whole.
"""
import os
import re
import subprocess
import threading
import time
from contextlib import contextmanager
from functools import wraps

import compose5_arms as T
import e3_compose4_box as _box

ARM_SCHEMA = 'compose5-arm/1'
GRADE_SCHEMA = 'compose5-grade/1'
SCOPE = 'the yield priced under the closed-loop replay and in a frozen pair, on one engine: crash, OOM, stock prefill and the price read by compose5_rule.py; not a release assessment and not a numerics result'
NOT_COVERED = _box.NOT_COVERED
replay = _box.replay
_report = _box.report


def report(result):
    text = _report(result).replace('q18a-compose-4', 'q18a-compose-throughput', 1).replace('compose4_rule.py', 'compose5_rule.py')
    # coverage-soak-fix: the geometry-tail class and the worker RSS read, one line per replay arm (the decode gap is the reader's)
    extra = []
    for label, cov in sorted((result.get('coverage') or {}).items()):
        geo = (cov or {}).get('geometry_tail')
        if geo:
            extra.append('- %s geometry-tail: turns %d, rows %d (%d live), max r %s, max rows %s (engine life)'
                         % (label, geo['turns'], geo['rows'], geo['live_rows'], geo['max_r'], geo['max_rows']))
            # coverage-throughput build-2 (c) (lead, 6 Nov): the tail rows and their share next to native share, every reading
            total = sum(sum(((cov.get('rows') or {}).get(s) or {}).get(p, {}).values()) for s in ('live', 'idle') for p in ('native', 'stock'))
            extra.append('- %s native share %s (strict %s) | geometry-tail rows %d of %d prefill rows, share %s'
                         % (label, cov.get('all_native_share'), cov.get('all_native_share_strict'), geo['rows'], total,
                            round(geo['rows'] / total, 6) if total else None))
    for label, row in sorted((result.get('worker_rss') or {}).items()):
        extra.append('- %s worker RSS: %s; native appends %s, history %s tokens'
                     % (label, ', '.join(row['fails']) or 'no failure', row['native_appends'], row['append_history_tokens']))
    return text + ('\n' + '\n'.join(extra) + '\n' if extra else '')


def __getattr__(name):
    """compose-4's box by its names, for a caller that only reads them."""
    return getattr(_box, name)


_status_row = _box.status_row


def status_row(tag, reply):
    """compose-4's status row, and the engine's decline record beside it: a1's yield_view keeps a fixed list of keys and drops it, so dnr's
    records had no decline counters. Bounded counts, no request UID; the coverage counter is in it."""
    row = _status_row(tag, reply)
    source = reply.get('append_yield') if type(reply) is dict else None
    if type(row.get('append_yield')) is dict and type(source) is dict:
        row['append_yield']['declines'] = source.get('declines')
    return row


@contextmanager
def _bound():
    """This table and these names in compose-4's box for one call, and back: a suite that runs both boxes in one process sees each with its own."""
    names = dict(T=T, ARM_SCHEMA=ARM_SCHEMA, GRADE_SCHEMA=GRADE_SCHEMA, SCOPE=SCOPE, report=report, status_row=status_row)
    saved = {name: getattr(_box, name) for name in names}
    # coverage-throughput: a1's yield_control confirms the engine's bounds against its own table's LAUNCH; this table raises max_new.
    a1_table = _box.a1.T
    a1_launch = a1_table.LAUNCH
    for name, value in names.items():
        setattr(_box, name, value)
    a1_table.LAUNCH = T.LAUNCH
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(_box, name, value)
        a1_table.LAUNCH = a1_launch


def _under_this_table(fn):
    @wraps(fn)
    def call(*args, **kwargs):
        with _bound():
            return fn(*args, **kwargs)
    return call


_grade, _smoke, _replay_arm = _box.grade, _box.smoke_check, _box.replay_arm


def cap_selected(rec):
    """What each rank's lane_chunk answered at the arm's select: requested is the row's flag, and the arm's flushes delta reads 0 with the cap on too (c4b). None if a rank did not answer."""
    select = (rec.get('reserved_cap') or {}).get('select') or {}
    states = [((select.get(str(rank)) or {}).get('after') or {}).get('reserved_cap') for rank in range(4)]
    if not all(type(state) is dict and type(state.get('selected')) is bool for state in states):
        return None
    return any(state['selected'] or state.get('cap_bytes') for state in states)


COVERAGE_SCHEMA = 's5-prefill-coverage/1'
MEMORY_SCHEMA = 'soak-memory/1'
CONTAINER = os.environ.get('Q17_TEST_CONTAINER') or 'qwen-flash-prod'      # the draft's engine container (its CONTAINER)
GPU_QUERY = ['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits']
VRAM_GROWTH_MIB = 64          # flagged: a rank's free VRAM down by this much from an nr arm's start to its settled end, or first nr start to last
SETTLE_STEP_S, SETTLE_TRIES = 2.0, 5   # the residue run's samples: in use rises ~1 GiB inside an arm and falls back when the lane goes idle (the flush)
RSS_GROWTH_KIB = 256 * 1024   # flagged: the engine container's process RSS up by this much
# coverage-soak-fix (lead, 6 Nov 06:25): per Worker_PP, an nr arm FAILS when its RSS grows by WORKER_RSS_FAIL_KIB, or by
# WORKER_RSS_FAIL_B_PER_TOKEN bytes or more per token of native-append history in the arm (the soak's leak: about 280).
WORKER_RSS_FAIL_KIB = 256 * 1024
WORKER_RSS_FAIL_B_PER_TOKEN = 32
WORKER_COMM = 'VLLM::Worker'
T_GEOMETRY_MAX = 64            # the engine's GEOMETRY_TAIL_MAX (scheduler_hooks, wire_v1): the most rows a geometry tail hands stock
# the /proc/<pid>/status fields that tell the leak's class (owner hunt, 6 Nov): heap (RssAnon, VmData), pinned (VmLck, VmPin),
# shared (RssShmem), plus the mapping count (lines of /proc/<pid>/maps)
STATUS_FIELDS = ('VmRSS', 'RssAnon', 'RssFile', 'RssShmem', 'VmLck', 'VmPin', 'VmData')


def worker_status(pid, proc='/proc', opener=open):
    """One Worker_PP's /proc status fields (KiB) and its mapping count. Read only; a failed read is an error in the row."""
    out = dict(pid=pid, errors=[])
    try:
        with opener('%s/%d/status' % (proc, pid)) as f:
            for line in f:
                key, _, value = line.partition(':')
                if key in STATUS_FIELDS:
                    parts = value.split()
                    out[key.lower() + '_kib'] = int(parts[0]) if parts and parts[0].isdigit() else None
    except Exception as exc:  # noqa: BLE001 - a read, never the arm's failure
        out['errors'].append('status:' + type(exc).__name__)
    try:
        with opener('%s/%d/maps' % (proc, pid)) as f:
            out['maps'] = sum(1 for _ in f)
    except Exception as exc:  # noqa: BLE001
        out['maps'] = None
        out['errors'].append('maps:' + type(exc).__name__)
    return out


def engine_memory(run=subprocess.run, clock=time.time, status=worker_status):
    """Per-rank VRAM and the engine container's process RSS, now; per Worker_PP its pid, RSS and /proc status (coverage-soak-fix).
    Read only; never raises: a failed read is an error in the row."""
    out = dict(schema=MEMORY_SCHEMA, wall_s=clock(), gpus=None, rss_kib=None, processes=None, top=None, workers=None, errors=[])
    try:
        got = run(GPU_QUERY, capture_output=True, text=True, timeout=20)
        gpus = {}
        for line in (got.stdout or '').splitlines():
            if line.strip():
                index, used, total = (int(x.strip()) for x in line.split(','))
                gpus[str(index)] = dict(used_mib=used, total_mib=total, free_mib=total - used)
        if got.returncode == 0 and gpus:
            out['gpus'] = gpus
        else:
            out['errors'].append('nvidia-smi-rc-%s' % got.returncode)
    except Exception as exc:  # noqa: BLE001 - a read, never the arm's failure
        out['errors'].append('nvidia-smi:' + type(exc).__name__)
    try:
        got = run(['docker', 'top', CONTAINER, '-eo', 'pid,rss,comm'], capture_output=True, text=True, timeout=20)
        rows, workers = [], []
        for line in (got.stdout or '').splitlines()[1:]:
            parts = line.split(None, 2)
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                comm = parts[2].strip() if len(parts) > 2 else '?'
                rows.append((comm, int(parts[1])))
                if comm.startswith(WORKER_COMM):
                    workers.append(dict(status(int(parts[0])), comm=comm, rss_kib=int(parts[1])))
        if got.returncode == 0 and rows:
            out['rss_kib'], out['processes'] = sum(r for _c, r in rows), len(rows)
            out['top'] = [list(r) for r in sorted(rows, key=lambda r: -r[1])[:6]]
            out['workers'] = sorted(workers, key=lambda x: x['pid'])
        else:
            out['errors'].append('docker-top-rc-%s' % got.returncode)
    except Exception as exc:  # noqa: BLE001
        out['errors'].append('docker-top:' + type(exc).__name__)
    return out


_AGENT = re.compile(r'You are coding agent (\d+), session (\d+)\.')


def tracking(stream, inflight, lock, clock=time.monotonic):
    """The replay's stream with each request in ``inflight`` while it is open: what the replay gives up on can be named."""
    def wrapped(url, body, key, origin, **kwargs):
        messages = body.get('messages') or ()
        found = _AGENT.match(str((messages[0] if messages else {}).get('content', '')))
        entry = dict(agent=int(found.group(1)) if found else None, session=int(found.group(2)) if found else None,
                     kind='fresh' if len(messages) <= 2 else 'append', turn_messages=len(messages), sent_s=round(clock() - origin, 3))
        name = threading.current_thread().name
        with lock:
            inflight[name] = entry
        try:
            return stream(url, body, key, origin, **kwargs)
        finally:
            with lock:
                inflight.pop(name, None)
    wrapped.around = lambda inner: tracking(inner, inflight, lock, clock)       # the same tracking over another stream (a test's scripted one)
    return wrapped


def _down(start, end):
    """Any rank's free VRAM down by VRAM_GROWTH_MIB or more; False if either side has no VRAM read."""
    a, b = (start or {}).get('gpus') or {}, (end or {}).get('gpus') or {}
    return any(r in b and a[r]['free_mib'] - b[r]['free_mib'] >= VRAM_GROWTH_MIB for r in a)


def settled_end(start, read=None, sleep=None):
    """The end read, after the lane's release: re-read every SETTLE_STEP_S while a rank is still down from the start, at most
    SETTLE_TRIES times. The first read is kept beside it; settle_s says how long it waited."""
    read, sleep = read or memory_read, sleep or time.sleep
    first = end = read()
    waited = 0.0
    for _ in range(SETTLE_TRIES):
        if not _down(start, end):
            break
        sleep(SETTLE_STEP_S)
        waited += SETTLE_STEP_S
        end = read()
    end = dict(end, settle_s=waited)
    if waited:
        end['first_gpus'] = first.get('gpus')
    return end


def memory_delta(start, end):
    """Per rank: free VRAM start, end, drop (MiB); RSS start, end, growth (KiB). None where a side was not read."""
    if not isinstance(start, dict) or not isinstance(end, dict):
        return None
    ranks = {}
    for rank in sorted(set((start.get('gpus') or {})) | set((end.get('gpus') or {}))):
        a, b = (start.get('gpus') or {}).get(rank), (end.get('gpus') or {}).get(rank)
        ranks[rank] = dict(free_start_mib=(a or {}).get('free_mib'), free_end_mib=(b or {}).get('free_mib'),
                           free_drop_mib=a['free_mib'] - b['free_mib'] if a and b else None)
    rss = (start.get('rss_kib'), end.get('rss_kib'))
    return dict(ranks=ranks or None, rss_start_kib=rss[0], rss_end_kib=rss[1],
                rss_growth_kib=rss[1] - rss[0] if None not in rss else None, workers=worker_delta(start, end),
                errors=list(start.get('errors') or ()) + list(end.get('errors') or ()))


def worker_delta(start, end):
    """Per Worker_PP (by pid, start and end): RSS and each /proc field at both ends and its growth (KiB), the mapping count.
    None when either side has no worker list; a pid on one side only is named in 'unmatched' (an engine restart)."""
    a = {w['pid']: w for w in (start or {}).get('workers') or ()}
    b = {w['pid']: w for w in (end or {}).get('workers') or ()}
    if not a or not b:
        return None
    rows = {}
    for pid in sorted(set(a) & set(b)):
        row = dict(comm=b[pid].get('comm'))
        for key in ('rss_kib',) + tuple(f.lower() + '_kib' for f in STATUS_FIELDS) + ('maps',):
            x, y = a[pid].get(key), b[pid].get(key)
            row[key] = dict(start=x, end=y, growth=y - x if type(x) is int and type(y) is int else None)
        rows[str(pid)] = row
    return dict(rows=rows, unmatched=sorted(set(a) ^ set(b)))


def worker_rss_fails(delta, append_history):
    """coverage-soak-fix: the nr arm's RSS failures, per Worker_PP. ``append_history``: the arm's native-append history tokens (the
    engine's coverage counter), None if not read. A missing worker read fails: unbounded growth must not pass unread."""
    workers = (delta or {}).get('workers')
    if not workers or not workers['rows']:
        return ['worker-rss-not-read']
    fails = ['worker-pid-changed-%s' % pid for pid in workers['unmatched']]
    for pid, row in workers['rows'].items():
        growth = row['rss_kib']['growth']
        if growth is None:
            fails.append('worker-rss-not-read-pid%s' % pid)
            continue
        if growth >= WORKER_RSS_FAIL_KIB:
            fails.append('worker-rss-up-pid%s-%dMiB' % (pid, growth // 1024))
        if type(append_history) is int and append_history > 0 and growth * 1024 / append_history >= WORKER_RSS_FAIL_B_PER_TOKEN:
            fails.append('worker-rss-per-append-token-pid%s-%dB' % (pid, round(growth * 1024 / append_history)))
    return fails


def memory_flags(delta):
    if delta is None or not delta.get('ranks') or delta.get('rss_growth_kib') is None:
        return ['memory-not-read']
    flags = ['vram-free-down-rank%s-%dMiB' % (r, v['free_drop_mib']) for r, v in delta['ranks'].items()
             if v['free_drop_mib'] is None or v['free_drop_mib'] >= VRAM_GROWTH_MIB]
    if delta['rss_growth_kib'] >= RSS_GROWTH_KIB:
        flags.append('engine-rss-up-%dMiB' % (delta['rss_growth_kib'] // 1024))
    return flags


def _coverage(status):
    found = (((status or {}).get('append_yield') or {}).get('declines') or {}).get('coverage')
    return found if type(found) is dict and found.get('schema') == COVERAGE_SCHEMA else None


def coverage_delta(start, end):
    """The engine's prefill coverage counter between two status rows: rows and requests by live|idle, native|stock, site. None if either
    row lacks it, or a count went down (another engine)."""
    a, b = _coverage(start), _coverage(end)
    if a is None or b is None:
        return None
    out = dict(steps=b.get('steps', 0) - a.get('steps', 0), errors=b.get('errors', 0) - a.get('errors', 0), open_native=b.get('open_native'),
               native_turns=b.get('native_turns', 0) - a.get('native_turns', 0),
               native_appends=b.get('native_appends', 0) - a.get('native_appends', 0),
               append_history_tokens=b.get('append_history_tokens', 0) - a.get('append_history_tokens', 0))
    for table in ('rows', 'requests'):
        out[table] = {}
        for state in ('live', 'idle'):
            out[table][state] = {}
            for path in ('native', 'stock'):
                was = ((a.get(table) or {}).get(state) or {}).get(path) or {}
                now = ((b.get(table) or {}).get(state) or {}).get(path) or {}
                diff = {site: now.get(site, 0) - was.get(site, 0) for site in sorted(set(now) | set(was))}
                if any(type(v) is not int or v < 0 for v in diff.values()) or out['steps'] < 0 or out['errors'] < 0:
                    return None
                out[table][state][path] = {site: v for site, v in diff.items() if v}
    live, idle = out['rows']['live'], out['rows']['idle']
    # The lead (6 Nov): the one-row handback of a native turn is native-served (one token in the decode batch for its logits), so the
    # shares count it native; 'strict' counts it stock. Any other stock row, 'handback-excess' and 'expired' included, is stock.
    hb_rows = sum(out['rows'][s]['stock'].get('handback', 0) for s in ('live', 'idle'))
    hb_requests = sum(out['requests'][s]['stock'].get('handback', 0) for s in ('live', 'idle'))
    excess = sum(out['rows'][s]['stock'].get('handback-excess', 0) for s in ('live', 'idle'))
    # every 'handback' row is the first stock row after its own native turn's note_handback (the counter only names it so then): one per
    # turn, so turns == rows. "A native turn before it", exact over the engine's life: handback rows never outrun native turns; a
    # window fails if that deficit grows inside it (an earlier window's own is that window's failure, not this one's).
    deficit = lambda c: sum(((c.get('rows') or {}).get(s) or {}).get('stock', {}).get('handback', 0) for s in ('live', 'idle')) - c.get('native_turns', 0)
    out['handback'] = dict(turns=hb_rows, rows=hb_rows + excess, excess_rows=excess, requests=hb_requests, native_turns=out['native_turns'],
                           max_rows_per_turn=(b.get('handback_max_rows') if excess else (1 if hb_rows else 0)),
                           every_row_after_a_native_turn=deficit(b) <= max(0, deficit(a)))
    # coverage-soak-fix (lead, 6 Nov 06:25): a geometry tail's r rows (2..64) after the request's own native turn that ended at the
    # last block boundary, once per turn, are native-served as their own class; the counter names them only so, any row past r is
    # 'handback-excess'. max_r and max_rows are the engine's life maxima, not this window's.
    # coverage-throughput build-2 (c) (lead, 6 Nov): the boundary is the prompt's last prefix unit (176 rows; a block boundary is
    # one); same class, same once-per-turn and 64-row limits, so the grade below is unchanged.
    ga, gb = a.get('geometry_tail') or {}, b.get('geometry_tail') or {}
    out['geometry_tail'] = dict(turns=gb.get('turns', 0) - ga.get('turns', 0),
                                rows=sum(out['rows'][s]['stock'].get('geometry-tail', 0) for s in ('live', 'idle')),
                                live_rows=out['rows']['live']['stock'].get('geometry-tail', 0),
                                requests=sum(out['requests'][s]['stock'].get('geometry-tail', 0) for s in ('live', 'idle')),
                                max_r=gb.get('max_r', 0), max_rows=gb.get('max_rows', 0))
    def share(states, served):
        native = sum(sum(out['rows'][s]['native'].values()) for s in states)
        stock = sum(sum(out['rows'][s]['stock'].values()) for s in states)
        if served:
            moved = sum(out['rows'][s]['stock'].get('handback', 0) + out['rows'][s]['stock'].get('geometry-tail', 0) for s in states)
            native, stock = native + moved, stock - moved
        return round(native / (native + stock), 6) if native + stock else None
    out['live_native_share'], out['all_native_share'] = share(('live',), True), share(('live', 'idle'), True)     # target 1.0 both
    out['live_native_share_strict'], out['all_native_share_strict'] = share(('live',), False), share(('live', 'idle'), False)
    return out


def coverage_windows(arms):
    """Per replay arm: its entry to the next arm's entry (the residue setting holds until the next arm's stage command, so the turns
    still open at 'after' stay in), the last arm to its own 'after'."""
    by = {x.get('label'): x for x in arms if x.get('label')}
    order = [row['label'] for row in T.arms()]
    status = lambda rec, label: next((s for s in (rec or {}).get('statuses') or () if s.get('label') == label), None)
    out = {}
    for k, row in enumerate(T.arms()):
        rec = by.get(row['label'])
        if row['kind'] != 'replay' or rec is None:
            continue
        following = by.get(order[k + 1]) if k + 1 < len(order) else None
        end, to = status(following, 'entry'), 'next-entry'
        if end is None:
            end, to = next((s for s in reversed(rec.get('statuses') or ()) if str(s.get('label')).startswith('after')), None), 'after'
        delta = coverage_delta(status(rec, 'entry'), end)
        out[row['label']] = None if delta is None else dict(delta, to=to)
    return out


def grade(arms, *args, **kwargs):
    """compose-4's grade, and each arm's bound acknowledged: a launch that ignored it would read as this run with no sign of it."""
    with _bound():
        result = _grade(arms, *args, **kwargs)
    by = {x.get('label'): x for x in arms if x.get('label')}
    for row in T.arms():
        rec = by.get(row['label'])
        if rec is not None and (rec.get('inwait') or {}).get('bound_acknowledged') != row['chunk_bound']:
            result['failures'] = sorted(set(result['failures']) | {row['label'] + ':chunk-bound-not-acknowledged'})
        if rec is not None and (rec.get('inwait') or {}).get('hold_acknowledged') != row['native_hold']:
            result['failures'] = sorted(set(result['failures']) | {row['label'] + ':native-hold-not-acknowledged'})
        if rec is not None and (rec.get('inwait') or {}).get('residue_acknowledged') != row['native_residue']:
            result['failures'] = sorted(set(result['failures']) | {row['label'] + ':native-residue-not-acknowledged'})
        if rec is not None and cap_selected(rec):
            result['failures'] = sorted(set(result['failures']) | {row['label'] + ':reserved-cap-selected'})
    # coverage-residue: the operator's rule, measured. On a residue arm any stock prefill row fails it, with a decoder live or idle (the lead,
    # 6 Nov 01:10: 100 % of all prefill rows). The lead's call (6 Nov): the one-row handback of a native turn is native-served, only as exactly
    # one row per turn after that turn (a longer one is 'handback-excess', stock); more handbacks than native turns fail.
    result['coverage'] = coverage_windows(arms)
    rows = T.by_label()
    for label, delta in result['coverage'].items():
        fails = []
        if delta is None:
            fails.append('coverage-not-read')
        else:
            if delta['errors']:
                fails.append('coverage-errors')
            for state, name in (('live', 'with-decoder-live'), ('idle', 'idle')):
                stock = delta['rows'][state]['stock']
                if rows[label]['native_residue'] and any(v for site, v in stock.items() if site not in ('handback', 'geometry-tail')):
                    fails.append('stock-prefill-' + name)
            hb = delta['handback']
            if rows[label]['native_residue'] and (hb['excess_rows'] or not hb['every_row_after_a_native_turn']):
                fails.append('handback-not-one-row-of-a-native-turn')
            geo = delta['geometry_tail']
            if rows[label]['native_residue'] and geo['rows'] and (not geo['turns'] or geo['rows'] > geo['turns'] * T_GEOMETRY_MAX
                                                                  or geo['max_rows'] > T_GEOMETRY_MAX):
                fails.append('geometry-tail-not-of-a-native-turn')
        result['failures'] = sorted(set(result['failures']) | {label + ':' + f for f in fails})
    # coverage-soak (lead, 6 Nov 03:55): no hang (every request sent in an arm completes within the arm and its drain; any that did not is
    # named), and memory at each nr arm's start and end, growth flagged, not failed. ny is read alike, never failed on these.
    result['open_requests'], result['memory'], result['memory_flags'] = {}, {}, {}
    nr = []
    for row in T.arms():
        rec = by.get(row['label'])
        if row['kind'] != 'replay' or rec is None:
            continue
        open_ = rec.get('replay_open_requests') or []
        if open_ or rec.get('replay_still_open'):
            result['open_requests'][row['label']] = dict(requests=open_, agents=rec.get('replay_still_open') or [])
            if row['c1'] == T.ON:
                result['failures'] = sorted(set(result['failures']) | {row['label'] + ':request-open-after-the-drain'})
        mem = rec.get('memory') or {}
        result['memory'][row['label']] = memory_delta(mem.get('start'), mem.get('end'))
        if row['c1'] == T.ON:
            nr.append(mem)
            # coverage-soak-fix: RSS per Worker_PP fails the nr arm (lead, 6 Nov 06:25): >= 256 MiB, or >= 32 B per append-history token
            cov = result['coverage'].get(row['label']) or {}
            rss_fails = worker_rss_fails(result['memory'][row['label']], cov.get('append_history_tokens'))
            result.setdefault('worker_rss', {})[row['label']] = dict(
                fails=rss_fails, append_history_tokens=cov.get('append_history_tokens'), native_appends=cov.get('native_appends'))
            if rss_fails:
                result['failures'] = sorted(set(result['failures']) | {row['label'] + ':' + f for f in rss_fails})
            flags = memory_flags(result['memory'][row['label']])
            if flags:
                result['memory_flags'][row['label']] = flags
    if len(nr) > 1:
        # both starts are idle reads (the lane released at the previous arm's end): their difference is growth that stayed
        result['memory']['nr-first-start-to-last-start'] = run_delta = memory_delta(nr[0].get('start'), nr[-1].get('start'))
        result['memory']['nr-first-start-to-last-end'] = memory_delta(nr[0].get('start'), nr[-1].get('end'))
        flags = memory_flags(run_delta)
        if flags:
            result['memory_flags']['nr-first-start-to-last-start'] = flags
    result['verdict'] = 'FAIL' if result['failures'] else 'INC' if result['missing'] else 'PASS'
    return result


def smoke_check(rec, row=None):
    """compose-4's smoke, and what this table adds: the bound acknowledged, chunks above 1024 rows really run, the host filter engaged on every rank."""
    with _bound():
        ok, reasons = _smoke(rec, row)
    if not isinstance(rec, dict):
        return ok, reasons
    row = row or T.by_label()[T.SMOKE_ARM]
    # dnr: the cap is off by design (the table's note). compose-4's box is frozen and asks for it, so its reason is replaced by the reverse: requested on an arm of this table is the failure.
    reasons = [r for r in reasons if r != 'smoke-reserved-cap-not-requested']
    if (rec.get('reserved_cap') or {}).get('requested') is not False:
        reasons.append('smoke-reserved-cap-not-off')
    if cap_selected(rec) is not False:
        reasons.append('smoke-reserved-cap-selected')
    if (rec.get('inwait') or {}).get('bound_acknowledged') != row['chunk_bound']:
        reasons.append('smoke-chunk-bound-not-acknowledged')
    if (rec.get('inwait') or {}).get('hold_acknowledged') != row['native_hold']:
        reasons.append('smoke-native-hold-not-acknowledged')
    if (rec.get('inwait') or {}).get('residue_acknowledged') != row['native_residue']:
        reasons.append('smoke-native-residue-not-acknowledged')
    # A silent fallback to 1024 would read as no gain. The arm's count holds the bulk, its returners and the append, so the line is the bulk's count at 1024: b13's 40 sit well under it.
    prompt, chunks = (rec.get('fresh') or {}).get('prompt_tokens_observed'), (rec.get('progress_delta') or {}).get('chunks')
    if type(prompt) is not int or type(chunks) is not int or chunks >= len(T.planned_chunks(prompt, 1024)):
        reasons.append('smoke-chunk-rows-not-requested')
    hf = (rec.get('cow_host_filter') or {}).get('counters') or {}
    kept, skipped = hf.get('kept'), hf.get('skipped')
    if not (type(kept) is dict and type(skipped) is dict and len(skipped) == 4 and all((kept.get(rank) or 0) + (skipped.get(rank) or 0) for rank in skipped)):
        reasons.append('smoke-host-filter-not-engaged')
    if any((hf.get('unfiltered') or {}).values()):
        reasons.append('smoke-host-filter-unfiltered')
    return not reasons, sorted(set(reasons))


run_arm = _under_this_table(_box.run_arm)


def replay_arm(a, row, client, poll_client, *, run=None):
    """compose-4's replay arm with this row's seed: in the run itself, and in the journal header, which reads the table's REPLAY."""
    base = run or _box.replay.run
    cfg = T.replay_config(row)
    inflight, lock, memory = {}, threading.Lock(), {}

    def seeded(*args, **kwargs):
        kwargs['stream'] = tracking(kwargs.get('stream') or _box.replay.stream_turn, inflight, lock)
        memory['start'] = memory_read()
        try:
            return base(*args, cfg=cfg, **kwargs)
        finally:
            memory['end'] = settled_end(memory['start'])
    saved = T.REPLAY
    T.REPLAY = cfg
    try:
        with _bound():
            record = _replay_arm(a, row, client, poll_client, run=seeded)
    finally:
        T.REPLAY = saved
    with lock:
        record['replay_open_requests'] = sorted(inflight.values(), key=lambda e: (e['sent_s'], e['agent'] or 0))
    record['memory'] = dict(start=memory.get('start'), end=memory.get('end'))
    return record


def memory_read():
    return engine_memory()


one_stream_arm = _under_this_table(_box.one_stream_arm)


def main(argv=None):
    # grade and smoke inside compose-4's main are this module's, for the call
    names = dict(grade=grade, smoke_check=smoke_check, replay_arm=replay_arm)
    saved = {name: getattr(_box, name) for name in names}
    with _bound():
        for name, value in names.items():
            setattr(_box, name, value)
        try:
            return _box.main(argv)
        finally:
            for name, value in saved.items():
                setattr(_box, name, value)

if __name__ == '__main__':
    raise SystemExit(main())
