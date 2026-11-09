#!/usr/bin/env python3
"""q18a-parity-o3f box arms and grade (lane-o3-fix: k2_settle acknowledged on every arm (FAIL), engaged on the arms that select it (INC)
and never on the others (FAIL); an O3 arm that kept runs but armed none is INC (n1: kept > 0, armed 0 read as working); per arm the fix's receipts
reported (O3 armed/kept/refused, yields split vs FULL, K2 barrier refusals, the lane parked on a held head). q18a-parity-na2: (lane-o3: the o3-pair arm kind, o3_split acknowledged on every arm, O3 engaged on no and po and never on
the controls, the stop rule over the pair arms' first-token top-20 sets; sp, parity-base's native-off baseline, ported from claude/parity-base-sp
e3a6a63f unchanged: keep_lease, small_status, native_off_read and its grade and report lines; lane-o3-na: na, nt with parity-ahead's
ahead_nosync, acknowledged on every arm (FAIL), its 'flash-s5 ahead-split' receipts engaged on na (INC) and never nosync=1 elsewhere (FAIL), and
this queue's own ACTIONS with both stage fields for the router smoke; tmp/decode-phase/READY-parity-o3.md). q18a-compose-throughput-p2: compose-4's box over this table (harness/compose5_arms.py). Discovery mode.
parity-2 adds the per-row terminal_successor, acknowledged (FAIL), and its counters per replay arm: nt that never admitted a terminal
successor is INC (not engaged), ny that did is FAIL; the counts are reported per arm. dnr added a per-row replay seed and the cap off; hold added the per-row native_hold, acknowledged;
coverage-residue adds the per-row native_residue, acknowledged, keeps the engine's decline record (with its prefill coverage counter) in each status row, and grades coverage per replay arm.
coverage-soak adds, per replay arm: the requests still open when the replay gave up on them, by agent, session and send time (a hang, named), and
the engine's memory at the traffic's start and end (per-rank free VRAM by nvidia-smi, the engine container's process RSS by docker top), read on
the box host, never raising; the grade flags growth on nr, it does not fail on it.

Every arm kind compose-5 runs, compose-4's box already runs: the live arm for sm, fon and foff, the replay arm for cmp and ny, the one-stream arm. Only the table and the names differ, so the table is bound in here
and nothing that ran is edited. The rule is compose5_rule.py on the fetched record; this grade only says the record is whole.
"""
import datetime
import hashlib
import os
import re
import subprocess
import threading
import time
from contextlib import contextmanager
from functools import wraps

import compose5_arms as T
import e3_compose4_box as _box

# lane-o3-na: the router smoke's list with this queue's two stage fields (compose-4's ACTIONS is shared by the older routers, unchanged)
ACTIONS = tuple((action, fields + ('o3_split', 'ahead_nosync', 'k2_settle')) if action == 'native_stage_local' else (action, fields) for action, fields in _box.ACTIONS)
ARM_SCHEMA = 'compose5-arm/1'
GRADE_SCHEMA = 'compose5-grade/1'
SCOPE = 'the yield priced under the closed-loop replay and in a frozen pair, on one engine: crash, OOM, stock prefill and the price read by compose5_rule.py; not a release assessment and not a numerics result'
NOT_COVERED = _box.NOT_COVERED
replay = _box.replay
_report = _box.report


def report(result):
    text = _report(result).replace('q18a-compose-4', 'q18a-parity-n3s24q2024p', 1).replace('compose4_rule.py', 'compose5_rule.py')
    # coverage-soak-fix: the geometry-tail class and the worker RSS read, one line per replay arm (the decode gap is the reader's)
    extra = []
    numerics = result.get('numerics_o3')
    if numerics:
        # lane-o3: the stop rule first; OUTSIDE means read nothing past it (lead, 7 Nov)
        extra.append('- O3 NUMERICS (stop rule, first-token top-20 sets, %s against %s): %s; envelope %s, allowance %s, candidate worst %s, token flips %s%s'
                     % (numerics.get('candidate'), ' | '.join(numerics.get('control') or ()), numerics['verdict'], numerics.get('envelope'),
                        numerics.get('allowance'), numerics.get('candidate_worst'), numerics.get('token_flips'),
                        '; ' + ', '.join(numerics['reasons']) if numerics.get('reasons') else ''))
    for label, row in (result.get('o3') or {}).items():
        extra.append('- %s O3: selected %s, kept %s, armed %s, route moves %s, gap %s, transitions %s, FULL reasons %s'
                     % (label, row.get('selected'), row.get('kept'), row.get('armed'), row.get('route_moves'), row.get('gap'),
                        row.get('transitions'), row.get('full')))
    for label, row in (result.get('fix') or {}).items():
        # lane-o3-fix: the fix's receipts over the arm (printed, not graded)
        extra.append('- %s FIX: O3 armed %s kept %s refused %s returned %s settle_ticks %s unparked %s | yields split %s FULL %s | '
                     'K2 settle %s settled %s bound %s, barrier refusals %s | parked %s ms (%s held ticks)'
                     % (label, row.get('o3_armed'), row.get('o3_kept'), row.get('o3_refused'), row.get('o3_returned'), row.get('o3_settle_ticks'),
                        row.get('o3_unparked'), row.get('yields_split'), row.get('yields_full'), row.get('k2_selected'), row.get('k2_settled'),
                        row.get('k2_bound'), row.get('barrier_refusals'), row.get('parked_ms'), row.get('held_ticks')))
    for label, row in sorted(((result.get('ahead') or {}).get('arms') or {}).items()):
        # lane-o3-na: the parity-ahead receipts over the arm
        extra.append('- %s ahead-split: %d lines, %d nosync, %d errors; busy_in %d, still busy_out %d; wait p50 ms nosync %s, sync %s'
                     % (label, row['lines'], row['nosync'], row['errors'], row['busy_in'], row['busy_out_of_busy_in'],
                        row['wait_p50_ms']['nosync'], row['wait_p50_ms']['sync']))
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
    for label, cov in sorted((result.get('coverage') or {}).items()):
        held = (cov or {}).get('yield_bulk_hold') or {}
        if any(held.values()):
            extra.append('- %s YIELD BULK HELD: bulks %d, deferrals %d, expired %d, errors %d (a yielded bulk with no re-entry plan)'
                         % (label, held.get('bulks', 0), held.get('deferrals', 0), held.get('expired', 0), held.get('errors', 0)))
    for label, row in sorted((result.get('worker_rss') or {}).items()):
        extra.append('- %s worker RSS: %s; native appends %s, history %s tokens%s'
                     % (label, ', '.join(row['fails']) or 'no failure', row['native_appends'], row['append_history_tokens'],
                        '; ' + row['read'] if row.get('read') else ''))
    for label, delta in sorted((result.get('memory') or {}).items()):
        # build-3: the arms the RSS rule does not grade (ny, the control), read the same way
        if label not in (result.get('worker_rss') or {}) and not label.startswith('nr-first-') and delta is not None:
            extra.append('- %s worker RSS (read, not graded): %s' % (label, rss_read(delta)))
    for label, row in sorted((result.get('terminal') or {}).items()):
        # parity-2 A: the terminal successor's counters over the arm (stage_delta), and the exact plan's terminal answers beside them
        extra.append('- %s terminal successor: selected %s, admitted %s, mismatch %s; exact admits %s, exact terminal answers %s'
                     % (label, row['selected'], row['terminal_admits'], row['terminal_mismatch'], row['exact_admits'], row['exact_terminal']))
    for label, row in sorted((result.get('native_off') or {}).items()):
        # parity-base: what proves the baseline plain, one line per sp arm
        dec = row.get('decode') or {}
        extra.append('- %s native off: %s; native rows %s, handback rows %s, native-served rows %s, native turns %s, chunks %s; lease %s (%s); '
                     'gate-manual-owned declines %s turns; decode features %s; attn-meta batched %s' % (
                         label, ', '.join(row['fails'] + row['missing']) or 'pure', row.get('native_rows'), row.get('handback_rows'), row.get('served_rows'),
                         row.get('native_turns'), row.get('chunks'), 'held' if row.get('lease_held') else 'NOT HELD' if row.get('lease_held') is False else 'not read',
                         row.get('lease_reads'), row.get('manual_owned_declines'),
                         ', '.join('%s %s' % kv for kv in sorted(dec.items())) or 'not read', row.get('attn_batched')))
    alloc = result.get('alloc_retries')
    if alloc:
        # build-2b24c64: the logits cap's retry rule, per rank over the run, then per arm
        per = lambda ranks: ', '.join('%s %d (max %.2f MiB)' % (k, v['retries'], v['max_ask_mib']) for k, v in sorted(ranks.items())) or 'none'
        mib = lambda peaks: ', '.join('%s %s' % (k, 'none' if v is None else '%.2f MiB' % (v / MIB)) for k, v in sorted(peaks.items())) or 'none'
        extra.append('- allocator retries, cap %d MiB, graded on the native lane: native %s; native indexer peak %s; %d native chunks; fatal CUDA OOM lines %d%s' % (
            alloc['cap_mib'], per(alloc.get('native_ranks') or {}), mib(alloc.get('native_indexer_peak') or {}), alloc.get('native_chunks', 0),
            alloc['cuda_oom_lines'], ('' if alloc['read'] else '; NO LOG READ') + ('; %d retry lines not parsed' % alloc['unparsed'] if alloc['unparsed'] else '')))
        extra.append('- allocator retries outside native chunks (stock or other; printed, not graded): %s' % per(alloc.get('other_ranks') or {}))
        for label, ranks in alloc['arms'].items():
            extra.append('- %s allocator retries: native %s | outside %s | native indexer peak %s' % (
                label, per((alloc.get('native_arms') or {}).get(label) or {}), per((alloc.get('other_arms') or {}).get(label) or {}),
                mib((alloc.get('native_indexer_peak_arms') or {}).get(label) or {})))
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
    if type(reply) is dict:
        row['route_generation'] = reply.get('route_generation')     # lane-o3: route moves per arm
    return row


_select_and_release, _small_status = _box.select_and_release, _box.small_status


def small_status(reply):
    """compose-4's poll row and the policy's lease beside it (parity-base: sp proves the lease held through every poll)."""
    row = _small_status(reply)
    row['manual_owned'] = (reply.get('auto_arm') or {}).get('manual_owned') if type(reply) is dict else None
    return row


def keep_lease(client, record):
    """parity-base, sp: after the arm's release (which every entry needs) take the manual lease back before any turn is sent. The client's
    pause sends the all-rank barrier, a non-read native control (native_auto_arm.manual_control: manual_owned = True); resume is stock's
    resume_generation and takes no lease. The status after must read the lease held, the policy idle, the route FULL, no request; else the arm
    stops before its traffic. Every step recorded."""
    out = record['native_off'] = dict(requested=True, paused=False, resumed=None, manual_owned=None)
    try:
        client.pause()
        out['paused'] = True
    finally:
        reply = client.resume()
        out['resumed'] = reply.get('resumed') is True or reply.get('already_running') is True
    status = client.call('status')
    record['statuses'].append(status_row('lease', status))
    auto = status.get('auto_arm') or {}
    out.update(manual_owned=auto.get('manual_owned'), phase=auto.get('phase'), active=auto.get('active'), selection_seq=auto.get('selection_seq'),
               route=status.get('route'), scheduler_requests=status.get('scheduler_requests'))
    if auto.get('manual_owned') is not True or auto.get('active') is not False or status.get('route') != 'full' or status.get('scheduler_requests') != 0:
        raise ValueError('native-off-lease-not-held')


def select_and_release(client, row, record):
    """compose-4's, then on a native-off row the lease kept for the arm (keep_lease); every other row records native_off requested False."""
    _select_and_release(client, row, record)
    if row.get('native_off'):
        keep_lease(client, record)
    else:
        record['native_off'] = dict(requested=False)


@contextmanager
def _bound():
    """This table and these names in compose-4's box for one call, and back: a suite that runs both boxes in one process sees each with its own."""
    names = dict(T=T, ARM_SCHEMA=ARM_SCHEMA, GRADE_SCHEMA=GRADE_SCHEMA, SCOPE=SCOPE, report=report, status_row=status_row, one_stream_arm=one_stream_arm,
                 select_and_release=select_and_release, small_status=small_status)
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
MEMORY_SCHEMA = 'soak-memory/2'       # /2: the box host's meminfo and cgroup in each read (build-3)
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
# coverage-soak-fix build-3 (lead, 6 Nov 11:45): the box host's own memory at the same reads, beside the workers' RssShmem: new memory
# (cgroup memory.current and Shmem up, MemAvailable down) or pool pages already charged, newly mapped into a worker (those flat)
CGROUP = '/sys/fs/cgroup'
MEMINFO_FIELDS = ('MemTotal', 'MemAvailable', 'Shmem')
CGROUP_STAT_FIELDS = ('shmem', 'anon', 'file')


def host_memory(cgroup=CGROUP, meminfo='/proc/meminfo', opener=open):
    """The box host's memory now: /proc/meminfo MemTotal, MemAvailable, Shmem (KiB) and the cgroup's memory.current, memory.max
    (bytes; 'max' is None) and memory.stat shmem, anon, file (bytes). Read only; never raises: a failed read is an error in the row."""
    out = dict(errors=[])
    try:
        with opener(meminfo) as f:
            for line in f:
                key, _, value = line.partition(':')
                if key in MEMINFO_FIELDS:
                    parts = value.split()
                    out[key.lower() + '_kib'] = int(parts[0]) if parts and parts[0].isdigit() else None
    except Exception as exc:  # noqa: BLE001 - a read, never the arm's failure
        out['errors'].append('meminfo:' + type(exc).__name__)
    for name in ('memory.current', 'memory.max'):
        key = name.replace('.', '_')
        try:
            with opener('%s/%s' % (cgroup, name)) as f:
                value = f.read().strip()
            out[key] = int(value) if value.isdigit() else None
        except Exception as exc:  # noqa: BLE001
            out[key] = None
            out['errors'].append('%s:%s' % (name, type(exc).__name__))
    try:
        with opener('%s/memory.stat' % cgroup) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2 and parts[0] in CGROUP_STAT_FIELDS:
                    out['cgroup_' + parts[0]] = int(parts[1]) if parts[1].isdigit() else None
    except Exception as exc:  # noqa: BLE001
        out['errors'].append('memory.stat:' + type(exc).__name__)
    return out


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


def engine_memory(run=subprocess.run, clock=time.time, status=worker_status, host=host_memory):
    """Per-rank VRAM and the engine container's process RSS, now; per Worker_PP its pid, RSS and /proc status (coverage-soak-fix);
    the box host's meminfo and cgroup (build-3). Read only; never raises: a failed read is an error in the row."""
    out = dict(schema=MEMORY_SCHEMA, wall_s=clock(), gpus=None, rss_kib=None, processes=None, top=None, workers=None, host=None, errors=[])
    try:
        out['host'] = host()
    except Exception as exc:  # noqa: BLE001
        out['errors'].append('host:' + type(exc).__name__)
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
                host=host_delta(start, end), errors=list(start.get('errors') or ()) + list(end.get('errors') or ()))


HOST_KEYS = tuple(f.lower() + '_kib' for f in MEMINFO_FIELDS) + ('memory_current', 'memory_max') + tuple('cgroup_' + f for f in CGROUP_STAT_FIELDS)


def host_delta(start, end):
    """The box host's fields at both ends and their growth (meminfo in KiB, cgroup in bytes); None when either side has no host read."""
    a, b = (start or {}).get('host'), (end or {}).get('host')
    if type(a) is not dict or type(b) is not dict:
        return None
    rows = {}
    for key in HOST_KEYS:
        x, y = a.get(key), b.get(key)
        rows[key] = dict(start=x, end=y, growth=y - x if type(x) is int and type(y) is int else None)
    return rows


def _mib(value, unit):
    return '?' if value is None else '%+d' % round(value * unit / 2 ** 20)


def rss_read(delta):
    """The RSS line's reading: each Worker_PP's RssShmem growth, then the host's memory.current, MemAvailable, Shmem and the
    cgroup's shmem growth (MiB). '?' where a side was not read."""
    workers = ((delta or {}).get('workers') or {}).get('rows') or {}
    shmem = ', '.join('pid%s %s' % (pid, _mib(row.get('rssshmem_kib', {}).get('growth'), 1024)) for pid, row in sorted(workers.items()))
    host = (delta or {}).get('host')
    if host is None:
        return 'worker RssShmem MiB %s; host not read' % (shmem or '-')
    g = lambda key: host.get(key, {}).get('growth')   # noqa: E731
    return ('worker RssShmem MiB %s; host MiB: cgroup memory.current %s, MemAvailable %s, meminfo Shmem %s, cgroup shmem %s'
            % (shmem or '-', _mib(g('memory_current'), 1), _mib(g('memavailable_kib'), 1024), _mib(g('shmem_kib'), 1024),
               _mib(g('cgroup_shmem'), 1)))


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


def _bulk_hold(status):
    found = (((status or {}).get('append_yield') or {}).get('declines') or {}).get('yield_bulk_hold')
    return found if type(found) is dict else {}


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
    # coverage-throughput build-2b (lead, 6 Nov): yielded bulks the re-entry could not plan, held at the positive frontier.
    ha, hb = _bulk_hold(start), _bulk_hold(end)
    out['yield_bulk_hold'] = {k: hb.get(k, 0) - ha.get(k, 0) for k in ('bulks', 'deferrals', 'expired', 'errors')}
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


# build-2b24c64 (lead, 6 Nov ~17:12): the indexer logits cap is 64 MiB for the whole run. The caching allocator's retry warning names the ask:
# '<stamp>Z [rank1]:[W1106 15:41:27.577 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 1 while trying to allocate
# 81788928 bytes (free: ..., total: ...)'. The retried asks above ~60 MiB on record are logits slices: under o4's cap 136 every one was <= 136
# MiB (q18a-o4-chunk2048), under stock 512 they reached 190 (build-2b24's death). So every retried ask on any rank must be <= the cap.
ALLOC_RETRY = re.compile(r'\[rank(\d+)\]:\[W[^\]]*CUDACachingAllocator[^\]]*\] memory allocation failed with OOM on device (\d+) '
                         r'while trying to allocate (\d+) bytes')
CUDA_OOM = re.compile(r'OutOfMemoryError: CUDA out of memory')
# build-2b24c64 scope (lead, 6 Nov ~21:50): the cap's rule covers the native lane, by identity, not every retried ask by size. On
# b2b24c64's box run ny's only asks over 64 MiB (118 MiB, ranks 0 and 2) came on the first stock step after a native run's handback,
# outside every native chunk; under the cap no indexer prefill slice can exceed it (ops/qsa_indexer.py rows_per_chunk), so they are not
# capped logits. Attribution is clock-free: the engine's own cumulative alloc_retries per rank at each native chunk's open and release
# ('flash-s5 memory point=open|released'), against each rank's warnings in order (warning k is retry k: the counts matched on every
# record read). A retry counted inside a chunk's (open, released] is native (a stock step overlapping the chunk counts too: toward FAIL).
NATIVE_MEMORY = re.compile(r'flash-s5 memory point=(open|released) rank=(\d+) txn_seq=(\d+) chunk=\d+-\d+ .*?alloc_retries=(\d+) ')
NATIVE_ALLOC = re.compile(r'flash-s5 native-alloc rank=(\d+) txn_seq=(\d+) chunk=\d+-\d+ .*?indexer_peak=(\d+) ')
LOG_STAMP = re.compile(r'^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?)Z ')
MIB = 1 << 20


def log_wall(line):
    """The log line's own UTC stamp (docker's), as epoch seconds; None when it has none."""
    m = LOG_STAMP.match(line)
    if not m:
        return None
    s = m.group(1)[:26]
    return datetime.datetime.strptime(s, '%Y-%m-%dT%H:%M:%S.%f' if '.' in s else '%Y-%m-%dT%H:%M:%S').replace(
        tzinfo=datetime.timezone.utc).timestamp()


def _ranks(rows):
    out = {}
    for _wall, rank, ask in rows:
        row = out.setdefault('rank%d' % rank, dict(retries=0, max_ask_bytes=0))
        row['retries'] += 1
        row['max_ask_bytes'] = max(row['max_ask_bytes'], ask)
    for row in out.values():
        row['max_ask_mib'] = round(row['max_ask_bytes'] / MIB, 2)
    return out


def alloc_retries(logs, arms=()):
    """Every allocator retry in the followed logs (each line once: the same warning in two streams counts once), per rank over the run and
    per arm window (the arm record's wall_start_s .. wall_end_s), and the fatal CUDA OOM lines. 'read' and 'unread' name the paths."""
    seen, rows, read, unread, oom, unparsed = set(), [], [], [], set(), set()
    opened, spans, peaks, counters = {}, {}, {}, {}
    for path in logs or ():
        try:
            handle = open(path, errors='replace')
        except OSError as exc:
            unread.append('%s: %s' % (path, type(exc).__name__))
            continue
        read.append(str(path))
        with handle:
            for line in handle:
                if 'flash-s5 memory point=' in line or 'flash-s5 native-alloc' in line:
                    m = NATIVE_MEMORY.search(line)
                    if m:
                        rank, retries, key = int(m.group(2)), int(m.group(4)), (int(m.group(2)), m.group(3))
                        counters[rank] = max(counters.get(rank, 0), retries)
                        if m.group(1) == 'open':
                            opened[key] = retries
                        elif key in opened:
                            spans.setdefault(rank, []).append((opened.pop(key), retries, log_wall(line)))
                    m = NATIVE_ALLOC.search(line)
                    if m:
                        rank, peak = int(m.group(1)), int(m.group(3))
                        peaks.setdefault(rank, []).append((log_wall(line), peak))
                    continue
                if 'CUDA out of memory' in line and CUDA_OOM.search(line):
                    oom.add(line.rstrip('\n'))
                if 'CUDACachingAllocator' not in line or 'memory allocation failed' not in line:
                    continue
                m = ALLOC_RETRY.search(line)
                if not m:
                    unparsed.add(line.rstrip('\n'))
                if not m or line in seen:
                    continue
                seen.add(line)
                rows.append((log_wall(line), int(m.group(1)), int(m.group(3))))
    # each rank's k-th warning is its k-th retry; native when some chunk's counter went from below k to k or more
    nth, native = {}, []
    for row in rows:
        nth[row[1]] = nth.get(row[1], 0) + 1
        native.append(any(lo < nth[row[1]] <= hi for lo, hi, _wall in spans.get(row[1], ())))
    windows, native_windows, other_windows, peak_windows = {}, {}, {}, {}
    for rec in arms or ():
        start, end = (rec or {}).get('wall_start_s'), (rec or {}).get('wall_end_s')
        if rec and rec.get('label') and type(start) in (int, float) and type(end) in (int, float):
            inside = lambda wall: wall is not None and start <= wall <= end
            windows[rec['label']] = _ranks([r for r in rows if inside(r[0])])
            native_windows[rec['label']] = _ranks([r for r, n in zip(rows, native) if n and inside(r[0])])
            other_windows[rec['label']] = _ranks([r for r, n in zip(rows, native) if not n and inside(r[0])])
            peak_windows[rec['label']] = {'rank%d' % k: max((p for wall, p in v if inside(wall)), default=None) for k, v in sorted(peaks.items())}
    counted = {'rank%d' % k: v for k, v in sorted(nth.items())}
    return dict(cap_mib=T.LOGITS_CAP_MB, read=read, unread=unread, ranks=_ranks(rows), arms=windows, unstamped=sum(1 for r in rows if r[0] is None),
                cuda_oom_lines=len(oom), unparsed=len(unparsed),
                native_ranks=_ranks([r for r, n in zip(rows, native) if n]), other_ranks=_ranks([r for r, n in zip(rows, native) if not n]),
                native_arms=native_windows, other_arms=other_windows, native_chunks=sum(len(v) for v in spans.values()),
                native_indexer_peak={'rank%d' % k: max(p for _w, p in v) for k, v in sorted(peaks.items())}, native_indexer_peak_arms=peak_windows,
                engine_retries={'rank%d' % k: v for k, v in sorted(counters.items())}, warned_retries=counted)


def alloc_fails(alloc):
    """The cap's rule on the native lane (scope, lead 6 Nov ~21:50): FAIL per rank on a native indexer peak over the cap (the lane's own
    native-alloc receipt) or a retried ask over the cap inside a native chunk; FAIL on a fatal CUDA OOM line. Asks outside native chunks
    (stock or other) are printed, never failed. INC: nothing read, a retry line it cannot size, no native receipt, or warnings that do not
    match the engine's own retry count (the attribution would not hold)."""
    cap = alloc['cap_mib'] * MIB
    fails = ['%s:native-alloc-retry-ask-over-%dMiB' % (rank, alloc['cap_mib']) for rank, row in sorted(alloc['native_ranks'].items())
             if row['max_ask_bytes'] > cap]
    fails += ['%s:native-indexer-peak-over-%dMiB' % (rank, alloc['cap_mib']) for rank, peak in sorted(alloc['native_indexer_peak'].items())
              if peak > cap]
    if alloc['cuda_oom_lines']:
        fails.append('cuda-out-of-memory')
    missing = [] if alloc['read'] else ['alloc-retries-not-read']
    if alloc['unparsed']:
        missing.append('alloc-retry-line-unparsed')       # a retry warning this rule cannot size: not a pass
    if alloc['read'] and not alloc['native_chunks']:
        missing.append('native-chunks-not-read')
    # every retry the engine counted has its warning (a later stock retry may follow the last receipt): fewer warnings and the k-th is not retry k
    if alloc['read'] and alloc['native_chunks'] and any(alloc['warned_retries'].get(rank, 0) < n for rank, n in alloc['engine_retries'].items()):
        missing.append('alloc-retry-count-mismatch')
    return fails, missing


# parity-base: the decode-side counters a plain arm must leave at 0 (stage_delta, before..after), and the stage command's two decode fields
SP_DECODE_COUNTERS = ('step_lazy_marked', 'step_merge_steps', 'step_iw_applied', 'step_iw_sent')


def _turns(status, site):
    turns = (((status or {}).get('append_yield') or {}).get('declines') or {}).get('turns')
    return turns.get(site) if type(turns) is dict and type(turns.get(site)) is int else None


def native_off_read(rec, cov):
    """parity-base, one sp arm: (fails, missing, facts). The arm is plain when its coverage window (entry to the next arm's entry) holds no
    native row, no handback (one-row or excess), no native-served geometry tail, no native turn or append, and its progress no chunk and no
    native selection; the lease read held at every read from keep_lease to the arm's end (its statuses and every poll); the lazy receive and
    the decode merge acknowledged off and their step counters at 0. The attn-meta leak is attn_check's FAIL already."""
    fails, missing = [], []
    status = {s.get('label'): s for s in rec.get('statuses') or ()}
    facts = dict(native_rows=None, handback_rows=None, served_rows=None, native_turns=None, chunks=None, selections=None)
    if cov is not None:
        rows = cov['rows']
        facts.update(native_rows=sum(sum(rows[s]['native'].values()) for s in ('live', 'idle')),
                     handback_rows=cov['handback']['rows'], served_rows=cov['geometry_tail']['rows'],
                     native_turns=cov['native_turns'] + cov['native_appends'])
    chunks = (rec.get('progress_delta') or {}).get('chunks')
    facts['chunks'] = chunks if type(chunks) is int else None
    seqs = [((status.get(k) or {}).get('auto_arm') or {}).get('selection_seq') for k in ('lease', 'after')]
    facts['selections'] = seqs[1] - seqs[0] if all(type(x) is int for x in seqs) else None
    counts = [facts[k] for k in ('native_rows', 'handback_rows', 'served_rows', 'native_turns', 'chunks', 'selections')]
    if any(type(x) is int and x for x in counts):
        fails.append('sp:native-row')
    elif any(x is None for x in counts):
        missing.append('sp:native-rows-not-read')
    reads = [((status.get(k) or {}).get('auto_arm') or {}).get('manual_owned') for k in ('lease', 'before', 'after')]
    polls = [r.get('manual_owned') for r in (rec.get('polled') or {}).get('rows') or ()]
    facts['lease_reads'] = 'lease, before, after and %d polls' % len(polls)
    facts['lease_held'] = False if any(x is False for x in reads + polls) else True if all(x is True for x in reads + polls) else None
    if facts['lease_held'] is False:
        fails.append('sp:lease-not-held')
    elif facts['lease_held'] is None:
        missing.append('sp:lease-not-read')
    step, delta = rec.get('step') or {}, rec.get('stage_delta')
    dec = {k[5:]: (delta or {}).get(k) for k in SP_DECODE_COUNTERS}
    dec['acknowledged_lazy'], dec['acknowledged_merge'] = step.get('acknowledged_lazy'), step.get('acknowledged_merge')
    facts['decode'] = dec
    if step.get('acknowledged_lazy') is not False or step.get('acknowledged_merge') != 'off' or any(type(dec[k[5:]]) is int and dec[k[5:]] for k in SP_DECODE_COUNTERS):
        fails.append('sp:native-decode-feature')
    elif delta is None or any(type(dec[k[5:]]) is not int for k in SP_DECODE_COUNTERS):
        missing.append('sp:decode-features-not-read')
    a, b = _turns(status.get('entry'), 'gate-manual-owned'), _turns(status.get('after'), 'gate-manual-owned')
    facts['manual_owned_declines'] = b - a if a is not None and b is not None else None
    return fails, missing, facts


def grade(arms, *args, **kwargs):
    """compose-4's grade, and each arm's bound acknowledged: a launch that ignored it would read as this run with no sign of it."""
    with _bound():
        result = _grade(arms, *args, **kwargs)
    # build-2b24c64: the logits cap's retry rule over the followed logs (compose-4's grade takes them as its fourth argument, 'logs')
    logs = kwargs['logs'] if 'logs' in kwargs else args[2] if len(args) > 2 else ()
    result['alloc_retries'] = alloc = alloc_retries(logs, [x for x in arms if x.get('label')])
    fails, missing = alloc_fails(alloc)
    result['failures'] = sorted(set(result['failures']) | set(fails))
    result['missing'] = sorted(set(result['missing']) | set(missing))
    by = {x.get('label'): x for x in arms if x.get('label')}
    for row in T.arms():
        rec = by.get(row['label'])
        if rec is not None and (rec.get('inwait') or {}).get('bound_acknowledged') != row['chunk_bound']:
            result['failures'] = sorted(set(result['failures']) | {row['label'] + ':chunk-bound-not-acknowledged'})
        if rec is not None and (rec.get('inwait') or {}).get('hold_acknowledged') != row['native_hold']:
            result['failures'] = sorted(set(result['failures']) | {row['label'] + ':native-hold-not-acknowledged'})
        if rec is not None and (rec.get('inwait') or {}).get('residue_acknowledged') != row['native_residue']:
            result['failures'] = sorted(set(result['failures']) | {row['label'] + ':native-residue-not-acknowledged'})
        if rec is not None and (rec.get('inwait') or {}).get('terminal_acknowledged') is not row['terminal_successor']:
            result['failures'] = sorted(set(result['failures']) | {row['label'] + ':terminal-successor-not-acknowledged'})
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
            held = delta.get('yield_bulk_hold') or {}
            if rows[label]['native_residue'] and (held.get('bulks') or held.get('expired') or held.get('errors')):
                fails.append('yield-bulk-held')       # build-2b: unreachable by plan_view; the lead: FAIL loudly if it fires
            geo = delta['geometry_tail']
            if rows[label]['native_residue'] and geo['rows'] and (not geo['turns'] or geo['rows'] > geo['turns'] * T_GEOMETRY_MAX
                                                                  or geo['max_rows'] > T_GEOMETRY_MAX):
                fails.append('geometry-tail-not-of-a-native-turn')
        result['failures'] = sorted(set(result['failures']) | {label + ':' + f for f in fails})
    # parity-base: sp, the native-off baseline, is plain (native_off_read); every arm ran the lane as its row asked
    result['native_off'] = {}
    for row in T.arms():
        rec = by.get(row['label'])
        if rec is None:
            continue
        if bool((rec.get('native_off') or {}).get('requested')) is not row['native_off']:
            result['failures'] = sorted(set(result['failures']) | {row['label'] + ':native-off-not-as-the-table'})
        if row['native_off']:
            fails, missing, facts = native_off_read(rec, result['coverage'].get(row['label']))
            facts.update(fails=fails, missing=missing, attn_batched=((result.get('attn_meta') or {}).get(row['label']) or {}).get('batched'))
            result['native_off'][row['label']] = facts
            result['failures'] = sorted(set(result['failures']) | {row['label'] + ':' + f for f in fails})
            result['missing'] = sorted(set(result['missing']) | {row['label'] + ':' + m for m in missing})
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
            if row['c1'] in T.ONS:
                result['failures'] = sorted(set(result['failures']) | {row['label'] + ':request-open-after-the-drain'})
        mem = rec.get('memory') or {}
        result['memory'][row['label']] = memory_delta(mem.get('start'), mem.get('end'))
        if row['c1'] in T.ONS:
            nr.append(mem)
            # coverage-soak-fix: RSS per Worker_PP fails the nr arm (lead, 6 Nov 06:25): >= 256 MiB, or >= 32 B per append-history token
            cov = result['coverage'].get(row['label']) or {}
            rss_fails = worker_rss_fails(result['memory'][row['label']], cov.get('append_history_tokens'))
            result.setdefault('worker_rss', {})[row['label']] = dict(
                fails=rss_fails, append_history_tokens=cov.get('append_history_tokens'), native_appends=cov.get('native_appends'),
                read=rss_read(result['memory'][row['label']]))
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
    # parity-2 A: per replay arm, the terminal successor's counters over the arm. On nt, none admitted is not engaged (INC: the fix did not
    # run, so the pair does not price it); on ny, any admitted is the control moved (FAIL). A mismatch (the plan named a chunk terminal that
    # was not) is counted by the engine, not raised, and reported here.
    result['terminal'] = terminal = {}
    for row in T.arms():
        rec = by.get(row['label'])
        if row['kind'] != 'replay' or rec is None:
            continue
        delta = rec.get('stage_delta') or {}
        terminal[row['label']] = read = dict(selected=row['terminal_successor'], **{k: delta.get('step_iw_' + k) for k in ('terminal_admits', 'terminal_mismatch', 'exact_admits', 'exact_terminal')})
        admits = read['terminal_admits']
        if type(admits) is not int and row.get('native_off'):
            pass        # lane-o3: sp's stage_delta unread is sp:decode-features-not-read already; an admit on sp still fails below
        elif type(admits) is not int:
            result['missing'] = sorted(set(result['missing']) | {row['label'] + ':terminal-counters-not-read'})
        elif row['terminal_successor'] and admits == 0:
            result['missing'] = sorted(set(result['missing']) | {row['label'] + ':terminal-successor-not-engaged'})
        elif not row['terminal_successor'] and admits:
            result['failures'] = sorted(set(result['failures']) | {row['label'] + ':terminal-successor-on-the-control'})
    # lane-o3: o3_split acknowledged on every arm (FAIL); per replay and pair arm its counters over the arm: selected with none kept is not
    # engaged (INC), any kept on an arm that did not select it is the control moved (FAIL). Then the stop rule over the pair arms' numerics.
    result['o3'] = o3 = {}
    for row in T.arms():
        rec = by.get(row['label'])
        if rec is None:
            continue
        if (rec.get('inwait') or {}).get('o3_acknowledged') is not row['o3_split']:
            result['failures'] = sorted(set(result['failures']) | {row['label'] + ':o3-split-not-acknowledged'})
        if row['kind'] not in ('replay', 'o3-pair'):
            continue
        o3[row['label']] = read = o3_arm_read(rec)
        read['selected'] = row['o3_split']
        if type(read['kept']) is not int:
            result['missing'] = sorted(set(result['missing']) | {row['label'] + ':o3-counters-not-read'})
        elif row['o3_split'] and read['kept'] == 0:
            result['missing'] = sorted(set(result['missing']) | {row['label'] + ':o3-not-engaged'})
        elif row['o3_split'] and read['armed'] == 0:
            # lane-o3-fix: n1 kept 3/13/27 runs and armed none on every arm; kept without a split continuation is not O3 working
            result['missing'] = sorted(set(result['missing']) | {row['label'] + ':o3-not-armed'})
        elif not row['o3_split'] and read['kept']:
            result['failures'] = sorted(set(result['failures']) | {row['label'] + ':o3-on-the-control'})
    # lane-o3-fix: k2_settle acknowledged on every arm (FAIL); per replay and pair arm the policy's k2 counters over the arm: selected with
    # none settled is not engaged (INC), any settle tick or settle on an arm that did not select it is the control moved (FAIL); unread on a
    # selecting arm is INC. The fix's receipts per arm go to the report.
    result['fix'] = fix = {}
    for row in T.arms():
        rec = by.get(row['label'])
        if rec is None:
            continue
        if (rec.get('inwait') or {}).get('k2_acknowledged') is not row['k2_settle']:
            result['failures'] = sorted(set(result['failures']) | {row['label'] + ':k2-settle-not-acknowledged'})
        if row['kind'] not in ('replay', 'o3-pair'):
            continue
        fix[row['label']] = read = fix_arm_read(rec)
        read['k2_selected'] = row['k2_settle']
        if type(read['k2_settled']) is not int:
            if row['k2_settle']:
                result['missing'] = sorted(set(result['missing']) | {row['label'] + ':k2-counters-not-read'})
        elif row['k2_settle'] and read['k2_settled'] == 0:
            result['missing'] = sorted(set(result['missing']) | {row['label'] + ':k2-not-engaged'})
        elif not row['k2_settle'] and (read['k2_settled'] or read['k2_settle_ticks']):
            result['failures'] = sorted(set(result['failures']) | {row['label'] + ':k2-on-the-control'})
    # lane-o3-na (parity-ahead's suggested grade): ahead_nosync acknowledged on every arm (FAIL); per replay and pair arm the engine's
    # 'flash-s5 ahead-split' receipts in its window: na with no nosync=1 line is not engaged (INC), a nosync=1 line on any other arm is
    # the control moved (FAIL). why=error lines are printed, never failed.
    result['ahead'] = ahead = ahead_split(logs, [x for x in arms if x.get('label')])
    for row in T.arms():
        rec = by.get(row['label'])
        if rec is None:
            continue
        if (rec.get('inwait') or {}).get('ahead_nosync_acknowledged') is not row['ahead_nosync']:
            result['failures'] = sorted(set(result['failures']) | {row['label'] + ':ahead-nosync-not-acknowledged'})
        if row['kind'] not in ('replay', 'o3-pair'):
            continue
        counts = ahead['arms'].get(row['label'])
        if counts is None or not ahead['read']:
            if row['ahead_nosync']:
                result['missing'] = sorted(set(result['missing']) | {row['label'] + ':ahead-split-not-read'})
        elif row['ahead_nosync'] and not counts['nosync']:
            result['missing'] = sorted(set(result['missing']) | {row['label'] + ':ahead-nosync-not-engaged'})
        elif not row['ahead_nosync'] and counts['nosync']:
            result['failures'] = sorted(set(result['failures']) | {row['label'] + ':ahead-nosync-on-the-control'})
    # parity-na2: the stop rule reads the pair arms; a table without them (q18a-parity-na2) has no O3 candidate to stop, so no rule.
    result['numerics_o3'] = numerics = o3_numerics(by) if any(row['kind'] == 'o3-pair' for row in T.arms()) else dict(verdict='NOT-IN-TABLE', reasons=[])
    if numerics['verdict'] == 'NOT-IN-TABLE':
        pass
    elif numerics['verdict'] == 'OUTSIDE':
        result['failures'] = sorted(set(result['failures']) | set(numerics['reasons']))
    elif numerics['verdict'] != 'WITHIN':
        result['missing'] = sorted(set(result['missing']) | set(numerics['reasons']))
    result['verdict'] = 'FAIL' if result['failures'] else 'INC' if result['missing'] else 'PASS'
    return result


AHEAD_SPLIT = re.compile(r'flash-s5 ahead-split rank=(-?\d+) .*?nosync=([01]) why=(\S*) busy_in=(-?\d+) busy_out=(-?\d+) wait_ns=(\d+) wait_cpu_ns=(\d+)')


def ahead_split(logs, arms=()):
    """lane-o3-na: the parity-ahead receipts ('flash-s5 ahead-split', one per successor build, each line once over the followed logs)
    per arm window (wall_start_s .. wall_end_s): lines, nosync=1 lines, why=error lines, busy_in=1 lines and those still busy_out=1, and
    the wait wall p50 in ms by nosync."""
    seen, rows, read = set(), [], []
    for path in logs or ():
        try:
            handle = open(path, errors='replace')
        except OSError:
            continue
        read.append(str(path))
        with handle:
            for line in handle:
                if 'flash-s5 ahead-split ' not in line or line in seen:
                    continue
                seen.add(line)
                m = AHEAD_SPLIT.search(line)
                if m:
                    rows.append((log_wall(line), int(m.group(1)), m.group(2) == '1', m.group(3), int(m.group(4)), int(m.group(5)), int(m.group(6))))
    out = {}
    for rec in arms or ():
        start, end = (rec or {}).get('wall_start_s'), (rec or {}).get('wall_end_s')
        if not (rec and rec.get('label') and type(start) in (int, float) and type(end) in (int, float)):
            continue
        mine = [r for r in rows if r[0] is not None and start <= r[0] <= end]
        waits = lambda flag: sorted(r[6] for r in mine if r[2] is flag)
        p50 = lambda v: round(v[len(v) // 2] / 1e6, 3) if v else None
        out[rec['label']] = dict(lines=len(mine), nosync=sum(1 for r in mine if r[2]), errors=sum(1 for r in mine if r[3] == 'error'),
                                 busy_in=sum(1 for r in mine if r[4] == 1), busy_out_of_busy_in=sum(1 for r in mine if r[4] == 1 and r[5] == 1),
                                 wait_p50_ms=dict(nosync=p50(waits(True)), sync=p50(waits(False))))
    return dict(read=read, lines=len(rows), unstamped=sum(1 for r in rows if r[0] is None), arms=out)


def _o3_status(rec, label):
    rows = [s for s in rec.get('statuses') or () if s.get('label') == label]
    if not rows:
        return None, None
    declines = (rows[-1].get('append_yield') or {}).get('declines')
    o3 = declines.get('o3') if type(declines) is dict else None
    return (o3 if type(o3) is dict else None), rows[-1].get('route_generation')


def _count_delta(after, before):
    """{key: int} or {key: [n, total, max]} after minus before; max is the arm's when it moved (a run's max cannot be split)."""
    out = {}
    for key, value in (after or {}).items():
        old = (before or {}).get(key)
        if type(value) is int:
            out[key] = value - (old if type(old) is int else 0)
        elif type(value) in (list, tuple) and len(value) == 3:
            old = old if type(old) in (list, tuple) and len(old) == 3 else (0, 0.0, 0.0)
            n = value[0] - old[0]
            out[key] = dict(n=n, ms=round(value[1] - old[1], 3), mean_ms=round((value[1] - old[1]) / n, 3) if n else None,
                            max_ms=value[2] if n else None)
    return out


def _declines(rec, label):
    rows = [s for s in rec.get('statuses') or () if s.get('label') == label]
    if not rows:
        return None
    declines = (rows[-1].get('append_yield') or {}).get('declines')
    return declines if type(declines) is dict else None


def _int_delta(after, before, *path):
    for key in path:
        after = after.get(key) if type(after) is dict else None
        before = before.get(key) if type(before) is dict else None
    if type(after) not in (int, float):
        return None
    return round(after - (before if type(before) in (int, float) else 0), 3)


def fix_arm_read(rec):
    """lane-o3-fix: the fix's receipts over one arm, from the policy's decline record in its before and after status rows: O3 armed / kept /
    refused by reason / FULL returns by reason / tail settle ticks / refusals not parked / yields kept split; yields on FULL (the driver's
    'yield' FULL reason); K2 settle ticks, settles and bound hits; K2 barrier refusals; the lane parked on a held head (ms, ticks)."""
    b, e = _declines(rec, 'before'), _declines(rec, 'after')
    if b is None or e is None:
        return dict(read=False, k2_settled=None, k2_settle_ticks=None)
    o3b, o3e = b.get('o3') or {}, e.get('o3') or {}
    full = _count_delta((o3e.get('driver') or {}).get('full'), (o3b.get('driver') or {}).get('full'))
    return dict(read=True, o3_armed=_int_delta(e, b, 'o3', 'armed'), o3_kept=_int_delta(e, b, 'o3', 'kept'),
                o3_refused=_count_delta(o3e.get('refused'), o3b.get('refused')), o3_returned=_count_delta(o3e.get('full'), o3b.get('full')),
                o3_settle_ticks=_int_delta(e, b, 'o3', 'settle_ticks'), o3_unparked=_int_delta(e, b, 'o3', 'unparked'),
                yields_split=_int_delta(e, b, 'o3', 'yield_kept'), yields_full=None if full.get('off') else full.get('yield', 0),     # off arms: every FULL reads 'off'
                k2_settle_ticks=_int_delta(e, b, 'k2', 'settle_ticks'), k2_settled=_int_delta(e, b, 'k2', 'settled'),
                k2_bound=_int_delta(e, b, 'k2', 'bound'),
                barrier_refusals=_int_delta(e, b, 'rollbacks', 'worker-barrier-cohort-refused') or 0,
                parked_ms=_int_delta(e, b, 'residue', 'held_ms'), held_ticks=_int_delta(e, b, 'residue', 'refused_held'))


def o3_arm_read(rec):
    """One arm's O3 counters, the run-end gaps and the route transitions over the arm, from its before and after status rows."""
    labels = ('before', 'after')
    (b, gb), (e, ge) = (_o3_status(rec, x) for x in labels)
    if b is None or e is None:
        return dict(kept=None, read=False)
    driver_b, driver_e = b.get('driver') or {}, e.get('driver') or {}
    return dict(read=True, kept=(e.get('kept') or 0) - (b.get('kept') or 0), armed=(e.get('armed') or 0) - (b.get('armed') or 0),
                tail_bound=(e.get('tail_bound') or 0) - (b.get('tail_bound') or 0),
                driver_kept=(driver_e.get('kept') or 0) - (driver_b.get('kept') or 0),
                full=_count_delta(driver_e.get('full'), driver_b.get('full')), returned=_count_delta(e.get('full'), b.get('full')),
                refused=_count_delta(e.get('refused'), b.get('refused')),
                gap=_count_delta(e.get('gap'), b.get('gap')), transitions=_count_delta(driver_e.get('transitions'), driver_b.get('transitions')),
                route_moves=ge - gb if type(ge) is int and type(gb) is int else None)


# The stop rule (lead, 7 Nov after 10:41): first-token top-20 sets with O3 on against off, judged against the control's self-disagreement,
# numerical gate rule v4's shape (e3_teacher_decode_rule4: allowance = min(cap, max(floor, 2E + 0.125)), the counting metric 2E + a
# margin): E is the off arms' own disagreement (pf against pg, the same prompt per native, the same position). Fixed before the run.
O3_NUMERICS = dict(position=0, lp_floor=0.375, lp_cap=1.0, lp_add=0.125, members_floor=4, members_cap=40, members_add=2)


def _top(native, position):
    top = (native or {}).get('top') or []
    return {int(k): float(v) for k, v in top[position]} if position < len(top) else None


def _pair_diff(x, y):
    shared = set(x) & set(y)
    return dict(members=len(set(x) ^ set(y)), max_abs_logprob=round(max((abs(x[k] - y[k]) for k in shared), default=0.0), 6))


def o3_numerics(by):
    """The pair arms' stop rule: WITHIN, OUTSIDE (a FAIL; read nothing past it) or INC (evidence missing)."""
    rule = O3_NUMERICS
    off_a, on, off_b = ('%s.g2-live-0' % sel for sel in T.O3_PAIR)
    out = dict(schema='parity-o3-numerics/1', is_gate=True, rule=dict(rule), control=[off_a, off_b], candidate=on, natives=[], reasons=[])
    arms = {name: by.get(name) for name in (off_a, on, off_b)}
    for name, rec in arms.items():
        if rec is None or len((rec or {}).get('natives') or ()) != T.O3_PAIR_NATIVES:
            out['reasons'].append('numerics-o3-evidence-missing:' + name)
    if out['reasons']:
        out.update(verdict='INC', allowance=None)
        return out
    prompts = {n.get('prompt_sha256') for rec in arms.values() for n in rec['natives']}
    if len(prompts) != 1:
        out['reasons'].append('numerics-o3-prompt-differs')
    control, across = [], []
    for i in range(T.O3_PAIR_NATIVES):
        tops = {name: _top(arms[name]['natives'][i], rule['position']) for name in arms}
        if any(v is None for v in tops.values()):
            out['reasons'].append('numerics-o3-position-missing:%d' % i)
            continue
        c = _pair_diff(tops[off_a], tops[off_b])
        d = [_pair_diff(tops[on], tops[off_a]), _pair_diff(tops[on], tops[off_b])]
        tokens = {name: (arms[name]['natives'][i].get('tokens') or [None])[rule['position']] for name in arms}
        control.append(c)
        across.extend(d)
        out['natives'].append(dict(index=i, control=c, candidate=d, tokens=tokens))
    if out['reasons']:
        out.update(verdict='INC', allowance=None)
        return out
    e_lp, e_m = max(c['max_abs_logprob'] for c in control), max(c['members'] for c in control)
    allowance = dict(max_abs_logprob=min(rule['lp_cap'], max(rule['lp_floor'], 2 * e_lp + rule['lp_add'])),
                     members=min(rule['members_cap'], max(rule['members_floor'], 2 * e_m + rule['members_add'])))
    worst = dict(max_abs_logprob=max(d['max_abs_logprob'] for d in across), members=max(d['members'] for d in across))
    out.update(envelope=dict(max_abs_logprob=e_lp, members=e_m), allowance=allowance, candidate_worst=worst,
               token_flips=dict(control=sum(1 for n in out['natives'] if n['tokens'][off_a] != n['tokens'][off_b]),
                                candidate=sum(1 for n in out['natives'] for ref in (off_a, off_b) if n['tokens'][on] != n['tokens'][ref])))
    if worst['max_abs_logprob'] > allowance['max_abs_logprob'] or worst['members'] > allowance['members']:
        out['reasons'].append('numerics-o3-outside:%s' % on)
        out['verdict'] = 'OUTSIDE'
    else:
        out['verdict'] = 'WITHIN'
    return out


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
    if (rec.get('inwait') or {}).get('terminal_acknowledged') is not row['terminal_successor']:
        reasons.append('smoke-terminal-successor-not-acknowledged')
    if (rec.get('inwait') or {}).get('o3_acknowledged') is not row['o3_split']:
        reasons.append('smoke-o3-split-not-acknowledged')     # lane-o3: the smoke sends False; an engine without FLASH_S5_O3 refuses it
    if (rec.get('inwait') or {}).get('ahead_nosync_acknowledged') is not row['ahead_nosync']:
        reasons.append('smoke-ahead-nosync-not-acknowledged')     # lane-o3-na: the same, for FLASH_S5_AHEAD_NOSYNC
    if (rec.get('inwait') or {}).get('k2_acknowledged') is not row['k2_settle']:
        reasons.append('smoke-k2-settle-not-acknowledged')        # lane-o3-fix: the smoke sends False; an engine without the field refuses it
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


_one_stream_arm = _under_this_table(_box.one_stream_arm)

# --- lane-o3: the design's first box test (kind o3-pair) --------------------------------------------------------------------------
O3_LANE = 'parity-o3-pair-'
O3_PAIR_SCHEMA = 'parity-o3-pair/1'


def o3_view(reply):
    """The policy's O3 record in a status reply (append_yield.declines.o3: bounded counts, the run-end gap and the transition ms by
    path; no request id), with the route and its generation beside it. None when absent."""
    if type(reply) is not dict:
        return None
    declines = (reply.get('append_yield') or {}).get('declines')
    o3 = declines.get('o3') if type(declines) is dict else None
    return dict(o3 if type(o3) is dict else {}, present=type(o3) is dict, route=reply.get('route'), route_generation=reply.get('route_generation'))


def _o3_native(a, row, cfg, prompt, index):
    """One native request of the pair arm: the band's fresh prompt under its own salt (so no cache hit, in this arm or another), greedy,
    top-20 logprobs on every one of O3_PAIR_TOKENS positions (c1b's checked instrument)."""
    shared, c1b = _box.shared, _box.iw.c1b
    salt = shared._cache_salt(a.run_id, O3_LANE + row['label'], index)
    workload = shared._workload_view(cfg, streams=1, generation_budget=T.O3_PAIR_TOKENS, prompt_seed=cfg['prefill']['fresh']['seed'],
                                     finish_timeout_s=cfg['timeouts']['finish_s'], temperature=0.0, top_p=1.0)
    body = dict(shared._request(a.model, prompt, workload, salt), logprobs=T.O3_LOGPROBS, return_tokens_as_token_ids=True)
    instrument = c1b.Numerics()
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    state = c1b.correctness._start_streams(a.url, a.model, [body], [{'prompt_sha256': digest, 'prompt_chars': len(prompt)}], workload,
                                           instrument=instrument)[0]
    return dict(index=index, state=state, workload=workload, instrument=instrument, prompt_sha256=digest,
                salt_sha256=hashlib.sha256(salt.encode()).hexdigest(),
                request=dict(temperature=body['temperature'], top_p=body['top_p'], max_tokens=body['max_tokens'], logprobs=body['logprobs']))


def _ns(value):
    return value if type(value) is int else None


def o3_native_row(native, standing):
    """The record row of one native: times (send, first output, finish), the tokens and their top-20, and how many standing decoders
    were still open at its first output (the cohort its handback ran beside)."""
    state, inst = native['state'], native['instrument']
    send, first, done = _ns(getattr(state, 'send_at_ns', None)), _ns(native.get('first_ns')), _ns(getattr(state, 'finish_at_ns', None))
    live = None
    if first is not None:
        live = sum(1 for d in standing if _ns(getattr(d, 'finish_at_ns', None)) is None or d.finish_at_ns > first)
    tokens = list(getattr(inst, 'tokens', ()))
    return dict(index=native['index'], prompt_sha256=native['prompt_sha256'], salt_sha256=native['salt_sha256'], request=native['request'],
                send_ns=send, first_ns=first, finish_ns=done,
                first_output_s=round((first - send) / 1e9, 6) if send is not None and first is not None else None,
                total_s=round((done - send) / 1e9, 6) if send is not None and done is not None else None,
                error=None if getattr(state, 'error', None) is None else str(state.error)[:200],
                malformed=None if getattr(state, 'malformed', None) is None else str(state.malformed)[:200],
                tokens=tokens, top=list(getattr(inst, 'top', ())), decoders_live_at_first=live,
                complete=len(tokens) == T.O3_PAIR_TOKENS and len(getattr(inst, 'top', ())) == T.O3_PAIR_TOKENS)


def o3_pair_arm(a, row, cfg, budget, client, *, sleep=time.sleep):
    """lane-o3 arm 1 (DESIGN-multi-prompt-lane O3's first box test): O3_PAIR_DECODERS standing decoders open (the 8-stream baseline's
    prompts and salts, so its cache serves them), the arm's levers selected (o3_split among them), then O3_PAIR_NATIVES native requests
    sent at once; the hold keeps stock off them, so the lane takes them back to back. Every exception is retained, never retried."""
    shared, streams, base, progress = _box.shared, _box.streams, _box.base, _box.progress
    record = _box.base_record(a, row, client)
    record.update(o3_schema=O3_PAIR_SCHEMA, o3_split=row['o3_split'], natives=[], o3={}, standing_generation_budget=budget)
    standing, natives, workload = [], [], None
    mark = len(getattr(client, 'history', ()))
    try:
        if shared._package_gate(os.environ).get('ok') is not True:
            raise ValueError('reviewed-package-readiness-not-pass')
        _box.entry_status(client, row, record)
        workload = shared._workload_view(cfg, generation_budget=budget, streams=T.O3_PAIR_DECODERS)
        rows = streams.build_cohort(workload)
        if len(rows) != T.O3_PAIR_DECODERS:
            raise ValueError('o3-pair-cohort-not-%d' % T.O3_PAIR_DECODERS)
        bodies = [shared._request(a.model, item['prompt'], workload, shared._cache_salt(a.run_id, progress.STANDING, i)) for i, item in enumerate(rows)]
        standing = streams._start_streams(a.url, a.model, bodies, rows, workload)
        opened = streams._wait_open(standing, workload)
        record['open'] = {k: opened.get(k) for k in ('complete', 'reasons')}
        if opened.get('complete') is not True:
            raise ValueError('standing-cohort-unopened')
        _box.select_and_release(client, row, record)
        sleep(base.SETTLE_S)
        before = client.call('status')
        record['statuses'].append(status_row('before', before))
        record['progress'].append(dict(progress.progress_view(before) or {}, label='before', native_stage_local=_box.iw.stage_view(before)))
        record['o3']['before'] = o3_view(before)
        prompt = shared._prefill_prompts(cfg)['fresh']
        record['prompt_sha256'] = hashlib.sha256(prompt.encode()).hexdigest()
        record['natives_sent_s'] = time.time()
        for index in range(T.O3_PAIR_NATIVES):
            natives.append(_o3_native(a, row, cfg, prompt, index))
        for native in natives:
            native['first_ns'] = shared._wait_first_advance(native['state'], cfg['timeouts']['read_s'])
        finish = shared._finish_or_cancel([n['state'] for n in natives], natives[0]['workload'])
        record['natives_finish'] = {k: finish.get(k) for k in ('timed_out', 'cancelled', 'complete')} if type(finish) is dict else None
        after = client.call('status')
        record['statuses'].append(status_row('after', after))
        record['progress'].append(dict(progress.progress_view(after) or {}, label='after', native_stage_local=_box.iw.stage_view(after)))
        record['o3']['after'] = o3_view(after)
        view = {r.get('label'): r for r in record['progress']}
        if view['before'].get('native_stage_local') and view['after'].get('native_stage_local'):
            record['stage_delta'] = _box.iw.stage_delta(view['after']['native_stage_local'], view['before']['native_stage_local'])
        record['control_actions'] = sorted({h.get('action') for h in list(getattr(client, 'history', ()))[mark:]})
        if not set(record['control_actions']) <= _box.ACTION_NAMES:
            record['reasons'].append('arm-sent-a-non-receipt-action')
    except Exception as exc:  # noqa: BLE001 - retained, never retried
        record['error'] = _box.exception_details(exc, getattr(client, 'control', None))
        if getattr(exc, 'refusal', None) is not None:
            record['controller_refusal'] = exc.refusal
        record['reasons'].append('arm-error')
    finally:
        record['natives'] = [o3_native_row(n, standing) for n in natives]
        if standing:
            streams._stop_streams({}, standing, workload)
        record['wall_end_s'] = time.time()
    for row_ in record['natives']:
        if not row_['complete'] or row_['error'] or row_['malformed']:
            record['reasons'].append('o3-pair-native-%d-incomplete' % row_['index'])
    if record['natives'] and len(record['natives']) != T.O3_PAIR_NATIVES:
        record['reasons'].append('o3-pair-natives-not-%d' % T.O3_PAIR_NATIVES)
    record['complete'] = not record['reasons']
    return record


def one_stream_arm(a, row, cfg, budget, client, **kwargs):
    """compose-4's run_arm sends every arm that is not a replay or a live arm here: the o3-pair kind is lane-o3's, the rest compose-4's."""
    if row.get('kind') == 'o3-pair':
        with _bound():
            return o3_pair_arm(a, row, cfg, budget, client, **kwargs)
    return _one_stream_arm(a, row, cfg, budget, client, **kwargs)


def main(argv=None):
    # grade and smoke inside compose-4's main are this module's, for the call
    names = dict(grade=grade, smoke_check=smoke_check, replay_arm=replay_arm, one_stream_arm=one_stream_arm)
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
