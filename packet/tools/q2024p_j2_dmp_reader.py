#!/usr/bin/env python3
"""Exploratory reader for the q2024p-j2 decode-merge probe records (dmp-off, dmp-offsmall). Read-only; no rule, no
thresholds, no verdict.

Per record and replay arm (every arm-*.json with a turn journal, in replay order: sp | n3 X X n3 on the dmp packets;
sp n3 n3 sp on a j2 record, which is how the reader is dry-run) it prints:
  * turns/h, x sp (against the record's first sp arm), native share (compose5-grade coverage), complete and reasons;
  * the stage command's echo (requested/acknowledged decode_merge) and the arm's merge counters (stage_delta);
  * median client token gap (ms) by overlap state x live decoders. States and filters are fable-levers
    decode-overlap/d_client_gaps.py's, ported: per ms of the arm window the state is pause / full / split_idle /
    armed_gap / txn_tiny (<256 rows) / txn_mid / txn_bulk (>=1536 rows); a gap counts when 4 <= gap <= 150 ms and >= 90 %
    of it lies in one state; live = median over the gap of the turns between first token and turn end. "quiet" in the
    12 Nov fits (13.7 + 0.61 n) is the ``full`` state; "bulk" (21.7 + 2.07 n) is ``txn_bulk``. The fit at the arm's
    live counts is printed beside the measured median;
  * decode token-gap p50/p95/p99 over every consecutive arrival pair (no filter; the hgb reader's measure);
  * stage forward (device forward of execute_model, handler anatomy 'm' rows kind 'e', dev_forward_ns) per rank: n,
    median, mean, and the median while no transaction is open vs inside a bulk transaction;
  * bulk-chunk (>=1536 rows) prefill busy ms per rank from the workers' 'flash-s5 anatomy' chunk rows (busy_us);
  * scheduler cohort stats (flash-sched stats lines, every 200 steps): dec_even / dec_odd / cap_skips over the arm and
    the mean running cohorts even/odd;
  * the liveness view: engine 'flash-s5 decode-liveness' lines in the window, the status rows' liveness blocks
    (before/after), the arm's decode_liveness_guard record;
  * the submit-cpu ride-along per worker rank: flash_s5_submit calls, wall ms, thread-CPU ms over the arm;
and per record: numerics_o3 verdict, queue/compose5 grade verdicts and rc, OOM max and fatal lines per rank, and the
engine identity's FLASH_S5_DECODE_MERGE / FLASH_S5_DECODE_LIVENESS / FLASH_S5_SUBMIT_CPU / FLASH_S5_SMALL_PIECE_BATCH.

Times: engine monotonic stamps are mapped to wall by the minimum over native-gap lines of (line stamp - observed_ns)
(as fable-levers parse.py does); a turn's arrivals_ms are ms since the arm's replay start (the journal's own base: the
first arrival is t + ttft), its live interval is [t + ttft, t + seconds].

Usage:
  python3 -B tools/q2024p_j2_dmp_reader.py off=<FETCH>/q18a-parity-n3s24q2024p-j2-dmp-off \
      offsmall=<FETCH>/q18a-parity-n3s24q2024p-j2-dmp-offsmall [--json-out <fresh.json>]
  (any label=<record dir> is accepted; the label is only printed)
"""
import argparse
import ast
import collections
import datetime
import json
import math
from pathlib import Path
import re

STATES = ('pause', 'full', 'split_idle', 'armed_gap', 'txn_tiny', 'txn_mid', 'txn_bulk')
TINY_ROWS, BULK_ROWS = 256, 1536
GAP_MIN_MS, GAP_MAX_MS, STATE_SHARE = 4, 150, 0.9
FITS = {'full': (13.7, 0.61), 'txn_bulk': (21.7, 2.07)}      # 12 Nov fable-levers fits (ms = a + b * live)
MIN_CELL = 30                                                 # cells with fewer gaps are printed as '-'
IDENTITY_KEYS = ('FLASH_S5_DECODE_MERGE', 'FLASH_S5_DECODE_LIVENESS', 'FLASH_S5_SUBMIT_CPU', 'FLASH_S5_SMALL_PIECE_BATCH')
STAMP = re.compile(r'^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d+))?Z ')
MEMORY_POINT = re.compile(r'flash-s5 memory point=(\w+) rank=(\d+) txn_seq=(\d+) chunk=(\d+)-(\d+)')
MEMORY = re.compile(r'flash-s5 memory .*?\brank=(\d+)\b.*?\booms=(\d+)\b')
FATAL = re.compile(r'FLASH_S5_WORKER_FAILURE|CUDA (?:error:.*(?:illegal|out of memory)|out of memory)|OutOfMemoryError|'
                   r'WorkerRPCError.*worker-fatal|Segmentation fault', re.I)
SCHED = re.compile(r'flash-sched stats step (\d+): (\{.*?\}) cohorts even=(\d+) odd=(\d+)')
SCHED_KEYS = ('dec_even', 'dec_odd', 'cap_skips', 'parity_changes', 'D', 'P', 'plain')


def load(path):
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return value if type(value) is dict else None


def stamp(line):
    """Docker's RFC 3339 stamp at the line head to Unix seconds, or None."""
    m = STAMP.match(line)
    if m is None:
        return None
    when = datetime.datetime.fromisoformat(m[1] + '+00:00').timestamp()
    return when + (float('0.' + m[2]) if m[2] else 0.0)


def quantile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]


def _json_after(line, marker):
    i = line.find(marker)
    if i < 0:
        return None
    try:
        value = json.loads(line[i + len(marker):])
    except ValueError:
        return None
    return value if type(value) is dict else None


class Engine:
    """One pass over engine-stream.log: the lines this reader uses, in engine monotonic seconds unless named wall."""

    def __init__(self, path):
        self.offset = None                 # wall - monotonic
        self.gap = []                      # (mono_s, event, dict)
        self.mem = []                      # (wall, point, rank, txn, c0, c1)
        self.forward = []                  # (mono_s, rank, dev_forward_ms)
        self.anatomy = []                  # (mono_s first, rank, rows, busy_ms, window_ms)
        self.sched = []                    # (wall, step, dict, even, odd)
        self.liveness = []                 # wall of each decode-liveness line
        self.submit = []                   # (wall, rank, dict)
        self.ooms, self.fatal, self.torn = {}, 0, collections.Counter()
        with Path(path).open(errors='replace') as handle:
            for line in handle:
                self._line(line)

    def _line(self, line):
        if 'flash-s5 ' not in line and 'flash-sched stats' not in line and not FATAL.search(line):
            return
        if FATAL.search(line):
            self.fatal += 1
        if 'flash-sched stats' in line:
            m = SCHED.search(line)
            if m:
                try:
                    self.sched.append((stamp(line), int(m[1]), ast.literal_eval(m[2]), int(m[3]), int(m[4])))
                except (ValueError, SyntaxError):
                    self.torn['sched'] += 1
            return
        if 'flash-s5 native-gap ' in line:
            d = _json_after(line, 'flash-s5 native-gap ')
            wall = stamp(line)
            if d is None or wall is None or type(d.get('observed_ns')) is not int:
                self.torn['native-gap'] += 1
                return
            mono = d['observed_ns'] / 1e9
            self.offset = wall - mono if self.offset is None else min(self.offset, wall - mono)
            self.gap.append((mono, d.get('event'), d))
        elif 'flash-s5 memory ' in line:
            m = MEMORY.search(line)
            if m:
                self.ooms[int(m[1])] = max(self.ooms.get(int(m[1]), 0), int(m[2]))
            p = MEMORY_POINT.search(line)
            if p:
                self.mem.append((stamp(line), p[1], int(p[2]), int(p[3]), int(p[4]), int(p[5])))
        elif 'flash-s5 handler ' in line:
            d = _json_after(line, 'flash-s5 handler ')
            if d is None:
                self.torn['handler'] += 1
                return
            base, rank = d.get('base_ns') or 0, d.get('rank')
            for m in (d.get('rows') or {}).get('m', ()):
                if len(m) > 9 and m[1] == 'e' and type(m[9]) is int and type(m[4]) is int:
                    self.forward.append(((base + m[4] * 1000) / 1e9, rank, m[9] / 1e6))
        elif 'flash-s5 anatomy ' in line:
            d = _json_after(line, 'flash-s5 anatomy ')
            if d is None:
                self.torn['anatomy'] += 1
                return
            if d.get('side') == 'engine':
                return
            c = ((d.get('rows') or {}).get('c') or [None])[0]
            if not c or type(c[4]) is not int or type(c[6]) is not int:
                return
            base = d.get('base_ns') or 0
            window = (c[5] - c[4]) / 1000 if type(c[5]) is int else None
            rows = (d.get('chunk_end') or 0) - (d.get('chunk_start') or 0)
            self.anatomy.append(((base + c[4] * 1000) / 1e9, d.get('rank'), rows, c[6] / 1000, window))
        elif 'flash-s5 decode-liveness ' in line:
            self.liveness.append(stamp(line))
        elif 'flash-s5 submit-cpu ' in line:
            d = _json_after(line, 'flash-s5 submit-cpu ')
            if d is not None:
                self.submit.append((stamp(line), d.get('rank'), d))

    # --- state intervals, in wall seconds --------------------------------------------------------------------------
    def wall(self, mono):
        return mono + (self.offset or 0.0)

    def intervals(self):
        """(start_wall, end_wall, state) in paint order: armed, txn tiny/mid/bulk, full, pause (later wins)."""
        out, armed, full, pause = [], None, None, None
        if self.gap:
            full = self.gap[0][0]
        for mono, event, d in self.gap:
            if event == 'admission_armed':
                armed = mono if armed is None else armed
                if full is not None:
                    out.append((full, mono, 'full'))
                    full = None
            elif event == 'admission_disarmed':
                if armed is not None:
                    out.append((armed, mono, 'armed_gap'))
                    armed = None
                if d.get('reason') == 'route-returned-full' and full is None:
                    full = mono
            elif event == 'pause':
                if d.get('pause') == 'PAUSED_ALL':
                    pause = mono if pause is None else pause
                elif pause is not None:
                    out.append((pause, mono, 'pause'))
                    pause = None
        if full is not None:
            out.append((full, self.gap[-1][0] + 3600, 'full'))
        result = [(self.wall(a), self.wall(b), s) for a, b, s in out]
        opened, released = {}, {}
        for wall, point, rank, txn, c0, c1 in self.mem:
            if point == 'open' and rank == 0:
                opened[txn] = (wall, c1 - c0)
            elif point != 'open':
                released[txn] = max(released.get(txn, wall), wall)
        txns = []
        for txn, (wall, rows) in opened.items():
            if txn in released:
                state = 'txn_tiny' if rows < TINY_ROWS else ('txn_mid' if rows < BULK_ROWS else 'txn_bulk')
                txns.append((wall, released[txn], state))
        order = {'armed_gap': 0, 'txn_tiny': 1, 'txn_mid': 2, 'txn_bulk': 3, 'full': 4, 'pause': 5}
        return sorted(result + txns, key=lambda x: order[x[2]])


def grid(intervals, origin, ms):
    """Per-ms state code over [origin, origin + ms/1000), default split_idle, paint order kept."""
    cells = bytearray([STATES.index('split_idle')]) * ms
    for a, b, state in intervals:
        i, j = max(0, int((a - origin) * 1000)), min(ms, int((b - origin) * 1000) + 1)
        if j > i:
            cells[i:j] = bytes([STATES.index(state)]) * (j - i)
    return cells


def turns_of(path):
    head, turns = None, []
    try:
        with Path(path).open() as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if type(row) is not dict:
                    continue
                if row.get('header'):
                    head = row
                else:
                    turns.append(row)
    except OSError:
        return None, None
    return head, turns


def client_gaps(turns, cells, ms):
    """Median token gap by state x live, the unfiltered gap quantiles, and the live-count stream seconds."""
    diff = [0] * (ms + 1)
    streams = []
    for t in turns:
        arr = t.get('arrivals_ms')
        if type(arr) is not list or len(arr) < 2 or type(t.get('t')) not in (int, float):
            continue
        base = t['t'] * 1000
        if type(t.get('ttft')) in (int, float) and type(t.get('seconds')) in (int, float):
            i, j = int(max(0, base + t['ttft'] * 1000)), int(min(ms, base + t['seconds'] * 1000))
            if j > i:
                diff[i] += 1
                diff[j] -= 1
        streams.append([int(a) for a in arr if type(a) in (int, float)])   # arrivals_ms: ms since the replay start
    live, run = bytearray(ms), 0
    for i in range(ms):
        run += diff[i]
        live[i] = max(0, min(255, run))
    cells_by = collections.defaultdict(list)
    raw = []
    for arrivals in streams:
        for x, y in zip(arrivals, arrivals[1:]):
            d = y - x
            raw.append(d)
            if x < 0 or y > ms or d < GAP_MIN_MS or d > GAP_MAX_MS:
                continue
            seg = cells[x:y]
            counts = [seg.count(k) for k in range(len(STATES))]
            k = max(range(len(STATES)), key=counts.__getitem__)
            if counts[k] < STATE_SHARE * d:
                continue
            n = sorted(live[x:y])[d // 2]
            cells_by[(STATES[k], n)].append(d)
            cells_by[(STATES[k], 'all')].append(d)
    table = {}
    for (state, n), values in cells_by.items():
        table['%s|%s' % (state, n)] = dict(n=len(values), p50_ms=quantile(values, .5))
    seconds = collections.Counter(live)
    return table, dict(n=len(raw), p50_ms=quantile(raw, .5), p95_ms=quantile(raw, .95), p99_ms=quantile(raw, .99)), \
        {str(k): round(v / 1000, 1) for k, v in sorted(seconds.items()) if k}


def forward_by_rank(engine, start, end, cells):
    rows = collections.defaultdict(lambda: dict(all=[], quiet=[], bulk=[]))
    for mono, rank, ms in engine.forward:
        i = int((engine.wall(mono) - start) * 1000)
        if 0 <= i < len(cells):
            slot = rows[rank]
            slot['all'].append(ms)
            state = STATES[cells[i]]
            if state in ('full', 'split_idle'):
                slot['quiet'].append(ms)
            elif state == 'txn_bulk':
                slot['bulk'].append(ms)
    out = {}
    for rank in sorted(rows, key=str):
        slot = rows[rank]
        out[str(rank)] = dict(n=len(slot['all']), p50_ms=_r(quantile(slot['all'], .5)),
                              mean_ms=_r(sum(slot['all']) / len(slot['all'])) if slot['all'] else None,
                              no_txn_p50_ms=_r(quantile(slot['quiet'], .5)), bulk_p50_ms=_r(quantile(slot['bulk'], .5)),
                              no_txn_n=len(slot['quiet']), bulk_n=len(slot['bulk']))
    return out


def bulk_busy_by_rank(engine, start, end):
    rows = collections.defaultdict(list)
    for mono, rank, n_rows, busy, window in engine.anatomy:
        if n_rows >= BULK_ROWS and start <= engine.wall(mono) <= end:
            rows[rank].append((busy, window))
    out = {}
    for rank in sorted(rows, key=str):
        busy = [b for b, _ in rows[rank]]
        windows = [w for _, w in rows[rank] if w is not None]
        out[str(rank)] = dict(chunks=len(busy), busy_mean_ms=_r(sum(busy) / len(busy)),
                              window_mean_ms=_r(sum(windows) / len(windows)) if windows else None)
    return out


def sched_over(engine, start, end):
    before = [r for r in engine.sched if r[0] is not None and r[0] < start]
    inside = [r for r in engine.sched if r[0] is not None and start <= r[0] <= end]
    if not inside:
        return None
    first = before[-1][2] if before else {}
    last = inside[-1][2]
    delta = {k: last.get(k, 0) - first.get(k, 0) for k in SCHED_KEYS if type(last.get(k)) is int}
    delta.update(lines=len(inside), steps=inside[-1][1] - (before[-1][1] if before else 0),
                 cohort_even_mean=_r(sum(r[3] for r in inside) / len(inside)),
                 cohort_odd_mean=_r(sum(r[4] for r in inside) / len(inside)))
    return delta


def submit_over(engine, start, end):
    out = {}
    for rank in sorted({r for _, r, _ in engine.submit}, key=str):
        rows = [(w, d) for w, r, d in engine.submit if r == rank and w is not None]
        before = [d for w, d in rows if w < start]
        inside = [d for w, d in rows if start <= w <= end]
        if not inside:
            continue
        a, b = inside[-1], (before[-1] if before else {})
        calls = a.get('n', 0) - b.get('n', 0)
        wall = (a.get('wall_ns', 0) - b.get('wall_ns', 0)) / 1e6
        cpu = (a.get('cpu_ns', 0) - b.get('cpu_ns', 0)) / 1e6
        out[str(rank)] = dict(calls=calls, wall_ms=_r(wall), cpu_ms=_r(cpu),
                              cpu_per_wall=_r(cpu / wall, 4) if wall else None,
                              max_wall_ms_lifetime=_r(a.get('max_wall_ns', 0) / 1e6))
    return out


def _r(value, digits=3):
    return None if value is None else round(value, digits)


def identity(root):
    values = {}
    for phase in ('start', 'end'):
        doc = load(root / ('engine.identity-%s.json' % phase))
        env = ((doc or {}).get('identity') or {}).get('env') if doc else None
        values[phase] = None if type(env) is not dict else {k: env.get(k, 'unset') for k in IDENTITY_KEYS}
    return values


def status_rows(arm):
    return {r.get('label'): r for r in arm.get('statuses') or () if type(r) is dict and r.get('label')}


def turns_per_hour(arm):
    cfg, summary = arm.get('replay_config') or {}, arm.get('replay_summary') or {}
    try:
        done = summary['warm_turns'] - summary['turns_finished_after_window']
        return done, done * 3600 / (cfg['DURATION'] - cfg['WARMUP'])
    except (KeyError, TypeError, ZeroDivisionError):
        return None, None


def replay_arms(root):
    arms = []
    for path in sorted(root.glob('arm-*.json')):
        if (root / (path.name + '.turns.jsonl')).is_file():
            arm = load(path)
            if arm is not None:
                arms.append(arm)
    return sorted(arms, key=lambda a: a.get('replay_start_s') or 0)


def collect(root, label):
    root = Path(root)
    notes = []
    grade, queue = load(root / 'compose5-grade.json') or {}, load(root / 'queue-grade.json') or {}
    log = root / 'engine-stream.log'
    engine = Engine(log) if log.is_file() else None
    if engine is None:
        notes.append('missing-engine-stream.log')
    elif engine.offset is None:
        notes.append('no-native-gap-lines: engine times not mapped')
    intervals = engine.intervals() if engine else []
    arms = {}
    for arm in replay_arms(root):
        name = arm.get('label')
        done, tph = turns_per_hour(arm)
        cfg = arm.get('replay_config') or {}
        start = arm.get('replay_start_s')
        duration = cfg.get('DURATION') or 480.0
        ms = int(duration * 1000)
        head, turns = turns_of(root / ('arm-%s.json.turns.jsonl' % name))
        coverage = (grade.get('coverage') or {}).get(name) or {}
        delta = arm.get('stage_delta') or {}
        rows = status_rows(arm)
        native_off = bool((arm.get('native_off') or {}).get('requested'))
        row = dict(native_off=native_off, completed_turns=done, turns_per_hour=tph, complete=arm.get('complete'), reasons=arm.get('reasons'),
                   seed=cfg.get('SEED'), step=arm.get('step'),
                   merge={k: delta.get(k) for k in ('step_merge_steps', 'step_merge_on', 'step_merge_off', 'step_lazy_marked')},
                   native_share=None if native_off else
                   {k: coverage.get(k) for k in ('live_native_share_strict', 'all_native_share_strict')},
                   liveness_status={k: rows[k].get('decode_liveness') for k in ('before', 'after') if k in rows},
                   liveness_guard=arm.get('decode_liveness_guard'))
        if type(start) in (int, float) and engine is not None and turns is not None:
            end = start + duration
            # A native-off arm runs the FULL route throughout (no native-gap events of its own): painted full.
            cells = (bytearray([STATES.index('full')]) * ms) if native_off else grid(intervals, start, ms)
            row['gaps'], row['decode_gaps'], row['live_seconds'] = client_gaps(turns, cells, ms)
            row['state_seconds'] = {STATES[k]: round(cells.count(k) / 1000, 1) for k in range(len(STATES))}
            row['forward'] = forward_by_rank(engine, start, end, cells)
            row['bulk_busy'] = bulk_busy_by_rank(engine, start, end)
            row['sched'] = sched_over(engine, start, end)
            row['liveness_lines'] = sum(1 for w in engine.liveness if w is not None and start <= w <= end)
            row['submit_cpu'] = submit_over(engine, start, end)
        else:
            notes.append('arm-unread:' + str(name))
        arms[name] = row
    sp = next((a for k, a in arms.items() if k.startswith('sp.')), None)
    for name, row in arms.items():
        if sp and sp['completed_turns'] and row['completed_turns'] is not None and not name.startswith('sp.'):
            row['x_sp'] = round(row['completed_turns'] / sp['completed_turns'], 4)
    by_sel = collections.defaultdict(list)
    for name, row in arms.items():
        if row['completed_turns'] is not None:
            by_sel[name.split('.')[0]].append(row['completed_turns'])
    return dict(record=str(root), label=label, arms=arms, notes=notes, identity=identity(root),
                turns_by_selection={k: v for k, v in sorted(by_sel.items())},
                numerics=(grade.get('numerics_o3') or {}).get('verdict'),
                compose5_grade=grade.get('verdict'), queue_grade=queue.get('verdict'), queue_rc=queue.get('rc'),
                engine_crashed=queue.get('engine_crashed'),
                engine=None if engine is None else dict(
                    offset_s=engine.offset, oom_max=max(engine.ooms.values()) if engine.ooms else None,
                    oom_by_rank={str(k): v for k, v in sorted(engine.ooms.items())}, fatal_lines=engine.fatal,
                    liveness_lines=len(engine.liveness), torn=dict(engine.torn)))


def _f(value, digits=1):
    if value is None:
        return '-'
    return ('%.' + str(digits) + 'f') % value if type(value) is float else str(value)


def render(results):
    lines = []
    for res in results:
        eng = res['engine'] or {}
        lines.append('== %s  %s' % (res['label'], res['record']))
        lines.append('   identity start=%s end=%s' % (json.dumps(res['identity']['start'], sort_keys=True),
                                                     json.dumps(res['identity']['end'], sort_keys=True)))
        lines.append('   numerics=%s grade=%s/%s rc=%s engine_crashed=%s oom_max=%s fatal=%s liveness_lines=%s turns=%s' % (
            res['numerics'], res['queue_grade'], res['compose5_grade'], res['queue_rc'], res['engine_crashed'],
            eng.get('oom_max'), eng.get('fatal_lines'), eng.get('liveness_lines'),
            json.dumps(res['turns_by_selection'], sort_keys=True)))
        for name, arm in res['arms'].items():
            share = arm['native_share'] or {}
            dg = arm.get('decode_gaps') or {}
            step = arm.get('step') or {}
            lines.append('  %-14s seed=%s turns=%s turns/h=%s x_sp=%s native=%s/%s complete=%s merge req/ack=%s/%s '
                         'merge_steps=%s on=%s off=%s gap p50/p95/p99=%s/%s/%s ms' % (
                             name, arm['seed'], arm['completed_turns'], _f(arm['turns_per_hour']), _f(arm.get('x_sp'), 4),
                             _f(share.get('live_native_share_strict'), 4), _f(share.get('all_native_share_strict'), 4),
                             arm['complete'], step.get('requested_merge'), step.get('acknowledged_merge'),
                             arm['merge'].get('step_merge_steps'), arm['merge'].get('step_merge_on'),
                             arm['merge'].get('step_merge_off'), dg.get('p50_ms'), dg.get('p95_ms'), dg.get('p99_ms')))
            if arm.get('reasons'):
                lines.append('      reasons=%s' % json.dumps(arm['reasons']))
            gaps = arm.get('gaps') or {}
            for state in ('full', 'split_idle', 'txn_tiny', 'txn_mid', 'txn_bulk'):
                cells = []
                for n in ['all'] + list(range(1, 9)):
                    cell = gaps.get('%s|%s' % (state, n))
                    if cell and cell['n'] >= MIN_CELL:
                        fit = FITS.get(state)
                        extra = '' if fit is None or n == 'all' else '(fit %.1f)' % (fit[0] + fit[1] * n)
                        cells.append('%s:%s%s' % (n, cell['p50_ms'], extra))
                if cells:
                    lines.append('      gap %-10s %s' % (state, ' '.join(cells)))
            if arm.get('state_seconds'):
                lines.append('      state_s=%s' % json.dumps(arm['state_seconds'], sort_keys=True))
            for rank, f in (arm.get('forward') or {}).items():
                lines.append('      fwd rank%s n=%s p50=%s mean=%s no_txn_p50=%s(n=%s) bulk_p50=%s(n=%s) ms' % (
                    rank, f['n'], f['p50_ms'], f['mean_ms'], f['no_txn_p50_ms'], f['no_txn_n'], f['bulk_p50_ms'], f['bulk_n']))
            busy = arm.get('bulk_busy') or {}
            if busy:
                lines.append('      bulk>=%d busy %s' % (BULK_ROWS, ' '.join(
                    'rank%s:%s/%sms(n=%s)' % (r, b['busy_mean_ms'], b['window_mean_ms'], b['chunks']) for r, b in busy.items())))
            if arm.get('sched'):
                lines.append('      sched %s' % json.dumps(arm['sched'], sort_keys=True))
            live = arm.get('liveness_status') or {}
            guard = arm.get('liveness_guard')
            lines.append('      liveness lines=%s before=%s after=%s guard=%s' % (
                arm.get('liveness_lines'), json.dumps(live.get('before'), sort_keys=True),
                json.dumps(live.get('after'), sort_keys=True),
                '-' if guard is None else json.dumps({k: guard.get(k) for k in ('trip', 'harness', 'trips_delta',
                                                                                  'max_live_stall_ms', 'polls',
                                                                                  'poll_failures', 'ended_replay')},
                                                     sort_keys=True)))
            if arm.get('submit_cpu'):
                lines.append('      submit-cpu %s' % json.dumps(arm['submit_cpu'], sort_keys=True))
        for note in res['notes']:
            lines.append('  NOTE ' + note)
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('records', nargs='+', help='label=<record dir>, e.g. off=<dir> offsmall=<dir>')
    parser.add_argument('--json-out', type=Path)
    args = parser.parse_args(argv)
    results = []
    for item in args.records:
        label, sep, path = item.partition('=')
        if not sep or not label:
            parser.error('records are label=<dir>')
        results.append(collect(Path(path), label))
    print(render(results))
    print('EXPLORATORY no rule, no thresholds, no verdict')
    if args.json_out:
        with args.json_out.open('x') as handle:
            json.dump(dict(schema='q2024p-j2-dmp-read/1', exploratory=True, results=results), handle, indent=2, sort_keys=True)
            handle.write('\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
