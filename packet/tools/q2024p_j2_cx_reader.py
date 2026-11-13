#!/usr/bin/env python3
"""Exploratory reader for a q2024p-j2 composition cohort (cx<tag>-b1 .. b4, or the dry packet; lead 12 Nov 2026).
Read-only; no rule, no thresholds, no verdict.

Every block of a cohort is built from one commit with one flag set (tools/q2024p_j2_cx_render.py writes it to
cohort.json); the reader is given that file (or the flags by --set) and the block records:

  python3 -I -B tools/q2024p_j2_cx_reader.py --cohort <render out>/cohort.json \\
      b1=<FETCH>/q18a-parity-n3s24q2024p-j2-cx<tag>-b1 b2=... b3=... b4=... [--full] [--json-out <fresh.json>]

Per block: everything the night-pack reader prints (tools/q2024p_j2_np_reader.py, told the cohort's five flags and
verify settings, so its identity NOTEs read against them), then from the decode-merge probe reader
(tools/q2024p_j2_dmp_reader.py) per arm the completed turns, seed, merge steps (on/off) and the decode liveness status
and guard record; per n3 arm the lever A append_pieces status delta (present only with FLASH_S5_APPEND_NO_BLOCK_STOP=1:
appends, pieces, fused pieces, stops skipped, fallbacks, errors; pieces per append); per arm (sp and n3) the
FLASH_S5_CONTROL_INWAIT status delta per transition class (native_admission.control_inwait, after less before: query
attempts, refusals, invalidations, armed, closed, skipped duplicate drains, mandatory drains by reason; pause-to-arm and
PAUSED_ALL ms, total and per armed; the max fields are the engine's since start, not per arm) beside the auto-arm
selections and completions in the same window, or 'off (block absent)' with the same auto-arm delta, so one read shows
whether the mechanism engaged; for the kept-SPLIT classes of FLASH_S5_CONTROL_O3_INWAIT (split-kept-entry,
split-kept-yield) also the finite cut's frames issued and applied and the arms with later decode frames still queued, and
per arm a lane-o3 line: whether the kept-SPLIT in-wait was installed, the stock O3 armed / yield_kept / kept counts and
the refusals and FULL returns by reason in the window (a scope-2 fallback ends in the stock refusal); identity NOTEs for
the composition settings (FLASH_S5_APPEND_NO_BLOCK_STOP, FLASH_S5_DECODE_GROUPS, FLASH_S5_DECODE_LIVENESS,
FLASH_S5_CONTROL_INWAIT, FLASH_S5_CONTROL_O3_INWAIT, lever D and the anatomy), for an n3 arm whose recorded decode_merge
is not the cohort's mode, for a control_inwait block absent under the flag (or present without it), for the kept-SPLIT
install absent or present against the flag, and for a kept class whose cut frames issued and applied differ; per arm the
lever
D sampler steps per sampling rank from engine-stream.log's docker-stamped 'flash-s5 sample-fast' lines (cumulative,
one per power of two of the fast count and per 1,024 sampler steps; the last line in the arm less the last one before
it, so each arm edge is within one period): sampler steps, D's fast steps, the rows drawn on the reviewed fix's exact
overflow route (2eba0dd0's retired counter, rows not steps, one line behind), the stock fallback steps by reason and
the fast share, with NOTES for an n3 arm without lines under FLASH_S5_DECODE_SAMPLE_FAST=1 or lines without it. With
FLASH_S5_DECODE_ANATOMY=1, dev_sample (ms) per sampling rank: per clean anatomy window (rows inside the core's
open..close, the sm smoke arm's window first) and per graded arm in native context, with a NOTE when no clean
sample row was read. A D pair, records off=<dir> on=<dir> each with --cohort-of <label>=<its cohort.json> differing in
FLASH_S5_DECODE_SAMPLE_FAST only, also prints the decode-step builder's adoption line per sampling rank: the first
clean window's dev_sample p50, off less on, against 0.1 ms.
Cohort lines (descriptive, the V1 proposal's arithmetic): per block R = n3 completed turns / sp completed turns pooled
over the block's two seeds (n3.g2-live-0 + n3.g2-live-1 over sp.g2-live-0 + sp.g2-live-1), and G = the geometric mean
of the blocks' R; a block with an unread arm has no R and is named.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, HERE / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


np = _load('q2024p_j2_np_reader', 'q2024p_j2_np_reader.py')
dmp = _load('q2024p_j2_dmp_reader', 'q2024p_j2_dmp_reader.py')
yr, hgb, ing = np.yr, np.hgb, np.ing

DEFAULTS = {
    'FLASH_S5_SMALL_PIECE_BATCH': '0', 'FLASH_S5_YIELD_REENTRY_SETTLE': '0', 'FLASH_S5_YIELD_NO_APPEND_OWNER': '0',
    'FLASH_S5_YIELD_WINDOW': '0', 'FLASH_S5_INGRESS_INCREMENTAL': '0', 'FLASH_S5_INGRESS_VERIFY_EVERY': None,
    'FLASH_S5_INGRESS_VERIFY_ASYNC': None, 'FLASH_S5_APPEND_NO_BLOCK_STOP': '0', 'FLASH_S5_DECODE_GROUPS': 'chunk',
    'FLASH_S5_DECODE_LIVENESS': '0', 'FLASH_S5_CONTROL_INWAIT': '0', 'FLASH_S5_CONTROL_O3_INWAIT': '0',
    'FLASH_S5_DECODE_SAMPLE_FAST': '0',
    'FLASH_S5_DECODE_ANATOMY': '0'}
NP_KEYS = (('batch', 'FLASH_S5_SMALL_PIECE_BATCH'), ('settle', 'FLASH_S5_YIELD_REENTRY_SETTLE'),
           ('y1', 'FLASH_S5_YIELD_NO_APPEND_OWNER'), ('y2', 'FLASH_S5_YIELD_WINDOW'),
           ('ingress', 'FLASH_S5_INGRESS_INCREMENTAL'))
COMPOSE_ENVS = ('FLASH_S5_APPEND_NO_BLOCK_STOP', 'FLASH_S5_DECODE_GROUPS', 'FLASH_S5_DECODE_LIVENESS',
                'FLASH_S5_CONTROL_INWAIT', 'FLASH_S5_CONTROL_O3_INWAIT', 'FLASH_S5_DECODE_SAMPLE_FAST',
                'FLASH_S5_DECODE_ANATOMY')
SP_ARMS = ('sp.g2-live-0', 'sp.g2-live-1')
N3_ARMS = ('n3.g2-live-0', 'n3.g2-live-1')
PIECES = ('appends', 'pieces', 'fused_pieces', 'stops_skipped', 'errors', 'stale_states_unpublished',
          'extra_blocks_unpublished')
INWAIT = ('attempts', 'refused', 'invalidated', 'armed', 'closed', 'drains_skipped')
INWAIT_MS = ('pause_to_arm_ms', 'decoder_stall_ms')            # totals: deltas, and per armed
INWAIT_MAX = ('pause_to_arm_max_ms', 'decoder_stall_max_ms')   # the engine's max since start (after row), not per arm
# scope 2 (FLASH_S5_CONTROL_O3_INWAIT, Codex a59b1e7f): only the kept-SPLIT classes carry these
INWAIT_CUT = ('cut_frames_issued', 'cut_frames_applied', 'arms_with_later_frames')
KEPT_CLASSES = ('split-kept-entry', 'split-kept-yield')
O3_SCALARS = ('armed', 'yield_kept', 'kept')
O3_MAPS = ('refused', 'yield_full', 'full')


def flags_of(pairs, cohort=None):
    flags = dict(DEFAULTS)
    if cohort is not None:
        doc = json.loads(Path(cohort).read_text())
        assert doc.get('schema') == 'q2024p-j2-cx-cohort/1', doc.get('schema')
        flags.update(doc['flags'])
    for pair in pairs:
        key, sep, value = pair.partition('=')
        if not sep or key not in DEFAULTS:
            raise ValueError('unknown flag %r' % pair)
        flags[key] = value
    return flags


def np_view(root, label, flags):
    """np_reader.collect with the cohort's flags as a temporary family member (its tables restored after)."""
    saved_family, saved_verify = dict(np.FAMILY), dict(np.VERIFY)
    try:
        np.FAMILY['cx'] = {k: flags[env] for k, env in NP_KEYS}
        if flags['FLASH_S5_INGRESS_INCREMENTAL'] == '1':
            np.VERIFY.clear()
            np.VERIFY.update({np.VERIFY_EVERY_ENV: flags['FLASH_S5_INGRESS_VERIFY_EVERY'],
                              np.VERIFY_ASYNC_ENV: flags['FLASH_S5_INGRESS_VERIFY_ASYNC']})
        result = np.collect(root, 'cx-' + label)
    finally:
        np.FAMILY.clear()
        np.FAMILY.update(saved_family)
        np.VERIFY.clear()
        np.VERIFY.update(saved_verify)
    result['label'] = label
    return result


def pieces_delta(arm):
    rows = hgb.status_rows(arm)
    b, a = ing.block(rows.get('before'), 'append_pieces'), ing.block(rows.get('after'), 'append_pieces')
    if type(a) is not dict:
        return None
    b = b if type(b) is dict else {}
    out = {k: (a.get(k) or 0) - (b.get(k) or 0) for k in PIECES}
    out['fallbacks'] = {k: v - (b.get('fallbacks') or {}).get(k, 0) for k, v in (a.get('fallbacks') or {}).items()
                        if v - (b.get('fallbacks') or {}).get(k, 0)}
    out['hist'] = {k: v - (b.get('hist') or {}).get(k, 0) for k, v in (a.get('hist') or {}).items()
                   if v - (b.get('hist') or {}).get(k, 0)}
    out['pieces_per_append'] = round(out['pieces'] / out['appends'], 3) if out['appends'] else None
    return out


def inwait_delta(arm):
    """control_inwait per transition class, after less before; None when the after row has no block (flag off)."""
    rows = hgb.status_rows(arm)
    b, a = ing.block(rows.get('before'), 'control_inwait'), ing.block(rows.get('after'), 'control_inwait')
    if type(a) is not dict:
        return None
    before = (b.get('transitions') if type(b) is dict else None) or {}
    out = {}
    for kind, row in sorted((a.get('transitions') or {}).items()):
        old = before.get(kind) or {}
        d = {k: (row.get(k) or 0) - (old.get(k) or 0) for k in INWAIT}
        d['drains_kept'] = {k: v - (old.get('drains_kept') or {}).get(k, 0)
                            for k, v in sorted((row.get('drains_kept') or {}).items())}
        for k in INWAIT_MS:
            d[k] = round((row.get(k) or 0.) - (old.get(k) or 0.), 3)
            d[k.replace('_ms', '_per_armed_ms')] = round(d[k] / d['armed'], 3) if d['armed'] else None
        for k in INWAIT_MAX:
            d[k + '_since_start'] = row.get(k)
        for k in INWAIT_CUT:
            if k in row:
                d[k] = (row.get(k) or 0) - (old.get(k) or 0)
        out[kind] = d
    return out


def o3_delta(arm):
    """Scope 2 beside the transition rows: whether the kept-SPLIT in-wait was installed (the after row's control_inwait
    o3_enabled, cut_pending at the after row) and the stock lane-o3 counters in the arm window (append_yield.declines.o3:
    armed, yield_kept, kept; refused / yield_full / full by reason), where a scope-2 fallback ends (its reason is the stock
    _o3_refuse reason). None when the after row has no o3 record."""
    rows = hgb.status_rows(arm)
    block = lambda row: (((row or {}).get('append_yield') or {}).get('declines') or {}).get('o3')
    b, a = block(rows.get('before')), block(rows.get('after'))
    ci = ing.block(rows.get('after'), 'control_inwait')
    out = dict(o3_inwait=bool(type(ci) is dict and ci.get('o3_enabled')),
               cut_pending_after=ci.get('cut_pending') if type(ci) is dict else None)
    if type(a) is not dict:
        out['o3'] = None
        return out
    b = b if type(b) is dict else {}
    d = {k: (a.get(k) or 0) - (b.get(k) or 0) for k in O3_SCALARS}
    for k in O3_MAPS:
        x, y = a.get(k) or {}, b.get(k) or {}
        d[k] = {r: x.get(r, 0) - y.get(r, 0) for r in sorted(set(x) | set(y)) if x.get(r, 0) - y.get(r, 0)}
    out['o3'] = d
    return out


SAMPLE_FAST = 'flash-s5 sample-fast '


def sample_fast_lines(root):
    """(wall, rank, counters, overflow) per lever D line in engine-stream.log; None without the log. ``overflow`` is the
    reviewed fix's ``overflow_rows_retired`` (2eba0dd0: rows drawn on the exact overflow route, read from a retired host
    copy, so one line behind; None while that copy is in flight), or the pre-fix ``tie_overflow_lagged``."""
    path = Path(root) / 'engine-stream.log'
    if not path.is_file():
        return None
    out = []
    with path.open(errors='replace') as handle:
        for line in handle:
            if SAMPLE_FAST not in line:
                continue
            d, wall = dmp._json_after(line, SAMPLE_FAST), dmp.stamp(line)
            if d is not None and wall is not None and type(d.get('counters')) is dict:
                out.append((wall, d.get('rank'), d['counters'],
                            d['overflow_rows_retired'] if 'overflow_rows_retired' in d else d.get('tie_overflow_lagged')))
    return out


def sample_fast_delta(lines, arm):
    """Per sampling rank over the arm's replay window: sampler steps, D's fast steps, the stock fallback steps (by
    reason), the fast share, and the rows D drew on the exact overflow route (the retired counter's last value in the
    arm less its last value before; rows, not steps: a step's row count is not on the host without a sync; None when no
    retired value was read in the arm)."""
    start = arm.get('replay_start_s')
    if lines is None or type(start) not in (int, float):
        return None
    end = start + ((arm.get('replay_config') or {}).get('DURATION') or 480.0)
    out = {}
    for rank in sorted({r for _, r, _, _ in lines}, key=str):
        rows = [(w, c, t) for w, r, c, t in lines if r == rank]
        before = [c for w, c, _ in rows if w < start]
        inside = [(c, t) for w, c, t in rows if start <= w <= end]
        if not inside:
            continue
        a, b = inside[-1][0], (before[-1] if before else {})
        d = {k: (a.get(k) or 0) - (b.get(k) or 0) for k in a}
        steps = d.get('fast', 0) + d.get('stock', 0)
        seen_in = [x for _, x in inside if type(x) is int]
        seen_before = [x for w, _, x in rows if w < start and type(x) is int]
        out[str(rank)] = dict(steps=steps, fast=d.get('fast', 0), stock=d.get('stock', 0),
                              stock_by_reason={k[len('stock_'):]: v for k, v in sorted(d.items())
                                               if k.startswith('stock_') and v},
                              fast_share=round(d.get('fast', 0) / steps, 4) if steps else None,
                              overflow_rows=(seen_in[-1] - (seen_before[-1] if seen_before else 0)) if seen_in else None,
                              lines=len(inside))
    return out


ANATOMY_CORE = 'flash-s5 decode-anatomy '
HANDLER = 'flash-s5 handler '
HANDLER_SCHEMA = 's5-handler-anatomy/1'      # == tools/stage1_anatomy_report.SCHEMA (patches/s5_engine/handler_anatomy.py)
# main-row fields and flags as the writer's (patches/s5_engine/handler_anatomy.M_FIELDS / FLAG_*), read the way
# tools/stage1_anatomy_report.py reads them; that tool and its harness imports are not in a packet bundle, so this
# reader parses the few fields it needs itself (test_q2024p_j2_cx pins these to the writer and the torn-line rule)
HANDLER_M_FIELDS = ('seq', 'kind', 'out', 'flags', 't_enter', 't_exit', 'wall_ns', 'cpu_ns', 'observer_ns',
                    'dev_forward_ns', 'dev_sample_ns', 'idle_before_launch', 'idle_after_launch', 'handle_output_ns',
                    'seams')
HANDLER_BAD_FLAGS = 1 | 2 | 4                # FLAG_FAILED | FLAG_ABORTED | FLAG_INSTALL
INNER_HEADER = re.compile(r'\((?:Worker_\w+|EngineCore\w*) pid=\d+\) ?')   # == harness/step_a_torn_lines.INNER_HEADER
SAMPLING_RANKS = ('2', '3')        # the last PP stage samples (TP2 x PP2)


def handler_body(line):
    """The handler line's body, or None when it is torn (no JSON dict, or another writer's header inside it)."""
    body = line[line.find(HANDLER) + len(HANDLER):]
    if INNER_HEADER.search(body):
        return None
    try:
        value = json.loads(body)
    except ValueError:
        return None
    return value if type(value) is dict else None


def handler_rows(records):
    """Main rows of the whole-or-partial flushes: (stamp, rank, row dict with t_enter in ns), as the anatomy report's
    assemble (t_enter = base_ns + t_enter_us * 1000); a flush's stamp is its earliest part's."""
    groups = {}
    for stamp, body in records:
        groups.setdefault((body.get('rank'), body.get('line'), body.get('base_ns')), []).append((stamp, body))
    out = []
    for (rank, _, base), parts in groups.items():
        stamps = [s for s, _ in parts if s is not None]
        when = min(stamps) if stamps else None
        for _, body in parts:
            for item in (body.get('rows') or {}).get('m', []):
                r = dict(zip(HANDLER_M_FIELDS, item))
                r['t_enter'] = (base or 0) + r['t_enter'] * 1000 if type(r.get('t_enter')) is int else None
                out.append((when, rank, r))
    return out


def _dist(values):
    if not values:
        return None
    ordered = sorted(values)
    return dict(n=len(ordered), p50=round(dmp.quantile(ordered, 0.5), 4), p90=round(dmp.quantile(ordered, 0.9), 4))


def dev_sample(root, arms):
    """dev_sample (device time of the sample_tokens RPC's sampling, handler-anatomy rows) per sampling rank:
    ``clean`` per FLASH_S5_DECODE_ANATOMY window (rows whose t_enter is inside the core's open_ns..close_ns, the same
    host monotonic clock; no native progress there by the window's own condition), each window placed in the graded arm
    whose replay window holds its core line's stamp, else 'before-arms' (the sm smoke arm's window); and ``native`` per
    graded arm (the other sample_tokens rows, by their flush's stamp). None without engine-stream.log."""
    path = Path(root) / 'engine-stream.log'
    if not path.is_file():
        return None
    windows, records, torn, errors = {}, [], 0, 0
    with path.open(errors='replace') as handle:
        for line in handle:
            if ANATOMY_CORE in line:
                d = dmp._json_after(line, ANATOMY_CORE)
                if d is not None and d.get('kind') == 'core-steps' and type(d.get('open_ns')) is int:
                    windows.setdefault(d.get('window'), (d['open_ns'], d.get('close_ns'), dmp.stamp(line)))
            elif HANDLER in line:
                body = handler_body(line)
                if body is None:
                    torn += 1
                    continue
                if body.get('schema') != HANDLER_SCHEMA:
                    errors += 1
                    continue
                records.append((dmp.stamp(line), body))
    incomplete = 0
    for key in {(b.get('rank'), b.get('line'), b.get('base_ns')) for _, b in records}:
        parts = sorted(b.get('part', -1) for _, b in records if (b.get('rank'), b.get('line'), b.get('base_ns')) == key)
        count = next((b.get('parts') for _, b in records if (b.get('rank'), b.get('line'), b.get('base_ns')) == key), 0)
        incomplete += parts != list(range(count or 0))
    spans = {}
    for name, arm in arms.items():
        start = arm.get('replay_start_s') if arm else None
        if type(start) in (int, float):
            spans[name] = (start, start + ((arm.get('replay_config') or {}).get('DURATION') or 480.0))

    def arm_at(wall):
        for name, (a, b) in spans.items():
            if wall is not None and a <= wall <= b:
                return name
        if wall is None:
            return 'unstamped'
        return 'before-arms' if spans and wall < min(a for a, _ in spans.values()) else 'between'

    clean, native = {}, {}
    for when, rank, r in handler_rows(records):
        if r.get('kind') != 's' or type(r.get('dev_sample_ns')) is not int or (r.get('flags') or 0) & HANDLER_BAD_FLAGS:
            continue
        ms = r['dev_sample_ns'] / 1e6
        hit = [w for w, (a, b, _) in windows.items() if type(b) is int and r['t_enter'] is not None and a <= r['t_enter'] <= b]
        if hit:
            clean.setdefault(hit[0], {}).setdefault(str(rank), []).append(ms)
        else:
            native.setdefault(arm_at(when), {}).setdefault(str(rank), []).append(ms)
    return dict(
        clean={str(w): dict(arm=arm_at(windows[w][2]), close_minus_open_s=round((windows[w][1] - windows[w][0]) / 1e9, 3)
                            if type(windows[w][1]) is int else None,
                            ranks={k: _dist(v) for k, v in sorted(clean.get(w, {}).items())})
               for w in sorted(windows, key=str)},
        native={a: {k: _dist(v) for k, v in sorted(ranks.items())} for a, ranks in sorted(native.items())},
        torn_lines=torn, schema_errors=errors, incomplete_flushes=incomplete)


def auto_arm_delta(arm):
    """Selections and completions in the arm's window (the status rows' auto_arm block), flag on or off."""
    rows = hgb.status_rows(arm)
    b, a = (rows.get('before') or {}).get('auto_arm'), (rows.get('after') or {}).get('auto_arm')
    if type(a) is not dict or type(b) is not dict:
        return None
    return {k: (a.get(k) or 0) - (b.get(k) or 0) for k in ('selection_seq', 'completed')}


def compose_notes(root, label, flags, dview):
    notes, got = [], {}
    for env in COMPOSE_ENVS:
        got[env] = yr.identity_env(root, env)
        want = flags[env]
        for phase, value in got[env].items():
            if value == 'unset':
                value = DEFAULTS[env]
            if value != want:
                notes.append('identity-%s-%s-%s-%s' % (label, env, phase, value))
    for name, arm in (dview.get('arms') or {}).items():
        step = (arm.get('step') or {}).get('requested_merge')
        if name.startswith('n3.') and step is not None and step != flags['FLASH_S5_DECODE_GROUPS']:
            notes.append('decode-merge-%s-%s-%s' % (label, name, step))
    return got, notes


def collect(root, label, flags):
    root = Path(root)
    result = np_view(root, label, flags)
    dview = dmp.collect(root, label)
    result['decode'] = {name: {k: arm.get(k) for k in ('completed_turns', 'seed', 'merge', 'step', 'liveness_status',
                                                       'liveness_guard', 'liveness_lines')}
                        for name, arm in dview['arms'].items()}
    result['decode_engine_crashed'] = dview.get('engine_crashed')
    done, notes_extra = {}, []
    fast_lines = sample_fast_lines(root)
    d_on = flags['FLASH_S5_DECODE_SAMPLE_FAST'] == '1'
    loaded = {}
    for name in SP_ARMS + N3_ARMS:
        arm = loaded[name] = hgb.load(root / ('arm-' + name + '.json'))
        # completed turns as the dmp reader counts them (warm turns less those finished after the window), per arm record
        done[name] = None if arm is None else dmp.turns_per_hour(arm)[0]
        if arm is not None and name in N3_ARMS and name in result['arms']:
            result['arms'][name]['append_pieces'] = pieces_delta(arm)
        if arm is not None:
            row = result['arms'].setdefault(name, {})
            row['control_inwait'], row['auto_arm_delta'] = inwait_delta(arm), auto_arm_delta(arm)
            on = flags['FLASH_S5_CONTROL_INWAIT'] == '1'
            if on and row['control_inwait'] is None:
                notes_extra.append('control-inwait-absent-%s-%s' % (label, name))
            elif not on and row['control_inwait'] is not None:
                notes_extra.append('control-inwait-present-%s-%s' % (label, name))
            row['control_o3'] = o3_delta(arm)
            o3_on = flags['FLASH_S5_CONTROL_O3_INWAIT'] == '1'
            if o3_on and not row['control_o3']['o3_inwait']:
                notes_extra.append('control-o3-inwait-absent-%s-%s' % (label, name))
            elif not o3_on and row['control_o3']['o3_inwait']:
                notes_extra.append('control-o3-inwait-present-%s-%s' % (label, name))
            for kind, d in (row['control_inwait'] or {}).items():
                if kind in KEPT_CLASSES and d.get('cut_frames_issued') != d.get('cut_frames_applied'):
                    notes_extra.append('control-o3-cut-unapplied-%s-%s' % (label, name))
            row['sample_fast'] = sample_fast_delta(fast_lines, arm)
            if d_on and name in N3_ARMS and not row['sample_fast']:
                notes_extra.append('sample-fast-silent-%s-%s' % (label, name))
            elif not d_on and row['sample_fast']:
                notes_extra.append('sample-fast-present-%s-%s' % (label, name))
    if flags['FLASH_S5_DECODE_ANATOMY'] == '1':
        result['dev_sample'] = dev_sample(root, loaded)
        if result['dev_sample'] is not None and not any(
                (w['ranks'].get(r) or {}).get('n') for w in result['dev_sample']['clean'].values() for r in SAMPLING_RANKS):
            notes_extra.append('decode-anatomy-no-clean-sample-rows-%s' % label)
    result['identity_compose'], notes = compose_notes(root, label, flags, dview)
    result['compose_notes'] = notes + notes_extra + ['dmp-' + n for n in dview['notes']]
    result['completed'] = done
    if all(type(v) is int for v in done.values()) and sum(done[n] for n in SP_ARMS):
        result['R'] = round(sum(done[n] for n in N3_ARMS) / sum(done[n] for n in SP_ARMS), 4)
    else:
        result['R'] = None
    return result


def cohort_line(results):
    rs = [r['R'] for r in results if r['R']]
    missing = [r['label'] for r in results if not r['R']]
    g = round(math.exp(sum(math.log(x) for x in rs) / len(rs)), 4) if rs else None
    return dict(R={r['label']: r['R'] for r in results}, G=g, blocks_with_R=len(rs), blocks_without_R=missing)


def inwait_lines(name, arm):
    if 'control_inwait' not in arm:
        return ['  %-13s control_inwait - (arm unread)' % name]
    ci, aa = arm['control_inwait'], arm.get('auto_arm_delta')
    aa = '-' if aa is None else 'selected=%s completed=%s' % (aa['selection_seq'], aa['completed'])
    if ci is None:
        return ['  %-13s control_inwait off (block absent)  auto-arm %s' % (name, aa)]
    if not ci:
        return ['  %-13s control_inwait on, no transition yet  auto-arm %s' % (name, aa)]
    lines = ['  %-13s control_inwait on  auto-arm %s' % (name, aa)]
    for kind, d in ci.items():
        lines.append('        %-18s attempts=%s refused=%s invalidated=%s armed=%s closed=%s drains_skipped=%s kept=%s '
                     'pause_to_arm_ms=%s/armed=%s (max since start %s) PAUSED_ALL_ms=%s/armed=%s (max since start %s)' % (
                         kind, d['attempts'], d['refused'], d['invalidated'], d['armed'], d['closed'],
                         d['drains_skipped'], np.fmt(d['drains_kept']), d['pause_to_arm_ms'],
                         d['pause_to_arm_per_armed_ms'], d['pause_to_arm_max_ms_since_start'], d['decoder_stall_ms'],
                         d['decoder_stall_per_armed_ms'], d['decoder_stall_max_ms_since_start']))
        if kind in KEPT_CLASSES:
            lines.append('          %-16s cut issued=%s applied=%s arms_with_later_frames=%s (kept-SPLIT: success leaves '
                         'typed-route-fence 0; a fallback counts fallback-full-cut and typed-route-fence)' % (
                             '', d.get('cut_frames_issued'), d.get('cut_frames_applied'),
                             d.get('arms_with_later_frames')))
    o3 = arm.get('control_o3')
    if o3 is not None:
        x = o3.get('o3')
        lines.append('        %-18s o3_inwait=%s cut_pending_after=%s %s' % (
            'lane-o3', 'on' if o3.get('o3_inwait') else 'off', o3.get('cut_pending_after'),
            '- (no o3 record)' if x is None else 'armed=%s yield_kept=%s kept=%s refused=%s yield_full=%s full=%s' % (
                x['armed'], x['yield_kept'], x['kept'], np.fmt(x['refused']), np.fmt(x['yield_full']),
                np.fmt(x['full']))))
    return lines


def sample_fast_render(name, arm):
    if 'sample_fast' not in arm:
        return []
    got = arm['sample_fast']
    if got is None:
        return ['  %-13s sample_fast - (no engine-stream.log or arm start)' % name]
    if not got:
        return ['  %-13s sample_fast no lines in the arm' % name]
    return ['  %-13s sample_fast rank%s steps=%s fast=%s overflow_rows=%s stock_fallback=%s share=%s '
            'stock_by_reason=%s lines=%s' % (
                name, rank, d.get('steps'), d['fast'], d.get('overflow_rows'), d['stock'], d['fast_share'],
                np.fmt(d['stock_by_reason']), d['lines']) for rank, d in got.items()]


def dev_sample_render(r):
    got = r.get('dev_sample')
    if 'dev_sample' not in r:
        return []
    if got is None:
        return ['  dev_sample - (no engine-stream.log)']
    lines = ['  dev_sample ms (sample_tokens rows, per sampling rank) torn=%s schema_errors=%s incomplete_flushes=%s' % (
        got['torn_lines'], got['schema_errors'], got['incomplete_flushes'])]
    for w, d in got['clean'].items():
        lines.append('    clean window %s in %s (%ss) %s' % (w, d['arm'], d['close_minus_open_s'], np.fmt(d['ranks'])))
    if not got['clean']:
        lines.append('    clean - (no decode-anatomy window)')
    for a, ranks in got['native'].items():
        lines.append('    native-context %s %s' % (a, np.fmt(ranks)))
    return lines


def d_pair(off, on):
    """The decode-step builder's adoption line (SIZING.md, STATUS.md): keep D only if dev_sample falls by at least
    0.1 ms, on ranks 2/3 (the sampling stage), D on against D off. Per sampling rank, the first clean window with rows
    on that rank is read in each packet (the sm arm's 20 s window; the 20 s budget usually ends the anatomy inside it),
    its p50 off less its p50 on. Descriptive: the lead applies the line."""
    out = {}
    for rank in SAMPLING_RANKS:
        got = []
        for r in (off, on):
            clean = (r.get('dev_sample') or {}).get('clean') or {}
            first = next((w['ranks'][rank] for _, w in sorted(clean.items(), key=lambda kv: kv[0])
                          if (w['ranks'].get(rank) or {}).get('n')), None)
            got.append(first)
        a, b = got
        drop = round(a['p50'] - b['p50'], 4) if a and b else None
        out[rank] = dict(off=a, on=b, drop_ms=drop, falls_by_0_1_ms=None if drop is None else drop >= 0.1)
    return out


def render(results, full=False):
    lines = [np.render(results, full=full)]
    for r in results:
        lines.append('== %s composition  completed=%s R(n3/sp pooled)=%s' % (r['label'], np.fmt(r['completed']), r['R']))
        for name, d in r['decode'].items():
            step, guard = d.get('step') or {}, d.get('liveness_guard')
            lines.append('  %-13s seed=%s merge req/ack=%s/%s steps=%s on=%s off=%s liveness lines=%s guard=%s' % (
                name, d.get('seed'), step.get('requested_merge'), step.get('acknowledged_merge'),
                (d.get('merge') or {}).get('step_merge_steps'), (d.get('merge') or {}).get('step_merge_on'),
                (d.get('merge') or {}).get('step_merge_off'), d.get('liveness_lines'),
                '-' if guard is None else np.fmt({k: guard.get(k) for k in ('trip', 'harness', 'trips_delta',
                                                                           'max_live_stall_ms', 'ended_replay')})))
            pieces = (r['arms'].get(name) or {}).get('append_pieces')
            if pieces is not None:
                lines.append('      append_pieces %s' % np.fmt(pieces))
        for name in SP_ARMS + N3_ARMS:                 # every arm, sp beside n3, in one place
            lines.extend(inwait_lines(name, r['arms'].get(name) or {}))
        for name in SP_ARMS + N3_ARMS:
            lines.extend(sample_fast_render(name, r['arms'].get(name) or {}))
        lines.extend(dev_sample_render(r))
        if r['compose_notes']:
            lines.append('  NOTES %s' % np.fmt(sorted(set(r['compose_notes']))))
    c = cohort_line(results)
    lines.append('== cohort (descriptive) R=%s G=%s blocks_with_R=%d without=%s' % (
        np.fmt(c['R']), c['G'], c['blocks_with_R'], np.fmt(c['blocks_without_R'])))
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('records', nargs='+', help='<block>=<dir>, e.g. b1=<dir> (labels are free; one per record)')
    parser.add_argument('--cohort', type=Path, help="the render's cohort.json (its flags)")
    parser.add_argument('--set', dest='flags', action='append', default=[], metavar='FLAG=VALUE')
    parser.add_argument('--cohort-of', action='append', default=[], metavar='LABEL=COHORT_JSON',
                        help="one record's own cohort.json (a D pair: off=<off cohort.json>, on=<on cohort.json>)")
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--json-out', type=Path)
    args = parser.parse_args(argv)
    try:
        flags = flags_of(args.flags, args.cohort)
    except ValueError as exc:
        parser.error(str(exc))
    own = {}
    for item in args.cohort_of:
        label, sep, path = item.partition('=')
        if not sep:
            parser.error('--cohort-of is LABEL=COHORT_JSON')
        own[label] = flags_of(args.flags, Path(path))
    results, used = [], {}
    for item in args.records:
        label, sep, path = item.partition('=')
        if not sep or not label or '-' in label:
            parser.error('records are <block>=<dir>, the block label without "-"')
        used[label] = own.get(label, flags)
        results.append(collect(Path(path), label, used[label]))
    pair = None
    if set(used) == {'off', 'on'} and set(own) == {'off', 'on'}:
        diff = sorted(k for k in DEFAULTS if used['off'][k] != used['on'][k])
        if diff != ['FLASH_S5_DECODE_SAMPLE_FAST'] or used['on']['FLASH_S5_DECODE_SAMPLE_FAST'] != '1' or \
                used['off']['FLASH_S5_DECODE_ANATOMY'] != '1':
            parser.error('a D pair differs in FLASH_S5_DECODE_SAMPLE_FAST only (on=1), anatomy on in both; got %s' % diff)
        by = {r['label']: r for r in results}
        pair = d_pair(by['off'], by['on'])
    print(render(results, full=args.full))
    if pair is not None:
        print('== D pair (decode-step builder: keep D only if dev_sample falls by >= 0.1 ms on ranks 2/3; '
              'first clean window per rank, p50)')
        for rank, d in pair.items():
            print('  rank%s off=%s on=%s drop_ms=%s falls_by_0.1=%s' % (
                rank, np.fmt(d['off']), np.fmt(d['on']), d['drop_ms'], d['falls_by_0_1_ms']))
    print('EXPLORATORY no rule, no thresholds, no verdict')
    if args.json_out:
        with args.json_out.open('x') as handle:
            json.dump(dict(schema='q2024p-j2-cx-read/1', exploratory=True, flags=flags, results=results,
                           cohort=cohort_line(results), flags_per_record=used, d_pair=pair),
                      handle, indent=2, sort_keys=True, default=str)
            handle.write('\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
