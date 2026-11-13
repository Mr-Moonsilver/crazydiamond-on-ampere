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
whether the mechanism engaged; identity NOTEs for the four composition settings (FLASH_S5_APPEND_NO_BLOCK_STOP,
FLASH_S5_DECODE_GROUPS, FLASH_S5_DECODE_LIVENESS, FLASH_S5_CONTROL_INWAIT), for an n3 arm whose recorded decode_merge is
not the cohort's mode, and for a control_inwait block absent under the flag (or present without it).
Cohort lines (descriptive, the V1 proposal's arithmetic): per block R = n3 completed turns / sp completed turns pooled
over the block's two seeds (n3.g2-live-0 + n3.g2-live-1 over sp.g2-live-0 + sp.g2-live-1), and G = the geometric mean
of the blocks' R; a block with an unread arm has no R and is named.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path

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
    'FLASH_S5_DECODE_LIVENESS': '0', 'FLASH_S5_CONTROL_INWAIT': '0', 'FLASH_S5_DECODE_SAMPLE_FAST': '0',
    'FLASH_S5_DECODE_ANATOMY': '0'}
NP_KEYS = (('batch', 'FLASH_S5_SMALL_PIECE_BATCH'), ('settle', 'FLASH_S5_YIELD_REENTRY_SETTLE'),
           ('y1', 'FLASH_S5_YIELD_NO_APPEND_OWNER'), ('y2', 'FLASH_S5_YIELD_WINDOW'),
           ('ingress', 'FLASH_S5_INGRESS_INCREMENTAL'))
COMPOSE_ENVS = ('FLASH_S5_APPEND_NO_BLOCK_STOP', 'FLASH_S5_DECODE_GROUPS', 'FLASH_S5_DECODE_LIVENESS',
                'FLASH_S5_CONTROL_INWAIT', 'FLASH_S5_DECODE_SAMPLE_FAST', 'FLASH_S5_DECODE_ANATOMY')
SP_ARMS = ('sp.g2-live-0', 'sp.g2-live-1')
N3_ARMS = ('n3.g2-live-0', 'n3.g2-live-1')
PIECES = ('appends', 'pieces', 'fused_pieces', 'stops_skipped', 'errors', 'stale_states_unpublished',
          'extra_blocks_unpublished')
INWAIT = ('attempts', 'refused', 'invalidated', 'armed', 'closed', 'drains_skipped')
INWAIT_MS = ('pause_to_arm_ms', 'decoder_stall_ms')            # totals: deltas, and per armed
INWAIT_MAX = ('pause_to_arm_max_ms', 'decoder_stall_max_ms')   # the engine's max since start (after row), not per arm


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
        out[kind] = d
    return out


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
    for name in SP_ARMS + N3_ARMS:
        arm = hgb.load(root / ('arm-' + name + '.json'))
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
    return lines


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
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--json-out', type=Path)
    args = parser.parse_args(argv)
    try:
        flags = flags_of(args.flags, args.cohort)
    except ValueError as exc:
        parser.error(str(exc))
    results = []
    for item in args.records:
        label, sep, path = item.partition('=')
        if not sep or not label or '-' in label:
            parser.error('records are <block>=<dir>, the block label without "-"')
        results.append(collect(Path(path), label, flags))
    print(render(results, full=args.full))
    print('EXPLORATORY no rule, no thresholds, no verdict')
    if args.json_out:
        with args.json_out.open('x') as handle:
            json.dump(dict(schema='q2024p-j2-cx-read/1', exploratory=True, flags=flags, results=results,
                           cohort=cohort_line(results)), handle, indent=2, sort_keys=True, default=str)
            handle.write('\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
