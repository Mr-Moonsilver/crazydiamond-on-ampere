#!/usr/bin/env python3
"""Final-engine serve (b16's serve_lane.py, unchanged logic): the n3 lane's typed control, exactly the cx runner's sequence (CT <ct-id>, loopback, the bundle's clients).

b15 sent a typed 'status' first.  On a fresh engine the native session has no run_id until the first typed 'prepare'
binds it (native_overlap_session.execute returns result['run_id'] = self.run_id, None before that), so the client's
identity check refused the reply.  The cx runner never does that: its first typed exchange is the startup grade.

  serve_lane.py cx-step --h <bundle> -- <module.py> <args...>
        run one cx step's module in this process, unchanged, with the identity diagnostic installed:
          e3_split_startup_grade.py --url U --run-id R --package <overlay> --out startup-grade.json
              (NativeRuntimeClient(url, run_id), no epoch: prepare('full', 'full-prefill-solo') binds the run_id,
               then identity, then resume; the EPOCH is the grade's engine_epoch, verdict PASS)
          e3_native39_trunk_checks.py full-idle --url U --run-id R --epoch E ...   (the arm entry: typed FULL, lease)
  serve_lane.py select --h <bundle> --url U --run-id R --epoch E --out F
        cxay12's n3 row (n3.g2-live-0): compose-4 entry_status, then compose-5 select_and_release, auto-arm released.
        No replay, no grading; nothing here re-selects.
  serve_lane.py status --h <bundle> --url U --run-id R [--epoch E]   one numbers-only status line (JSON); same run_id

Identity diagnostic: when a reply fails the client's identity check (response-identity-differs), one stderr line
  SERVE_IDENTITY_DIFF action=<sent> fields=<name:sent->got,...>
names each differing field (schema, action, run_id, command_seq, client_id) with the sent and returned values (labels
and integers only; never the key, never the body).  The exception itself is re-raised unchanged.
The key comes from VLLM_API_KEY in the environment (the bundle's LoopbackControl reads it); it is never printed.
"""
import argparse
import json
import os
import sys
import threading
import time

ROW = ('n3', 2, 'live', 0)          # cxfx42's (and cxay12's) n3 arms ran n3.g2-live-0 and n3.g2-live-1: the same selection, index 0
_LAST = threading.local()



def _packet_error(exc):
    import sys, traceback
    frame=traceback.extract_tb(exc.__traceback__)[-1]
    print(f'PACKET_CLIENT_ERROR {type(exc).__name__}: {str(exc)[:256]} at {frame.filename}:{frame.lineno}', file=sys.stderr)

class _JsonShim:
    """The real json module with loads() remembering its last result per thread (for the diagnostic only)."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def loads(self, *args, **kwargs):
        result = self._real.loads(*args, **kwargs)
        _LAST.result = result
        return result


def _show(value):
    if value is None or type(value) in (int, bool):
        return repr(value)
    if type(value) is str:
        return repr(value[:80])
    return '<%s>' % type(value).__name__


def identity_diff(command, result):
    """The differing identity fields as 'name:sent->got' (the checks of LoopbackControl/RefusalKeepingControl._post)."""
    if type(result) is not dict:
        return ['reply:<%s>' % type(result).__name__]
    out = []
    if result.get('schema') != 1:
        out.append('schema:1->%s' % _show(result.get('schema')))
    for name in ('action', 'run_id'):
        if result.get(name) != command.get(name):
            out.append('%s:%s->%s' % (name, _show(command.get(name)), _show(result.get(name))))
    for name in ('command_seq', 'client_id'):
        if name in result and result[name] != command.get(name):
            out.append('%s:%s->%s' % (name, _show(command.get(name)), _show(result[name])))
    if 'error' in result:
        out.append('error:%s' % _show(result.get('error')))
    return out or ['none-of-the-session-fields(adapter-path-echo)']


def install_diag():
    import e3_native_qualification_driver as qd
    import e3_native_refusal_client as rc
    for module in (qd, rc):
        if not isinstance(module.json, _JsonShim):
            module.json = _JsonShim(module.json)
    for cls in (qd.LoopbackControl, rc.RefusalKeepingControl):
        original = cls.__dict__.get('_post')
        if original is None or getattr(original, '_serve_diag', False):
            continue

        def wrapped(self, path, command, timeout_s=None, _original=original):
            _LAST.result = None
            try:
                return _original(self, path, command, timeout_s)
            except qd.QualificationRefused as exc:
                if str(exc) == 'response-identity-differs':
                    print('SERVE_IDENTITY_DIFF action=%s fields=%s' % (
                        _show(command.get('action') if type(command) is dict else None),
                        ','.join(identity_diff(command if type(command) is dict else {}, getattr(_LAST, 'result', None)))),
                        file=sys.stderr, flush=True)
                raise
        wrapped._serve_diag = True
        cls._post = wrapped


def load(h):
    sys.path[:0] = [h + '/e3_builder', h]
    install_diag()
    import n3j2_box  # noqa: F401  binds the n3j2 table into compose-5/compose-4 (FLASH_S5_DECODE_GROUPS=chunk)
    import e3_compose5_box as c5
    return c5


def client_for(c5, a, epoch=None):
    return c5._box.refusal.RefusalKeepingClient(a.url, a.run_id, epoch=epoch, timeout_s=a.control_timeout_s)


def compact(status):
    """Numbers and short states only: the lane hit counter and the lease/route."""
    adm = status.get('native_admission') or {}
    auto = status.get('auto_arm') or {}
    out = dict(engine_epoch=status.get('engine_epoch'), route=status.get('route'),
               scheduler_requests=status.get('scheduler_requests'))
    for k in ('admitted', 'refused', 'alloc_failures', 'split_stock_deferrals', 'split_hold_timeouts', 'refusal_seq'):
        v = adm.get(k)
        out['native_' + k] = v if type(v) in (int, float) else None
    for k in ('active', 'manual_owned', 'phase', 'selection_seq'):
        v = auto.get(k)
        out['auto_arm_' + k] = v if type(v) in (int, float, bool, str) or v is None else type(v).__name__
    prog = status.get('native_progress') or {}
    for k in ('chunks', 'commands', 'direct_commands', 'stock_rpcs', 'mode', 'recording', 'line_errors'):
        v = prog.get(k)
        out['progress_' + k] = v if type(v) in (int, float, bool, str) or v is None else type(v).__name__
    last = adm.get('last_refusal')
    out['native_last_refusal_reason'] = (last or {}).get('reason') if type(last) is dict else None
    return out


def cx_step(h, argv):
    """One cx step's module main, unchanged, in this process (diagnostic installed first)."""
    if not argv:
        raise SystemExit('cx-step needs <module.py> <args...>')
    name = argv[0]
    if name not in ('e3_split_startup_grade.py', 'e3_native39_trunk_checks.py'):
        raise SystemExit('cx-step: module not in this packet\'s list: %s' % name)
    sys.path[:0] = [h + '/e3_builder', h]
    install_diag()
    import importlib
    module = importlib.import_module(name[:-3])
    sys.argv = [os.path.join(h, name)] + list(argv[1:])
    return module.main(list(argv[1:]))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ['cx-step']:
        q = argparse.ArgumentParser(prog='serve_lane.py cx-step')
        q.add_argument('--h', required=True)
        q.add_argument('rest', nargs=argparse.REMAINDER)
        b = q.parse_args(argv[1:])
        rest = b.rest[1:] if b.rest[:1] == ['--'] else b.rest
        return cx_step(b.h, rest)
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('action', choices=('select', 'status'))
    p.add_argument('--h', required=True)
    p.add_argument('--url', default='http://127.0.0.1:8000')
    p.add_argument('--run-id', required=True)
    p.add_argument('--epoch', type=int)
    p.add_argument('--control-timeout-s', type=float, default=60.0)
    p.add_argument('--out')
    p.add_argument('--dry', action='store_true', help='select: resolve the row and print it, no control call')
    a = p.parse_args(argv)
    c5 = load(a.h)
    if a.action == 'status':
        client = client_for(c5, a, a.epoch)
        status = client.call('status')
        print(json.dumps(dict(compact(status), t=time.time()), sort_keys=True))
        return 0
    if not a.dry and (a.epoch is None or not a.out):
        p.error('select needs --epoch and --out')
    with c5._bound():
        T = c5._box.T
        name = T.label(*ROW)
        row = T.by_label().get(name)
        if row is None or row.get('native_off') or row.get('kind') != 'replay':
            print('SERVE_LANE REFUSED row %s not the n3 replay row' % name)
            return 2
        if a.dry:
            keys = ('kind', 'label', 'attn', 'yield_on', 'o3_split', 'ahead_nosync', 'k2_settle', 'lazy_recv', 'decode_merge',
                    'stage_local', 'inwait', 'next_levers', 'native_off', 'size', 'selection')
            print('SERVE_LANE_DRY ' + json.dumps({k: row.get(k) for k in keys}, sort_keys=True, default=str))
            return 0
        client = client_for(c5, a, a.epoch)
        record = c5._box.base_record(argparse.Namespace(run_id=a.run_id), row, client)
        record['serve'] = dict(schema='serve-lane/2', row=name, attn=row.get('attn'), o3_split=row.get('o3_split'),
                               ahead_nosync=row.get('ahead_nosync'), k2_settle=row.get('k2_settle'))
        rc = 0
        try:
            c5._box.entry_status(client, row, record)
            c5.select_and_release(client, row, record)
            after = client.call('status')
            record['statuses'].append(c5.status_row('after-select', after))
            record['serve']['after'] = compact(after)
            auto = after.get('auto_arm') or {}
            if auto.get('manual_owned') is not False:
                record['reasons'].append('auto-arm-not-released')
            record['complete'] = not record['reasons']
        except Exception as exc:  # noqa: BLE001 - recorded, never retried
            _packet_error(exc)
            record['reasons'].append('select-refused:%s:%s' % (type(exc).__name__, str(exc)[:160]))
            record['complete'] = False
        if record['complete'] is not True:
            rc = 2
        record['wall_end_s'] = time.time()
        with open(a.out, 'x') as handle:
            json.dump(record, handle, sort_keys=True, default=str)
        print('SERVE_LANE %s %s %s' % (name, 'SELECTED' if rc == 0 else 'REFUSED', ';'.join(record['reasons'])))
        print('SERVE_LANE_AFTER ' + json.dumps(record['serve'].get('after'), sort_keys=True))
        return rc


if __name__ == '__main__':
    raise SystemExit(main())
