#!/usr/bin/env python3
"""C1 connected CPU probe: two chunks in flight through the real generated executor path.

C0's probe composition (e3_alpha_c0_probe: held-f's e3_native41_progress_probe.run, production install_engine, real
EngineCore.step_with_batch_queue, scheduler with the real KV manager, controller, driver, worker bridges and dispatchers, the
auto-arm policy) with FLASH_S5_STAGE_OVERLAP=1 installed beside FLASH_S5_STAGE_LOCAL=1. Modes:

  overlap-burst, overlap-burst-solo, overlap-burst-skew   selection local + overlap
  local-burst, serial-burst                                the overlap switch installed, overlap not selected

A local or serial mode must reproduce C0's record of the same mode (the self-run compares them): installing the overlap switch
moves nothing when the arm does not select it.

What an overlap mode adds over C0's checks, over the same run:
- successors were admitted, began and committed in order (the controller's and driver's counters, and the engine journal);
- stage 1's open of every successor returned exactly the lanes stage 0 had bound it to;
- each rank received the same device commands in the same order as the serial path: the per-rank list of accepted segment
  commands (chunk range, kind, stage, layer) equals the serial mode's list, and the request's outputs are the same;
- both stages had commands in flight at once (both_stages_ns), and never two commands on one stage;
- every progress row is attributed to exactly one chunk by the stage-overlap lines, and per chunk the C0 stand-in flow holds.

Fixture leaves: C0's (declared there), unchanged. The CPU-worker driver class goes under the real OverlapSegmentDriver the same
way C0's goes under StagedSegmentDriver. Declared limit, as C0: no device stream, no real QSA layer, no real collective.
"""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import re
import traceback
from unittest import mock

import e3_alpha_c0_probe as c0
import e3_native41_progress_probe as pp

MODES = ('overlap-burst', 'overlap-burst-solo', 'overlap-burst-skew', 'local-burst', 'serial-burst')
ENV = dict(c0.ENV, FLASH_S5_STAGE_OVERLAP='1')
OVERLAP_LINE = re.compile(r'^flash-s5 stage-overlap txn=(\d+) seqs=(\S+) successor=(\S+)$')


class OverlapLines(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.lines = []

    def emit(self, record):
        message = record.getMessage()
        if message.startswith('flash-s5 stage-overlap '):
            self.lines.append(message)


def seqs_of(text):
    out = []
    for part in text.split(','):
        if part == '-':
            continue
        a, _, b = part.partition('-')
        out += list(range(int(a), int(b or a) + 1))
    return out


def run(mode):
    assert mode in MODES, mode
    overlap = mode.startswith('overlap')
    local = mode.startswith(('overlap', 'local'))
    state = {}

    def connected(setting):
        import s5_startup as startup
        real = startup.install_engine

        def install(engine, *, env=None, **kwargs):
            env = dict(env or {}, FLASH_S5_NATIVE_PROGRESS=setting, **ENV)
            import vllm.v1.flash_e3.engine_driver as ed
            import vllm.v1.flash_e3.stage_progress as sp
            import vllm.v1.flash_e3.stage_overlap as so

            class OverlapWithCPUWorkers(so.OverlapSegmentDriver, ed.NativeSegmentDriver):
                pass
            sp.ring_has_headroom = lambda sender, slots: True   # C0's leaf, unchanged
            with mock.patch.object(so, 'OverlapSegmentDriver', OverlapWithCPUWorkers):
                return real(engine, env=env, **kwargs)
        with mock.patch.object(startup, 'install_engine', install):
            conn = pp.ap.Connected('1')
        state['conn'] = conn
        import vllm.v1.flash_e3.engine_driver as ed
        state['logger'] = ed._progress_logger()
        state['logger'].addHandler(lines)
        conn.transport.submit = c0.scoped_submit(conn.transport)
        return conn

    def set_mode(conn, selected):
        from vllm.v1.core.sched.interface import PauseState
        scheduler = conn.scheduler
        old = scheduler.pause_state
        scheduler.set_pause_state(PauseState.PAUSED_ALL)
        epoch = conn.driver.engine_epoch
        try:
            reply = conn.driver.submit_control(dict(action='native_progress', mode=selected, engine_epoch=epoch))
            state['selection'] = conn.driver.submit_control(dict(action='native_stage_local', local=local,
                                                                 overlap=overlap, engine_epoch=epoch))
            leased = conn.driver.auto_arm.manual_owned
            release = conn.driver.submit_control(dict(action='release_auto_arm', engine_epoch=epoch))
        finally:
            scheduler.set_pause_state(old)
        return reply, leased, release

    lines = OverlapLines()
    accepted = {}
    inflight_peak = {0: 0, 1: 0}

    def watch(conn):
        """Per rank, accepted segment commands in order; per stage, chunk commands in flight."""
        driver = conn.driver
        real_apply = driver._apply_segments

        def apply_segments(values):
            out = real_apply(values)
            tx, command = driver.tx, driver._command
            for item in values:
                if item.accepted and not item.skipped and not item.pending:
                    accepted.setdefault(item.worker_rank, []).append(
                        (tx.chunk_start, tx.chunk_end, command.kind.value, command.stage,
                         command.descriptor.layer_start))
            return out
        driver._apply_segments = apply_segments
        real_observed = driver._submit_observed

        def observed(kind, envelope, ranks, **kwargs):
            sequence = real_observed(kind, envelope, ranks, **kwargs)
            for stage in (0, 1):
                count = sum(1 for seq in driver._inflight if driver._route_stage(seq) == stage)
                inflight_peak[stage] = max(inflight_peak[stage], count)
            return sequence
        driver._submit_observed = observed

    parent, waived = c0.local_parent() if local else (pp, [])
    real_connected_fn = connected

    def connected_and_watched(setting):
        conn = real_connected_fn(setting)
        watch(conn)
        return conn
    try:
        with mock.patch.object(parent, 'connected', connected_and_watched), mock.patch.object(parent, 'set_mode', set_mode), \
                mock.patch.object(pp.ap, 'Skew', c0.stage_skew(pp.ap) if local else pp.ap.Skew):
            result = parent.run(mode.split('-', 1)[1])
    finally:
        if 'logger' in state:
            state['logger'].removeHandler(lines)
    conn = state['conn']
    checks = result['checks']

    def check(name, ok, **details):
        checks.append(dict(name=name, ok=bool(ok), **details))
        assert ok, name + ' ' + json.dumps(details, sort_keys=True, default=str)[:2000]
    status = conn.driver._status_record()['native_stage_local']
    counters, over = status['counters'], status['overlap']
    chunks = len(conn.native)
    selection = state['selection']
    check('selection-through-the-arm-sequence', selection['local'] is local and selection['overlap'] is overlap,
          reply={k: selection.get(k) for k in ('local', 'overlap', 'previous', 'previous_overlap', 'route')})
    check('switch-installed-overlap-driver', type(conn.driver).__name__ == 'OverlapWithCPUWorkers'
          and type(conn.controller).__name__ == 'StagedAdmissionController')
    check('native-chunks-ran', chunks > 0, chunks=chunks)
    check('one-command-in-flight-per-stage', max(inflight_peak.values()) <= 1, peak=inflight_peak)
    result['per_rank_segments_sha256'] = {rank: hashlib.sha256(json.dumps(rows).encode()).hexdigest()
                                          for rank, rows in sorted(accepted.items())}
    result['per_rank_segments'] = {rank: len(rows) for rank, rows in sorted(accepted.items())}
    if local:
        rows_by_line = {}
        import native_measurement as nm
        at = nm.PROGRESS_N
        bodies = [json.loads(t[len(nm.PROGRESS_LINE):]) for t in result.pop('alpha_c0_lines')]
        for body in bodies:
            rows_by_line.setdefault(body['line'], []).extend(body['rows'].get('n', []))
        rows = {r[at['seq']]: r for line in rows_by_line.values() for r in line}
        result['waived_parent_checks'] = sorted({row['name'] for row in waived})
        check('stage-counters', status['selected'] is True and counters['chunks'] == chunks
              and counters['local_releases'] == 2 * chunks, counters=counters, chunks=chunks)
    if overlap:
        parsed = [OVERLAP_LINE.match(text) for text in lines.lines]
        check('overlap-lines-parse', parsed and all(parsed), lines=len(lines.lines))
        owner = {}
        for match in parsed:
            for seq in seqs_of(match.group(2)):
                check('row-attributed-once-%d' % seq, seq not in owner)
                owner[seq] = int(match.group(1))
        check('every-row-attributed', set(rows) <= set(owner), unattributed=sorted(set(rows) - set(owner))[:8])
        per_chunk = {}
        for seq, txn in owner.items():
            if seq in rows:
                per_chunk.setdefault(txn, []).append(rows[seq])
        for txn, chunk_rows in sorted(per_chunk.items()):
            kinds = [r[at['kind']] for r in sorted(chunk_rows, key=lambda r: r[at['seq']])]
            check('stage-chunk-command-flow-%d' % txn, kinds.count('open') == 2 and kinds[0] == 'open'
                  and kinds.count('release') == 2 and 'segment' in kinds, kinds=sorted(set(kinds)))
        ctl = over['controller']
        check('successors-admitted-and-committed-in-order', ctl['successors'] > 0
              and over['counters']['successor_frames'] == ctl['successors']
              and over['counters']['overlapped_commits'] == ctl['promoted'] and ctl['promoted'] > 0,
              overlap=over)
        check('stage1-lane-equals-the-bound-expectation', over['counters']['lane_checks'] >= ctl['promoted'],
              overlap=over['counters'])
        check('both-stages-in-flight', counters['both_stages_ns'] > 0, counters=counters)
        check('native-chunks-contiguous', all(a['end'] == b['start'] for a, b in zip(conn.native, conn.native[1:])))
        result['overlap'] = over
    elif local:
        check('no-overlap-lines-when-unselected', not lines.lines, lines=len(lines.lines))
        check('overlap-counters-all-zero', not any(over['counters'].values())
              and not any(over['controller'].values()) and not any(over['refusals'].values()), overlap=over)
    result.update(native_stage_local=counters, selection=dict(local=selection.get('local'), overlap=selection.get('overlap')),
                  overlap_lines=len(lines.lines))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mode', choices=MODES, required=True)
    a = p.parse_args()
    try:
        r = run(a.mode)
        rc = 0
    except Exception as exc:
        r = dict(verdict='FAIL', mode=a.mode, error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        rc = 1
    r.update(mode=a.mode, schema='alpha-c1-connected-cpu/1', gpu_execution=False, box_evidence=False,
             probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with a.out.open('x') as f:
        json.dump(r, f, sort_keys=True, indent=2, default=str)
        f.write('\n')
    print(json.dumps({k: r.get(k) for k in ('verdict', 'mode', 'error_type', 'error')}))
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
