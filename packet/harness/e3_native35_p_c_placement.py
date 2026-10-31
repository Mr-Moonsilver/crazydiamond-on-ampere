#!/usr/bin/env python3
"""Native35 P-c row4: the L2 placement runner with the decode cohort stopped at
the client window boundary, plus the unchanged row4 CLI.

On the box (q18a-native35-p-b, 31 Oct 2026) the L2 runner completed its client
window (8/8 streams, +40 tokens, token_bound), then left the eight decode
streams running to their 64-token budget while it waited ~80 s for the native
prefill's first token.  The replay observer is declared for a 40-replay decode
window (``PLACEMENT_REPLAYS``); the open cohort put ~125 decode-split replays
per rank into it, enough to overflow the 131072-record kernel ring, and the
observer_end reply exceeded the 112 MiB transport cap.

``StoppedWindowPlacementRunner.run`` is ``ResumeWindowPlacementRunner.run``
(e3_native35_l2_placement, unchanged) byte for byte plus one inserted block
directly after the ``placement-window-incomplete`` refusal: the owned decode
cohort is stopped (client close, engine abort) and a surviving reader refuses
``placement-readers-alive``, before the prefill open-wait.  With the driver's
``PLACEMENT_WINDOW_TOKENS`` = 20 the window is two parity cohorts x 20 tokens =
the declared 40 replays; the driver checks the observed per-rank replays
against [``PLACEMENT_REPLAYS_LOW``, ``PLACEMENT_REPLAYS_HIGH``].  Workload,
budgets, every later check, the engine measure/observer intervals and the row4
grader are unchanged.  The later stop at the observer boundary is the parent's
and is a no-op on the already-stopped cohort (test_native35_p_c_window.py proves
the source relation and the timing).

``main`` is ``e3_q18a_qualification.main`` with the same argv and record; for the
duration of the call the driver module's ``NativePlacementRunner`` name is bound
to the P-c runner and restored afterwards.
"""
import copy
import sys

import e3_decode_correctness as correctness
import e3_native_qualification_driver as driver_module
from e3_native35_l2_placement import ResumeWindowPlacementRunner
from e3_native_qualification_driver import (
    PhaseUnavailable, QualificationError,
    QualificationRefused, PLACEMENT_PHASE, PLACEMENT_STREAMS,
    PLACEMENT_WINDOW_TOKENS, RANKS, _replay_delta, _text, canonical_sha,
    exception_details, freeze_prefill_plan)


class StoppedWindowPlacementRunner(ResumeWindowPlacementRunner):
    """L2 runner; the decode cohort stops when the client window completes."""

    def run(self, driver, *, window_id, run_id, engine_epoch, model):
        if driver.control is None:
            raise PhaseUnavailable('missing-control-transport')
        if not _text(self.url):
            raise PhaseUnavailable('missing-placement-loopback-url')
        full_bank, split_bank = driver.route_banks()
        # Missing queue plan -> named INCONCLUSIVE before any engine action.
        prefill = freeze_prefill_plan(self.prefill_plan, run_id=run_id,
                                      model=model)
        record: dict = {'calls': [], 'streams': [],
                        'readers_alive_after_join': [], 'open': None,
                        'window': None, 'prefill_request': None}
        states = []
        prefill_states = []
        prefill_api = {'plan_sha256': prefill['plan_sha256'],
                       'manifest_sha256': prefill['manifest_sha256'],
                       'request_sha256': list(prefill['request_sha256']),
                       'streams': len(prefill['bodies']), 'open': None,
                       'streams_detail': [], 'complete': False}
        window_generation = None
        transition_attempted = False
        returned_full = False
        cleanup = None
        try:
            # 1. Stock full-route preparation.  No admission/native field is
            # ever part of this command (see driver.command_keys()).
            prep = driver._session('row4', 'prepare', target_route='full')
            if prep.get('route') != 'full':
                raise QualificationRefused('prepare-route-is-not-full')
            self._require_acks(prep, bank=full_bank, label='prepare-full',
                               epoch=engine_epoch)
            # 2. Frontend plan cleared: the shipping path must not admit a
            # native prefill or keep a teacher plan for the placement window.
            frontend_before = driver._frontend('row4', False, None)
            if frontend_before.get('enabled') is not False:
                raise QualificationRefused('shipping-path-instrument-enabled')
            # 3. Install the observer only on an empty, drained, full worker.
            configure = driver._worker('row4', 'teacher_configure', {
                'action': 'teacher_configure', 'enabled': True,
                'snapshot_max_bytes': self.snapshot_max_bytes})
            configure_rows = self._qualifications(
                configure, action='teacher_configure', epoch=engine_epoch)
            for rank in RANKS:
                if configure_rows[rank].get('route') != 'full':
                    raise QualificationRefused('teacher-configure-route-differs')
            # 4. Submit the stock cohort while generation is paused, then
            # resume before waiting for its first tokens.  Qualification's
            # worker utility re-paused the engine after FULL preparation; an
            # omitted resume accepts all eight HTTP streams but schedules none.
            cohort = correctness.build_cohort(self.workload)
            if (type(cohort) is not list or len(cohort) != PLACEMENT_STREAMS):
                raise QualificationRefused('placement-cohort-shape-differs')
            salt = correctness.build_cache_salt(run_id, 9000)
            bodies = [correctness.request_body(model, row['prompt'],
                                               self.workload, salt)
                      for row in cohort]
            states = correctness._start_streams(
                self.url, model, bodies, cohort, self.workload)
            resumed = driver._session('row4', 'resume', require_epoch=False)
            if not (resumed.get('resumed') is True
                    or resumed.get('already_running') is True):
                raise QualificationRefused('placement-stock-route-not-resumed')
            opened = correctness._wait_open(states, self.workload)
            record['open'] = opened
            if opened.get('complete') is not True:
                raise QualificationRefused('placement-open-incomplete')
            # 5. Real all-rank pause + split transition, arming the declared
            # native placement phase only for this window.  The session enters
            # the split route first and then arms the Core-owned intent; the
            # stock preparation above carried no phase and never armed it.
            transition_attempted = True
            split = driver._session('row4', 'pause_enter_split',
                                    phase=PLACEMENT_PHASE)
            if split.get('route') != 'split':
                raise QualificationRefused('split-transition-not-applied')
            admission = self._placement_admission(split, engine_epoch)
            window_generation = split.get('route_generation_after')
            if type(window_generation) is not int or window_generation < 0:
                raise QualificationRefused('split-route-generation-missing')
            self._require_acks(split, bank=split_bank, label='enter-split',
                               epoch=engine_epoch)
            # 6. Engine-side measurement marker, then the worker observer.
            begin = driver._session('row4', 'overlap_begin',
                                    route='split',
                                    decode_streams=len(cohort))
            if begin.get('route') != 'split':
                raise QualificationRefused('overlap-begin-route-differs')
            start_ns = begin.get('measure_start_ns')
            proof = begin.get('clock_proof') or {}
            if proof.get('proved') is not True:
                raise QualificationRefused('overlap-begin-clock-unproved')
            observer_begin = driver._worker('row4', 'observer_begin', {
                'action': 'observer_begin', 'run_id': run_id,
                'window_id': window_id})
            begin_rows = self._qualifications(
                observer_begin, action='observer_begin', epoch=engine_epoch)
            for rank in RANKS:
                if begin_rows[rank].get('route') != 'split':
                    raise QualificationRefused('observer-begin-route-differs')
            # 7. Submit the actual native prefill request *inside* this window,
            # then resume.  Its prompts/config come only from the frozen plan;
            # the engine offer ledger must attribute it separately.
            prefill_states = correctness._start_streams(
                self.url, model, prefill['bodies'], prefill['cohort'],
                prefill['workload'])
            record['prefill_request'] = prefill_api
            resume = driver._session('row4', 'resume', require_epoch=False)
            if not (resume.get('resumed') is True
                    or resume.get('already_running') is True):
                raise QualificationRefused('resume-not-accepted')
            starts = {state.index: state.count() for state in states}
            window = correctness._wait_window(states, starts, self.workload)
            record['window'] = window
            if window.get('complete') is not True:
                raise QualificationRefused('placement-window-incomplete')
            # P-c: stop the owned decode cohort at the window boundary, before
            # the native prefill's first token, so the observer window holds the
            # declared decode replays (2 parity cohorts x PLACEMENT_WINDOW_TOKENS).
            correctness._stop_streams(record, states, self.workload)
            if record.get('readers_alive_after_join'):
                raise QualificationRefused('placement-readers-alive')
            prefill_api['open'] = correctness._wait_open(
                prefill_states, prefill['workload'])
            if prefill_api['open'].get('complete') is not True:
                raise QualificationRefused('native-prefill-not-opened')
            prefill_api['streams_detail'] = self._prefill_progress(
                prefill_states, prefill['workload'])
            prefill_api['advanced'] = any(
                row['first_advance_at_ns'] is not None
                for row in prefill_api['streams_detail'])
            if prefill_api['advanced'] is not True:
                raise QualificationRefused('native-prefill-not-started')
            prefill_api['token_count_proved'] = any(
                row['prompt_tokens_proven']
                for row in prefill_api['streams_detail'])
            if prefill_api['token_count_proved'] is not True:
                # No relabelling: an unproved prompt length is not a native
                # prefill request for the declared plan.
                raise QualificationRefused('native-prefill-token-count-unproved')
            end = driver._session('row4', 'overlap_end')
            if end.get('route') not in ('split', 'full'):
                raise QualificationRefused('overlap-end-route-differs')
            end_ns = end.get('measure_end_ns')
            if not (type(start_ns) is int and type(end_ns) is int
                    and end_ns > start_ns):
                raise QualificationRefused('placement-interval-unproved')
            # 8. Stop the owned cohort at the window boundary, then close the
            # observer on a quiet split worker.
            correctness._stop_streams(record, states, self.workload)
            correctness._stop_streams(prefill_api, prefill_states,
                                      prefill['workload'])
            prefill_api['complete'] = (
                prefill_api.get('readers_alive_after_join') == []
                and prefill_api.get('advanced') is True
                and prefill_api.get('token_count_proved') is True)
            if record.get('readers_alive_after_join'):
                raise QualificationRefused('placement-readers-alive')
            if prefill_api.get('complete') is not True:
                raise QualificationRefused('native-prefill-readers-alive')
            observer_end = driver._worker('row4', 'observer_end', {
                'action': 'observer_end', 'run_id': run_id,
                'window_id': window_id})
            end_rows = self._qualifications(
                observer_end, action='observer_end', epoch=engine_epoch)
            raw_rows, engine_rows, raw_shas = {}, {}, {}
            bounds, census = {}, {}
            graph_receipts, accounting_receipts, native_packets = {}, {}, {}
            for rank in RANKS:
                row = end_rows[rank]
                current = row.get('current_route_generation', row.get('route_generation'))
                route = row.get('current_route', row.get('route'))
                if not ((route == 'split' and current == window_generation)
                        or (route == 'full' and current == window_generation + 1
                            and row.get('state') == 'closed'
                            and row.get('observer_stopped') is True)):
                    raise QualificationRefused('observer-end-route-differs')
                if row.get('route_generation') != window_generation:
                    # A window that spans a route transition cannot be
                    # flattened into one split interval: retain the receipts and
                    # refuse instead of relabelling generations.
                    raise QualificationRefused(
                        'placement-window-spans-route-generation',
                        evidence={'window_generation': window_generation,
                                  'rank': rank,
                                  'observed_generation':
                                      row.get('route_generation')})
                if end.get('route') == 'full' and not (
                        row.get('current_route') == 'full'
                        and row.get('current_route_generation') == window_generation + 1):
                    raise QualificationRefused('full-return-worker-window-contradiction')
                receipt = row.get('receipt')
                accounting = row.get('engine_accounting')
                if type(receipt) is not dict:
                    raise PhaseUnavailable('raw-graph-receipt-unavailable:%d' % rank)
                if type(accounting) is not dict:
                    raise PhaseUnavailable('engine-accounting-unavailable:%d' % rank)
                raw = copy.deepcopy(receipt)
                raw_rows[rank] = raw
                engine_rows[rank] = copy.deepcopy(accounting)
                raw_shas[rank] = canonical_sha(raw)
                native_packets[rank] = copy.deepcopy(row)
                graph_receipts[rank] = copy.deepcopy(receipt)
                accounting_receipts[rank] = copy.deepcopy(accounting)
                bounds[rank] = {
                    'drained_streams': row.get('drained_streams'),
                    'route_generation': row.get('route_generation'),
                    'current_route_generation': row.get('current_route_generation'),
                    'storage_epoch': row.get('storage_epoch'),
                    'worker_incarnation': row.get('worker_incarnation'),
                    'replay_census': copy.deepcopy(row.get('replay_census')),
                    'window_bounds': copy.deepcopy(receipt.get('bounds')),
                }
                census[rank] = copy.deepcopy(row.get('replay_census'))
            # 9. Read the teacher trace state separately.  It is never used as
            # placement evidence and any retained row is reported, not hidden.
            trace = driver._worker('row4', 'teacher_evidence', {
                'action': 'teacher_evidence'})
            trace_rows = self._qualifications(
                trace, action='teacher_evidence', epoch=engine_epoch)
            teacher_rows = sum(len(row.get('positions') or [])
                               for row in trace_rows.values())
            teacher_trace_receipt = {str(rank): trace_rows[rank] for rank in RANKS}
            # 10. Return to full through the engine's own typed ACKs.
            left, cleanup = self._leave_full(driver, engine_epoch)
            returned_full = True
            # 11. Clear the observer/teacher instrumentation on the empty
            # worker and prove the shipping frontend is disabled.
            teacher_clear = driver._worker('row4', 'teacher_configure', {
                'action': 'teacher_configure', 'enabled': False,
                'snapshot_max_bytes': self.snapshot_max_bytes})
            clear_rows = self._qualifications(
                teacher_clear, action='teacher_configure', epoch=engine_epoch)
            # Clearing the worker observer is another qualification intent and
            # therefore pauses generation.  Restore ordinary FULL sampling
            # before handing the engine to the next same-queue gate.
            resumed = driver._session('row4', 'resume', require_epoch=False)
            if not (resumed.get('resumed') is True
                    or resumed.get('already_running') is True):
                raise QualificationRefused('placement-full-return-not-resumed')
            frontend_after = driver._frontend('row4', False, None)
            disabled = (frontend_after.get('enabled') is False
                        and frontend_after.get('admitted') is not True
                        and frontend_after.get('pending') in (0, None)
                        and all(row.get('route') == 'full'
                                for row in clear_rows.values()))
            window = {
                'kind': 'short', 'run_id': run_id,
                'window_id': window_id, 'engine_epoch': engine_epoch,
                'start_ns': start_ns, 'end_ns': end_ns,
                'host_boundary_basis': 'engine control observations; per-rank native bounds retained separately',
                'window_tokens': PLACEMENT_WINDOW_TOKENS,
                'replays': self.replicates,
                'observed_replays': {
                    rank: _replay_delta(((native_packets[rank].get('decode_window') or {})
                        .get('binding') or {}).get('router_census')) for rank in RANKS},
                'replay_count_basis': 'actual native binding router delta, not the client token target',
                'teacher_overlap': False,
                'teacher_window_end_ns': (driver._teacher_window or {}).get('ended_ns'),
                'cohort_id': 'placement-' + canonical_sha(
                    {'run_id': run_id, 'window_id': window_id})[:24],
                'decode_progress': end.get('decode_progress'),
                'native_phase': PLACEMENT_PHASE,
                'admission': admission,
                'route_generation': window_generation,
                'native_prefill_request': copy.deepcopy(prefill_api),
            }
            return {
                'window': window,
                'native_prefill_request': copy.deepcopy(prefill_api),
                'window_route_generation': window_generation,
                'raw_placement': raw_rows, 'engine_accounting': engine_rows,
                'native_observer_packets': native_packets,
                'raw_sha256': {str(rank): raw_shas[rank] for rank in RANKS},
                'raw_graph_receipts': graph_receipts,
                'engine_accounting_receipts': accounting_receipts,
                'per_rank_bounds': bounds, 'replay_census': census,
                'teacher_rows_during_placement': teacher_rows,
                'teacher_trace_receipt': teacher_trace_receipt,
                'shipping_path_instrument_disabled': bool(disabled),
                'cleanup': cleanup,
                'stock_prefix': {
                    'prepare_keys': [row['keys'] for row in
                                     driver.command_keys('row4', 'prepare')],
                    'native_admission_intent': False,
                },
                'frontend': {'before': frontend_before, 'after': frontend_after},
                'overlap_end': end, 'leave_full': left,
                'calls': record.get('calls'),
            }
        except BaseException as exc:  # noqa: BLE001 - preserve the first failure
            if transition_attempted and not returned_full:
                cleanup = self._attempt_cleanup(driver, engine_epoch)
            partial = {'kind': 'placement-partial',
                       'run_id': run_id, 'window_id': window_id,
                       'engine_epoch': engine_epoch,
                       'collector_error': exception_details(exc, getattr(driver, 'control', None)),
                       'open': record.get('open'),
                       'client_window': record.get('window'),
                       'reads_alive_after_join':
                           record.get('readers_alive_after_join'),
                       'prefill_request': copy.deepcopy(prefill_api),
                       'calls': [call for call in record.get('calls') or []]}
            if isinstance(exc, QualificationRefused):
                exc.cleanup = cleanup
                if exc.evidence is None:
                    exc.evidence = partial
                raise
            if isinstance(exc, QualificationError):
                raise QualificationRefused(
                    type(exc).__name__, cleanup=cleanup,
                    evidence=partial) from None
            raise QualificationRefused(
                'placement-exception:' + type(exc).__name__,
                cleanup=cleanup, evidence=partial) from None
        finally:
            if states:
                correctness._stop_streams(record, states, self.workload)
            if prefill_states:
                correctness._stop_streams(prefill_api, prefill_states,
                                          prefill['workload'])


def main(argv=None):
    import e3_q18a_qualification as qualification
    original = driver_module.NativePlacementRunner
    driver_module.NativePlacementRunner = StoppedWindowPlacementRunner
    try:
        return qualification.main(argv)
    finally:
        driver_module.NativePlacementRunner = original


if __name__ == '__main__':
    sys.exit(main())
