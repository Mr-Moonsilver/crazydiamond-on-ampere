"""One native A1/A2/B/C teacher sequence over actual session/worker actions.

The existing instrument owns requests, sampler-row/SSE joins and frozen rule4.
This module supplies native lifecycle receipts, never probe acknowledgements.
"""
import copy
import time

import e3_decode_correctness as correctness
import e3_teacher_decode_client as teacher


class NativeTeacherSequence:
    def __init__(self, driver, instrument, workload):
        from e3_native_qualification_driver import NativePlacementRunner
        self.driver, self.instrument, self.workload = driver, instrument, workload
        self.lifecycle = NativePlacementRunner

    def _worker(self, action, **fields):
        reply = self.driver._worker('rule4_controls', action, {'action': action, **fields})
        rows = self.lifecycle._qualifications(reply, action=action,
                                              epoch=self.driver.engine_epoch)
        return reply, rows

    def _session(self, action, **fields):
        return self.driver._session('rule4_controls', action,
                                    require_epoch=action != 'resume', **fields)

    def _resume(self):
        reply = self._session('resume')
        if not (reply.get('resumed') is True or reply.get('already_running') is True):
            raise ValueError('native-teacher-resume-unproved')
        return reply

    def _require_acks(self, *args, **kwargs):
        return self.lifecycle._require_acks(*args, **kwargs)

    def _leave_full(self, driver, epoch):
        return self.lifecycle._leave_full(self, driver, epoch)

    def run(self):
        driver, instrument, workload = self.driver, self.instrument, self.workload
        cohort = instrument.cohort(driver.url, driver.model)
        full, split = driver.route_banks()
        runs = []
        for repeat, (label, route) in enumerate((
                ('full-A1', 'full'), ('full-A2', 'full'),
                ('split-B', 'split'), ('full-C', 'full'))):
            record = {'label': label, 'route': route, 'complete': False,
                      'contract': False, 'streams': [], 'calls': []}
            states, completed, cleanup = [], False, None
            native = {'schema': 's5-native-teacher-window/1', 'repeat': repeat,
                      'engine_epoch': driver.engine_epoch, 'lifecycle': {}}
            record['native_teacher_window'] = native
            life = native['lifecycle']
            try:
                prep = self._session('prepare', target_route='full')
                self.lifecycle._require_acks(prep, bank=full, label='teacher-prepare',
                                              epoch=driver.engine_epoch)
                if prep.get('route') != 'full':
                    raise ValueError('native-teacher-prepare-route')
                life['prepare'] = prep
                configure, _ = self._worker('teacher_configure', enabled=True,
                    snapshot_max_bytes=instrument.snapshot_max_bytes)
                life['configure'] = configure
                salt = correctness.build_cache_salt(driver.run_id, repeat)
                bodies = [correctness.request_body(driver.model, row['prompt'],
                                                  workload, salt) for row in cohort]
                instrument.requests(bodies, cohort, repeat)
                native['frontend_configure'] = driver._frontend(
                    'rule4_controls', True, copy.deepcopy(instrument.plan))
                states = correctness._start_streams(driver.url, driver.model,
                    bodies, cohort, workload, instrument=instrument)
                life['initial_resume'] = self._resume()
                life['open'] = correctness._wait_open(states, workload)
                if life['open'].get('complete') is not True:
                    raise ValueError('native-teacher-cohort-unopened')
                # The named qualification utility first awaits keep-pause and
                # queries every worker drain. No legacy pause_full is sent.
                initial, initial_rows = self._worker('teacher_evidence')
                life['entry_pause'] = initial
                if any(row.get('route') != 'full' for row in initial_rows.values()):
                    raise ValueError('native-teacher-entry-route')
                if route == 'split':
                    entered = self._session('pause_enter_split')
                    self.lifecycle._require_acks(entered, bank=split,
                        label='teacher-enter-split', epoch=driver.engine_epoch)
                    if entered.get('route') != 'split':
                        raise ValueError('native-teacher-split-not-applied')
                    life['enter_split'] = entered
                life['window_resume'] = self._resume()
                starts = {state.index: state.count() for state in states}
                window = correctness._wait_window(states, starts, workload)
                life['client_window'] = window
                if window.get('complete') is not True:
                    raise ValueError('native-teacher-window-incomplete')
                reply, rows = self._worker('teacher_evidence')
                if any(row.get('route') != route for row in rows.values()):
                    raise ValueError('native-teacher-window-route-changed')
                native['worker_reply'] = reply
                native['frontend_clear'] = driver._frontend('rule4_controls', False, None)
                # The pure instrument validates admission and all physical row
                # identities before performing any numerical comparison.
                teacher._native_teacher_binding(record, instrument.plan_shas)
                left = self._session('leave_full')
                self.lifecycle._require_acks(left, bank=full, label='teacher-return',
                                              epoch=driver.engine_epoch)
                if left.get('route') != 'full':
                    raise ValueError('native-teacher-full-return-unproved')
                life['leave_full'] = left
                life['tail_resume'] = self._resume()
                life['finish_timeouts'] = correctness._wait_finish(states, workload)
                if life['finish_timeouts']:
                    raise ValueError('native-teacher-tail-incomplete')
                alive = correctness._stop_streams(record, states, workload)
                record['streams'] = correctness._trajectory_rows(states, window, workload)
                if (alive or len(record['streams']) != 8
                        or any(row.get('complete') is not True for row in record['streams'])):
                    raise ValueError('native-teacher-stream-close-unproved')
                clear, clear_rows = self._worker('teacher_configure', enabled=False,
                    snapshot_max_bytes=instrument.snapshot_max_bytes)
                life['clear'] = clear
                if any(row.get('route') != 'full' or row.get('enabled') is not False
                       for row in clear_rows.values()):
                    raise ValueError('native-teacher-clear-unproved')
                record['complete'] = completed = True
            except Exception as exc:
                # Preserve the first refusal and every actual receipt. No
                # subsequent teacher run, retry or alternate sample is chosen.
                from e3_native_qualification_driver import exception_details, QualificationRefused
                record['error'] = exception_details(exc, getattr(driver, 'control', None))
                cleanup = self.lifecycle._attempt_cleanup(self, driver, driver.engine_epoch)
                raise QualificationRefused('native-teacher:' + record['error']['error_message'],
                    evidence={'label': label, 'native_teacher_window': native},
                    cleanup=cleanup) from None
            finally:
                if states:
                    correctness._stop_streams(record, states, workload)
                driver.store.write_once('teacher-native-' + label + '.json', {
                    'label': label, 'route': route, 'complete': completed,
                    'native_teacher_window': native, 'cleanup': cleanup,
                    'error': record.get('error'), 'recorded_ns': time.monotonic_ns()})
            runs.append(record)
        return {'teacher_forced': instrument.finish(runs), 'native_runs': runs,
                'native_frontend_calls': [dict(
                    repeat=run['native_teacher_window']['repeat'],
                    receipt=run['native_teacher_window']['frontend_configure'])
                    for run in runs]}
