"""Owned q17b replay observation at the real worker/controller boundaries.

CPU-only import. No arbitrary RPC paths or handles. One short window on the
first split cohort; stop/close completes before an unprofiled cohort can run.
"""
import os
import secrets
from collections.abc import Mapping

from replay_observer import (DrainedProof, ObserverSession, ReplayBinding,
                             ReplayObserverError, helper_source_sha256)

ENABLE_ENV = 'FLASH_S5_REPLAY_OBSERVER'
LIBRARY_ENV = 'FLASH_S5_REPLAY_LIB'
LIBRARY_SHA_ENV = 'FLASH_S5_REPLAY_LIB_SHA256'
CUPTI_SHA256 = 'e2f9ed861fe27c492b8bb52b5e3220ef5120f3edcda36312e96b7fd8a186be3e'


def plain(value):
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise ReplayObserverError('malformed_native_report')


class WorkerObserver:
    def __init__(self, session, rank):
        if type(rank) is not int or rank not in (0, 1, 2, 3):
            raise ReplayObserverError('bad_proof')
        self.session = session
        self.rank = rank
        self.incarnation = secrets.token_hex(16)
        self.engine_epoch = None
        self.generation = None
        self.receipt = None
        self.started_before_split = False
        self.closed = False

    def start(self):
        self.session.start()
        self.started_before_split = True

    def status(self):
        return {'enabled': True, 'state': self.session.state,
                'started_before_split': self.started_before_split,
                'closed': self.closed, 'worker_incarnation': self.incarnation,
                'reserved_native_host_bytes': 0 if self.closed else self.session._native_reserved,
                'peak_native_host_bytes': self.session._native_reserved,
                'binding': plain(self.session.binding.identity()),
                'window_completed': self.receipt is not None}

    def _check(self, probe, epoch):
        if type(epoch) is not int or epoch < 1:
            raise ReplayObserverError('bad_proof')
        if self.engine_epoch is not None and self.engine_epoch != epoch:
            raise ReplayObserverError('bad_proof')
        probe._require_not_down()
        if (probe.in_flight or getattr(probe.runner, 'execute_model_state', None) is not None
                or not probe.controller.admitting):
            raise ReplayObserverError('bad_state')
        self.engine_epoch = epoch

    def begin(self, probe, epoch):
        self._check(probe, epoch)
        if self.closed or probe.route != 'split' or self.session.state != 'started':
            raise ReplayObserverError('bad_state')
        # RPC arrives only after frontend keep-pause and transition ACKs, but
        # this own worker proof additionally queries all relevant stream events.
        drained = probe.controller.drain()
        if type(drained) is not int or drained <= 0:
            raise ReplayObserverError('bad_proof')
        self.session.bind_bank(probe)
        self.session.begin_window(route_generation=probe.route_generation)
        self.generation = probe.route_generation
        return {'ack': 'flash_s5_observer_begin', 'rank': self.rank,
                'engine_epoch': epoch, 'route_generation': self.generation,
                'storage_epoch': probe.storage_epoch, 'drained_streams': drained,
                'state': 'window', 'worker_incarnation': self.incarnation}

    def end(self, probe, epoch):
        self._check(probe, epoch)
        if (self.closed or probe.route != 'split' or self.session.state != 'window'
                or self.generation != probe.route_generation):
            raise ReplayObserverError('bad_state')
        drained = probe.controller.drain()
        if type(drained) is not int or drained <= 0:
            raise ReplayObserverError('bad_proof')
        proof = DrainedProof(self.rank, self.incarnation, 'engine:' + str(epoch),
                             probe.route_generation, probe.storage_epoch,
                             drained, True)
        receipt = plain(self.session.end_window(proof))
        self.receipt = receipt
        # Stops only this owned subscriber. Stop failure propagates/quarantines;
        # no performance interval may proceed with unproven instrumentation off.
        self.session.stop()
        self.session.close()
        self.closed = True
        return {'ack': 'flash_s5_observer_end', 'rank': self.rank,
                'engine_epoch': epoch, 'route_generation': self.generation,
                'storage_epoch': probe.storage_epoch, 'drained_streams': drained,
                'state': 'closed', 'observer_stopped': True,
                'worker_incarnation': self.incarnation, 'receipt': receipt}

    def shutdown(self, probe=None):
        if self.closed:
            return
        if self.session.state == 'window':
            if probe is None or self.engine_epoch is None:
                raise ReplayObserverError('window_active')
            self.end(probe, self.engine_epoch)
        else:
            self.session.close()
            self.closed = True


def start_worker_observer(worker, runner, env=None, *, binding_factory=ReplayBinding,
                          session_factory=ObserverSession):
    values = os.environ if env is None else env
    enabled = values.get(ENABLE_ENV, '0')
    if enabled not in ('0', '1'):
        raise ReplayObserverError('bad_state')
    if enabled == '0':
        return None
    if (values.get('FLASH_S5_DECODE_ONLY_PROBE') != '1'
            or values.get('FLASH_S5_OVERLAP', '0') != '0'
            or getattr(runner, '_flash_s5_split', None) is not None
            or getattr(runner, '_flash_s5_replay_observer', None) is not None):
        raise ReplayObserverError('bad_state')
    path, sha = values.get(LIBRARY_ENV), values.get(LIBRARY_SHA_ENV)
    if not path or not sha:
        raise ReplayObserverError('bad_binding')
    binding = binding_factory(path, sha, helper_source_sha256(), CUPTI_SHA256)
    owner = WorkerObserver(session_factory(binding), getattr(worker, 'rank', None))
    # Retain ownership before start: partial native setup failure must not lose
    # the subscriber/buffer owner or permit a second start.
    runner._flash_s5_replay_observer = owner
    owner.start()
    return owner
