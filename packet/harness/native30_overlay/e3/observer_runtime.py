"""Owned q17b replay observation at the real worker/controller boundaries.

CPU-only import. No arbitrary RPC paths or handles. One short window on the
first split cohort; stop/close completes before an unprofiled cohort can run.
"""
import os
import secrets
from graph_router import GraphRoutingError
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
    def __init__(self, session, rank, *, engine_offers=False):
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
        self.full_launch=None
        self.full_launch_active=False
        self.launch_diagnostic=None
        self.engine_offers_enabled = engine_offers
        self.bridge = self.ledger = None
        self.engine_accounting = self.composed_placement = None
        self.window_identity = None
        self.census_start = None

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
                'window_completed': self.receipt is not None,
                'engine_offers_enabled': self.engine_offers_enabled}

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

    def launch_begin(self,probe,epoch):
        self._check(probe,epoch)
        if self.closed or probe.route!='full' or self.full_launch is not None or self.full_launch_active:
            raise ReplayObserverError('bad_state')
        drained=probe.controller.drain()
        if type(drained) is not int or drained<=0:raise ReplayObserverError('bad_proof')
        from launch_diagnostic import inventory
        try:self.full_launch_inventory=inventory(self.session,probe,True)
        except (AttributeError,KeyError,ReplayObserverError,GraphRoutingError):
            self.full_launch_inventory=None
            return {'rank':self.rank,'ack':'flash_s5_launch_begin','engine_epoch':epoch,'drained_streams':drained,'diagnostic':'INCONCLUSIVE'}
        rc=self.session.binding._lib.s5ro_launch_begin(self.session._token)
        if rc:raise ReplayObserverError('launch_sample_begin_failed')
        self.full_launch_active=True
        return {'rank':self.rank,'ack':'flash_s5_launch_begin','engine_epoch':epoch,'drained_streams':drained}

    def launch_end(self,probe,epoch):
        self._check(probe,epoch)
        if probe.route!='full':raise ReplayObserverError('bad_state')
        if self.full_launch_inventory is None:
            return {'rank':self.rank,'ack':'flash_s5_launch_end','engine_epoch':epoch,'drained_streams':probe.controller.drain(),'diagnostic':'INCONCLUSIVE'}
        if not self.full_launch_active:raise ReplayObserverError('bad_state')
        drained=probe.controller.drain()
        if type(drained) is not int or drained<=0:raise ReplayObserverError('bad_proof')
        report=self.session._report_call(self.session.binding._lib.s5ro_launch_report,(self.session._token,1))
        self.full_launch=(report,self.full_launch_inventory)
        self.full_launch_active=False
        return {'rank':self.rank,'ack':'flash_s5_launch_end','engine_epoch':epoch,'drained_streams':drained}

    def begin(self, probe, epoch, *, run_id=None, window_id=None):
        self._check(probe, epoch)
        if self.closed or probe.route != 'split' or self.session.state != 'started':
            raise ReplayObserverError('bad_state')
        # RPC arrives only after frontend keep-pause and transition ACKs, but
        # this own worker proof additionally queries all relevant stream events.
        drained = probe.controller.drain()
        if type(drained) is not int or drained <= 0:
            raise ReplayObserverError('bad_proof')
        if self.engine_offers_enabled:
            from activity_accounting import CorrelationBridge
            self.bridge = CorrelationBridge(self.session, epoch=epoch,
                generation=probe.route_generation, rank=self.rank)
        self.session.bind_bank(probe)
        replay_census = self.session.replay_census()
        target = None
        if self.engine_offers_enabled:
            from engine_offers import graph_descriptor, descriptor_key
            prior = getattr(probe, '_last_decode_descriptor', None)
            if prior is None:
                raise ReplayObserverError('engine_target_unproved')
            key = descriptor_key(graph_descriptor(None, prior))
            matches = [d for d in self.session._descriptor_objects
                       if descriptor_key(graph_descriptor(None, d)) == key]
            if len(matches) != 1:
                raise ReplayObserverError('engine_target_unproved')
            target = matches[0]
        # Native context/stream queries are observation setup, before tracing.
        roles = self._stream_roles(probe) if self.engine_offers_enabled else None
        self.session.begin_window(route_generation=probe.route_generation, descriptor=target)
        if self.engine_offers_enabled:
            self._begin_ledger(probe, epoch, run_id, window_id, target, replay_census, roles)
        self.generation = probe.route_generation
        return {'ack': 'flash_s5_observer_begin', 'rank': self.rank,
                'engine_epoch': epoch, 'route_generation': self.generation,
                'storage_epoch': probe.storage_epoch, 'drained_streams': drained,
                'state': 'window', 'worker_incarnation': self.incarnation,
                'replay_census': replay_census}

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
        replay_census = self.session.replay_census()
        if self.ledger is not None:
            self.ledger.deactivate()
        receipt = plain(self.session.end_window(proof))
        if self.ledger is not None:
            from engine_offers import compose_placement
            self.ledger.finish_window(self._census_delta(replay_census))
            self.engine_accounting = self.ledger.seal()
            self.composed_placement = compose_placement(receipt, self.engine_accounting)
        from launch_diagnostic import inventory,compare
        try:
            launch_report=self.session._report_call(self.session.binding._lib.s5ro_launch_report,(self.session._token,0))
            self.launch_diagnostic=compare(self.full_launch,(launch_report,inventory(self.session,probe,False)))
        except (AttributeError,KeyError,ReplayObserverError,GraphRoutingError):
            self.launch_diagnostic={'schema':1,'verdict':'INCONCLUSIVE','gates_w1':False,'missing':['launch configuration report unavailable']}
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
                'worker_incarnation': self.incarnation, 'receipt': receipt,
                'replay_census': replay_census,'launch_config':self.launch_diagnostic,
                **({'engine_accounting': self.engine_accounting,
                    'composed_placement': self.composed_placement}
                   if self.engine_offers_enabled else {})}

    def _stream_roles(self, probe):
        runner = probe.runner
        query = self.session.binding.stream_identity
        main = query(runner.main_stream)
        roles = {'main': main}
        handler = getattr(runner, 'pp_handler', None)
        for role, stream in (
                ('output_copy', getattr(runner, 'output_copy_stream', None)),
                ('pp', getattr(handler, 'broadcast_stream', None))):
            if stream is not None:
                roles[role] = dict(query(stream))
                roles[role]['non_green_control'] = roles[role]['context_id'] != main['context_id']
        # Both driver ownership and CUPTI's context namespace must be bound.
        owned = query(probe.split.decode_stream)
        if main != owned:
            raise ReplayObserverError('engine_main_stream_not_owned')
        return roles

    def _begin_ledger(self, probe, epoch, run_id, window_id, target, census, roles):
        from engine_offers import (EngineOfferLedger, bind_window, graph_descriptor,
            descriptor_key, bank_identity, ENABLE_ENV)
        from replay_observer import NATIVE_SCHEMA_STATIC, NATIVE_SCHEMA_WINDOW, CUPTI_API
        runner = probe.runner
        main = roles['main']
        descs = self.session._descriptor_objects
        labels = {'graph:' + str(i): descriptor_key(graph_descriptor(None, d))
                  for i, d in enumerate(descs)}
        required = {int(row['slot']): dict(row['required'])
                    for row in self.session._static_report['graph_slots']}
        if set(required) != set(range(len(descs))):
            raise ReplayObserverError('engine_census_unproved')
        node_census = {labels['graph:' + str(i)]: required[i] for i in range(len(descs))}
        target_label = 'graph:' + str(next(i for i, d in enumerate(descs) if d is target))
        start = self.bridge.window()
        binding = bind_window(run_id=run_id, window_id=window_id, worker='rank'+str(self.rank),
            rank=self.rank, worker_incarnation=self.incarnation, engine_incarnation='engine:'+str(epoch),
            engine_epoch=epoch, route_generation=probe.route_generation, storage_epoch=probe.storage_epoch,
            green_context_id=main['context_id'], native_start_ns=start['start_ns'], native_end_ns=None,
            observer_identity=self.session.binding.identity(), helper_identity={
                'native_static_schema': NATIVE_SCHEMA_STATIC, 'native_window_schema': NATIVE_SCHEMA_WINDOW,
                'compiled_api': CUPTI_API, 'runtime_api': CUPTI_API}, declared_target=target_label,
            descriptor_labels=labels, router_census=None, bank_identity_sha256=bank_identity(self.session._bank_meta),
            stream_roles=roles, pp_stage=self._pp_stage(runner))
        if type(census) is not dict:
            raise ReplayObserverError('engine_census_unproved')
        self.census_start = census
        self.window_identity = (run_id, window_id)
        self.ledger = EngineOfferLedger(self.bridge, worker='rank'+str(self.rank), rank=self.rank,
            epoch=epoch, generation=probe.route_generation, route_generation=probe.route_generation,
            green_context_id=main['context_id'], identities=roles, pp_stage=self._pp_stage(runner),
            census=node_census, env={ENABLE_ENV:'1'}, binding=binding)
        self.ledger.activate(runner)
        from engine_offers import install_census
        install_census(runner, node_census)

    @staticmethod
    def _pp_stage(runner):
        parallel = getattr(runner, 'parallel_config', None)
        size = getattr(parallel, 'pipeline_parallel_size', None)
        first, last = getattr(runner, 'is_first_pp_rank', None), getattr(runner, 'is_last_pp_rank', None)
        if type(size) is not int or size < 1 or type(first) is not bool or type(last) is not bool:
            raise ReplayObserverError('engine_pp_stage_unproved')
        if first and last and size == 1:
            return 0
        if first and not last and size > 1:
            return 0
        if last and not first and size > 1:
            return size - 1
        raise ReplayObserverError('engine_pp_stage_unproved')

    def _census_delta(self, end):
        start = self.census_start
        if (type(end) is not dict or type(start) is not dict
                or set(end['graphs']) != set(start['graphs'])):
            raise ReplayObserverError('engine_census_unproved')
        graphs = {key:end['graphs'][key]-start['graphs'][key] for key in start['graphs']}
        total = end['total']-start['total']
        if any(type(n) is not int or n < 0 for n in graphs.values()) or sum(graphs.values()) != total:
            raise ReplayObserverError('engine_census_unproved')
        return {'total':total, 'graphs':{k:v for k,v in graphs.items() if v}}

    def shutdown(self, probe=None):
        if self.closed:
            return
        if self.ledger is not None:
            self.ledger.deactivate()
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
    probe_mode = values.get('FLASH_S5_DECODE_ONLY_PROBE', '0')
    overlap_mode = values.get('FLASH_S5_OVERLAP', '0')
    if (getattr(runner, '_flash_s5_split', None) is not None
            or getattr(runner, '_flash_s5_replay_observer', None) is not None):
        raise ReplayObserverError('bad_state')
    if (probe_mode, overlap_mode) == ('0', '1'):
        # Native observation uses the same owned subscriber, only after the
        # separate native readiness check. This still hard-refuses today;
        # an environment flag is not an authorization to allocate or subscribe.
        from s5_startup import require_native_ready
        require_native_ready()
    elif (probe_mode, overlap_mode) != ('1', '0'):
        raise ReplayObserverError('bad_state')
    path, sha = values.get(LIBRARY_ENV), values.get(LIBRARY_SHA_ENV)
    if not path or not sha:
        raise ReplayObserverError('bad_binding')
    binding = binding_factory(path, sha, helper_source_sha256(), CUPTI_SHA256)
    ledger_flag = values.get('FLASH_S5_ENGINE_OFFERS', '0')
    if ledger_flag not in ('0', '1'):
        raise ReplayObserverError('bad_state')
    owner = WorkerObserver(session_factory(binding), getattr(worker, 'rank', None),
                           engine_offers=ledger_flag == '1')
    # Retain ownership before start: partial native setup failure must not lose
    # the subscriber/buffer owner or permit a second start.
    runner._flash_s5_replay_observer = owner
    owner.start()
    return owner
