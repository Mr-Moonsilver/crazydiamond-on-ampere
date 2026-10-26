"""Stateful W1 receipt adapter over the fixed, awaited diagnostic controller.

No synthetic worker ACKs. Counter evidence covers the enclosing worker interval;
client receive timestamps remain a separate, narrower measurement window.

Measurement validity and safe cleanup are deliberately separate:

* :func:`interval` never invents a zero and never quarantines on unavailable
  counters; it returns explicit unavailable evidence. A route/identity
  contradiction between the two paused snapshots is the one case that raises
  (and therefore quarantines).
* ``pause_end`` still requires the completed keep/clear_cache=False pause,
  four typed drain barriers and unchanged route/storage/generation identity.
  A proven lifecycle is returned as ``lifecycle_proven`` even when the
  measured interval is invalid, so the driver may safely restore full.
* The prior-receipt hash is computed over the same bounded metadata core that
  the driver retains, so the chain never refers to discarded material.
"""
import asyncio
import copy
import hashlib
import json
import time

from probe_control import ProbeControlError, RANKS, _label, _positive


# Fields bound by the prior-receipt chain. Both the session and the client
# retain this metadata; the client retains a superset of it.  The projection
# below mirrors the client's retained receipt so every hashed byte is present
# in the artifact and the SHA is recomputable from retained bytes.
RECEIPT_CORE_KEYS = (
    'schema', 'run_id', 'repeat', 'command_seq', 'action', 'cohort_id',
    'status', 'engine_epoch', 'storage_epoch', 'route_before', 'route_after',
    'route_generation_before', 'route_generation_after', 'expected_rank_set',
    'observed_clock_ns', 'clock_basis', 'pause', 'resume', 'ingress',
    'barrier_replies', 'interval', 'decode_eligibility', 'lifecycle_proven', 'observer_sha256',
)

CORE_COUNTER_KEYS = (
    'mode', 'state', 'route', 'route_generation', 'storage_epoch',
    'replays', 'replay_streams_match_owner', 'transition_count',
    'transition_latencies_s', 'interval', 'interval_index',
    'interval_invalidations', 'invalidation_reasons', 'quarantined',
    'in_flight', 'sampling_pending', 'transition_pending', 'warmup_replays',
)
CORE_BATCH_KEYS = (
    'batches', 'decode_batches', 'non_decode_batches', 'empty_batches',
    'full_graph_dispatches', 'non_full_dispatches', 'scheduler_observations',
    'preempted_requests', 'unknown_preemption_observations',
)
CORE_LAST_KEYS = ('sequence', 'route', 'route_generation', 'num_reqs',
                  'num_tokens', 'decode_only', 'full_graph')
CORE_BARRIER_KEYS = ('rank', 'ack', 'mode', 'route', 'route_generation',
                     'storage_epoch', 'engine_epoch', 'drained_streams')
CORE_ELIGIBILITY_RANK_KEYS = (
    'rank', 'batch_reqs', 'scheduled_tokens', 'prefill', 'sequence',
    'num_reqs', 'num_tokens', 'decode_only', 'full_graph', 'route',
    'route_generation',
)


def _project_counter(counters):
    if type(counters) is not dict:
        return None
    out = {key: copy.deepcopy(counters[key]) for key in CORE_COUNTER_KEYS
           if key in counters}
    for key in ('replays_by_bank', 'replays_by_stream', 'replays_by_descriptor'):
        value = counters.get(key)
        if type(value) is dict:
            out[key] = {str(k)[:64]: copy.deepcopy(v)
                        for k, v in list(value.items())[:64]}
    batches = counters.get('batch_observations')
    if type(batches) is dict:
        observed = {key: copy.deepcopy(batches.get(key))
                    for key in CORE_BATCH_KEYS}
        last = batches.get('last')
        observed['last'] = ({key: copy.deepcopy(last.get(key))
                             for key in CORE_LAST_KEYS}
                            if type(last) is dict else None)
        out['batch_observations'] = observed
    return out


def _project_interval(value):
    if type(value) is not dict:
        return None
    out = {key: copy.deepcopy(value[key]) for key in (
        'available', 'valid', 'reasons', 'unavailable', 'observer_basis',
        'invalidation', 'invalidation_complete', 'invalidation_lower_bounds',
        'invalidation_basis') if key in value}
    for key in ('counters_before', 'counters_after'):
        rows = value.get(key)
        if type(rows) is not list:
            continue
        out[key] = [{'rank': copy.deepcopy(row.get('rank')),
                     'decode_steps': copy.deepcopy(row.get('decode_steps')),
                     'counters': _project_counter(row.get('counters'))}
                    for row in rows[:64] if type(row) is dict]
    return out


def _project_eligibility(value):
    if type(value) is not dict:
        return None
    out = {key: copy.deepcopy(value[key]) for key in
           ('cohort_id', 'cohort_cardinality', 'reason') if key in value}
    rows = value.get('ranks')
    if type(rows) is list:
        out['ranks'] = [{key: copy.deepcopy(row[key])
                         for key in CORE_ELIGIBILITY_RANK_KEYS if key in row}
                        for row in rows[:64] if type(row) is dict]
    return out



_TEXT_FIELDS = (('clock_basis',), ('decode_eligibility', 'reason'),
                ('interval', 'observer_basis'), ('ingress', 'basis'),
                ('interval', 'invalidation_basis'))
_TEXT_BASES = {
    'frontend monotonic completion; offset to client clock unknown':
        'frontend_monotonic_client_offset_unknown_v1',
    'paused live request/queued-token counts and last actual per-worker batch':
        'paused_live_counts_actual_batch_v1',
}


def _bind_receipt_text(core, original):
    """Bind narrative bytes without persisting arbitrary text in the core.

    Validate the original wire receipt separately. Hash before truncation, and
    retain this exact projected core so its chain digest is reproducible even
    when a display receipt redacts narrative fields.
    """
    for path in _TEXT_FIELDS:
        source, target = original, core
        for key in path[:-1]:
            source = source.get(key) if type(source) is dict else None
            target = target.get(key) if type(target) is dict else None
        key = path[-1]
        if type(source) is not dict or key not in source or type(target) is not dict:
            continue
        value = source[key]
        if type(value) is str:
            raw = value.encode('utf-8')
            marker = {'kind': 'text', 'sha256': hashlib.sha256(raw).hexdigest(),
                      'utf8_bytes': len(raw), 'nonblank': bool(value.strip()),
                      'known_basis': _TEXT_BASES.get(value)}
        else:
            marker = {'kind': 'null' if value is None else 'invalid_type'}
        target[key] = marker
    return core

def receipt_core(result):
    """Bounded projection identical to the retained client receipt core."""
    if type(result) is not dict:
        return None
    out = {}
    for key in RECEIPT_CORE_KEYS:
        if key not in result:
            continue
        value = result[key]
        if key == 'pause' and type(value) is dict:
            out[key] = {k: copy.deepcopy(value[k]) for k in
                        ('mode', 'clear_cache', 'completed', 'drained') if k in value}
        elif key == 'resume' and type(value) is dict:
            out[key] = {k: copy.deepcopy(value[k]) for k in ('completed',)
                        if k in value}
        elif key == 'ingress' and type(value) is dict:
            out[key] = {k: copy.deepcopy(value[k]) for k in
                        ('sealed', 'pending', 'rejected', 'basis') if k in value}
        elif key == 'barrier_replies' and type(value) is list:
            out[key] = [{k: copy.deepcopy(row[k]) for k in CORE_BARRIER_KEYS
                         if k in row} for row in value[:64] if type(row) is dict]
        elif key == 'interval':
            out[key] = _project_interval(value)
        elif key == 'decode_eligibility':
            out[key] = _project_eligibility(value)
        else:
            out[key] = copy.deepcopy(value)
    return _bind_receipt_text(out, result)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def receipt_digest(result):
    return digest(receipt_core(result))


# Bounded per-rank counter identity retained in interval evidence.
INTERVAL_COUNTER_KEYS = (
    'route', 'route_generation', 'storage_epoch', 'replays',
    'replay_streams_match_owner', 'transition_count', 'interval_invalidations',
    'in_flight', 'sampling_pending', 'transition_pending', 'quarantined',
)
INTERVAL_BATCH_KEYS = (
    'batches', 'decode_batches', 'non_decode_batches', 'empty_batches',
    'full_graph_dispatches', 'non_full_dispatches', 'scheduler_observations',
    'preempted_requests', 'unknown_preemption_observations',
)
INTERVAL_DELTA_KEYS = (
    'decode_batches', 'non_decode_batches', 'non_full_dispatches',
    'empty_batches', 'preempted_requests', 'unknown_preemption_observations',
    'scheduler_observations',
)


def _counter_row(row):
    """Actual bounded counter metadata; never fills a missing value with zero."""
    counters = row.get('counters') if type(row) is dict else None
    if type(counters) is not dict:
        return None
    out = {}
    for key in INTERVAL_COUNTER_KEYS:
        value = counters.get(key)
        if value is not None:
            out[key] = value if type(value) in (bool, int) else str(value)[:120]
    batches = counters.get('batch_observations')
    if type(batches) is dict:
        observed = {key: batches.get(key) for key in INTERVAL_BATCH_KEYS}
        last = batches.get('last')
        observed['last'] = ({key: last.get(key) for key in
                             ('sequence', 'route', 'route_generation', 'num_reqs',
                              'num_tokens', 'decode_only', 'full_graph')}
                            if type(last) is dict else None)
        out['batch_observations'] = observed
    return out


def _ranked(rows, label):
    if type(rows) is not list:
        raise ProbeControlError('interval-%s-ranks-missing' % label)
    mapped = {}
    for row in rows:
        if type(row) is not dict or type(row.get('rank')) is not int or row['rank'] not in RANKS:
            raise ProbeControlError('interval-%s-rank-invalid' % label)
        if row['rank'] in mapped:
            raise ProbeControlError('interval-%s-rank-duplicate' % label)
        mapped[row['rank']] = row
    if set(mapped) != set(RANKS):
        raise ProbeControlError('interval-%s-rank-coverage' % label)
    return mapped


# Each independently observable invalidation total and the counters that feed
# it. A missing/regressed contributor makes that total an explicitly labelled
# lower bound, never an exact zero; a known positive contribution stays a
# known positive contribution.
INVALIDATION_PARTS = {
    'prefill': ('non_decode_batches',),
    'mixed': ('non_decode_batches',),
    'unsupported': ('non_full_dispatches', 'empty_batches'),
    'preempted': ('preempted_requests',),
}


def interval(before, after):
    """Bounded interval evidence with explicit unavailable/contradictory cases.

    Every observable delta is processed independently. A missing or regressed
    counter does not erase a known positive invalidation in the same rank.
    Totals whose contributors are incomplete are reported as lower bounds via
    ``invalidation_complete``; they are never presented as exact zeros.
    """
    initial, final = [], []
    unavailable = []
    reasons = []
    totals = dict(prefill=0, mixed=0, unsupported=0, preempted=0)
    complete = dict(prefill=True, mixed=True, unsupported=True, preempted=True)
    before_ranks = _ranked(before.get('ranks'), 'before')
    after_ranks = _ranked(after.get('ranks'), 'after')
    counters_available = True
    for rank in RANKS:
        a, b = before_ranks[rank].get('counters'), after_ranks[rank].get('counters')
        if type(a) is not dict or type(b) is not dict:
            counters_available = False
            unavailable.append('counters:%d' % rank)
            for key in complete:
                complete[key] = False
            initial.append({'rank': rank, 'decode_steps': None, 'counters': None})
            final.append({'rank': rank, 'decode_steps': None, 'counters': None})
            continue
        # A changed route/storage/generation between two paused snapshots is an
        # ownership contradiction, not missing measurement evidence.
        if any(a.get(k) != b.get(k) for k in ('route', 'route_generation', 'storage_epoch')):
            raise ProbeControlError('interval-route-changed')
        x, y = a.get('batch_observations', {}), b.get('batch_observations', {})
        if type(x) is not dict or type(y) is not dict:
            counters_available = False
            unavailable.append('batch_observations:%d' % rank)
            for key in complete:
                complete[key] = False
            initial.append({'rank': rank, 'decode_steps': None, 'counters': _counter_row(before_ranks[rank])})
            final.append({'rank': rank, 'decode_steps': None, 'counters': _counter_row(after_ranks[rank])})
            continue
        delta = {}
        for key in INTERVAL_DELTA_KEYS:
            if (type(x.get(key)) is not int or type(y.get(key)) is not int
                    or x[key] < 0 or y[key] < x[key]):
                counters_available = False
                unavailable.append('%s:%d' % (key, rank))
                delta[key] = None
                continue
            delta[key] = y[key] - x[key]
        steps_before, steps_after = x.get('decode_batches'), y.get('decode_batches')
        initial.append({'rank': rank,
                        'decode_steps': steps_before if type(steps_before) is int else None,
                        'counters': _counter_row(before_ranks[rank])})
        final.append({'rank': rank,
                      'decode_steps': steps_after if type(steps_after) is int else None,
                      'counters': _counter_row(after_ranks[rank])})
        # Process every observable delta independently: an unknown counter must
        # not discard a known positive preemption/nondecode observation.
        for total_key, sources in INVALIDATION_PARTS.items():
            parts = [delta.get(source) for source in sources]
            if any(part is None for part in parts):
                complete[total_key] = False
            totals[total_key] += sum(part for part in parts
                                     if type(part) is int)
        unknown_preemption = delta.get('unknown_preemption_observations')
        scheduler = delta.get('scheduler_observations')
        if unknown_preemption is None or scheduler is None:
            reasons.append('interval-preemption-proof-unavailable')
        elif unknown_preemption > 0 or scheduler == 0:
            reasons.append('interval-preemption-proof-unavailable')
    if not counters_available:
        reasons.append('interval-counter-unavailable-or-regressed')
    if unavailable:
        reasons.append('interval-evidence-incomplete')
    for key in ('prefill', 'mixed', 'unsupported', 'preempted'):
        if totals[key] > 0:
            reasons.append('interval-invalidated:' + key)
    deduped = []
    for reason in reasons:
        if reason not in deduped:
            deduped.append(reason)
    return {
        'available': counters_available,
        'valid': counters_available and not deduped,
        'reasons': deduped,
        'unavailable': unavailable,
        'counters_before': initial,
        'counters_after': final,
        'invalidation': totals,
        'invalidation_complete': complete,
        'invalidation_lower_bounds': sorted(key for key, done in complete.items()
                                            if not done),
        'invalidation_basis': ('known positive contributions are retained even when '
                               'another counter is unavailable; totals marked incomplete '
                               'in invalidation_complete are lower bounds, never exact zeros'),
        'observer_basis': ('actual worker batch/dispatch/preemption counts between paused '
                           'snapshots; not client token clocks; missing values stay unavailable, never zero'),
    }


class ProbeSession:
    def __init__(self, controller):
        self.controller = controller
        self.lock = asyncio.Lock()
        self.seq = 0
        self.prior = None
        self.repeat = -1
        self.cohort = None
        self.size = None
        self.phase = 'new'
        self.before = None
        self.failed = False

    async def _call(self, action, **kwargs):
        ctl = self.controller
        return await ctl.execute(dict(action=action, run_id=ctl.run_id,
                                      engine_epoch=ctl.epoch, seq=ctl.seq + 1, **kwargs))

    async def _status(self, command):
        """Read-only, sequence-independent observation; allowed after failure."""
        if (type(command) is not dict
                or set(command) != {'schema', 'run_id', 'action', 'read_only'}
                or type(command.get('schema')) is not int or command['schema'] != 1
                or not _label(command.get('run_id'))
                or command.get('action') != 'status' or command.get('read_only') is not True):
            raise ProbeControlError('invalid-status-command')
        async with self.lock:
            ctl = self.controller
            if ctl.run_id is not None and ctl.run_id != command['run_id']:
                raise ProbeControlError('session-owner-changed')
            observed = await ctl.execute({'action': 'status', 'run_id': command['run_id'],
                                          'read_only': True})
            # Deliberately does not touch seq/prior/phase/failed and never
            # clears controller quarantine.
            return {
                'schema': 1, 'action': 'status', 'run_id': command['run_id'],
                'read_only': True, 'sequence_independent': True,
                'status': 'ok',
                'session': {
                    'failed': self.failed, 'phase': self.phase,
                    'command_seq': self.seq, 'repeat': self.repeat,
                    'cohort_id': self.cohort, 'cohort_size': self.size,
                    'prior_receipt': copy.deepcopy(self.prior),
                },
                'quarantined': bool(ctl.quarantined),
                'quarantine_reason': ctl.quarantine_reason,
                'controller': observed,
                'observed_clock_ns': time.monotonic_ns(),
            }

    async def execute(self, command):
        if type(command) is dict and command.get('action') == 'status':
            return await self._status(command)
        fields = {'schema', 'run_id', 'repeat', 'command_seq', 'action', 'target_route',
                  'cohort_id', 'cohort_size', 'prior_receipt', 'expected_rank_set'}
        actions = ('prepare', 'pause_enter_split', 'pause_full', 'resume', 'pause_end', 'leave_full')
        if (type(command) is not dict or set(command) != fields
                or type(command['schema']) is not int or command['schema'] != 1
                or not _label(command['run_id']) or not _label(command['cohort_id'])
                or not _positive(command['command_seq']) or not _positive(command['cohort_size'])
                or type(command['repeat']) is not int or command['repeat'] < 0
                or command['action'] not in actions or command['target_route'] not in ('full', 'split')
                or command['expected_rank_set'] != list(RANKS)
                or any(type(r) is not int for r in command['expected_rank_set'])):
            raise ProbeControlError('invalid-session-command')
        async with self.lock:
            if self.failed:
                raise ProbeControlError('session-quarantined')
            if command['command_seq'] != self.seq + 1 or command['prior_receipt'] != self.prior:
                raise ProbeControlError('stale-session-command')
            if self.controller.run_id not in (None, command['run_id']):
                raise ProbeControlError('session-owner-changed')
            action = command['action']
            if action == 'prepare':
                if self.phase not in ('new', 'cleanup') or command['repeat'] != self.repeat + 1:
                    raise ProbeControlError('invalid-repeat-boundary')
            elif (command['repeat'] != self.repeat or command['cohort_id'] != self.cohort
                  or command['cohort_size'] != self.size):
                raise ProbeControlError('session-cohort-changed')
            self.seq = command['command_seq']
            try:
                result = await self._execute(command)
            except BaseException as exc:
                self.failed = True
                self.controller.quarantined = True
                self.controller.ingress.sealed = True
                if self.controller.quarantine_reason is None:
                    self.controller.quarantine_reason = (
                        str(exc)[:120] if isinstance(exc, ProbeControlError)
                        else 'session-operation-failed:' + type(exc).__name__)
                raise
            self.prior = {'command_seq': self.seq,
                          'receipt_sha256': receipt_digest(result)}
            return copy.deepcopy(result)

    def _observer_phase(self, snapshot):
        observers = [row.get('inventory', {}).get('replay_observer')
                     for row in snapshot['ranks']]
        if all(value is None or value == {'enabled': False} for value in observers):
            return 'disabled'
        if any(type(value) is not dict or value.get('enabled') is not True
               for value in observers):
            raise ProbeControlError('mixed-observer-declaration')
        phases = {value.get('state') for value in observers}
        if len(phases) != 1 or not phases <= {'started', 'window', 'closed'}:
            raise ProbeControlError('mixed-or-unexpected-observer-state')
        if any(value.get('started_before_split') is not True for value in observers):
            raise ProbeControlError('observer-did-not-start-before-split')
        phase = next(iter(phases))
        if phase == 'closed' and any(value.get('closed') is not True for value in observers):
            raise ProbeControlError('observer-close-not-proven')
        return phase

    async def _execute(self, c):
        ctl, action = self.controller, c['action']
        if ctl.run_id is None:
            raw = await ctl.execute({'action':'open', 'run_id':c['run_id']})
        else:
            raw = await self._call('status')
        before = raw['snapshot']
        after = before
        acks = None
        extra = {}
        if action == 'prepare':
            if self.phase == 'cleanup':
                # Client connections have closed; wait for the ordinary abort
                # path to reach the output processor, without admitting a new
                # cohort or manufacturing a completion.
                deadline = time.monotonic() + ctl.timeout_s
                while ctl.engine.get_num_unfinished_requests() != 0:
                    if time.monotonic() >= deadline:
                        raise ProbeControlError('previous-cohort-not-drained')
                    await asyncio.sleep(.01)
            if (before['route'] != 'full' or raw['paused'] is not False
                    or ctl.engine.get_num_unfinished_requests() != 0):
                raise ProbeControlError('prepare-requires-empty-running-full-engine')
            ctl.ingress.unseal()
            self.repeat, self.cohort, self.size = c['repeat'], c['cohort_id'], c['cohort_size']
            self.phase = 'prepared'
        elif action in ('pause_enter_split', 'pause_full'):
            if self.phase != 'prepared' or before['route'] != 'full':
                raise ProbeControlError('invalid-entry-phase')
            paused = await self._call('pause')
            eligibility = ctl._eligibility(paused['snapshot'], self.size)
            extra['pause'] = dict(paused['pause'], drained=True)
            extra['barrier_replies'] = paused['barriers']
            extra['ingress'] = paused['ingress']
            extra['decode_eligibility'] = {
                'cohort_id': self.cohort, 'cohort_cardinality': self.size,
                'reason': eligibility['basis'],
                'ranks': [{'rank': r['rank'],
                           'batch_reqs': r['batch']['num_reqs'],
                           'scheduled_tokens': r['batch']['num_tokens'],
                           'prefill': not r['batch']['decode_only'],
                           'sequence': r['batch'].get('sequence'),
                           'num_reqs': r['batch'].get('num_reqs'),
                           'num_tokens': r['batch'].get('num_tokens'),
                           'decode_only': r['batch'].get('decode_only'),
                           'full_graph': r['batch'].get('full_graph'),
                           'route': r['batch'].get('route'),
                           'route_generation': r['batch'].get('route_generation')}
                          for r in eligibility['ranks']]}
            after = paused['snapshot']
            if action == 'pause_enter_split':
                changed = await self._call('transition', target='split',
                    expected_route_generation=after['route_generation'], cohort_size=self.size)
                after, acks = changed['snapshot'], changed['replies']
                observer_phase = self._observer_phase(after)
                if observer_phase == 'started':
                    observed = await self._call('observer_begin')
                    after = observed['snapshot']
                    extra['observer'] = observed['observer']
                elif observer_phase not in ('disabled', 'closed'):
                    raise ProbeControlError('observer-window-already-active')
            self.phase = 'entered'
        elif action == 'resume':
            if self.phase not in ('entered', 'left'):
                raise ProbeControlError('invalid-resume-phase')
            measurement = self.phase == 'entered'
            resumed = await self._call('resume', cohort_size=self.size)
            after = resumed['snapshot']
            extra['resume'] = resumed['resume']
            extra['barrier_replies'] = resumed['barriers']
            if measurement:
                self.before = copy.deepcopy(after)
            self.phase = 'measuring' if measurement else 'cleanup'
        elif action == 'pause_end':
            if self.phase != 'measuring':
                raise ProbeControlError('invalid-end-phase')
            paused = await self._call('pause')
            after = paused['snapshot']
            extra['pause'] = dict(paused['pause'], drained=True)
            extra['barrier_replies'] = paused['barriers']
            extra['ingress'] = paused['ingress']
            # Lifecycle proof is independent of measurement validity. The raw
            # pause action already required the completed keep clear_cache=False
            # pause, four typed drain barriers, unchanged route/storage/epoch and
            # no ambiguous pending transition. An unavailable/invalid interval
            # is preserved as evidence but does not quarantine this session.
            extra['lifecycle_proven'] = True
            extra['lifecycle'] = {
                'pause_completed': True,
                'pause_drained': True,
                'barrier_ranks': [row.get('rank') for row in paused['barriers']],
                'route_unchanged': before['route'] == after['route'],
                'route_generation_unchanged': (before['route_generation']
                                               == after['route_generation']),
                'storage_epoch': after['storage_epoch'],
                'engine_epoch': ctl.epoch,
            }
            if self._observer_phase(after) == 'window':
                observed = await self._call('observer_end')
                after = observed['snapshot']
                extra['observer'] = observed['observer']
            sample = interval(self.before, after)
            extra['interval'] = sample
            extra['measurement'] = {
                'valid': sample['valid'],
                'available': sample['available'],
                'reasons': list(sample['reasons']),
            }
            self.phase = 'ended'
        elif action == 'leave_full':
            if self.phase != 'ended' or not ctl.paused:
                raise ProbeControlError('invalid-leave-phase')
            if before['route'] == 'split':
                changed = await self._call('transition', target='full',
                    expected_route_generation=before['route_generation'], cohort_size=self.size)
                after, acks = changed['snapshot'], changed['replies']
            self.phase = 'left'
        if c['target_route'] != after['route']:
            raise ProbeControlError('session-target-route-differs')
        ranks = [{'rank': r, 'status':'ok'} for r in RANKS]
        if acks is not None:
            for row, ack in zip(ranks, sorted(acks, key=lambda a:a['rank'])):
                row['transition_ack'] = copy.deepcopy(ack)
        if 'observer' in extra:
            extra['observer_sha256'] = digest(extra['observer'])
        return dict(schema=1, run_id=c['run_id'], repeat=c['repeat'], command_seq=self.seq,
                    action=action, cohort_id=c['cohort_id'], status='ok', engine_epoch=ctl.epoch,
                    storage_epoch=after['storage_epoch'], route_before=before['route'], route_after=after['route'],
                    route_generation_before=before['route_generation'], route_generation_after=after['route_generation'],
                    expected_rank_set=list(RANKS), ranks=ranks, observed_clock_ns=time.monotonic_ns(),
                    worker_observations=after['ranks'],
                    clock_basis='frontend monotonic completion; offset to client clock unknown', **extra)
