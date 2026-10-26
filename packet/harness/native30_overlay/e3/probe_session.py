"""Stateful W1 receipt adapter over the fixed, awaited diagnostic controller.

No synthetic worker ACKs. Counter evidence covers the enclosing worker interval;
client receive timestamps remain a separate, narrower measurement window.
"""
import asyncio
import copy
import hashlib
import json
import time

from probe_control import ProbeControlError, RANKS, _label, _positive


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def interval(before, after):
    keys = ('decode_batches', 'non_decode_batches', 'non_full_dispatches',
            'empty_batches', 'preempted_requests', 'unknown_preemption_observations',
            'scheduler_observations')
    initial, final = [], []
    totals = dict(prefill=0, mixed=0, unsupported=0, preempted=0)
    for rank in RANKS:
        a, b = before['ranks'][rank]['counters'], after['ranks'][rank]['counters']
        if any(a[k] != b[k] for k in ('route', 'route_generation', 'storage_epoch')):
            raise ProbeControlError('interval-route-changed')
        x, y = a.get('batch_observations', {}), b.get('batch_observations', {})
        delta = {}
        for key in keys:
            if (type(x.get(key)) is not int or type(y.get(key)) is not int
                    or x[key] < 0 or y[key] < x[key]):
                raise ProbeControlError('interval-counter-unavailable-or-regressed')
            delta[key] = y[key] - x[key]
        if delta['unknown_preemption_observations'] or not delta['scheduler_observations']:
            raise ProbeControlError('interval-preemption-proof-unavailable')
        initial.append({'rank': rank, 'decode_steps': x['decode_batches']})
        final.append({'rank': rank, 'decode_steps': y['decode_batches']})
        # A non-decode batch is conservatively disqualifying; these two labels
        # intentionally share one counter, not two independent classifications.
        totals['prefill'] += delta['non_decode_batches']
        totals['mixed'] += delta['non_decode_batches']
        totals['unsupported'] += delta['non_full_dispatches'] + delta['empty_batches']
        totals['preempted'] += delta['preempted_requests']
    return {'counters_before': initial, 'counters_after': final, 'invalidation': totals,
            'observer_basis': 'actual worker batch/dispatch/preemption counts between paused snapshots; not client token clocks'}


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

    async def execute(self, command):
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
            except BaseException:
                self.failed = True
                self.controller.quarantined = True
                raise
            self.prior = {'command_seq': self.seq, 'receipt_sha256': digest(result)}
            return copy.deepcopy(result)

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
            if (before['route'] != 'full' or raw['paused'] is not False
                    or ctl.engine.get_num_unfinished_requests() != 0):
                raise ProbeControlError('prepare-requires-empty-running-full-engine')
            self.repeat, self.cohort, self.size = c['repeat'], c['cohort_id'], c['cohort_size']
            self.phase = 'prepared'
        elif action in ('pause_enter_split', 'pause_full'):
            if self.phase != 'prepared' or before['route'] != 'full':
                raise ProbeControlError('invalid-entry-phase')
            paused = await self._call('pause')
            eligibility = ctl._eligibility(paused['snapshot'], self.size)
            extra['pause'] = dict(paused['pause'], drained=True)
            extra['decode_eligibility'] = {
                'cohort_id': self.cohort, 'cohort_cardinality': self.size,
                'reason': eligibility['basis'],
                'ranks': [{'rank': r['rank'], 'batch_reqs': r['batch']['num_reqs'],
                           'scheduled_tokens': r['batch']['num_tokens'], 'prefill': not r['batch']['decode_only']}
                          for r in eligibility['ranks']]}
            after = paused['snapshot']
            if action == 'pause_enter_split':
                changed = await self._call('transition', target='split',
                    expected_route_generation=after['route_generation'], cohort_size=self.size)
                after, acks = changed['snapshot'], changed['replies']
            self.phase = 'entered'
        elif action == 'resume':
            if self.phase not in ('entered', 'left'):
                raise ProbeControlError('invalid-resume-phase')
            measurement = self.phase == 'entered'
            resumed = await self._call('resume', cohort_size=self.size)
            after = resumed['snapshot']
            extra['resume'] = resumed['resume']
            if measurement:
                self.before = copy.deepcopy(after)
            self.phase = 'measuring' if measurement else 'cleanup'
        elif action == 'pause_end':
            if self.phase != 'measuring':
                raise ProbeControlError('invalid-end-phase')
            paused = await self._call('pause')
            after = paused['snapshot']
            extra['pause'] = dict(paused['pause'], drained=True)
            extra['interval'] = interval(self.before, after)
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
        return dict(schema=1, run_id=c['run_id'], repeat=c['repeat'], command_seq=self.seq,
                    action=action, cohort_id=c['cohort_id'], status='ok', engine_epoch=ctl.epoch,
                    storage_epoch=after['storage_epoch'], route_before=before['route'], route_after=after['route'],
                    route_generation_before=before['route_generation'], route_generation_after=after['route_generation'],
                    expected_rank_set=list(RANKS), ranks=ranks, observed_clock_ns=time.monotonic_ns(),
                    clock_basis='frontend monotonic completion; offset to client clock unknown', **extra)
