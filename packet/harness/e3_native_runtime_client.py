"""Bounded synchronous client for the actual native session, no probe aliases."""
import secrets
from e3_native_qualification_driver import LoopbackControl, NativePlacementRunner
from wire_v1 import ROUTE_BANK_FULL, ROUTE_BANK_SPLIT

RANKS = (0, 1, 2, 3)


class NativeRuntimeClient:
    def __init__(self, url, run_id, *, epoch=None, timeout_s=120, control=None):
        self.control = control or LoopbackControl(url, timeout_s=timeout_s)
        self.run_id, self.epoch = run_id, epoch
        self.timeout_s = timeout_s
        self.client_id = 'native-measure-' + secrets.token_hex(8)
        self.sequence = 0
        self.history = []

    def call(self, action, **fields):
        self.sequence += 1
        command = dict(schema=1, action=action, run_id=self.run_id,
                       client_id=self.client_id, command_seq=self.sequence)
        if self.epoch is not None:
            command['engine_epoch'] = self.epoch
        command.update(fields)
        reply = self.control.session(command, self.timeout_s)
        if (type(reply) is not dict or reply.get('schema') != 1
                or reply.get('action') != action or reply.get('run_id') != self.run_id):
            raise ValueError('native-response-identity')
        if action != 'resume':
            epoch = reply.get('engine_epoch')
            if type(epoch) is not int or epoch < 1 or (self.epoch is not None and epoch != self.epoch):
                raise ValueError('native-response-epoch')
            self.epoch = epoch
        self.history.append(reply)
        return reply

    def prepare(self, route='full', phase=None):
        fields = dict(target_route=route)
        if phase is not None:
            fields['phase'] = phase
        reply = self.call('prepare', **fields)
        self.require_acks(reply, route)
        if reply.get('route') != route:
            raise ValueError('native-prepare-route')
        return reply

    def require_acks(self, reply, route):
        NativePlacementRunner._require_acks(reply,
            bank=ROUTE_BANK_FULL if route == 'full' else ROUTE_BANK_SPLIT,
            label='native-' + str(reply.get('action')), epoch=self.epoch)

    def pause(self):
        reply = self.call('pause')
        pause = reply.get('pause') or {}
        if (pause.get('paused') is not True or pause.get('mode') != 'keep'
                or pause.get('clear_cache') is not False):
            raise ValueError('native-awaited-keep-pause-unproved')
        rows = self.workers(reply.get('barrier'))
        if any(row.get('barrier_clean') is not True or row.get('drained_streams', 0) < 1
               or row.get('pending_native_replays') != 0 or row.get('retained_rows') != 0
               or row.get('pending_prefill_streams') != 0 or row.get('pending_output_copies') != 0
               for row in rows.values()):
            raise ValueError('native-queried-drain-unproved')
        return reply

    def workers(self, reply):
        rows = reply.get('workers') if type(reply) is dict else None
        if (type(rows) is not list or len(rows) != len(RANKS)
                or any(type(r) is not dict or type(r.get('worker_rank')) is not int for r in rows)
                or {r['worker_rank'] for r in rows} != set(RANKS)
                or any(r.get('engine_epoch') != self.epoch or r.get('engine_epoch_ok') is not True
                       for r in rows)):
            raise ValueError('native-worker-identity-coverage')
        return {r['worker_rank']: r for r in rows}

    def enter(self, phase):
        reply = self.call('pause_enter_split', phase=phase)
        self.require_acks(reply, 'split')
        if reply.get('route') != 'split':
            raise ValueError('native-split-not-applied')
        return reply

    def resume(self):
        reply = self.call('resume')
        if reply.get('resumed') is not True and reply.get('already_running') is not True:
            raise ValueError('native-resume-unproved')
        return reply

    def leave(self):
        reply = self.call('leave_full')
        self.require_acks(reply, 'full')
        if reply.get('route') != 'full':
            raise ValueError('native-full-return-unproved')
        return reply
