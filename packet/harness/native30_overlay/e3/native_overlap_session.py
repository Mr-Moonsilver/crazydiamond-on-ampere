"""Owner-side native overlap session: prepare, measure, leave.

The session is a thin orchestration over :class:`native_overlap_control.
NativeControlClient`.  It never touches the driver directly, never installs an
RPC and never treats an enqueue, a submission counter or a declared boolean as
a completion.  Its receipts are built from

* the engine control journal (``overlap_evidence``) with real admitted segment
  intervals, queried worker reports, PP receipts, retired source holds and
  committed frontiers;
* the client-visible decode completion delta the caller supplies for the same
  window (an int, not a rate);
* host-monotonic clock-domain proofs for both boundary commands.

``native_ready`` describes the independently checked implementation/build
closure only. Missing closure remains refused; hardware/release qualification
is never supplied by this session's status metadata.
"""
import time
import os

from native_overlap_control import NativeControlClient
from native_measurement import select_window_evidence, control_path_delta
from native_overlap_metrics import (
    clock_identity,
    measure_windows,
    throughput,
)
from wire_v1 import ROUTE_BANK_FULL

SCHEMA = 1
RUN_ID_CHARS = frozenset('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_')
# The engine's typed phase names.  The engine remains the authority: it refuses
# an unknown, non-native, foreign-epoch or already-armed-for-another-phase
# intent.  Stock full-route preparation, split W1 decode-only and the split
# decode-only phase deliberately leave admission off (no env/config switch).
ADMISSION_PHASES = ('full-prefill-solo', 'split-prefill-solo',
                    'split-w1-decode', 'row1-native', 'w2-overlap')
NATIVE_ADMISSION_PHASES = ('split-prefill-solo', 'row1-native', 'w2-overlap')
NATIVE_READY_REASON = (
    'native readiness is refused: require_native_ready() still raises and the '
    'reviewed construction gaps (private QSA scratch/UID binding, source/oracle '
    'transport, real qualification provider) remain open')


def _label(value):
    return (type(value) is str and 1 <= len(value) <= 80
            and all(char in RUN_ID_CHARS for char in value))


class NativeOverlapSessionError(RuntimeError):
    pass


REFUSAL_POLICY = ('conservative: any native-admission refusal counted between begin and end '
                  'marks the window incomplete (engine-global counters, newest receipt)')
# The driver's refusal of an evidence window that journaled no native work
# (engine_driver.overlap_evidence).  The only evidence failure prefill_end may
# type as an admission refusal (native39, Codex P2).
EMPTY_WINDOW_ERROR = 'overlap-journal-empty-window'


class NativeOverlapSession:
    def __init__(self, engine, *, control=None, clock=time.monotonic_ns,
                 timeout_s=120.0):
        self.engine = engine
        self.control = (control if control is not None
                        else NativeControlClient(engine, timeout_s=timeout_s,
                                                 clock=clock))
        self.clock = clock
        self.run_id = None
        self.route = None
        self.measure_start_ns = None
        self.begin_receipt = None
        self.begin_proof = None

    async def execute(self, command):
        if type(command) is not dict:
            raise NativeOverlapSessionError('session-command-not-an-object')
        if command.get('schema') != SCHEMA:
            raise NativeOverlapSessionError('session-command-schema-differs')
        action = command.get('action')
        run_id = command.get('run_id')
        if run_id is not None and not _label(run_id):
            raise NativeOverlapSessionError('session-run-id-invalid')
        handler = getattr(self, '_action_' + str(action), None)
        if action == 'release_auto_arm' and os.environ.get('FLASH_S5_AUTO_ARM', '0') != '1':
            handler = None
        if action == 'native_progress' and not (
                os.environ.get('FLASH_S5_NATIVE_PROGRESS', '') == 'burst'
                and os.environ.get('FLASH_S5_AUTO_ARM', '0') == '1'):
            handler = None
        if action == 'native_stage_local' and not (
                os.environ.get('FLASH_S5_STAGE_LOCAL', '0') == '1'
                and os.environ.get('FLASH_S5_AUTO_ARM', '0') == '1'):
            handler = None
        if handler is None:
            raise NativeOverlapSessionError('session-action-not-exposed')
        if self.run_id is not None and run_id is not None and run_id != self.run_id:
            raise NativeOverlapSessionError('session-run-id-changed')
        result = await handler(command)
        if type(result) is not dict:
            raise NativeOverlapSessionError('session-action-returned-no-record')
        result.setdefault('schema', SCHEMA)
        epoch = getattr(self.control, 'engine_epoch', None)
        if 'engine_epoch' not in result and type(epoch) is int and epoch > 0:
            result['engine_epoch'] = epoch  # the utility-bound immutable owner
        result.setdefault('action', action)
        result['run_id'] = self.run_id
        from native_readiness import implementation_status
        implementation = implementation_status()
        result['native_ready'] = implementation['native_ready']
        result['native_ready_reason'] = implementation.get('reason') or implementation['scope']
        result['armable'] = implementation['armable']
        result['implementation_closure'] = implementation
        return result

    # -- prepare ---------------------------------------------------------
    async def _action_release_auto_arm(self, command):
        if self.measure_start_ns is not None:
            raise NativeOverlapSessionError('release-with-open-measurement')
        return await self.control.release_auto_arm()

    async def _action_native_progress(self, command):
        if self.measure_start_ns is not None:
            raise NativeOverlapSessionError('native-progress-with-open-measurement')
        return await self.control.native_progress(command.get('mode'))

    async def _action_native_stage_local(self, command):
        if self.measure_start_ns is not None:
            raise NativeOverlapSessionError('native-stage-local-with-open-measurement')
        if 'overlap' in command and os.environ.get('FLASH_S5_STAGE_OVERLAP', '0') == '1':
            # C1 only; with the switch off the command is C0's, field and all, as before.
            return await self.control.native_stage_overlap(command.get('local'), command['overlap'])
        return await self.control.native_stage_local(command.get('local'))

    async def _action_prepare(self, command):
        target = command.get('target_route')
        if target not in ('full', 'split'):
            raise NativeOverlapSessionError('session-target-route-invalid')
        if command.get('run_id') is None:
            raise NativeOverlapSessionError('session-run-id-required')
        phase = command.get('phase')
        if phase is not None and phase not in ADMISSION_PHASES:
            raise NativeOverlapSessionError('session-admission-phase-invalid')
        if target == 'full' and phase in NATIVE_ADMISSION_PHASES:
            # A full-route preparation ends the native phase before the all-rank
            # return; arming one here would only race its own auto-close.
            raise NativeOverlapSessionError(
                'session-full-prepare-requires-admission-off')
        self.run_id = command['run_id']
        pause = await self.control.ensure_paused()
        admission = None
        initial_full_fence = None
        if target == 'full':
            # The native phase (if any) ends before the all-rank return.
            admission = await self._set_admission(phase or 'full-prefill-solo')
            reply = await self.control.leave_split()
            if reply.get('bootstrapped') is False:
                # A cold FULL route has no TransitionAck: wire-v1 transitions
                # advance generation0 to1 or later. Do not invent an ACK or
                # accept an empty typed_acks flag. Bootstrap the real workers,
                # prove that they are empty/drained, then perform one actual
                # empty SPLIT->FULL warmup inside this preparation.
                reply, initial_full_fence = await self._initial_full_fence(reply)
        else:
            reply = await self.control.enter_split()
            if phase is not None:
                admission = await self._set_admission(phase)
        resume = await self.control.resume()
        self.route = reply.get('route_after')
        epoch = reply.get('engine_epoch')
        if type(epoch) is not int or epoch < 1:
            raise NativeOverlapSessionError('session-engine-epoch-unavailable')
        acks = reply.get('transition_acks') or []
        return {
            'target_route': target,
            'route': self.route,
            'engine_epoch': epoch,
            'phase': phase,
            'admission': admission,
            'paused': pause,
            'resumed': resume,
            'changed': reply.get('changed'),
            'bootstrap_acks': reply.get('bootstrap_acks') or [],
            'transition_acks': acks,
            'typed_acks': reply.get('typed_acks') is True,
            'workers': ((self.control.last_snapshot or {}).get('workers')),
            'snapshot': self.control.last_snapshot,
            'initial_full_fence': initial_full_fence,
        }

    async def _initial_full_fence(self, initial):
        if (initial.get('route_after') != 'full'
                or type(initial.get('route_generation_after')) is not int
                or initial.get('route_generation_after') != 0
                or initial.get('transition_acks') != []
                or initial.get('bootstrap_acks') != []):
            raise NativeOverlapSessionError('initial-full-state-contradictory')
        bootstrap = await self.control.bootstrap_engine_owner()
        barrier = await self.control.barrier()
        rows = barrier.get('workers')
        ranks = (bootstrap.get('snapshot') or {}).get('ranks')
        if (type(rows) is not list or not rows or type(ranks) is not list
                or any(type(row) is not dict for row in rows)
                or any(type(rank) is not int or rank < 0 for rank in ranks)
                or len(set(ranks)) != len(ranks)
                or {row.get('worker_rank') for row in rows} != set(ranks)
                or len(rows) != len(ranks)):
            raise NativeOverlapSessionError('initial-full-worker-coverage-unproved')
        zeros = ('live_decode_requests', 'running_requests', 'prefill_requests',
                 'retained_rows', 'pending_native_replays',
                 'pending_prefill_streams', 'pending_output_copies')
        for row in rows:
            if (row.get('engine_epoch') != initial.get('engine_epoch')
                    or row.get('route') != 'full' or row.get('route_generation') != 0
                    or row.get('barrier_clean') is not True
                    or (row.get('live_decode') or {}).get('available') is not True
                    or any(type(row.get(key)) is not int or row[key] != 0 for key in zeros)):
                raise NativeOverlapSessionError('initial-full-warmup-needs-empty-drained-workers')
        entered = await self.control.enter_split()
        if entered.get('route_after') != 'split' or entered.get('typed_acks') is not True:
            raise NativeOverlapSessionError('initial-full-split-transition-unproved')
        returned = await self.control.leave_split()
        if returned.get('route_after') != 'full' or returned.get('typed_acks') is not True:
            raise NativeOverlapSessionError('initial-full-return-unproved')
        return returned, {'initial': initial, 'bootstrap': bootstrap,
                          'barrier': barrier, 'entered': entered, 'returned': returned}

    # -- pause / enter split ---------------------------------------------
    async def _action_pause_enter_split(self, command):
        phase = command.get('phase')
        if phase is not None and phase not in ADMISSION_PHASES:
            raise NativeOverlapSessionError('session-admission-phase-invalid')
        pause = await self.control.ensure_paused()
        reply = await self.control.enter_split()
        admission = (await self._set_admission(phase)
                     if phase is not None else None)
        self.route = reply.get('route_after')
        return {
            'route': self.route,
            'pause': pause,
            'phase': phase,
            'admission': admission,
            'engine_epoch': reply.get('engine_epoch'),
            'changed': reply.get('changed'),
            'route_generation_before': reply.get('route_generation_before'),
            'route_generation_after': reply.get('route_generation_after'),
            'bootstrap_acks': reply.get('bootstrap_acks') or [],
            'transition_acks': reply.get('transition_acks') or [],
            'typed_acks': reply.get('typed_acks') is True,
        }

    # -- typed admission intent ------------------------------------------
    async def _set_admission(self, phase):
        if phase in NATIVE_ADMISSION_PHASES:
            reply = await self.control.arm_admission(phase)
            armed = True
        elif phase in ADMISSION_PHASES:
            # Stock full-route preparation and the two decode-only phases
            # explicitly keep the native admission policy off.
            reply = await self.control.disarm_admission()
            armed = False
        else:
            raise NativeOverlapSessionError('session-admission-phase-invalid')
        return {'phase': phase,
                'armed': armed,
                'admission_intent': reply.get('admission_intent'),
                'previous_admission_intent': reply.get(
                    'previous_admission_intent'),
                'route': reply.get('route'),
                'route_generation': reply.get('route_generation')}

    async def _action_admission(self, command):
        phase = command.get('phase')
        if phase not in ADMISSION_PHASES:
            raise NativeOverlapSessionError('session-admission-phase-invalid')
        return await self._set_admission(phase)

    async def _action_arm_admission(self, command):
        phase = command.get('phase')
        if phase not in NATIVE_ADMISSION_PHASES:
            raise NativeOverlapSessionError('session-admission-phase-not-native')
        return await self._set_admission(phase)

    async def _action_disarm_admission(self, command):
        reply = await self.control.disarm_admission()
        return {'armed': False,
                'admission_intent': reply.get('admission_intent'),
                'previous_admission_intent': reply.get(
                    'previous_admission_intent'),
                'route': reply.get('route'),
                'route_generation': reply.get('route_generation')}

    async def _action_resume(self, command):
        return await self.control.resume()

    async def _action_pause(self, command):
        """An actual awaited keep-pause plus all-rank queried barrier."""
        pause = await self.control.ensure_paused()
        barrier = await self.control.barrier()
        snapshot = await self.control.status()
        return {'pause': pause, 'barrier': barrier, 'snapshot': snapshot,
                'engine_epoch': snapshot.get('engine_epoch'),
                'route': snapshot.get('route'),
                'route_generation': snapshot.get('route_generation')}

    async def _action_identity(self, command):
        """Actual worker construction records while paused; never a probe open."""
        pause = await self.control.ensure_paused()
        # identity uses the same ordered worker-reply path, with its explicit
        # worker action; no additional RPC or reply-queue owner is installed.
        rows = await self.control.worker_identity()
        return {'pause': pause, 'workers': rows.get('workers'),
                'engine_epoch': rows.get('engine_epoch'),
                'snapshot': rows.get('snapshot')}

    # -- measurement -----------------------------------------------------
    async def _action_overlap_begin(self, command):
        if command.get('route') not in (None, 'split'):
            raise NativeOverlapSessionError('session-begin-requires-split-route')
        status = await self.control.status()
        if status.get('route') != 'split':
            raise NativeOverlapSessionError('session-begin-route-is-not-split')
        proof = (self.control.history[-1].get('clock_proof')
                 if self.control.history else None)
        if type(proof) is not dict or proof.get('proved') is not True:
            raise NativeOverlapSessionError('session-begin-clock-unproved')
        self.begin_receipt = status
        self.begin_proof = proof
        self.measure_start_ns = int(self.clock())
        return {
            'route': status.get('route'),
            'engine_epoch': status.get('engine_epoch'),
            'measure_start_ns': self.measure_start_ns,
            'clock_proof': proof,
            'decode_streams': command.get('decode_streams'),
            'snapshot': status,
        }

    async def _action_overlap_end(self, command):
        if self.measure_start_ns is None or self.begin_receipt is None:
            raise NativeOverlapSessionError('session-end-without-begin')
        end_ns = int(self.clock())
        evidence = await self.control.overlap_evidence(self.measure_start_ns,
                                                       end_ns)
        end_proof = (self.control.history[-1].get('clock_proof')
                     if self.control.history else None)
        decode_progress = command.get('decode_progress')
        if decode_progress is not None and type(decode_progress) is not int:
            raise NativeOverlapSessionError('session-decode-progress-not-an-int')
        measured, binding = select_window_evidence(
            evidence, self.begin_receipt, self.measure_start_ns, end_ns)
        windows = measure_windows(
            begin_receipt=self.begin_receipt, end_receipt=evidence,
            begin_proof=self.begin_proof, end_proof=end_proof,
            segments=measured.get('segments'), completions=measured.get('completions'),
            decode_progress=decode_progress)
        prefill = dict(measured.get('prefill') or {})
        prefill['prefill_tokens_per_s'] = throughput(
            prefill.get('useful_completed_tokens'), prefill.get('prefill_wall_s'))
        serialized = bool(measured.get('serialized_transport')) or any(
            segment.get('transport') != 'native-split'
            for segment in windows.get('segments') or []
            if segment.get('accepted') is True)
        enabled = (windows.get('valid') is True
                   and binding.get('valid') is True
                   and not serialized)
        reasons = list(windows.get('reasons') or [])
        reasons.extend(binding.get('reasons') or [])
        phase = (self.begin_receipt.get('admission_intent') or {}).get('phase')
        if phase == 'w2-overlap' and (windows.get('append_prefill') or {}).get('valid') is not True:
            enabled = False
            reasons.append('native-append-window-unproved')
        if serialized:
            reasons.append('serialized-or-uva-transport-not-accepted-overlap')
        raw_loss = (evidence.get('journal') or {}).get('complete_in_window') is not True
        if raw_loss and binding.get('complete') is not True:
            reasons.append('engine-journal-evicted-before-window')
        return {
            'measure_start_ns': self.measure_start_ns,
            'measure_end_ns': end_ns,
            'route': evidence.get('route'),
            'engine_epoch': evidence.get('engine_epoch'),
            'enabled': enabled,
            'reasons': sorted(set(reasons)),
            'segments': windows.get('segments') or [],
            'outer': windows.get('outer'),
            'fresh_prefill': windows.get('fresh_prefill'),
            'append_prefill': windows.get('append_prefill'),
            'completions': windows.get('completions') or [],
            'decode_progress': windows.get('decode_progress'),
            'prefill': prefill,
            'source_holds': measured.get('source_holds') or [],
            'committed_frontiers': measured.get('committed_frontiers') or [],
            'pp_consumer_receipts': measured.get('pp_consumer_receipts') or [],
            'cancellations': measured.get('cancellations') or [],
            'workers': evidence.get('workers') or [],
            'journal': evidence.get('journal') or {},
            'measurement': evidence.get('measurement'),
            'control_path': control_path_delta(
                self.begin_receipt.get('control_path'), evidence.get('control_path')),
            'measurement_binding': binding,
            'raw_journal_evidence': {key: evidence.get(key) for key in (
                'segments', 'completions', 'prefill', 'source_holds',
                'committed_frontiers', 'pp_consumer_receipts', 'cancellations',
                'reasons', 'construction_valid')},
            'notes': (['raw journal eviction retained descriptively; independent census required']
                      if raw_loss else []),
            'allocator_peaks': _allocator_peaks(evidence.get('workers') or []),
            'clock': {'begin': self.begin_proof, 'end': end_proof,
                      'identity': clock_identity()},
            'interval': {'reset_count': None,
                         'basis': ('the control journal does not expose a '
                                   'scheduler interval reset counter')},
        }

    async def _action_prefill_begin(self, command):
        """One native request family, measured separately across a FULL tail.

        Unlike overlap_begin/end this also serves the matched prefill-only
        denominator: it makes no claim that a decode cohort exists.
        """
        if self.measure_start_ns is not None:
            raise NativeOverlapSessionError('prefill-window-already-open')
        phase = command.get('phase')
        if phase not in ('split-prefill-solo', 'w2-overlap'):
            raise NativeOverlapSessionError('prefill-phase-invalid')
        status = await self.control.status()
        if (status.get('route') != 'split'
                or (status.get('admission_intent') or {}).get('phase') != phase):
            raise NativeOverlapSessionError('prefill-admission-not-armed')
        proof = self.control.history[-1].get('clock_proof') if self.control.history else None
        if type(proof) is not dict or proof.get('proved') is not True:
            raise NativeOverlapSessionError('prefill-begin-clock-unproved')
        self.measure_start_ns = int(self.clock())
        self.begin_receipt, self.begin_proof = status, proof
        return {'engine_epoch': status.get('engine_epoch'), 'route': status.get('route'),
                'phase': phase, 'snapshot': status, 'clock_proof': proof,
                'measure_start_ns': self.measure_start_ns}

    async def _action_prefill_end(self, command):
        if self.measure_start_ns is None or self.begin_receipt is None:
            raise NativeOverlapSessionError('prefill-end-without-begin')
        end_ns = int(self.clock())
        try:
            raw = await self.control.overlap_evidence(self.measure_start_ns, end_ns)
        except Exception as error:
            # native37: a request native admission refused leaves no journaled
            # native work, so the evidence read refuses the empty window.  Type
            # that refusal from the queried admission record.  native39: only
            # that exact refusal, and only for a refusal receipt observed inside
            # this window on this engine; every other evidence failure (transport,
            # identity, evidence defects) is raised unchanged.
            if EMPTY_WINDOW_ERROR not in str(error):
                raise
            refused = await self._refused_prefill_end(end_ns, error)
            if refused is None:
                raise error
            return refused
        proof = self.control.history[-1].get('clock_proof') if self.control.history else None
        measured, binding = select_window_evidence(raw, self.begin_receipt,
                                                   self.measure_start_ns, end_ns)
        result = {'engine_epoch': raw.get('engine_epoch'), 'route': raw.get('route'),
                  'begin': self.begin_receipt, 'clock': {'begin': self.begin_proof, 'end': proof},
                  'measure_start_ns': self.measure_start_ns, 'measure_end_ns': end_ns,
                  'measurement_binding': binding, 'measurement': raw.get('measurement'),
                  'raw_journal_evidence': raw,
                  'complete': binding.get('valid') is True
                              and type(proof) is dict and proof.get('proved') is True}
        refusal = native_admission_refusal(self.begin_receipt.get('native_admission'),
                                           raw.get('native_admission'))
        if refusal is not None:
            # Typed: native admission refused work inside this window (the
            # request waited unallocated on the armed split route).  The policy
            # is conservative: the counters are engine-global, so ANY refusal
            # counted in the window marks it incomplete, also when the measured
            # request itself ran; it is never read as "the request never ran".
            result['refusal'] = refusal
            result['refusal_policy'] = REFUSAL_POLICY
            result['complete'] = False
        self.measure_start_ns, self.begin_receipt, self.begin_proof = None, None, None
        return result

    async def _refused_prefill_end(self, end_ns, error):
        """The typed refusal result when the evidence read failed, else ``None``."""
        try:
            status = await self.control.status()
        except Exception:  # noqa: BLE001 - the original evidence error is raised
            return None
        if status.get('engine_epoch') != self.begin_receipt.get('engine_epoch'):
            return None
        refusal = native_admission_refusal(self.begin_receipt.get('native_admission'),
                                           status.get('native_admission'))
        if refusal is None:
            return None
        # Correlated to this window: the newest typed receipt was observed
        # between begin and end on the same engine.  The untyped
        # stock-deferred-without-admission case has no receipt and is kept as
        # before (counts after begin, nothing admitted).
        last = refusal.get('last_refusal')
        if last is not None:
            observed = last.get('observed_ns')
            if (type(observed) is not int or last.get('engine_epoch') != status.get('engine_epoch')
                    or not self.measure_start_ns <= observed <= end_ns):
                return None
        result = {'engine_epoch': status.get('engine_epoch'), 'route': status.get('route'),
                  'begin': self.begin_receipt, 'clock': {'begin': self.begin_proof, 'end': None},
                  'measure_start_ns': self.measure_start_ns, 'measure_end_ns': end_ns,
                  'measurement_binding': None, 'measurement': None, 'raw_journal_evidence': None,
                  'evidence_error': {'type': type(error).__name__, 'message': str(error)[:512]},
                  'end_status': status, 'refusal': refusal, 'refusal_policy': REFUSAL_POLICY,
                  'complete': False}
        self.measure_start_ns, self.begin_receipt, self.begin_proof = None, None, None
        return result

    # -- return to the full route ----------------------------------------
    async def _action_leave_full(self, command):
        pause = await self.control.ensure_paused()
        # Disable the explicit intent before the all-rank return so no stale
        # arm survives into stock sampling, even if the route is already full.
        admission = await self._set_admission('full-prefill-solo')
        census = await self.control.census()
        reply = await self.control.leave_split()
        self.route = reply.get('route_after')
        acks = reply.get('transition_acks') or []
        if reply.get('typed_acks') is not True:
            raise NativeOverlapSessionError('session-leave-acks-not-typed')
        drain = []
        for row in acks:
            value = row.get('drained_streams')
            if type(value) is not int or value < 1:
                raise NativeOverlapSessionError(
                    'session-leave-ack-drain-unproved')
            # ``applied_bank`` is the wire-v1 route bank the worker switched
            # to (an int), never a transport label.
            if type(row.get('applied_bank')) is not int \
                    or row['applied_bank'] != ROUTE_BANK_FULL:
                raise NativeOverlapSessionError(
                    'session-leave-ack-bank-unproved')
            drain.append({'worker_rank': row.get('worker_rank'),
                          'drained_streams': value,
                          'applied_route_generation':
                              row.get('applied_route_generation'),
                          'decode_replay_watermark': row.get('decode_replay_watermark')})
        resume = await self.control.resume()
        self.measure_start_ns = None
        self.begin_receipt = None
        self.begin_proof = None
        return {
            'route': self.route,
            'pause': pause,
            'resume': resume,
            'admission': admission,
            'census_before_return': census,
            'transition_acks': acks,
            'worker_drain': drain,
            'typed_acks': True,
        }

    # -- read-only -------------------------------------------------------
    async def _action_status(self, command):
        status = await self.control.status()
        status.pop('native_ready', None)
        return status


def native_admission_refusal(begin, end):
    """Typed refusal between two engine admission records, or ``None``.

    ``begin``/``end`` are the driver's ``native_admission`` records at the
    window's begin and end (status and overlap evidence).  A refusal counted
    after begin is reported with the newest typed receipt; nothing is inferred
    when either record is missing (an older engine has none).
    """
    if type(begin) is not dict or type(end) is not dict:
        return None
    counts = {}
    for key in ('refused', 'alloc_failures', 'admitted', 'plan_searches',
                'plan_memo_hits', 'split_stock_deferrals', 'refusal_seq'):
        a, b = begin.get(key), end.get(key)
        if type(a) is not int or type(b) is not int or b < a:
            return {'reason': 'native-admission-record-invalid', 'field': key}
        counts[key] = b - a
    if counts['refused'] == 0 and counts['alloc_failures'] == 0:
        if counts['split_stock_deferrals'] > 0 and counts['admitted'] == 0:
            # Stock prefill held on the armed split route and nothing admitted
            # natively: the window's request never ran (e.g. an untyped
            # environment refusal such as a speculative config).
            return {'reason': 'native-admission-refused:stock-deferred-without-admission',
                    'deltas': counts, 'last_refusal': None}
        return None
    last = end.get('last_refusal')
    reason = last.get('reason') if type(last) is dict else None
    return {'reason': 'native-admission-refused:' + (reason if type(reason) is str else 'untyped'),
            'deltas': counts,
            'last_refusal': dict(last) if type(last) is dict else None}


def _allocator_peaks(workers):
    """Peak allocator bytes copied from the workers' own queried census."""
    rows = [row for row in workers
            if type(row) is dict and type(row.get('allocator')) is dict]
    ranks = [row.get('worker_rank') for row in rows]
    if not rows or not all(type(rank) is int for rank in ranks):
        return {'measured': False,
                'basis': 'worker allocator census unavailable',
                'ranks': []}
    return {
        'measured': True,
        'basis': 'torch-cuda-allocator-queried-v1',
        'ranks': sorted(ranks),
        'allocated_peak_bytes': {row['worker_rank']:
                                 row['allocator'].get('allocated_peak_bytes')
                                 for row in rows},
        'reserved_peak_bytes': {row['worker_rank']:
                                row['allocator'].get('reserved_peak_bytes')
                                for row in rows},
        'devices': {row['worker_rank']: row['allocator'].get('device_index')
                    for row in rows},
    }


__all__ = ['ADMISSION_PHASES', 'NATIVE_ADMISSION_PHASES', 'NATIVE_READY_REASON',
           'NativeOverlapSession', 'NativeOverlapSessionError', 'SCHEMA']
