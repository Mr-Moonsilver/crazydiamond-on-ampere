"""Native prefill half of one shared observer window, default-off.

The owner primes actual stream identities before tracing, pins every shadow to
persisted package/source identities, and restores all instance shadows before
sealing. Graph-only evidence and the shared native report remain separate.
No readiness, engine control or numerical policy is implemented here.
"""
import ctypes
from pathlib import Path

from native_prefill_offers import NativePrefillLedger, install_seams


class NativeObservationError(RuntimeError):
    pass


def native_clock(binding):
    """Resolve the pinned CUPTI dependency's clock before tracing starts."""
    query = binding._lib.cuptiGetTimestamp
    query.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
    query.restype = ctypes.c_int
    def read():
        value = ctypes.c_uint64()
        if query(ctypes.byref(value)) != 0 or value.value == 0:
            raise NativeObservationError('native-observation-timestamp-unproved')
        return value.value
    read()
    return read


class PrimedIdentities:
    """Queried before tracing; no driver/CUPTI setup calls inside an offer."""
    def __init__(self, query, streams):
        self.rows = {}
        for stream in streams:
            if stream is None:
                continue
            handle = getattr(stream, 'cuda_stream', None)
            if type(handle) is not int or handle <= 0:
                raise NativeObservationError('native-stream-handle-unproved')
            if handle not in self.rows:
                row = dict(query(stream))
                if (set(row) != {'context_id', 'stream_id'}
                        or any(type(v) is not int or v < 1 for v in row.values())):
                    raise NativeObservationError('native-stream-identity-unproved')
                self.rows[handle] = row

    def __call__(self, stream):
        handle = getattr(stream, 'cuda_stream', None)
        if handle not in self.rows:
            raise NativeObservationError('native-stream-not-primed')
        return dict(self.rows[handle])


def persisted_source_pins():
    """Read the generated attestation, never hash live code into a new pin."""
    import decode_probe
    import capture_streams
    import vllm
    evidence = decode_probe.verify_runtime_sources()
    module_root = Path(decode_probe.__file__).resolve().parent
    pins = {str(module_root / (name + '.py')): digest
            for name, digest in evidence['generated_source_shas'].items()}
    vllm_root = Path(vllm.__file__).resolve().parent
    pins.update({str(vllm_root / rel): digest
                 for rel, digest in capture_streams.PINS.items()})
    return pins


class NativeObservation:
    """Source-declared prefill offers beside the existing decode ledger."""
    MAX_ATTACHMENTS = 8192

    def __init__(self, native, runner, *, query, pins, clock):
        from native_startup import WorkerNative
        if not isinstance(native, WorkerNative) or native.runner is not runner:
            raise NativeObservationError('native-worker-owner-unproved')
        if not isinstance(pins, dict) or not pins:
            raise NativeObservationError('native-source-pins-missing')
        if not callable(clock):
            raise NativeObservationError('native-timestamp-provider-missing')
        if getattr(native, 'observation', None) is not None:
            raise NativeObservationError('native-observation-already-owned')
        owners = native.prefill_streams
        if owners is None or owners.state != 'ready' or owners._in_scope:
            raise NativeObservationError('native-prefill-owners-not-ready')
        if native.private_bindings is None:
            raise NativeObservationError('native-private-bindings-missing')
        dispatcher = getattr(runner, '_flash_s5_dispatcher', None)
        if dispatcher is None:
            raise NativeObservationError('native-dispatcher-missing')
        if len(owners.owners) != len(owners.replacements):
            raise NativeObservationError('native-prefill-owner-census-differs')
        aux = {}
        for (owner, attr), replacement in zip(owners.owners, owners.replacements):
            if replacement is None:
                continue  # A stock-disabled auxiliary remains disabled.
            role = {'_stream': 'shared_experts_aux',
                    '_prefetch_stream': 'ple_aux'}.get(attr)
            if role is None:
                raise NativeObservationError('native-prefill-owner-role-unknown')
            aux.setdefault(role, []).append(replacement)
        copy_stream = getattr(native, 'copy_stream', None)
        if copy_stream is None:
            raise NativeObservationError('native-copy-stream-missing')
        self.native, self.runner, self.pins = native, runner, dict(pins)
        self.clock = clock
        self.owners, self.dispatcher = owners, dispatcher
        self.streams = {'prefill_main': native.split.prefill_stream,
                        'copy': copy_stream}
        self.aux = aux
        for role, candidates in aux.items():
            self.streams[role] = candidates[0]
        all_streams = list(self.streams.values()) + [s for ss in aux.values() for s in ss]
        self.identities = PrimedIdentities(query, all_streams)
        self.ledger = None
        self.attachments = {}
        self.source_records = []
        self.closed = False
        self.retained_report = None
        self.accounting = None

    def begin(self, bridge, *, worker, rank, epoch, generation, storage_epoch,
              run_id, window_id):
        if self.ledger is not None or self.closed:
            raise NativeObservationError('native-observation-not-new')
        ledger = NativePrefillLedger(
            bridge, worker=worker, rank=rank, epoch=epoch,
            generation=generation, route_generation=generation,
            storage_epoch=storage_epoch, run_id=run_id, window_id=window_id,
            stream_identity=self.identities, torch_module=self.native.torch,
            clock=self.clock,
            env={'FLASH_S5_NATIVE_PREFILL_OFFERS': '1'})
        self.ledger = ledger
        ledger.declare_streams(prefill_stream=self.streams['prefill_main'],
                               copy_stream=self.streams['copy'], aux_streams=self.aux)
        for role in self.streams:
            if role != 'prefill_main':
                ledger.owner_start('owned:' + role, role, stream=self.streams[role])
        self.native.observation = self
        self.native.private_bindings.attach_ledger(ledger)
        self.attach(self.dispatcher, 'lane_dispatcher')
        for owner, attr in self.owners.owners:
            self.attach(owner, 'shared_experts' if attr == '_stream' else 'ple_embedding')
        return ledger.window()

    def attach(self, target, kind):
        """Attach a real dynamically-created operation once; never skip its body."""
        if self.ledger is None or self.closed:
            return target
        key = (id(target), kind)
        if key in self.attachments:
            return target
        if len(self.attachments) >= self.MAX_ATTACHMENTS:
            self.ledger._invalidate('native-shadow-capacity')
            return target
        try:
            shadow = install_seams(target, self.ledger, kind, pins=self.pins)
        except Exception as exc:
            # Model/source ownership remains intact. The resulting unoffered
            # work and invalid ledger cannot qualify this window.
            self.ledger._invalidate('native-shadow-refused:' + kind + ':' + type(exc).__name__)
        else:
            self.attachments[key] = shadow
            self.source_records.append(shadow.record())
        return target

    def before_end(self):
        """Fence actual role streams and restore observers before native flush."""
        if self.ledger is None or self.closed:
            raise NativeObservationError('native-observation-not-open')
        ledger = self.ledger
        for role, stream in self.streams.items():
            event = self.native.torch.cuda.Event(enable_timing=False)
            event.record(stream)
            event.synchronize()
            if not event.query():
                ledger._invalidate('native-role-drain-unproved:' + role)
            ledger.owner_finish('lane' if role == 'prefill_main' else 'owned:' + role,
                                events=(event,))
        ledger.close_shadows()
        # Every event was queried above. Keep source records, not references to
        # completed per-chunk forwards/operations through performance testing.
        self.attachments.clear()
        self.native.private_bindings.attach_ledger(None)
        if self.native.observation is not self:
            raise NativeObservationError('native-observation-owner-changed')
        self.native.observation = None
        self.closed = True

    def after_end(self, decode):
        """Retain/regrade both exact partitions of the one shared native report."""
        if not self.closed or self.ledger is None:
            raise NativeObservationError('native-observation-not-drained')
        from engine_offers import grade_engine
        from native_prefill_offers import partition_view
        from native_placement import compose_overlap
        self.ledger.finish()
        prefill = self.ledger.seal()
        whole = self.ledger.snapshot()
        native, offers = whole['native'], whole['offers']
        declared = {oid: dict(meta) for oid, meta in decode._meta.items()}
        view = partition_view(native, offers, set(declared), allow_shared=True)
        decode_receipt = grade_engine(view,
            [row for row in offers if row.get('id') in declared], declared, decode.window())
        paired = compose_overlap(native, offers, decode_declared=declared,
            decode_window=decode.window(), prefill_declared=prefill['declared'],
            prefill_window=prefill['window'], decode_receipt=decode_receipt,
            prefill_receipt=prefill)
        self.retained_report = whole
        self.accounting = paired
        return {'engine_accounting': decode_receipt, 'native_prefill_accounting': prefill,
                'native_overlap_accounting': paired,
                'native_report': native, 'offer_report': offers,
                'native_report_read': whole.get('native_report_read'),
                'native_source_seams': list(self.source_records),
                'decode_declared': {str(k): v for k, v in declared.items()},
                'decode_window': decode.window()}
