"""E3 green-context split for one GPU worker (D4, worker/native side).

Binds the header-compiled helper ``native/flash_s5_green.cu`` through ctypes on
our own exported C ABI; no CUDA call happens at import, and no driver struct
layout is guessed in Python. The helper library is built on the box by the
overlay generation step and its identity (source sha, library sha, ABI) is
recorded next to the measured split.

Evidence boundary: ``create`` returns the SM counts the driver reports for the
two partitions; ``membership`` measures which SMs each stream's kernels land on
with the probe kernel. Both are recorded, neither is a performance claim. Tests
inject a fake binding and a fake torch; they cannot establish native readiness.
"""
import hashlib
import os

from e3_types import E3ContractError

ABI_VERSION = 1
HELPER_SOURCE = 'native/flash_s5_green.cu'


class GreenContextError(E3ContractError):
    pass


def helper_source_sha256():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), HELPER_SOURCE)
    with open(path, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


class DriverBinding:
    """ctypes binding of the compiled helper. Only our exported ABI is typed."""

    def __init__(self, library_path, *, ctypes_module=None):
        if ctypes_module is None:
            import ctypes as ctypes_module
        if not isinstance(library_path, str) or not os.path.isfile(library_path):
            raise GreenContextError('green helper library is missing')
        with open(library_path, 'rb') as handle:
            self.library_sha256 = hashlib.sha256(handle.read()).hexdigest()
        self.library_path = library_path
        c = ctypes_module
        lib = c.CDLL(library_path)
        U, I, P = c.c_uint64, c.c_int, c.c_void_p
        lib.s5_abi.argtypes = []
        lib.s5_abi.restype = I
        lib.s5_create.argtypes = [I, c.c_uint, c.POINTER(P), c.POINTER(U), c.POINTER(U),
                                  c.POINTER(I), c.POINTER(I), c.POINTER(U), c.POINTER(U)]
        lib.s5_create.restype = I
        lib.s5_close.argtypes = [P]
        lib.s5_close.restype = I
        lib.s5_probe.argtypes = [U, I, U, U]
        lib.s5_probe.restype = I
        self._c = c
        self._lib = lib
        abi = int(lib.s5_abi())
        if abi != ABI_VERSION:
            raise GreenContextError(f'green helper ABI {abi} != expected {ABI_VERSION}')
        self.abi = abi

    def create(self, device_index, requested_decode_sms):
        c = self._c
        handle, ds, ps = c.c_void_p(), c.c_uint64(), c.c_uint64()
        dc, pc = c.c_int(), c.c_int()
        dctx, pctx = c.c_uint64(), c.c_uint64()
        rc = self._lib.s5_create(int(device_index), int(requested_decode_sms),
                                 c.byref(handle), c.byref(ds), c.byref(ps),
                                 c.byref(dc), c.byref(pc), c.byref(dctx), c.byref(pctx))
        if rc:
            raise GreenContextError(f'green context create failed: CUresult={rc}')
        return (handle, int(ds.value), int(ps.value), int(dc.value), int(pc.value),
                int(dctx.value), int(pctx.value))

    def close(self, handle):
        return int(self._lib.s5_close(handle))

    def probe(self, pointer, blocks, stream_token, cycles):
        return int(self._lib.s5_probe(int(pointer), int(blocks), int(stream_token), int(cycles)))

    def identity(self):
        return {'library_sha256': self.library_sha256, 'abi': self.abi,
                'helper_source_sha256': helper_source_sha256()}


class MembershipResult:
    __slots__ = ('decode_sms', 'prefill_sms', 'total_sms', 'ok', 'rounds')

    def __init__(self, decode_sms, prefill_sms, total_sms, ok, rounds):
        self.decode_sms = frozenset(decode_sms)
        self.prefill_sms = frozenset(prefill_sms)
        self.total_sms = total_sms
        self.ok = ok
        self.rounds = rounds

    def as_record(self):
        return {'decode_seen': len(self.decode_sms), 'prefill_seen': len(self.prefill_sms),
                'total': self.total_sms, 'disjoint': not (self.decode_sms & self.prefill_sms),
                'ok': self.ok, 'rounds': self.rounds}


class GreenSplit:
    """Two green contexts (decode N SMs, prefill the rest) and one stream each."""

    def __init__(self, binding, device_index, requested_decode_sms, *, torch_module=None):
        if not all(callable(getattr(binding, m, None)) for m in ('create', 'close', 'probe')):
            raise GreenContextError('green split needs a driver binding')
        if type(device_index) is not int or device_index < 0:
            raise GreenContextError('device index must be a non-negative int')
        if type(requested_decode_sms) is not int or requested_decode_sms < 1:
            raise GreenContextError('requested decode SM count must be a positive int')
        self.binding = binding
        self.device_index = device_index
        self.requested_decode_sms = requested_decode_sms
        if torch_module is None:
            import torch as torch_module
        self.torch = torch_module
        self.state = 'new'
        self._handle = None
        self.decode_stream = None
        self.prefill_stream = None
        self.decode_stream_token = None
        self.prefill_stream_token = None
        self.decode_context_id = None
        self.prefill_context_id = None
        self.decode_sms = None
        self.prefill_sms = None
        self.total_sms = None
        self.membership_result = None

    # -- lifecycle --------------------------------------------------------
    def create(self):
        if self.state != 'new':
            raise GreenContextError(f'green split cannot be created from state {self.state!r}')
        props = self.torch.cuda.get_device_properties(self.device_index)
        total = int(getattr(props, 'multi_processor_count', 0))
        if total < 2:
            raise GreenContextError('device reports no usable SM count')
        if self.requested_decode_sms >= total:
            raise GreenContextError('requested decode SMs leave no prefill partition')
        handle, ds, ps, dc, pc, dctx, pctx = self.binding.create(
            self.device_index, self.requested_decode_sms)
        if dc < self.requested_decode_sms or pc < 1 or dc + pc != total:
            # The helper already rejects this; unwind if a binding lies.
            self.binding.close(handle)
            raise GreenContextError(f'driver split {dc}+{pc} does not cover {total} SMs')
        if ds == ps or dctx == pctx or not ds or not ps or not dctx or not pctx:
            self.binding.close(handle)
            raise GreenContextError('green helper returned aliased or null stream/context handles')
        self._handle = handle
        self.decode_stream_token, self.prefill_stream_token = ds, ps
        self.decode_context_id, self.prefill_context_id = dctx, pctx
        self.decode_sms, self.prefill_sms, self.total_sms = dc, pc, total
        self.decode_stream = self.torch.cuda.ExternalStream(ds, device=self.device_index)
        self.prefill_stream = self.torch.cuda.ExternalStream(ps, device=self.device_index)
        for stream, token in ((self.decode_stream, ds), (self.prefill_stream, ps)):
            if int(getattr(stream, 'cuda_stream', -1)) != token:
                self.binding.close(handle)
                self._handle = None
                raise GreenContextError('torch external stream does not carry the helper stream token')
        self.state = 'open'
        return self

    def membership(self, rounds=3, cycles=100000):
        """Measure SM membership of both streams with the probe kernel."""
        self._require_open()
        torch = self.torch
        sets, evidence = [], []
        for stream, wanted in ((self.decode_stream, self.decode_sms),
                               (self.prefill_stream, self.prefill_sms)):
            seen, per_round = set(), []
            for _ in range(rounds):
                out = torch.full((self.total_sms * 32,), -1, dtype=torch.int32,
                                 device=torch.device('cuda', self.device_index))
                torch.cuda.synchronize(self.device_index)
                rc = self.binding.probe(out.data_ptr(), out.numel(), stream.cuda_stream, cycles)
                if rc:
                    raise GreenContextError(f'SM probe launch failed: status={rc}')
                stream.synchronize()
                sample = set(int(v) for v in out.cpu().tolist())
                if -1 in sample:
                    raise GreenContextError('SM probe left unwritten output')
                seen |= sample
                per_round.append(len(sample))
                if len(seen) >= wanted:
                    break
            sets.append(seen)
            evidence.append(per_round)
        ok = (len(sets[0]) == self.decode_sms and len(sets[1]) == self.prefill_sms
              and not (sets[0] & sets[1]) and len(sets[0] | sets[1]) == self.total_sms)
        self.membership_result = MembershipResult(sets[0], sets[1], self.total_sms, ok, evidence)
        return self.membership_result

    def close(self):
        if self.state != 'open':
            raise GreenContextError(f'green split cannot be closed from state {self.state!r}')
        rc = self.binding.close(self._handle)
        if rc:
            self.state = 'quarantined'
            raise GreenContextError(f'green split close failed: CUresult={rc}')
        self._handle = None
        self.decode_stream = self.prefill_stream = None
        self.state = 'closed'
        return True

    # -- identity ---------------------------------------------------------
    def _require_open(self):
        if self.state != 'open':
            raise GreenContextError('green split is not open')

    def describe(self):
        """Counts and identities only; no device addresses in records."""
        self._require_open()
        record = {'requested_decode_sms': self.requested_decode_sms,
                  'decode_sms': self.decode_sms, 'prefill_sms': self.prefill_sms,
                  'total_sms': self.total_sms, 'basis': 'driver-reported'}
        if self.membership_result is not None:
            record['membership'] = self.membership_result.as_record()
            if self.membership_result.ok:
                record['basis'] = 'measured'
        identity = getattr(self.binding, 'identity', None)
        if callable(identity):
            record['helper'] = identity()
        return record


def declared_decode_sms(env=None):
    """The declared fixed split; a missing declaration fails closed."""
    env = os.environ if env is None else env
    value = env.get('FLASH_S5_DECODE_SMS')
    if value is None:
        raise GreenContextError('FLASH_S5_DECODE_SMS is not declared')
    try:
        count = int(value)
    except ValueError as exc:
        raise GreenContextError('FLASH_S5_DECODE_SMS is not an integer') from exc
    if count < 1:
        raise GreenContextError('FLASH_S5_DECODE_SMS must be positive')
    return count


def open_split(device_index, *, env=None, torch_module=None, binding=None):
    """Create the declared split from the environment; used by worker startup."""
    env = os.environ if env is None else env
    if binding is None:
        library = env.get('FLASH_S5_GREEN_LIB')
        if not library:
            raise GreenContextError('FLASH_S5_GREEN_LIB is not declared')
        binding = DriverBinding(library)
    split = GreenSplit(binding, device_index, declared_decode_sms(env), torch_module=torch_module)
    return split.create()


__all__ = ['ABI_VERSION', 'DriverBinding', 'GreenContextError', 'GreenSplit',
           'MembershipResult', 'declared_decode_sms', 'helper_source_sha256', 'open_split']
