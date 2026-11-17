"""Native descriptor slots never rotate the ordinary runner's UVA pools."""
from contextlib import contextmanager


class NativeUvaError(RuntimeError):
    pass


_UNRECORDED = object()


class _Pool:
    def __init__(self, stock, owner):
        self.owner = owner
        # Two B members may publish before their first consumer, plus the prior scalar view.
        self.pool = type(stock)(stock.size, stock.dtype, max(3, stock.max_concurrency))
        self.readers = [None] * self.pool.max_concurrency

    def __getattr__(self, name):
        return getattr(self.pool, name)

    def copy_to_uva(self, x):
        if not self.owner.active:
            raise NativeUvaError('native UVA publication outside its stream scope')
        slot = (self.pool._curr + 1) % self.pool.max_concurrency
        if (self, slot) in self.owner.used:
            raise NativeUvaError('native UVA slot reused inside an unfinished segment')
        reader = self.readers[slot]
        if reader is _UNRECORDED:
            raise NativeUvaError('native UVA slot has unrecorded readers')
        if reader is not None and not reader.query():
            reader.synchronize()  # Retire before the CPU overwrite OR pool growth.
            self.owner.waits += 1
        self.readers[slot] = _UNRECORDED
        self.owner.used.add((self, slot))
        return self.pool.copy_to_uva(x)

    def copy_to_gpu(self, x, out=None):
        uva = self.copy_to_uva(x)
        return uva.clone() if out is None else out.copy_(uva, non_blocking=True)


class NativeUvaSlots:
    """Scoped pool bindings and scalar views; GPU destinations stay shared/row-owned.

    Worker model commands run serially on the host. Only the native stream uses
    these bindings; ordinary calls and captured graphs retain their stock slots.
    """
    def __init__(self, runner, torch_module, buffers=None):
        if getattr(runner, '_flash_s5_native_uva', None) is not None:
            raise NativeUvaError('runner already owns native UVA slots')
        if buffers is None:
            from vllm.v1.worker.gpu import buffer_utils as buffers
        self.torch = torch_module
        self.active = False
        self.failed = False
        self.used = set()
        self.waits = 0
        self.events = 0
        self.bindings = []
        self.scalars = []
        self.pools = {}
        seen = set()

        def pool(owner, name, path):
            stock = getattr(owner, name)
            if stock is None:
                return
            if type(stock) is not buffers.UvaBufferPool:
                raise NativeUvaError('unclassified UVA pool: ' + path)
            private = self.pools.get(id(stock))
            if private is None:
                private = self.pools[id(stock)] = _Pool(stock, self)
            self.bindings.append((owner, name, stock, private, path))

        def visit(value, path):
            if value is None or id(value) in seen:
                return
            seen.add(id(value))
            if type(value) is buffers.StagedWriteTensor:
                for name in ('write_indices', 'write_starts', 'write_cu_lens', 'write_contents'):
                    pool(value, name, path + '.' + name)
            elif type(value) is buffers.FusedStagedWriter:
                for name in ('group_ids', 'indices', 'starts', 'cu_lens'):
                    pool(value, name, path + '.' + name)
            elif type(value) is buffers.UvaBackedTensor:
                pool(value, 'pool', path + '.pool')
                private = self.bindings[-1][3]
                if any(entry[2] is private for entry in self.scalars):
                    raise NativeUvaError('aliased native scalar UVA pool: ' + path)
                # A startup CPU snapshot, before any GPU consumer exists.
                gpu = private.pool.copy_to_uva(value.np)
                self.scalars.append([value, gpu, private, path])
            elif isinstance(value, (tuple, list)):
                for i, item in enumerate(value):
                    visit(item, path + '.' + str(i))
            elif hasattr(value, '__dict__'):
                # Only pinned state containers, never model weights/config/tensors.
                module = type(value).__module__
                if not module.startswith('vllm.v1.worker.gpu.'):
                    raise NativeUvaError('unclassified UVA state container: ' + path)
                for name, item in vars(value).items():
                    if isinstance(item, (buffers.StagedWriteTensor, buffers.FusedStagedWriter,
                                         buffers.UvaBackedTensor)) or name in ('block_tables', 'fused_writer'):
                        visit(item, path + '.' + name)

        visit(runner.req_states, 'req_states')
        visit(runner.block_tables, 'block_tables')
        for name in ('rope_state', 'prompt_embeds_state'):
            visit(getattr(runner.model_state, name, None), 'model_state.' + name)
        if not self.bindings:
            raise NativeUvaError('native runner has no classified UVA pools')
        runner._flash_s5_native_uva = self

    def require_active(self):
        if not self.active or self.failed:
            raise NativeUvaError('native publication requires its owning UVA scope')

    @contextmanager
    def scope(self):
        if self.active or self.failed:
            raise NativeUvaError('native UVA scope is nested or has failed')
        for owner, name, stock, private, path in self.bindings:
            if getattr(owner, name) is not stock:
                raise NativeUvaError('stock UVA binding changed: ' + path)
        self.active = True
        self.used.clear()
        saved = [(entry, entry[0].gpu) for entry in self.scalars]
        try:
            for owner, name, stock, private, path in self.bindings:
                setattr(owner, name, private)
            for entry, stock_gpu in saved:
                owner, gpu, private, path = entry
                owner.gpu = gpu
                self.used.add((private, private.pool._curr))
            yield
        finally:
            # Include scalar views still consumed by later native segments.
            for entry, stock_gpu in saved:
                owner, gpu, private, path = entry
                entry[1] = owner.gpu
                self.used.add((private, private.pool._curr))
                owner.gpu = stock_gpu
            for owner, name, stock, private, path in self.bindings:
                setattr(owner, name, stock)
            try:
                if self.used:
                    event = self.torch.cuda.Event(enable_timing=False)
                    event.record()  # _scope still holds the owning prefill stream.
                    self.events += 1
                    for private, slot in self.used:
                        private.readers[slot] = event
            except BaseException:
                self.failed = True  # Retain every allocation; never reuse an unproved slot.
                raise
            finally:
                self.active = False

    def inventory(self):
        host_bytes = sum(int(buf.cpu.numel() * buf.cpu.element_size())
                         for private in self.pools.values() for buf in private.pool._uva_bufs)
        return dict(pools=len(self.pools), host_bytes=host_bytes, gpu_bytes=0,
                    retire_waits=self.waits, events_recorded=self.events, failed=self.failed,
                    slot_depths=[p.pool.max_concurrency for p in self.pools.values()],
                    fields=[path for _, _, _, _, path in self.bindings])
