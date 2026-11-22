"""Drained rung-2 byte invariants, NOT advancing model-state exactness.

CPU snapshots are bounded and private. Shared physical prefix pages are deduped.
Comparison uses actual byte equality, never digest equality. Dynamic KV suffix,
GDN/conv/PLE state, graph input/output contents and allocator scratch are excluded.
Graph *definitions* (nodes, dependencies, captured scalar/pointer arguments) are
immutable; pointers' advancing pointees are not mislabelled immutable bank data.
"""
import ctypes

MAX_SNAPSHOT_BYTES = 2 * 1024**3  # per worker; exact required bytes checked first
MAX_GRAPH_BYTES = 16 * 1024**2
CHUNK_BYTES = 1024**2


class InvariantUnavailable(RuntimeError):
    pass


def require(ok, reason):
    if not ok: raise InvariantUnavailable(reason)


class GraphBytes:
    def __init__(self, library):
        # Reuse the already hash-verified, worker-owned observer binding.
        self.fn = library.s5_graph_bytes
        self.fn.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t,
                            ctypes.POINTER(ctypes.c_size_t)]
        self.fn.restype = ctypes.c_int

    def __call__(self, graph):
        raw = graph.raw_cuda_graph()
        require(type(raw) is int and raw > 0, 'raw-graph-unavailable')
        size = ctypes.c_size_t()
        require(self.fn(raw, None, 0, ctypes.byref(size)) == 0
                and 0 < size.value <= MAX_GRAPH_BYTES, 'graph-definition-unavailable')
        buffer = ctypes.create_string_buffer(size.value)
        require(self.fn(raw, buffer, size.value, ctypes.byref(size)) == 0
                and size.value == len(buffer), 'graph-definition-read-incomplete')
        return buffer.raw


class PrefixSources:
    """Discover authoritative QSA CPU history and freeze only prompt rows."""
    def __init__(self, runner):
        import torch
        self.torch, self.runner = torch, runner
        states = runner.req_states
        mapping = states.req_id_to_index
        require(type(mapping) is dict and len(mapping) == 8, 'prefix-cohort-incomplete')
        self.mapping = dict(mapping)  # private, never serialized
        self.prompts = {slot:int(states.prompt_len.np[slot]) for slot in mapping.values()}
        require(all(0 < n <= 262144 for n in self.prompts.values()), 'prefix-length-invalid')
        self.tables, self.layers, self.coverage = {}, {}, {}
        ctx = runner.compilation_config.static_forward_context
        for name, layer in ctx.items():
            host = getattr(layer, '_qsa_host_kv', None)
            if host is None: continue
            groups = [i for i,g in enumerate(runner.kv_cache_config.kv_cache_groups)
                      if name in g.layer_names]
            require(len(groups) == 1, 'prefix-layer-group-ambiguous')
            group = groups[0]
            require(host.device.type == 'cpu' and host.dtype == torch.bfloat16
                    and tuple(host.shape[1:]) == (1,4048,512) and host.is_contiguous(),
                    'prefix-host-layout-unsupported')
            require(runner.block_tables.block_sizes[group] == 4048
                    and runner.block_tables.blocks_per_kv_block[group] == 1,
                    'prefix-page-layout-unsupported')
            if group not in self.tables:
                rows = {}
                for slot,n in self.prompts.items():
                    count=(n+4047)//4048
                    require(int(runner.block_tables.num_blocks.np[group,slot]) >= count,
                            'prefix-table-incomplete')
                    row=runner.block_tables.block_tables[group].gpu[slot,:count].cpu().tolist()
                    require(len(row)==count and all(type(b) is int and 0 < b < host.shape[0] for b in row),
                            'prefix-table-null-or-invalid')
                    rows[slot]=row
                self.tables[group]=rows
            coverage={}
            for slot,row in self.tables[group].items():
                for j,physical in enumerate(row):
                    coverage[physical]=max(coverage.get(physical,0),min(4048,self.prompts[slot]-j*4048))
            self.layers[name]=(layer,host,group)
            self.coverage[name]=coverage
        require(len(self.layers)==6, 'prefix-six-local-histories-required')
        pointers=[host.data_ptr() for _,host,_ in self.layers.values()]
        require(len(set(pointers))==6, 'prefix-layer-alias')
        self.total_bytes=sum(sum(c.values())*1024 for c in self.coverage.values())

    def identity(self):
        r=self.runner
        require(r.req_states.req_id_to_index==self.mapping, 'prefix-cohort-changed')
        require(all(int(r.req_states.prompt_len.np[s])==n for s,n in self.prompts.items()),
                'prefix-length-changed')
        for group,rows in self.tables.items():
            for slot,row in rows.items():
                actual=r.block_tables.block_tables[group].gpu[slot,:len(row)].cpu().tolist()
                require(actual==row, 'prefix-mapping-changed')
        for layer,host,_ in self.layers.values():
            require(layer._qsa_host_kv is host, 'prefix-allocation-changed')

    def chunks(self):
        self.identity()
        for name in sorted(self.layers):
            host=self.layers[name][1]
            for page,n in sorted(self.coverage[name].items()):
                raw=host[page,0,:n,:].view(self.torch.uint8).reshape(-1)
                for start in range(0,raw.numel(),CHUNK_BYTES):
                    # numpy exposes an exact CPU view; tobytes is one bounded copy.
                    yield raw[start:start+CHUNK_BYTES].numpy().tobytes()

    def metadata(self):
        return dict(local_layers=6,cohort_size=8,
                    unique_layer_pages=sum(len(c) for c in self.coverage.values()),
                    history_bytes=self.total_bytes)


class Snapshot:
    def __init__(self, probe, sources=None, graph_bytes=None):
        self.probe=probe;self.failure=None;self.checks=[]
        self.sources=sources or PrefixSources(probe.runner)
        self.graphs=tuple(probe.bank.manager.graphs.items())
        require(self.graphs and len(self.graphs)<=64, 'split-bank-unavailable')
        if graph_bytes is None:
            owner=probe.runner._flash_s5_replay_observer
            graph_bytes=GraphBytes(owner.session.binding._lib)
        self.graph_bytes=graph_bytes
        self.cap=getattr(probe,'_snapshot_max_bytes',MAX_SNAPSHOT_BYTES)
        require(type(self.cap) is int and 64*1024**2<=self.cap<=MAX_SNAPSHOT_BYTES,'invalid-snapshot-budget')
        # Charge actual graph bytes before history allocation. One bounded
        # native/ctypes serialization is transient; retained bytes never exceed
        # the run cap, even if the graph inventory grows.
        used=self.sources.total_bytes
        require(used<=self.cap,'byte-snapshot-budget-exceeded')
        self.bank_bytes=[]
        for _,graph in self.graphs:
            data=graph_bytes(graph)
            require(type(data) is bytes and 0<len(data)<=MAX_GRAPH_BYTES,'graph-definition-size-invalid')
            used+=len(data)
            require(used<=self.cap,'byte-snapshot-budget-exceeded')
            self.bank_bytes.append(data)
        self.history=list(self.sources.chunks())
        require(sum(map(len,self.history))==self.sources.total_bytes,'prefix-snapshot-incomplete')
        self.storage_epoch=probe.storage_epoch
        self.captured_generation=probe.route_generation

    def check(self, phase):
        require(phase in ('entered','measured','returned'), 'invariant-phase-invalid')
        expected={'entered':[], 'measured':['entered'], 'returned':['entered','measured']}[phase]
        require(self.checks==expected, 'invariant-phase-order')
        if self.failure is None:
            try:
                require(self.probe.storage_epoch==self.storage_epoch,'invariant-storage-changed')
                current=tuple(self.probe.bank.manager.graphs.items())
                require(len(current)==len(self.graphs) and all(a==b and g is h
                    for (a,g),(b,h) in zip(current,self.graphs)), 'split-bank-replaced')
                for (_,g),prior in zip(self.graphs,self.bank_bytes):
                    require(self.graph_bytes(g)==prior, 'split-bank-byte-mismatch')
                count=0
                for actual in self.sources.chunks():
                    require(count<len(self.history) and actual==self.history[count], 'prefix-byte-mismatch')
                    count+=1
                require(count==len(self.history),'prefix-snapshot-coverage-changed')
            except InvariantUnavailable as exc:
                self.failure=str(exc)
        self.checks.append(phase)
        result=dict(schema=1,verdict='FAIL' if self.failure else 'PASS',failure=self.failure,
                    phase=phase,checks=list(self.checks),storage_epoch=self.storage_epoch,
                    captured_generation=self.captured_generation,
                    snapshot_bytes=sum(map(len,self.history))+sum(map(len,self.bank_bytes)),
                    graph_count=len(self.graphs),snapshot_cap_bytes=self.cap,**self.sources.metadata(),
                    scope='prompt-prefix QSA KV/retained CPU history and captured split graph definitions; advancing state and pointees excluded')
        if phase=='returned':self.history=[];self.bank_bytes=[]
        return result
