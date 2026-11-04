"""Shared, device-generic cases for the attn-meta overlay (patches/s5_engine/attn_meta_batch.py).

Used by harness/test_attn_meta_overlay.py (pinned image, CPU) and tools/attn_meta_vm_probe.py (pinned image, CPU on
the hub self-run, CUDA on VM <vm-id>): the same fixture, the same comparison, the same native-clone check.

Fixture: tools/attn_meta_witness.py's Fixture (real VllmConfig from the real model config, the box's per-stage
39-group layout, real AttentionGroup / builders, real InputBatch, real Qwen4ExpModelState.prepare_attn); its leaves
are declared there (LEAVES). The overlay is installed exactly as the engine would: ``attn_meta_batch.install`` on a
runner object whose ``model_state`` is the fixture's real model state.

"Equal" (stock vs overlay, two independently built fixtures, every builder tensor poisoned with the same sentinel
first): same dict keys in the same order; same metadata class per layer; every scalar field equal; every tensor
field equal in dtype / shape / values and in location (the same builder persistent buffer, field name + storage
offset, or the same input tensor; never a fresh tensor where stock used a buffer); the same number of distinct
metadata objects; every tensor of every builder byte-equal after the call.
"""
import dataclasses
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
SENTINEL = -7
B = 4048
CAPTURE_SIZES = (1, 2, 3, 4, 6, 8, 9, 12, 15, 18, 21, 24)
HISTS = {
    'h65k': [65000 + 997 * i for i in range(8)],
    'h110k': [110000 + 997 * i for i in range(8)],
    'boundary': [B - 2, B - 1, B, B + 1, 2 * B - 1, 2 * B, 16 * B, 27 * B + 1],
    'one': [70001],
    'ten': [60000 + 4049 * i for i in range(10)],
}
PADDED = ((3, 4), (7, 8), (1, 2), (9, 10))

LEAVES = [
    dict(leaf='runner', real='GPUModelRunner (V2) whose model_state is the real Qwen4ExpModelState',
         fixture='types.SimpleNamespace(model_state=<the witness fixture\'s real Qwen4ExpModelState>)',
         why='install() reads only runner.model_state; everything below is the real model state and builders'),
    dict(leaf='FULL padding', real='V2 prepare_inputs pads seq_lens with 0, query_start_loc(_np) with num_tokens to '
         'R+1, seq_lens_cpu_upper_bound with 0, tokens to R (uniform decode); block tables gathered to R rows',
         fixture='pad_batch() writes the same padded values into the InputBatch fields prepare_attn reads',
         why='prepare_attn reads exactly these fields (mamba_hybrid.py:232ff); num_scheduled_tokens and '
         'is_prefilling_np stay unpadded as in V2'),
    dict(leaf='native lane clones', real='lane_chunk.private_attention_groups over the stock attention groups at '
         'LaneChunkFactory construction', fixture='the same function, real code, over the fixture\'s stock groups',
         why='the clone is the real code path; only its caller (LaneChunkFactory) is not constructed'),
]


def paths():
    for p in (ROOT / 'tools', ROOT / 'patches/s5_engine'):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


class Cases:
    """Fixture factory and checks over one device."""

    def __init__(self, W, cfg, text, torch, np, device):
        self.W, self.cfg, self.text, self.torch, self.np, self.device = W, cfg, text, torch, np, device

    # -- fixtures --------------------------------------------------------------------------------------------
    def fixture(self, stage, hist, pad_to=None):
        fx = self.W.Fixture(stage, self.device, self.cfg, self.text, self.torch, self.np, hist, {})
        if pad_to is not None and pad_to > len(hist):
            self.pad_batch(fx, len(hist), pad_to)
        return fx

    def pad_batch(self, fx, n, R):
        torch, np, dev = self.torch, self.np, self.device
        b = fx.batch
        seq = torch.zeros(R, dtype=torch.int32, device=dev)
        seq[:n] = b.seq_lens[:n]
        qsl = torch.full((R + 1,), n, dtype=torch.int32, device=dev)
        qsl[:n + 1] = b.query_start_loc[:n + 1]
        ub = torch.zeros(R, dtype=torch.int32)
        ub[:n] = b.seq_lens_cpu_upper_bound[:n]
        qsl_np = np.full(R + 1, n, dtype=np.int32)
        qsl_np[:n + 1] = b.query_start_loc_np[:n + 1]
        fx.batch = dataclasses.replace(b, num_reqs_after_padding=R, num_tokens_after_padding=R,
                                       seq_lens=seq, query_start_loc=qsl, query_start_loc_np=qsl_np,
                                       seq_lens_cpu_upper_bound=ub)
        fx.tables, fx.slots = self.W.make_tables(fx.kv_cfg, self.cfg, R, torch, dev)

    @staticmethod
    def builders(fx, attn=None):
        for gid, row in enumerate(fx.attn if attn is None else attn):
            for g in row:
                for b in g.metadata_builders:
                    yield gid, b

    def poison(self, fx, attn=None):
        torch = self.torch
        for _, b in self.builders(fx, attn):
            for v in vars(b).values():
                if isinstance(v, torch.Tensor) and v.numel():
                    v.fill_(True if v.dtype == torch.bool else SENTINEL)

    def tensors(self, fx, attn=None):
        torch = self.torch
        for gid, b in self.builders(fx, attn):
            for k, v in sorted(vars(b).items()):
                if isinstance(v, torch.Tensor):
                    yield gid, type(b).__name__, k, v

    def buffers(self, fx, attn=None):
        """Every builder tensor's bytes (what a captured FULL graph can read)."""
        torch = self.torch
        rows = []
        for gid, cls, k, v in self.tensors(fx, attn):
            data = v.detach().contiguous().view(-1)
            raw = data.view(torch.uint8) if data.numel() and data.dtype != torch.bool else data.to(torch.uint8)
            rows.append((gid, cls, k, str(v.dtype), tuple(v.shape), raw.cpu().numpy().tobytes()))
        return rows

    def gdn_buffers(self, fx):
        rows = []
        for gid, cls, k, v in self.tensors(fx):
            if cls == 'GDNAttentionMetadataBuilder' and k in ('non_spec_state_indices_tensor',
                                                               'non_spec_query_start_loc'):
                rows.append((gid, k, v.cpu().numpy().tobytes()))
        return rows

    def locate(self, fx, t):
        torch = self.torch
        ptr = t.untyped_storage().data_ptr()
        for gid, cls, k, v in self.tensors(fx):
            if v.numel() and v.untyped_storage().data_ptr() == ptr:
                return ('builder', gid, cls, k, t.storage_offset())
        named = dict(seq_lens=fx.batch.seq_lens, query_start_loc=fx.batch.query_start_loc,
                     positions=fx.batch.positions, slot_mappings=fx.slots)
        for i, bt in enumerate(fx.tables):
            named[f'block_table[{i}]'] = bt
        for k, v in named.items():
            if isinstance(v, torch.Tensor) and v.numel() and v.untyped_storage().data_ptr() == ptr:
                return ('input', k, t.storage_offset())
        return ('fresh',)

    def describe(self, fx, md):
        torch = self.torch
        out = {}
        for name, v in sorted(vars(md).items()):
            if isinstance(v, torch.Tensor):
                out[name] = ('tensor', str(v.dtype), tuple(v.shape), v.cpu().tolist(), self.locate(fx, v))
            elif isinstance(v, dict):
                out[name] = ('dict', repr(sorted(v.items())))
            else:
                out[name] = ('value', repr(v))
        return type(md).__name__, out

    def snapshot(self, fx, md):
        return dict(keys=list(md), objects=len({id(v) for v in md.values()}),
                    meta={k: self.describe(fx, v) for k, v in md.items()}, buffers=self.buffers(fx))

    # -- the two sides ---------------------------------------------------------------------------------------
    @staticmethod
    def install(mod, fx, env=None):
        runner = types.SimpleNamespace(model_state=fx.mstate)
        return mod.install(runner, env={mod.ENV: '1'} if env is None else env), runner

    def call(self, fx, **kw):
        return fx.mstate.prepare_attn(fx.batch, fx.mode, fx.tables, fx.slots, fx.attn, fx.kv_cfg,
                                      for_capture=kw.get('for_capture', False))

    def compare(self, mod, stage, hist, pad_to=None, mode=None, mutate=None, env=None, poison=True):
        a, b = self.fixture(stage, hist, pad_to), self.fixture(stage, hist, pad_to)
        for fx in (a, b):
            if mode is not None:
                fx.mode = mode
            if mutate is not None:
                mutate(fx)
        batch, _ = self.install(mod, b, env)
        for fx in (a, b):
            if poison:
                self.poison(fx)
        sa = self.snapshot(a, self.call(a))
        sb = self.snapshot(b, self.call(b))
        return sa, sb, batch

    @staticmethod
    def first_difference(sa, sb):
        if sa['keys'] != sb['keys']:
            return 'keys/order'
        if sa['objects'] != sb['objects']:
            return 'objects'
        if sa['meta'] != sb['meta']:
            return 'metadata'
        if sa['buffers'] != sb['buffers']:
            return 'buffers'
        return None

    # -- native lane clones ----------------------------------------------------------------------------------
    def native_clones(self, fx):
        import lane_chunk
        groups, nbytes, ntensors, _ = lane_chunk.private_attention_groups(fx.attn, torch_module=self.torch)
        return groups, nbytes, ntensors

    def shared_storage(self, fx, clones):
        """(stock field, clone field) pairs whose tensors share a storage: must be empty."""
        stock = {}
        for gid, cls, k, v in self.tensors(fx):
            if v.numel():
                stock[v.untyped_storage().data_ptr()] = (gid, cls, k)
        out = []
        for gid, cls, k, v in self.tensors(fx, clones):
            if v.numel() and v.untyped_storage().data_ptr() in stock:
                out.append((stock[v.untyped_storage().data_ptr()], (gid, cls, k)))
        return out


# -- dynamic decode sequence: block crossings, admissions, finishes ----------------------------------------------
@dataclasses.dataclass
class Req:
    rid: int
    seq: int          # tokens already in the cache (history)
    blocks: list      # per-request block ids (one table row; the same row shape for every group)


def padded(n):
    for s in CAPTURE_SIZES:
        if s >= n:
            return s
    raise ValueError('no capture size for %d' % n)


def sequence_plan(steps=48):
    """A deterministic decode plan: starts just below block boundaries so rows cross a 4048-token block within a
    few steps; admits a request at steps 6 and 20 and finishes (removes, compacting the rows) at 13, 27 and 34."""
    hist = [B - 3, 2 * B - 1, 5 * B - 6, 3 * B - 2, 9 * B - 9, 4 * B - 4, B - 1]
    admit = {6: 7 * B - 2, 20: B - 5}
    finish = {13: 2, 27: 0, 34: 3}
    return dict(hist=hist, admit=admit, finish=finish, steps=steps)


class Sequence:
    """Drives two fixtures (stock, overlay) through the same decode plan, step by step.

    Each step rebuilds the InputBatch and block tables from the live requests as V2 does for a FULL decode step:
    rows in admission order (finished rows removed, the rest compacted), padded to the next capture size;
    every request's table row grows by one block id when its history crosses a block boundary.
    """

    def __init__(self, cases, stage, plan):
        self.c, self.stage, self.plan = cases, stage, plan
        self.next_rid = 0
        self.next_block = 1
        self.reqs = [self.new_req(h) for h in plan['hist']]
        self.crossings = 0

    def new_req(self, hist):
        n_blocks = hist // B + 1
        r = Req(self.next_rid, hist, list(range(self.next_block, self.next_block + n_blocks)))
        self.next_rid += 1
        self.next_block += n_blocks + 3
        return r

    def advance(self, step):
        for r in self.reqs:
            before = r.seq // B
            r.seq += 1
            if r.seq // B != before:
                r.blocks.append(self.next_block)
                self.next_block += 1
                self.crossings += 1
        if step in self.plan['finish'] and self.plan['finish'][step] < len(self.reqs):
            del self.reqs[self.plan['finish'][step]]
        if step in self.plan['admit']:
            self.reqs.append(self.new_req(self.plan['admit'][step]))

    def load(self, fx):
        """Write the live requests into a fixture's batch and tables (fresh tensors, as one V2 step)."""
        c, torch, np = self.c, self.c.torch, self.c.np
        hist = [r.seq for r in self.reqs]
        n = len(hist)
        R = padded(n)
        base = c.W.make_batch(hist, torch, np, c.device)[0]
        fx.batch = base
        if R > n:
            c.pad_batch(fx, n, R)
        tables = []
        for gid, t in enumerate(c.W.make_tables(fx.kv_cfg, c.cfg, R, torch, c.device)[0]):
            t = torch.zeros_like(t)
            for i, r in enumerate(self.reqs):
                row = torch.tensor([b + 100000 * gid for b in r.blocks][:t.shape[1]], dtype=t.dtype)
                t[i, :row.numel()] = row.to(t.device)
            tables.append(t)
        fx.tables = tuple(tables)
        fx.slots = c.W.make_tables(fx.kv_cfg, c.cfg, R, torch, c.device)[1]
        return n, R


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=1, sort_keys=True, default=str) + '\n')
