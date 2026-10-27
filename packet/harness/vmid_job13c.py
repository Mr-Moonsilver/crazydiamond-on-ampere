#!/usr/bin/env python3
"""DRAFT, NOT RUN, NOT AUTHORIZED: native prefill positive accounting, VM only.

One shared correlation window carries two declared partitions on separate
green contexts: the stock runner/sampler decode offers on the decode green
stream and the source-declared prefill offers on the prefill green stream and
its owned prefill-green auxiliaries.  A real retained graph is replayed inside
the decode partition; real pinned ``SharedExperts``/PLE owner methods run with
synthetic BF16 payloads inside their source-pinned prefill offers; a copy-only
role does one history D2D copy on an ordinary primary-context stream.  The
whole ``s5ro-correlation/2`` report is retained and the paired composer proves
the exact-id union, independent regrades of both partitions and raw-loss
retention.  A second window puts a prefill offer's kernel on the ordinary
primary-context stream and must FAIL on context and stream.

Stream identities are queried through CUPTI *before* the window opens; the
declared provider inside the window is an exact handle lookup, because the
pinned context probe uses ``cuCtxFromGreenCtx``/``cuGreenCtxGetId``, which the
reviewed ``api_kind`` vocabulary does not classify as host control.

The draft cannot establish native readiness, engine call sites or SM
containment; no engine, model, PP stage, box or GPU action is taken here.
"""
import argparse
import copy
import ctypes as C
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace as NS
import vmid_support as s
from vmid_job2 import GraphBinding
from vmid_job6 import HostQueries

SCHEMA = 'vm<vm-id>-job13c/1'
RUN_ID = 'vm-job13c'
WINDOW_ID = 'native-overlap'
EPOCH = 1
GENERATION = 1
STORAGE_EPOCH = 1
ROUTE = 1
DECODE_SMS = '44'
ENGINE_ENV = 'FLASH_S5_ENGINE_OFFERS'
PREFILL_ENV = 'FLASH_S5_NATIVE_PREFILL_OFFERS'
# The retained graph probe: 1386 kernel nodes, 26 copy nodes, 7 memset nodes.
MIX = dict(kernel=1386, memcpy=26, memset=7)
MODULES = ('native_prefill_offers.py', 'native_placement.py')
PINNED_MODULES = ('activity_accounting.py', 'capture_streams.py',
                  'e3_types.py', 'engine_offers.py', 'green_context.py',
                  'model_hooks.py', 'replay_observer.py')
CANNOT_EXECUTE = (
    'TP>1 sharded prefill: one GPU drives TP1 only; sharded prefill needs '
    'peer ranks and collective transports',
    'PP>1 prefill, pp_receive and pp_send: one GPU has no pipeline peer, so '
    'the PP channel is never exercised',
    'full Qwen4ExpModel prefill: no engine, weights or scheduler; only the '
    'pinned owner methods run with synthetic payloads',
    'real overlap performance: the measured work is synthetic and bounded',
)
SCOPE = ('real correlation window, graph replay and eager submissions on '
         'distinct green contexts; real pinned SharedExperts/PLE owner '
         'methods with synthetic BF16 payloads; no engine, model, PP or '
         'full-model prefill qualification')


class Identities:
    """Pre-window CUPTI queries; inside the window, exact handle lookups."""

    def __init__(self, host):
        self.host = host
        self.by_handle = {}
        self.rows = {}

    def prime(self, stream, label):
        handle = int(getattr(stream, 'cuda_stream', 0))
        if not handle or handle in self.by_handle or label in self.rows:
            raise RuntimeError('identity_prime:' + label)
        raw = self.host.stream(stream)
        green = bool(raw.get('green_present'))
        context = (raw.get('green_cupti_context_id') if green
                   else raw.get('primary_cupti_context_id'))
        stream_id = raw.get('cupti_stream_id')
        if type(context) is not int or context < 1:
            raise RuntimeError('context_query:' + label)
        if type(stream_id) is not int or stream_id < 1:
            raise RuntimeError('stream_query:' + label)
        row = {'context_id': context, 'stream_id': stream_id, 'green': green,
               'label': label}
        self.by_handle[handle] = row
        self.rows[label] = row
        return dict(row)

    def __call__(self, stream):
        handle = int(getattr(stream, 'cuda_stream', 0))
        row = self.by_handle.get(handle)
        if row is None:
            raise RuntimeError('stream_identity_not_primed')
        return {'context_id': row['context_id'], 'stream_id': row['stream_id']}


def _pins(modules):
    """Verify the bundled module bytes against the immutable build closure."""
    pins = {name: s.sha(modules / name) for name in MODULES + PINNED_MODULES}
    closure = json.loads((modules.parent / 'e3_build_closure.json').read_text())
    sources = closure.get('builder_sources')
    if not isinstance(sources, dict):
        raise RuntimeError('closure_sources')
    for name, digest in pins.items():
        if sources.get(name) != digest:
            raise RuntimeError('module_pin:' + name)
    return pins


def _owner_run(shared, ple, x, ids, ple_out):
    """One reference owner run on the stock side streams; no offers."""
    held = []
    for owner in shared:
        if owner.maybe_forward_async(x) is not True:
            raise RuntimeError('shared_path_changed')
        owner.wait()
        held.append(owner.output)
    ple.start_prefetch(x, ids)
    ple._finalize_prefetch(ple._prefetch_buffer, ple_out)
    return held


def fixture(torch, types):
    """Real pinned SharedExperts/PLE instances with small synthetic operands."""
    shared_cls, ple_cls = types

    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.up = torch.full((1280, 2560), .001, device='cuda',
                                 dtype=torch.bfloat16)
            self.down = torch.full((2560, 640), .001, device='cuda',
                                   dtype=torch.bfloat16)
            self.gate = torch.full((1, 2560), .001, device='cuda',
                                   dtype=torch.bfloat16)

        def forward(self, value):
            import torch.nn.functional as F
            z = F.linear(value, self.up)
            first, second = z.chunk(2, dim=-1)
            return (F.linear(F.silu(first) * second, self.down)
                    * torch.sigmoid(F.linear(value, self.gate)))

    cfg = NS(moe_parallel_config=NS(
        enable_eplb=False, all2all_backend='allgather_reducescatter',
        use_fi_nvl_two_sided_kernels=False))
    mlp = MLP()
    shared = [shared_cls(mlp, cfg, False, lambda: False) for _ in range(2)]
    x = torch.full((4, 2560), .125, device='cuda', dtype=torch.bfloat16)
    if not all(int(o._determine_shared_experts_order(x)) == 3 for o in shared):
        raise RuntimeError('shared_overlap_branch_unavailable')
    # Bounded PLE constructor state; the real lookup/fork/join methods remain.
    ple = ple_cls.__new__(ple_cls)
    torch.nn.Module.__init__(ple)
    weight = torch.full((32, 160), .25, dtype=torch.bfloat16, pin_memory=True)
    ple.weight = torch.nn.Parameter(weight, requires_grad=False)
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
    ple._uva_weight = get_accelerator_view_from_cpu_tensor(weight)
    ple.embedding_dim = 160
    ple._block_d = 256
    ple.tp_size = 1
    ple.etp_data_parallel_size = 1
    ple.data_parallel_rank = 0
    ple.shard_indices = NS(org_vocab_start_index=0, org_vocab_end_index=32)
    ple._prefetch_stream = torch.cuda.Stream()
    ple._prefetch_buffer = torch.empty((4, 2, 160), device='cuda',
                                       dtype=torch.bfloat16)
    ple._output_dim = 320
    ids = torch.tensor([[0, 1], [2, 3], [4, 5], [30, 31]], device='cuda',
                       dtype=torch.int64)
    ple_out = torch.empty((4, 320), device='cuda', dtype=torch.bfloat16)
    # At ETP-DP=1 the id gather is an identity; a kernel there would run on the
    # caller (prefill main) inside the ple_aux offer and must not be claimed.
    gathered = ple._gather_dp_ids(ids, 4)
    if gathered is not ids:
        raise RuntimeError('ple_gather_not_identity')
    return shared, ple, x, ids, ple_out


def discover(torch, cs, mh, split, shared, ple):
    """Real pinned type/census discovery; the model container is never run."""
    from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpModel
    model = Qwen4ExpModel.__new__(Qwen4ExpModel)
    object.__setattr__(model, 'start_layer', 0)
    object.__setattr__(model, 'end_layer', 2)
    object.__setattr__(model, 'config', NS(ple_layer_ids=[1]))
    runner = NS(model=NS(modules=lambda: [model, shared[0], shared[1], ple]))
    owners = mh.PrefillStreamOwners.discover(runner, split,
                                             pinned=cs.pinned_types())
    got = [(id(obj), attr) for obj, attr in owners.owners]
    want = ([(id(obj), '_stream') for obj in shared]
            + [(id(ple), '_prefetch_stream')])
    if got != want:
        raise RuntimeError('prefill_owner_discovery')
    return owners


def _kind_count(families, family, suboperation, kind):
    row = (families or {}).get(family, {}).get(suboperation) or {}
    return int(row.get(kind, 0) or 0)


def _activity_count(families, family, suboperation):
    row = (families or {}).get(family, {}).get(suboperation) or {}
    return sum(int(row.get(kind, 0) or 0) for kind in ('kernel', 'memcpy',
                                                       'memset'))


def case_grade(record):
    f = list(record.get('failures', []))
    m = list(record.get('missing', []))
    required = ['source_pins_verified', 'graph_bytes_stable', 'cleanup_ok',
                'contexts_disjoint', 'raw_unchanged', 'graph_output_exact']
    if record.get('mode') == 'positive':
        required.extend(('output_exact', 'owners_restored', 'sharing_preserved'))
    for key in required:
        if record.get(key) is False:
            f.append(key)
        elif record.get(key) is not True:
            m.append(key)
    mode = record.get('mode')
    if mode == 'positive':
        composed = record.get('composed')
        receipt = record.get('prefill_receipt') or {}
        if not isinstance(composed, dict):
            m.append('composition')
        else:
            if composed.get('verdict') == 'FAIL':
                f.extend('composed:' + str(reason)
                         for reason in composed.get('failures', []))
            elif composed.get('verdict') != 'PASS':
                m.append('composition_inconclusive')
            m.extend('composed_missing:' + str(reason)
                     for reason in composed.get('missing', []))
            conservation = composed.get('conservation') or {}
            if conservation.get('proved') is not True:
                f.append('conservation_not_proved')
            whole = composed.get('whole') or {}
            if whole.get('complete') is not True:
                m.append('whole_raw_incomplete')
            if any(whole.get(key) for key in ('errors', 'overflow',
                                              'open_scopes',
                                              'boundary_records')):
                m.append('whole_raw_loss')
            decode = (composed.get('partitions') or {}).get('decode') or {}
            prefill = (composed.get('partitions') or {}).get('prefill') or {}
            for label, row in (('decode', decode), ('prefill', prefill)):
                if row.get('verdict') != 'PASS':
                    m.append(label + '_partition')
                if not row.get('activity_records'):
                    m.append(label + '_empty')
                summary = composed.get(label) or {}
                m.extend(label + '_missing:' + str(reason)
                         for reason in summary.get('missing') or ())
            p_kinds = prefill.get('by_kind') or {}
            if not p_kinds.get('kernel'):
                m.append('prefill_kernel')
            if not p_kinds.get('memcpy'):
                m.append('prefill_copy')
            completion = prefill.get('completion') or {}
            submitted = completion.get('offers_submitted') or 0
            completed = completion.get('offers_completed') or 0
            if not submitted:
                m.append('prefill_completion_missing')
            elif submitted != completed:
                m.append('prefill_completion_unproven')
            families = prefill.get('families') or {}
            if not _kind_count(families, 'shared_experts',
                               'maybe_forward_async', 'kernel'):
                f.append('shared_experts_unjoined')
            if not _kind_count(families, 'ple', 'start_prefetch', 'kernel'):
                f.append('ple_prefetch_unjoined')
            if not _activity_count(families, 'ple', 'finalize_prefetch'):
                f.append('ple_finalize_unjoined')
            if not _kind_count(families, 'qsa', 'bucket_fill', 'memcpy'):
                f.append('bucket_fill_unjoined')
            for sub in ('selection_readback', 'sparse_bucket_fill'):
                if not _kind_count(families, 'qsa', sub, 'memcpy'):
                    f.append(sub + '_unjoined')
        if receipt.get('verdict') != 'PASS':
            m.append('prefill_receipt')
            if receipt.get('verdict') == 'FAIL':
                f.extend('prefill:' + str(reason)
                         for reason in receipt.get('failures') or ())
            m.extend('prefill_missing:' + str(reason)
                     for reason in receipt.get('missing') or ())
        raw = record.get('native_report')
        if not isinstance(raw, dict) or raw.get('schema') != 's5ro-correlation/2':
            m.append('whole_report_retained')
        elif isinstance(composed, dict):
            if (raw.get('window') != (composed.get('whole') or {}).get('window')
                    or raw.get('activity_records')
                    != (composed.get('whole') or {}).get('activity_records')):
                f.append('whole_report_rewritten')
        if record.get('copy_role_control') is not True:
            f.append('copy_role_control')
        if not record.get('shadow_pins'):
            m.append('source_shadow_pins')
        else:
            for row in record['shadow_pins']:
                digest = row.get('source_sha256')
                if not isinstance(digest, str) or len(digest) != 64:
                    m.append('source_shadow_pin')
                    break
    elif mode == 'ordinary':
        negative = record.get('negative') or {}
        if negative.get('verdict') != 'FAIL':
            f.append('negative_not_rejected')
        else:
            reasons = set(negative.get('failures') or ())
            for want in ('wrong_context', 'wrong_stream'):
                if want not in reasons:
                    f.append('negative_missing:' + want)
        if (negative.get('joined_records') or 0) < 1:
            m.append('negative_vacuous')
        if negative.get('raw_complete') is not True:
            m.append('negative_raw_incomplete')
    else:
        m.append('case_mode')
    return s.verdict(f, m)


def grade(raw):
    f = list(raw.get('failures', []))
    m = list(raw.get('missing', []))
    cases = {}
    if [row.get('mode') for row in raw.get('cases', [])] != ['positive',
                                                            'ordinary']:
        m.append('case_coverage')
    for row in raw.get('cases', []):
        name = row.get('mode', 'unknown')
        g = case_grade(row)
        cases[name] = g
        f.extend(name + ':' + reason for reason in g['failures'])
        m.extend(name + ':' + reason for reason in g['missing'])
    for name in ('job1', 'job2b_flat'):
        g = raw.get('regressions', {}).get(name, {})
        if g.get('verdict') == 'FAIL':
            f.append(name + ':regression')
        elif g.get('verdict') != 'PASS':
            m.append(name + ':regression_missing')
    if raw.get('build_verified') is not True:
        m.append('build_pin')
    if not isinstance(raw.get('cannot_execute'), list) or not raw['cannot_execute']:
        m.append('cannot_execute')
    out = s.verdict(f, m)
    out['cases'] = cases
    return out


def replay_owned_graph(binding, owner, graph_handle):
    """The helper replays its retained Graphs owner, not a raw cudaGraph_t."""
    if (type(owner) is not int or owner <= 0 or type(graph_handle) is not int
            or graph_handle <= 0 or owner == graph_handle):
        raise RuntimeError('graph_owner_identity_unproved')
    return binding.lib.vmg_replay(owner)


def progress(out, phase):
    # Append-only CPU breadcrumb before native calls, retained on process crash.
    with (out / 'progress.jsonl').open('a') as handle:
        handle.write(json.dumps({'phase': phase, 'wall_s': time.time()}) + '\n')
        handle.flush()


def case(torch, ro, gc, cs, mh, native, receipt, modules, out, mode):
    import activity_accounting as aa
    import e3_types
    import engine_offers as eo
    import native_placement as np_
    import native_prefill_offers as npo

    r = dict(mode=mode, failures=[], missing=[], source_pins={},
             shadow_pins=[], contexts={})
    progress(out, mode + ':begin')
    gl = None
    split = session = None
    graph_owner = 0
    owners_obj = None
    shared = ple = None
    shadows = []
    prefill = decode = None
    proof = None
    run_id = RUN_ID + '-' + mode
    try:
        r['source_pins'] = _pins(modules)
        r['source_pins_verified'] = True
        gl = GraphBinding(native / 'vmid_graph_probe.so')
        eager = C.CDLL(str(native / 'vmid_eager_probe.so'))
        eager.vm8_eager_submit.argtypes = [C.c_uint64] * 5 + [C.c_int] * 2
        eager.vm8_eager_submit.restype = C.c_int
        host = HostQueries(native / 'vmid_context_probe.so')
        identities = Identities(host)
        limits = ro.ObserverLimits(max_receipt_bytes=64 * 1024 ** 2)
        binding = ro.ReplayBinding(
            str(native / 'flash_s5_replay_observer.so'),
            receipt['files']['flash_s5_replay_observer.so'],
            receipt['files']['flash_s5_replay_observer.cpp'], s.CUPTI,
            cupti_library_path=('/usr/local/lib/python3.12/dist-packages/'
                                'nvidia/cu13/lib/libcupti.so.13'))
        session = ro.ObserverSession(binding, limits)
        session.start()
        bridge = aa.CorrelationBridge(session, epoch=EPOCH,
                                      generation=GENERATION)
        split = gc.open_split(
            0, env={'FLASH_S5_DECODE_SMS': DECODE_SMS}, torch_module=torch,
            binding=gc.DriverBinding(str(native / 'flash_s5_green.so')))
        split.retain_allocator_context()
        r['split'] = dict(decode=split.decode_sms, prefill=split.prefill_sms,
                          total=split.total_sms,
                          decode_context_handle=split.decode_context_id,
                          prefill_context_handle=split.prefill_context_id)
        ordinary = torch.cuda.Stream()
        src = torch.full((1024,), 7, dtype=torch.int32, pin_memory=True)
        dst = torch.empty_like(src, pin_memory=True)
        scratch = torch.empty_like(src, pin_memory=True)
        a, b, z, e, f = [torch.zeros((1024,), dtype=torch.int32,
                                     device='cuda') for _ in range(5)]

        def submit(family, count=1, target=None):
            target = split.prefill_stream if target is None else target
            s.checked(eager.vm8_eager_submit(
                int(target.cuda_stream), e.data_ptr(), f.data_ptr(),
                src.data_ptr(), scratch.data_ptr(), family, count), 'eager')

        # One real retained graph, captured on the decode green stream.  It is
        # created before the window and replayed inside the decode offer.
        bank = NS(captured=True, pairs=[], handles={})
        bank.manager = NS(graphs={}, _flash_s5_capture_owner=split,
                          _flash_s5_capture_context=lambda: None)
        descriptor, local = object(), object()
        desc = dict(num_tokens=4, num_reqs=1, uniform_token_count=4,
                    num_ubatches=1)
        own, graph = C.c_uint64(), C.c_uint64()
        s.checked(gl.lib.vmg_create(
            int(split.decode_stream.cuda_stream), a.data_ptr(), b.data_ptr(),
            z.data_ptr(), src.data_ptr(), dst.data_ptr(), 1024,
            MIX['kernel'], MIX['memcpy'], MIX['memset'], 0,
            C.byref(own), C.byref(graph)), 'create')
        graph_owner = own.value
        graph_handle = graph.value
        bank.pairs.append((descriptor, local))
        bank.handles[local] = NS(local=local, bank=bank, destroyed=False)
        bank.manager.graphs[descriptor] = NS(raw_cuda_graph=lambda: graph_handle)
        window_probe = NS(bank=bank,
                          router=NS(storage_epoch=STORAGE_EPOCH,
                                    route_generation=ROUTE),
                          split=split, storage_epoch=STORAGE_EPOCH,
                          route_generation=ROUTE, route='split',
                          interval='active')
        lib = binding._lib
        before_bytes = s.graph_bytes(lib, graph_handle)

        # Warm every measured path before the window: the no-offer owner run
        # (reference values plus operator/library caches), the owned auxiliary
        # replacements, all eager families on every role stream and one graph
        # replay.  No warm submission is inside a measured window.
        reference = reference_ple = None
        if mode == 'positive':
            shared, ple, x, ids, ple_out = fixture(torch, cs.pinned_types())
            if shared[0]._stream is not shared[1]._stream:
                raise RuntimeError('stock_shared_stream_not_shared')
            owners_obj = discover(torch, cs, mh, split, shared, ple)
            ref_held = _owner_run(shared, ple, x, ids, ple_out)
            torch.cuda.synchronize()
            reference = [t.detach().float().cpu().clone() for t in ref_held]
            reference_ple = ple_out.detach().float().cpu().clone()
            owners_obj.prepare()
            for _ in range(2):
                with owners_obj.scope():
                    with torch.cuda.stream(split.prefill_stream):
                        warm = _owner_run(shared, ple, x, ids, ple_out)
                torch.cuda.synchronize()
                del warm
        for family in range(1, 8):
            submit(family, 1, split.prefill_stream)
            submit(family, 1, ordinary)
            submit(family, 1, split.decode_stream)
        progress(out, mode + ':before_warm_replay')
        s.checked(replay_owned_graph(gl, graph_owner, graph_handle), 'warm_replay')
        progress(out, mode + ':after_warm_replay')
        torch.cuda.synchronize()

        # Queried CUPTI identities, before any offer of the measured window.
        contexts = {
            'prefill_main': identities.prime(split.prefill_stream,
                                             'prefill_main'),
            'decode_main': identities.prime(split.decode_stream, 'decode_main'),
            'copy': identities.prime(ordinary, 'copy'),
        }
        if mode == 'positive':
            contexts['shared_experts_aux'] = identities.prime(
                owners_obj.replacements[0], 'shared_experts_aux')
            contexts['ple_aux'] = identities.prime(
                owners_obj.replacements[2], 'ple_aux')
        r['contexts'] = contexts
        streams = [row['stream_id'] for row in contexts.values()]
        greens = all(row['green'] for name, row in contexts.items()
                     if name != 'copy')
        r['contexts_disjoint'] = bool(
            greens and not contexts['copy']['green']
            and contexts['prefill_main']['context_id']
            != contexts['decode_main']['context_id']
            and contexts['copy']['context_id']
            != contexts['prefill_main']['context_id']
            and len(set(streams)) == len(streams))
        if mode == 'positive':
            r['copy_role_control'] = (
                contexts['copy']['context_id']
                != contexts['prefill_main']['context_id'])

        progress(out, mode + ':before_bind_bank')
        session.bind_bank(window_probe)
        session.begin_window(route_generation=ROUTE,
                             descriptor=bank.pairs[0][0])
        start = bridge.window()
        tx = e3_types.TransactionId(
            engine_epoch=EPOCH, request_uid='vm-job13c-prefill-request',
            request_generation=1, chunk_start=0, chunk_end=8, txn_seq=1)
        prefill = npo.NativePrefillLedger(
            bridge, worker='rank0', rank=0, epoch=EPOCH,
            generation=GENERATION, route_generation=ROUTE,
            storage_epoch=STORAGE_EPOCH, run_id=run_id, window_id=WINDOW_ID,
            stream_identity=identities, torch_module=torch,
            clock=host.timestamp, env={PREFILL_ENV: '1', ENGINE_ENV: '1'})

        if mode == 'positive':
            roles = {
                'main': {'context_id': contexts['decode_main']['context_id'],
                         'stream_id': contexts['decode_main']['stream_id']},
                'output_copy': {
                    'context_id': contexts['decode_main']['context_id'],
                    'stream_id': contexts['decode_main']['stream_id']},
            }
            wb = eo.bind_window(
                run_id=run_id, window_id=WINDOW_ID, worker='rank0', rank=0,
                worker_incarnation='vm-worker-1',
                engine_incarnation='synthetic-no-engine',
                engine_epoch=EPOCH, route_generation=ROUTE,
                storage_epoch=STORAGE_EPOCH,
                green_context_id=contexts['decode_main']['context_id'],
                native_start_ns=start['start_ns'], native_end_ns=None,
                observer_identity=binding.identity(),
                helper_identity=dict(
                    native_static_schema=ro.NATIVE_SCHEMA_STATIC,
                    native_window_schema=ro.NATIVE_SCHEMA_WINDOW,
                    compiled_api=130001, runtime_api=130001),
                declared_target='graph:0',
                descriptor_labels={'graph:0': eo.descriptor_key(desc)},
                router_census=None,
                bank_identity_sha256=eo.bank_identity(session._bank_meta),
                stream_roles=roles, pp_stage=0)
            decode = eo.EngineOfferLedger(
                bridge, worker='rank0', rank=0, epoch=EPOCH,
                generation=GENERATION, route_generation=ROUTE,
                green_context_id=contexts['decode_main']['context_id'],
                identities=roles, pp_stage=0, env={ENGINE_ENV: '1'},
                binding=wb)
            decode.activate()
            rep_shared = owners_obj.replacements[0]
            rep_ple = owners_obj.replacements[2]
            prefill.declare_streams(
                prefill_stream=split.prefill_stream, copy_stream=ordinary,
                aux_streams={'shared_experts_aux': rep_shared,
                             'ple_aux': rep_ple})
            prefill.owner_start('qsa_fill', 'copy', transaction=tx,
                                stream=ordinary)
            prefill.owner_start('shared_experts', 'shared_experts_aux',
                                transaction=tx, stream=rep_shared)
            prefill.owner_start('ple', 'ple_aux', transaction=tx,
                                stream=rep_ple)
            pins_shared = npo.live_pins(shared[0])
            for owner in shared:
                shadows.append(npo.install_seams(
                    owner, prefill, 'shared_experts', pins=pins_shared))
            shadows.append(npo.install_seams(
                ple, prefill, 'ple_embedding', pins=npo.live_pins(ple)))
            shadow_records = [shadow.record() for shadow in shadows]

            held = []
            with owners_obj.scope():
                with torch.cuda.stream(split.prefill_stream):
                    with prefill.segment('prepare', transaction=tx, op_seq=1):
                        submit(1, 1)
                    with prefill.segment('prefix', transaction=tx, op_seq=2):
                        submit(1, 2)
                    with prefill.segment('layer', transaction=tx, op_seq=3,
                                         branch='driver'):
                        with prefill.qsa('pre'):
                            submit(2, 1)
                        with prefill.qsa_fill():
                            submit(5, 1, ordinary)
                        with prefill.qsa_fill(suboperation='selection_readback'):
                            submit(6, 2, ordinary)
                        with prefill.qsa_fill(suboperation='sparse_bucket_fill'):
                            submit(4, 1, ordinary)
                        with prefill.qsa('bucket'):
                            submit(3, 1)
                        with prefill.qsa('tail'):
                            submit(1, 1)
                    with prefill.segment('layer', transaction=tx, op_seq=4,
                                         branch='direct'):
                        submit(2, 2)
                    with prefill.segment('suffix', transaction=tx, op_seq=5):
                        submit(3, 1)
                    # GDN/PLE state finalize: the real owner methods run here,
                    # each inside its own source-pinned offer.
                    with prefill.segment('state_finalize', transaction=tx,
                                         op_seq=6):
                        for owner in shared:
                            owner.maybe_forward_async(x)
                            owner.wait()
                            held.append(owner.output)
                        ple.start_prefetch(x, ids)
                        ple._finalize_prefetch(ple._prefetch_buffer, ple_out)

            with decode.step('execute'):
                with decode.offer_scope('request_state_table', 'prep_state'):
                    submit(1, 1, split.decode_stream)
                with decode.offer_scope('input_attention', 'prepare_inputs'):
                    submit(4, 1, split.decode_stream)
                with decode.offer_scope('input_attention', 'prepare_attn',
                                        branch='ordinary'):
                    submit(5, 1, split.decode_stream)
                with decode.offer_scope('forward_dispatch', 'graph_replay',
                                        descriptor=desc, census=MIX):
                    progress(out, mode + ':before_measured_replay')
                    s.checked(replay_owned_graph(gl, graph_owner, graph_handle), 'replay')
                    progress(out, mode + ':after_measured_replay')
            with decode.step('sample'):
                with decode.offer_scope('logits', 'compute_full'):
                    submit(2, 2, split.decode_stream)
                with decode.offer_scope('sampler', 'completion_counts'):
                    submit(7, 1, split.decode_stream)
                with decode.offer_scope('output_copy', 'async_output',
                                        branch='ordinary'):
                    submit(6, 1, split.decode_stream)
            events = {}
            for name, stream in (('lane', split.prefill_stream),
                                 ('qsa_fill', ordinary),
                                 ('shared_experts', rep_shared),
                                 ('ple', rep_ple)):
                events[name] = torch.cuda.Event(enable_timing=False)
                events[name].record(stream)
            torch.cuda.synchronize()
            for name in ('lane', 'qsa_fill', 'shared_experts', 'ple'):
                prefill.owner_finish(name, events=(events[name],))
            decode.deactivate()
            for shadow in reversed(shadows):
                shadow.close()
            shadows = []
            proof = ro.DrainedProof(
                worker_rank=0, worker_incarnation='vm-worker-1',
                engine_incarnation='synthetic-no-engine',
                route_generation=ROUTE, storage_epoch=STORAGE_EPOCH,
                relevant_consumers=2, controller_drained=True)
        else:
            prefill.declare_streams(prefill_stream=split.prefill_stream)
            with prefill.segment('prefix', transaction=tx, op_seq=1):
                submit(1, 1, ordinary)
            event = torch.cuda.Event(enable_timing=False)
            event.record(split.prefill_stream)
            torch.cuda.synchronize()
            prefill.owner_finish('lane', events=(event,))
            proof = ro.DrainedProof(
                worker_rank=0, worker_incarnation='vm-worker-1',
                engine_incarnation='synthetic-no-engine',
                route_generation=ROUTE, storage_epoch=STORAGE_EPOCH,
                relevant_consumers=1, controller_drained=True)

        # Every value comparison runs after the native window has closed: a
        # D2H copy or comparison launched inside it would be a real activity
        # outside all offers.
        progress(out, mode + ':before_window_end')
        raw_report = ro._receipt_dict(session.end_window(proof))
        progress(out, mode + ':after_window_end')
        r['window_receipt'] = raw_report
        if mode == 'positive':
            r['shadow_pins'] = shadow_records
            r['discovery'] = owners_obj.record()
            r['sharing_preserved'] = bool(
                owners_obj.originals[0] is owners_obj.originals[1]
                and owners_obj.replacements[0] is owners_obj.replacements[1])
            r['owners_restored'] = bool(
                all(getattr(owner, '_stream', None) is orig
                    for owner, orig in zip(shared, owners_obj.originals))
                and ple._prefetch_stream is owners_obj.originals[2])
            r['output_exact'] = bool(
                all(torch.equal(held[index].float().cpu(), reference[index])
                    for index in range(len(reference)))
                and torch.equal(ple_out.float().cpu(), reference_ple))
            decode.finish_window({'total': 1, 'graphs': {'graph:0': 1}})
            prefill.finish()
            receipt_prefill = prefill.seal()
            snapshot = prefill.snapshot()
            raw_before = json.dumps(snapshot['native'], sort_keys=True)
            composed = np_.compose_overlap(
                copy.deepcopy(snapshot['native']),
                copy.deepcopy(snapshot['offers']),
                decode_declared=copy.deepcopy(
                    {oid: dict(meta) for oid, meta in decode._meta.items()}),
                decode_window=copy.deepcopy(decode.window()),
                prefill_declared=copy.deepcopy(receipt_prefill['declared']),
                prefill_window=copy.deepcopy(receipt_prefill['window']),
                prefill_receipt=copy.deepcopy(receipt_prefill))
            r.update(
                composed=composed, prefill_receipt=receipt_prefill,
                native_report=snapshot['native'],
                offer_report=snapshot['offers'],
                decode_declared={str(oid): dict(meta) for oid, meta in decode._meta.items()},
                decode_window=decode.window(),
                raw_unchanged=(json.dumps(snapshot['native'], sort_keys=True)
                               == raw_before),
                offers={'decode': len(decode._meta),
                        'prefill': len(receipt_prefill['declared']),
                        'shared_report': len(snapshot['offers'])})
        else:
            prefill.finish()
            native_before = json.dumps(prefill.snapshot()['native'], sort_keys=True)
            receipt_negative = prefill.seal()
            r['raw_unchanged'] = native_before == json.dumps(
                prefill.snapshot()['native'], sort_keys=True)
            r['negative'] = {
                'verdict': receipt_negative['verdict'],
                'failures': receipt_negative['failures'],
                'missing': receipt_negative['missing'],
                'joined_records': receipt_negative['joined_records'],
                'counts': receipt_negative['counts'],
                'raw_complete': (receipt_negative.get('native') or {}).get(
                    'complete') is True,
                'raw_window': receipt_negative.get('native_window'),
            }
            snapshot = prefill.snapshot()
            r['native_report'] = snapshot['native']
        after_bytes = s.graph_bytes(lib, graph_handle)
        r['graph_bytes_stable'] = before_bytes == after_bytes
        r['graph_bytes'] = {'before': len(before_bytes),
                            'after': len(after_bytes)}
        r['graph_output_exact'] = bool(
            torch.all(dst == 7 + MIX['kernel']).item()
            and torch.all(z.cpu() == 0).item())
        r['allocator_peak_reserved'] = torch.cuda.max_memory_reserved(0)
    except Exception as exc:
        r['missing'].append('case_exception')
        r['error_type'] = type(exc).__name__
        r['error_message'] = str(exc).replace('\n', ' ')[:512]
        r['error_stack'] = [
            dict(file=Path(frame.filename).name, line=frame.lineno,
                 function=frame.name)
            for frame in traceback.extract_tb(exc.__traceback__)]
    finally:
        errors = []
        if decode is not None:
            try:
                decode.deactivate()
            except Exception:
                errors.append('ledger_deactivate')
        try:
            torch.cuda.synchronize()
        except Exception:
            errors.append('drain')
        for shadow in reversed(shadows):
            try:
                shadow.close()
            except Exception:
                errors.append('shadow_close')
        try:
            if session:
                if session.state == 'window':
                    session.end_window(ro.DrainedProof(
                        worker_rank=0, worker_incarnation='vm-worker-1',
                        engine_incarnation='synthetic-no-engine',
                        route_generation=ROUTE,
                        storage_epoch=STORAGE_EPOCH,
                        relevant_consumers=1, controller_drained=True))
                if session.state in ('started', 'bound', 'ended'):
                    session.stop()
                session.close()
        except Exception:
            errors.append('observer_close')
        try:
            if graph_owner and gl is not None:
                s.checked(gl.lib.vmg_close(graph_owner), 'graph_close')
        except Exception:
            errors.append('graph_close')
        try:
            if split is not None:
                split.close()
        except Exception:
            errors.append('native_close')
        r['cleanup_ok'] = not errors
        r['cleanup_errors'] = errors
        r['ended_wall_s'] = time.time()
    r['grade'] = case_grade(r)
    s.write(out / (mode + '.json'), r)
    return r


def probe(modules, native, out):
    r = dict(schema=SCHEMA, image=s.IMAGE, cases=[], regressions={},
             failures=[], missing=[], memory_fraction=.125,
             cannot_execute=list(CANNOT_EXECUTE), scope=SCOPE)
    try:
        if os.environ.get(ENGINE_ENV) != '1' or os.environ.get(PREFILL_ENV) != '1':
            raise RuntimeError('lane_enable_env')
        import torch
        torch.cuda.set_per_process_memory_fraction(.125, 0)
        torch.set_num_threads(4)
        if (torch.cuda.device_count() != 1
                or torch.cuda.get_device_capability(0) != (8, 6)):
            raise RuntimeError('device')
        sys.path.insert(0, str(modules))
        import replay_observer as ro
        import green_context as gc
        import capture_streams as cs
        import model_hooks as mh
        receipt = json.loads((native / 'q17-helper-build.json').read_text())
        if receipt['image'] != s.IMAGE or \
                receipt['cupti_library_sha256'] != s.CUPTI:
            raise RuntimeError('build_pin')
        for name, digest in receipt['files'].items():
            if s.sha(native / name) != digest:
                raise RuntimeError('artifact_pin')
        for name in ('flash_s5_green.cu', 'flash_s5_replay_observer.cpp',
                     'flash_s5_graph_bytes.cpp'):
            if s.sha(modules / 'native' / name) != receipt['files'][name]:
                raise RuntimeError('source_pin')
        r['build'] = receipt
        r['build_verified'] = True
        from vllm.config import VllmConfig, set_current_vllm_config
        with set_current_vllm_config(VllmConfig()):
            for mode in ('positive', 'ordinary'):
                r['cases'].append(case(torch, ro, gc, cs, mh, native, receipt,
                                       modules, out, mode))
        # Unmodified job1 and job2b-flat regressions, isolated processes and
        # the same helper binaries.  The child-node result remains separate.
        for name in ('job1', 'job2'):
            directory = out / ('regression-' + name)
            directory.mkdir()
            cmd = [sys.executable,
                   str(Path(__file__).with_name('vmid_' + name + '.py')),
                   '--mode', 'probe', '--modules', str(modules),
                   '--native', str(native), '--out', str(directory)]
            with (directory / 'process.log').open('x') as log:
                rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=120).returncode
            raw_regression = json.loads((directory / 'probe.json').read_text())
            if name == 'job1':
                import vmid_job1 as reg
                g = reg.grade(raw_regression)
                r['regressions']['job1'] = g
            else:
                import vmid_job2 as reg
                g = reg.grade(raw_regression)
                r['regressions']['job2b_flat'] = g['q17_flat_node_families']
                r['regressions']['job2_child_scope'] = g
            if rc != g['rc']:
                r['missing'].append(name + ':process_verdict')
    except Exception as exc:
        r['error_type'] = type(exc).__name__
        r['error_message'] = str(exc).replace('\n', ' ')[:512]
        r['missing'].append('probe_exception')
        r['error_stack'] = [
            dict(file=Path(frame.filename).name, line=frame.lineno,
                 function=frame.name)
            for frame in traceback.extract_tb(exc.__traceback__)]
    r['grade'] = grade(r)
    s.write(out / 'probe.json', r)
    return r['grade']['rc']


def finalize(out, process_rc):
    try:
        raw = json.loads((out / 'probe.json').read_text())
        g = grade(raw)
    except (OSError, ValueError, TypeError, KeyError):
        raw = {}
        g = s.verdict([], ['probe_result'])
        g['cases'] = {}
    try:
        memory = json.loads((out / 'monitor.json').read_text())
    except (OSError, ValueError):
        memory = {}
    f = list(g['failures'])
    m = list(g['missing'])
    if memory.get('stop_triggered') or memory.get('own_peak_mib', 0) > \
            s.CAP_MIB:
        f.append('resource_stop_or_cap')
    if process_rc in (139, 134):
        f.append('native_process_crash')
    elif process_rc not in (0, 1, 2) or process_rc != g['rc']:
        m.append('process_verdict_mismatch')
    if not memory.get('own_samples') or memory.get('errors', 1):
        m.append('memory_sampling')
    result = s.verdict(f, m)
    result.update(schema=SCHEMA, process_rc=process_rc, memory=memory,
                  cases=g['cases'],
                  regressions=raw.get('regressions', {}),
                  cannot_execute=raw.get('cannot_execute', []),
                  scope=SCOPE)
    s.write(out / 'result.json', result)
    print('VMID_JOB13C ' + result['verdict'])
    return result['rc']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('probe', 'monitor', 'finalize'),
                        required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--modules', type=Path)
    parser.add_argument('--native', type=Path)
    parser.add_argument('--process-rc', type=int, default=2)
    args = parser.parse_args()
    if args.mode == 'monitor':
        return s.monitor(args.out, 'flashnext-job13c')
    if args.mode == 'finalize':
        return finalize(args.out, args.process_rc)
    return probe(args.modules, args.native, args.out)


if __name__ == '__main__':
    sys.exit(main())
