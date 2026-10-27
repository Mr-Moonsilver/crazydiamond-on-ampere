#!/usr/bin/env python3
"""DRAFT, NOT RUN, NOT AUTHORIZED: prefill side-stream ownership, VM only.

Real pinned SharedExperts/PLE owner methods and real green-context helpers with
small synthetic operands.  The same owner run is observed three times: with the
stock primary-context side streams (failure-before), inside
``model_hooks.PrefillStreamOwners.scope()`` on prefill-green auxiliaries
(pass-after) and after exact restoration (failure-again).  A decode capture
regression proves the qualified decoder owners still use decode auxiliaries and
that the two pools are disjoint.  job1 and job2b run unmodified in isolated
processes.  No engine, no model load, no box action.

The PINNED_* receipt fields and the two real-owner results are VM-only
evidence.  This CPU-authored draft cannot establish native readiness or SM
containment; Codex must review and bundle it with the candidate before any GO.
"""
import argparse
import ctypes as C
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
import vmid_support as s
from vmid_job6 import HostQueries

HERE = Path(__file__).resolve().parent
_S5 = HERE / 'e3_builder'
if not _S5.is_dir():
    _S5 = HERE.parent / 'patches' / 's5_engine'
if str(_S5) not in sys.path:
    sys.path.insert(0, str(_S5))

FAMILIES = ('shared', 'ple', 'composed24')
PHASES = ('before', 'after', 'restored')
REPLAYS = 3
PREFILL_SYMBOLS = ('s5_prefill_aux_create', 's5_prefill_stream_check',
                   's5_prefill_aux_destroy')
DECODE_SYMBOLS = ('s5_decode_aux_create', 's5_decode_stream_check',
                  's5_decode_aux_destroy')
SOURCE_FILES = ('model_hooks.py', 'green_context.py', 'e3_types.py',
                'capture_streams.py')
SCOPE = ('real owner methods; synthetic small arithmetic and TP1/DP1 PLE fixture; '
         'uninitialized real Qwen4ExpModel used only for the pinned type/census check')


class NativeCallFailure(RuntimeError):
    def __init__(self, operation, rc):
        self.operation, self.rc = operation, rc
        super().__init__('native_call_failed')


def native_call(record, operation, function, *args):
    allowed = ('bind_prefill_context', 'bind_prefill_graph', 'begin_prefill_window',
               'bind_decode_context', 'bind_decode_graph', 'begin_decode_window')
    if operation not in allowed:
        raise ValueError('unknown_native_operation')
    rc = int(function(*args))
    record.setdefault('native_calls', []).append(dict(operation=operation, rc=rc))
    if rc:
        raise NativeCallFailure(operation, rc)


def cupti_context(facts):
    """Return an execution ID, never a CUgreenCtx/CUcontext handle."""
    value = facts.get('green_cupti_context_id')
    if (facts.get('green_present') is not True or type(value) is not int
            or not 0 <= value < 2**32):
        raise ValueError('green_cupti_context_unproved')
    return value


def subscribed_split(session, create_split):
    session.start()
    return create_split()


def bind_window_graph(record, operation, session, graph):
    """Satisfy the existing native window contract with a real retained graph."""
    c = session.binding._c
    nodes = c.c_uint32()
    native_call(record, operation, session.binding._lib.s5ro_bind_graph,
                session._token, graph.raw_cuda_graph(), 0, c.byref(nodes))
    if nodes.value < 1:
        raise ValueError('bound_graph_empty')
    static = session._report_call(session.binding._lib.s5ro_static, (session._token,))
    return dict(bound=True, slot=0, nodes=int(nodes.value),
                static_complete=static.get('complete'),
                static_reasons=static.get('reasons', []))


def valid_census(c):
    if not isinstance(c, dict):
        return False
    if c.get('schema') != 's5ro-activity-census/1' or c.get('complete') is not True:
        return False
    if any(c.get(k) != 0 for k in ('boundary_records', 'invalid_records',
                                   'omitted_records')):
        return False
    if type(c.get('rows')) is not list:
        return False
    if not all(type(x.get('count')) is int and x['count'] > 0 for x in c['rows']):
        return False
    return sum(x['count'] for x in c['rows']) == c.get('matched_records')


def context_rule(label, row, *, expected, allowed, side, required, want):
    """One phase's census rule; no count is guessed for the real-owner kernels."""
    f, m = [], []
    census = row.get('census', {})
    if not valid_census(census):
        return [], [label + ':census_incomplete']
    rows = census['rows']
    if not rows:
        return [], [label + ':unobserved']
    seen = {x['stream_id'] for x in rows}
    if not seen <= set(allowed):
        f.append(label + ':unexpected_side_stream')
    if want == 'expected':
        if any(x['context_id'] != expected for x in rows):
            f.append(label + ':wrong_context')
        for stream in required:
            if stream not in seen:
                m.append(label + ':side_stream_unobserved')
    elif want == 'other':
        side_rows = [x for x in rows if x['stream_id'] in set(side)]
        if any(x['context_id'] == expected for x in side_rows):
            f.append(label + ':stock_side_stream_is_green')
        for stream in required:
            if not any(x['context_id'] != expected and x['stream_id'] == stream
                       for x in rows):
                m.append(label + ':negative_side_unobserved')
    else:
        raise ValueError('unknown context rule')
    return f, m


def phase_grade(name, phase):
    f, m = [], []
    if not isinstance(phase, dict):
        return [], [name + ':missing']
    if phase.get('output_exact') is False:
        f.append(name + ':output')
    elif phase.get('output_exact') is not True:
        m.append(name + ':output')
    if phase.get('predicate') is None:
        m.append(name + ':predicate')
    elif name == 'after':
        if phase['predicate'] != 'PASS':
            f.append(name + ':owned_side_streams_not_green')
    elif phase['predicate'] != 'FAIL':
        m.append(name + ':negative_guard_not_reproduced')
    f.extend(phase.get('rule_failures', []))
    m.extend(phase.get('rule_missing', []))
    return f, m


def decode_graph_rule(row, *, expected, sms, bound):
    """Canonical graph/node accounting, not capture-stream equality.

    CUDA may replay a captured side branch on an internal stream. The strict
    native alias/node/kind/context guards and exact per-node multiplicity prove
    the branch; the census independently conserves its executed records. Stream
    IDs remain in the raw receipt. Eager phases still use context_rule.
    """
    from collections import Counter
    from replay_observer import (ObserverLimits, ReplayObserverError,
                                 NATIVE_FAIL_REASONS, _validate_window)
    f, m = [], []
    if row.get('subscriber_started_before_split') is not True:
        m.append('decode:subscriber_start_order')
    native = row.get('native_window')
    if type(native) is not dict:
        return f, m + ['decode:native_window']
    try:
        counts, flags, reasons = _validate_window(native, ObserverLimits(), 1)
    except (ReplayObserverError, TypeError, KeyError, ValueError):
        return f, m + ['decode:native_window_invalid']
    f.extend('decode:native:' + reason for reason in reasons
             if reason in NATIVE_FAIL_REASONS)
    m.extend('decode:native:' + reason for reason in reasons
             if reason not in NATIVE_FAIL_REASONS)
    if native.get('complete') is not True:
        m.append('decode:native_incomplete')
    if (type(bound) is not dict or type(bound.get('nodes')) is not int
            or bound['nodes'] < 1):
        return f, m + ['decode:native_graph_binding']
    context = native['context_diagnostics']
    if context['expected_context_id'] != expected:
        f.append('decode:native_context_binding')
    if context['complete'] is not True:
        m.append('decode:native_context_diagnostics')
    execution_nodes = Counter()
    known_graphs = set()
    for item in context['executions']:
        if item['slot'] != 0 or item['node_index'] >= bound['nodes']:
            m.append('decode:kernel_node_binding')
        if item['context_id'] != expected:
            f.append('decode:kernel_node_binding')
        execution_nodes[(item['graph_id'], item['graph_node_id'],
                         item['context_id'], item['stream_id'])] += item['count']
        known_graphs.add(item['graph_id'])
    if (type(sms) is not int or sms < 1
            or native['expected_context']['expected_sms'] != sms):
        f.append('decode:native_sm_binding')
    slots = native['graph_slots']
    if len(slots) != 1 or slots[0]['slot'] != 0:
        return f, m + ['decode:native_graph_slot']
    slot = slots[0]
    if type(bound) is not dict or bound.get('nodes') != slot['node_count']:
        return f, m + ['decode:native_graph_binding']
    if not slot['static_complete'] or not slot['complete']:
        m.append('decode:native_graph_incomplete')
    census = row.get('census')
    if not valid_census(census):
        return f, m + ['decode:census_incomplete']
    totals, nodes, kernels = Counter(), Counter(), Counter()
    for item in census['rows']:
        if item.get('context_id') != expected:
            f.append('decode:wrong_context')
        kind = item.get('kind')
        if kind not in ('kernel', 'memcpy', 'memset'):
            m.append('decode:unsupported_kind')
            continue
        graph, node = item.get('graph_id'), item.get('node_id')
        if (type(graph) is not int or graph <= 0
                or type(node) is not int or node <= 0):
            f.append('decode:not_graph_activity')
            continue
        if graph not in known_graphs:
            f.append('decode:unknown_graph')
        totals[kind] += item['count']
        nodes[(kind, graph, node)] += item['count']
        if kind == 'kernel':
            kernels[(graph, node, item['context_id'], item['stream_id'])] += item['count']
    if kernels != execution_nodes:
        m.append('decode:kernel_census_join')
    if counts['records_in_window'] != census['matched_records']:
        m.append('decode:native_census_conservation')
    for kind, field in (('kernel', 'kernel_executions'),
                        ('memcpy', 'copy_executions'),
                        ('memset', 'memset_executions')):
        required = slot['required'][kind]
        expected_bound = REPLAYS if required else 0
        if (slot[field] != required * REPLAYS
                or totals[kind] != required * REPLAYS
                or slot['node_execution_bounds'][kind]
                   != dict(min=expected_bound, max=expected_bound)
                or sum(key[0] == kind for key in nodes) != required
                or any(value != REPLAYS for key, value in nodes.items()
                       if key[0] == kind)):
            m.append('decode:node_multiplicity:' + kind)
    return sorted(set(f)), sorted(set(m))


def raw_context_grade(record):
    """Recompute context/coverage from raw rows, never from a cached predicate."""
    failures, missing = [], []
    if record.get('context_id_namespace') != 'CUPTI':
        return [], ['context_id_namespace']
    if record.get('subscriber_started_before_split') is not True:
        missing.append('subscriber_start_order')
    for role in ('prefill', 'decode'):
        graph = record.get(role + '_binding_graph', {})
        if (graph.get('bound') is not True or graph.get('slot') != 0
                or type(graph.get('nodes')) is not int or graph['nodes'] < 1):
            missing.append(role + ':binding_graph')
        if graph.get('static_complete') is not True:
            missing.append(role + ':binding_graph_static')
    if record.get('prefill_binding_graph', {}).get('replays_in_window') != 0:
        failures.append('prefill:sentinel_replayed')
    for role in ('prefill', 'decode'):
        try:
            observed = cupti_context(record['context_queries'][role])
        except (KeyError, TypeError, ValueError):
            missing.append(role + ':context_query')
        else:
            if observed != record.get(role + '_context_id'):
                failures.append(role + ':context_query_mismatch')
    inventory = record.get('stream_inventory')
    if not isinstance(inventory, dict):
        return [], ['stream_inventory']
    for key in ('prefill_main', 'decode_main'):
        if type(inventory.get(key)) is not int or inventory[key] < 0:
            return [], ['stream_inventory:' + key]
    for key in ('stock_side', 'prefill_aux', 'decode_aux',
                'required_stock_side', 'required_prefill_aux'):
        values = inventory.get(key)
        if (type(values) is not list or not values
                or any(type(x) is not int or x < 0 for x in values)):
            return [], ['stream_inventory:' + key]
    for key in ('prefill_context_id', 'decode_context_id'):
        if type(record.get(key)) is not int or record[key] < 0:
            return [], [key]
    for name in PHASES:
        phase = record.get('phases', {}).get(name)
        if not isinstance(phase, dict):
            missing.append(name + ':raw_phase')
            continue
        owned = name == 'after'
        side = inventory['prefill_aux' if owned else 'stock_side']
        required = inventory['required_prefill_aux' if owned else 'required_stock_side']
        f, m = context_rule(name, phase, expected=record['prefill_context_id'],
                            allowed=[inventory['prefill_main'], *side], side=side,
                            required=required, want='expected' if owned else 'other')
        failures.extend(f)
        missing.extend(m)
    decode = record.get('decode')
    if not isinstance(decode, dict):
        missing.append('decode:raw_phase')
    else:
        f, m = decode_graph_rule(decode, expected=record['decode_context_id'],
                                 sms=record.get('decode_sms'),
                                 bound=record.get('decode_binding_graph'))
        failures.extend(f)
        missing.extend(m)
    return failures, missing


def case_grade(r):
    f = list(r.get('failures', []))
    m = list(r.get('missing', []))
    for key in ('source_pins_verified', 'owners_restored', 'pools_disjoint'):
        if r.get(key) is False:
            f.append(key)
        elif r.get(key) is not True:
            m.append(key)
    if r.get('role_checks') is False:
        f.append('role_context_check')
    elif r.get('role_checks') is not True:
        m.append('role_context_check')
    discovery = r.get('discovery', {})
    if discovery.get('shared_owners') != 2 or discovery.get('ple_owners') != 1:
        m.append('owner_census')
    if discovery.get('sharing_preserved') is not True:
        f.append('original_sharing_lost')
    if discovery.get('basis') != 'prefill-green':
        f.append('prefill_stream_basis')
    aux = r.get('prefill_aux_tokens') or []
    decode_aux = r.get('decode_aux_tokens') or []
    if len(aux) != discovery.get('auxiliary_streams'):
        m.append('auxiliary_inventory')
    if not aux or not decode_aux:
        m.append('auxiliary_inventory')
    native_symbols = r.get('helper', {}).get('native_symbols', {})
    for name in PREFILL_SYMBOLS + DECODE_SYMBOLS:
        if native_symbols.get(name) is not True:
            m.append('helper_symbol')
    for name in PHASES:
        a, b = phase_grade(name, r.get('phases', {}).get(name))
        f.extend(name + ':' + x for x in a)
        m.extend(name + ':' + x for x in b)
    if r.get('window_ended') is not True:
        m.append('prefill_window')
    decode = r.get('decode', {})
    for key in ('capture_ok', 'owners_restored', 'graph_retired', 'output_exact',
                'window_ended'):
        if decode.get(key) is False:
            f.append('decode:' + key)
        elif decode.get(key) is not True:
            m.append('decode:' + key)
    if decode.get('predicate') == 'FAIL':
        f.append('decode:wrong_context')
    elif decode.get('predicate') != 'PASS':
        m.append('decode:predicate')
    f.extend('decode:' + x for x in decode.get('rule_failures', []))
    m.extend('decode:' + x for x in decode.get('rule_missing', []))
    lifetime = r.get('native_lifetime', {})
    if lifetime.get('native_release') != 'process-exit' or \
            lifetime.get('native_retained') is not True:
        m.append('native_allocator_lifetime')
    if lifetime.get('retained_prefill_auxiliary_streams') != len(aux) or not aux:
        m.append('prefill_auxiliary_retention')
    if r.get('late_tensor_release_after_close') is not True:
        m.append('late_tensor_release')
    if r.get('cleanup_ok') is False:
        f.append('cleanup')
    elif r.get('cleanup_ok') is not True:
        m.append('cleanup')
    a, b = raw_context_grade(r)
    f.extend(a)
    m.extend(b)
    return s.verdict(f, m)


def grade(r):
    f = list(r.get('failures', []))
    m = list(r.get('missing', []))
    rows = r.get('cases', [])
    if [x.get('family') for x in rows] != list(FAMILIES):
        m.append('case_inventory')
    grades = {}
    for x in rows:
        name = x.get('family', 'unknown')
        g = case_grade(x)
        grades[name] = g
        f.extend(name + ':' + v for v in g['failures'])
        m.extend(name + ':' + v for v in g['missing'])
    for name in ('job1', 'job2b_flat'):
        g = r.get('regressions', {}).get(name, {})
        if g.get('verdict') == 'FAIL':
            f.append(name + ':regression')
        elif g.get('verdict') != 'PASS':
            m.append(name + ':regression_missing')
    g = s.verdict(f, m)
    g['cases'] = grades
    return g


def fixture(torch, types):
    """Real SharedExperts/PLE instances, synthetic operands (as job7b/fixture)."""
    shared_cls, ple_cls = types

    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.up = torch.full((1280, 2560), .001, device='cuda', dtype=torch.bfloat16)
            self.down = torch.full((2560, 640), .001, device='cuda', dtype=torch.bfloat16)
            self.gate = torch.full((1, 2560), .001, device='cuda', dtype=torch.bfloat16)

        def forward(self, x):
            import torch.nn.functional as F
            z = F.linear(x, self.up)
            a, b = z.chunk(2, dim=-1)
            return F.linear(F.silu(a) * b, self.down) * torch.sigmoid(F.linear(x, self.gate))

    cfg = SimpleNamespace(moe_parallel_config=SimpleNamespace(
        enable_eplb=False, all2all_backend='allgather_reducescatter',
        use_fi_nvl_two_sided_kernels=False))
    mlp = MLP()
    shared = [shared_cls(mlp, cfg, False, lambda: False) for _ in range(2)]
    x = torch.full((4, 2560), .125, device='cuda', dtype=torch.bfloat16)
    if not all(int(o._determine_shared_experts_order(x)) == 3 for o in shared):
        raise RuntimeError('real_shared_overlap_branch_unavailable')
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
    ple.shard_indices = SimpleNamespace(org_vocab_start_index=0, org_vocab_end_index=32)
    ple._prefetch_stream = torch.cuda.Stream()
    ple._prefetch_buffer = torch.empty((4, 2, 160), device='cuda', dtype=torch.bfloat16)
    ple._output_dim = 320
    ids = torch.tensor([[0, 1], [2, 3], [4, 5], [30, 31]], device='cuda', dtype=torch.int64)
    ple_out = torch.empty((4, 320), device='cuda', dtype=torch.bfloat16)
    outputs = [torch.empty_like(x) for _ in range(24)]
    marker = torch.zeros((32,), device='cuda')
    owners = [(shared[0], '_stream'), (shared[1], '_stream'),
              (ple, '_prefetch_stream')]

    def run(family):
        marker.fill_(2)
        if family != 'shared':
            ple.start_prefetch(x, ids)
        if family != 'ple':
            for i, out in enumerate(outputs):
                o = shared[i % 2]
                if o.maybe_forward_async(x) is not True:
                    raise RuntimeError('shared_path_changed')
                o.wait()
                out.copy_(o.output)
        if family != 'shared':
            ple._finalize_prefetch(ple._prefetch_buffer, ple_out)
        marker.add_(1)

    def take(family):
        selected = [marker]
        if family != 'ple':
            selected.extend(outputs)
        if family != 'shared':
            selected.append(ple_out)
        return [t.detach().cpu().clone() for t in selected]

    return owners, shared, ple, run, take


def discover(torch, cs, mh, split, shared, ple):
    """Real type/census discovery; the model container is never executed."""
    from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpModel
    model = Qwen4ExpModel.__new__(Qwen4ExpModel)
    object.__setattr__(model, 'start_layer', 0)
    object.__setattr__(model, 'end_layer', 2)
    object.__setattr__(model, 'config', SimpleNamespace(ple_layer_ids=[1]))
    runner = SimpleNamespace(model=SimpleNamespace(
        modules=lambda: [model, shared[0], shared[1], ple]))
    owners = mh.PrefillStreamOwners.discover(
        runner, split, pinned=cs.pinned_types())
    got = [(id(o), a) for o, a in owners.owners]
    want = [(id(o), '_stream') for o in shared] + [(id(ple), '_prefetch_stream')]
    if got != want:
        raise RuntimeError('prefill owner discovery differs from the fixture')
    return owners


def sharing_preserved(owners):
    groups = {}
    for old, new in zip(owners.originals, owners.replacements):
        if old is None:
            if new is not None:
                return False
            continue
        key = old.cuda_stream
        if key in groups and groups[key] is not new:
            return False
        groups[key] = new
    return True


def census(session, lib, token, start, end):
    return session._report_call(lib.s5ro_activity_census, (token, start, end))


def sync_streams(streams):
    seen = set()
    for stream in streams:
        if stream is not None and id(stream) not in seen:
            seen.add(id(stream))
            stream.synchronize()


def run_phase(torch, host, split, scope, run, family, take, reference, side_objects):
    """One eager interval; census is read after the observer window closes."""
    with scope:
        begin = host.timestamp()
        with torch.cuda.stream(split.prefill_stream):
            run(family)
        sync_streams([split.prefill_stream, *side_objects])
        end = host.timestamp()
    return dict(start=begin, end=end,
                output_exact=all(torch.equal(a, b) for a, b in zip(reference, take(family))))


def case(torch, ro, gc, cs, mh, native, receipt, out, family, vconfig):
    r = dict(family=family, failures=[], missing=[], phases={}, decode={},
             source_scope=SCOPE, source_pins={},
             substitutions=['bounded PLE constructor state (32x160 pinned BF16)',
                            'synthetic BF16 gated MLP weights; no model load',
                            'uninitialized Qwen4ExpModel container for the type/census check only'],
             started_wall_s=time.time())
    prefill_session = decode_session = split = adapter = None
    graphs = []
    attached = []
    window = None
    late_tensor = None
    try:
        for name in SOURCE_FILES:
            r['source_pins'][name] = s.sha(Path(receipt['modules']) / name)
        if r['source_pins'] != receipt['sources']:
            raise RuntimeError('source_pin')
        r['source_pins_verified'] = True
        # Match the qualified job7b/engine ordering: subscribe before creating
        # green contexts, rather than attaching after their creation/query.
        limits = ro.ObserverLimits()
        binding_o = ro.ReplayBinding(
            str(native / 'flash_s5_replay_observer.so'),
            receipt['files']['flash_s5_replay_observer.so'],
            receipt['files']['flash_s5_replay_observer.cpp'], s.CUPTI,
            cupti_library_path='/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib/libcupti.so.13')
        prefill_session = ro.ObserverSession(binding_o, limits)
        def create_split():
            binding = gc.DriverBinding(str(native / 'flash_s5_green.so'))
            return gc.open_split(0, env={'FLASH_S5_DECODE_SMS': '44'},
                                 torch_module=torch, binding=binding)
        split = subscribed_split(prefill_session, create_split)
        binding = split.binding
        r['subscriber_started_before_split'] = True
        r['helper'] = {'abi': binding.abi,
                       'prefill_symbols': {n: callable(getattr(binding, n, None))
                                           for n in PREFILL_SYMBOLS},
                       'decode_symbols': {n: callable(getattr(binding, n, None))
                                          for n in DECODE_SYMBOLS}}
        native_symbols = {}
        for name in PREFILL_SYMBOLS + DECODE_SYMBOLS:
            try:
                getattr(binding._lib, name)
                native_symbols[name] = True
            except AttributeError:
                native_symbols[name] = False
        r['helper']['native_symbols'] = native_symbols
        r['prefill_sms'] = split.prefill_sms
        r['decode_sms'] = split.decode_sms
        split.retain_allocator_context()  # VM7b before any tensor can see an aux
        ordinary = torch.cuda.Stream()
        owners, shared, ple, run, take = fixture(torch, cs.pinned_types())
        owners_obj = discover(torch, cs, mh, split, shared, ple)
        owners_obj.prepare()
        r['discovery'] = owners_obj.record()
        r['discovery']['sharing_preserved'] = sharing_preserved(owners_obj)
        r['prefill_aux_tokens'] = sorted(x.cuda_stream for x in owners_obj.streams)
        host = HostQueries(native / 'vmid_context_probe.so')
        prefill_identity = host.stream(split.prefill_stream)
        decode_identity = host.stream(split.decode_stream)
        r['context_id_namespace'] = 'CUPTI'
        r['context_queries'] = dict(prefill=prefill_identity, decode=decode_identity)
        r['prefill_context_id'] = cupti_context(prefill_identity)
        r['decode_context_id'] = cupti_context(decode_identity)
        main_cupti = prefill_identity['cupti_stream_id']
        stock_side = [host.stream(getattr(o, a))['cupti_stream_id']
                      for o, a in owners]
        aux_side = [host.stream(x)['cupti_stream_id'] for x in owners_obj.streams]
        # The family's side owners are required to appear in its own census; a
        # disabled stock stream is never required (and never re-enabled).
        indexed = {'shared': [(0, 0)], 'ple': [(2, 2)],
                   'composed24': [(0, 0), (2, 2)]}[family]
        required_aux = [owners_obj.streams.index(owners_obj.replacements[i])
                        for i, _ in indexed if owners_obj.replacements[i] is not None]
        required_stock = [stock_side[j] for i, j in indexed
                          if getattr(owners[i][0], owners[i][1], None) is not None]
        # Deterministic reference with the stock owners on an ordinary stream.
        with torch.cuda.stream(ordinary):
            run(family)
        torch.cuda.synchronize()
        reference = take(family)
        # Decode regression: the qualified capture repair and decode pool.
        adapter = cs.CaptureStreams(split, [(o, a, 'ple' if a == '_prefetch_stream'
                                             else 'shared') for o, a in owners])
        adapter.prepare()
        r['decode_aux_tokens'] = sorted(x.cuda_stream for x in adapter.streams)
        r['pools_disjoint'] = bool(
            set(r['prefill_aux_tokens']) and set(r['decode_aux_tokens'])
            and not set(r['prefill_aux_tokens']) & set(r['decode_aux_tokens']))
        role = []
        try:
            split.check_prefill_stream(adapter.streams[0])
            role.append('prefill_accepts_decode')
        except gc.GreenContextError:
            pass
        try:
            split.check_decode_stream(owners_obj.streams[0])
            role.append('decode_accepts_prefill')
        except gc.GreenContextError:
            pass
        r['role_checks'] = not role
        r['role_errors'] = role

        def capture(stream):
            from vllm.forward_context import set_forward_context
            from vllm.config import CUDAGraphMode
            with set_forward_context(None, vconfig, num_tokens=4,
                                     cudagraph_runtime_mode=CUDAGraphMode.NONE):
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        run(family)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph(keep_graph=True)
                graphs.append(g)
                with torch.cuda.graph(g, stream=stream):
                    run(family)
                g.instantiate()
                stream.synchronize()
                return g

        decode_main = host.stream(split.decode_stream)['cupti_stream_id']
        decode_side = [host.stream(x)['cupti_stream_id'] for x in adapter.streams]
        r['stream_inventory'] = dict(
            prefill_main=main_cupti, stock_side=stock_side, prefill_aux=aux_side,
            required_stock_side=required_stock,
            required_prefill_aux=[aux_side[i] for i in required_aux],
            decode_main=decode_main, decode_aux=decode_side)
        # Keep the initial subscriber (started BEFORE split creation) for the
        # strict decode graph window. The eager prefill negatives get a fresh
        # subscriber afterwards so their sticky primary-context findings cannot
        # poison the independent decode placement proof.
        decode_binding = binding_o
        decode_session, prefill_session = prefill_session, None
        r['decode']['subscriber_started_before_split'] = True
        # Retain clone/resource provenance for the actual decode graph too.
        # Subscribe before capture, then bind that graph before begin_window.
        with adapter.capture():
            graph = capture(split.decode_stream)
        adapter.attach_graph(graph)
        attached.append(graph)
        r['decode']['capture_ok'] = True
        r['decode']['owners_restored'] = adapter.record()['original_owners_restored']
        # Re-query after subscribing the second owned observer. Compare only
        # CUPTI execution IDs, not raw green-context handles from the split.
        decode_identity = host.stream(split.decode_stream)
        r['context_queries']['decode'] = decode_identity
        r['decode_context_id'] = cupti_context(decode_identity)
        decode_main = decode_identity['cupti_stream_id']
        decode_side = [host.stream(x)['cupti_stream_id'] for x in adapter.streams]
        r['stream_inventory'].update(decode_main=decode_main, decode_aux=decode_side)
        dlib, dtoken = decode_binding._lib, decode_session._token
        dlib.s5ro_activity_census.argtypes = [C.c_uint64] * 3 + [
            C.c_void_p, C.c_uint64, C.POINTER(C.c_uint64)]
        dlib.s5ro_activity_census.restype = C.c_int
        native_call(r, 'bind_decode_context', dlib.s5ro_bind_context,
                    dtoken, split.decode_context_id, split.decode_sms)
        r['decode_binding_graph'] = bind_window_graph(
            r, 'bind_decode_graph', decode_session, graph)
        dstart = C.c_uint64()
        native_call(r, 'begin_decode_window', dlib.s5ro_begin_window,
                    dtoken, 1, C.byref(dstart))
        dwindow = dstart.value
        begin = host.timestamp()
        with torch.cuda.stream(split.decode_stream):
            for _ in range(REPLAYS):
                graph.replay()
        sync_streams([split.decode_stream, *adapter.streams])
        end = host.timestamp()
        r['decode']['native_window'] = decode_session._report_call(
            dlib.s5ro_end_window, (dtoken, dwindow, 1), retry=False)
        r['decode']['window_ended'] = True
        r['decode']['census'] = census(decode_session, dlib, dtoken, begin, end)
        r['decode']['output_exact'] = all(
            torch.equal(a, b) for a, b in zip(reference, take(family)))
        rows = r['decode']['census'].get('rows', [])
        r['decode']['predicate'] = (
            'FAIL' if any(x['context_id'] != r['decode_context_id'] for x in rows)
            else 'PASS' if rows else 'INCONCLUSIVE')
        f, m = decode_graph_rule(r['decode'], expected=r['decode_context_id'],
                                 sms=r['decode_sms'],
                                 bound=r['decode_binding_graph'])
        r['decode']['rule_failures'] = f
        r['decode']['rule_missing'] = m
        decode_session.close()
        decode_session = None
        graph.reset()
        adapter.retire_graph(graph)
        attached.remove(graph)
        adapter.close_after_graphs()
        r['decode']['graph_retired'] = True
        # Eager ownership phases keep their strict source-stream rule.
        binding_o = ro.ReplayBinding(
            str(native / 'flash_s5_replay_observer.so'),
            receipt['files']['flash_s5_replay_observer.so'],
            receipt['files']['flash_s5_replay_observer.cpp'], s.CUPTI,
            cupti_library_path='/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib/libcupti.so.13')
        prefill_session = ro.ObserverSession(binding_o, limits)
        prefill_session.start()
        # Prefill observation: one window, three intervals, one expected context.
        lib, token = binding_o._lib, prefill_session._token
        lib.s5ro_activity_census.argtypes = [C.c_uint64] * 3 + [
            C.c_void_p, C.c_uint64, C.POINTER(C.c_uint64)]
        lib.s5ro_activity_census.restype = C.c_int
        r['observer_host_reserved_bytes'] = prefill_session._native_reserved
        native_call(r, 'bind_prefill_context', lib.s5ro_bind_context, token,
                    split.prefill_context_id, split.prefill_sms)
        # begin_window requires both a context AND a retained graph, even when
        # the scoped evidence is eager activity. Bind a separate tiny graph
        # but do not replay it inside the three eager census intervals.
        sentinel = torch.empty((32,), device='cuda')
        with torch.cuda.stream(split.prefill_stream):
            sentinel.zero_()
        split.prefill_stream.synchronize()
        binding_graph = torch.cuda.CUDAGraph(keep_graph=True)
        graphs.append(binding_graph)
        with torch.cuda.graph(binding_graph, stream=split.prefill_stream):
            sentinel.add_(0)
        binding_graph.instantiate()
        split.prefill_stream.synchronize()
        r['prefill_binding_graph'] = bind_window_graph(
            r, 'bind_prefill_graph', prefill_session, binding_graph)
        r['prefill_binding_graph'].update(kind='one_kernel_sentinel', replays_in_window=0)
        start = C.c_uint64()
        native_call(r, 'begin_prefill_window', lib.s5ro_begin_window,
                    token, 1, C.byref(start))
        window = start.value
        r['phases']['before'] = run_phase(
            torch, host, split, nullcontext(), run, family, take, reference,
            [getattr(o, a) for o, a in owners])
        r['phases']['after'] = run_phase(
            torch, host, split, owners_obj.scope(), run, family, take, reference,
            owners_obj.streams)
        r['phases']['restored'] = run_phase(
            torch, host, split, nullcontext(), run, family, take, reference,
            [getattr(o, a) for o, a in owners])
        r['owners_restored'] = all(getattr(o, a) is old for (o, a), old
                                   in zip(owners, owners_obj.originals))
        r['prefill_native_window'] = prefill_session._report_call(
            lib.s5ro_end_window, (token, window, 1), retry=False)
        window = None
        r['window_ended'] = True
        for name in PHASES:
            phase = r['phases'][name]
            phase['census'] = census(prefill_session, lib, token,
                                     phase.pop('start'), phase.pop('end'))
        for name in PHASES:
            phase = r['phases'][name]
            side = aux_side if name == 'after' else stock_side
            want = 'expected' if name == 'after' else 'other'
            req = [aux_side[i] for i in required_aux] if name == 'after' else required_stock
            f, m = context_rule(name, phase, expected=r['prefill_context_id'],
                                allowed=[main_cupti, *side], side=side,
                                required=req, want=want)
            phase['rule_failures'] = f
            phase['rule_missing'] = m
            rows = phase.get('census', {}).get('rows', [])
            if name == 'after':
                off = any(x['context_id'] != r['prefill_context_id'] for x in rows)
            else:
                off = any(x['context_id'] != r['prefill_context_id']
                          for x in rows if x['stream_id'] in set(side))
            phase['predicate'] = 'FAIL' if off else 'PASS' if rows else 'INCONCLUSIVE'
        prefill_session.close()
        prefill_session = None
        binding_graph.reset()
        # Explicit late free after logical close (VM7b).
        late_tensor = torch.ones((1024,), device='cuda')
        late_stream = owners_obj.streams[0]
        with torch.cuda.stream(late_stream):
            late_tensor.add_(1)
        late_tensor.record_stream(late_stream)
        late_stream.synchronize()
    except Exception as exc:
        r['error_type'] = type(exc).__name__
        r['error_message'] = str(exc).replace('\n', ' ')[:512]
        if isinstance(exc, NativeCallFailure):
            r['native_error'] = dict(operation=exc.operation, rc=exc.rc)
        r['error_stack'] = [dict(file=Path(x.filename).name, line=x.lineno,
                                 function=x.name)
                            for x in traceback.extract_tb(exc.__traceback__)]
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            r['failures'].append('cuda_oom')
        else:
            r['missing'].append('case_exception')
    finally:
        errors = []
        drained = False
        try:
            torch.cuda.synchronize()
            drained = True
        except Exception:
            errors.append('drain')
        try:
            if prefill_session and window is not None:
                prefill_session._report_call(
                    prefill_session.binding._lib.s5ro_end_window,
                    (prefill_session._token, window, 1), retry=False)
            for session in (prefill_session, decode_session):
                if session:
                    session.close()
        except Exception:
            errors.append('observer_close')
        if drained:
            for g in reversed(graphs):
                try:
                    g.reset()
                    if adapter and g in attached:
                        adapter.retire_graph(g)
                except Exception:
                    errors.append('graph_reset')
            try:
                if adapter:
                    adapter.close_after_graphs()
                if split:
                    split.close()
                    r['native_lifetime'] = split.lifetime_record()
                if late_tensor is not None:
                    del late_tensor
                    import gc as python_gc
                    python_gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    r['late_tensor_release_after_close'] = True
            except Exception:
                errors.append('stream_close')
        r['cleanup_ok'] = not errors
        r['cleanup_errors'] = errors
        r['ended_wall_s'] = time.time()
    r['grade'] = case_grade(r)
    s.write(out / (family + '.json'), r)
    return r


def probe(modules, native, out):
    r = dict(schema='vm<vm-id>-job7f/1', image=s.IMAGE, cases=[], regressions={},
             failures=[], missing=[], memory_fraction=.125,
             scope='VM-only prefill side-stream ownership draft; not a box GO')
    try:
        import torch
        torch.cuda.set_per_process_memory_fraction(.125, 0)
        torch.set_num_threads(4)
        if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (8, 6):
            raise RuntimeError('device')
        sys.path.insert(0, str(modules))
        import replay_observer as ro
        import green_context as gc
        import capture_streams as cs
        import model_hooks as mh
        receipt = json.loads((native / 'q17-helper-build.json').read_text())
        receipt['modules'] = str(modules)
        if receipt['image'] != s.IMAGE or receipt['cupti_library_sha256'] != s.CUPTI:
            raise RuntimeError('build_pin')
        for name, h in receipt['files'].items():
            if s.sha(native / name) != h:
                raise RuntimeError('artifact_pin')
        for name in ('flash_s5_green.cu', 'flash_s5_replay_observer.cpp',
                     'flash_s5_graph_bytes.cpp'):
            if s.sha(modules / 'native' / name) != receipt['files'][name]:
                raise RuntimeError('source_pin')
        if set(receipt.get('prefill_symbols', ())) != set(PREFILL_SYMBOLS):
            raise RuntimeError('helper_extension_pin')
        r['build'] = receipt
        from vllm.config import VllmConfig, set_current_vllm_config
        with set_current_vllm_config(VllmConfig()):
            from vllm.config import get_current_vllm_config
            cfg = get_current_vllm_config()
            for family in FAMILIES:
                r['cases'].append(case(torch, ro, gc, cs, mh, native, receipt,
                                       out, family, cfg))
        for name in ('job1', 'job2'):
            d = out / ('regression-' + name)
            d.mkdir()
            cmd = [sys.executable,
                   str(Path(__file__).with_name('vmid_' + name + '.py')),
                   '--mode', 'probe', '--modules', str(modules),
                   '--native', str(native), '--out', str(d)]
            with (d / 'process.log').open('x') as log:
                rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=120).returncode
            raw = json.loads((d / 'probe.json').read_text())
            if name == 'job1':
                import vmid_job1 as reg
                g = reg.grade(raw)
                r['regressions']['job1'] = g
            else:
                import vmid_job2 as reg
                g = reg.grade(raw)
                r['regressions']['job2b_flat'] = g['q17_flat_node_families']
                r['regressions']['job2_child_scope'] = g
            if rc != g['rc']:
                r['missing'].append(name + ':process_verdict')
    except Exception as exc:
        r['error_type'] = type(exc).__name__
        r['error_message'] = str(exc).replace('\n', ' ')[:512]
        r['missing'].append('probe_exception')
        r['error_stack'] = [dict(file=Path(x.filename).name, line=x.lineno,
                                 function=x.name)
                            for x in traceback.extract_tb(exc.__traceback__)]
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
        m = json.loads((out / 'monitor.json').read_text())
    except (OSError, ValueError):
        m = {}
    f = list(g['failures'])
    missing = list(g['missing'])
    if m.get('stop_triggered') or m.get('own_peak_mib', 0) > 3072:
        f.append('resource_stop_or_cap')
    if process_rc in (139, 134):
        f.append('native_process_crash')
    elif process_rc not in (0, 1, 2) or process_rc != g['rc']:
        missing.append('process_verdict_mismatch')
    if not m.get('own_samples') or m.get('errors', 1):
        missing.append('memory_sampling')
    result = s.verdict(f, missing)
    result.update(schema='vm<vm-id>-job7f/1', process_rc=process_rc, memory=m,
                  cases=g['cases'], regressions=raw.get('regressions', {}),
                  scope='real owner methods with synthetic payloads; no engine/box qualification')
    s.write(out / 'result.json', result)
    print('VMID_JOB7F ' + result['verdict'])
    return result['rc']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('probe', 'monitor', 'finalize'), required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--modules', type=Path)
    p.add_argument('--native', type=Path)
    p.add_argument('--process-rc', type=int, default=2)
    a = p.parse_args()
    if a.mode == 'monitor':
        return s.monitor(a.out, 'flashnext-job7f')
    if a.mode == 'finalize':
        return finalize(a.out, a.process_rc)
    return probe(a.modules, a.native, a.out)


if __name__ == '__main__':
    sys.exit(main())
