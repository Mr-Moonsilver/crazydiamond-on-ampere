#!/usr/bin/env python3
"""DRAFT, NOT RUN, NOT AUTHORIZED: real private initialization + pending query.

VM-only positive accounting for the two U source leaves added after VM13c:

* ``LanePrivateBindings.discover`` is built on the real ``LaneRuntime`` and
  ``WorkerState`` with small *actual* CUDA allocations (private UID buffer,
  top-k scratch, decode UID, request slot table, retained Marlin lock) and
  synthetic runner/model ownership metadata.  No engine and no weights.
* ``bindings.prepare_workspace`` funds the already-provisioned extra workspace
  lane from the stock high-water allocation; the real ``chunk_scope(...).scope``
  then runs the unmodified ``_populate_uid_once`` and ``_initialize_lock_once``
  inside an actual ``layer`` segment offer, with the real
  ``NativePrefillLedger`` attached.  The QSA layer parent stays a declared
  zero-activity container; uid_init/lock_init are its leaves.
* In the same measured native ``s5ro-correlation/2`` report a real
  ``cudaEventQuery`` is observed pending (return 600, zero joined GPU work)
  and later completes with a real queried success.  Pending is never
  completion: the offer/owner completion counts come from the later queries.
* A real retained graph is replayed inside the decode partition.  Both
  partitions are regraded through ``native_placement.compose_overlap``; the
  whole report is retained unmodified and conservation must be exact.

The bounded delay that makes the pending query deterministic is
``torch.cuda._sleep(cycles)`` on the declared prefill-green stream, inside a
declared ``prefix`` offer.  If it is unavailable the probe leaves the pending
requirement missing (INCONCLUSIVE) rather than fabricating a native row.

Negatives (primary-context work, unoffered eager work, 600 on a non-query or
with GPU activity) are CPU mutation tests in ``test_vmid_job13g.py``; they
are never injected into a real CUDA call here.

This draft cannot establish native readiness, engine call sites or SM
containment; no engine, model, PP stage, box or GPU action is taken by it.
"""
import argparse
import ast
import copy
import ctypes as C
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import MethodType, SimpleNamespace as NS

import vmid_support as s
from vmid_job2 import GraphBinding
from vmid_job6 import HostQueries

SCHEMA = 'vm<vm-id>-job13g/1'
RUN_ID = 'vm-job13g'
WINDOW_ID = 'native-overlap-13g'
EPOCH = 1
GENERATION = 1
STORAGE_EPOCH = 1
ROUTE = 1
DECODE_SMS = '44'
WORKSPACE_SHA256 = 'd21c08167d1d0d1c0cddaa80453a53a0b4dec5d81b5a537c94a21324344976b3'
ENGINE_ENV = 'FLASH_S5_ENGINE_OFFERS'
PREFILL_ENV = 'FLASH_S5_NATIVE_PREFILL_OFFERS'
# Same retained graph probe as VM13c: 1386 kernel, 26 copy, 7 memset nodes.
MIX = dict(kernel=1386, memcpy=26, memset=7)
CHUNK_TOKENS = 8
WIDTH = 4
REQ_INDEX = 2
SEED = 7
# Bounded finite delay for a deterministic pending query (~1.3 s at 1.5 GHz).
SLEEP_CYCLES = 2_000_000_000
MODULES = ('lane_private.py', 'native_prefill_offers.py', 'native_placement.py', 'native_graph_placement.py')
PINNED_MODULES = ('activity_accounting.py', 'capture_streams.py',
                  'e3_types.py', 'engine_offers.py', 'green_context.py',
                  'lane_chunk.py', 'lane_dispatch.py', 'lane_hooks.py',
                  'lane_state.py', 'model_hooks.py', 'native_resources.py',
                  'native_startup.py', 'replay_observer.py', 'qsa_driver.py')
CANNOT_EXECUTE = (
    'TP>1 sharded prefill: one GPU drives TP1 only; sharded prefill needs '
    'peer ranks and collective transports',
    'PP>1 prefill, pp_receive and pp_send: one GPU has no pipeline peer',
    'full Qwen4ExpModel prefill/decoder: no engine, weights or scheduler; '
    'only the pinned private-scope methods and synthetic ownership metadata run',
    'real overlap performance: the measured work is synthetic and bounded',
)
SCOPE = ('real pinned vLLM conditional/causal/backbone class shells with synthetic '
         'local layers/configuration, repaired ownership discovery and LaneForward '
         'construction (no model forward or PP arithmetic); real '
         'LaneRuntime/WorkerState/LanePrivateBindings discovery, real '
         'workspace funding, real uid_init/lock_init leaves over actual CUDA '
         'allocations, real pending-then-complete EventQuery, retained graph '
         'replay and paired compose/regrade; no engine, model, PP or readiness '
         'qualification')


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
    """Verify bundled bytes against the immutable build closure (root bundle)."""
    pins = {name: s.sha(modules / name) for name in MODULES + PINNED_MODULES}
    closure = json.loads((modules.parent / 'e3_build_closure.json').read_text())
    sources = closure.get('builder_sources')
    if not isinstance(sources, dict):
        raise RuntimeError('closure_sources')
    for name, digest in pins.items():
        if sources.get(name) != digest:
            raise RuntimeError('module_pin:' + name)
    return pins


def progress(out, phase):
    """Append-only CPU breadcrumb before native calls, retained on crash."""
    with (out / 'progress.jsonl').open('a') as handle:
        handle.write(json.dumps({'phase': phase, 'wall_s': time.time()}) + '\n')
        handle.flush()


# --- CPU-testable pure evidence readers -----------------------------------

def _accounting():
    import activity_accounting
    return activity_accounting


def pending_evidence(native):
    """Classify every exact EventQuery row in a retained ``/2`` report.

    ``pending``/``success`` hold correlation ids of exact event-query rows;
    ``joined`` and ``with_activity`` prove whether any offer/activity consumed
    that query.  A pending query with joined GPU work is never accepted.
    """
    aa = _accounting()
    apis = native.get('apis') or []
    external = {row.get('correlation_id'): row.get('external_id')
                for row in native.get('external') or []}
    activity = {row.get('correlation_id')
                for row in native.get('activities') or []}
    rows = {'pending': [], 'success': [], 'refused': [], 'joined': [],
            'with_activity': []}
    for row in apis:
        name, value = row.get('name'), row.get('return_value')
        cid = row.get('correlation_id')
        kind = aa.host_return_class(name, value)
        if kind == 'pending':
            rows['pending'].append(cid)
        elif kind == 'success' and aa.is_host_query(name):
            rows['success'].append(cid)
        elif kind == 'refused' and aa.api_kind(name) == 'host':
            rows['refused'].append(cid)
        if cid in external:
            rows['joined'].append(cid)
        if cid in activity:
            rows['with_activity'].append(cid)
    return rows


def uid_lock_counts(receipt):
    """The qsa uid_init/lock_init leaf counts from a sealed prefill receipt."""
    families = receipt.get('families') or {}
    qsa = families.get('qsa') or {}
    return {'uid_init': dict(qsa.get('uid_init') or {}),
            'lock_init': dict(qsa.get('lock_init') or {})}


def _container_offers(receipt):
    return [row for row in receipt.get('offers') or []
            if row.get('expectation') == 'container']


def _joined_offers(receipt):
    return {row.get('offer') for row in receipt.get('joins') or []}


def pending_not_joined(evidence):
    """Every pending query is host-only: no external join and no activity."""
    return bool(evidence.get('pending')) and not (
        set(evidence['pending']) & set(evidence.get('joined') or ())) and not (
        set(evidence['pending']) & set(evidence.get('with_activity') or ()))


def case_grade(record):
    f = list(record.get('failures', []))
    m = list(record.get('missing', []))
    ownership = ownership_grade(record.get('model_ownership'))
    f.extend('model_ownership:' + x for x in ownership['failures'])
    m.extend('model_ownership:' + x for x in ownership['missing'])
    required = ['source_pins_verified', 'graph_bytes_stable', 'cleanup_ok',
                'contexts_disjoint', 'raw_unchanged', 'graph_output_exact',
                'private_scope_first', 'private_scope_repeat',
                'uid_bytes_exact', 'lock_zero', 'owners_restored',
                'decoder_contents_unchanged', 'sharing_preserved',
                'queried_completion']
    for key in required:
        if record.get(key) is False:
            f.append(key)
        elif record.get(key) is not True:
            m.append(key)
    receipt = record.get('prefill_receipt') or {}
    if not isinstance(receipt, dict) or not receipt:
        m.append('prefill_receipt')
    else:
        counts = uid_lock_counts(receipt)
        uid, lock = counts['uid_init'], counts['lock_init']
        if uid.get('kernel') != 2:
            f.append('uid_init_kernel_count')
        if uid.get('memcpy') != 2:
            f.append('uid_init_copy_count')
        if sum(int(v or 0) for v in lock.values()) != 1:
            f.append('lock_init_count:' + json.dumps(lock, sort_keys=True))
        containers = _container_offers(receipt)
        if len(containers) != 1:
            m.append('qsa_container_count')
        elif containers[0].get('id') in _joined_offers(receipt):
            f.append('qsa_container_has_activity')
        leaves = [row for row in receipt.get('offers') or []
                  if row.get('suboperation') in ('uid_init', 'lock_init')]
        if {row.get('suboperation') for row in leaves} != {'uid_init',
                                                           'lock_init'}:
            m.append('private_leaf_offers')
        elif containers and any(row.get('parent') != containers[0].get('id')
                                for row in leaves):
            f.append('private_leaf_parent')
        if receipt.get('verdict') != 'PASS':
            m.append('prefill_receipt_verdict')
            if receipt.get('verdict') == 'FAIL':
                f.extend('prefill:' + str(reason)
                         for reason in receipt.get('failures') or ())
            m.extend('prefill_missing:' + str(reason)
                     for reason in receipt.get('missing') or ())
    composed = record.get('composed')
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
        if (composed.get('conservation') or {}).get('proved') is not True:
            f.append('conservation_not_proved')
        whole = composed.get('whole') or {}
        if whole.get('complete') is not True:
            m.append('whole_raw_incomplete')
        if any(whole.get(key) for key in ('errors', 'overflow', 'open_scopes',
                                          'boundary_records')):
            m.append('whole_raw_loss')
        parts = composed.get('partitions') or {}
        for label in ('decode', 'prefill'):
            row = parts.get(label) or {}
            if row.get('verdict') != 'PASS':
                m.append(label + '_partition')
            if not row.get('activity_records'):
                m.append(label + '_empty')
        completion = (parts.get('prefill') or {}).get('completion') or {}
        submitted = completion.get('offers_submitted') or 0
        completed = completion.get('offers_completed') or 0
        if not submitted:
            m.append('offer_completion_missing')
        elif submitted != completed:
            m.append('offer_completion_unproven')
        owners = (composed.get('prefill') or {}).get('completion') or {}
        if 'completion' in (composed.get('prefill') or {}):
            owners = composed['prefill']['completion']
        unproven = owners.get('unproven_owners') if isinstance(owners, dict) \
            else None
        if unproven:
            m.append('owner_completion_unproven')
    raw = record.get('native_report')
    if isinstance(raw, dict) and raw.get('schema') == 's5ro-correlation/2':
        actual_pending = pending_evidence(raw)
        if actual_pending != record.get('pending_evidence'):
            m.append('pending_evidence_report_mismatch')
        evidence = actual_pending
    else:
        evidence = None
    if not isinstance(evidence, dict):
        m.append('pending_evidence')
    else:
        if not evidence.get('pending'):
            m.append('event_query_600_missing')
        elif not pending_not_joined(evidence):
            f.append('pending_query_joined_activity')
        if not evidence.get('success'):
            m.append('event_query_completion_missing')
    raw = record.get('native_report')
    if not isinstance(raw, dict) or raw.get('schema') != 's5ro-correlation/2':
        m.append('whole_report_retained')
    elif isinstance(composed, dict):
        if (raw.get('window') != (composed.get('whole') or {}).get('window')
                or raw.get('activity_records')
                != (composed.get('whole') or {}).get('activity_records')):
            f.append('whole_report_rewritten')
    method_pins = record.get('private_method_pins')
    methods = {'LanePrivateChunk._populate_uid_once',
               'LanePrivateBindings._initialize_lock_once'}
    if not isinstance(method_pins, list) or {row.get('method') for row in method_pins} != methods:
        m.append('private_method_source_pins')
    elif any(row.get('source_sha256') != record.get('source_pins', {}).get('lane_private.py')
             for row in method_pins):
        f.append('private_method_source_mismatch')
    try:
        import native_placement
        rebuilt = native_placement.compose_overlap(
            record['native_report'], record['offer_report'],
            decode_declared=record['decode_declared'],
            decode_window=record['decode_window'],
            prefill_declared=receipt['declared'], prefill_window=receipt['window'],
            prefill_receipt=receipt)
        if rebuilt != composed:
            m.append('composition_regrade_mismatch')
        if rebuilt.get('verdict') == 'FAIL':
            f.append('composition_regrade_fail')
        elif rebuilt.get('verdict') != 'PASS':
            m.append('composition_regrade_incomplete')
        actual_families = rebuilt.get('partitions', {}).get('prefill', {}).get('families', {})
        if actual_families != receipt.get('families'):
            f.append('private_families_native_mismatch')
    except (KeyError, TypeError, ValueError):
        m.append('composition_regrade_inputs')
    if not isinstance(record.get('window_receipt'), dict):
        m.append('raw_graph_receipt_missing')
    else:
        try:
            import native_graph_placement as ngp
            graph_grade = ngp.compose_native_placement(
                record['window_receipt'], record['native_report'], record['offer_report'],
                decode_declared=record['decode_declared'], decode_window=record['decode_window'],
                prefill_declared=receipt['declared'], prefill_window=receipt['window'],
                prefill_receipt=receipt)
            if graph_grade != record.get('graph_placement'):
                m.append('raw_graph_composition_regrade_mismatch')
            if graph_grade['verdict'] == 'FAIL':
                f.extend('raw_graph:' + reason for reason in graph_grade['failures'])
            elif graph_grade['verdict'] != 'PASS':
                m.extend('raw_graph:' + reason for reason in graph_grade['missing'])
        except (KeyError, TypeError, ValueError):
            m.append('raw_graph_composition_inputs')
    return s.verdict(f, m)


def grade(raw):
    f = list(raw.get('failures', []))
    m = list(raw.get('missing', []))
    cases = {}
    if [row.get('mode') for row in raw.get('cases', [])] != ['positive']:
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
    child = raw.get('regressions', {}).get('job2_child_scope') or {}
    if child.get('verdict') != 'INCONCLUSIVE':
        m.append('job2_child_scope_not_declared_inc')
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


# --- real objects (CPU-testable with controlled device leaves) ------------

class QsaAttention:
    """One local QSA layer with real storage and synthetic ownership metadata."""

    def __init__(self, original, *, layer_name, width, tokens, host_kv):
        self.layer_name = layer_name
        self.topk_indices_buffer = original
        self.indexer = NS(packed_output_width=width, skip_topk=False,
                          raw_key_cache=NS(prefix='side.raw'),
                          compressed_key_cache=NS(prefix='side.compressed'))
        self.impl = NS()
        self.kv_cache = host_kv
        self._qsa_host_kv = host_kv
        self._qsa_rowcache = None
        self._qsa_gpu_slots = object()
        self._qsa_kv_offload = True
        self._max_decode_query_len = 1
        self.tokens = tokens

    def _flash_e3_attention_pre(self, positions, hidden):
        return {'num_tokens': self.tokens}

    def _flash_e3_attention_post(self, state):
        return None


class LockOwner:
    def __init__(self, workspace):
        self.workspace = workspace


class Layer:
    def __init__(self, attn, lock):
        self.layer_type = 'full_attention'
        self.self_attn = attn
        self.ple = None
        self.lock = lock

    def named_modules(self):
        return (('', self), ('self_attn', self.self_attn), ('lock', self.lock))


class Model:
    """Synthetic model container: no forward is ever run."""

    def __init__(self, layers):
        self.layers = list(layers)
        self.start_layer = 0
        self.end_layer = len(self.layers)


def ownership_grade(proof):
    f, m = [], []
    required = ('pinned_class_module', 'pinned_model_source', 'legacy_width_refused',
                'direct_resolved', 'causal_resolved', 'conditional_resolved',
                'root_preserved', 'private_owners_exact', 'lane_forward_backbone',
                'stock_forward_unchanged', 'generated_methods_exact_inverse')
    if not isinstance(proof, dict):
        return dict(verdict='INCONCLUSIVE', failures=[], missing=['proof_missing'])
    for key in required:
        if proof.get(key) is False:f.append(key)
        elif proof.get(key) is not True:m.append(key)
    if (type(proof.get('hc_width')) is not int or proof['hc_width'] < 1
            or proof.get('hc_width') != proof.get('text_hc_width')):
        f.append('hc_width')
    return dict(verdict='FAIL' if f else 'INCONCLUSIVE' if m else 'PASS',
                failures=f, missing=m)


def pinned_owner_shells(torch_module, model, *, generated_dir=None):
    """Real pinned classes without constructors/weights; synthetic layer data.

    This is ownership/API proof only. Generated native methods are bound on
    the one instance, never used to run model arithmetic, and stock forward
    methods are neither replaced nor called.
    """
    import model_hooks as mh
    import native_startup as ns
    module = importlib.import_module('vllm.models.qwen4_exp.nvidia.model')
    path = Path(module.__file__).resolve()
    source = path.read_text()
    expected = mh.PINS['models/qwen4_exp/nvidia/model.py']
    if s.sha(path) != expected:raise RuntimeError('model_class_source_pin')
    classes = tuple(getattr(module, name) for name in (
        'Qwen4ExpModel', 'Qwen4ExpForCausalLM', 'Qwen4ExpForConditionalGeneration'))
    if any(cls.__module__ != module.__name__ for cls in classes):
        raise RuntimeError('model_class_module')
    originals = tuple(cls.forward for cls in classes)
    objects = tuple(object.__new__(cls) for cls in classes)
    for obj in objects:torch_module.nn.Module.__init__(obj)
    backbone, causal, wrapper = objects
    # Same field ownership as the pinned constructors; values/layers are the
    # pre-existing bounded VM13f synthetic fixture, not production weights.
    backbone.layers = model.layers
    backbone.start_layer, backbone.end_layer = model.start_layer, model.end_layer
    backbone.config = NS(hc_count=4, hidden_size=2560)
    causal.model, causal.config = backbone, backbone.config
    wrapper.language_model, wrapper.config = causal, NS(text_config=backbone.config)
    generated = mh.generate_one('models/qwen4_exp/nvidia/model.py', source)
    tree = ast.parse(generated)
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Qwen4ExpModel')
    methods = [n for n in owner.body if isinstance(n, ast.FunctionDef)
               and n.name.startswith('_flash_e3_')]
    if not methods:raise RuntimeError('generated_model_methods_missing')
    namespace = dict(module.__dict__)
    unit = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')],level=0), *methods], type_ignores=[])
    ast.fix_missing_locations(unit)
    filename = '<vm13g-pinned-native-methods>'
    if generated_dir is not None:
        generated_dir.mkdir(parents=True,exist_ok=True)
        filename = str(generated_dir/'model_native_methods.py')
        with open(filename,'x') as out:out.write(generated)
    exec(compile(unit,filename,'exec'),namespace)
    for method in methods:setattr(backbone,method.name,MethodType(namespace[method.name],backbone))
    refused = False
    try:ns._hc_width(wrapper)
    except ns.NativeStartupError:refused = True
    proof = dict(pinned_class_module=True, pinned_model_source=True,
        model_source_sha256=expected, legacy_width_refused=refused,
        direct_resolved=mh.local_text_model(backbone) is backbone,
        causal_resolved=mh.local_text_model(causal) is backbone,
        conditional_resolved=mh.local_text_model(wrapper) is backbone,
        hc_width=ns._hc_width(mh.local_text_model(wrapper)),
        text_hc_width=backbone.config.hc_count*backbone.config.hidden_size,
        stock_forward_unchanged=all(cls.forward is original for cls,original in zip(classes,originals)),
        generated_methods_exact_inverse=mh.revert_one('models/qwen4_exp/nvidia/model.py',generated)==source,
        classes=[cls.__name__ for cls in classes],
        scope='real class shells, synthetic layer/config metadata; no weights or model arithmetic')
    return wrapper, backbone, proof


class ForwardContextModule:
    """The UID namespace the runtime installs ``QSA_REQ_UID`` into."""

    def __init__(self, decode_uid):
        self._forward_context = None
        self.QSA_REQ_UID = decode_uid


def _chunk_type(lane_chunk):
    """Subclass the bundled ``LaneChunk`` at call time (module path is lazy)."""
    cached = getattr(_chunk_type, '_cls', None)
    if cached is not None and issubclass(cached, lane_chunk.LaneChunk):
        return cached

    class AdmittedChunk(lane_chunk.LaneChunk):
        """An admitted chunk surface over the real transaction/snapshot."""

        def __init__(self, runner, req_index, transaction, num_tokens):
            self.runner = runner
            self.req_index = req_index
            self.admission = lane_chunk.ChunkAdmission(
                transaction, transaction.chunk_end + 1, (1,) * num_tokens, (),
                ((1, 2),), state_epoch=1)
            self.forward_context = NS(attn_metadata=None)

        def forward_scope(self):
            import contextlib
            return contextlib.nullcontext(self.forward_context)

        def lane_inputs(self):
            return dict(positions='positions', input_ids='input_ids',
                        query_start_loc='query_start_loc',
                        ngram_context='ngram')

    _chunk_type._cls = AdmittedChunk
    return AdmittedChunk


def _set(tensor, index, value):
    writer = getattr(tensor, 'write', None)
    if callable(writer):
        writer(index, value)
    else:
        tensor[index] = value


def private_method_pins(bindings, lane_private, expected):
    """Verify the explicit source methods; no nonexistent shadow is claimed."""
    import inspect
    methods = (lane_private.LanePrivateChunk._populate_uid_once,
               type(bindings)._initialize_lock_once)
    result = []
    for method in methods:
        actual = s.sha(Path(inspect.getfile(method)))
        if actual != expected:
            raise RuntimeError('private_method_source_pin')
        result.append(dict(method=method.__qualname__, source_sha256=actual,
                           basis='explicit method; no instance shadow'))
    return result


def initialize_workspace(torch_module, workspace_module):
    """Use the pinned module's real reset/init API before private discovery."""
    workspace_module.reset_workspace_manager()
    workspace_module.init_workspace_manager(torch_module.device('cuda:0'),
                                           num_ubatches=1, num_lanes=2)
    manager = workspace_module.current_workspace_manager()
    if manager._num_ubatches != 1 or manager._num_lanes != 2 or manager._locked:
        raise RuntimeError('workspace_initialization_unproved')
    return manager


def drain_split_streams(split):
    """Synchronize both actual measured streams; return the deduplicated count."""
    streams = {int(stream.cuda_stream): stream
               for stream in (split.decode_stream, split.prefill_stream)}
    if len(streams) != 2 or 0 in streams:
        raise RuntimeError('split_consumer_streams_unproved')
    for stream in streams.values():
        stream.synchronize()
    return len(streams)


def collect_pending(record, native):
    """The scanner consumes the retained correlation /2 report, never graphs."""
    if native.get('schema') != 's5ro-correlation/2':
        raise RuntimeError('pending_native_report_schema')
    evidence = pending_evidence(native)
    record['pending_evidence'] = evidence
    record['pending_query_observed'] = bool(evidence['pending'])
    record['pending_query_not_joined'] = pending_not_joined(evidence)
    return evidence


def build_bindings(torch_module, *, workspace_module, prefill_stream=None,
                   chunk_tokens=CHUNK_TOKENS, width=WIDTH, req_index=REQ_INDEX,
                   seed=SEED, device='cuda:0', topk_rows=16,
                   topk_value=900, lock_value=12345, fund_stock_bytes=128,
                   request_uid='vm-job13g-request', owner_factory=pinned_owner_shells,
                   generated_dir=None):
    """Construct the real runtime/bindings over actual (or controlled) tensors.

    Called with real ``torch`` and the real vLLM workspace module on the VM;
    the CPU suite calls the identical path with controlled device leaves and
    the pinned ``WorkspaceManager`` source.  No engine, weights or forward.
    """
    import lane_chunk
    import lane_hooks
    import lane_private
    import lane_state
    import native_resources

    device = torch_module.device(device)
    uid = torch_module.full((4,), -2, dtype=torch_module.int32, device=device)
    decode_uid = torch_module.full((4,), -2, dtype=torch_module.int32,
                                   device=device)
    scratch = torch_module.empty((chunk_tokens * width * 4,),
                                 dtype=torch_module.uint8, device=device)
    slots = torch_module.full((chunk_tokens,), -2, dtype=torch_module.int32,
                              device=device)
    _set(slots, req_index, seed)
    host_kv = torch_module.empty((8,), dtype=torch_module.uint8,
                                 device=device)
    original_a = torch_module.full((topk_rows, width), topk_value,
                                   dtype=torch_module.int32, device=device)
    original_b = torch_module.full((topk_rows, width), topk_value + 1,
                                   dtype=torch_module.int32, device=device)
    lock_original = torch_module.full((4,), lock_value, dtype=torch_module.int32,
                                      device=device)
    lock = LockOwner(lock_original)
    attn_a = QsaAttention(original_a, layer_name='model.layers.0.attn',
                          width=width, tokens=chunk_tokens, host_kv=host_kv)
    attn_b = QsaAttention(original_b, layer_name='model.layers.1.attn',
                          width=width, tokens=chunk_tokens, host_kv=host_kv)
    model = Model([Layer(attn_a, lock), Layer(attn_b, lock)])
    root_model, model, ownership = owner_factory(torch_module, model,
                                               generated_dir=generated_dir)
    fc = ForwardContextModule(decode_uid)

    state = lane_state.WorkerState(0, engine_epoch=EPOCH)
    transaction = state.new_transaction(request_uid, 1, 0, chunk_tokens)
    rows = (lane_state.BatchRow(request_uid, 1, 0, 0, chunk_tokens, 0),)
    snapshot = lane_state.MetadataSnapshot.capture(
        transaction, rows, block_ids=(10, 11), slot_mappings=(0,))
    runner = NS(model=root_model, device=device, _fc_module=fc,
                max_num_tokens=chunk_tokens, _qsa_uid_slots=slots,
                _qsa_uid_next=seed,
                req_states=NS(req_id_to_index={transaction.request_uid:
                                               req_index}))
    runtime = lane_hooks.LaneRuntime(
        state,
        forward_context=native_resources._ForwardContextAdapter(fc),
        workspace=native_resources._WorkspaceLaneAdapter(workspace_module, 1),
        prefill_forward_context=NS(prefill=True), prefill_workspace=1,
        uid_namespace=fc, prefill_uid_buffer=uid)
    runtime._native_owned_buffers = (
        NS(name=lane_private.UID_BUFFER_NAME, tensor=uid),
        NS(name=lane_private.TOPK_SCRATCH_NAME, tensor=scratch))
    runtime.open_prefill(transaction, snapshot)

    manager = workspace_module.current_workspace_manager()
    with workspace_module.use_workspace_lane(0):
        manager._ensure_workspace_size(fund_stock_bytes)
    bindings = lane_private.LanePrivateBindings.discover(
        runner, runtime, chunk_tokens=chunk_tokens, torch_module=torch_module)
    workspace_bytes = bindings.prepare_workspace()
    manager.lock()
    chunk = _chunk_type(lane_chunk)(runner, req_index, transaction,
                                    chunk_tokens)
    import native_startup as ns
    native = ns.WorkerNative(runner, NS(prefill_stream=prefill_stream), None, None,
        None, NS(tp=None, pp=None), None, None, chunk_tokens, None, torch_module,
        private_bindings=bindings, text_model=model)
    forward = native.lane_forward(chunk.lane_inputs(), forward_context=chunk.forward_scope)
    ownership.update(root_preserved=runner.model is root_model,
        private_owners_exact=[owner.module for owner in bindings.owners]==[attn_a,attn_b],
        lane_forward_backbone=forward.model is model)
    return dict(torch=torch_module, workspace=workspace_module, runner=runner,
                model=model, root_model=root_model, model_ownership=ownership,
                state=state, runtime=runtime, fc=fc,
                transaction=transaction, snapshot=snapshot, uid=uid,
                decode_uid=decode_uid, scratch=scratch, slots=slots,
                original=(original_a, original_b), lock_original=lock_original,
                lock_owner=lock, attn=(attn_a, attn_b), bindings=bindings,
                workspace_bytes=workspace_bytes, chunk=chunk,
                prefill_stream=prefill_stream, lane_private=lane_private)


def new_chunk(bundle):
    import lane_chunk
    return _chunk_type(lane_chunk)(
        bundle['runner'], bundle['chunk'].req_index, bundle['transaction'],
        CHUNK_TOKENS)


def run_private_scope(bundle, ledger, chunk, transaction, *, op_seq,
                      layer_idx=0, torch_module=None):
    """Run the real scope methods inside one actual layer segment offer."""
    torch = torch_module if torch_module is not None else bundle['torch']
    stream = bundle.get('prefill_stream')
    with torch.cuda.stream(stream):
        with ledger.segment('layer', transaction=transaction, op_seq=op_seq,
                            branch='driver'):
            with bundle['bindings'].chunk_scope(chunk).scope(layer_idx):
                pass


def run_private_scopes(bundle, ledger, scopes, transaction, *, op_seq,
                       torch_module=None):
    """First and repeated scopes inside one actual layer segment offer."""
    torch = torch_module if torch_module is not None else bundle['torch']
    stream = bundle.get('prefill_stream')
    with torch.cuda.stream(stream):
        with ledger.segment('layer', transaction=transaction, op_seq=op_seq,
                            branch='driver'):
            for chunk, layer_idx in scopes:
                with bundle['bindings'].chunk_scope(chunk).scope(layer_idx):
                    pass


def _value(tensor):
    """Read one tensor to host after the native window has closed."""
    return tensor.detach().float().cpu() if hasattr(tensor, 'detach') else tensor


def _tensor_values(tensor):
    import torch
    return torch.flatten(_value(tensor)).tolist()


# --- the measured native window -------------------------------------------

def start_prefill_ledger(npo, bridge, *, run_id, identities, torch_module,
                         clock, prefill_stream):
    """Declare once; declare_streams owns the single lane owner mark."""
    ledger = npo.NativePrefillLedger(
        bridge, worker='rank0', rank=0, epoch=EPOCH,
        generation=GENERATION, route_generation=ROUTE,
        storage_epoch=STORAGE_EPOCH, run_id=run_id, window_id=WINDOW_ID,
        stream_identity=identities, torch_module=torch_module,
        clock=clock, env={PREFILL_ENV: '1', ENGINE_ENV: '1'})
    ledger.declare_streams(prefill_stream=prefill_stream)
    return ledger


def case(torch, ro, gc, cs, mh, native, receipt, modules, out):
    import activity_accounting as aa
    import e3_types
    import engine_offers as eo
    import native_placement as np_
    import native_prefill_offers as npo

    r = dict(mode='positive', failures=[], missing=[], source_pins={},
             shadow_pins=[], contexts={})
    progress(out, 'positive:begin')
    gl = session = split = ws_module = None
    graph_owner = 0
    bindings = None
    shadows = []
    decode = prefill = None
    proof = None
    run_id = RUN_ID
    try:
        import vllm.v1.worker.workspace as ws_module
        r['workspace_source_sha256'] = s.sha(
            Path(ws_module.__file__).resolve())
        if r['workspace_source_sha256'] != WORKSPACE_SHA256:
            raise RuntimeError('workspace_source_pin')
        r['source_pins'] = _pins(modules)
        r['source_pins_verified'] = True
        gl = GraphBinding(native / 'vmid_graph_probe.so')
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
        src = torch.full((1024,), 7, dtype=torch.int32, pin_memory=True)
        dst = torch.empty_like(src, pin_memory=True)
        scratch = torch.empty_like(src, pin_memory=True)
        a, b, z = [torch.zeros((1024,), dtype=torch.int32, device='cuda')
                   for _ in range(3)]

        # Warm allocation/library caches only.  The private scope is never run
        # here: the first measured scope must be the real first use.
        initialize_workspace(torch, ws_module)
        bundle = build_bindings(torch, workspace_module=ws_module,
                                prefill_stream=split.prefill_stream,
                                generated_dir=out/'generated')
        r['model_ownership'] = bundle['model_ownership']
        bindings = bundle['bindings']
        r['private_method_pins'] = private_method_pins(bindings, bundle['lane_private'], r['source_pins']['lane_private.py'])
        r['discovery'] = bindings.audit()
        r['workspace_bytes'] = bundle['workspace_bytes']
        torch.cuda.synchronize()

        # One real retained graph captured on the decode green stream.
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
        progress(out, 'positive:before_warm_replay')
        s.checked(replay_owned_graph(gl, graph_owner, graph_handle),
                  'warm_replay')
        progress(out, 'positive:after_warm_replay')
        torch.cuda.synchronize()

        # Queried CUPTI identities, before any offer of the measured window.
        contexts = {
            'prefill_main': identities.prime(split.prefill_stream,
                                             'prefill_main'),
            'decode_main': identities.prime(split.decode_stream, 'decode_main'),
        }
        r['contexts'] = contexts
        r['contexts_disjoint'] = bool(
            all(row['green'] for row in contexts.values())
            and contexts['prefill_main']['context_id']
            != contexts['decode_main']['context_id']
            and contexts['prefill_main']['stream_id']
            != contexts['decode_main']['stream_id'])

        progress(out, 'positive:before_bind_bank')
        session.bind_bank(window_probe)
        session.begin_window(route_generation=ROUTE,
                             descriptor=bank.pairs[0][0])
        start = bridge.window()
        tx = bundle['transaction']
        prefill = start_prefill_ledger(npo, bridge, run_id=run_id,
            identities=identities, torch_module=torch, clock=host.timestamp,
            prefill_stream=split.prefill_stream)

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

        bindings.attach_ledger(prefill)
        chunk_first = new_chunk(bundle)
        chunk_repeat = new_chunk(bundle)
        # First and repeated real private scopes inside one layer container.
        run_private_scopes(bundle, prefill,
                           [(chunk_first, 0), (chunk_repeat, 0)], tx,
                           op_seq=1, torch_module=torch)
        r['private_scope_first'] = True
        r['lock_initialized_after_first'] = len(bindings._lock_initialized)
        r['private_scope_repeat'] = True

        # A bounded finite delay inside a declared prefix offer.  The offer's
        # completion event is then queried for real while the stream is busy.
        sleep = getattr(torch.cuda, '_sleep', None)
        r['pending_delay_helper'] = callable(sleep)
        if callable(sleep):
            with torch.cuda.stream(split.prefill_stream):
                with prefill.segment('prefix', transaction=tx, op_seq=3):
                    sleep(SLEEP_CYCLES)
            progress(out, 'positive:before_pending_poll')
            prefill.poll_completions()
            progress(out, 'positive:after_pending_poll')
            event = torch.cuda.Event(enable_timing=False)
            event.record(split.prefill_stream)
            torch.cuda.synchronize()
            prefill.owner_finish('lane', events=(event,))
            prefill.poll_completions()
            r['queried_completion'] = bool(
                all(meta.get('completed') for meta in prefill._meta.values()
                    if meta.get('submitted')))
        else:
            r['missing'].append('pending_delay_helper')
            torch.cuda.synchronize()
            event = torch.cuda.Event(enable_timing=False)
            event.record(split.prefill_stream)
            torch.cuda.synchronize()
            prefill.owner_finish('lane', events=(event,))

        with decode.step('execute'):
            with decode.offer_scope('forward_dispatch', 'graph_replay',
                                    descriptor=desc, census=MIX):
                progress(out, 'positive:before_measured_replay')
                s.checked(replay_owned_graph(gl, graph_owner, graph_handle),
                          'replay')
                progress(out, 'positive:after_measured_replay')
        decode.deactivate()
        r['drained_streams'] = drain_split_streams(split)
        proof = ro.DrainedProof(
            worker_rank=0, worker_incarnation='vm-worker-1',
            engine_incarnation='synthetic-no-engine',
            route_generation=ROUTE, storage_epoch=STORAGE_EPOCH,
            relevant_consumers=r['drained_streams'], controller_drained=True)

        # Every value comparison runs after the native window has closed.
        progress(out, 'positive:before_window_end')
        raw_report = ro._receipt_dict(session.end_window(proof))
        progress(out, 'positive:after_window_end')
        r['window_receipt'] = raw_report

        uid_values = _tensor_values(bundle['uid'])
        r['uid_values'] = uid_values
        r['uid_bytes_exact'] = bool(
            uid_values[:1] == [SEED]
            and uid_values[1:] == [-2] * (len(uid_values) - 1))
        lock_values = _tensor_values(bindings.marlin_locks[0][4])
        r['lock_values'] = lock_values
        r['lock_zero'] = bool(lock_values and all(v == 0 for v in lock_values))
        r['owners_restored'] = bool(
            bundle['fc'].QSA_REQ_UID is bundle['decode_uid']
            and all(getattr(owner, 'topk_indices_buffer', None) is original
                    for owner, original in zip(bundle['attn'],
                                                bundle['original']))
            and bundle['lock_owner'].workspace is bundle['lock_original'])
        r['decoder_contents_unchanged'] = bool(
            _tensor_values(bundle['decode_uid']) == [-2] * 4
            and all(_tensor_values(original) ==
                    [topk_value] * (original.numel())
                    for original, topk_value in zip(
                        bundle['original'], (900, 901))))
        rows = {row[0]: row for row in bindings.marlin_locks}
        r['sharing_preserved'] = bool(
            len(bindings.marlin_locks) == 2
            and bindings.marlin_locks[0][3] is bindings.marlin_locks[1][3]
            and bindings.marlin_locks[0][4] is bindings.marlin_locks[1][4]
            and len({id(row[4]) for row in bindings.marlin_locks}) == 1)
        del rows

        decode.finish_window({'total': 1, 'graphs': {'graph:0': 1}})
        prefill.finish()
        receipt_prefill = prefill.seal()
        snapshot = prefill.snapshot()
        collect_pending(r, snapshot['native'])
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
            native_report=snapshot['native'], offer_report=snapshot['offers'],
            decode_declared={str(oid): dict(meta)
                             for oid, meta in decode._meta.items()},
            decode_window=decode.window(),
            raw_unchanged=(json.dumps(snapshot['native'], sort_keys=True)
                           == raw_before),
            offers={'decode': len(decode._meta),
                    'prefill': len(receipt_prefill['declared']),
                    'shared_report': len(snapshot['offers'])})
        import native_graph_placement as ngp
        r['graph_placement'] = ngp.compose_native_placement(
            raw_report, snapshot['native'], snapshot['offers'],
            decode_declared=r['decode_declared'], decode_window=r['decode_window'],
            prefill_declared=receipt_prefill['declared'], prefill_window=receipt_prefill['window'],
            prefill_receipt=receipt_prefill)
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
        drained = 0
        try:
            if split is not None:
                drained = drain_split_streams(split)
            else:
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
                    if drained != 2:
                        raise RuntimeError('window_cleanup_consumer_drain_unproved')
                    session.end_window(ro.DrainedProof(
                        worker_rank=0, worker_incarnation='vm-worker-1',
                        engine_incarnation='synthetic-no-engine',
                        route_generation=ROUTE,
                        storage_epoch=STORAGE_EPOCH,
                        relevant_consumers=drained, controller_drained=True))
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
        try:
            if ws_module is not None:
                ws_module.reset_workspace_manager()
        except Exception:
            errors.append('workspace_reset')
        r['cleanup_ok'] = not errors
        r['cleanup_errors'] = errors
        r['ended_wall_s'] = time.time()
    r['grade'] = case_grade(r)
    s.write(out / 'positive.json', r)
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
            # Resolve/import real class definitions before starting observation;
            # no model constructor, weights or engine is executed.
            importlib.import_module('vllm.models.qwen4_exp.nvidia.model')
            r['cases'].append(case(torch, ro, gc, cs, mh, native, receipt,
                                   modules, out))
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
    except (OSError, ValueError, TypeError, KeyError, ImportError) as exc:
        raw = {}
        g = s.verdict([], ['probe_result:' + type(exc).__name__])
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
    print('VMID_JOB13G ' + result['verdict'])
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
    if args.mode in ('probe', 'finalize'):
        modules = args.modules
        if modules is None:
            modules = Path(__file__).resolve().parent / 'e3_builder'
        if modules.is_dir():
            sys.path.insert(0, str(modules.resolve()))
    if args.mode == 'monitor':
        return s.monitor(args.out, 'flashnext-job13g')
    if args.mode == 'finalize':
        return finalize(args.out, args.process_rc)
    return probe(args.modules, args.native, args.out)


if __name__ == '__main__':
    sys.exit(main())
