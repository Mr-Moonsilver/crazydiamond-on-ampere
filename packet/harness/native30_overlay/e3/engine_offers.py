"""Source-pinned engine offer ledger for the short q17b placement window.

This module is deliberately split in two halves:

* runtime seams (``EngineOfferLedger``, ``run``, ``scoped_call``,
  ``offer_method``, ``step_method``, ``mark_zero_work``, ``run_graph_replay``)
  used by the generated model-runner/sampler seams. Offers are always created
  by :class:`activity_accounting.CorrelationBridge`; this module never builds
  an offer dict and never invents ``thread``/``system_thread``.
* offline transforms (``PINS``/``ACCEPTED``/``generate_one``/``revert_one``)
  for the pinned ``model_runner.py`` and ``sampler.py`` call sites.

The engine-accounting receipt is separate from the raw graph-only observer
receipt. Graph replay offers declare the retained graph-node census; graph
records are owned by the raw placement receipt, while every in-window eager
(graph-ID-zero) record must join exactly one declared engine offer.
"""
import ast
from collections import Counter, defaultdict
from contextlib import contextmanager
import contextvars
import functools
import hashlib
import json
import os
import sys
import threading

try:  # imported from the generated package when installed there
    from activity_accounting import (KINDS, api_kind as _api_kind,
                                     host_return_ok as _host_return_ok)
except ImportError:  # pragma: no cover - package-relative import
    from .activity_accounting import (KINDS, api_kind as _api_kind,
                                      host_return_ok as _host_return_ok)

SCHEMA = 's5-engine-accounting/1'
NATIVE_SCHEMA = 's5ro-correlation/2'
NATIVE_SCHEMA_LEGACY = 's5ro-correlation/1'
PLACEMENT_SCHEMA = 's5ro-replay-observer/1'
COMPOSED_SCHEMA = 's5-placement-composed/1'
ENABLE_ENV = 'FLASH_S5_ENGINE_OFFERS'
MAX_OFFERS = 32768
MAX_STEPS = 65536
MAX_GAPS = 4096
MAX_RECEIPT_BYTES = 16 * 1024 * 1024
THREAD_ID_TYPES = ('default-pthread-low32', 'system-gettid')

# CUPTI_ACTIVITY_MEMCPY_KIND_* values from the pinned CUPTI 13 header.
COPY_KIND = {'unknown': 0, 'H2D': 1, 'D2H': 2, 'H2A': 3, 'A2H': 4,
             'A2A': 5, 'A2D': 6, 'D2A': 7, 'D2D': 8, 'H2H': 9, 'P2P': 10}
ALL_COPY_KINDS = tuple(sorted(set(COPY_KIND.values())))
STREAM_ROLES = ('main', 'pp', 'output_copy')
STEP_KINDS = ('execute', 'sample')

_EXPECTATIONS = ('activity', 'activity_or_zero', 'zero_work', 'census',
                 'container', 'unsupported')

_ALL = tuple(KINDS)
_ZERO = ()
_NONE = ()

# --- vocabulary -------------------------------------------------------------
# Families and suboperations are the plan's table, declared before measuring.
# ``expectation`` means:
#   activity          at least one joined eager record for the offer
#   activity_or_zero  zero or more joined records (host-only/no-op branch)
#   zero_work         exactly zero joined records, explicitly declared
#   census            graph replay; the retained graph-node census is declared
#   container         no records of its own; nested offers carry the work
#   unsupported       an active, unimplemented branch. Its records are
#                     consumed so they cannot be mistaken for another family,
#                     but the offer always makes the window INCONCLUSIVE.
# ``kinds`` is always explicit. ``copy_kinds`` is always explicit whenever
# memcpy is an allowed kind; no declaration inherits an all-kinds default.
# ``supported_branches`` is a host-only declaration used only for branches
# whose activity shape depends on the installed object. ``noop_branch`` names
# the branch that is known to submit no GPU work.
FAMILIES = {
    'request_state_table': {
        'suboperations': {
            'prep_state': {'expectation': 'activity_or_zero',
                           'kinds': _ALL, 'copy_kinds': ALL_COPY_KINDS},
            # Pinned ordinary path is CPU/numpy request state; the adaptive
            # branch is host-only too (numpy in adaptive_verification.py).
            'gather_batch_req_state': {'expectation': 'activity_or_zero',
                                       'kinds': _ZERO,
                                       'supported_branches': (
                                           'dummy', 'ordinary', 'adaptive')},
            'no_work_step': {'expectation': 'zero_work', 'kinds': _ZERO},
        },
    },
    'input_attention': {
        'suboperations': {
            # runner.prepare_inputs (model_runner.py:1425-1622): async_copy_to_gpu
            # H2D staging, device torch.arange/zeros, the prepare_prefill_inputs/
            # prepare_pos_seq_lens/combine_sampled_and_draft_tokens kernels and
            # the prompt_len.gpu[idx_mapping] gather.
            'prepare_inputs': {'expectation': 'activity_or_zero',
                               'kinds': ('kernel', 'memcpy', 'memset'),
                               'copy_kinds': (COPY_KIND['H2D'],)},
            # runner.prepare_attn (model_runner.py:1624): _qsa_write_req_uids
            # fill_/gather/D2D copy_, then the _gather_block_tables_kernel and
            # _compute_slot_mappings_kernel Triton launches (block_table.py).
            'prepare_attn': {'expectation': 'activity_or_zero',
                             'kinds': ('kernel', 'memcpy'),
                             'copy_kinds': (COPY_KIND['D2D'],),
                             'supported_branches': ('ordinary', 'pcp')},
            # model_state.prepare_attn stages attention metadata on GPU.
            'attn_metadata': {'expectation': 'activity_or_zero',
                              'kinds': ('kernel', 'memcpy', 'memset'),
                              'copy_kinds': (COPY_KIND['D2D'],
                                             COPY_KIND['H2D'])},
            'preprocess_state': {'expectation': 'activity_or_zero',
                                 'kinds': ('kernel', 'memcpy', 'memset'),
                                 'copy_kinds': (COPY_KIND['D2D'],)},
        },
    },
    'forward_dispatch': {
        'suboperations': {
            # This is the whole dispatch_cg_and_sync_dp call: descriptor
            # selection is host-side, DP padding is a CPU all_reduce on the
            # pinned dp_utils path, and a reuse may allocate/fill a GPU
            # tensor. It is a composite leaf, not a catch-all.
            # Descriptor selection and DP sync are host-side in the pinned
            # dp_utils path (CPU all_reduce; DP1 returns before any collective).
            'dp_dispatch': {'expectation': 'activity_or_zero',
                            'kinds': _ZERO,
                            'supported_branches': ('dp1', 'collective',
                                                   'reuse')},
            # Pinned eplb_utils.prepare_forward returns without CUDA work when
            # EPLB is disabled. An enabled EPLB is not reviewed here.
            'eplb_prepare': {'expectation': 'activity_or_zero',
                             'kinds': _ZERO},
            'prepare_inputs_embeds': {'expectation': 'activity_or_zero',
                                      'kinds': ('kernel', 'memcpy', 'memset'),
                                      'copy_kinds': (COPY_KIND['H2D'],
                                                     COPY_KIND['D2D'])},
            'prepare_model_inputs': {'expectation': 'activity_or_zero',
                                     'kinds': ('kernel', 'memcpy', 'memset'),
                                     'copy_kinds': (COPY_KIND['D2D'],)},
            # PP intermediate_tensors .copy_ on the main stream.
            'pp_intermediate_copy': {'expectation': 'activity_or_zero',
                                     'kinds': ('memcpy',),
                                     'copy_kinds': (COPY_KIND['D2D'],)},
            # Stock run_fullgraph's offloader sync. The installed offloader
            # implementation is not in the pinned subset, so this branch is
            # explicitly unsupported rather than guessed host-only/copy.
            'offloader_sync': {'expectation': 'unsupported',
                               'kinds': _ALL,
                               'copy_kinds': ALL_COPY_KINDS,
                               'supported_branches': ('stock_unreviewed',)},
            'graph_replay': {'expectation': 'census', 'graph': True,
                             'kinds': _ALL, 'copy_kinds': ALL_COPY_KINDS},
        },
    },
    'pp': {
        'suboperations': {
            'pp_receive': {'expectation': 'activity_or_zero',
                           'kinds': ('kernel', 'memcpy', 'memset'),
                           'copy_kinds': ALL_COPY_KINDS,
                           'stream_role': 'pp', 'requires_pp': True},
            # pinned pp_utils.py: only receive/broadcast enter
            # ``torch.cuda.stream(self.broadcast_stream)``; these two runner
            # calls launch on the current (main/decode) stream.
            'postprocess_num_computed': {'expectation': 'activity_or_zero',
                                         'kinds': ('kernel', 'memcpy'),
                                         'copy_kinds': (COPY_KIND['D2D'],),
                                         'requires_pp': True},
            'postprocess_state': {'expectation': 'activity_or_zero',
                                  'kinds': ('kernel', 'memcpy', 'memset'),
                                  'copy_kinds': (COPY_KIND['D2D'],),
                                  'requires_pp': True},
            'pp_broadcast': {'expectation': 'activity_or_zero',
                             'kinds': ('kernel', 'memcpy', 'memset'),
                             'copy_kinds': ALL_COPY_KINDS,
                             'stream_role': 'pp', 'requires_pp': True},
            'pp_broadcast_drafts': {'expectation': 'activity_or_zero',
                                    'kinds': ('kernel', 'memcpy', 'memset'),
                                    'copy_kinds': ALL_COPY_KINDS,
                                    'stream_role': 'pp', 'requires_pp': True},
        },
    },
    # Orchestration containers. They must have no GPU submission of their own.
    'logits_sampler': {
        'suboperations': {
            'runner_sample': {'expectation': 'container', 'kinds': _ZERO},
            'sampler_call': {'expectation': 'container', 'kinds': _ZERO,
                              'supported_branches': ('active', 'empty_owner',
                                                     'rejection')},
            'sampler_entry': {'expectation': 'container', 'kinds': _ZERO},
            'sampler_sample': {'expectation': 'container', 'kinds': _ZERO},
        },
    },
    # Leaves inside model_runner.sample (pinned 1666-1742).
    'logits': {
        'suboperations': {
            'select_hidden': {'expectation': 'activity',
                              'kinds': ('kernel', 'memcpy'),
                              'copy_kinds': (COPY_KIND['D2D'],)},
            'compute_local': {'expectation': 'activity', 'kinds': ('kernel',)},
            'compute_full': {'expectation': 'activity', 'kinds': ('kernel',)},
            'tp_all_to_all': {'expectation': 'activity',
                              'kinds': ('kernel', 'memcpy'),
                              'copy_kinds': (COPY_KIND['A2A'],)},
            'grammar': {'expectation': 'activity_or_zero',
                        'kinds': ('kernel', 'memcpy'),
                        'copy_kinds': (COPY_KIND['H2D'],),
                        'supported_branches': ('active', 'absent')},
        },
    },
    # Leaves inside sample/sampler.py and the runner's shard/gather helpers.
    'sampler': {
        'suboperations': {
            'shard_inputs': {'expectation': 'activity_or_zero',
                             'kinds': ('kernel', 'memcpy'),
                             'copy_kinds': (COPY_KIND['D2D'],),
                             'supported_branches': ('active', 'no_sharder')},
            'gather_output': {'expectation': 'activity_or_zero',
                              'kinds': ('kernel', 'memcpy', 'memset'),
                              'copy_kinds': ALL_COPY_KINDS,
                              'supported_branches': ('sharded', 'unsharded')},
            'select_inputs': {'expectation': 'activity',
                              'kinds': ('kernel', 'memcpy'),
                              'copy_kinds': (COPY_KIND['D2D'],)},
            'nan_check': {'expectation': 'activity_or_zero',
                          'kinds': ('kernel',),
                          'supported_branches': ('computed', 'disabled')},
            'apply_params': {'expectation': 'activity_or_zero',
                             'kinds': ('kernel', 'memcpy'),
                             'copy_kinds': (COPY_KIND['D2D'],),
                             'supported_branches': ('none', 'processing')},
            'topk_topp_inputs': {'expectation': 'activity_or_zero',
                                  'kinds': ('kernel', 'memcpy'),
                                  'copy_kinds': (COPY_KIND['D2D'],)},
            'draw': {'expectation': 'activity', 'kinds': ('kernel',),
                     'supported_branches': ('flashinfer', 'gumbel')},
            'trace_apply': {'expectation': 'activity_or_zero',
                            'kinds': ('kernel', 'memcpy', 'memset'),
                            'copy_kinds': (COPY_KIND['D2D'],),
                            'supported_branches': ('applied', 'absent')},
            'topk_logprobs': {'expectation': 'activity_or_zero',
                              'kinds': ('kernel', 'memcpy', 'memset'),
                              'copy_kinds': (COPY_KIND['D2D'],),
                              'supported_branches': ('requested',
                                                     'not_requested')},
            'completion_counts': {'expectation': 'activity_or_zero',
                                  'kinds': ('kernel', 'memset')},
            'sampling_mask': {'expectation': 'activity_or_zero',
                              'kinds': ('kernel',),
                              'supported_branches': ('enabled', 'disabled')},
            'rejection': {'expectation': 'unsupported',
                          'kinds': _ALL, 'copy_kinds': ALL_COPY_KINDS},
            'prompt_logprobs': {'expectation': 'activity_or_zero',
                                'kinds': ('kernel', 'memcpy', 'memset'),
                                'copy_kinds': (COPY_KIND['D2D'],
                                               COPY_KIND['D2H'])},
        },
    },
    'speculator': {
        'suboperations': {
            # The draft graph census is not installed in this pass; an active
            # MTP/draft path is explicit missing coverage, never target census.
            'propose': {'expectation': 'unsupported',
                        'kinds': _ALL, 'copy_kinds': ALL_COPY_KINDS},
            'draft_writeback': {'expectation': 'activity_or_zero',
                                'kinds': ('kernel', 'memcpy', 'memset'),
                                'copy_kinds': (COPY_KIND['D2D'],)},
            'watermarking': {'expectation': 'activity_or_zero',
                             'kinds': ('kernel', 'memcpy', 'memset'),
                             'copy_kinds': ALL_COPY_KINDS},
            'adaptive_record': {'expectation': 'activity_or_zero',
                                'kinds': ('kernel', 'memcpy', 'memset'),
                                'copy_kinds': ALL_COPY_KINDS},
            'set_draft_tokens': {'expectation': 'activity_or_zero',
                                 'kinds': ('kernel', 'memcpy'),
                                 'copy_kinds': (COPY_KIND['D2D'],)},
        },
    },
    'kv_connector': {
        'suboperations': {
            # Installed stock KVConnector methods are exact no-ops. An active
            # connector is not qualified: its branch is named missing, and
            # any activity it submits is reported but cannot make PASS.
            'pre_forward': {'expectation': 'activity_or_zero',
                            'kinds': _ALL, 'copy_kinds': ALL_COPY_KINDS,
                            'supported_branches': ('noop',),
                            'noop_branch': 'noop'},
            'finish_forward': {'expectation': 'activity_or_zero',
                               'kinds': _ALL, 'copy_kinds': ALL_COPY_KINDS,
                               'supported_branches': ('noop',),
                               'noop_branch': 'noop'},
            'post_forward': {'expectation': 'activity_or_zero',
                             'kinds': _ALL, 'copy_kinds': ALL_COPY_KINDS,
                             'supported_branches': ('noop',),
                             'noop_branch': 'noop'},
            'no_forward': {'expectation': 'activity_or_zero',
                           'kinds': _ALL, 'copy_kinds': ALL_COPY_KINDS,
                           'supported_branches': ('noop',),
                           'noop_branch': 'noop'},
            'ec_no_forward': {'expectation': 'activity_or_zero',
                              'kinds': _ALL,
                              'copy_kinds': ALL_COPY_KINDS,
                              'supported_branches': ('noop',),
                              'noop_branch': 'noop'},
        },
    },
    'output_copy': {
        'suboperations': {
            # AsyncOutput's stream scope: ordinary branch is D2H copies.
            'async_output': {'expectation': 'activity_or_zero',
                             'stream_role': 'output_copy',
                             'kinds': ('memcpy', 'memset'),
                             'copy_kinds': (COPY_KIND['D2H'],),
                             'supported_branches': ('ordinary',)},
            # check_ep_fault/query_fault is a separate conditional operation.
            'ep_fault_query': {'expectation': 'activity_or_zero',
                               'stream_role': 'output_copy',
                               'kinds': ('kernel', 'memcpy'),
                               'copy_kinds': (COPY_KIND['D2H'],
                                              COPY_KIND['D2D']),
                               'supported_branches': ('fault_check',
                                                      'disabled')},
        },
    },
    'postprocess': {
        'suboperations': {
            'execute_postprocess': {'expectation': 'container', 'kinds': _ZERO},
            'postprocess_sampled': {'expectation': 'activity_or_zero',
                                    'kinds': ('kernel', 'memcpy', 'memset'),
                                    'copy_kinds': (COPY_KIND['D2D'],)},
            'postprocess_state': {'expectation': 'activity_or_zero',
                                  'kinds': ('kernel', 'memcpy', 'memset'),
                                  'copy_kinds': (COPY_KIND['D2D'],)},
        },
    },
    # Never a catch-all: an explicit own helper site must issue this offer.
    'diagnostics': {
        'suboperations': {
            'sentinel': {'expectation': 'zero_work', 'kinds': _ZERO,
                         'exclusive': True},
        },
    },
}

# The main stream role carries model/decode compute and must be green. Only
# these family/suboperation pairs may run on a declared non-green control role.
_NON_GREEN_LEAVES = {
    'output_copy.async_output': 'output_copy',
    'output_copy.ep_fault_query': 'output_copy',
    'pp.pp_receive': 'pp',
    'pp.pp_broadcast': 'pp',
    'pp.pp_broadcast_drafts': 'pp',
}

_FORBIDDEN_FAMILIES = ('unknown', 'other', 'catch_all', 'default', 'misc')


class OfferError(RuntimeError):
    """Base class for engine-ledger refusals."""


class OfferDisabled(OfferError):
    pass


class OfferCapacityExceeded(OfferError):
    pass


class OfferPopIdentity(OfferError):
    pass


class GenerationMismatch(OfferError):
    pass


class VocabularyError(OfferError):
    pass


def _enabled(env=None):
    env = os.environ if env is None else env
    return env.get(ENABLE_ENV, '0') == '1'


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _validate_vocabulary():
    for family, body in FAMILIES.items():
        if family in _FORBIDDEN_FAMILIES or not isinstance(family, str):
            raise VocabularyError('reserved family')
        subs = body.get('suboperations')
        if not isinstance(subs, dict) or not subs:
            raise VocabularyError('family without suboperations: ' + family)
        for sub, decl in subs.items():
            exp = decl.get('expectation')
            if exp not in _EXPECTATIONS:
                raise VocabularyError('bad expectation: ' + family + '.' + sub)
            if 'kinds' not in decl:
                raise VocabularyError('explicit kinds required: %s.%s'
                                      % (family, sub))
            kinds = decl['kinds']
            if not isinstance(kinds, tuple) or not set(kinds) <= set(KINDS):
                raise VocabularyError('bad kinds: ' + family + '.' + sub)
            ck = decl.get('copy_kinds')
            if 'memcpy' in kinds:
                if not isinstance(ck, tuple) or not ck \
                        or not set(ck) <= set(ALL_COPY_KINDS):
                    raise VocabularyError(
                        'explicit copy directions required: %s.%s'
                        % (family, sub))
            elif ck:
                raise VocabularyError('copy kinds without memcpy: %s.%s'
                                      % (family, sub))
            role = decl.get('stream_role', 'main')
            if role not in STREAM_ROLES:
                raise VocabularyError('bad stream role: ' + family + '.' + sub)
            want_role = _NON_GREEN_LEAVES.get(family + '.' + sub, 'main')
            if role != want_role:
                raise VocabularyError('stream role is not source-authorized: '
                                      '%s.%s' % (family, sub))
            if exp in ('zero_work', 'container') and kinds:
                raise VocabularyError('zero-work/container declares kinds')
            if exp == 'census' and not decl.get('graph'):
                raise VocabularyError('census without graph')
            if decl.get('graph') and exp != 'census':
                raise VocabularyError('graph offer is not census')
            if family != 'diagnostics' and decl.get('exclusive'):
                raise VocabularyError('exclusive suboperation outside diagnostics')
            if exp == 'unsupported' and not kinds:
                raise VocabularyError('unsupported branch without kinds')
            branches = decl.get('supported_branches')
            if branches is not None and (
                    not isinstance(branches, tuple) or not branches):
                raise VocabularyError('bad supported branches: %s.%s'
                                      % (family, sub))
            noop = decl.get('noop_branch')
            if noop is not None and noop not in (branches or ()):
                raise VocabularyError('noop branch not supported: %s.%s'
                                      % (family, sub))


_validate_vocabulary()


def _manifest_body():
    out = {}
    for family in sorted(FAMILIES):
        subs = FAMILIES[family]['suboperations']
        out[family] = {
            key: {
                'expectation': decl['expectation'],
                'kinds': list(decl['kinds']),
                'copy_kinds': (None if decl.get('copy_kinds') is None
                               else list(decl['copy_kinds'])),
                'stream_role': decl.get('stream_role', 'main'),
                'graph': bool(decl.get('graph', False)),
                'requires_pp': bool(decl.get('requires_pp', False)),
                'supported_branches': (None if decl.get('supported_branches')
                                       is None else
                                       list(decl['supported_branches'])),
            }
            for key, decl in sorted(subs.items())
        }
    return out


def family_manifest_identity():
    """The declared family-manifest identity bound before the window."""
    body = _manifest_body()
    encoded = json.dumps(body, sort_keys=True, separators=(',', ':'))
    return {'sha256': _sha(encoded),
            'families': sorted(FAMILIES)}


FAMILY_MANIFEST_SHA256 = family_manifest_identity()['sha256']


def _digest(value, reason):
    if (type(value) is not str or len(value) != 64
            or any(c not in '0123456789abcdef' for c in value)):
        raise OfferError(reason)
    return value


def _bounded_text(value, reason, limit=128):
    if (type(value) is not str or not value or len(value) > limit
            or any(ord(c) < 32 for c in value)):
        raise OfferError(reason)
    return value


def _strict_int(value, reason, minimum=0):
    if type(value) is not int or value < minimum:
        raise OfferError(reason)
    return value


def bank_identity(bank):
    """Stable identity of the retained bank record, never inferred."""
    if not isinstance(bank, dict):
        raise OfferError('binding_bank_shape')
    keys = ('storage_epoch', 'route_generation', 'descriptor_count',
            'split_decode_sms', 'split_requested_sms',
            'green_context_bound', 'capture_order_labels')
    body = {key: bank.get(key) for key in keys}
    return _sha(json.dumps(body, sort_keys=True, separators=(',', ':')))


def source_identities_sha256(source_identities):
    if not isinstance(source_identities, dict):
        raise OfferError('binding_source_identities')
    return _sha(json.dumps(source_identities, sort_keys=True,
                           separators=(',', ':')))


class WindowBinding:
    """The single WorkerObserver binding shared by both receipts.

    Identity/start are bound before offers. End and router replay delta are
    finalized after the drained window; neither is predicted in advance. The raw graph receipt remains
    byte-for-byte untouched; this record is retained beside it.
    """
    SCHEMA = 's5-window-binding/1'

    def __init__(self, *, run_id, window_id, worker, rank, worker_incarnation,
                 engine_incarnation, engine_epoch, route_generation,
                 storage_epoch, green_context_id, native_start_ns,
                 native_end_ns, observer_identity, helper_identity,
                 declared_target, descriptor_labels, router_census,
                 bank_identity_sha256, family_manifest=None,
                 stream_roles=None, pp_stage=None):
        self.run_id = _bounded_text(run_id, 'binding_run_id')
        self.window_id = _bounded_text(window_id, 'binding_window_id')
        self.worker = _bounded_text(worker, 'binding_worker')
        self.rank = _strict_int(rank, 'binding_rank')
        self.worker_incarnation = _bounded_text(worker_incarnation,
                                                'binding_worker_incarnation')
        self.engine_incarnation = _bounded_text(engine_incarnation,
                                                'binding_engine_incarnation')
        self.engine_epoch = _strict_int(engine_epoch, 'binding_engine_epoch', 1)
        self.route_generation = _strict_int(route_generation,
                                            'binding_route_generation', 1)
        self.storage_epoch = _strict_int(storage_epoch,
                                         'binding_storage_epoch', 1)
        self.green_context_id = _strict_int(green_context_id,
                                            'binding_green_context', 1)
        self.native_start_ns = _strict_int(native_start_ns,
                                           'binding_window_start')
        self.native_end_ns = (None if native_end_ns is None else
                              _strict_int(native_end_ns, 'binding_window_end'))
        if (native_end_ns is None) != (router_census is None):
            raise OfferError('binding_partial_finalization')
        if native_end_ns is not None and native_end_ns <= self.native_start_ns:
            raise OfferError('binding_window_order')
        if not isinstance(observer_identity, dict):
            raise OfferError('binding_observer_identity')
        for key in ('abi', 'library_sha256', 'source_sha256',
                    'cupti_library_sha256'):
            if key not in observer_identity:
                raise OfferError('binding_observer_identity')
        self.observer_identity = json.loads(json.dumps(observer_identity))
        if not isinstance(helper_identity, dict):
            raise OfferError('binding_helper_identity')
        self.helper_identity = json.loads(json.dumps(helper_identity))
        self.declared_target = _bounded_text(declared_target,
                                             'binding_declared_target')
        if not isinstance(descriptor_labels, dict) or not descriptor_labels:
            raise OfferError('binding_descriptor_labels')
        self.descriptor_labels = {}
        for label, key in descriptor_labels.items():
            self.descriptor_labels[_bounded_text(
                label, 'binding_descriptor_label')] = _bounded_text(
                    key, 'binding_descriptor_key', 512)
        if self.declared_target not in self.descriptor_labels:
            raise OfferError('binding_declared_target')
        self.router_census = None
        if router_census is not None:
            if not isinstance(router_census, dict):
                raise OfferError('binding_router_census')
            total = _strict_int(router_census.get('total'),
                                'binding_router_census', 1)
            graphs = router_census.get('graphs')
            if not isinstance(graphs, dict) or not graphs:
                raise OfferError('binding_router_census')
            clean_graphs = {}
            for label, count in graphs.items():
                label = _bounded_text(label, 'binding_router_label')
                clean_graphs[label] = _strict_int(count,
                                                  'binding_router_count', 1)
            if total != sum(clean_graphs.values()):
                raise OfferError('binding_router_census')
            self.router_census = {'total': total, 'graphs': clean_graphs}
        self.bank_identity = _digest(bank_identity_sha256,
                                     'binding_bank_identity')
        manifest = family_manifest_identity()
        if family_manifest is not None:
            if not isinstance(family_manifest, dict):
                raise OfferError('binding_family_manifest')
            if (family_manifest.get('sha256') != manifest['sha256']
                    or family_manifest.get('families')
                    != manifest['families']):
                raise OfferError('binding_family_manifest')
        self.family_manifest = manifest
        if stream_roles is not None:
            self.stream_roles = EngineOfferLedger._validate_identities(
                stream_roles)
        else:
            self.stream_roles = {}
        self.pp_stage = None if pp_stage is None else int(pp_stage)

    def as_record(self):
        return {
            'schema': self.SCHEMA,
            'run_id': self.run_id,
            'window_id': self.window_id,
            'worker': self.worker,
            'rank': self.rank,
            'worker_incarnation': self.worker_incarnation,
            'engine_incarnation': self.engine_incarnation,
            'engine_epoch': self.engine_epoch,
            'route_generation': self.route_generation,
            'storage_epoch': self.storage_epoch,
            'green_context_id': self.green_context_id,
            'native_start_ns': self.native_start_ns,
            'native_end_ns': self.native_end_ns,
            'observer_identity': self.observer_identity,
            'helper_identity': self.helper_identity,
            'declared_target': self.declared_target,
            'descriptor_labels': dict(self.descriptor_labels),
            'router_census': (None if self.router_census is None else
                              {'total': self.router_census['total'],
                               'graphs': dict(self.router_census['graphs'])}),
            'bank_identity_sha256': self.bank_identity,
            'family_manifest': self.family_manifest,
            'stream_roles': {role: dict(row)
                             for role, row in self.stream_roles.items()},
            'pp_stage': self.pp_stage,
        }


def bind_window(**kwargs):
    return WindowBinding(**kwargs).as_record()


def _coerce_binding(binding):
    if isinstance(binding, WindowBinding):
        return binding
    if isinstance(binding, dict):
        if binding.get('schema') != WindowBinding.SCHEMA:
            raise OfferError('binding_schema')
        plain = dict(binding)
        plain.pop('schema', None)
        plain['family_manifest'] = plain.get('family_manifest')
        plain['stream_roles'] = plain.get('stream_roles') or None
        plain['pp_stage'] = plain.get('pp_stage')
        return WindowBinding(**plain)
    raise OfferError('binding_shape')


def _decl(family, suboperation):
    body = FAMILIES.get(family)
    if body is None:
        raise VocabularyError('undeclared family: ' + str(family))
    decl = body['suboperations'].get(suboperation)
    if decl is None:
        raise VocabularyError('undeclared suboperation: %s.%s' %
                              (family, suboperation))
    return decl


def _wire(family, suboperation):
    return family + '.' + suboperation


def _split_wire(value):
    if not isinstance(value, str) or value.count('.') != 1:
        raise VocabularyError('undeclared wire family')
    family, sub = value.split('.', 1)
    _decl(family, sub)
    return family, sub


# --- runtime seam helpers ---------------------------------------------------
_ACTIVE = None
_ACTIVE_LOCK = threading.Lock()
_CENSUS_LOCK = threading.Lock()


def current_ledger(owner=None):
    """The ledger installed on ``owner``, else the process active ledger."""
    ledger = None
    if owner is not None:
        ledger = getattr(owner, '_flash_s5_engine_ledger', None)
    if ledger is not None and getattr(ledger, 'active', False):
        return ledger
    with _ACTIVE_LOCK:
        ledger = _ACTIVE
    if ledger is not None and getattr(ledger, 'active', False):
        return ledger
    return None


def _safe_descriptor(descriptor):
    """Only bounded scalars; never request text, tokens or pointers."""
    if descriptor is None:
        return None
    allowed = ('num_tokens', 'num_reqs', 'num_toks', 'has_lora', 'uniform',
               'pp_stage', 'graph_id', 'uniform_token_count', 'max_query_len',
               'num_active_loras', 'num_ubatches')
    out = {}
    for key in allowed:
        if key not in descriptor:
            continue
        value = descriptor[key]
        if value is None:
            continue
        if isinstance(value, bool):
            out[key] = value
        elif type(value) is int and abs(value) < (1 << 63):
            out[key] = value
        else:
            raise VocabularyError('descriptor value is not a bounded scalar: ' +
                                  key)
    return out


def descriptor_key(descriptor):
    return json.dumps(descriptor, sort_keys=True, separators=(',', ':'))


def pp_branch(owner):
    first = bool(getattr(owner, 'is_first_pp_rank', False))
    last = bool(getattr(owner, 'is_last_pp_rank', False))
    if not hasattr(owner, 'is_first_pp_rank') and not hasattr(
            owner, 'is_last_pp_rank'):
        return 'unknown'
    if first and last:
        return 'single'
    if first:
        return 'first'
    if last:
        return 'last'
    return 'middle'


def gather_branch(owner, scheduler_output, dummy_run):
    """Source branch of ``gather_batch_req_state`` (host-only in all three)."""
    if dummy_run:
        return 'dummy'
    adaptive = getattr(owner, 'adaptive_verification', None)
    if adaptive is not None and getattr(
            scheduler_output, 'scheduled_spec_decode_tokens', None):
        return 'adaptive'
    return 'ordinary'


def dp_branch(owner, dp_sync=None):
    """Source branch of ``dispatch_cg_and_sync_dp``."""
    if dp_sync is not None:
        return 'reuse'
    if int(getattr(owner, 'dp_size', 1)) == 1:
        return 'dp1'
    return 'collective'


def connector_branch(connector):
    """Exact installed connector family; names alone do not authorize work.

    Only the stock in-tree ``KVConnector`` (and its disabled methods) is the
    reviewed no-op. Anything else is an active/unreviewed branch, which keeps
    the window INCONCLUSIVE rather than silently admitting its activity.
    """
    cls = type(connector)
    if (cls.__name__ == 'KVConnector'
            and cls.__module__ == 'vllm.v1.worker.gpu.kv_connector'):
        return 'noop'
    if (cls.__name__ == 'ECConnector'
            and cls.__module__ == 'vllm.v1.worker.gpu.ec_connector'):
        return 'noop'
    return 'active'


def offloader_branch(offloader=None):
    """The pinned subset does not include the installed offloader."""
    return 'stock_unreviewed'


def params_branch(owner, idx_mapping_np):
    """Source branch of ``apply_sampling_params``: exact early return or work."""
    needs = getattr(owner, 'needs_logits_processing', None)
    if needs is None:
        return 'unknown'
    return 'processing' if bool(needs[idx_mapping_np].any()) else 'none'


_GRAPH_DESCRIPTOR_FIELDS = ('num_tokens', 'num_reqs', 'uniform_token_count',
                            'max_query_len', 'num_active_loras',
                            'num_ubatches')


def graph_descriptor(scheduler_output, batch_desc):
    """Census key of the replayed graph: pinned ``BatchExecutionDescriptor``
    scalar fields only (cudagraph_utils.py:60-73). Per-step scheduler counts
    (actual requests/tokens) vary under one padded graph and are not part of
    the graph identity; ``scheduler_output`` is accepted for the seam shape.
    """
    return _safe_descriptor({name: getattr(batch_desc, name, None)
                             for name in _GRAPH_DESCRIPTOR_FIELDS})


def _validate_census(mapping):
    if not isinstance(mapping, dict) or not mapping:
        raise VocabularyError('census declaration must be a non-empty mapping')
    clean = {}
    for key, counts in mapping.items():
        if not isinstance(key, str) or not key:
            raise VocabularyError('census key must be a string')
        if not isinstance(counts, dict) or not counts:
            raise VocabularyError('census entry must be non-empty')
        row = {}
        for kind, count in counts.items():
            if kind not in KINDS or type(count) is not int or count < 0:
                raise VocabularyError('census entry has invalid kind/count')
            row[kind] = count
        clean[key] = row
    return clean


def install_census(owner, mapping):
    """Install a pre-window retained graph-node census on the runner.

    The census key is :func:`descriptor_key` of the batch descriptor. Values
    are ``{kind: count}``. This is a declaration, not a measurement: it must
    come from the retained graph inventory before the short window starts.
    """
    clean = _validate_census(mapping)
    if owner is not None:
        setattr(owner, '_flash_s5_engine_census', clean)
    with _CENSUS_LOCK:
        global _CENSUS
        _CENSUS = clean
    return clean


_CENSUS = {}


def census_for(owner, descriptor):
    mapping = None
    if owner is not None:
        mapping = getattr(owner, '_flash_s5_engine_census', None)
    if mapping is None:
        with _CENSUS_LOCK:
            mapping = _CENSUS
    if not isinstance(mapping, dict) or not mapping:
        return None
    key = descriptor_key(descriptor)
    counts = mapping.get(key)
    if counts is None:
        return None
    row = {}
    for kind, count in counts.items():
        if kind not in KINDS or type(count) is not int or count < 0:
            raise VocabularyError('invalid retained census')
        row[kind] = count
    return row


class EngineOfferLedger:
    """Bounded offer ledger over one CorrelationBridge (one worker/rank).

    The bridge owns offer creation and the thread fields. This class owns the
    declared family/suboperation/expectation metadata, the bounded step
    context, the declared stream identities, the census and the separate
    engine-accounting receipt.
    """

    def __init__(self, bridge, *, worker, rank, epoch, generation,
                 route_generation, green_context_id, identities=None,
                 pp_stage=None, census=None, env=None, max_offers=MAX_OFFERS,
                 binding=None):
        if not _enabled(env):
            raise OfferDisabled('engine offers disabled')
        if bridge is None:
            raise OfferError('correlation bridge required')
        if type(rank) is not int or rank < 0:
            raise OfferError('invalid rank')
        if type(epoch) is not int or epoch < 1:
            raise OfferError('invalid epoch')
        if type(generation) is not int or generation < 1:
            raise OfferError('invalid generation')
        if type(route_generation) is not int or route_generation < 1:
            raise OfferError('invalid route generation')
        if generation != route_generation:
            raise GenerationMismatch('engine and route generation differ')
        if getattr(bridge, 'epoch', epoch) != epoch:
            raise GenerationMismatch('bridge epoch differs')
        if getattr(bridge, 'generation', generation) != generation:
            raise GenerationMismatch('bridge generation differs')
        if getattr(bridge, 'rank', rank) != rank:
            raise GenerationMismatch('bridge rank differs')
        if type(green_context_id) is not int or green_context_id < 1:
            raise OfferError('invalid green context id')
        if type(max_offers) is not int or not 0 < max_offers <= MAX_OFFERS:
            raise OfferError('invalid offer capacity')
        self.bridge = bridge
        self.worker = str(worker)
        self.rank = rank
        self.epoch = epoch
        self.generation = generation
        self.route_generation = route_generation
        self.green_context_id = green_context_id
        self.pp_stage = None if pp_stage is None else int(pp_stage)
        self.identities = self._validate_identities(identities)
        self.census = _validate_census(census) if census else {}
        self.max_offers = max_offers
        self.env = dict(os.environ if env is None else env)
        self._lock = threading.RLock()
        self._meta = {}
        self._gaps = []
        self._steps = []
        self._step_seq = 0
        self._step = contextvars.ContextVar('s5_engine_step', default=None)
        self._owner = None
        self._sealed = None
        self._branches = {}
        self._invalid = []
        self._invalidated = False
        self.binding = _coerce_binding(binding) if binding is not None else None
        if self.binding is not None:
            self._check_binding(self.binding)
        self.active = False

    def _check_binding(self, binding):
        if binding.rank != self.rank:
            raise GenerationMismatch('binding rank differs')
        if binding.engine_epoch != self.epoch:
            raise GenerationMismatch('binding epoch differs')
        if binding.route_generation != self.route_generation:
            raise GenerationMismatch('binding route generation differs')
        if binding.green_context_id != self.green_context_id:
            raise GenerationMismatch('binding green context differs')
        if binding.stream_roles and self.identities:
            for role, row in binding.stream_roles.items():
                mine = self.identities.get(role)
                if mine is None or mine.get('context_id') != row.get(
                        'context_id') or mine.get('stream_id') != row.get(
                            'stream_id'):
                    raise GenerationMismatch(
                        'binding stream identity differs: ' + role)

    def bind(self, binding):
        """Bind the single owner-supplied window identity before the seal."""
        if self._sealed is not None or self._meta:
            raise OfferError('binding arrived after offers were admitted')
        self.binding = _coerce_binding(binding)
        self._check_binding(self.binding)
        return self

    def finish_window(self, router_census):
        """Finalize only observed bounds, after native end and ledger deactivation.

        All pre-window identity fields remain fixed. The caller supplies the
        independent router delta; the native bridge supplies its own timestamps.
        """
        if (self.active or self._sealed is not None or self.binding is None
                or self.binding.native_end_ns is not None or self.open_offers()):
            raise OfferError('window_not_ready_for_finalization')
        native = self.bridge.window()
        if (native.get('start_ns') != self.binding.native_start_ns
                or native.get('route_generation') != self.route_generation
                or type(native.get('end_ns')) is not int
                or native['end_ns'] <= self.binding.native_start_ns):
            raise OfferError('window_end_not_observed')
        record = self.binding.as_record()
        record.update(native_end_ns=native['end_ns'], router_census=router_census)
        final = _coerce_binding(record)
        self._check_binding(final)
        self.binding = final
        return final.as_record()

    def start(self, binding=None):
        if binding is not None:
            self.bind(binding)
        return self.activate(self._owner)

    def receipt(self):
        return self.seal()

    @staticmethod
    def _validate_identities(identities):
        if identities is None:
            return {}
        if not isinstance(identities, dict):
            raise OfferError('identities must be a mapping')
        clean = {}
        for role, row in identities.items():
            if role not in STREAM_ROLES:
                raise OfferError('undeclared stream role: ' + str(role))
            if not isinstance(row, dict):
                raise OfferError('stream identity must be a mapping')
            context_id = row.get('context_id')
            stream_id = row.get('stream_id')
            if context_id is not None and (type(context_id) is not int
                                           or context_id < 1):
                raise OfferError('invalid context id')
            if stream_id is not None and (type(stream_id) is not int
                                          or stream_id < 1):
                raise OfferError('invalid stream id')
            if role == 'main' and row.get('non_green_control'):
                raise OfferError('main role carries model/decode compute; '
                                 'it cannot be a non-green control')
            clean[role] = {
                'context_id': context_id,
                'stream_id': stream_id,
                'non_green_control': bool(row.get('non_green_control', False)),
            }
        return clean

    # -- lifecycle ---------------------------------------------------------
    def activate(self, owner=None):
        if self.binding is not None and self.binding.native_end_ns is None:
            native = self.bridge.window()
            if native != dict(start_ns=self.binding.native_start_ns, end_ns=0,
                              route_generation=self.route_generation):
                raise OfferError('window_start_not_observed')
        global _ACTIVE
        with _ACTIVE_LOCK:
            if self.active:
                raise OfferError('ledger already active')
            if _ACTIVE is not None and _ACTIVE is not self:
                raise OfferError('another engine ledger is active')
            self._owner = owner
            self.active = True
            _ACTIVE = self
        if owner is not None:
            setattr(owner, '_flash_s5_engine_ledger', self)
            if self.census:
                setattr(owner, '_flash_s5_engine_census', dict(self.census))
        return self

    def deactivate(self):
        global _ACTIVE
        with _ACTIVE_LOCK:
            self.active = False
            if _ACTIVE is self:
                _ACTIVE = None
        owner = self._owner
        if owner is not None and getattr(
                owner, '_flash_s5_engine_ledger', None) is self:
            try:
                delattr(owner, '_flash_s5_engine_ledger')
            except AttributeError:
                pass
            if self.census and getattr(
                    owner, '_flash_s5_engine_census', None) == self.census:
                try:
                    delattr(owner, '_flash_s5_engine_census')
                except AttributeError:
                    pass
        return self

    def __enter__(self):
        return self.activate(self._owner)

    def __exit__(self, exc_type, exc, tb):
        self.deactivate()
        return False

    # -- bounded notes -----------------------------------------------------
    def _invalidate(self, kind, family=None, suboperation=None, detail=None):
        """Latch a bounded sticky instrumentation-invalid reason.

        This is deliberately not an exception: the engine operation must run
        once even when observation fails. The window can never qualify.
        """
        reason = str(kind)
        with self._lock:
            if len(self._invalid) < 64:
                row = {'kind': reason}
                if family is not None:
                    row['family'] = str(family)
                if suboperation is not None:
                    row['suboperation'] = str(suboperation)
                if detail is not None:
                    row['detail'] = str(detail)[:120]
                if row not in self._invalid:
                    self._invalid.append(row)
            self._invalidated = True
        self._gap(kind, family, suboperation, detail)

    def mark_branch(self, family, suboperation, branch):
        """Bounded host-only branch-presence record; never a CUPTI offer."""
        decl = _decl(family, suboperation)
        branch = _bounded_text(branch, 'branch_name', 32)
        allowed = decl.get('supported_branches')
        if allowed is not None and branch not in allowed:
            raise VocabularyError('undeclared branch: %s.%s=%s'
                                  % (family, suboperation, branch))
        with self._lock:
            key = _wire(family, suboperation)
            seen = self._branches.get(key)
            if seen is None:
                seen = self._branches[key] = {}
            if len(seen) < 16:
                seen[branch] = seen.get(branch, 0) + 1

    def _gap(self, kind, family=None, suboperation=None, detail=None):
        with self._lock:
            if len(self._gaps) >= MAX_GAPS:
                if not any(g['kind'] == 'gap_capacity' for g in self._gaps):
                    self._gaps.append({'kind': 'gap_capacity'})
                return
            if any(g['kind'] == kind and g.get('family') == family
                   and g.get('suboperation') == suboperation
                   for g in self._gaps):
                return
            row = {'kind': str(kind)}
            if family is not None:
                row['family'] = str(family)
            if suboperation is not None:
                row['suboperation'] = str(suboperation)
            if detail is not None:
                row['detail'] = str(detail)[:120]
            self._gaps.append(row)

    # -- step identity -----------------------------------------------------
    @contextmanager
    def step(self, kind):
        if kind not in STEP_KINDS:
            raise OfferError('invalid step kind')
        row = None
        with self._lock:
            self._step_seq += 1
            if len(self._steps) >= MAX_STEPS:
                self._gap('step_capacity')
            else:
                row = {'step': self._step_seq, 'kind': kind}
                self._steps.append(row)
        token = self._step.set(row)
        try:
            yield row
        finally:
            self._step.reset(token)

    # -- offers ------------------------------------------------------------
    def _resolve(self, decl):
        role = decl.get('stream_role', 'main')
        identity = self.identities.get(role)
        if identity is None or identity.get('context_id') is None \
                or identity.get('stream_id') is None:
            return role, None, None
        return role, identity['context_id'], identity['stream_id']

    @contextmanager
    def offer_scope(self, family, suboperation, *, descriptor=None,
                    census=None, explicit_zero=False, branch=None):
        """Open one reviewed engine offer, or run the body uninstrumented.

        Instrumentation failures never escape into the engine operation:
        capacity, push and pop failures are typed, latched, bounded reasons.
        The operation itself is executed exactly once and is never retried.
        An exception from the original engine operation propagates unchanged.
        """
        if not self.active or self._invalidated:
            if self._invalidated:
                self._gap('offer_admission_stopped', family, suboperation)
            yield None
            return
        try:
            decl = _decl(family, suboperation)
            role, context_id, stream_id = self._resolve(decl)
            if context_id is None or stream_id is None:
                self._invalidate('stream_identity_missing', family, suboperation, role)
                raise OfferError('stream_identity_missing')
            branch = None if branch is None else _bounded_text(
                branch, 'branch_name', 32)
            supported = decl.get('supported_branches')
            if branch is not None and supported is not None \
                    and branch not in supported:
                self._gap('branch_unsupported', family, suboperation, branch)
            step_row = self._step.get()
            pp = pp_branch(self._owner)
            if decl.get('requires_pp'):
                if self.pp_stage is None:
                    self._gap('pp_stage_missing', family, suboperation)
                if pp == 'unknown':
                    self._gap('pp_branch_missing', family, suboperation)
            if decl.get('expectation') == 'census':
                if census is not None:
                    census = {k: int(v) for k, v in census.items()}
                if census is None:
                    self._gap('census_missing', family, suboperation)
            meta = {
                'family': family,
                'suboperation': suboperation,
                'expectation': decl['expectation'],
                'allowed_kinds': list(decl['kinds']),
                'copy_kinds': (None if decl.get('copy_kinds') is None
                               else list(decl['copy_kinds'])),
                'stream_role': role,
                'context_id': context_id,
                'stream_id': stream_id,
                'graph': bool(decl.get('graph', False)),
                'pp_stage': self.pp_stage,
                'pp_branch': pp,
                'branch': branch,
                'step': step_row['step'] if step_row else None,
                'step_kind': step_row['kind'] if step_row else None,
                'descriptor': _safe_descriptor(descriptor),
                'census': census,
                'explicit_zero_work': bool(explicit_zero),
                'rank': self.rank,
                'epoch': self.epoch,
                'generation': self.generation,
            }
        except (OfferError, TypeError, ValueError, KeyError, AttributeError) as exc:
            self._invalidate('offer_metadata_invalid', detail=type(exc).__name__)
            yield None
            return
        with self._lock:
            full = len(self._meta) >= self.max_offers
        if full:
            self._invalidate('offer_capacity_exceeded', family,
                             suboperation)
            yield None
            return
        expected = dict(census) if census else {}
        bridge_copy = (ALL_COPY_KINDS if meta['copy_kinds'] is None
                       else tuple(meta['copy_kinds']))
        oid = None
        offer = None
        cm = None
        body_info = (None, None, None)
        try:
            cm = self.bridge.offer(
                _wire(family, suboperation), expected,
                context_id=context_id, stream_id=stream_id,
                graph=meta['graph'], copy_kinds=bridge_copy)
            offer = cm.__enter__()
            oid = offer.get('id')
            if type(oid) is not int or oid < 1:
                raise OfferError('bridge offer identity')
            with self._lock:
                if oid in self._meta:
                    raise OfferError('duplicate engine offer id')
                self._meta[oid] = meta
        except BaseException as exc:
            # Never retry a push and never guess a native stack repair. If the
            # bridge returned an offer, that push is ours; pop it exactly once.
            if offer is not None and cm is not None:
                try:
                    cm.__exit__(*sys.exc_info())
                except BaseException:
                    pass
            self._invalidate('offer_push_failed', family, suboperation,
                             type(exc).__name__)
            yield None
            return
        try:
            yield offer
        except BaseException:
            body_info = sys.exc_info()
            raise
        finally:
            try:
                cm.__exit__(*body_info)
            except BaseException as cleanup:
                # Any cleanup failure other than the body's own exception is
                # a pop failure, whether or not the body raised; never rely on
                # the bridge's row flags alone to latch it.
                if body_info[0] is None or cleanup is not body_info[1]:
                    self._invalidate('offer_pop_failed', family,
                                     suboperation,
                                     type(cleanup).__name__)
                # A body exception always propagates unchanged.
            if oid is not None and offer is not None:
                verified = (offer.get('closed') is True)
                popped = offer.get('popped_id')
                pop_rc = offer.get('pop_rc')
                if popped is not None and popped != oid:
                    verified = False
                if pop_rc is not None and pop_rc != 0:
                    verified = False
                meta['pop_verified'] = bool(verified)
                meta['pop_identity'] = 'bridge+verified'
                if not verified:
                    self._invalidate('offer_pop_unverified', family,
                                     suboperation)

    # -- receipt -----------------------------------------------------------
    def window(self):
        with self._lock:
            gaps = [dict(g) for g in self._gaps]
            steps = [dict(s) for s in self._steps]
            offer_count = len(self._meta)
        return {
            'kind': 'short-placement',
            'worker': self.worker,
            'rank': self.rank,
            'epoch': self.epoch,
            'generation': self.generation,
            'route_generation': self.route_generation,
            'green_context_id': self.green_context_id,
            'pp_stage': self.pp_stage,
            'stream_roles': {k: dict(v) for k, v in self.identities.items()},
            'steps': steps,
            'max_offers': self.max_offers,
            'offer_count': offer_count,
            'gaps': gaps,
            'branches': {key: dict(value)
                         for key, value in self._branches.items()},
            'invalid': [dict(row) for row in self._invalid],
            'invalidated': bool(self._invalidated),
            'binding': (None if self.binding is None
                        else self.binding.as_record()),
        }

    def seal(self):
        if self._sealed is not None:
            return self._sealed
        window = self.window()
        try:
            raw = self.bridge.report()
        except BaseException as exc:
            receipt = _empty_receipt(window,
                missing=['bridge_report_failed:' + type(exc).__name__],
                offer_count=len(self._meta))
            # Only trusted native error codes and bounded scalar diagnostics.
            # Never copy arbitrary exception text, paths or engine payloads.
            code = getattr(exc, 'code', None)
            from replay_observer import _STATUS_NAMES
            known = set(_STATUS_NAMES.values()) | {
                'receipt_too_large', 'malformed_native_report', 'duplicate_json_key', 'native_error'}
            detail = {'type':type(exc).__name__[:80]}
            if type(code) is str and code in known:
                detail['message'] = code
            source = getattr(exc, 'report_diagnostics', {})
            if isinstance(source, dict):
                for key in ('return_code','required_bytes','capacity_bytes'):
                    value=source.get(key)
                    if type(value) is int and 0 <= value < 2**63:
                        detail[key]=value
                for key in ('function','session_state'):
                    value=source.get(key)
                    allowed = ({'s5ro_correlation_report','s5ro_correlation_window','native_report'}
                               if key == 'function' else
                               {'created','started','bound','window','ended','stopped','closed'})
                    if type(value) is str and value in allowed:
                        detail[key]=value
            receipt['bridge_report_error'] = detail
            self._sealed = receipt
            return receipt
        native = raw.get('native') if isinstance(raw, dict) else None
        offers = raw.get('offers') if isinstance(raw, dict) else None
        if not isinstance(native, dict) or not isinstance(offers, list):
            receipt = _empty_receipt(window, missing=['bridge_report_shape'])
            self._sealed = receipt
            return receipt
        with self._lock:
            declared = {oid: dict(meta) for oid, meta in self._meta.items()}
        try:
            receipt = grade_engine(native, offers, declared, window)
        except Exception as exc:
            receipt = _empty_receipt(
                window, missing=['grade_failed:' + type(exc).__name__],
                native_summary=_native_summary(native),
                offer_count=len(offers))
            self._sealed = receipt
            return receipt
        try:
            encoded = json.dumps(receipt, sort_keys=True,
                                 separators=(',', ':'))
        except (TypeError, ValueError):
            encoded = None
        if encoded is None:
            receipt = _empty_receipt(window, missing=['receipt_not_serializable'],
                                     native_summary=_native_summary(native),
                                     offer_count=len(offers))
        elif len(encoded.encode()) > MAX_RECEIPT_BYTES:
            receipt = _empty_receipt(window, missing=['receipt_too_large'],
                                     native_summary=_native_summary(native),
                                     offer_count=len(offers))
        self._sealed = receipt
        return receipt

    report = seal

    def open_offers(self):
        with self._lock:
            return sorted(oid for oid, meta in self._meta.items()
                          if not meta.get('pop_verified'))


# --- runtime functions called by generated source ---------------------------
def run(owner, family, suboperation, fn, *args, descriptor=None, branch=None,
        **kwargs):
    """One declared offer around ``fn(*args, **kwargs)`` or inert passthrough."""
    ledger = current_ledger(owner)
    if ledger is None:
        return fn(*args, **kwargs)
    with ledger.offer_scope(family, suboperation, descriptor=descriptor,
                            branch=branch):
        return fn(*args, **kwargs)


def scoped_call(owner, family, suboperation, descriptor, fn, *args, **kwargs):
    return run(owner, family, suboperation, fn, *args, descriptor=descriptor,
               **kwargs)


@contextmanager
def leaf(owner, family, suboperation, *, descriptor=None, branch=None):
    """Context-manager leaf for a block of source operations."""
    ledger = current_ledger(owner)
    if ledger is None:
        yield None
        return
    with ledger.offer_scope(family, suboperation, descriptor=descriptor,
                            branch=branch):
        yield


def mark_branch(owner, family, suboperation, branch):
    """Host-only branch-not-taken record; creates no CUPTI offer."""
    ledger = current_ledger(owner)
    if ledger is None:
        return None
    try:
        ledger.mark_branch(family, suboperation, branch)
    except (OfferError, TypeError, ValueError, KeyError, AttributeError) as exc:
        ledger._invalidate('branch_metadata_invalid', detail=type(exc).__name__)
    return None


def mark_zero_work(owner, family, suboperation):
    """Explicit zero-work declaration for a reviewed no-work branch."""
    ledger = current_ledger(owner)
    if ledger is None:
        return None
    with ledger.offer_scope(family, suboperation, explicit_zero=True):
        return None


def offer_method(family, suboperation, descriptor=None, branch=None):
    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            ledger = current_ledger(self)
            if ledger is None:
                return fn(self, *args, **kwargs)
            with ledger.offer_scope(family, suboperation,
                                    descriptor=descriptor, branch=branch):
                return fn(self, *args, **kwargs)
        return wrapper
    return decorate


def step_method(kind):
    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            ledger = current_ledger(self)
            if ledger is None:
                return fn(self, *args, **kwargs)
            with ledger.step(kind):
                return fn(self, *args, **kwargs)
        return wrapper
    return decorate


def run_graph_replay(owner, scheduler_output, batch_desc, fn, *args):
    """Census offer around the reviewed graph replay call."""
    ledger = current_ledger(owner)
    if ledger is None:
        return fn(*args)
    try:
        descriptor = graph_descriptor(scheduler_output, batch_desc)
        census = census_for(owner, descriptor)
    except (OfferError, TypeError, ValueError, KeyError, AttributeError) as exc:
        ledger._invalidate('graph_metadata_invalid', detail=type(exc).__name__)
        return fn(*args)
    with ledger.offer_scope('forward_dispatch', 'graph_replay',
                            descriptor=descriptor, census=census):
        return fn(*args)


def open_ledger(session, *, worker, rank, epoch, generation,
                route_generation, green_context_id, identities=None,
                pp_stage=None, census=None, env=None, max_offers=MAX_OFFERS,
                binding=None):
    """Runtime entry point: configure the bridge, then the ledger.

    The caller must already have started the native observer and must call
    ``seal`` after the drained window ends. This function performs no CUDA or
    observer work itself beyond what ``CorrelationBridge`` does.
    """
    try:
        from activity_accounting import CorrelationBridge
    except ImportError:
        from .activity_accounting import CorrelationBridge
    bridge = CorrelationBridge(session, epoch=epoch, generation=generation,
                               rank=rank)
    return EngineOfferLedger(
        bridge, worker=worker, rank=rank, epoch=epoch, generation=generation,
        route_generation=route_generation, green_context_id=green_context_id,
        identities=identities, pp_stage=pp_stage, census=census, env=env,
        max_offers=max_offers, binding=binding)


# --- grading ----------------------------------------------------------------
class _BadEvidence(Exception):
    pass


def _need(condition, reason):
    if not condition:
        raise _BadEvidence(reason)


def _int(value, reason, minimum=0):
    _need(type(value) is int and value >= minimum, reason)
    return value


def _native_summary(native):
    keys = ('schema', 'thread_id_type', 'complete', 'capacity',
            'reserved_bytes', 'errors', 'overflow', 'open_scopes',
            'boundary_records', 'activity_records', 'window')
    return {k: native.get(k) for k in keys}


def _empty_receipt(window, *, missing=None, native_summary=None,
                   offer_count=0):
    return {
        'schema': SCHEMA,
        'verdict': 'INCONCLUSIVE' if missing else 'FAIL',
        'window': window,
        'binding': window.get('binding'),
        'native_window': (native_summary or {}).get('window'),
        'thread_id_type': None,
        'native': native_summary or {},
        'offers': [],
        'declared': {},
        'joins': [],
        'counts': {},
        'families': {},
        'graph': {'joined_records': 0, 'outside_engine_ledger': [],
                  'ids': []},
        'capacity': {'offer_capacity': window.get('max_offers'),
                     'offers': offer_count},
        'eager_records': 0,
        'eager_green_records': 0,
        'eager_other_records': 0,
        'eager_green_by_kind': {},
        'eager_other_by_kind': {},
        'eager_other_by_role': {},
        'eager_by_kind': {},
        'engine_by_kind': {},
        'native_by_kind': {},
        'unjoined_graph_records': 0,
        'graph_records': 0,
        'joined_records': 0,
        'unexplained': [],
        'failures': [] if missing else ['empty_receipt'],
        'missing': sorted(set(missing or [])),
        'notes': [],
    }


def grade_engine(native, offers, declared, window):
    """Grade the engine-accounting evidence; FAIL beats INCONCLUSIVE."""
    failures = []
    missing = []
    notes = []
    joins = []
    graph_outside = []
    missing_graph_rows = []
    by_offer_counts = defaultdict(Counter)
    window_totals = Counter()
    census_declared = {}
    thread_type = None

    def fail(reason):
        if reason not in failures:
            failures.append(reason)

    def miss(reason):
        if reason not in missing:
            missing.append(reason)

    try:
        # window declaration
        _need(isinstance(window, dict), 'window_shape')
        _need(isinstance(window.get('worker'), str) and window['worker'],
              'window_worker')
        rank = _int(window.get('rank'), 'window_rank')
        epoch = _int(window.get('epoch'), 'window_epoch', 1)
        generation = _int(window.get('generation'), 'window_generation', 1)
        route_generation = _int(window.get('route_generation'),
                                'window_route_generation', 1)
        green = _int(window.get('green_context_id'),
                     'window_green_context', 1)
        _need(generation == route_generation, 'window_generation_mismatch')
        _need(isinstance(window.get('stream_roles'), dict),
              'window_stream_roles')
        _need(isinstance(window.get('steps'), list), 'window_steps')
        gaps = window.get('gaps', [])
        _need(isinstance(gaps, list), 'window_gaps')
        for gap in gaps:
            _need(isinstance(gap, dict) and isinstance(gap.get('kind'), str),
                  'window_gap_shape')
            miss('ledger_gap:' + gap['kind'])
        if window.get('invalidated'):
            miss('instrumentation_invalid')
        branches = window.get('branches', {})
        _need(isinstance(branches, dict), 'window_branches')
        for wire, seen in branches.items():
            _need(isinstance(wire, str) and isinstance(seen, dict),
                  'window_branch_shape')
            try:
                family, sub = _split_wire(wire)
            except VocabularyError:
                raise _BadEvidence('window_branch_family')
            decl = _decl(family, sub)
            allowed = decl.get('supported_branches')
            for name, count in seen.items():
                _need(isinstance(name, str) and name,
                      'window_branch_name')
                _need(type(count) is int and count > 0,
                      'window_branch_count')
                if allowed is not None and name not in allowed:
                    miss('branch_unqualified:%s=%s' % (wire, name))

        binding = window.get('binding')
        if binding is None:
            miss('binding_missing')
        else:
            try:
                binding = _coerce_binding(binding)
            except OfferError:
                raise _BadEvidence('binding_shape')
            if binding.worker != window['worker']:
                raise _BadEvidence('binding_worker_mismatch')
            if binding.rank != rank:
                raise _BadEvidence('binding_rank_mismatch')
            if binding.engine_epoch != epoch:
                raise _BadEvidence('binding_epoch_mismatch')
            if binding.route_generation != route_generation:
                raise _BadEvidence('binding_route_mismatch')
            if binding.green_context_id != green:
                raise _BadEvidence('binding_green_mismatch')
            if binding.pp_stage != window.get('pp_stage'):
                raise _BadEvidence('binding_pp_mismatch')
            for role, row in window['stream_roles'].items():
                binding_role = binding.stream_roles.get(role)
                if binding_role is None:
                    continue
                if (binding_role.get('context_id') != row.get('context_id')
                        or binding_role.get('stream_id')
                        != row.get('stream_id')):
                    raise _BadEvidence('binding_stream_mismatch')

        # native evidence shape and loss
        _need(isinstance(native, dict), 'native_shape')
        schema = native.get('schema')
        if schema == NATIVE_SCHEMA:
            thread_type = native.get('thread_id_type')
            if thread_type not in THREAD_ID_TYPES:
                miss('thread_id_type_invalid')
                thread_type = None
        elif schema == NATIVE_SCHEMA_LEGACY:
            thread_type = None
            miss('thread_id_type_missing')
        else:
            raise _BadEvidence('correlation_schema')
        if binding is not None and binding.native_end_ns is None:
            miss('window_not_finalized')
        native_window = native.get('window')
        if not isinstance(native_window, dict):
            miss('native_window_missing')
        else:
            _need(type(native_window.get('start_ns')) is int
                  and native_window['start_ns'] >= 0,
                  'native_window_start')
            _need(type(native_window.get('end_ns')) is int
                  and native_window['end_ns'] >= native_window['start_ns'],
                  'native_window_end')
            _need(type(native_window.get('route_generation')) is int
                  and native_window['route_generation'] >= 0,
                  'native_window_route')
            if binding is not None:
                if (native_window['start_ns'] != binding.native_start_ns
                        or native_window['end_ns'] != binding.native_end_ns):
                    raise _BadEvidence('native_window_binding_mismatch')
                if native_window['route_generation'] != binding.route_generation:
                    raise _BadEvidence('native_window_route_mismatch')
        if native.get('complete') is not True:
            miss('correlation_incomplete')
        for key, reason in (('overflow', 'correlation_overflow'),
                            ('errors', 'correlation_errors'),
                            ('open_scopes', 'correlation_open_scopes'),
                            ('boundary_records', 'correlation_boundary')):
            value = native.get(key)
            _need(type(value) is int and value >= 0, key + '_shape')
            if value:
                miss(reason)
        native_capacity = native.get('capacity')
        native_reserved = native.get('reserved_bytes')
        _need(type(native_capacity) is int and native_capacity >= 0,
              'capacity_shape')
        _need(type(native_reserved) is int and native_reserved >= 0,
              'reserved_shape')

        # offer ledger
        _need(isinstance(offers, list), 'offers_shape')
        _need(isinstance(declared, dict), 'declared_shape')
        if not offers:
            miss('no_offers')
        om = {}
        for offer in offers:
            _need(isinstance(offer, dict), 'offer_shape')
            oid = offer.get('id')
            _need(type(oid) is int and oid > 0 and oid not in om,
                  'offer_identity')
            om[oid] = offer
        if len(om) > MAX_OFFERS:
            miss('offer_capacity_exceeded')
        contract_ok = thread_type in THREAD_ID_TYPES
        if contract_ok:
            for offer in offers:
                if not (type(offer.get('system_thread')) is int
                        and offer['system_thread'] > 0):
                    contract_ok = False
                    break
        if offers and not contract_ok:
            miss('thread_contract_missing')

        for oid, offer in om.items():
            meta = declared.get(oid)
            if not isinstance(meta, dict):
                fail('undeclared_offer')
                continue
            if offer.get('epoch') != epoch or offer.get('generation') != generation:
                fail('stale_generation')
            if offer.get('rank') != rank:
                fail('offer_rank')
            if offer.get('closed') is not True:
                miss('offer_not_closed')
            if not (type(offer.get('thread')) is int and offer['thread'] > 0):
                miss('offer_thread_missing')
            if offer.get('body_exception') is True:
                miss('offer_body_exception')
            try:
                family, sub = _split_wire(offer.get('family'))
            except VocabularyError:
                fail('undeclared_family')
                continue
            if (meta.get('family') != family
                    or meta.get('suboperation') != sub):
                fail('offer_family_metadata_mismatch')
            decl = _decl(family, sub)
            if meta.get('expectation') != decl['expectation']:
                fail('offer_expectation_mismatch')
            if list(meta.get('allowed_kinds', [])) != list(decl['kinds']):
                fail('offer_kinds_metadata_mismatch')
            if decl.get('copy_kinds') is not None and list(
                    meta.get('copy_kinds') or []) != list(decl['copy_kinds']):
                fail('offer_copy_kinds_metadata_mismatch')
            if meta.get('graph') is not bool(decl.get('graph', False)):
                fail('offer_graph_metadata_mismatch')
            branch = meta.get('branch')
            supported = decl.get('supported_branches')
            if branch is not None:
                _need(isinstance(branch, str) and branch, 'offer_branch_shape')
                if supported is not None and branch not in supported:
                    miss('branch_unqualified:%s=%s'
                         % (_wire(family, sub), branch))
            if decl.get('noop_branch') is not None and branch == decl[
                    'noop_branch']:
                meta['noop_branch_checked'] = True
            if meta.get('step') is None or meta.get('step_kind') not in STEP_KINDS:
                miss('offer_without_step')
            if meta.get('context_id') is None or meta.get('stream_id') is None:
                miss('stream_identity_missing')
            else:
                _need(type(meta['context_id']) is int
                      and type(meta['stream_id']) is int,
                      'offer_identity_shape')
                role = meta.get('stream_role')
                _need(role in STREAM_ROLES, 'offer_role_shape')
                declared_role = window['stream_roles'].get(role)
                if declared_role is None:
                    fail('declared_stream_role_missing')
                elif (declared_role.get('context_id') != meta['context_id']
                      or declared_role.get('stream_id') != meta['stream_id']):
                    fail('declared_stream_identity_mismatch')
                elif (declared_role.get('context_id') != green
                      and (role == 'main'
                           or not declared_role.get('non_green_control'))):
                    fail('role_context_not_green')
                controls = window.setdefault('non_green_roles', {})
                if declared_role and declared_role.get('context_id') != green:
                    controls[role] = dict(declared_role)
            if decl.get('requires_pp'):
                if meta.get('pp_stage') is None:
                    miss('pp_stage_missing')
                if meta.get('pp_branch') not in ('single', 'first', 'last',
                                                 'middle'):
                    miss('pp_branch_missing')
            if meta.get('pop_verified') is not True:
                miss('offer_pop_unverified')
            expected = offer.get('expected')
            _need(isinstance(expected, dict), 'offer_expected_shape')
            if decl['expectation'] in ('zero_work', 'container'):
                if expected:
                    fail('offer_expected_not_empty')
            elif decl['expectation'] == 'census':
                if meta.get('census') is None:
                    miss('census_missing')
                elif expected != meta['census']:
                    fail('offer_expected_census_mismatch')

        for oid, offer in om.items():
            parent = offer.get('parent')
            if parent:
                parent_offer = om.get(parent)
                if parent_offer is None:
                    fail('offer_parent_missing')
                elif parent_offer.get('thread') != offer.get('thread'):
                    fail('offer_parent_thread')

        # native scopes: per-thread push/pop and exact coverage
        stacks = defaultdict(list)
        opened = set()
        closed = set()
        _need(isinstance(native.get('scopes'), list), 'scopes_shape')
        for event in native['scopes']:
            _need(isinstance(event, dict), 'scope_shape')
            oid = event.get('id')
            if oid not in om:
                fail('native_scope_unknown_offer')
                continue
            offer = om[oid]
            meta = declared.get(oid, {})
            if event.get('thread') != offer.get('thread'):
                if contract_ok:
                    fail('native_scope_thread_mismatch')
                else:
                    miss('native_scope_thread_unverified')
            elif (contract_ok and event.get('system_thread') != offer.get('system_thread')):
                fail('native_scope_system_thread_mismatch')
            group = offer.get('thread')
            stack = stacks[group]
            action = event.get('action')
            if action == 'push':
                if oid in opened:
                    fail('native_scope_duplicate_push')
                    continue
                want = stack[-1] if stack else 0
                if event.get('parent') != offer.get('parent', 0) \
                        or event.get('parent') != want:
                    fail('native_scope_parent')
                opened.add(oid)
                stack.append(oid)
            elif action == 'pop':
                if not stack or stack[-1] != oid or oid in closed:
                    fail('native_scope_pop')
                else:
                    stack.pop()
                    closed.add(oid)
            else:
                fail('native_scope_action')
        if opened != set(om) or closed != set(om) or any(stacks.values()):
            fail('native_scope_coverage')

        # APIs
        _need(isinstance(native.get('apis'), list), 'apis_shape')
        apis = {}
        for api in native['apis']:
            _need(isinstance(api, dict), 'api_shape')
            cid = api.get('correlation_id')
            _need(type(cid) is int and cid > 0 and cid not in apis,
                  'api_identity')
            _need(api.get('domain') in ('runtime', 'driver'), 'api_domain')
            _need(isinstance(api.get('name'), str) and api['name'], 'api_name')
            _need(type(api.get('return_value')) is int, 'api_return')
            _need(type(api.get('thread')) is int and api['thread'] > 0,
                  'api_thread')
            _need(type(api.get('start')) is int and type(api.get('end')) is int
                  and 0 < api['start'] <= api['end'], 'api_time')
            apis[cid] = api

        # external CUSTOM0 joins
        _need(isinstance(native.get('external'), list), 'external_shape')
        external = {}
        for item in native['external']:
            _need(isinstance(item, dict), 'external_row')
            cid = item.get('correlation_id')
            _need(type(cid) is int and cid > 0, 'external_identity')
            if cid in external:
                fail('duplicate_external_join')
                continue
            if item.get('kind') != 'custom0':
                fail('foreign_external_kind')
                continue
            if cid not in apis:
                fail('external_api_missing')
                continue
            oid = item.get('external_id')
            if oid not in om:
                fail('unknown_offer')
                continue
            external[cid] = oid
            if contract_ok and apis[cid]['thread'] != om[oid]['thread']:
                fail('external_thread_mismatch')

        # A reviewed no-work/container boundary may not submit real GPU work
        # under its own correlation id, even if the deferred row is missing.
        for cid, oid in external.items():
            meta = declared.get(oid)
            if not isinstance(meta, dict):
                continue
            if meta.get('expectation') not in ('zero_work', 'container'):
                continue
            api = apis.get(cid)
            if api is None or api.get('return_value') != 0:
                continue
            kind = _api_kind(api.get('name'))
            if kind in KINDS or kind == 'graph':
                fail('zero_work_has_submission' if
                     meta['expectation'] == 'zero_work'
                     else 'container_has_submission')

        # activities: every eager record joins exactly once
        _need(isinstance(native.get('activities'), list), 'activities_shape')
        offer_threads = set()
        for offer in om.values():
            for key in ('thread',):
                value = offer.get(key)
                if type(value) is int and value > 0:
                    offer_threads.add(value)
        records = 0
        seen_rows = set()
        by_api = Counter()
        census_joined = defaultdict(Counter)
        graph_offer_launches = defaultdict(set)
        eager_green_by_kind = Counter()
        eager_other_by_kind = Counter()
        eager_other_by_role = Counter()
        graph_by_kind = Counter()
        for row in native['activities']:
            _need(isinstance(row, dict), 'activity_row')
            kind = row.get('kind')
            _need(kind in KINDS, 'activity_kind')
            count = row.get('count')
            _need(type(count) is int and count > 0, 'activity_count')
            cid = row.get('correlation_id')
            _need(type(cid) is int and cid >= 0, 'activity_identity')
            graph_id = row.get('graph_id', 0)
            _need(type(graph_id) is int and graph_id >= 0, 'activity_graph')
            context_id = row.get('context_id')
            stream_id = row.get('stream_id')
            _need(type(context_id) is int and type(stream_id) is int,
                  'activity_context')
            window_totals[kind] += count
            records += count
            # A graph launch shares one correlation id across its node kinds
            # and directions, so uniqueness is the full aggregated row, not
            # the correlation id alone.
            signature = (cid, kind, graph_id, context_id, stream_id,
                         row.get('copy_kind'), row.get('name'))
            if signature in seen_rows:
                fail('duplicate_activity_join')
                continue
            seen_rows.add(signature)
            oid = external.get(cid)
            api = apis.get(cid)
            if oid is None:
                if graph_id:
                    # A graph row with no CUSTOM0/external join is missing
                    # evidence. It is never reclassified eager and never
                    # silently "outside" the ledger.
                    missing_graph_rows.append({'correlation_id': cid,
                                               'kind': kind, 'count': count,
                                               'graph_id': graph_id})
                    miss('graph_row_without_join')
                    continue
                # An eager submission with no CUSTOM0 push on a thread that
                # issued no engine offer is foreign async/library work. It is
                # never borrowed into an offer; it is insufficient evidence,
                # not wrong ownership.
                if (api is not None and offer_threads
                        and api['thread'] not in offer_threads):
                    miss('unaccounted_activity_thread')
                else:
                    fail('unoffered_eager_activity')
                continue
            if api is None:
                fail('activity_without_api')
                continue
            meta = declared.get(oid)
            if not isinstance(meta, dict):
                fail('undeclared_offer')
                continue
            unsupported = meta.get('expectation') == 'unsupported'
            if api.get('return_value') != 0:
                fail('failed_api_has_activity')
            expected_api = 'graph' if graph_id else kind
            if not unsupported and _api_kind(api['name']) != expected_api:
                fail('api_activity_kind')
            if not unsupported and bool(graph_id) != bool(meta.get('graph')):
                fail('offer_graph_scope')
            if meta.get('context_id') is None:
                miss('stream_identity_missing')
            elif context_id != meta['context_id']:
                fail('wrong_context')
            if not graph_id:
                if meta.get('stream_id') is None:
                    miss('stream_identity_missing')
                elif stream_id != meta['stream_id']:
                    fail('wrong_stream')
            if unsupported:
                miss('unsupported_coverage:' + _wire(meta['family'],
                                                     meta['suboperation']))
            if graph_id:
                census_joined[oid][kind] += count
            if not unsupported and kind not in meta.get('allowed_kinds', ()):
                fail('wrong_kind')
            if not unsupported and kind == 'memcpy' \
                    and meta.get('copy_kinds') is not None:
                if row.get('copy_kind') not in meta['copy_kinds']:
                    fail('copy_direction')
            by_offer_counts[oid][kind] += count
            by_api[cid] += count
            if graph_id:
                graph_by_kind[kind] += count
            elif context_id == green:
                eager_green_by_kind[kind] += count
            else:
                eager_other_by_kind[kind] += count
                role = meta.get('stream_role')
                eager_other_by_role[role] += count
            joins.append({'offer': oid, 'correlation_id': cid, 'kind': kind,
                          'count': count, 'graph_id': graph_id,
                          'context_id': context_id,
                          'stream_id': stream_id,
                          'stream_role': meta.get('stream_role')})

        if records != native.get('activity_records'):
            miss('activity_census')
        native_totals = native.get('window_totals')
        if native_totals is not None:
            if not isinstance(native_totals, dict) or any(
                    type(v) is not int or v < 0 for v in native_totals.values()):
                miss('window_totals_shape')
            elif Counter(native_totals) != window_totals:
                miss('window_totals_mismatch')
        else:
            notes.append('window_totals_from_activity_rows')

        # every successful submission covered; runtime/driver wrapper dedup
        peer_index = {}
        for cid, api in apis.items():
            if by_api[cid]:
                key = (api['thread'], external.get(cid),
                       _api_kind(api['name']))
                peer_index.setdefault(key, []).append((cid, api))
        wrappers = []
        wrapper_peers = set()
        graph_api_offers = set()
        graph_apis_by_offer = defaultdict(set)
        graph_launches_by_offer = defaultdict(set)
        for cid, api in apis.items():
            kind = _api_kind(api['name'])
            if kind is None:
                miss('unsupported_api:' + api['name'])
                continue
            if kind == 'host':
                # Success, or the documented NOT_READY of the exact event
                # query; a pending query observes zero GPU work and is never
                # completion. Any other nonzero host return still fails.
                if not _host_return_ok(api['name'], api['return_value']):
                    fail('host_api_failed')
                if by_api[cid]:
                    fail('host_api_has_activity')
                continue
            if api['return_value'] != 0:
                fail('api_submission_failed')
                if by_api[cid]:
                    fail('failed_api_has_activity')
                continue
            if kind == 'graph':
                oid = external.get(cid)
                if oid is None:
                    fail('unoffered_graph_submission')
                    continue
                if not declared.get(oid, {}).get('graph'):
                    fail('unoffered_graph_submission')
                else:
                    graph_api_offers.add(oid)
                    graph_apis_by_offer[oid].add(cid)
                continue
            if by_api[cid]:
                continue
            peers = []
            for other, peer in peer_index.get(
                    (api['thread'], external.get(cid), kind), []):
                nested = ((api['start'] <= peer['start']
                           and peer['end'] <= api['end'])
                          or (peer['start'] <= api['start']
                              and api['end'] <= peer['end']))
                if (by_api[other] and api['domain'] != peer['domain']
                        and api['thread'] == peer['thread']
                        and external.get(cid) is not None
                        and external.get(cid) == external.get(other)
                        and _api_kind(peer['name']) == kind and nested):
                    peers.append(other)
            if len(peers) != 1:
                miss('submission_without_unique_activity')
            elif peers[0] in wrapper_peers:
                miss('wrapper_peer_reused')
            else:
                wrapper_peers.add(peers[0])
                wrappers.append({'wrapper': cid, 'activity_api': peers[0]})

        # A single graph launch may be visible as a runtime API plus a nested
        # driver API. Collapse each overlapping cross-domain pair to one launch;
        # anything else remains a distinct launch for the census.
        for oid, cids in graph_apis_by_offer.items():
            remaining = set(cids)
            while remaining:
                cid = min(remaining)
                api = apis[cid]
                partner = None
                for other in sorted(remaining - {cid}):
                    peer = apis[other]
                    nested = ((api['start'] <= peer['start']
                               and peer['end'] <= api['end'])
                              or (peer['start'] <= api['start']
                                  and api['end'] <= peer['end']))
                    if (api['domain'] != peer['domain']
                            and api['thread'] == peer['thread']
                            and _api_kind(peer['name']) == 'graph'
                            and nested):
                        partner = other
                        break
                group = {cid}
                if partner is not None:
                    group.add(partner)
                graph_launches_by_offer[oid].add(min(group))
                remaining -= group

        # per-offer expectation checks
        for oid, offer in om.items():
            meta = declared.get(oid)
            if not isinstance(meta, dict):
                continue
            counts = by_offer_counts[oid]
            expectation = meta.get('expectation')
            if expectation == 'zero_work':
                if counts:
                    fail('zero_work_has_activity')
            elif expectation == 'container':
                if counts:
                    fail('container_has_activity')
            elif expectation == 'census':
                census = meta.get('census')
                census_declared[str(oid)] = census
                if census is None:
                    miss('census_missing')
                    continue
                launches = sorted(graph_launches_by_offer.get(oid, ()))
                if not launches:
                    miss('census_launch_missing')
                elif len(launches) != 1:
                    # One census offer owns exactly one actual replay after
                    # runtime/driver wrapper dedup.
                    fail('census_launch_multiplicity')
                joined = census_joined[oid]
                if joined:
                    if Counter(census) != joined:
                        fail('census_mismatch')
                    else:
                        notes.append('census_verified:' + str(oid))
                elif launches:
                    if any(census.values()):
                        miss('census_records_missing')
                    else:
                        notes.append('census_launch_only:' + str(oid))
            elif expectation == 'unsupported':
                miss('unsupported_coverage:' + _wire(meta['family'],
                                                     meta['suboperation']))
            elif expectation == 'activity':
                if not counts:
                    miss('activity_missing')
            else:  # activity_or_zero
                if meta.get('noop_branch_checked') and counts:
                    fail('noop_branch_has_activity')

        # aggregate per family/suboperation/rank
        families = {}
        for oid, counts in by_offer_counts.items():
            meta = declared.get(oid)
            if not isinstance(meta, dict):
                continue
            key = str(meta.get('rank', rank))
            family = families.setdefault(key, {})
            sub = family.setdefault(meta['family'], {})
            row = sub.setdefault(meta['suboperation'], Counter())
            row.update(counts)
        families = {rank_key: {family: {sub: dict(counts)
                                        for sub, counts in subs.items()}
                               for family, subs in body.items()}
                    for rank_key, body in families.items()}

        eager_records = sum(j['count'] for j in joins
                            if j['graph_id'] == 0)
        eager_green_records = sum(eager_green_by_kind.values())
        eager_other_records = sum(eager_other_by_kind.values())
        graph_joined = sum(j['count'] for j in joins if j['graph_id'])
        graph_out_count = sum(row['count'] for row in graph_outside)
        unjoined_graph_count = sum(row['count'] for row in missing_graph_rows)
        graph_ids = sorted({row['graph_id'] for row in graph_outside}
                           | {row['graph_id'] for row in missing_graph_rows}
                           | {j['graph_id'] for j in joins if j['graph_id']})
        engine_by_kind = Counter()
        for kind in KINDS:
            engine_by_kind[kind] = (eager_green_by_kind[kind]
                                    + eager_other_by_kind[kind]
                                    + graph_by_kind[kind])
        native_by_kind = Counter()
        for row in native.get('activities', []):
            native_by_kind[row['kind']] += row['count']
        if engine_by_kind != native_by_kind:
            miss('engine_partition_mismatch')
        graph_offer_rows = {}
        for oid in graph_api_offers:
            graph_offer_rows[str(oid)] = {
                'launch_count': len(graph_launches_by_offer.get(oid, ())),
                'node_counts': dict(census_joined.get(oid, {})),
                'census': declared.get(oid, {}).get('census'),
                'descriptor': declared.get(oid, {}).get('descriptor'),
            }
        unexplained = []
        if failures:
            unexplained = sorted(set(failures))
    except _BadEvidence as exc:
        missing.append('malformed_engine_evidence:' + str(exc))
        families = {}
        eager_records = 0
        graph_joined = 0
        graph_out_count = 0
        graph_ids = []
        census_declared = {}
        wrappers = []
        native_capacity = None
        native_reserved = None
        thread_type = None
        graph_api_offers = set()
        eager_green_records = 0
        eager_other_records = 0
        eager_green_by_kind = Counter()
        eager_other_by_kind = Counter()
        eager_other_by_role = Counter()
        graph_by_kind = Counter()
        engine_by_kind = Counter()
        native_by_kind = Counter()
        unjoined_graph_count = 0
        graph_offer_rows = {}
        unexplained = sorted(set(failures))
    except (KeyError, TypeError, ValueError):
        missing.append('malformed_engine_evidence')
        families = {}
        eager_records = 0
        graph_joined = 0
        graph_out_count = 0
        graph_ids = []
        census_declared = {}
        wrappers = []
        native_capacity = None
        native_reserved = None
        thread_type = None
        graph_api_offers = set()
        eager_green_records = 0
        eager_other_records = 0
        eager_green_by_kind = Counter()
        eager_other_by_kind = Counter()
        eager_other_by_role = Counter()
        graph_by_kind = Counter()
        engine_by_kind = Counter()
        native_by_kind = Counter()
        unjoined_graph_count = 0
        graph_offer_rows = {}
        unexplained = sorted(set(failures))

    verdict = 'FAIL' if failures else ('INCONCLUSIVE' if missing
                                       else 'PASS')
    return {
        'schema': SCHEMA,
        'verdict': verdict,
        'window': window,
        'thread_id_type': thread_type,
        'native': _native_summary(native) if isinstance(native, dict) else {},
        'offers': offers,
        'declared': declared,
        'joins': joins,
        'counts': families,
        'families': families,
        'binding': (None if window.get('binding') is None
                    else window['binding']),
        'native_window': (native.get('window')
                          if isinstance(native, dict) else None),
        'graph': {'joined_records': graph_joined,
                  'outside_engine_ledger': graph_outside,
                  'outside_count': graph_out_count,
                  'unjoined_rows': missing_graph_rows,
                  'unjoined_count': unjoined_graph_count,
                  'ids': graph_ids,
                  'census_declared': census_declared,
                  'offers': graph_offer_rows,
                  'by_kind': dict(graph_by_kind)},
        'capacity': {'offer_capacity': window.get('max_offers'),
                     'offers': len(offers),
                     'native_capacity': native_capacity,
                     'native_reserved_bytes': native_reserved,
                     'native_errors': (native.get('errors')
                                       if isinstance(native, dict) else None),
                     'native_overflow': (native.get('overflow')
                                         if isinstance(native, dict) else None)},
        'eager_records': eager_records,
        'eager_green_records': eager_green_records,
        'eager_other_records': eager_other_records,
        'eager_green_by_kind': dict(eager_green_by_kind),
        'eager_other_by_kind': dict(eager_other_by_kind),
        'eager_other_by_role': dict(eager_other_by_role),
        'eager_by_kind': {kind: (eager_green_by_kind[kind]
                                 + eager_other_by_kind[kind])
                          for kind in KINDS},
        'graph_records': graph_joined + graph_out_count,
        'unjoined_graph_records': unjoined_graph_count,
        'engine_by_kind': dict(engine_by_kind),
        'native_by_kind': dict(native_by_kind),
        'joined_records': sum(j['count'] for j in joins),
        'wrappers': wrappers,
        'unexplained': unexplained,
        'failures': sorted(set(failures)),
        'missing': sorted(set(missing)),
        'notes': notes,
    }


def validate_receipt(receipt):
    """Reject a tampered or internally inconsistent engine receipt."""
    if not isinstance(receipt, dict) or receipt.get('schema') != SCHEMA:
        raise ValueError('engine receipt schema')
    verdict = receipt.get('verdict')
    if verdict not in ('PASS', 'FAIL', 'INCONCLUSIVE'):
        raise ValueError('engine receipt verdict')
    failures = receipt.get('failures')
    missing = receipt.get('missing')
    if not isinstance(failures, list) or not isinstance(missing, list):
        raise ValueError('engine receipt reasons')
    if not all(isinstance(x, str) for x in failures + missing):
        raise ValueError('engine receipt reason type')
    want = 'FAIL' if failures else ('INCONCLUSIVE' if missing else 'PASS')
    if verdict != want:
        raise ValueError('engine receipt verdict/reason mismatch')
    joins = receipt.get('joins')
    if not isinstance(joins, list):
        raise ValueError('engine receipt joins')
    total = 0
    for join in joins:
        if not isinstance(join, dict):
            raise ValueError('engine receipt join')
        count = join.get('count')
        if type(count) is not int or count < 1:
            raise ValueError('engine receipt join count')
        total += count
    if receipt.get('joined_records') != total:
        raise ValueError('engine receipt join total')
    if verdict == 'PASS' and receipt.get('unexplained'):
        raise ValueError('engine receipt unexplained ids')
    binding = receipt.get('binding')
    if binding is None:
        raise ValueError('engine receipt binding')
    try:
        binding = _coerce_binding(binding)
    except OfferError:
        raise ValueError('engine receipt binding') from None
    if binding.native_end_ns is None:
        raise ValueError('engine receipt window not finalized')
    window = receipt.get('window')
    if not isinstance(window, dict) or window.get('binding') != binding.as_record():
        raise ValueError('engine receipt window binding')
    native_window = receipt.get('native_window')
    if not isinstance(native_window, dict) \
            or native_window.get('start_ns') != binding.native_start_ns \
            or native_window.get('end_ns') != binding.native_end_ns \
            or native_window.get('route_generation') != binding.route_generation:
        raise ValueError('engine receipt native window')
    if verdict == 'PASS' and receipt.get('native', {}).get('complete') is not True:
        raise ValueError('engine receipt correlation completeness')
    for key in ('eager_green_records', 'eager_other_records',
                'eager_records', 'unjoined_graph_records'):
        value = receipt.get(key)
        if type(value) is not int or value < 0:
            raise ValueError('engine receipt counts')
    if receipt['eager_records'] != (receipt['eager_green_records']
                                    + receipt['eager_other_records']):
        raise ValueError('engine receipt eager total')
    if verdict == 'PASS' and receipt['unjoined_graph_records']:
        raise ValueError('engine receipt unjoined graph')
    return True


# --- composed placement predicate ------------------------------------------
# The raw receipt is evidence and is never edited. These arrays are the exact
# raw vocabulary; a composed result is refused when a new/unknown field or
# reason appears.
_RAW_COUNT_KEYS = (
    'context_records', 'kernel_records', 'copy_records', 'resource_events',
    'unknown_activity_records', 'unknown_resource_events',
    'records_before_window', 'records_after_window', 'records_in_window',
    'outside_scope_records', 'outside_scope_graph_records',
    'unattributed_expected_context', 'unattributed_other_context',
    'unknown_graph_records', 'ambiguous_graph_records', 'unknown_node_records',
    'ambiguous_node_records', 'kind_mismatch_records',
    'missing_node_id_records', 'boundary_overlap_records',
    'unknown_timestamp_records', 'incomplete_records',
    'invalid_interval_records', 'known_wrong_context_records',
    'buffers_requested', 'buffers_issued', 'buffers_completed',
    'buffers_issued_in_window', 'buffers_completed_in_window',
    'buffers_refused', 'buffers_outstanding', 'dropped_records_total',
    'drop_query_errors', 'parse_errors', 'invalid_records', 'callback_errors',
    'callback_reentrant_refused', 'duplicate_buffer_completions',
    'dependency_events', 'outside_scope_updates', 'graph_passes',
    'alias_type_disagreements', 'alias_type_memcpy_disagreements',
    'alias_type_memset_disagreements',
    'alias_type_reported_kernel_disagreements',
)
_RAW_FLAG_KEYS = (
    'permanent_overflow', 'buffer_refused', 'buffer_outstanding', 'parse_error',
    'unknown_activity', 'unknown_resource', 'drop_seen', 'drop_query_error',
    'flush_failed', 'update_seen', 'destroy_seen', 'inventory_changed',
    'expected_context_missing', 'expected_context_duplicate',
    'expected_context_contradictory', 'known_wrong_context',
    'known_wrong_greenness', 'known_wrong_sm', 'context_destroyed',
    'outside_scope_graph_records', 'no_exercised_graph', 'link_overflow',
    'reentrant_refused',
)
_RAW_RING_KEYS = ('context', 'kernel', 'copy', 'resource', 'aliases')
_RAW_ZERO_COUNTS = (
    'unknown_activity_records', 'unknown_resource_events',
    'outside_scope_graph_records', 'unknown_graph_records',
    'ambiguous_graph_records', 'unknown_node_records',
    'ambiguous_node_records', 'kind_mismatch_records',
    'missing_node_id_records', 'boundary_overlap_records',
    'unknown_timestamp_records', 'incomplete_records',
    'invalid_interval_records', 'known_wrong_context_records',
    'buffers_refused', 'buffers_outstanding', 'dropped_records_total',
    'drop_query_errors', 'parse_errors', 'invalid_records',
    'callback_errors', 'callback_reentrant_refused',
    'duplicate_buffer_completions',
)
_DISCHARGE_NATIVE = 'unattributed_expected_context'
_DISCHARGE_DERIVED = frozenset({'unattributed_expected_context',
                                'replay_incomplete'})
_WORK_KINDS = ('kernel', 'memcpy', 'memset')


def _reason_list(value, code):
    _need(isinstance(value, (list, tuple)), code)
    _need(all(isinstance(item, str) and item for item in value), code)
    return list(value)


def _kind_int_map(value, code):
    _need(isinstance(value, dict), code)
    out = {}
    for kind in _WORK_KINDS:
        out[kind] = _int(value.get(kind, 0), code)
    _need(set(value).issubset(set(_WORK_KINDS)), code)
    return out


def _raw_receipt_consistency(placement):
    """Shape and cross-field consistency of the unchanged raw receipt."""
    _need(isinstance(placement, dict), 'shape')
    _need(placement.get('schema') == PLACEMENT_SCHEMA, 'schema')
    verdict = placement.get('verdict')
    _need(verdict in ('PASS', 'FAIL', 'INCONCLUSIVE'), 'verdict')
    reasons = _reason_list(placement.get('reasons'), 'reasons')
    complete = placement.get('complete')
    _need(type(complete) is bool, 'complete')
    if verdict == 'PASS':
        _need(not reasons and complete is True, 'forged_pass')
    elif verdict == 'INCONCLUSIVE':
        _need(bool(reasons) and complete is False, 'inconclusive_reasons')
    else:
        _need(bool(reasons), 'fail_reasons')
    replay = placement.get('replay')
    _need(isinstance(replay, dict), 'replay_shape')
    _need(replay.get('verdict') == verdict, 'replay_verdict')
    _need(type(replay.get('complete')) is bool, 'replay_complete')
    native_reasons = _reason_list(replay.get('reasons'), 'native_reasons')
    _need(replay['complete'] is (not native_reasons), 'native_complete_reasons')
    _need(('replay_incomplete' in reasons) is (not replay['complete']), 'derived_incomplete')
    _need(set(native_reasons) <= set(reasons), 'native_reasons_not_retained')
    static = placement.get('static_structure')
    _need(isinstance(static, dict), 'static_shape')
    _need(static.get('verdict') in ('PASS', 'INCONCLUSIVE'), 'static_verdict')
    _need(type(static.get('complete')) is bool, 'static_complete_shape')
    _need((static.get('verdict') == 'PASS') is bool(static.get('complete')),
          'static_verdict_complete')
    _reason_list(static.get('reasons'), 'static_reasons')
    for key in ('counts', 'flags', 'window', 'context', 'mapping',
                'ring_overflows', 'graph_slots'):
        _need(isinstance(replay.get(key), (dict, list, tuple)),
              'replay_' + key)
    for key in ('coverage', 'bank', 'caller_proof', 'source_identities'):
        _need(isinstance(placement.get(key), dict), key + '_shape')
    _need(isinstance(placement.get('binding'), dict), 'binding_shape')
    limits = placement.get('limits')
    _need(isinstance(limits, dict) and limits, 'limits_shape')
    _need(all(type(value) is int and value >= 0
              for value in limits.values()), 'limits_shape')
    _need(type(placement.get('reserved_bytes')) is int
          and placement['reserved_bytes'] >= 0, 'reserved_bytes')
    return verdict, reasons, native_reasons


def _raw_fail_signals(placement):
    """Known placement violations outrank unrelated missing evidence."""
    signals = []
    replay = placement.get('replay')
    if not isinstance(replay, dict):
        return signals
    flags = replay.get('flags')
    if isinstance(flags, dict):
        for key in ('known_wrong_context', 'known_wrong_greenness',
                    'known_wrong_sm', 'context_destroyed'):
            if flags.get(key) is True:
                signals.append(key)
    counts = replay.get('counts')
    if isinstance(counts, dict):
        if counts.get('known_wrong_context_records', 0):
            signals.append('known_wrong_context_records')
    context = replay.get('context')
    if isinstance(context, dict):
        if context.get('is_green') is False and context.get('records', 0):
            signals.append('expected_context_not_green')
        if context.get('green_true') == 0 and context.get('records', 0):
            signals.append('expected_context_not_green')
        if context.get('sm_mismatched', 0) and context.get('sm_matching') == 0:
            signals.append('expected_context_sm_mismatch')
    return sorted(set(signals))


def _raw_label_index(label, count, code):
    _need(isinstance(label, str), code)
    for prefix in ('graph:', 'descriptor:'):
        if label.startswith(prefix):
            rest = label[len(prefix):]
            _need(rest.isdigit(), code)
            index = int(rest)
            _need(0 <= index < count, code)
            return index
    raise _BadEvidence(code)


def _raw_identity_binding(placement, binding):
    raw_binding = placement.get('binding')
    _need(raw_binding == binding.observer_identity, 'observer_identity_mismatch')
    _need(placement.get('source_identities') == binding.helper_identity,
          'helper_identity_mismatch')
    bank = placement.get('bank')
    _need(bank_identity(bank) == binding.bank_identity,
          'bank_identity_mismatch')
    _need(_int(bank.get('storage_epoch'), 'bank_storage', 1)
          == binding.storage_epoch, 'bank_storage_mismatch')
    _need(_int(bank.get('route_generation'), 'bank_route', 1)
          == binding.route_generation, 'bank_route_mismatch')
    _need(bank.get('green_context_bound') is True,
          'bank_green_context_unbound')
    proof = placement.get('caller_proof')
    _need(_int(proof.get('worker_rank'), 'proof_rank') == binding.rank,
          'proof_rank_mismatch')
    _need(proof.get('worker_incarnation') == binding.worker_incarnation,
          'proof_worker_incarnation_mismatch')
    _need(proof.get('engine_incarnation') == binding.engine_incarnation,
          'proof_engine_incarnation_mismatch')
    _need(_int(proof.get('route_generation'), 'proof_route')
          == binding.route_generation, 'proof_route_mismatch')
    _need(_int(proof.get('storage_epoch'), 'proof_storage', 1)
          == binding.storage_epoch, 'proof_storage_mismatch')
    _need(proof.get('route_generation_matches') is True,
          'proof_route_unmatched')
    _need(proof.get('storage_epoch_matches') is True,
          'proof_storage_unmatched')
    _need(proof.get('controller_drained') is True, 'drain_not_completed')
    _need(_int(proof.get('relevant_consumers'), 'proof_consumers') > 0,
          'drain_consumers_missing')
    _need(proof.get('verified_by_observer') is False,
          'proof_verifier_shape')
    replay = placement['replay']
    window = replay.get('window')
    _need(_int(window.get('start_ns'), 'window_start', 1)
          == binding.native_start_ns, 'window_start_mismatch')
    _need(_int(window.get('end_ns'), 'window_end', 1)
          == binding.native_end_ns, 'window_end_mismatch')
    _need(window.get('flushed') is True, 'window_not_flushed')
    _need(window.get('flush_forced') is False, 'window_flush_forced')
    _need(_int(window.get('route_generation'), 'window_route')
          == binding.route_generation, 'window_route_mismatch')
    coverage = placement.get('coverage')
    inventory = coverage.get('descriptor_inventory')
    _need(isinstance(inventory, (list, tuple)), 'inventory_shape')
    labels = list(bank.get('capture_order_labels') or ())
    _need(len(labels) == len(inventory), 'inventory_bank_mismatch')
    _need(labels == list(inventory), 'inventory_bank_mismatch')
    return inventory


def _raw_graph_proof(placement, binding, engine, raw_verdict):
    """Independently prove every retained graph condition.

    Static inventory, per-slot node records, exact per-node multiplicity, the
    router census and the whole-window counters must agree. Raw PASS additionally
    requires the raw receipt's own replay-qualified set; an INC discharge does
    not, because the wrapper clears it when the eager condition is unresolved.
    """
    replay = placement['replay']
    static = placement['static_structure']
    coverage = placement['coverage']
    _need(static.get('verdict') == 'PASS' and static.get('complete') is True,
          'static_not_complete')
    _need(not static.get('reasons'), 'static_reasons')
    counts = replay.get('counts')
    flags = replay.get('flags')
    context = replay.get('context')
    mapping = replay.get('mapping')
    rings = replay.get('ring_overflows')
    _need(set(counts) == set(_RAW_COUNT_KEYS), 'counts_keys')
    for key in _RAW_COUNT_KEYS:
        _int(counts[key], 'counts_' + key)
    _need(set(flags) == set(_RAW_FLAG_KEYS), 'flags_keys')
    for key in _RAW_FLAG_KEYS:
        _need(type(flags[key]) is bool, 'flags_' + key)
        _need(flags[key] is False, 'raw_flag_' + key)
    _need(isinstance(rings, dict) and set(rings) == set(_RAW_RING_KEYS),
          'ring_keys')
    for key in _RAW_RING_KEYS:
        _int(rings[key], 'ring_' + key)
        _need(rings[key] == 0, 'ring_overflow_' + key)
    for key in _RAW_ZERO_COUNTS:
        _need(counts[key] == 0, 'raw_count_' + key)
    _need(counts['buffers_issued_in_window']
          == counts['buffers_completed_in_window'], 'buffer_outstanding')
    _need(counts['records_in_window']
          == counts['kernel_records'] + counts['copy_records'],
          'records_by_kind')

    _need(context.get('bound') is True, 'context_bound')
    _need(context.get('duplicate') is False, 'context_duplicate')
    _need(context.get('contradictory') is False, 'context_contradictory')
    _need(context.get('is_green') is True, 'context_green')
    _need(_int(context.get('expected_sms'), 'context_sms', 1),
          'context_sms')
    _need(_int(context.get('records'), 'context_records', 1),
          'context_records')
    _need(_int(context.get('green_true'), 'context_green_true', 1),
          'context_green_true')
    _need(_int(context.get('green_false'), 'context_green_false') == 0,
          'context_green_false')
    _need(_int(context.get('sm_known'), 'context_sm_known'),
          'context_sm_known')
    _need(_int(context.get('sm_matching'), 'context_sm_matching'),
          'context_sm_matching')
    _need(_int(context.get('sm_mismatched'), 'context_sm_mismatched') == 0,
          'context_sm_mismatched')
    _need(context['sm_mismatched'] == 0, 'context_sm_mismatch')
    _need(_int(context.get('num_multiprocessors'), 'context_sms'),
          'context_sms')
    _need(type(context.get('parent_context_present')) is bool,
          'context_parent')

    _need(mapping.get('ambiguous_graph_ids') is False,
          'mapping_ambiguous_graph')
    _need(mapping.get('node_overlap') is False, 'mapping_node_overlap')
    _need(_int(mapping.get('alias_unknown'), 'mapping_alias_unknown')
          == 0, 'mapping_alias_unknown')
    for key in ('graph_ids', 'node_keys', 'node_aliases'):
        _int(mapping.get(key), 'mapping_' + key)
    _need(mapping['graph_ids'] > 0, 'mapping_graph_ids')

    static_slots = static.get('graph_slots')
    window_slots = replay.get('graph_slots')
    _need(isinstance(static_slots, (list, tuple)), 'static_slots_shape')
    _need(isinstance(window_slots, (list, tuple)), 'window_slots_shape')
    _need(_int(static.get('graph_count'), 'graph_count')
          == len(static_slots), 'graph_count')
    graph_count = len(static_slots)
    _need(graph_count > 0, 'graph_count_zero')
    _need(len(window_slots) == graph_count, 'graph_slot_count')

    inventory = _raw_identity_binding(placement, binding)
    _need(len(inventory) == graph_count, 'inventory_count')
    for index, label in enumerate(inventory):
        _need(_raw_label_index(label, graph_count, 'inventory_label')
              == index, 'inventory_order')
    declared_target = coverage.get('declared_target')
    if declared_target is not None:
        _need(isinstance(declared_target, str) and declared_target,
              'declared_target')
        target = _raw_label_index(declared_target, graph_count,
                                  'declared_target')
        _need(binding.declared_target
              == 'graph:' + str(target), 'binding_target_mismatch')
    else:
        # The owner binding carries the target; older native receipts may
        # omit it. A present raw target must still agree with the binding.
        target = _raw_label_index(binding.declared_target, graph_count,
                                  'binding_target')
        _need('graph:' + str(target) == binding.declared_target,
              'binding_target')
    # Native slot descriptors are inventory labels (``descriptor:N``), not
    # engine scheduler descriptors. The owner-supplied binding is the only
    # place that maps each retained graph label to its engine descriptor key,
    # and that mapping must be one-to-one over the inventory.
    _need(set(binding.descriptor_labels) == {
        'graph:' + str(i) for i in range(graph_count)
    }, 'binding_descriptor_labels')
    _need(len(set(binding.descriptor_labels.values())) == graph_count,
          'descriptor_ambiguous')

    static_by_index = {}
    for slot in static_slots:
        _need(isinstance(slot, dict), 'static_slot_shape')
        index = _raw_label_index(slot.get('label'), graph_count,
                                 'static_slot_label')
        _need(index not in static_by_index, 'static_slot_duplicate')
        static_by_index[index] = slot
    _need(set(static_by_index) == set(range(graph_count)),
          'static_slot_coverage')
    window_by_index = {}
    for slot in window_slots:
        _need(isinstance(slot, dict), 'window_slot_shape')
        index = _raw_label_index(slot.get('label'), graph_count,
                                 'window_slot_label')
        _need(index not in window_by_index, 'window_slot_duplicate')
        window_by_index[index] = slot
    _need(set(window_by_index) == set(range(graph_count)),
          'window_slot_coverage')

    observed = coverage.get('observed_descriptors')
    unexercised = coverage.get('unexercised_descriptors')
    qualified = coverage.get('replay_qualified')
    _need(isinstance(observed, (list, tuple)), 'observed_shape')
    _need(isinstance(unexercised, (list, tuple)), 'unexercised_shape')
    _need(isinstance(qualified, (list, tuple)), 'qualified_shape')
    observed_idx = {_raw_label_index(x, graph_count, 'observed_label')
                    for x in observed}
    unexercised_idx = {_raw_label_index(x, graph_count, 'unexercised_label')
                       for x in unexercised}
    qualified_idx = {_raw_label_index(x, graph_count, 'qualified_label')
                     for x in qualified}
    _need(not (observed_idx & unexercised_idx), 'coverage_overlap')
    _need(observed_idx | unexercised_idx == set(range(graph_count)),
          'coverage_partition')
    _need(target in observed_idx, 'target_unexercised')
    _need(qualified_idx <= observed_idx, 'qualified_unobserved')
    if raw_verdict == 'PASS':
        _need(target in qualified_idx, 'target_not_qualified')
    else:
        _need(not qualified_idx, 'stale_replay_qualified')

    router = binding.router_census
    router_idx = {}
    for label, count in router['graphs'].items():
        index = _raw_label_index(label, graph_count, 'router_label')
        _need(index not in router_idx, 'router_label_duplicate')
        router_idx[index] = _int(count, 'router_count', 1)
    # ``graph_passes`` is the native graph-mapping fixpoint pass count
    # (flash_s5_replay_observer.cpp, kMaxGraphPasses loop), not a replay
    # count: job9b reports 2 passes for 1 replay. Router multiplicity is
    # proven per slot below from executions and per-node bounds.
    _need(_int(counts['graph_passes'], 'graph_passes', 1),
          'graph_passes')
    _need(set(router_idx) == observed_idx, 'router_coverage')
    _need(sum(router_idx.values()) == router['total'], 'router_sum')

    graph_by_kind = Counter()
    slot_proofs = {}
    for index in range(graph_count):
        sslot = static_by_index[index]
        wslot = window_by_index[index]
        label = 'graph:' + str(index)
        _need(sslot.get('label') == label and wslot.get('label') == label,
              'slot_label')
        _need(sslot.get('static_complete') is True, 'static_incomplete')
        _need(not sslot.get('reasons'), 'static_slot_reasons')
        _need(not sslot.get('unsupported'), 'static_unsupported')
        _need(not sslot.get('unknown_type'), 'static_unknown_type')
        _need(sslot.get('graph_id_known') is True, 'static_graph_id')
        _need(sslot.get('node_ids_known') is True, 'static_node_ids')
        node_count = _int(sslot.get('node_count'), 'static_node_count', 1)
        node_types = sslot.get('node_types')
        _need(isinstance(node_types, dict), 'static_node_types')
        _need(all(type(v) is int and v >= 0 for v in node_types.values()),
              'static_node_types')
        _need(sum(node_types.values()) == node_count, 'static_node_total')
        required = _kind_int_map(sslot.get('required'),
                                 'static_required')
        for kind in _WORK_KINDS:
            _need(required[kind] == node_types.get(kind, 0),
                  'static_required')

        _need(wslot.get('descriptor') == 'descriptor:' + str(index),
              'slot_descriptor_label')
        _need(wslot.get('node_count') == node_count, 'slot_node_count')
        _need(wslot.get('node_types') == node_types, 'slot_node_types')
        _need(_kind_int_map(wslot.get('required'), 'slot_required')
              == required, 'slot_required')
        exercised = wslot.get('exercised')
        _need(type(exercised) is bool, 'slot_exercised')
        _need(exercised is (index in observed_idx), 'slot_exercised_coverage')
        replays = router_idx.get(index, 0)
        _need(exercised is (replays > 0), 'slot_router_coverage')
        observed_kinds = _kind_int_map(wslot.get('observed'), 'slot_observed')
        missing_kinds = _kind_int_map(wslot.get('missing'), 'slot_missing')
        for kind in _WORK_KINDS:
            _need(observed_kinds[kind] + missing_kinds[kind] == required[kind],
                  'slot_observed_total')
            if exercised:
                _need(missing_kinds[kind] == 0, 'slot_missing_node')
            else:
                # Unused bank entries are retained inventory. Native reports
                # every node as unobserved, not as a qualified replay.
                _need(observed_kinds[kind] == 0, 'unexercised_observed_node')
                _need(missing_kinds[kind] == required[kind],
                      'unexercised_inventory_mismatch')
        if exercised:
            _need(not wslot.get('missing_nodes'), 'slot_missing_nodes')
        else:
            _need(isinstance(wslot.get('missing_nodes'), (list, tuple)),
                  'unexercised_missing_shape')
            _need(bool(wslot['missing_nodes']) is bool(sum(required.values())),
                  'unexercised_missing_inventory')
        _need(not wslot.get('context_mismatch_nodes'),
              'slot_context_mismatch')
        _need(wslot.get('graph_id_known') is True, 'slot_graph_id')
        _need(wslot.get('node_ids_known') is True, 'slot_node_ids')
        _need(wslot.get('static_complete') is True, 'slot_static_complete')
        _need(not wslot.get('unsupported'), 'slot_unsupported')
        _need(not wslot.get('unknown_type'), 'slot_unknown_type')
        _need(wslot.get('inventory_changed') is False,
              'slot_inventory_changed')
        _need(wslot.get('update_seen') is False, 'slot_update_seen')
        _need(wslot.get('destroy_seen') is False, 'slot_destroy_seen')
        _need(wslot.get('mapping_ambiguous') is False,
              'slot_mapping_ambiguous')
        _need(wslot.get('complete') is exercised, 'slot_incomplete')
        _need(not wslot.get('reasons'), 'slot_reasons')
        if raw_verdict == 'PASS':
            _need(wslot.get('replay_qualified') is (index in qualified_idx),
                  'slot_qualified')
        else:
            _need(wslot.get('replay_qualified') is False,
                  'slot_stale_qualified')
        expected_exec = {kind: node_types.get(kind, 0) * replays
                         for kind in _WORK_KINDS}
        for kind, key in (('kernel', 'kernel_executions'),
                          ('memcpy', 'copy_executions'),
                          ('memset', 'memset_executions')):
            _need(_int(wslot.get(key), 'slot_' + key)
                  == expected_exec[kind], 'slot_executions')
            graph_by_kind[kind] += expected_exec[kind]
        bounds = wslot.get('node_execution_bounds')
        _need(isinstance(bounds, dict)
              and set(bounds) == set(_WORK_KINDS), 'slot_bounds_shape')
        for kind in _WORK_KINDS:
            row = bounds[kind]
            _need(isinstance(row, dict) and set(row) == {'min', 'max'},
                  'slot_bounds_shape')
            low = _int(row['min'], 'slot_bound_min')
            high = _int(row['max'], 'slot_bound_max')
            # Per-node execution bounds: every node of this kind executed
            # exactly ``replays`` times (native emits min/max of per-node
            # executions; job9b: 1/1 for 1386 kernels under one replay).
            if node_types.get(kind, 0):
                _need(low == high == replays, 'slot_multiplicity')
            else:
                _need(low == high == 0, 'slot_multiplicity')
        slot_proofs[index] = {
            'descriptor': binding.descriptor_labels[label],
            'node_types': {kind: node_types.get(kind, 0)
                           for kind in _WORK_KINDS},
            'replays': replays,
        }

    target_nodes = window_by_index[target].get('node_types', {})
    _need(sum(target_nodes.get(kind, 0)
              for kind in _WORK_KINDS) > 0, 'target_no_work')
    graph_records = sum(graph_by_kind.values())
    _need(graph_by_kind['kernel'] <= counts['kernel_records'],
          'graph_kernel_exceeds')
    _need(graph_by_kind['memcpy'] + graph_by_kind['memset']
          <= counts['copy_records'], 'graph_copy_exceeds')
    expected = counts['unattributed_expected_context']
    other = counts['unattributed_other_context']
    _need(expected + other + graph_records
          == counts['records_in_window'], 'raw_partition')
    _need(counts['outside_scope_records'] == other,
          'outside_scope_partition')
    if raw_verdict == 'PASS':
        _need(expected == 0, 'raw_pass_with_expected')
    return {
        'graph_count': graph_count,
        'target': target,
        'target_label': 'graph:' + str(target),
        'observed': sorted('graph:' + str(i) for i in observed_idx),
        'qualified': sorted('graph:' + str(i) for i in qualified_idx),
        'router': {str(k): v for k, v in sorted(router_idx.items())},
        'graph_by_kind': {kind: graph_by_kind[kind]
                          for kind in _WORK_KINDS},
        'slots': slot_proofs,
        'raw_counts': {key: counts[key] for key in _RAW_COUNT_KEYS},
        'native_reasons': list(replay['reasons']),
        'static_verdict': static.get('verdict'),
        'context': dict(context),
        'flags': dict(flags),
    }


def _engine_graph_reconciliation(engine, binding, proof):
    """Join every graph offer to exactly one raw replay/census."""
    graph = engine.get('graph')
    _need(isinstance(graph, dict), 'engine_graph')
    offers = graph.get('offers')
    _need(isinstance(offers, dict) and offers, 'engine_graph_offers')
    _need(_int(graph.get('unjoined_count'), 'engine_graph_unjoined') == 0,
          'engine_graph_unjoined')
    descriptors = {}
    for index, slot in proof['slots'].items():
        label = 'graph:' + str(index)
        descriptor = slot['descriptor']
        _need(isinstance(descriptor, str), 'slot_descriptor')
        _need(descriptor not in descriptors, 'descriptor_ambiguous')
        descriptors[descriptor] = label
    _need(len(descriptors) == proof['graph_count'], 'descriptor_ambiguous')
    offers_by_label = Counter()
    offer_node_kinds = Counter()
    for key, row in offers.items():
        _need(isinstance(row, dict), 'engine_graph_offer_shape')
        descriptor = row.get('descriptor')
        if isinstance(descriptor, dict):
            # The engine receipt retains the scheduler descriptor fields; the
            # binding names the same descriptor by its compact key.
            descriptor = descriptor_key(descriptor)
        _need(isinstance(descriptor, str) and descriptor in descriptors,
              'engine_graph_descriptor')
        label = descriptors[descriptor]
        index = int(label.split(':', 1)[1])
        _need(_int(row.get('launch_count'), 'engine_launch', 1) == 1,
              'census_launch_multiplicity')
        census = _kind_int_map(row.get('census'), 'engine_census')
        _need(any(census.values()), 'engine_census_empty')
        _need(census == proof['slots'][index]['node_types'],
              'engine_census_mismatch')
        node_counts = _kind_int_map(row.get('node_counts'),
                                    'engine_node_counts')
        _need(census == node_counts, 'engine_census_mismatch')
        offers_by_label[label] += 1
        for kind in _WORK_KINDS:
            offer_node_kinds[kind] += census[kind]
    for index, slot in proof['slots'].items():
        label = 'graph:' + str(index)
        _need(offers_by_label[label] == slot['replays'],
              'census_replay_multiplicity')
    _need({kind: offer_node_kinds[kind] for kind in _WORK_KINDS}
          == proof['graph_by_kind'], 'graph_node_multiset')
    _need(_int(graph.get('joined_records'), 'graph_joined')
          == sum(proof['graph_by_kind'].values()), 'graph_joined_total')
    by_kind = _kind_int_map(graph.get('by_kind'), 'engine_graph_by_kind')
    _need(by_kind == proof['graph_by_kind'], 'engine_graph_by_kind')
    return offers_by_label


def _engine_binding_reconciliation(engine, binding):
    _need(engine.get('binding') == binding.as_record(),
          'engine_binding_mismatch')
    window = engine.get('window')
    _need(isinstance(window, dict), 'engine_window')
    _need(window.get('binding') == binding.as_record(),
          'engine_window_binding')
    _need(window.get('worker') == binding.worker, 'engine_worker_mismatch')
    _need(_int(window.get('rank'), 'engine_rank') == binding.rank,
          'engine_rank_mismatch')
    _need(_int(window.get('epoch'), 'engine_epoch', 1)
          == binding.engine_epoch, 'engine_epoch_mismatch')
    _need(_int(window.get('route_generation'), 'engine_route', 1)
          == binding.route_generation, 'engine_route_mismatch')
    _need(window.get('green_context_id') == binding.green_context_id,
          'engine_green_mismatch')
    _need(window.get('pp_stage') == binding.pp_stage,
          'engine_pp_mismatch')
    native_window = engine.get('native_window')
    _need(isinstance(native_window, dict), 'engine_native_window')
    _need(native_window.get('start_ns') == binding.native_start_ns,
          'engine_native_start')
    _need(native_window.get('end_ns') == binding.native_end_ns,
          'engine_native_end')
    _need(native_window.get('route_generation') == binding.route_generation,
          'engine_native_route')
    _need(engine.get('thread_id_type') in THREAD_ID_TYPES,
          'engine_thread_contract')


def _engine_other_context_owned(engine, binding):
    """Other-context eager records must be individually owned output/control.

    No generic non-green/main allowance: only reviewed non-green leaves, on
    their declared stream role, with that role marked non-green in the binding.
    """
    declared = engine.get('declared')
    _need(isinstance(declared, dict), 'engine_declared')
    joins = engine.get('joins')
    _need(isinstance(joins, list), 'engine_joins')
    for join in joins:
        if not isinstance(join, dict):
            raise _BadEvidence('engine_join_shape')
        if join.get('graph_id') or join.get('context_id') \
                == binding.green_context_id:
            continue
        meta = declared.get(join.get('offer'))
        if meta is None:
            meta = declared.get(str(join.get('offer')))
        _need(isinstance(meta, dict), 'engine_other_meta')
        wire = _wire(meta.get('family'), meta.get('suboperation'))
        role = meta.get('stream_role')
        _need(_NON_GREEN_LEAVES.get(wire) == role,
              'other_context_not_owned')
        stream = binding.stream_roles.get(role)
        _need(isinstance(stream, dict)
              and stream.get('non_green_control') is True,
              'other_context_role_undeclared')
        _need(stream.get('context_id') == join.get('context_id'),
              'other_context_identity')
        _need(stream.get('stream_id') == join.get('stream_id'),
              'other_context_stream')


def _engine_multiset_reconciliation(engine, proof):
    counts = proof['raw_counts']
    engine_kinds = _kind_int_map(engine.get('engine_by_kind'),
                                 'engine_by_kind')
    native_kinds = _kind_int_map(engine.get('native_by_kind'),
                                 'native_by_kind')
    _need(engine_kinds == native_kinds, 'engine_native_partition')
    _need(engine_kinds['kernel'] == counts['kernel_records'],
          'kernel_multiset')
    _need(engine_kinds['memcpy'] + engine_kinds['memset']
          == counts['copy_records'], 'copy_multiset')
    _need(_int(engine.get('joined_records'), 'engine_joined')
          == counts['records_in_window'], 'window_multiset')
    _need(_int(engine.get('eager_green_records'), 'engine_eager_green')
          == counts['unattributed_expected_context'],
          'eager_green_multiset')
    _need(_int(engine.get('eager_other_records'), 'engine_eager_other')
          == counts['unattributed_other_context'],
          'eager_other_multiset')
    _need(_int(engine.get('eager_records'), 'engine_eager')
          == counts['unattributed_expected_context']
          + counts['unattributed_other_context'], 'eager_multiset')


def compose_placement(placement, engine_receipt):
    """Compose the unchanged raw graph receipt with engine accounting.

    PASS requires independently proved graph qualification plus complete
    in-window engine accounting. A raw INCONCLUSIVE is discharged only when its
    sole native condition is a positive ``unattributed_expected_context`` and
    all derived consequences are independently explained. The raw receipt is
    never modified.
    """
    composed = {'schema': COMPOSED_SCHEMA, 'verdict': 'INCONCLUSIVE',
                'reasons': []}

    if isinstance(placement, dict) and placement.get('schema') == PLACEMENT_SCHEMA:
        if placement.get('verdict') == 'FAIL':
            composed.update(verdict='FAIL', placement_verdict='FAIL',
                            reasons=['raw_placement_fail'])
            return composed

    if not isinstance(engine_receipt, dict) \
            or engine_receipt.get('schema') != SCHEMA:
        composed['reasons'] = ['engine_schema']
        return composed
    try:
        validate_receipt(engine_receipt)
    except ValueError:
        composed['reasons'] = ['engine_receipt_invalid']
        return composed

    engine_verdict = engine_receipt.get('verdict')
    composed['engine_verdict'] = engine_verdict
    if engine_verdict == 'FAIL':
        composed['verdict'] = 'FAIL'
        composed['reasons'] = ['engine_fail:' + ','.join(
            engine_receipt.get('failures', []))]
        return composed

    try:
        raw_verdict, raw_reasons, native_reasons = _raw_receipt_consistency(
            placement)
    except _BadEvidence as exc:
        composed['reasons'] = ['placement_malformed:' + str(exc)]
        return composed
    composed['placement_verdict'] = raw_verdict
    composed['raw'] = {
        'schema': placement.get('schema'),
        'verdict': raw_verdict,
        'reasons': list(raw_reasons),
        'complete': placement.get('complete'),
        'native_reasons': list(native_reasons),
        'sha256': _sha(json.dumps(placement, sort_keys=True,
                                 separators=(',', ':'))),
    }

    fail_signals = _raw_fail_signals(placement)
    if raw_verdict == 'FAIL' or fail_signals:
        composed['verdict'] = 'FAIL'
        composed['reasons'] = sorted(set(
            ['placement_fail:' + ','.join(raw_reasons)] +
            ['placement_wrong_context:' + signal for signal in fail_signals]))
        return composed

    if engine_verdict != 'PASS':
        composed['reasons'] = ['engine_inconclusive']
        return composed

    window = engine_receipt.get('window')
    binding_record = engine_receipt.get('binding')
    if binding_record is None or not isinstance(window, dict) \
            or window.get('binding') is None:
        composed['reasons'] = ['binding_missing']
        return composed
    try:
        binding = _coerce_binding(binding_record)
    except OfferError:
        composed['reasons'] = ['binding_invalid']
        return composed
    composed['binding'] = binding.as_record()

    try:
        _engine_binding_reconciliation(engine_receipt, binding)
        proof = _raw_graph_proof(placement, binding, engine_receipt,
                                 raw_verdict)
        offers_by_label = _engine_graph_reconciliation(
            engine_receipt, binding, proof)
        _engine_other_context_owned(engine_receipt, binding)
        _engine_multiset_reconciliation(engine_receipt, proof)
    except _BadEvidence as exc:
        composed['verdict'] = 'INCONCLUSIVE'
        reason = str(exc)
        if reason.startswith('other_context_') \
                and reason != 'other_context_role_undeclared':
            # Ownership is evidence, not instrumentation: a known unowned
            # context is a placement violation, not missing data. A role
            # absent from the owner binding is missing evidence (INC).
            composed['verdict'] = 'FAIL'
        composed['reasons'] = ['placement_refused:' + reason]
        return composed

    origin = 'raw_pass'
    if raw_verdict == 'INCONCLUSIVE':
        if list(native_reasons) != [_DISCHARGE_NATIVE]:
            composed['reasons'] = [
                'placement_reason_not_dischargeable:'
                + ','.join(native_reasons)]
            return composed
        if not set(raw_reasons).issubset(_DISCHARGE_DERIVED) \
                or _DISCHARGE_NATIVE not in raw_reasons:
            composed['reasons'] = [
                'placement_reason_not_dischargeable:'
                + ','.join(raw_reasons)]
            return composed
        expected = proof['raw_counts']['unattributed_expected_context']
        if expected <= 0:
            composed['reasons'] = ['placement_expected_count_not_positive']
            return composed
        origin = 'discharged_eager_expected_context'

    composed['verdict'] = 'PASS'
    composed['reasons'] = []
    composed['discharge'] = {
        'origin': origin,
        'native_reasons': list(native_reasons),
        'raw_reasons': list(raw_reasons),
        'derived_reasons': sorted(set(raw_reasons)
                                 - {_DISCHARGE_NATIVE}),
        'expected_context_records':
            proof['raw_counts']['unattributed_expected_context'],
    }
    composed['graph'] = {
        'count': proof['graph_count'],
        'target': proof['target_label'],
        'observed': proof['observed'],
        'replay_qualified': proof['qualified'],
        'router': proof['router'],
        'by_kind': proof['graph_by_kind'],
        'offers_by_label': {k: v for k, v in sorted(offers_by_label.items())},
    }
    composed['reconciliation'] = {
        'raw_counts': proof['raw_counts'],
        'engine_by_kind': _kind_int_map(engine_receipt['engine_by_kind'],
                                        'engine_by_kind'),
        'native_by_kind': _kind_int_map(engine_receipt['native_by_kind'],
                                        'native_by_kind'),
        'joined_records': engine_receipt['joined_records'],
        'eager_green_records': engine_receipt['eager_green_records'],
        'eager_other_records': engine_receipt['eager_other_records'],
    }
    return composed


# --- pinned transforms ------------------------------------------------------
PINS = {
    'v1/worker/gpu/model_runner.py':
        'a49588ad1ab173b1b5b1695d28fa64da7bbfc5a13d7d32d7aaaf9781879069a2',
    'v1/worker/gpu/sample/sampler.py':
        '832c9945d201a1e730ff9b4b52076aa83bd4a5064e1e4cfea952cf84d4a8d1f0',
    'v1/worker/gpu/async_utils.py':
        '77e17a4570ead2be30ae9b00888cf077a3b873018c12ca0549b853bfda02c1ba',
    'v1/worker/gpu/cudagraph_utils.py':
        'b1423812da6d07933bbaf014fa21236c944af1c5f8e553aefb2de89af8afb384',
}
# Exact package inputs, reproduced from the pinned E2/A/B/D2 pipeline.
# The existing standalone A+B variant remains accepted for offline fixtures.
MODEL_RUNNER_AFTER_AB_SHA256 = (
    '1841a6dc838554c9f2df410c5a809f7eed8136b10de517d115b9563c8e598ffa')
MODEL_RUNNER_PACKAGE_SHA256 = '61aeb2053b423e4278b43bef90bcd582b762a4282777b3425751af19b25c36a9'
CUDAGRAPH_AFTER_WORKER_SHA256 = '283a654aa858fce0f29b1cbd761685dfee32bc3a02de731305452ddeadafc584'
ACCEPTED = {
    'v1/worker/gpu/model_runner.py': (
        PINS['v1/worker/gpu/model_runner.py'],
        MODEL_RUNNER_AFTER_AB_SHA256, MODEL_RUNNER_PACKAGE_SHA256,
    ),
    'v1/worker/gpu/sample/sampler.py': (
        PINS['v1/worker/gpu/sample/sampler.py'],
    ),
    'v1/worker/gpu/async_utils.py': (
        PINS['v1/worker/gpu/async_utils.py'],
    ),
    'v1/worker/gpu/cudagraph_utils.py': (
        PINS['v1/worker/gpu/cudagraph_utils.py'], CUDAGRAPH_AFTER_WORKER_SHA256,
    ),
}
SOURCE_PINS = dict(PINS)

_MARKER = {
    'v1/worker/gpu/model_runner.py':
        'import vllm.v1.flash_e3.engine_offers as _flash_s5_engine\n',
    'v1/worker/gpu/sample/sampler.py':
        'import vllm.v1.flash_e3.engine_offers as _flash_s5_engine\n',
    'v1/worker/gpu/async_utils.py':
        'import vllm.v1.flash_e3.engine_offers as _flash_s5_engine\n',
    'v1/worker/gpu/cudagraph_utils.py':
        'import vllm.v1.flash_e3.engine_offers as _flash_s5_engine\n',
}


def _replace(text, old, new, count=1):
    if text.count(old) != count:
        raise ValueError('source anchor changed or duplicated: ' +
                         old[:80].replace('\n', '\\n'))
    return text.replace(old, new)


def _transform_rows(transforms):
    for row in transforms:
        if len(row) == 2:
            yield row[0], row[1], 1
        else:
            yield row[0], row[1], int(row[2])


def _run_call(indent, family, sub, target, *args):
    pad = ' ' * indent
    tail = ''.join(', ' + a for a in args)
    return (pad + '_flash_s5_engine.run(\n' +
            pad + '    self, "%s", "%s", %s%s\n' % (family, sub, target, tail) +
            pad + ')\n')


_RUNNER_SHARED = (
    ('import vllm.forward_context as _qsa_fc\n',
     'import vllm.forward_context as _qsa_fc\n'
     'import vllm.v1.flash_e3.engine_offers as _flash_s5_engine\n'),
    ('    def execute_model(\n',
     '    @_flash_s5_engine.step_method("execute")\n'
     '    def execute_model(\n'),
    ('    def sample_tokens(\n',
     '    @_flash_s5_engine.step_method("sample")\n'
     '    def sample_tokens(\n'),
    ('            self.update_pp_decode_requests()\n',
     _run_call(12, 'request_state_table', 'prep_state',
               'self.update_pp_decode_requests')),
    ('            self.finish_requests(scheduler_output)\n',
     _run_call(12, 'request_state_table', 'prep_state',
               'self.finish_requests', 'scheduler_output')),
    ('            self.free_states(scheduler_output)\n',
     _run_call(12, 'request_state_table', 'prep_state',
               'self.free_states', 'scheduler_output')),
    ('            self.add_requests(scheduler_output)\n',
     _run_call(12, 'request_state_table', 'prep_state',
               'self.add_requests', 'scheduler_output')),
    ('            self.update_requests(scheduler_output)\n',
     _run_call(12, 'request_state_table', 'prep_state',
               'self.update_requests', 'scheduler_output')),
    ('            self.block_tables.apply_staged_writes()\n',
     _run_call(12, 'request_state_table', 'prep_state',
               'self.block_tables.apply_staged_writes')),
    ('            if scheduler_output.total_num_scheduled_tokens == 0:\n'
     '                # No need to run the model.\n',
     '            if scheduler_output.total_num_scheduled_tokens == 0:\n'
     '                # No need to run the model.\n'
     '                _flash_s5_engine.mark_zero_work(\n'
     '                    self, "request_state_table", "no_work_step"\n'
     '                )\n'),
    # Both no-work paths call the stock connector. The branch distinguishes
    # the exact installed no-op from an active connector, which refuses.
    ('                # No need to run the model.\n'
     '                _flash_s5_engine.mark_zero_work(\n'
     '                    self, "request_state_table", "no_work_step"\n'
     '                )\n'
     '                empty_output = self.kv_connector.no_forward(scheduler_output)\n',
     '                # No need to run the model.\n'
     '                _flash_s5_engine.mark_zero_work(\n'
     '                    self, "request_state_table", "no_work_step"\n'
     '                )\n'
     '                empty_output = _flash_s5_engine.run(\n'
     '                    self, "kv_connector", "no_forward",\n'
     '                    self.kv_connector.no_forward, scheduler_output,\n'
     '                    branch=_flash_s5_engine.connector_branch(\n'
     '                        self.kv_connector),\n'
     '                )\n'),
    ('        if batch_desc.num_tokens == 0:\n'
     '            # All DP ranks have zero tokens to run.\n'
     '            empty_output = self.kv_connector.no_forward(scheduler_output)\n',
     '        if batch_desc.num_tokens == 0:\n'
     '            # All DP ranks have zero tokens to run.\n'
     '            empty_output = _flash_s5_engine.run(\n'
     '                self, "kv_connector", "no_forward",\n'
     '                self.kv_connector.no_forward, scheduler_output,\n'
     '                branch=_flash_s5_engine.connector_branch(\n'
     '                    self.kv_connector),\n'
     '            )\n'),
    # EC connector no-work merge is its own conditional operation.
    ('        return ModelRunnerOutput.with_ec_conn_output(\n'
     '            output,\n'
     '            self.ec_connector.no_forward(scheduler_output).ec_connector_output,\n'
     '        )\n',
     '        return ModelRunnerOutput.with_ec_conn_output(\n'
     '            output,\n'
     '            _flash_s5_engine.run(\n'
     '                self, "kv_connector", "ec_no_forward",\n'
     '                lambda: self.ec_connector.no_forward(\n'
     '                    scheduler_output\n'
     '                ).ec_connector_output,\n'
     '                branch=_flash_s5_engine.connector_branch(\n'
     '                    self.ec_connector),\n'
     '            ),\n'
     '        )\n'),
    # Ordinary pinned path gathers CPU/numpy state; dummy/adaptive are host-
    # only branches too. Never assert GPU work here.
    ('        batch_req_state, uniform_tok_count = self.gather_batch_req_state(\n'
     '            scheduler_output, dummy_run\n'
     '        )\n',
     '        batch_req_state, uniform_tok_count = _flash_s5_engine.run(\n'
     '            self, "request_state_table", "gather_batch_req_state",\n'
     '            self.gather_batch_req_state, scheduler_output, dummy_run,\n'
     '            branch=_flash_s5_engine.gather_branch(\n'
     '                self, scheduler_output, dummy_run),\n'
     '        )\n'),
    # DP descriptor selection plus conditional CPU-group collective. DP size 1
    # returns without a collective; a supplied sync is a reuse branch.
    ('        batch_desc, dp_sync = dispatch_cg_and_sync_dp(\n',
     '        batch_desc, dp_sync = _flash_s5_engine.run(\n'
     '            self, "forward_dispatch", "dp_dispatch",\n'
     '            dispatch_cg_and_sync_dp,\n'),
    ('            uniform_decode=uniform_tok_count == self.decode_query_len,\n'
     '        )\n',
     '            uniform_decode=uniform_tok_count == self.decode_query_len,\n'
     '            branch=_flash_s5_engine.dp_branch(self),\n'
     '        )\n'),
    ('            input_batch = self.prepare_inputs(\n',
     '            input_batch = _flash_s5_engine.run(\n'
     '                self, "input_attention", "prepare_inputs", '
     'self.prepare_inputs,\n'),
    ('            block_tables, slot_mappings = self.prepare_attn(input_batch)\n',
     '            block_tables, slot_mappings = _flash_s5_engine.run(\n'
     '                self, "input_attention", "prepare_attn", '
     'self.prepare_attn, input_batch,\n'
     '                branch=("pcp" if self.pcp_manager is not None\n'
     '                        else "ordinary"),\n'
     '            )\n'),
    ('            self.model_state.preprocess_state(\n',
     '            _flash_s5_engine.run(\n'
     '                self, "input_attention", "preprocess_state",\n'
     '                self.model_state.preprocess_state,\n'),
    ('            attn_metadata = self.model_state.prepare_attn(\n',
     '            attn_metadata = _flash_s5_engine.run(\n'
     '                self, "input_attention", "attn_metadata",\n'
     '                self.model_state.prepare_attn,\n'),
    ('                    inputs_embeds = self.model_state.prepare_inputs_embeds(\n'
     '                        scheduled_encoder_inputs, input_batch, self.req_states\n'
     '                    )\n',
     '                    inputs_embeds = _flash_s5_engine.run(\n'
     '                        self, "forward_dispatch", "prepare_inputs_embeds",\n'
     '                        self.model_state.prepare_inputs_embeds,\n'
     '                        scheduled_encoder_inputs, input_batch, self.req_states,\n'
     '                    )\n'),
    ('            **self.model_state.prepare_inputs(input_batch, self.req_states),\n',
     '            **_flash_s5_engine.run(\n'
     '                self, "forward_dispatch", "prepare_model_inputs",\n'
     '                self.model_state.prepare_inputs, input_batch, '
     'self.req_states,\n'
     '            ),\n'),
    # PP intermediate tensors are copied into per-step buffers on the main
    # stream; each copy is one declared leaf, not the whole forward.
    ('                else v[:n].copy_(intermediate_tensors.tensors[k][:n])\n',
     '                else _flash_s5_engine.run(\n'
     '                    self, "forward_dispatch", "pp_intermediate_copy",\n'
     '                    lambda: v[:n].copy_(\n'
     '                        intermediate_tensors.tensors[k][:n]\n'
     '                    ),\n'
     '                )\n'),
    ('        self.eplb.prepare_forward(\n'
     '            self.model_config, input_batch.num_tokens, ubatch_slices\n'
     '        )\n',
     '        _flash_s5_engine.run(\n'
     '            self, "forward_dispatch", "eplb_prepare",\n'
     '            self.eplb.prepare_forward,\n'
     '            self.model_config, input_batch.num_tokens, ubatch_slices,\n'
     '        )\n'),
    # Both installed connector paths: exact no-op or named active branch.
    ('            self.kv_connector.pre_forward(\n'
     '                **connector_kwargs, attn_metadata=attn_metadata\n'
     '            )\n',
     '            _flash_s5_engine.run(\n'
     '                self, "kv_connector", "pre_forward",\n'
     '                self.kv_connector.pre_forward,\n'
     '                **connector_kwargs, attn_metadata=attn_metadata,\n'
     '                branch=_flash_s5_engine.connector_branch(\n'
     '                    self.kv_connector),\n'
     '            )\n'),
    ('                self.kv_connector.pre_forward(**connector_kwargs)\n',
     '                _flash_s5_engine.run(\n'
     '                    self, "kv_connector", "pre_forward",\n'
     '                    self.kv_connector.pre_forward, **connector_kwargs,\n'
     '                    branch=_flash_s5_engine.connector_branch(\n'
     '                        self.kv_connector),\n'
     '                )\n'),
    ('        self.kv_connector.finish_forward()\n',
     '        _flash_s5_engine.run(\n'
     '            self, "kv_connector", "finish_forward",\n'
     '            self.kv_connector.finish_forward,\n'
     '            branch=_flash_s5_engine.connector_branch(\n'
     '                self.kv_connector),\n'
     '        )\n'),
    ('            # Post-step KV connector related operations.\n'
     '            kv_connector_output = self.kv_connector.post_forward(finished_req_ids)\n',
     '            # Post-step KV connector related operations.\n'
     '            kv_connector_output = _flash_s5_engine.run(\n'
     '                self, "kv_connector", "post_forward",\n'
     '                self.kv_connector.post_forward, finished_req_ids,\n'
     '                branch=_flash_s5_engine.connector_branch(\n'
     '                    self.kv_connector),\n'
     '            )\n'),
    ('        # Post-step KV connector related operations.\n'
     '        kv_connector_output = self.kv_connector.post_forward(finished_req_ids)\n'
     '        model_runner_output.kv_connector_output = kv_connector_output\n',
     '        # Post-step KV connector related operations.\n'
     '        kv_connector_output = _flash_s5_engine.run(\n'
     '            self, "kv_connector", "post_forward",\n'
     '            self.kv_connector.post_forward, finished_req_ids,\n'
     '            branch=_flash_s5_engine.connector_branch(\n'
     '                self.kv_connector),\n'
     '        )\n'
     '        model_runner_output.kv_connector_output = kv_connector_output\n'),
    # pool path uses the same declared connector operation.
    ('        # Post-step KV connector related operations.\n'
     '        kv_connector_output = self.kv_connector.post_forward(finished_req_ids)\n'
     '\n'
     '        if not self.is_last_pp_rank:\n',
     '        # Post-step KV connector related operations.\n'
     '        kv_connector_output = _flash_s5_engine.run(\n'
     '            self, "kv_connector", "post_forward",\n'
     '            self.kv_connector.post_forward, finished_req_ids,\n'
     '            branch=_flash_s5_engine.connector_branch(\n'
     '                self.kv_connector),\n'
     '        )\n'
     '\n'
     '        if not self.is_last_pp_rank:\n'),
    ('            all_decode_next = self.pp_handler.receive(input_batch)\n',
     '            all_decode_next = _flash_s5_engine.run(\n'
     '                self, "pp", "pp_receive", self.pp_handler.receive, '
     'input_batch\n'
     '            )\n'),
    ('            # Optimistically update num_computed_tokens for entire batch here.\n'
     '            # Will be adjusted for rejections if necessary in update_requests.\n'
     '            self.postprocess_num_computed_tokens(input_batch)\n',
     '            # Optimistically update num_computed_tokens for entire batch here.\n'
     '            # Will be adjusted for rejections if necessary in update_requests.\n'
     '            _flash_s5_engine.run(\n'
     '                self, "pp", "postprocess_num_computed",\n'
     '                self.postprocess_num_computed_tokens, input_batch,\n'
     '            )\n'),
    ('                self.model_state.postprocess_state(input_batch.idx_mapping, 0)\n',
     '                _flash_s5_engine.run(\n'
     '                    self, "pp", "postprocess_state",\n'
     '                    self.model_state.postprocess_state, '
     'input_batch.idx_mapping, 0,\n'
     '                )\n'),
    ('        self.model_state.postprocess_state(\n'
     '            idx_mapping, num_sampled, self.req_states.num_computed_tokens.gpu\n'
     '        )\n',
     '        _flash_s5_engine.run(\n'
     '            self, "postprocess", "postprocess_state",\n'
     '            self.model_state.postprocess_state,\n'
     '            idx_mapping, num_sampled, '
     'self.req_states.num_computed_tokens.gpu,\n'
     '        )\n'),
    # Runner sample is a container; its ordinary Sampler call is a different
    # container. Rejection sampling is a separately declared unsupported branch.
    ('    def sample(\n        self,\n        hidden_states: torch.Tensor,\n',
     '    @_flash_s5_engine.offer_method("logits_sampler", "runner_sample")\n'
     '    def sample(\n        self,\n        hidden_states: torch.Tensor,\n'),
    ('        if input_batch.num_reqs == 0:\n'
     '            # This rank owns no requests this step. It contributes an\n',
     '        if input_batch.num_reqs == 0:\n'
     '            _flash_s5_engine.mark_branch(\n'
     '                self, "logits_sampler", "sampler_call", "empty_owner"\n'
     '            )\n'
     '            # This rank owns no requests this step. It contributes an\n'),
    ('            sampler_output = self.sampler(logits, input_batch)\n',
     '            sampler_output = _flash_s5_engine.run(\n'
     '                self, "logits_sampler", "sampler_call",\n'
     '                self.sampler, logits, input_batch,\n'
     '            )\n'),
    ('            sampler_output = self.rejection_sampler(\n'
     '                logits,\n'
     '                input_batch,\n'
     '                # Draft logits are needed for probabilistic rejection sampling.\n'
     '                self.speculator.draft_logits,\n'
     '            )\n',
     '            _flash_s5_engine.mark_branch(\n'
     '                self, "logits_sampler", "sampler_call", "rejection"\n'
     '            )\n'
     '            sampler_output = _flash_s5_engine.run(\n'
     '                self, "sampler", "rejection",\n'
     '                self.rejection_sampler, logits, input_batch,\n'
     '                # Draft logits are needed for probabilistic rejection sampling.\n'
     '                self.speculator.draft_logits,\n'
     '            )\n'),
    ('            input_batch, sorted_logits_indices, grammar_output, shard_metadata = (\n'
     '                self.batch_sharder.shard_sampler_inputs(input_batch, grammar_output)\n'
     '            )\n',
     '            input_batch, sorted_logits_indices, grammar_output, shard_metadata = (\n'
     '                _flash_s5_engine.run(\n'
     '                    self, "sampler", "shard_inputs",\n'
     '                    self.batch_sharder.shard_sampler_inputs,\n'
     '                    input_batch, grammar_output, branch="active",\n'
     '                )\n'
     '            )\n'),
    ('            sample_hidden_states = hidden_states[sorted_logits_indices]\n'
     '            local_logits = self.model.compute_logits_local(sample_hidden_states)\n'
     '            logits = all_to_all_logits(local_logits, shard_metadata)\n',
     '            sample_hidden_states = _flash_s5_engine.run(\n'
     '                self, "logits", "select_hidden",\n'
     '                lambda: hidden_states[sorted_logits_indices],\n'
     '            )\n'
     '            local_logits = _flash_s5_engine.run(\n'
     '                self, "logits", "compute_local",\n'
     '                self.model.compute_logits_local, sample_hidden_states,\n'
     '            )\n'
     '            logits = _flash_s5_engine.run(\n'
     '                self, "logits", "tp_all_to_all",\n'
     '                all_to_all_logits, local_logits, shard_metadata,\n'
     '            )\n'),
    ('            sample_hidden_states = hidden_states[input_batch.logits_indices]\n'
     '            logits = self.model.compute_logits(sample_hidden_states)\n',
     '            sample_hidden_states = _flash_s5_engine.run(\n'
     '                self, "logits", "select_hidden",\n'
     '                lambda: hidden_states[input_batch.logits_indices],\n'
     '            )\n'
     '            logits = _flash_s5_engine.run(\n'
     '                self, "logits", "compute_full",\n'
     '                self.model.compute_logits, sample_hidden_states,\n'
     '            )\n'
     '            _flash_s5_engine.mark_branch(\n'
     '                self, "sampler", "shard_inputs", "no_sharder"\n'
     '            )\n'),
    ('            self.structured_outputs_worker.apply_grammar_bitmask(\n'
     '                logits,\n'
     '                input_batch,\n'
     '                grammar_output.structured_output_request_ids,\n'
     '                grammar_output.grammar_bitmask,\n'
     '            )\n',
     '            _flash_s5_engine.run(\n'
     '                self, "logits", "grammar",\n'
     '                self.structured_outputs_worker.apply_grammar_bitmask,\n'
     '                logits,\n'
     '                input_batch,\n'
     '                grammar_output.structured_output_request_ids,\n'
     '                grammar_output.grammar_bitmask,\n'
     '            )\n'),
    # Host-only branch truth for the branches that submit no leaf offer.
    ('            )\n\n        sampler_output: SamplerOutput | None\n',
     '            )\n'
     '        else:\n'
     '            _flash_s5_engine.mark_branch(\n'
     '                self, "logits", "grammar", "absent"\n'
     '            )\n\n'
     '        sampler_output: SamplerOutput | None\n'),
    ('            sampler_output = gather_sampler_output(\n'
     '                sampler_output,\n'
     '                shard_metadata,\n'
     '                device=self.device,\n'
     '                global_batch=global_input_batch,\n'
     '                local_batch=input_batch,\n'
     '                gather_num_nans=self.sampler.compute_nans,\n'
     '                logprobs_dims=self.sampler.get_logprobs_dims(\n'
     '                    global_input_batch.idx_mapping_np,\n'
     '                    # Rejection sampler does not return logprob token ids.\n'
     '                    include_token_ids=(\n'
     '                        global_input_batch.num_draft_tokens == 0\n'
     '                        or self.rejection_sampler is None\n'
     '                    ),\n'
     '                ),\n'
     '            )\n',
     '            sampler_output = _flash_s5_engine.run(\n'
     '                self, "sampler", "gather_output", gather_sampler_output,\n'
     '                sampler_output,\n'
     '                shard_metadata,\n'
     '                device=self.device,\n'
     '                global_batch=global_input_batch,\n'
     '                local_batch=input_batch,\n'
     '                gather_num_nans=self.sampler.compute_nans,\n'
     '                logprobs_dims=self.sampler.get_logprobs_dims(\n'
     '                    global_input_batch.idx_mapping_np,\n'
     '                    # Rejection sampler does not return logprob token ids.\n'
     '                    include_token_ids=(\n'
     '                        global_input_batch.num_draft_tokens == 0\n'
     '                        or self.rejection_sampler is None\n'
     '                    ),\n'
     '                ),\n'
     '            )\n'),
    ('                ),\n'
     '            )\n\n'
     '        assert sampler_output is not None\n',
     '                ),\n'
     '            )\n'
     '        else:\n'
     '            _flash_s5_engine.mark_branch(\n'
     '                self, "sampler", "gather_output", "unsharded"\n'
     '            )\n\n'
     '        assert sampler_output is not None\n'),
    ('        sampler_output, num_sampled, num_rejected = self.sample(\n',
     '        sampler_output, num_sampled, num_rejected = self.sample(\n'),
    ('            self.pp_handler.broadcast(\n',
     '            _flash_s5_engine.run(\n'
     '                self, "pp", "pp_broadcast", self.pp_handler.broadcast,\n'),
    ('        self.postprocess_sampled(\n',
     '        _flash_s5_engine.run(\n'
     '            self, "postprocess", "execute_postprocess", '
     'self.postprocess_sampled,\n'),
    ('    def postprocess_sampled(\n',
     '    @_flash_s5_engine.offer_method("postprocess", '
     '"postprocess_sampled")\n'
     '    def postprocess_sampled(\n'),
    # Conditional prompt logprob pass: not generation logprobs, not lm_head.
    ('        prompt_logprobs_dict = self.prompt_logprobs_worker.compute_prompt_logprobs(\n'
     '            self.model.compute_logits,\n'
     '            hidden_states,\n'
     '            input_batch,\n'
     '            self.req_states.all_token_ids.gpu,\n'
     '            self.req_states.num_computed_tokens.gpu,\n'
     '            self.req_states.prompt_len.np,\n'
     '        )\n',
     '        prompt_logprobs_dict = _flash_s5_engine.run(\n'
     '            self, "sampler", "prompt_logprobs",\n'
     '            self.prompt_logprobs_worker.compute_prompt_logprobs,\n'
     '            self.model.compute_logits,\n'
     '            hidden_states,\n'
     '            input_batch,\n'
     '            self.req_states.all_token_ids.gpu,\n'
     '            self.req_states.num_computed_tokens.gpu,\n'
     '            self.req_states.prompt_len.np,\n'
     '        )\n'),
    # Speculator: distinct proposal and writeback, plus the active branches.
    # The draft graph census is not installed, so proposal is unsupported.
    ('                self.speculator.prepare_watermarking(\n'
     '                    self.sampler._get_contexts(input_batch.idx_mapping),\n'
     '                    self.sampler.watermarking.gpu[input_batch.idx_mapping],\n'
     '                )\n',
     '                _flash_s5_engine.run(\n'
     '                    self, "speculator", "watermarking",\n'
     '                    lambda: self.speculator.prepare_watermarking(\n'
     '                        self.sampler._get_contexts(\n'
     '                            input_batch.idx_mapping\n'
     '                        ),\n'
     '                        self.sampler.watermarking.gpu[\n'
     '                            input_batch.idx_mapping\n'
     '                        ],\n'
     '                    ),\n'
     '                )\n', 2),
    ('                self.speculator.propose(\n'
     '                    input_batch=input_batch,\n'
     '                    attn_metadata=attn_metadata,\n'
     '                    slot_mappings=slot_mappings_by_layer,\n'
     '                    last_hidden_states=spec_hidden_states,\n'
     '                    aux_hidden_states=aux_hidden_states,\n'
     '                    num_sampled=torch.ones(\n',
     '                _flash_s5_engine.run(\n'
     '                    self, "speculator", "propose",\n'
     '                    lambda: self.speculator.propose(\n'
     '                    input_batch=input_batch,\n'
     '                    attn_metadata=attn_metadata,\n'
     '                    slot_mappings=slot_mappings_by_layer,\n'
     '                    last_hidden_states=spec_hidden_states,\n'
     '                    aux_hidden_states=aux_hidden_states,\n'
     '                    num_sampled=torch.ones(\n'),
    ('                    is_profile=is_profile,\n'
     '                )\n'
     '            self.step_timing.drafter_end()\n',
     '                    is_profile=is_profile,\n'
     '                    ),\n'
     '                )\n'
     '            self.step_timing.drafter_end()\n'),
    ('                draft_tokens = self.speculator.propose(\n'
     '                    input_batch,\n'
     '                    attn_metadata,\n'
     '                    slot_mappings_by_layer,\n'
     '                    spec_hidden_states,\n'
     '                    aux_hidden_states,\n'
     '                    num_sampled,\n'
     '                    num_rejected,\n'
     '                    self.req_states.last_sampled_tokens,\n'
     '                    self.req_states.next_prefill_tokens,\n'
     '                    self.sampler.sampling_states.temperature.gpu,\n'
     '                    self.sampler.sampling_states.seeds.gpu,\n',
     '                draft_tokens = _flash_s5_engine.run(\n'
     '                    self, "speculator", "propose",\n'
     '                    lambda: self.speculator.propose(\n'
     '                    input_batch,\n'
     '                    attn_metadata,\n'
     '                    slot_mappings_by_layer,\n'
     '                    spec_hidden_states,\n'
     '                    aux_hidden_states,\n'
     '                    num_sampled,\n'
     '                    num_rejected,\n'
     '                    self.req_states.last_sampled_tokens,\n'
     '                    self.req_states.next_prefill_tokens,\n'
     '                    self.sampler.sampling_states.temperature.gpu,\n'
     '                    self.sampler.sampling_states.seeds.gpu,\n'),
    ('                    mm_inputs=mm_inputs,\n'
     '                )\n'
     '            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens\n',
     '                    mm_inputs=mm_inputs,\n'
     '                    ),\n'
     '                )\n'
     '            with _flash_s5_engine.leaf(\n'
     '                self, "speculator", "draft_writeback"\n'
     '            ):\n'
     '                self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens\n'),
    ('                self.adaptive_verification.record_confidences(\n'
     '                    self.speculator.draft_token_confidence_probs, input_batch\n'
     '                )\n',
     '                _flash_s5_engine.run(\n'
     '                    self, "speculator", "adaptive_record",\n'
     '                    self.adaptive_verification.record_confidences,\n'
     '                    self.speculator.draft_token_confidence_probs, input_batch,\n'
     '                )\n'),
    ('            self.draft_tokens_handler.set_draft_tokens(\n'
     '                input_batch,\n'
     '                self.req_states.draft_tokens[input_batch.idx_mapping],\n'
     '            )\n',
     # The draft_tokens[idx_mapping] argument is itself a GPU gather; it stays
     # inside the offer (lambda), never evaluated before the push.
     '            _flash_s5_engine.run(\n'
     '                self, "speculator", "set_draft_tokens",\n'
     '                lambda: self.draft_tokens_handler.set_draft_tokens(\n'
     '                    input_batch,\n'
     '                    self.req_states.draft_tokens[input_batch.idx_mapping],\n'
     '                ),\n'
     '            )\n'),
    ('                self.pp_handler.broadcast_drafts(\n'
     '                    self.req_states.draft_tokens, input_batch\n'
     '                )\n',
     '                _flash_s5_engine.run(\n'
     '                    self, "pp", "pp_broadcast_drafts",\n'
     '                    self.pp_handler.broadcast_drafts,\n'
     '                    self.req_states.draft_tokens, input_batch,\n'
     '                )\n'),
)

_RUNNER_GRAPH_PINNED = (
    ('            model_output = self.cudagraph_manager.run_fullgraph(batch_desc)\n',
     '            model_output = _flash_s5_engine.run_graph_replay(\n'
     '                self, scheduler_output, batch_desc,\n'
     '                self.cudagraph_manager.run_fullgraph, batch_desc,\n'
     '            )\n'),
)

_RUNNER_GRAPH_AB = (
    ('            model_output = _flash_e3_graph.replay_fullgraph(\n'
     '                self, scheduler_output, batch_desc,\n'
     '                self.cudagraph_manager.run_fullgraph,\n'
     '            )\n',
     '            model_output = _flash_s5_engine.run_graph_replay(\n'
     '                self, scheduler_output, batch_desc,\n'
     '                lambda: _flash_e3_graph.replay_fullgraph(\n'
     '                    self, scheduler_output, batch_desc,\n'
     '                    self.cudagraph_manager.run_fullgraph,\n'
     '                ),\n'
     '            )\n'),
)

_SAMPLER_TRANSFORMS = (
    ('import vllm.envs as envs\n',
     'import vllm.envs as envs\n'
     'import vllm.v1.flash_e3.engine_offers as _flash_s5_engine\n'),
    # Container around the ordinary Sampler call. It must submit nothing
    # itself; the leaves below own every GPU operation.
    ('    def __call__(\n'
     '        self,\n'
     '        logits: torch.Tensor,\n'
     '        input_batch: InputBatch,\n',
     '    @_flash_s5_engine.offer_method("logits_sampler", "sampler_entry")\n'
     '    def __call__(\n'
     '        self,\n'
     '        logits: torch.Tensor,\n'
     '        input_batch: InputBatch,\n'),
    # The two index gathers are one source-declared leaf. Statements stay in
    # the same order and none of the expressions moves outside the offer.
    ('        pos = input_batch.positions[input_batch.logits_indices]\n'
     '        input_ids = input_batch.input_ids[input_batch.logits_indices]\n',
     '        with _flash_s5_engine.leaf(\n'
     '            self, "sampler", "select_inputs"\n'
     '        ):\n'
     '            pos = input_batch.positions[input_batch.logits_indices]\n'
     '            input_ids = input_batch.input_ids[input_batch.logits_indices]\n'),
    # Conditional NaN check. The disabled branch records a bounded host field
    # and creates no empty offer/CUPTI pair.
    ('        num_nans = get_num_nans(logits) if self.compute_nans else None\n',
     '        if self.compute_nans:\n'
     '            num_nans = _flash_s5_engine.run(\n'
     '                self, "sampler", "nan_check", get_num_nans, logits\n'
     '            )\n'
     '        else:\n'
     '            _flash_s5_engine.mark_branch(\n'
     '                self, "sampler", "nan_check", "disabled"\n'
     '            )\n'
     '            num_nans = None\n'),
    # Container around the public sample body. Its leaves are not re-labelled
    # as the whole container.
    ('    def sample(\n'
     '        self,\n'
     '        logits: torch.Tensor,\n'
     '        expanded_idx_mapping: torch.Tensor,\n',
     '    @_flash_s5_engine.offer_method("logits_sampler", "sampler_sample")\n'
     '    def sample(\n'
     '        self,\n'
     '        logits: torch.Tensor,\n'
     '        expanded_idx_mapping: torch.Tensor,\n'),
    # apply_sampling_params is the actual copy/bias/penalty/masking work; it is
    # NOT compute_logits. The branch is the source's own early-return test.
    ('        processed_logits = self.apply_sampling_params(\n'
     '            logits,\n'
     '            expanded_idx_mapping,\n'
     '            idx_mapping,\n'
     '            idx_mapping_np,\n'
     '            pos,\n'
     '            input_ids,\n'
     '            expanded_local_pos,\n'
     '            skip_top_k_top_p=True,\n'
     '        )\n',
     '        processed_logits = _flash_s5_engine.run(\n'
     '            self, "sampler", "apply_params",\n'
     '            self.apply_sampling_params,\n'
     '            logits,\n'
     '            expanded_idx_mapping,\n'
     '            idx_mapping,\n'
     '            idx_mapping_np,\n'
     '            pos,\n'
     '            input_ids,\n'
     '            expanded_local_pos,\n'
     '            skip_top_k_top_p=True,\n'
     '            branch=_flash_s5_engine.params_branch(self, idx_mapping_np),\n'
     '        )\n'),
    ('        top_k, top_p = self.sampling_states.get_top_k_top_p(\n'
     '            expanded_idx_mapping, idx_mapping_np\n'
     '        )\n',
     '        top_k, top_p = _flash_s5_engine.run(\n'
     '            self, "sampler", "topk_topp_inputs",\n'
     '            self.sampling_states.get_top_k_top_p,\n'
     '            expanded_idx_mapping, idx_mapping_np,\n'
     '        )\n'),
    # One draw leaf covers the selected implementation branch. Do not label
    # FlashInfer/Gumbel work as logits compute.
    ('        return self._sample_random(\n'
     '            processed_logits,\n'
     '            expanded_idx_mapping,\n'
     '            idx_mapping_np,\n'
     '            pos,\n'
     '            top_k,\n'
     '            top_p,\n'
     '            use_flashinfer,\n'
     '        )\n',
     '        return _flash_s5_engine.run(\n'
     '            self, "sampler", "draw", self._sample_random,\n'
     '            processed_logits,\n'
     '            expanded_idx_mapping,\n'
     '            idx_mapping_np,\n'
     '            pos,\n'
     '            top_k,\n'
     '            top_p,\n'
     '            use_flashinfer,\n'
     '            branch=("flashinfer" if use_flashinfer else "gumbel"),\n'
     '        )\n'),
    # Optional trace branch: a real leaf only when installed. The absent
    # branch is a host-only bounded record, never an empty offer.
    ('        if self.trace_replay_state is not None:\n'
     '            # Overwrite sampled tokens with the replay trace up-front so that\n'
     '            # computed logprobs reflect the real distribution of the forced token.\n'
     '            self.trace_replay_state.apply_trace(sampled, idx_mapping)\n',
     '        if self.trace_replay_state is not None:\n'
     '            # Overwrite sampled tokens with the replay trace up-front so that\n'
     '            # computed logprobs reflect the real distribution of the forced token.\n'
     '            _flash_s5_engine.run(\n'
     '                self, "sampler", "trace_apply",\n'
     '                self.trace_replay_state.apply_trace, sampled, idx_mapping,\n'
     '            )\n'
     '        else:\n'
     '            _flash_s5_engine.mark_branch(\n'
     '                self, "sampler", "trace_apply", "absent"\n'
     '            )\n'),
    # Top-k logprobs are conditional on the request. Absent means a bounded
    # branch record and no push/pop pair.
    ('            logprobs_tensors = compute_topk_scores(\n'
     '                logits,\n'
     '                num_logprobs,\n'
     '                sampled,\n'
     '                cu_num_logits,\n'
     '                logprob_token_ids_state=self.logprob_token_ids_state,\n'
     '                expanded_idx_mapping=input_batch.expanded_idx_mapping,\n'
     '                max_per_req_token_ids=max_per_req_token_ids,\n'
     '                logits_mode=self.logprobs_mode in ("raw_logits", "processed_logits"),\n'
     '            )\n'
     '        else:\n'
     '            logprobs_tensors = None\n',
     '            logprobs_tensors = _flash_s5_engine.run(\n'
     '                self, "sampler", "topk_logprobs", compute_topk_scores,\n'
     '                logits,\n'
     '                num_logprobs,\n'
     '                sampled,\n'
     '                cu_num_logits,\n'
     '                logprob_token_ids_state=self.logprob_token_ids_state,\n'
     '                expanded_idx_mapping=input_batch.expanded_idx_mapping,\n'
     '                max_per_req_token_ids=max_per_req_token_ids,\n'
     '                logits_mode=self.logprobs_mode in ("raw_logits", "processed_logits"),\n'
     '            )\n'
     '        else:\n'
     '            _flash_s5_engine.mark_branch(\n'
     '                self, "sampler", "topk_logprobs", "not_requested"\n'
     '            )\n'
     '            logprobs_tensors = None\n'),
    # new_ones is an argument expression that allocates/writes, so it must be
    # evaluated inside the leaf scope, not before entering run().
    ('        num_sampled, num_rejected = get_num_sampled_and_rejected(\n'
     '            input_batch.seq_lens.new_ones(input_batch.num_reqs),\n'
     '            input_batch.seq_lens,\n'
     '            input_batch.cu_num_logits,\n'
     '            input_batch.idx_mapping,\n'
     '            self.req_states.prefill_len.gpu,\n'
     '        )\n',
     '        with _flash_s5_engine.leaf(\n'
     '            self, "sampler", "completion_counts"\n'
     '        ):\n'
     '            num_sampled, num_rejected = get_num_sampled_and_rejected(\n'
     '                input_batch.seq_lens.new_ones(input_batch.num_reqs),\n'
     '                input_batch.seq_lens,\n'
     '                input_batch.cu_num_logits,\n'
     '                input_batch.idx_mapping,\n'
     '                self.req_states.prefill_len.gpu,\n'
     '            )\n'),
    # Sampling masks are conditional; the disabled branch must not create an
    # empty offer. The enabled branch owns the mask construction once.
    ('        sampling_mask_tensors = None\n'
     '        if self.return_sampling_mask:\n'
     '            # Size by the validated top_k batch max; wider supports use the bitmask.\n'
     '            max_num_kept = int(np.max(self.sampling_states.top_k.np[idx_mapping_np]))\n'
     '            sampling_mask_tensors = SamplingMaskTensors.from_logits(\n'
     '                processed_logits, num_sampled, max_num_kept\n'
     '            )\n',
     '        sampling_mask_tensors = None\n'
     '        if self.return_sampling_mask:\n'
     '            # Size by the validated top_k batch max; wider supports use the bitmask.\n'
     '            with _flash_s5_engine.leaf(\n'
     '                self, "sampler", "sampling_mask"\n'
     '            ):\n'
     '                max_num_kept = int(\n'
     '                    np.max(self.sampling_states.top_k.np[idx_mapping_np])\n'
     '                )\n'
     '                sampling_mask_tensors = SamplingMaskTensors.from_logits(\n'
     '                    processed_logits, num_sampled, max_num_kept\n'
     '                )\n'
     '        else:\n'
     '            _flash_s5_engine.mark_branch(\n'
     '                self, "sampler", "sampling_mask", "disabled"\n'
     '            )\n'),
)

_ASYNC_TRANSFORMS = (
    ('import torch\n',
     'import torch\n'
     'import vllm.v1.flash_e3.engine_offers as _flash_s5_engine\n'),
    # The copy stream is the output-copy role. The ordinary branch is the
    # D2H copy scope; the fault query is a separate nested conditional leaf.
    ('        with stream(copy_stream, main_stream):\n'
     '            copy_stream.wait_stream(main_stream)\n'
     '\n'
     '            self.sampled_token_ids = async_copy_to_np(sampler_output.sampled_token_ids)\n',
     '        with stream(copy_stream, main_stream), _flash_s5_engine.leaf(\n'
     '            self, "output_copy", "async_output", branch="ordinary"\n'
     '        ):\n'
     '            copy_stream.wait_stream(main_stream)\n'
     '\n'
     '            self.sampled_token_ids = async_copy_to_np(sampler_output.sampled_token_ids)\n'),
    ('            if check_ep_fault:\n'
     '                has_fault = get_ep_all2all_manager().query_fault()\n'
     '                self._has_fault = has_fault.to("cpu", non_blocking=True)\n',
     '            if check_ep_fault:\n'
     '                with _flash_s5_engine.leaf(\n'
     '                    self, "output_copy", "ep_fault_query",\n'
     '                    branch="fault_check",\n'
     '                ):\n'
     '                    has_fault = get_ep_all2all_manager().query_fault()\n'
     '                    self._has_fault = has_fault.to(\n'
     '                        "cpu", non_blocking=True\n'
     '                    )\n'
     '            else:\n'
     '                _flash_s5_engine.mark_branch(\n'
     '                    self, "output_copy", "ep_fault_query", "disabled"\n'
     '                )\n'),
)

_CUDAGRAPH_TRANSFORMS = (
    ('from vllm.config import VllmConfig, set_current_vllm_config\n',
     'from vllm.config import VllmConfig, set_current_vllm_config\n'
     'import vllm.v1.flash_e3.engine_offers as _flash_s5_engine\n'),
    # Stock run_fullgraph's pre-replay offloader sync. The installed offloader
    # is not in the pinned subset, so the leaf is explicitly unsupported and
    # keeps the window INCONCLUSIVE. The split graph_hooks replay does not call
    # this stock function, so no stock-sync offer is required or added there.
    ('        # while those copies are still in flight.\n'
     '        get_offloader().sync_prev_onload()\n'
     '        self.graphs[desc].replay()\n',
     '        # while those copies are still in flight.\n'
     '        _flash_s5_engine.run(\n'
     '            self, "forward_dispatch", "offloader_sync",\n'
     '            lambda: get_offloader().sync_prev_onload(),\n'
     '            branch=_flash_s5_engine.offloader_branch(),\n'
     '        )\n'
     '        self.graphs[desc].replay()\n'),
)


def _runner_variant(source):
    if _sha(source) == PINS['v1/worker/gpu/model_runner.py']:
        return 'pinned'
    if _sha(source) in (MODEL_RUNNER_AFTER_AB_SHA256, MODEL_RUNNER_PACKAGE_SHA256):
        return 'ab'
    raise ValueError('source is not a reviewed model runner')


def _transforms(path, source):
    if path == 'v1/worker/gpu/model_runner.py':
        variant = _runner_variant(source)
        return (_RUNNER_SHARED +
                (_RUNNER_GRAPH_PINNED if variant == 'pinned'
                 else _RUNNER_GRAPH_AB)), variant
    if path == 'v1/worker/gpu/sample/sampler.py':
        if _sha(source) != PINS[path]:
            raise ValueError('source is not the pinned sampler')
        return _SAMPLER_TRANSFORMS, 'pinned'
    if path == 'v1/worker/gpu/async_utils.py':
        if _sha(source) != PINS[path]:
            raise ValueError('source is not the pinned async utils')
        return _ASYNC_TRANSFORMS, 'pinned'
    if path == 'v1/worker/gpu/cudagraph_utils.py':
        if _sha(source) not in ACCEPTED[path]:
            raise ValueError('source is not the pinned cudagraph utils')
        return _CUDAGRAPH_TRANSFORMS, 'pinned'
    raise ValueError('engine offers do not transform ' + str(path))


def generate_one(path, source):
    """Transform one reviewed source; refuse drift or double application."""
    if path not in PINS:
        raise ValueError('engine offers do not transform ' + path)
    if _MARKER[path] in source:
        raise ValueError('source already carries engine offers: ' + path)
    transforms, _variant = _transforms(path, source)
    text = source
    for old, new, count in _transform_rows(transforms):
        text = _replace(text, old, new, count)
    try:
        ast.parse(text)
    except SyntaxError as exc:
        raise ValueError('generated source syntax invalid: ' + path) from exc
    return text


def revert_one(path, source, expected=None):
    """Exact inverse of :func:`generate_one`."""
    if path not in PINS:
        raise ValueError('engine offers do not transform ' + path)
    if _MARKER[path] not in source:
        raise ValueError('source does not carry engine offers: ' + path)
    if path == 'v1/worker/gpu/model_runner.py':
        variant = ('ab' if '_flash_e3_graph.replay_fullgraph(' in source
                   else 'pinned')
        transforms = (_RUNNER_SHARED +
                      (_RUNNER_GRAPH_PINNED if variant == 'pinned'
                       else _RUNNER_GRAPH_AB))
    elif path == 'v1/worker/gpu/sample/sampler.py':
        transforms = _SAMPLER_TRANSFORMS
    elif path == 'v1/worker/gpu/async_utils.py':
        transforms = _ASYNC_TRANSFORMS
    elif path == 'v1/worker/gpu/cudagraph_utils.py':
        transforms = _CUDAGRAPH_TRANSFORMS
    else:
        raise ValueError('engine offers do not transform ' + str(path))
    text = source
    for old, new, count in reversed(list(_transform_rows(transforms))):
        text = _replace(text, new, old, count)
    if expected is not None and text != expected:
        raise ValueError('revert did not recover the supplied source')
    if _sha(text) not in ACCEPTED[path]:
        raise ValueError('revert did not recover a reviewed source: ' + path)
    return text


def generate(files):
    """``{path: source}`` for reviewed inputs -> transformed outputs."""
    out = {}
    for path in PINS:
        if path not in files:
            raise ValueError('missing reviewed input ' + path)
        out[path] = generate_one(path, files[path])
    return out


__all__ = [
    'ACCEPTED', 'ALL_COPY_KINDS', 'COMPOSED_SCHEMA', 'COPY_KIND', 'ENABLE_ENV',
    'EngineOfferLedger', 'FAMILIES', 'FAMILY_MANIFEST_SHA256',
    'GenerationMismatch', 'KINDS', 'MAX_OFFERS', 'MODEL_RUNNER_AFTER_AB_SHA256',
    'NATIVE_SCHEMA', 'NATIVE_SCHEMA_LEGACY', 'OfferCapacityExceeded',
    'OfferDisabled', 'OfferError', 'OfferPopIdentity', 'PINS', 'PLACEMENT_SCHEMA',
    'SCHEMA', 'SOURCE_PINS', 'STREAM_ROLES', 'THREAD_ID_TYPES',
    'VocabularyError', 'WindowBinding', 'bank_identity', 'bind_window',
    'census_for', 'compose_placement', 'connector_branch', 'current_ledger',
    'descriptor_key', 'dp_branch', 'family_manifest_identity', 'gather_branch',
    'generate', 'generate_one', 'grade_engine', 'graph_descriptor',
    'install_census', 'leaf', 'mark_branch', 'mark_zero_work',
    'offer_method', 'offloader_branch', 'open_ledger', 'params_branch',
    'pp_branch', 'revert_one', 'run', 'run_graph_replay', 'scoped_call',
    'source_identities_sha256', 'step_method', 'validate_receipt',
]
