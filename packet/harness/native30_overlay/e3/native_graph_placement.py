"""Raw graph proof composed with both native accounting partitions.

``native_placement.compose_overlap`` proves that a retained
``s5ro-correlation/2`` report is covered by the decode and source-declared
prefill partitions and regrades each partition.  It never looks at the raw
graph-only receipt (``s5ro-replay-observer/1``).  ``engine_offers.compose_placement``
proves the raw receipt against a *single* engine receipt (the decode
partition).  This module closes the remaining gap for the paired case:

* the unchanged ``compose_overlap`` is called first and its verdict is never
  improved on;
* the unchanged raw receipt is bound to the same worker rank, native window,
  engine epoch, route generation, storage epoch, observer/helper identities,
  retained bank and router census as the decode window binding;
* every original raw graph condition (static bank, observed/target coverage,
  per-node multiplicity against the router replays, context/SM, mapping
  ambiguity, ring overflows, sticky flags, whole-window counters) is proved by
  the unchanged ``engine_offers._raw_graph_proof``;
* the raw graph records are then reconciled against the actual joined graph
  activity rows and declared graph offers of the whole report, and every
  graph-id-zero raw record (``unattributed_expected_context`` /
  ``unattributed_other_context``) against exactly one actual joined non-graph
  activity row of a source-declared decode or prefill offer.

The only raw ``INCONCLUSIVE`` this module may discharge is the documented
VM13c shape: the native reason set is exactly
``['unattributed_expected_context']`` and the derived reasons are a subset of
``{'unattributed_expected_context', 'replay_incomplete'}``.  The discharge is
recorded beside the untouched raw digest, and it is only issued when both
partitions pass independently and both ledgers are complete and positive.

Reused source helpers (read-only, unchanged)
--------------------------------------------
``engine_offers._raw_receipt_consistency``, ``_raw_fail_signals``,
``_raw_identity_binding``, ``_raw_graph_proof``, ``_coerce_binding``,
``descriptor_key``, ``_kind_int_map``, ``_api_kind``, ``_NON_GREEN_LEAVES``;
``native_placement.compose_overlap`` / ``validate_overlap`` and its declared-map
parser.

Named seam (not reused, smallest separate proof implemented here)
-----------------------------------------------------------------
``engine_offers._engine_graph_reconciliation`` and
``_engine_multiset_reconciliation`` read fields that only a single-partition
engine receipt retains (``engine['graph']['offers'][...]['launch_count']`` /
``['census']`` / ``['node_counts']``, ``engine['engine_by_kind']``,
``engine['native_by_kind']``, ``engine['eager_green_records']``,
``engine['eager_other_records']``).  ``compose_overlap`` deliberately retains
only per-partition activity totals, so those predicates cannot be called
unchanged.  ``_graph_offer_proof`` and ``_native_ledger`` below re-prove the
same predicates directly over the actual declared offers, the actual native
activity rows and the unchanged raw proof.  They never rewrite a raw flag, a
raw count or a raw verdict.

Nothing here mutates its inputs, resets a native loss flag, or relaxes a
readiness/closure/guard predicate.
"""
import hashlib
import json
from collections import Counter

try:
    from activity_accounting import KINDS
    from engine_offers import (_BadEvidence, _NON_GREEN_LEAVES, _api_kind,
                               _coerce_binding, _kind_int_map, _raw_fail_signals,
                               _raw_graph_proof, _raw_identity_binding,
                               _raw_receipt_consistency, _wire,
                               descriptor_key)
    from native_placement import (OverlapError, compose_overlap,
                                  _declared_ids)
except ImportError:  # pragma: no cover - package-relative import
    from .activity_accounting import KINDS
    from .engine_offers import (_BadEvidence, _NON_GREEN_LEAVES, _api_kind,
                                _coerce_binding, _kind_int_map,
                                _raw_fail_signals, _raw_graph_proof,
                                _raw_identity_binding, _raw_receipt_consistency,
                                _wire, descriptor_key)
    from .native_placement import (OverlapError, compose_overlap,
                                   _declared_ids)

SCHEMA = 's5-native-graph-placement/1'
RAW_SCHEMA = 's5ro-replay-observer/1'
BASELINE_SCHEMA = 's5-native-overlap-accounting/1'
WORK_KINDS = tuple(KINDS)
# The only raw native reason this composer may explain away, and the exact
# derived consequences it may leave to the two accounting ledgers.
DISCHARGE_NATIVE = 'unattributed_expected_context'
DISCHARGE_DERIVED = frozenset({'unattributed_expected_context',
                               'replay_incomplete'})
EXERCISED_ROLES = ('decode', 'prefill')


class NativeGraphError(ValueError):
    pass


class _LedgerViolation(Exception):
    """Actual evidence contradicts another retained receipt."""


class _LedgerMissing(Exception):
    """The evidence needed for a positive ledger is absent."""


def _uint(value, reason, minimum=0):
    if type(value) is not int or value < minimum:
        raise _LedgerMissing(reason)
    return value


def _sha(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _reason_list(value):
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if isinstance(item, str)]


def _proof_disposition(code):
    """Classify a proof refusal: contradiction (FAIL) or missing (INC).

    Only refusals that are unambiguously *absent or malformed* evidence stay
    INCONCLUSIVE.  A wrong identity, bank, window, context, SM, node kind,
    multiplicity, census, sticky flag or loss counter is a contradiction.
    Unrecognised refusals default to INCONCLUSIVE: they can never PASS and are
    never discharged.
    """
    code = str(code)
    missing = (
        'shape', '_keys', 'counts_', 'flags_keys', 'ring_keys',
        'graph_count_zero', 'static_not_complete', 'static_reasons',
        'slot_qualified', 'stale_replay_qualified', 'context_records',
        'context_bound', 'mapping_graph_ids', 'inventory_shape',
    )
    if any(token in code for token in missing):
        return 'INCONCLUSIVE'
    return 'FAIL'


def _declared(declared, label):
    """Reuse the unchanged declared-map parser; report its refusal as INC."""
    try:
        return {oid: row for oid, row in
                _declared_ids(declared, label).items()}
    except OverlapError as exc:
        raise _LedgerMissing(label + '_declared:' + str(exc))


def _binding_record(window):
    if not isinstance(window, dict):
        raise _LedgerMissing('window_shape')
    record = window.get('binding')
    if not isinstance(record, dict):
        raise _LedgerMissing('binding_missing')
    return record


def _binding_summary(binding):
    return {
        'run_id': binding.run_id,
        'window_id': binding.window_id,
        'worker': binding.worker,
        'rank': binding.rank,
        'worker_incarnation': binding.worker_incarnation,
        'engine_incarnation': binding.engine_incarnation,
        'engine_epoch': binding.engine_epoch,
        'route_generation': binding.route_generation,
        'storage_epoch': binding.storage_epoch,
        'green_context_id': binding.green_context_id,
        'native_start_ns': binding.native_start_ns,
        'native_end_ns': binding.native_end_ns,
        'observer_identity_sha256': _sha(binding.observer_identity),
        'helper_identity_sha256': _sha(binding.helper_identity),
        'declared_target': binding.declared_target,
        'descriptor_labels': dict(binding.descriptor_labels),
        'router_census': (None if binding.router_census is None else {
            'total': binding.router_census['total'],
            'graphs': dict(binding.router_census['graphs']),
        }),
        'bank_identity_sha256': binding.bank_identity,
    }


def _raw_summary(raw):
    return {
        'schema': raw.get('schema'),
        'verdict': raw.get('verdict'),
        'reasons': _reason_list(raw.get('reasons')),
        'complete': raw.get('complete'),
        'sha256': _sha(raw),
        'unmodified': True,
        'scope': 'raw-graph-only-placement',
    }


def _baseline_summary(baseline):
    baseline = baseline if isinstance(baseline, dict) else {}
    parts = baseline.get('partitions')
    parts = parts if isinstance(parts, dict) else {}
    out = {
        'schema': baseline.get('schema'),
        'verdict': baseline.get('verdict'),
        'failures': list(baseline.get('failures', [])),
        'missing': list(baseline.get('missing', [])),
        'sha256': _sha(baseline),
        'conservation_proved': bool(
            (baseline.get('conservation') or {}).get('proved')),
        'partitions': {},
    }
    for name in EXERCISED_ROLES:
        row = parts.get(name)
        if not isinstance(row, dict):
            row = {}
        out['partitions'][name] = {
            'verdict': row.get('verdict'),
            'offers': row.get('offers'),
            'activity_records': row.get('activity_records'),
            'by_kind': dict(row.get('by_kind') or {}),
            'offers_partitioned': list(row.get('offers_partitioned') or []),
            'receipt_sha256': row.get('receipt_sha256'),
        }
    return out


def _external_map(native):
    rows = native.get('external')
    if not isinstance(rows, list):
        raise _LedgerMissing('external_shape')
    out = {}
    for row in rows:
        if not isinstance(row, dict):
            raise _LedgerMissing('external_row')
        cid = row.get('correlation_id')
        if type(cid) is not int or cid < 1:
            raise _LedgerMissing('external_identity')
        if cid in out:
            raise _LedgerViolation('duplicate_external_join')
        if row.get('kind') != 'custom0':
            raise _LedgerViolation('foreign_external_kind')
        oid = row.get('external_id')
        if type(oid) is not int or oid < 1:
            raise _LedgerMissing('external_offer')
        out[cid] = oid
    return out


def _graph_api_launches(native, external, graph_offers):
    """Successful graph-launch APIs joined to each declared graph offer."""
    rows = native.get('apis')
    if not isinstance(rows, list):
        raise _LedgerMissing('apis_shape')
    out = Counter()
    for row in rows:
        if not isinstance(row, dict):
            raise _LedgerMissing('api_row')
        cid = row.get('correlation_id')
        if _api_kind(row.get('name')) != 'graph':
            continue
        oid = external.get(cid)
        if oid is None:
            raise _LedgerViolation('unoffered_graph_submission')
        if oid not in graph_offers:
            raise _LedgerViolation('graph_api_without_graph_offer')
        if row.get('return_value') != 0:
            raise _LedgerViolation('failed_graph_submission')
        out[oid] += 1
    return out


def _graph_offer_proof(offers, decode_declared, prefill_declared, binding,
                       proof, graph_by_offer, graph_apis):
    """Re-prove raw graph/replay/census predicates over the paired offers.

    Mirrors ``engine_offers._engine_graph_reconciliation`` over the fields the
    paired accounting really retains.  Every graph offer resolves to exactly
    one retained bank slot, its declared census equals the static node types,
    the number of graph offers per slot equals the router replay count, and the
    actual joined graph activity of each offer equals its census (one launch
    per offer, ``replays`` launches per observed slot).
    """
    offer_index = {}
    for row in offers:
        if isinstance(row, dict) and type(row.get('id')) is int:
            offer_index[row['id']] = row
    graph_offers = {}
    for oid in sorted(set(decode_declared) | set(prefill_declared)):
        meta = decode_declared.get(oid)
        if meta is None:
            meta = prefill_declared.get(oid)
        if isinstance(meta, dict) and meta.get('graph') is True:
            graph_offers[oid] = meta
    if not graph_offers:
        raise _LedgerMissing('no_graph_offer')
    reverse = {}
    for label, key in binding.descriptor_labels.items():
        if key in reverse:
            raise _LedgerViolation('descriptor_ambiguous')
        reverse[key] = label
    offers_by_label = Counter()
    offer_proofs = {}
    for oid, meta in graph_offers.items():
        descriptor = meta.get('descriptor')
        if isinstance(descriptor, dict):
            descriptor = descriptor_key(descriptor)
        if not isinstance(descriptor, str) or descriptor not in reverse:
            raise _LedgerViolation('graph_offer_descriptor:' + str(oid))
        label = reverse[descriptor]
        index = int(label.split(':', 1)[1])
        slot = proof['slots'].get(index)
        if slot is None or label not in proof['observed']:
            raise _LedgerViolation('graph_offer_unobserved:' + str(oid))
        census = _kind_int_map(meta.get('census'), 'graph_offer_census')
        if not any(census.values()):
            raise _LedgerViolation('graph_offer_census_empty:' + str(oid))
        if census != slot['node_types']:
            raise _LedgerViolation('graph_offer_census:' + str(oid))
        offer = offer_index.get(oid)
        if not isinstance(offer, dict):
            raise _LedgerMissing('graph_offer_absent:' + str(oid))
        if offer.get('expected') != census:
            raise _LedgerViolation('graph_offer_expected:' + str(oid))
        if offer.get('closed') is not True:
            raise _LedgerMissing('graph_offer_open:' + str(oid))
        actual = graph_by_offer.get(oid, Counter())
        if actual != Counter(census):
            raise _LedgerViolation('graph_activity_mismatch:' + str(oid))
        if not graph_apis.get(oid):
            raise _LedgerMissing('graph_launch_missing:' + str(oid))
        offers_by_label[label] += 1
        offer_proofs[str(oid)] = {
            'partition': ('decode' if oid in decode_declared else 'prefill'),
            'descriptor': descriptor,
            'label': label,
            'census': census,
            'joined_records': sum(actual.values()),
            'graph_apis': graph_apis[oid],
        }
    for index in range(proof['graph_count']):
        label = 'graph:' + str(index)
        expected = proof['router'].get(str(index), 0)
        if offers_by_label.get(label, 0) != expected:
            raise _LedgerViolation('census_replay_multiplicity:' + label)
    total = Counter()
    for row in offer_proofs.values():
        for kind in WORK_KINDS:
            total[kind] += row['census'][kind]
    if any(total[kind] != proof['graph_by_kind'].get(kind, 0)
           for kind in WORK_KINDS):
        raise _LedgerViolation('graph_node_multiset')
    return offer_proofs, dict(offers_by_label)


def _role_identity(meta, known, key, label):
    role = meta.get(key)
    row = known.get(role) if isinstance(known, dict) else None
    if not isinstance(row, dict):
        raise _LedgerMissing(label + '_role_missing')
    return role, row


def _native_ledger(native, decode_declared, prefill_declared, binding, proof,
                   prefill_roles, baseline):
    """Join every raw record to exactly one actual joined activity row.

    Graph rows (``graph_id != 0``) must belong to one declared graph offer of
    one partition.  Graph-id-zero rows in the bound green context are the raw
    ``unattributed_expected_context`` records and must be decode-partition
    offers on their declared stream role.  Rows in any other context are the
    raw ``unattributed_other_context`` records and must be prefill-partition
    offers whose *verified* role identity names exactly that context/stream;
    the only other allowance is the reviewed non-green control leaf on a
    decode stream role the binding marks ``non_green_control``.
    """
    rows = native.get('activities')
    if not isinstance(rows, list):
        raise _LedgerMissing('activities_shape')
    external = _external_map(native)
    green = binding.green_context_id
    graph_by_kind = Counter()
    expected_by_kind = Counter()
    other_by_kind = Counter()
    graph_by_offer = {}
    decode_by_kind = Counter()
    prefill_by_kind = Counter()
    by_partition = {'decode': Counter(), 'prefill': Counter()}
    ledger_rows = []
    for row in rows:
        if not isinstance(row, dict):
            raise _LedgerMissing('activity_row')
        kind = row.get('kind')
        if kind not in WORK_KINDS:
            raise _LedgerMissing('unknown_activity_kind')
        count = _uint(row.get('count'), 'activity_count', 1)
        cid = _uint(row.get('correlation_id'), 'activity_identity', 1)
        graph_id = _uint(row.get('graph_id', 0), 'activity_graph')
        context_id = _uint(row.get('context_id'), 'activity_context')
        stream_id = _uint(row.get('stream_id'), 'activity_stream')
        oid = external.get(cid)
        if oid is None:
            raise _LedgerMissing('activity_not_joined')
        if oid in decode_declared:
            partition = 'decode'
            meta = decode_declared[oid]
        elif oid in prefill_declared:
            partition = 'prefill'
            meta = prefill_declared[oid]
        else:
            raise _LedgerMissing('activity_undeclared_offer')
        if not isinstance(meta, dict):
            raise _LedgerMissing('activity_offer_meta')
        declared_context = meta.get('context_id')
        declared_stream = meta.get('stream_id')
        if declared_context is None or declared_stream is None:
            raise _LedgerMissing('activity_declared_identity')
        if context_id != declared_context or stream_id != declared_stream:
            raise _LedgerViolation('declared_context_stream:' + str(oid))
        if graph_id:
            if meta.get('graph') is not True:
                raise _LedgerViolation('graph_row_owned_by_eager_offer')
            allowed = meta.get('allowed_kinds')
            if isinstance(allowed, (list, tuple)) and kind not in allowed:
                raise _LedgerViolation('graph_row_kind:' + str(oid))
            graph_by_kind[kind] += count
            graph_by_offer.setdefault(oid, Counter())[kind] += count
            role = meta.get('stream_role')
        else:
            if meta.get('graph') is True:
                raise _LedgerViolation('eager_row_owned_by_graph_offer')
            if context_id == green:
                # The raw expected context is the bound green context.  A
                # graph-id-zero record there is owned only by a decode offer on
                # a decode stream role, never by a prefill or generic role.
                if partition != 'decode':
                    raise _LedgerViolation('expected_context_not_decode')
                role, role_row = _role_identity(
                    meta, binding.stream_roles, 'stream_role', 'decode')
                if (role_row.get('context_id') != context_id
                        or role_row.get('stream_id') != stream_id):
                    raise _LedgerViolation('decode_role_identity:' + str(oid))
                expected_by_kind[kind] += count
            else:
                if partition == 'prefill':
                    # A second green context exists only because the verified
                    # prefill partition declared this role/context/stream.
                    role, role_row = _role_identity(
                        meta, prefill_roles, 'role', 'prefill')
                    if (role_row.get('context_id') != context_id
                            or role_row.get('stream_id') != stream_id):
                        raise _LedgerViolation(
                            'prefill_role_identity:' + str(oid))
                else:
                    # Decode-owned other-context work is allowed only for the
                    # reviewed non-green control leaves on a binding role that
                    # declares itself non-green.  No generic allowance.
                    wire = _wire(meta.get('family'), meta.get('suboperation'))
                    role = _NON_GREEN_LEAVES.get(wire)
                    role_row = (binding.stream_roles.get(role)
                                if role is not None else None)
                    if not isinstance(role_row, dict) \
                            or role_row.get('non_green_control') is not True:
                        raise _LedgerViolation(
                            'other_context_not_owned:' + str(oid))
                    if (role_row.get('context_id') != context_id
                            or role_row.get('stream_id') != stream_id):
                        raise _LedgerViolation(
                            'other_context_identity:' + str(oid))
                    role = meta.get('stream_role')
                other_by_kind[kind] += count
        by_partition[partition][kind] += count
        (decode_by_kind if partition == 'decode'
         else prefill_by_kind)[kind] += count
        ledger_rows.append({
            'offer': oid,
            'partition': partition,
            'correlation_id': cid,
            'kind': kind,
            'count': count,
            'graph_id': graph_id,
            'context_id': context_id,
            'stream_id': stream_id,
            'role': role,
        })
    records = sum(row['count'] for row in ledger_rows)
    activity_records = native.get('activity_records')
    if type(activity_records) is not int:
        raise _LedgerMissing('activity_records')
    if records != activity_records:
        raise _LedgerMissing('activity_census')
    counts = proof['raw_counts']
    if sum(expected_by_kind.values()) != counts['unattributed_expected_context']:
        raise _LedgerViolation('raw_expected_mismatch')
    if sum(other_by_kind.values()) != counts['unattributed_other_context']:
        raise _LedgerViolation('raw_other_mismatch')
    if any(graph_by_kind[kind] != proof['graph_by_kind'].get(kind, 0)
           for kind in WORK_KINDS):
        raise _LedgerViolation('raw_graph_kind_mismatch')
    if records != counts['records_in_window']:
        raise _LedgerViolation('raw_records_mismatch')
    parts = baseline.get('partitions') or {}
    for name, mine in (('decode', decode_by_kind), ('prefill', prefill_by_kind)):
        row = parts.get(name) or {}
        theirs = row.get('by_kind') or {}
        for kind in WORK_KINDS:
            if mine[kind] != theirs.get(kind):
                raise _LedgerViolation('partition_totals:' + name + ':' + kind)
        if sum(mine.values()) != row.get('activity_records'):
            raise _LedgerViolation('partition_records:' + name)
    return {
        'proved': True,
        'records': records,
        'graph_records': sum(graph_by_kind.values()),
        'expected_context_records': sum(expected_by_kind.values()),
        'other_context_records': sum(other_by_kind.values()),
        'graph_by_kind': {kind: graph_by_kind[kind] for kind in WORK_KINDS},
        'expected_by_kind': {kind: expected_by_kind[kind]
                             for kind in WORK_KINDS},
        'other_by_kind': {kind: other_by_kind[kind] for kind in WORK_KINDS},
        'partitions': {
            name: {
                'records': sum(by_partition[name].values()),
                'by_kind': {kind: by_partition[name][kind]
                            for kind in WORK_KINDS},
            } for name in EXERCISED_ROLES
        },
        'graph_by_offer': {
            str(oid): {kind: row[kind] for kind in WORK_KINDS}
            for oid, row in sorted(graph_by_offer.items())
        },
        'rows': ledger_rows,
    }


def _result(window, failures, missing, *, raw=None, baseline=None, binding=None,
            graph=None, reconciliation=None, partitions=None, discharge=None,
            ledger_rows=None, notes=None):
    verdict = 'FAIL' if failures else (
        'INCONCLUSIVE' if missing else 'PASS')
    return {
        'schema': SCHEMA,
        'verdict': verdict,
        'row4': verdict,
        'window': window,
        'raw': raw,
        'baseline': baseline,
        'binding': binding,
        'graph': graph,
        'reconciliation': reconciliation,
        'partitions': partitions or {},
        'discharge': discharge,
        'ledger': {'proved': False, 'rows': list(ledger_rows or [])},
        'failures': sorted(set(failures)),
        'missing': sorted(set(missing)),
        'raw_unchanged': True,
        'notes': list(notes or []),
    }


def compose_native_placement(raw, native, offers, *, decode_declared,
                             decode_window, prefill_declared, prefill_window,
                             decode_receipt=None, prefill_receipt=None,
                             historical_placement=None):
    """Compose the unchanged raw graph receipt with both native partitions.

    The function is pure: no input is mutated and no raw flag, count or verdict
    is rewritten.  ``compose_overlap`` is called unchanged and first; this
    composer can only keep or worsen its verdict.
    """
    failures = []
    missing = []
    notes = [
        'compose_overlap is called unchanged; its verdict is never improved on',
        'the raw graph receipt is proved by the unchanged _raw_graph_proof '
        'against the decode window binding',
        'graph-id-zero records are joined by exact offer/context/stream ids; '
        'no symbol, time or current-scope heuristic is used',
        'the only dischargeable raw INCONCLUSIVE is exactly '
        'unattributed_expected_context + replay_incomplete, and only when both '
        'partition ledgers are complete and positive',
    ]
    raw_summary = None
    if isinstance(raw, dict):
        try:
            raw_summary = _raw_summary(raw)
        except (TypeError, ValueError):
            return _result(None, [], ['raw_malformed:digest'],
                           notes=notes)
    window = None

    try:
        baseline = compose_overlap(
            native, offers, decode_declared=decode_declared,
            decode_window=decode_window, prefill_declared=prefill_declared,
            prefill_window=prefill_window, decode_receipt=decode_receipt,
            prefill_receipt=prefill_receipt,
            historical_placement=historical_placement)
    except (OverlapError, TypeError, ValueError, KeyError) as exc:
        return _result(None, [], ['composition:compose_overlap:' + type(exc).__name__],
                       raw=raw_summary,
                       notes=notes + ['compose_overlap refused to run'])
    baseline_summary = _baseline_summary(baseline)
    window = {k: (decode_window or {}).get(k) for k in (
        'worker', 'rank', 'epoch', 'generation', 'route_generation',
        'storage_epoch', 'run_id', 'window_id')
    } if isinstance(decode_window, dict) else None

    def fail(reason):
        if reason not in failures:
            failures.append(reason)

    def miss(reason):
        if reason not in missing:
            missing.append(reason)

    # Independent known raw contradictions retain FAIL precedence even when
    # another partition is missing. An incomplete ledger cannot hide a raw
    # graph/context violation as a merely incomplete composed result.
    if isinstance(raw, dict) and raw.get('schema') == 's5ro-replay-observer/1':
        try:
            signals = _raw_fail_signals(raw)
        except (TypeError, KeyError, ValueError):
            signals = []
        if raw.get('verdict') == 'FAIL' or signals:
            return _result(window, ['raw_fail:' + ','.join(raw.get('reasons') or ['raw_verdict'])] +
                ['raw_known_violation:' + value for value in signals], [],
                raw=raw_summary, baseline=baseline_summary, notes=notes)

    if baseline.get('verdict') == 'FAIL':
        return _result(window, ['baseline:' + reason
                                for reason in baseline.get('failures', [])],
                       ['baseline:' + reason
                        for reason in baseline.get('missing', [])],
                       raw=raw_summary, baseline=baseline_summary, notes=notes)
    if baseline.get('verdict') != 'PASS':
        return _result(window, [],
                       ['baseline:' + reason
                        for reason in (baseline.get('missing')
                                       or ['baseline_inconclusive'])],
                       raw=raw_summary, baseline=baseline_summary, notes=notes)

    # 1. Unchanged raw receipt shape/consistency.
    try:
        raw_verdict, raw_reasons, native_reasons = _raw_receipt_consistency(raw)
    except _BadEvidence as exc:
        return _result(window, [], ['raw_malformed:' + str(exc)],
                       raw=raw_summary, baseline=baseline_summary, notes=notes)
    except (TypeError, KeyError, ValueError) as exc:
        return _result(window, [], ['raw_malformed:' + type(exc).__name__],
                       raw=raw_summary, baseline=baseline_summary, notes=notes)
    raw_summary = _raw_summary(raw)
    raw_summary['native_reasons'] = list(native_reasons)
    fail_signals = _raw_fail_signals(raw)
    if raw_verdict == 'FAIL' or fail_signals:
        return _result(window,
                       ['raw_fail:' + ','.join(raw_reasons or ['raw_verdict'])] +
                       ['raw_known_violation:' + signal
                        for signal in fail_signals],
                       [], raw=raw_summary, baseline=baseline_summary,
                       notes=notes)
    if raw_verdict == 'INCONCLUSIVE':
        if list(native_reasons) != [DISCHARGE_NATIVE]:
            return _result(
                window, [],
                ['raw_not_dischargeable:' + ','.join(native_reasons or
                                                     ['no_native_reason'])],
                raw=raw_summary, baseline=baseline_summary, notes=notes)
        if not set(raw_reasons).issubset(DISCHARGE_DERIVED):
            return _result(
                window, [],
                ['raw_not_dischargeable:' + ','.join(raw_reasons)],
                raw=raw_summary, baseline=baseline_summary, notes=notes)

    # 2. The same binding the decode ledger used, coerced by the source helper.
    try:
        record = _binding_record(decode_window)
        binding = _coerce_binding(record)
    except _LedgerMissing as exc:
        return _result(window, [], ['binding:' + str(exc)],
                       raw=raw_summary, baseline=baseline_summary, notes=notes)
    except Exception as exc:  # OfferError and malformed binding records
        return _result(window, [], ['binding_invalid:' + type(exc).__name__],
                       raw=raw_summary, baseline=baseline_summary, notes=notes)
    binding_summary = _binding_summary(binding)

    # 3. Raw identity/bank/window must be the decode binding's.
    try:
        _raw_identity_binding(raw, binding)
    except _BadEvidence as exc:
        code = str(exc)
        disposition = _proof_disposition(code)
        target = fail if disposition == 'FAIL' else miss
        target('raw_identity_' + ('violation' if disposition == 'FAIL'
                                  else 'missing') + ':' + code)
        return _result(window, failures, missing, raw=raw_summary,
                       baseline=baseline_summary, binding=binding_summary,
                       notes=notes)

    # 4. Every original raw graph condition, independently of the partitions.
    try:
        proof = _raw_graph_proof(raw, binding, None, raw_verdict)
    except _BadEvidence as exc:
        code = str(exc)
        disposition = _proof_disposition(code)
        target = fail if disposition == 'FAIL' else miss
        target('raw_graph_' + ('violation' if disposition == 'FAIL'
                              else 'missing') + ':' + code)
        return _result(window, failures, missing, raw=raw_summary,
                       baseline=baseline_summary, binding=binding_summary,
                       notes=notes)
    graph_summary = {
        'count': proof['graph_count'],
        'target': proof['target_label'],
        'observed': list(proof['observed']),
        'replay_qualified': list(proof['qualified']),
        'router': dict(proof['router']),
        'by_kind': dict(proof['graph_by_kind']),
        'raw_counts': dict(proof['raw_counts']),
        'slots': {str(index): dict(slot)
                  for index, slot in proof['slots'].items()},
        'static_verdict': proof['static_verdict'],
        'context': dict(proof['context']),
    }

    # 5. Both partitions must pass on their own views (baseline PASS already
    #    implies this, but the composed receipt states it explicitly).
    baseline_parts = baseline.get('partitions') or {}
    for name in EXERCISED_ROLES:
        row = baseline_parts.get(name)
        if not isinstance(row, dict) or row.get('verdict') != 'PASS':
            miss('partition_not_pass:' + name)

    # 6. Native ledgers.
    try:
        d_decl = _declared(decode_declared, 'decode')
        p_decl = _declared(prefill_declared, 'prefill')
        prefill_roles = (baseline_parts.get('prefill') or {}).get('roles')
        if not isinstance(prefill_roles, dict) or not prefill_roles:
            raise _LedgerMissing('prefill_roles_missing')
        ledger = _native_ledger(native, d_decl, p_decl, binding, proof,
                                prefill_roles, baseline)
        external = _external_map(native)
        graph_apis = _graph_api_launches(native, external,
                                         {oid for oid, meta in
                                          list(d_decl.items())
                                          + list(p_decl.items())
                                          if isinstance(meta, dict)
                                          and meta.get('graph') is True})
        graph_offers, offers_by_label = _graph_offer_proof(
            offers, d_decl, p_decl, binding, proof,
            {int(oid): Counter(row) for oid, row
             in ledger['graph_by_offer'].items()}, graph_apis)
    except _LedgerViolation as exc:
        fail('ledger_violation:' + str(exc))
        return _result(window, failures, missing, raw=raw_summary,
                       baseline=baseline_summary, binding=binding_summary,
                       graph=graph_summary, notes=notes)
    except _LedgerMissing as exc:
        miss('ledger_missing:' + str(exc))
        return _result(window, failures, missing, raw=raw_summary,
                       baseline=baseline_summary, binding=binding_summary,
                       graph=graph_summary, notes=notes)
    except (TypeError, KeyError, ValueError) as exc:
        miss('ledger_malformed:' + type(exc).__name__)
        return _result(window, failures, missing, raw=raw_summary,
                       baseline=baseline_summary, binding=binding_summary,
                       graph=graph_summary, notes=notes)

    # 7. Discharge only the documented derived INCONCLUSIVE.
    discharge = None
    if raw_verdict == 'INCONCLUSIVE':
        expected = ledger['expected_context_records']
        if expected <= 0:
            return _result(window, [],
                           ['raw_expected_count_not_positive'],
                           raw=raw_summary, baseline=baseline_summary,
                           binding=binding_summary, graph=graph_summary,
                           notes=notes)
        discharge = {
            'origin': 'discharged_eager_expected_context',
            'native_reasons': list(native_reasons),
            'raw_reasons': list(raw_reasons),
            'derived_reasons': sorted(set(raw_reasons) - {DISCHARGE_NATIVE}),
            'expected_context_records': expected,
            'other_context_records': ledger['other_context_records'],
            'graph_records': ledger['graph_records'],
        }
    else:
        if ledger['expected_context_records'] != 0:
            fail('raw_pass_with_expected_context')
        discharge = {
            'origin': 'raw_pass',
            'native_reasons': [],
            'raw_reasons': [],
            'derived_reasons': [],
            'expected_context_records': 0,
            'other_context_records': ledger['other_context_records'],
            'graph_records': ledger['graph_records'],
        }

    reconciliation = {
        'proved': ledger['proved'] and not failures,
        'records': ledger['records'],
        'graph_records': ledger['graph_records'],
        'expected_context_records': ledger['expected_context_records'],
        'other_context_records': ledger['other_context_records'],
        'graph_by_kind': ledger['graph_by_kind'],
        'expected_by_kind': ledger['expected_by_kind'],
        'other_by_kind': ledger['other_by_kind'],
        'native_by_kind': {
            kind: ledger['graph_by_kind'][kind]
            + ledger['expected_by_kind'][kind]
            + ledger['other_by_kind'][kind] for kind in WORK_KINDS},
        'graph_by_offer': ledger['graph_by_offer'],
        'graph_offers': graph_offers,
        'offers_by_label': offers_by_label,
        'raw_records_in_window': proof['raw_counts']['records_in_window'],
    }
    partitions = {
        'decode': {
            **baseline_summary['partitions'].get('decode', {}),
            'records': ledger['partitions']['decode']['records'],
            'by_kind': ledger['partitions']['decode']['by_kind'],
            'roles': {role: dict(row) for role, row in
                      binding.stream_roles.items()},
            'binding_sha256': _sha(binding.as_record()),
        },
        'prefill': {
            **baseline_summary['partitions'].get('prefill', {}),
            'records': ledger['partitions']['prefill']['records'],
            'by_kind': ledger['partitions']['prefill']['by_kind'],
            'roles': {role: dict(row) for role, row in
                      prefill_roles.items()},
        },
    }
    all_failures = failures
    verdict = 'FAIL' if all_failures else (
        'INCONCLUSIVE' if missing else 'PASS')
    return {
        'schema': SCHEMA,
        'verdict': verdict,
        'row4': verdict,
        'window': window,
        'raw': raw_summary,
        'baseline': baseline_summary,
        'binding': binding_summary,
        'graph': graph_summary,
        'reconciliation': reconciliation,
        'partitions': partitions,
        'discharge': discharge,
        'ledger': {'proved': ledger['proved'], 'rows': ledger['rows']},
        'failures': sorted(set(all_failures)),
        'missing': sorted(set(missing)),
        'raw_unchanged': True,
        'notes': notes,
    }


def validate_native_placement(receipt):
    """Reject a mutated composed receipt; PASS needs both complete ledgers."""
    if not isinstance(receipt, dict) or receipt.get('schema') != SCHEMA:
        raise NativeGraphError('receipt_schema')
    verdict = receipt.get('verdict')
    if verdict not in ('PASS', 'INCONCLUSIVE', 'FAIL'):
        raise NativeGraphError('receipt_verdict')
    if receipt.get('row4') not in ('PASS', 'INCONCLUSIVE', 'FAIL'):
        raise NativeGraphError('receipt_row4')
    if receipt.get('raw_unchanged') is not True:
        raise NativeGraphError('receipt_raw_unchanged')
    raw = receipt.get('raw')
    if not isinstance(raw, dict) or raw.get('schema') != RAW_SCHEMA:
        raise NativeGraphError('receipt_raw')
    if raw.get('unmodified') is not True:
        raise NativeGraphError('receipt_raw_unmodified')
    if not isinstance(raw.get('sha256'), str) or len(raw['sha256']) != 64:
        raise NativeGraphError('receipt_raw_digest')
    if raw.get('verdict') not in ('PASS', 'INCONCLUSIVE', 'FAIL'):
        raise NativeGraphError('receipt_raw_verdict')
    if raw.get('verdict') == 'FAIL':
        raise NativeGraphError('receipt_raw_fail')
    if verdict == 'PASS':
        if receipt.get('failures') or receipt.get('missing'):
            raise NativeGraphError('receipt_pass_reasons')
        baseline = receipt.get('baseline')
        if not isinstance(baseline, dict) or baseline.get('verdict') != 'PASS' \
                or baseline.get('conservation_proved') is not True:
            raise NativeGraphError('receipt_baseline')
        parts = receipt.get('partitions')
        if not isinstance(parts, dict):
            raise NativeGraphError('receipt_partitions')
        for name in EXERCISED_ROLES:
            row = parts.get(name)
            if not isinstance(row, dict) or row.get('verdict') != 'PASS':
                raise NativeGraphError('receipt_partition:' + name)
        graph = receipt.get('graph')
        if not isinstance(graph, dict) or graph.get('count', 0) < 1:
            raise NativeGraphError('receipt_graph')
        if graph.get('target') not in graph.get('observed', []):
            raise NativeGraphError('receipt_target')
        ledger = receipt.get('ledger')
        if not isinstance(ledger, dict) or ledger.get('proved') is not True:
            raise NativeGraphError('receipt_ledger')
        rows = ledger.get('rows')
        if not isinstance(rows, list) or not rows:
            raise NativeGraphError('receipt_ledger_rows')
        reconciliation = receipt.get('reconciliation')
        if not isinstance(reconciliation, dict) \
                or reconciliation.get('proved') is not True:
            raise NativeGraphError('receipt_reconciliation')
        discharge = receipt.get('discharge')
        if not isinstance(discharge, dict) or discharge.get('origin') not in (
                'raw_pass', 'discharged_eager_expected_context'):
            raise NativeGraphError('receipt_discharge')
        if discharge['origin'] == 'discharged_eager_expected_context':
            if raw.get('verdict') != 'INCONCLUSIVE':
                raise NativeGraphError('receipt_discharge_raw')
            if discharge.get('native_reasons') != [DISCHARGE_NATIVE]:
                raise NativeGraphError('receipt_discharge_reasons')
            if not set(discharge.get('raw_reasons', ())).issubset(
                    DISCHARGE_DERIVED):
                raise NativeGraphError('receipt_discharge_derived')
            if discharge.get('expected_context_records', 0) <= 0:
                raise NativeGraphError('receipt_discharge_positive')
        else:
            if raw.get('verdict') != 'PASS':
                raise NativeGraphError('receipt_discharge_raw')
        for key in ('graph_by_kind', 'expected_by_kind', 'other_by_kind',
                    'native_by_kind'):
            row = reconciliation.get(key)
            if not isinstance(row, dict) or any(
                    type(value) is not int or value < 0
                    for value in row.values()):
                raise NativeGraphError('receipt_kind_map:' + key)
        total = sum(reconciliation['native_by_kind'].values())
        if total != reconciliation.get('records'):
            raise NativeGraphError('receipt_records')
        if sum(reconciliation['graph_by_kind'].values()) != \
                reconciliation.get('graph_records'):
            raise NativeGraphError('receipt_graph_records')
        if sum(reconciliation['expected_by_kind'].values()) != \
                reconciliation.get('expected_context_records'):
            raise NativeGraphError('receipt_expected_records')
        if sum(reconciliation['other_by_kind'].values()) != \
                reconciliation.get('other_context_records'):
            raise NativeGraphError('receipt_other_records')
    return True


__all__ = ['DISCHARGE_DERIVED', 'DISCHARGE_NATIVE', 'NativeGraphError',
           'RAW_SCHEMA', 'SCHEMA', 'compose_native_placement',
           'validate_native_placement']
