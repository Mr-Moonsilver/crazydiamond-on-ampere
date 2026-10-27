"""Paired native overlap accounting (``s5-native-overlap-accounting/1``).

One retained ``s5ro-correlation/2`` report carries the decode engine offers and
the source-declared prefill offers of the same window.  This module proves that
the two declared partitions cover the whole report:

* every offer, scope event, CUSTOM0 join, API record and activity row is
  assigned to exactly one partition (or, for a correlated host-only API, to the
  host-control bucket),
* the multiset of rows in the two views equals the multiset in the retained
  whole report,
* activity counts by kind and in total add up exactly,
* the decode partition is regraded by the unchanged ``grade_engine`` on its
  own filtered view, and the prefill partition by ``grade_prefill``.

Views are built only by exact CUSTOM0/external/offer ids.  Nothing is selected
by context, symbol, name, time or "last active scope".  The whole raw report
and its sticky loss flags are never modified; a raw loss can only make the
composed verdict worse.

The decode ledger cannot seal on a bridge that also carries prefill offers
(``grade_engine`` would report ``undeclared_offer`` for them).  The caller
therefore captures the decode ledger's declared metadata and window before the
shared window ends and passes them here; this composer rebuilds the decode view
and regrades it.  That wiring is intentionally outside this module.
"""
import hashlib
import json
from collections import Counter

try:
    from activity_accounting import api_kind as _api_kind
    from activity_accounting import KINDS
    from engine_offers import grade_engine, validate_receipt as validate_engine
    from native_prefill_offers import (MAX_OFFERS, NATIVE_SCHEMA,
                                       grade_prefill, partition_view,
                                       validate_receipt as validate_prefill)
except ImportError:  # pragma: no cover - package-relative import
    from .activity_accounting import api_kind as _api_kind
    from .activity_accounting import KINDS
    from .engine_offers import grade_engine, validate_receipt as validate_engine
    from .native_prefill_offers import (MAX_OFFERS, NATIVE_SCHEMA,
                                        grade_prefill, partition_view,
                                        validate_receipt as validate_prefill)

SCHEMA = 's5-native-overlap-accounting/1'
PLACEMENT_SCHEMA = 's5-engine-accounting/1'
MAX_RECEIPT_BYTES = 64 * 1024 * 1024
HOST = 'host'


class OverlapError(ValueError):
    pass


def _int(value, reason, minimum=1):
    if type(value) is not int or value < minimum:
        raise OverlapError(reason)
    return value


def _declared_ids(declared, label):
    if not isinstance(declared, dict) or not declared:
        raise OverlapError(label + '_declared_empty')
    out = {}
    for raw, row in declared.items():
        try:
            oid = int(raw)
        except (TypeError, ValueError):
            raise OverlapError(label + '_declared_identity')
        if oid < 1 or oid in out or not isinstance(row, dict):
            raise OverlapError(label + '_declared_identity')
        out[oid] = row
    return out


def _offer_map(offers):
    if not isinstance(offers, list) or not offers:
        raise OverlapError('offers_empty')
    seen = {}
    for row in offers:
        if not isinstance(row, dict):
            raise OverlapError('offer_shape')
        oid = row.get('id')
        if type(oid) is not int or oid < 1 or oid in seen:
            raise OverlapError('offer_identity')
        seen[oid] = row
    return seen


def _sig_scope(row):
    return (row.get('id'), row.get('action'), row.get('thread'),
            row.get('system_thread'), row.get('parent'))


def _sig_api(row):
    return (row.get('correlation_id'), row.get('domain'), row.get('name'),
            row.get('return_value'), row.get('thread'), row.get('start'),
            row.get('end'))


def _sig_external(row):
    return (row.get('correlation_id'), row.get('kind'),
            row.get('external_id'))


def _sig_activity(row):
    return (row.get('correlation_id'), row.get('kind'), row.get('graph_id', 0),
            row.get('context_id'), row.get('stream_id'),
            row.get('copy_kind'), row.get('name'), row.get('count'))


def _rows(native, key):
    rows = native.get(key)
    if not isinstance(rows, list):
        raise OverlapError(key + '_shape')
    for row in rows:
        if not isinstance(row, dict):
            raise OverlapError(key + '_row')
    return rows


def _external_by_cid(rows, ids):
    """Validate CUSTOM0 joins and return ``{correlation_id: offer_id}``."""
    out = {}
    for row in rows:
        cid = _int(row.get('correlation_id'), 'external_identity')
        if cid in out:
            raise OverlapError('duplicate_external_join')
        if row.get('kind') != 'custom0':
            raise OverlapError('foreign_external_kind')
        oid = row.get('external_id')
        if oid not in ids:
            raise OverlapError('unpartitioned_external:' + str(oid))
        out[cid] = oid
    return out


def _api_bucket(row, external):
    cid = _int(row.get('correlation_id'), 'api_identity')
    if cid in external:
        return external[cid]
    if _api_kind(row.get('name')) == 'host':
        return HOST
    raise OverlapError('unpartitioned_api:' + str(cid))


def _window_identity(window):
    """Merge a ledger window with its binding (the decode ledger nests them)."""
    if not isinstance(window, dict):
        raise OverlapError('window_shape')
    binding = window.get('binding')
    binding = binding if isinstance(binding, dict) else {}
    return {
        'worker': window.get('worker', binding.get('worker')),
        'rank': window.get('rank', binding.get('rank')),
        'epoch': window.get('epoch', binding.get('engine_epoch')),
        'generation': window.get('generation',
                                 binding.get('route_generation')),
        'route_generation': window.get('route_generation',
                                       binding.get('route_generation')),
        'storage_epoch': window.get('storage_epoch',
                                    binding.get('storage_epoch')),
        'run_id': window.get('run_id', binding.get('run_id')),
        'window_id': window.get('window_id', binding.get('window_id')),
    }


def _window_envelope(native, window, label):
    whole = native.get('window')
    if not isinstance(whole, dict):
        raise OverlapError('native_window_missing')
    ws = _int(whole.get('start_ns'), 'native_window_start', 0)
    we = _int(whole.get('end_ns'), 'native_window_end', 0)
    wr = _int(whole.get('route_generation'), 'native_window_route', 0)
    if we < ws:
        raise OverlapError('native_window_bounds')
    binding = window.get('binding') if isinstance(window, dict) else None
    binding = binding if isinstance(binding, dict) else {}
    start = window.get('native_start_ns', binding.get('native_start_ns'))
    end = window.get('native_end_ns', binding.get('native_end_ns'))
    start = _int(start, label + '_window_start', 0)
    end = _int(end, label + '_window_end', 0)
    route = _int(window.get('route_generation',
                            binding.get('route_generation')),
                 label + '_window_route', 0)
    if end <= start:
        raise OverlapError(label + '_window_bounds')
    if start != ws or end != we:
        raise OverlapError(label + '_window_differs_from_whole')
    if route != wr:
        raise OverlapError(label + '_window_route')
    return {'start_ns': start, 'end_ns': end, 'route_generation': route}


def _view(native, offers, ids, envelope):
    if not ids:
        view = {k: native.get(k) for k in (
            'schema', 'thread_id_type', 'complete', 'capacity',
            'reserved_bytes', 'errors', 'overflow', 'open_scopes',
            'boundary_records', 'window')}
        view.update(scopes=[], apis=[], external=[], activities=[],
                    activity_records=0,
                    partition_totals={kind: 0 for kind in KINDS})
    else:
        view = partition_view(native, offers, ids, allow_shared=True)
    view['window'] = dict(envelope)
    return view


def _summary(receipt, name):
    body = json.dumps(receipt, sort_keys=True, separators=(',', ':'),
                      default=str).encode()
    return {
        'name': name,
        'verdict': receipt.get('verdict'),
        'joined_records': receipt.get('joined_records'),
        'counts': receipt.get('counts') or receipt.get('eager_by_kind'),
        'failures': list(receipt.get('failures', [])),
        'missing': list(receipt.get('missing', [])),
        'receipt_sha256': hashlib.sha256(body).hexdigest(),
    }


def _fail_result(window, failures, missing, conservation):
    verdict = 'FAIL' if failures else 'INCONCLUSIVE'
    return {
        'schema': SCHEMA,
        'verdict': verdict,
        'row4': verdict,
        'window': window,
        'partitions': {},
        'conservation': conservation,
        'decode': None,
        'prefill': None,
        'raw': None,
        'historical_placement': None,
        'failures': sorted(set(failures)),
        'missing': sorted(set(missing)),
        'notes': [],
    }


def compose_overlap(native, offers, *, decode_declared, decode_window,
                    prefill_declared, prefill_window,
                    decode_receipt=None, prefill_receipt=None,
                    historical_placement=None):
    """Prove the two partitions cover the retained whole report and regrade.

    ``decode_declared`` / ``prefill_declared`` are the bounded per-offer
    metadata maps captured from the live ledgers (int or decimal-string ids).
    ``*_window`` are the ledgers' captured window declarations.  The function
    is pure: no input object is mutated and no native loss flag is reset.
    """
    failures = []
    missing = []
    notes = []
    conservation = {'proved': False, 'offers': 0, 'scopes': 0, 'apis': 0,
                    'external': 0, 'activities': 0, 'host_apis': 0,
                    'by_kind': {}, 'failures': []}

    def fail(reason):
        if reason not in failures:
            failures.append(reason)

    def miss(reason):
        if reason not in missing:
            missing.append(reason)

    def cons(reason):
        if reason not in conservation['failures']:
            conservation['failures'].append(reason)
        fail('conservation:' + reason)

    if not isinstance(native, dict) or native.get('schema') != NATIVE_SCHEMA:
        return _fail_result(None, ['native_schema'], [], conservation)
    try:
        whole_window = native.get('window')
        if not isinstance(whole_window, dict):
            raise OverlapError('native_window_missing')
        d_decl = _declared_ids(decode_declared, 'decode')
        p_decl = _declared_ids(prefill_declared, 'prefill')
        d_ids = set(d_decl)
        p_ids = set(p_decl)
        if d_ids & p_ids:
            cons('partition_overlap')
        if not d_ids:
            miss('decode_partition_empty')
        if not p_ids:
            miss('prefill_partition_empty')
        union = d_ids | p_ids
        offered = _offer_map(offers)
        if union - set(offered):
            cons('declared_offer_missing')
        for oid in offered:
            if oid not in union:
                cons('unpartitioned_offer:' + str(oid))
        if len(offered) > MAX_OFFERS:
            miss('offer_capacity_exceeded')
        conservation['offers'] = len(offered)

        d_env = _window_envelope(native, decode_window, 'decode')
        p_env = _window_envelope(native, prefill_window, 'prefill')
        d_ident = _window_identity(decode_window)
        p_ident = _window_identity(prefill_window)
        for name in d_ident:
            if d_ident[name] != p_ident[name]:
                fail('partition_identity_mismatch:' + name)
            if d_ident[name] is None:
                miss('partition_identity_missing:' + name)

        # offers and window loss
        if native.get('complete') is not True:
            miss('correlation_incomplete')
        for key, reason in (('overflow', 'correlation_overflow'),
                            ('errors', 'correlation_errors'),
                            ('open_scopes', 'correlation_open_scopes'),
                            ('boundary_records', 'correlation_boundary')):
            value = native.get(key)
            if type(value) is not int or value < 0:
                raise OverlapError(key + '_shape')
            if value:
                miss(reason)
        activity_records = _int(native.get('activity_records'),
                                'activity_records_shape', 0)

        # rows and their exact partition assignment
        scopes = _rows(native, 'scopes')
        apis = _rows(native, 'apis')
        external_rows = _rows(native, 'external')
        activities = _rows(native, 'activities')
        for row in scopes:
            if row.get('id') not in union:
                cons('unpartitioned_scope:' + str(row.get('id')))
        try:
            external = _external_by_cid(external_rows, union)
        except OverlapError as exc:
            cons(str(exc))
            external = {}
        api_part = {}
        for row in apis:
            try:
                bucket = _api_bucket(row, external)
            except OverlapError as exc:
                cons(str(exc))
                continue
            api_part[row.get('correlation_id')] = bucket
            if bucket == HOST:
                conservation['host_apis'] += 1
        act_part = {}
        by_kind = Counter()
        for row in activities:
            cid = row.get('correlation_id')
            if cid not in external:
                cons('unpartitioned_activity:' + str(cid))
                continue
            if cid not in api_part:
                cons('activity_without_api:' + str(cid))
                continue
            act_part[cid] = external[cid]
            kind = row.get('kind')
            if kind not in KINDS:
                cons('activity_kind')
                continue
            count = row.get('count')
            if type(count) is not int or count <= 0:
                cons('activity_count')
                continue
            by_kind[kind] += count
        conservation['scopes'] = len(scopes)
        conservation['apis'] = len(apis)
        conservation['external'] = len(external_rows)
        conservation['activities'] = len(activities)
        conservation['by_kind'] = {kind: by_kind[kind] for kind in KINDS}
        if conservation['host_apis'] and not conservation['apis']:
            cons('host_api_shape')

        d_view = _view(native, offers, d_ids, d_env)
        p_view = _view(native, offers, p_ids, p_env)

        # multiset union / disjointness, per row class
        def multiset(rows, sig):
            return Counter(sig(row) for row in rows)

        d_scopes = [row for row in scopes if row.get('id') in d_ids]
        p_scopes = [row for row in scopes if row.get('id') in p_ids]
        d_api = [row for row in apis
                 if api_part.get(row.get('correlation_id')) in d_ids]
        p_api = [row for row in apis
                 if api_part.get(row.get('correlation_id')) in p_ids]
        h_api = [row for row in apis if api_part.get(row.get('correlation_id'))
                 == HOST]
        d_ext = [row for row in external_rows
                 if external.get(row.get('correlation_id')) in d_ids]
        p_ext = [row for row in external_rows
                 if external.get(row.get('correlation_id')) in p_ids]
        d_act = [row for row in activities
                 if act_part.get(row.get('correlation_id')) in d_ids]
        p_act = [row for row in activities
                 if act_part.get(row.get('correlation_id')) in p_ids]
        for label, parts, whole, sig in (
                ('scopes', d_scopes + p_scopes, scopes, _sig_scope),
                ('apis', d_api + p_api + h_api, apis, _sig_api),
                ('external', d_ext + p_ext, external_rows, _sig_external),
                ('activities', d_act + p_act, activities, _sig_activity)):
            if multiset(parts, sig) != multiset(whole, sig):
                cons(label + '_union')
        if len(d_act) + len(p_act) != len(activities):
            cons('activities_duplicated')
        if len(d_scopes) + len(p_scopes) != len(scopes):
            cons('scopes_duplicated')
        if len(d_ext) + len(p_ext) != len(external_rows):
            cons('external_duplicated')
        if len(d_api) + len(p_api) + len(h_api) != len(apis):
            cons('apis_duplicated')
        d_records = sum(row['count'] for row in d_act)
        p_records = sum(row['count'] for row in p_act)
        if d_records + p_records != activity_records:
            cons('activity_records_union')
        if sum(by_kind.values()) != activity_records:
            cons('activity_counts_union')
        conservation['proved'] = not conservation['failures']

        # regrade both partitions on their own views
        d_offers = [row for row in offers if row.get('id') in d_ids]
        decode_out = grade_engine(d_view, d_offers, d_decl, dict(decode_window))
        prefill_out = grade_prefill(p_view, p_decl, dict(prefill_window))
        if decode_receipt is not None:
            try:
                validate_engine(decode_receipt)
            except Exception:
                fail('decode_receipt_invalid')
            else:
                if (decode_receipt.get('verdict') != decode_out.get('verdict')
                        or decode_receipt.get('joined_records')
                        != decode_out.get('joined_records')):
                    fail('decode_receipt_mismatch')
        if prefill_receipt is not None:
            try:
                validate_prefill(prefill_receipt)
            except Exception:
                fail('prefill_receipt_invalid')
            else:
                if (prefill_receipt.get('verdict') != prefill_out.get('verdict')
                        or prefill_receipt.get('joined_records')
                        != prefill_out.get('joined_records')):
                    fail('prefill_receipt_mismatch')

        historical = None
        if historical_placement is not None:
            try:
                validate_engine(historical_placement)
            except Exception:
                fail('historical_placement_invalid')
            else:
                raw = json.dumps(historical_placement, sort_keys=True,
                                 separators=(',', ':'),
                                 default=str).encode()
                historical = {
                    'verdict': historical_placement.get('verdict'),
                    'failures': list(historical_placement.get('failures', [])),
                    'missing': list(historical_placement.get('missing', [])),
                    'sha256': hashlib.sha256(raw).hexdigest(),
                    'scope': 'historical-decode-placement',
                    'modified': False,
                }
                if historical_placement.get('verdict') == 'FAIL':
                    fail('historical_placement_fail')

        for label, out in (('decode', decode_out), ('prefill', prefill_out)):
            if out.get('failures'):
                fail(label + '_grade_fail:' + out['failures'][0])
            elif out.get('missing'):
                miss(label + '_grade_inconclusive')
        if not conservation['proved']:
            cons('not_proved')

        raw_loss = any(native.get(k) for k in (
            'errors', 'overflow', 'open_scopes', 'boundary_records'))
        if raw_loss:
            miss('whole_raw_loss')
        if native.get('complete') is not True:
            miss('whole_raw_incomplete')

        d_sum = _summary(decode_out, 'decode')
        p_sum = _summary(prefill_out, 'prefill')
        verdict = 'FAIL' if failures else (
            'INCONCLUSIVE' if missing else 'PASS')
        if verdict == 'PASS' and not conservation['proved']:
            verdict = 'FAIL'
        return {
            'schema': SCHEMA,
            'verdict': verdict,
            'row4': verdict,
            'window': d_ident,
            'whole': {
                'schema': native.get('schema'),
                'window': whole_window,
                'complete': native.get('complete'),
                'errors': native.get('errors'),
                'overflow': native.get('overflow'),
                'open_scopes': native.get('open_scopes'),
                'boundary_records': native.get('boundary_records'),
                'activity_records': activity_records,
                'offers': len(offered),
                'by_kind': {kind: by_kind[kind] for kind in KINDS},
            },
            'partitions': {
                'decode': {
                    'offers': len(d_ids), 'window': d_env,
                    'activity_records': d_records,
                    'by_kind': {kind: sum(
                        row['count'] for row in d_act
                        if row.get('kind') == kind) for kind in KINDS},
                    'offers_partitioned': sorted(d_ids),
                    'verdict': decode_out.get('verdict'),
                    'receipt_sha256': d_sum['receipt_sha256'],
                },
                'prefill': {
                    'offers': len(p_ids), 'window': p_env,
                    'activity_records': p_records,
                    'by_kind': {kind: sum(
                        row['count'] for row in p_act
                        if row.get('kind') == kind) for kind in KINDS},
                    'offers_partitioned': sorted(p_ids),
                    'verdict': prefill_out.get('verdict'),
                    'receipt_sha256': p_sum['receipt_sha256'],
                    'roles': prefill_out.get('roles'),
                    'families': prefill_out.get('families'),
                    'completion': prefill_out.get('completion'),
                },
            },
            'conservation': conservation,
            'decode': d_sum,
            'prefill': p_sum,
            'historical_placement': historical,
            'raw_unchanged': True,
            'failures': sorted(set(failures)),
            'missing': sorted(set(missing)),
            'notes': notes + [
                'partition views are selected by exact offer/external ids; '
                'no symbol, context or time heuristic is used',
                'the retained whole report and every sticky native flag are '
                'unmodified; raw loss can only make this verdict worse',
                'the historical decode-only placement receipt is not '
                'rewritten or discharged',
            ],
        }
    except OverlapError as exc:
        return _fail_result(
            {k: (decode_window or {}).get(k) for k in (
                'worker', 'rank', 'epoch', 'generation', 'route_generation',
                'storage_epoch', 'run_id', 'window_id')}
            if isinstance(decode_window, dict) else None,
            failures, missing + ['compose_failed:' + str(exc)], conservation)


def validate_overlap(receipt):
    """Reject a mutated composed receipt; PASS needs proven conservation."""
    if not isinstance(receipt, dict) or receipt.get('schema') != SCHEMA:
        raise OverlapError('receipt_schema')
    verdict = receipt.get('verdict')
    if verdict not in ('PASS', 'INCONCLUSIVE', 'FAIL'):
        raise OverlapError('receipt_verdict')
    if receipt.get('row4') not in ('PASS', 'INCONCLUSIVE', 'FAIL'):
        raise OverlapError('receipt_row4')
    conservation = receipt.get('conservation')
    if not isinstance(conservation, dict):
        raise OverlapError('receipt_conservation')
    if verdict == 'PASS':
        if conservation.get('proved') is not True:
            raise OverlapError('receipt_pass_without_conservation')
        if receipt.get('failures') or receipt.get('missing'):
            raise OverlapError('receipt_pass_reasons')
        parts = receipt.get('partitions')
        if not isinstance(parts, dict):
            raise OverlapError('receipt_partitions')
        ids = []
        for name in ('decode', 'prefill'):
            row = parts.get(name)
            if not isinstance(row, dict) or row.get('verdict') != 'PASS':
                raise OverlapError('receipt_partition_verdict:' + name)
            ids.append(set(row.get('offers_partitioned', [])))
        if ids[0] & ids[1]:
            raise OverlapError('receipt_partition_overlap')
        whole = receipt.get('whole')
        if not isinstance(whole, dict):
            raise OverlapError('receipt_whole')
        if whole.get('complete') is not True:
            raise OverlapError('receipt_raw_incomplete')
        if any(whole.get(k) for k in ('errors', 'overflow', 'open_scopes',
                                      'boundary_records')):
            raise OverlapError('receipt_raw_loss')
        total = 0
        for _kind, count in (whole.get('by_kind') or {}).items():
            if type(count) is not int or count < 0:
                raise OverlapError('receipt_whole_kind')
            total += count
        if total != whole.get('activity_records'):
            raise OverlapError('receipt_whole_records')
        part_total = 0
        for name in ('decode', 'prefill'):
            part = parts[name]
            if type(part.get('activity_records')) is not int:
                raise OverlapError('receipt_partition_records')
            part_total += part['activity_records']
            subtotal = sum((part.get('by_kind') or {}).values())
            if subtotal != part['activity_records']:
                raise OverlapError('receipt_partition_kind_total')
        if part_total != whole['activity_records']:
            raise OverlapError('receipt_record_union')
        if receipt.get('raw_unchanged') is not True:
            raise OverlapError('receipt_raw_unchanged')
    return True


__all__ = ['HOST', 'OverlapError', 'SCHEMA', 'compose_overlap',
           'validate_overlap']
