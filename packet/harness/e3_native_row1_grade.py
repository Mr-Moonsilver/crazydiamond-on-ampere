"""Grade the native oracle's actual flat records; never manufacture proof.

The producer directly compares bytes. Digests here bind that producer's raw
records, not tensor contents. This pure CPU grader has no GPU/engine side effect.
The phase owner must separately retain the collective configure/disable ACKs.
"""
import hashlib
import json

SCHEMA = 's5-native-row1-store/1'
RANKS = (0, 1, 2, 3)
IDENTITY = ('rank', 'worker_incarnation', 'engine_epoch', 'route_generation',
            'storage_epoch', 'source_epoch', 'state_epoch', 'mapping_version',
            'layer', 'operation')
BUCKETS = {'dense_page_bucket': 'dense_dma', 'sparse_row_bucket': 'sparse_dma'}
OUTPUT = 'qsa_output_before_moe'


def _int(value, minimum=0):
    return type(value) is int and value >= minimum


def _text(value):
    return type(value) is str and 0 < len(value) <= 128


def _total(rows, field):
    values = [row.get(field) for row in rows]
    return sum(values) if all(_int(value) for value in values) else None


def _hash(kind, payload):
    return hashlib.sha256(json.dumps(dict(payload, kind=kind), sort_keys=True,
        default=str, separators=(',', ':')).encode()).hexdigest()


def _identity(row):
    return {key: row.get(key) for key in IDENTITY}


def _digest(row, layer=False):
    """Reconstruct the producer's exact digest envelope, including JSON lists."""
    identity = _identity(row)
    if layer:
        keys = ('passed', 'aborted', 'route', 'bucket_checks', 'bucket_rows',
                'bucket_bytes', 'output_checks', 'output_rows', 'output_bytes',
                'hold_released', 'hold_stats')
        return _hash('row1_layer', dict(identity=identity,
            **{key: row.get(key) for key in keys}))
    keys = ('route', 'rows', 'pieces', 'compared_bytes', 'different_bytes',
            'equal', 'observed_ns')
    payload = dict(identity=identity, **{key: row.get(key) for key in keys})
    if row.get('operation') == OUTPUT:
        payload.update(bucket=None, finite=row.get('finite'))
        return _hash('row1_output', payload)
    payload.update(bucket=row.get('bucket'), compact_start=row.get('compact_start'))
    return _hash('row1_bucket', payload)


def grade(evidence, *, run_id, engine_epoch):
    failures, missing, checks = [], [], {}

    def fail(reason):
        if reason not in failures:
            failures.append(reason)

    def absent(reason):
        if reason not in missing:
            missing.append(reason)

    def require(label, condition):
        checks[label] = checks.get(label, True) and bool(condition)
        if not condition:
            fail(label)
        return bool(condition)

    workers = evidence.get('workers') if type(evidence) is dict else None
    require('phase', type(evidence) is dict and evidence.get('phase') == 'row1'
            and type(evidence.get('engine_epoch')) is int
            and evidence['engine_epoch'] == engine_epoch)
    if type(evidence) is dict and evidence.get('vm_receipt_used_as_engine_evidence') is True:
        fail('vm_receipt_as_engine_evidence')
    if not _text(run_id) or not _int(engine_epoch, 1):
        absent('phase_identity')
    if type(workers) not in (list, tuple) or not workers:
        absent('worker_packets')
        workers = []
    seen, generations, disabled_generations = set(), set(), set()
    for packet in workers:
        if type(packet) is not dict:
            fail('packet_not_object')
            continue
        rank = packet.get('rank')
        if type(rank) is not int or rank not in RANKS or rank in seen:
            fail('rank_identity')
            continue
        seen.add(rank)
        prefix = 'rank%d:' % rank
        require(prefix + 'schema', packet.get('schema') == SCHEMA)
        require(prefix + 'run', packet.get('run_id') == run_id)
        require(prefix + 'engine', type(packet.get('engine_epoch')) is int
                and packet['engine_epoch'] == engine_epoch)
        require(prefix + 'packet_identity', _text(packet.get('worker_incarnation'))
                and _int(packet.get('route_generation')))
        require(prefix + 'disabled', packet.get('enabled') is False)
        generation = packet.get('route_generation')
        disabled = packet.get('disabled_route_generation')
        if _int(generation):
            generations.add(generation)
        if require(prefix + 'disable_generation', _int(generation)
                   and _int(disabled) and disabled >= generation):
            disabled_generations.add(disabled)
        names = packet.get('local_layers')
        if not (type(names) in (tuple, list) and 0 < len(names) <= 64
                and all(_text(name) for name in names) and len(set(names)) == len(names)):
            absent(prefix + 'local_layer_census')
            continue
        raw = packet.get('raw')
        if type(raw) is not dict:
            absent(prefix + 'raw_store')
            continue
        require(prefix + 'raw_disabled', raw.get('enabled') is False)
        collections = [raw.get(key) for key in ('records', 'layer_records', 'failure_records')]
        if any(type(rows) not in (tuple, list) for rows in collections):
            absent(prefix + 'raw_records')
            continue
        records, layers, errors = collections
        capacity = raw.get('capacity')
        require(prefix + 'capacity', _int(capacity, 1) and capacity <= 65536)
        for key, rows in zip(('comparisons', 'layers', 'failures'), collections):
            require(prefix + key + '_census', _int(raw.get(key))
                    and raw[key] == len(rows) and _int(capacity, 1) and len(rows) <= capacity)
        if errors:
            fail(prefix + 'oracle_failure')
        if not records or not layers:
            absent(prefix + 'comparison_or_layer_records')
        groups, terminals = {}, {}

        def identity(row):
            if type(row) is not dict:
                fail(prefix + 'record_not_object')
                return None
            good = require(prefix + 'record_identity',
                row.get('rank') == rank and type(row.get('rank')) is int
                and row.get('worker_incarnation') == packet.get('worker_incarnation')
                and _text(row.get('worker_incarnation'))
                and row.get('engine_epoch') == engine_epoch
                and type(row.get('engine_epoch')) is int
                and row.get('route_generation') == packet.get('route_generation')
                and _int(row.get('route_generation'))
                and row.get('layer') in names
                and all(_int(row.get(key), 1) for key in (
                    'storage_epoch', 'source_epoch', 'state_epoch', 'mapping_version')))
            if not good:
                return None
            return tuple(row[key] for key in IDENTITY[:-1])

        for row in records:
            key = identity(row)
            if key is None:
                continue
            op = row.get('operation')
            if not require(prefix + 'operation', type(op) is str and op in (*BUCKETS, OUTPUT)):
                continue
            require(prefix + 'direct_comparison', row.get('equal') is True
                    and (row.get('different_bytes') is None
                         or (type(row.get('different_bytes')) is int
                             and row['different_bytes'] == 0)))
            require(prefix + 'byte_coverage', _int(row.get('compared_bytes'), 1)
                    and _int(row.get('rows'), 1)
                    and row.get('coverage_bytes') == row.get('compared_bytes')
                    and row.get('coverage_rows') == row.get('rows'))
            require(prefix + 'duration', _int(row.get('observed_ns')))
            require(prefix + 'record_digest', row.get('digest') == _digest(row))
            if op == OUTPUT:
                require(prefix + 'output', row.get('finite') is True
                        and row.get('bucket') == -1 and row.get('compact_start') is None
                        and row.get('route') in ('dense_dma', 'sparse_dma', 'empty')
                        and _int(row.get('pieces')))
            elif op in BUCKETS:
                require(prefix + 'bucket', row.get('route') == BUCKETS[op]
                        and _int(row.get('bucket')) and _int(row.get('pieces'), 1)
                        and row.get('finite') is None
                        and (_int(row.get('compact_start')) if op == 'sparse_row_bucket'
                             else row.get('compact_start') is None))
            group = groups.setdefault(key, [])
            if any((old.get('operation'), old.get('bucket')) == (op, row.get('bucket'))
                   for old in group):
                fail(prefix + 'duplicate_comparison')
            group.append(row)

        covered = set()
        for row in layers:
            key = identity(row)
            if key is None:
                continue
            if key in terminals:
                fail(prefix + 'duplicate_layer')
            terminals[key] = row
            require(prefix + 'layer_operation', row.get('operation') == 'layer_row1')
            require(prefix + 'layer_digest', row.get('digest') == _digest(row, layer=True))
            if row.get('aborted') is not None:
                absent(prefix + 'aborted_layer')
                continue
            require(prefix + 'layer_passed', row.get('passed') is True)
            require(prefix + 'hold_release', row.get('hold_released') is True)
            observations = groups.get(key, [])
            buckets = [c for c in observations if c.get('operation') in BUCKETS]
            outputs = [c for c in observations if c.get('operation') == OUTPUT]
            require(prefix + 'output_multiplicity', len(outputs) == 1)
            for kind, subset in (('bucket', buckets), ('output', outputs)):
                require(prefix + kind + '_totals',
                    all(_int(c.get(field)) for c in subset for field in ('rows', 'compared_bytes'))
                    and all(_int(row.get(kind + '_' + field))
                            for field in ('checks', 'rows', 'bytes'))
                    and row.get(kind + '_checks') == len(subset)
                    and row.get(kind + '_rows') == _total(subset, 'rows')
                    and row.get(kind + '_bytes') == _total(subset, 'compared_bytes'))
            route = row.get('route')
            require(prefix + 'layer_route', route in ('dense_dma', 'sparse_dma', 'empty')
                    and all(c.get('route') == route for c in observations))
            require(prefix + 'reference_fill_coverage', len(outputs) == 1
                    and outputs[0].get('pieces') == (len(buckets) if route == 'dense_dma' else 0))
            stats = row.get('hold_stats')
            if (type(stats) not in (list, tuple)
                    or any(type(item) not in (list, tuple) or len(item) != 2
                           or type(item[0]) is not str or not _int(item[1]) for item in stats)):
                absent(prefix + 'hold_census')
            else:
                counts = dict(stats)
                require(prefix + 'hold_census', len(counts) == len(stats)
                        and counts.get('released') == 1 and counts.get('abandoned') == 0
                        and _int(counts.get('buckets'))
                        and all(counts.get(key) == counts.get('buckets')
                                for key in ('copied', 'published', 'consumed')))
                if route == 'sparse_dma':
                    require(prefix + 'sparse_hold_totals', counts.get('buckets') == len(buckets)
                            and counts.get('rows') == _total(buckets, 'rows'))
            if route != 'empty' and not buckets:
                absent(prefix + 'bucket_coverage')
            if sorted(c.get('bucket') for c in buckets if _int(c.get('bucket'))) != list(range(len(buckets))):
                fail(prefix + 'bucket_sequence')
            covered.add((row.get('layer'), route))
        if set(groups) != set(terminals):
            absent(prefix + 'unclosed_comparison_groups')
        required = {(name, route) for name in names for route in ('dense_dma', 'sparse_dma')}
        if not required <= covered:
            absent(prefix + 'layer_route_cartesian_coverage')
    if seen != set(RANKS):
        absent('rank_coverage')
    if len(generations) != 1 or len(disabled_generations) != 1:
        absent('collective_route_identity')
    verdict = 'FAIL' if failures else ('INCONCLUSIVE' if missing else 'PASS')
    return dict(schema='s5-native-row1-grade/1', verdict=verdict,
                failures=failures, missing=missing, checks=checks,
                scope='raw direct-byte oracle and local hold release; collective ACKs are retained separately')
