"""Regrade a retained native worker window; no collection or cached verdicts.

The native JSON read is a bounded report read, not a read of bank data.  The
raw graph receipt and the complete shared correlation report remain separate.
"""
import hashlib
import json

from native_graph_placement import compose_native_placement

REPORT_LIMIT = 64 * 1024 ** 2


def digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def regrade_packet(packet, *, rank, engine_epoch, run_id=None, window_id=None,
                   route_generation=None):
    """Return FAIL for contradictions, INC for absent proof, never infer work."""
    failures, missing, checks = [], [], {}

    def require(name, condition):
        checks[name] = bool(condition)
        if not condition:
            failures.append(name)

    def result(composed=None):
        return {'verdict': 'FAIL' if failures else (
                    'INCONCLUSIVE' if missing else 'PASS'),
                'failures': sorted(set(failures)),
                'missing': sorted(set(missing)), 'checks': checks,
                'binding': (composed or {}).get('binding'),
                'composed': composed}

    if type(packet) is not dict:
        missing.append('native_observer_packet')
        return result()
    raw = packet.get('receipt')
    if type(raw) is dict and raw.get('verdict') == 'FAIL':
        failures.append('raw_placement_fail')
    native = packet.get('native_report')
    metadata = packet.get('native_report_read')
    if type(native) is not dict:
        missing.append('native_report')
    if type(metadata) is not dict:
        missing.append('native_report_read')
    else:
        require('native_read_function', metadata.get('function') == 's5ro_correlation_report')
        require('native_read_completed', metadata.get('completed') is True)
        require('native_read_return', type(metadata.get('return_code')) is int
                and metadata['return_code'] == 0)
        require('native_read_limit', metadata.get('limit_bytes') == REPORT_LIMIT)
        require('native_read_capacity', metadata.get('capacity_bytes') == REPORT_LIMIT)
        size = metadata.get('required_bytes')
        require('native_read_size', type(size) is int and 0 < size <= REPORT_LIMIT)
        require('native_read_schema', metadata.get('schema') == 's5ro-correlation/2')
        if type(native) is dict:
            require('native_read_digest', metadata.get('decoded_report_sha256') == digest(native))
    structural_start = len(missing)
    fields = ('receipt', 'native_report', 'offer_report', 'decode_declared',
              'decode_window', 'engine_accounting', 'native_prefill_accounting')
    for key in fields:
        expected_type = list if key == 'offer_report' else dict
        if type(packet.get(key)) is not expected_type:
            missing.append(key)
    prefill = packet.get('native_prefill_accounting')
    if type(prefill) is dict and (type(prefill.get('declared')) is not dict
                                 or type(prefill.get('window')) is not dict):
        missing.append('native_prefill_declaration')
    if len(missing) > structural_start or type(native) is not dict:
        return result()
    before = digest(raw)
    try:
        composed = compose_native_placement(
            raw, native, packet['offer_report'],
            decode_declared=packet['decode_declared'],
            decode_window=packet['decode_window'],
            prefill_declared=prefill['declared'], prefill_window=prefill['window'],
            decode_receipt=packet['engine_accounting'], prefill_receipt=prefill)
    except (KeyError, TypeError, ValueError) as exc:
        missing.append('composition:' + type(exc).__name__)
        return result()
    require('raw_unchanged', before == digest(raw))
    if composed.get('verdict') == 'FAIL':
        failures.extend('composition:' + value for value in composed.get('failures', []))
    elif composed.get('verdict') != 'PASS':
        missing.extend('composition:' + value for value in composed.get('missing', []))
        if not composed.get('missing'):
            missing.append('composition_unproved')
    binding = composed.get('binding')
    if type(binding) is not dict:
        missing.append('binding')
        return result(composed)
    expected = {'rank': rank, 'engine_epoch': engine_epoch,
                'engine_incarnation': 'engine:' + str(engine_epoch)}
    for key, value in (('run_id', run_id), ('window_id', window_id),
                       ('route_generation', route_generation)):
        if value is not None:
            expected[key] = value
    for key, value in expected.items():
        require('binding_' + key, binding.get(key) == value)
    # The closed window retains its original route generation. A later FULL
    # telemetry sample is checked separately and never re-keys the proof.
    generation = binding.get('route_generation')
    require('packet_rank', packet.get('rank') == rank)
    require('packet_epoch', packet.get('engine_epoch') == engine_epoch)
    require('packet_window_generation', packet.get('route_generation') == generation)
    require('observer_stopped', packet.get('observer_stopped') is True)
    current = packet.get('current_route_generation', generation)
    route = packet.get('current_route', packet.get('route'))
    same_split = current == generation and route == 'split'
    returned_full = (type(generation) is int and current == generation + 1
                     and route == 'full' and packet.get('state') == 'closed')
    require('current_route_bound', same_split or returned_full)
    return result(composed)
