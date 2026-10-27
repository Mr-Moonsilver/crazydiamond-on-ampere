#!/usr/bin/env python3
"""q18a qualification phases over the native session (offline client).

One phase per invocation, matching the reviewed queue:

  row1             the other worker's raw direct-byte comparison records: the
                   native QSA bucket bytes compared with the original UVA
                   gather in the same arena, BEFORE MoE, under a real engine
                   source hold released only after the comparison.
  rule4_controls   ONE teacher instrument sequence A1,A2,B,C.  The retained
                   raw schema/2 artifact is graded by the frozen v4 policy.
  rule4_candidate  the candidate artifact, which must be byte-identical to the
                   controls artifact; no second teacher sequence.
  row4             raw ReplayObserver receipts composed with independently
                   validated engine offer accounting for all four ranks.

The engine/worker side owns the actual operations.  This client sends one
schema-1 command to the authenticated loopback session and grades the raw
record it returns.  It never trusts a cached verdict, a flag or a digest-only
claim: the teacher artifact goes through ``e3_teacher_decode_rule4.grade`` and
row4 is recomputed locally with ``native_graph_placement.compose_native_placement``.

Exit: 0 PASS, 1 FAIL, 2 INCONCLUSIVE.  ``--out`` is only created once, with
mode 0600.
"""
import argparse
import copy
import hashlib
import json
import os
import sys
import time

import e3_teacher_decode_rule4 as rule4

_RUNTIME = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'patches', 's5_engine')
if _RUNTIME not in sys.path:
    sys.path.insert(0, _RUNTIME)
import engine_offers  # noqa: E402  (frozen composition arithmetic)

PHASES = ('row1', 'rule4_controls', 'rule4_candidate', 'row4')
TEACHER_SEQUENCE = ['full-A1', 'full-A2', 'split-B', 'full-C']
TEACHER_ROUTES = {'full-A1': 'full', 'full-A2': 'full',
                  'split-B': 'split', 'full-C': 'full'}
FROZEN_TEACHER = 'v4'
TEACHER_SCHEMA = 's5-teacher-decode/2'
ROW_SCHEMA = 's5-teacher-row-mapping/1'
ROW_BASIS = 'sampler-invocation/1'
STREAMS = 8
MEASURED = 256
NATIVE_READ_BYTES = 64 * 1024 * 1024
RANKS = (0, 1, 2, 3)
MODEL_SETTING_KEYS = ('max_model_len', 'max_num_seqs', 'gpu_memory_utilization',
                      'enforce_eager', 'kv_cache_dtype')


def _digest(value):
    return (type(value) is str and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


def _int(value):
    return type(value) is int


def _text(value):
    return type(value) is str and bool(value.strip())


def _canonical_sha(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class Grade:
    def __init__(self, phase):
        self.phase = phase
        self.failures = []
        self.missing = []
        self.checks = {}

    def fail(self, reason):
        if reason not in self.failures:
            self.failures.append(reason)

    def absent(self, reason):
        if reason not in self.missing:
            self.missing.append(reason)

    def require(self, label, condition):
        self.checks[label] = bool(condition)
        if not condition:
            self.fail(label)
        return bool(condition)

    def need(self, label, value, predicate):
        if value is None:
            self.absent(label + ':missing')
            return False
        if not predicate(value):
            self.fail(label + ':invalid')
            return False
        return True

    def verdict(self):
        if self.failures:
            return 'FAIL'
        if self.missing:
            return 'INCONCLUSIVE'
        return 'PASS'


def _specification(phase, model, run_id, artifact_sha256, row1_sha256=None):
    base = {'phase': phase, 'model': model, 'run_id': run_id,
            'immutable_harness': True, 'no_diagnostic_rerun': True,
            'model_settings_unchanged': True}
    if phase == 'row1':
        base.update({
            'transport': {'direct_bytes': True, 'vm_receipt_is_not_engine': True},
            'qsa': {'before_moe': True, 'same_selected_rows': True,
                    'same_arithmetic': True, 'same_arena': True,
                    'source_hold_required': True,
                    'bounded_serial_oracle': True,
                    'second_history_buffer': False},
        })
    elif phase in ('rule4_controls', 'rule4_candidate'):
        base.update({
            'teacher_sequence': list(TEACHER_SEQUENCE),
            'teacher_version': FROZEN_TEACHER,
            'frozen': True,
            'single_artifact': True,
            'retries': 0,
            'picked': False,
            'teacher_runs': 1 if phase == 'rule4_controls' else 0,
        })
        if phase == 'rule4_candidate':
            base['derived_from_controls_artifact'] = True
            base['artifact_sha256'] = artifact_sha256
    else:
        base.update({
            'compose_from': ['row1', 'rule4_controls'],
            'row1_sha256': row1_sha256,
            'artifact_sha256': artifact_sha256,
            'native_report_capacity_bytes': NATIVE_READ_BYTES,
            'graph_receipt': {'raw': True, 'source': 'native'},
        })
    return base


def _grade_row1(g, evidence, *, run_id, engine_epoch):
    # The worker's flat records are the only source. No synthetic nested
    # digest/shape packet is an alternative byte oracle.
    from e3_native_row1_grade import grade
    result = grade(evidence, run_id=run_id, engine_epoch=engine_epoch)
    g.checks.update({'row1:' + key:value for key,value in result['checks'].items()})
    for value in result['failures']:
        g.fail('row1:' + value)
    for value in result['missing']:
        g.absent('row1:' + value)


def _grade_row_mapping(g, name, run, positions):
    mapping = run.get('row_mapping')
    if type(mapping) is not dict:
        g.absent('teacher:row-mapping:' + name)
        return
    g.require('teacher:row-mapping-schema:' + name, mapping.get('schema') == ROW_SCHEMA)
    g.require('teacher:row-mapping-basis:' + name, mapping.get('basis') == ROW_BASIS)
    g.require('teacher:row-mapping-overflow:' + name, mapping.get('overflow') is False)
    g.require('teacher:row-mapping-defects:' + name, mapping.get('defects') == [])
    records = mapping.get('records')
    if type(records) is not list:
        g.absent('teacher:row-records:' + name)
        return
    expected = {(rank, stream, position) for rank in (2, 3)
                for stream in range(STREAMS) for position in positions}
    seen = set()
    for record in records:
        if type(record) is not dict:
            g.fail('teacher:row-record-not-object:' + name)
            continue
        if (record.get('run') != name
                or record.get('repeat') != run.get('repeat')):
            g.fail('teacher:row-record-run:' + name)
        key = (record.get('rank'), record.get('stream'),
               record.get('position'))
        if key in seen:
            g.fail('teacher:row-record-duplicate:' + name)
        seen.add(key)
        if not _digest(record.get('logprob_sha256')):
            g.fail('teacher:row-record-digest:' + name)
    if seen != expected:
        g.absent('teacher:row-coverage:' + name)
    g.require('teacher:row-count:' + name, len(records) == len(expected))


def _grade_teacher(g, evidence, phase, expected_artifact):
    artifact = evidence.get('artifact')
    if type(artifact) is not dict:
        g.absent('teacher:artifact')
        return None
    sha = _canonical_sha(artifact)
    g.require('teacher:artifact-schema', artifact.get('schema') == TEACHER_SCHEMA)
    g.require('teacher:artifact-digest', _digest(sha))
    echo = evidence.get('artifact_sha256')
    if not _digest(echo):
        g.absent('teacher:artifact-sha-echo')
    else:
        g.require('teacher:artifact-sha-echo', echo == sha)
    runs = artifact.get('runs')
    if type(runs) is not dict or set(runs) != set(TEACHER_SEQUENCE):
        g.fail('teacher:sequence-is-not-A1-A2-B-C')
        return sha
    positions = artifact.get('fixed_positions')
    if (type(positions) is not list or len(positions) != MEASURED
            or len(set(positions)) != MEASURED
            or any(type(position) is not int for position in positions)):
        g.absent('teacher:measured-positions')
        positions = list(range(len(positions))) if type(positions) is list else []
    top_basis = artifact.get('top_order_basis')
    if top_basis != rule4.TOP_ORDER_BASIS:
        g.absent('teacher:top-order-basis')
    epochs, generations, storages, repeats = set(), {}, set(), set()
    for name in TEACHER_SEQUENCE:
        run = runs[name]
        if type(run) is not dict:
            g.fail('teacher:run-not-object:' + name)
            continue
        g.require('teacher:route:' + name, run.get('route') == TEACHER_ROUTES[name])
        epoch = run.get('engine_epoch')
        if _int(epoch) and epoch > 0:
            epochs.add(epoch)
        else:
            g.fail('teacher:engine-epoch:' + name)
        generation = run.get('route_generation')
        if _int(generation) and generation >= 0:
            generations[name] = generation
        else:
            g.fail('teacher:route-generation:' + name)
        storage = run.get('storage_epoch')
        if _int(storage) and storage > 0:
            storages.add(storage)
        else:
            g.fail('teacher:storage-epoch:' + name)
        repeat = run.get('repeat')
        if _int(repeat) and repeat >= 0:
            repeats.add(repeat)
        else:
            g.fail('teacher:repeat:' + name)
        g.require('teacher:lifecycle:' + name, run.get('lifecycle_verdict') == 'PASS')
        _grade_row_mapping(g, name, run, positions)
    if len(repeats) != len(TEACHER_SEQUENCE):
        g.fail('teacher:repeat-identities-not-single-sequence')
    if len(epochs) != 1:
        g.fail('teacher:engine-epochs-differ')
    if len(storages) != 1:
        g.fail('teacher:storage-epochs-differ')
    if len(generations) == len(TEACHER_SEQUENCE):
        if not (generations['full-A1'] == generations['full-A2']
                and generations['split-B'] > generations['full-A1']
                and generations['full-C'] != generations['split-B']):
            g.fail('teacher:route-generation-order')
    if evidence.get('retries') not in (0, None):
        g.fail('teacher:retries-present')
    if evidence.get('picked') is True:
        g.fail('teacher:picked')
    if phase == 'rule4_controls':
        g.require('teacher:controls-kind', evidence.get('kind') == 'controls')
        g.require('teacher:one-run', evidence.get('teacher_runs') == 1)
    else:
        g.require('teacher:candidate-kind', evidence.get('kind') == 'candidate')
        g.require('teacher:no-new-run', evidence.get('teacher_runs') == 0)
        g.require('teacher:same-artifact',
                  evidence.get('selected_from_same_artifact') is True)
        g.require('teacher:c-after-c', evidence.get('post_c_before_c') is not True)
        if _digest(expected_artifact):
            g.require('teacher:candidate-artifact-equals-controls',
                      sha == expected_artifact)
        else:
            g.absent('teacher:controls-artifact-sha')
    grade = rule4.grade(artifact)
    g.checks['teacher:rule4-verdict'] = grade.get('verdict')
    g.checks['teacher:rule4-calibration'] = grade.get('calibration_sha256')
    if grade.get('verdict') == 'FAIL':
        for reason in grade.get('failures') or ['rule4-failure']:
            g.fail('teacher:rule4-fail:' + str(reason))
    elif grade.get('verdict') != 'PASS':
        for reason in grade.get('missing') or ['rule4-inconclusive']:
            g.absent('teacher:rule4-inc:' + str(reason))
    return sha


# -- row4: recomposed placement --------------------------------------------
def _rank_rows(value):
    if type(value) is list:
        rows = {}
        for index, row in enumerate(value):
            if type(row) is not dict:
                continue
            rank = row.get('rank') if row.get('rank') in RANKS else index
            if rank in RANKS:
                rows[rank] = row
        return rows
    if type(value) is dict:
        rows = {}
        for key, row in value.items():
            try:
                rank = int(key)
            except (TypeError, ValueError):
                continue
            if rank in RANKS and type(row) is dict:
                rows[rank] = row
        return rows
    return {}


def _rank_keys(value):
    """Normalise a rank keyed map, including the string keys JSON writes."""
    if type(value) is not dict:
        return None
    rows = {}
    for key, row in value.items():
        try:
            rank = int(key)
        except (TypeError, ValueError):
            continue
        if rank in RANKS:
            rows[rank] = row
    return rows


def _grade_row4(g, evidence, expected_teacher, expected_row1, expected_run=None):
    row1_echo = evidence.get('row1_sha256')
    teacher_echo = evidence.get('teacher_artifact_sha256')
    if _digest(row1_echo):
        g.require('row4:row1-echo', True)
    else:
        g.absent('row4:row1-echo')
    if _digest(teacher_echo):
        g.require('row4:teacher-echo', True)
    else:
        g.absent('row4:teacher-echo')
    if _digest(expected_row1):
        g.require('row4:row1-binding',
                  evidence.get('row1_sha256') == expected_row1)
    else:
        g.absent('row4:row1-sha')
    if _digest(expected_teacher):
        g.require('row4:teacher-binding',
                  evidence.get('teacher_artifact_sha256') == expected_teacher)
    else:
        g.absent('row4:teacher-sha')
    raw_rows = _rank_rows(evidence.get('raw_placement'))
    engine_rows = _rank_rows(evidence.get('engine_accounting'))
    for rank in RANKS:
        if rank not in raw_rows:
            g.absent('row4:raw-placement:%d' % rank)
        if rank not in engine_rows:
            g.absent('row4:engine-accounting:%d' % rank)
    supplied_shas = _rank_keys(evidence.get('raw_sha256'))
    if supplied_shas is None:
        g.absent('row4:raw-sha256')
    bindings = {}
    epoch = evidence.get('engine_epoch')
    for rank in RANKS:
        raw = raw_rows.get(rank)
        engine = engine_rows.get(rank)
        if type(raw) is not dict or type(engine) is not dict:
            continue
        label = 'row4:r%d' % rank
        g.require(label + ':raw-schema',
                  raw.get('schema') == engine_offers.PLACEMENT_SCHEMA)
        g.require(label + ':engine-schema', engine.get('schema') == engine_offers.SCHEMA)
        before = copy.deepcopy(raw)
        raw_sha = _canonical_sha(raw)
        if supplied_shas is not None:
            g.require(label + ':raw-sha',
                      supplied_shas.get(rank) == raw_sha)
        if raw.get('verdict') == 'FAIL':
            g.fail(label + ':raw-placement-fail')
        from e3_native_placement_evidence import regrade_packet
        packet = _rank_rows(evidence.get('native_observer_packets')).get(rank)
        if type(packet) is not dict:
            g.absent(label + ':native-observer-packet')
            # Retain identity/refusal diagnostics from historical decode-only
            # evidence, but the absent native packet can never qualify row4.
            composed = engine_offers.compose_placement(raw, engine)
        else:
            g.require(label + ':packet-raw-binding', packet.get('receipt') == raw)
            g.require(label + ':packet-engine-binding', packet.get('engine_accounting') == engine)
            window = evidence.get('window') or {}
            proof = regrade_packet(packet, rank=rank, engine_epoch=epoch,
                run_id=expected_run, window_id=window.get('window_id'),
                route_generation=window.get('route_generation'))
            for reason in proof['failures']:
                g.fail(label + ':' + reason)
            for reason in proof['missing']:
                g.absent(label + ':' + reason)
            composed = proof['composed'] or {}
        if raw != before or _canonical_sha(raw) != raw_sha:
            g.fail(label + ':raw-receipt-changed')
        verdict = composed.get('verdict')
        g.checks[label + ':composed-verdict'] = verdict
        if verdict == 'FAIL':
            g.fail(label + ':composed-fail:' + ','.join(composed.get('reasons') or []))
        elif verdict != 'PASS':
            g.absent(label + ':composed-inc:' + ','.join(composed.get('reasons') or []))
        binding = composed.get('binding')
        if type(binding) is not dict:
            g.absent(label + ':binding')
            continue
        bindings[rank] = binding
        g.require(label + ':binding-rank', binding.get('rank') == rank)
        if expected_run is not None:
            g.require(label + ':binding-run', binding.get('run_id') == expected_run)
        g.require(label + ':binding-incarnation',
                  _text(binding.get('worker_incarnation')))
        g.require(label + ':binding-engine-incarnation',
                  binding.get('engine_incarnation') == 'engine:' + str(epoch))
        g.require(label + ':binding-engine-epoch',
                  binding.get('engine_epoch') == epoch)
        g.require(label + ':binding-generation',
                  _int(binding.get('route_generation'))
                  and binding['route_generation'] >= 1)
        g.require(label + ':binding-storage-epoch',
                  _int(binding.get('storage_epoch'))
                  and binding['storage_epoch'] >= 1)
        start, end = binding.get('native_start_ns'), binding.get('native_end_ns')
        g.require(label + ':binding-window-order',
                  _int(start) and _int(end) and end > start)
    if set(bindings) == set(RANKS):
        windows = {(row.get('run_id'), row.get('window_id'),
                    row.get('engine_epoch'), row.get('route_generation'),
                    row.get('storage_epoch')) for row in bindings.values()}
        # CUPTI begin/end timestamps are independently queried on each rank.
        # Bind run/window/generation collectively and raw/native endpoints
        # locally; equality of four independently taken clock samples is not
        # an ownership proof and is never an achievable real-engine gate.
        g.require('row4:same-window', len(windows) == 1)
        incarnations = {row.get('worker_incarnation') for row in bindings.values()}
        g.require('row4:distinct-incarnations', len(incarnations) == len(RANKS))
    # The per-rank actual report reads were validated and digest-bound above.
    # A 64MiB file read or a cached native-read flag is not a substitute.
    g.require('row4:shipping-instrument-disabled',
              evidence.get('shipping_path_instrument_disabled') is True)


def grade_evidence(phase, evidence, specification, expected_artifact=None,
                   expected_row1=None):
    """Grade one raw phase record; no cached verdict field is trusted."""
    g = Grade(phase)
    if type(evidence) is not dict:
        g.absent(phase + ':evidence')
        return g
    if evidence.get('phase') != phase:
        g.fail(phase + ':phase-differs')
    epoch = evidence.get('engine_epoch')
    if not (_int(epoch) and epoch > 0):
        g.absent(phase + ':engine-epoch')
    expected_epoch = specification.get('engine_epoch')
    if _int(expected_epoch) and epoch != expected_epoch:
        g.fail(phase + ':engine-epoch-differs')
    settings = evidence.get('model_settings')
    if settings is not None and type(settings) is not dict:
        g.fail(phase + ':model-settings-not-object')
    if evidence.get('model_settings_unchanged') is not True:
        g.fail(phase + ':model-settings-unchanged')
    if type(settings) is dict:
        for key in MODEL_SETTING_KEYS:
            if settings.get(key + '_changed') is True:
                g.fail(phase + ':model-setting-changed:' + key)
    if phase == 'row1':
        _grade_row1(g, evidence, run_id=specification.get('run_id'),
                    engine_epoch=specification.get('engine_epoch'))
    elif phase in ('rule4_controls', 'rule4_candidate'):
        _grade_teacher(g, evidence, phase, expected_artifact)
    else:
        _grade_row4(g, evidence, expected_artifact, expected_row1,
                    specification.get('run_id'))
    return g


def _write_record(path, record):
    out = os.path.abspath(path)
    if os.path.exists(out):
        raise SystemExit('refusing to overwrite existing qualification artifact')
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as handle:
        json.dump(record, handle, sort_keys=True, indent=1)
        handle.write('\n')


def _workload_object(value):
    """Accept a queue manifest object (or a validated Workload) exactly."""
    if value is None:
        return None
    from e3_decode_correctness import Workload, validate_manifest
    if isinstance(value, Workload):
        return value
    problems, workload, _sha = validate_manifest(value)
    if problems or workload is None:
        raise ValueError('workload-manifest-invalid:' + ','.join(problems[:4]))
    return workload


def _phase_grading_evidence(phase, collected, engine_epoch):
    """Project the driver's actual return into the pure grader's input.

    This adds no observation: teacher phases are graded from the *retained*
    artifact and the driver's own flags; row1/row4 are graded from the raw
    evidence the driver returned.  A cached or server-supplied verdict is never
    read.
    """
    if phase in ('rule4_controls', 'rule4_candidate'):
        artifact = collected.get('artifact')
        evidence = {
            'phase': phase, 'engine_epoch': engine_epoch,
            'model_settings_unchanged': True,
            'artifact': artifact,
            'artifact_sha256': collected.get('artifact_sha256'),
            'teacher_runs': 1 if phase == 'rule4_controls' else 0,
            'retries': 0, 'picked': False,
        }
        if phase == 'rule4_controls':
            evidence['kind'] = 'controls'
        else:
            evidence.update({'kind': 'candidate',
                             'selected_from_same_artifact': True,
                             'post_c_before_c': False})
        return evidence
    evidence = collected.get('evidence')
    return evidence


def run(phase, url, model, run_id, epoch, out, artifact_sha256=None,
        evidence_path=None, env=None, row1_sha256=None, *, store=None,
        teacher_baseline=None, teacher_workload=None, teacher_instrument=None,
        request_plan=None, prefill_plan=None, placement=None,
        placement_workload=None, native_read_provider=None,
        native_prefill_provider=None, native_source_provider=None,
        window_id=None, key=None, timeout_s=None,
        snapshot_max_bytes=2*1024**3):
    """Collect one phase over the actual native transport, then grade it.

    ``evidence_path`` remains the offline pure-grading path.  Without it the
    local ``QualificationDriver`` composes the reviewed source modules over
    authenticated loopback; no phase command is sent to the decode-probe RPC
    endpoint, which has no phase handler.  The phase artifact is retained once
    by the driver's immutable store and this process writes only its own
    qualification record.
    """
    env = os.environ if env is None else env
    started = time.monotonic_ns()
    specification = _specification(phase, model, run_id, artifact_sha256,
                                   row1_sha256)
    specification['engine_epoch'] = epoch
    record = {
        'schema': 1, 'phase': phase, 'run_id': run_id, 'model': model,
        'started_monotonic_ns': started, 'target_label': 's5-overlap-80-80',
        'scope': 'qualification phase; not performance and not armability',
        'artifact_sha256': artifact_sha256, 'row1_sha256': row1_sha256,
    }
    if phase == 'rule4_candidate' and not _digest(artifact_sha256):
        record.update({'verdict': 'INCONCLUSIVE',
                       'missing': ['missing-retained-teacher-artifact-sha'],
                       'failures': [], 'evidence': None})
        _write_record(out, record)
        return 2
    if phase == 'row4' and not (_digest(artifact_sha256) and _digest(row1_sha256)):
        record.update({'verdict': 'INCONCLUSIVE',
                       'missing': ['missing-retained-row1-or-teacher-sha'],
                       'failures': [], 'evidence': None})
        _write_record(out, record)
        return 2
    if evidence_path:
        try:
            with open(evidence_path) as handle:
                grading_input = json.load(handle)
        except (OSError, ValueError) as exc:
            record.update({'verdict': 'INCONCLUSIVE',
                           'missing': ['qualification-evidence-unreadable:'
                                       + type(exc).__name__],
                           'failures': []})
            _write_record(out, record)
            return 2
        if type(grading_input) is not dict:
            record.update({'verdict': 'INCONCLUSIVE',
                           'missing': ['evidence-not-object'], 'failures': []})
            _write_record(out, record)
            return 2
        evidence = grading_input.get('evidence')
        record['response'] = grading_input
        if type(evidence) is dict:
            if phase == 'row1' and type(evidence.get('workers')) is list:
                record['row1_sha256'] = _canonical_sha(evidence['workers'])
            elif phase in ('rule4_controls', 'rule4_candidate') and type(
                    evidence.get('artifact')) is dict:
                record['teacher_artifact_sha256'] = _canonical_sha(
                    evidence['artifact'])
        g = grade_evidence(phase, evidence, specification, artifact_sha256,
                           row1_sha256)
    else:
        if not _int(epoch) or epoch <= 0:
            record.update({'verdict': 'INCONCLUSIVE',
                           'missing': ['engine-epoch-not-bound'],
                           'failures': [], 'evidence': None})
            _write_record(out, record)
            return 2
        if not _text(store):
            record.update({'verdict': 'INCONCLUSIVE',
                           'missing': ['missing-qualification-store'],
                           'failures': [], 'evidence': None})
            _write_record(out, record)
            return 2
        try:
            from e3_native_qualification_driver import (
                NativePlacementRunner, PhaseConflict, PhaseUnavailable,
                QualificationDriver, QualificationRefused)
        except ImportError as exc:
            record.update({'verdict': 'INCONCLUSIVE',
                           'missing': ['qualification-module-unavailable:'
                                       + type(exc).__name__],
                           'failures': [], 'evidence': None})
            _write_record(out, record)
            return 2
        try:
            workload = _workload_object(teacher_workload)
            placement_workload_object = _workload_object(placement_workload)
            if (placement is None and phase == 'row4'
                    and placement_workload_object is not None):
                placement = NativePlacementRunner(
                    workload=placement_workload_object, url=url,
                    prefill_plan=prefill_plan,snapshot_max_bytes=snapshot_max_bytes)
            driver = QualificationDriver(
                store, run_id=run_id, model=model, engine_epoch=epoch,
                url=url, key=key if key is not None else env.get('VLLM_API_KEY'),
                teacher_baseline=teacher_baseline,
                teacher_workload=workload,
                teacher_instrument=teacher_instrument,
                snapshot_max_bytes=snapshot_max_bytes,
                placement=placement,
                native_read_provider=native_read_provider,
                native_prefill_provider=native_prefill_provider,
                native_source_provider=native_source_provider,
                timeout_s=(timeout_s if timeout_s is not None
                           else float(env.get('Q18A_QUAL_TIMEOUT_S', '900'))))
            kwargs = {}
            if phase == 'row1':
                kwargs['request_plan'] = request_plan
            elif phase == 'rule4_candidate':
                kwargs['artifact_sha256'] = artifact_sha256
            elif phase == 'row4':
                kwargs.update({'artifact_sha256': artifact_sha256,
                               'row1_sha256': row1_sha256,
                               'window_id': window_id})
            collected = driver.collect_phase(phase, **kwargs)
        except (QualificationRefused, PhaseUnavailable, PhaseConflict,
                ValueError) as exc:
            code = getattr(exc, 'code', None) or str(exc)[:80] or type(exc).__name__
            record.update({
                'verdict': 'INCONCLUSIVE',
                'missing': ['collect:' + str(code)],
                'failures': [],
                'refusal_evidence': copy.deepcopy(getattr(exc, 'evidence', None)),
                'refusal_cleanup': copy.deepcopy(getattr(exc, 'cleanup', None)),
                'finished_monotonic_ns': time.monotonic_ns(),
            })
            _write_record(out, record)
            return 2
        evidence = _phase_grading_evidence(phase, collected, epoch)
        record['phase_record'] = collected.get('record')
        if phase == 'row1':
            record['row1_sha256'] = collected.get('row1_sha256')
        elif phase in ('rule4_controls', 'rule4_candidate'):
            record['teacher_artifact_sha256'] = collected.get('artifact_sha256')
        record['driver_missing'] = list(collected.get('missing') or [])
        g = grade_evidence(phase, evidence, specification, artifact_sha256,
                           row1_sha256)
        for reason in record['driver_missing']:
            g.absent('collector:' + str(reason))
        if collected.get('complete') is not True:
            g.absent('collector:phase-incomplete')
    record['evidence'] = evidence
    verdict = g.verdict()
    record.update({
        'verdict': verdict, 'failures': g.failures, 'missing': g.missing,
        'checks': g.checks, 'specification_sha256': _canonical_sha(specification),
        'finished_monotonic_ns': time.monotonic_ns(),
    })
    _write_record(out, record)
    return {'PASS': 0, 'FAIL': 1}.get(verdict, 2)


def _json_arg(path):
    if path is None:
        return None
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError) as exc:
        raise ValueError('plan-unreadable:' + type(exc).__name__) from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=(__doc__ or '').splitlines()[0])
    parser.add_argument('--phase', required=True, choices=PHASES)
    parser.add_argument('--url', default=os.environ.get('PROBE_URL',
                                                       'http://127.0.0.1:8000'))
    parser.add_argument('--model', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--artifact-sha256',
                        default=os.environ.get('Q18A_TEACHER_ARTIFACT_SHA'))
    parser.add_argument('--row1-sha256',
                        default=os.environ.get('Q18A_ROW1_SHA'))
    parser.add_argument('--evidence', default=os.environ.get('Q18A_QUAL_EVIDENCE'))
    parser.add_argument('--out', required=True)
    # Queue-owned collection inputs.  Without ``--evidence`` these define the
    # actual phase producer; nothing here falls back to a built-in workload.
    parser.add_argument('--store', default=os.environ.get('Q18A_STORE'))
    parser.add_argument('--request-plan',
                        default=os.environ.get('Q18A_ROW1_REQUEST_PLAN'))
    parser.add_argument('--workload', default=os.environ.get('Q18A_TEACHER_WORKLOAD'))
    parser.add_argument('--placement-workload',
                        default=os.environ.get('Q18A_PLACEMENT_WORKLOAD'))
    parser.add_argument('--prefill-plan',
                        default=os.environ.get('Q18A_PREFILL_PLAN'))
    parser.add_argument('--teacher-baseline',
                        default=os.environ.get('Q18A_TEACHER_BASELINE'))
    parser.add_argument('--window-id', default=os.environ.get('Q18A_WINDOW_ID'))
    parser.add_argument('--timeout-s', type=float, default=None)
    parser.add_argument('--snapshot-max-bytes',type=int,default=2*1024**3)
    args = parser.parse_args(argv)
    try:
        request_plan = _json_arg(args.request_plan)
        teacher_workload = _json_arg(args.workload)
        placement_workload = _json_arg(args.placement_workload)
        prefill_plan = _json_arg(args.prefill_plan)
    except ValueError as exc:
        record = {'schema': 1, 'phase': args.phase, 'run_id': args.run_id,
                  'model': args.model, 'verdict': 'INCONCLUSIVE',
                  'missing': [str(exc)], 'failures': [], 'evidence': None,
                  'started_monotonic_ns': time.monotonic_ns()}
        _write_record(args.out, record)
        return 2
    return run(args.phase, args.url, args.model, args.run_id, args.epoch,
               args.out, args.artifact_sha256, args.evidence,
               row1_sha256=args.row1_sha256,
               store=args.store, teacher_baseline=args.teacher_baseline,
               teacher_workload=teacher_workload, request_plan=request_plan,
               prefill_plan=prefill_plan,
               placement_workload=placement_workload,
               window_id=args.window_id, timeout_s=args.timeout_s,
               snapshot_max_bytes=args.snapshot_max_bytes)


if __name__ == '__main__':
    sys.exit(main())
