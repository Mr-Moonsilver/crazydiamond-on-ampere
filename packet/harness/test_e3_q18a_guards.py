"""CPU-only guards for the composed q18a qualification driver and graders.

Nothing here talks to an engine, a GPU or the network.  Row1 carries the
actual flat ``s5-native-row1-store/1`` disabled packets and is graded by
``e3_native_row1_grade`` through q18a; rule4 reuses the reviewed schema/2
teacher fixture and the frozen v4 policy; row4 composes the reviewed
``test_e3_engine_offers`` ledger receipts through
``engine_offers.compose_placement``.

The negative cases require FAIL-over-missing, no empty successful oracle, and
no cached/server-supplied verdict.  Startup and lifecycle grading live with
their owning work packages in this tree, not in this file.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))
sys.path.insert(0, str(ROOT / 'harness'))

import engine_offers                                          # noqa: E402
import row1_oracle as row1_oracle                             # noqa: E402
import e3_native_qualification_driver as driver_module        # noqa: E402
import e3_q18a_qualification as qualification                 # noqa: E402
import test_e3_engine_offers as offers                        # noqa: E402
import test_e3_teacher_decode_rule4 as rules                  # noqa: E402

RUN = 'q18a-run'
MODEL = 'fake-model'
PACKAGE_DIR = ROOT / 'patches/s5_engine'
EPOCH = 3
STORAGE = 2
RANKS = (0, 1, 2, 3)
ROW4_EPOCH = 3
TEACHER_SPEC = {'engine_epoch': EPOCH}
ROW1_SPEC = {'engine_epoch': EPOCH, 'run_id': RUN}
ROW4_SPEC = {'engine_epoch': ROW4_EPOCH}


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True))
    return str(path)


def source_sha(name):
    return hashlib.sha256((PACKAGE_DIR / (name + '.py')).read_bytes()).hexdigest()


def response(phase, evidence):
    return {'schema': 1, 'action': 'qualification', 'run_id': RUN,
            'phase': phase, 'evidence': evidence}


# ---------------------------------------------------------------------------
# row1: the actual flat ``s5-native-row1-store/1`` disabled packets
#
# The packets use the source-owned dataclass fields (identity, digest
# envelopes) and disabled route generation.  They are graded by the actual
# ``e3_native_row1_grade`` through q18a; no nested comparison/source-hold
# schema and no peer_inputs copy is imported anywhere.
# ---------------------------------------------------------------------------
ROW1_GENERATION = 2
ROW1_LAYERS = ('model.layers.0.self_attn', 'model.layers.1.self_attn')


def row1_packet(rank, *, enabled=False, generation=ROW1_GENERATION,
                disabled_generation=ROW1_GENERATION + 1):
    # Actual Row1Qualifier comparisons across dense/sparse operation epochs,
    # rather than route marginals on two different layers in one epoch.
    import test_e3_native_row1_grade as producer
    packet = copy.deepcopy(producer.evidence()['workers'][rank])
    packet.update(run_id=RUN, route_generation=generation,
                  disabled_route_generation=disabled_generation, enabled=enabled)
    packet['raw']['enabled'] = enabled
    return packet


def row1_evidence(workers=None):
    return {
        'phase': 'row1', 'engine_epoch': EPOCH,
        'model_settings_unchanged': True,
        'workers': ([row1_packet(rank) for rank in RANKS]
                    if workers is None else workers),
    }


def grade_row1(evidence, expected_row1=None):
    return qualification.grade_evidence('row1', evidence, ROW1_SPEC,
                                        expected_row1=expected_row1)


class Row1Tests(unittest.TestCase):
    def test_flat_disabled_packets_are_graded_by_the_actual_source_grader(self):
        grade = grade_row1(row1_evidence())
        self.assertEqual(grade.verdict(), 'PASS', (grade.failures, grade.missing))
        for rank in RANKS:
            self.assertTrue(grade.checks['row1:rank%d:schema' % rank])
            self.assertTrue(grade.checks['row1:rank%d:disabled' % rank])
            self.assertTrue(grade.checks['row1:rank%d:record_digest' % rank])

    def test_empty_workers_never_pass(self):
        grade = grade_row1(row1_evidence(workers=[]))
        self.assertEqual(grade.verdict(), 'INCONCLUSIVE')
        self.assertIn('row1:worker_packets', grade.missing)
        self.assertIn('row1:rank_coverage', grade.missing)

    def test_tampered_raw_digest_is_a_failure(self):
        evidence = copy.deepcopy(row1_evidence())
        evidence['workers'][0]['raw']['records'][0]['digest'] = '0' * 64
        grade = grade_row1(evidence)
        self.assertEqual(grade.verdict(), 'FAIL')
        self.assertIn('row1:rank0:record_digest', grade.failures)

    def test_oracle_enabled_packet_is_a_failure(self):
        evidence = copy.deepcopy(row1_evidence())
        for packet in evidence['workers']:
            packet['enabled'] = True
            packet['raw']['enabled'] = True
        grade = grade_row1(evidence)
        self.assertEqual(grade.verdict(), 'FAIL')
        self.assertIn('row1:rank0:disabled', grade.failures)
        self.assertIn('row1:rank0:raw_disabled', grade.failures)

    def test_missing_rank_is_inconclusive(self):
        evidence = row1_evidence(workers=[row1_packet(rank)
                                          for rank in (0, 1, 2)])
        grade = grade_row1(evidence)
        self.assertEqual(grade.verdict(), 'INCONCLUSIVE')
        self.assertIn('row1:rank_coverage', grade.missing)

    def test_wrong_run_or_disabled_generation_is_refused(self):
        evidence = copy.deepcopy(row1_evidence())
        evidence['workers'][1]['run_id'] = 'other-run'
        grade = grade_row1(evidence)
        self.assertEqual(grade.verdict(), 'FAIL')
        self.assertIn('row1:rank1:run', grade.failures)

        evidence = copy.deepcopy(row1_evidence())
        evidence['workers'][2]['disabled_route_generation'] = 1
        grade = grade_row1(evidence)
        self.assertEqual(grade.verdict(), 'FAIL')
        self.assertIn('row1:rank2:disable_generation', grade.failures)

    def test_vm_receipt_is_never_engine_evidence(self):
        evidence = row1_evidence(workers=[])
        evidence['vm_receipt_used_as_engine_evidence'] = True
        evidence['vm_receipt'] = {'schema': 'vm-job12c/1', 'verdict': 'PASS'}
        grade = grade_row1(evidence)
        self.assertEqual(grade.verdict(), 'FAIL')
        self.assertIn('row1:vm_receipt_as_engine_evidence', grade.failures)
        self.assertNotEqual(grade.verdict(), 'PASS')

    def test_cached_verdict_is_never_trusted(self):
        evidence = row1_evidence(workers=[])
        evidence['verdict'] = 'PASS'
        evidence['grade'] = {'verdict': 'PASS'}
        grade = grade_row1(evidence)
        self.assertEqual(grade.verdict(), 'INCONCLUSIVE')
        self.assertIn('row1:worker_packets', grade.missing)


# ---------------------------------------------------------------------------
# rule4: ONE A1/A2/B/C sequence over the frozen v4 policy
# ---------------------------------------------------------------------------
def teacher_artifact():
    artifact = rules.valid_evidence()
    rules.controls_equal(artifact)
    return artifact


def teacher_evidence(phase, artifact, **overrides):
    sha = canonical(artifact)
    evidence = {'phase': phase, 'engine_epoch': EPOCH,
                'model_settings_unchanged': True, 'artifact': artifact,
                'artifact_sha256': sha}
    if phase == 'rule4_controls':
        evidence.update(kind='controls', teacher_runs=1)
    else:
        evidence.update(kind='candidate', teacher_runs=0,
                        selected_from_same_artifact=True,
                        post_c_before_c=False)
    evidence.update(overrides)
    return evidence


def grade_teacher(phase, evidence, expected=None):
    return qualification.grade_evidence(phase, evidence, TEACHER_SPEC,
                                        expected_artifact=expected)


class TeacherRule4Tests(unittest.TestCase):
    def test_canonical_sequence_goes_through_frozen_rule4(self):
        artifact = teacher_artifact()
        self.assertEqual(artifact['schema'], 's5-teacher-decode/2')
        self.assertEqual(set(artifact['runs']),
                         set(qualification.TEACHER_SEQUENCE))
        grade = grade_teacher('rule4_controls',
                              teacher_evidence('rule4_controls', artifact))
        self.assertEqual(grade.verdict(), 'PASS',
                         (grade.failures, grade.missing))
        self.assertEqual(grade.checks['teacher:rule4-verdict'], 'PASS')

    def test_g_false_pass_cached_fail_verdict_is_ignored(self):
        """The delivered fixture carried a FAIL verdict yet graded PASS."""
        artifact = teacher_artifact()
        evidence = teacher_evidence('rule4_controls', artifact)
        evidence['rule4_grade'] = {
            'verdict': 'FAIL',
            'failures': ['candidate-outside-control-envelope']}
        grade = grade_teacher('rule4_controls', evidence)
        self.assertEqual(grade.verdict(), 'PASS',
                         (grade.failures, grade.missing))
        self.assertEqual(grade.checks['teacher:rule4-verdict'], 'PASS')

    def test_candidate_numeric_mutation_is_a_failure(self):
        artifact = teacher_artifact()
        values = dict(rules.VALUES)
        values['token_id:19'] = -4.8
        rules.set_all(artifact, 'split-B', values)
        rules.bind(artifact)
        evidence = teacher_evidence('rule4_controls', artifact)
        evidence['rule4_grade'] = {'verdict': 'PASS', 'failures': []}
        grade = grade_teacher('rule4_controls', evidence)
        self.assertEqual(grade.verdict(), 'FAIL')
        self.assertTrue(any(reason.startswith('teacher:rule4-fail:')
                            for reason in grade.failures), grade.failures)
        self.assertEqual(grade.checks['teacher:rule4-verdict'], 'FAIL')

    def test_control_instability_stays_inconclusive(self):
        artifact = teacher_artifact()
        rules.set_all(artifact, 'full-C', rules.shifted(-1.3))
        rules.bind(artifact)
        grade = grade_teacher('rule4_controls',
                              teacher_evidence('rule4_controls', artifact))
        self.assertEqual(grade.verdict(), 'INCONCLUSIVE',
                         (grade.failures, grade.missing))
        self.assertEqual(grade.checks['teacher:rule4-verdict'], 'INCONCLUSIVE')
        self.assertTrue(any(reason.startswith('teacher:rule4-inc:')
                            for reason in grade.missing), grade.missing)

    def test_unavailable_control_metrics_stay_inconclusive(self):
        artifact = teacher_artifact()
        artifact['runs']['full-A2']['positions'][0].pop('top', None)
        grade = grade_teacher('rule4_controls',
                              teacher_evidence('rule4_controls', artifact))
        self.assertEqual(grade.verdict(), 'INCONCLUSIVE',
                         (grade.failures, grade.missing))
        self.assertEqual(grade.checks['teacher:rule4-verdict'], 'INCONCLUSIVE')

    def test_candidate_must_echo_the_controls_artifact(self):
        artifact = teacher_artifact()
        controls_sha = canonical(artifact)
        other = teacher_artifact()
        other['fixed_positions'] = other['fixed_positions'][:-1] + [999]
        rules.bind(other)
        evidence = teacher_evidence('rule4_candidate', other)
        grade = grade_teacher('rule4_candidate', evidence, controls_sha)
        self.assertEqual(grade.verdict(), 'FAIL')
        self.assertIn('teacher:candidate-artifact-equals-controls',
                      grade.failures)

    def test_candidate_cannot_rerun_the_sequence(self):
        artifact = teacher_artifact()
        evidence = teacher_evidence('rule4_candidate', artifact,
                                    teacher_runs=1, post_c_before_c=True)
        grade = grade_teacher('rule4_candidate', evidence)
        self.assertEqual(grade.verdict(), 'FAIL')
        self.assertIn('teacher:no-new-run', grade.failures)
        self.assertIn('teacher:c-after-c', grade.failures)

    def test_sequence_must_be_exactly_a1_a2_b_c(self):
        artifact = teacher_artifact()
        artifact['runs'].pop('full-C')
        grade = grade_teacher('rule4_controls',
                              teacher_evidence('rule4_controls', artifact))
        self.assertEqual(grade.verdict(), 'FAIL')
        self.assertIn('teacher:sequence-is-not-A1-A2-B-C', grade.failures)

    def test_run_mapping_requires_all_8x256_rows(self):
        artifact = teacher_artifact()
        artifact['runs']['split-B']['row_mapping']['records'].pop()
        grade = grade_teacher('rule4_controls',
                              teacher_evidence('rule4_controls', artifact))
        self.assertEqual(grade.verdict(), 'FAIL')
        self.assertIn('teacher:row-count:split-B', grade.failures)
        self.assertIn('teacher:row-coverage:split-B', grade.missing)

    def test_row_mapping_digest_defect_fails(self):
        artifact = teacher_artifact()
        artifact['runs']['split-B']['row_mapping']['records'][0][
            'logprob_sha256'] = 'not-a-digest'
        grade = grade_teacher('rule4_controls',
                              teacher_evidence('rule4_controls', artifact))
        self.assertEqual(grade.verdict(), 'FAIL')
        self.assertIn('teacher:row-record-digest:split-B', grade.failures)

    def test_row_record_bound_to_another_run_fails(self):
        artifact = teacher_artifact()
        artifact['runs']['split-B']['row_mapping']['records'][0]['run'] = 'full-A1'
        grade = grade_teacher('rule4_controls',
                              teacher_evidence('rule4_controls', artifact))
        self.assertEqual(grade.verdict(), 'FAIL')
        self.assertIn('teacher:row-record-run:split-B', grade.failures)

    def test_wrong_generation_order_fails(self):
        artifact = teacher_artifact()
        artifact['runs']['split-B']['route_generation'] = 0
        grade = grade_teacher('rule4_controls',
                              teacher_evidence('rule4_controls', artifact))
        self.assertEqual(grade.verdict(), 'FAIL')
        self.assertIn('teacher:route-generation-order', grade.failures)

    def test_retries_and_picking_fail(self):
        artifact = teacher_artifact()
        evidence = teacher_evidence('rule4_controls', artifact,
                                    retries=1, picked=True)
        grade = grade_teacher('rule4_controls', evidence)
        self.assertEqual(grade.verdict(), 'FAIL')
        self.assertIn('teacher:retries-present', grade.failures)
        self.assertIn('teacher:picked', grade.failures)

    def test_token_only_artifact_is_not_a_numeric_grade(self):
        artifact = teacher_artifact()
        for run in artifact['runs'].values():
            for row in run['positions']:
                row.pop('top', None)
        grade = grade_teacher('rule4_controls',
                              teacher_evidence('rule4_controls', artifact))
        self.assertNotEqual(grade.verdict(), 'PASS',
                            (grade.failures, grade.missing))
        self.assertIn(grade.checks['teacher:rule4-verdict'],
                      ('FAIL', 'INCONCLUSIVE'))

    def test_all_runs_must_bind_one_engine_epoch(self):
        artifact = teacher_artifact()
        artifact['runs']['full-C']['engine_epoch'] = EPOCH + 1
        grade = grade_teacher('rule4_controls',
                              teacher_evidence('rule4_controls', artifact))
        self.assertEqual(grade.verdict(), 'FAIL')
        self.assertIn('teacher:engine-epochs-differ', grade.failures)

    def test_missing_artifact_is_inconclusive(self):
        grade = grade_teacher('rule4_controls',
                              {'phase': 'rule4_controls',
                               'engine_epoch': EPOCH,
                               'model_settings_unchanged': True})
        self.assertEqual(grade.verdict(), 'INCONCLUSIVE')
        self.assertIn('teacher:artifact', grade.missing)


# ---------------------------------------------------------------------------
# row4: raw ReplayObserver receipts + validated engine offer accounting
# ---------------------------------------------------------------------------
def row4_material(*, include_eager=False, verdict='PASS',
                  native_reasons=None, expected=None):
    _, bridge, ledger, _ = offers.positive(include_eager=include_eager)
    engine = offers.grade(ledger, bridge)
    raw = offers.raw_receipt(engine, expected=expected, verdict=verdict,
                             native_reasons=native_reasons)
    return raw, engine


def four_ranks(raw, engine):
    raws, engines = [], []
    for rank in range(4):
        accounting = copy.deepcopy(engine)
        binding = accounting['binding']
        binding.update(run_id=RUN, rank=rank, worker='w%d' % rank,
                       worker_incarnation='worker-inc-%d' % rank,
                       engine_incarnation='engine:%d' % ROW4_EPOCH)
        accounting['window']['binding'] = copy.deepcopy(binding)
        accounting['window'].update(run_id=RUN, rank=rank, worker='w%d' % rank)
        receipt = copy.deepcopy(raw)
        proof = receipt['caller_proof']
        proof.update(worker_rank=rank,
                     worker_incarnation='worker-inc-%d' % rank,
                     engine_incarnation='engine:%d' % ROW4_EPOCH)
        raws.append(receipt)
        engines.append(accounting)
    return raws, engines


def row4_evidence(raws, engines, *, row1_sha=None, teacher_sha=None,
                  string_keys=False, **overrides):
    row1_sha = row1_sha or digest('row1-artifact')
    teacher_sha = teacher_sha or digest('teacher-artifact')
    raw_sha = {}
    for rank, raw in enumerate(raws):
        raw_sha[str(rank) if string_keys else rank] = canonical(raw)
    evidence = {
        'phase': 'row4', 'engine_epoch': ROW4_EPOCH,
        'model_settings_unchanged': True,
        'row1_sha256': row1_sha, 'teacher_artifact_sha256': teacher_sha,
        'raw_placement': raws, 'engine_accounting': engines,
        'raw_sha256': raw_sha,
        'native_read': {'bytes': qualification.NATIVE_READ_BYTES,
                        'basis': 'actual-native-read', 'completed': True,
                        'buffer_sha256': digest('native-read'),
                        'source': 'retained-split-bank'},
        'shipping_path_instrument_disabled': True,
    }
    evidence.update(overrides)
    return evidence


def grade_row4(evidence, teacher_sha=None, row1_sha=None):
    return qualification.grade_evidence(
        'row4', evidence, ROW4_SPEC,
        expected_artifact=teacher_sha or evidence.get('teacher_artifact_sha256'),
        expected_row1=row1_sha or evidence.get('row1_sha256'))


class Row4Tests(unittest.TestCase):
    def tearDown(self):
        if engine_offers._ACTIVE is not None:
            engine_offers._ACTIVE.deactivate()
        engine_offers._CENSUS = {}

    def test_independent_rank_timestamps_keep_semantic_window_binding(self):
        raws, engines = [], []
        start, end = offers.START, offers.END
        for rank in range(4):
            # Build and independently grade each native/ledger pair with actual
            # per-rank bounds; changing a sealed digest afterward is not enough.
            with patch.object(offers, 'START', start + rank), patch.object(offers, 'END', end + rank):
                raw, engine = row4_material()
                rank_raws, rank_engines = four_ranks(raw, engine)
                raws.append(rank_raws[rank])
                engines.append(rank_engines[rank])
        grade = grade_row4(row4_evidence(raws, engines))
        self.assertEqual(grade.verdict(), 'INCONCLUSIVE', (grade.failures, grade.missing))
        engines[3]['binding']['window_id'] = 'different-window'
        self.assertNotEqual(grade_row4(row4_evidence(raws, engines)).verdict(), 'PASS')

    def test_valid_receipts_from_another_run_do_not_qualify_this_command(self):
        raws, engines = four_ranks(*row4_material())
        evidence = row4_evidence(raws, engines)
        grade = qualification.grade_evidence('row4', evidence,
            dict(ROW4_SPEC, run_id='other-run'),
            expected_artifact=evidence['teacher_artifact_sha256'],
            expected_row1=evidence['row1_sha256'])
        self.assertEqual(grade.verdict(), 'FAIL')
        self.assertIn('row4:r0:binding-run', grade.failures)

    def test_legacy_decode_only_cannot_qualify_native_placement(self):
        raws, engines = four_ranks(*row4_material())
        grade = grade_row4(row4_evidence(raws, engines))
        self.assertEqual(grade.verdict(), 'INCONCLUSIVE',
                         (grade.failures, grade.missing))
        self.assertEqual(grade.checks['row4:r0:composed-verdict'], 'PASS')

    def test_json_string_raw_sha_keys_are_accepted(self):
        raws, engines = four_ranks(*row4_material())
        grade = grade_row4(row4_evidence(raws, engines, string_keys=True))
        self.assertEqual(grade.verdict(), 'INCONCLUSIVE',
                         (grade.failures, grade.missing))

    def test_g_false_pass_raw_fail_and_engine_fail_are_ignored(self):
        """The delivered row4 fixture supplied a composed PASS beside FAILs."""
        raw, engine = row4_material(verdict='FAIL',
                                    include_eager=True,
                                    native_reasons=['wrong_context'])
        engine['verdict'] = 'FAIL'
        engine['failures'] = ['unoffered_eager_activity']
        raws, engines = four_ranks(raw, engine)
        evidence = row4_evidence(
            raws, engines,
            composed_placement={'schema': 's5-placement-composed/1',
                                'verdict': 'PASS'},
            placement=True, native_accounting=True)
        grade = grade_row4(evidence)
        self.assertEqual(grade.verdict(), 'FAIL')
        self.assertIn('row4:r0:raw-placement-fail', grade.failures)
        self.assertTrue(any('row4:r0:composed-fail:raw_placement_fail' == reason
                            for reason in grade.failures), grade.failures)

    def test_engine_offer_failure_is_never_discharged(self):
        raw, engine = row4_material()
        engine['verdict'] = 'FAIL'
        engine['failures'] = ['unoffered_eager_activity']
        raws, engines = four_ranks(raw, engine)
        grade = grade_row4(row4_evidence(raws, engines))
        self.assertEqual(grade.verdict(), 'FAIL')
        self.assertTrue(any('unoffered_eager_activity' in reason
                            for reason in grade.failures), grade.failures)

    def test_raw_inc_is_discharged_only_by_a_complete_join(self):
        raw, engine = row4_material(
            include_eager=True, verdict='INCONCLUSIVE',
            native_reasons=['unattributed_expected_context'],
            expected=None)
        raw['replay']['counts']['unattributed_expected_context'] = (
            engine['eager_green_records'])
        # The raw receipt and the engine partition must agree after composition.
        composed = engine_offers.compose_placement(copy.deepcopy(raw), engine)
        self.assertEqual(composed['verdict'], 'PASS', composed['reasons'])
        self.assertEqual(composed['discharge']['origin'],
                         'discharged_eager_expected_context')
        raws, engines = four_ranks(raw, engine)
        grade = grade_row4(row4_evidence(raws, engines))
        self.assertEqual(grade.verdict(), 'INCONCLUSIVE',
                         (grade.failures, grade.missing))

    def test_raw_inc_without_native_accounting_stays_inc(self):
        raw, engine = row4_material(
            include_eager=False, verdict='INCONCLUSIVE',
            native_reasons=['unattributed_expected_context'], expected=13)
        raws, engines = four_ranks(raw, engine)
        evidence = row4_evidence(raws, engines, native_accounting=True)
        grade = grade_row4(evidence)
        self.assertEqual(grade.verdict(), 'INCONCLUSIVE',
                         (grade.failures, grade.missing))
        self.assertTrue(any('row4:r0:composed-inc:' in reason
                            for reason in grade.missing), grade.missing)

    def test_raw_inc_with_an_extra_reason_is_not_dischargeable(self):
        raw, engine = row4_material(
            include_eager=True, verdict='INCONCLUSIVE',
            native_reasons=['unattributed_expected_context', 'graph_incomplete'],
            expected=None)
        raw['replay']['counts']['unattributed_expected_context'] = (
            engine['eager_green_records'])
        raws, engines = four_ranks(raw, engine)
        grade = grade_row4(row4_evidence(raws, engines))
        self.assertEqual(grade.verdict(), 'INCONCLUSIVE')
        self.assertTrue(any('not_dischargeable' in reason
                            for reason in grade.missing), grade.missing)

    def test_raw_receipt_must_be_byte_unchanged(self):
        raws, engines = four_ranks(*row4_material())
        evidence = row4_evidence(raws, engines)
        evidence['raw_sha256'][0] = digest('stale-raw')
        grade = grade_row4(evidence)
        self.assertEqual(grade.verdict(), 'FAIL')
        self.assertIn('row4:r0:raw-sha', grade.failures)

    def test_wrong_rank_is_not_discharged(self):
        raws, engines = four_ranks(*row4_material())
        raws[0]['caller_proof']['worker_rank'] = 1
        grade = grade_row4(row4_evidence(raws, engines))
        self.assertEqual(grade.verdict(), 'INCONCLUSIVE')
        self.assertTrue(any('proof_rank_mismatch' in reason
                            for reason in grade.missing), grade.missing)

    def test_wrong_window_is_not_discharged(self):
        raws, engines = four_ranks(*row4_material())
        raws[0]['replay']['window']['start_ns'] += 1
        grade = grade_row4(row4_evidence(raws, engines))
        self.assertEqual(grade.verdict(), 'INCONCLUSIVE')
        self.assertTrue(any('window_start_mismatch' in reason
                            for reason in grade.missing), grade.missing)

    def test_wrong_epoch_fails(self):
        raws, engines = four_ranks(*row4_material())
        engines[0]['binding']['engine_epoch'] = ROW4_EPOCH + 1
        engines[0]['window']['binding']['engine_epoch'] = ROW4_EPOCH + 1
        grade = grade_row4(row4_evidence(raws, engines))
        self.assertEqual(grade.verdict(), 'FAIL')
        self.assertIn('row4:r0:binding-engine-epoch', grade.failures)

    def test_wrong_engine_incarnation_fails(self):
        raws, engines = four_ranks(*row4_material())
        raws[0]['caller_proof']['engine_incarnation'] = 'engine:99'
        engines[0]['binding']['engine_incarnation'] = 'engine:99'
        engines[0]['window']['binding']['engine_incarnation'] = 'engine:99'
        grade = grade_row4(row4_evidence(raws, engines))
        self.assertEqual(grade.verdict(), 'FAIL')
        self.assertIn('row4:r0:binding-engine-incarnation', grade.failures)

    def test_missing_rank_is_inconclusive(self):
        raws, engines = four_ranks(*row4_material())
        grade = grade_row4(row4_evidence(raws[:3], engines[:3]))
        self.assertEqual(grade.verdict(), 'INCONCLUSIVE')
        self.assertIn('row4:raw-placement:3', grade.missing)
        self.assertIn('row4:engine-accounting:3', grade.missing)

    def test_missing_native_accounting_is_not_a_true_flag(self):
        raws, engines = four_ranks(*row4_material())
        evidence = row4_evidence(raws, engines, native_accounting=True,
                                 placement={'decode': True})
        evidence.pop('engine_accounting')
        grade = grade_row4(evidence)
        self.assertEqual(grade.verdict(), 'INCONCLUSIVE')
        self.assertIn('row4:engine-accounting:0', grade.missing)

    def test_a_bank_file_claim_cannot_replace_a_native_report_packet(self):
        raws, engines = four_ranks(*row4_material())
        evidence = row4_evidence(raws, engines)
        grade = grade_row4(evidence)
        self.assertEqual(grade.verdict(), 'INCONCLUSIVE')
        self.assertIn('row4:r0:native-observer-packet', grade.missing)

    def test_missing_native_report_packet_is_inconclusive(self):
        raws, engines = four_ranks(*row4_material())
        evidence = row4_evidence(raws, engines)
        evidence.pop('native_read')
        grade = grade_row4(evidence)
        self.assertEqual(grade.verdict(), 'INCONCLUSIVE')
        self.assertIn('row4:r0:native-observer-packet', grade.missing)

    def test_row1_and_teacher_echoes_are_bound(self):
        raws, engines = four_ranks(*row4_material())
        evidence = row4_evidence(raws, engines)
        grade = grade_row4(evidence, teacher_sha=digest('other-teacher'),
                           row1_sha=digest('other-row1'))
        self.assertEqual(grade.verdict(), 'FAIL')
        self.assertIn('row4:row1-binding', grade.failures)
        self.assertIn('row4:teacher-binding', grade.failures)

    def test_missing_echoes_are_inconclusive(self):
        raws, engines = four_ranks(*row4_material())
        evidence = row4_evidence(raws, engines)
        for key in ('row1_sha256', 'teacher_artifact_sha256'):
            evidence.pop(key)
        grade = qualification.grade_evidence('row4', evidence, ROW4_SPEC)
        self.assertEqual(grade.verdict(), 'INCONCLUSIVE')
        self.assertIn('row4:row1-sha', grade.missing)
        self.assertIn('row4:teacher-sha', grade.missing)
        self.assertIn('row4:row1-echo', grade.missing)
        self.assertIn('row4:teacher-echo', grade.missing)

    def test_instrument_must_be_disabled_on_the_shipping_path(self):
        raws, engines = four_ranks(*row4_material())
        evidence = row4_evidence(raws, engines,
                                 shipping_path_instrument_disabled=False)
        grade = grade_row4(evidence)
        self.assertEqual(grade.verdict(), 'FAIL')
        self.assertIn('row4:shipping-instrument-disabled', grade.failures)


# ---------------------------------------------------------------------------
# qualification client run plumbing
# ---------------------------------------------------------------------------
class PairedNativeRow4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from native_placement_evidence_support import packet
        cls.packets = {str(rank): packet(rank) for rank in RANKS}

    def setUp(self):
        packets = copy.deepcopy(self.packets)
        self.evidence = row4_evidence(
            [packets[str(rank)]['receipt'] for rank in RANKS],
            [packets[str(rank)]['engine_accounting'] for rank in RANKS],
            native_observer_packets=packets,
            window={'window_id': 'win-1', 'route_generation': 7})

    def grade(self):
        return qualification.grade_evidence('row4', self.evidence,
            dict(ROW4_SPEC, run_id='run-1'),
            expected_artifact=self.evidence['teacher_artifact_sha256'],
            expected_row1=self.evidence['row1_sha256'])

    def test_four_real_paired_ledgers_and_small_actual_native_reads_pass(self):
        before = copy.deepcopy(self.evidence)
        grade = self.grade()
        self.assertEqual(grade.verdict(), 'PASS', (grade.failures, grade.missing))
        self.assertEqual(self.evidence, before)
        starts = {p['decode_window']['binding']['native_start_ns']
                  for p in self.evidence['native_observer_packets'].values()}
        self.assertEqual(len(starts), 4)

    def test_retained_split_receipts_after_full_return_keep_window_identity(self):
        for packet in self.evidence['native_observer_packets'].values():
            packet.update(current_route='full', current_route_generation=8)
        grade = self.grade()
        self.assertEqual(grade.verdict(), 'PASS', (grade.failures, grade.missing))

    def test_decode_only_cached_pass_cannot_replace_missing_prefill(self):
        packet = self.evidence['native_observer_packets']['0']
        packet.pop('native_prefill_accounting')
        packet['composed_placement'] = {'verdict': 'PASS'}
        grade = self.grade()
        self.assertEqual(grade.verdict(), 'INCONCLUSIVE', grade.failures)

    def test_native_read_digest_and_window_are_independent_guards(self):
        self.evidence['native_observer_packets']['0']['native_report_read'][
            'decoded_report_sha256'] = '0' * 64
        grade = self.grade()
        self.assertIn('row4:r0:native_read_digest', grade.failures)

    def test_raw_fail_is_not_hidden_by_missing_packet(self):
        self.evidence['raw_placement'][0]['verdict'] = 'FAIL'
        self.evidence['native_observer_packets'].pop('0')
        grade = self.grade()
        self.assertIn('row4:r0:raw-placement-fail', grade.failures)

    def test_packet_cannot_substitute_another_raw_receipt(self):
        self.evidence['native_observer_packets']['0']['receipt']['reserved_bytes'] += 1
        # The retained raw digest is independent and remains the original.
        grade = self.grade()
        self.assertEqual(grade.verdict(), 'FAIL')
        self.assertIn('row4:r0:raw-sha', grade.failures)


class QualificationRunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_phase(self, phase, evidence, artifact=None, row1=None, name=None,
                  epoch=EPOCH):
        evidence_path = write_json(
            self.root / ((name or phase) + '-response.json'),
            response(phase, evidence))
        out = self.root / ((name or phase) + '-out.json')
        code = qualification.run(
            phase, 'http://127.0.0.1:8000', MODEL, RUN, epoch, str(out),
            artifact, evidence_path, row1_sha256=row1)
        return code, json.loads(out.read_text())

    def test_row1_run_records_the_raw_artifact_sha(self):
        evidence = row1_evidence()
        code, record = self.run_phase('row1', evidence)
        self.assertEqual(code, 0, record)
        self.assertEqual(record['verdict'], 'PASS')
        self.assertEqual(record['row1_sha256'], canonical(evidence['workers']))

    def test_teacher_candidate_reuses_the_controls_artifact(self):
        artifact = teacher_artifact()
        teacher_sha = canonical(artifact)
        code, controls = self.run_phase(
            'rule4_controls', teacher_evidence('rule4_controls', artifact),
            name='controls')
        self.assertEqual(code, 0, controls)
        self.assertEqual(controls['teacher_artifact_sha256'], teacher_sha)
        code, candidate = self.run_phase(
            'rule4_candidate', teacher_evidence('rule4_candidate', artifact),
            artifact=controls['teacher_artifact_sha256'], name='candidate')
        self.assertEqual(code, 0, candidate)
        self.assertEqual(candidate['teacher_artifact_sha256'], teacher_sha)

    def test_candidate_without_the_controls_artifact_is_inconclusive(self):
        out = self.root / 'candidate.json'
        code = qualification.run('rule4_candidate', 'http://127.0.0.1:8000',
                                 MODEL, RUN, EPOCH, str(out), None,
                                 None)
        record = json.loads(out.read_text())
        self.assertEqual(code, 2)
        self.assertEqual(record['verdict'], 'INCONCLUSIVE')
        self.assertIn('missing-retained-teacher-artifact-sha',
                      record['missing'])

    def test_collection_without_a_store_never_constructs_a_driver(self):
        out = self.root / 'no-store.json'
        with patch.object(driver_module, 'QualificationDriver',
                          side_effect=AssertionError('driver constructed')):
            code = qualification.run(
                'row1', 'http://127.0.0.1:8000', MODEL, RUN, EPOCH, str(out),
                request_plan={'schema': 's5-native-row1-request-plan/1'},
                store=None)
        record = json.loads(out.read_text())
        self.assertEqual(code, 2)
        self.assertEqual(record['verdict'], 'INCONCLUSIVE')
        self.assertEqual(record['missing'], ['missing-qualification-store'])
        self.assertIsNone(record['evidence'])

    def test_evidence_path_is_pure_grading_and_never_composes_a_driver(self):
        evidence = row1_evidence()
        evidence_path = write_json(self.root / 'raw-response.json',
                                   response('row1', evidence))
        out = self.root / 'offline.json'
        with patch.object(driver_module, 'QualificationDriver',
                          side_effect=AssertionError('driver constructed')):
            code = qualification.run(
                'row1', 'http://127.0.0.1:8000', MODEL, RUN, EPOCH, str(out),
                None, evidence_path, store=str(self.root / 'store'))
        record = json.loads(out.read_text())
        # The actual packets are retained and graded; the source grader's
        # cartesian coverage stays INCONCLUSIVE rather than becoming a PASS.
        self.assertEqual(code, 0)
        self.assertEqual(record['verdict'], 'PASS')
        self.assertEqual(record['row1_sha256'], canonical(evidence['workers']))
        self.assertEqual(canonical(record['response']['evidence']),
                         canonical(evidence))
        self.assertEqual(record['missing'], [])

    def test_local_driver_composition_passes_queue_plan_and_retains_once(self):
        captured = {}
        evidence = row1_evidence()

        class FakeDriver:
            def __init__(self, store, **kwargs):
                captured['store'] = store
                captured['kwargs'] = kwargs

            def collect_phase(self, phase, **kwargs):
                captured['phase'] = phase
                captured['collect'] = kwargs
                return {'record': {'path': 'row1.json'}, 'evidence': evidence,
                        'row1_sha256': canonical(evidence['workers']),
                        'missing': []}

        plan = {'schema': 's5-native-row1-request-plan/1',
                'run_id': RUN, 'requests': []}
        out = self.root / 'composed.json'
        with patch.object(driver_module, 'QualificationDriver', FakeDriver):
            code = qualification.run(
                'row1', 'http://127.0.0.1:8000', MODEL, RUN, EPOCH, str(out),
                request_plan=plan, store=str(self.root / 'store'),
                timeout_s=12.5)
        record = json.loads(out.read_text())
        self.assertEqual(captured['store'], str(self.root / 'store'))
        self.assertEqual(captured['kwargs']['run_id'], RUN)
        self.assertEqual(captured['kwargs']['model'], MODEL)
        self.assertEqual(captured['kwargs']['engine_epoch'], EPOCH)
        self.assertEqual(captured['kwargs']['timeout_s'], 12.5)
        self.assertEqual(captured['phase'], 'row1')
        self.assertEqual(captured['collect'], {'request_plan': plan})
        # The raw evidence came back from the driver; grading is local and
        # the immutable phase record is referenced, not rewritten.
        self.assertEqual(code, 2)
        self.assertEqual(record['phase_record'], {'path': 'row1.json'})
        self.assertEqual(record['driver_missing'], [])
        self.assertEqual(record['row1_sha256'], canonical(evidence['workers']))
        self.assertEqual(record['verdict'], 'INCONCLUSIVE')

    def test_cli_reads_queue_plan_and_keeps_the_missing_store_inconclusive(self):
        plan_path = write_json(self.root / 'plan.json',
                               {'schema': 's5-native-row1-request-plan/1'})
        out = self.root / 'cli.json'
        code = qualification.main([
            '--phase', 'row1', '--model', MODEL, '--run-id', RUN,
            '--epoch', str(EPOCH), '--out', str(out),
            '--request-plan', str(plan_path)])
        record = json.loads(out.read_text())
        self.assertEqual(code, 2)
        self.assertEqual(record['missing'], ['missing-qualification-store'])
        self.assertIsNone(record['evidence'])

    def test_q18a_has_no_qualification_phase_rpc(self):
        source = Path(qualification.__file__).read_text()
        self.assertNotIn('e3_probe_rpc', source)
        self.assertNotIn('action="qualification"', source)
        self.assertNotIn("action='qualification'", source)

    def test_row4_run_binds_both_retained_shas(self):
        raws, engines = four_ranks(*row4_material())
        row1_sha = digest('row1-artifact')
        teacher_sha = digest('teacher-artifact')
        evidence = row4_evidence(raws, engines, row1_sha=row1_sha,
                                 teacher_sha=teacher_sha)
        code, record = self.run_phase('row4', evidence, teacher_sha,
                                      row1=row1_sha, epoch=ROW4_EPOCH)
        self.assertEqual(code, 2, record)
        self.assertEqual(record['row1_sha256'], row1_sha)
        self.assertEqual(record['artifact_sha256'], teacher_sha)

    def test_row4_without_the_row1_sha_is_inconclusive(self):
        out = self.root / 'row4.json'
        code = qualification.run('row4', 'http://127.0.0.1:8000', MODEL, RUN,
                                 EPOCH, str(out), digest('teacher'), None)
        record = json.loads(out.read_text())
        self.assertEqual(code, 2)
        self.assertIn('missing-retained-row1-or-teacher-sha',
                      record['missing'])

    def test_qualification_refuses_to_overwrite(self):
        out = self.root / 'out.json'
        out.write_text('{}')
        evidence_path = write_json(self.root / 'response.json',
                                   response('row1', row1_evidence()))
        with self.assertRaises(SystemExit):
            qualification.run('row1', 'http://127.0.0.1:8000', MODEL, RUN,
                              EPOCH, str(out), None, evidence_path)


if __name__ == '__main__':
    unittest.main()
