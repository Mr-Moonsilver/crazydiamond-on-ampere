"""Native43 l-f policy-aware row1 grade and its qualification wrapper (CPU only).

Dense-policy evidence comes from the actual Row1Qualifier producer, as in test_e3_native_row1_grade: per rank
two layers, each with a dense chunk at the arena (one bucket, one reference fill) and a dense chunk beyond it
(two buckets, two reference fills: the reference on its own dense branch).  The base grade over that evidence
is INCONCLUSIVE only for the cartesian dense x sparse coverage; native43_row1_grade with policy dense replaces
exactly that rule and PASSes; with policy current it is the base grade.
Leaves: harness/native43_row1_dense_fixture_leaves.json (row1-evidence-producer, qualification-main).
"""
import copy
from dataclasses import replace
import json
import types
import unittest
from unittest import mock

import e3_native_row1_grade as base
import e3_q18a_qualification as qualification
import native43_row1_grade as policy_grade
import native43_row1_qualification as wrapper
import row1_oracle as oracle
import test_e3_native_row1_grade as base_tests
import test_e3_native_source_oracle as source

RUN = base_tests.RUN
EPOCH = base_tests.EPOCH


def dense_evidence(*, beyond=True):
    packets = []
    for rank in base.RANKS:
        store = oracle.Row1RecordStore()
        names = ['qsa.%d' % (24 * (rank // 2) + layer) for layer in (0, 1)]
        for name in names:
            for epoch, buckets in enumerate((1, 2 if beyond else 1), 2):
                rig = source.Row1QualifierTests('test_a_matching_bucket_records_the_actual_bytes')
                rig.setUp()
                rig.qualifier.store = store
                rig.qualifier.lane = replace(rig.lane, worker_rank=rank)
                rig.epochs['worker_incarnation'] = 'worker-%d' % rank
                hold = types.SimpleNamespace(released=False)
                run = rig.make_run(layer_name=name, state_epoch=epoch, sparse_hold=hold,
                                   row1_reference=lambda: None)
                for index in range(buckets):
                    bucket = replace(rig.bucket(index=index), route='dense_dma', compact_start=None)
                    rig.fill_actual(bucket)
                    rig.qualifier.bucket(run, bucket, rig.arena)

                def reference(session, buckets=buckets):
                    session.route('dense_dma', 25 if buckets == 2 else 24)
                    for index in range(buckets):
                        session.fill(rig.arena, index, index + 1, source.T((2,), values=[0, 0]))

                rig.output(run, reference, rig.out_tensor([0.0, 0.0]), expected_route='dense_dma',
                           expected_fills=buckets)
                rig.qualifier.check_releasable(run)
                hold.released = True
                rig.qualifier.finish_layer(run, hold_stats=dict(
                    buckets=0, copied=0, published=0, consumed=0, abandoned=0, rows=0, descriptors=0, nbytes=0,
                    released=1))
        packets.append(dict(schema=base.SCHEMA, run_id=RUN, rank=rank, worker_incarnation='worker-%d' % rank,
                            engine_epoch=EPOCH, route_generation=2, disabled_route_generation=3, local_layers=names,
                            enabled=False,
                            raw=dict(store.summary(), enabled=False, records=[r.as_dict() for r in store.records()],
                                     layer_records=[r.as_dict() for r in store.layer_records()],
                                     failure_records=[])))
    return json.loads(json.dumps(dict(phase='row1', engine_epoch=EPOCH, workers=packets)))


class PolicyGradeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dense = dense_evidence()
        cls.dense_at_arena = dense_evidence(beyond=False)
        cls.current = base_tests.evidence()

    def grade(self, evidence, policy):
        return policy_grade.grade(copy.deepcopy(evidence), run_id=RUN, engine_epoch=EPOCH, policy=policy)

    def test_dense_policy_evidence_passes_and_the_base_misses_only_the_cartesian_rule(self):
        result = self.grade(self.dense, 'dense')
        self.assertEqual(result['verdict'], 'PASS', (result['failures'], result['missing']))
        self.assertEqual(result['policy'], 'dense')
        base_result = base.grade(copy.deepcopy(self.dense), run_id=RUN, engine_epoch=EPOCH)
        self.assertEqual(base_result['verdict'], 'INCONCLUSIVE')
        self.assertEqual(base_result['failures'], [])
        self.assertEqual(base_result['missing'], ['rank%d:layer_route_cartesian_coverage' % r for r in base.RANKS])
        self.assertEqual(result['replaced'], base_result['missing'])
        for rank in base.RANKS:
            self.assertIs(result['checks']['rank%d:dense_policy_route_coverage' % rank], True)
            self.assertIs(result['checks']['rank%d:dense_policy_beyond_arena_coverage' % rank], True)
        for key, value in base_result['checks'].items():
            self.assertEqual(result['checks'][key], value, key)

    def test_current_policy_is_the_base_grade(self):
        for evidence in (self.current, self.dense):
            with self.subTest(evidence='current' if evidence is self.current else 'dense'):
                result = self.grade(evidence, 'current')
                expected = base.grade(copy.deepcopy(evidence), run_id=RUN, engine_epoch=EPOCH)
                self.assertEqual(result, dict(expected, policy='current'))
        self.assertEqual(self.grade(self.current, 'current')['verdict'], 'PASS')
        self.assertEqual(self.grade(self.dense, 'current')['verdict'], 'INCONCLUSIVE')

    def test_a_sparse_route_under_the_dense_policy_fails(self):
        result = self.grade(self.current, 'dense')
        self.assertEqual(result['verdict'], 'FAIL')
        for rank in base.RANKS:
            self.assertIn('rank%d:dense_policy_sparse_route' % rank, result['failures'])
            self.assertIs(result['checks']['rank%d:dense_policy_sparse_route' % rank], False)

    def test_dense_without_a_beyond_arena_comparison_is_inconclusive(self):
        result = self.grade(self.dense_at_arena, 'dense')
        self.assertEqual(result['verdict'], 'INCONCLUSIVE')
        self.assertEqual(result['failures'], [])
        self.assertEqual(result['missing'], ['rank%d:dense_policy_beyond_arena_coverage' % r for r in base.RANKS])

    def test_a_layer_without_dense_records_is_inconclusive(self):
        data = copy.deepcopy(self.dense)
        raw = data['workers'][1]['raw']
        name = data['workers'][1]['local_layers'][1]
        raw['records'] = [r for r in raw['records'] if r['layer'] != name]
        raw['layer_records'] = [r for r in raw['layer_records'] if r['layer'] != name]
        raw.update(comparisons=len(raw['records']), layers=len(raw['layer_records']))
        result = self.grade(data, 'dense')
        self.assertEqual(result['verdict'], 'INCONCLUSIVE')
        self.assertIn('rank1:dense_policy_route_coverage', result['missing'])
        self.assertIn('rank1:dense_policy_beyond_arena_coverage', result['missing'])

    def test_every_base_failure_still_fails_under_the_dense_policy(self):
        data = copy.deepcopy(self.dense)
        row = data['workers'][0]['raw']['records'][0]
        row.update(equal=False, different_bytes=1)
        row['digest'] = base._digest(row)
        result = self.grade(data, 'dense')
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('rank0:direct_comparison', result['failures'])
        data = copy.deepcopy(self.dense)
        data['workers'].pop()
        self.assertEqual(self.grade(data, 'dense')['verdict'], 'INCONCLUSIVE')
        self.assertIn('rank_coverage', self.grade(data, 'dense')['missing'])

    def test_a_relabelled_beyond_arena_record_does_not_count(self):
        data = copy.deepcopy(self.dense)
        for row in data['workers'][2]['raw']['records']:
            if row['operation'] == base.OUTPUT and row['pieces'] == 2:
                row['worker_incarnation'] = 'foreign'
        result = self.grade(data, 'dense')
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('rank2:dense_policy_beyond_arena_coverage', result['missing'])

    def test_unknown_policy_refuses(self):
        for policy in ('DENSE', '', None):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                self.grade(self.dense, policy)


class WrapperTests(unittest.TestCase):
    def test_policy_grade_row1_writes_the_same_grade_names_into_the_phase_grade(self):
        g = qualification.Grade('row1')
        wrapper.policy_grade_row1('dense')(g, dense_evidence(), run_id=RUN, engine_epoch=EPOCH)
        self.assertEqual(g.failures, [])
        self.assertEqual(g.missing, [])
        self.assertIs(g.checks['row1:policy-dense'], True)
        self.assertIs(g.checks['row1:rank0:dense_policy_beyond_arena_coverage'], True)
        g = qualification.Grade('row1')
        qualification._grade_row1(g, dense_evidence(), run_id=RUN, engine_epoch=EPOCH)
        self.assertEqual(g.missing, ['row1:rank%d:layer_route_cartesian_coverage' % r for r in base.RANKS])

    def test_the_real_row1_phase_grade_passes_dense_only_with_the_policy_binding(self):
        evidence = dict(dense_evidence(), model_settings_unchanged=True)
        spec = dict(qualification._specification('row1', 'm', RUN, None), engine_epoch=EPOCH)
        g = qualification.grade_evidence('row1', copy.deepcopy(evidence), spec)
        self.assertEqual(g.verdict(), 'INCONCLUSIVE')
        self.assertEqual(g.missing, ['row1:rank%d:layer_route_cartesian_coverage' % r for r in base.RANKS])
        with mock.patch.object(qualification, '_grade_row1', wrapper.policy_grade_row1('dense')):
            g = qualification.grade_evidence('row1', copy.deepcopy(evidence), spec)
        self.assertEqual(g.verdict(), 'PASS', (g.failures, g.missing))
        with mock.patch.object(qualification, '_grade_row1', wrapper.policy_grade_row1('dense')):
            g = qualification.grade_evidence('row1', dict(base_tests.evidence(), model_settings_unchanged=True), spec)
        self.assertEqual(g.verdict(), 'FAIL')

    def test_main_binds_the_policy_grade_for_the_call_and_restores_it(self):
        original = qualification._grade_row1
        seen = {}

        def fake_main(argv):
            seen['argv'] = argv
            seen['bound'] = qualification._grade_row1
            return 0
        argv = ['--qsa-policy', 'dense', '--phase', 'row1', '--model', 'm', '--run-id', RUN, '--out', '/nonexistent']
        with mock.patch.object(qualification, 'main', fake_main):
            self.assertEqual(wrapper.main(argv), 0)
        self.assertEqual(seen['argv'], argv[2:])
        self.assertIsNot(seen['bound'], original)
        self.assertIs(qualification._grade_row1, original)

    def test_main_restores_on_error_and_refuses_other_phases_or_no_policy(self):
        original = qualification._grade_row1
        with mock.patch.object(qualification, 'main', side_effect=RuntimeError('client')):
            with self.assertRaises(RuntimeError):
                wrapper.main(['--qsa-policy', 'dense', '--phase', 'row1'])
        self.assertIs(qualification._grade_row1, original)
        for argv in (['--qsa-policy', 'dense', '--phase', 'row4'], ['--phase', 'row1'],
                     ['--qsa-policy', 'sparse', '--phase', 'row1']):
            with self.subTest(argv=argv), mock.patch('sys.stderr'), self.assertRaises(SystemExit):
                wrapper.main(argv)
        self.assertIs(qualification._grade_row1, original)


if __name__ == '__main__':
    unittest.main()
