"""Raw grader acceptance with the actual oracle producer and CPU byte fakes.

No GPU, engine or transport. Positive records come from Row1Qualifier's direct
CPU comparisons and finish_layer, not handwritten equality/digest verdicts.
"""
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))
import e3_native_row1_grade as grader
import row1_oracle as oracle
import test_e3_native_source_oracle as source

RUN = 'row1-scope'
EPOCH = 3


def evidence():
    packets = []
    for rank in grader.RANKS:
        store = oracle.Row1RecordStore()
        names = ['qsa.%d' % (24 * (rank // 2) + layer) for layer in (0, 1)]
        for name in names:
            for epoch, route in enumerate(('dense_dma', 'sparse_dma'), 2):
                rig = source.Row1QualifierTests('test_a_matching_bucket_records_the_actual_bytes')
                rig.setUp()
                rig.qualifier.store = store
                rig.qualifier.lane = replace(rig.lane, worker_rank=rank)
                rig.epochs['worker_incarnation'] = 'worker-%d' % rank
                hold = types.SimpleNamespace(released=False)
                run = rig.make_run(layer_name=name, state_epoch=epoch,
                                   sparse_hold=hold, row1_reference=lambda: None)
                bucket = replace(rig.bucket(), route=route,
                                 compact_start=0 if route == 'sparse_dma' else None)
                rig.fill_actual(bucket)
                rig.qualifier.bucket(run, bucket, rig.arena)

                def reference(session):
                    session.route('dense_dma' if route == 'dense_dma' else 'sparse_uva', 2)
                    if route == 'dense_dma':
                        session.fill(rig.arena, 0, 1, source.T((1,), values=[0]))

                rig.output(run, reference, rig.out_tensor([0.0, 0.0]),
                           expected_route='dense_dma' if route == 'dense_dma' else 'sparse_uva',
                           expected_fills=1 if route == 'dense_dma' else 0)
                rig.qualifier.check_releasable(run)
                hold.released = True
                count = int(route == 'sparse_dma')
                rig.qualifier.finish_layer(run, hold_stats=dict(
                    buckets=count, copied=count, published=count, consumed=count,
                    abandoned=0, rows=bucket.rows * count, descriptors=bucket.pieces * count,
                    nbytes=bucket.nbytes * count, released=1))
        packets.append(dict(schema=grader.SCHEMA, run_id=RUN, rank=rank,
            worker_incarnation='worker-%d' % rank, engine_epoch=EPOCH,
            route_generation=2, disabled_route_generation=3, local_layers=names, enabled=False,
            raw=dict(store.summary(), enabled=False,
                records=[r.as_dict() for r in store.records()],
                layer_records=[r.as_dict() for r in store.layer_records()],
                failure_records=[])))
    # Exercise the actual tuple -> JSON list transport representation.
    return json.loads(json.dumps(dict(phase='row1', engine_epoch=EPOCH, workers=packets)))


class RawGradeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = evidence()

    def setUp(self):
        self.data = copy.deepcopy(self.raw)

    def grade(self):
        return grader.grade(self.data, run_id=RUN, engine_epoch=EPOCH)

    def assert_verdict(self, expected):
        result = self.grade()
        self.assertEqual(result['verdict'], expected, (result['failures'], result['missing']))
        return result

    def test_actual_flat_producer_records_and_distinct_pp_layers_pass(self):
        self.assert_verdict('PASS')

    def test_byte_mismatch_fails_even_when_other_records_pass(self):
        row = self.data['workers'][0]['raw']['records'][0]
        row.update(equal=False, different_bytes=1)
        row['digest'] = grader._digest(row)
        result = self.assert_verdict('FAIL')
        self.assertIn('rank0:direct_comparison', result['failures'])
        self.assertFalse(result['checks']['rank0:direct_comparison'])

    def test_missing_worker_is_inconclusive(self):
        self.data['workers'].pop()
        self.assert_verdict('INCONCLUSIVE')

    def test_source_epoch_or_worker_cannot_be_relabelled(self):
        self.data['workers'][0]['raw']['records'][0]['source_epoch'] = 0
        self.assert_verdict('FAIL')

    def test_foreign_scope_cannot_be_graded_under_this_run(self):
        self.data['workers'][0]['run_id'] = 'foreign'
        self.assert_verdict('FAIL')

    def test_oracle_must_be_disabled_before_performance(self):
        self.data['workers'][0]['enabled'] = True
        self.assert_verdict('FAIL')

    def test_mixed_route_generations_do_not_prove_collective_disable(self):
        self.data['workers'][0]['disabled_route_generation'] += 1
        self.assert_verdict('INCONCLUSIVE')

    def test_record_digest_and_coverage_are_independent_guards(self):
        row = self.data['workers'][0]['raw']['records'][0]
        row['coverage_bytes'] -= 1
        self.assertEqual(row['digest'], grader._digest(row))
        self.assert_verdict('FAIL')

    def test_retained_failure_always_fails(self):
        raw = self.data['workers'][0]['raw']
        raw['failure_records'].append({'kind': 'arena_bytes'})
        raw['failures'] = 1
        self.assert_verdict('FAIL')

    def test_duplicate_records_cannot_inflate_coverage(self):
        raw = self.data['workers'][0]['raw']
        raw['records'].append(copy.deepcopy(raw['records'][0]))
        raw['comparisons'] += 1
        self.assert_verdict('FAIL')

    def test_release_and_consumer_census_must_agree(self):
        raw = self.data['workers'][0]['raw']
        row = next(r for r in raw['layer_records'] if r['route'] == 'sparse_dma')
        stats = dict(row['hold_stats'])
        stats['consumed'] = 0
        row['hold_stats'] = list(stats.items())
        row['digest'] = grader._digest(row, layer=True)
        self.assert_verdict('FAIL')

    def test_missing_output_or_missing_layer_is_not_a_pass(self):
        raw = self.data['workers'][0]['raw']
        raw['records'] = [r for r in raw['records'] if r['layer'] != 'qsa.0']
        raw['layer_records'] = [r for r in raw['layer_records'] if r['layer'] != 'qsa.0']
        raw.update(comparisons=len(raw['records']), layers=len(raw['layer_records']))
        self.assert_verdict('INCONCLUSIVE')

    def test_layer_route_marginals_cannot_replace_cartesian_coverage(self):
        raw = self.data['workers'][0]['raw']
        keep = lambda r: (r['layer'], r['route']) in (
            ('qsa.0', 'dense_dma'), ('qsa.1', 'sparse_dma'))
        raw['records'] = [r for r in raw['records'] if keep(r)]
        raw['layer_records'] = [r for r in raw['layer_records'] if keep(r)]
        raw.update(comparisons=len(raw['records']), layers=len(raw['layer_records']))
        self.assert_verdict('INCONCLUSIVE')

    def test_raw_store_truncation_cannot_pass(self):
        self.data['workers'][0]['raw']['records'].pop()
        self.assert_verdict('FAIL')

    def test_malformed_arithmetic_returns_a_verdict_not_an_exception(self):
        for field, value in (('rows', None), ('operation', {}), ('compared_bytes', [])):
            with self.subTest(field=field):
                self.data = copy.deepcopy(self.raw)
                self.data['workers'][0]['raw']['records'][0][field] = value
                self.assert_verdict('FAIL')

    def test_vm_or_hash_only_artifact_is_never_engine_evidence(self):
        self.data = dict(phase='row1', vm_receipt_used_as_engine_evidence=True,
                         records=[{'sha256': '0' * 64, 'equal': True}])
        self.assert_verdict('FAIL')


if __name__ == '__main__':
    unittest.main()
