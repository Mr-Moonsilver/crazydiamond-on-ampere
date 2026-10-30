"""CPU contracts for teacher-forced numerical rule v4; no GPU claim.

Rule v4 is a separate comparison policy over the unchanged schema/2 collector
artifact.  These tests exercise the frozen control-envelope arithmetic, all
three split comparisons, tie order, missing support, calibration pinning,
structural-defect propagation and the optional client seam.
"""
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))

import e3_teacher_decode_rule4 as r4
import test_e3_teacher_decode as fixture

VALUES = {f'token_id:{i}': -i * .2 for i in range(20)}


def valid_evidence():
    e = fixture.base_evidence()
    e['top_order_basis'] = r4.TOP_ORDER_BASIS
    return e


def set_all(e, name, values=None, order=None):
    values = dict(VALUES if values is None else values)
    for row in e['runs'][name]['positions']:
        row['top'] = dict(values)
        row['top_order'] = list(order if order is not None else values)
    return e


def set_at(e, name, position, values, order=None):
    values = dict(values)
    for row in e['runs'][name]['positions']:
        if row['position'] == position:
            row['top'] = dict(values)
            row['top_order'] = list(order if order is not None else values)
    return e


def shifted(delta):
    return {token: value + delta for token, value in VALUES.items()}


def bind(e):
    return fixture.bind_numeric_fixture(e)


def controls_equal(e, values=None):
    values = dict(VALUES if values is None else values)
    for name in ('full-A1', 'full-A2', 'full-C', 'split-B'):
        set_all(e, name, values)
    return bind(e)


class GradeTests(unittest.TestCase):
    def test_valid_evidence_passes_and_labels_both_policies(self):
        e = valid_evidence()
        set_all(e, 'split-B')
        bind(e)
        out = r4.grade(e)
        self.assertEqual(out['verdict'], 'PASS', out['reasons'])
        self.assertEqual(out['version'], 4)
        self.assertEqual(out['policy'], r4.POLICY)
        self.assertEqual(out['collector_schema'], 's5-teacher-decode/2')
        self.assertEqual(out['collector_rule_version'], 3)
        self.assertEqual(out['calibration_sha256'], r4.PINNED_CALIBRATION_SHA256)
        self.assertEqual(len(out['calibration_file_sha256']), 64)
        self.assertEqual(set(out['controls']), {'full-A1|full-A2', 'full-A2|full-C'})
        self.assertEqual(set(out['comparisons']),
                         {'split-B|full-A1', 'split-B|full-A2', 'split-B|full-C'})
        self.assertEqual(out['allowances'],
                         {'winners': .375, 'top5': .5, 'shared_info': .5,
                          'flip_margin': .375, 'flips': 41})
        self.assertIn('still declares rule v3', out['note'])
        self.assertEqual(e['rule']['version'], 3)

    def test_control_envelope_uses_both_pairs_and_their_max(self):
        e = valid_evidence()
        set_all(e, 'full-A1')
        set_all(e, 'full-A2', shifted(-.125))
        set_all(e, 'full-C', shifted(-.125))
        set_all(e, 'split-B')
        bind(e)
        out = r4.grade(e)
        self.assertEqual(out['verdict'], 'PASS', out['reasons'])
        for metric in ('winners', 'top5', 'shared_info'):
            self.assertAlmostEqual(out['controls']['full-A1|full-A2'][metric], .125)
            self.assertEqual(out['controls']['full-A2|full-C'][metric], 0.0)
            self.assertAlmostEqual(out['control_max'][metric], .125)
            self.assertAlmostEqual(out['allowances'][metric],
                                   .5 if metric != 'winners' else .375)
        self.assertEqual(out['control_max']['flips'], 0)
        self.assertEqual(out['allowances']['flips'], 41)

    def test_split_is_compared_against_all_three_control_runs(self):
        e = valid_evidence()
        set_all(e, 'full-A1')
        set_all(e, 'full-A2', shifted(-.125))
        set_all(e, 'full-C', shifted(-.25))
        set_all(e, 'split-B')
        bind(e)
        out = r4.grade(e)
        self.assertEqual(out['verdict'], 'PASS', out['reasons'])
        self.assertEqual(out['comparisons']['split-B|full-A1']['winners'], 0.0)
        self.assertAlmostEqual(out['comparisons']['split-B|full-A2']['winners'], .125)
        self.assertAlmostEqual(out['comparisons']['split-B|full-C']['winners'], .25)

    def test_one_metric_failure_is_labelled_without_collateral_tolerance(self):
        e = valid_evidence()
        set_all(e, 'full-A1')
        set_all(e, 'full-A2', shifted(-.125))
        set_all(e, 'full-C', shifted(-.125))
        values = dict(VALUES)
        values['token_id:19'] = -4.8  # outside both top-5 sets, one nat below A1
        set_all(e, 'split-B', values)
        bind(e)
        out = r4.grade(e)
        self.assertEqual(out['verdict'], 'FAIL')
        self.assertEqual(len(out['reasons']), 3)
        self.assertTrue(all('shared_info' in reason for reason in out['reasons']),
                        out['reasons'])
        self.assertTrue(all('winners' not in reason and 'top5' not in reason
                            for reason in out['reasons']), out['reasons'])

    def test_exact_allowance_edge_passes_then_fails(self):
        e = valid_evidence()
        controls_equal(e)
        values = dict(VALUES)
        values['token_id:10'] = -1.5  # exactly 0.5 from the -2.0 baseline
        set_all(e, 'split-B', values)
        bind(e)
        out = r4.grade(e)
        self.assertEqual(out['verdict'], 'PASS', out['reasons'])
        self.assertEqual(out['allowances']['shared_info'], .5)
        self.assertEqual(out['comparisons']['split-B|full-A1']['shared_info'], .5)
        values['token_id:10'] = -1.499999  # 0.500001 above the baseline
        set_all(e, 'split-B', values)
        bind(e)
        out = r4.grade(e)
        self.assertEqual(out['verdict'], 'FAIL')
        self.assertTrue(all('shared_info' in reason for reason in out['reasons']),
                        out['reasons'])

    def test_missing_or_infinite_support_is_never_dropped(self):
        e = valid_evidence()
        disjoint = {f'token_id:{100 + i}': -i * .2 for i in range(20)}
        set_all(e, 'full-A1')
        set_all(e, 'full-A2', disjoint)
        set_all(e, 'full-C')
        set_all(e, 'split-B')
        bind(e)
        out = r4.grade(e)
        self.assertEqual(out['verdict'], 'INCONCLUSIVE')
        self.assertTrue(any('full-A1|full-A2' in reason and 'nonfinite' in reason
                            for reason in out['reasons']), out['reasons'])
        self.assertEqual(out['controls']['full-A1|full-A2']['winners'], 'inf')
        self.assertNotIn('Infinity', json.dumps(out))
        self.assertEqual(out['comparisons'], {})

        e = valid_evidence()
        controls_equal(e)
        set_all(e, 'split-B', disjoint)
        bind(e)
        out = r4.grade(e)
        self.assertEqual(out['verdict'], 'FAIL')
        self.assertTrue(any('split-B|full-A1' in reason and 'winners' in reason
                            and 'nonfinite' in reason for reason in out['reasons']),
                        out['reasons'])

    def test_flip_over_margin_allowance_fails_even_when_deltas_are_inside(self):
        control = dict(VALUES)
        control['token_id:0'], control['token_id:1'] = .4, 0.0
        split = dict(control)
        split['token_id:0'], split['token_id:1'] = .1, .3
        e = valid_evidence()
        for name in ('full-A1', 'full-A2', 'full-C'):
            set_all(e, name, control)
        set_all(e, 'split-B', control)
        set_at(e, 'split-B', fixture.MEASURED[0], split)
        bind(e)
        out = r4.grade(e)
        self.assertEqual(out['verdict'], 'FAIL')
        metrics = out['comparisons']['split-B|full-A1']
        self.assertEqual(metrics['flips'], 8)
        self.assertAlmostEqual(metrics['winners'], .3)
        self.assertAlmostEqual(metrics['top5'], .3)
        self.assertAlmostEqual(metrics['shared_info'], .3)
        self.assertAlmostEqual(metrics['flip_margin'], .4)
        self.assertEqual(metrics['flips_over'], 8)
        self.assertEqual(len(out['reasons']), 3)
        self.assertTrue(all('flip_margin' in reason for reason in out['reasons']),
                        out['reasons'])

    def test_tie_order_is_preserved_and_zero_margin_flip_is_admissible(self):
        self.assertEqual(r4._winner([('token_id:0', 0.0), ('token_id:1', 0.0)]),
                         'token_id:0')
        self.assertEqual(r4._winner([('token_id:1', 0.0), ('token_id:0', 0.0)]),
                         'token_id:1')
        seq = [(f'token_id:{i}', 0.0) for i in range(20)]
        self.assertEqual(r4._top5(seq), {f'token_id:{i}' for i in range(15, 20)})
        self.assertEqual(r4._top5(list(reversed(seq))),
                         {f'token_id:{i}' for i in range(5)})

        e = valid_evidence()
        controls_equal(e)
        values = dict(VALUES)
        values['token_id:1'] = 0.0
        order = ['token_id:1', 'token_id:0'] + [
            f'token_id:{i}' for i in range(2, 20)]
        set_at(e, 'split-B', fixture.MEASURED[0], values, order=order)
        bind(e)
        out = r4.grade(e)
        self.assertEqual(out['verdict'], 'PASS', out['reasons'])
        metrics = out['comparisons']['split-B|full-A1']
        self.assertEqual(metrics['flips'], 8)
        self.assertAlmostEqual(metrics['flip_margin'], .2)
        self.assertLessEqual(metrics['flip_margin'], out['allowances']['flip_margin'])

    def test_control_over_hard_cap_or_nonfinite_is_inconclusive(self):
        e = valid_evidence()
        set_all(e, 'full-A1')
        set_all(e, 'full-A2')
        set_all(e, 'full-C', shifted(-1.3))
        set_all(e, 'split-B')
        bind(e)
        out = r4.grade(e)
        self.assertEqual(out['verdict'], 'INCONCLUSIVE')
        self.assertTrue(any('full-A2|full-C' in reason and 'winners' in reason
                            and 'hard cap' in reason for reason in out['reasons']),
                        out['reasons'])
        self.assertEqual(out['comparisons'], {})

        e = valid_evidence()
        reverse = {f'token_id:{i}': -(19 - i) * .2 for i in range(20)}
        set_all(e, 'full-A1')
        set_all(e, 'full-A2', reverse)
        set_all(e, 'full-C')
        set_all(e, 'split-B')
        bind(e)
        out = r4.grade(e)
        self.assertEqual(out['verdict'], 'INCONCLUSIVE')
        self.assertTrue(any('flips' in reason and 'hard cap' in reason
                            for reason in out['reasons']), out['reasons'])

    def test_forged_or_missing_calibration_is_refused(self):
        e = valid_evidence()
        controls_equal(e)
        with tempfile.TemporaryDirectory() as directory:
            forged = Path(directory) / 'forged.json'
            calibration = json.loads(r4.DEFAULT_CALIBRATION_PATH.read_text())
            calibration['floors']['winners'] = .1
            forged.write_text(json.dumps(calibration))
            out = r4.grade(e, calibration_path=forged)
            self.assertEqual(out['verdict'], 'INCONCLUSIVE')
            self.assertIn('calibration identity differs', out['missing'])
            self.assertEqual(out['controls'], {})

            exact = Path(directory) / 'exact.json'
            exact.write_bytes(r4.DEFAULT_CALIBRATION_PATH.read_bytes())
            out = r4.grade(e, calibration_path=exact)
            self.assertEqual(out['verdict'], 'PASS', out['reasons'])
            self.assertEqual(out['calibration_sha256'], r4.PINNED_CALIBRATION_SHA256)

            out = r4.grade(e, calibration_path=Path(directory) / 'absent.json')
            self.assertEqual(out['verdict'], 'INCONCLUSIVE')
            self.assertIn('calibration unreadable', out['missing'])

    def test_structural_defects_are_propagated_unchanged(self):
        e = valid_evidence()
        controls_equal(e)
        e['runs']['split-B']['row_mapping'] = fixture.row_mapping_fixture(
            'split-B', 'split', generation=1, repeat=2, steps=fixture.swapped_steps())
        bind(e)
        out = r4.grade(e)
        self.assertEqual(out['verdict'], 'INCONCLUSIVE')
        self.assertTrue(any('one-position shift' in reason for reason in out['reasons']),
                        out['reasons'])

        e = valid_evidence()
        controls_equal(e)
        e['runs']['full-A2']['row_mapping']['overflow'] = True
        out = r4.grade(e)
        self.assertEqual(out['verdict'], 'INCONCLUSIVE')
        self.assertTrue(any('row mapping overflow' in reason for reason in out['reasons']),
                        out['reasons'])

        e = valid_evidence()
        controls_equal(e)
        position = fixture.MEASURED[0]
        for row in e['runs']['split-B']['positions']:
            if row['position'] == position:
                row['forced_token_exact'] = False
        for record in e['runs']['split-B']['row_mapping']['records']:
            if record['position'] == position:
                record['forced_token_exact'] = False
        out = r4.grade(e)
        self.assertEqual(out['verdict'], 'FAIL')
        self.assertTrue(any('worker observed forced token mismatch' in reason
                            for reason in out['failures']), out['failures'])

    def test_raw_evidence_is_not_mutated(self):
        e = valid_evidence()
        controls_equal(e)
        before = copy.deepcopy(e)
        out = r4.grade(e)
        self.assertEqual(out['verdict'], 'PASS', out['reasons'])
        self.assertEqual(e, before)
        self.assertEqual(e['rule']['version'], 3)
        self.assertEqual(e['schema'], 's5-teacher-decode/2')

    def test_pair_metrics_uses_both_flip_directions(self):
        left_values = dict(VALUES)
        left_values['token_id:0'], left_values['token_id:1'] = .2, 0.0
        right_values = dict(VALUES)
        right_values['token_id:0'], right_values['token_id:1'] = .1, .4
        left = {p: {'top': dict(left_values)} for p in range(r4.N)}
        right = {p: {'top': dict(right_values)} for p in range(r4.N)}
        metrics = r4.pair_metrics(left, right)
        self.assertEqual(metrics['flips'], r4.N)
        self.assertAlmostEqual(metrics['flip_margin'], .3)
        self.assertEqual(r4.pair_metrics(left, right, flip_margin_limit=.25)['flips_over'],
                         r4.N)
        self.assertEqual(r4.pair_metrics(left, right, flip_margin_limit=.35)['flips_over'], 0)


class AllowanceTests(unittest.TestCase):
    def setUp(self):
        self.calibration, _, _ = r4.load_calibration()
        self.control_max = {
            metric: max(metrics[metric] for metrics in self.calibration['controls'].values())
            for metric in r4.CONTINUOUS + r4.COUNTING}

    def test_formula_floor_and_cap_clamps(self):
        allowances = r4._allowances(self.calibration, self.control_max)
        self.assertAlmostEqual(
            allowances['winners'],
            self.calibration['factor'] * self.control_max['winners'] + self.calibration['offset'])
        self.assertAlmostEqual(
            allowances['top5'],
            self.calibration['factor'] * self.control_max['top5'] + self.calibration['offset'])
        self.assertAlmostEqual(
            allowances['shared_info'],
            self.calibration['factor'] * self.control_max['shared_info'] + self.calibration['offset'])
        self.assertEqual(allowances['flip_margin'], 1.0)
        self.assertEqual(allowances['flips'], 1577)

        zero = {metric: 0.0 for metric in r4.CONTINUOUS}
        zero['flips'] = 0
        floors = r4._allowances(self.calibration, zero)
        self.assertEqual(floors, {'winners': .375, 'top5': .5, 'shared_info': .5,
                                  'flip_margin': .375, 'flips': 41})

        huge = {metric: 100.0 for metric in r4.CONTINUOUS}
        huge['flips'] = 10000
        caps = r4._allowances(self.calibration, huge)
        self.assertEqual(caps, self.calibration['caps'])

    def test_comparison_edges_are_inclusive(self):
        allowances = r4._allowances(self.calibration, self.control_max)
        metrics = {'n': r4.N, 'winners': allowances['winners'],
                   'top5': allowances['top5'], 'shared_info': allowances['shared_info'],
                   'flips': allowances['flips'], 'flip_margin': allowances['flip_margin'],
                   'flips_over': 0}
        self.assertEqual(r4._comparison_violations('edge', metrics, allowances), [])
        for metric in ('winners', 'top5', 'shared_info'):
            over = dict(metrics, **{metric: allowances[metric] + 1e-9})
            violations = r4._comparison_violations('edge', over, allowances)
            self.assertEqual(len(violations), 1)
            self.assertIn(metric, violations[0])
        over = dict(metrics, flips=allowances['flips'] + 1)
        self.assertEqual(len(r4._comparison_violations('edge', over, allowances)), 1)
        flip = dict(metrics, flips=1, flip_margin=allowances['flip_margin'],
                    flips_over=0)
        self.assertEqual(r4._comparison_violations('edge', flip, allowances), [])
        over_flip = dict(flip, flips_over=1)
        self.assertEqual(len(r4._comparison_violations('edge', over_flip, allowances)), 1)

    def test_control_caps_are_inclusive(self):
        exact = {metric: self.calibration['caps'][metric] for metric in r4.CONTINUOUS}
        exact['flips'] = self.calibration['caps']['flips']
        self.assertEqual(r4._control_violations(self.calibration, {'pair': exact}), [])
        over = dict(exact, winners=exact['winners'] + 1e-9)
        violations = r4._control_violations(self.calibration, {'pair': over})
        self.assertEqual(len(violations), 1)
        self.assertIn('winners', violations[0])
        over = dict(exact, flips=exact['flips'] + 1)
        self.assertTrue(r4._control_violations(self.calibration, {'pair': over}))


class CalibrationTests(unittest.TestCase):
    def test_pinned_calibration_is_self_consistent(self):
        calibration = json.loads(r4.DEFAULT_CALIBRATION_PATH.read_text())
        self.assertEqual(calibration['version'], 4)
        self.assertEqual(calibration['policy'], r4.POLICY)
        self.assertEqual(calibration['source_sha256'], r4.SOURCE_SHA256)
        self.assertEqual(calibration['evidence_schema'], 's5-teacher-decode/2')
        self.assertEqual(calibration['evidence_rule_version'], 3)
        self.assertEqual(calibration['geometry'],
                         {'N': 2048, 'streams': 8, 'positions': 256})
        self.assertEqual(r4._canonical_sha(calibration), r4.PINNED_CALIBRATION_SHA256)
        expected = {
            'full-A1|full-A2': {'winners': 0.5495443940162659,
                                'top5': 0.7004556655883789,
                                'shared_info': 5.48369026184082,
                                'flip_margin': 0.4375, 'flips': 768},
            'full-A2|full-C': {'winners': 0.38385462760925293,
                               'top5': 0.39290809631347656,
                               'shared_info': 4.618258476257324,
                               'flip_margin': 0.375, 'flips': 678}}
        self.assertEqual(calibration['controls'], expected)
        for metric in r4.CONTINUOUS:
            maximum = max(metrics[metric] for metrics in expected.values())
            self.assertEqual(calibration['envelope'][metric],
                             math.ceil(maximum * 1e6) / 1e6)
            self.assertEqual(calibration['caps'][metric],
                             round(2 * calibration['envelope'][metric] + .125, 6))
        self.assertEqual(calibration['envelope']['flips'], 768)
        self.assertEqual(calibration['caps']['flips'],
                         min(r4.N, 2 * 768 + math.ceil(.02 * r4.N)))
        self.assertEqual(calibration['floors'],
                         {'winners': .375, 'top5': .5, 'shared_info': .5,
                          'flip_margin': .375})


class ClientSeamTests(unittest.TestCase):
    def test_default_is_rule3_and_four_labels_the_new_policy(self):
        cases = fixture.ClientTests()
        instrument = cases.instrument()
        instrument.numerical_rule = 4
        instrument.calibration_path = None
        out = instrument.finish(cases.fixture_runs(instrument))
        self.assertEqual(out['grade']['verdict'], 'PASS', out['grade']['reasons'])
        self.assertEqual(out['grade']['version'], 4)
        self.assertEqual(out['evidence']['rule']['version'], 3)
        self.assertEqual(out['evidence']['schema'], 's5-teacher-decode/2')
        self.assertEqual(out['numerical_rule'], 4)
        self.assertEqual(out['gate_policy']['name'], 'control-envelope-v4')
        self.assertEqual(out['gate_policy']['collector_rule_version'], 3)
        self.assertEqual(out['gate_policy']['calibration_sha256'],
                         r4.PINNED_CALIBRATION_SHA256)

        instrument = cases.instrument()
        out = instrument.finish(cases.fixture_runs(instrument))
        reasons = out['grade'].get('missing', []) + out['grade'].get('failures', [])
        self.assertEqual(out['grade']['verdict'], 'PASS', reasons)
        self.assertEqual(out['grade']['rule']['version'], 3)
        self.assertNotIn('numerical_rule', out)

    def test_instrument_rejects_unknown_rule_selection(self):
        import e3_teacher_decode_client as client
        baseline = ROOT / 'harness/baselines/q15-teacher-forced-rule3.json'
        with self.assertRaisesRegex(ValueError, 'numerical rule selection invalid'):
            client.Instrument(baseline, post=lambda *args: None, numerical_rule=5)
        with self.assertRaisesRegex(ValueError, 'calibration path invalid'):
            client.Instrument(baseline, post=lambda *args: None,
                              numerical_rule=4, calibration_path=object())


if __name__ == '__main__':
    unittest.main()
