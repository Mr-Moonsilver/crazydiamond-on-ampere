"""CPU contract tests for paired native decode/prefill overlap accounting.

The retained whole report is a controlled ``s5ro-correlation/2`` sink carrying
a real decode ``EngineOfferLedger`` partition and the source-declared native
prefill partition.  Passing here proves the composer's partition conservation,
never native CUDA evidence.
"""
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'harness'))
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))

import engine_offers  # noqa: E402
import native_placement as np_  # noqa: E402
import native_prefill_support as support  # noqa: E402


def positive(*, extra_copy_families=False):
    bridge = support.FakeBridge()
    torch_module = support.make_torch()
    prefill = support.make_ledger(bridge, torch_module=torch_module)
    support.declare(prefill)
    parts = support.build_lane(bridge, torch_module=torch_module)
    shadows = support.install_lane_seams(parts, prefill)
    events = support.start_role_owners(prefill, torch_module)
    support.submit_all(parts['dispatcher'])
    if extra_copy_families:
        with prefill.segment('state_finalize', transaction=support.TXN,
                             op_seq=100):
            parts['operation'].stage_sparse_selection(None, None)
            parts['operation'].fill_sparse_bucket(None, None, 0, None)
            parts['ple']._finalize_prefetch(None, None)
    decode, decode_binding = support.decode_ledger(bridge)
    support.run_decode_offers(decode, bridge)
    support.finish_role_owners(prefill, events=events[:1])
    prefill.owner_finish('lane', events=events[:1])
    for shadow in reversed(shadows):
        shadow.close()
    bridge.close_window()
    decode_declared, decode_window = support.finish_decode(decode, bridge)
    prefill.finish()
    prefill_receipt = prefill.seal()
    snapshot = prefill.snapshot()
    native = copy.deepcopy(snapshot['native'])
    offers = copy.deepcopy(snapshot['offers'])
    return {
        'bridge': bridge, 'prefill': prefill, 'decode': decode,
        'binding': decode_binding, 'native': native, 'offers': offers,
        'decode_declared': decode_declared, 'decode_window': decode_window,
        'prefill_receipt': prefill_receipt,
        'prefill_declared': prefill_receipt['declared'],
        'prefill_window': prefill_receipt['window'],
    }


def compose(fix, *, native=None, offers=None, decode_declared=None,
            decode_window=None, prefill_declared=None, prefill_window=None,
            **kw):
    return np_.compose_overlap(
        copy.deepcopy(fix['native'] if native is None else native),
        copy.deepcopy(fix['offers'] if offers is None else offers),
        decode_declared=(copy.deepcopy(fix['decode_declared'])
                         if decode_declared is None else copy.deepcopy(
                             decode_declared)),
        decode_window=(copy.deepcopy(fix['decode_window'])
                       if decode_window is None else copy.deepcopy(
                           decode_window)),
        prefill_declared=(copy.deepcopy(fix['prefill_declared'])
                          if prefill_declared is None else copy.deepcopy(
                              prefill_declared)),
        prefill_window=(copy.deepcopy(fix['prefill_window'])
                        if prefill_window is None else copy.deepcopy(
                            prefill_window)),
        **kw)


class PositiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fix = positive()
        cls.receipt = compose(cls.fix)

    def test_composed_verdict_is_pass_with_proven_conservation(self):
        r = self.receipt
        self.assertEqual(r['verdict'], 'PASS', r)
        self.assertEqual(r['failures'], [], r)
        self.assertEqual(r['missing'], [], r)
        self.assertEqual(r['row4'], 'PASS')
        self.assertTrue(r['conservation']['proved'])
        self.assertEqual(r['conservation']['failures'], [])
        self.assertTrue(np_.validate_overlap(r))

    def test_partitions_are_disjoint_and_union_the_whole_report(self):
        r = self.receipt
        whole = r['whole']
        d = r['partitions']['decode']
        p = r['partitions']['prefill']
        self.assertFalse(set(d['offers_partitioned'])
                         & set(p['offers_partitioned']))
        self.assertEqual(whole['offers'],
                         d['offers'] + p['offers'])
        self.assertEqual(whole['activity_records'],
                         d['activity_records'] + p['activity_records'])
        self.assertEqual(whole['activity_records'],
                         sum(whole['by_kind'].values()))
        for kind in ('kernel', 'memcpy', 'memset'):
            self.assertEqual(whole['by_kind'][kind],
                             d['by_kind'][kind] + p['by_kind'][kind])
        self.assertEqual(whole['activity_records'],
                         self.fix['native']['activity_records'])
        self.assertEqual(whole['offers'], len(self.fix['offers']))

    def test_both_partition_receipts_pass_independently(self):
        r = self.receipt
        self.assertEqual(r['decode']['verdict'], 'PASS')
        self.assertEqual(r['prefill']['verdict'], 'PASS')
        self.assertEqual(r['partitions']['decode']['verdict'], 'PASS')
        self.assertEqual(r['partitions']['prefill']['verdict'], 'PASS')
        self.assertEqual(r['decode']['joined_records'], 2)
        self.assertEqual(r['prefill']['joined_records'], 9)
        self.assertTrue(r['partitions']['prefill']['families'][
            'shared_experts']['maybe_forward_async'])
        self.assertEqual(r['partitions']['prefill']['completion'][
            'offers_completed'], 13)

    def test_whole_raw_report_is_retained_and_unmodified(self):
        r = self.receipt
        self.assertTrue(r['raw_unchanged'])
        self.assertEqual(r['whole'],
                         {key: self.fix['native'].get(key) for key in (
                             'schema', 'window', 'complete', 'errors',
                             'overflow', 'open_scopes', 'boundary_records',
                             'activity_records')} |
                         {'offers': len(self.fix['offers']),
                          'by_kind': r['whole']['by_kind']})
        self.assertEqual(self.fix['native']['errors'], 0)
        self.assertEqual(self.fix['native']['overflow'], 0)

    def test_no_heuristic_selection_is_claimed(self):
        notes = ' '.join(self.receipt['notes'])
        self.assertIn('exact offer/external ids', notes)
        self.assertIn('no symbol, context or time heuristic', notes)


class ConservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fix = positive()

    def test_duplicate_activity_row_breaks_multiset(self):
        native = copy.deepcopy(self.fix['native'])
        native['activities'].append(dict(native['activities'][0]))
        out = compose(self.fix, native=native)
        self.assertEqual(out['verdict'], 'FAIL')
        self.assertTrue(any(reason.startswith('conservation:activity')
                            for reason in out['failures']), out['failures'])
        self.assertFalse(out['conservation']['proved'])

    def test_unpartitioned_offer_is_refused(self):
        offers = copy.deepcopy(self.fix['offers'])
        extra = dict(offers[0])
        extra['id'] = max(row['id'] for row in offers) + 1
        offers.append(extra)
        out = compose(self.fix, offers=offers)
        self.assertEqual(out['verdict'], 'FAIL')
        self.assertTrue(any(reason.startswith(
            'conservation:unpartitioned_offer') for reason in out['failures']),
            out['failures'])

    def test_declared_partition_overlap_is_refused(self):
        decode_declared = copy.deepcopy(self.fix['decode_declared'])
        prefill_id = int(next(iter(self.fix['prefill_declared'])))
        decode_declared[prefill_id] = dict(
            next(iter(self.fix['prefill_declared'].values())))
        out = compose(self.fix, decode_declared=decode_declared)
        self.assertEqual(out['verdict'], 'FAIL')
        self.assertIn('conservation:partition_overlap', out['failures'])

    def test_unpartitioned_activity_and_scope_are_refused(self):
        native = copy.deepcopy(self.fix['native'])
        native['activities'].append({
            'correlation_id': 99999, 'kind': 'kernel', 'count': 1,
            'graph_id': 0, 'context_id': support.MAIN_CTX,
            'stream_id': support.MAIN_STREAM, 'copy_kind': 0,
            'name': 'cudaLaunchKernel'})
        out = compose(self.fix, native=native)
        self.assertEqual(out['verdict'], 'FAIL')
        self.assertTrue(any(reason.startswith(
            'conservation:unpartitioned_activity') for reason in out['failures']),
            out['failures'])

        native = copy.deepcopy(self.fix['native'])
        native['scopes'].append({'id': 99999, 'parent': 0,
                                 'thread': support.THREAD,
                                 'system_thread': support.SYSTEM_THREAD,
                                 'action': 'push'})
        out = compose(self.fix, native=native)
        self.assertEqual(out['verdict'], 'FAIL')
        self.assertTrue(any(reason.startswith(
            'conservation:unpartitioned_scope') for reason in out['failures']),
            out['failures'])

    def test_foreign_external_join_is_refused(self):
        native = copy.deepcopy(self.fix['native'])
        native['external'][0]['external_id'] = 99999
        out = compose(self.fix, native=native)
        self.assertEqual(out['verdict'], 'FAIL')
        self.assertTrue(any(reason.startswith(
            'conservation:unpartitioned_external')
            for reason in out['failures']), out['failures'])

    def test_activity_census_mismatch_is_refused(self):
        native = copy.deepcopy(self.fix['native'])
        row = native['activities'].pop()
        cid = row['correlation_id']
        native['apis'] = [api for api in native['apis']
                          if api['correlation_id'] != cid]
        native['external'] = [ext for ext in native['external']
                              if ext['correlation_id'] != cid]
        out = compose(self.fix, native=native)
        self.assertEqual(out['verdict'], 'FAIL')
        self.assertIn('conservation:activity_records_union', out['failures'])


class RawEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fix = positive()

    def test_raw_loss_and_incompleteness_are_never_discharged(self):
        for field, reason in (('errors', 'correlation_errors'),
                              ('overflow', 'correlation_overflow'),
                              ('open_scopes', 'correlation_open_scopes'),
                              ('boundary_records', 'correlation_boundary'),
                              ('complete', 'correlation_incomplete')):
            native = copy.deepcopy(self.fix['native'])
            native[field] = False if field == 'complete' else 1
            out = compose(self.fix, native=native)
            self.assertEqual(out['verdict'], 'INCONCLUSIVE', field)
            self.assertIn(reason, out['missing'], field)
            self.assertTrue(out['conservation']['proved'], field)
            self.assertIn('whole_raw_loss' if field != 'complete'
                          else 'whole_raw_incomplete', out['missing'])

    def test_window_outside_the_whole_report_is_refused(self):
        window = copy.deepcopy(self.fix['prefill_window'])
        window['native_end_ns'] = self.fix['native']['window']['end_ns'] + 1
        out = compose(self.fix, prefill_window=window)
        self.assertTrue(any('prefill_window_differs_from_whole' in reason
                            for reason in out['missing']), out['missing'])

    def test_partition_identity_mismatch_is_fail(self):
        window = copy.deepcopy(self.fix['prefill_window'])
        window['rank'] = 1
        out = compose(self.fix, prefill_window=window)
        self.assertEqual(out['verdict'], 'FAIL')
        self.assertIn('partition_identity_mismatch:rank', out['failures'])

    def test_partition_grade_failure_propagates(self):
        native = copy.deepcopy(self.fix['native'])
        for row in native['activities']:
            if row['kind'] == 'memcpy' and row['stream_id'] == \
                    support.COPY_STREAM:
                row['kind'] = 'kernel'
                row['name'] = 'cudaLaunchKernel'
                for api in native['apis']:
                    if api['correlation_id'] == row['correlation_id']:
                        api['name'] = 'cudaLaunchKernel'
        out = compose(self.fix, native=native)
        self.assertEqual(out['verdict'], 'FAIL')
        self.assertTrue(any(reason.startswith('prefill_grade_fail:')
                            for reason in out['failures']), out['failures'])


class ReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fix = positive()
        cls.receipt = compose(cls.fix)

    def test_prefill_receipt_is_regraded_identically(self):
        out = compose(self.fix,
                      prefill_receipt=self.fix['prefill_receipt'])
        self.assertEqual(out['verdict'], 'PASS', out)
        self.assertNotIn('prefill_receipt_mismatch', out['failures'])

    def test_mutated_partition_receipt_is_rejected(self):
        bad = copy.deepcopy(self.fix['prefill_receipt'])
        bad['verdict'] = 'FAIL'
        out = compose(self.fix, prefill_receipt=bad)
        self.assertEqual(out['verdict'], 'FAIL')
        self.assertTrue('prefill_receipt_invalid' in out['failures']
                        or 'prefill_receipt_mismatch' in out['failures'],
                        out['failures'])

    def _historical(self, verdict='FAIL', failures=('raw_fail',)):
        binding = copy.deepcopy(self.fix['decode_window']['binding'])
        return {
            'schema': engine_offers.SCHEMA,
            'verdict': verdict,
            'failures': list(failures),
            'missing': [] if verdict != 'INCONCLUSIVE' else ['raw_missing'],
            'joins': [], 'joined_records': 0,
            'binding': binding,
            'window': copy.deepcopy(self.fix['decode_window']),
            'native_window': copy.deepcopy(self.fix['native']['window']),
            'eager_green_records': 0, 'eager_other_records': 0,
            'eager_records': 0, 'unjoined_graph_records': 0,
        }

    def test_historical_decode_placement_is_recorded_never_discharged(self):
        historical = self._historical('INCONCLUSIVE', ())
        original = copy.deepcopy(historical)
        out = compose(self.fix, historical_placement=historical)
        self.assertEqual(out['verdict'], 'PASS', out)
        row = out['historical_placement']
        self.assertEqual(row['verdict'], 'INCONCLUSIVE')
        self.assertEqual(row['scope'], 'historical-decode-placement')
        self.assertFalse(row['modified'])
        self.assertEqual(historical, original)

    def test_historical_fail_cannot_be_discharged(self):
        out = compose(self.fix,
                      historical_placement=self._historical('FAIL'))
        self.assertEqual(out['verdict'], 'FAIL')
        self.assertIn('historical_placement_fail', out['failures'])
        self.assertEqual(out['historical_placement']['verdict'], 'FAIL')

    def test_invalid_historical_receipt_is_rejected(self):
        historical = self._historical('INCONCLUSIVE', ())
        historical['schema'] = 'other'
        out = compose(self.fix, historical_placement=historical)
        self.assertEqual(out['verdict'], 'FAIL')
        self.assertIn('historical_placement_invalid', out['failures'])

    def test_pass_receipt_with_mutated_conservation_is_rejected(self):
        bad = copy.deepcopy(self.receipt)
        bad['conservation']['proved'] = False
        with self.assertRaises(np_.OverlapError):
            np_.validate_overlap(bad)

        bad = copy.deepcopy(self.receipt)
        bad['whole']['activity_records'] += 1
        with self.assertRaises(np_.OverlapError):
            np_.validate_overlap(bad)

        bad = copy.deepcopy(self.receipt)
        bad['partitions']['decode']['verdict'] = 'INCONCLUSIVE'
        with self.assertRaises(np_.OverlapError):
            np_.validate_overlap(bad)

    def test_inputs_are_not_mutated(self):
        native = copy.deepcopy(self.fix['native'])
        offers = copy.deepcopy(self.fix['offers'])
        declared = copy.deepcopy(self.fix['decode_declared'])
        out = compose(self.fix, native=native, offers=offers,
                      decode_declared=declared)
        self.assertEqual(native, self.fix['native'])
        self.assertEqual(offers, self.fix['offers'])
        self.assertEqual(declared, self.fix['decode_declared'])
        self.assertEqual(out['verdict'], 'PASS')



class ReviewWindowBindingTests(unittest.TestCase):
    def test_each_partition_must_bind_the_actual_whole_window(self):
        fix = positive()
        for label in ('decode', 'prefill'):
            for bound in ('native_start_ns', 'native_end_ns'):
                window = copy.deepcopy(fix[label + '_window'])
                envelope = window if bound in window else window['binding']
                envelope[bound] += 1 if bound.endswith('start_ns') else -1
                out = compose(fix, **{label + '_window': window})
                self.assertEqual(out['verdict'], 'INCONCLUSIVE', out)
                self.assertNotEqual(out['row4'], 'PASS')
                self.assertIn('compose_failed:' + label + '_window_differs_from_whole', out['missing'])


if __name__ == '__main__':
    unittest.main()
