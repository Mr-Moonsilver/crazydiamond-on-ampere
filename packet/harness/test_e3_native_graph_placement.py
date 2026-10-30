"""CPU contract tests for raw graph proof + both native accounting partitions.

Two layers:

* ``HardwareRegradeTests`` regrade the retained, read-only job13c receipts
  (``positive.json`` / ``ordinary.json``) and mutate deep copies of them.  They
  prove nothing about CUDA; they prove that the composer's predicates hold on
  the actual producer-shaped receipts and that every listed mutation is
  refused.  They skip (never silently pass) when the receipt directory is not
  mounted.
* ``PortableTests`` build a real decode + prefill partition pair from
  ``native_prefill_support`` and show that a graph-less raw receipt or a raw
  FAIL can never compose to PASS, and that the validator is fail-closed.

The mutation fixtures never derive an asserted count from the rows being
checked: every expectation is a property of the unchanged raw receipt (its
recorded expected/other/graph census) or of a deliberately corrupted copy.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'harness'))
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))

import engine_offers as eo  # noqa: E402
import native_graph_placement as ng  # noqa: E402
import native_prefill_support as support  # noqa: E402

RECEIPTS = Path(os.environ.get(
    'FLASH_E3_JOB13C_RECEIPTS',
    '<hub-dir>/agents/flash-next/tmp/vm<vm-id>-job13c-receipts'))
POSITIVE = RECEIPTS / 'positive.json'
ORDINARY = RECEIPTS / 'ordinary.json'


def digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def hw_positive():
    raw = load(POSITIVE)
    return {
        'raw': raw['window_receipt'],
        'native': raw['native_report'],
        'offers': raw['offer_report'],
        'decode_declared': raw['decode_declared'],
        'decode_window': raw['decode_window'],
        'prefill_declared': raw['prefill_receipt']['declared'],
        'prefill_window': raw['prefill_receipt']['window'],
        'whole': raw,
    }


def compose(fix, *, raw=None, native=None, offers=None, decode_declared=None,
            decode_window=None, prefill_declared=None, prefill_window=None):
    return ng.compose_native_placement(
        raw if raw is not None else fix['raw'],
        native if native is not None else fix['native'],
        offers if offers is not None else fix['offers'],
        decode_declared=(decode_declared if decode_declared is not None
                         else fix['decode_declared']),
        decode_window=(decode_window if decode_window is not None
                       else fix['decode_window']),
        prefill_declared=(prefill_declared if prefill_declared is not None
                          else fix['prefill_declared']),
        prefill_window=(prefill_window if prefill_window is not None
                        else fix['prefill_window']))


def mutation(fix, section, mutate):
    """Deep-copy the fixture and apply ``mutate`` to one section."""
    out = {key: copy.deepcopy(value) if key != 'whole' else None
           for key, value in fix.items()}
    mutate(out[section])
    return out


class ValidatorTests(unittest.TestCase):
    def test_rejects_non_receipts(self):
        for value in (None, {}, {'schema': 'x'}, []):
            with self.assertRaises(ng.NativeGraphError):
                ng.validate_native_placement(value)

    def test_rejects_pass_without_ledger_and_raw_unchanged(self):
        receipt = {
            'schema': ng.SCHEMA, 'verdict': 'PASS', 'row4': 'PASS',
            'raw': {'schema': ng.RAW_SCHEMA, 'verdict': 'INCONCLUSIVE',
                    'sha256': '0' * 64, 'unmodified': True},
            'raw_unchanged': False, 'failures': [], 'missing': [],
        }
        with self.assertRaises(ng.NativeGraphError):
            ng.validate_native_placement(receipt)
        receipt['raw_unchanged'] = True
        with self.assertRaises(ng.NativeGraphError):
            ng.validate_native_placement(receipt)

    def test_rejects_pass_with_raw_fail(self):
        receipt = {
            'schema': ng.SCHEMA, 'verdict': 'PASS', 'row4': 'PASS',
            'raw': {'schema': ng.RAW_SCHEMA, 'verdict': 'FAIL',
                    'sha256': '0' * 64, 'unmodified': True},
            'raw_unchanged': True, 'failures': [], 'missing': [],
        }
        with self.assertRaises(ng.NativeGraphError):
            ng.validate_native_placement(receipt)


# --- portable: real ledgers, graph-less raw receipt ------------------------
BANK = {
    'storage_epoch': 1, 'route_generation': 3, 'descriptor_count': 0,
    'split_decode_sms': 44, 'split_requested_sms': 44,
    'green_context_bound': True, 'capture_order_labels': [],
}


def portable_fixture():
    bridge = support.FakeBridge()
    torch_module = support.make_torch()
    prefill = support.make_ledger(bridge, torch_module=torch_module)
    support.declare(prefill)
    parts = support.build_lane(bridge, torch_module=torch_module)
    shadows = support.install_lane_seams(parts, prefill)
    events = support.start_role_owners(prefill, torch_module)
    support.submit_all(parts['dispatcher'])
    decode, _ = support.decode_ledger(bridge)
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
    # Bind the raw receipt's bank to a record whose identity we can recompute.
    # grade_engine compares binding fields, never the bank hash, so the
    # partition regrade is unaffected.
    window = copy.deepcopy(decode_window)
    window['binding']['bank_identity_sha256'] = eo.bank_identity(BANK)
    return {
        'native': copy.deepcopy(snapshot['native']),
        'offers': copy.deepcopy(snapshot['offers']),
        'decode_declared': decode_declared,
        'decode_window': window,
        'prefill_declared': prefill_receipt['declared'],
        'prefill_window': prefill_receipt['window'],
    }


def graphless_raw(binding, native):
    activities = native['activities']
    kernels = sum(row['count'] for row in activities
                  if row['kind'] == 'kernel')
    copies = sum(row['count'] for row in activities
                 if row['kind'] in ('memcpy', 'memset'))
    records = sum(row['count'] for row in activities)
    counts = {key: 0 for key in eo._RAW_COUNT_KEYS}
    counts.update({
        'context_records': 1,
        'kernel_records': kernels,
        'copy_records': copies,
        'records_in_window': records,
        'buffers_requested': 4,
        'buffers_issued': 4,
        'buffers_completed': 4,
        'buffers_issued_in_window': 4,
        'buffers_completed_in_window': 4,
        'graph_passes': 1,
    })
    return {
        'schema': eo.PLACEMENT_SCHEMA,
        'verdict': 'PASS',
        'reasons': [],
        'complete': True,
        'static_structure': {
            'verdict': 'PASS', 'complete': True, 'graph_count': 0,
            'reasons': [], 'graph_slots': [],
        },
        'replay': {
            'verdict': 'PASS',
            'window': {'start_ns': binding['native_start_ns'],
                       'end_ns': binding['native_end_ns'],
                       'route_generation': binding['route_generation'],
                       'flushed': True, 'flush_forced': False},
            'context': {
                'bound': True, 'context_known': True, 'expected_sms': 44,
                'records': 1, 'green_true': 1, 'green_false': 0,
                'sm_known': 1, 'sm_matching': 1, 'sm_mismatched': 0,
                'is_green': True, 'num_multiprocessors': 44,
                'parent_context_present': True, 'duplicate': False,
                'contradictory': False,
            },
            'counts': counts,
            'ring_overflows': {key: 0 for key in eo._RAW_RING_KEYS},
            'mapping': {'graph_ids': 0, 'node_keys': 0, 'node_aliases': 0,
                        'alias_unknown': 0, 'ambiguous_graph_ids': False,
                        'node_overlap': False},
            'context_diagnostics': {},
            'graph_slots': [],
            'flags': {key: False for key in eo._RAW_FLAG_KEYS},
            'complete': True,
            'reasons': [],
        },
        'coverage': {
            'scope': 'exercised descriptors only',
            'descriptor_inventory': [],
            'observed_descriptors': [],
            'replay_qualified': [],
            'unexercised_descriptors': [],
            'declared_target': None,
        },
        'binding': copy.deepcopy(binding['observer_identity']),
        'limits': {'max_graphs': 64},
        'reserved_bytes': 0,
        'bank': copy.deepcopy(BANK),
        'caller_proof': {
            'controller_drained': True,
            'engine_incarnation': binding['engine_incarnation'],
            'relevant_consumers': 2,
            'route_generation': binding['route_generation'],
            'route_generation_matches': True,
            'storage_epoch': binding['storage_epoch'],
            'storage_epoch_matches': True,
            'verified_by_observer': False,
            'worker_incarnation': binding['worker_incarnation'],
            'worker_rank': binding['rank'],
        },
        'source_identities': copy.deepcopy(binding['helper_identity']),
        'limitations': [],
    }


class PortableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fix = portable_fixture()
        cls.raw = graphless_raw(cls.fix['decode_window']['binding'],
                                cls.fix['native'])

    def test_graphless_raw_never_passes_despite_both_partitions(self):
        out = compose(self.fix, raw=copy.deepcopy(self.raw))
        self.assertEqual(out['verdict'], 'INCONCLUSIVE', out)
        self.assertIsNone(out['discharge'])
        self.assertTrue(any(item.startswith('raw_graph_missing:')
                            for item in out['missing']), out['missing'])
        self.assertEqual(out['baseline']['verdict'], 'PASS')
        self.assertTrue(out['baseline']['conservation_proved'])
        self.assertFalse(out['ledger']['proved'])
        self.assertTrue(ng.validate_native_placement(out))
        forged = copy.deepcopy(out)
        forged.update(verdict='PASS', row4='PASS', failures=[], missing=[])
        with self.assertRaises(ng.NativeGraphError):
            ng.validate_native_placement(forged)

    def test_raw_fail_is_never_discharged(self):
        raw = copy.deepcopy(self.raw)
        raw.update(verdict='FAIL', complete=False,
                   reasons=['no_exercised_graph', 'replay_incomplete'])
        raw['replay'].update(verdict='FAIL', complete=False,
                             reasons=['no_exercised_graph'])
        out = compose(self.fix, raw=raw)
        self.assertEqual(out['verdict'], 'FAIL', out)
        self.assertIsNone(out['discharge'])
        self.assertEqual(out['raw']['verdict'], 'FAIL')
        self.assertEqual(out['raw']['sha256'], digest(raw))

    def test_raw_fail_precedence_survives_incomplete_native_partition(self):
        raw=copy.deepcopy(self.raw)
        raw.update(verdict='FAIL',complete=False,reasons=['known_graph_context'])
        native=copy.deepcopy(self.fix['native']);native['complete']=False
        out=compose(self.fix,raw=raw,native=native)
        self.assertEqual(out['verdict'],'FAIL',out)
        self.assertEqual(out['raw']['sha256'],digest(raw))
        self.assertTrue(any(value.startswith('raw_fail:') for value in out['failures']))


@unittest.skipUnless(POSITIVE.exists() and ORDINARY.exists(),
                     'job13c read-only receipts not mounted')
class HardwareRegradeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fix = hw_positive()
        cls.receipt = compose(cls.fix)

    def test_positive_passes_with_both_partitions_and_graph_proof(self):
        r = self.receipt
        self.assertEqual(r['verdict'], 'PASS', (r['failures'], r['missing']))
        self.assertEqual(r['row4'], 'PASS')
        self.assertEqual(r['failures'], [])
        self.assertEqual(r['missing'], [])
        self.assertTrue(ng.validate_native_placement(r))
        self.assertEqual(r['baseline']['verdict'], 'PASS')
        self.assertTrue(r['baseline']['conservation_proved'])
        for name in ('decode', 'prefill'):
            self.assertEqual(r['partitions'][name]['verdict'], 'PASS')
            self.assertGreater(r['partitions'][name]['records'], 0)
        self.assertEqual(r['graph']['count'], 1)
        self.assertEqual(r['graph']['target'], 'graph:0')
        self.assertEqual(r['graph']['observed'], ['graph:0'])
        self.assertEqual(r['graph']['router'], {'0': 1})
        self.assertEqual(r['graph']['by_kind'],
                         {'kernel': 1386, 'memcpy': 26, 'memset': 7})
        self.assertEqual(r['reconciliation']['graph_records'], 1419)
        self.assertEqual(r['reconciliation']['graph_by_offer'],
                         {'22': {'kernel': 1386, 'memcpy': 26,
                                 'memset': 7}})
        self.assertEqual(r['reconciliation']['offers_by_label'],
                         {'graph:0': 1})
        self.assertEqual(r['reconciliation']['records'], 1459)

    def test_only_the_documented_derived_inconclusive_is_discharged(self):
        r = self.receipt
        self.assertEqual(r['raw']['verdict'], 'INCONCLUSIVE')
        self.assertEqual(r['raw']['native_reasons'],
                         ['unattributed_expected_context'])
        self.assertEqual(r['raw']['reasons'],
                         ['unattributed_expected_context', 'replay_incomplete'])
        self.assertEqual(r['discharge']['origin'],
                         'discharged_eager_expected_context')
        self.assertEqual(r['discharge']['derived_reasons'],
                         ['replay_incomplete'])
        self.assertEqual(r['discharge']['expected_context_records'], 7)
        self.assertEqual(r['discharge']['other_context_records'], 33)
        self.assertEqual(r['discharge']['graph_records'], 1419)
        self.assertEqual(
            r['reconciliation']['expected_by_kind'],
            {'kernel': 3, 'memcpy': 3, 'memset': 1})
        self.assertEqual(
            r['reconciliation']['other_by_kind'],
            {'kernel': 28, 'memcpy': 5, 'memset': 0})

    def test_every_graphidzero_record_has_exactly_one_joined_owner(self):
        rows = [row for row in self.receipt['ledger']['rows']
                if row['graph_id'] == 0]
        self.assertEqual(len(rows), 40)
        self.assertEqual(sum(row['count'] for row in rows), 40)
        self.assertEqual(
            {row['partition'] for row in rows}, {'decode', 'prefill'})
        for row in rows:
            self.assertIsInstance(row['offer'], int)
            self.assertIn(row['partition'], ('decode', 'prefill'))
            self.assertEqual(row['count'], 1)
        expected = [row for row in rows
                    if row['context_id'] == 2]
        other = [row for row in rows if row['context_id'] != 2]
        self.assertEqual(len(expected), 7)
        self.assertEqual(len(other), 33)
        self.assertEqual({row['partition'] for row in expected}, {'decode'})
        self.assertEqual({row['partition'] for row in other}, {'prefill'})
        self.assertIn('copy', {row['role'] for row in other})

    def test_raw_receipt_and_inputs_are_unmodified(self):
        before = digest(self.fix['whole'])
        expected_raw_sha = digest(self.fix['raw'])
        out = compose(self.fix)
        again = compose(self.fix)
        self.assertEqual(out['raw']['sha256'], expected_raw_sha)
        self.assertTrue(out['raw']['unmodified'])
        self.assertTrue(out['raw_unchanged'])
        self.assertEqual(before, digest(self.fix['whole']))
        self.assertEqual(out['baseline']['sha256'],
                         again['baseline']['sha256'])
        for key in ('verdict', 'reasons', 'complete'):
            self.assertEqual(out['raw'][key], self.fix['raw'][key])

    def test_graphless_ordinary_raw_is_not_dischargeable(self):
        ordinary = load(ORDINARY)['window_receipt']
        out = compose(self.fix, raw=ordinary)
        self.assertNotEqual(out['verdict'], 'PASS', out)
        self.assertIsNone(out['discharge'])
        self.assertNotEqual(out['raw']['verdict'], 'FAIL')
        self.assertTrue(out['missing'] or out['failures'], out)


class HardwareMutationTests(HardwareRegradeTests):
    """Every listed mutation must never compose to PASS."""

    def assertRefused(self, out, *, verdict=None, reason=None):
        self.assertNotEqual(out['verdict'], 'PASS', out)
        self.assertIsNone(out['discharge'], out)
        if verdict is not None:
            self.assertEqual(out['verdict'], verdict, out)
        if reason is not None:
            self.assertTrue(
                any(item.startswith(reason)
                    for item in out['failures'] + out['missing']),
                (reason, out['failures'], out['missing']))

    def test_extra_eager_row_is_a_ledger_violation(self):
        fix = mutation(self.fix, 'native', lambda native: (
            native['activities'][0].__setitem__('count', 2),
            native.__setitem__('activity_records',
                               native['activity_records'] + 1)))
        out = compose(fix)
        self.assertEqual(out['baseline']['verdict'], 'PASS',
                         'the baseline must not catch this by itself')
        self.assertRefused(out, verdict='FAIL',
                           reason='ledger_violation:raw_other_mismatch')

    def test_raw_expected_count_mismatch(self):
        fix = mutation(self.fix, 'raw', lambda raw: raw['replay']['counts']
                       .__setitem__('unattributed_expected_context', 8))
        self.assertRefused(compose(fix), verdict='FAIL',
                           reason='raw_graph_violation:raw_partition')

    def test_wrong_native_window(self):
        fix = mutation(self.fix, 'raw', lambda raw: raw['replay']['window']
                       .__setitem__('start_ns', raw['replay']['window']
                                    ['start_ns'] + 1))
        self.assertRefused(compose(fix), verdict='FAIL',
                           reason='raw_identity_violation:window_start_mismatch')

    def test_wrong_rank(self):
        fix = mutation(self.fix, 'raw', lambda raw: raw['caller_proof']
                       .__setitem__('worker_rank', 1))
        self.assertRefused(compose(fix), verdict='FAIL',
                           reason='raw_identity_violation:proof_rank_mismatch')

    def test_wrong_engine_epoch_route_and_storage(self):
        for key, value in (('engine_incarnation', 'other-engine'),
                           ('route_generation', 2),
                           ('storage_epoch', 2)):
            fix = mutation(self.fix, 'raw',
                           lambda raw, k=key, v=value:
                           raw['caller_proof'].__setitem__(k, v))
            self.assertRefused(compose(fix), verdict='FAIL')
        fix = mutation(self.fix, 'raw', lambda raw: raw['caller_proof']
                       .__setitem__('storage_epoch_matches', False))
        self.assertRefused(compose(fix), verdict='FAIL',
                           reason='raw_identity_violation:'
                                  'proof_storage_unmatched')

    def test_wrong_retained_bank(self):
        fix = mutation(self.fix, 'raw', lambda raw: raw['bank']
                       .__setitem__('route_generation', 2))
        self.assertRefused(compose(fix), verdict='FAIL',
                           reason='raw_identity_violation:'
                                  'bank_identity_mismatch')

    def test_wrong_declared_target(self):
        fix = mutation(self.fix, 'raw', lambda raw: raw['coverage']
                       .__setitem__('declared_target', 'descriptor:1'))
        self.assertRefused(compose(fix), verdict='FAIL',
                           reason='raw_graph_violation:declared_target')

    def test_per_node_multiplicity_loss_despite_node_seen(self):
        fix = mutation(self.fix, 'raw', lambda raw: raw['replay']
                       ['graph_slots'][0].__setitem__('kernel_executions', 1))
        self.assertRefused(compose(fix), verdict='FAIL',
                           reason='raw_graph_violation:slot_executions')
        fix = mutation(self.fix, 'raw', lambda raw: raw['replay']
                       ['graph_slots'][0]['node_execution_bounds']
                       .__setitem__('kernel', {'min': 0, 'max': 1}))
        self.assertRefused(compose(fix), verdict='FAIL',
                           reason='raw_graph_violation:slot_multiplicity')

    def test_wrong_replay_census(self):
        fix = mutation(self.fix, 'decode_window', lambda window: window
                       .__setitem__('binding', dict(
                           window['binding'],
                           router_census={'total': 2,
                                          'graphs': {'graph:0': 2}})))
        self.assertRefused(compose(fix), verdict='FAIL')

    def test_raw_fail(self):
        def fail(raw):
            raw.update(verdict='FAIL', complete=False,
                       reasons=['no_exercised_graph', 'replay_incomplete'])
            raw['replay'].update(verdict='FAIL', complete=False,
                                 reasons=['no_exercised_graph'])
        self.assertRefused(compose(mutation(self.fix, 'raw', fail)),
                           verdict='FAIL', reason='raw_fail:')

    def test_context_and_sm_contradictions(self):
        fix = mutation(self.fix, 'raw', lambda raw: raw['replay']['context']
                       .update(sm_mismatched=1, sm_matching=0))
        self.assertRefused(compose(fix), verdict='FAIL',
                           reason='raw_known_violation:'
                                  'expected_context_sm_mismatch')
        fix = mutation(self.fix, 'raw', lambda raw: raw['replay']
                       ['graph_slots'][0].__setitem__('context_mismatch_nodes',
                                                      ['node']))
        self.assertRefused(compose(fix), verdict='FAIL',
                           reason='raw_graph_violation:slot_context_mismatch')

    def test_sticky_loss_flags_and_ring_overflow(self):
        fix = mutation(self.fix, 'raw', lambda raw: raw['replay']['flags']
                       .__setitem__('drop_seen', True))
        self.assertRefused(compose(fix), verdict='FAIL',
                           reason='raw_graph_violation:raw_flag_drop_seen')
        fix = mutation(self.fix, 'raw', lambda raw: raw['replay']
                       ['ring_overflows'].__setitem__('kernel', 1))
        self.assertRefused(compose(fix), verdict='FAIL',
                           reason='raw_graph_violation:ring_overflow_kernel')

    def test_unsupported_or_unknown_node_evidence(self):
        fix = mutation(self.fix, 'raw', lambda raw: raw['static_structure']
                       ['graph_slots'][0].__setitem__('unsupported', 1))
        self.assertRefused(compose(fix), verdict='FAIL',
                           reason='raw_graph_violation:static_unsupported')
        fix = mutation(self.fix, 'raw', lambda raw: raw['static_structure']
                       ['graph_slots'][0].__setitem__('unknown_type', 1))
        self.assertRefused(compose(fix), verdict='FAIL')
        fix = mutation(self.fix, 'native', lambda native: native['activities']
                       [0].__setitem__('kind', 'host'))
        self.assertRefused(compose(fix), verdict='FAIL')

    def test_wrong_context_stream_in_declared_offer(self):
        fix = mutation(self.fix, 'native', lambda native: native['activities']
                       [0].__setitem__('context_id', 9))
        self.assertRefused(compose(fix), verdict='FAIL')

    def test_no_prefill_partition(self):
        fix = mutation(self.fix, 'prefill_declared', lambda declared:
                       declared.clear())
        out = compose(fix)
        self.assertRefused(out, verdict='INCONCLUSIVE',
                           reason='baseline:compose_failed:'
                                  'prefill_declared_empty')

    def test_raw_pass_with_positive_expected_context(self):
        def pass_raw(raw):
            raw.update(verdict='PASS', complete=True, reasons=[])
            raw['replay'].update(verdict='PASS', complete=True, reasons=[])
            raw['coverage']['replay_qualified'] = ['graph:0']
            raw['replay']['graph_slots'][0]['replay_qualified'] = True
        self.assertRefused(compose(mutation(self.fix, 'raw', pass_raw)),
                           verdict='FAIL',
                           reason='raw_graph_violation:'
                                  'raw_pass_with_expected')

    def test_non_dischargeable_inconclusive(self):
        def inc(raw):
            raw.update(verdict='INCONCLUSIVE', complete=False,
                       reasons=['no_exercised_graph', 'replay_incomplete'])
            raw['replay'].update(verdict='INCONCLUSIVE', complete=False,
                                 reasons=['no_exercised_graph'])
        self.assertRefused(compose(mutation(self.fix, 'raw', inc)),
                           verdict='INCONCLUSIVE',
                           reason='raw_not_dischargeable:no_exercised_graph')


if __name__ == '__main__':
    unittest.main()
