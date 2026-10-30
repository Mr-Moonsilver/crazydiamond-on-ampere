#!/usr/bin/env python3
"""CPU-only tests for the q18a real E3 overlap driver and grader.

No GPU, no engine, no box access.  The integration cases use a loopback fake
SSE completion server and an injected controller that returns schema-1
receipts.  They prove the client-side gate behaves as declared:

  * true concurrent decode progress during accepted prefill is accepted;
  * a serialized execution that stalls decode during prefill is refused and
    its invalid interval is retained;
  * useful-work accounting without a completed prefill cannot pass;
  * a controller transport failure is marked as an engine crash;
  * grading refuses wrong baselines, zero denominators, counter resets,
    missing/stale proofs, memory breach, cleanup that restores production,
    and any target label that is missing or contradictory.
"""
import json
import copy
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import e3_overlap_grade as grading
import e3_overlap_performance as performance

SHA_A = 'a' * 64
SHA_B = 'b' * 64
SHA_W = 'c' * 64
PACKAGE = 'd' * 64
LAUNCHER = 'e' * 64
LABEL = 's5-overlap-80-80'


# ---------------------------------------------------------------------------
# grade fixtures
# ---------------------------------------------------------------------------

def _rank(index, engine_epoch=7):
    return {
        'rank': index,
        'ownership': {'worker_incarnation': 'w1', 'engine_epoch': engine_epoch,
                      'storage_epoch': 3, 'route_generation': 11},
        'copy': {'buckets': 2, 'bytes': 4096, 'events': 2,
                 'completion': 'queried'},
        'attention': {'ops': 4, 'complete': True},
        'state_write': {'proof': 'queried_event', 'queried': True,
                        'writers': ['layer-0', 'layer-23'],
                        'joined_streams': 8},
        'pp': {'receive': True, 'consume': True},
        'replay': {'submitted_before': 10, 'submitted_after': 20,
                   'completed_before': 9, 'completed_after': 19},
    }


def segment(**overrides):
    value = {
        'segment_seq': 1, 'start_monotonic_ns': 1000,
        'end_monotonic_ns': 2000, 'accepted': True,
        'path': 'e3_overlap', 'copy_route': 'dense', 'decode_arrivals': 50,
        'committed_tokens_before': 0, 'committed_tokens_after': 200000,
        'useful_tokens_before': 0, 'useful_tokens_after': 200000,
        'prefill_completed': True,
        'ranks': [_rank(index) for index in range(4)],
    }
    value.update(overrides)
    return value


def overlap(**overrides):
    value = {
        'valid': True, 'engine_crash': False, 'workload_sha256': SHA_W,
        'worker_incarnation': 'w1', 'engine_epoch': 7, 'storage_epoch': 3,
        'route_generation': 11, 'segments': [segment()],
        'owners': {str(rank): dict(_rank(rank)['ownership']) for rank in range(4)},
        'invalid_intervals': [],
        'summary_basis': {'filtered_invalid': False, 'retry_selected': False,
                          'picked': False,
                          'all_declared_repeats_retained': True},
        'claims': {'speed_claim': False},
        'interval_counters': {'reset_count': 0},
        'useful_completed_tokens': 200000, 'prompt_tokens': 200000,
        'prefill_wall_s': 2000.0, 'prefill_tokens_per_s': 100.0,
        'completed_prefills': 1, 'decode_tokens_per_s': 90.0,
        'gap_p99_ms': 2.0, 'fresh_ttft_s': 10.0, 'append_ttft_s': 2.0,
        'stream_rates_tokens_per_s': [90.0] * 8,
        'valid_completed_turns': 8, 'cancellations': 0, 'errors': 0,
        'target_label': LABEL,
    }
    value.update(overrides)
    return value


def baseline(route, decode=None, prefill=None, valid=True):
    row = {'route': route, 'valid': valid,
           'serving_baseline': route == 'full',
           'workload_sha256': SHA_W}
    if decode is not None:
        row['decode_tokens_per_s'] = decode
    if prefill is not None:
        row['prefill_tokens_per_s'] = prefill
    return row


def evidence(**overrides):
    value = {
        'schema': 1, 'target_label': LABEL, 'overlap': overlap(),
        'baselines': {
            'w1_full': baseline('full', decode=100.0),
            'w1_split': baseline('split', decode=60.0),
            'prefill_full': baseline('full', prefill=120.0),
            'prefill_split': baseline('split', prefill=100.0),
        },
        'memory': {
            'reserve_mib': 512,
            'samples': [{'pressure': True,
                         'gpus': [{'index': index, 'free_mib': 600.0}
                                  for index in range(4)]}],
            'allocator_peaks': {'gpu'+str(rank): 1234 for rank in range(4)},
        },
        'cleanup': {
            'engine_state': 'inactive', 'engine_stopped': True,
            'production_restored': False, 'automatic_back': False,
            'restore_withheld': False,
            'launcher_found_sha256': SHA_A,
            'launcher_restored_sha256': SHA_A,
            'defaults_found_sha256': SHA_B,
            'defaults_restored_sha256': SHA_B,
        },
    }
    value.update(overrides)
    return value


class GradeTests(unittest.TestCase):
    def test_w1_baseline_does_not_select_only_valid_survivors(self):
        raw = w1_fixture('full', 100)
        bad = copy.deepcopy(raw['repeats'][0])
        bad['valid'] = False
        raw['repeats'].append(bad)
        result = grading._w1_route(raw, 'full')
        self.assertFalse(result['valid'])
        self.assertEqual(result['invalid_repeats'], 1)

    def test_real_w1_gap_field_is_retained(self):
        raw = w1_fixture('full', 100)
        row = raw['summary']['routes']['full']
        row.pop('gap_p99_ms')
        row['gap_ms_p99'] = 4.5
        self.assertEqual(grading._w1_route(raw, 'full')['gap_p99_ms'], 4.5)

    def test_serialized_transport_cannot_be_counted_as_accepted_overlap(self):
        data = evidence()
        data['overlap']['segments'][0]['copy_route'] = 'serialized'
        result = grading.grade(data)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertTrue(any('unqualified_transport_admitted_to_overlap' in s for s in result['failures']))

    def test_missing_gpu_in_pressure_sample_cannot_pass(self):
        data = evidence()
        data['memory']['samples'][0]['gpus'].pop()
        result = grading.grade(data)
        self.assertNotEqual(result['verdict'], 'PASS', result)

    def test_replay_completion_cannot_exceed_submission(self):
        data = evidence()
        data['overlap']['segments'][0]['ranks'][0]['replay']['completed_after'] = 1000
        result = grading.grade(data)
        self.assertEqual(result['verdict'], 'FAIL', result)

    def test_segment_cannot_override_authoritative_owner_with_its_own_top(self):
        data = evidence()
        foreign = dict(worker_incarnation='foreign', engine_epoch=7,
                       storage_epoch=3, route_generation=11)
        data['overlap']['segments'][0]['top'] = {'owner': foreign}
        for row in data['overlap']['segments'][0]['ranks']:
            row['ownership'] = dict(foreign)
        result = grading.grade(data)
        self.assertEqual(result['verdict'], 'FAIL', result)

    def test_grade_preserves_the_raw_receipt(self):
        data = evidence()
        original = copy.deepcopy(data)
        grading.grade(data)
        self.assertEqual(data, original)

    def test_complete_evidence_passes(self):
        result = grading.grade(evidence())
        self.assertEqual(result['verdict'], 'PASS', result)
        self.assertEqual(result['routes']['accepted_segments'], 1)
        self.assertEqual(result['metrics']['decode_share_of_full'], 0.9)

    def test_wrong_baseline_route_fails(self):
        data = evidence()
        data['baselines']['w1_full'] = baseline('split', decode=100.0)
        result = grading.grade(data)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('wrong_baseline_route:w1_full=\'split\'',
                      result['failures'])

    def test_split_baseline_claimed_as_serving_fails(self):
        data = evidence()
        data['baselines']['w1_split']['serving_baseline'] = True
        result = grading.grade(data)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('wrong_baseline_route:w1_split_claimed_serving',
                      result['failures'])

    def test_w1_route_context_mismatch_fails(self):
        data = evidence()
        data['baselines']['w1_split']['workload_sha256'] = SHA_A
        result = grading.grade(data)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('baseline_context_mismatch:w1', result['failures'])

    def test_overlap_context_mismatch_fails(self):
        data = evidence()
        data['overlap']['workload_sha256'] = SHA_A
        result = grading.grade(data)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('baseline_context_mismatch:overlap_lane',
                      result['failures'])

    def test_zero_decode_denominator_is_inconclusive(self):
        data = evidence()
        data['baselines']['w1_full'] = baseline('full', decode=0.0)
        result = grading.grade(data)
        self.assertEqual(result['verdict'], 'INCONCLUSIVE')
        self.assertIn('zero_decode_denominator', result['missing'])

    def test_zero_prefill_denominator_is_inconclusive(self):
        data = evidence()
        data['baselines']['prefill_split'] = baseline('split', prefill=0.0)
        result = grading.grade(data)
        self.assertEqual(result['verdict'], 'INCONCLUSIVE')
        self.assertIn('zero_prefill_denominator', result['missing'])

    def test_serialized_replay_progress_is_refused(self):
        data = evidence()
        row = data['overlap']['segments'][0]['ranks'][2]
        row['replay']['submitted_after'] = row['replay']['submitted_before']
        row['replay']['completed_after'] = row['replay']['completed_before']
        result = grading.grade(data)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertTrue(any('serialized_false_positive' in item
                            for item in result['failures']), result['failures'])

    def test_engine_side_zero_decode_arrivals_is_refused(self):
        data = evidence()
        data['overlap']['segments'][0]['decode_arrivals'] = 0
        result = grading.grade(data)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertTrue(any('serialized_false_positive' in item
                            for item in result['failures']))

    def test_missing_prefill_completion_is_inconclusive(self):
        data = evidence()
        row = data['overlap']
        row['useful_completed_tokens'] = 1000
        row['prompt_tokens'] = 200000
        row['completed_prefills'] = 0
        row['segments'][0]['useful_tokens_after'] = 1000
        row['segments'][0]['committed_tokens_after'] = 1000
        row['segments'][0]['prefill_completed'] = False
        result = grading.grade(data)
        self.assertEqual(result['verdict'], 'INCONCLUSIVE')
        self.assertIn('prefill_not_completed', result['missing'])

    def test_offered_only_work_is_refused(self):
        data = evidence()
        row = data['overlap']
        row['segments'][0]['useful_tokens_after'] = 0
        row['useful_completed_tokens'] = 0
        result = grading.grade(data)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertTrue(any('near_tie_envelope_offered_only' in item
                            for item in result['failures']))

    def test_invalid_intervals_are_retained(self):
        data = evidence()
        data['overlap']['valid'] = False
        data['overlap']['invalid_intervals'] = [
            {'reasons': ['open_incomplete']}]
        result = grading.grade(data)
        self.assertEqual(result['verdict'], 'INCONCLUSIVE')
        self.assertIn('overlap_lane_not_valid', result['missing'])
        self.assertEqual(result['coverage']['accepted'], 1)
        self.assertEqual(data['overlap']['invalid_intervals'],
                         [{'reasons': ['open_incomplete']}])

    def test_filtering_invalid_intervals_fails(self):
        data = evidence()
        data['overlap']['summary_basis']['filtered_invalid'] = True
        result = grading.grade(data)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('invalid_intervals_filtered_from_result',
                      result['failures'])

    def test_counter_reset_is_refused(self):
        data = evidence()
        data['overlap']['segments'][0]['committed_tokens_after'] = 10
        data['overlap']['segments'][0]['committed_tokens_before'] = 100
        result = grading.grade(data)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertTrue(any(item.startswith('counter_reset:segment')
                            for item in result['failures']))

    def test_interval_counter_reset_is_refused(self):
        data = evidence()
        data['overlap']['interval_counters'] = {'reset_count': 1}
        result = grading.grade(data)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('counter_reset:interval_counters', result['failures'])

    def test_missing_rank_proof_is_inconclusive(self):
        data = evidence()
        del data['overlap']['segments'][0]['ranks'][1]['copy']
        result = grading.grade(data)
        self.assertEqual(result['verdict'], 'INCONCLUSIVE')
        self.assertTrue(any('missing_proof:rank1' in item
                            for item in result['missing']))

    def test_stale_ownership_proof_is_refused(self):
        data = evidence()
        data['overlap']['segments'][0]['top'] = {
            'owner': {'worker_incarnation': 'w1', 'engine_epoch': 8,
                      'storage_epoch': 3, 'route_generation': 11}}
        result = grading.grade(data)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertTrue(any('stale_proof:rank0.seg1.ownership.engine_epoch'
                            in item for item in result['failures']))

    def test_512mib_reserve_breach_is_refused(self):
        data = evidence()
        data['memory']['samples'][0]['gpus'][2]['free_mib'] = 511.0
        result = grading.grade(data)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('512MiB_reserve_breached:gpu2:511.0', result['failures'])

    def test_reserve_change_is_refused(self):
        data = evidence()
        data['memory']['reserve_mib'] = 1024
        result = grading.grade(data)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertTrue(any(item.startswith('reserve_changed:')
                            for item in result['failures']))

    def test_missing_pressure_samples_is_inconclusive(self):
        data = evidence()
        data['memory']['samples'] = [{'pressure': False, 'gpus': []}]
        result = grading.grade(data)
        self.assertEqual(result['verdict'], 'INCONCLUSIVE')
        self.assertIn('memory_pressure_samples_missing', result['missing'])

    def test_engine_crash_partial_evidence_is_inconclusive(self):
        data = evidence()
        data['overlap']['engine_crash'] = True
        result = grading.grade(data)
        self.assertEqual(result['verdict'], 'INCONCLUSIVE')
        self.assertIn('engine_crash_partial_evidence', result['missing'])

    def test_cleanup_restore_withheld_is_refused(self):
        data = evidence()
        data['cleanup']['restore_withheld'] = True
        data['cleanup']['launcher_restored_sha256'] = None
        result = grading.grade(data)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('cleanup_restore_withheld_under_live_engine',
                      result['failures'])

    def test_production_restored_is_refused(self):
        data = evidence()
        data['cleanup']['production_restored'] = True
        result = grading.grade(data)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('production_restored_without_operator',
                      result['failures'])

    def test_stock_forward_fallback_is_measured_not_accepted(self):
        data = evidence()
        data['overlap']['segments'].append({
            'segment_seq': 2, 'start_monotonic_ns': 2000,
            'end_monotonic_ns': 3000, 'accepted': False,
            'stock_forward_used': True, 'path': 'stock_forward',
            'decode_arrivals': 0, 'prefill_completed': False})
        result = grading.grade(data)
        self.assertEqual(result['verdict'], 'PASS', result)
        self.assertEqual(result['routes']['stock_forward_fallback'], 1)
        self.assertEqual(result['routes']['accepted_segments'], 1)

    def test_hidden_stock_forward_as_overlap_is_refused(self):
        data = evidence()
        data['overlap']['segments'][0]['stock_forward_used'] = True
        result = grading.grade(data)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertTrue(any(
            'whole_stock_forward_fallback_claimed_as_overlap' in item
            for item in result['failures']))

    def test_target_label_is_required(self):
        data = evidence()
        data.pop('target_label')
        data['overlap'].pop('target_label')
        result = grading.grade(data)
        self.assertEqual(result['verdict'], 'INCONCLUSIVE')
        self.assertIn('target_label_required', result['missing'])

    def test_unknown_target_label_is_refused(self):
        data = evidence()
        data['target_label'] = 'hand-picked'
        result = grading.grade(data)
        self.assertEqual(result['verdict'], 'INCONCLUSIVE')
        self.assertIn('unknown_target_label', result['missing'])

    def test_target_definition_conflict_is_exposed(self):
        data = evidence()
        data['target'] = {'label': LABEL, 'decode_share_min': 0.95}
        result = grading.grade(data)
        self.assertEqual(result['verdict'], 'INCONCLUSIVE')
        self.assertIn('target_definition_conflict:decode_share_min',
                      result['missing'])

    def test_target_shortfall_is_retained(self):
        data = evidence()
        data['baselines']['w1_full'] = baseline('full', decode=200.0)
        result = grading.grade(data)
        self.assertEqual(result['verdict'], 'PASS')
        self.assertFalse(result['target_met'])
        self.assertEqual(result['target_status'], 'SHORTFALL')
        self.assertTrue(any(item.startswith('decode_share_below_target:')
                            for item in result['shortfall']))

    def test_readiness_not_pass_is_refused(self):
        data = evidence()
        data['readiness'] = {'verdict': 'INCONCLUSIVE'}
        result = grading.grade(data)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('candidate_readiness_not_pass:INCONCLUSIVE',
                      result['failures'])

    def test_direct_target_policies_do_not_mix(self):
        self.assertTrue(grading.target_definition('s5-w2-target')['ok'])
        self.assertTrue(grading.target_definition('s5-overlap-80-80')['ok'])
        conflict = grading.target_definition(
            's5-overlap-80-80', {'decode_share_min': 0.95,
                                 'prefill_denominator': 'prefill_full'})
        self.assertFalse(conflict['ok'])
        self.assertIn('target_definition_conflict:decode_share_min',
                      conflict['problems'])
        self.assertIn('target_definition_conflict:prefill_denominator',
                      conflict['problems'])


# ---------------------------------------------------------------------------
# candidate readiness
# ---------------------------------------------------------------------------

def readiness_receipt(trusted=LAUNCHER, package=PACKAGE):
    receipt = {
        'reviewed': True, 'reviewed_by': 'owner-review',
        'require_native_ready': True, 'native_ready': True,
        'full_native_overlap': True, 'sparse_dma_lease': True,
        'armable': True, 'package_sha256': package,
        'launcher_sha256': trusted,
        'path': 'e3/native-overlap-readiness.json',
        'content_sha256': SHA_A,
    }
    receipt['receipt_sha256'] = grading._receipt_digest(receipt)
    return receipt


def generation(**overrides):
    value = {'armable': True, 'require_native_ready': True,
             'default_feature': 'FLASH_S5_OVERLAP=0',
             'package_sha256': PACKAGE,
             'native_overlap': {'enabled': True, 'sparse_dma_lease': True,
                                'stock_forward_fallback_hidden': False}}
    value.update(overrides)
    return value


class ReadinessTests(unittest.TestCase):
    def test_review_metadata_cannot_enable_the_feature_by_default(self):
        result = self.call(generation(default_feature='FLASH_S5_OVERLAP=1'), readiness_receipt())
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('package_default_not_off', result['failures'])

    def call(self, gen, ready, **kwargs):
        entries = kwargs.pop('manifest_entries',
                             {'e3/native-overlap-readiness.json': SHA_A})
        return grading.candidate_readiness(
            gen, ready, launcher_sha=kwargs.pop('launcher_sha', LAUNCHER),
            package_sha256=kwargs.pop('package_sha256', PACKAGE),
            manifest_entries=entries,
            env_launcher_sha=kwargs.pop('env_launcher_sha', None),
            trusted_launcher_sha=LAUNCHER)

    def test_reviewed_package_passes(self):
        result = self.call(generation(), readiness_receipt())
        self.assertEqual(result['verdict'], 'PASS', result)

    def test_armable_false_is_refused(self):
        result = self.call(generation(armable=False), readiness_receipt())
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('candidate_not_armable:armable_false',
                      result['failures'])

    def test_missing_require_native_ready_is_refused(self):
        gen = generation()
        del gen['require_native_ready']
        result = self.call(gen, readiness_receipt())
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('require_native_ready_not_declared', result['failures'])

    def test_bypass_is_refused(self):
        result = self.call(generation(bypass_require_native_ready=True),
                           readiness_receipt())
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('require_native_ready_bypass_requested',
                      result['failures'])

    def test_fabricated_receipt_digest_is_refused(self):
        ready = readiness_receipt()
        ready['receipt_sha256'] = SHA_B
        result = self.call(generation(), ready)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('readiness_receipt_digest_differs', result['failures'])

    def test_unreviewed_receipt_is_inconclusive(self):
        ready = readiness_receipt()
        ready['reviewed'] = False
        ready['receipt_sha256'] = grading._receipt_digest(
            {key: value for key, value in ready.items()
             if key != 'receipt_sha256'})
        result = self.call(generation(), ready)
        self.assertEqual(result['verdict'], 'INCONCLUSIVE')
        self.assertIn('readiness.review_not_proved', result['missing'])

    def test_untracked_receipt_is_refused(self):
        result = self.call(generation(), readiness_receipt(),
                           manifest_entries={'other.json': SHA_A})
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('readiness_receipt_not_in_package_manifest',
                      result['failures'])

    def test_env_launcher_pin_is_refused(self):
        result = self.call(generation(), readiness_receipt(),
                           env_launcher_sha=SHA_B)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('arbitrary_env_launcher_pin_refused', result['failures'])

    def test_untrusted_launcher_is_refused(self):
        result = self.call(generation(launcher={'name': 'x'}),
                           readiness_receipt(), launcher_sha=SHA_B)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('candidate_launcher_not_trusted_pin', result['failures'])

    def test_package_sha_mismatch_is_refused(self):
        ready = readiness_receipt(package=SHA_B)
        result = self.call(generation(), ready)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('package_sha_mismatch:readiness', result['failures'])


# ---------------------------------------------------------------------------
# manifest validation
# ---------------------------------------------------------------------------

def manifest(**overrides):
    value = {
        'schema': 1, 'workload': 'q18a-test', 'target_label': LABEL,
        'decode': {'streams': 8, 'prompt_tokens': 153537,
                   'generation_budget': 8192, 'seed': 1,
                   'open_timeout_s': 900.0,
                   'sampling': {'seed': 1, 'temperature': 0.7, 'top_p': 1.0,
                                'ignore_eos': True, 'min_tokens': 'equal_max'}},
        'prefill': {'fresh': {'prompt_tokens': 200000,
                              'generation_budget': 1, 'seed': 2},
                    'append': {'prompt_tokens': 4096,
                               'generation_budget': 1, 'seed': 7}},
        'timeouts': {'controller_s': 120.0, 'read_s': 180.0,
                     'drain_s': 120.0, 'join_s': 120.0, 'finish_s': 120.0},
        'window_s': 20.0, 'repeats': 1, 'reserve_mib': 512,
        'route_plan': ['prefill_full', 'prefill_split', 'overlap'],
        'claims': {'speed_claim': False},
    }
    value.update(overrides)
    return value


class ManifestTests(unittest.TestCase):
    def test_declared_manifest_is_valid(self):
        problems, cfg = performance.validate_manifest(manifest())
        self.assertEqual(problems, [])
        self.assertEqual(cfg['route_plan'],
                         ('prefill_full', 'prefill_split', 'overlap'))
        self.assertEqual(cfg['reserve_mib'], 512)

    def test_repository_manifest_is_valid(self):
        path = performance.Path(__file__).with_name('workloads') / \
            'q18a-w2-overlap.json'
        problems, cfg = performance.validate_manifest(
            json.loads(path.read_text()))
        self.assertEqual(problems, [])
        self.assertEqual(cfg['target_label'], LABEL)

    def test_reserve_change_is_invalid(self):
        problems, _cfg = performance.validate_manifest(
            manifest(reserve_mib=1024))
        self.assertIn('reserve_mib_must_stay_512', problems)

    def test_route_plan_change_is_invalid(self):
        problems, _cfg = performance.validate_manifest(
            manifest(route_plan=['full', 'split', 'overlap']))
        self.assertIn('route_plan_must_be_prefill_full_prefill_split_overlap',
                      problems)

    def test_missing_target_label_is_invalid(self):
        bad = manifest()
        del bad['target_label']
        problems, _cfg = performance.validate_manifest(bad)
        self.assertIn('manifest_field_missing:target_label', problems)

    def test_sampling_must_be_fixed(self):
        bad = manifest()
        bad['decode']['sampling']['ignore_eos'] = False
        problems, _cfg = performance.validate_manifest(bad)
        self.assertIn('manifest_sampling_ignore_eos_must_be_true', problems)


# ---------------------------------------------------------------------------
# loopback SSE + injected controller integration
# ---------------------------------------------------------------------------

class FakeCompletionServer:
    """Fake /v1/completions SSE source with an optional decode stall."""

    def __init__(self):
        self.pause = threading.Event()
        self.stop = threading.Event()
        self.prefill_delay_s = 0.30
        self.prefill_lock = threading.Lock()
        self.active_prefills = self.max_active_prefills = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, format, *args):  # noqa: A002 - base signature
                pass

            def do_POST(self):
                length = int(self.headers.get('Content-Length', '0'))
                body = json.loads(self.rfile.read(length))
                prefill = body.get('max_tokens') == 1
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Connection', 'close')
                self.end_headers()
                self.close_connection = True

                def emit(tokens, finish, completion):
                    event = {
                        'choices': [{'index': 0, 'token_ids': tokens,
                                     'finish_reason': finish}],
                        'usage': {'completion_tokens': completion,
                                  'prompt_tokens': 16},
                    }
                    self.wfile.write(
                        ('data: ' + json.dumps(event) + '\n\n').encode())
                    self.wfile.flush()

                try:
                    if prefill:
                        with outer.prefill_lock:
                            outer.active_prefills += 1
                            outer.max_active_prefills = max(outer.max_active_prefills, outer.active_prefills)
                        try:
                            time.sleep(outer.prefill_delay_s)
                            emit([1], None, 0)
                            emit([], 'length', 1)
                        finally:
                            with outer.prefill_lock:
                                outer.active_prefills -= 1
                        return
                    count = 1
                    emit([1], None, 0)
                    for _ in range(1000):
                        if outer.stop.is_set():
                            return
                        while outer.pause.is_set() and not outer.stop.is_set():
                            time.sleep(0.005)
                        time.sleep(0.015)
                        count += 1
                        emit([count % 1000], None, count)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return

        self.httpd = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.url = 'http://127.0.0.1:%d' % self.httpd.server_address[1]

    def close(self):
        self.stop.set()
        self.httpd.shutdown()
        self.httpd.server_close()


class FakeController:
    """Schema-1 receipt source; no worker or engine is involved."""

    def __init__(self, server, *, stall=False, useful=512, completed=1,
                 crash=False):
        self.server = server
        self.stall = stall
        self.useful = useful
        self.completed = completed
        self.crash = crash
        self.start_ns = None

    def execute(self, command, _timeout):
        action = command['action']
        receipt = {'schema': 1, 'action': action,
                   'run_id': command['run_id'], 'engine_epoch': 7,
                   'command_seq': command.get('command_seq', 0)}
        if action == 'overlap_begin':
            if self.stall:
                self.server.pause.set()
                time.sleep(0.05)
            self.start_ns = time.monotonic_ns()
            receipt['measure_start_ns'] = self.start_ns
        elif action == 'overlap_end':
            if self.crash:
                raise performance.protocol.ControllerTransportError(
                    'injected-transport-failure')
            end_ns = time.monotonic_ns()
            if self.stall:
                self.server.pause.clear()
            receipt['measure_end_ns'] = end_ns
            receipt['interval'] = {'reset_count': 0}
            receipt['valid_completed_turns'] = 8
            receipt['cancellations'] = 0
            receipt['errors'] = 0
            receipt['worker_incarnation'] = 'w1'
            receipt['storage_epoch'] = 3
            receipt['route_generation'] = 11
            receipt['overlap'] = {
                'enabled': True, 'native_ready': True,
                'segments': [{
                    'segment_seq': 1,
                    'start_monotonic_ns': self.start_ns,
                    'end_monotonic_ns': end_ns,
                    'accepted': True, 'path': 'e3_overlap',
                    'copy_route': 'dense', 'decode_arrivals': 10,
                    'committed_tokens_before': 0,
                    'committed_tokens_after': self.useful,
                    'useful_tokens_before': 0,
                    'useful_tokens_after': self.useful,
                    'prefill_completed': self.completed >= 1,
                    'ranks': [_rank(index) for index in range(4)],
                }],
                'prefill': {'useful_completed_tokens': self.useful,
                            'prefill_wall_s': 0.7,
                            'prompt_tokens': 512,
                            'completed_prefills': self.completed},
                'allocator_peaks': {'gpu0': 1},
            }
        return receipt, {'kind': 'injected'}


def test_config():
    data = manifest()
    data['decode']['prompt_tokens'] = 64
    data['decode']['generation_budget'] = 200
    data['prefill']['fresh']['prompt_tokens'] = 64
    data['prefill']['append']['prompt_tokens'] = 8
    data['timeouts'] = {'controller_s': 5.0, 'read_s': 5.0,
                        'drain_s': 5.0, 'join_s': 5.0, 'finish_s': 5.0}
    problems, cfg = performance.validate_manifest(data)
    if problems:
        raise AssertionError(problems)
    return cfg


def w1_fixture(route, rate, gap=2.0):
    return {
        'summary': {'routes': {route: {
            'route': route, 'valid': True,
            'aggregate_tokens_per_s_median': rate,
            'stream_rates_tokens_per_s': [rate] * 8,
            'gap_p99_ms': gap, 'workload_sha256': SHA_W}}},
        'config': {'WORKLOAD_SHA256': SHA_W},
        'repeats': [{'route': route, 'valid': True,
                     'metrics': {'stream_rates_tokens_per_s': [rate] * 8,
                                 'aggregate_tokens_per_s': rate * 8},
                     'streams': [{'window': {'gaps_ms': [gap]}}] * 8}],
    }


class IntegrationTests(unittest.TestCase):
    def test_append_is_submitted_after_fresh_completion(self):
        out = self.lane(FakeController(self.server))
        self.assertTrue(out['valid'], out)
        self.assertEqual(self.server.max_active_prefills, 1)

    def test_missing_native_clock_is_not_replaced_by_client_time(self):
        server = self.server
        class MissingClock(FakeController):
            def execute(self, command, timeout):
                receipt, transport = super().execute(command, timeout)
                receipt.pop('measure_start_ns', None)
                return receipt, transport
        out = self.lane(MissingClock(server))
        self.assertFalse(out['valid'])
        self.assertEqual(out['invalid_intervals'][0]['reasons'], ['measurement_start_unproved'])

    def test_receipt_from_a_different_engine_epoch_is_refused(self):
        server = self.server
        class ForeignEngine(FakeController):
            def execute(self, command, timeout):
                receipt, transport = super().execute(command, timeout)
                receipt['engine_epoch'] = 99
                return receipt, transport
        out = self.lane(ForeignEngine(server))
        self.assertFalse(out['valid'])
        self.assertEqual(out['invalid_intervals'][0]['reasons'], ['receipt_engine_epoch_mismatch'])

    def setUp(self):
        self.server = FakeCompletionServer()

    def tearDown(self):
        self.server.close()

    def lane(self, controller):
        return performance.run_lane_overlap(
            test_config(), controller, self.server.url, 'fake-model',
            {'engine_epoch': 7}, 'q18a-test-run')

    def test_true_concurrent_progress_is_accepted(self):
        out = self.lane(FakeController(self.server))
        self.assertTrue(out['valid'], out)
        self.assertEqual(out['invalid_intervals'], [])
        self.assertGreater(out['decode_tokens_per_s'], 0)
        self.assertGreater(
            sum(row['in_window_arrivals'] for row in out['window']['streams']),
            0)
        self.assertTrue(all(row['completion_delta'] > 0
                            for row in out['window']['streams']), out['window'])

    def test_serialized_false_positive_is_refused(self):
        out = self.lane(FakeController(self.server, stall=True))
        self.assertFalse(out['valid'])
        reasons = out['invalid_intervals'][0]['reasons']
        self.assertIn('no_client_decode_arrivals_during_prefill', reasons)
        self.assertIn('serialized_segment_no_client_arrivals', reasons)
        self.assertEqual(
            sum(row['in_window_arrivals'] for row in out['window']['streams']),
            0)

    def test_missing_prefill_completion_is_inconclusive_at_grade(self):
        out = self.lane(FakeController(self.server, useful=256, completed=0))
        out['workload_sha256'] = SHA_W
        artifact = {'run_id': 'q18a-test-run', 'target_label': LABEL,
                    'launcher_sha256': LAUNCHER, 'package_sha256': PACKAGE,
                    'lanes': {
                        'prefill_full': baseline('full', prefill=100.0),
                        'prefill_split': baseline('split', prefill=200.0)},
                    'overlap': out,
                    'memory': {'reserve_mib': 512,
                               'allocator_peaks': {'gpu0': 1}}}
        w1 = w1_fixture('full', 100.0)
        split = w1_fixture('split', 60.0)
        w1['summary']['routes'].update(split['summary']['routes'])
        w1['repeats'].extend(split['repeats'])
        data = grading.compose(w1, artifact,
                               samples=[{'pressure': True,
                                         'gpus': [{'index': index,
                                                   'free_mib': 600.0}
                                                  for index in range(4)]}],
                               cleanup=evidence()['cleanup'])
        result = grading.grade(data)
        self.assertEqual(result['verdict'], 'INCONCLUSIVE', result)
        self.assertIn('prefill_not_completed', result['missing'])

    def test_transport_failure_does_not_invent_engine_crash(self):
        out = self.lane(FakeController(self.server, crash=True))
        self.assertIsNone(out['engine_crash'])
        self.assertTrue(out['controller_transport_failed'])
        self.assertFalse(out['valid'])
        self.assertEqual(out['invalid_intervals'][0]['reasons'],
                         ['ControllerTransportError'])


if __name__ == '__main__':
    unittest.main()
