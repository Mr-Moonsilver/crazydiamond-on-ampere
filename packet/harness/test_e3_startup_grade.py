#!/usr/bin/env python3
"""CPU-only tests for the frozen q17b startup/capture grade.

Synthetic status records only: no torch, CUDA, docker, engine or network.  These
tests prove the verdict rules and artifact handling, not GPU behaviour.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from e3_startup_grade import grade, REQUESTED_DECODE_SMS

HERE = Path(__file__).resolve().parent
LIB = 'b' * 64
HELPER = 'a' * 64
RUN = 'q17s-test-run'


def stream_check(label):
    return {'descriptor': label, 'bank': 'decode-split',
            'stream_label': 'decode-split-stream',
            'context_label': 'decode-split-context',
            'vllm_matches_owner': True, 'torch_matches_owner': True,
            'basis': 'checked at graph stream argument; not node placement'}


def memory(**overrides):
    value = {'schema': 1, 'wall_s': 1.0, 'monotonic_ns': 1,
             'sample_end_wall_s': 1.1, 'sample_end_monotonic_ns': 2,
             'allocator': {'allocated_current_bytes': 100,
                           'reserved_current_bytes': 200,
                           'allocated_peak_bytes': 300,
                           'reserved_peak_bytes': 400,
                           'inactive_split_current_bytes': 10,
                           'inactive_split_peak_bytes': 20},
             'device_free_bytes': 1000, 'device_total_bytes': 16000,
             'peak_scope': 'allocator lifetime since last allocator reset; this probe never resets',
             'device_scope': 'point sample, not peak; allocator and device reads are sequential',
             'synchronized_by_snapshot': False, 'errors': []}
    value.update(overrides)
    return value


def rank_row(rank, generation=7, **overrides):
    labels = ['full:tokens=1:reqs=1:uniform=1', 'full:tokens=2:reqs=2:uniform=1']
    split = {'requested_decode_sms': REQUESTED_DECODE_SMS,
             'decode_sms': 48, 'prefill_sms': 34, 'total_sms': 82,
             'basis': 'measured',
             'membership': {'decode_seen': 48, 'prefill_seen': 34, 'total': 82,
                            'disjoint': True, 'ok': True, 'rounds': [[48], [34]]},
             'helper': {'library_sha256': LIB, 'abi': 1,
                        'helper_source_sha256': HELPER}}
    inventory = {'mode': 'decode-only-probe',
                 'helper_source_sha256': HELPER, 'library_sha256': LIB,
                 'binding_identity': {'library_sha256': LIB, 'abi': 1,
                                      'helper_source_sha256': HELPER},
                 'split': split,
                 'descriptors': [
                     {'label': labels[0], 'cg_mode': 'full', 'num_tokens': 1,
                      'num_reqs': 1, 'uniform_token_count': 1},
                     {'label': labels[1], 'cg_mode': 'full', 'num_tokens': 2,
                      'num_reqs': 2, 'uniform_token_count': 1}],
                 'descriptors_captured': 2, 'raw_graphs_retained': True,
                 'split_pool_bytes': 4096,
                 'split_pool_basis': 'device-free delta around capture; not isolated allocator pool bytes',
                 'split_pool_label': 'decode-split-pool',
                 'split_context_label': 'decode-split-context',
                 'split_stream_label': 'decode-split-stream',
                 'identity_scope': 'worker incarnation and storage epoch; opaque labels, not CUDA handles',
                 'capture_stream_checks': [stream_check(label) for label in labels],
                 'capture_memory': {'free_before_bytes': 8192, 'free_after_bytes': 4096,
                                    'delta_bytes': 4096,
                                    'basis': 'device-free samples around synchronized capture, not allocator pool attribution'},
                 'storage_epoch': 1, 'route': 'full', 'route_generation': generation,
                 'communicators': 'stock TP/PP retained; no additional prefill communicators',
                 'pp_channel_bytes': 0, 'ple_scratch_bytes': 0, 'excluded': []}
    counters = {'mode': 'decode-only-probe', 'route': 'full',
                'route_generation': generation, 'storage_epoch': 1,
                'quarantined': None, 'in_flight': 0, 'sampling_pending': False,
                'transition_pending': False, 'transition_count': 0,
                'invalidation_reasons': [], 'state': 'ready'}
    row = {'rank': rank, 'inventory': inventory, 'counters': counters, 'memory': memory()}
    row.update(overrides)
    return row


def status(generation=7, rows=None, **overrides):
    value = {'schema': 1, 'run_id': RUN, 'engine_epoch': 9, 'command_seq': 1,
             'action': 'status', 'quarantined': False, 'paused': False,
             'snapshot': {'route': 'full', 'route_generation': generation,
                          'storage_epoch': 1,
                          'ranks': rows if rows is not None else [rank_row(i, generation) for i in range(4)]}}
    value.update(overrides)
    return value


def opened(generation=7):
    return {'schema': 1, 'run_id': RUN, 'engine_epoch': 9, 'command_seq': 0,
            'action': 'open', 'quarantined': False, 'paused': False,
            'snapshot': {'route': 'full', 'route_generation': generation,
                         'storage_epoch': 1,
                         'ranks': [rank_row(i, generation) for i in range(4)]}}


def evidence():
    return {'pins': {'v1/worker/gpu_worker.py': 'c' * 64},
            'helper_source_sha256': HELPER,
            'generated_source_shas': {'decode_probe': 'd' * 64},
            'build_only_source_shas': {'operation_source': 'e' * 64},
            'tests': ['test_decode_probe.py']}


def samples(valid=True):
    rows = [{'kind': 'sample', 'clock': 'unix_realtime_seconds', 'start_s': 1.0,
             'end_s': 1.1, 'elapsed_s': 0.1, 'gpus': {str(i): {'used_mib': 100, 'total_mib': 1000}
                                    for i in range(4)}}]
    if valid:
        rows.append({'kind': 'stopped', 'wall_s': 2.0, 'clock': 'unix_realtime_seconds'})
    return rows


class VerdictRules(unittest.TestCase):
    def run_grade(self, status_record=None, open_record=None, ev=None, samples_value=None):
        return grade(opened() if open_record is None else open_record,
                     status() if status_record is None else status_record,
                     evidence() if ev is None else ev, LIB,
                     samples() if samples_value is None else samples_value)

    def assertVerdict(self, result, want):
        self.assertEqual(result['verdict'], want,
                         (result['failures'], result['missing']))
        if want == 'PASS':
            self.assertFalse(result['failures'] or result['missing'])
        return result

    def test_open_workers_and_pending_transitions_are_checked(self):
        bad = opened()
        bad['snapshot']['ranks'][0]['counters']['quarantined'] = 'failure'
        self.assertVerdict(self.run_grade(open_record=bad), 'FAIL')
        for value in (True, None):
            bad = status()
            bad['snapshot']['ranks'][0]['counters']['transition_pending'] = value
            self.assertVerdict(self.run_grade(bad), 'FAIL')
        bad = status()
        bad['snapshot']['ranks'][0]['counters']['in_flight'] = False
        self.assertVerdict(self.run_grade(bad), 'FAIL')
        bad = opened()
        bad['snapshot']['ranks'] = [{'rank': i} for i in range(4)]
        self.assertVerdict(self.run_grade(open_record=bad), 'INCONCLUSIVE')

    def test_inventory_and_membership_mutations_cannot_pass(self):
        mutations = (
            lambda i: i['descriptors'][0].update(num_tokens=999, num_reqs=999),
            lambda i: i['descriptors'][0].update(uniform_token_count=8),
            lambda i: i['split']['membership'].update(rounds=[None]),
            lambda i: i['split'].update(decode_sms=1, prefill_sms=81),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                bad = status()
                mutate(bad['snapshot']['ranks'][0]['inventory'])
                self.assertVerdict(self.run_grade(bad), 'FAIL')

    def test_memory_errors_bounds_and_clock_mutations(self):
        for mutate, expected in (
            (lambda r: r['memory'].update(errors=['device:RuntimeError']), 'INCONCLUSIVE'),
            (lambda r: r['memory'].update(wall_s=None), 'INCONCLUSIVE'),
            (lambda r: r['memory']['allocator'].update(allocated_current_bytes=300), 'FAIL'),
            (lambda r: r['inventory']['capture_memory'].update(free_before_bytes=999999), 'FAIL'),
        ):
            bad = status()
            mutate(bad['snapshot']['ranks'][0])
            self.assertVerdict(self.run_grade(bad), expected)
        bad_samples = samples()
        bad_samples[-1]['wall_s'] = -1
        self.assertVerdict(self.run_grade(samples_value=bad_samples), 'FAIL')
        bad_samples = samples()
        bad_samples[0]['error'] = 'ClockStep'
        self.assertVerdict(self.run_grade(samples_value=bad_samples), 'INCONCLUSIVE')

    def test_actual_controller_replies_compose(self):
        import asyncio
        from test_e3_probe_control import Engine, ProbeController
        engine = Engine()
        engine.generation = 7
        engine.corrupt = lambda command, rows: [rank_row(i) for i in range(4)]
        async def run():
            controller = ProbeController(engine, epoch=9)
            first = await controller.execute({'action': 'open', 'run_id': RUN})
            last = await controller.execute({'action': 'status', 'run_id': RUN,
                                             'engine_epoch': 9, 'seq': 1})
            return grade(first, last, evidence(), LIB, samples())
        self.assertVerdict(asyncio.run(run()), 'PASS')

    def test_pass_records_requested_and_rounded_measurement_separately(self):
        result = self.assertVerdict(self.run_grade(), 'PASS')
        self.assertEqual(result['measured']['requested_decode_sms'], 44)
        self.assertEqual(result['measured']['decode_sms'], 48)
        self.assertEqual(result['measured']['prefill_sms'], 34)
        self.assertEqual(result['measured']['total_sms'], 82)
        self.assertTrue(result['requested_decode_sms_is_unchanged'])
        self.assertTrue(result['measured_decode_sms_recorded_separately'])
        self.assertFalse(result['full_package_armable'])
        self.assertIn('startup/capture only', result['scope'])
        self.assertEqual(set(result['ranks']['rank']), {'0', '1', '2', '3'})
        self.assertEqual(result['ranks']['device_samples']['valid'], 1)
        rank0 = result['ranks']['rank']['0']
        self.assertEqual(rank0['allocator']['reserved_peak_bytes'], 400)
        self.assertEqual(rank0['allocator']['inactive_split_current_bytes'], 10)
        self.assertEqual(rank0['device_free_bytes'], 1000)
        self.assertIn('not peak', rank0['device_scope'])

    def test_measured_split_must_agree_across_ranks(self):
        bad = status()
        split = bad['snapshot']['ranks'][3]['inventory']['split']
        split.update(decode_sms=47, prefill_sms=35)
        split['membership'].update(decode_seen=47, prefill_seen=35)
        self.assertVerdict(self.run_grade(bad), 'FAIL')
        # An unknown memory observation on another rank must not hide the
        # measured split disagreement: contradiction still dominates.
        bad['snapshot']['ranks'][0].pop('memory')
        result = self.assertVerdict(self.run_grade(bad), 'FAIL')
        self.assertTrue(result['missing'])
        self.assertIn('measured split allocation differs between ranks', result['failures'])

    def test_requested_split_44_cannot_drift_while_rounding_is_allowed(self):
        bad = status()
        for row in bad['snapshot']['ranks']:
            row['inventory']['split']['requested_decode_sms'] = 43
        self.assertVerdict(self.run_grade(bad), 'FAIL')
        good = status()
        for row in good['snapshot']['ranks']:
            row['inventory']['split'].update(decode_sms=44, prefill_sms=38, total_sms=82)
            row['inventory']['split']['membership'].update(decode_seen=44, prefill_seen=38, rounds=[[44], [38]])
        good_open = opened()
        good_open['snapshot'] = __import__('copy').deepcopy(good['snapshot'])
        result = self.assertVerdict(self.run_grade(good, open_record=good_open), 'PASS')
        self.assertEqual(result['measured']['decode_sms'], 44)

    def test_missing_or_malformed_status_is_inconclusive(self):
        self.assertVerdict(self.run_grade(open_record={}), 'INCONCLUSIVE')
        bad = status()
        del bad['snapshot']
        self.assertVerdict(self.run_grade(bad), 'INCONCLUSIVE')

    def test_missing_quarantine_state_is_not_accepted(self):
        bad = status()
        del bad['quarantined']
        self.assertVerdict(self.run_grade(bad), 'FAIL')
        bad = status()
        del bad['snapshot']['ranks'][0]['counters']['quarantined']
        self.assertVerdict(self.run_grade(bad), 'FAIL')

    def test_duplicate_or_missing_rank_fails(self):
        rows = [rank_row(i) for i in range(4)]
        rows[3]['rank'] = 2
        self.assertVerdict(self.run_grade(status(rows=rows)), 'FAIL')
        rows = [rank_row(i) for i in range(4)]
        rows.pop()
        self.assertVerdict(self.run_grade(status(rows=rows)), 'FAIL')

    def test_empty_or_duplicate_capture_fails(self):
        bad = status()
        for row in bad['snapshot']['ranks']:
            row['inventory']['descriptors'] = []
            row['inventory']['descriptors_captured'] = 0
            row['inventory']['capture_stream_checks'] = []
        self.assertVerdict(self.run_grade(bad), 'FAIL')
        bad = status()
        bad['snapshot']['ranks'][0]['inventory']['descriptors'].append(
            dict(bad['snapshot']['ranks'][0]['inventory']['descriptors'][0]))
        self.assertVerdict(self.run_grade(bad), 'FAIL')

    def test_wrong_helper_or_library_identity_fails(self):
        for field, value in (('helper_source_sha256', 'f' * 64),
                             ('library_sha256', 'f' * 64)):
            bad = status()
            bad['snapshot']['ranks'][2]['inventory'][field] = value
            self.assertVerdict(self.run_grade(bad), 'FAIL')
        bad = status()
        bad['snapshot']['ranks'][0]['inventory']['binding_identity']['abi'] = 2
        self.assertVerdict(self.run_grade(bad), 'FAIL')
        bad_ev = evidence()
        bad_ev['helper_source_sha256'] = 'f' * 64
        self.assertVerdict(self.run_grade(ev=bad_ev), 'FAIL')

    def test_unknown_memory_is_inconclusive_but_invalid_is_fail(self):
        unknown = status()
        unknown['snapshot']['ranks'][1]['memory']['allocator']['allocated_peak_bytes'] = None
        unknown['snapshot']['ranks'][1]['memory']['errors'] = ['allocator:RuntimeError']
        self.assertVerdict(self.run_grade(unknown), 'INCONCLUSIVE')
        missing_device = status()
        missing_device['snapshot']['ranks'][1]['memory']['device_free_bytes'] = None
        self.assertVerdict(self.run_grade(missing_device), 'INCONCLUSIVE')
        del missing_device['snapshot']['ranks'][2]['memory']
        self.assertVerdict(self.run_grade(missing_device), 'INCONCLUSIVE')
        invalid = status()
        invalid['snapshot']['ranks'][0]['memory']['device_free_bytes'] = 17000
        self.assertVerdict(self.run_grade(invalid), 'FAIL')
        invalid = status()
        invalid['snapshot']['ranks'][0]['memory']['allocator']['reserved_peak_bytes'] = -1
        self.assertVerdict(self.run_grade(invalid), 'FAIL')
        unknown_split = status()
        unknown_split['snapshot']['ranks'][0]['inventory']['split_pool_bytes'] = None
        self.assertVerdict(self.run_grade(unknown_split), 'INCONCLUSIVE')
        unknown_capture = status()
        unknown_capture['snapshot']['ranks'][0]['inventory']['capture_memory'] = None
        self.assertVerdict(self.run_grade(unknown_capture), 'INCONCLUSIVE')
        unknown_capture = status()
        unknown_capture['snapshot']['ranks'][1]['inventory']['capture_memory']['free_before_bytes'] = None
        self.assertVerdict(self.run_grade(unknown_capture), 'INCONCLUSIVE')
        invalid = status()
        invalid['snapshot']['ranks'][0]['inventory']['split_pool_bytes'] = -1
        self.assertVerdict(self.run_grade(invalid), 'FAIL')

    def test_capture_and_stream_contradictions_fail(self):
        bad = status()
        r = bad['snapshot']['ranks'][0]['inventory']
        r['capture_memory']['delta_bytes'] = 123
        self.assertVerdict(self.run_grade(bad), 'FAIL')
        bad = status()
        bad['snapshot']['ranks'][0]['inventory']['capture_stream_checks'][0]['torch_matches_owner'] = False
        self.assertVerdict(self.run_grade(bad), 'FAIL')
        bad = status()
        del bad['snapshot']['ranks'][0]['inventory']['capture_stream_checks'][0]
        self.assertVerdict(self.run_grade(bad), 'FAIL')
        bad = status()
        bad['snapshot']['ranks'][0]['inventory']['split']['membership']['disjoint'] = False
        self.assertVerdict(self.run_grade(bad), 'FAIL')

    def test_incarnation_route_and_epoch_changes_fail(self):
        bad = status(engine_epoch=10)
        self.assertVerdict(self.run_grade(bad), 'FAIL')
        bad = status()
        bad['snapshot']['storage_epoch'] = 2
        self.assertVerdict(self.run_grade(bad), 'FAIL')
        bad = status(generation=8)
        self.assertVerdict(self.run_grade(bad), 'FAIL')
        bad = status()
        bad['paused'] = True
        self.assertVerdict(self.run_grade(bad), 'FAIL')

    def test_no_replay_or_transition_during_startup(self):
        bad = status()
        for row in bad['snapshot']['ranks']:
            row['counters']['transition_count'] = 1
        self.assertVerdict(self.run_grade(bad), 'FAIL')
        bad = status()
        bad['snapshot']['ranks'][0]['counters']['invalidation_reasons'] = ['unsupported_descriptor']
        self.assertVerdict(self.run_grade(bad), 'FAIL')

    def test_device_sample_evidence_is_required_and_clean(self):
        self.assertVerdict(self.run_grade(samples_value=[]), 'INCONCLUSIVE')
        self.assertVerdict(self.run_grade(samples_value=samples(valid=False)), 'INCONCLUSIVE')
        bad = samples()
        bad[0]['gpus'].pop('3')
        self.assertVerdict(self.run_grade(samples_value=bad), 'INCONCLUSIVE')
        bad = samples()
        bad[0]['gpus']['0']['used_mib'] = 2000
        self.assertVerdict(self.run_grade(samples_value=bad), 'FAIL')
        untimed = samples()
        untimed[0].pop('start_s')
        untimed[0].pop('end_s')
        self.assertVerdict(self.run_grade(samples_value=untimed), 'INCONCLUSIVE')
        untimed = samples()
        untimed[-1].pop('wall_s')
        self.assertVerdict(self.run_grade(samples_value=untimed), 'INCONCLUSIVE')

    def test_fail_dominates_inconclusive(self):
        bad = status()
        bad['snapshot']['ranks'][1]['memory']['allocator']['reserved_peak_bytes'] = None
        bad['snapshot']['ranks'][2]['inventory']['library_sha256'] = 'f' * 64
        result = self.assertVerdict(self.run_grade(bad), 'FAIL')
        self.assertTrue(result['missing'])


class CommandLine(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.write('open.json', opened())
        self.write('status.json', status())
        self.write('evidence.json', evidence())
        self.write_samples(samples())

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, value):
        (self.root / name).write_text(json.dumps(value))

    def write_samples(self, rows):
        (self.root / 'samples.jsonl').write_text(
            ''.join(json.dumps(row) + '\n' for row in rows))

    def run_cli(self, out='grade.json', samples='samples.jsonl', status='status.json'):
        return subprocess.run(
            [sys.executable, str(HERE / 'e3_startup_grade.py'),
             '--open', str(self.root / 'open.json'),
             '--status', str(self.root / status),
             '--evidence', str(self.root / 'evidence.json'),
             '--library-sha256', LIB,
             '--samples', str(self.root / samples),
             '--out', str(self.root / out)],
            capture_output=True, text=True, timeout=30)

    def test_cli_pass_writes_exclusive_result(self):
        done = self.run_cli()
        self.assertEqual(done.returncode, 0, done.stderr)
        result = json.loads((self.root / 'grade.json').read_text())
        self.assertEqual(result['verdict'], 'PASS')
        self.assertIn('PASS startup/capture-only', done.stdout)
        again = self.run_cli()
        self.assertEqual(again.returncode, 2)
        self.assertEqual(json.loads((self.root / 'grade.json').read_text()), result)

    def test_cli_missing_or_malformed_input_is_inconclusive(self):
        (self.root / 'samples.jsonl').write_text('not json')
        done = self.run_cli(out='missing.json')
        self.assertEqual(done.returncode, 2, done.stderr)
        self.assertEqual(json.loads((self.root / 'missing.json').read_text())['verdict'],
                         'INCONCLUSIVE')
        done = self.run_cli(out='absent.json', status='does-not-exist.json')
        self.assertEqual(done.returncode, 2, done.stderr)
        result = json.loads((self.root / 'absent.json').read_text())
        self.assertIn('INCONCLUSIVE', result['verdict'])
        self.assertFalse(result['measured_decode_sms_recorded_separately'])

    def test_cli_hard_failure_exit_one(self):
        bad = status()
        bad['snapshot']['ranks'][0]['inventory']['helper_source_sha256'] = 'f' * 64
        self.write('bad.json', bad)
        done = self.run_cli(out='fail.json', status='bad.json')
        self.assertEqual(done.returncode, 1, done.stderr)
        self.assertEqual(json.loads((self.root / 'fail.json').read_text())['verdict'], 'FAIL')

    def test_import_has_no_runtime_side_effect(self):
        code = ('import sys; sys.path.insert(0, %r); import e3_startup_grade; '
                'assert "torch" not in sys.modules; assert "docker" not in sys.modules'
                % str(HERE))
        done = subprocess.run([sys.executable, '-c', code], capture_output=True,
                              text=True, timeout=30)
        self.assertEqual(done.returncode, 0, done.stderr)


if __name__ == '__main__':
    unittest.main()
